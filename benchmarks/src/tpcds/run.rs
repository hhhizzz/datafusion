// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::util::metrics_object_store::{
    MetricsObjectStore, OBJECT_STORE_COALESCE_PARALLEL_DEFAULT,
    OBJECT_STORE_COALESCE_PARALLEL_MAX, ObjectStoreMetrics,
};
use crate::util::{BenchmarkRun, CommonOpt, print_memory_stats};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::{self, pretty_format_batches};
use arrow_row::{RowConverter, SortField};
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::error::Result;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::*;
use datafusion_common::instant::Instant;
use datafusion_common::utils::get_available_parallelism;
use datafusion_common::{
    Constraint, Constraints, DEFAULT_PARQUET_EXTENSION, DataFusionError, plan_err,
};
use object_store::OBJECT_STORE_COALESCE_DEFAULT;
use object_store::aws::AmazonS3Builder;
use sha2::{Digest, Sha256};
use url::Url;

use clap::Args;
use log::info;

// hack to avoid `default_value is meaningless for bool` errors
type BoolDefaultTrue = bool;
pub const TPCDS_QUERY_START_ID: usize = 1;
pub const TPCDS_QUERY_END_ID: usize = 99;
const OBJECT_STORE_COALESCE_GAP_ENV: &str = "TPCDS_OBJECT_STORE_COALESCE_GAP_BYTES";
const OBJECT_STORE_COALESCE_PARALLELISM_ENV: &str =
    "TPCDS_OBJECT_STORE_COALESCE_PARALLELISM";

struct QueryOutput {
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
}

struct TpcdsIterationEvidence {
    query: usize,
    iteration: usize,
    elapsed: std::time::Duration,
    row_count: usize,
    result_hash: String,
}

fn commit_query_evidence(
    benchmark_run: &mut BenchmarkRun,
    committed_evidence: &mut Vec<TpcdsIterationEvidence>,
    query_run: Result<Vec<TpcdsIterationEvidence>>,
) -> Result<()> {
    let evidence = match query_run {
        Ok(evidence) => evidence,
        Err(error) => {
            benchmark_run.mark_failed();
            return Err(error);
        }
    };
    for record in &evidence {
        benchmark_run.write_iter_with_result_hash(
            record.elapsed,
            record.row_count,
            Some(record.result_hash.clone()),
        );
    }
    committed_evidence.extend(evidence);
    Ok(())
}

fn publish_evidence_records(
    evidence: &[TpcdsIterationEvidence],
    output: &mut impl Write,
) -> Result<()> {
    let lines = evidence
        .iter()
        .map(|record| {
            format!(
                "TPCDS_RESULT_HASH query={} iteration={} sha256={} rows={}",
                record.query, record.iteration, record.result_hash, record.row_count
            )
        })
        .collect::<Vec<_>>();

    for line in lines {
        writeln!(output, "{line}")
            .map_err(|error| DataFusionError::External(Box::new(error)))?;
    }
    Ok(())
}

fn commit_benchmark_and_publish(
    benchmark_run: &BenchmarkRun,
    output_path: Option<&Path>,
    evidence: &[TpcdsIterationEvidence],
    output: &mut impl Write,
) -> Result<()> {
    benchmark_run.maybe_write_json(output_path)?;
    publish_evidence_records(evidence, output)
}

pub const TPCDS_TABLES: &[&str] = &[
    "call_center",
    "customer_address",
    "household_demographics",
    "promotion",
    "store_sales",
    "web_page",
    "catalog_page",
    "customer_demographics",
    "income_band",
    "reason",
    "store",
    "web_returns",
    "catalog_returns",
    "customer",
    "inventory",
    "ship_mode",
    "time_dim",
    "web_sales",
    "catalog_sales",
    "date_dim",
    "item",
    "store_returns",
    "warehouse",
    "web_site",
];

static TPCDS_PRIMARY_KEYS: &[(&str, &[&str])] = &[
    ("call_center", &["cc_call_center_sk"]),
    ("catalog_page", &["cp_catalog_page_sk"]),
    ("catalog_returns", &["cr_item_sk", "cr_order_number"]),
    ("catalog_sales", &["cs_item_sk", "cs_order_number"]),
    ("customer", &["c_customer_sk"]),
    ("customer_address", &["ca_address_sk"]),
    ("customer_demographics", &["cd_demo_sk"]),
    ("date_dim", &["d_date_sk"]),
    ("household_demographics", &["hd_demo_sk"]),
    ("income_band", &["ib_income_band_sk"]),
    (
        "inventory",
        &["inv_date_sk", "inv_item_sk", "inv_warehouse_sk"],
    ),
    ("item", &["i_item_sk"]),
    ("promotion", &["p_promo_sk"]),
    ("reason", &["r_reason_sk"]),
    ("ship_mode", &["sm_ship_mode_sk"]),
    ("store", &["s_store_sk"]),
    ("store_returns", &["sr_item_sk", "sr_ticket_number"]),
    ("store_sales", &["ss_item_sk", "ss_ticket_number"]),
    ("time_dim", &["t_time_sk"]),
    ("warehouse", &["w_warehouse_sk"]),
    ("web_page", &["wp_web_page_sk"]),
    ("web_returns", &["wr_item_sk", "wr_order_number"]),
    ("web_sales", &["ws_item_sk", "ws_order_number"]),
    ("web_site", &["web_site_sk"]),
];

/// Get the constraints for a TPC-DS table. Only primary keys are returned;
/// TPC-DS also defines foreign keys, but those are currently unsupported.
fn table_constraints(table: &str, schema: &Schema) -> Constraints {
    let columns = TPCDS_PRIMARY_KEYS
        .iter()
        .find(|(name, _)| *name == table)
        .map(|(_, columns)| *columns)
        .unwrap_or_else(|| unimplemented!("unknown TPC-DS table: {table}"));

    Constraints::new_unverified(vec![primary_key(schema, columns)])
}

fn primary_key(schema: &Schema, column_names: &[&str]) -> Constraint {
    let indices = column_names
        .iter()
        .map(|column_name| {
            schema.index_of(column_name).unwrap_or_else(|_| {
                panic!("primary key column '{column_name}' not found in schema")
            })
        })
        .collect();

    Constraint::PrimaryKey(indices)
}

/// Get the SQL statements from the specified query file
pub fn get_query_sql(base_query_path: &str, query: usize) -> Result<Vec<String>> {
    if query > 0 && query < 100 {
        let filename = format!("{base_query_path}/{query}.sql");
        let mut errors = vec![];
        match fs::read_to_string(&filename) {
            Ok(contents) => {
                return Ok(contents
                    .split(';')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect());
            }
            Err(e) => errors.push(format!("{filename}: {e}")),
        };

        plan_err!("invalid query. Could not find query: {:?}", errors)
    } else {
        plan_err!("invalid query. Expected value between 1 and 99")
    }
}

/// Run the tpcds benchmark.
#[derive(Debug, Args, Clone)]
#[command(verbatim_doc_comment)]
pub struct RunOpt {
    /// Query number. If not specified, runs all queries
    #[arg(short, long)]
    pub query: Option<usize>,

    /// Common options
    #[command(flatten)]
    common: CommonOpt,

    /// Path to data files
    #[arg(required = true, short = 'p', long = "path")]
    path: PathBuf,

    /// Path to query files
    #[arg(required = true, short = 'Q', long = "query_path")]
    query_path: PathBuf,

    /// Load the data into a MemTable before executing the query
    #[arg(short = 'm', long = "mem-table")]
    mem_table: bool,

    /// Path to machine readable output file
    #[arg(short = 'o', long = "output")]
    output_path: Option<PathBuf>,

    /// Whether to disable collection of statistics (and cost based optimizations) or not.
    #[arg(short = 'S', long = "disable-statistics")]
    disable_statistics: bool,

    /// If true then hash join used, if false then sort merge join
    /// True by default.
    #[arg(short = 'j', long = "prefer_hash_join", default_value = "true")]
    prefer_hash_join: BoolDefaultTrue,

    /// If true then Piecewise Merge Join can be used, if false then it will opt for Nested Loop Join
    /// False by default.
    #[arg(
        short = 'w',
        long = "enable_piecewise_merge_join",
        default_value = "false"
    )]
    enable_piecewise_merge_join: BoolDefaultTrue,

    /// Mark the first column of each table as sorted in ascending order.
    /// The tables should have been created with the `--sort` option for this to have any effect.
    #[arg(short = 't', long = "sorted")]
    sorted: bool,

    /// How many bytes to buffer on the probe side of hash joins.
    #[arg(long, default_value = "0")]
    hash_join_buffering_capacity: usize,
}

impl RunOpt {
    pub async fn run(self) -> Result<()> {
        println!("Running benchmarks with the following options: {self:?}");
        let query_range = match self.query {
            Some(query_id) => query_id..=query_id,
            None => TPCDS_QUERY_START_ID..=TPCDS_QUERY_END_ID,
        };

        let mut benchmark_run = BenchmarkRun::new();
        let mut committed_evidence = Vec::new();
        let mut config = self
            .common
            .config()?
            .with_collect_statistics(!self.disable_statistics);
        config.options_mut().optimizer.prefer_hash_join = self.prefer_hash_join;
        config.options_mut().optimizer.enable_piecewise_merge_join =
            self.enable_piecewise_merge_join;
        config.options_mut().execution.hash_join_buffering_capacity =
            self.hash_join_buffering_capacity;
        let rt = self.common.build_runtime()?;
        let ctx = SessionContext::new_with_config_rt(config, rt);

        let object_store_metrics =
            register_s3_object_store(&ctx, self.path.to_str().unwrap())?;

        // register tables
        self.register_tables(&ctx).await?;
        if let Some(metrics) = object_store_metrics.as_ref() {
            metrics.reset();
        }

        for query_id in query_range {
            benchmark_run.start_new_case(&format!("Query {query_id}"));
            let query_run = self
                .benchmark_query(query_id, &ctx, object_store_metrics.as_ref())
                .await;
            if let Err(error) = commit_query_evidence(
                &mut benchmark_run,
                &mut committed_evidence,
                query_run,
            ) {
                eprintln!("Query {query_id} failed: {error}");
            }
        }
        {
            let stdout = std::io::stdout();
            let mut stdout = stdout.lock();
            commit_benchmark_and_publish(
                &benchmark_run,
                self.output_path.as_deref(),
                &committed_evidence,
                &mut stdout,
            )?;
        }
        benchmark_run.maybe_print_failures();
        Ok(())
    }

    async fn benchmark_query(
        &self,
        query_id: usize,
        ctx: &SessionContext,
        object_store_metrics: Option<&ObjectStoreMetrics>,
    ) -> Result<Vec<TpcdsIterationEvidence>> {
        let mut millis = vec![];
        let mut evidence = vec![];

        let sql = &get_query_sql(self.query_path.to_str().unwrap(), query_id)?;

        if self.common.debug {
            println!("=== SQL for query {query_id} ===\n{}\n", sql.join(";\n"));
        }

        for i in 0..self.iterations() {
            if let Some(metrics) = object_store_metrics {
                metrics.reset();
            }
            let start = Instant::now();

            // query 15 is special, with 3 statements. the second statement is the one from which we
            // want to capture the results
            let mut result = None;

            for query in sql {
                result = Some(self.execute_query(ctx, query).await?);
            }
            let result = result.ok_or_else(|| {
                DataFusionError::Plan("TPC-DS query is empty".to_string())
            })?;

            let elapsed = start.elapsed();
            if let Some(metrics) = object_store_metrics {
                let snapshot = metrics.snapshot();
                let json = serde_json::to_string(&snapshot)
                    .map_err(|error| DataFusionError::External(Box::new(error)))?;
                println!(
                    "TPCDS_OBJECT_STORE_METRICS query={query_id} iteration={i} {json}"
                );
            }
            let ms = elapsed.as_secs_f64() * 1000.0;
            millis.push(ms);
            info!("output:\n\n{}\n\n", pretty_format_batches(&result.batches)?);
            let row_count = result.batches.iter().map(|b| b.num_rows()).sum();
            let result_hash = canonical_result_hash(&result.schema, &result.batches)?;
            println!(
                "Query {query_id} iteration {i} took {ms:.1} ms and returned {row_count} rows"
            );
            evidence.push(TpcdsIterationEvidence {
                query: query_id,
                iteration: i,
                elapsed,
                row_count,
                result_hash,
            });
        }

        let avg = millis.iter().sum::<f64>() / millis.len() as f64;
        println!("Query {query_id} avg time: {avg:.2} ms");

        // Print memory stats using mimalloc (only when compiled with --features mimalloc_extended)
        print_memory_stats();

        Ok(evidence)
    }

    async fn register_tables(&self, ctx: &SessionContext) -> Result<()> {
        for table in TPCDS_TABLES {
            let table_provider = { self.get_table(ctx, table).await? };

            if self.mem_table {
                println!("Loading table '{table}' into memory");
                let start = Instant::now();
                let memtable =
                    MemTable::load(table_provider, Some(self.partitions()), &ctx.state())
                        .await?;
                println!(
                    "Loaded table '{}' into memory in {} ms",
                    table,
                    start.elapsed().as_millis()
                );
                ctx.register_table(*table, Arc::new(memtable))?;
            } else {
                ctx.register_table(*table, table_provider)?;
            }
        }
        Ok(())
    }

    async fn execute_query(
        &self,
        ctx: &SessionContext,
        sql: &str,
    ) -> Result<QueryOutput> {
        let debug = self.common.debug;
        let plan = ctx.sql(sql).await?;
        let (state, plan) = plan.into_parts();

        if debug {
            println!("=== Logical plan ===\n{plan}\n");
        }

        let plan = state.optimize(&plan)?;
        if debug {
            println!("=== Optimized logical plan ===\n{plan}\n");
        }
        let physical_plan = state.create_physical_plan(&plan).await?;
        if debug {
            println!(
                "=== Physical plan ===\n{}\n",
                displayable(physical_plan.as_ref()).indent(true)
            );
        }
        let schema = physical_plan.schema();
        let result = collect(physical_plan.clone(), state.task_ctx()).await?;
        if debug {
            println!(
                "=== Physical plan with metrics ===\n{}\n",
                DisplayableExecutionPlan::with_metrics(physical_plan.as_ref())
                    .indent(true)
            );
            if !result.is_empty() {
                // do not call print_batches if there are no batches as the result is confusing
                // and makes it look like there is a batch with no columns
                pretty::print_batches(&result)?;
            }
        }
        Ok(QueryOutput {
            batches: result,
            schema,
        })
    }

    async fn get_table(
        &self,
        ctx: &SessionContext,
        table: &str,
    ) -> Result<Arc<dyn TableProvider>> {
        let path = self.path.to_str().unwrap();

        // Obtain a snapshot of the SessionState
        let state = ctx.state();
        let path = format!("{path}/{table}.parquet");

        // Check if the file exists
        if !Path::new(&path).exists() {
            eprintln!("Warning registering {table}: Table file does not exist: {path}");
        }

        let format = ParquetFormat::default()
            .with_options(ctx.state().table_options().parquet.clone());

        let table_path = ListingTableUrl::parse(path)?;
        let options = ListingOptions::new(Arc::new(format))
            .with_file_extension(DEFAULT_PARQUET_EXTENSION);

        let schema = options.infer_schema(&state, &table_path).await?;
        let constraints = table_constraints(table, schema.as_ref());

        if self.common.debug {
            println!(
                "Inferred schema from {table_path} for table '{table}':\n{schema:#?}\n"
            );
        }

        let options = if self.sorted {
            let key_column_name = schema.fields()[0].name();
            options
                .with_file_sort_order(vec![vec![col(key_column_name).sort(true, false)]])
        } else {
            options
        };

        let config = ListingTableConfig::new(table_path)
            .with_listing_options(options)
            .with_schema(schema);

        let provider = ListingTable::try_new(config)?
            .with_constraints(constraints)
            .with_cache(ctx.runtime_env().cache_manager.get_file_statistic_cache());

        Ok(Arc::new(provider))
    }

    fn iterations(&self) -> usize {
        self.common.iterations
    }

    fn partitions(&self) -> usize {
        self.common
            .partitions
            .unwrap_or_else(get_available_parallelism)
    }
}

fn canonical_result_hash(schema: &SchemaRef, batches: &[RecordBatch]) -> Result<String> {
    let mut hasher = Sha256::new();
    hash_length_prefixed(&mut hasher, b"tpcds-result-v3-arrow-row");
    write_schema_identity(&mut hasher, schema);
    let converter = RowConverter::new(
        schema
            .fields()
            .iter()
            .map(|field| SortField::new(field.data_type().clone()))
            .collect(),
    )?;

    let mut rows = Vec::new();
    for batch in batches {
        if batch.schema() != *schema {
            return plan_err!("TPC-DS result batches have inconsistent schemas");
        }
        if schema.fields().is_empty() {
            rows.resize_with(rows.len().saturating_add(batch.num_rows()), Vec::new);
            continue;
        }
        let converted = converter.convert_columns(batch.columns())?;
        if converted.num_rows() != batch.num_rows() {
            return plan_err!("TPC-DS result row conversion changed the row count");
        }
        rows.extend(converted.iter().map(|row| row.as_ref().to_vec()));
    }
    rows.sort_unstable();
    hash_u64(&mut hasher, u64::try_from(rows.len()).unwrap_or(u64::MAX));
    for row in rows {
        hash_length_prefixed(&mut hasher, &row);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn write_schema_identity(hasher: &mut Sha256, schema: &SchemaRef) {
    hash_u64(
        hasher,
        u64::try_from(schema.fields().len()).unwrap_or(u64::MAX),
    );
    for field in schema.fields() {
        write_field_identity(hasher, field);
    }
    write_metadata(hasher, schema.metadata());
}

fn write_field_identity(hasher: &mut Sha256, field: &Field) {
    hash_length_prefixed(hasher, field.name().as_bytes());
    hasher.update([u8::from(field.is_nullable())]);
    write_metadata(hasher, field.metadata());
    write_data_type_identity(hasher, field.data_type());
}

fn write_data_type_identity(hasher: &mut Sha256, data_type: &DataType) {
    hash_length_prefixed(hasher, data_type.to_string().as_bytes());
    match data_type {
        DataType::List(field)
        | DataType::ListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::LargeList(field)
        | DataType::LargeListView(field)
        | DataType::Map(field, _) => write_field_identity(hasher, field),
        DataType::Struct(fields) => {
            hash_u64(hasher, u64::try_from(fields.len()).unwrap_or(u64::MAX));
            for field in fields {
                write_field_identity(hasher, field);
            }
        }
        DataType::Union(fields, _) => {
            hash_u64(hasher, u64::try_from(fields.len()).unwrap_or(u64::MAX));
            for (type_id, field) in fields.iter() {
                hasher.update(type_id.to_be_bytes());
                write_field_identity(hasher, field);
            }
        }
        DataType::Dictionary(key, value) => {
            write_data_type_identity(hasher, key);
            write_data_type_identity(hasher, value);
        }
        DataType::RunEndEncoded(run_ends, values) => {
            write_field_identity(hasher, run_ends);
            write_field_identity(hasher, values);
        }
        _ => {}
    }
}

fn write_metadata(
    hasher: &mut Sha256,
    metadata: &std::collections::HashMap<String, String>,
) {
    let mut entries = metadata.iter().collect::<Vec<_>>();
    entries.sort_unstable_by(|left, right| {
        left.0.cmp(right.0).then_with(|| left.1.cmp(right.1))
    });
    hash_u64(hasher, u64::try_from(entries.len()).unwrap_or(u64::MAX));
    for (key, value) in entries {
        hash_length_prefixed(hasher, key.as_bytes());
        hash_length_prefixed(hasher, value.as_bytes());
    }
}

fn hash_u64(target: &mut Sha256, value: u64) {
    target.update(value.to_be_bytes());
}

fn hash_length_prefixed(target: &mut Sha256, value: &[u8]) {
    hash_u64(target, u64::try_from(value.len()).unwrap_or(u64::MAX));
    target.update(value);
}

fn register_s3_object_store(
    ctx: &SessionContext,
    path: &str,
) -> Result<Option<ObjectStoreMetrics>> {
    let Some(object_store_url) = s3_object_store_url(path)? else {
        return Ok(None);
    };

    let object_store_url_ref: &Url = object_store_url.as_ref();
    let bucket_name = object_store_url_ref.host_str().ok_or_else(|| {
        DataFusionError::Plan(format!("S3 path must include a bucket name: {path}"))
    })?;

    let store = AmazonS3Builder::from_env()
        .with_bucket_name(bucket_name)
        .build()?;

    let metrics_enabled = std::env::var("TPCDS_OBJECT_STORE_METRICS")
        .is_ok_and(|value| parse_bool_flag(&value));
    let coalesce_gap = object_store_coalesce_gap_from_env()?;
    let coalesce_parallelism = object_store_coalesce_parallelism_from_env()?;

    if metrics_enabled {
        let effective_gap = coalesce_gap.unwrap_or(OBJECT_STORE_COALESCE_DEFAULT);
        let effective_parallelism =
            coalesce_parallelism.unwrap_or(OBJECT_STORE_COALESCE_PARALLEL_DEFAULT);
        let store = MetricsObjectStore::new_with_coalesce_options(
            store,
            effective_gap,
            effective_parallelism,
        );
        let metrics = store.metrics();
        ctx.register_object_store(object_store_url_ref, Arc::new(store));
        println!(
            "Registered instrumented S3 object store for {object_store_url} \
             coalesce_gap_bytes={effective_gap} \
             coalesce_parallelism={effective_parallelism}"
        );
        Ok(Some(metrics))
    } else if coalesce_gap.is_some() || coalesce_parallelism.is_some() {
        let effective_gap = coalesce_gap.unwrap_or(OBJECT_STORE_COALESCE_DEFAULT);
        let effective_parallelism =
            coalesce_parallelism.unwrap_or(OBJECT_STORE_COALESCE_PARALLEL_DEFAULT);
        let store = MetricsObjectStore::new_coalescing_with_options(
            store,
            effective_gap,
            effective_parallelism,
        );
        ctx.register_object_store(object_store_url_ref, Arc::new(store));
        println!(
            "Registered coalescing S3 object store for {object_store_url} \
             coalesce_gap_bytes={effective_gap} \
             coalesce_parallelism={effective_parallelism} metrics=false"
        );
        Ok(None)
    } else {
        ctx.register_object_store(object_store_url_ref, Arc::new(store));
        println!("Registered S3 object store for {object_store_url}");
        Ok(None)
    }
}

fn parse_bool_flag(value: &str) -> bool {
    matches!(value.to_ascii_lowercase().as_str(), "true" | "1" | "yes")
}

fn object_store_coalesce_gap_from_env() -> Result<Option<u64>> {
    match std::env::var(OBJECT_STORE_COALESCE_GAP_ENV) {
        Ok(value) => parse_object_store_coalesce_gap(Some(&value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(DataFusionError::Configuration(format!(
                "{OBJECT_STORE_COALESCE_GAP_ENV} must contain a UTF-8 u64 byte value"
            )))
        }
    }
}

fn parse_object_store_coalesce_gap(value: Option<&str>) -> Result<Option<u64>> {
    value
        .map(|value| {
            value.parse::<u64>().map_err(|error| {
                DataFusionError::Configuration(format!(
                    "invalid {OBJECT_STORE_COALESCE_GAP_ENV} value '{value}': {error}"
                ))
            })
        })
        .transpose()
}

fn object_store_coalesce_parallelism_from_env() -> Result<Option<usize>> {
    match std::env::var(OBJECT_STORE_COALESCE_PARALLELISM_ENV) {
        Ok(value) => parse_object_store_coalesce_parallelism(Some(&value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(DataFusionError::Configuration(format!(
                "{OBJECT_STORE_COALESCE_PARALLELISM_ENV} must contain a UTF-8 integer between 1 and {OBJECT_STORE_COALESCE_PARALLEL_MAX}"
            )))
        }
    }
}

fn parse_object_store_coalesce_parallelism(value: Option<&str>) -> Result<Option<usize>> {
    value
        .map(|value| {
            let parallelism = value.parse::<usize>().map_err(|error| {
                DataFusionError::Configuration(format!(
                    "invalid {OBJECT_STORE_COALESCE_PARALLELISM_ENV} value '{value}': {error}"
                ))
            })?;
            if !(1..=OBJECT_STORE_COALESCE_PARALLEL_MAX).contains(&parallelism) {
                return Err(DataFusionError::Configuration(format!(
                    "invalid {OBJECT_STORE_COALESCE_PARALLELISM_ENV} value '{value}': expected 1..={OBJECT_STORE_COALESCE_PARALLEL_MAX}"
                )));
            }
            Ok(parallelism)
        })
        .transpose()
}

fn s3_object_store_url(path: &str) -> Result<Option<ObjectStoreUrl>> {
    if !path.starts_with("s3://") && !path.starts_with("s3a://") {
        return Ok(None);
    }

    let listing_url = ListingTableUrl::parse(path)?;
    Ok(Some(listing_url.object_store()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io, str};

    use arrow::array::builder::{StringDictionaryBuilder, StringViewBuilder};
    use arrow::array::{Array, ArrayRef, Int32Array};
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};

    #[test]
    fn result_hash_ignores_batch_and_row_order() {
        let schema = hash_schema("value", true);
        let one_batch = vec![int_batch(
            Arc::clone(&schema),
            vec![Some(1), None, Some(2), Some(2)],
        )];
        let reordered_batches = vec![
            int_batch(Arc::clone(&schema), vec![Some(2), Some(1)]),
            int_batch(Arc::clone(&schema), vec![None, Some(2)]),
        ];

        let hash = canonical_result_hash(&schema, &one_batch).unwrap();
        assert_eq!(hash.len(), 64);
        assert_eq!(
            hash,
            canonical_result_hash(&schema, &reordered_batches).unwrap()
        );
    }

    #[test]
    fn failed_query_discards_all_staged_evidence_before_commit() {
        let query_run: Result<Vec<_>> = vec![
            Ok(test_evidence(0, 'a')),
            Err(DataFusionError::Execution("iteration 1 failed".into())),
        ]
        .into_iter()
        .collect();
        let mut benchmark_run = BenchmarkRun::new();
        benchmark_run.start_new_case("Query 72");
        let mut committed_evidence = Vec::new();

        assert!(
            commit_query_evidence(
                &mut benchmark_run,
                &mut committed_evidence,
                query_run,
            )
            .is_err()
        );
        assert!(committed_evidence.is_empty());

        let benchmark: serde_json::Value =
            serde_json::from_str(&benchmark_run.to_json()).unwrap();
        assert_eq!(benchmark["queries"][0]["success"], false);
        assert_eq!(benchmark["queries"][0]["iterations"], serde_json::json!([]));

        let mut published = Vec::new();
        commit_benchmark_and_publish(
            &benchmark_run,
            None,
            &committed_evidence,
            &mut published,
        )
        .unwrap();
        assert!(published.is_empty());
    }

    #[test]
    fn successful_query_publishes_hashes_after_json_commit() {
        let mut benchmark_run = BenchmarkRun::new();
        benchmark_run.start_new_case("Query 72");
        let mut committed_evidence = Vec::new();
        commit_query_evidence(
            &mut benchmark_run,
            &mut committed_evidence,
            Ok(vec![test_evidence(0, 'a'), test_evidence(1, 'b')]),
        )
        .unwrap();

        let output_dir = tempfile::tempdir().unwrap();
        let json_path = output_dir.path().join("run.json");
        let mut publisher = JsonCommitObserver::new(json_path.clone());
        commit_benchmark_and_publish(
            &benchmark_run,
            Some(json_path.as_path()),
            &committed_evidence,
            &mut publisher,
        )
        .unwrap();

        assert!(publisher.saw_committed_json_before_output);
        let benchmark: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(json_path).unwrap()).unwrap();
        let lines = str::from_utf8(&publisher.output)
            .unwrap()
            .lines()
            .collect::<Vec<_>>();
        assert_eq!(lines.len(), committed_evidence.len());
        for (index, evidence) in committed_evidence.iter().enumerate() {
            assert_eq!(
                lines[index],
                format!(
                    "TPCDS_RESULT_HASH query={} iteration={} sha256={} rows={}",
                    evidence.query,
                    evidence.iteration,
                    evidence.result_hash,
                    evidence.row_count,
                )
            );
            assert_eq!(
                benchmark["queries"][0]["iterations"][index]["result_hash"],
                evidence.result_hash
            );
        }
    }

    #[test]
    fn result_hash_changes_for_values_nulls_duplicates_and_schema() {
        let schema = hash_schema("value", true);
        let baseline = canonical_result_hash(
            &schema,
            &[int_batch(Arc::clone(&schema), vec![Some(1), None, Some(2)])],
        )
        .unwrap();
        let changed_value = canonical_result_hash(
            &schema,
            &[int_batch(Arc::clone(&schema), vec![Some(1), None, Some(3)])],
        )
        .unwrap();
        let changed_null = canonical_result_hash(
            &schema,
            &[int_batch(
                Arc::clone(&schema),
                vec![Some(1), Some(0), Some(2)],
            )],
        )
        .unwrap();
        let changed_duplicate = canonical_result_hash(
            &schema,
            &[int_batch(
                Arc::clone(&schema),
                vec![Some(1), None, Some(2), Some(2)],
            )],
        )
        .unwrap();
        let schema_changed = hash_schema("different_value", true);
        let changed_schema = canonical_result_hash(
            &schema_changed,
            &[int_batch(
                Arc::clone(&schema_changed),
                vec![Some(1), None, Some(2)],
            )],
        )
        .unwrap();

        for changed in [
            changed_value,
            changed_null,
            changed_duplicate,
            changed_schema,
        ] {
            assert_ne!(baseline, changed);
        }
    }

    #[test]
    fn result_hash_normalizes_utf8_view_backing_layout_and_batches() {
        let logical_first = "the first logical string stored out of line";
        let logical_second = "the second logical string stored out of line";
        let mut compact = StringViewBuilder::new().with_fixed_block_size(64);
        compact.append_value(logical_first);
        compact.append_value(logical_second);
        let compact = Arc::new(compact.finish()) as ArrayRef;

        let mut first_backing = StringViewBuilder::new().with_fixed_block_size(128);
        first_backing.append_value("an unused prefix in the first backing buffer");
        first_backing.append_value(logical_first);
        let first_backing = Arc::new(first_backing.finish().slice(1, 1)) as ArrayRef;
        let mut second_backing = StringViewBuilder::new().with_fixed_block_size(256);
        second_backing.append_value("an unused prefix in another backing buffer");
        second_backing.append_value(logical_second);
        let second_backing = Arc::new(second_backing.finish().slice(1, 1)) as ArrayRef;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "view",
            DataType::Utf8View,
            false,
        )]));

        let compact_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![compact]).unwrap();
        let first_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![first_backing]).unwrap();
        let second_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![second_backing]).unwrap();
        assert_eq!(
            canonical_result_hash(&schema, &[compact_batch]).unwrap(),
            canonical_result_hash(&schema, &[second_batch, first_batch]).unwrap(),
        );
    }

    #[test]
    fn result_hash_hydrates_dictionary_values_and_ignores_unused_entries() {
        let mut compact = StringDictionaryBuilder::<Int32Type>::new();
        compact.append("same logical value").unwrap();
        let compact = Arc::new(compact.finish()) as ArrayRef;

        let mut expanded = StringDictionaryBuilder::<Int32Type>::new();
        expanded.append("unused dictionary value").unwrap();
        expanded.append("same logical value").unwrap();
        let expanded = Arc::new(expanded.finish().slice(1, 1)) as ArrayRef;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "dictionary",
            compact.data_type().clone(),
            false,
        )]));

        let compact_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![compact]).unwrap();
        let expanded_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![expanded]).unwrap();
        assert_eq!(
            canonical_result_hash(&schema, &[compact_batch]).unwrap(),
            canonical_result_hash(&schema, &[expanded_batch]).unwrap(),
        );
    }

    #[test]
    fn result_hash_rejects_invalid_map_shape_cleanly() {
        let entries = Field::new("entries", DataType::Int32, false);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "invalid_map",
            DataType::Map(Arc::new(entries), false),
            true,
        )]));
        let error = canonical_result_hash(&schema, &[]).unwrap_err();

        assert!(error.to_string().contains("expected struct field in map"));
    }

    fn test_evidence(iteration: usize, hash_byte: char) -> TpcdsIterationEvidence {
        TpcdsIterationEvidence {
            query: 72,
            iteration,
            elapsed: std::time::Duration::from_millis(iteration as u64 + 1),
            row_count: iteration + 10,
            result_hash: hash_byte.to_string().repeat(64),
        }
    }

    struct JsonCommitObserver {
        json_path: PathBuf,
        output: Vec<u8>,
        saw_committed_json_before_output: bool,
    }

    impl JsonCommitObserver {
        fn new(json_path: PathBuf) -> Self {
            Self {
                json_path,
                output: Vec::new(),
                saw_committed_json_before_output: false,
            }
        }
    }

    impl Write for JsonCommitObserver {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.output.is_empty() && !buf.is_empty() {
                self.saw_committed_json_before_output =
                    fs::read_to_string(&self.json_path)
                        .ok()
                        .and_then(|json| {
                            serde_json::from_str::<serde_json::Value>(&json).ok()
                        })
                        .is_some();
            }
            self.output.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn hash_schema(name: &str, nullable: bool) -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(
            name,
            DataType::Int32,
            nullable,
        )]))
    }

    fn int_batch(schema: SchemaRef, values: Vec<Option<i32>>) -> RecordBatch {
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values)) as ArrayRef])
            .unwrap()
    }

    #[test]
    fn derives_s3_object_store_url_from_tpcds_path() {
        let url = s3_object_store_url("s3://datafusion-bench/tpcds/sf10/parquet")
            .unwrap()
            .unwrap();

        assert_eq!(url.to_string(), "s3://datafusion-bench/");
    }

    #[test]
    fn ignores_local_path_for_s3_registration() {
        assert!(s3_object_store_url("/tmp/tpcds_sf10").unwrap().is_none());
    }

    #[test]
    fn parses_object_store_metrics_flag() {
        for enabled in ["true", "TRUE", "1", "yes", "YES"] {
            assert!(parse_bool_flag(enabled));
        }
        for disabled in ["false", "0", "no", "", "unexpected"] {
            assert!(!parse_bool_flag(disabled));
        }
    }

    #[test]
    fn parses_object_store_coalesce_gap() {
        assert_eq!(parse_object_store_coalesce_gap(None).unwrap(), None);
        assert_eq!(parse_object_store_coalesce_gap(Some("0")).unwrap(), Some(0));
        assert_eq!(
            parse_object_store_coalesce_gap(Some("65536")).unwrap(),
            Some(65536)
        );

        let error = parse_object_store_coalesce_gap(Some("invalid")).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("TPCDS_OBJECT_STORE_COALESCE_GAP_BYTES")
        );
    }
    #[test]
    fn parses_object_store_coalesce_parallelism() {
        assert_eq!(parse_object_store_coalesce_parallelism(None).unwrap(), None);
        assert_eq!(
            parse_object_store_coalesce_parallelism(Some("1")).unwrap(),
            Some(1)
        );
        assert_eq!(
            parse_object_store_coalesce_parallelism(Some("24")).unwrap(),
            Some(24)
        );

        for value in ["0", "25", "invalid"] {
            let error = parse_object_store_coalesce_parallelism(Some(value)).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("TPCDS_OBJECT_STORE_COALESCE_PARALLELISM")
            );
        }
    }
}
