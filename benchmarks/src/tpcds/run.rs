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
use std::path::PathBuf;
use std::sync::Arc;

use super::q39_reuse::{self, Q39_REUSE_TABLE};
use crate::util::metrics_object_store::{MetricsObjectStore, ObjectStoreMetrics};
use crate::util::{BenchmarkRun, CommonOpt, QueryResult, print_memory_stats};

use arrow::record_batch::RecordBatch;
use arrow::util::pretty::{self, pretty_format_batches};
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
use datafusion_common::{DEFAULT_PARQUET_EXTENSION, DataFusionError, plan_err};
use object_store::OBJECT_STORE_COALESCE_DEFAULT;
use object_store::aws::AmazonS3Builder;
use url::Url;

use clap::Args;
use log::info;

// hack to avoid `default_value is meaningless for bool` errors
type BoolDefaultTrue = bool;
pub const TPCDS_QUERY_START_ID: usize = 1;
pub const TPCDS_QUERY_END_ID: usize = 99;
const OBJECT_STORE_COALESCE_GAP_ENV: &str = "TPCDS_OBJECT_STORE_COALESCE_GAP_BYTES";
const Q39_REUSE_CONTROL_ENV: &str = "TPCDS_Q39_REUSE_CONTROL";

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
            match query_run {
                Ok(query_results) => {
                    for iter in query_results {
                        benchmark_run.write_iter(iter.elapsed, iter.row_count);
                    }
                }
                Err(e) => {
                    benchmark_run.mark_failed();
                    eprintln!("Query {query_id} failed: {e}");
                }
            }
        }
        benchmark_run.maybe_write_json(self.output_path.as_ref())?;
        benchmark_run.maybe_print_failures();
        Ok(())
    }

    async fn benchmark_query(
        &self,
        query_id: usize,
        ctx: &SessionContext,
        object_store_metrics: Option<&ObjectStoreMetrics>,
    ) -> Result<Vec<QueryResult>> {
        let mut millis = vec![];
        // run benchmark
        let mut query_results = vec![];

        let sql = &get_query_sql(self.query_path.to_str().unwrap(), query_id)?;

        if self.common.debug {
            println!("=== SQL for query {query_id} ===\n{}\n", sql.join(";\n"));
        }

        let q39_reuse_control = q39_reuse_control_enabled(
            query_id,
            std::env::var(Q39_REUSE_CONTROL_ENV).ok().as_deref(),
        );

        for i in 0..self.iterations() {
            if let Some(metrics) = object_store_metrics {
                metrics.reset();
            }
            let start = Instant::now();

            // query 15 is special, with 3 statements. the second statement is the one from which we
            // want to capture the results
            let result = if q39_reuse_control {
                self.execute_q39_reuse_consumers(ctx, i, &q39_reuse::consumer_sql())
                    .await?
            } else {
                let mut result = vec![];

                for query in sql {
                    result = self.execute_query(ctx, query).await?;
                }

                result
            };

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
            info!("output:\n\n{}\n\n", pretty_format_batches(&result)?);
            let row_count = result.iter().map(|b| b.num_rows()).sum();
            println!(
                "Query {query_id} iteration {i} took {ms:.1} ms and returned {row_count} rows"
            );
            query_results.push(QueryResult { elapsed, row_count });
        }

        let avg = millis.iter().sum::<f64>() / millis.len() as f64;
        println!("Query {query_id} avg time: {avg:.2} ms");

        // Print memory stats using mimalloc (only when compiled with --features mimalloc_extended)
        print_memory_stats();

        Ok(query_results)
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
    ) -> Result<Vec<RecordBatch>> {
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
        Ok(result)
    }

    async fn execute_q39_reuse_consumers(
        &self,
        ctx: &SessionContext,
        iteration: usize,
        consumers: &[String],
    ) -> Result<Vec<RecordBatch>> {
        let materialized = q39_reuse::materialize(ctx).await?;
        ctx.register_table(Q39_REUSE_TABLE, materialized.table)?;
        println!(
            "TPCDS_Q39_REUSE_CONTROL iteration={iteration} rows={} batches={} bytes={}",
            materialized.stats.rows,
            materialized.stats.batches,
            materialized.stats.estimated_bytes,
        );

        let execution = async {
            let mut result = vec![];
            for query in consumers {
                result = self.execute_query(ctx, query).await?;
            }
            Ok(result)
        }
        .await;
        let cleanup = ctx.deregister_table(Q39_REUSE_TABLE).map(|_| ());

        match (execution, cleanup) {
            (Ok(result), Ok(())) => Ok(result),
            (Err(execution), Ok(())) => Err(execution),
            (Ok(_), Err(cleanup)) => Err(cleanup),
            (Err(execution), Err(cleanup)) => {
                Err(DataFusionError::Collection(vec![execution, cleanup]))
            }
        }
    }

    async fn get_table(
        &self,
        ctx: &SessionContext,
        table: &str,
    ) -> Result<Arc<dyn TableProvider>> {
        let path = self.path.to_str().unwrap();
        let target_partitions = self.partitions();

        // Obtain a snapshot of the SessionState
        let state = ctx.state();
        let path = format!("{path}/{table}.parquet");

        // Check if the file exists
        if !std::path::Path::new(&path).exists() {
            eprintln!("Warning registering {table}: Table file does not exist: {path}");
        }

        let format = ParquetFormat::default()
            .with_options(ctx.state().table_options().parquet.clone());

        let table_path = ListingTableUrl::parse(path)?;
        let options = ListingOptions::new(Arc::new(format))
            .with_file_extension(DEFAULT_PARQUET_EXTENSION)
            .with_target_partitions(target_partitions)
            .with_collect_stat(state.config().collect_statistics());
        let schema = options.infer_schema(&state, &table_path).await?;

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

        Ok(Arc::new(ListingTable::try_new(config)?.with_cache(
            ctx.runtime_env().cache_manager.get_file_statistic_cache(),
        )))
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

    if metrics_enabled {
        let effective_gap = coalesce_gap.unwrap_or(OBJECT_STORE_COALESCE_DEFAULT);
        let store = match coalesce_gap {
            Some(gap) => MetricsObjectStore::new_with_coalesce_gap(store, gap),
            None => MetricsObjectStore::new(store),
        };
        let metrics = store.metrics();
        ctx.register_object_store(object_store_url_ref, Arc::new(store));
        println!(
            "Registered instrumented S3 object store for {object_store_url} \
             coalesce_gap_bytes={effective_gap}"
        );
        Ok(Some(metrics))
    } else if let Some(coalesce_gap) = coalesce_gap {
        let store = MetricsObjectStore::new_coalescing(store, coalesce_gap);
        ctx.register_object_store(object_store_url_ref, Arc::new(store));
        println!(
            "Registered coalescing S3 object store for {object_store_url} \
             coalesce_gap_bytes={coalesce_gap} metrics=false"
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

fn q39_reuse_control_enabled(query_id: usize, value: Option<&str>) -> bool {
    query_id == 39 && value == Some("true")
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
    fn enables_q39_reuse_control_only_for_query_39_with_true_flag() {
        assert!(q39_reuse_control_enabled(39, Some("true")));

        for value in [None, Some("false"), Some("TRUE"), Some("1"), Some("yes")] {
            assert!(!q39_reuse_control_enabled(39, value));
        }

        assert!(!q39_reuse_control_enabled(38, Some("true")));
        assert!(!q39_reuse_control_enabled(40, Some("true")));
    }

    #[tokio::test]
    async fn q39_reuse_consumer_failure_deregisters_temporary_table() {
        let ctx = SessionContext::new();
        register_q39_fixture(&ctx).await.unwrap();

        let result = q39_reuse_runner()
            .execute_q39_reuse_consumers(
                &ctx,
                0,
                &["SELECT * FROM missing_q39_reuse_input".to_string()],
            )
            .await;

        assert!(result.is_err());
        assert!(ctx.table(Q39_REUSE_TABLE).await.is_err());
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

    fn q39_reuse_runner() -> RunOpt {
        RunOpt {
            query: Some(39),
            common: CommonOpt {
                iterations: 1,
                partitions: Some(1),
                batch_size: None,
                mem_pool_type: "fair".to_string(),
                memory_limit: None,
                sort_spill_reservation_bytes: None,
                debug: false,
                simulate_latency: false,
            },
            path: PathBuf::new(),
            query_path: PathBuf::new(),
            mem_table: false,
            output_path: None,
            disable_statistics: false,
            prefer_hash_join: true,
            enable_piecewise_merge_join: false,
            sorted: false,
            hash_join_buffering_capacity: 0,
        }
    }

    async fn register_q39_fixture(ctx: &SessionContext) -> Result<()> {
        for sql in [
            "CREATE TABLE inventory (\
             inv_item_sk BIGINT, inv_warehouse_sk BIGINT, inv_date_sk BIGINT, \
             inv_quantity_on_hand BIGINT) AS VALUES \
             (10, 1, 1, 0), (10, 1, 2, 0), (10, 1, 3, 3), \
             (10, 1, 4, 0), (10, 1, 5, 0), (10, 1, 6, 3), \
             (20, 1, 7, 0), (20, 1, 8, 2), \
             (20, 1, 9, 0), (20, 1, 10, 2)",
            "CREATE TABLE item (i_item_sk BIGINT) AS VALUES (10), (20)",
            "CREATE TABLE warehouse (w_warehouse_sk BIGINT, w_warehouse_name VARCHAR) \
             AS VALUES (1, 'warehouse 1')",
            "CREATE TABLE date_dim (d_date_sk BIGINT, d_year BIGINT, d_moy BIGINT) \
             AS VALUES \
             (1, 1998, 4), (2, 1998, 4), (3, 1998, 4), \
             (4, 1998, 5), (5, 1998, 5), (6, 1998, 5), \
             (7, 1998, 4), (8, 1998, 4), (9, 1998, 5), (10, 1998, 5)",
        ] {
            ctx.sql(sql).await?.collect().await?;
        }

        Ok(())
    }
}
