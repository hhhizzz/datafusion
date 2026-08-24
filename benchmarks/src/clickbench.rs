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
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::util::{BenchmarkRun, CommonOpt, QueryResult, print_memory_stats};
use arrow::array::{Array, RecordBatch};
use arrow::util::display::array_value_to_string;
use clap::Args;
use datafusion::logical_expr::{ExplainFormat, ExplainOption};
use datafusion::{
    error::{DataFusionError, Result},
    prelude::SessionContext,
};
use datafusion_common::exec_datafusion_err;
use datafusion_common::instant::Instant;
use serde::Serialize;

/// SQL to create the hits view with proper EventDate casting.
///
/// ClickBench stores EventDate as UInt16 (days since 1970-01-01) for
/// storage efficiency (2 bytes vs 4-8 bytes for date types).
/// This view transforms it to SQL DATE type for query compatibility.
const HITS_VIEW_DDL: &str = r#"CREATE VIEW hits AS
SELECT * EXCEPT ("EventDate"),
       CAST(CAST("EventDate" AS INTEGER) AS DATE) AS "EventDate"
FROM hits_raw"#;

/// Driver program to run the ClickBench benchmark
///
/// The ClickBench[1] benchmarks are widely cited in the industry and
/// focus on grouping / aggregation / filtering. This runner uses the
/// scripts and queries from [2].
///
/// [1]: https://github.com/ClickHouse/ClickBench
/// [2]: https://github.com/ClickHouse/ClickBench/tree/main/datafusion
#[derive(Debug, Args, Clone)]
#[command(verbatim_doc_comment)]
pub struct RunOpt {
    /// Query number (between 0 and 42). If not specified, runs all queries
    #[arg(short, long)]
    pub query: Option<usize>,

    /// If specified, enables Parquet Filter Pushdown.
    ///
    /// Specifically, it enables:
    /// * `pushdown_filters = true`
    /// * `reorder_filters = true`
    #[arg(long = "pushdown")]
    pushdown: bool,

    /// Common options
    #[command(flatten)]
    common: CommonOpt,

    /// Path to hits.parquet (single file) or `hits_partitioned`
    /// (partitioned, 100 files)
    #[arg(
        short = 'p',
        long = "path",
        default_value = "benchmarks/data/hits.parquet"
    )]
    path: PathBuf,

    /// Path to queries directory
    #[arg(
        short = 'r',
        long = "queries-path",
        default_value = "benchmarks/queries/clickbench/queries"
    )]
    pub queries_path: PathBuf,

    /// If present, write results json here
    #[arg(short = 'o', long = "output")]
    output_path: Option<PathBuf>,

    /// Capture the first iteration's complete ordered result next to `--output`.
    /// This is a correctness sidecar and is not included in the measured time.
    #[arg(long, requires = "output_path")]
    capture_results: bool,

    /// Column name that the data is sorted by (e.g., "EventTime")
    /// If specified, DataFusion will be informed that the data has this sort order
    /// using CREATE EXTERNAL TABLE with WITH ORDER clause.
    ///
    /// Recommended to use with: -c datafusion.optimizer.prefer_existing_sort=true
    /// This allows DataFusion to optimize away redundant sorts while maintaining
    /// multi-core parallelism for other operations.
    #[arg(long = "sorted-by")]
    sorted_by: Option<String>,

    /// Sort order: ASC or DESC (default: ASC)
    #[arg(long = "sort-order", default_value = "ASC")]
    sort_order: String,

    /// Configuration options in the format key=value
    /// Can be specified multiple times.
    ///
    /// Example: -c datafusion.optimizer.prefer_existing_sort=true
    #[arg(short = 'c', long = "config")]
    config_options: Vec<String>,
}

/// Get the SQL file path
pub fn get_query_path(query_dir: &Path, query: usize) -> PathBuf {
    query_dir.join(format!("q{query}.sql"))
}

/// Get the SQL statement from the specified query file
pub fn get_query_sql(query_path: &Path) -> Result<Option<String>> {
    if fs::exists(query_path)? {
        Ok(Some(fs::read_to_string(query_path)?))
    } else {
        Ok(None)
    }
}

impl RunOpt {
    pub async fn run(self) -> Result<()> {
        println!("Running benchmarks with the following options: {self:?}");

        let query_dir_metadata = fs::metadata(&self.queries_path).map_err(|e| {
            if e.kind() == ErrorKind::NotFound {
                exec_datafusion_err!(
                    "Query path '{}' does not exist.",
                    &self.queries_path.to_str().unwrap()
                )
            } else {
                DataFusionError::External(Box::new(e))
            }
        })?;

        if !query_dir_metadata.is_dir() {
            return Err(exec_datafusion_err!(
                "Query path '{}' is not a directory.",
                &self.queries_path.to_str().unwrap()
            ));
        }

        let query_range = match self.query {
            Some(query_id) => query_id..=query_id,
            None => 0..=usize::MAX,
        };

        // configure parquet options
        let mut config = self.common.config()?;

        if self.sorted_by.is_some() {
            println!("ℹ️  Data is registered with sort order");

            let has_prefer_sort = self
                .config_options
                .iter()
                .any(|opt| opt.contains("prefer_existing_sort=true"));

            if !has_prefer_sort {
                println!(
                    "ℹ️  Consider using -c datafusion.optimizer.prefer_existing_sort=true"
                );
                println!("ℹ️  to optimize queries while maintaining parallelism");
            }
        }

        // Apply user-provided configuration options
        for config_opt in &self.config_options {
            let parts: Vec<&str> = config_opt.splitn(2, '=').collect();
            if parts.len() != 2 {
                return Err(exec_datafusion_err!(
                    "Invalid config option format: '{}'. Expected 'key=value'",
                    config_opt
                ));
            }
            let key = parts[0];
            let value = parts[1];

            println!("Setting config: {key} = {value}");
            config = config.set_str(key, value);
        }

        {
            let parquet_options = &mut config.options_mut().execution.parquet;
            // The hits_partitioned dataset specifies string columns
            // as binary due to how it was written. Force it to strings
            parquet_options.binary_as_string = true;

            // Turn on Parquet filter pushdown if requested
            if self.pushdown {
                parquet_options.pushdown_filters = true;
                parquet_options.reorder_filters = true;
            }

            if self.sorted_by.is_some() {
                // We should compare the dynamic topk optimization when data is sorted, so we make the
                // assumption that filter pushdown is also enabled in this case.
                parquet_options.pushdown_filters = true;
                parquet_options.reorder_filters = true;
            }
        }

        let rt = self.common.build_runtime()?;
        let ctx = SessionContext::new_with_config_rt(config, rt);

        self.register_hits(&ctx).await?;

        let mut benchmark_run = BenchmarkRun::new();
        benchmark_run.set_memory_pool(&ctx.runtime_env().memory_pool);
        for query_id in query_range {
            let query_path = get_query_path(&self.queries_path, query_id);
            let Some(sql) = get_query_sql(&query_path)? else {
                if self.query.is_some() {
                    return Err(exec_datafusion_err!(
                        "Could not load query file '{}'.",
                        &query_path.to_str().unwrap()
                    ));
                }
                break;
            };
            benchmark_run.start_new_case(&format!("Query {query_id}"));
            let query_run = self.benchmark_query(&sql, query_id, &ctx).await;
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
        sql: &str,
        query_id: usize,
        ctx: &SessionContext,
    ) -> Result<Vec<QueryResult>> {
        println!("Q{query_id}: {sql}");

        let mut millis = Vec::with_capacity(self.iterations());
        let mut query_results = vec![];
        for i in 0..self.iterations() {
            let start = Instant::now();
            let results = ctx.sql(sql).await?.collect().await?;
            let elapsed = start.elapsed();
            let ms = elapsed.as_secs_f64() * 1000.0;
            millis.push(ms);

            if i == 0 && self.capture_results {
                self.write_capture(query_id, &results)?;
            }

            let row_count: usize = results.iter().map(|b| b.num_rows()).sum();
            println!(
                "Query {query_id} iteration {i} took {ms:.1} ms and returned {row_count} rows"
            );
            query_results.push(QueryResult { elapsed, row_count })
        }
        if self.common.debug {
            ctx.sql(sql)
                .await?
                .explain_with_options(
                    ExplainOption::default().with_format(ExplainFormat::Tree),
                )?
                .show()
                .await?;
        }
        let avg = millis.iter().sum::<f64>() / millis.len() as f64;
        println!("Query {query_id} avg time: {avg:.2} ms");

        // Print memory usage stats using mimalloc (only when compiled with --features mimalloc_extended)
        print_memory_stats(&*ctx.runtime_env().memory_pool);

        Ok(query_results)
    }

    fn write_capture(&self, query_id: usize, results: &[RecordBatch]) -> Result<()> {
        let output_path = self
            .output_path
            .as_ref()
            .ok_or_else(|| exec_datafusion_err!("--capture-results requires --output"))?;
        let capture_path =
            output_path.with_file_name(format!("capture-q{query_id}.json"));
        write_capture(&capture_path, query_id, results)?;
        println!(
            "Captured Q{query_id} ordered result to {}",
            capture_path.display()
        );
        Ok(())
    }

    /// Registers the `hits.parquet` as a table named `hits`
    /// If sorted_by is specified, uses CREATE EXTERNAL TABLE with WITH ORDER
    async fn register_hits(&self, ctx: &SessionContext) -> Result<()> {
        let path = self.path.as_os_str().to_str().unwrap();

        // If sorted_by is specified, use CREATE EXTERNAL TABLE with WITH ORDER
        if let Some(ref sort_column) = self.sorted_by {
            println!(
                "Registering table with sort order: {} {}",
                sort_column, self.sort_order
            );

            // Escape column name with double quotes
            let escaped_column = if sort_column.contains('"') {
                sort_column.clone()
            } else {
                format!("\"{sort_column}\"")
            };

            // Build CREATE EXTERNAL TABLE DDL with WITH ORDER clause
            // Schema will be automatically inferred from the Parquet file
            let create_table_sql = format!(
                "CREATE EXTERNAL TABLE hits_raw \
                 STORED AS PARQUET \
                 LOCATION '{}' \
                 WITH ORDER ({} {})",
                path,
                escaped_column,
                self.sort_order.to_uppercase()
            );

            println!("Executing: {create_table_sql}");

            // Execute the CREATE EXTERNAL TABLE statement
            ctx.sql(&create_table_sql).await?.collect().await?;
        } else {
            // Original registration without sort order
            let options = Default::default();
            ctx.register_parquet("hits_raw", path, options)
                .await
                .map_err(|e| {
                    DataFusionError::Context(
                        format!("Registering 'hits_raw' as {path}"),
                        Box::new(e),
                    )
                })?;
        }

        // Create the hits view with EventDate transformation
        Self::create_hits_view(ctx).await
    }

    /// Creates the hits view with EventDate transformation from UInt16 to DATE.
    ///
    /// ClickBench encodes EventDate as UInt16 days since epoch (1970-01-01).
    async fn create_hits_view(ctx: &SessionContext) -> Result<()> {
        ctx.sql(HITS_VIEW_DDL).await?.collect().await.map_err(|e| {
            DataFusionError::Context(
                "Creating 'hits' view with EventDate transformation".to_string(),
                Box::new(e),
            )
        })?;
        Ok(())
    }

    fn iterations(&self) -> usize {
        self.common.iterations
    }
}

#[derive(Serialize)]
struct CapturedField<'a> {
    name: &'a str,
    data_type: String,
    nullable: bool,
}

#[derive(Serialize)]
struct CapturedQuery<'a> {
    query: usize,
    schema: Vec<CapturedField<'a>>,
    rows: Vec<Vec<Option<String>>>,
}

fn write_capture(path: &Path, query: usize, batches: &[RecordBatch]) -> Result<()> {
    let schema = batches
        .first()
        .map(RecordBatch::schema)
        .ok_or_else(|| exec_datafusion_err!("Q{query} produced no batches to capture"))?;

    let fields = schema
        .fields()
        .iter()
        .map(|field| CapturedField {
            name: field.name(),
            data_type: field.data_type().to_string(),
            nullable: field.is_nullable(),
        })
        .collect();
    let mut rows = Vec::new();
    for batch in batches {
        if batch.schema() != schema {
            return Err(exec_datafusion_err!(
                "Q{query} produced batches with inconsistent schemas"
            ));
        }
        for row_index in 0..batch.num_rows() {
            let mut row = Vec::with_capacity(batch.num_columns());
            for column in batch.columns() {
                let value = if column.is_null(row_index) {
                    None
                } else {
                    Some(array_value_to_string(column.as_ref(), row_index)?)
                };
                row.push(value);
            }
            rows.push(row);
        }
    }

    let capture = CapturedQuery {
        query,
        schema: fields,
        rows,
    };
    let mut encoded = serde_json::to_vec_pretty(&capture)
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
    encoded.push(b'\n');
    fs::write(path, encoded)?;
    fs::write(
        path.with_file_name("effective-cargo-lock.txt"),
        include_str!("../../Cargo.lock"),
    )?;
    Ok(())
}
