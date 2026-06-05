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

//! ParquetSource implementation for reading parquet files
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use crate::DefaultParquetFileReaderFactory;
use crate::ParquetFileReaderFactory;
use crate::opener::ParquetMorselizer;
use crate::opener::build_pruning_predicates;
use crate::row_filter::can_expr_be_pushed_down_with_schemas;
use datafusion_common::config::ConfigOptions;
#[cfg(feature = "parquet_encryption")]
use datafusion_common::config::EncryptionFactoryOptions;
use datafusion_datasource::as_file_source;
use datafusion_datasource::file_stream::FileOpener;
use datafusion_datasource::morsel::Morselizer;

use arrow::array::timezone::Tz;
use arrow::datatypes::TimeUnit;
use datafusion_common::DataFusionError;
use datafusion_common::config::TableParquetOptions;
use datafusion_datasource::TableSchema;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr::{EquivalenceProperties, conjunction};
use datafusion_physical_expr_adapter::DefaultPhysicalExprAdapterFactory;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
use datafusion_physical_expr_common::physical_expr::fmt_sql;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::SortOrderPushdownResult;
use datafusion_physical_plan::filter_pushdown::PushedDown;
use datafusion_physical_plan::filter_pushdown::{
    FilterPushdownPropagation, PushedDownPredicate,
};
use datafusion_physical_plan::metrics::Count;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use log::warn;

#[cfg(feature = "parquet_encryption")]
use datafusion_execution::parquet_encryption::EncryptionFactory;
use datafusion_physical_expr_common::sort_expr::{LexOrdering, PhysicalSortExpr};
use itertools::Itertools;
use object_store::ObjectStore;
#[cfg(feature = "parquet_encryption")]
use parquet::encryption::decrypt::FileDecryptionProperties;

const ROW_FILTER_DISABLE_LABELS_ENV: &str =
    "DATAFUSION_PARQUET_ROW_FILTER_DISABLE_LABELS";
const ROW_FILTER_ADMISSION_RULES_ENV: &str =
    "DATAFUSION_PARQUET_ROW_FILTER_ADMISSION_RULES";

/// Execution plan for reading one or more Parquet files.
///
/// ```text
///             ▲
///             │
///             │  Produce a stream of
///             │  RecordBatches
///             │
/// ┌───────────────────────┐
/// │                       │
/// │     DataSourceExec    │
/// │                       │
/// └───────────────────────┘
///             ▲
///             │  Asynchronously read from one
///             │  or more parquet files via
///             │  ObjectStore interface
///             │
///             │
///   .───────────────────.
///  │                     )
///  │`───────────────────'│
///  │    ObjectStore      │
///  │.───────────────────.│
///  │                     )
///   `───────────────────'
/// ```
///
/// # Example: Create a `DataSourceExec`
/// ```
/// # use std::sync::Arc;
/// # use arrow::datatypes::Schema;
/// # use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
/// # use datafusion_datasource_parquet::source::ParquetSource;
/// # use datafusion_datasource::PartitionedFile;
/// # use datafusion_execution::object_store::ObjectStoreUrl;
/// # use datafusion_physical_expr::expressions::lit;
/// # use datafusion_datasource::source::DataSourceExec;
/// # use datafusion_common::config::TableParquetOptions;
///
/// # let file_schema = Arc::new(Schema::empty());
/// # let object_store_url = ObjectStoreUrl::local_filesystem();
/// # let predicate = lit(true);
/// let source = Arc::new(
///     ParquetSource::new(Arc::clone(&file_schema))
///         .with_predicate(predicate)
/// );
/// // Create a DataSourceExec for reading `file1.parquet` with a file size of 100MB
/// let config = FileScanConfigBuilder::new(object_store_url, source)
///    .with_file(PartitionedFile::new("file1.parquet", 100*1024*1024)).build();
/// let exec = DataSourceExec::from_data_source(config);
/// ```
///
/// # Features
///
/// Supports the following optimizations:
///
/// * Concurrent reads: reads from one or more files in parallel as multiple
///   partitions, including concurrently reading multiple row groups from a single
///   file.
///
/// * Predicate push down: skips row groups, pages, rows based on metadata
///   and late materialization. See "Predicate Pushdown" below.
///
/// * Projection pushdown: reads and decodes only the columns required.
///
/// * Limit pushdown: stop execution early after some number of rows are read.
///
/// * Custom readers: customize reading  parquet files, e.g. to cache metadata,
///   coalesce I/O operations, etc. See [`ParquetFileReaderFactory`] for more
///   details.
///
/// * Schema evolution: read parquet files with different schemas into a unified
///   table schema. See [`DefaultPhysicalExprAdapterFactory`] for more details.
///
/// * metadata_size_hint: controls the number of bytes read from the end of the
///   file in the initial I/O when the default [`ParquetFileReaderFactory`]. If a
///   custom reader is used, it supplies the metadata directly and this parameter
///   is ignored. [`ParquetSource::with_metadata_size_hint`] for more details.
///
/// * User provided  `ParquetAccessPlan`s to skip row groups and/or pages
///   based on external information. See "Implementing External Indexes" below
///
/// # Predicate Pushdown
///
/// `DataSourceExec` uses the provided [`PhysicalExpr`] predicate as a filter to
/// skip reading unnecessary data and improve query performance using several techniques:
///
/// * Row group pruning: skips entire row groups based on min/max statistics
///   found in [`ParquetMetaData`] and any Bloom filters that are present.
///
/// * Page pruning: skips individual pages within a ColumnChunk using the
///   [Parquet PageIndex], if present.
///
/// * Row filtering: skips rows within a page using a form of late
///   materialization. When possible, predicates are applied by the parquet
///   decoder *during* decode (see [`ArrowPredicate`] and [`RowFilter`] for more
///   details). This is only enabled if `ParquetScanOptions::pushdown_filters` is set to true.
///
/// Note: If the predicate can not be used to accelerate the scan, it is ignored
/// (no error is raised on predicate evaluation errors).
///
/// [`ArrowPredicate`]: parquet::arrow::arrow_reader::ArrowPredicate
/// [`RowFilter`]: parquet::arrow::arrow_reader::RowFilter
/// [Parquet PageIndex]: https://github.com/apache/parquet-format/blob/master/PageIndex.md
///
/// # Example: rewriting `DataSourceExec`
///
/// You can modify a `DataSourceExec` using [`ParquetSource`], for example
/// to change files or add a predicate.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use arrow::datatypes::Schema;
/// # use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
/// # use datafusion_datasource::PartitionedFile;
/// # use datafusion_datasource::source::DataSourceExec;
///
/// # fn parquet_exec() -> DataSourceExec { unimplemented!() }
/// // Split a single DataSourceExec into multiple DataSourceExecs, one for each file
/// let exec = parquet_exec();
/// let data_source = exec.data_source();
/// let base_config = data_source.downcast_ref::<FileScanConfig>().unwrap();
/// let existing_file_groups = &base_config.file_groups;
/// let new_execs = existing_file_groups
///   .iter()
///   .map(|file_group| {
///     // create a new exec by copying the existing exec's source config
///     let new_config = FileScanConfigBuilder::from(base_config.clone())
///        .with_file_groups(vec![file_group.clone()])
///       .build();
///
///     (DataSourceExec::from_data_source(new_config))
///   })
///   .collect::<Vec<_>>();
/// ```
///
/// # Implementing External Indexes
///
/// It is possible to restrict the row groups and selections within those row
/// groups that the DataSourceExec will consider by providing an initial
/// `ParquetAccessPlan` as `extensions` on `PartitionedFile`. This can be
/// used to implement external indexes on top of parquet files and select only
/// portions of the files.
///
/// The `DataSourceExec` will try and reduce any provided `ParquetAccessPlan`
/// further based on the contents of `ParquetMetadata` and other settings.
///
/// ## Example of providing a ParquetAccessPlan
///
/// ```
/// # use std::sync::Arc;
/// # use arrow::datatypes::{Schema, SchemaRef};
/// # use datafusion_datasource::PartitionedFile;
/// # use datafusion_datasource_parquet::ParquetAccessPlan;
/// # use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
/// # use datafusion_datasource_parquet::source::ParquetSource;
/// # use datafusion_execution::object_store::ObjectStoreUrl;
/// # use datafusion_datasource::source::DataSourceExec;
///
/// # fn schema() -> SchemaRef {
/// #   Arc::new(Schema::empty())
/// # }
/// // create an access plan to scan row group 0, 1 and 3 and skip row groups 2 and 4
/// let mut access_plan = ParquetAccessPlan::new_all(5);
/// access_plan.skip(2);
/// access_plan.skip(4);
/// // provide the plan as extension to the FileScanConfig
/// let partitioned_file = PartitionedFile::new("my_file.parquet", 1234)
///   .with_extension(access_plan);
/// // create a FileScanConfig to scan this file
/// let config = FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), Arc::new(ParquetSource::new(schema())))
///     .with_file(partitioned_file).build();
/// // this parquet DataSourceExec will not even try to read row groups 2 and 4. Additional
/// // pruning based on predicates may also happen
/// let exec = DataSourceExec::from_data_source(config);
/// ```
///
/// For a complete example, see the [`advanced_parquet_index` example]).
///
/// [`parquet_index_advanced` example]: https://github.com/apache/datafusion/blob/main/datafusion-examples/examples/data_io/parquet_advanced_index.rs
///
/// # Execution Overview
///
/// * Step 1: `DataSourceExec::execute` is called, returning a `FileStream`
///   configured to morselize parquet files with a `ParquetMorselizer`.
///
/// * Step 2: When the stream is polled, the `ParquetMorselizer` is called to
///   plan the file.
///
/// * Step 3: The `ParquetMorselizer` gets the [`ParquetMetaData`] (file metadata)
///   via [`ParquetFileReaderFactory`], creating a `ParquetAccessPlan` by
///   applying predicates to metadata. The plan and projections are used to
///   determine what pages must be read.
///
/// * Step 4: The stream begins reading data, fetching the required parquet
///   pages incrementally decoding them, and applying any row filters (see
///   [`Self::with_pushdown_filters`]).
///
/// * Step 5: As each [`RecordBatch`] is read, it may be adapted by a
///   [`DefaultPhysicalExprAdapterFactory`] to match the table schema. By default missing columns are
///   filled with nulls, but this can be customized via [`PhysicalExprAdapterFactory`].
///
/// [`RecordBatch`]: arrow::record_batch::RecordBatch
/// [`ParquetMetadata`]: parquet::file::metadata::ParquetMetaData
/// [`PhysicalExprAdapterFactory`]: datafusion_physical_expr_adapter::PhysicalExprAdapterFactory
#[derive(Clone, Debug)]
pub struct ParquetSource {
    /// Options for reading Parquet files
    pub(crate) table_parquet_options: TableParquetOptions,
    /// Optional metrics
    pub(crate) metrics: ExecutionPlanMetricsSet,
    /// The schema of the file.
    /// In particular, this is the schema of the table without partition columns,
    /// *not* the physical schema of the file.
    pub(crate) table_schema: TableSchema,
    /// Optional predicate for row filtering during parquet scan
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    /// Optional user defined parquet file reader factory
    pub(crate) parquet_file_reader_factory: Option<Arc<dyn ParquetFileReaderFactory>>,
    /// Batch size configuration
    pub(crate) batch_size: Option<usize>,
    /// Optional hint for the size of the parquet metadata
    pub(crate) metadata_size_hint: Option<usize>,
    /// Projection to apply to the output.
    pub(crate) projection: ProjectionExprs,
    #[cfg(feature = "parquet_encryption")]
    pub(crate) encryption_factory: Option<Arc<dyn EncryptionFactory>>,
    /// If true, the opener flips row-group iteration order. Within-
    /// row-group order is on-disk order, so the scan is `Inexact` and
    /// a `SortExec` is kept in the plan.
    reverse_row_groups: bool,
    /// Sort order driving `PreparedAccessPlan::reorder_by_statistics`
    /// in the opener.
    sort_order_for_reorder: Option<LexOrdering>,
    /// Diagnostic label used by benchmark oracle runs to disable row-level
    /// filtering for a selected scan while preserving metadata pruning.
    diagnostic_file_label: Option<Arc<str>>,
}

impl ParquetSource {
    /// Create a new ParquetSource to read the data specified in the file scan
    /// configuration with the provided schema.
    ///
    /// Uses default `TableParquetOptions`.
    /// To set custom options, use [ParquetSource::with_table_parquet_options`].
    pub fn new(table_schema: impl Into<TableSchema>) -> Self {
        let table_schema = table_schema.into();
        // Projection over the full table schema (file columns + partition columns)
        let full_schema = table_schema.table_schema();
        let indices: Vec<usize> = (0..full_schema.fields().len()).collect();
        Self {
            projection: ProjectionExprs::from_indices(&indices, full_schema),
            table_schema,
            table_parquet_options: TableParquetOptions::default(),
            metrics: ExecutionPlanMetricsSet::new(),
            predicate: None,
            parquet_file_reader_factory: None,
            batch_size: None,
            metadata_size_hint: None,
            #[cfg(feature = "parquet_encryption")]
            encryption_factory: None,
            reverse_row_groups: false,
            sort_order_for_reorder: None,
            diagnostic_file_label: None,
        }
    }

    /// Set the `TableParquetOptions` for this ParquetSource.
    pub fn with_table_parquet_options(
        mut self,
        table_parquet_options: TableParquetOptions,
    ) -> Self {
        self.table_parquet_options = table_parquet_options;
        self
    }

    /// Set the metadata size hint
    ///
    /// This value determines how many bytes at the end of the file the default
    /// [`ParquetFileReaderFactory`] will request in the initial IO. If this is
    /// too small, the ParquetSource will need to make additional IO requests to
    /// read the footer.
    pub fn with_metadata_size_hint(mut self, metadata_size_hint: usize) -> Self {
        self.metadata_size_hint = Some(metadata_size_hint);
        self
    }

    pub(crate) fn with_diagnostic_file_label(
        mut self,
        diagnostic_file_label: Option<Arc<str>>,
    ) -> Self {
        self.diagnostic_file_label = diagnostic_file_label;
        self
    }

    /// Set predicate information
    #[expect(clippy::needless_pass_by_value)]
    pub fn with_predicate(&self, predicate: Arc<dyn PhysicalExpr>) -> Self {
        let mut conf = self.clone();
        conf.predicate = Some(Arc::clone(&predicate));
        conf
    }

    /// Set the encryption factory to use to generate file decryption properties
    #[cfg(feature = "parquet_encryption")]
    pub fn with_encryption_factory(
        mut self,
        encryption_factory: Arc<dyn EncryptionFactory>,
    ) -> Self {
        self.encryption_factory = Some(encryption_factory);
        self
    }

    /// Options passed to the parquet reader for this scan
    pub fn table_parquet_options(&self) -> &TableParquetOptions {
        &self.table_parquet_options
    }

    /// Optional predicate.
    #[deprecated(since = "50.2.0", note = "use `filter` instead")]
    pub fn predicate(&self) -> Option<&Arc<dyn PhysicalExpr>> {
        self.predicate.as_ref()
    }

    /// return the optional file reader factory
    pub fn parquet_file_reader_factory(
        &self,
    ) -> Option<&Arc<dyn ParquetFileReaderFactory>> {
        self.parquet_file_reader_factory.as_ref()
    }

    /// Optional user defined parquet file reader factory.
    pub fn with_parquet_file_reader_factory(
        mut self,
        parquet_file_reader_factory: Arc<dyn ParquetFileReaderFactory>,
    ) -> Self {
        self.parquet_file_reader_factory = Some(parquet_file_reader_factory);
        self
    }

    /// If true, the predicate will be used during the parquet scan.
    /// Defaults to false.
    pub fn with_pushdown_filters(mut self, pushdown_filters: bool) -> Self {
        self.table_parquet_options.global.pushdown_filters = pushdown_filters;
        self
    }

    /// Return the value described in [`Self::with_pushdown_filters`]
    pub(crate) fn pushdown_filters(&self) -> bool {
        self.table_parquet_options.global.pushdown_filters
    }

    fn row_filter_can_skip_projected_columns(
        &self,
        filter: &Arc<dyn PhysicalExpr>,
    ) -> bool {
        if DynamicFilterTracking::classify(filter).contains_dynamic_filter() {
            return true;
        }

        let projected_columns = self.projection.column_indices();
        if projected_columns.is_empty() {
            return true;
        }

        let filter_columns = collect_columns(filter);
        projected_columns
            .iter()
            .any(|index| !filter_columns.iter().any(|column| column.index() == *index))
    }

    fn row_filter_set_can_skip_projected_columns(
        &self,
        filters: &[Arc<dyn PhysicalExpr>],
    ) -> bool {
        let projected_columns = self.projection.column_indices();
        if projected_columns.is_empty()
            || filters.iter().any(|filter| {
                DynamicFilterTracking::classify(filter).contains_dynamic_filter()
            })
        {
            return true;
        }

        let filter_columns = filters
            .iter()
            .flat_map(collect_columns)
            .map(|column| column.index())
            .collect_vec();

        projected_columns.iter().any(|index| {
            !filter_columns
                .iter()
                .any(|filter_index| filter_index == index)
        })
    }

    /// If true, the `RowFilter` made by `pushdown_filters` may try to
    /// minimize the cost of filter evaluation by reordering the
    /// predicate [`Expr`]s. If false, the predicates are applied in
    /// the same order as specified in the query. Defaults to false.
    ///
    /// [`Expr`]: datafusion_expr::Expr
    pub fn with_reorder_filters(mut self, reorder_filters: bool) -> Self {
        self.table_parquet_options.global.reorder_filters = reorder_filters;
        self
    }

    /// Return the value described in [`Self::with_reorder_filters`]
    fn reorder_filters(&self) -> bool {
        self.table_parquet_options.global.reorder_filters
    }

    /// Return the value of [`datafusion_common::config::ParquetOptions::force_filter_selections`]
    fn force_filter_selections(&self) -> bool {
        self.table_parquet_options.global.force_filter_selections
    }

    /// If enabled, the reader will read the page index
    /// This is used to optimize filter pushdown
    /// via `RowSelector` and `RowFilter` by
    /// eliminating unnecessary IO and decoding
    pub fn with_enable_page_index(mut self, enable_page_index: bool) -> Self {
        self.table_parquet_options.global.enable_page_index = enable_page_index;
        self
    }

    /// Return the value described in [`Self::with_enable_page_index`]
    fn enable_page_index(&self) -> bool {
        self.table_parquet_options.global.enable_page_index
    }

    /// If enabled, the reader will read by the bloom filter
    pub fn with_bloom_filter_on_read(mut self, bloom_filter_on_read: bool) -> Self {
        self.table_parquet_options.global.bloom_filter_on_read = bloom_filter_on_read;
        self
    }

    /// If enabled, the writer will write by the bloom filter
    pub fn with_bloom_filter_on_write(
        mut self,
        enable_bloom_filter_on_write: bool,
    ) -> Self {
        self.table_parquet_options.global.bloom_filter_on_write =
            enable_bloom_filter_on_write;
        self
    }

    /// Return the value described in [`Self::with_bloom_filter_on_read`]
    fn bloom_filter_on_read(&self) -> bool {
        self.table_parquet_options.global.bloom_filter_on_read
    }

    /// Return the maximum predicate cache size, in bytes, used when
    /// `pushdown_filters`
    pub fn max_predicate_cache_size(&self) -> Option<usize> {
        self.table_parquet_options.global.max_predicate_cache_size
    }

    #[cfg(feature = "parquet_encryption")]
    fn get_encryption_factory_with_config(
        &self,
    ) -> Option<(Arc<dyn EncryptionFactory>, EncryptionFactoryOptions)> {
        match &self.encryption_factory {
            None => None,
            Some(factory) => Some((
                Arc::clone(factory),
                self.table_parquet_options.crypto.factory_options.clone(),
            )),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_reverse_row_groups(mut self, reverse_row_groups: bool) -> Self {
        self.reverse_row_groups = reverse_row_groups;
        self
    }
    #[cfg(test)]
    pub(crate) fn reverse_row_groups(&self) -> bool {
        self.reverse_row_groups
    }
}

/// Parses datafusion.common.config.ParquetOptions.coerce_int96 String to a arrow_schema.datatype.TimeUnit
pub(crate) fn parse_coerce_int96_string(
    str_setting: &str,
) -> datafusion_common::Result<TimeUnit> {
    let str_setting_lower: &str = &str_setting.to_lowercase();

    match str_setting_lower {
        "ns" => Ok(TimeUnit::Nanosecond),
        "us" => Ok(TimeUnit::Microsecond),
        "ms" => Ok(TimeUnit::Millisecond),
        "s" => Ok(TimeUnit::Second),
        _ => Err(DataFusionError::Configuration(format!(
            "Unknown or unsupported parquet coerce_int96: \
        {str_setting}. Valid values are: ns, us, ms, and s."
        ))),
    }
}

/// Validates that `tz` is a parseable IANA timezone and returns it as an
/// `Arc<str>` for use in `Timestamp(_, Some(tz))` types.
pub(crate) fn parse_coerce_int96_tz_string(
    tz: &str,
) -> datafusion_common::Result<Arc<str>> {
    tz.parse::<Tz>().map_err(|e| {
        DataFusionError::Configuration(format!(
            "Invalid parquet coerce_int96_tz {tz:?}: {e}"
        ))
    })?;
    Ok(Arc::<str>::from(tz))
}

/// Allows easy conversion from ParquetSource to Arc&lt;dyn FileSource&gt;
impl From<ParquetSource> for Arc<dyn FileSource> {
    fn from(source: ParquetSource) -> Self {
        as_file_source(source)
    }
}

impl FileSource for ParquetSource {
    fn create_file_opener(
        &self,
        _object_store: Arc<dyn ObjectStore>,
        _base_config: &FileScanConfig,
        _partition: usize,
    ) -> datafusion_common::Result<Arc<dyn FileOpener>> {
        datafusion_common::internal_err!(
            "ParquetSource::create_file_opener called but it supports the Morsel API, please use that instead"
        )
    }

    fn create_morselizer(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> datafusion_common::Result<Box<dyn Morselizer>> {
        let expr_adapter_factory = base_config
            .expr_adapter_factory
            .clone()
            .unwrap_or_else(|| Arc::new(DefaultPhysicalExprAdapterFactory) as _);

        let parquet_file_reader_factory =
            self.parquet_file_reader_factory.clone().unwrap_or_else(|| {
                Arc::new(DefaultParquetFileReaderFactory::new(object_store)) as _
            });

        #[cfg(feature = "parquet_encryption")]
        let file_decryption_properties = self
            .table_parquet_options()
            .crypto
            .file_decryption
            .clone()
            .map(FileDecryptionProperties::try_from)
            .transpose()?
            .map(Arc::new);

        let coerce_int96 = self
            .table_parquet_options
            .global
            .coerce_int96
            .as_ref()
            .map(|time_unit| parse_coerce_int96_string(time_unit.as_str()).unwrap());
        let coerce_int96_tz = self
            .table_parquet_options
            .global
            .coerce_int96_tz
            .as_ref()
            .map(|tz| parse_coerce_int96_tz_string(tz))
            .transpose()?;
        if coerce_int96_tz.is_some() && coerce_int96.is_none() {
            warn!(
                "coerce_int96_tz is set but coerce_int96 is not; the timezone will be ignored"
            );
        }

        Ok(Box::new(ParquetMorselizer {
            partition_index: partition,
            projection: self.projection.clone(),
            batch_size: self
                .batch_size
                .expect("Batch size must set before creating ParquetMorselizer"),
            limit: base_config.limit,
            preserve_order: base_config.preserve_order,
            predicate: self.predicate.clone(),
            table_schema: self.table_schema.clone(),
            metadata_size_hint: self.metadata_size_hint,
            metrics: self.metrics().clone(),
            parquet_file_reader_factory,
            pushdown_filters: self.pushdown_filters(),
            reorder_filters: self.reorder_filters(),
            force_filter_selections: self.force_filter_selections(),
            enable_page_index: self.enable_page_index(),
            enable_bloom_filter: self.bloom_filter_on_read(),
            enable_row_group_stats_pruning: self.table_parquet_options.global.pruning,
            coerce_int96,
            coerce_int96_tz,
            #[cfg(feature = "parquet_encryption")]
            file_decryption_properties,
            expr_adapter_factory,
            #[cfg(feature = "parquet_encryption")]
            encryption_factory: self.get_encryption_factory_with_config(),
            max_predicate_cache_size: self.max_predicate_cache_size(),
            reverse_row_groups: self.reverse_row_groups,
            sort_order_for_reorder: self.sort_order_for_reorder.clone(),
        }))
    }

    fn reorder_files(
        &self,
        files: Vec<datafusion_datasource::PartitionedFile>,
    ) -> Vec<datafusion_datasource::PartitionedFile> {
        crate::sort::reorder_files_by_min_statistics(
            files,
            self.sort_order_for_reorder.as_ref(),
            self.reverse_row_groups,
            self.table_schema.table_schema(),
        )
    }

    fn table_schema(&self) -> &TableSchema {
        &self.table_schema
    }

    fn filter(&self) -> Option<Arc<dyn PhysicalExpr>> {
        self.predicate.clone()
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        let mut conf = self.clone();
        conf.batch_size = Some(batch_size);
        Arc::new(conf)
    }

    fn try_pushdown_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> datafusion_common::Result<Option<Arc<dyn FileSource>>> {
        let mut source = self.clone();
        source.projection = self.projection.try_merge(projection)?;
        Ok(Some(Arc::new(source)))
    }

    fn projection(&self) -> Option<&ProjectionExprs> {
        Some(&self.projection)
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    fn file_type(&self) -> &str {
        "parquet"
    }

    fn fmt_extra(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let predicate_string = self
                    .filter()
                    .map(|p| format!(", predicate={p}"))
                    .unwrap_or_default();

                write!(f, "{predicate_string}")?;

                // Inexact sort-pushdown markers: surface both flags so
                // readers can see the optimization fired.
                if let Some(sort_order) = &self.sort_order_for_reorder {
                    write!(f, ", sort_order_for_reorder=[{sort_order}]")?;
                }
                if self.reverse_row_groups {
                    write!(f, ", reverse_row_groups=true")?;
                }

                // Try to build the pruning predicates.
                // These are only generated here because it's useful to have *some*
                // idea of what pushdown is happening when viewing plans.
                // However, it is important to note that these predicates are *not*
                // necessarily the predicates that are actually evaluated:
                // the actual predicates are built in reference to the physical schema of
                // each file, which we do not have at this point and hence cannot use.
                // Instead, we use the logical schema of the file (the table schema without partition columns).
                if let Some(predicate) = &self.predicate {
                    let predicate_creation_errors = Count::new();
                    if let Some(pruning_predicate) = build_pruning_predicates(
                        Some(predicate),
                        self.table_schema.table_schema(),
                        &predicate_creation_errors,
                    ) {
                        let mut guarantees = pruning_predicate
                            .literal_guarantees()
                            .iter()
                            .map(|item| format!("{item}"))
                            .collect_vec();
                        guarantees.sort();
                        write!(
                            f,
                            ", pruning_predicate={}, required_guarantees=[{}]",
                            pruning_predicate.predicate_expr(),
                            guarantees.join(", ")
                        )?;
                    }
                };
                Ok(())
            }
            DisplayFormatType::TreeRender => {
                if let Some(predicate) = self.filter() {
                    writeln!(f, "predicate={}", fmt_sql(predicate.as_ref()))?;
                }
                Ok(())
            }
        }
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        config: &ConfigOptions,
    ) -> datafusion_common::Result<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        let table_schema = self.table_schema.table_schema();
        // Determine if based on configs we should push filters down.
        // If either the table / scan itself or the config has pushdown enabled,
        // we will push down the filters.
        // If both are disabled, we will not push down the filters.
        // By default they are both disabled.
        // Regardless of pushdown, we will update the predicate to include the filters
        // because even if scan pushdown is disabled we can still use the filters for stats pruning.
        let config_pushdown_enabled = config.execution.parquet.pushdown_filters;
        let table_pushdown_enabled = self.pushdown_filters();
        let pushdown_filters = table_pushdown_enabled || config_pushdown_enabled;

        let mut source = self.clone();
        let row_filter_set_can_skip_columns =
            self.row_filter_set_can_skip_projected_columns(&filters);
        let pushdown_filter_predicates: Vec<(PushedDownPredicate, bool)> = filters
            .into_iter()
            .map(|filter| {
                if can_expr_be_pushed_down_with_schemas(&filter, table_schema) {
                    let row_filter_can_skip_columns = row_filter_set_can_skip_columns
                        && self.row_filter_can_skip_projected_columns(&filter);
                    (
                        if row_filter_can_skip_columns {
                            PushedDownPredicate::supported(filter)
                        } else {
                            PushedDownPredicate::unsupported(filter)
                        },
                        true,
                    )
                } else {
                    (PushedDownPredicate::unsupported(filter), false)
                }
            })
            .collect();
        if pushdown_filter_predicates
            .iter()
            .all(|(_, can_prune)| !can_prune)
        {
            // No filters can be pushed down, so we can just return the remaining filters
            // and avoid replacing the source in the physical plan.
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![PushedDown::No; pushdown_filter_predicates.len()],
            ));
        }
        let has_pruning_only_filters =
            pushdown_filter_predicates
                .iter()
                .any(|(filter, can_prune)| {
                    *can_prune && matches!(filter.discriminant, PushedDown::No)
                });
        let row_filter_disabled_for_oracle =
            row_filter_disabled_by_label_spec(self.diagnostic_file_label.as_deref());
        let admission_mode = row_filter_admission_mode(
            self.diagnostic_file_label.as_deref(),
            &pushdown_filter_predicates,
            table_schema,
            &self.projection,
            row_filter_set_can_skip_columns,
            pushdown_filters,
            has_pruning_only_filters,
            row_filter_disabled_for_oracle,
        );

        if matches!(admission_mode, RowFilterAdmissionMode::ResidualOnly) {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![PushedDown::No; pushdown_filter_predicates.len()],
            ));
        }

        let allowed_filters = pushdown_filter_predicates
            .iter()
            .filter_map(|(filter, can_prune)| {
                (*can_prune).then(|| Arc::clone(&filter.predicate))
            })
            .collect_vec();
        let predicate = match source.predicate {
            Some(predicate) => {
                conjunction(std::iter::once(predicate).chain(allowed_filters))
            }
            None => conjunction(allowed_filters),
        };
        source.predicate = Some(predicate);
        let pushdown_filters =
            matches!(admission_mode, RowFilterAdmissionMode::RowFilter);
        source = source.with_pushdown_filters(pushdown_filters);
        let source = Arc::new(source);
        // If pushdown_filters is false we tell our parents that they still have to handle the filters,
        // even if we updated the predicate to include the filters (they will only be used for stats pruning).
        if !pushdown_filters {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![PushedDown::No; pushdown_filter_predicates.len()],
            )
            .with_updated_node(source));
        }
        Ok(FilterPushdownPropagation::with_parent_pushdown_result(
            pushdown_filter_predicates
                .iter()
                .map(|(filter, _)| filter.discriminant)
                .collect(),
        )
        .with_updated_node(source))
    }

    /// Try to optimize the scan to produce data in the requested sort order.
    ///
    /// Inputs:
    /// 1. The query's required ordering (`order` parameter)
    /// 2. The source's equivalence properties (`eq_properties`)
    ///
    /// # Returns
    /// - `Exact`: the source's natural ordering already satisfies the
    ///   request. The surrounding `SortExec` can be eliminated provided
    ///   files within each group are non-overlapping (verified by
    ///   `FileScanConfig`).
    /// - `Inexact`: the source can approximate the request via two
    ///   composable runtime steps — stats-based row-group reorder
    ///   (skipped when the leading sort key isn't a plain `Column`
    ///   in the file schema) and row-group iteration reverse. A
    ///   `SortExec` is still required for full correctness, but limit
    ///   pushdown and `TopK` benefit immediately.
    /// - `Unsupported`: no approximation is available.
    ///
    /// # How the Inexact result is communicated
    ///
    /// The result is carried through two fields on `ParquetSource`:
    ///
    /// - `sort_order_for_reorder`: set to the request's `LexOrdering`
    ///   whenever the pushdown fires, regardless of whether the
    ///   leading expression is a plain `Column`. The opener invokes
    ///   `PreparedAccessPlan::reorder_by_statistics`, which skips
    ///   when the expression can't be looked up in parquet metadata.
    ///   Exposing the field unconditionally keeps `EXPLAIN` honest
    ///   about what the source was asked to approximate.
    /// - `reverse_row_groups`: drives the opener's iteration flip.
    ///   When stats reorder applies (column-in-schema), this is just
    ///   the request's direction — the reorder produces ASC-by-min,
    ///   so reverse iff the query asks for DESC. When stats reorder
    ///   doesn't apply but the reversed source ordering satisfies
    ///   the request (function-wrapped case), this is always `true`
    ///   because we're flipping the file's natural order.
    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
        eq_properties: &EquivalenceProperties,
    ) -> datafusion_common::Result<SortOrderPushdownResult<Arc<dyn FileSource>>> {
        if order.is_empty() {
            return Ok(SortOrderPushdownResult::Unsupported);
        }

        // Check if the natural (non-reversed) ordering already satisfies the request.
        // Parquet metadata guarantees within-file ordering, so if the ordering matches
        // we can return Exact. FileScanConfig will verify that files within each group
        // are non-overlapping before declaring the entire scan as Exact.
        if eq_properties.ordering_satisfy(order.iter().cloned())? {
            return Ok(SortOrderPushdownResult::Exact {
                inner: Arc::new(self.clone()) as Arc<dyn FileSource>,
            });
        }

        // If the source's declared ordering is a non-empty *proper* prefix
        // of the request (e.g. source `[a DESC, b ASC]`, request
        // `[a DESC, b ASC, c DESC]`), decline pushdown so the outer
        // `SortExec`'s `sort_prefix` optimisation — prefix-aware early
        // termination in `TopK` — can still fire. Firing the Inexact
        // pipeline below would invalidate the source's `output_ordering`
        // (the runtime row-group reorder is approximate, so we can't
        // honour the declared ordering anymore), which is exactly what
        // `EnforceSorting` needs to derive `sort_prefix`. On data that
        // is already in prefix order the stats-based reorder is mostly
        // a no-op anyway, so the trade-off is plainly bad.
        for prefix_len in 1..order.len() {
            let prefix = order[..prefix_len].to_vec();
            if eq_properties.ordering_satisfy(prefix.iter().cloned())? {
                return Ok(SortOrderPushdownResult::Unsupported);
            }
        }

        // Inexact pushdown. Two independent signals; either is enough
        // to produce an approximate ordering, and they compose when
        // both apply:
        //
        // 1. `column_in_file_schema`: the request's leading sort key is
        //    a plain `Column` present in the file schema. The opener
        //    can sort row groups by that column's `min` via parquet
        //    statistics. Drives `sort_order_for_reorder`'s actual use.
        //
        // 2. `reversed_satisfies`: the source's declared ordering,
        //    when reversed, satisfies the request. This is strictly
        //    more powerful than the column-in-schema check because it
        //    runs the request through `EquivalenceProperties`'s full
        //    reasoning machinery:
        //
        //    - Function monotonicity: e.g. file declares
        //      `[extract_year_month(ws) DESC, ws DESC]`, request is
        //      `[ws ASC]`; the reversed ordering still satisfies the
        //      request via `extract_year_month`'s monotonicity even
        //      though parquet has no stats keyed by the function
        //      expression itself.
        //    - Constant columns from filters: equivalence classes can
        //      mark columns as constant under a predicate, allowing
        //      requested orderings on those columns to be trivially
        //      satisfied.
        //    - Other equivalence relationships (e.g. `a = b` transfers
        //      ordering between `a` and `b`).
        //
        //    `reorder_by_statistics` can't substitute for any of the
        //    above because it can only look up min/max for a plain
        //    physical column.
        //
        // `sort_order_for_reorder` is set in both cases so EXPLAIN
        // shows what the source was asked to approximate; the opener
        // skips stats-based reorder when the leading expression isn't
        // a plain `Column`.
        //
        // The reversal flips each `PhysicalSortExpr` (both descending
        // and nulls_first) and rebuilds an `EquivalenceProperties` so
        // the request can be tested against the reversed orderings
        // via the same `ordering_satisfy` API.
        let reversed_eq_properties = {
            let mut new = eq_properties.clone();
            new.clear_orderings();
            let reversed_orderings = eq_properties
                .oeq_class()
                .iter()
                .map(|ordering| {
                    ordering
                        .iter()
                        .map(|expr| expr.reverse())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            new.add_orderings(reversed_orderings);
            new
        };
        let reversed_satisfies =
            reversed_eq_properties.ordering_satisfy(order.iter().cloned())?;
        let sort_order = LexOrdering::new(order.iter().cloned());
        let column_in_file_schema = sort_order.as_ref().is_some_and(|s| {
            s.first()
                .expr
                .downcast_ref::<datafusion_physical_expr::expressions::Column>()
                .is_some_and(|col| {
                    self.table_schema
                        .file_schema()
                        .field_with_name(col.name())
                        .is_ok()
                })
        });

        if !column_in_file_schema && !reversed_satisfies {
            return Ok(SortOrderPushdownResult::Unsupported);
        }

        // `reverse_row_groups` has different starting points in the
        // two cases:
        // - With stats reorder (column-in-schema): the reorder produces
        //   ASC-by-min, so reverse iff the request is DESC.
        // - Without stats reorder (reversed-eq fallback): we flip the
        //   file's natural order, so always reverse.
        let is_descending = sort_order
            .as_ref()
            .is_some_and(|s| s.first().options.descending);
        let mut new_source = self.clone();
        new_source.sort_order_for_reorder = sort_order;
        new_source.reverse_row_groups = if column_in_file_schema {
            is_descending
        } else {
            true
        };
        Ok(SortOrderPushdownResult::Inexact {
            inner: Arc::new(new_source) as Arc<dyn FileSource>,
        })
    }
}

fn row_filter_disabled_by_label_spec(label: Option<&str>) -> bool {
    std::env::var(ROW_FILTER_DISABLE_LABELS_ENV)
        .ok()
        .is_some_and(|spec| row_filter_disable_spec_matches(label, &spec))
}

fn row_filter_disable_spec_matches(label: Option<&str>, spec: &str) -> bool {
    let Some(label) = label else {
        return false;
    };
    let label = label.trim();
    if label.is_empty() {
        return false;
    }

    spec.split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .any(|token| {
            let token = token.strip_suffix(".parquet").unwrap_or(token);
            let label_name = label
                .rsplit('/')
                .next()
                .map(|name| name.strip_suffix(".parquet").unwrap_or(name));
            token == "*"
                || token.eq_ignore_ascii_case("all")
                || token.eq_ignore_ascii_case(label)
                || label_name.is_some_and(|name| token.eq_ignore_ascii_case(name))
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowFilterAdmissionMode {
    RowFilter,
    MetadataPruneOnly,
    ResidualOnly,
}

fn row_filter_admission_mode(
    label: Option<&str>,
    predicates: &[(PushedDownPredicate, bool)],
    table_schema: &arrow::datatypes::Schema,
    projection: &ProjectionExprs,
    row_filter_set_can_skip_projected_columns: bool,
    pushdown_filters_enabled: bool,
    has_pruning_only_filters: bool,
    row_filter_disabled_for_oracle: bool,
) -> RowFilterAdmissionMode {
    let default_mode = if pushdown_filters_enabled
        && !has_pruning_only_filters
        && !row_filter_disabled_for_oracle
    {
        RowFilterAdmissionMode::RowFilter
    } else {
        RowFilterAdmissionMode::MetadataPruneOnly
    };

    let Some(requested_mode) = row_filter_admission_rule_mode_from_env(
        label,
        predicates,
        table_schema,
        projection,
        row_filter_set_can_skip_projected_columns,
    ) else {
        return default_mode;
    };

    match requested_mode {
        RowFilterAdmissionMode::RowFilter
            if pushdown_filters_enabled && !has_pruning_only_filters =>
        {
            RowFilterAdmissionMode::RowFilter
        }
        RowFilterAdmissionMode::RowFilter => RowFilterAdmissionMode::MetadataPruneOnly,
        other => other,
    }
}

fn row_filter_admission_rule_mode_from_env(
    label: Option<&str>,
    predicates: &[(PushedDownPredicate, bool)],
    table_schema: &arrow::datatypes::Schema,
    projection: &ProjectionExprs,
    row_filter_set_can_skip_projected_columns: bool,
) -> Option<RowFilterAdmissionMode> {
    std::env::var(ROW_FILTER_ADMISSION_RULES_ENV)
        .ok()
        .and_then(|spec| {
            row_filter_admission_rule_mode_from_spec(
                label,
                predicates,
                table_schema,
                projection,
                row_filter_set_can_skip_projected_columns,
                &spec,
            )
        })
}

fn row_filter_admission_rule_mode_from_spec(
    label: Option<&str>,
    predicates: &[(PushedDownPredicate, bool)],
    table_schema: &arrow::datatypes::Schema,
    projection: &ProjectionExprs,
    row_filter_set_can_skip_projected_columns: bool,
    spec: &str,
) -> Option<RowFilterAdmissionMode> {
    let context = RowFilterAdmissionContext::new(
        label,
        predicates,
        table_schema,
        projection,
        row_filter_set_can_skip_projected_columns,
    );

    spec.split(';')
        .map(str::trim)
        .filter(|rule| !rule.is_empty())
        .find_map(|rule| {
            let (mode, conditions) = rule.split_once(':').unwrap_or((rule, ""));
            let mode = row_filter_admission_mode_from_token(mode.trim())?;
            row_filter_admission_conditions_match(&context, conditions).then_some(mode)
        })
}

fn row_filter_admission_mode_from_token(token: &str) -> Option<RowFilterAdmissionMode> {
    match token.trim().to_ascii_lowercase().as_str() {
        "row_filter" | "rowfilter" | "row" | "enable" | "enabled" => {
            Some(RowFilterAdmissionMode::RowFilter)
        }
        "metadata_prune_only" | "metadata" | "prune_only" | "pruning_only" => {
            Some(RowFilterAdmissionMode::MetadataPruneOnly)
        }
        "residual_only" | "residual" | "parent_only" | "none" => {
            Some(RowFilterAdmissionMode::ResidualOnly)
        }
        _ => None,
    }
}

#[derive(Debug)]
struct RowFilterAdmissionContext<'a> {
    label: Option<&'a str>,
    contains_dynamic_filter: bool,
    row_filter_set_can_skip_projected_columns: bool,
    filter_columns: BTreeSet<String>,
    projected_columns: BTreeSet<String>,
}

impl<'a> RowFilterAdmissionContext<'a> {
    fn new(
        label: Option<&'a str>,
        predicates: &[(PushedDownPredicate, bool)],
        table_schema: &arrow::datatypes::Schema,
        projection: &ProjectionExprs,
        row_filter_set_can_skip_projected_columns: bool,
    ) -> Self {
        let contains_dynamic_filter = predicates.iter().any(|(predicate, _)| {
            DynamicFilterTracking::classify(&predicate.predicate)
                .contains_dynamic_filter()
        });
        let filter_columns = predicates
            .iter()
            .flat_map(|(predicate, _)| collect_columns(&predicate.predicate))
            .map(|column| column.name().to_owned())
            .collect();
        let projected_columns = projection
            .column_indices()
            .into_iter()
            .filter_map(|index| table_schema.fields().get(index))
            .map(|field| field.name().to_owned())
            .collect();

        Self {
            label,
            contains_dynamic_filter,
            row_filter_set_can_skip_projected_columns,
            filter_columns,
            projected_columns,
        }
    }
}

fn row_filter_admission_conditions_match(
    context: &RowFilterAdmissionContext<'_>,
    conditions: &str,
) -> bool {
    conditions
        .split(',')
        .map(str::trim)
        .filter(|condition| !condition.is_empty())
        .all(|condition| row_filter_admission_condition_matches(context, condition))
}

fn row_filter_admission_condition_matches(
    context: &RowFilterAdmissionContext<'_>,
    condition: &str,
) -> bool {
    match condition.to_ascii_lowercase().as_str() {
        "all" | "*" => true,
        "dynamic" | "has_dynamic" => context.contains_dynamic_filter,
        "static" | "no_dynamic" => !context.contains_dynamic_filter,
        "skip_projected" | "can_skip_projected" => {
            context.row_filter_set_can_skip_projected_columns
        }
        "covers_projection" | "cannot_skip_projected" => {
            !context.row_filter_set_can_skip_projected_columns
        }
        _ => {
            let Some((key, value)) = condition.split_once('=') else {
                return false;
            };
            row_filter_admission_key_value_matches(context, key.trim(), value.trim())
        }
    }
}

fn row_filter_admission_key_value_matches(
    context: &RowFilterAdmissionContext<'_>,
    key: &str,
    value: &str,
) -> bool {
    match key.to_ascii_lowercase().as_str() {
        "label" | "table" | "file" => {
            row_filter_disable_spec_matches(context.label, value)
        }
        "filter_col" | "filter_column" => {
            set_contains_ignore_ascii_case(&context.filter_columns, value)
        }
        "projected_col" | "projected_column" | "output_col" | "output_column" => {
            set_contains_ignore_ascii_case(&context.projected_columns, value)
        }
        _ => false,
    }
}

fn set_contains_ignore_ascii_case(values: &BTreeSet<String>, target: &str) -> bool {
    values
        .iter()
        .any(|value| value.eq_ignore_ascii_case(target))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{
        BinaryExpr, Column, DynamicFilterPhysicalExpr, lit,
    };

    #[test]
    #[expect(deprecated)]
    fn test_parquet_source_predicate_same_as_filter() {
        let predicate = lit(true);

        let parquet_source =
            ParquetSource::new(Arc::new(Schema::empty())).with_predicate(predicate);
        // same value. but filter() call Arc::clone internally
        assert_eq!(parquet_source.predicate(), parquet_source.filter().as_ref());
    }

    fn clickbench_search_phrase_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("SearchPhrase", DataType::Utf8, true),
            Field::new("URL", DataType::Utf8, true),
        ]))
    }

    fn search_phrase_not_empty_filter() -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new("SearchPhrase", 0)),
            Operator::NotEq,
            lit(""),
        ))
    }

    fn dynamic_search_phrase_filter() -> Arc<dyn PhysicalExpr> {
        let column = Arc::new(Column::new("SearchPhrase", 0)) as Arc<dyn PhysicalExpr>;
        Arc::new(DynamicFilterPhysicalExpr::new(vec![column], lit(true)))
    }

    fn url_not_empty_filter() -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new("URL", 1)),
            Operator::NotEq,
            lit(""),
        ))
    }

    #[test]
    fn row_filter_disable_spec_matches_label_tokens() {
        assert!(row_filter_disable_spec_matches(
            Some("store_sales"),
            "inventory,store_sales"
        ));
        assert!(row_filter_disable_spec_matches(
            Some("tpcds_sf10/store_sales.parquet"),
            "store_sales.parquet"
        ));
        assert!(row_filter_disable_spec_matches(
            Some("store_sales"),
            "store_sales.parquet"
        ));
        assert!(row_filter_disable_spec_matches(Some("catalog_sales"), "*"));
        assert!(row_filter_disable_spec_matches(
            Some("catalog_sales"),
            "all"
        ));
        assert!(!row_filter_disable_spec_matches(
            Some("store_sales"),
            "web_sales"
        ));
        assert!(!row_filter_disable_spec_matches(None, "store_sales"));
    }

    fn supported_predicate(
        predicate: Arc<dyn PhysicalExpr>,
    ) -> (PushedDownPredicate, bool) {
        (PushedDownPredicate::supported(predicate), true)
    }

    #[test]
    fn row_filter_admission_rules_match_scan_shape() {
        let schema = clickbench_search_phrase_schema();
        let projection = ProjectionExprs::from_indices(&[0, 1], &schema);
        let predicates = vec![supported_predicate(dynamic_search_phrase_filter())];

        let mode = row_filter_admission_rule_mode_from_spec(
            Some("catalog_sales"),
            &predicates,
            &schema,
            &projection,
            true,
            "metadata_prune_only:label=catalog_sales,dynamic,filter_col=SearchPhrase,projected_col=URL",
        );
        assert_eq!(mode, Some(RowFilterAdmissionMode::MetadataPruneOnly));

        let mode = row_filter_admission_rule_mode_from_spec(
            Some("catalog_sales"),
            &predicates,
            &schema,
            &projection,
            true,
            "metadata_prune_only:label=web_sales,dynamic;residual_only:label=catalog_sales,dynamic",
        );
        assert_eq!(mode, Some(RowFilterAdmissionMode::ResidualOnly));

        let mode = row_filter_admission_rule_mode_from_spec(
            Some("catalog_sales"),
            &predicates,
            &schema,
            &projection,
            true,
            "metadata_prune_only:label=catalog_sales,static",
        );
        assert_eq!(mode, None);
    }

    #[test]
    fn row_filter_admission_rules_match_projection_skip_shape() {
        let schema = clickbench_search_phrase_schema();
        let projection = ProjectionExprs::from_indices(&[0], &schema);
        let predicates = vec![supported_predicate(url_not_empty_filter())];

        let mode = row_filter_admission_rule_mode_from_spec(
            Some("clickbench/hits.parquet"),
            &predicates,
            &schema,
            &projection,
            true,
            "row_filter:label=hits,static,skip_projected,filter_col=URL",
        );
        assert_eq!(mode, Some(RowFilterAdmissionMode::RowFilter));

        let mode = row_filter_admission_rule_mode_from_spec(
            Some("clickbench/hits.parquet"),
            &predicates,
            &schema,
            &projection,
            false,
            "metadata_prune_only:label=hits,covers_projection,filter_col=URL",
        );
        assert_eq!(mode, Some(RowFilterAdmissionMode::MetadataPruneOnly));
    }

    #[test]
    fn try_pushdown_filters_keeps_parent_filter_when_no_payload_columns_can_be_skipped() {
        let schema = clickbench_search_phrase_schema();
        let source = ParquetSource::new(Arc::clone(&schema));
        let projection = ProjectionExprs::from_indices(&[0], &schema);
        let source = source
            .try_pushdown_projection(&projection)
            .unwrap()
            .unwrap();
        let source = source
            .downcast_ref::<ParquetSource>()
            .expect("projected source is ParquetSource");

        let mut config = ConfigOptions::default();
        config.execution.parquet.pushdown_filters = true;

        let result = source
            .try_pushdown_filters(vec![search_phrase_not_empty_filter()], &config)
            .unwrap();

        assert_eq!(result.filters.len(), 1);
        assert!(matches!(result.filters[0], PushedDown::No));
        let updated = result
            .updated_node
            .expect("predicate is still used for pruning");
        let updated = updated
            .downcast_ref::<ParquetSource>()
            .expect("updated source is ParquetSource");
        assert!(updated.filter().is_some());
        assert!(!updated.pushdown_filters());
    }

    #[test]
    fn try_pushdown_filters_keeps_dynamic_filter_on_row_filter_path() {
        let schema = clickbench_search_phrase_schema();
        let source = ParquetSource::new(Arc::clone(&schema));
        let projection = ProjectionExprs::from_indices(&[0], &schema);
        let source = source
            .try_pushdown_projection(&projection)
            .unwrap()
            .unwrap();
        let source = source
            .downcast_ref::<ParquetSource>()
            .expect("projected source is ParquetSource");

        let mut config = ConfigOptions::default();
        config.execution.parquet.pushdown_filters = true;

        let result = source
            .try_pushdown_filters(vec![dynamic_search_phrase_filter()], &config)
            .unwrap();

        assert_eq!(result.filters.len(), 1);
        assert!(matches!(result.filters[0], PushedDown::Yes));
        let updated = result.updated_node.expect("source should be updated");
        let updated = updated
            .downcast_ref::<ParquetSource>()
            .expect("updated source is ParquetSource");
        assert!(updated.filter().is_some());
        assert!(updated.pushdown_filters());
    }

    #[test]
    fn try_pushdown_filters_pushes_when_payload_columns_can_be_skipped() {
        let schema = clickbench_search_phrase_schema();
        let source = ParquetSource::new(Arc::clone(&schema));
        let projection = ProjectionExprs::from_indices(&[0, 1], &schema);
        let source = source
            .try_pushdown_projection(&projection)
            .unwrap()
            .unwrap();
        let source = source
            .downcast_ref::<ParquetSource>()
            .expect("projected source is ParquetSource");

        let mut config = ConfigOptions::default();
        config.execution.parquet.pushdown_filters = true;

        let result = source
            .try_pushdown_filters(vec![search_phrase_not_empty_filter()], &config)
            .unwrap();

        assert_eq!(result.filters.len(), 1);
        assert!(matches!(result.filters[0], PushedDown::Yes));
        let updated = result.updated_node.expect("source should be updated");
        let updated = updated
            .downcast_ref::<ParquetSource>()
            .expect("updated source is ParquetSource");
        assert!(updated.filter().is_some());
        assert!(updated.pushdown_filters());
    }

    #[test]
    fn try_pushdown_filters_keeps_parent_filter_when_filter_set_covers_projection() {
        let schema = clickbench_search_phrase_schema();
        let source = ParquetSource::new(Arc::clone(&schema));
        let projection = ProjectionExprs::from_indices(&[0, 1], &schema);
        let source = source
            .try_pushdown_projection(&projection)
            .unwrap()
            .unwrap();
        let source = source
            .downcast_ref::<ParquetSource>()
            .expect("projected source is ParquetSource");

        let mut config = ConfigOptions::default();
        config.execution.parquet.pushdown_filters = true;

        let result = source
            .try_pushdown_filters(
                vec![search_phrase_not_empty_filter(), url_not_empty_filter()],
                &config,
            )
            .unwrap();

        assert_eq!(result.filters.len(), 2);
        assert!(
            result
                .filters
                .iter()
                .all(|filter| matches!(filter, PushedDown::No))
        );
        let updated = result
            .updated_node
            .expect("predicate is still used for pruning");
        let updated = updated
            .downcast_ref::<ParquetSource>()
            .expect("updated source is ParquetSource");
        assert!(updated.filter().is_some());
        assert!(!updated.pushdown_filters());
    }

    #[test]
    fn test_reverse_scan_default_value() {
        use arrow::datatypes::Schema;

        let schema = Arc::new(Schema::empty());
        let source = ParquetSource::new(schema);

        assert!(!source.reverse_row_groups());
    }

    #[test]
    fn test_reverse_scan_with_setter() {
        use arrow::datatypes::Schema;

        let schema = Arc::new(Schema::empty());

        let source = ParquetSource::new(schema.clone()).with_reverse_row_groups(true);
        assert!(source.reverse_row_groups());

        let source = source.with_reverse_row_groups(false);
        assert!(!source.reverse_row_groups());
    }

    #[test]
    fn test_reverse_scan_clone_preserves_value() {
        use arrow::datatypes::Schema;

        let schema = Arc::new(Schema::empty());

        let source = ParquetSource::new(schema).with_reverse_row_groups(true);
        let cloned = source.clone();

        assert!(cloned.reverse_row_groups());
        assert_eq!(source.reverse_row_groups(), cloned.reverse_row_groups());
    }

    #[test]
    fn test_reverse_scan_with_other_options() {
        use arrow::datatypes::Schema;

        let schema = Arc::new(Schema::empty());
        let options = TableParquetOptions::default();

        let source = ParquetSource::new(schema)
            .with_table_parquet_options(options)
            .with_metadata_size_hint(8192)
            .with_reverse_row_groups(true);

        assert!(source.reverse_row_groups());
        assert_eq!(source.metadata_size_hint, Some(8192));
    }

    #[test]
    fn test_reverse_scan_builder_pattern() {
        use arrow::datatypes::Schema;

        let schema = Arc::new(Schema::empty());

        let source = ParquetSource::new(schema)
            .with_reverse_row_groups(true)
            .with_reverse_row_groups(false)
            .with_reverse_row_groups(true);

        assert!(source.reverse_row_groups());
    }

    #[test]
    fn test_reverse_scan_independent_of_predicate() {
        use arrow::datatypes::Schema;
        use datafusion_physical_expr::expressions::lit;

        let schema = Arc::new(Schema::empty());
        let predicate = lit(true);

        let source = ParquetSource::new(schema)
            .with_predicate(predicate)
            .with_reverse_row_groups(true);

        assert!(source.reverse_row_groups());
        assert!(source.filter().is_some());
    }

    /// Helpers for the `try_pushdown_sort` regression tests below.
    mod pushdown_sort_helpers {
        use super::*;
        use arrow::compute::SortOptions;
        use arrow::datatypes::{DataType, Field, Schema};
        use datafusion_physical_expr::expressions::Column;
        use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;

        pub(super) fn schema_with_a_int() -> Arc<Schema> {
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]))
        }

        pub(super) fn sort_expr_on(
            schema: &Arc<Schema>,
            name: &str,
            descending: bool,
        ) -> PhysicalSortExpr {
            let idx = schema.index_of(name).unwrap();
            PhysicalSortExpr {
                expr: Arc::new(Column::new(name, idx)),
                options: SortOptions {
                    descending,
                    nulls_first: true,
                },
            }
        }
    }

    /// When neither natural nor reversed ordering matches the request,
    /// but the sort column is a plain `Column` present in the file
    /// schema, `try_pushdown_sort` returns `Inexact` with
    /// `sort_order_for_reorder` set so the opener can reorder row
    /// groups by min/max statistics at runtime.
    #[test]
    fn try_pushdown_sort_returns_inexact_when_column_in_schema_asc() {
        use datafusion_physical_expr::EquivalenceProperties;
        use pushdown_sort_helpers::*;

        let schema = schema_with_a_int();
        let source = ParquetSource::new(Arc::clone(&schema));
        let order = vec![sort_expr_on(&schema, "a", false)];
        // No declared natural ordering on the source.
        let eq = EquivalenceProperties::new(Arc::clone(&schema));

        let result = source.try_pushdown_sort(&order, &eq).unwrap();

        let SortOrderPushdownResult::Inexact { inner } = result else {
            panic!("expected Inexact, got a different variant");
        };
        // Downcast back to `ParquetSource` to inspect the fields the
        // opener reads to drive `reorder_by_statistics` / `reverse`.
        let inner_parquet = inner
            .downcast_ref::<ParquetSource>()
            .expect("inner is ParquetSource");
        let sort_order = inner_parquet
            .sort_order_for_reorder
            .as_ref()
            .expect("sort_order_for_reorder must be set so the opener can reorder");
        assert_eq!(sort_order.first().expr.to_string(), "a@0");
        // ASC request must not flip RG iteration order.
        assert!(
            !inner_parquet.reverse_row_groups(),
            "ASC request must not set reverse_row_groups",
        );
    }

    /// Same as above but for DESC. `reverse_row_groups` must also be
    /// `true` so the opener flips iteration order.
    #[test]
    fn try_pushdown_sort_returns_inexact_when_column_in_schema_desc() {
        use datafusion_physical_expr::EquivalenceProperties;
        use pushdown_sort_helpers::*;

        let schema = schema_with_a_int();
        let source = ParquetSource::new(Arc::clone(&schema));
        let order = vec![sort_expr_on(&schema, "a", true)];
        let eq = EquivalenceProperties::new(Arc::clone(&schema));

        let result = source.try_pushdown_sort(&order, &eq).unwrap();

        let SortOrderPushdownResult::Inexact { inner } = result else {
            panic!("expected Inexact, got a different variant");
        };
        let inner_parquet = inner
            .downcast_ref::<ParquetSource>()
            .expect("inner is ParquetSource");
        assert!(inner_parquet.sort_order_for_reorder.is_some());
        assert!(
            inner_parquet.reverse_row_groups(),
            "DESC request must set reverse_row_groups",
        );
    }

    /// A non-`Column` leading sort expression (e.g. `a + 1`,
    /// `date_trunc('hour', ts)`) with no declared source ordering
    /// yields `Unsupported` — parquet stats need a column name to
    /// look up min/max, and there's no source ordering to reverse.
    #[test]
    fn try_pushdown_sort_returns_unsupported_for_non_column_sort_expr() {
        use arrow::compute::SortOptions;
        use datafusion_physical_expr::EquivalenceProperties;
        use datafusion_physical_expr::expressions::{BinaryExpr, Column, lit};
        use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;
        use pushdown_sort_helpers::*;

        let schema = schema_with_a_int();
        let source = ParquetSource::new(Arc::clone(&schema));

        // `a + 1` — not a plain Column.
        let order = vec![PhysicalSortExpr {
            expr: Arc::new(BinaryExpr::new(
                Arc::new(Column::new("a", 0)),
                Operator::Plus,
                lit(1i32),
            )),
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        }];
        let eq = EquivalenceProperties::new(Arc::clone(&schema));

        let result = source.try_pushdown_sort(&order, &eq).unwrap();
        assert!(
            matches!(result, SortOrderPushdownResult::Unsupported),
            "non-Column sort expression must yield Unsupported",
        );
    }

    /// A sort column missing from the file schema with no declared
    /// source ordering yields `Unsupported` — there are no parquet
    /// stats for that column and no source ordering to reverse.
    #[test]
    fn try_pushdown_sort_returns_unsupported_when_column_not_in_file_schema() {
        use arrow::compute::SortOptions;
        use datafusion_physical_expr::EquivalenceProperties;
        use datafusion_physical_expr::expressions::Column;
        use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;
        use pushdown_sort_helpers::*;

        let schema = schema_with_a_int();
        let source = ParquetSource::new(Arc::clone(&schema));

        // Reference a column ("b") that does not exist in the file
        // schema (which only has "a"). The Column expression itself is
        // well-formed; only its name is unknown to the file.
        let order = vec![PhysicalSortExpr {
            expr: Arc::new(Column::new("b", 0)),
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        }];
        let eq = EquivalenceProperties::new(Arc::clone(&schema));

        let result = source.try_pushdown_sort(&order, &eq).unwrap();
        assert!(
            matches!(result, SortOrderPushdownResult::Unsupported),
            "column not in file schema must yield Unsupported",
        );
    }

    /// Regression: when the source declares `[a DESC]` and the request is
    /// `[a ASC]`, both `column_in_file_schema` and `reversed_satisfies`
    /// are true. `reverse_row_groups` must follow the *request's*
    /// direction (false for ASC) — not the source's, and not the OR of
    /// both signals. The opener's stats-based reorder produces
    /// ASC-by-min, so an ASC request needs no further flip; flipping
    /// would incorrectly emit DESC.
    #[test]
    fn try_pushdown_sort_source_desc_request_asc_does_not_reverse() {
        use datafusion_physical_expr::EquivalenceProperties;
        use pushdown_sort_helpers::*;

        let schema = schema_with_a_int();
        let source = ParquetSource::new(Arc::clone(&schema));
        // Source declares `[a DESC]`.
        let mut eq = EquivalenceProperties::new(Arc::clone(&schema));
        eq.add_ordering(vec![sort_expr_on(&schema, "a", true)]);
        // Request `[a ASC]`.
        let order = vec![sort_expr_on(&schema, "a", false)];

        let result = source.try_pushdown_sort(&order, &eq).unwrap();

        let SortOrderPushdownResult::Inexact { inner } = result else {
            panic!("expected Inexact, got a different variant");
        };
        let inner_parquet = inner
            .downcast_ref::<ParquetSource>()
            .expect("inner is ParquetSource");
        assert!(
            inner_parquet.sort_order_for_reorder.is_some(),
            "sort_order_for_reorder must be set",
        );
        assert!(
            !inner_parquet.reverse_row_groups(),
            "ASC request on source-DESC must not set reverse_row_groups; \
             a stale `reversed_satisfies || is_descending` formula would \
             incorrectly flip iteration to DESC after the stats reorder",
        );
    }

    /// A sort column that is *not* in the file schema (here: a partition
    /// column `b`) but the source's *reversed* declared ordering does
    /// satisfy the request. Pushdown fires via the reversed-equivalence
    /// path; `sort_order_for_reorder` is still set (so EXPLAIN reflects
    /// what the source was asked to approximate, even though the opener
    /// will skip its stats reorder because `b` has no per-RG min/max in
    /// the parquet file), and `reverse_row_groups` is `true` because we
    /// flip the file's natural order rather than re-sort by stats.
    #[test]
    fn try_pushdown_sort_returns_inexact_via_reversed_eq_when_column_not_in_file_schema()
    {
        use arrow::compute::SortOptions;
        use arrow::datatypes::{DataType, Field, Schema};
        use datafusion_datasource::TableSchema;
        use datafusion_physical_expr::EquivalenceProperties;
        use datafusion_physical_expr::expressions::Column;
        use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;

        // File schema is just `[a]`; `b` lives as a partition column on
        // top, so it appears in the table schema but not the file schema
        // — the same shape `column_in_file_schema` discards.
        let file_schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        let partition_b = Arc::new(Field::new("b", DataType::Int32, true));
        let table_schema = TableSchema::builder(file_schema)
            .with_table_partition_cols(vec![partition_b])
            .build();
        let source = ParquetSource::new(table_schema);

        // EquivalenceProperties is built on the *full* table schema so
        // it can carry an ordering on `b`.
        let full_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]));
        // Construct the request first, then derive the declared
        // ordering as its reverse, so `ordering_satisfy` on the
        // reversed-eq holds exactly. `PhysicalSortExpr::reverse` flips
        // both `descending` and `nulls_first`, so spelling the
        // declared ordering directly is error-prone.
        let request_expr = PhysicalSortExpr {
            expr: Arc::new(Column::new("b", 1)),
            options: SortOptions {
                descending: true,
                nulls_first: true,
            },
        };
        let declared = request_expr.reverse();
        let mut eq = EquivalenceProperties::new(Arc::clone(&full_schema));
        eq.add_ordering(vec![declared]);
        let order = vec![request_expr];

        let result = source.try_pushdown_sort(&order, &eq).unwrap();

        let SortOrderPushdownResult::Inexact { inner } = result else {
            panic!("expected Inexact, got a different variant");
        };
        let inner_parquet = inner
            .downcast_ref::<ParquetSource>()
            .expect("inner is ParquetSource");
        assert!(
            inner_parquet.sort_order_for_reorder.is_some(),
            "sort_order_for_reorder must be set so EXPLAIN reflects the request",
        );
        assert!(
            inner_parquet.reverse_row_groups(),
            "request reached via reversed_satisfies (column-not-in-file-schema) \
             must set reverse_row_groups to flip the file's natural order",
        );
    }

    /// Regression: when the source's declared ordering is a non-empty
    /// *proper* prefix of the request, `try_pushdown_sort` must return
    /// `Unsupported` so that the outer `SortExec` can keep its
    /// `sort_prefix` annotation and `TopK` can early-terminate within
    /// each prefix block. Letting the Phase 3 Inexact pipeline fire
    /// would drop the source's `output_ordering`, destroying the
    /// information `EnforceSorting` needs to compute `sort_prefix`.
    #[test]
    fn try_pushdown_sort_preserves_sort_prefix_when_source_declares_prefix_ordering() {
        use arrow::compute::SortOptions;
        use arrow::datatypes::{DataType, Field, Schema};
        use datafusion_physical_expr::EquivalenceProperties;
        use datafusion_physical_expr::expressions::Column;
        use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
            Field::new("c", DataType::Int32, true),
        ]));
        let source = ParquetSource::new(Arc::clone(&schema));

        // Source declares `[a DESC, b ASC NULLS LAST]` — the same prefix
        // the SortExec input will see.
        let mut eq = EquivalenceProperties::new(Arc::clone(&schema));
        eq.add_ordering(vec![
            PhysicalSortExpr {
                expr: Arc::new(Column::new("a", 0)),
                options: SortOptions {
                    descending: true,
                    nulls_first: true,
                },
            },
            PhysicalSortExpr {
                expr: Arc::new(Column::new("b", 1)),
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            },
        ]);

        // Request `[a DESC, b ASC NULLS LAST, c DESC]` — three columns;
        // source's two-column declaration is a strict prefix.
        let order = vec![
            PhysicalSortExpr {
                expr: Arc::new(Column::new("a", 0)),
                options: SortOptions {
                    descending: true,
                    nulls_first: true,
                },
            },
            PhysicalSortExpr {
                expr: Arc::new(Column::new("b", 1)),
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            },
            PhysicalSortExpr {
                expr: Arc::new(Column::new("c", 2)),
                options: SortOptions {
                    descending: true,
                    nulls_first: true,
                },
            },
        ];

        let result = source.try_pushdown_sort(&order, &eq).unwrap();
        assert!(
            matches!(result, SortOrderPushdownResult::Unsupported),
            "source ordering [a DESC, b ASC NULLS LAST] is a proper prefix \
             of the request — `try_pushdown_sort` must return Unsupported so \
             the SortExec sort_prefix optimisation can fire",
        );
    }

    /// Helpers for the `reorder_files` regression tests below.
    mod reorder_files_helpers {
        use super::*;
        use datafusion_common::stats::Precision;
        use datafusion_common::{ColumnStatistics, ScalarValue, Statistics};
        use datafusion_datasource::PartitionedFile;

        pub(super) fn file_with_min(name: &str, min: Option<i32>) -> PartitionedFile {
            let mut pf = PartitionedFile::new(name.to_string(), 0);
            let min_value = min
                .map(|v| Precision::Exact(ScalarValue::Int32(Some(v))))
                .unwrap_or(Precision::Absent);
            pf.statistics = Some(Arc::new(Statistics {
                num_rows: Precision::Absent,
                total_byte_size: Precision::Absent,
                column_statistics: vec![ColumnStatistics {
                    null_count: Precision::Absent,
                    max_value: Precision::Absent,
                    min_value,
                    sum_value: Precision::Absent,
                    distinct_count: Precision::Absent,
                    byte_size: Precision::Absent,
                }],
            }));
            pf
        }

        pub(super) fn names(files: &[PartitionedFile]) -> Vec<&str> {
            files
                .iter()
                .map(|f| f.object_meta.location.as_ref())
                .collect()
        }
    }

    /// ASC TopK: `reorder_files` keys off file `min` and sorts ASC,
    /// so the file with the smallest `min` is read first. This
    /// matches `PreparedAccessPlan::reorder_by_statistics` at the
    /// row-group level (also `min ASC`).
    #[test]
    fn reorder_files_sorts_asc_by_min_for_asc_request() {
        use pushdown_sort_helpers::*;
        use reorder_files_helpers::*;

        let schema = schema_with_a_int();
        let mut source = ParquetSource::new(Arc::clone(&schema));
        source.sort_order_for_reorder =
            Some(LexOrdering::new(vec![sort_expr_on(&schema, "a", false)]).unwrap());
        // ASC request → `reverse_row_groups` left at its default `false`.

        let reordered = source.reorder_files(vec![
            file_with_min("middle", Some(50)),
            file_with_min("small", Some(10)),
            file_with_min("large", Some(100)),
        ]);

        assert_eq!(names(&reordered), vec!["small", "middle", "large"]);
    }

    /// DESC TopK: same `min` key, but sorted DESC — file with the
    /// largest `min` first. Mirrors the row-group strategy of
    /// "ASC-by-min then `reverse`".
    #[test]
    fn reorder_files_sorts_desc_by_min_for_desc_request() {
        use pushdown_sort_helpers::*;
        use reorder_files_helpers::*;

        let schema = schema_with_a_int();
        let mut source =
            ParquetSource::new(Arc::clone(&schema)).with_reverse_row_groups(true);
        source.sort_order_for_reorder =
            Some(LexOrdering::new(vec![sort_expr_on(&schema, "a", true)]).unwrap());

        let reordered = source.reorder_files(vec![
            file_with_min("middle", Some(50)),
            file_with_min("small", Some(10)),
            file_with_min("large", Some(100)),
        ]);

        assert_eq!(names(&reordered), vec!["large", "middle", "small"]);
    }

    /// Files without statistics sort to the *end* so present-stats
    /// files run first regardless of direction. Verified for both
    /// ASC and DESC.
    #[test]
    fn reorder_files_pushes_missing_stats_to_the_end() {
        use pushdown_sort_helpers::*;
        use reorder_files_helpers::*;

        let schema = schema_with_a_int();
        let mut source = ParquetSource::new(Arc::clone(&schema));
        source.sort_order_for_reorder =
            Some(LexOrdering::new(vec![sort_expr_on(&schema, "a", false)]).unwrap());

        let reordered = source.reorder_files(vec![
            file_with_min("no_stats", None),
            file_with_min("has_min", Some(10)),
        ]);
        assert_eq!(names(&reordered), vec!["has_min", "no_stats"]);

        // Same for DESC.
        let mut source =
            ParquetSource::new(Arc::clone(&schema)).with_reverse_row_groups(true);
        source.sort_order_for_reorder =
            Some(LexOrdering::new(vec![sort_expr_on(&schema, "a", true)]).unwrap());
        let reordered = source.reorder_files(vec![
            file_with_min("no_stats", None),
            file_with_min("has_min", Some(10)),
        ]);
        assert_eq!(names(&reordered), vec!["has_min", "no_stats"]);
    }

    /// When no sort pushdown has fired (`sort_order_for_reorder` is
    /// `None`), `reorder_files` is a no-op and preserves input order.
    #[test]
    fn reorder_files_is_a_no_op_without_pushdown() {
        use pushdown_sort_helpers::*;
        use reorder_files_helpers::*;

        let schema = schema_with_a_int();
        let source = ParquetSource::new(schema);
        // No `sort_order_for_reorder` set on the source.

        let input = vec![
            file_with_min("c", Some(30)),
            file_with_min("a", Some(10)),
            file_with_min("b", Some(20)),
        ];
        let reordered = source.reorder_files(input.clone());
        assert_eq!(names(&reordered), names(&input));
    }

    /// `sort_order_for_reorder` is surfaced in both `EXPLAIN` (Default)
    /// and `EXPLAIN VERBOSE` / `EXPLAIN ANALYZE` (Verbose) so readers
    /// and snapshot tests can see the inexact sort-pushdown fired.
    #[test]
    fn sort_order_for_reorder_shown_in_explain() {
        use pushdown_sort_helpers::*;

        // `std::fmt::Formatter` can't be constructed outside core fmt
        // machinery, so we drive `fmt_extra` through a Display adapter
        // and read the rendered string back with `format!`.
        struct DisplayHelper<'a> {
            source: &'a ParquetSource,
            mode: DisplayFormatType,
        }
        impl std::fmt::Display for DisplayHelper<'_> {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                self.source.fmt_extra(self.mode, f)
            }
        }

        let schema = schema_with_a_int();
        let mut source = ParquetSource::new(Arc::clone(&schema));
        let order = LexOrdering::new(vec![sort_expr_on(&schema, "a", false)]).unwrap();
        source.sort_order_for_reorder = Some(order);

        for mode in [DisplayFormatType::Default, DisplayFormatType::Verbose] {
            let out = format!(
                "{}",
                DisplayHelper {
                    source: &source,
                    mode,
                },
            );
            assert!(
                out.contains("sort_order_for_reorder=[a@0 ASC]"),
                "{mode:?} display must surface sort_order_for_reorder, got: {out}",
            );
        }
    }
}
