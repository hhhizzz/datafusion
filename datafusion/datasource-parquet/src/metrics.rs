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

use std::sync::Arc;

use datafusion_physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, Gauge, Label, MetricBuilder, MetricCategory,
    MetricType, PruningMetrics, RatioMergeStrategy, RatioMetrics, Time,
};
use parquet::arrow::arrow_reader::metrics::ArrowReaderMetrics;

/// Stores metrics about the parquet execution for a particular parquet file.
///
/// This component is a subject to **change** in near future and is exposed for low level integrations
/// through [`ParquetFileReaderFactory`].
///
/// [`ParquetFileReaderFactory`]: super::ParquetFileReaderFactory
#[derive(Debug, Clone)]
pub struct ParquetFileMetrics {
    /// Number of file **ranges** pruned or matched by partition or file level statistics.
    /// Pruning of files often happens at planning time but may happen at execution time
    /// if dynamic filters (e.g. from a join) result in additional pruning.
    ///
    /// This does **not** necessarily equal the number of files pruned:
    /// files may be scanned in sub-ranges to increase parallelism,
    /// in which case this will represent the number of sub-ranges pruned, not the number of files.
    /// The number of files pruned will always be less than or equal to this number.
    ///
    /// A single file may have some ranges that are not pruned and some that are pruned.
    /// For example, with a query like `ORDER BY col LIMIT 10`, the TopK dynamic filter
    /// pushdown optimization may fill up the TopK heap when reading the first part of a file,
    /// then skip the second part if file statistics indicate it cannot contain rows
    /// that would be in the TopK.
    pub files_ranges_pruned_statistics: PruningMetrics,
    /// Number of times the predicate could not be evaluated
    pub predicate_evaluation_errors: Count,
    /// Number of row groups pruned by bloom filters
    pub row_groups_pruned_bloom_filter: PruningMetrics,
    /// Number of row groups pruned due to limit pruning.
    pub limit_pruned_row_groups: PruningMetrics,
    /// Number of row groups pruned by statistics
    pub row_groups_pruned_statistics: PruningMetrics,
    /// Total number of bytes scanned
    pub bytes_scanned: Count,
    /// Total rows filtered out by predicates pushed into parquet scan
    pub pushdown_rows_pruned: Count,
    /// Total rows passed predicates pushed into parquet scan
    pub pushdown_rows_matched: Count,
    /// Total time spent evaluating row-level pushdown filters
    pub row_pushdown_eval_time: Time,
    /// Total time spent evaluating row group-level statistics filters
    pub statistics_eval_time: Time,
    /// Total time spent evaluating row group Bloom Filters
    pub bloom_filter_eval_time: Time,
    /// Total rows filtered or matched by parquet page index
    pub page_index_rows_pruned: PruningMetrics,
    /// Total pages filtered or matched by parquet page index
    pub page_index_pages_pruned: PruningMetrics,
    /// Total time spent evaluating parquet page index filters
    pub page_index_eval_time: Time,
    /// Total time spent reading and parsing metadata from the footer
    pub metadata_load_time: Time,
    /// Scan Efficiency Ratio, calculated as bytes_scanned / total_file_size
    pub scan_efficiency_ratio: RatioMetrics,
    /// Predicate Cache: Total number of rows physically read and decoded from the Parquet file.
    ///
    /// This metric tracks "cache misses" in the predicate pushdown optimization.
    /// When the specialized predicate reader cannot find the requested data in its cache,
    /// it must fall back to the "inner reader" to physically decode the data from the
    /// Parquet.
    ///
    /// This is the expensive path (IO + Decompression + Decoding).
    ///
    /// We use a Gauge here as arrow-rs reports absolute numbers rather
    /// than incremental readings, we want a `set` operation here rather
    /// than `add`. Earlier it was `Count`, which led to this issue:
    /// github.com/apache/datafusion/issues/19334
    pub predicate_cache_inner_records: Gauge,
    /// Predicate Cache: number of records read from the cache. This is the
    /// number of rows that were stored in the cache after evaluating predicates
    /// reused for the output.
    pub predicate_cache_records: Gauge,
    /// Row filter: total rows entering row-level filter selections.
    pub row_filter_input_rows: Gauge,
    /// Row filter: rows selected by row-level filter selections.
    pub row_filter_selected_rows: Gauge,
    /// Row filter: rows skipped by row-level filter selections.
    pub row_filter_skipped_rows: Gauge,
    /// Row filter: selected rows divided by input rows.
    pub row_filter_selected_ratio: RatioMetrics,
    /// Row selection: selected rows recorded in planned selections.
    pub row_selection_selected_rows: Gauge,
    /// Row selection: skipped rows recorded in planned selections.
    pub row_selection_skipped_rows: Gauge,
    /// Row selection: non-empty selectors recorded in planned selections.
    pub row_selection_selector_count: Gauge,
    /// Row selection: selected runs recorded in planned selections.
    pub row_selection_selected_run_count: Gauge,
    /// Row selection: skipped runs recorded in planned selections.
    pub row_selection_skipped_run_count: Gauge,
    /// Row selection: selected runs divided by selected rows.
    pub row_selection_fragmentation_ratio: RatioMetrics,
    /// Row selection: plans materialized with masks.
    pub row_selection_mask_plan_count: Gauge,
    /// Row selection: plans materialized with selectors.
    pub row_selection_selector_plan_count: Gauge,
    /// Row selection: plans forced to masks.
    pub row_selection_forced_mask_plan_count: Gauge,
    /// Row selection: plans forced to selectors.
    pub row_selection_forced_selector_plan_count: Gauge,
    /// Row selection: Auto plans choosing masks for empty selections.
    pub row_selection_auto_mask_empty_plan_count: Gauge,
    /// Row selection: Auto plans choosing masks for short selected runs.
    pub row_selection_auto_mask_short_run_plan_count: Gauge,
    /// Row selection: Auto plans choosing masks for fragmented selections.
    pub row_selection_auto_mask_fragmented_plan_count: Gauge,
    /// Row selection: Auto plans choosing masks for high selected-row ratios.
    pub row_selection_auto_mask_high_ratio_plan_count: Gauge,
    /// Row selection: Auto plans choosing selectors for clustered selections.
    pub row_selection_auto_selector_clustered_plan_count: Gauge,
    /// Row selection: Auto plans choosing selectors for long selected runs.
    pub row_selection_auto_selector_long_run_plan_count: Gauge,
    /// Cost model: row groups included in the observation window.
    pub cost_model_observed_row_group_count: Gauge,
    /// Cost model: row groups executed with pushdown.
    pub cost_model_pushdown_row_group_count: Gauge,
    /// Cost model: row groups executed with post-filter.
    pub cost_model_post_filter_row_group_count: Gauge,
    /// Cost model: incomplete observation-window decisions.
    pub cost_model_observation_incomplete_count: Gauge,
    /// Cost model: decisions that kept pushdown.
    pub cost_model_pushdown_still_preferred_count: Gauge,
    /// Cost model: high-selectivity no-pruning triggers.
    pub cost_model_high_selectivity_no_pruning_count: Gauge,
    /// Cost model: projected-predicate moderate-selectivity triggers.
    pub cost_model_projected_predicate_moderate_selectivity_count: Gauge,
    /// Cost model: fragmented moderate-selectivity triggers.
    pub cost_model_fragmented_moderate_selectivity_count: Gauge,
    /// Cost model: fragmented high-selectivity triggers.
    pub cost_model_fragmented_high_selectivity_count: Gauge,
    /// Predicate cache: cached records divided by total predicate records.
    pub predicate_cache_hit_ratio: RatioMetrics,
}

impl ParquetFileMetrics {
    /// Create new metrics
    pub fn new(
        partition: usize,
        filename: &str,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Self {
        // Share the filename label across all per-file metrics to avoid
        // allocating the same filename string for each metric.
        let filename_label = Label::new("filename", Arc::<str>::from(filename));
        let builder = MetricBuilder::new(metrics).with_label(filename_label);

        // -----------------------
        // 'summary' level metrics
        // -----------------------
        let row_groups_pruned_bloom_filter = builder
            .clone()
            .with_type(MetricType::Summary)
            .pruning_metrics("row_groups_pruned_bloom_filter", partition);

        let limit_pruned_row_groups = builder
            .clone()
            .with_type(MetricType::Summary)
            .pruning_metrics("limit_pruned_row_groups", partition);

        let row_groups_pruned_statistics = builder
            .clone()
            .with_type(MetricType::Summary)
            .pruning_metrics("row_groups_pruned_statistics", partition);

        let page_index_pages_pruned = builder
            .clone()
            .with_type(MetricType::Summary)
            .pruning_metrics("page_index_pages_pruned", partition);

        let bytes_scanned = builder
            .clone()
            .with_type(MetricType::Summary)
            .with_category(MetricCategory::Bytes)
            .counter("bytes_scanned", partition);

        let metadata_load_time = builder
            .clone()
            .with_type(MetricType::Summary)
            .subset_time("metadata_load_time", partition);

        let files_ranges_pruned_statistics = MetricBuilder::new(metrics)
            .with_type(MetricType::Summary)
            .pruning_metrics("files_ranges_pruned_statistics", partition);

        let scan_efficiency_ratio = builder
            .clone()
            .with_type(MetricType::Summary)
            .ratio_metrics_with_strategy(
                "scan_efficiency_ratio",
                partition,
                RatioMergeStrategy::AddPartSetTotal,
            );

        // -----------------------
        // 'dev' level metrics
        // -----------------------
        let predicate_evaluation_errors = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .counter("predicate_evaluation_errors", partition);

        let pushdown_rows_pruned = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .counter("pushdown_rows_pruned", partition);
        let pushdown_rows_matched = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .counter("pushdown_rows_matched", partition);

        let row_pushdown_eval_time = builder
            .clone()
            .subset_time("row_pushdown_eval_time", partition);
        let statistics_eval_time = builder
            .clone()
            .subset_time("statistics_eval_time", partition);
        let bloom_filter_eval_time = builder
            .clone()
            .subset_time("bloom_filter_eval_time", partition);

        let page_index_eval_time = builder
            .clone()
            .subset_time("page_index_eval_time", partition);

        let page_index_rows_pruned = builder
            .clone()
            .pruning_metrics("page_index_rows_pruned", partition);

        let predicate_cache_inner_records = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("predicate_cache_inner_records", partition);

        let predicate_cache_records = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("predicate_cache_records", partition);

        let row_filter_input_rows = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_filter_input_rows", partition);
        let row_filter_selected_rows = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_filter_selected_rows", partition);
        let row_filter_skipped_rows = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_filter_skipped_rows", partition);
        let row_filter_selected_ratio = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .ratio_metrics("row_filter_selected_ratio", partition);

        let row_selection_selected_rows = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_selection_selected_rows", partition);
        let row_selection_skipped_rows = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_selection_skipped_rows", partition);
        let row_selection_selector_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_selection_selector_count", partition);
        let row_selection_selected_run_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_selection_selected_run_count", partition);
        let row_selection_skipped_run_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_selection_skipped_run_count", partition);
        let row_selection_fragmentation_ratio = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .ratio_metrics("row_selection_fragmentation_ratio", partition);
        let row_selection_mask_plan_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_selection_mask_plan_count", partition);
        let row_selection_selector_plan_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_selection_selector_plan_count", partition);
        let row_selection_forced_mask_plan_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_selection_forced_mask_plan_count", partition);
        let row_selection_forced_selector_plan_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_selection_forced_selector_plan_count", partition);
        let row_selection_auto_mask_empty_plan_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_selection_auto_mask_empty_plan_count", partition);
        let row_selection_auto_mask_short_run_plan_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_selection_auto_mask_short_run_plan_count", partition);
        let row_selection_auto_mask_fragmented_plan_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_selection_auto_mask_fragmented_plan_count", partition);
        let row_selection_auto_mask_high_ratio_plan_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_selection_auto_mask_high_ratio_plan_count", partition);
        let row_selection_auto_selector_clustered_plan_count =
            builder.clone().with_category(MetricCategory::Rows).gauge(
                "row_selection_auto_selector_clustered_plan_count",
                partition,
            );
        let row_selection_auto_selector_long_run_plan_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("row_selection_auto_selector_long_run_plan_count", partition);

        let cost_model_observed_row_group_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("cost_model_observed_row_group_count", partition);
        let cost_model_pushdown_row_group_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("cost_model_pushdown_row_group_count", partition);
        let cost_model_post_filter_row_group_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("cost_model_post_filter_row_group_count", partition);
        let cost_model_observation_incomplete_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("cost_model_observation_incomplete_count", partition);
        let cost_model_pushdown_still_preferred_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("cost_model_pushdown_still_preferred_count", partition);
        let cost_model_high_selectivity_no_pruning_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("cost_model_high_selectivity_no_pruning_count", partition);
        let cost_model_projected_predicate_moderate_selectivity_count =
            builder.clone().with_category(MetricCategory::Rows).gauge(
                "cost_model_projected_predicate_moderate_selectivity_count",
                partition,
            );
        let cost_model_fragmented_moderate_selectivity_count =
            builder.clone().with_category(MetricCategory::Rows).gauge(
                "cost_model_fragmented_moderate_selectivity_count",
                partition,
            );
        let cost_model_fragmented_high_selectivity_count = builder
            .clone()
            .with_category(MetricCategory::Rows)
            .gauge("cost_model_fragmented_high_selectivity_count", partition);

        let predicate_cache_hit_ratio = builder
            .with_category(MetricCategory::Rows)
            .ratio_metrics("predicate_cache_hit_ratio", partition);

        Self {
            files_ranges_pruned_statistics,
            predicate_evaluation_errors,
            row_groups_pruned_bloom_filter,
            row_groups_pruned_statistics,
            limit_pruned_row_groups,
            bytes_scanned,
            pushdown_rows_pruned,
            pushdown_rows_matched,
            row_pushdown_eval_time,
            page_index_rows_pruned,
            page_index_pages_pruned,
            statistics_eval_time,
            bloom_filter_eval_time,
            page_index_eval_time,
            metadata_load_time,
            scan_efficiency_ratio,
            predicate_cache_inner_records,
            predicate_cache_records,
            row_filter_input_rows,
            row_filter_selected_rows,
            row_filter_skipped_rows,
            row_filter_selected_ratio,
            row_selection_selected_rows,
            row_selection_skipped_rows,
            row_selection_selector_count,
            row_selection_selected_run_count,
            row_selection_skipped_run_count,
            row_selection_fragmentation_ratio,
            row_selection_mask_plan_count,
            row_selection_selector_plan_count,
            row_selection_forced_mask_plan_count,
            row_selection_forced_selector_plan_count,
            row_selection_auto_mask_empty_plan_count,
            row_selection_auto_mask_short_run_plan_count,
            row_selection_auto_mask_fragmented_plan_count,
            row_selection_auto_mask_high_ratio_plan_count,
            row_selection_auto_selector_clustered_plan_count,
            row_selection_auto_selector_long_run_plan_count,
            cost_model_observed_row_group_count,
            cost_model_pushdown_row_group_count,
            cost_model_post_filter_row_group_count,
            cost_model_observation_incomplete_count,
            cost_model_pushdown_still_preferred_count,
            cost_model_high_selectivity_no_pruning_count,
            cost_model_projected_predicate_moderate_selectivity_count,
            cost_model_fragmented_moderate_selectivity_count,
            cost_model_fragmented_high_selectivity_count,
            predicate_cache_hit_ratio,
        }
    }

    /// Copy absolute counters from arrow-rs reader metrics into DataFusion
    /// gauges and derived ratios.
    pub(crate) fn copy_arrow_reader_metrics(
        &self,
        arrow_reader_metrics: &ArrowReaderMetrics,
    ) {
        let inner_records = arrow_reader_metrics.records_read_from_inner();
        let cached_records = arrow_reader_metrics.records_read_from_cache();
        set_gauge(&self.predicate_cache_inner_records, inner_records);
        set_gauge(&self.predicate_cache_records, cached_records);
        if let (Some(inner), Some(cached)) = (inner_records, cached_records) {
            set_ratio(&self.predicate_cache_hit_ratio, cached, inner + cached);
        }

        let selected_rows = arrow_reader_metrics.row_selection_selected_rows();
        let skipped_rows = arrow_reader_metrics.row_selection_skipped_rows();
        set_gauge(&self.row_selection_selected_rows, selected_rows);
        set_gauge(&self.row_selection_skipped_rows, skipped_rows);
        if let (Some(selected), Some(skipped)) = (selected_rows, skipped_rows) {
            self.row_filter_selected_rows.set(selected);
            self.row_filter_skipped_rows.set(skipped);
            self.row_filter_input_rows.set(selected + skipped);
            set_ratio(
                &self.row_filter_selected_ratio,
                selected,
                selected + skipped,
            );
        }

        set_gauge(
            &self.row_selection_selector_count,
            arrow_reader_metrics.row_selection_selector_count(),
        );
        let selected_run_count = arrow_reader_metrics.row_selection_selected_run_count();
        set_gauge(&self.row_selection_selected_run_count, selected_run_count);
        set_gauge(
            &self.row_selection_skipped_run_count,
            arrow_reader_metrics.row_selection_skipped_run_count(),
        );
        if let (Some(selected_runs), Some(selected)) = (selected_run_count, selected_rows)
        {
            set_ratio(
                &self.row_selection_fragmentation_ratio,
                selected_runs,
                selected,
            );
        }

        set_gauge(
            &self.row_selection_mask_plan_count,
            arrow_reader_metrics.row_selection_mask_plan_count(),
        );
        set_gauge(
            &self.row_selection_selector_plan_count,
            arrow_reader_metrics.row_selection_selector_plan_count(),
        );
        set_gauge(
            &self.row_selection_forced_mask_plan_count,
            arrow_reader_metrics.row_selection_forced_mask_plan_count(),
        );
        set_gauge(
            &self.row_selection_forced_selector_plan_count,
            arrow_reader_metrics.row_selection_forced_selector_plan_count(),
        );
        set_gauge(
            &self.row_selection_auto_mask_empty_plan_count,
            arrow_reader_metrics.row_selection_auto_mask_empty_plan_count(),
        );
        set_gauge(
            &self.row_selection_auto_mask_short_run_plan_count,
            arrow_reader_metrics.row_selection_auto_mask_short_run_plan_count(),
        );
        set_gauge(
            &self.row_selection_auto_mask_fragmented_plan_count,
            arrow_reader_metrics.row_selection_auto_mask_fragmented_plan_count(),
        );
        set_gauge(
            &self.row_selection_auto_mask_high_ratio_plan_count,
            arrow_reader_metrics.row_selection_auto_mask_high_ratio_plan_count(),
        );
        set_gauge(
            &self.row_selection_auto_selector_clustered_plan_count,
            arrow_reader_metrics.row_selection_auto_selector_clustered_plan_count(),
        );
        set_gauge(
            &self.row_selection_auto_selector_long_run_plan_count,
            arrow_reader_metrics.row_selection_auto_selector_long_run_plan_count(),
        );

        set_gauge(
            &self.cost_model_observed_row_group_count,
            arrow_reader_metrics.cost_model_observed_row_group_count(),
        );
        set_gauge(
            &self.cost_model_pushdown_row_group_count,
            arrow_reader_metrics.cost_model_pushdown_row_group_count(),
        );
        set_gauge(
            &self.cost_model_post_filter_row_group_count,
            arrow_reader_metrics.cost_model_post_filter_row_group_count(),
        );
        set_gauge(
            &self.cost_model_observation_incomplete_count,
            arrow_reader_metrics.cost_model_observation_incomplete_count(),
        );
        set_gauge(
            &self.cost_model_pushdown_still_preferred_count,
            arrow_reader_metrics.cost_model_pushdown_still_preferred_count(),
        );
        set_gauge(
            &self.cost_model_high_selectivity_no_pruning_count,
            arrow_reader_metrics.cost_model_high_selectivity_no_pruning_count(),
        );
        set_gauge(
            &self.cost_model_projected_predicate_moderate_selectivity_count,
            arrow_reader_metrics
                .cost_model_projected_predicate_moderate_selectivity_count(),
        );
        set_gauge(
            &self.cost_model_fragmented_moderate_selectivity_count,
            arrow_reader_metrics.cost_model_fragmented_moderate_selectivity_count(),
        );
        set_gauge(
            &self.cost_model_fragmented_high_selectivity_count,
            arrow_reader_metrics.cost_model_fragmented_high_selectivity_count(),
        );
    }

    /// Record pages whose page-index pruning was skipped because the containing
    /// row group was fully matched by row-group statistics.
    ///
    /// The counter is only registered when there is a non-zero value. This keeps
    /// [`ParquetFileMetrics::new`] from cloning the filename and metrics set for
    /// files that never use this metric.
    pub(crate) fn add_page_index_pages_skipped_by_fully_matched(
        metrics: &ExecutionPlanMetricsSet,
        partition: usize,
        filename: &str,
        n: usize,
    ) {
        if n == 0 {
            return;
        }

        let count = MetricBuilder::new(metrics)
            .with_new_label("filename", filename.to_string())
            .with_type(MetricType::Summary)
            .with_category(MetricCategory::Rows)
            .counter("page_index_pages_skipped_by_fully_matched", partition);
        count.add(n);
    }
}

fn set_gauge(gauge: &Gauge, value: Option<usize>) {
    if let Some(value) = value {
        gauge.set(value);
    }
}

fn set_ratio(ratio: &RatioMetrics, part: usize, total: usize) {
    ratio.set_part(part);
    ratio.set_total(total);
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_physical_plan::metrics::MetricValue;

    fn metric_names(metrics: &ExecutionPlanMetricsSet) -> Vec<String> {
        metrics
            .clone_inner()
            .iter()
            .map(|metric| metric.value().name().to_string())
            .collect()
    }

    #[test]
    fn parquet_file_metrics_register_arrow_reader_bridge_metrics() {
        let metrics = ExecutionPlanMetricsSet::new();
        let file_metrics = ParquetFileMetrics::new(0, "file.parquet", &metrics);

        file_metrics.row_filter_input_rows.set(10);
        file_metrics.row_filter_selected_rows.set(4);
        file_metrics.row_filter_skipped_rows.set(6);
        file_metrics.row_filter_selected_ratio.set_part(4);
        file_metrics.row_filter_selected_ratio.set_total(10);

        let names = metric_names(&metrics);
        for expected in [
            "row_filter_input_rows",
            "row_filter_selected_rows",
            "row_filter_skipped_rows",
            "row_filter_selected_ratio",
            "row_selection_selected_run_count",
            "row_selection_skipped_run_count",
            "row_selection_fragmentation_ratio",
            "row_selection_mask_plan_count",
            "row_selection_selector_plan_count",
            "cost_model_observed_row_group_count",
            "cost_model_pushdown_row_group_count",
            "cost_model_post_filter_row_group_count",
            "cost_model_fragmented_high_selectivity_count",
            "predicate_cache_hit_ratio",
        ] {
            assert!(
                names.iter().any(|name| name == expected),
                "missing metric {expected}; registered metrics: {names:?}"
            );
        }

        assert!(
            metrics
                .clone_inner()
                .sum(|metric| {
                    matches!(
                        metric.value(),
                        MetricValue::Ratio { name, .. }
                            if name.as_ref() == "row_filter_selected_ratio"
                    )
                })
                .is_some()
        );
    }
}
