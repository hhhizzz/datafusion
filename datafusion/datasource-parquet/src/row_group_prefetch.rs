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

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use parquet::arrow::ProjectionMask;
use parquet::file::metadata::ParquetMetaData;

use datafusion_physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, Gauge, MetricBuilder, MetricCategory,
};

const MAX_MERGE_GAP: u64 = 256 * 1024;
const MAX_MERGED_RANGE: u64 = 4 * 1024 * 1024;
const MIN_ADMISSION_BYTES: usize = 1024 * 1024;

/// Plans the projected compressed column spans for one prepared decoder run.
#[derive(Debug, Clone)]
pub(crate) struct RowGroupPrefetchPlan {
    row_group_order: Vec<usize>,
    ranges_by_row_group: BTreeMap<usize, Vec<Range<u64>>>,
    projected_payload_bytes: usize,
}

impl RowGroupPrefetchPlan {
    /// Creates a plan from the final decoder row-group order and projected leaves.
    pub(crate) fn new(
        metadata: &ParquetMetaData,
        projection_mask: &ProjectionMask,
        row_group_order: Vec<usize>,
    ) -> Self {
        let mut ranges_by_row_group = BTreeMap::new();

        for &row_group_index in &row_group_order {
            if ranges_by_row_group.contains_key(&row_group_index) {
                continue;
            }
            let Some(row_group) = metadata.row_groups().get(row_group_index) else {
                continue;
            };

            let mut ranges = row_group
                .columns()
                .iter()
                .enumerate()
                .filter_map(|(leaf_index, column)| {
                    projection_mask
                        .leaf_included(leaf_index)
                        .then(|| column.byte_range())
                        .and_then(|(start, length)| {
                            start.checked_add(length).map(|end| start..end)
                        })
                })
                .collect::<Vec<_>>();
            ranges.sort_unstable_by_key(|range| (range.start, range.end));
            ranges_by_row_group.insert(row_group_index, ranges);
        }

        let projected_payload_bytes = unique_range_bytes(
            ranges_by_row_group
                .values()
                .flat_map(|ranges| ranges.iter().cloned()),
        );

        Self {
            row_group_order,
            ranges_by_row_group,
            projected_payload_bytes,
        }
    }

    /// Returns the final decoder order, including a reverse scan order.
    pub(crate) fn row_group_order(&self) -> &[usize] {
        &self.row_group_order
    }

    /// Returns the unique projected compressed bytes across this decoder run.
    pub(crate) fn projected_payload_bytes(&self) -> usize {
        self.projected_payload_bytes
    }

    /// Returns deterministic physical ranges for the requested row groups.
    pub(crate) fn ranges_for(&self, row_group_indexes: &[usize]) -> Vec<Range<u64>> {
        let requested = row_group_indexes.iter().copied().collect::<BTreeSet<_>>();
        let mut ranges = requested
            .into_iter()
            .filter_map(|index| self.ranges_by_row_group.get(&index))
            .flat_map(|ranges| ranges.iter().cloned())
            .collect::<Vec<_>>();
        ranges.sort_unstable_by_key(|range| (range.start, range.end));
        merge_ranges(ranges)
    }

    /// Creates policy state that observes exact requests for this plan's spans.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Task 5 drives density admission.")
    )]
    pub(crate) fn density_admission(&self) -> DensityAdmission {
        let candidate_ranges = merge_ranges_without_gap(
            self.ranges_by_row_group
                .values()
                .flat_map(|ranges| ranges.iter().cloned())
                .collect(),
        );
        DensityAdmission::new(candidate_ranges)
    }

    /// Creates admission state for exact requests from one file-level row group.
    pub(crate) fn density_admission_for(
        &self,
        row_group_index: usize,
    ) -> DensityAdmission {
        DensityAdmission::new(
            self.ranges_by_row_group
                .get(&row_group_index)
                .cloned()
                .unwrap_or_default(),
        )
    }

    /// Returns the next unclaimed row groups after `row_group_index` in final
    /// decoder order, preserving reverse scans.
    pub(crate) fn next_window_after(
        &self,
        row_group_index: usize,
        window: usize,
        claimed: &BTreeSet<usize>,
    ) -> Vec<usize> {
        self.row_group_order
            .iter()
            .position(|index| *index == row_group_index)
            .map(|position| {
                self.row_group_order[position + 1..]
                    .iter()
                    .copied()
                    .filter(|index| !claimed.contains(index))
                    .take(window)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns the part of staged physical ranges attributable to each row
    /// group. Coalescing gaps remain unused staged bytes.
    pub(crate) fn staged_window_useful_bytes(
        &self,
        row_group_indexes: &[usize],
        staged_ranges: &[Range<u64>],
    ) -> BTreeMap<usize, usize> {
        row_group_indexes
            .iter()
            .copied()
            .map(|row_group_index| {
                let bytes = self
                    .ranges_by_row_group
                    .get(&row_group_index)
                    .into_iter()
                    .flatten()
                    .flat_map(|candidate| {
                        staged_ranges.iter().filter_map(move |staged| {
                            let start = candidate.start.max(staged.start);
                            let end = candidate.end.min(staged.end);
                            (start < end).then_some(start..end)
                        })
                    });
                (row_group_index, unique_range_bytes(bytes))
            })
            .collect()
    }
}

/// The density-gated prefetch policy state for one decoder run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdmissionState {
    /// More exact coverage is required before deciding.
    Observing,
    /// Candidate spans have enough dense exact coverage to permit prefetching.
    Enabled,
    /// Candidate spans have enough sparse exact coverage to deny prefetching.
    Disabled,
}

/// A terminal density-admission transition, emitted at most once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdmissionDecision {
    /// Candidate spans have sufficient dense exact coverage.
    Enabled,
    /// Candidate spans have sufficient sparse exact coverage.
    Denied,
}

/// Tracks unique exact coverage against the candidate projected spans.
#[derive(Debug, Clone)]
pub(crate) struct DensityAdmission {
    candidate_ranges: Vec<Range<u64>>,
    observed_ranges: Vec<Range<u64>>,
    candidate_payload_bytes: usize,
    observed_payload_bytes: usize,
    state: AdmissionState,
    admission_metrics_recorded: bool,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Task 5 drives density admission.")
)]
impl DensityAdmission {
    fn new(candidate_ranges: Vec<Range<u64>>) -> Self {
        let candidate_payload_bytes =
            unique_range_bytes(candidate_ranges.iter().cloned());
        Self {
            candidate_ranges,
            observed_ranges: Vec::new(),
            candidate_payload_bytes,
            observed_payload_bytes: 0,
            state: AdmissionState::Observing,
            admission_metrics_recorded: false,
        }
    }

    /// Adds exact decoder requests and returns a terminal transition once.
    pub(crate) fn observe_exact_ranges(
        &mut self,
        exact_ranges: &[Range<u64>],
    ) -> Option<AdmissionDecision> {
        if self.state != AdmissionState::Observing {
            return None;
        }

        for exact_range in exact_ranges {
            for candidate_range in &self.candidate_ranges {
                let start = exact_range.start.max(candidate_range.start);
                let end = exact_range.end.min(candidate_range.end);
                if start < end {
                    self.observed_ranges.push(start..end);
                }
            }
        }
        self.observed_payload_bytes =
            unique_range_bytes(self.observed_ranges.iter().cloned());

        if self.observed_payload_bytes < MIN_ADMISSION_BYTES
            || self.candidate_payload_bytes == 0
        {
            return None;
        }

        let observed = self.observed_payload_bytes as u128;
        let candidate = self.candidate_payload_bytes as u128;
        if observed * 5 >= candidate * 4 {
            self.state = AdmissionState::Enabled;
            Some(AdmissionDecision::Enabled)
        } else if observed * 2 < candidate {
            self.state = AdmissionState::Disabled;
            Some(AdmissionDecision::Denied)
        } else {
            None
        }
    }

    /// Returns the unique exact bytes observed within the candidate spans.
    pub(crate) fn observed_payload_bytes(&self) -> usize {
        self.observed_payload_bytes
    }

    /// Returns the unique candidate bytes used as the density denominator.
    pub(crate) fn candidate_payload_bytes(&self) -> usize {
        self.candidate_payload_bytes
    }

    /// Returns the current policy state.
    pub(crate) fn state(&self) -> AdmissionState {
        self.state
    }

    /// Records the terminal admission decision at most once for this instance.
    pub(crate) fn record_admission_metrics(
        &mut self,
        metrics: &RowGroupPrefetchMetrics,
    ) -> Option<AdmissionDecision> {
        if self.admission_metrics_recorded {
            return None;
        }
        let decision = match self.state {
            AdmissionState::Observing => return None,
            AdmissionState::Enabled => AdmissionDecision::Enabled,
            AdmissionState::Disabled => AdmissionDecision::Denied,
        };
        metrics.record_admission(decision);
        self.admission_metrics_recorded = true;
        Some(decision)
    }
}

/// Per-file execution metrics for row-group prefetch planning and staging.
#[derive(Debug, Clone)]
pub(crate) struct RowGroupPrefetchMetrics {
    observed_exact_bytes: Count,
    candidate_bytes: Count,
    prefetch_windows: Count,
    prefetched_ranges: Count,
    prefetched_bytes: Count,
    useful_staged_bytes: Count,
    unused_staged_bytes: Count,
    admission_enables: Count,
    admission_denials: Count,
    peak_staged_bytes: Gauge,
}

impl RowGroupPrefetchMetrics {
    /// Registers file-scoped prefetch metrics for one execution partition.
    pub(crate) fn new(
        partition: usize,
        filename: &str,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Self {
        let bytes_counter = |name| {
            MetricBuilder::new(metrics)
                .with_new_label("filename", filename.to_string())
                .with_category(MetricCategory::Bytes)
                .counter(name, partition)
        };
        let counter = |name| {
            MetricBuilder::new(metrics)
                .with_new_label("filename", filename.to_string())
                .counter(name, partition)
        };
        let peak_staged_bytes = MetricBuilder::new(metrics)
            .with_new_label("filename", filename.to_string())
            .with_category(MetricCategory::Bytes)
            .gauge("prefetch_peak_staged_bytes", partition);

        Self {
            observed_exact_bytes: bytes_counter("prefetch_observed_exact_bytes"),
            candidate_bytes: bytes_counter("prefetch_candidate_bytes"),
            prefetch_windows: counter("prefetch_windows"),
            prefetched_ranges: counter("prefetched_ranges"),
            prefetched_bytes: bytes_counter("prefetched_bytes"),
            useful_staged_bytes: bytes_counter("useful_staged_bytes"),
            unused_staged_bytes: bytes_counter("unused_staged_bytes"),
            admission_enables: counter("prefetch_admission_enables"),
            admission_denials: counter("prefetch_admission_denials"),
            peak_staged_bytes,
        }
    }

    pub(crate) fn record_observed_exact_bytes(&self, bytes: usize) {
        self.observed_exact_bytes.add(bytes);
    }

    pub(crate) fn record_candidate_bytes(&self, bytes: usize) {
        self.candidate_bytes.add(bytes);
    }

    pub(crate) fn record_prefetch(&self, range_count: usize, bytes: usize) {
        self.prefetch_windows.add(1);
        self.prefetched_ranges.add(range_count);
        self.prefetched_bytes.add(bytes);
    }

    pub(crate) fn record_useful_staged_bytes(&self, bytes: usize) {
        self.useful_staged_bytes.add(bytes);
    }

    pub(crate) fn record_unused_staged_bytes(&self, bytes: usize) {
        self.unused_staged_bytes.add(bytes);
    }

    fn record_admission(&self, decision: AdmissionDecision) {
        match decision {
            AdmissionDecision::Enabled => self.admission_enables.add(1),
            AdmissionDecision::Denied => self.admission_denials.add(1),
        }
    }

    pub(crate) fn record_peak_staged_bytes(&self, bytes: usize) {
        self.peak_staged_bytes.set_max(bytes);
    }
}

fn merge_ranges(ranges: Vec<Range<u64>>) -> Vec<Range<u64>> {
    let mut merged = Vec::with_capacity(ranges.len());
    for range in ranges {
        let Some(previous) = merged.last_mut() else {
            merged.push(range);
            continue;
        };

        let gap = range.start.saturating_sub(previous.end);
        let merged_end = previous.end.max(range.end);
        let merged_span = merged_end.saturating_sub(previous.start);
        if gap <= MAX_MERGE_GAP && merged_span <= MAX_MERGED_RANGE {
            previous.end = merged_end;
        } else {
            merged.push(range);
        }
    }
    merged
}

fn unique_range_bytes(ranges: impl IntoIterator<Item = Range<u64>>) -> usize {
    merge_ranges_without_gap(ranges.into_iter().collect())
        .into_iter()
        .map(|range| range.end.saturating_sub(range.start))
        .try_fold(0usize, |total, length| {
            usize::try_from(length)
                .ok()
                .and_then(|length| total.checked_add(length))
        })
        .unwrap_or(usize::MAX)
}

fn merge_ranges_without_gap(mut ranges: Vec<Range<u64>>) -> Vec<Range<u64>> {
    ranges.sort_unstable_by_key(|range| (range.start, range.end));
    let mut merged = Vec::with_capacity(ranges.len());
    for range in ranges {
        let Some(previous) = merged.last_mut() else {
            merged.push(range);
            continue;
        };
        if range.start <= previous.end {
            previous.end = previous.end.max(range.end);
        } else {
            merged.push(range);
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use parquet::arrow::ProjectionMask;
    use parquet::basic::Type as PhysicalType;
    use parquet::file::metadata::{
        ColumnChunkMetaData, FileMetaData, ParquetMetaData, RowGroupMetaData,
    };
    use parquet::schema::types::{SchemaDescPtr, SchemaDescriptor, Type};

    use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;

    use super::{
        AdmissionDecision, AdmissionState, DensityAdmission, MAX_MERGE_GAP,
        MAX_MERGED_RANGE, RowGroupPrefetchMetrics, RowGroupPrefetchPlan,
    };

    #[test]
    fn extracts_only_projected_leaf_ranges_from_four_row_groups() {
        let metadata = metadata();
        let projection_mask =
            ProjectionMask::leaves(metadata.file_metadata().schema_descr(), [0, 2]);
        let plan =
            RowGroupPrefetchPlan::new(&metadata, &projection_mask, vec![0, 1, 2, 3]);

        assert_eq!(plan.row_group_order(), &[0, 1, 2, 3]);
        assert_eq!(plan.projected_payload_bytes(), 800);
        assert_eq!(plan.ranges_for(&[0]), vec![0..300],);
    }

    #[test]
    fn merges_only_forward_ranges_within_gap_and_span_limits() {
        let exact_gap = metadata_with_leaf_ranges(vec![
            0..1,
            (1 + MAX_MERGE_GAP)..(2 + MAX_MERGE_GAP),
        ]);
        let plan = leaf_plan(&exact_gap, vec![0, 1]);
        assert_eq!(plan.ranges_for(&[1, 0]), vec![0..(2 + MAX_MERGE_GAP)]);

        let over_gap = metadata_with_leaf_ranges(vec![
            0..1,
            (2 + MAX_MERGE_GAP)..(3 + MAX_MERGE_GAP),
        ]);
        let plan = leaf_plan(&over_gap, vec![0, 1]);
        assert_eq!(
            plan.ranges_for(&[0, 1]),
            vec![0..1, (2 + MAX_MERGE_GAP)..(3 + MAX_MERGE_GAP)],
        );

        let first_end = MAX_MERGED_RANGE - MAX_MERGE_GAP - 1;
        let at_max = metadata_with_leaf_ranges(vec![
            0..first_end,
            (first_end + MAX_MERGE_GAP)..MAX_MERGED_RANGE,
        ]);
        let plan = leaf_plan(&at_max, vec![0, 1]);
        assert_eq!(plan.ranges_for(&[0, 1]), vec![0..MAX_MERGED_RANGE]);

        let over_max = metadata_with_leaf_ranges(vec![
            0..first_end,
            (first_end + MAX_MERGE_GAP)..(MAX_MERGED_RANGE + 1),
        ]);
        let plan = leaf_plan(&over_max, vec![0, 1]);
        assert_eq!(
            plan.ranges_for(&[0, 1]),
            vec![
                0..first_end,
                (first_end + MAX_MERGE_GAP)..(MAX_MERGED_RANGE + 1),
            ],
        );
        assert!(
            plan.ranges_for(&[0, 1])
                .iter()
                .all(|range| range.end - range.start <= MAX_MERGED_RANGE)
        );
    }

    #[test]
    fn keeps_reverse_decoder_order_but_returns_sorted_physical_ranges() {
        let metadata = metadata_with_leaf_ranges(vec![
            4_000_000..4_000_100,
            3_000_000..3_000_100,
            2_000_000..2_000_100,
            1_000_000..1_000_100,
        ]);
        let plan = leaf_plan(&metadata, vec![3, 2, 1, 0]);

        assert_eq!(plan.row_group_order(), &[3, 2, 1, 0]);
        assert_eq!(
            plan.ranges_for(&[3, 2, 1, 0]),
            vec![
                1_000_000..1_000_100,
                2_000_000..2_000_100,
                3_000_000..3_000_100,
                4_000_000..4_000_100,
            ],
        );
    }

    #[test]
    fn ignores_unknown_and_duplicate_row_groups_without_inflating_coverage() {
        let metadata = metadata_with_leaf_ranges(vec![0..1_048_576, 524_288..1_572_864]);
        let plan = leaf_plan(&metadata, vec![0, 1, 0, 99]);

        assert_eq!(plan.row_group_order(), &[0, 1, 0, 99]);
        assert_eq!(plan.projected_payload_bytes(), 1_572_864);
        assert_eq!(plan.ranges_for(&[]), Vec::<std::ops::Range<u64>>::new());
        assert_eq!(plan.ranges_for(&[1, 0, 1, 99]), vec![0..1_572_864]);
    }

    #[test]
    fn preserves_an_individual_projected_chunk_larger_than_merge_limit() {
        let metadata = metadata_with_leaf_ranges(
            std::iter::once(0..(MAX_MERGED_RANGE + 1)).collect(),
        );
        let plan = leaf_plan(&metadata, vec![0]);

        assert_eq!(plan.ranges_for(&[0]), vec![0..(MAX_MERGED_RANGE + 1)]);
    }

    #[test]
    fn density_requires_one_mib_and_obeys_enable_and_disable_boundaries() {
        const MIB: u64 = 1024 * 1024;

        let dense_metadata = metadata_with_leaf_ranges(single_range(0..(MIB * 5 / 4)));
        let dense_plan = leaf_plan(&dense_metadata, vec![0]);
        let mut admission = dense_plan.density_admission();
        assert_eq!(observe_single(&mut admission, 0..(MIB - 1)), None);
        assert_eq!(observe_single(&mut admission, 0..(MIB - 1)), None);
        assert_eq!(admission.observed_payload_bytes(), (MIB - 1) as usize);
        assert_eq!(
            observe_single(&mut admission, (MIB - 1)..MIB),
            Some(AdmissionDecision::Enabled)
        );
        assert_eq!(observe_single(&mut admission, 0..MIB), None);
        assert_eq!(admission.observed_payload_bytes(), MIB as usize);
        assert_eq!(admission.candidate_payload_bytes(), (MIB * 5 / 4) as usize);
        assert_eq!(admission.state(), AdmissionState::Enabled);

        let half_metadata = metadata_with_leaf_ranges(single_range(0..(MIB * 2)));
        let half_plan = leaf_plan(&half_metadata, vec![0]);
        let mut admission = half_plan.density_admission();
        assert_eq!(observe_single(&mut admission, 0..MIB), None);

        let sparse_metadata = metadata_with_leaf_ranges(single_range(0..(MIB * 2 + 1)));
        let sparse_plan = leaf_plan(&sparse_metadata, vec![0]);
        let mut admission = sparse_plan.density_admission();
        assert_eq!(
            observe_single(&mut admission, 0..MIB),
            Some(AdmissionDecision::Denied)
        );
    }

    #[test]
    fn density_is_scoped_to_the_exact_row_group_and_windows_follow_plan_order() {
        const MIB: u64 = 1024 * 1024;
        let metadata = metadata_with_leaf_ranges(vec![
            0..(MIB * 5 / 4),
            (MIB * 2)..(MIB * 3),
            (MIB * 4)..(MIB * 5),
            (MIB * 6)..(MIB * 7),
        ]);
        let plan = leaf_plan(&metadata, vec![3, 2, 1, 0]);
        let mut admission = plan.density_admission_for(3);

        assert_eq!(
            observe_single(&mut admission, (MIB * 6)..(MIB * 7)),
            Some(AdmissionDecision::Enabled)
        );
        assert_eq!(plan.next_window_after(3, 2, &BTreeSet::new()), vec![2, 1]);
        assert_eq!(plan.next_window_after(2, 4, &BTreeSet::from([1])), vec![0]);
    }

    #[test]
    fn prefetch_metrics_are_file_scoped_and_peak_is_not_reset() {
        let metrics_set = ExecutionPlanMetricsSet::new();
        let first = RowGroupPrefetchMetrics::new(3, "first.parquet", &metrics_set);
        let second = RowGroupPrefetchMetrics::new(4, "second.parquet", &metrics_set);

        first.record_observed_exact_bytes(10);
        first.record_candidate_bytes(20);
        first.record_prefetch(2, 30);
        first.record_useful_staged_bytes(8);
        first.record_unused_staged_bytes(22);
        const MIB: u64 = 1024 * 1024;
        let mut enabled_admission = DensityAdmission::new(single_range(0..MIB));
        assert_eq!(
            observe_single(&mut enabled_admission, 0..MIB),
            Some(AdmissionDecision::Enabled)
        );
        assert_eq!(
            enabled_admission.record_admission_metrics(&first),
            Some(AdmissionDecision::Enabled)
        );
        assert_eq!(enabled_admission.record_admission_metrics(&first), None);

        let mut denied_admission = DensityAdmission::new(single_range(0..(MIB * 2 + 1)));
        assert_eq!(
            observe_single(&mut denied_admission, 0..MIB),
            Some(AdmissionDecision::Denied)
        );
        assert_eq!(
            denied_admission.record_admission_metrics(&second),
            Some(AdmissionDecision::Denied)
        );
        assert_eq!(denied_admission.record_admission_metrics(&second), None);
        first.record_peak_staged_bytes(30);
        first.record_peak_staged_bytes(10);

        assert_eq!(first.observed_exact_bytes.value(), 10);
        assert_eq!(first.candidate_bytes.value(), 20);
        assert_eq!(first.prefetch_windows.value(), 1);
        assert_eq!(first.prefetched_ranges.value(), 2);
        assert_eq!(first.prefetched_bytes.value(), 30);
        assert_eq!(first.useful_staged_bytes.value(), 8);
        assert_eq!(first.unused_staged_bytes.value(), 22);
        assert_eq!(first.admission_enables.value(), 1);
        assert_eq!(first.admission_denials.value(), 0);
        assert_eq!(second.admission_enables.value(), 0);
        assert_eq!(second.admission_denials.value(), 1);
        assert_eq!(first.peak_staged_bytes.value(), 30);

        let rendered = metrics_set
            .clone_inner()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("filename=first.parquet"));
        assert!(rendered.contains("filename=second.parquet"));
        assert!(rendered.contains("partition=3"));
        assert!(rendered.contains("partition=4"));
    }

    fn metadata() -> ParquetMetaData {
        let schema = schema(3);
        let row_groups = (0..4)
            .map(|row_group| {
                let base = (row_group * 1_000) as i64;
                let columns = (0..3)
                    .map(|leaf| {
                        ColumnChunkMetaData::builder(schema.column(leaf))
                            .set_num_values(10)
                            .set_total_compressed_size(100)
                            .set_data_page_offset(base + (leaf * 100) as i64)
                            .build()
                            .unwrap()
                    })
                    .collect();
                RowGroupMetaData::builder(Arc::clone(&schema))
                    .set_num_rows(10)
                    .set_column_metadata(columns)
                    .build()
                    .unwrap()
            })
            .collect();
        let file_metadata = FileMetaData::new(1, 40, None, None, schema, None);
        ParquetMetaData::new(file_metadata, row_groups)
    }

    fn leaf_plan(
        metadata: &ParquetMetaData,
        row_group_order: Vec<usize>,
    ) -> RowGroupPrefetchPlan {
        let projection_mask =
            ProjectionMask::leaves(metadata.file_metadata().schema_descr(), [0]);
        RowGroupPrefetchPlan::new(metadata, &projection_mask, row_group_order)
    }

    fn observe_single(
        admission: &mut DensityAdmission,
        range: std::ops::Range<u64>,
    ) -> Option<AdmissionDecision> {
        admission.observe_exact_ranges(&[range])
    }

    fn single_range(range: std::ops::Range<u64>) -> Vec<std::ops::Range<u64>> {
        std::iter::once(range).collect()
    }

    fn metadata_with_leaf_ranges(ranges: Vec<std::ops::Range<u64>>) -> ParquetMetaData {
        let schema = schema(1);
        let row_count = ranges.len() as i64;
        let row_groups = ranges
            .into_iter()
            .map(|range| {
                let length = i64::try_from(range.end - range.start).unwrap();
                let offset = i64::try_from(range.start).unwrap();
                let column = ColumnChunkMetaData::builder(schema.column(0))
                    .set_num_values(10)
                    .set_total_compressed_size(length)
                    .set_data_page_offset(offset)
                    .build()
                    .unwrap();
                RowGroupMetaData::builder(Arc::clone(&schema))
                    .set_num_rows(10)
                    .set_column_metadata(vec![column])
                    .build()
                    .unwrap()
            })
            .collect();
        let file_metadata =
            FileMetaData::new(1, row_count * 10, None, None, schema, None);
        ParquetMetaData::new(file_metadata, row_groups)
    }

    fn schema(leaf_count: usize) -> SchemaDescPtr {
        let fields = (0..leaf_count)
            .map(|index| {
                Arc::new(
                    Type::primitive_type_builder(
                        &format!("leaf_{index}"),
                        PhysicalType::INT32,
                    )
                    .build()
                    .unwrap(),
                )
            })
            .collect();
        let schema = Type::group_type_builder("schema")
            .with_fields(fields)
            .build()
            .unwrap();
        Arc::new(SchemaDescriptor::new(Arc::new(schema)))
    }
}
