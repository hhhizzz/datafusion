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

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;

use arrow::array::RecordBatch;
use datafusion_common::{DataFusionError, Result};
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use parquet::DecodeResult;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::push_decoder::ParquetPushDecoder;
use tokio::sync::OwnedSemaphorePermit;

use super::PushDecoderOutputState;
use crate::lookahead::{
    LookaheadFileContext, MAX_IN_FLIGHT_RANGES, MAX_RANGES_PER_FILE_FETCH,
    SpeculativeLease,
};
use crate::row_group_prefetch::{
    AdmissionDecision, RowGroupPrefetchMetrics, RowGroupPrefetchPlan,
};

/// A bounded push-decoder driver that overlaps upcoming row-group fetches with
/// synchronous decoding of the current row group.
pub(crate) struct LookaheadPushDecoderStreamState {
    decoder: Option<ParquetPushDecoder>,
    pending_decoders: VecDeque<ParquetPushDecoder>,
    active_prefetch_plan: Option<Arc<Mutex<PrefetchRunState>>>,
    pending_prefetch_plans: VecDeque<Arc<Mutex<PrefetchRunState>>>,
    reader: Option<Box<dyn AsyncFileReader>>,
    output: PushDecoderOutputState,
    lookahead: LookaheadFileContext,
    active_reader: Option<ParquetRecordBatchReader>,
    next_reader_future: Option<BoxFuture<'static, NextReaderOutcome>>,
    prefetched_readers: VecDeque<PrefetchedReader>,
    foreground_resources: SpeculativeResources,
    deferred_error: Option<DataFusionError>,
    run_finished: bool,
    speculation_disabled: bool,
    terminated: bool,
}

/// All prefetch state that advances in lockstep with decoder runs.
pub(crate) struct PrefetchPlanQueue {
    pub(crate) active: RowGroupPrefetchPlan,
    pub(crate) pending: VecDeque<RowGroupPrefetchPlan>,
    pub(crate) metrics: RowGroupPrefetchMetrics,
    pub(crate) staging_enabled: bool,
}

struct PrefetchedReader {
    reader: ParquetRecordBatchReader,
    resources: SpeculativeResources,
    staged_window: Option<StagedWindowFinalResources>,
}

#[derive(Default)]
struct SpeculativeResources {
    leases: Vec<SpeculativeLease>,
    range_permits: Vec<OwnedSemaphorePermit>,
}

struct StagedByteTracker {
    staged_bytes: AtomicUsize,
    metrics: RowGroupPrefetchMetrics,
}

impl StagedByteTracker {
    fn reserve(self: &Arc<Self>, bytes: usize) -> StagedReservation {
        let staged_bytes = self.staged_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.metrics
            .record_peak_staged_bytes(staged_bytes.saturating_add(bytes));
        StagedReservation {
            tracker: Arc::clone(self),
            bytes,
        }
    }
}

struct StagedReservation {
    tracker: Arc<StagedByteTracker>,
    bytes: usize,
}

impl Drop for StagedReservation {
    fn drop(&mut self) {
        self.tracker
            .staged_bytes
            .fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// The complete reservation for one non-overlapping staged row-group window.
struct StagedWindowResources {
    remaining_row_groups: VecDeque<usize>,
    resources: SpeculativeResources,
    staged_bytes: usize,
    useful_bytes_by_row_group: BTreeMap<usize, usize>,
    useful_staged_bytes: usize,
    reservation: StagedReservation,
}

struct StagedWindowFinalResources {
    resources: SpeculativeResources,
    unused_staged_bytes: usize,
    metrics: RowGroupPrefetchMetrics,
    _reservation: StagedReservation,
}

impl Drop for StagedWindowFinalResources {
    fn drop(&mut self) {
        self.metrics
            .record_unused_staged_bytes(self.unused_staged_bytes);
        self.resources.leases.clear();
        self.resources.range_permits.clear();
    }
}

struct StagingRequest {
    row_groups: Vec<usize>,
    ranges: Vec<std::ops::Range<u64>>,
    staged_bytes: usize,
    useful_bytes_by_row_group: BTreeMap<usize, usize>,
}

struct PrefetchRunState {
    plan: RowGroupPrefetchPlan,
    metrics: RowGroupPrefetchMetrics,
    window: usize,
    staging_enabled: bool,
    admissions: BTreeMap<usize, crate::row_group_prefetch::DensityAdmission>,
    claimed_row_groups: BTreeSet<usize>,
    staged_windows: VecDeque<StagedWindowResources>,
    tracker: Arc<StagedByteTracker>,
}

impl PrefetchRunState {
    fn new(
        plan: RowGroupPrefetchPlan,
        metrics: RowGroupPrefetchMetrics,
        window: usize,
        staging_enabled: bool,
    ) -> Self {
        Self {
            plan,
            metrics: metrics.clone(),
            window,
            staging_enabled: staging_enabled && window > 0,
            admissions: BTreeMap::new(),
            claimed_row_groups: BTreeSet::new(),
            staged_windows: VecDeque::new(),
            tracker: Arc::new(StagedByteTracker {
                staged_bytes: AtomicUsize::new(0),
                metrics,
            }),
        }
    }

    fn plan(&self) -> &RowGroupPrefetchPlan {
        &self.plan
    }

    fn observe_exact_ranges(
        &mut self,
        row_group_index: usize,
        exact_ranges: &[std::ops::Range<u64>],
    ) -> Option<StagingRequest> {
        if !self.staging_enabled {
            return None;
        }

        let (observed_bytes, decision) = {
            let admission = self
                .admissions
                .entry(row_group_index)
                .or_insert_with(|| self.plan.density_admission_for(row_group_index));
            let before = admission.observed_payload_bytes();
            let decision = admission.observe_exact_ranges(exact_ranges);
            (
                admission.observed_payload_bytes().saturating_sub(before),
                decision,
            )
        };
        self.metrics.record_observed_exact_bytes(observed_bytes);

        if decision.is_some() {
            self.admissions
                .get_mut(&row_group_index)
                .expect("row-group admission inserted above")
                .record_admission_metrics(&self.metrics);
        }
        if decision != Some(AdmissionDecision::Enabled) {
            return None;
        }

        let row_groups = self.plan.next_window_after(
            row_group_index,
            self.window,
            &self.claimed_row_groups,
        );
        if row_groups.is_empty() {
            return None;
        }
        self.claimed_row_groups.extend(row_groups.iter().copied());

        let ranges = self.plan.ranges_for(&row_groups);
        let staged_bytes = checked_range_bytes(&ranges)?;
        let useful_bytes_by_row_group =
            self.plan.staged_window_useful_bytes(&row_groups, &ranges);
        Some(StagingRequest {
            row_groups,
            ranges,
            staged_bytes,
            useful_bytes_by_row_group,
        })
    }

    fn add_staged_window(
        &mut self,
        request: StagingRequest,
        resources: SpeculativeResources,
    ) {
        self.metrics
            .record_prefetch(request.ranges.len(), request.staged_bytes);
        self.staged_windows.push_back(StagedWindowResources {
            remaining_row_groups: request.row_groups.into(),
            resources,
            staged_bytes: request.staged_bytes,
            useful_bytes_by_row_group: request.useful_bytes_by_row_group,
            useful_staged_bytes: 0,
            reservation: self.tracker.reserve(request.staged_bytes),
        });
    }

    fn reader_ready(
        &mut self,
        row_group_index: usize,
    ) -> Option<StagedWindowFinalResources> {
        let window = self.staged_windows.front_mut()?;
        if window.remaining_row_groups.front().copied() != Some(row_group_index) {
            return None;
        }
        window.remaining_row_groups.pop_front();
        let useful_bytes = window
            .useful_bytes_by_row_group
            .remove(&row_group_index)
            .unwrap_or_default();
        window.useful_staged_bytes = window
            .useful_staged_bytes
            .saturating_add(useful_bytes)
            .min(window.staged_bytes);
        self.metrics.record_useful_staged_bytes(useful_bytes);

        if !window.remaining_row_groups.is_empty() {
            return None;
        }
        let window = self
            .staged_windows
            .pop_front()
            .expect("front staged window checked above");
        Some(StagedWindowFinalResources {
            resources: window.resources,
            unused_staged_bytes: window
                .staged_bytes
                .saturating_sub(window.useful_staged_bytes),
            metrics: self.metrics.clone(),
            _reservation: window.reservation,
        })
    }

    fn discard_staged_windows(&mut self) {
        for window in self.staged_windows.drain(..) {
            self.metrics.record_unused_staged_bytes(
                window
                    .staged_bytes
                    .saturating_sub(window.useful_staged_bytes),
            );
        }
    }
}

impl Drop for PrefetchRunState {
    fn drop(&mut self) {
        self.discard_staged_windows();
    }
}

struct NextReaderOutcome {
    decoder: ParquetPushDecoder,
    reader: Box<dyn AsyncFileReader>,
    result: NextReaderResult,
    resources: SpeculativeResources,
}

enum NextReaderResult {
    Reader {
        reader: ParquetRecordBatchReader,
        staged_window: Option<StagedWindowFinalResources>,
    },
    Finished,
    Denied,
    Error(DataFusionError),
}

enum ForegroundProgress {
    ReaderReady,
    RunFinished,
}

impl LookaheadPushDecoderStreamState {
    pub(crate) fn new(
        decoder: ParquetPushDecoder,
        pending_decoders: VecDeque<ParquetPushDecoder>,
        reader: Box<dyn AsyncFileReader>,
        output: PushDecoderOutputState,
        lookahead: LookaheadFileContext,
    ) -> Self {
        Self {
            decoder: Some(decoder),
            pending_decoders,
            active_prefetch_plan: None,
            pending_prefetch_plans: VecDeque::new(),
            reader: Some(reader),
            output,
            lookahead,
            active_reader: None,
            next_reader_future: None,
            prefetched_readers: VecDeque::new(),
            foreground_resources: SpeculativeResources::default(),
            deferred_error: None,
            run_finished: false,
            speculation_disabled: false,
            terminated: false,
        }
    }

    /// Creates a lookahead state with one prefetch plan for every decoder run.
    pub(crate) fn new_with_prefetch_plans(
        decoder: ParquetPushDecoder,
        pending_decoders: VecDeque<ParquetPushDecoder>,
        prefetch_plans: PrefetchPlanQueue,
        reader: Box<dyn AsyncFileReader>,
        output: PushDecoderOutputState,
        lookahead: LookaheadFileContext,
    ) -> Result<Self> {
        if pending_decoders.len() != prefetch_plans.pending.len() {
            return Err(prefetch_plan_queue_drift());
        }
        let PrefetchPlanQueue {
            active,
            pending,
            metrics,
            staging_enabled,
        } = prefetch_plans;
        let window = lookahead.prefetch_window();
        let pending_metrics = metrics.clone();
        let mut state = Self::new(decoder, pending_decoders, reader, output, lookahead);
        state.active_prefetch_plan = Some(Arc::new(Mutex::new(PrefetchRunState::new(
            active,
            metrics,
            window,
            staging_enabled,
        ))));
        state.pending_prefetch_plans = pending
            .into_iter()
            .map(|plan| {
                Arc::new(Mutex::new(PrefetchRunState::new(
                    plan,
                    pending_metrics.clone(),
                    window,
                    staging_enabled,
                )))
            })
            .collect();
        Ok(state)
    }

    /// Returns the plan paired with the active decoder run for Task 5.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Task 5 reads the active prefetch plan.")
    )]
    pub(crate) fn active_prefetch_plan(&self) -> Option<RowGroupPrefetchPlan> {
        self.active_prefetch_plan.as_ref().map(|state| {
            state
                .lock()
                .expect("prefetch run state must not be poisoned")
                .plan()
                .clone()
        })
    }

    pub(crate) fn into_stream(self) -> BoxStream<'static, Result<RecordBatch>> {
        futures::stream::unfold(self, |state| async move { state.transition().await })
            .fuse()
            .boxed()
    }

    async fn transition(mut self) -> Option<(Result<RecordBatch>, Self)> {
        loop {
            if self.terminated {
                return None;
            }
            if self.output.limit_reached() {
                self.terminate();
                return None;
            }

            if self.active_reader.is_none() {
                if let Some(prefetched) = self.prefetched_readers.pop_front() {
                    let PrefetchedReader {
                        reader,
                        resources,
                        staged_window,
                    } = prefetched;
                    self.active_reader = Some(reader);
                    drop(resources);
                    drop(staged_window);
                    continue;
                }

                if let Some(error) = self.deferred_error.take() {
                    self.terminate();
                    return Some((Err(error), self));
                }

                if self.run_finished {
                    match self.advance_decoder_run() {
                        Ok(true) => continue,
                        Ok(false) => {
                            self.terminate();
                            return None;
                        }
                        Err(error) => {
                            self.terminate();
                            return Some((Err(error), self));
                        }
                    }
                }

                if let Some(future) = self.next_reader_future.take() {
                    let outcome = future.await;
                    self.accept_next_reader_outcome(outcome);
                    continue;
                }

                match self.load_foreground_reader().await {
                    Ok(ForegroundProgress::ReaderReady) => continue,
                    Ok(ForegroundProgress::RunFinished) => continue,
                    Err(error) => {
                        self.terminate();
                        return Some((Err(error), self));
                    }
                }
            }

            self.start_speculation();
            if let Some(future) = self.next_reader_future.as_mut()
                && let Some(outcome) = poll_once(future).await
            {
                self.next_reader_future = None;
                self.accept_next_reader_outcome(outcome);
            }

            let next_batch = self
                .active_reader
                .as_mut()
                .expect("active reader checked above")
                .next();
            match next_batch {
                Some(Ok(batch)) => match self.output.finalize_batch(batch) {
                    Ok(batch) => {
                        if self.output.limit_reached() {
                            self.terminate();
                        }
                        return Some((Ok(batch), self));
                    }
                    Err(error) => {
                        self.terminate();
                        return Some((Err(error), self));
                    }
                },
                Some(Err(error)) => {
                    self.terminate();
                    return Some((Err(DataFusionError::from(error)), self));
                }
                None => {
                    self.active_reader = None;
                }
            }
        }
    }

    fn start_speculation(&mut self) {
        if self.speculation_disabled
            || self.next_reader_future.is_some()
            || self.prefetched_readers.len() >= self.lookahead.depth()
            || self.run_finished
            || self.deferred_error.is_some()
            || self.active_reader.is_none()
        {
            return;
        }

        let decoder = self
            .decoder
            .take()
            .expect("decoder is owned by state when no future exists");
        let reader = self
            .reader
            .take()
            .expect("reader is owned by state when no future exists");
        self.next_reader_future = Some(
            drive_speculative_next_reader(
                decoder,
                reader,
                self.lookahead.clone(),
                self.active_prefetch_plan.clone(),
            )
            .boxed(),
        );
    }

    fn accept_next_reader_outcome(&mut self, outcome: NextReaderOutcome) {
        let NextReaderOutcome {
            decoder,
            reader,
            result,
            resources,
        } = outcome;
        match result {
            NextReaderResult::Reader {
                reader: next_reader,
                staged_window,
            } => {
                self.decoder = Some(decoder);
                self.reader = Some(reader);
                self.prefetched_readers.push_back(PrefetchedReader {
                    reader: next_reader,
                    resources,
                    staged_window,
                });
            }
            NextReaderResult::Finished => {
                self.decoder = Some(decoder);
                self.reader = Some(reader);
                self.run_finished = true;
                drop(resources);
            }
            NextReaderResult::Denied => {
                self.decoder = Some(decoder);
                self.reader = Some(reader);
                self.foreground_resources = resources;
                self.speculation_disabled = true;
            }
            NextReaderResult::Error(error) => {
                self.deferred_error = Some(error);
                self.speculation_disabled = true;
                drop(decoder);
                drop(reader);
                drop(resources);
            }
        }
    }

    async fn load_foreground_reader(&mut self) -> Result<ForegroundProgress> {
        loop {
            let next_row_group = self
                .decoder
                .as_ref()
                .expect("foreground decoder must be present")
                .peek_next_row_group()
                .map_err(DataFusionError::from)?;
            let decode_result = self
                .decoder
                .as_mut()
                .expect("foreground decoder must be present")
                .try_next_reader()
                .map_err(DataFusionError::from)?;
            match decode_result {
                DecodeResult::NeedsData(ranges) => {
                    if let Some(row_group_index) = next_row_group {
                        stage_admitted_window(
                            self.decoder
                                .as_mut()
                                .expect("foreground decoder must be present"),
                            self.reader
                                .as_mut()
                                .expect("foreground reader must be present"),
                            &self.lookahead,
                            self.active_prefetch_plan.as_ref(),
                            row_group_index,
                            &ranges,
                        )
                        .await?;
                    }
                    let data = self
                        .reader
                        .as_mut()
                        .expect("foreground reader must be present")
                        .get_byte_ranges(ranges.clone())
                        .await
                        .map_err(DataFusionError::from)?;
                    self.decoder
                        .as_mut()
                        .expect("foreground decoder must be present")
                        .push_ranges(ranges, data)
                        .map_err(DataFusionError::from)?;
                }
                DecodeResult::Data(reader) => {
                    let staged_window = next_row_group.and_then(|row_group_index| {
                        self.active_prefetch_plan.as_ref().and_then(|state| {
                            state
                                .lock()
                                .expect("prefetch run state must not be poisoned")
                                .reader_ready(row_group_index)
                        })
                    });
                    self.active_reader = Some(reader);
                    self.foreground_resources = SpeculativeResources::default();
                    drop(staged_window);
                    return Ok(ForegroundProgress::ReaderReady);
                }
                DecodeResult::Finished => {
                    self.run_finished = true;
                    self.foreground_resources = SpeculativeResources::default();
                    return Ok(ForegroundProgress::RunFinished);
                }
            }
        }
    }

    fn advance_decoder_run(&mut self) -> Result<bool> {
        self.run_finished = false;
        if self.active_prefetch_plan.is_none() {
            if !self.pending_prefetch_plans.is_empty() {
                return Err(prefetch_plan_queue_drift());
            }
            let Some(decoder) = self.pending_decoders.pop_front() else {
                return Ok(false);
            };
            self.decoder = Some(decoder);
            return Ok(true);
        }
        let active_prefetch_plan = self
            .active_prefetch_plan
            .take()
            .expect("active prefetch plan checked above");
        active_prefetch_plan
            .lock()
            .expect("prefetch run state must not be poisoned")
            .discard_staged_windows();
        if self.pending_decoders.is_empty() != self.pending_prefetch_plans.is_empty() {
            return Err(prefetch_plan_queue_drift());
        }
        let (Some(decoder), Some(prefetch_plan)) = (
            self.pending_decoders.pop_front(),
            self.pending_prefetch_plans.pop_front(),
        ) else {
            return Ok(false);
        };
        self.decoder = Some(decoder);
        self.active_prefetch_plan = Some(prefetch_plan);
        Ok(true)
    }

    fn terminate(&mut self) {
        self.terminated = true;
        self.prefetched_readers.clear();
        self.next_reader_future = None;
        self.foreground_resources = SpeculativeResources::default();
        self.active_reader = None;
        self.decoder = None;
        self.reader = None;
        self.pending_decoders.clear();
        self.active_prefetch_plan = None;
        self.pending_prefetch_plans.clear();
        self.deferred_error = None;
        self.run_finished = false;
    }
}

fn prefetch_plan_queue_drift() -> DataFusionError {
    DataFusionError::Internal(
        "row-group prefetch plan queue drifted from decoder runs".to_string(),
    )
}

impl Drop for LookaheadPushDecoderStreamState {
    fn drop(&mut self) {
        self.terminate();
    }
}

async fn poll_once(
    future: &mut BoxFuture<'static, NextReaderOutcome>,
) -> Option<NextReaderOutcome> {
    futures::future::poll_fn(|context| {
        Poll::Ready(match future.as_mut().poll(context) {
            Poll::Ready(outcome) => Some(outcome),
            Poll::Pending => None,
        })
    })
    .await
}

async fn drive_speculative_next_reader(
    mut decoder: ParquetPushDecoder,
    mut reader: Box<dyn AsyncFileReader>,
    lookahead: LookaheadFileContext,
    prefetch_plan: Option<Arc<Mutex<PrefetchRunState>>>,
) -> NextReaderOutcome {
    let mut resources = SpeculativeResources::default();
    let result = loop {
        let next_row_group = decoder.peek_next_row_group().map_err(DataFusionError::from);
        let next_row_group = match next_row_group {
            Ok(row_group_index) => row_group_index,
            Err(error) => break NextReaderResult::Error(error),
        };
        match decoder.try_next_reader() {
            Ok(DecodeResult::NeedsData(ranges)) => {
                if let Some(row_group_index) = next_row_group
                    && let Err(error) = stage_admitted_window(
                        &mut decoder,
                        &mut reader,
                        &lookahead,
                        prefetch_plan.as_ref(),
                        row_group_index,
                        &ranges,
                    )
                    .await
                {
                    break NextReaderResult::Error(error);
                }
                if ranges.len() > MAX_RANGES_PER_FILE_FETCH {
                    break NextReaderResult::Denied;
                }
                let Some(bytes) = checked_range_bytes(&ranges) else {
                    break NextReaderResult::Denied;
                };
                let permit_count = u32::try_from(ranges.len())
                    .expect("speculative range set is bounded by four");
                let Ok(range_permit) = Arc::clone(&lookahead.coordinator.range_permits)
                    .try_acquire_many_owned(permit_count)
                else {
                    break NextReaderResult::Denied;
                };
                let Some(lease) = lookahead.try_reserve(bytes) else {
                    drop(range_permit);
                    break NextReaderResult::Denied;
                };
                let data = match reader.get_byte_ranges(ranges.clone()).await {
                    Ok(data) => data,
                    Err(error) => {
                        drop(lease);
                        drop(range_permit);
                        break NextReaderResult::Error(DataFusionError::from(error));
                    }
                };
                if let Err(error) = decoder.push_ranges(ranges, data) {
                    drop(lease);
                    drop(range_permit);
                    break NextReaderResult::Error(DataFusionError::from(error));
                }
                resources.leases.push(lease);
                resources.range_permits.push(range_permit);
            }
            Ok(DecodeResult::Data(reader)) => {
                let staged_window = next_row_group.and_then(|row_group_index| {
                    prefetch_plan.as_ref().and_then(|state| {
                        state
                            .lock()
                            .expect("prefetch run state must not be poisoned")
                            .reader_ready(row_group_index)
                    })
                });
                break NextReaderResult::Reader {
                    reader,
                    staged_window,
                };
            }
            Ok(DecodeResult::Finished) => {
                break NextReaderResult::Finished;
            }
            Err(error) => {
                break NextReaderResult::Error(DataFusionError::from(error));
            }
        }
    };

    NextReaderOutcome {
        decoder,
        reader,
        result,
        resources,
    }
}

fn checked_range_bytes(ranges: &[std::ops::Range<u64>]) -> Option<usize> {
    ranges.iter().try_fold(0usize, |sum, range| {
        let length = usize::try_from(range.end.checked_sub(range.start)?).ok()?;
        sum.checked_add(length)
    })
}

async fn stage_admitted_window(
    decoder: &mut ParquetPushDecoder,
    reader: &mut Box<dyn AsyncFileReader>,
    lookahead: &LookaheadFileContext,
    prefetch_plan: Option<&Arc<Mutex<PrefetchRunState>>>,
    row_group_index: usize,
    exact_ranges: &[std::ops::Range<u64>],
) -> Result<()> {
    let Some(prefetch_plan) = prefetch_plan else {
        return Ok(());
    };
    let request = prefetch_plan
        .lock()
        .expect("prefetch run state must not be poisoned")
        .observe_exact_ranges(row_group_index, exact_ranges);
    let Some(request) = request else {
        return Ok(());
    };

    if request.ranges.is_empty() || request.ranges.len() > MAX_IN_FLIGHT_RANGES {
        return Ok(());
    }
    let permit_count = match u32::try_from(request.ranges.len()) {
        Ok(permit_count) => permit_count,
        Err(_) => return Ok(()),
    };
    let Ok(range_permit) = Arc::clone(&lookahead.coordinator.range_permits)
        .try_acquire_many_owned(permit_count)
    else {
        return Ok(());
    };
    let Some(lease) = lookahead.try_reserve(request.staged_bytes) else {
        drop(range_permit);
        return Ok(());
    };
    let data = reader
        .get_byte_ranges(request.ranges.clone())
        .await
        .map_err(DataFusionError::from)?;
    decoder
        .push_ranges(request.ranges.clone(), data)
        .map_err(DataFusionError::from)?;

    prefetch_plan
        .lock()
        .expect("prefetch run state must not be poisoned")
        .add_staged_window(
            request,
            SpeculativeResources {
                leases: vec![lease],
                range_permits: vec![range_permit],
            },
        );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet, VecDeque};
    use std::ops::Range;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::Poll;
    use std::time::Duration;

    use arrow::array::{ArrayRef, Int32Array, RecordBatch};
    use arrow::compute::kernels::cmp::gt;
    use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use bytes::Bytes;
    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryConsumer, MemoryPool, UnboundedMemoryPool,
    };
    use datafusion_physical_expr::projection::ProjectionExprs;
    use datafusion_physical_plan::metrics::{
        BaselineMetrics, ExecutionPlanMetricsSet, MetricBuilder,
    };
    use futures::future::BoxFuture;
    use futures::{FutureExt, StreamExt, TryStreamExt};
    use parquet::arrow::arrow_reader::metrics::ArrowReaderMetrics;
    use parquet::arrow::arrow_reader::{ArrowPredicateFn, ArrowReaderOptions, RowFilter};
    use parquet::arrow::async_reader::AsyncFileReader;
    use parquet::arrow::push_decoder::{ParquetPushDecoder, ParquetPushDecoderBuilder};
    use parquet::arrow::{ArrowWriter, ProjectionMask};
    use parquet::errors::Result as ParquetResult;
    use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
    use parquet::file::properties::WriterProperties;
    use tokio::sync::Notify;
    use tokio::time::timeout;

    use super::super::{
        LookaheadPushDecoderStreamState, PushDecoderOutputState, PushDecoderStreamState,
    };
    use super::{PrefetchPlanQueue, PrefetchRunState};
    use crate::lookahead::{
        LookaheadFileContext, MAX_IN_FLIGHT_RANGES, MAX_SPECULATIVE_BYTES,
        ParquetLookaheadCoordinator,
    };
    use crate::row_group_prefetch::{RowGroupPrefetchMetrics, RowGroupPrefetchPlan};

    const FILTER_COLUMNS: [&str; 6] = [
        "value", "filter_1", "filter_2", "filter_3", "filter_4", "filter_5",
    ];
    const LOOKAHEAD_FILTER_COLUMNS: [&str; 4] =
        ["value", "filter_1", "filter_2", "filter_3"];
    const DISJOINT_FILTER_COLUMNS: [&str; 4] =
        ["filter_1", "filter_2", "filter_3", "filter_4"];

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct RequestRecord {
        row_group: usize,
        ranges: Vec<Range<u64>>,
    }

    #[derive(Debug)]
    struct ScriptControlInner {
        requests: Mutex<Vec<RequestRecord>>,
        request_started: Notify,
        release_gate: Notify,
        gated_row_group: Option<usize>,
        failed_row_group: Option<usize>,
        successful_requests_before_failure: usize,
        corrupted_row_group: Option<usize>,
        gate_released: AtomicBool,
        active_fetches: AtomicUsize,
    }

    #[derive(Debug, Clone)]
    struct ScriptControl {
        inner: Arc<ScriptControlInner>,
    }

    impl ScriptControl {
        fn new(gated_row_group: Option<usize>) -> Self {
            Self::with_script(gated_row_group, None, 0, None)
        }

        fn failing(failed_row_group: usize) -> Self {
            Self::with_script(None, Some(failed_row_group), 0, None)
        }

        fn failing_after(
            failed_row_group: usize,
            successful_requests_before_failure: usize,
        ) -> Self {
            Self::with_script(
                None,
                Some(failed_row_group),
                successful_requests_before_failure,
                None,
            )
        }

        fn corrupting(corrupted_row_group: usize) -> Self {
            Self::with_script(None, None, 0, Some(corrupted_row_group))
        }

        fn with_script(
            gated_row_group: Option<usize>,
            failed_row_group: Option<usize>,
            successful_requests_before_failure: usize,
            corrupted_row_group: Option<usize>,
        ) -> Self {
            Self {
                inner: Arc::new(ScriptControlInner {
                    requests: Mutex::new(vec![]),
                    request_started: Notify::new(),
                    release_gate: Notify::new(),
                    gated_row_group,
                    failed_row_group,
                    successful_requests_before_failure,
                    corrupted_row_group,
                    gate_released: AtomicBool::new(gated_row_group.is_none()),
                    active_fetches: AtomicUsize::new(0),
                }),
            }
        }

        fn record_fetch(
            &self,
            ranges_by_row_group: BTreeMap<usize, Vec<Range<u64>>>,
        ) -> ActiveFetchGuard {
            self.inner.requests.lock().unwrap().extend(
                ranges_by_row_group
                    .into_iter()
                    .map(|(row_group, ranges)| RequestRecord { row_group, ranges }),
            );
            self.inner.active_fetches.fetch_add(1, Ordering::SeqCst);
            self.inner.request_started.notify_waiters();
            ActiveFetchGuard {
                inner: Arc::clone(&self.inner),
            }
        }

        async fn wait_for_row_group(&self, row_group: usize) {
            timeout(Duration::from_secs(5), async {
                loop {
                    let notified = self.inner.request_started.notified();
                    if self
                        .inner
                        .requests
                        .lock()
                        .unwrap()
                        .iter()
                        .any(|request| request.row_group == row_group)
                    {
                        return;
                    }
                    notified.await;
                }
            })
            .await
            .expect("timed out waiting for scripted row-group request");
        }

        async fn wait_if_gated(&self, row_group: usize) {
            if self.inner.gated_row_group != Some(row_group) {
                return;
            }
            loop {
                let notified = self.inner.release_gate.notified();
                if self.inner.gate_released.load(Ordering::SeqCst) {
                    return;
                }
                notified.await;
            }
        }

        fn release(&self) {
            self.inner.gate_released.store(true, Ordering::SeqCst);
            self.inner.release_gate.notify_waiters();
        }

        fn requests_for(&self, row_group: usize) -> Vec<RequestRecord> {
            self.inner
                .requests
                .lock()
                .unwrap()
                .iter()
                .filter(|request| request.row_group == row_group)
                .cloned()
                .collect()
        }

        fn requests(&self) -> Vec<RequestRecord> {
            self.inner.requests.lock().unwrap().clone()
        }

        fn has_request_for(&self, row_group: usize) -> bool {
            !self.requests_for(row_group).is_empty()
        }

        fn active_fetches(&self) -> usize {
            self.inner.active_fetches.load(Ordering::SeqCst)
        }

        fn should_fail(&self, row_groups: &[usize]) -> bool {
            self.inner.failed_row_group.is_some_and(|row_group| {
                row_groups.contains(&row_group)
                    && self.requests_for(row_group).len()
                        > self.inner.successful_requests_before_failure
            })
        }

        fn should_corrupt(&self, row_groups: &[usize]) -> bool {
            self.inner
                .corrupted_row_group
                .is_some_and(|row_group| row_groups.contains(&row_group))
        }
    }

    struct ActiveFetchGuard {
        inner: Arc<ScriptControlInner>,
    }

    impl Drop for ActiveFetchGuard {
        fn drop(&mut self) {
            self.inner.active_fetches.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[derive(Clone)]
    struct ReaderDropProbe {
        context: LookaheadFileContext,
        reader_dropped: Arc<AtomicBool>,
        resources_released_before_reader_drop: Arc<AtomicBool>,
    }

    impl ReaderDropProbe {
        fn new(context: LookaheadFileContext) -> Self {
            Self {
                context,
                reader_dropped: Arc::new(AtomicBool::new(false)),
                resources_released_before_reader_drop: Arc::new(AtomicBool::new(false)),
            }
        }

        fn record_reader_drop(&self) {
            let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
            let reservation = Arc::new(
                MemoryConsumer::new("push-decoder-lookahead-reader-drop-probe")
                    .register(&pool),
            );
            let byte_budget_context = LookaheadFileContext::new(
                Arc::clone(&self.context.coordinator),
                reservation,
            );
            let byte_budget_released = byte_budget_context
                .try_reserve(MAX_SPECULATIVE_BYTES)
                .is_some_and(|lease| {
                    drop(lease);
                    true
                });
            self.resources_released_before_reader_drop.store(
                self.context.reservation.size() == 0
                    && self.context.coordinator.range_permits.available_permits()
                        == MAX_IN_FLIGHT_RANGES
                    && byte_budget_released,
                Ordering::SeqCst,
            );
            self.reader_dropped.store(true, Ordering::SeqCst);
        }
    }

    struct ScriptedAsyncFileReader {
        data: Bytes,
        metadata: Arc<ParquetMetaData>,
        row_group_spans: Arc<Vec<Range<u64>>>,
        control: ScriptControl,
        drop_probe: Option<ReaderDropProbe>,
    }

    impl ScriptedAsyncFileReader {
        fn ranges_by_row_group(
            &self,
            ranges: &[Range<u64>],
        ) -> BTreeMap<usize, Vec<Range<u64>>> {
            ranges
                .iter()
                .cloned()
                .fold(BTreeMap::new(), |mut grouped, range| {
                    let row_group = self
                        .row_group_spans
                        .iter()
                        .position(|span| {
                            range.start >= span.start && range.end <= span.end
                        })
                        .unwrap_or_else(|| {
                            panic!("range {range:?} is outside row-group data")
                        });
                    grouped.entry(row_group).or_default().push(range);
                    grouped
                })
        }
    }

    impl Drop for ScriptedAsyncFileReader {
        fn drop(&mut self) {
            if let Some(probe) = &self.drop_probe {
                probe.record_reader_drop();
            }
        }
    }

    impl AsyncFileReader for ScriptedAsyncFileReader {
        fn get_bytes(
            &mut self,
            range: Range<u64>,
        ) -> BoxFuture<'_, ParquetResult<Bytes>> {
            let data = self.data.slice(range.start as usize..range.end as usize);
            futures::future::ready(Ok(data)).boxed()
        }

        fn get_byte_ranges(
            &mut self,
            ranges: Vec<Range<u64>>,
        ) -> BoxFuture<'_, ParquetResult<Vec<Bytes>>> {
            let ranges_by_row_group = self.ranges_by_row_group(&ranges);
            let row_groups = ranges_by_row_group.keys().copied().collect::<Vec<_>>();
            let data = self.data.clone();
            let control = self.control.clone();
            async move {
                let _active_fetch = control.record_fetch(ranges_by_row_group);
                for row_group in &row_groups {
                    control.wait_if_gated(*row_group).await;
                }
                if control.should_fail(&row_groups) {
                    return Err(parquet::errors::ParquetError::General(format!(
                        "scripted row-group {} failure",
                        control.inner.failed_row_group.unwrap_or_default()
                    )));
                }
                Ok(ranges
                    .into_iter()
                    .map(|range| {
                        if control.should_corrupt(&row_groups) {
                            let len = usize::try_from(range.end - range.start).unwrap();
                            Bytes::from(vec![0_u8; len])
                        } else {
                            data.slice(range.start as usize..range.end as usize)
                        }
                    })
                    .collect())
            }
            .boxed()
        }

        fn get_metadata<'a>(
            &'a mut self,
            _options: Option<&'a ArrowReaderOptions>,
        ) -> BoxFuture<'a, ParquetResult<Arc<ParquetMetaData>>> {
            futures::future::ready(Ok(Arc::clone(&self.metadata))).boxed()
        }
    }

    struct ThreeRowGroupFixture {
        data: Bytes,
        metadata: Arc<ParquetMetaData>,
        row_group_spans: Arc<Vec<Range<u64>>>,
        output_schema: SchemaRef,
        filter_columns: Vec<&'static str>,
        rows_per_group: usize,
    }

    impl ThreeRowGroupFixture {
        fn new() -> Self {
            Self::with_filter_columns(&LOOKAHEAD_FILTER_COLUMNS)
        }

        fn with_filter_columns(filter_columns: &[&'static str]) -> Self {
            Self::with_row_group_count_and_filter_columns(3, filter_columns)
        }

        fn with_row_group_count_and_filter_columns(
            row_group_count: usize,
            filter_columns: &[&'static str],
        ) -> Self {
            Self::with_row_group_count_and_rows_per_group_and_filter_columns(
                row_group_count,
                3,
                filter_columns,
            )
        }

        fn with_row_group_count_and_rows_per_group_and_filter_columns(
            row_group_count: usize,
            rows_per_group: usize,
            filter_columns: &[&'static str],
        ) -> Self {
            let schema = Arc::new(Schema::new(
                FILTER_COLUMNS
                    .iter()
                    .map(|name| Field::new(*name, DataType::Int32, false))
                    .collect::<Vec<_>>(),
            ));
            let row_count = i32::try_from(row_group_count * rows_per_group).unwrap();
            let values = Arc::new(Int32Array::from_iter_values(0..row_count)) as ArrayRef;
            let columns = (0..FILTER_COLUMNS.len())
                .map(|_| Arc::clone(&values))
                .collect::<Vec<_>>();
            let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
            let properties = WriterProperties::builder()
                .set_max_row_group_row_count(Some(rows_per_group))
                .build();
            let mut output = Vec::new();
            let mut writer =
                ArrowWriter::try_new(&mut output, schema, Some(properties)).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();

            let data = Bytes::from(output);
            let metadata = Arc::new(
                ParquetMetaDataReader::new()
                    .parse_and_finish(&data)
                    .unwrap(),
            );
            assert_eq!(metadata.num_row_groups(), row_group_count);
            let row_group_spans = Arc::new(
                metadata
                    .row_groups()
                    .iter()
                    .map(|row_group| {
                        let start = row_group
                            .columns()
                            .iter()
                            .map(|column| column.byte_range().0)
                            .min()
                            .unwrap();
                        let end = row_group
                            .columns()
                            .iter()
                            .map(|column| {
                                let (start, len) = column.byte_range();
                                start + len
                            })
                            .max()
                            .unwrap();
                        start..end
                    })
                    .collect(),
            );
            let output_schema = Arc::new(Schema::new(vec![Field::new(
                "value",
                DataType::Int32,
                false,
            )]));
            Self {
                data,
                metadata,
                row_group_spans,
                output_schema,
                filter_columns: filter_columns.to_vec(),
                rows_per_group,
            }
        }

        fn with_disjoint_filter_and_projection() -> Self {
            Self::with_filter_columns(&DISJOINT_FILTER_COLUMNS)
        }

        fn dense_prefetch(row_group_count: usize) -> Self {
            let rows_per_group = 300_000;
            let schema = Arc::new(Schema::new(
                FILTER_COLUMNS
                    .iter()
                    .map(|name| Field::new(*name, DataType::Int32, false))
                    .collect::<Vec<_>>(),
            ));
            let row_count = row_group_count * rows_per_group;
            let mut state = 0x9e37_79b9_u32;
            let values =
                Arc::new(Int32Array::from_iter_values((0..row_count).map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    state as i32
                }))) as ArrayRef;
            let columns = (0..FILTER_COLUMNS.len())
                .map(|_| Arc::clone(&values))
                .collect::<Vec<_>>();
            let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
            let properties = WriterProperties::builder()
                .set_max_row_group_row_count(Some(rows_per_group))
                .set_compression(parquet::basic::Compression::UNCOMPRESSED)
                .set_encoding(parquet::basic::Encoding::PLAIN)
                .build();
            let mut output = Vec::new();
            let mut writer =
                ArrowWriter::try_new(&mut output, schema, Some(properties)).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();

            let data = Bytes::from(output);
            let metadata = Arc::new(
                ParquetMetaDataReader::new()
                    .parse_and_finish(&data)
                    .unwrap(),
            );
            let row_group_spans = Arc::new(
                metadata
                    .row_groups()
                    .iter()
                    .map(|row_group| {
                        let start = row_group
                            .columns()
                            .iter()
                            .map(|column| column.byte_range().0)
                            .min()
                            .unwrap();
                        let end = row_group
                            .columns()
                            .iter()
                            .map(|column| {
                                let (start, length) = column.byte_range();
                                start + length
                            })
                            .max()
                            .unwrap();
                        start..end
                    })
                    .collect(),
            );
            Self {
                data,
                metadata,
                row_group_spans,
                output_schema: Arc::new(Schema::new(vec![Field::new(
                    "value",
                    DataType::Int32,
                    false,
                )])),
                filter_columns: Vec::new(),
                rows_per_group,
            }
        }

        fn reader(&self, control: ScriptControl) -> Box<dyn AsyncFileReader> {
            self.reader_with_drop_probe(control, None)
        }

        fn reader_with_drop_probe(
            &self,
            control: ScriptControl,
            drop_probe: Option<ReaderDropProbe>,
        ) -> Box<dyn AsyncFileReader> {
            Box::new(ScriptedAsyncFileReader {
                data: self.data.clone(),
                metadata: Arc::clone(&self.metadata),
                row_group_spans: Arc::clone(&self.row_group_spans),
                control,
                drop_probe,
            })
        }

        fn decoder(&self) -> (ParquetPushDecoder, ArrowReaderMetrics) {
            let arrow_reader_metrics = ArrowReaderMetrics::enabled();
            let decoder = self.decoder_for_row_groups(None, &arrow_reader_metrics);
            (decoder, arrow_reader_metrics)
        }

        fn decoder_for_row_groups(
            &self,
            row_groups: Option<Vec<usize>>,
            arrow_reader_metrics: &ArrowReaderMetrics,
        ) -> ParquetPushDecoder {
            self.decoder_for_row_groups_with_filter(
                row_groups,
                arrow_reader_metrics,
                Some(0),
            )
        }

        fn decoder_for_row_groups_with_filter(
            &self,
            row_groups: Option<Vec<usize>>,
            arrow_reader_metrics: &ArrowReaderMetrics,
            filter_threshold: Option<i32>,
        ) -> ParquetPushDecoder {
            self.decoder_for_row_groups_with_filter_and_batch_size(
                row_groups,
                arrow_reader_metrics,
                filter_threshold,
                1,
            )
        }

        fn decoder_for_row_groups_with_filter_and_batch_size(
            &self,
            row_groups: Option<Vec<usize>>,
            arrow_reader_metrics: &ArrowReaderMetrics,
            filter_threshold: Option<i32>,
            batch_size: usize,
        ) -> ParquetPushDecoder {
            let mut builder =
                ParquetPushDecoderBuilder::try_new_decoder(Arc::clone(&self.metadata))
                    .unwrap();
            let schema_descr = builder.metadata().file_metadata().schema_descr_ptr();
            if let Some(row_groups) = row_groups {
                builder = builder.with_row_groups(row_groups);
            }
            builder = builder
                .with_projection(ProjectionMask::columns(&schema_descr, ["value"]))
                .with_batch_size(batch_size)
                .with_metrics(arrow_reader_metrics.clone());
            if let Some(filter_threshold) = filter_threshold {
                let predicate = ArrowPredicateFn::new(
                    ProjectionMask::columns(
                        &schema_descr,
                        self.filter_columns.iter().copied(),
                    ),
                    move |batch| {
                        let values = batch
                            .column(0)
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap();
                        gt(values, &Int32Array::new_scalar(filter_threshold))
                    },
                );
                builder =
                    builder.with_row_filter(RowFilter::new(vec![Box::new(predicate)]));
            }
            builder.build().unwrap()
        }

        fn prefetch_plan(&self, row_group_order: Vec<usize>) -> RowGroupPrefetchPlan {
            let schema_descr = self.metadata.file_metadata().schema_descr();
            let projection_mask = ProjectionMask::columns(schema_descr, ["value"]);
            RowGroupPrefetchPlan::new(
                self.metadata.as_ref(),
                &projection_mask,
                row_group_order,
            )
        }

        fn prefetch_metrics(&self) -> RowGroupPrefetchMetrics {
            RowGroupPrefetchMetrics::new(
                0,
                "scripted.parquet",
                &ExecutionPlanMetricsSet::new(),
            )
        }

        fn output_state(
            &self,
            arrow_reader_metrics: ArrowReaderMetrics,
        ) -> PushDecoderOutputState {
            let metrics = ExecutionPlanMetricsSet::new();
            let projector = ProjectionExprs::from_indices(&[0], &self.output_schema)
                .make_projector(&self.output_schema)
                .unwrap();
            PushDecoderOutputState {
                remaining_limit: None,
                projector,
                output_schema: Arc::clone(&self.output_schema),
                replace_schema: false,
                arrow_reader_metrics,
                predicate_cache_inner_records: MetricBuilder::new(&metrics)
                    .gauge("predicate_cache_inner_records", 0),
                predicate_cache_records: MetricBuilder::new(&metrics)
                    .gauge("predicate_cache_records", 0),
                baseline_metrics: BaselineMetrics::new(&metrics, 0),
            }
        }

        fn lookahead_context(
            &self,
        ) -> (Arc<ParquetLookaheadCoordinator>, LookaheadFileContext) {
            self.lookahead_context_with_depth(1)
        }

        fn lookahead_context_with_depth(
            &self,
            depth: usize,
        ) -> (Arc<ParquetLookaheadCoordinator>, LookaheadFileContext) {
            self.lookahead_context_with_depth_and_window(depth, 0)
        }

        fn lookahead_context_with_depth_and_window(
            &self,
            depth: usize,
            window: usize,
        ) -> (Arc<ParquetLookaheadCoordinator>, LookaheadFileContext) {
            let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
            self.lookahead_context_with_depth_window_and_pool(depth, window, pool)
        }

        fn lookahead_context_with_pool(
            &self,
            pool: Arc<dyn MemoryPool>,
        ) -> (Arc<ParquetLookaheadCoordinator>, LookaheadFileContext) {
            self.lookahead_context_with_depth_and_pool(1, pool)
        }

        fn lookahead_context_with_depth_and_pool(
            &self,
            depth: usize,
            pool: Arc<dyn MemoryPool>,
        ) -> (Arc<ParquetLookaheadCoordinator>, LookaheadFileContext) {
            self.lookahead_context_with_depth_window_and_pool(depth, 0, pool)
        }

        fn lookahead_context_with_depth_window_and_pool(
            &self,
            depth: usize,
            window: usize,
            pool: Arc<dyn MemoryPool>,
        ) -> (Arc<ParquetLookaheadCoordinator>, LookaheadFileContext) {
            let coordinator = Arc::new(ParquetLookaheadCoordinator::new(depth, window));
            let reservation = Arc::new(
                MemoryConsumer::new("push-decoder-lookahead-test").register(&pool),
            );
            let context =
                LookaheadFileContext::new(Arc::clone(&coordinator), reservation);
            (coordinator, context)
        }

        async fn serial_oracle(&self) -> (Vec<RecordBatch>, ScriptControl) {
            let control = ScriptControl::new(None);
            let (decoder, arrow_reader_metrics) = self.decoder();
            let stream = PushDecoderStreamState {
                decoder,
                pending_decoders: VecDeque::new(),
                reader: self.reader(control.clone()),
                output: self.output_state(arrow_reader_metrics),
            }
            .into_stream();
            (stream.try_collect().await.unwrap(), control)
        }

        async fn serial_oracle_without_filter(&self) -> Vec<RecordBatch> {
            let control = ScriptControl::new(None);
            let arrow_reader_metrics = ArrowReaderMetrics::enabled();
            let decoder = self.decoder_for_row_groups_with_filter_and_batch_size(
                None,
                &arrow_reader_metrics,
                None,
                self.rows_per_group,
            );
            PushDecoderStreamState {
                decoder,
                pending_decoders: VecDeque::new(),
                reader: self.reader(control),
                output: self.output_state(arrow_reader_metrics),
            }
            .into_stream()
            .try_collect()
            .await
            .unwrap()
        }

        fn lookahead_stream(
            &self,
            control: ScriptControl,
            context: LookaheadFileContext,
        ) -> futures::stream::BoxStream<'static, datafusion_common::Result<RecordBatch>>
        {
            self.lookahead_stream_with_limit(control, context, None)
        }

        fn lookahead_stream_with_limit(
            &self,
            control: ScriptControl,
            context: LookaheadFileContext,
            remaining_limit: Option<usize>,
        ) -> futures::stream::BoxStream<'static, datafusion_common::Result<RecordBatch>>
        {
            let (decoder, arrow_reader_metrics) = self.decoder();
            let mut output = self.output_state(arrow_reader_metrics);
            output.remaining_limit = remaining_limit;
            LookaheadPushDecoderStreamState::new(
                decoder,
                VecDeque::new(),
                self.reader(control),
                output,
                context,
            )
            .into_stream()
        }

        fn staged_stream_without_filter(
            &self,
            control: ScriptControl,
            context: LookaheadFileContext,
        ) -> futures::stream::BoxStream<'static, datafusion_common::Result<RecordBatch>>
        {
            self.staged_stream_without_filter_with_options(
                control,
                context,
                (0..self.metadata.num_row_groups()).collect(),
                None,
                true,
            )
        }

        fn staged_stream_without_filter_with_options(
            &self,
            control: ScriptControl,
            context: LookaheadFileContext,
            row_group_order: Vec<usize>,
            remaining_limit: Option<usize>,
            staging_enabled: bool,
        ) -> futures::stream::BoxStream<'static, datafusion_common::Result<RecordBatch>>
        {
            let arrow_reader_metrics = ArrowReaderMetrics::enabled();
            let decoder = self.decoder_for_row_groups_with_filter_and_batch_size(
                Some(row_group_order.clone()),
                &arrow_reader_metrics,
                None,
                self.rows_per_group,
            );
            let mut output = self.output_state(arrow_reader_metrics);
            output.remaining_limit = remaining_limit;
            LookaheadPushDecoderStreamState::new_with_prefetch_plans(
                decoder,
                VecDeque::new(),
                PrefetchPlanQueue {
                    active: self.prefetch_plan(row_group_order),
                    pending: VecDeque::new(),
                    metrics: self.prefetch_metrics(),
                    staging_enabled,
                },
                self.reader(control),
                output,
                context,
            )
            .unwrap()
            .into_stream()
        }

        fn lookahead_stream_without_filter(
            &self,
            control: ScriptControl,
            context: LookaheadFileContext,
        ) -> futures::stream::BoxStream<'static, datafusion_common::Result<RecordBatch>>
        {
            let arrow_reader_metrics = ArrowReaderMetrics::enabled();
            let decoder = self.decoder_for_row_groups_with_filter_and_batch_size(
                None,
                &arrow_reader_metrics,
                None,
                self.rows_per_group,
            );
            LookaheadPushDecoderStreamState::new(
                decoder,
                VecDeque::new(),
                self.reader(control),
                self.output_state(arrow_reader_metrics),
                context,
            )
            .into_stream()
        }
    }

    #[test]
    fn prefetch_plans_advance_with_split_decoder_runs_without_io() {
        let fixture = ThreeRowGroupFixture::with_row_group_count_and_filter_columns(
            5,
            &LOOKAHEAD_FILTER_COLUMNS,
        );
        let control = ScriptControl::new(None);
        let (_, context) = fixture.lookahead_context();
        let arrow_reader_metrics = ArrowReaderMetrics::enabled();
        let active_decoder =
            fixture.decoder_for_row_groups(Some(vec![0, 1]), &arrow_reader_metrics);
        let pending_decoders = VecDeque::from([
            fixture.decoder_for_row_groups(Some(vec![2]), &arrow_reader_metrics),
            fixture.decoder_for_row_groups(Some(vec![3, 4]), &arrow_reader_metrics),
        ]);
        let mut state = LookaheadPushDecoderStreamState::new_with_prefetch_plans(
            active_decoder,
            pending_decoders,
            PrefetchPlanQueue {
                active: fixture.prefetch_plan(vec![0, 1]),
                pending: VecDeque::from([
                    fixture.prefetch_plan(vec![2]),
                    fixture.prefetch_plan(vec![3, 4]),
                ]),
                metrics: fixture.prefetch_metrics(),
                staging_enabled: false,
            },
            fixture.reader(control.clone()),
            fixture.output_state(arrow_reader_metrics),
            context,
        )
        .unwrap();

        assert_eq!(
            state.active_prefetch_plan().unwrap().row_group_order(),
            &[0, 1]
        );
        assert_eq!(state.pending_prefetch_plans.len(), 2);
        assert!((0..5).all(|row_group| !control.has_request_for(row_group)));

        state.run_finished = true;
        assert!(state.advance_decoder_run().unwrap());
        assert_eq!(
            state.active_prefetch_plan().unwrap().row_group_order(),
            &[2]
        );
        assert_eq!(state.pending_prefetch_plans.len(), 1);

        state.run_finished = true;
        assert!(state.advance_decoder_run().unwrap());
        assert_eq!(
            state.active_prefetch_plan().unwrap().row_group_order(),
            &[3, 4]
        );
        assert!(state.pending_prefetch_plans.is_empty());
        assert!((0..5).all(|row_group| !control.has_request_for(row_group)));
    }

    #[test]
    fn prefetch_plans_retain_reverse_order_and_reject_drift() {
        let fixture = ThreeRowGroupFixture::with_row_group_count_and_filter_columns(
            5,
            &LOOKAHEAD_FILTER_COLUMNS,
        );
        let control = ScriptControl::new(None);
        let (_, context) = fixture.lookahead_context();
        let arrow_reader_metrics = ArrowReaderMetrics::enabled();
        let active_decoder =
            fixture.decoder_for_row_groups(Some(vec![4, 3]), &arrow_reader_metrics);
        let pending_decoder =
            fixture.decoder_for_row_groups(Some(vec![2, 1, 0]), &arrow_reader_metrics);
        let mut state = LookaheadPushDecoderStreamState::new_with_prefetch_plans(
            active_decoder,
            VecDeque::from([pending_decoder]),
            PrefetchPlanQueue {
                active: fixture.prefetch_plan(vec![4, 3]),
                pending: VecDeque::from([fixture.prefetch_plan(vec![2, 1, 0])]),
                metrics: fixture.prefetch_metrics(),
                staging_enabled: false,
            },
            fixture.reader(control),
            fixture.output_state(arrow_reader_metrics),
            context,
        )
        .unwrap();

        assert_eq!(
            state.active_prefetch_plan().unwrap().row_group_order(),
            &[4, 3]
        );
        state.run_finished = true;
        assert!(state.advance_decoder_run().unwrap());
        assert_eq!(
            state.active_prefetch_plan().unwrap().row_group_order(),
            &[2, 1, 0]
        );

        state.pending_prefetch_plans.push_back(Arc::new(Mutex::new(
            PrefetchRunState::new(
                fixture.prefetch_plan(vec![0]),
                fixture.prefetch_metrics(),
                0,
                false,
            ),
        )));
        state.run_finished = true;
        let error = state.advance_decoder_run().unwrap_err();
        assert!(error.to_string().contains("prefetch plan queue drift"));
    }

    #[test]
    fn prefetch_plan_constructor_rejects_decoder_queue_drift() {
        let fixture = ThreeRowGroupFixture::new();
        let control = ScriptControl::new(None);
        let (_, context) = fixture.lookahead_context();
        let arrow_reader_metrics = ArrowReaderMetrics::enabled();
        let decoder =
            fixture.decoder_for_row_groups(Some(vec![0]), &arrow_reader_metrics);
        let pending_decoder =
            fixture.decoder_for_row_groups(Some(vec![1]), &arrow_reader_metrics);

        let result = LookaheadPushDecoderStreamState::new_with_prefetch_plans(
            decoder,
            VecDeque::from([pending_decoder]),
            PrefetchPlanQueue {
                active: fixture.prefetch_plan(vec![0]),
                pending: VecDeque::new(),
                metrics: fixture.prefetch_metrics(),
                staging_enabled: false,
            },
            fixture.reader(control),
            fixture.output_state(arrow_reader_metrics),
            context,
        );
        let error = match result {
            Ok(_) => panic!("mismatched queues must fail"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("prefetch plan queue drift"));
    }

    #[tokio::test]
    async fn prefetch_window_zero_adds_no_requests_and_matches_depth_only() {
        let fixture = ThreeRowGroupFixture::new();

        let depth_only_control = ScriptControl::new(None);
        let (_, depth_only_context) = fixture.lookahead_context_with_depth(2);
        let depth_only = fixture
            .lookahead_stream_without_filter(
                depth_only_control.clone(),
                depth_only_context,
            )
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let planned_control = ScriptControl::new(None);
        let (_, planned_context) = fixture.lookahead_context_with_depth_and_window(2, 0);
        let planned = fixture
            .staged_stream_without_filter(planned_control.clone(), planned_context)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(planned, depth_only);
        assert_eq!(planned_control.requests(), depth_only_control.requests());
    }

    #[tokio::test]
    async fn sparse_first_row_group_never_stages_a_window() {
        let fixture = ThreeRowGroupFixture::new();
        let control = ScriptControl::new(None);
        let (_, context) = fixture.lookahead_context_with_depth_and_window(2, 2);
        let mut stream = fixture.staged_stream_without_filter(control.clone(), context);

        let first = stream.next().await.unwrap().unwrap();

        assert_eq!(first.num_rows(), fixture.rows_per_group);
        assert!(control.has_request_for(1));
        assert!(!control.has_request_for(2));
        drop(stream);
    }

    #[tokio::test]
    async fn dense_window_two_is_reused_and_held_until_final_row_group_foreground() {
        let fixture = ThreeRowGroupFixture::dense_prefetch(3);
        let serial = fixture.serial_oracle_without_filter().await;
        let control = ScriptControl::new(None);
        let (coordinator, context) =
            fixture.lookahead_context_with_depth_and_window(2, 2);
        let reservation = Arc::clone(&context.reservation);
        let mut stream = fixture.staged_stream_without_filter(control.clone(), context);

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first, serial[0]);
        assert!(control.has_request_for(1));
        assert!(control.has_request_for(2));
        assert_eq!(control.requests_for(1).len(), 1);
        assert_eq!(control.requests_for(2).len(), 1);
        assert!(reservation.size() > 0);

        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(second, serial[1]);
        assert_eq!(control.requests_for(1).len(), 1);
        assert_eq!(control.requests_for(2).len(), 1);
        assert!(reservation.size() > 0);

        let third = stream.next().await.unwrap().unwrap();
        assert_eq!(third, serial[2]);
        assert_eq!(reservation.size(), 0);
        assert_eq!(
            coordinator.range_permits.available_permits(),
            MAX_IN_FLIGHT_RANGES
        );
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn dense_window_four_fetches_the_next_four_row_groups_before_exact_needs() {
        let fixture = ThreeRowGroupFixture::dense_prefetch(5);
        let control = ScriptControl::new(None);
        let (_, context) = fixture.lookahead_context_with_depth_and_window(4, 4);
        let mut stream = fixture.staged_stream_without_filter(control.clone(), context);

        let first = stream.next().await.unwrap().unwrap();

        assert_eq!(first.num_rows(), fixture.rows_per_group);
        for row_group in 1..=4 {
            assert!(control.has_request_for(row_group));
            assert_eq!(control.requests_for(row_group).len(), 1);
        }
        drop(stream);
    }

    #[tokio::test]
    async fn staged_window_denial_and_fetch_error_release_every_reservation() {
        let fixture = ThreeRowGroupFixture::dense_prefetch(3);
        let control = ScriptControl::new(None);
        let (coordinator, context) =
            fixture.lookahead_context_with_depth_and_window(2, 2);
        let reservation = Arc::clone(&context.reservation);
        let range_permits = Arc::clone(&coordinator.range_permits)
            .try_acquire_many_owned(MAX_IN_FLIGHT_RANGES as u32)
            .unwrap();
        let mut stream = fixture.staged_stream_without_filter(control.clone(), context);

        let first = stream.next().await.unwrap().unwrap();

        assert_eq!(first.num_rows(), fixture.rows_per_group);
        assert!(!control.has_request_for(1));
        assert!(!control.has_request_for(2));
        assert_eq!(reservation.size(), 0);
        drop(stream);
        drop(range_permits);
        assert_shared_budgets_released(&coordinator);

        let byte_denied_control = ScriptControl::new(None);
        let byte_denied_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(0));
        let (byte_denied_coordinator, byte_denied_context) =
            fixture.lookahead_context_with_depth_window_and_pool(2, 2, byte_denied_pool);
        let byte_denied_reservation = Arc::clone(&byte_denied_context.reservation);
        let mut byte_denied_stream = fixture.staged_stream_without_filter(
            byte_denied_control.clone(),
            byte_denied_context,
        );

        let first = byte_denied_stream.next().await.unwrap().unwrap();

        assert_eq!(first.num_rows(), fixture.rows_per_group);
        assert!(!byte_denied_control.has_request_for(1));
        assert!(!byte_denied_control.has_request_for(2));
        assert_eq!(byte_denied_reservation.size(), 0);
        drop(byte_denied_stream);
        assert_shared_budgets_released(&byte_denied_coordinator);

        let failing_control = ScriptControl::failing(1);
        let (failing_coordinator, failing_context) =
            fixture.lookahead_context_with_depth_and_window(2, 2);
        let failing_reservation = Arc::clone(&failing_context.reservation);
        let mut failing_stream =
            fixture.staged_stream_without_filter(failing_control, failing_context);

        let error = failing_stream.next().await.unwrap().unwrap_err();

        assert!(error.to_string().contains("scripted row-group 1 failure"));
        assert_eq!(failing_reservation.size(), 0);
        assert_shared_budgets_released(&failing_coordinator);
        assert!(failing_stream.next().await.is_none());
    }

    #[tokio::test]
    async fn staged_window_decode_error_releases_every_reservation() {
        let fixture = ThreeRowGroupFixture::dense_prefetch(3);
        let control = ScriptControl::corrupting(1);
        let (coordinator, context) =
            fixture.lookahead_context_with_depth_and_window(2, 2);
        let reservation = Arc::clone(&context.reservation);
        let mut stream = fixture.staged_stream_without_filter(control, context);

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.num_rows(), fixture.rows_per_group);

        let error = stream.next().await.unwrap().unwrap_err();

        assert!(error.to_string().contains("Parquet error"));
        assert_eq!(reservation.size(), 0);
        assert_shared_budgets_released(&coordinator);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn cancelling_a_staged_window_releases_all_shared_resources() {
        let fixture = ThreeRowGroupFixture::dense_prefetch(3);
        let control = ScriptControl::new(None);
        let (coordinator, context) =
            fixture.lookahead_context_with_depth_and_window(2, 2);
        let reservation = Arc::clone(&context.reservation);
        let mut stream = fixture.staged_stream_without_filter(control, context);

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.num_rows(), fixture.rows_per_group);
        assert!(reservation.size() > 0);

        drop(stream);

        assert_eq!(reservation.size(), 0);
        assert_shared_budgets_released(&coordinator);
    }

    #[tokio::test]
    async fn staged_windows_follow_reverse_row_group_order() {
        let fixture = ThreeRowGroupFixture::dense_prefetch(3);
        let control = ScriptControl::new(None);
        let (_, context) = fixture.lookahead_context_with_depth_and_window(2, 2);
        let mut stream = fixture.staged_stream_without_filter_with_options(
            control.clone(),
            context,
            vec![2, 1, 0],
            None,
            true,
        );

        let first = stream.next().await.unwrap().unwrap();

        assert_eq!(first.num_rows(), fixture.rows_per_group);
        assert!(control.has_request_for(1));
        assert!(control.has_request_for(0));
        drop(stream);
    }

    #[tokio::test]
    async fn staged_windows_do_not_cross_decoder_run_boundaries() {
        let fixture = ThreeRowGroupFixture::dense_prefetch(3);
        let control = ScriptControl::new(None);
        let (_, context) = fixture.lookahead_context_with_depth_and_window(2, 2);
        let arrow_reader_metrics = ArrowReaderMetrics::enabled();
        let active_decoder = fixture.decoder_for_row_groups_with_filter_and_batch_size(
            Some(vec![0]),
            &arrow_reader_metrics,
            None,
            fixture.rows_per_group,
        );
        let pending_decoder = fixture.decoder_for_row_groups_with_filter_and_batch_size(
            Some(vec![1, 2]),
            &arrow_reader_metrics,
            None,
            fixture.rows_per_group,
        );
        let mut stream = LookaheadPushDecoderStreamState::new_with_prefetch_plans(
            active_decoder,
            VecDeque::from([pending_decoder]),
            PrefetchPlanQueue {
                active: fixture.prefetch_plan(vec![0]),
                pending: VecDeque::from([fixture.prefetch_plan(vec![1, 2])]),
                metrics: fixture.prefetch_metrics(),
                staging_enabled: true,
            },
            fixture.reader(control.clone()),
            fixture.output_state(arrow_reader_metrics),
            context,
        )
        .unwrap()
        .into_stream();

        let first = stream.next().await.unwrap().unwrap();

        assert_eq!(first.num_rows(), fixture.rows_per_group);
        assert!(!control.has_request_for(1));
        assert!(!control.has_request_for(2));
        drop(stream);
    }

    #[tokio::test]
    async fn limited_scans_stay_exact_without_staged_window_requests() {
        let fixture = ThreeRowGroupFixture::dense_prefetch(3);
        let control = ScriptControl::new(None);
        let (_, context) = fixture.lookahead_context_with_depth_and_window(2, 2);
        let mut stream = fixture.staged_stream_without_filter_with_options(
            control.clone(),
            context,
            vec![0, 1, 2],
            Some(1),
            false,
        );

        let first = stream.next().await.unwrap().unwrap();

        assert_eq!(first.num_rows(), 1);
        assert!(!control.has_request_for(2));
        drop(stream);
    }

    fn batch_values(batch: &RecordBatch) -> Vec<i32> {
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    fn assert_shared_budgets_released(coordinator: &Arc<ParquetLookaheadCoordinator>) {
        assert_eq!(
            coordinator.range_permits.available_permits(),
            MAX_IN_FLIGHT_RANGES
        );
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let reservation = Arc::new(
            MemoryConsumer::new("push-decoder-lookahead-byte-budget-probe")
                .register(&pool),
        );
        let context = LookaheadFileContext::new(Arc::clone(coordinator), reservation);
        drop(
            context
                .try_reserve(MAX_SPECULATIVE_BYTES)
                .expect("the full shared byte budget must be released"),
        );
    }

    #[tokio::test]
    async fn depth_four_queues_multiple_readers_and_preserves_order() {
        let fixture = ThreeRowGroupFixture::with_row_group_count_and_rows_per_group_and_filter_columns(
            4,
            5,
            &LOOKAHEAD_FILTER_COLUMNS,
        );
        let (serial, _) = fixture.serial_oracle().await;

        let depth_one_control = ScriptControl::new(None);
        let (_, depth_one_context) = fixture.lookahead_context_with_depth(1);
        let mut depth_one_stream =
            fixture.lookahead_stream(depth_one_control.clone(), depth_one_context);
        for expected in 1..=3 {
            let batch = depth_one_stream.next().await.unwrap().unwrap();
            assert_eq!(batch_values(&batch), vec![expected]);
        }
        assert!(depth_one_control.has_request_for(1));
        assert!(!depth_one_control.has_request_for(2));
        assert!(!depth_one_control.has_request_for(3));
        drop(depth_one_stream);

        let depth_four_control = ScriptControl::new(None);
        let (_, depth_four_context) = fixture.lookahead_context_with_depth(4);
        let mut depth_four_stream =
            fixture.lookahead_stream(depth_four_control.clone(), depth_four_context);
        let mut output = Vec::new();
        for expected in 1..=3 {
            let batch = depth_four_stream.next().await.unwrap().unwrap();
            assert_eq!(batch_values(&batch), vec![expected]);
            output.push(batch);
        }

        assert!(depth_four_control.has_request_for(1));
        assert!(depth_four_control.has_request_for(2));
        assert!(depth_four_control.has_request_for(3));
        assert_eq!(batch_values(&serial[3]), vec![4]);
        assert_eq!(output.len(), 3, "an RG0 output batch remains pending");

        output.extend(depth_four_stream.try_collect::<Vec<_>>().await.unwrap());
        assert_eq!(output, serial);
    }

    #[tokio::test]
    async fn lookahead_overlaps_at_depth_one_and_matches_serial_order() {
        let fixture = ThreeRowGroupFixture::new();
        let (serial, _) = fixture.serial_oracle().await;
        let control = ScriptControl::new(Some(1));
        let (coordinator, context) = fixture.lookahead_context();
        let reservation = Arc::clone(&context.reservation);
        let mut stream = fixture.lookahead_stream(control.clone(), context);

        let first = timeout(Duration::from_secs(5), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(batch_values(&first), vec![1]);
        control.wait_for_row_group(1).await;
        assert!(!control.has_request_for(2));

        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(batch_values(&second), vec![2]);
        assert!(!control.has_request_for(2));

        let mut next = Box::pin(stream.next());
        assert!(matches!(futures::poll!(next.as_mut()), Poll::Pending));
        assert!(!control.has_request_for(2));

        control.release();
        let third = timeout(Duration::from_secs(5), next)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(batch_values(&third), vec![3]);
        control.wait_for_row_group(2).await;

        let mut lookahead = vec![first, second, third];
        lookahead.extend(stream.try_collect::<Vec<_>>().await.unwrap());
        assert_eq!(lookahead, serial);
        assert!(
            lookahead
                .iter()
                .all(|batch| batch.schema() == fixture.output_schema)
        );
        assert!(
            control
                .requests_for(1)
                .iter()
                .chain(control.requests_for(2).iter())
                .all(|request| request.ranges.len() <= 4)
        );
        assert_eq!(reservation.size(), 0);
        assert_eq!(
            coordinator.range_permits.available_permits(),
            MAX_IN_FLIGHT_RANGES
        );
    }

    #[tokio::test]
    async fn dropping_pending_lookahead_releases_fetch_and_budgets() {
        let fixture = ThreeRowGroupFixture::new();
        for depth in [1, 4] {
            let control = ScriptControl::new(Some(1));
            let (coordinator, context) = fixture.lookahead_context_with_depth(depth);
            let reservation = Arc::clone(&context.reservation);
            let budget_context = context.clone();
            let mut stream = fixture.lookahead_stream(control.clone(), context);

            let first = stream.next().await.unwrap().unwrap();
            assert_eq!(batch_values(&first), vec![1]);
            control.wait_for_row_group(1).await;
            assert_eq!(control.active_fetches(), 1);
            assert!(reservation.size() > 0);
            assert!(coordinator.range_permits.available_permits() < MAX_IN_FLIGHT_RANGES);

            drop(stream);

            assert_eq!(control.active_fetches(), 0);
            assert_eq!(reservation.size(), 0);
            assert_eq!(
                coordinator.range_permits.available_permits(),
                MAX_IN_FLIGHT_RANGES
            );
            drop(
                budget_context
                    .try_reserve(MAX_SPECULATIVE_BYTES)
                    .expect("the full byte budget must be released"),
            );
        }
    }

    #[tokio::test]
    async fn final_limit_batch_synchronously_releases_pending_lookahead() {
        let fixture = ThreeRowGroupFixture::with_row_group_count_and_filter_columns(
            4,
            &LOOKAHEAD_FILTER_COLUMNS,
        );
        for (depth, limit, requested_row_groups) in
            [(1, 1, &[1][..]), (4, 3, &[1, 2, 3][..])]
        {
            let control = ScriptControl::new(None);
            let (coordinator, context) = fixture.lookahead_context_with_depth(depth);
            let reservation = Arc::clone(&context.reservation);
            let mut stream = fixture.lookahead_stream_with_limit(
                control.clone(),
                context,
                Some(limit),
            );

            let mut output = Vec::new();
            for _ in 0..limit {
                output.push(stream.next().await.unwrap().unwrap());
            }

            assert_eq!(batch_values(&output[0]), vec![1]);
            for row_group in requested_row_groups {
                assert!(control.has_request_for(*row_group));
            }
            assert_eq!(control.active_fetches(), 0);
            assert_eq!(reservation.size(), 0);
            assert_eq!(
                coordinator.range_permits.available_permits(),
                MAX_IN_FLIGHT_RANGES
            );
            assert_shared_budgets_released(&coordinator);
            assert!(stream.next().await.is_none());
        }
    }

    #[tokio::test]
    async fn speculative_error_is_deferred_until_foreground_reader_drains() {
        let depth_one_fixture = ThreeRowGroupFixture::new();
        let depth_one_control = ScriptControl::failing(1);
        let (depth_one_coordinator, depth_one_context) =
            depth_one_fixture.lookahead_context_with_depth(1);
        let depth_one_reservation = Arc::clone(&depth_one_context.reservation);
        let mut depth_one_stream = depth_one_fixture
            .lookahead_stream(depth_one_control.clone(), depth_one_context);
        let depth_one_error = depth_one_stream.next().await.unwrap().unwrap();
        let depth_one_second = depth_one_stream.next().await.unwrap().unwrap();
        let depth_one_failure = depth_one_stream.next().await.unwrap().unwrap_err();

        assert_eq!(batch_values(&depth_one_error), vec![1]);
        assert_eq!(batch_values(&depth_one_second), vec![2]);
        assert!(
            depth_one_failure
                .to_string()
                .contains("scripted row-group 1 failure")
        );
        assert!(!depth_one_control.has_request_for(2));
        assert_eq!(depth_one_reservation.size(), 0);
        assert_shared_budgets_released(&depth_one_coordinator);

        let fixture = ThreeRowGroupFixture::with_row_group_count_and_rows_per_group_and_filter_columns(
            4,
            5,
            &LOOKAHEAD_FILTER_COLUMNS,
        );
        let healthy_control = ScriptControl::new(None);
        let (_, healthy_context) = fixture.lookahead_context_with_depth(4);
        let mut healthy_stream =
            fixture.lookahead_stream(healthy_control.clone(), healthy_context);
        for expected in 1..=3 {
            let batch = healthy_stream.next().await.unwrap().unwrap();
            assert_eq!(batch_values(&batch), vec![expected]);
        }
        assert!(healthy_control.has_request_for(3));
        drop(healthy_stream);

        let control = ScriptControl::failing(2);
        let (coordinator, context) = fixture.lookahead_context_with_depth(4);
        let reservation = Arc::clone(&context.reservation);
        let mut stream = fixture.lookahead_stream(control.clone(), context);
        let mut output = Vec::new();
        let error = loop {
            match stream.next().await {
                Some(Ok(batch)) => output.extend(batch_values(&batch)),
                Some(Err(error)) => break error,
                None => panic!("speculative error must be surfaced"),
            }
        };

        assert_eq!(output, (1..=9).collect::<Vec<_>>());
        assert!(error.to_string().contains("scripted row-group 2 failure"));
        assert!(control.has_request_for(1));
        assert!(control.has_request_for(2));
        assert!(!control.has_request_for(3));
        assert_eq!(control.active_fetches(), 0);
        assert_eq!(reservation.size(), 0);
        assert_shared_budgets_released(&coordinator);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn foreground_fetch_error_during_bootstrap_is_immediate_and_terminal() {
        let fixture = ThreeRowGroupFixture::new();
        let control = ScriptControl::failing(0);
        let (coordinator, context) = fixture.lookahead_context();
        let reservation = Arc::clone(&context.reservation);
        let mut stream = fixture.lookahead_stream(control.clone(), context);

        let error = stream.next().await.unwrap().unwrap_err();

        assert!(error.to_string().contains("scripted row-group 0 failure"));
        assert!(!control.has_request_for(1));
        assert_eq!(control.active_fetches(), 0);
        assert_eq!(reservation.size(), 0);
        assert_eq!(
            coordinator.range_permits.available_permits(),
            MAX_IN_FLIGHT_RANGES
        );
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn foreground_fetch_error_after_partial_byte_denial_is_immediate_and_terminal()
    {
        let fixture = ThreeRowGroupFixture::with_disjoint_filter_and_projection();
        let (_, serial_control) = fixture.serial_oracle().await;
        let first_chunk_bytes = serial_control.requests_for(1)[0].ranges[..4]
            .iter()
            .map(|range| usize::try_from(range.end - range.start).unwrap())
            .sum();
        let control = ScriptControl::failing_after(1, 1);
        let pool = Arc::new(GreedyMemoryPool::new(first_chunk_bytes));
        let (coordinator, context) = fixture.lookahead_context_with_pool(pool.clone());
        let reservation = Arc::clone(&context.reservation);
        let mut stream = fixture.lookahead_stream(control.clone(), context);

        let first = stream.next().await.unwrap().unwrap();
        let second = stream.next().await.unwrap().unwrap();
        let error = stream.next().await.unwrap().unwrap_err();

        assert_eq!(batch_values(&first), vec![1]);
        assert_eq!(batch_values(&second), vec![2]);
        assert!(error.to_string().contains("scripted row-group 1 failure"));
        assert_eq!(
            control
                .requests_for(1)
                .iter()
                .map(|request| request.ranges.len())
                .collect::<Vec<_>>(),
            vec![4, 1]
        );
        assert!(!control.has_request_for(2));
        assert_eq!(control.active_fetches(), 0);
        assert_eq!(reservation.size(), 0);
        assert_eq!(pool.reserved(), 0);
        assert_eq!(
            coordinator.range_permits.available_permits(),
            MAX_IN_FLIGHT_RANGES
        );
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn foreground_record_reader_decode_error_cancels_prefetch_and_terminates() {
        let fixture = ThreeRowGroupFixture::new();
        for depth in [1, 4] {
            let control = ScriptControl::corrupting(0);
            let (coordinator, context) = fixture.lookahead_context_with_depth(depth);
            let reservation = Arc::clone(&context.reservation);
            let arrow_reader_metrics = ArrowReaderMetrics::enabled();
            let decoder = fixture.decoder_for_row_groups_with_filter(
                None,
                &arrow_reader_metrics,
                None,
            );
            let mut stream = LookaheadPushDecoderStreamState::new(
                decoder,
                VecDeque::new(),
                fixture.reader(control.clone()),
                fixture.output_state(arrow_reader_metrics),
                context,
            )
            .into_stream();

            let error = stream.next().await.unwrap().unwrap_err();

            assert!(control.has_request_for(1));
            assert!(!error.to_string().is_empty());
            assert_eq!(control.active_fetches(), 0);
            assert_eq!(reservation.size(), 0);
            assert_eq!(
                coordinator.range_permits.available_permits(),
                MAX_IN_FLIGHT_RANGES
            );
            assert!(stream.next().await.is_none());
        }
    }

    #[tokio::test]
    async fn replacement_schema_error_cancels_pending_prefetch_and_terminates() {
        let fixture = ThreeRowGroupFixture::new();
        for depth in [1, 4] {
            let control = ScriptControl::new(Some(1));
            let (coordinator, context) = fixture.lookahead_context_with_depth(depth);
            let reservation = Arc::clone(&context.reservation);
            let (decoder, arrow_reader_metrics) = fixture.decoder();
            let mut output = fixture.output_state(arrow_reader_metrics);
            output.replace_schema = true;
            output.output_schema = Arc::new(Schema::new(vec![
                Field::new("value", DataType::Int32, false),
                Field::new("unexpected", DataType::Int32, false),
            ]));
            let mut stream = LookaheadPushDecoderStreamState::new(
                decoder,
                VecDeque::new(),
                fixture.reader(control.clone()),
                output,
                context,
            )
            .into_stream();

            let error = stream.next().await.unwrap().unwrap_err();

            assert!(error.to_string().contains("number of columns"));
            assert!(control.has_request_for(1));
            assert_eq!(control.active_fetches(), 0);
            assert_eq!(reservation.size(), 0);
            assert_eq!(
                coordinator.range_permits.available_permits(),
                MAX_IN_FLIGHT_RANGES
            );
            assert!(stream.next().await.is_none());
        }
    }

    #[tokio::test]
    async fn pending_decoder_run_is_not_prefetched_across_boundary() {
        let fixture = ThreeRowGroupFixture::with_row_group_count_and_filter_columns(
            5,
            &LOOKAHEAD_FILTER_COLUMNS,
        );
        let (serial, _) = fixture.serial_oracle().await;
        let control = ScriptControl::new(Some(3));
        let (_, context) = fixture.lookahead_context_with_depth(4);
        let arrow_reader_metrics = ArrowReaderMetrics::enabled();
        let decoder =
            fixture.decoder_for_row_groups(Some(vec![0, 1, 2]), &arrow_reader_metrics);
        let pending_decoder =
            fixture.decoder_for_row_groups(Some(vec![3, 4]), &arrow_reader_metrics);
        let mut stream = LookaheadPushDecoderStreamState::new(
            decoder,
            VecDeque::from([pending_decoder]),
            fixture.reader(control.clone()),
            fixture.output_state(arrow_reader_metrics),
            context,
        )
        .into_stream();

        let mut output = Vec::new();
        for expected in 1..=8 {
            let batch = stream.next().await.unwrap().unwrap();
            assert_eq!(batch_values(&batch), vec![expected]);
            output.push(batch);
            assert!(!control.has_request_for(3));
        }
        assert!(control.has_request_for(1));
        assert!(control.has_request_for(2));

        let mut next = Box::pin(stream.next());
        assert!(matches!(futures::poll!(next.as_mut()), Poll::Pending));
        assert!(control.has_request_for(3));
        assert!(!control.has_request_for(4));

        control.release();
        output.push(next.await.unwrap().unwrap());
        output.extend(stream.try_collect::<Vec<_>>().await.unwrap());
        assert_eq!(output, serial);
    }

    #[tokio::test]
    async fn depth_one_pending_decoder_run_is_not_prefetched_across_boundary() {
        let fixture = ThreeRowGroupFixture::with_row_group_count_and_filter_columns(
            5,
            &LOOKAHEAD_FILTER_COLUMNS,
        );
        let (serial, _) = fixture.serial_oracle().await;
        let control = ScriptControl::new(Some(3));
        let (_, context) = fixture.lookahead_context_with_depth(1);
        let arrow_reader_metrics = ArrowReaderMetrics::enabled();
        let decoder =
            fixture.decoder_for_row_groups(Some(vec![0, 1, 2]), &arrow_reader_metrics);
        let pending_decoder =
            fixture.decoder_for_row_groups(Some(vec![3, 4]), &arrow_reader_metrics);
        let mut stream = LookaheadPushDecoderStreamState::new(
            decoder,
            VecDeque::from([pending_decoder]),
            fixture.reader(control.clone()),
            fixture.output_state(arrow_reader_metrics),
            context,
        )
        .into_stream();

        let mut output = Vec::new();
        for expected in 1..=8 {
            let batch = stream.next().await.unwrap().unwrap();
            assert_eq!(batch_values(&batch), vec![expected]);
            output.push(batch);
            assert!(!control.has_request_for(3));
        }
        assert!(control.has_request_for(1));
        assert!(control.has_request_for(2));

        let mut next = Box::pin(stream.next());
        assert!(matches!(futures::poll!(next.as_mut()), Poll::Pending));
        assert!(control.has_request_for(3));
        assert!(!control.has_request_for(4));

        control.release();
        output.push(next.await.unwrap().unwrap());
        output.extend(stream.try_collect::<Vec<_>>().await.unwrap());
        assert_eq!(output, serial);
    }

    #[tokio::test]
    async fn partial_range_permit_denial_continues_without_duplicate_ranges() {
        let fixture = ThreeRowGroupFixture::new();
        let (serial, serial_control) = fixture.serial_oracle().await;
        for depth in [1, 4] {
            let control = ScriptControl::new(None);
            let (coordinator, context) = fixture.lookahead_context_with_depth(depth);
            let reservation = Arc::clone(&context.reservation);
            let held = Arc::clone(&coordinator.range_permits)
                .try_acquire_many_owned(21)
                .unwrap();
            let stream = fixture.lookahead_stream(control.clone(), context);

            let lookahead = stream.try_collect::<Vec<_>>().await.unwrap();

            assert_eq!(lookahead, serial);
            let denied_requests = control.requests_for(1);
            assert_eq!(
                denied_requests
                    .iter()
                    .map(|request| request.ranges.len())
                    .collect::<Vec<_>>(),
                vec![4]
            );
            let actual_ranges = denied_requests
                .iter()
                .flat_map(|request| request.ranges.iter())
                .map(|range| (range.start, range.end))
                .collect::<HashSet<_>>();
            let expected_ranges = serial_control
                .requests_for(1)
                .iter()
                .flat_map(|request| request.ranges.iter())
                .map(|range| (range.start, range.end))
                .collect::<HashSet<_>>();
            assert_eq!(actual_ranges, expected_ranges);
            assert_eq!(actual_ranges.len(), 4);
            assert_eq!(reservation.size(), 0);
            assert_eq!(coordinator.range_permits.available_permits(), 3);

            drop(held);
            assert_eq!(
                coordinator.range_permits.available_permits(),
                MAX_IN_FLIGHT_RANGES
            );
            assert_shared_budgets_released(&coordinator);
        }
    }

    #[tokio::test]
    async fn oversized_range_set_falls_back_without_fragmenting() {
        let fixture = ThreeRowGroupFixture::with_filter_columns(&FILTER_COLUMNS);
        let (serial, serial_control) = fixture.serial_oracle().await;
        let control = ScriptControl::new(Some(1));
        let (coordinator, context) = fixture.lookahead_context();
        let reservation = Arc::clone(&context.reservation);
        let mut stream = fixture.lookahead_stream(control.clone(), context);

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(batch_values(&first), vec![1]);
        assert!(!control.has_request_for(1));

        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(batch_values(&second), vec![2]);
        assert!(!control.has_request_for(1));

        let mut next = Box::pin(stream.next());
        assert!(matches!(futures::poll!(next.as_mut()), Poll::Pending));
        assert!(control.has_request_for(1));
        assert_eq!(
            control
                .requests_for(1)
                .iter()
                .map(|request| request.ranges.len())
                .collect::<Vec<_>>(),
            vec![6]
        );

        control.release();
        let third = timeout(Duration::from_secs(5), next)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(batch_values(&third), vec![3]);

        let mut lookahead = vec![first, second, third];
        lookahead.extend(stream.try_collect::<Vec<_>>().await.unwrap());

        assert_eq!(lookahead, serial);
        assert_eq!(
            serial_control
                .requests_for(1)
                .iter()
                .map(|request| request.ranges.len())
                .collect::<Vec<_>>(),
            vec![6]
        );
        assert_eq!(
            control
                .requests_for(2)
                .iter()
                .map(|request| request.ranges.len())
                .collect::<Vec<_>>(),
            vec![6]
        );
        assert_eq!(reservation.size(), 0);
        assert_eq!(
            coordinator.range_permits.available_permits(),
            MAX_IN_FLIGHT_RANGES
        );
    }

    #[tokio::test]
    async fn reservation_denial_falls_back_to_foreground_without_result_change() {
        let fixture = ThreeRowGroupFixture::new();
        let (serial, _) = fixture.serial_oracle().await;
        for depth in [1, 4] {
            let control = ScriptControl::new(None);
            let pool = Arc::new(GreedyMemoryPool::new(0));
            let (coordinator, context) =
                fixture.lookahead_context_with_depth_and_pool(depth, pool.clone());
            let reservation = Arc::clone(&context.reservation);
            let stream = fixture.lookahead_stream(control.clone(), context);

            let lookahead = stream.try_collect::<Vec<_>>().await.unwrap();

            assert_eq!(lookahead, serial);
            assert_eq!(
                control
                    .requests_for(1)
                    .iter()
                    .map(|request| request.ranges.len())
                    .collect::<Vec<_>>(),
                vec![4]
            );
            assert_eq!(reservation.size(), 0);
            assert_eq!(pool.reserved(), 0);
            assert_eq!(
                coordinator.range_permits.available_permits(),
                MAX_IN_FLIGHT_RANGES
            );
            assert_shared_budgets_released(&coordinator);
        }
    }

    #[tokio::test]
    async fn partial_memory_denial_preserves_ranges_and_disables_later_speculation() {
        let fixture = ThreeRowGroupFixture::with_disjoint_filter_and_projection();
        let (serial, serial_control) = fixture.serial_oracle().await;
        let first_chunk_bytes = serial_control.requests_for(1)[0].ranges[..4]
            .iter()
            .map(|range| usize::try_from(range.end - range.start).unwrap())
            .sum();
        for depth in [1, 4] {
            let control = ScriptControl::new(None);
            let pool = Arc::new(GreedyMemoryPool::new(first_chunk_bytes));
            let (coordinator, context) =
                fixture.lookahead_context_with_depth_and_pool(depth, pool.clone());
            let reservation = Arc::clone(&context.reservation);
            let stream = fixture.lookahead_stream(control.clone(), context);

            let lookahead = stream.try_collect::<Vec<_>>().await.unwrap();

            assert_eq!(lookahead, serial);
            assert_eq!(
                control
                    .requests_for(1)
                    .iter()
                    .map(|request| request.ranges.len())
                    .collect::<Vec<_>>(),
                vec![4, 1]
            );
            assert_eq!(
                control
                    .requests_for(2)
                    .iter()
                    .map(|request| request.ranges.len())
                    .collect::<Vec<_>>(),
                vec![4, 1]
            );
            let actual_ranges = control
                .requests_for(1)
                .iter()
                .flat_map(|request| request.ranges.iter())
                .map(|range| (range.start, range.end))
                .collect::<HashSet<_>>();
            let expected_ranges = serial_control
                .requests_for(1)
                .iter()
                .flat_map(|request| request.ranges.iter())
                .map(|range| (range.start, range.end))
                .collect::<HashSet<_>>();
            assert_eq!(actual_ranges, expected_ranges);
            assert_eq!(actual_ranges.len(), 5);
            assert_eq!(reservation.size(), 0);
            assert_eq!(pool.reserved(), 0);
            assert_eq!(
                coordinator.range_permits.available_permits(),
                MAX_IN_FLIGHT_RANGES
            );
            assert_shared_budgets_released(&coordinator);
        }
    }

    #[tokio::test]
    async fn future_reader_can_be_ready_before_current_reader_drains() {
        let fixture = ThreeRowGroupFixture::new();
        let control = ScriptControl::new(None);
        let (coordinator, context) = fixture.lookahead_context();
        let reservation = Arc::clone(&context.reservation);
        let mut stream = fixture.lookahead_stream(control.clone(), context);

        let first = stream.next().await.unwrap().unwrap();

        assert_eq!(batch_values(&first), vec![1]);
        assert!(control.has_request_for(1));
        assert!(!control.has_request_for(2));
        assert!(reservation.size() > 0);
        assert!(coordinator.range_permits.available_permits() < MAX_IN_FLIGHT_RANGES);

        let mut batches = vec![first];
        batches.extend((&mut stream).try_collect::<Vec<_>>().await.unwrap());
        assert_eq!(
            batches.iter().flat_map(batch_values).collect::<Vec<_>>(),
            (1..9).collect::<Vec<_>>()
        );
        assert_eq!(reservation.size(), 0);
        assert_eq!(
            coordinator.range_permits.available_permits(),
            MAX_IN_FLIGHT_RANGES
        );
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn dropping_queued_readers_releases_all_lookahead_budgets() {
        let fixture = ThreeRowGroupFixture::with_row_group_count_and_filter_columns(
            4,
            &LOOKAHEAD_FILTER_COLUMNS,
        );
        for depth in [1, 4] {
            let control = ScriptControl::new(None);
            let (coordinator, context) = fixture.lookahead_context_with_depth(depth);
            let reservation = Arc::clone(&context.reservation);
            let budget_context = context.clone();
            let mut stream = fixture.lookahead_stream(control.clone(), context);

            assert_eq!(
                batch_values(&stream.next().await.unwrap().unwrap()),
                vec![1]
            );
            assert_eq!(
                batch_values(&stream.next().await.unwrap().unwrap()),
                vec![2]
            );
            assert!(control.has_request_for(1));
            if depth == 4 {
                assert!(control.has_request_for(2));
            }
            assert!(reservation.size() > 0);
            assert!(coordinator.range_permits.available_permits() < MAX_IN_FLIGHT_RANGES);

            drop(stream);

            assert_eq!(control.active_fetches(), 0);
            assert_eq!(reservation.size(), 0);
            assert_eq!(
                coordinator.range_permits.available_permits(),
                MAX_IN_FLIGHT_RANGES
            );
            drop(
                budget_context
                    .try_reserve(MAX_SPECULATIVE_BYTES)
                    .expect("the full byte budget must be released"),
            );
            assert_shared_budgets_released(&coordinator);
        }
    }

    #[tokio::test]
    async fn direct_stream_drop_releases_speculation_before_reader_teardown() {
        let fixture = ThreeRowGroupFixture::with_row_group_count_and_rows_per_group_and_filter_columns(
            5,
            4,
            &LOOKAHEAD_FILTER_COLUMNS,
        );
        let control = ScriptControl::new(Some(4));
        let (coordinator, context) = fixture.lookahead_context_with_depth(4);
        let probe = ReaderDropProbe::new(context.clone());
        let (decoder, arrow_reader_metrics) = fixture.decoder();
        let mut stream = LookaheadPushDecoderStreamState::new(
            decoder,
            VecDeque::new(),
            fixture.reader_with_drop_probe(control.clone(), Some(probe.clone())),
            fixture.output_state(arrow_reader_metrics),
            context,
        )
        .into_stream();

        for expected in 1..=4 {
            let batch = stream.next().await.unwrap().unwrap();
            assert_eq!(batch_values(&batch), vec![expected]);
        }
        control.wait_for_row_group(4).await;
        assert!(control.has_request_for(1));
        assert!(control.has_request_for(2));
        assert!(control.has_request_for(3));
        assert!(control.has_request_for(4));

        drop(stream);

        assert!(probe.reader_dropped.load(Ordering::SeqCst));
        assert!(
            probe
                .resources_released_before_reader_drop
                .load(Ordering::SeqCst)
        );
        assert_eq!(control.active_fetches(), 0);
        assert_eq!(
            coordinator.range_permits.available_permits(),
            MAX_IN_FLIGHT_RANGES
        );
        assert_shared_budgets_released(&coordinator);
    }

    #[tokio::test]
    async fn direct_stream_drop_releases_foreground_resources_before_reader_teardown() {
        let fixture = ThreeRowGroupFixture::with_disjoint_filter_and_projection();
        let (_, serial_control) = fixture.serial_oracle().await;
        let first_chunk_bytes = serial_control.requests_for(1)[0].ranges[..4]
            .iter()
            .map(|range| usize::try_from(range.end - range.start).unwrap())
            .sum();
        let control = ScriptControl::new(None);
        let pool = Arc::new(GreedyMemoryPool::new(first_chunk_bytes));
        let (coordinator, context) =
            fixture.lookahead_context_with_depth_and_pool(4, pool.clone());
        let probe = ReaderDropProbe::new(context.clone());
        let reservation = Arc::clone(&context.reservation);
        let (decoder, arrow_reader_metrics) = fixture.decoder();
        let mut stream = LookaheadPushDecoderStreamState::new(
            decoder,
            VecDeque::new(),
            fixture.reader_with_drop_probe(control.clone(), Some(probe.clone())),
            fixture.output_state(arrow_reader_metrics),
            context,
        )
        .into_stream();

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(batch_values(&first), vec![1]);
        assert_eq!(
            control
                .requests_for(1)
                .iter()
                .map(|request| request.ranges.len())
                .collect::<Vec<_>>(),
            vec![4]
        );
        assert!(reservation.size() > 0);
        assert!(coordinator.range_permits.available_permits() < MAX_IN_FLIGHT_RANGES);

        drop(stream);

        assert!(probe.reader_dropped.load(Ordering::SeqCst));
        assert!(
            probe
                .resources_released_before_reader_drop
                .load(Ordering::SeqCst)
        );
        assert_eq!(reservation.size(), 0);
        assert_eq!(pool.reserved(), 0);
        assert_shared_budgets_released(&coordinator);
    }

    #[tokio::test]
    async fn fully_filtered_row_group_is_skipped_before_following_data() {
        let fixture = ThreeRowGroupFixture::new();
        let control = ScriptControl::new(None);
        let (coordinator, context) = fixture.lookahead_context();
        let reservation = Arc::clone(&context.reservation);
        let arrow_reader_metrics = ArrowReaderMetrics::enabled();
        let decoder = fixture.decoder_for_row_groups_with_filter(
            None,
            &arrow_reader_metrics,
            Some(2),
        );
        let stream = LookaheadPushDecoderStreamState::new(
            decoder,
            VecDeque::new(),
            fixture.reader(control.clone()),
            fixture.output_state(arrow_reader_metrics),
            context,
        )
        .into_stream();

        let batches = stream.try_collect::<Vec<_>>().await.unwrap();

        assert_eq!(
            batches.iter().flat_map(batch_values).collect::<Vec<_>>(),
            (3..9).collect::<Vec<_>>()
        );
        assert!(control.has_request_for(0));
        assert!(control.has_request_for(1));
        assert_eq!(reservation.size(), 0);
        assert_eq!(
            coordinator.range_permits.available_permits(),
            MAX_IN_FLIGHT_RANGES
        );
    }

    #[tokio::test]
    async fn immediately_finished_decoder_returns_terminal_eof_without_fetching() {
        let fixture = ThreeRowGroupFixture::new();
        let control = ScriptControl::new(None);
        let (coordinator, context) = fixture.lookahead_context();
        let reservation = Arc::clone(&context.reservation);
        let arrow_reader_metrics = ArrowReaderMetrics::enabled();
        let decoder = fixture.decoder_for_row_groups_with_filter(
            Some(vec![]),
            &arrow_reader_metrics,
            Some(0),
        );
        let mut stream = LookaheadPushDecoderStreamState::new(
            decoder,
            VecDeque::new(),
            fixture.reader(control.clone()),
            fixture.output_state(arrow_reader_metrics),
            context,
        )
        .into_stream();

        assert!(stream.next().await.is_none());
        assert!(stream.next().await.is_none());
        assert!(control.inner.requests.lock().unwrap().is_empty());
        assert_eq!(reservation.size(), 0);
        assert_eq!(
            coordinator.range_permits.available_permits(),
            MAX_IN_FLIGHT_RANGES
        );
    }

    #[tokio::test]
    async fn reverse_row_group_order_is_preserved_by_lookahead() {
        let fixture = ThreeRowGroupFixture::new();
        for depth in [1, 4] {
            let control = ScriptControl::new(None);
            let (coordinator, context) = fixture.lookahead_context_with_depth(depth);
            let reservation = Arc::clone(&context.reservation);
            let arrow_reader_metrics = ArrowReaderMetrics::enabled();
            let decoder = fixture
                .decoder_for_row_groups(Some(vec![2, 1, 0]), &arrow_reader_metrics);
            let stream = LookaheadPushDecoderStreamState::new(
                decoder,
                VecDeque::new(),
                fixture.reader(control),
                fixture.output_state(arrow_reader_metrics),
                context,
            )
            .into_stream();

            let batches = stream.try_collect::<Vec<_>>().await.unwrap();

            assert_eq!(
                batches.iter().flat_map(batch_values).collect::<Vec<_>>(),
                vec![6, 7, 8, 3, 4, 5, 1, 2]
            );
            assert_eq!(reservation.size(), 0);
            assert_eq!(
                coordinator.range_permits.available_permits(),
                MAX_IN_FLIGHT_RANGES
            );
        }
    }

    #[tokio::test]
    async fn filtered_and_fully_matched_decoder_runs_preserve_output_order() {
        let fixture = ThreeRowGroupFixture::new();
        let control = ScriptControl::new(None);
        let (coordinator, context) = fixture.lookahead_context();
        let reservation = Arc::clone(&context.reservation);
        let arrow_reader_metrics = ArrowReaderMetrics::enabled();
        let first = fixture.decoder_for_row_groups_with_filter(
            Some(vec![0]),
            &arrow_reader_metrics,
            Some(0),
        );
        let fully_matched = fixture.decoder_for_row_groups_with_filter(
            Some(vec![1]),
            &arrow_reader_metrics,
            None,
        );
        let last = fixture.decoder_for_row_groups_with_filter(
            Some(vec![2]),
            &arrow_reader_metrics,
            Some(6),
        );
        let stream = LookaheadPushDecoderStreamState::new(
            first,
            VecDeque::from([fully_matched, last]),
            fixture.reader(control),
            fixture.output_state(arrow_reader_metrics),
            context,
        )
        .into_stream();

        let batches = stream.try_collect::<Vec<_>>().await.unwrap();

        assert_eq!(
            batches.iter().flat_map(batch_values).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 7, 8]
        );
        assert_eq!(reservation.size(), 0);
        assert_eq!(
            coordinator.range_permits.available_permits(),
            MAX_IN_FLIGHT_RANGES
        );
    }
}
