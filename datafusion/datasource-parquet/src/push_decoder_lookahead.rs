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

use std::collections::VecDeque;
use std::sync::Arc;
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
    LookaheadFileContext, MAX_RANGES_PER_FILE_FETCH, SpeculativeLease,
};

/// A depth-one push-decoder driver that overlaps the next row-group fetch with
/// synchronous decoding of the current row group.
pub(crate) struct LookaheadPushDecoderStreamState {
    decoder: Option<ParquetPushDecoder>,
    pending_decoders: VecDeque<ParquetPushDecoder>,
    reader: Option<Box<dyn AsyncFileReader>>,
    output: PushDecoderOutputState,
    lookahead: LookaheadFileContext,
    active_reader: Option<ParquetRecordBatchReader>,
    next_reader_future: Option<BoxFuture<'static, NextReaderOutcome>>,
    prefetched_reader: Option<PrefetchedReader>,
    foreground_resources: SpeculativeResources,
    deferred_error: Option<DataFusionError>,
    run_finished: bool,
    speculation_disabled: bool,
    terminated: bool,
}

struct PrefetchedReader {
    reader: ParquetRecordBatchReader,
    resources: SpeculativeResources,
}

#[derive(Default)]
struct SpeculativeResources {
    leases: Vec<SpeculativeLease>,
    range_permits: Vec<OwnedSemaphorePermit>,
}

struct NextReaderOutcome {
    decoder: ParquetPushDecoder,
    reader: Box<dyn AsyncFileReader>,
    result: NextReaderResult,
    resources: SpeculativeResources,
}

enum NextReaderResult {
    Reader(ParquetRecordBatchReader),
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
            reader: Some(reader),
            output,
            lookahead,
            active_reader: None,
            next_reader_future: None,
            prefetched_reader: None,
            foreground_resources: SpeculativeResources::default(),
            deferred_error: None,
            run_finished: false,
            speculation_disabled: false,
            terminated: false,
        }
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
                if let Some(prefetched) = self.prefetched_reader.take() {
                    self.active_reader = Some(prefetched.reader);
                    drop(prefetched.resources);
                    continue;
                }

                if let Some(error) = self.deferred_error.take() {
                    self.terminate();
                    return Some((Err(error), self));
                }

                if self.run_finished {
                    if !self.advance_decoder_run() {
                        self.terminate();
                        return None;
                    }
                    continue;
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
            || self.prefetched_reader.is_some()
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
            drive_speculative_next_reader(decoder, reader, self.lookahead.clone())
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
            NextReaderResult::Reader(next_reader) => {
                self.decoder = Some(decoder);
                self.reader = Some(reader);
                self.prefetched_reader = Some(PrefetchedReader {
                    reader: next_reader,
                    resources,
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
            let decode_result = self
                .decoder
                .as_mut()
                .expect("foreground decoder must be present")
                .try_next_reader()
                .map_err(DataFusionError::from)?;
            match decode_result {
                DecodeResult::NeedsData(ranges) => {
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
                    self.active_reader = Some(reader);
                    self.foreground_resources = SpeculativeResources::default();
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

    fn advance_decoder_run(&mut self) -> bool {
        self.run_finished = false;
        let Some(decoder) = self.pending_decoders.pop_front() else {
            return false;
        };
        self.decoder = Some(decoder);
        true
    }

    fn terminate(&mut self) {
        self.terminated = true;
        self.next_reader_future = None;
        self.prefetched_reader = None;
        self.active_reader = None;
        self.decoder = None;
        self.reader = None;
        self.pending_decoders.clear();
        self.foreground_resources = SpeculativeResources::default();
        self.deferred_error = None;
        self.run_finished = false;
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
) -> NextReaderOutcome {
    let mut resources = SpeculativeResources::default();
    let result = loop {
        match decoder.try_next_reader() {
            Ok(DecodeResult::NeedsData(ranges)) => {
                if ranges.len() > MAX_RANGES_PER_FILE_FETCH {
                    break NextReaderResult::Denied;
                }
                let Some(bytes) = ranges.iter().try_fold(0usize, |sum, range| {
                    let len =
                        usize::try_from(range.end.checked_sub(range.start)?).ok()?;
                    sum.checked_add(len)
                }) else {
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
                break NextReaderResult::Reader(reader);
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

#[cfg(test)]
mod tests {
    use std::collections::{HashSet, VecDeque};
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
    use crate::lookahead::{
        LookaheadFileContext, MAX_IN_FLIGHT_RANGES, ParquetLookaheadCoordinator,
    };

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
            row_group: usize,
            ranges: Vec<Range<u64>>,
        ) -> ActiveFetchGuard {
            self.inner
                .requests
                .lock()
                .unwrap()
                .push(RequestRecord { row_group, ranges });
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

        fn has_request_for(&self, row_group: usize) -> bool {
            !self.requests_for(row_group).is_empty()
        }

        fn active_fetches(&self) -> usize {
            self.inner.active_fetches.load(Ordering::SeqCst)
        }

        fn should_fail(&self, row_group: usize) -> bool {
            self.inner.failed_row_group == Some(row_group)
                && self.requests_for(row_group).len()
                    > self.inner.successful_requests_before_failure
        }

        fn should_corrupt(&self, row_group: usize) -> bool {
            self.inner.corrupted_row_group == Some(row_group)
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

    struct ScriptedAsyncFileReader {
        data: Bytes,
        metadata: Arc<ParquetMetaData>,
        row_group_spans: Arc<Vec<Range<u64>>>,
        control: ScriptControl,
    }

    impl ScriptedAsyncFileReader {
        fn row_group_for_ranges(&self, ranges: &[Range<u64>]) -> usize {
            let mut row_groups = ranges.iter().map(|range| {
                self.row_group_spans
                    .iter()
                    .position(|span| range.start >= span.start && range.end <= span.end)
                    .unwrap_or_else(|| {
                        panic!("range {range:?} is outside row-group data")
                    })
            });
            let row_group = row_groups.next().expect("non-empty range request");
            assert!(
                row_groups.all(|candidate| candidate == row_group),
                "one request must not span row groups"
            );
            row_group
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
            let row_group = self.row_group_for_ranges(&ranges);
            let data = self.data.clone();
            let control = self.control.clone();
            async move {
                let _active_fetch = control.record_fetch(row_group, ranges.clone());
                control.wait_if_gated(row_group).await;
                if control.should_fail(row_group) {
                    return Err(parquet::errors::ParquetError::General(format!(
                        "scripted row-group {row_group} failure"
                    )));
                }
                Ok(ranges
                    .into_iter()
                    .map(|range| {
                        if control.should_corrupt(row_group) {
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
    }

    impl ThreeRowGroupFixture {
        fn new() -> Self {
            Self::with_filter_columns(&LOOKAHEAD_FILTER_COLUMNS)
        }

        fn with_filter_columns(filter_columns: &[&'static str]) -> Self {
            let schema = Arc::new(Schema::new(
                FILTER_COLUMNS
                    .iter()
                    .map(|name| Field::new(*name, DataType::Int32, false))
                    .collect::<Vec<_>>(),
            ));
            let values = Arc::new(Int32Array::from_iter_values(0..9)) as ArrayRef;
            let columns = (0..FILTER_COLUMNS.len())
                .map(|_| Arc::clone(&values))
                .collect::<Vec<_>>();
            let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
            let properties = WriterProperties::builder()
                .set_max_row_group_row_count(Some(3))
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
            assert_eq!(metadata.num_row_groups(), 3);
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
            }
        }

        fn with_disjoint_filter_and_projection() -> Self {
            Self::with_filter_columns(&DISJOINT_FILTER_COLUMNS)
        }

        fn reader(&self, control: ScriptControl) -> Box<dyn AsyncFileReader> {
            Box::new(ScriptedAsyncFileReader {
                data: self.data.clone(),
                metadata: Arc::clone(&self.metadata),
                row_group_spans: Arc::clone(&self.row_group_spans),
                control,
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
            let mut builder =
                ParquetPushDecoderBuilder::try_new_decoder(Arc::clone(&self.metadata))
                    .unwrap();
            let schema_descr = builder.metadata().file_metadata().schema_descr_ptr();
            if let Some(row_groups) = row_groups {
                builder = builder.with_row_groups(row_groups);
            }
            builder = builder
                .with_projection(ProjectionMask::columns(&schema_descr, ["value"]))
                .with_batch_size(1)
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
            let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
            self.lookahead_context_with_pool(pool)
        }

        fn lookahead_context_with_pool(
            &self,
            pool: Arc<dyn MemoryPool>,
        ) -> (Arc<ParquetLookaheadCoordinator>, LookaheadFileContext) {
            let coordinator = Arc::new(ParquetLookaheadCoordinator::new(1));
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
        let control = ScriptControl::new(Some(1));
        let (coordinator, context) = fixture.lookahead_context();
        let reservation = Arc::clone(&context.reservation);
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
    }

    #[tokio::test]
    async fn final_limit_batch_synchronously_releases_pending_lookahead() {
        let fixture = ThreeRowGroupFixture::new();
        let control = ScriptControl::new(Some(1));
        let (coordinator, context) = fixture.lookahead_context();
        let reservation = Arc::clone(&context.reservation);
        let mut stream =
            fixture.lookahead_stream_with_limit(control.clone(), context, Some(1));

        let first = stream.next().await.unwrap().unwrap();

        assert_eq!(batch_values(&first), vec![1]);
        assert!(control.has_request_for(1));
        assert_eq!(control.active_fetches(), 0);
        assert_eq!(reservation.size(), 0);
        assert_eq!(
            coordinator.range_permits.available_permits(),
            MAX_IN_FLIGHT_RANGES
        );
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn speculative_error_is_deferred_until_foreground_reader_drains() {
        let fixture = ThreeRowGroupFixture::new();
        let control = ScriptControl::failing(1);
        let (coordinator, context) = fixture.lookahead_context();
        let reservation = Arc::clone(&context.reservation);
        let mut stream = fixture.lookahead_stream(control.clone(), context);

        let first = stream.next().await.unwrap().unwrap();
        let second = stream.next().await.unwrap().unwrap();
        let error = stream.next().await.unwrap().unwrap_err();

        assert_eq!(batch_values(&first), vec![1]);
        assert_eq!(batch_values(&second), vec![2]);
        assert!(error.to_string().contains("scripted row-group 1 failure"));
        assert!(!control.has_request_for(2));
        assert_eq!(control.active_fetches(), 0);
        assert_eq!(reservation.size(), 0);
        assert_eq!(
            coordinator.range_permits.available_permits(),
            MAX_IN_FLIGHT_RANGES
        );
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
        let control = ScriptControl::corrupting(0);
        let (coordinator, context) = fixture.lookahead_context();
        let reservation = Arc::clone(&context.reservation);
        let arrow_reader_metrics = ArrowReaderMetrics::enabled();
        let decoder =
            fixture.decoder_for_row_groups_with_filter(None, &arrow_reader_metrics, None);
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

    #[tokio::test]
    async fn replacement_schema_error_cancels_pending_prefetch_and_terminates() {
        let fixture = ThreeRowGroupFixture::new();
        let control = ScriptControl::new(Some(1));
        let (coordinator, context) = fixture.lookahead_context();
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

    #[tokio::test]
    async fn pending_decoder_run_is_not_prefetched_across_boundary() {
        let fixture = ThreeRowGroupFixture::new();
        let (serial, _) = fixture.serial_oracle().await;
        let control = ScriptControl::new(Some(1));
        let (_, context) = fixture.lookahead_context();
        let arrow_reader_metrics = ArrowReaderMetrics::enabled();
        let decoder =
            fixture.decoder_for_row_groups(Some(vec![0]), &arrow_reader_metrics);
        let pending_decoder =
            fixture.decoder_for_row_groups(Some(vec![1, 2]), &arrow_reader_metrics);
        let mut stream = LookaheadPushDecoderStreamState::new(
            decoder,
            VecDeque::from([pending_decoder]),
            fixture.reader(control.clone()),
            fixture.output_state(arrow_reader_metrics),
            context,
        )
        .into_stream();

        let first = stream.next().await.unwrap().unwrap();
        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(batch_values(&first), vec![1]);
        assert_eq!(batch_values(&second), vec![2]);
        assert!(!control.has_request_for(1));

        let mut next = Box::pin(stream.next());
        assert!(matches!(futures::poll!(next.as_mut()), Poll::Pending));
        assert!(control.has_request_for(1));
        assert!(!control.has_request_for(2));

        control.release();
        let third = next.await.unwrap().unwrap();
        let mut lookahead = vec![first, second, third];
        lookahead.extend(stream.try_collect::<Vec<_>>().await.unwrap());
        assert_eq!(lookahead, serial);
    }

    #[tokio::test]
    async fn partial_range_permit_denial_continues_without_duplicate_ranges() {
        let fixture = ThreeRowGroupFixture::new();
        let (serial, serial_control) = fixture.serial_oracle().await;
        let control = ScriptControl::new(None);
        let (coordinator, context) = fixture.lookahead_context();
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
        let control = ScriptControl::new(None);
        let pool = Arc::new(GreedyMemoryPool::new(0));
        let (coordinator, context) = fixture.lookahead_context_with_pool(pool.clone());
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
    }

    #[tokio::test]
    async fn partial_memory_denial_preserves_ranges_and_disables_later_speculation() {
        let fixture = ThreeRowGroupFixture::with_disjoint_filter_and_projection();
        let (serial, serial_control) = fixture.serial_oracle().await;
        let first_chunk_bytes = serial_control.requests_for(1)[0].ranges[..4]
            .iter()
            .map(|range| usize::try_from(range.end - range.start).unwrap())
            .sum();
        let control = ScriptControl::new(None);
        let pool = Arc::new(GreedyMemoryPool::new(first_chunk_bytes));
        let (coordinator, context) = fixture.lookahead_context_with_pool(pool.clone());
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
        let control = ScriptControl::new(None);
        let (coordinator, context) = fixture.lookahead_context();
        let reservation = Arc::clone(&context.reservation);
        let arrow_reader_metrics = ArrowReaderMetrics::enabled();
        let decoder =
            fixture.decoder_for_row_groups(Some(vec![2, 1, 0]), &arrow_reader_metrics);
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
