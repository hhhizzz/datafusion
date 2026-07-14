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

//! Diagnostic object-store metrics for benchmark runs.

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use datafusion_common::instant::Instant;
use futures::stream::BoxStream;
use futures::{Stream, StreamExt};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetRange, GetResult, GetResultPayload, ListResult,
    MultipartUpload, OBJECT_STORE_COALESCE_DEFAULT, ObjectMeta, ObjectStore,
    ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    RenameOptions, Result, coalesce_ranges,
};
use serde::Serialize;

#[derive(Debug, Clone, Copy)]
enum RequestKind {
    Head,
    Range,
    Full,
}

#[derive(Debug, Default)]
struct PathMetrics {
    head_requests: u64,
    range_get_requests: u64,
    full_get_requests: u64,
    response_range_bytes: u64,
    body_bytes: u64,
    errors: u64,
    in_flight: u64,
    peak_in_flight: u64,
    header_latency_ns: u64,
    body_latency_ns: u64,
    max_body_latency_ns: u64,
    first_request_started: Option<Instant>,
    last_request_finished: Option<Instant>,
}

#[derive(Debug, Default)]
struct RequestWindow {
    first_started: Option<Instant>,
    last_finished: Option<Instant>,
}

#[derive(Debug, Default)]
struct MetricsState {
    coalesce_gap_bytes: AtomicU64,
    get_ranges_calls: AtomicU64,
    logical_ranges: AtomicU64,
    logical_range_bytes: AtomicU64,
    head_requests: AtomicU64,
    range_get_requests: AtomicU64,
    full_get_requests: AtomicU64,
    response_range_bytes: AtomicU64,
    body_bytes: AtomicU64,
    list_requests: AtomicU64,
    errors: AtomicU64,
    in_flight: AtomicU64,
    peak_in_flight: AtomicU64,
    header_latencies_ns: Mutex<Vec<u64>>,
    body_latencies_ns: Mutex<Vec<u64>>,
    wire_range_sizes: Mutex<Vec<u64>>,
    request_window: Mutex<RequestWindow>,
    paths: Mutex<HashMap<String, PathMetrics>>,
}

/// A clonable handle for resetting and reading diagnostic object-store metrics.
#[derive(Debug, Clone, Default)]
pub struct ObjectStoreMetrics {
    state: Arc<MetricsState>,
}

/// Serializable metrics for one object path.
#[derive(Debug, Serialize)]
pub struct PathMetricsSnapshot {
    pub path: String,
    pub head_requests: u64,
    pub range_get_requests: u64,
    pub full_get_requests: u64,
    pub response_range_bytes: u64,
    pub body_bytes: u64,
    pub errors: u64,
    pub peak_in_flight: u64,
    pub header_latency_total_ms: f64,
    pub body_latency_total_ms: f64,
    pub body_latency_mean_ms: f64,
    pub body_latency_max_ms: f64,
    pub request_window_ms: f64,
    pub body_throughput_mib_per_s: f64,
}

/// Serializable metrics collected since the last reset.
#[derive(Debug, Serialize)]
pub struct ObjectStoreMetricsSnapshot {
    pub coalesce_gap_bytes: u64,
    pub get_ranges_calls: u64,
    pub logical_ranges: u64,
    pub logical_range_bytes: u64,
    pub head_requests: u64,
    pub range_get_requests: u64,
    pub full_get_requests: u64,
    pub response_range_bytes: u64,
    pub body_bytes: u64,
    pub range_overfetch_bytes: u64,
    pub list_requests: u64,
    pub errors: u64,
    pub in_flight: u64,
    pub peak_in_flight: u64,
    pub wire_range_size_p50_bytes: u64,
    pub wire_range_size_p95_bytes: u64,
    pub wire_range_size_max_bytes: u64,
    pub header_latency_p50_ms: f64,
    pub header_latency_p95_ms: f64,
    pub header_latency_p99_ms: f64,
    pub header_latency_max_ms: f64,
    pub body_latency_p50_ms: f64,
    pub body_latency_p95_ms: f64,
    pub body_latency_p99_ms: f64,
    pub body_latency_max_ms: f64,
    pub request_window_ms: f64,
    pub body_throughput_mib_per_s: f64,
    pub paths: Vec<PathMetricsSnapshot>,
}

impl ObjectStoreMetrics {
    /// Reset all metrics. Call this only when no instrumented requests are active.
    pub fn reset(&self) {
        for metric in [
            &self.state.get_ranges_calls,
            &self.state.logical_ranges,
            &self.state.logical_range_bytes,
            &self.state.head_requests,
            &self.state.range_get_requests,
            &self.state.full_get_requests,
            &self.state.response_range_bytes,
            &self.state.body_bytes,
            &self.state.list_requests,
            &self.state.errors,
            &self.state.in_flight,
            &self.state.peak_in_flight,
        ] {
            metric.store(0, Ordering::Relaxed);
        }
        self.state
            .header_latencies_ns
            .lock()
            .expect("metrics mutex poisoned")
            .clear();
        self.state
            .body_latencies_ns
            .lock()
            .expect("metrics mutex poisoned")
            .clear();
        self.state
            .wire_range_sizes
            .lock()
            .expect("metrics mutex poisoned")
            .clear();
        *self
            .state
            .request_window
            .lock()
            .expect("metrics mutex poisoned") = RequestWindow::default();
        self.state
            .paths
            .lock()
            .expect("metrics mutex poisoned")
            .clear();
    }

    /// Return a consistent-enough diagnostic snapshot after requests have completed.
    pub fn snapshot(&self) -> ObjectStoreMetricsSnapshot {
        let header = self
            .state
            .header_latencies_ns
            .lock()
            .expect("metrics mutex poisoned")
            .clone();
        let body = self
            .state
            .body_latencies_ns
            .lock()
            .expect("metrics mutex poisoned")
            .clone();
        let range_sizes = self
            .state
            .wire_range_sizes
            .lock()
            .expect("metrics mutex poisoned")
            .clone();
        let global_request_window_ns = request_window_ns(
            &self
                .state
                .request_window
                .lock()
                .expect("metrics mutex poisoned"),
        );
        let mut paths = self
            .state
            .paths
            .lock()
            .expect("metrics mutex poisoned")
            .iter()
            .map(|(path, metrics)| {
                let body_requests =
                    metrics.range_get_requests + metrics.full_get_requests;
                let request_window_ns = request_window_ns(&RequestWindow {
                    first_started: metrics.first_request_started,
                    last_finished: metrics.last_request_finished,
                });
                PathMetricsSnapshot {
                    path: path.clone(),
                    head_requests: metrics.head_requests,
                    range_get_requests: metrics.range_get_requests,
                    full_get_requests: metrics.full_get_requests,
                    response_range_bytes: metrics.response_range_bytes,
                    body_bytes: metrics.body_bytes,
                    errors: metrics.errors,
                    peak_in_flight: metrics.peak_in_flight,
                    header_latency_total_ms: ns_to_ms(metrics.header_latency_ns),
                    body_latency_total_ms: ns_to_ms(metrics.body_latency_ns),
                    body_latency_mean_ms: if body_requests == 0 {
                        0.0
                    } else {
                        ns_to_ms(metrics.body_latency_ns) / body_requests as f64
                    },
                    body_latency_max_ms: ns_to_ms(metrics.max_body_latency_ns),
                    request_window_ms: ns_to_ms(request_window_ns),
                    body_throughput_mib_per_s: body_throughput_mib_per_s(
                        metrics.body_bytes,
                        request_window_ns,
                    ),
                }
            })
            .collect::<Vec<_>>();
        paths.sort_unstable_by(|a, b| {
            b.body_bytes
                .cmp(&a.body_bytes)
                .then_with(|| a.path.cmp(&b.path))
        });

        let logical_range_bytes = self.state.logical_range_bytes.load(Ordering::Relaxed);
        let response_range_bytes =
            self.state.response_range_bytes.load(Ordering::Relaxed);
        let body_bytes = self.state.body_bytes.load(Ordering::Relaxed);
        ObjectStoreMetricsSnapshot {
            coalesce_gap_bytes: self.state.coalesce_gap_bytes.load(Ordering::Relaxed),
            get_ranges_calls: self.state.get_ranges_calls.load(Ordering::Relaxed),
            logical_ranges: self.state.logical_ranges.load(Ordering::Relaxed),
            logical_range_bytes,
            head_requests: self.state.head_requests.load(Ordering::Relaxed),
            range_get_requests: self.state.range_get_requests.load(Ordering::Relaxed),
            full_get_requests: self.state.full_get_requests.load(Ordering::Relaxed),
            response_range_bytes,
            body_bytes,
            range_overfetch_bytes: response_range_bytes
                .saturating_sub(logical_range_bytes),
            list_requests: self.state.list_requests.load(Ordering::Relaxed),
            errors: self.state.errors.load(Ordering::Relaxed),
            in_flight: self.state.in_flight.load(Ordering::Relaxed),
            peak_in_flight: self.state.peak_in_flight.load(Ordering::Relaxed),
            wire_range_size_p50_bytes: percentile(&range_sizes, 50),
            wire_range_size_p95_bytes: percentile(&range_sizes, 95),
            wire_range_size_max_bytes: percentile(&range_sizes, 100),
            header_latency_p50_ms: ns_to_ms(percentile(&header, 50)),
            header_latency_p95_ms: ns_to_ms(percentile(&header, 95)),
            header_latency_p99_ms: ns_to_ms(percentile(&header, 99)),
            header_latency_max_ms: ns_to_ms(percentile(&header, 100)),
            body_latency_p50_ms: ns_to_ms(percentile(&body, 50)),
            body_latency_p95_ms: ns_to_ms(percentile(&body, 95)),
            body_latency_p99_ms: ns_to_ms(percentile(&body, 99)),
            body_latency_max_ms: ns_to_ms(percentile(&body, 100)),
            request_window_ms: ns_to_ms(global_request_window_ns),
            body_throughput_mib_per_s: body_throughput_mib_per_s(
                body_bytes,
                global_request_window_ns,
            ),
            paths,
        }
    }

    fn request_started(&self, path: &str, kind: RequestKind) -> RequestTracker {
        match kind {
            RequestKind::Head => &self.state.head_requests,
            RequestKind::Range => &self.state.range_get_requests,
            RequestKind::Full => &self.state.full_get_requests,
        }
        .fetch_add(1, Ordering::Relaxed);

        let in_flight = self.state.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        self.state
            .peak_in_flight
            .fetch_max(in_flight, Ordering::Relaxed);
        let started = Instant::now();
        if !matches!(kind, RequestKind::Head) {
            let mut window = self
                .state
                .request_window
                .lock()
                .expect("metrics mutex poisoned");
            window.first_started.get_or_insert(started);
        }
        let mut paths = self.state.paths.lock().expect("metrics mutex poisoned");
        let path_metrics = paths.entry(path.to_string()).or_default();
        match kind {
            RequestKind::Head => path_metrics.head_requests += 1,
            RequestKind::Range => path_metrics.range_get_requests += 1,
            RequestKind::Full => path_metrics.full_get_requests += 1,
        }
        path_metrics.in_flight += 1;
        path_metrics.peak_in_flight =
            path_metrics.peak_in_flight.max(path_metrics.in_flight);
        if !matches!(kind, RequestKind::Head) {
            path_metrics.first_request_started.get_or_insert(started);
        }
        drop(paths);

        RequestTracker {
            metrics: self.clone(),
            path: path.to_string(),
            kind,
            started,
            body_bytes: 0,
            failed: false,
            finished: false,
        }
    }

    fn header_finished(
        &self,
        path: &str,
        kind: RequestKind,
        elapsed: Duration,
        response_bytes: u64,
    ) {
        let elapsed = duration_ns(elapsed);
        self.state
            .header_latencies_ns
            .lock()
            .expect("metrics mutex poisoned")
            .push(elapsed);
        if matches!(kind, RequestKind::Range) {
            self.state
                .response_range_bytes
                .fetch_add(response_bytes, Ordering::Relaxed);
            self.state
                .wire_range_sizes
                .lock()
                .expect("metrics mutex poisoned")
                .push(response_bytes);
        }
        let mut paths = self.state.paths.lock().expect("metrics mutex poisoned");
        let metrics = paths.entry(path.to_string()).or_default();
        metrics.header_latency_ns = metrics.header_latency_ns.saturating_add(elapsed);
        if matches!(kind, RequestKind::Range) {
            metrics.response_range_bytes =
                metrics.response_range_bytes.saturating_add(response_bytes);
        }
    }

    fn request_failed(&self, path: &str) {
        self.state.errors.fetch_add(1, Ordering::Relaxed);
        self.state
            .paths
            .lock()
            .expect("metrics mutex poisoned")
            .entry(path.to_string())
            .or_default()
            .errors += 1;
    }

    fn request_finished(&self, tracker: &RequestTracker) {
        let elapsed = duration_ns(tracker.started.elapsed());
        let finished = Instant::now();
        self.state
            .body_bytes
            .fetch_add(tracker.body_bytes, Ordering::Relaxed);
        self.state
            .body_latencies_ns
            .lock()
            .expect("metrics mutex poisoned")
            .push(elapsed);
        self.state.in_flight.fetch_sub(1, Ordering::Relaxed);
        if !matches!(tracker.kind, RequestKind::Head) {
            self.state
                .request_window
                .lock()
                .expect("metrics mutex poisoned")
                .last_finished = Some(finished);
        }

        let mut paths = self.state.paths.lock().expect("metrics mutex poisoned");
        let metrics = paths.entry(tracker.path.clone()).or_default();
        metrics.body_bytes = metrics.body_bytes.saturating_add(tracker.body_bytes);
        metrics.body_latency_ns = metrics.body_latency_ns.saturating_add(elapsed);
        metrics.max_body_latency_ns = metrics.max_body_latency_ns.max(elapsed);
        metrics.in_flight = metrics.in_flight.saturating_sub(1);
        if !matches!(tracker.kind, RequestKind::Head) {
            metrics.last_request_finished = Some(finished);
        }
    }
}

/// An [`ObjectStore`] wrapper that records logical and wire-level read metrics.
#[derive(Debug)]
pub struct MetricsObjectStore<T: ObjectStore> {
    inner: T,
    metrics: ObjectStoreMetrics,
    coalesce_gap_bytes: u64,
    metrics_enabled: bool,
}

impl<T: ObjectStore> MetricsObjectStore<T> {
    pub fn new(inner: T) -> Self {
        Self::new_with_coalesce_gap(inner, OBJECT_STORE_COALESCE_DEFAULT)
    }

    pub fn new_with_coalesce_gap(inner: T, coalesce_gap_bytes: u64) -> Self {
        Self::new_inner(inner, coalesce_gap_bytes, true)
    }

    pub fn new_coalescing(inner: T, coalesce_gap_bytes: u64) -> Self {
        Self::new_inner(inner, coalesce_gap_bytes, false)
    }

    fn new_inner(inner: T, coalesce_gap_bytes: u64, metrics_enabled: bool) -> Self {
        let metrics = ObjectStoreMetrics::default();
        metrics
            .state
            .coalesce_gap_bytes
            .store(coalesce_gap_bytes, Ordering::Relaxed);
        Self {
            inner,
            metrics,
            coalesce_gap_bytes,
            metrics_enabled,
        }
    }

    pub fn metrics(&self) -> ObjectStoreMetrics {
        self.metrics.clone()
    }

    async fn instrumented_get_opts(
        &self,
        location: &Path,
        options: GetOptions,
        count_as_logical_range: bool,
    ) -> Result<GetResult> {
        let kind = if options.head {
            RequestKind::Head
        } else if options.range.is_some() {
            RequestKind::Range
        } else {
            RequestKind::Full
        };
        if count_as_logical_range
            && let Some(GetRange::Bounded(range)) = options.range.as_ref()
        {
            self.metrics
                .state
                .logical_ranges
                .fetch_add(1, Ordering::Relaxed);
            self.metrics
                .state
                .logical_range_bytes
                .fetch_add(range.end - range.start, Ordering::Relaxed);
        }

        let mut tracker = self.metrics.request_started(location.as_ref(), kind);
        let result = self.inner.get_opts(location, options).await;
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                tracker.mark_failed();
                return Err(error);
            }
        };
        let response_bytes = result.range.end - result.range.start;
        self.metrics.header_finished(
            location.as_ref(),
            kind,
            tracker.started.elapsed(),
            response_bytes,
        );

        let payload = match result.payload {
            payload @ GetResultPayload::File(_, _) => {
                tracker.finish();
                payload
            }
            GetResultPayload::Stream(stream) => GetResultPayload::Stream(
                MetricsStream {
                    inner: stream,
                    tracker: Some(tracker),
                }
                .boxed(),
            ),
        };
        Ok(GetResult { payload, ..result })
    }

    async fn get_coalesced_range(
        &self,
        location: &Path,
        range: Range<u64>,
    ) -> Result<Bytes> {
        if !self.metrics_enabled {
            return self.inner.get_range(location, range).await;
        }
        let options = GetOptions::new().with_range(Some(range));
        self.instrumented_get_opts(location, options, false)
            .await?
            .bytes()
            .await
    }
}

impl<T: ObjectStore> fmt::Display for MetricsObjectStore<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetricsObjectStore({})", self.inner)
    }
}

#[async_trait]
impl<T: ObjectStore> ObjectStore for MetricsObjectStore<T> {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        if self.metrics_enabled {
            self.instrumented_get_opts(location, options, true).await
        } else {
            self.inner.get_opts(location, options).await
        }
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> Result<Vec<Bytes>> {
        if self.metrics_enabled {
            self.metrics
                .state
                .get_ranges_calls
                .fetch_add(1, Ordering::Relaxed);
            self.metrics
                .state
                .logical_ranges
                .fetch_add(ranges.len() as u64, Ordering::Relaxed);
            self.metrics.state.logical_range_bytes.fetch_add(
                ranges.iter().map(|range| range.end - range.start).sum(),
                Ordering::Relaxed,
            );
        }
        coalesce_ranges(
            ranges,
            |range| self.get_coalesced_range(location, range),
            self.coalesce_gap_bytes,
        )
        .await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        if self.metrics_enabled {
            self.metrics
                .state
                .list_requests
                .fetch_add(1, Ordering::Relaxed);
        }
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        if self.metrics_enabled {
            self.metrics
                .state
                .list_requests
                .fetch_add(1, Ordering::Relaxed);
        }
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> Result<()> {
        self.inner.copy_opts(from, to, options).await
    }

    async fn rename_opts(
        &self,
        from: &Path,
        to: &Path,
        options: RenameOptions,
    ) -> Result<()> {
        self.inner.rename_opts(from, to, options).await
    }
}

struct RequestTracker {
    metrics: ObjectStoreMetrics,
    path: String,
    kind: RequestKind,
    started: Instant,
    body_bytes: u64,
    failed: bool,
    finished: bool,
}

impl RequestTracker {
    fn mark_failed(&mut self) {
        if !self.failed {
            self.failed = true;
            self.metrics.request_failed(&self.path);
        }
    }

    fn finish(&mut self) {
        if !self.finished {
            self.finished = true;
            self.metrics.request_finished(self);
        }
    }
}

impl Drop for RequestTracker {
    fn drop(&mut self) {
        self.finish();
    }
}

struct MetricsStream {
    inner: BoxStream<'static, Result<Bytes>>,
    tracker: Option<RequestTracker>,
}

impl Stream for MetricsStream {
    type Item = Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                if let Some(tracker) = this.tracker.as_mut() {
                    tracker.body_bytes =
                        tracker.body_bytes.saturating_add(bytes.len() as u64);
                }
                Poll::Ready(Some(Ok(bytes)))
            }
            Poll::Ready(Some(Err(error))) => {
                if let Some(tracker) = this.tracker.as_mut() {
                    tracker.mark_failed();
                }
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                if let Some(mut tracker) = this.tracker.take() {
                    tracker.finish();
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

fn request_window_ns(window: &RequestWindow) -> u64 {
    match (window.first_started, window.last_finished) {
        (Some(started), Some(finished)) => {
            duration_ns(finished.saturating_duration_since(started))
        }
        _ => 0,
    }
}

fn body_throughput_mib_per_s(body_bytes: u64, request_window_ns: u64) -> f64 {
    if request_window_ns == 0 {
        return 0.0;
    }
    body_bytes as f64 * 1_000_000_000.0 / request_window_ns as f64 / (1024.0 * 1024.0)
}

fn percentile(values: &[u64], percentile: usize) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut values = values.to_vec();
    values.sort_unstable();
    let index = (values.len() * percentile).div_ceil(100).saturating_sub(1);
    values[index.min(values.len() - 1)]
}
