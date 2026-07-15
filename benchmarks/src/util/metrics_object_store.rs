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
use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use datafusion::execution::memory_pool::MemoryReservation;
use datafusion_common::instant::Instant;
use futures::stream::BoxStream;
use futures::{Stream, StreamExt, TryStreamExt};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetRange, GetResult, GetResultPayload, ListResult,
    MultipartUpload, OBJECT_STORE_COALESCE_DEFAULT, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions, Result,
};
use serde::Serialize;
use tokio::sync::{OnceCell, OwnedSemaphorePermit, Semaphore};

/// Default range-request concurrency used by object_store 0.13.2.
pub const OBJECT_STORE_COALESCE_PARALLEL_DEFAULT: usize = 10;
/// Benchmark guardrail for configurable range-request concurrency.
pub const OBJECT_STORE_COALESCE_PARALLEL_MAX: usize = 24;

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ExactRangeCacheKey {
    path: String,
    start: u64,
    end: u64,
}

type ExactRangeCacheValue = std::result::Result<Bytes, Arc<object_store::Error>>;

#[derive(Debug)]
struct ExactRangeCacheEntry {
    value: OnceCell<ExactRangeCacheValue>,
    reserved_bytes: usize,
}

impl ExactRangeCacheEntry {
    fn new(reserved_bytes: usize) -> Self {
        Self {
            value: OnceCell::new(),
            reserved_bytes,
        }
    }
}

#[derive(Debug, Default)]
struct ExactRangeCacheInner {
    entries: HashMap<ExactRangeCacheKey, Arc<ExactRangeCacheEntry>>,
    retained_bytes: usize,
}

enum ExactRangeCacheAccess {
    Ready(Bytes),
    Inflight(Arc<ExactRangeCacheEntry>),
    Failed(Arc<ExactRangeCacheEntry>, Arc<object_store::Error>),
    Miss {
        entry: Arc<ExactRangeCacheEntry>,
        retained_bytes: usize,
    },
    AdmissionDenied,
}

#[derive(Debug)]
struct ExactRangeCache {
    max_bytes: usize,
    reservation: Arc<MemoryReservation>,
    inner: Mutex<ExactRangeCacheInner>,
}

impl ExactRangeCache {
    fn new(max_bytes: usize, reservation: Arc<MemoryReservation>) -> Self {
        Self {
            max_bytes,
            reservation,
            inner: Mutex::new(ExactRangeCacheInner::default()),
        }
    }

    fn access(
        &self,
        key: ExactRangeCacheKey,
        requested_bytes: usize,
    ) -> ExactRangeCacheAccess {
        let mut inner = self.inner.lock().expect("exact range cache mutex poisoned");
        if let Some(entry) = inner.entries.get(&key) {
            return match entry.value.get() {
                Some(Ok(bytes)) => ExactRangeCacheAccess::Ready(bytes.clone()),
                Some(Err(error)) => {
                    ExactRangeCacheAccess::Failed(Arc::clone(entry), Arc::clone(error))
                }
                None => ExactRangeCacheAccess::Inflight(Arc::clone(entry)),
            };
        }

        if requested_bytes > self.max_bytes.saturating_sub(inner.retained_bytes)
            || self.reservation.try_grow(requested_bytes).is_err()
        {
            return ExactRangeCacheAccess::AdmissionDenied;
        }

        let entry = Arc::new(ExactRangeCacheEntry::new(requested_bytes));
        inner.retained_bytes += requested_bytes;
        let retained_bytes = inner.retained_bytes;
        inner.entries.insert(key, Arc::clone(&entry));
        ExactRangeCacheAccess::Miss {
            entry,
            retained_bytes,
        }
    }

    fn remove_failed(
        &self,
        key: &ExactRangeCacheKey,
        failed_entry: &Arc<ExactRangeCacheEntry>,
    ) -> Option<usize> {
        let mut inner = self.inner.lock().expect("exact range cache mutex poisoned");
        let should_remove = inner
            .entries
            .get(key)
            .is_some_and(|entry| Arc::ptr_eq(entry, failed_entry));
        if !should_remove {
            return None;
        }

        let removed = inner
            .entries
            .remove(key)
            .expect("checked exact range cache entry must exist");
        inner.retained_bytes =
            inner.retained_bytes.saturating_sub(removed.reserved_bytes);
        let retained_bytes = inner.retained_bytes;
        drop(inner);
        self.reservation.shrink(removed.reserved_bytes);
        Some(retained_bytes)
    }

    fn clear(&self) {
        let mut inner = self.inner.lock().expect("exact range cache mutex poisoned");
        let retained_bytes = inner.retained_bytes;
        inner.entries.clear();
        inner.retained_bytes = 0;
        drop(inner);
        if retained_bytes != 0 {
            self.reservation.shrink(retained_bytes);
        }
    }
}

#[derive(Debug, Default)]
struct MetricsState {
    coalesce_gap_bytes: AtomicU64,
    coalesce_parallelism: AtomicU64,
    get_ranges_calls: AtomicU64,
    get_ranges_max_logical_ranges: AtomicU64,
    get_ranges_max_coalesced_ranges: AtomicU64,
    get_ranges_parallelism_saturated_calls: AtomicU64,
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
    exact_range_cache_capacity_bytes: AtomicU64,
    exact_range_cache_ready_hits: AtomicU64,
    exact_range_cache_inflight_joins: AtomicU64,
    exact_range_cache_misses: AtomicU64,
    exact_range_cache_admission_denied: AtomicU64,
    exact_range_cache_saved_gets: AtomicU64,
    exact_range_cache_saved_bytes: AtomicU64,
    exact_range_cache_retained_bytes: AtomicU64,
    exact_range_cache_retained_bytes_high_watermark: AtomicU64,
    exact_range_cache: Mutex<Option<Arc<ExactRangeCache>>>,
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
    pub coalesce_parallelism: u64,
    pub get_ranges_calls: u64,
    pub get_ranges_max_logical_ranges: u64,
    pub get_ranges_max_coalesced_ranges: u64,
    pub get_ranges_parallelism_saturated_calls: u64,
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
    pub exact_range_cache_capacity_bytes: u64,
    pub exact_range_cache_ready_hits: u64,
    pub exact_range_cache_inflight_joins: u64,
    pub exact_range_cache_misses: u64,
    pub exact_range_cache_admission_denied: u64,
    pub exact_range_cache_saved_gets: u64,
    pub exact_range_cache_saved_bytes: u64,
    pub exact_range_cache_retained_bytes: u64,
    pub exact_range_cache_retained_bytes_high_watermark: u64,
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
            &self.state.get_ranges_max_logical_ranges,
            &self.state.get_ranges_max_coalesced_ranges,
            &self.state.get_ranges_parallelism_saturated_calls,
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
            &self.state.exact_range_cache_ready_hits,
            &self.state.exact_range_cache_inflight_joins,
            &self.state.exact_range_cache_misses,
            &self.state.exact_range_cache_admission_denied,
            &self.state.exact_range_cache_saved_gets,
            &self.state.exact_range_cache_saved_bytes,
            &self.state.exact_range_cache_retained_bytes,
            &self.state.exact_range_cache_retained_bytes_high_watermark,
        ] {
            metric.store(0, Ordering::Relaxed);
        }
        if let Some(cache) = self.exact_range_cache() {
            cache.clear();
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
            coalesce_parallelism: self.state.coalesce_parallelism.load(Ordering::Relaxed),
            get_ranges_calls: self.state.get_ranges_calls.load(Ordering::Relaxed),
            get_ranges_max_logical_ranges: self
                .state
                .get_ranges_max_logical_ranges
                .load(Ordering::Relaxed),
            get_ranges_max_coalesced_ranges: self
                .state
                .get_ranges_max_coalesced_ranges
                .load(Ordering::Relaxed),
            get_ranges_parallelism_saturated_calls: self
                .state
                .get_ranges_parallelism_saturated_calls
                .load(Ordering::Relaxed),
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
            exact_range_cache_capacity_bytes: self
                .state
                .exact_range_cache_capacity_bytes
                .load(Ordering::Relaxed),
            exact_range_cache_ready_hits: self
                .state
                .exact_range_cache_ready_hits
                .load(Ordering::Relaxed),
            exact_range_cache_inflight_joins: self
                .state
                .exact_range_cache_inflight_joins
                .load(Ordering::Relaxed),
            exact_range_cache_misses: self
                .state
                .exact_range_cache_misses
                .load(Ordering::Relaxed),
            exact_range_cache_admission_denied: self
                .state
                .exact_range_cache_admission_denied
                .load(Ordering::Relaxed),
            exact_range_cache_saved_gets: self
                .state
                .exact_range_cache_saved_gets
                .load(Ordering::Relaxed),
            exact_range_cache_saved_bytes: self
                .state
                .exact_range_cache_saved_bytes
                .load(Ordering::Relaxed),
            exact_range_cache_retained_bytes: self
                .state
                .exact_range_cache_retained_bytes
                .load(Ordering::Relaxed),
            exact_range_cache_retained_bytes_high_watermark: self
                .state
                .exact_range_cache_retained_bytes_high_watermark
                .load(Ordering::Relaxed),
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

    fn exact_range_cache(&self) -> Option<Arc<ExactRangeCache>> {
        self.state
            .exact_range_cache
            .lock()
            .expect("exact range cache configuration mutex poisoned")
            .clone()
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
    coalesce_parallelism: usize,
    request_permits: Arc<Semaphore>,
    metrics_enabled: bool,
}

impl<T: ObjectStore> MetricsObjectStore<T> {
    pub fn new(inner: T) -> Self {
        Self::new_with_coalesce_options(
            inner,
            OBJECT_STORE_COALESCE_DEFAULT,
            OBJECT_STORE_COALESCE_PARALLEL_DEFAULT,
        )
    }

    pub fn new_with_coalesce_gap(inner: T, coalesce_gap_bytes: u64) -> Self {
        Self::new_with_coalesce_options(
            inner,
            coalesce_gap_bytes,
            OBJECT_STORE_COALESCE_PARALLEL_DEFAULT,
        )
    }

    pub fn new_coalescing(inner: T, coalesce_gap_bytes: u64) -> Self {
        Self::new_coalescing_with_options(
            inner,
            coalesce_gap_bytes,
            OBJECT_STORE_COALESCE_PARALLEL_DEFAULT,
        )
    }

    /// Construct an instrumented store with explicit range coalescing options.
    pub fn new_with_coalesce_options(
        inner: T,
        coalesce_gap_bytes: u64,
        coalesce_parallelism: usize,
    ) -> Self {
        Self::new_inner(
            inner,
            coalesce_gap_bytes,
            coalesce_parallelism,
            OBJECT_STORE_COALESCE_PARALLEL_MAX,
            true,
        )
    }

    /// Construct an instrumented store with explicit physical request limits.
    pub fn new_with_limits(
        inner: T,
        coalesce_gap_bytes: u64,
        coalesce_parallelism: usize,
        global_parallelism: usize,
    ) -> Self {
        assert!(
            (1..=OBJECT_STORE_COALESCE_PARALLEL_MAX).contains(&global_parallelism),
            "global parallelism must be between 1 and {OBJECT_STORE_COALESCE_PARALLEL_MAX}"
        );
        Self::new_inner(
            inner,
            coalesce_gap_bytes,
            coalesce_parallelism,
            global_parallelism,
            true,
        )
    }

    /// Construct a metrics-free store with explicit range coalescing options.
    pub fn new_coalescing_with_options(
        inner: T,
        coalesce_gap_bytes: u64,
        coalesce_parallelism: usize,
    ) -> Self {
        Self::new_inner(
            inner,
            coalesce_gap_bytes,
            coalesce_parallelism,
            OBJECT_STORE_COALESCE_PARALLEL_MAX,
            false,
        )
    }

    fn new_inner(
        inner: T,
        coalesce_gap_bytes: u64,
        coalesce_parallelism: usize,
        global_parallelism: usize,
        metrics_enabled: bool,
    ) -> Self {
        assert!(
            (1..=OBJECT_STORE_COALESCE_PARALLEL_MAX).contains(&coalesce_parallelism),
            "coalesce parallelism must be between 1 and {OBJECT_STORE_COALESCE_PARALLEL_MAX}"
        );
        let metrics = ObjectStoreMetrics::default();
        metrics
            .state
            .coalesce_gap_bytes
            .store(coalesce_gap_bytes, Ordering::Relaxed);
        metrics
            .state
            .coalesce_parallelism
            .store(coalesce_parallelism as u64, Ordering::Relaxed);
        Self {
            inner,
            metrics,
            coalesce_gap_bytes,
            coalesce_parallelism,
            request_permits: Arc::new(Semaphore::new(global_parallelism)),
            metrics_enabled,
        }
    }

    pub fn metrics(&self) -> ObjectStoreMetrics {
        self.metrics.clone()
    }

    pub fn with_exact_range_cache(
        self,
        max_bytes: usize,
        reservation: Arc<MemoryReservation>,
    ) -> Self {
        assert!(max_bytes > 0, "exact range cache capacity must be positive");
        self.metrics
            .state
            .exact_range_cache_capacity_bytes
            .store(max_bytes as u64, Ordering::Relaxed);
        *self
            .metrics
            .state
            .exact_range_cache
            .lock()
            .expect("exact range cache configuration mutex poisoned") =
            Some(Arc::new(ExactRangeCache::new(max_bytes, reservation)));
        self
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

        let request_permit = Arc::clone(&self.request_permits)
            .acquire_owned()
            .await
            .expect("request semaphore must not be closed");
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

        let payload =
            Self::with_request_permit(result.payload, Some(tracker), request_permit);
        Ok(GetResult { payload, ..result })
    }

    fn with_request_permit(
        payload: GetResultPayload,
        tracker: Option<RequestTracker>,
        request_permit: OwnedSemaphorePermit,
    ) -> GetResultPayload {
        match payload {
            payload @ GetResultPayload::File(_, _) => {
                if let Some(mut tracker) = tracker {
                    tracker.finish();
                }
                payload
            }
            GetResultPayload::Stream(stream) => GetResultPayload::Stream(
                MetricsStream {
                    inner: stream,
                    tracker,
                    request_permit: Some(request_permit),
                }
                .boxed(),
            ),
        }
    }

    async fn get_coalesced_range(
        &self,
        location: &Path,
        range: Range<u64>,
    ) -> Result<Bytes> {
        let Some(cache) = self.metrics.exact_range_cache() else {
            return self.fetch_physical_range(location, range).await;
        };
        let cache_key = ExactRangeCacheKey {
            path: location.to_string(),
            start: range.start,
            end: range.end,
        };
        let requested_bytes =
            usize::try_from(range.end.saturating_sub(range.start)).unwrap_or(usize::MAX);
        let entry = match cache.access(cache_key.clone(), requested_bytes) {
            ExactRangeCacheAccess::Ready(bytes) => {
                let saved_bytes = range.end.saturating_sub(range.start);
                self.metrics
                    .state
                    .exact_range_cache_ready_hits
                    .fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .state
                    .exact_range_cache_saved_gets
                    .fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .state
                    .exact_range_cache_saved_bytes
                    .fetch_add(saved_bytes, Ordering::Relaxed);
                return Ok(bytes);
            }
            ExactRangeCacheAccess::Inflight(entry) => {
                let saved_bytes = range.end.saturating_sub(range.start);
                self.metrics
                    .state
                    .exact_range_cache_inflight_joins
                    .fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .state
                    .exact_range_cache_saved_gets
                    .fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .state
                    .exact_range_cache_saved_bytes
                    .fetch_add(saved_bytes, Ordering::Relaxed);
                entry
            }
            ExactRangeCacheAccess::Failed(entry, error) => {
                self.remove_failed_exact_range(&cache, &cache_key, &entry);
                return Err(Self::shared_cache_error(error));
            }
            ExactRangeCacheAccess::Miss {
                entry,
                retained_bytes,
            } => {
                self.metrics
                    .state
                    .exact_range_cache_misses
                    .fetch_add(1, Ordering::Relaxed);
                self.record_exact_range_retained_bytes(retained_bytes);
                entry
            }
            ExactRangeCacheAccess::AdmissionDenied => {
                self.metrics
                    .state
                    .exact_range_cache_misses
                    .fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .state
                    .exact_range_cache_admission_denied
                    .fetch_add(1, Ordering::Relaxed);
                return self.fetch_physical_range(location, range).await;
            }
        };

        let value = entry
            .value
            .get_or_init(|| async {
                self.fetch_physical_range(location, range)
                    .await
                    .map_err(Arc::new)
            })
            .await;
        match value {
            Ok(bytes) => Ok(bytes.clone()),
            Err(error) => {
                self.remove_failed_exact_range(&cache, &cache_key, &entry);
                Err(Self::shared_cache_error(Arc::clone(error)))
            }
        }
    }

    async fn fetch_physical_range(
        &self,
        location: &Path,
        range: Range<u64>,
    ) -> Result<Bytes> {
        let options = GetOptions::new().with_range(Some(range));
        if self.metrics_enabled {
            self.instrumented_get_opts(location, options, false)
                .await?
                .bytes()
                .await
        } else {
            self.get_opts(location, options).await?.bytes().await
        }
    }

    fn record_exact_range_retained_bytes(&self, retained_bytes: usize) {
        self.metrics
            .state
            .exact_range_cache_retained_bytes
            .store(retained_bytes as u64, Ordering::Relaxed);
        self.metrics
            .state
            .exact_range_cache_retained_bytes_high_watermark
            .fetch_max(retained_bytes as u64, Ordering::Relaxed);
    }

    fn remove_failed_exact_range(
        &self,
        cache: &ExactRangeCache,
        cache_key: &ExactRangeCacheKey,
        entry: &Arc<ExactRangeCacheEntry>,
    ) {
        if let Some(retained_bytes) = cache.remove_failed(cache_key, entry) {
            self.metrics
                .state
                .exact_range_cache_retained_bytes
                .store(retained_bytes as u64, Ordering::Relaxed);
        }
    }

    fn shared_cache_error(error: Arc<object_store::Error>) -> object_store::Error {
        object_store::Error::Generic {
            store: "exact range cache",
            source: Box::new(error),
        }
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
            let request_permit = Arc::clone(&self.request_permits)
                .acquire_owned()
                .await
                .expect("request semaphore must not be closed");
            let result = self.inner.get_opts(location, options).await?;
            let payload = Self::with_request_permit(result.payload, None, request_permit);
            Ok(GetResult { payload, ..result })
        }
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> Result<Vec<Bytes>> {
        let fetch_ranges = merge_ranges(ranges, self.coalesce_gap_bytes);
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
            self.metrics
                .state
                .get_ranges_max_logical_ranges
                .fetch_max(ranges.len() as u64, Ordering::Relaxed);
            self.metrics
                .state
                .get_ranges_max_coalesced_ranges
                .fetch_max(fetch_ranges.len() as u64, Ordering::Relaxed);
            if fetch_ranges.len() > self.coalesce_parallelism {
                self.metrics
                    .state
                    .get_ranges_parallelism_saturated_calls
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        coalesce_ranges_with_parallelism(
            ranges,
            &fetch_ranges,
            |range| self.get_coalesced_range(location, range),
            self.coalesce_parallelism,
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

async fn coalesce_ranges_with_parallelism<F, E, Fut>(
    ranges: &[Range<u64>],
    fetch_ranges: &[Range<u64>],
    fetch: F,
    parallelism: usize,
) -> std::result::Result<Vec<Bytes>, E>
where
    F: Send + FnMut(Range<u64>) -> Fut,
    E: Send,
    Fut: Future<Output = std::result::Result<Bytes, E>> + Send,
{
    let fetched: Vec<_> = futures::stream::iter(fetch_ranges.iter().cloned())
        .map(fetch)
        .buffered(parallelism)
        .try_collect()
        .await?;

    Ok(ranges
        .iter()
        .map(|range| {
            let idx = fetch_ranges
                .partition_point(|candidate| candidate.start <= range.start)
                - 1;
            let fetch_range = &fetch_ranges[idx];
            let fetch_bytes = &fetched[idx];
            let start = range.start - fetch_range.start;
            let end = range.end - fetch_range.start;
            fetch_bytes.slice((start as usize)..(end as usize).min(fetch_bytes.len()))
        })
        .collect())
}

fn merge_ranges(ranges: &[Range<u64>], coalesce_gap_bytes: u64) -> Vec<Range<u64>> {
    if ranges.is_empty() {
        return vec![];
    }

    let mut ranges = ranges.to_vec();
    ranges.sort_unstable_by_key(|range| range.start);
    let mut merged = Vec::with_capacity(ranges.len());
    let mut start_idx = 0;
    let mut end_idx = 1;

    while start_idx != ranges.len() {
        let mut range_end = ranges[start_idx].end;
        while end_idx != ranges.len()
            && ranges[end_idx]
                .start
                .checked_sub(range_end)
                .map(|gap| gap <= coalesce_gap_bytes)
                .unwrap_or(true)
        {
            range_end = range_end.max(ranges[end_idx].end);
            end_idx += 1;
        }

        merged.push(ranges[start_idx].start..range_end);
        start_idx = end_idx;
        end_idx += 1;
    }
    merged
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
    request_permit: Option<OwnedSemaphorePermit>,
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
                drop(this.request_permit.take());
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

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::execution::memory_pool::{
        MemoryConsumer, MemoryPool, MemoryReservation, UnboundedMemoryPool,
    };
    use object_store::ObjectStoreExt;
    use object_store::memory::InMemory;
    use tokio::time::timeout;

    fn cache_reservation() -> Arc<MemoryReservation> {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        Arc::new(MemoryConsumer::new("exact-range-cache-test").register(&pool))
    }

    #[derive(Debug)]
    struct GatedRangeStore {
        inner: InMemory,
        started: Arc<Semaphore>,
        release: Arc<Semaphore>,
    }

    #[derive(Debug)]
    struct FailOnceRangeStore {
        inner: InMemory,
        fail_next_range: AtomicU64,
    }

    impl fmt::Display for GatedRangeStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("GatedRangeStore")
        }
    }

    impl fmt::Display for FailOnceRangeStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("FailOnceRangeStore")
        }
    }

    #[async_trait]
    impl ObjectStore for GatedRangeStore {
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

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> Result<GetResult> {
            if matches!(options.range.as_ref(), Some(GetRange::Bounded(_))) {
                self.started.add_permits(1);
                self.release
                    .acquire()
                    .await
                    .expect("test release semaphore must remain open")
                    .forget();
            }
            self.inner.get_opts(location, options).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, Result<Path>>,
        ) -> BoxStream<'static, Result<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
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

    #[async_trait]
    impl ObjectStore for FailOnceRangeStore {
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

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> Result<GetResult> {
            if matches!(options.range.as_ref(), Some(GetRange::Bounded(_)))
                && self.fail_next_range.swap(0, Ordering::Relaxed) == 1
            {
                return Err(object_store::Error::Generic {
                    store: "fail-once test store",
                    source: Box::new(std::io::Error::other("injected range failure")),
                });
            }
            self.inner.get_opts(location, options).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, Result<Path>>,
        ) -> BoxStream<'static, Result<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
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

    #[tokio::test]
    async fn exact_range_cache_reuses_completed_physical_range() {
        let path = Path::from("table.parquet");
        let inner = InMemory::new();
        inner
            .put(&path, PutPayload::from_static(b"0123456789"))
            .await
            .unwrap();
        let store = MetricsObjectStore::new_with_coalesce_options(inner, 0, 1)
            .with_exact_range_cache(128, cache_reservation());
        let metrics = store.metrics();

        let first = store.get_ranges(&path, &[2..6]).await.unwrap();
        let second = store.get_ranges(&path, &[2..6]).await.unwrap();

        assert_eq!(first, vec![Bytes::from_static(b"2345")]);
        assert_eq!(second, first);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.range_get_requests, 1);
        assert_eq!(snapshot.exact_range_cache_misses, 1);
        assert_eq!(snapshot.exact_range_cache_ready_hits, 1);
        assert_eq!(snapshot.exact_range_cache_inflight_joins, 0);
        assert_eq!(snapshot.exact_range_cache_saved_gets, 1);
        assert_eq!(snapshot.exact_range_cache_saved_bytes, 4);
        assert_eq!(snapshot.exact_range_cache_retained_bytes, 4);
    }

    #[tokio::test]
    async fn exact_range_cache_reset_releases_memory_and_starts_new_query() {
        let path = Path::from("table.parquet");
        let inner = InMemory::new();
        inner
            .put(&path, PutPayload::from_static(b"0123456789"))
            .await
            .unwrap();
        let reservation = cache_reservation();
        let store = MetricsObjectStore::new_with_coalesce_options(inner, 0, 1)
            .with_exact_range_cache(128, Arc::clone(&reservation));
        let metrics = store.metrics();

        store.get_ranges(&path, &[2..6]).await.unwrap();
        store.get_ranges(&path, &[2..6]).await.unwrap();
        assert_eq!(reservation.size(), 4);

        metrics.reset();
        assert_eq!(reservation.size(), 0);
        store.get_ranges(&path, &[2..6]).await.unwrap();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.range_get_requests, 1);
        assert_eq!(snapshot.exact_range_cache_misses, 1);
        assert_eq!(snapshot.exact_range_cache_ready_hits, 0);
        assert_eq!(snapshot.exact_range_cache_saved_gets, 0);
        assert_eq!(snapshot.exact_range_cache_retained_bytes, 4);
        assert_eq!(reservation.size(), 4);
    }

    #[tokio::test]
    async fn exact_range_cache_budget_denial_preserves_read_behavior() {
        let path = Path::from("table.parquet");
        let inner = InMemory::new();
        inner
            .put(&path, PutPayload::from_static(b"0123456789"))
            .await
            .unwrap();
        let reservation = cache_reservation();
        let store = MetricsObjectStore::new_with_coalesce_options(inner, 0, 1)
            .with_exact_range_cache(3, Arc::clone(&reservation));
        let metrics = store.metrics();

        let first = store.get_ranges(&path, &[2..6]).await.unwrap();
        let second = store.get_ranges(&path, &[2..6]).await.unwrap();

        assert_eq!(first, vec![Bytes::from_static(b"2345")]);
        assert_eq!(second, first);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.range_get_requests, 2);
        assert_eq!(snapshot.exact_range_cache_misses, 2);
        assert_eq!(snapshot.exact_range_cache_admission_denied, 2);
        assert_eq!(snapshot.exact_range_cache_ready_hits, 0);
        assert_eq!(snapshot.exact_range_cache_saved_gets, 0);
        assert_eq!(snapshot.exact_range_cache_retained_bytes, 0);
        assert_eq!(reservation.size(), 0);
    }

    #[tokio::test]
    async fn exact_range_cache_joins_inflight_physical_range() {
        let path = Path::from("table.parquet");
        let inner = InMemory::new();
        inner
            .put(&path, PutPayload::from_static(b"0123456789"))
            .await
            .unwrap();
        let started = Arc::new(Semaphore::new(0));
        let release = Arc::new(Semaphore::new(0));
        let store = Arc::new(
            MetricsObjectStore::new_with_coalesce_options(
                GatedRangeStore {
                    inner,
                    started: Arc::clone(&started),
                    release: Arc::clone(&release),
                },
                0,
                1,
            )
            .with_exact_range_cache(128, cache_reservation()),
        );
        let metrics = store.metrics();

        let first_store = Arc::clone(&store);
        let first_path = path.clone();
        let first =
            tokio::spawn(
                async move { first_store.get_ranges(&first_path, &[2..6]).await },
            );
        started
            .acquire()
            .await
            .expect("first physical request must start")
            .forget();

        let second_store = Arc::clone(&store);
        let second_path = path.clone();
        let second =
            tokio::spawn(
                async move { second_store.get_ranges(&second_path, &[2..6]).await },
            );
        let second_physical_started =
            timeout(Duration::from_millis(100), started.acquire())
                .await
                .is_ok();

        release.add_permits(if second_physical_started { 2 } else { 1 });
        let first = first.await.unwrap().unwrap();
        let second = second.await.unwrap().unwrap();

        assert!(!second_physical_started, "duplicate physical GET started");
        assert_eq!(first, vec![Bytes::from_static(b"2345")]);
        assert_eq!(second, first);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.range_get_requests, 1);
        assert_eq!(snapshot.exact_range_cache_misses, 1);
        assert_eq!(snapshot.exact_range_cache_ready_hits, 0);
        assert_eq!(snapshot.exact_range_cache_inflight_joins, 1);
        assert_eq!(snapshot.exact_range_cache_saved_gets, 1);
        assert_eq!(snapshot.exact_range_cache_saved_bytes, 4);
        assert_eq!(snapshot.exact_range_cache_retained_bytes, 4);
    }

    #[tokio::test]
    async fn exact_range_cache_failed_fill_releases_budget_and_allows_retry() {
        let path = Path::from("table.parquet");
        let inner = InMemory::new();
        inner
            .put(&path, PutPayload::from_static(b"0123456789"))
            .await
            .unwrap();
        let reservation = cache_reservation();
        let store = MetricsObjectStore::new_with_coalesce_options(
            FailOnceRangeStore {
                inner,
                fail_next_range: AtomicU64::new(1),
            },
            0,
            1,
        )
        .with_exact_range_cache(128, Arc::clone(&reservation));
        let metrics = store.metrics();

        let error = store.get_ranges(&path, &[2..6]).await.unwrap_err();
        assert!(error.to_string().contains("injected range failure"));
        assert_eq!(reservation.size(), 0);
        assert_eq!(metrics.snapshot().exact_range_cache_retained_bytes, 0);

        let retry = store.get_ranges(&path, &[2..6]).await.unwrap();
        let cached = store.get_ranges(&path, &[2..6]).await.unwrap();

        assert_eq!(retry, vec![Bytes::from_static(b"2345")]);
        assert_eq!(cached, retry);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.range_get_requests, 2);
        assert_eq!(snapshot.errors, 1);
        assert_eq!(snapshot.exact_range_cache_misses, 2);
        assert_eq!(snapshot.exact_range_cache_ready_hits, 1);
        assert_eq!(snapshot.exact_range_cache_saved_gets, 1);
        assert_eq!(snapshot.exact_range_cache_retained_bytes, 4);
        assert_eq!(reservation.size(), 4);
    }

    #[tokio::test]
    async fn exact_range_cache_key_includes_path_and_complete_range() {
        let first_path = Path::from("first.parquet");
        let second_path = Path::from("second.parquet");
        let inner = InMemory::new();
        inner
            .put(&first_path, PutPayload::from_static(b"0123456789"))
            .await
            .unwrap();
        inner
            .put(&second_path, PutPayload::from_static(b"abcdefghij"))
            .await
            .unwrap();
        let store = MetricsObjectStore::new_with_coalesce_options(inner, 0, 1)
            .with_exact_range_cache(128, cache_reservation());
        let metrics = store.metrics();

        let first = store.get_ranges(&first_path, &[2..6]).await.unwrap();
        let other_path = store.get_ranges(&second_path, &[2..6]).await.unwrap();
        let other_range = store.get_ranges(&first_path, &[3..6]).await.unwrap();
        let first_again = store.get_ranges(&first_path, &[2..6]).await.unwrap();

        assert_eq!(first, vec![Bytes::from_static(b"2345")]);
        assert_eq!(other_path, vec![Bytes::from_static(b"cdef")]);
        assert_eq!(other_range, vec![Bytes::from_static(b"345")]);
        assert_eq!(first_again, first);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.range_get_requests, 3);
        assert_eq!(snapshot.exact_range_cache_misses, 3);
        assert_eq!(snapshot.exact_range_cache_ready_hits, 1);
        assert_eq!(snapshot.exact_range_cache_retained_bytes, 11);
    }
}
