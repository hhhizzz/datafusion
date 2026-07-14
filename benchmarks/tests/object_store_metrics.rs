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

use bytes::Bytes;
use datafusion_benchmarks::util::metrics_object_store::MetricsObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::throttle::{ThrottleConfig, ThrottledStore};
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn records_logical_ranges_and_coalesced_wire_gets() {
    let store = MetricsObjectStore::new(InMemory::new());
    let path = Path::from("data.parquet");
    store
        .put(&path, PutPayload::from_static(b"abcdef"))
        .await
        .unwrap();

    let metrics = store.metrics();
    metrics.reset();
    let result = store.get_ranges(&path, &[0..2, 4..6]).await.unwrap();

    assert_eq!(
        result,
        vec![Bytes::from_static(b"ab"), Bytes::from_static(b"ef")]
    );
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.coalesce_parallelism, 10);
    assert_eq!(snapshot.get_ranges_calls, 1);
    assert_eq!(snapshot.logical_ranges, 2);
    assert_eq!(snapshot.logical_range_bytes, 4);
    assert_eq!(snapshot.range_get_requests, 1);
    assert_eq!(snapshot.response_range_bytes, 6);
    assert_eq!(snapshot.body_bytes, 6);
    assert_eq!(snapshot.peak_in_flight, 1);
    assert!(snapshot.request_window_ms > 0.0);
    assert!(snapshot.body_throughput_mib_per_s > 0.0);
    assert_eq!(snapshot.paths.len(), 1);
    assert_eq!(snapshot.paths[0].path, "data.parquet");
    assert_eq!(snapshot.paths[0].range_get_requests, 1);
    assert_eq!(snapshot.paths[0].response_range_bytes, 6);
    assert_eq!(snapshot.paths[0].body_bytes, 6);
    assert!(snapshot.paths[0].request_window_ms > 0.0);
    assert!(snapshot.paths[0].body_throughput_mib_per_s > 0.0);

    metrics.reset();
    let reset = metrics.snapshot();
    assert_eq!(reset.request_window_ms, 0.0);
    assert_eq!(reset.body_throughput_mib_per_s, 0.0);
}

#[tokio::test]
async fn configurable_gap_can_disable_range_merging() {
    let store = MetricsObjectStore::new_with_coalesce_gap(InMemory::new(), 0);
    let path = Path::from("data.parquet");
    store
        .put(&path, PutPayload::from_static(b"abcdef"))
        .await
        .unwrap();

    let metrics = store.metrics();
    metrics.reset();
    let result = store.get_ranges(&path, &[0..2, 4..6]).await.unwrap();

    assert_eq!(
        result,
        vec![Bytes::from_static(b"ab"), Bytes::from_static(b"ef")]
    );
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.coalesce_gap_bytes, 0);
    assert_eq!(snapshot.logical_ranges, 2);
    assert_eq!(snapshot.logical_range_bytes, 4);
    assert_eq!(snapshot.range_get_requests, 2);
    assert_eq!(snapshot.response_range_bytes, 4);
    assert_eq!(snapshot.body_bytes, 4);
    assert_eq!(snapshot.range_overfetch_bytes, 0);
}

#[tokio::test]
async fn coalescing_only_store_does_not_collect_metrics() {
    let wire_store = MetricsObjectStore::new(InMemory::new());
    let wire_metrics = wire_store.metrics();
    let store = MetricsObjectStore::new_coalescing(wire_store, 0);
    let path = Path::from("data.parquet");
    store
        .put(&path, PutPayload::from_static(b"abcdef"))
        .await
        .unwrap();

    let outer_metrics = store.metrics();
    outer_metrics.reset();
    wire_metrics.reset();
    let result = store.get_ranges(&path, &[0..2, 4..6]).await.unwrap();

    assert_eq!(
        result,
        vec![Bytes::from_static(b"ab"), Bytes::from_static(b"ef")]
    );
    let outer_snapshot = outer_metrics.snapshot();
    assert_eq!(outer_snapshot.logical_ranges, 0);
    assert_eq!(outer_snapshot.range_get_requests, 0);

    let wire_snapshot = wire_metrics.snapshot();
    assert_eq!(wire_snapshot.range_get_requests, 2);
    assert_eq!(wire_snapshot.response_range_bytes, 4);
}

#[tokio::test]
async fn configurable_parallelism_bounds_in_flight_range_gets() {
    let inner = ThrottledStore::new(
        InMemory::new(),
        ThrottleConfig {
            wait_get_per_call: Duration::from_millis(50),
            ..Default::default()
        },
    );
    let store = MetricsObjectStore::new_with_coalesce_options(inner, 0, 4);
    let path = Path::from("data.parquet");
    store
        .put(&path, PutPayload::from_static(b"abcdefghijklmnop"))
        .await
        .unwrap();

    let metrics = store.metrics();
    metrics.reset();
    let result = store
        .get_ranges(
            &path,
            &[0..1, 2..3, 4..5, 6..7, 8..9, 10..11, 12..13, 14..15],
        )
        .await
        .unwrap();

    assert_eq!(
        result,
        b"acegikmo"
            .iter()
            .map(|value| Bytes::copy_from_slice(&[*value]))
            .collect::<Vec<_>>()
    );
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.coalesce_parallelism, 4);
    assert_eq!(snapshot.get_ranges_max_logical_ranges, 8);
    assert_eq!(snapshot.get_ranges_max_coalesced_ranges, 8);
    assert_eq!(snapshot.get_ranges_parallelism_saturated_calls, 1);
    assert_eq!(snapshot.range_get_requests, 8);
    assert_eq!(snapshot.peak_in_flight, 4);
}

#[tokio::test]
async fn global_limit_bounds_concurrent_get_ranges_calls() {
    let inner = ThrottledStore::new(
        InMemory::new(),
        ThrottleConfig {
            wait_get_per_call: Duration::from_millis(50),
            ..Default::default()
        },
    );
    let store = Arc::new(MetricsObjectStore::new_with_limits(inner, 0, 4, 3));
    let path = Path::from("data.parquet");
    store
        .put(&path, PutPayload::from_bytes(Bytes::from(vec![0_u8; 32])))
        .await
        .unwrap();
    store.metrics().reset();

    let left = Arc::clone(&store);
    let right = Arc::clone(&store);
    let left_path = path.clone();
    let right_path = path.clone();
    let (left_result, right_result) = tokio::join!(
        async move { left.get_ranges(&left_path, &[0..1, 2..3, 4..5, 6..7]).await },
        async move {
            right
                .get_ranges(&right_path, &[8..9, 10..11, 12..13, 14..15])
                .await
        },
    );
    left_result.unwrap();
    right_result.unwrap();
    assert_eq!(store.metrics().snapshot().peak_in_flight, 3);
}
