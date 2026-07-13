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
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};

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
    assert_eq!(snapshot.get_ranges_calls, 1);
    assert_eq!(snapshot.logical_ranges, 2);
    assert_eq!(snapshot.logical_range_bytes, 4);
    assert_eq!(snapshot.range_get_requests, 1);
    assert_eq!(snapshot.response_range_bytes, 6);
    assert_eq!(snapshot.body_bytes, 6);
    assert_eq!(snapshot.peak_in_flight, 1);
    assert_eq!(snapshot.paths.len(), 1);
    assert_eq!(snapshot.paths[0].path, "data.parquet");
    assert_eq!(snapshot.paths[0].range_get_requests, 1);
    assert_eq!(snapshot.paths[0].response_range_bytes, 6);
    assert_eq!(snapshot.paths[0].body_bytes, 6);
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
