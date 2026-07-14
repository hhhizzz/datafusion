# S3 Query-local Quantum Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and measure a benchmark-only, query-local adaptive S3 quantum cache that preserves Parquet semantics while converting repeated small ranges into bounded 512 KiB, 1 MiB, or 4 MiB reads.

**Architecture:** `dfbench` registers an instrumented physical S3 store with one global 24-request semaphore, then optionally wraps it in `QuantumCacheObjectStore`. The cache records decoder-level logical requests, uses aligned single-flight blocks, accounts resident and in-flight bytes in the DataFusion memory pool, and is cleared before every query iteration. Physical wire metrics remain inside the cache.

**Tech Stack:** Rust, DataFusion benchmark crate, `object_store` 0.13, Tokio `Semaphore` and `OnceCell`, `bytes::Bytes`, DataFusion `MemoryPool`, shell benchmark wrappers.

## Global Constraints

- Run remote builds and benchmarks on `sz-data-b-1` with CPU affinity `0-23`.
- Apply `ulimit -Sv 50331648` to every compile, link, and benchmark process.
- Require at least 64 GiB `MemAvailable` and remote disk usage below 80% before starting.
- Refuse to start while competing Cargo, rustc, clippy, or dfbench processes exist.
- Use at most 16 Cargo jobs and exactly one global physical S3 semaphore with 24 permits.
- Keep the existing 256 MiB lookahead budget and at most 512 MiB resident plus in-flight cache bytes.
- Never retain cache data across benchmark iterations, queries, or rounds.
- Use `apply_patch` for manual edits and do not alter unrelated user changes.

---

### Task 1: Bound Physical S3 Concurrency

**Files:**
- Modify: `benchmarks/src/util/metrics_object_store.rs`
- Modify: `benchmarks/tests/object_store_metrics.rs`

**Interfaces:**
- Consumes: `OBJECT_STORE_COALESCE_PARALLEL_MAX == 24`.
- Produces: `MetricsObjectStore::new_with_limits(inner, gap, per_call, global)` and a permit retained until the response body is consumed.

- [ ] **Step 1: Write the failing global-concurrency test**

Add a test that starts two concurrent `get_ranges` calls, each containing four disjoint ranges, against a 50 ms `ThrottledStore`:

```rust
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
        async move { right.get_ranges(&right_path, &[8..9, 10..11, 12..13, 14..15]).await },
    );
    left_result.unwrap();
    right_result.unwrap();
    assert_eq!(store.metrics().snapshot().peak_in_flight, 3);
}
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
rtk cargo test -p datafusion-benchmarks --test object_store_metrics global_limit_bounds_concurrent_get_ranges_calls -- --nocapture
```

Expected: compile failure because `new_with_limits` does not exist.

- [ ] **Step 3: Add a body-lifetime request permit**

Add `request_permits: Arc<Semaphore>` to `MetricsObjectStore`. Extend `MetricsStream` so the permit is held through EOF or stream drop:

```rust
struct MetricsStream {
    inner: BoxStream<'static, Result<Bytes>>,
    tracker: Option<RequestTracker>,
    _request_permit: OwnedSemaphorePermit,
}

pub fn new_with_limits(
    inner: T,
    coalesce_gap_bytes: u64,
    coalesce_parallelism: usize,
    global_parallelism: usize,
) -> Self {
    assert!((1..=OBJECT_STORE_COALESCE_PARALLEL_MAX).contains(&global_parallelism));
    Self::new_inner(
        inner,
        coalesce_gap_bytes,
        coalesce_parallelism,
        global_parallelism,
        true,
    )
}
```

Acquire `Arc::clone(&self.request_permits).acquire_owned().await` before calling
`request_started` or `inner.get_opts`, so queued callers are not counted as
physical in-flight requests. Move the permit into `MetricsStream` for streaming
payloads; let it drop after a file payload or failed request. The metrics-free
path must use the same permit-carrying stream wrapper. Make existing
constructors pass `OBJECT_STORE_COALESCE_PARALLEL_MAX`.

- [ ] **Step 4: Run all object-store metric tests**

Run:

```bash
rtk cargo test -p datafusion-benchmarks --test object_store_metrics -- --nocapture
```

Expected: all tests pass, including exact `peak_in_flight == 3`.

- [ ] **Step 5: Commit**

```bash
rtk git add benchmarks/src/util/metrics_object_store.rs benchmarks/tests/object_store_metrics.rs
rtk git commit -m "bench: bound global S3 request concurrency"
```

### Task 2: Define Cache Configuration, Density, and Metrics

**Files:**
- Create: `benchmarks/src/util/quantum_cache_object_store.rs`
- Modify: `benchmarks/src/util/mod.rs`
- Create: `benchmarks/tests/quantum_cache_object_store.rs`

**Interfaces:**
- Produces: `QuantumCacheMode`, `QuantumCacheConfig`, `QuantumCacheMetrics`, `QuantumCacheMetricsSnapshot`, and `parse_quantum_cache_config`.
- Required values: modes `off|forced|adaptive`; blocks `524288|1048576|4194304`; cap default `536870912`; enable density `0.80`; disable density `0.50`; minimum observed payload `1048576`; disable epoch `32` blocks.

- [ ] **Step 1: Write parser and density RED tests**

```rust
#[test]
fn parses_only_supported_quantum_cache_values() {
    let config = parse_quantum_cache_config(
        Some("adaptive"),
        Some("1048576"),
        Some("536870912"),
    )
    .unwrap()
    .unwrap();
    assert_eq!(config.mode, QuantumCacheMode::Adaptive);
    assert_eq!(config.block_bytes, 1_048_576);
    assert_eq!(config.capacity_bytes, 536_870_912);

    for block in ["0", "131072", "8388608", "invalid"] {
        assert!(parse_quantum_cache_config(Some("forced"), Some(block), None).is_err());
    }
}

#[test]
fn adaptive_gate_enables_after_dense_unique_coverage() {
    let mut gate = DensityGate::new(1_048_576);
    gate.observe_exact(0..524_288);
    gate.observe_exact(524_288..1_048_576);
    assert!(gate.admit_blocks());
    assert_eq!(gate.unique_requested_bytes(), 1_048_576);
    assert_eq!(gate.touched_block_bytes(), 1_048_576);
}
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
rtk cargo test -p datafusion-benchmarks --test quantum_cache_object_store -- --nocapture
```

Expected: compile failure because the module and types do not exist.

- [ ] **Step 3: Implement pure configuration and density tracking**

Use merged block-relative intervals so duplicate requests do not inflate density:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantumCacheMode {
    Forced,
    Adaptive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantumCacheConfig {
    pub mode: QuantumCacheMode,
    pub block_bytes: u64,
    pub capacity_bytes: usize,
}

#[derive(Debug, Default)]
struct DensityGate {
    block_bytes: u64,
    coverage: HashMap<u64, Vec<Range<u64>>>,
    unique_requested_bytes: u64,
    admitted: bool,
    low_density_epochs: u8,
}

impl DensityGate {
    fn admit_blocks(&self) -> bool {
        self.admitted
    }

    fn density(&self) -> f64 {
        let touched = self.coverage.len() as u64 * self.block_bytes;
        if touched == 0 { 0.0 } else { self.unique_requested_bytes as f64 / touched as f64 }
    }
}
```

`observe_exact` must split each request by aligned block, merge intervals, add only newly covered bytes, and set `admitted` when unique bytes are at least 1 MiB and density is at least 0.80. Add a 32-block useful-byte epoch method that disables new admissions after two epochs below 0.50.

`parse_quantum_cache_config(Some("off"), ..)` returns `Ok(None)`; `off` is
therefore represented by not registering the wrapper rather than by a third
runtime enum branch.

- [ ] **Step 4: Add serializable cache metrics**

Expose atomics for logical requests/bytes, hit bytes, partial-hit bytes, miss bytes, block fetches/bytes, useful block bytes, overread bytes, single-flight joins, adaptive enables/disables, current reserved bytes, and peak reserved bytes. `reset_counters` must not change live reservation accounting.

- [ ] **Step 5: Run unit tests and commit**

```bash
rtk cargo test -p datafusion-benchmarks --test quantum_cache_object_store -- --nocapture
rtk git add benchmarks/src/util/mod.rs benchmarks/src/util/quantum_cache_object_store.rs benchmarks/tests/quantum_cache_object_store.rs
rtk git commit -m "bench: define adaptive S3 quantum cache policy"
```

Expected: parser, interval de-duplication, enable, disable, and metrics reset tests pass.

### Task 3: Implement Bounded Single-flight Blocks

**Files:**
- Modify: `benchmarks/src/util/quantum_cache_object_store.rs`
- Modify: `benchmarks/tests/quantum_cache_object_store.rs`

**Interfaces:**
- Consumes: `QuantumCacheConfig`, `QuantumCacheMetrics`, `Arc<dyn ObjectStore>`, `Arc<MemoryReservation>`.
- Produces: `QuantumCacheObjectStore::new`, `handle`, `clear_iteration`, and the full `ObjectStore` implementation.

- [ ] **Step 1: Write range, overlap, single-flight, and failure RED tests**

Cover these exact assertions:

```rust
#[tokio::test]
async fn forced_cache_slices_unaligned_cross_block_ranges_in_order() {
    let fixture = CacheFixture::new(QuantumCacheMode::Forced, 4, 16).await;
    let actual = fixture.store.get_ranges(&fixture.path, &[2..6, 0..2, 5..8]).await.unwrap();
    assert_eq!(actual, vec![Bytes::from_static(b"cdef"), Bytes::from_static(b"ab"), Bytes::from_static(b"fgh")]);
    let snapshot = fixture.handle.snapshot();
    assert_eq!(snapshot.logical_range_bytes, 8);
    assert_eq!(snapshot.block_fetches, 2);
    assert_eq!(snapshot.block_fetch_bytes, 8);
}

#[tokio::test]
async fn concurrent_misses_join_one_block_load() {
    let fixture = CacheFixture::throttled(QuantumCacheMode::Forced, 8, 32).await;
    let (left, right) = tokio::join!(
        fixture.store.get_range(&fixture.path, 1..3),
        fixture.store.get_range(&fixture.path, 4..7),
    );
    left.unwrap();
    right.unwrap();
    assert_eq!(fixture.wire_metrics.snapshot().range_get_requests, 1);
    assert_eq!(fixture.handle.snapshot().single_flight_joins, 1);
}
```

Add a `FailOnceStore` test proving the failed entry is removed and a second request retries successfully.

- [ ] **Step 2: Run RED tests**

```bash
rtk cargo test -p datafusion-benchmarks --test quantum_cache_object_store -- --nocapture
```

Expected: compile failure because `QuantumCacheObjectStore` is not implemented.

- [ ] **Step 3: Implement cache state without holding locks across await**

Use these core types:

```rust
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct BlockKey {
    path: Path,
    start: u64,
}

struct CacheState {
    entries: HashMap<BlockKey, Arc<OnceCell<Bytes>>>,
    lru: VecDeque<BlockKey>,
    gates: HashMap<Path, DensityGate>,
}

pub struct QuantumCacheObjectStore {
    inner: Arc<dyn ObjectStore>,
    config: QuantumCacheConfig,
    state: Arc<tokio::sync::Mutex<CacheState>>,
    byte_permits: Arc<Semaphore>,
    reservation: Arc<MemoryReservation>,
    metrics: QuantumCacheMetrics,
}
```

Under the mutex, find or insert `Arc<OnceCell<Bytes>>`, mark a join when the cell already exists, then release the mutex before `get_or_try_init`. Before loading, acquire `block_len` byte permits and call `reservation.try_grow(block_len)`. On error, release both resources and remove the same cell using `Arc::ptr_eq`.

- [ ] **Step 4: Keep reservations alive through returned `Bytes`**

Use a `Bytes::from_owner` owner so eviction cannot release memory accounting while a caller still owns a slice:

```rust
struct ReservedBlock {
    bytes: Bytes,
    reservation: Arc<MemoryReservation>,
    reserved_bytes: usize,
    _permits: OwnedSemaphorePermit,
}

impl AsRef<[u8]> for ReservedBlock {
    fn as_ref(&self) -> &[u8] { self.bytes.as_ref() }
}

impl Drop for ReservedBlock {
    fn drop(&mut self) { self.reservation.shrink(self.reserved_bytes); }
}
```

Store `Bytes::from_owner(ReservedBlock { ... })` in the cell. Return `Bytes::slice` for exact caller ranges. LRU removal drops only the cache's clone; active returned slices retain the owner and reservation.

- [ ] **Step 5: Implement exact, forced, and adaptive request paths**

`off` is represented by not installing the wrapper. In adaptive exact mode, call the inner store with the original ranges and update `DensityGate`. In forced/admitted mode, compute aligned block keys, fetch unique blocks concurrently through the shared physical store, and reconstruct results in original order. Never cache full GETs, HEADs, writes, list, copy, or rename operations.

- [ ] **Step 6: Implement iteration clearing**

`QuantumCacheHandle::clear_iteration()` clears entries, LRU, and density state, then resets counters. It returns an error if called while cache-owned in-flight loads remain; the TPC-DS runner only calls it after a completed iteration and before the next one.

- [ ] **Step 7: Run all cache tests and commit**

```bash
rtk cargo test -p datafusion-benchmarks --test quantum_cache_object_store -- --nocapture
rtk git add benchmarks/src/util/quantum_cache_object_store.rs benchmarks/tests/quantum_cache_object_store.rs
rtk git commit -m "bench: add bounded query-local S3 quantum cache"
```

Expected: ordered slicing, short final block, duplicate ranges, concurrent single-flight, retry after error, cancellation, capacity denial, eviction ownership, and adaptive fallback all pass.

### Task 4: Wire Per-iteration Cache Lifecycle into TPC-DS

**Files:**
- Modify: `benchmarks/src/tpcds/run.rs`
- Modify: `benchmarks/src/util/quantum_cache_object_store.rs`
- Modify: `benchmarks/tests/quantum_cache_object_store.rs`

**Interfaces:**
- Produces environment controls `TPCDS_S3_QUANTUM_CACHE_MODE`, `TPCDS_S3_QUANTUM_CACHE_BLOCK_BYTES`, and `TPCDS_S3_QUANTUM_CACHE_CAPACITY_BYTES`.
- Produces one `RegisteredS3Diagnostics` handle with `start_iteration` and `finish_iteration`.

- [ ] **Step 1: Write TPC-DS environment parser tests**

Add pure parser tests in `tpcds/run.rs` for absent configuration, each supported block, and invalid mode/cap. Invalid partial configuration must return `DataFusionError::Configuration` rather than silently disabling the cache.

- [ ] **Step 2: Replace the metrics-only registration result**

Use this shape:

```rust
struct RegisteredS3Diagnostics {
    wire_metrics: Option<ObjectStoreMetrics>,
    quantum_cache: Option<QuantumCacheHandle>,
}

impl RegisteredS3Diagnostics {
    fn start_iteration(&self) -> Result<()> {
        if let Some(cache) = &self.quantum_cache { cache.clear_iteration()?; }
        if let Some(metrics) = &self.wire_metrics { metrics.reset(); }
        Ok(())
    }

    fn finish_iteration(&self, query: usize, iteration: usize) -> Result<()> {
        if let Some(metrics) = &self.wire_metrics {
            println!("TPCDS_OBJECT_STORE_METRICS query={query} iteration={iteration} {}", serde_json::to_string(&metrics.snapshot())?);
        }
        if let Some(cache) = &self.quantum_cache {
            println!("TPCDS_QUANTUM_CACHE_METRICS query={query} iteration={iteration} {}", serde_json::to_string(&cache.snapshot())?);
        }
        Ok(())
    }
}
```

Create a `MemoryConsumer::new("TPCDS S3 quantum cache")` reservation from `ctx.runtime_env().memory_pool`. Register `QuantumCacheObjectStore` outside `MetricsObjectStore<AmazonS3>` so wire metrics see only physical block requests.

- [ ] **Step 3: Reset before every timed iteration**

Call `diagnostics.start_iteration()?` immediately before `let start = Instant::now()`. Call `finish_iteration` immediately after query execution and before printing elapsed time. Table registration and metadata reads remain outside timed iteration metrics.

- [ ] **Step 4: Test, Clippy, and commit**

```bash
rtk cargo test -p datafusion-benchmarks tpcds::run::tests -- --nocapture
rtk cargo test -p datafusion-benchmarks --test quantum_cache_object_store -- --nocapture
rtk cargo clippy -p datafusion-benchmarks --all-targets -- -D warnings
rtk git add benchmarks/src/tpcds/run.rs benchmarks/src/util/quantum_cache_object_store.rs benchmarks/tests/quantum_cache_object_store.rs
rtk git commit -m "bench: expose per-iteration S3 quantum cache controls"
```

Expected: tests and Clippy pass with no warning; logs contain separate logical-cache and physical-wire JSON.

### Task 5: Add Guarded Remote Sweep Support

**Files:**
- Modify: `/Users/qiwei.huang/Source/datafusion-workspace/codex/scripts/run-df54-interleaved-ranges-remote-bench.sh`
- Create: `/Users/qiwei.huang/Source/datafusion-workspace/codex/scripts/run-df54-quantum-cache-sweep.sh`
- Modify: `/Users/qiwei.huang/Source/datafusion-workspace/codex/scripts/summarize-interleaved-range-sweep.py`
- Create: `/Users/qiwei.huang/Source/datafusion-workspace/codex/agents/2026-07-14-s3-quantum-cache-runbook.md`

**Interfaces:**
- Consumes the three environment controls from Task 4.
- Produces manifests with source SHA, cache mode/block/cap, resource guards, wire/cache metrics, and query timing.

- [ ] **Step 1: Add wrapper arguments**

Add `--quantum-cache-mode`, `--quantum-cache-block-bytes`, and `--quantum-cache-capacity-bytes`. Validate modes and block values locally, export only supplied values, and record all three in the manifest.

- [ ] **Step 2: Preserve all resource guards**

Before build or run, the remote script must check `MemAvailable >= 67108864 kB`, disk `<80%`, no competing process, set `CARGO_BUILD_JOBS=16`, run under `taskset -c 0-23`, and set `ulimit -Sv 50331648` in the same shell as Cargo or dfbench.

- [ ] **Step 3: Add the exact screening matrix**

The sweep script runs baseline and forced blocks `524288`, `1048576`, `4194304` for q72 one round; then alternates baseline and the best forced candidate for three rounds. It does not choose the winner solely from old logs.

- [ ] **Step 4: Parse cache metrics**

Extend the summarizer to report median time, speedup, physical bytes, amplification, GET count, block size, hit ratio, useful-block ratio, adaptive transitions, and peak reserved bytes.

- [ ] **Step 5: Syntax-check and commit only source-repository changes**

```bash
rtk bash -n codex/scripts/run-df54-interleaved-ranges-remote-bench.sh
rtk bash -n codex/scripts/run-df54-quantum-cache-sweep.sh
rtk python3 -m py_compile codex/scripts/summarize-interleaved-range-sweep.py
```

The workspace root is not a Git repository; retain the runbook and scripts there as Codex-owned artifacts. Commit only DataFusion branch files in prior tasks.

### Task 6: Execute Route A Screening and Regression Gates

**Files:**
- Create results under: `/Users/qiwei.huang/Source/datafusion-workspace/codex/logs/datafusion-tpcds/quantum_cache_20260714/`

**Interfaces:**
- Produces controlled q72, q21, and q82 evidence and a route decision.

- [ ] **Step 1: Run remote unit tests and release-nonlto build**

Use the existing guarded remote-step wrapper. Expected: all cache and metric tests pass, benchmark Clippy passes, release-nonlto `dfbench` builds under CPU/memory/disk guards.

- [ ] **Step 2: Run q72 forced one-round screen**

Run baseline plus 512 KiB, 1 MiB, and 4 MiB forced blocks with factor 8, 256 KiB coalesce gap, per-call parallelism 10, and global physical limit 24.

- [ ] **Step 3: Run adaptive three-round alternation**

Use the best forced block. Advance only if forced improves q72 by at least 5%. The adaptive candidate must improve median q72 by at least 10%, read no more than 1.25x physical bytes, reserve no more than 512 MiB cache and 768 MiB cache plus lookahead, and return identical results.

- [ ] **Step 4: Run q21/q82 three-round regression screen**

Each query must regress by no more than 5%. Record request shapes even when timing passes.

- [ ] **Step 5: Archive and write route decision**

Download raw logs, JSON, manifests, and summaries; verify checksums; write `README.md` separating measured timing, profile evidence, and inference. A failed gate is a valid negative result and must not be hidden or promoted to formal evidence.
