# Multi-row-group Parquet Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend DataFusion's current depth-one Parquet push-decoder lookahead into a bounded depth queue and then add density-gated prefetch of projected compressed spans across two or four upcoming row groups.

**Architecture:** B1 changes only DataFusion's existing lookahead state machine and keeps exact decoder requests. B2 adds a small non-mutating next-row-group API to the pinned arrow-rs 58.3 source, computes filter-plus-output column spans from DataFusion's prepared access plan and Parquet metadata, and stages coalesced ranges into the existing push decoder. DataFusion owns admission, physical I/O, ordering, cancellation, and memory-pool reservations; arrow-rs continues to own row filtering and decoding.

**Tech Stack:** Rust, DataFusion datasource-parquet, arrow-rs parquet push decoder, Tokio, `object_store`, DataFusion execution metrics and memory pool, guarded remote benchmark scripts.

## Global Constraints

- First execute Task 1 of the quantum-cache plan (the shared physical
  24-request limiter) on `codex/df54-interleaved-file-ranges`. Create both
  DataFusion route worktrees from that common descendant of `c82dd31` so the
  baseline and both candidates use the same limiter. Use an isolated arrow-rs
  worktree from commit `913bab26ba9bed8fc2bc1acda300cc52345b0da1`.
- Run remote builds and benchmarks on `sz-data-b-1` with CPU affinity `0-23`.
- Apply `ulimit -Sv 50331648` to every compile, link, and benchmark process.
- Require at least 64 GiB `MemAvailable`, remote disk below 80%, and no competing Cargo, rustc, clippy, or dfbench process.
- Use at most 16 Cargo jobs, 24 physical S3 requests, 24 speculative logical ranges, and 256 MiB speculative bytes.
- Preserve row order, reverse scans, selections, limits, decoder-run boundaries, deferred errors, and cancellation.
- Keep `row_group_lookahead=false` behavior unchanged and depth 1 behavior byte-for-byte compatible with the current candidate.
- Use `apply_patch` for manual edits and do not alter unrelated user changes.

---

### Task 1: Add Lookahead Depth Configuration

**Files:**
- Modify: `datafusion/common/src/config.rs`
- Modify: `datafusion/datasource-parquet/src/lookahead.rs`
- Modify: `datafusion/datasource-parquet/src/source.rs`
- Modify: `datafusion/proto-common/src/to_proto/mod.rs`
- Modify: `datafusion/proto-common/src/from_proto/mod.rs`
- Modify generated proto files using the repository generator
- Modify: `datafusion/proto/tests/cases/roundtrip_logical_plan.rs`
- Modify: `docs/source/user-guide/configs.md`

**Interfaces:**
- Produces `datafusion.execution.parquet.row_group_lookahead_depth: usize`, default 1, valid range 1 through 4 when lookahead is enabled.
- Produces `ParquetLookaheadCoordinator::new(depth: usize)`,
  `ParquetLookaheadCoordinator::validate()`, and
  `LookaheadFileContext::depth()`.

- [ ] **Step 1: Write RED config tests**

Add tests for default 1, environment value 4, and runtime rejection of 0 and 5:

```rust
#[test]
fn row_group_lookahead_depth_defaults_to_one() {
    assert_eq!(
        ConfigOptions::default().execution.parquet.row_group_lookahead_depth,
        1
    );
}

#[test]
fn coordinator_rejects_depth_outside_one_through_four() {
    assert!(ParquetLookaheadCoordinator::new(0).validate().is_err());
    assert!(ParquetLookaheadCoordinator::new(5).validate().is_err());
    let coordinator = ParquetLookaheadCoordinator::new(4);
    coordinator.validate().unwrap();
    assert_eq!(coordinator.depth(), 4);
}
```

- [ ] **Step 2: Run RED tests**

```bash
rtk cargo test -p datafusion-datasource-parquet row_group_lookahead_depth -- --nocapture
```

Expected: compile failure because the config field and coordinator APIs do not exist.

- [ ] **Step 3: Add the config and coordinator field**

```rust
/// (reading) Maximum number of decoded row-group readers held by bounded
/// speculative lookahead. Used only when `row_group_lookahead` is true.
pub row_group_lookahead_depth: usize, default = 1
```

Store `depth: usize` in `ParquetLookaheadCoordinator` and construct it in
`ParquetSource::create_execution_state` from table parquet options. Change the
context lookup used by `create_morselizer_with_context` to validate the shared
coordinator and return `Result<Option<LookaheadScanContext>>`; this is the first
fallible boundary before any file I/O. Return a configuration error for values
outside `1..=4`; do not silently clamp or disable lookahead.

- [ ] **Step 4: Propagate through protobuf and docs**

Add the field adjacent to `row_group_lookahead`, regenerate protobuf Rust/JSON, add round-trip values 1 and 4, and update the generated config table.

- [ ] **Step 5: Run focused tests and commit**

```bash
rtk cargo test -p datafusion-datasource-parquet row_group_lookahead_depth -- --nocapture
rtk cargo test -p datafusion-proto test_parquet_options_row_group_lookahead -- --nocapture
rtk git add datafusion/common/src/config.rs datafusion/datasource-parquet/src/lookahead.rs datafusion/datasource-parquet/src/source.rs datafusion/proto-common datafusion/proto docs/source/user-guide/configs.md
rtk git commit -m "feat: configure parquet row-group lookahead depth"
```

Expected: default, env, validation, and proto round-trip tests pass.

### Task 2: Replace the Depth-one Slot with an Ordered Queue

**Files:**
- Modify: `datafusion/datasource-parquet/src/push_decoder_lookahead.rs`

**Interfaces:**
- Consumes `LookaheadFileContext::depth()`.
- Produces `prefetched_readers: VecDeque<PrefetchedReader>` with at most `depth` entries and one active `next_reader_future`.

- [ ] **Step 1: Write a RED depth-four overlap test**

Use the existing scripted reader fixture with a small record-batch size. Consume one batch from row group 0, poll subsequent batches without draining row group 0, and assert requests for row groups 1, 2, and 3 occur for depth 4 but only row group 1 occurs for depth 1. The final concatenated output must equal the serial fixture.

```rust
#[tokio::test]
async fn depth_four_queues_multiple_readers_and_preserves_order() {
    let fixture = LookaheadFixture::with_batch_size(1);
    let (_, context) = fixture.lookahead_context_with_depth(4);
    let mut stream = fixture.lookahead_stream(context);

    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first, fixture.expected_batches()[0]);
    for _ in 0..8 {
        let _ = stream.next().await.unwrap().unwrap();
        if fixture.control.has_request_for(3) { break; }
    }
    assert!(fixture.control.has_request_for(1));
    assert!(fixture.control.has_request_for(2));
    assert!(fixture.control.has_request_for(3));
    let output = collect_with_prefix(first, stream).await.unwrap();
    assert_eq!(output, fixture.expected_output());
}
```

- [ ] **Step 2: Run RED test**

```bash
rtk cargo test -p datafusion-datasource-parquet depth_four_queues_multiple_readers_and_preserves_order -- --nocapture
```

Expected: compile failure because the fixture and state support only depth one.

- [ ] **Step 3: Change the state machine to a queue**

Replace `prefetched_reader: Option<PrefetchedReader>` with `prefetched_readers: VecDeque<PrefetchedReader>`. The key transitions are:

```rust
if self.active_reader.is_none()
    && let Some(prefetched) = self.prefetched_readers.pop_front()
{
    self.active_reader = Some(prefetched.reader);
    drop(prefetched.resources);
    continue;
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
    // Preserve the existing single future and decoder ownership transfer.
}
```

Push successful outcomes to the back. On termination, clear the queue before dropping the decoder and reader. Surface a deferred error only after all earlier queued readers are consumed.

- [ ] **Step 4: Add denial, error, limit, reverse-order, and cleanup tests at depth 4**

Extend existing depth-one tests so each runs for depths 1 and 4 where applicable. Exact assertions: no queue crosses a pending decoder-run boundary; final limit drops all pending futures/readers synchronously; a speculative error follows preceding queued output; reverse row groups remain reversed; all 24 range and 256 MiB permits are restored after drop.

- [ ] **Step 5: Run tests and commit**

```bash
rtk cargo test -p datafusion-datasource-parquet lookahead_driver -- --nocapture
rtk cargo clippy -p datafusion-datasource-parquet --all-targets -- -D warnings
rtk git add datafusion/datasource-parquet/src/push_decoder_lookahead.rs
rtk git commit -m "feat: queue bounded parquet row-group lookahead"
```

Expected: depth 1 and 4 tests pass with no leaked resources or ordering change.

### Task 3: Add a Minimal arrow-rs Next-row-group API

**Files:**
- Modify in the isolated arrow-rs worktree: `parquet/src/arrow/push_decoder/remaining.rs`
- Modify in the isolated arrow-rs worktree: `parquet/src/arrow/push_decoder/mod.rs`

**Interfaces:**
- Produces `ParquetPushDecoder::peek_next_row_group() -> Result<Option<usize>, ParquetError>` on arrow-rs commit `913bab26b`.
- This API reports the next queued file-level row group without mutating selection state.

- [ ] **Step 1: Create the isolated arrow branch**

```bash
rtk git -C /Users/qiwei.huang/Source/datafusion-workspace/arrow-rs worktree add -b codex/df54-row-group-prefetch-arrow /Users/qiwei.huang/Source/datafusion-workspace/codex/worktrees/arrow-df54-row-group-prefetch-20260714 913bab26ba9bed8fc2bc1acda300cc52345b0da1
```

Expected: clean worktree at the exact pinned SHA.

- [ ] **Step 2: Write RED peek tests**

Add tests for initial row group 0, explicit `with_row_groups(vec![1])`, an empty row selection that skips row group 0, repeated non-mutating calls, and `None` after finish.

```rust
#[test]
fn test_peek_next_row_group_skips_empty_selection() {
    let decoder = ParquetPushDecoderBuilder::try_new_decoder(test_file_parquet_metadata())
        .unwrap()
        .with_row_selection(RowSelection::from(vec![
            RowSelector::skip(250),
            RowSelector::select(100),
        ]))
        .build()
        .unwrap();
    assert_eq!(decoder.peek_next_row_group().unwrap(), Some(1));
    assert_eq!(decoder.peek_next_row_group().unwrap(), Some(1));
}
```

- [ ] **Step 3: Run RED test**

```bash
rtk cargo test -p parquet test_peek_next_row_group -- --nocapture
```

Expected: compile failure because `peek_next_row_group` does not exist.

- [ ] **Step 4: Implement a read-only queue/selection simulation**

On the 58.3 state shape, clone only `row_groups` and `selection`, split the cloned selection by metadata row counts, and return the first row group with non-zero selected rows:

```rust
pub fn peek_next_row_group(&self) -> Result<Option<usize>, ParquetError> {
    let mut selection = self.selection.clone();
    for &row_group_idx in &self.row_groups {
        let rows = usize::try_from(
            self.parquet_metadata.row_group(row_group_idx).num_rows()
        ).map_err(|e| ParquetError::General(format!("Row count overflow: {e}")))?;
        let selected = selection
            .as_mut()
            .map(|value| value.split_off(rows).row_count())
            .unwrap_or(rows);
        if selected != 0 { return Ok(Some(row_group_idx)); }
    }
    Ok(None)
}
```

Delegate from both reading and decoding states; return `None` when finished. Document that this pinned version has no decoder-level offset budget and that DataFusion disables multi-row-group span prefetch for file scans with a limit.

- [ ] **Step 5: Run parquet tests and commit**

```bash
rtk cargo test -p parquet test_peek_next_row_group -- --nocapture
rtk cargo clippy -p parquet --all-targets -- -D warnings
rtk git add parquet/src/arrow/push_decoder/remaining.rs parquet/src/arrow/push_decoder/mod.rs
rtk git commit -m "feat(parquet): expose next push-decoder row group"
```

Expected: all new peek tests and existing push-decoder tests pass.

### Task 4: Build Projected Row-group Span Plans

**Files:**
- Create: `datafusion/datasource-parquet/src/row_group_prefetch.rs`
- Modify: `datafusion/datasource-parquet/src/lib.rs`
- Modify: `datafusion/datasource-parquet/src/row_filter.rs`
- Modify: `datafusion/datasource-parquet/src/opener/mod.rs`

**Interfaces:**
- Produces `RowGroupPrefetchPlan`, `DensityAdmission`, and `RowGroupPrefetchMetrics`.
- Consumes final `PreparedAccessPlan::row_group_indexes`, a union `ProjectionMask` for filter and output expressions, and `ParquetMetaData`.

- [ ] **Step 1: Write RED span and density tests**

Construct metadata with four row groups and projected/non-projected columns. Assert only projected leaf `byte_range()` values are included, ranges are sorted, adjacent ranges merge up to 256 KiB gap, merged ranges never exceed 4 MiB, reverse row-group order is retained, and density requires at least 1 MiB unique exact payload at 0.80.

```rust
#[test]
fn window_ranges_merge_projected_chunks_across_row_groups() {
    let plan = RowGroupPrefetchPlan::new(
        metadata(),
        ProjectionMask::leaves(schema(), [0, 2]),
        vec![0, 1, 2, 3],
        256 * 1024,
        4 * 1024 * 1024,
    );
    let ranges = plan.ranges_for(&[0, 1, 2, 3]);
    assert!(ranges.iter().all(|r| r.end - r.start <= 4 * 1024 * 1024));
    assert_eq!(plan.row_group_order(), &[0, 1, 2, 3]);
    assert_eq!(plan.projected_payload_bytes(), expected_projected_bytes());
}
```

- [ ] **Step 2: Run RED tests**

```bash
rtk cargo test -p datafusion-datasource-parquet row_group_prefetch -- --nocapture
```

Expected: compile failure because the module does not exist.

- [ ] **Step 3: Implement projected span extraction**

For every selected row group, iterate columns with `projection_mask.leaf_included(index)`, convert each `(offset, length)` from `ColumnChunkMetaData::byte_range()` into `offset..offset + length`, and store per-row-group exact ranges. `ranges_for` flattens requested row groups in supplied order and merges only forward-adjacent ranges within the configured gap and maximum size.

- [ ] **Step 4: Build the filter-plus-output projection union**

Before moving `prepared.predicate` or projection expressions, call `build_projection_read_plan` with the output projection expressions chained with the optional predicate expression. This intentionally includes every leaf needed by either dynamic filtering or output decoding. Create one `RowGroupPrefetchPlan` for each prepared decoder run, using that run's final ordered `row_group_indexes`.

- [ ] **Step 5: Add metrics and commit**

Create execution metrics for observed exact bytes, candidate bytes, prefetch windows, prefetched ranges/bytes, useful staged bytes, unused staged bytes, admission enables, admission denials, and peak staged bytes. Register them with filename and partition labels so `EXPLAIN ANALYZE`/debug physical plans expose them.

```bash
rtk cargo test -p datafusion-datasource-parquet row_group_prefetch -- --nocapture
rtk git add datafusion/datasource-parquet/src/lib.rs datafusion/datasource-parquet/src/row_group_prefetch.rs datafusion/datasource-parquet/src/row_filter.rs datafusion/datasource-parquet/src/opener/mod.rs
rtk git commit -m "feat: plan projected parquet row-group prefetch spans"
```

### Task 5: Stage Density-gated Windows into the Push Decoder

**Files:**
- Modify: `datafusion/common/src/config.rs`
- Modify protobuf/config docs for `row_group_prefetch_window`
- Modify: `datafusion/datasource-parquet/src/lookahead.rs`
- Modify: `datafusion/datasource-parquet/src/opener/mod.rs`
- Modify: `datafusion/datasource-parquet/src/push_decoder_lookahead.rs`
- Modify: `datafusion/datasource-parquet/src/row_group_prefetch.rs`

**Interfaces:**
- Produces `datafusion.execution.parquet.row_group_prefetch_window: usize`, default 0, valid values 0, 2, 4.
- Uses fixed density thresholds 0.80 enable, 0.50 disable and fixed merge controls 256 KiB gap, 4 MiB maximum range.

- [ ] **Step 1: Write RED admission and staging tests**

Add tests proving: window 0 makes no extra request; a sparse first row group never enables staging; a dense first row group causes the next two/four groups to be fetched before their exact `NeedsData`; staged bytes satisfy later decoder requests without another fetch; limit-bearing scans stay exact; cancellation and errors release all permits; output equals serial pushdown output.

- [ ] **Step 2: Add and validate the window config**

Add the config field, environment/proto round trips, and validation. The source must reject values other than 0, 2, 4. Window prefetch requires `row_group_lookahead=true`; otherwise return a configuration error rather than silently enabling an inert option.

- [ ] **Step 3: Pass one plan per decoder run**

Extend `LookaheadPushDecoderStreamState::new` with active and pending `RowGroupPrefetchPlan` values aligned 1:1 with active and pending decoders. `advance_decoder_run` pops both queues together and asserts neither can drift.

- [ ] **Step 4: Observe exact requests and prefetch the admitted window**

Before exact driving, capture `decoder.peek_next_row_group()`. Every exact `NeedsData(ranges)` updates the active plan's unique interval coverage. Once admission enables, call `plan.ranges_for(next_window_indices)`, reserve all bytes and range permits before I/O, fetch via `AsyncFileReader::get_byte_ranges`, and call `decoder.push_ranges` with those coalesced ranges. The push decoder already accepts data before asking for it.

- [ ] **Step 5: Retain resources until the owning row group is handed off**

Represent staged resources as one non-overlapping window. Coalesced ranges may
span several row groups, so their reservation cannot be released per row group:

```rust
struct StagedWindowResources {
    remaining_row_groups: VecDeque<usize>,
    resources: SpeculativeResources,
    staged_bytes: usize,
}
```

Use non-overlapping windows. Each time `try_next_reader` yields the row group
reported by peek, pop that index from `remaining_row_groups`. Keep the full
window reservation attached to decoder state while any staged future row group
remains. When the final staged row group is built, move the window resources
into that row group's `PrefetchedReader`; release them only when that reader
becomes foreground. Drop the full remaining window on cancellation, error,
denial, run transition, or termination. Never release its memory-pool
reservation merely because an earlier reader becomes active.

- [ ] **Step 6: Run state-machine tests and commit**

```bash
rtk cargo test -p datafusion-datasource-parquet lookahead_driver -- --nocapture
rtk cargo test -p datafusion-datasource-parquet row_group_prefetch -- --nocapture
rtk cargo clippy -p datafusion-datasource-parquet --all-targets -- -D warnings
rtk git add datafusion/common/src/config.rs datafusion/datasource-parquet datafusion/proto-common datafusion/proto docs/source/user-guide/configs.md
rtk git commit -m "feat: prefetch dense parquet row-group windows"
```

Expected: exact, depth-only, sparse, dense-window, reverse, limit, error, cancellation, and memory-denial paths pass.

### Task 6: Add Dual-source Remote Setup and Sweep Scripts

**Files:**
- Create: `/Users/qiwei.huang/Source/datafusion-workspace/codex/scripts/setup-df54-row-group-prefetch-remote.sh`
- Modify: `/Users/qiwei.huang/Source/datafusion-workspace/codex/scripts/run-df54-interleaved-ranges-remote-bench.sh`
- Create: `/Users/qiwei.huang/Source/datafusion-workspace/codex/scripts/run-df54-row-group-prefetch-sweep.sh`
- Modify: `/Users/qiwei.huang/Source/datafusion-workspace/codex/scripts/summarize-interleaved-range-sweep.py`
- Create: `/Users/qiwei.huang/Source/datafusion-workspace/codex/agents/2026-07-14-row-group-prefetch-runbook.md`

**Interfaces:**
- Accepts verified DataFusion and arrow-rs bundles, exact 40-character SHAs, and safe absolute remote paths.
- Produces manifests recording both repositories and all resource/config controls.

- [ ] **Step 1: Implement dual-bundle setup**

Verify SHA-256 for both bundles, `git bundle verify`, fetch exact heads, create or safely fast-forward clean detached worktrees, and point the DataFusion `.cargo/config.toml` Arrow patch path at the isolated candidate arrow worktree. Never mutate the shared clean arrow 58.3 source.

- [ ] **Step 2: Add depth/window arguments**

Add `--lookahead-depth 1|2|4` and `--prefetch-window 0|2|4`. Export DataFusion environment names, record them in manifests, and reject window > depth. The sweep sets depth to `max(best_b1_depth, window)` so both windows are always exercised.

- [ ] **Step 3: Preserve remote guards**

Use the same CPU `0-23`, 48 GiB virtual-memory, 64 GiB availability, disk `<80%`, 16 build jobs, and competing-process checks for both Cargo source trees and dfbench.

- [ ] **Step 4: Add exact B1/B2 matrices and summarization**

B1 runs q72 depth 1, 2, 4 with window 0. B2 runs windows 2 and 4 at depth `max(best_b1_depth, window)` and uses a matching window-0 baseline at that depth. Summaries include wall time, physical throughput, GET count/size, observed density, prefetched/useful/unused bytes, read amplification, peak staged memory, and output rows.

- [ ] **Step 5: Syntax-check scripts**

```bash
rtk bash -n codex/scripts/setup-df54-row-group-prefetch-remote.sh
rtk bash -n codex/scripts/run-df54-interleaved-ranges-remote-bench.sh
rtk bash -n codex/scripts/run-df54-row-group-prefetch-sweep.sh
rtk python3 -m py_compile codex/scripts/summarize-interleaved-range-sweep.py
```

Expected: all checks pass. Root scripts remain Codex-owned files because the workspace root is not a Git repository.

### Task 7: Execute Route B Screening and Regression Gates

**Files:**
- Create results under: `/Users/qiwei.huang/Source/datafusion-workspace/codex/logs/datafusion-tpcds/row_group_prefetch_20260714/`

**Interfaces:**
- Produces controlled B1/B2 q72 evidence, q21/q82 regressions, and a route decision comparable with Route A.

- [ ] **Step 1: Run remote arrow/DataFusion tests, Clippy, and release-nonlto build**

Run pinned arrow-rs push-decoder tests, DataFusion lookahead/prefetch/config/proto tests, both Clippy targets, and build dfbench with both exact SHAs recorded.

- [ ] **Step 2: Re-run the alternating baseline**

Use factor 8, 256 KiB object-store coalesce gap, per-call parallelism 10, global physical limit 24, lookahead depth 1, window 0, and q72. Do not use an old baseline as the sole comparator.

- [ ] **Step 3: Run B1 depth screening**

Run q72 depths 1, 2, 4 one round; run the best depth for three alternating rounds. Record that B1 request sizes should remain near the existing shape; its purpose is to test pipeline starvation.

- [ ] **Step 4: Run B2 window screening**

Run windows 2 and 4 one round at depth `max(best_b1_depth, window)`, each against a matching window-0 baseline at the same depth. Run the best for three alternating rounds. Advance only with at least 10% median q72 improvement, no more than 1.25x physical bytes, no more than 256 MiB staged memory, identical results, and no failure.

- [ ] **Step 5: Run q21/q82 regression and selected-query validation**

Each query must regress by no more than 5%. Every passing Route A or B candidate then receives five alternating rounds on q21/q72/q82.

- [ ] **Step 6: Select the complete-suite candidate**

If both routes pass and differ by less than 3% on q72, prefer Route B. Run one complete 99-query S3 screening round. Advance to the standard release-profile complete run only when aggregate time does not regress by more than 3%, no unexplained query regresses by more than 20%, and every result matches.

- [ ] **Step 7: Archive evidence and upstream recommendation**

Download and checksum raw logs, manifests, metrics, and summaries. Document measured timing separately from profiles and inference. Identify the arrow-rs API commit, DataFusion queue/prefetch commits, rejected variants, resource compliance, and the minimal upstreamable patch series.
