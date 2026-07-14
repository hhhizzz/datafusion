# S3 Parquet Prefetch Experiments

Date: 2026-07-14

Status: Design approved in conversation; written review pending

## Context

The current DataFusion candidate improves S3 scan parallelism by interleaving
file byte ranges. On SF10 TPC-DS, factor 16 improves q21 and q82 by about 4.06x
and 2.75x respectively. It improves q72 much less: the best measured q72 shape
is about 2.46 seconds at factor 8 and object-store range parallelism 10.

The q72 `inventory` scan provides the key evidence:

- It reads about 707.5 MiB of 710.2 MiB of candidate projected bytes (99.61%).
- It issues about 5,419 S3 GETs, with requests near 128 KiB.
- Each `get_ranges` call contains only one to three logical ranges and never
  saturates the configurable per-call range concurrency.
- Disabling row pushdown changes the request shape but regresses q72 from about
  2.46 seconds to about 9.06 seconds because unfiltered rows reach decoding and
  joins.
- The same object-store client reaches about 800 MiB/s for 128 KiB requests,
  1.84 GiB/s for 512 KiB requests, and 2.77 GiB/s for 4 MiB requests at up to
  24 concurrent requests.

The transport and Ceph endpoint are therefore not the primary limit. The
remaining problem is to preserve dynamic row-filter reduction while turning
multi-phase Parquet reads into larger or better-pipelined remote reads.

## Goals

1. Measure the upper bound from query-local aligned block reuse without
   changing Parquet decoding semantics.
2. Measure a production-oriented, density-gated multi-row-group prefetch path
   that uses Parquet metadata and the existing push decoder.
3. Keep all speculative memory and I/O bounded and accounted.
4. Preserve result rows, ordering contracts, cancellation, errors, and row
   pushdown behavior.
5. Select candidates using controlled q72, q21, and q82 experiments before any
   complete TPC-DS run.

## Non-goals

- No cross-query or cross-round warm cache is permitted.
- No local NVMe cache, data rewrite, partition-layout change, or Ceph tuning is
  part of the claimed S3 comparison.
- The benchmark-only quantum cache is not proposed directly as a public
  DataFusion feature.
- A candidate that only makes q72 faster by disabling row pushdown is invalid.
- Increasing HTTP concurrency, object-store coalesce gap, or page-index toggles
  alone will not be repeated; controlled experiments already rejected them.

## Shared Resource Contract

Every build and benchmark must satisfy all of the following:

- Run on `sz-data-b-1` with CPU affinity `0-23`.
- Use at most 24 asynchronous S3 requests globally. The instrumented store
  applies one physical-request semaphore shared by foreground, lookahead, and
  cache-fill reads; per-call range concurrency remains capped at 10 unless a
  named experiment changes it to another value no greater than 24.
- Use no more than 16 Cargo jobs unless an existing wrapper enforces a lower
  limit.
- Apply `ulimit -Sv 50331648` before compilation, linking, or execution.
- Require at least 64 GiB `MemAvailable` immediately before starting work.
- Require filesystem usage below 80% for
  `/home/inspur/arrow-rs-codex` before starting work.
- Refuse to start when a competing Cargo, rustc, clippy, or dfbench process is
  present.
- Keep the existing 256 MiB global speculative-read budget. A cache prototype
  may reserve at most another 512 MiB for resident plus in-flight blocks, and
  must expose its current and peak reserved bytes. Both budgets are registered
  with the DataFusion `MemoryPool`.
- Record all guards, source SHAs, binary path, environment controls, query
  list, rounds, output rows, and failures in the run manifest.

## Source Isolation

The current DataFusion experiment branch remains the common parent. Each route
uses a separate child worktree and branch:

- Route A changes only the DataFusion benchmark harness and its tests.
- Route B changes DataFusion parquet lookahead and, where required, an isolated
  arrow-rs worktree pinned by an explicit path override.
- Baseline and candidates use the same release-nonlto target family during
  screening. Formal evidence requires a release rebuild.

This keeps route A's benchmark instrumentation out of route B's production
code comparison and gives every result exact source lineage.

## Route A: Query-local Quantum Cache

### Placement and lifetime

Add a benchmark-only `ObjectStore` wrapper between the TPC-DS scan and the
instrumented S3 store. The wrapper exists for one benchmark query iteration.
It is dropped or explicitly reset before the next iteration, query, or round.
Cache keys include the object path and aligned block offset. The immutable
benchmark bucket permits query-local path keys; a production implementation
would additionally require ETag or version identity.

The wrapper supports `get_range` and `get_ranges`. Metadata, list, write, and
conditional operations pass through unchanged. Coalesced input ranges are
sliced back into the exact byte sequences and order requested by the caller.

### Modes

The prototype has three explicit modes:

- `off`: unchanged baseline.
- `forced`: every eligible range is served from aligned blocks. This measures
  the upper bound and is not an automatic policy.
- `adaptive`: begin with exact reads and track interval coverage inside the
  aligned blocks touched by each object. The observation density is unique
  requested bytes divided by the total bytes in those touched blocks. Enable
  aligned reads only after at least 1 MiB of unique payload and density at or
  above 0.80. Once enabled, evaluate non-overlapping 32-block epochs and stop
  new block admission after two consecutive epochs below 0.50 useful-byte
  density. Existing admitted blocks may still satisfy hits.

Test block sizes are 512 KiB, 1 MiB, and 4 MiB. The retained cache cap is
512 MiB including blocks being loaded. A memory-pool reservation is acquired
before the S3 request and held until the resident block is evicted. A shared
future or equivalent single-flight entry ensures concurrent misses for one
block perform one S3 GET. Failed loads release their reservation and are not
cached.

The forced sweep is run first. If no forced block size improves q72 by at least
5%, the adaptive path is still unit-tested but does not advance to expensive
benchmark matrices because its upper bound has already failed.

### Metrics

Route A records at least:

- logical request count and bytes before the cache;
- physical S3 GET count and bytes after the cache;
- cache hit, partial-hit, and miss bytes;
- unique requested payload bytes;
- block bytes fetched and consumed;
- alignment and unused overread bytes;
- single-flight joins;
- current and peak retained bytes;
- adaptive transitions and their observed density.

Physical object-store metrics remain inside the cache wrapper so they describe
actual S3 traffic. Cache metrics describe the original decoder requests.

### Correctness and failure behavior

- Unit tests cover aligned, unaligned, cross-block, overlapping, duplicate,
  concurrent, short-final-block, and ordered multi-range reads.
- Cancellation must release byte permits and waiter state.
- An S3 error is returned to all current waiters and the failed block entry is
  removed so a later request may retry.
- Eviction must not invalidate bytes already returned to callers.
- Metrics tests verify physical versus logical byte accounting.

## Route B: Density-gated Multi-row-group Prefetch

Route B extends the existing depth-one push-decoder lookahead. It is split into
two independently measurable increments so request starvation and request size
are not conflated.

### B1: Configurable lookahead queue depth

Replace the single `prefetched_reader` slot with an ordered bounded queue. Test
depths 1, 2, and 4. The driver may continue advancing the decoder and building
future row-group readers while the active reader is decoded, until the queue is
full or a shared resource guard denies speculation.

Readers are consumed strictly in row-group order. Deferred errors remain
ordered behind already completed preceding readers. A foreground denial falls
back to the existing non-speculative path. The existing global limits remain
24 in-flight ranges and 256 MiB speculative bytes, reserved through the
DataFusion `MemoryPool`.

B1 does not enlarge individual requests. Its purpose is to determine how much
of the gap from about 343 MiB/s to the measured 128 KiB ceiling of about
800 MiB/s is caused by insufficient pipeline depth.

### B2: Density-gated row-group span prefetch

At a row-group boundary, the decoder exposes the next readable row-group
indices without mutating the frontier. For a bounded window of two or four row
groups, a planner derives conservative compressed byte spans from Parquet
metadata for the filter and output projection.

The path has two stages:

1. Observe exact decoder requests and calculate unique requested bytes against
   the candidate projected spans.
2. After density reaches 0.80, fetch coalesced candidate spans for upcoming row
   groups and push those bytes into the decoder before they are requested.

The push decoder already permits data to arrive before `NeedsData`. Later
predicate and projection phases therefore reuse staged bytes while dynamic
row filters and downstream row reduction remain unchanged. Sparse scans stay
on exact ranges. Speculation stops immediately when the memory pool, 256 MiB
byte budget, 24-range budget, cancellation state, row ordering, or decoder-run
boundary makes the operation unsafe.

The first prototype may conservatively use the full row-group compressed span
when every leaf column is projected, as in q72 `inventory`. General projected
column-span planning is required before this can be recommended upstream.

### Arrow/DataFusion boundary

The smallest arrow-rs API addition should expose a non-mutating bounded peek of
future readable row-group indices and enough metadata to construct candidate
spans. DataFusion owns asynchronous object-store scheduling, memory-pool
reservations, and admission policy. Arrow-rs continues to own row selection,
predicate evaluation, decoding, and ordered reader production.

This boundary avoids cloning predicate closures or maintaining multiple
independent predicate state machines.

### Correctness and failure behavior

- Tests cover depth, ordering, reverse row-group order, global selection,
  offset/limit boundaries, decoder-run boundaries, deferred errors,
  cancellation, budget denial, and no-prefetch sparse admission.
- Prefetched bytes are released when consumed, cancelled, denied, or when the
  stream terminates.
- A speculative error is deferred until its row group becomes foreground; it
  must not suppress valid preceding batches.
- The depth-one setting must preserve the current behavior exactly.

## Experiment Sequence

1. Re-run the unchanged q72 baseline with factor 8, coalesce gap 256 KiB, range
   parallelism 10, and row-group lookahead enabled.
2. Route A forced screen: q72, one round, block sizes 512 KiB, 1 MiB, 4 MiB.
3. Route A adaptive screen using the best forced block size: q72, three
   alternating baseline/candidate rounds.
4. Route A regression screen: q21 and q82, three alternating rounds.
5. Route B1 screen: q72 at depths 1, 2, 4, one round, followed by three rounds
   for the best depth.
6. Route B2 screen: q72 windows 2 and 4, one round, followed by three rounds for
   the best window.
7. Route B regression screen: q21 and q82, three alternating rounds.
8. Run five-round selected-query validation for every candidate that passes
   the admission gate.
9. Run a complete 99-query S3 screening round for the selected candidate. If
   correctness and aggregate regression gates pass, run the formal complete
   suite under the standard experiment contract.

Baseline and candidate runs alternate within one host session. No candidate is
compared with an old result as its sole timing baseline.

## Admission Gates

A candidate advances from q72 screening only when all conditions hold:

- result row count and values match baseline;
- no failures, panics, hangs, or leaked remote process;
- q72 median improves by at least 10% over the alternating baseline;
- physical S3 bytes are no more than 1.25x baseline;
- peak resident plus in-flight cache blocks and speculative bytes remain within
  768 MiB;
- process virtual-memory and host-availability guards remain satisfied.

A candidate advances to a complete suite only when q21 and q82 each regress by
no more than 5% and all request/memory accounting reconciles. A complete-suite
candidate is rejected when aggregate wall time regresses by more than 3%, any
individual query regresses by more than 20% without an explained tradeoff, or
any result differs.

If both routes pass and their q72 medians differ by less than 3%, prefer route B
because it has explicit Parquet semantics and a clearer upstream path. Route A
remains an upper-bound control.

## Deliverables

- Isolated source branches and exact SHAs for baseline and both routes.
- Unit and integration tests for each new state transition and memory guard.
- Guarded setup, build, run, sweep, and summarization scripts.
- Raw logs, manifests, JSON results, metrics, and comparison summaries under
  `codex/logs/datafusion-tpcds`.
- A final report separating measured timing, profile evidence, and inference.
- An upstream recommendation describing which parts belong in DataFusion and
  which require arrow-rs API work.
