# Current Row Group Tail Prefetch Design

## Objective

Reduce fragmented S3 reads for dense Parquet row filtering when DataFusion's
file repartitioning and pruning leave only one selected row group in each
decoder run. Preserve query results, output order, limits, cancellation, and
the existing resource contract.

## Evidence

The Route B q72 run selected about 762 MB of projected payload and observed the
same amount through exact decoder requests. It recorded 25 admission enables
but zero prefetch windows. The SF10 `inventory.parquet` file has 1,084 row
groups, while q72 exposes about 25 selected row groups across 24 scan
partitions. Consequently, most decoder runs have no future selected row group.

At the object-store boundary q72 issues 5,650 physical GETs for 762.1 MB, with
a 78-byte median wire range and a request window near 2.7 seconds. Increasing
per-call `get_ranges` parallelism cannot help because each call has at most two
coalesced ranges. Fixed quantum caching reduced GET count but regressed q72 to
about 22 seconds, so it is not the next implementation path.

## Alternatives

1. Prefetch the unread projected tail of the current row group after exact
   byte coverage reaches 80%. This is the selected approach because it reuses
   the existing push-decoder staging API and bounds physical-byte amplification.
2. Predict dense output pages before I/O and prefetch output-column spans while
   predicate columns are read. This has a higher ceiling but needs a larger
   Arrow/DataFusion planning interface and cannot yet prove a byte bound from
   row-selection density alone.
3. Repair and retune the query-level fixed-block cache. This is broader and the
   current prototype has a large scheduling regression, so it remains a
   separate diagnostic path.

## Design

Each `DensityAdmission` retains the unique exact ranges observed inside one
row group's projected compressed spans. When coverage first reaches 80%, it
returns the projected ranges not already covered by exact requests. The ranges
are sorted and coalesced with the existing 256 KiB gap and 4 MiB maximum span.

`PrefetchRunState::observe_exact_ranges` first tries the current-row-group tail.
If the tail is non-empty, it creates a staging request owned by that same row
group. Future-row-group staging remains a separate fallback and is attempted
only when the current tail has no work. A staged request records whether it is
a current tail or a future window so metrics and lifecycle logic cannot confuse
the two.

The existing speculative range semaphore, memory reservation, and staged
window lease own the fetched bytes. Current-tail bytes are retired when the
current row-group reader is handed off. Cancellation, denial, read error,
decoder transition, and stream termination clear decoder buffers before
releasing the reservation and permits.

No detached tasks are introduced. The first prototype keeps the existing
ordered fetch path so correctness and resource ownership remain explicit.

## Read Amplification Bound

Let `C` be all projected candidate bytes for a row group, `O` the unique exact
bytes observed when admission fires, and `E` the exact bytes the baseline would
eventually read. Admission requires `O >= 0.8C`, and necessarily `E >= O`.
Fetching only `C - O` yields at most `C / E <= C / O <= 1.25` payload
amplification before the small, separately measured coalescing-gap overfetch.

Already observed intervals must be subtracted before staging. If subtraction
or byte accounting overflows, staging is denied and the exact path continues.

## Metrics

Add separate counters for current-tail admissions, requests, ranges, bytes,
and empty tails. Keep future-window metrics unchanged. The benchmark gate
requires at least one current-tail request and non-zero bytes; a configuration
that only enables admission is not considered exercised.

## Tests

1. A single-row-group decoder with dense exact coverage must reproduce the
   current q72 failure before the change: admission enables but no future
   window exists.
2. After the change it stages only the uncovered tail and does not fetch an
   already observed interval twice.
3. Sparse coverage, empty tails, oversized range sets, permit denial, memory
   denial, limits, cancellation, read errors, and reverse scans preserve the
   exact fallback and release every resource.
4. Existing multi-row-group, decoder-run-boundary, and output-equality tests
   remain green.

## Experiment Gates

Run q72 with factor 8, 256 KiB object-store coalescing, per-call parallelism 10,
24 global physical requests, 24 partitions, CPU 0-23, 256 MiB speculative
bytes, and the 48 GiB process limit.

The candidate advances from one-round screening only when results match,
current-tail metrics are non-zero, physical bytes are no more than 1.25x the
fresh baseline, peak staged bytes are no more than 256 MiB, and wall time is at
least 5% better. Three alternating pairs require at least 10% median q72
improvement. q21 and q82 may regress by no more than 5%. Only a passing
candidate receives five selected-query rounds and a complete S3 TPC-DS run.
