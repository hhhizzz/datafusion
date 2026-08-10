# Shape-aware selected decoding — DataFusion side

**Status: closed. Not proposed as an upstream feature.**

This branch holds the DataFusion half of an investigation whose main write-up
lives in the companion arrow-rs branch. Read that first:

> **arrow-rs branch** `exp/v21-rle-selected-fill-20260807`, file
> `experiments/shape-aware-selected-decoding/README.md`
> (tag `exp/shape-aware-selected-decoding-final`)

## What is on this branch

| Change | Location |
|---|---|
| Session option `datafusion.execution.parquet.selected_decode` (**default false**) | `datafusion/common/src/config.rs` |
| Plumbing to the Parquet reader | `datafusion/datasource-parquet/src/{source,opener/mod,push_decoder}.rs` |
| Per-query coverage reporting (`DFEXP_SELECTED_DECODE_COVERAGE=`) | `benchmarks/src/{clickbench,tpcds/run}.rs` |
| Leaf benchmark harnesses, including the `production_shape` comparator | `benchmarks/v21-rle-selected-fill/` |

The option is **off by default** and gates an experimental arrow-rs reader path.
With the companion arrow-rs branch absent it has no effect.

## Commit pins

| Branch | SHA | Role |
|---|---|---|
| `exp/v21-rle-selected-fill-20260807` | see tag below | shared base, option defaults **false** |
| `exp/v26-gw2-selected-decode-on-20260809` | `a5765d32c8` | measurement-only arm, option defaults **true**; differs from the base in exactly one file. Never intended for merge |
| `exp/v26-gw1-selected-decode-on-20260809` | `ef358c22d0` | earlier measurement-only arm, retained as the evidence pin for the correctness gate's first (failing) run |

## Outcome

The kernel is 2.6x–4.6x faster than the production-shaped baseline at the leaf,
and the integration is correct — but under the conservative v0 admission rule
(all projected columns flat, required, primitive) only **0 of 99 TPC-DS** and
**4 of 42 ClickBench** queries ever reach it, and no repeatable query-level
benefit was established. Full reasoning, data and caveats are in the arrow-rs
report.
