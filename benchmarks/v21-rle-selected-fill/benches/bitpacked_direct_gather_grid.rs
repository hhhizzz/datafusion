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

//! Full synthetic-grid Criterion bench for experiment `arrow-bitpacked-direct-gather-gate-v23`.
//! See `codex/experiments/arrow-bitpacked-direct-gather-gate-v23.md` -- this is the "Synthetic
//! grid" fixture family the frozen contract requires, matching
//! `arrow-paper-select-fourarm-v18`'s own axes (ticket 14) exactly, now measured through the
//! real `RleDecoder::get_batch_with_dict_selected_direct_gather`/`_checked` seam instead of
//! v18's clean-room standalone kernel. `bitpacked_direct_gather.rs` (the smoke stage, already
//! run and passed on real hardware) is this file's correctness prerequisite; this file adds no
//! new arm and reuses its exact 5-arm set, just at v18's paper-comparable scale and grid.
//!
//! ## Arms (unchanged from the smoke stage)
//!
//! `cursor` (reference, R1.5-admitted baseline) / `direct_gather` / `direct_gather_checked` /
//! `decode_all_indices_compact` / `materialize_then_filter`. See `bitpacked_direct_gather.rs`'s
//! own module doc comment for what each does.
//!
//! ## Grid: matched to v18/ticket 14, not re-derived
//!
//! - `k in {2, 5, 8, 12}` (v18's exact set -- note this is *not* the smoke stage's `{2, 8, 12,
//!   16}`; the frozen contract says "k x survival x shape as in ticket 14", so this file matches
//!   ticket 14's k values precisely rather than reusing the smoke stage's own choice).
//! - Selection: survival `s in {1/64, 1/16, 1/4}` x shape `{random (iid Bernoulli), clustered
//!   (geometric runs, mean run 64)}`, plus a dense control `s=1`, per `k`.
//! - Page = one bit-packed hybrid run, `n_values = floor(2^23/k)` rounded down to a multiple of
//!   512 (v18's exact `long_run_n_values` formula -- same per-cell packed working set, ~32 MiB
//!   across 32 pages, large enough to defeat single-page cache residency, matching v18's own
//!   measurement methodology). 8 zero bytes appended after the payload (v18's own fixture
//!   contract: "8-byte tail padding after every packed buffer"), which is *why* v18 could always
//!   measure the unchecked fast path -- without it, `direct_gather`'s own tail-safety check would
//!   incidentally force a page whose single run ends exactly at the buffer's end into the
//!   fallback branch (see `bitpacked_direct_gather.rs`'s module doc comment, group 1, for a case
//!   where the smoke stage deliberately did *not* pad and measured that fallback path instead).
//! - Guard cell `writer_real`: `k=8, s=1/16, random`, same total payload as the `k=8` long-run
//!   twin but encoded as `WRITER_REAL_RUN_LEN=504`-value runs back-to-back (v18's own writer
//!   -realistic run cap; also matches this program's own v22 finding that real bit-packed runs
//!   are all <=512 values). Unlike v18 -- whose clean-room arms have no notion of run boundaries
//!   and must be invoked once per run to measure per-run entry tax -- this file's arms all go
//!   through the real `RleDecoder`, whose `reload()` already dispatches across run boundaries
//!   *inside* one top-level call (the smoke stage's "mixed" cell group already proved this
//!   dispatch is correct for alternating run kinds). So `writer_real` here needs no per-run
//!   splitting machinery at all: one flat page-level mask, one call per page, exactly like every
//!   other cell -- the entry tax this measures is now "does resuming through many small runs cost
//!   the real decode seam anything", not v18's "does the clean-room kernel's per-call phase-table
//!   setup cost 3x", a different and arguably more production-relevant question.
//! - `P = 32` pages per cell, round-robin, freshly generated per page (distinct seeds).
//!
//! Cells: `4k x (3s x 2shape + 1 dense) + 1 guard = 29` (identical count to v18).
//!
//! ## v23-specific addition beyond v18: timed multi-call incremental consumption
//!
//! v18 never measured this (its clean-room arms have no notion of "call" at all beyond one
//! invocation per run). The smoke stage proved *correctness* for it (2 cells, untimed). This
//! file adds a small *timed* extension: `k=8` (v18's anchor k) crossed with the grid's own 3
//! survival points, each page's single run consumed via 3 successive calls at arbitrary,
//! non-64-aligned split points (`MULTI_CALL_SPLITS`), instead of the grid's usual one call per
//! page -- the shape a real batched caller (an 8192-row `RecordBatch` read against a <=512-value
//! run) actually uses. 3 cells, `cursor`/`direct_gather`/`direct_gather_checked` only (arms C/D
//! are single-call constructs in every other cell here too, so a multi-call variant of them adds
//! nothing this file's single-call cells don't already cover for those two arms).
//!
//! ## Protocol
//!
//! Untimed cross-arm FNV-1a-64 digest verification per page at setup (every arm against
//! `cursor`), panicking on any mismatch, before any timed measurement -- unchanged from every
//! prior stage in this program. Criterion `sample_size=12`, `warm_up_time=1s`,
//! `measurement_time=2.5s`, matching v18/R1/R1.5 exactly. Two full rounds are run as separate
//! K8s Jobs (not by this file), with the same direction-agreement rule v18/R1 used.

use std::hint;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, criterion_group, criterion_main};
use parquet::encodings::rle::{PackedSelection, RleDecoder};

// kernel.rs is shared across this crate's bench binaries; not every helper it exports is used
// by every binary.
#[allow(dead_code)]
#[path = "rle_fill/kernel.rs"]
mod kernel;

/// Distinct from every other bench file's seed constant in this crate, so this file's fixtures
/// are independently generated, not a reused subset of any prior stage's exact bytes.
const TOP_SEED: u64 = 0xC0FF_EE15_2026_0807_u64 ^ 0x9723;

const PAGES_PER_CELL: usize = 32;
const RLE_CHUNK: usize = 1024;

/// Real-world Parquet writer bit-packed-run cap (v18's own constant, re-derived independently
/// in this program's v22 static census: real bit-packed runs are all <=512 values). 504 = 63
/// groups of 8, the largest multiple of 8 whose LEB128 run-header indicator, `(63 << 1) | 1 =
/// 127`, still fits in a single ULEB128 byte.
const WRITER_REAL_RUN_LEN: usize = 504;

/// Clustered guard/cell target mean selected-run length, matching v18's own hardcoded Markov
/// -chain tuning ("mean selected-run length ~64").
const CLUSTERED_TARGET_MEAN_RUN: f64 = 64.0;

/// Arbitrary, non-64-aligned, non-8-aligned split points for the multi-call cell group, scaled
/// to `long_run_n_values(8)` (the k=8 page's own single-run value count): three chunks summing
/// to it exactly.
fn multi_call_splits(n_values: usize) -> [usize; 3] {
    let a = (n_values as f64 * 0.31) as usize;
    let b = (n_values as f64 * 0.36) as usize;
    [a, b, n_values - a - b]
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    Random,
    Clustered,
    Dense,
    WriterReal,
}

impl Shape {
    fn label(self) -> &'static str {
        match self {
            Shape::Random => "random",
            Shape::Clustered => "clustered",
            Shape::Dense => "dense",
            Shape::WriterReal => "writer_real",
        }
    }
}

struct CellSpec {
    k: u8,
    shape: Shape,
    survival: f64,
    denom: u32,
}

/// 4 k-values x (3 survivals x 2 shapes + 1 dense) + 1 guard = 29 cells (matches
/// `arrow-paper-select-fourarm-v18`'s own count exactly).
fn build_cells() -> Vec<CellSpec> {
    const KS: [u8; 4] = [2, 5, 8, 12];
    const SURVIVALS: [(f64, u32); 3] = [(1.0 / 64.0, 64), (1.0 / 16.0, 16), (1.0 / 4.0, 4)];

    let mut cells = Vec::with_capacity(29);
    for &k in &KS {
        for shape in [Shape::Random, Shape::Clustered] {
            for &(survival, denom) in &SURVIVALS {
                cells.push(CellSpec { k, shape, survival, denom });
            }
        }
        cells.push(CellSpec { k, shape: Shape::Dense, survival: 1.0, denom: 1 });
    }
    cells.push(CellSpec { k: 8, shape: Shape::WriterReal, survival: 1.0 / 16.0, denom: 16 });
    debug_assert_eq!(cells.len(), 29);
    cells
}

/// `n = floor(2^23 / k)` rounded down to a multiple of 512 -- v18's exact long-run page value
/// count formula, reused unchanged for a paper-comparable per-cell packed working set.
fn long_run_n_values(k: u8) -> usize {
    let raw = (1usize << 23) / k as usize;
    raw - raw % 512
}

// -----------------------------------------------------------------------------------------
// Fixture.
// -----------------------------------------------------------------------------------------

struct Page {
    /// Bit-packed run(s) (header+payload, one run for every shape except `WriterReal`, which
    /// packs many `WRITER_REAL_RUN_LEN`-value runs back-to-back) plus 8 zero pad bytes.
    buffer: Bytes,
    n_values: usize,
    mask_bytes: Vec<u8>,
    mask_words: Vec<u64>,
}

fn build_page(k: u8, n_values: usize, shape: Shape, survival: f64, seed: u64) -> Page {
    debug_assert_ne!(shape, Shape::WriterReal, "writer_real pages use build_writer_real_page");
    let mut rng = kernel::Xorshift64Star::new(seed);
    let values = kernel::generate_bitpacked_values(n_values, k, &mut rng);
    let mut buffer = Vec::new();
    kernel::write_bit_packed_run(&mut buffer, &values, k);
    buffer.extend_from_slice(&[0u8; 8]);

    let mask_words = match shape {
        Shape::Random => kernel::generate_random_mask(n_values, survival, &mut rng),
        Shape::Clustered => {
            kernel::generate_clustered_mask(n_values, survival, CLUSTERED_TARGET_MEAN_RUN, &mut rng)
        }
        Shape::Dense => kernel::generate_dense_mask(n_values),
        Shape::WriterReal => unreachable!(),
    };
    let mask_bytes = kernel::words_to_packed_bytes(&mask_words);
    Page { buffer: Bytes::from(buffer), n_values, mask_bytes, mask_words }
}

fn build_writer_real_page(k: u8, survival: f64, seed: u64) -> Page {
    // Same total payload as the k=8 long-run twin: floor-divide its value count into as many
    // whole WRITER_REAL_RUN_LEN-value runs as fit (1_048_576 / 504 = 2080 runs, 1_048_320
    // values, ~99.98% of the twin -- v18's own noted approximation, reused unchanged).
    let twin_n_values = long_run_n_values(8);
    let num_runs = twin_n_values / WRITER_REAL_RUN_LEN;
    let n_values = num_runs * WRITER_REAL_RUN_LEN;

    let mut rng = kernel::Xorshift64Star::new(seed);
    let mut buffer = Vec::new();
    for _ in 0..num_runs {
        let values = kernel::generate_bitpacked_values(WRITER_REAL_RUN_LEN, k, &mut rng);
        kernel::write_bit_packed_run(&mut buffer, &values, k);
    }
    buffer.extend_from_slice(&[0u8; 8]);

    let mask_words = kernel::generate_random_mask(n_values, survival, &mut rng);
    let mask_bytes = kernel::words_to_packed_bytes(&mask_words);
    Page { buffer: Bytes::from(buffer), n_values, mask_bytes, mask_words }
}

// -----------------------------------------------------------------------------------------
// Per-arm runners: single call, consuming the whole page's selection in one shot.
// -----------------------------------------------------------------------------------------

fn run_cursor(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("cursor: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, page.n_values)
        .expect("cursor: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_cursor(dict, out, selection)
        .expect("cursor: get_batch_with_dict_selected_cursor failed");
    assert_eq!(consumed, page.n_values, "cursor: RleDecoder did not consume the whole page");
    written
}

fn run_direct_gather(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("direct_gather: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, page.n_values)
        .expect("direct_gather: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_direct_gather(dict, out, selection)
        .expect("direct_gather: get_batch_with_dict_selected_direct_gather failed");
    assert_eq!(consumed, page.n_values, "direct_gather: RleDecoder did not consume the whole page");
    written
}

fn run_direct_gather_checked(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("direct_gather_checked: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, page.n_values)
        .expect("direct_gather_checked: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_direct_gather_checked(dict, out, selection)
        .expect("direct_gather_checked: get_batch_with_dict_selected_direct_gather_checked failed");
    assert_eq!(consumed, page.n_values, "direct_gather_checked: RleDecoder did not consume the whole page");
    written
}

fn run_decode_all_indices_compact(
    decoder: &mut RleDecoder,
    page: &Page,
    dict: &[i64],
    idx_buf: &mut [i32; RLE_CHUNK],
    out: &mut Vec<i64>,
) {
    decoder.set_data(page.buffer.clone()).expect("decode_all_indices_compact: set_data failed");
    let mut processed = 0usize;
    while processed < page.n_values {
        let chunk_len = (page.n_values - processed).min(RLE_CHUNK);
        let got = decoder
            .get_batch::<i32>(&mut idx_buf[..chunk_len])
            .expect("decode_all_indices_compact: get_batch failed");
        assert_eq!(got, chunk_len, "decode_all_indices_compact: fewer values than the page promised");

        let word_idx0 = processed / 64;
        let words_in_chunk = chunk_len.div_ceil(64);
        for wi in 0..words_in_chunk {
            let mut word = page.mask_words[word_idx0 + wi];
            let base = wi * 64;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                word &= word - 1;
                let local = base + bit;
                if local < chunk_len {
                    out.push(dict[idx_buf[local] as usize]);
                }
            }
        }
        processed += chunk_len;
    }
}

fn run_materialize_then_filter(
    decoder: &mut RleDecoder,
    page: &Page,
    dict: &[i64],
    val_buf: &mut [i64; RLE_CHUNK],
    out: &mut Vec<i64>,
) {
    decoder.set_data(page.buffer.clone()).expect("materialize_then_filter: set_data failed");
    let mut processed = 0usize;
    while processed < page.n_values {
        let chunk_len = (page.n_values - processed).min(RLE_CHUNK);
        let got = decoder
            .get_batch_with_dict::<i64>(dict, &mut val_buf[..chunk_len], chunk_len)
            .expect("materialize_then_filter: get_batch_with_dict failed");
        assert_eq!(got, chunk_len, "materialize_then_filter: fewer values than the page promised");

        let word_idx0 = processed / 64;
        let words_in_chunk = chunk_len.div_ceil(64);
        for wi in 0..words_in_chunk {
            let mut word = page.mask_words[word_idx0 + wi];
            let base = wi * 64;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                word &= word - 1;
                let local = base + bit;
                if local < chunk_len {
                    out.push(val_buf[local]);
                }
            }
        }
        processed += chunk_len;
    }
}

// -----------------------------------------------------------------------------------------
// Per-arm runners: multi-call incremental consumption (the v23-specific addition).
// -----------------------------------------------------------------------------------------

fn run_multi_call_cursor(
    decoder: &mut RleDecoder,
    page: &Page,
    dict: &[i64],
    splits: &[usize; 3],
    out: &mut [i64],
) -> usize {
    decoder.set_data(page.buffer.clone()).expect("multi-call cursor: set_data failed");
    let mut chunk_start = 0usize;
    let mut written = 0usize;
    for &chunk_len in splits {
        let selection = PackedSelection::new(&page.mask_bytes, chunk_start, chunk_len)
            .expect("multi-call cursor: PackedSelection::new failed");
        let (consumed, w) = decoder
            .get_batch_with_dict_selected_cursor(dict, &mut out[written..], selection)
            .expect("multi-call cursor: get_batch_with_dict_selected_cursor failed");
        assert_eq!(consumed, chunk_len, "multi-call cursor: did not consume the whole chunk");
        written += w;
        chunk_start += chunk_len;
    }
    written
}

fn run_multi_call_direct_gather(
    decoder: &mut RleDecoder,
    page: &Page,
    dict: &[i64],
    splits: &[usize; 3],
    out: &mut [i64],
) -> usize {
    decoder.set_data(page.buffer.clone()).expect("multi-call direct_gather: set_data failed");
    let mut chunk_start = 0usize;
    let mut written = 0usize;
    for &chunk_len in splits {
        let selection = PackedSelection::new(&page.mask_bytes, chunk_start, chunk_len)
            .expect("multi-call direct_gather: PackedSelection::new failed");
        let (consumed, w) = decoder
            .get_batch_with_dict_selected_direct_gather(dict, &mut out[written..], selection)
            .expect("multi-call direct_gather: get_batch_with_dict_selected_direct_gather failed");
        assert_eq!(consumed, chunk_len, "multi-call direct_gather: did not consume the whole chunk");
        written += w;
        chunk_start += chunk_len;
    }
    written
}

fn run_multi_call_direct_gather_checked(
    decoder: &mut RleDecoder,
    page: &Page,
    dict: &[i64],
    splits: &[usize; 3],
    out: &mut [i64],
) -> usize {
    decoder.set_data(page.buffer.clone()).expect("multi-call direct_gather_checked: set_data failed");
    let mut chunk_start = 0usize;
    let mut written = 0usize;
    for &chunk_len in splits {
        let selection = PackedSelection::new(&page.mask_bytes, chunk_start, chunk_len)
            .expect("multi-call direct_gather_checked: PackedSelection::new failed");
        let (consumed, w) = decoder
            .get_batch_with_dict_selected_direct_gather_checked(dict, &mut out[written..], selection)
            .expect("multi-call direct_gather_checked: get_batch_with_dict_selected_direct_gather_checked failed");
        assert_eq!(consumed, chunk_len, "multi-call direct_gather_checked: did not consume the whole chunk");
        written += w;
        chunk_start += chunk_len;
    }
    written
}

// -----------------------------------------------------------------------------------------
// Cross-arm digest verification (untimed, at setup).
// -----------------------------------------------------------------------------------------

fn verify_digests(bench_name: &str, cell_desc: &str, page_idx: usize, named: &[(&str, u64)]) {
    let reference = named[0].1;
    let mut diverged: Vec<String> = Vec::new();
    for &(name, d) in named.iter().skip(1) {
        if d != reference {
            diverged.push(format!("{name} (digest {d:#018x})"));
        }
    }
    if !diverged.is_empty() {
        panic!(
            "{bench_name} cell {cell_desc} page={page_idx}: cross-arm digest mismatch. Reference \
             ({}) digest = {reference:#018x}. Diverged: [{}]",
            named[0].0,
            diverged.join(", ")
        );
    }
}

// -----------------------------------------------------------------------------------------
// Per-cell benchmarking.
// -----------------------------------------------------------------------------------------

fn bench_cell(c: &mut Criterion, cell: &CellSpec, cell_index: usize) {
    let dict_seed = TOP_SEED.wrapping_add(cell_index as u64);
    let dict = kernel::generate_dict(cell.k as u32, dict_seed);
    assert_eq!(dict.len(), 1usize << cell.k, "fixture invariant: dict.len() == 1<<k");

    let cell_base_seed = kernel::derive_seed(TOP_SEED, cell_index as u64);
    let pages: Vec<Page> = (0..PAGES_PER_CELL)
        .map(|page_idx| {
            let seed = kernel::derive_seed(cell_base_seed, page_idx as u64);
            if cell.shape == Shape::WriterReal {
                build_writer_real_page(cell.k, cell.survival, seed)
            } else {
                let n_values = long_run_n_values(cell.k);
                build_page(cell.k, n_values, cell.shape, cell.survival, seed)
            }
        })
        .collect();

    let max_selected = pages.iter().map(|p| kernel::popcount_words(&p.mask_words)).max().unwrap_or(0);

    let k = cell.k;
    let shape = cell.shape.label();
    let denom = cell.denom;
    let cell_desc = format!("k{k}/{shape}/s{denom}");

    // --- untimed cross-arm correctness check (setup only, never inside a timed closure) ---
    {
        let mut decoder_cursor = RleDecoder::new(k);
        let mut decoder_dg = RleDecoder::new(k);
        let mut decoder_dgc = RleDecoder::new(k);
        let mut decoder_c = RleDecoder::new(k);
        let mut decoder_d = RleDecoder::new(k);
        let mut out_cursor = vec![0i64; max_selected];
        let mut out_dg = vec![0i64; max_selected];
        let mut out_dgc = vec![0i64; max_selected];
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut val_buf = [0i64; RLE_CHUNK];
        let mut out_c = Vec::with_capacity(max_selected);
        let mut out_d = Vec::with_capacity(max_selected);

        for (page_idx, page) in pages.iter().enumerate() {
            let written_cursor = run_cursor(&mut decoder_cursor, page, &dict, &mut out_cursor);
            let written_dg = run_direct_gather(&mut decoder_dg, page, &dict, &mut out_dg);
            let written_dgc = run_direct_gather_checked(&mut decoder_dgc, page, &dict, &mut out_dgc);
            out_c.clear();
            run_decode_all_indices_compact(&mut decoder_c, page, &dict, &mut idx_buf, &mut out_c);
            out_d.clear();
            run_materialize_then_filter(&mut decoder_d, page, &dict, &mut val_buf, &mut out_d);

            verify_digests(
                "bitpacked_direct_gather_grid",
                &cell_desc,
                page_idx,
                &[
                    ("cursor", kernel::fnv1a64(&out_cursor[..written_cursor])),
                    ("direct_gather", kernel::fnv1a64(&out_dg[..written_dg])),
                    ("direct_gather_checked", kernel::fnv1a64(&out_dgc[..written_dgc])),
                    ("decode_all_indices_compact", kernel::fnv1a64(&out_c)),
                    ("materialize_then_filter", kernel::fnv1a64(&out_d)),
                ],
            );
        }
    }

    // --- timed loop: one Criterion BenchmarkGroup for this cell ---
    let mut group = c.benchmark_group("bitpacked_direct_gather_grid");
    group.sample_size(12);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs_f64(2.5));

    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_cursor(&mut decoder, &pages[0], &dict, &mut out);
        group.bench_function(format!("cursor/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_cursor(&mut decoder, page, &dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_direct_gather(&mut decoder, &pages[0], &dict, &mut out);
        group.bench_function(format!("direct_gather/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_direct_gather(&mut decoder, page, &dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_direct_gather_checked(&mut decoder, &pages[0], &dict, &mut out);
        group.bench_function(format!("direct_gather_checked/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_direct_gather_checked(&mut decoder, page, &dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut out = Vec::with_capacity(max_selected);
        run_decode_all_indices_compact(&mut decoder, &pages[0], &dict, &mut idx_buf, &mut out);
        group.bench_function(format!("decode_all_indices_compact/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                out.clear();
                run_decode_all_indices_compact(&mut decoder, page, &dict, &mut idx_buf, &mut out);
                hint::black_box(out.as_slice());
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut val_buf = [0i64; RLE_CHUNK];
        let mut out = Vec::with_capacity(max_selected);
        run_materialize_then_filter(&mut decoder, &pages[0], &dict, &mut val_buf, &mut out);
        group.bench_function(format!("materialize_then_filter/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                out.clear();
                run_materialize_then_filter(&mut decoder, page, &dict, &mut val_buf, &mut out);
                hint::black_box(out.as_slice());
            });
        });
    }
    group.finish();

    // `pages`, `dict`, and everything above are dropped here, before the next cell's fixtures
    // are generated (peak memory bounded to one cell's working set at a time).
}

/// The v23-specific multi-call timing addition: `k=8` crossed with the grid's own 3 survival
/// points, 3 cells, `cursor`/`direct_gather`/`direct_gather_checked` only.
fn bench_multi_call_cell(c: &mut Criterion, survival: f64, denom: u32, cell_index: usize) {
    const K: u8 = 8;
    let dict_seed = TOP_SEED.wrapping_add(cell_index as u64);
    let dict = kernel::generate_dict(K as u32, dict_seed);
    assert_eq!(dict.len(), 1usize << K, "fixture invariant: dict.len() == 1<<k");

    let cell_base_seed = kernel::derive_seed(TOP_SEED, cell_index as u64);
    let n_values = long_run_n_values(K);
    let splits = multi_call_splits(n_values);
    let pages: Vec<Page> = (0..PAGES_PER_CELL)
        .map(|page_idx| {
            let seed = kernel::derive_seed(cell_base_seed, page_idx as u64);
            build_page(K, n_values, Shape::Random, survival, seed)
        })
        .collect();

    let max_selected = pages.iter().map(|p| kernel::popcount_words(&p.mask_words)).max().unwrap_or(0);
    let cell_desc = format!("k{K}/multicall/s{denom}");

    // --- untimed cross-arm correctness check ---
    {
        let mut decoder_cursor = RleDecoder::new(K);
        let mut decoder_dg = RleDecoder::new(K);
        let mut decoder_dgc = RleDecoder::new(K);
        let mut out_cursor = vec![0i64; max_selected];
        let mut out_dg = vec![0i64; max_selected];
        let mut out_dgc = vec![0i64; max_selected];

        for (page_idx, page) in pages.iter().enumerate() {
            let written_cursor = run_multi_call_cursor(&mut decoder_cursor, page, &dict, &splits, &mut out_cursor);
            let written_dg = run_multi_call_direct_gather(&mut decoder_dg, page, &dict, &splits, &mut out_dg);
            let written_dgc =
                run_multi_call_direct_gather_checked(&mut decoder_dgc, page, &dict, &splits, &mut out_dgc);

            verify_digests(
                "bitpacked_direct_gather_grid",
                &cell_desc,
                page_idx,
                &[
                    ("cursor", kernel::fnv1a64(&out_cursor[..written_cursor])),
                    ("direct_gather", kernel::fnv1a64(&out_dg[..written_dg])),
                    ("direct_gather_checked", kernel::fnv1a64(&out_dgc[..written_dgc])),
                ],
            );
        }
    }

    // --- timed loop ---
    let mut group = c.benchmark_group("bitpacked_direct_gather_grid_multi_call");
    group.sample_size(12);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs_f64(2.5));

    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(K);
        let mut out = vec![0i64; max_selected];
        let _ = run_multi_call_cursor(&mut decoder, &pages[0], &dict, &splits, &mut out);
        group.bench_function(format!("cursor/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_multi_call_cursor(&mut decoder, page, &dict, &splits, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(K);
        let mut out = vec![0i64; max_selected];
        let _ = run_multi_call_direct_gather(&mut decoder, &pages[0], &dict, &splits, &mut out);
        group.bench_function(format!("direct_gather/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_multi_call_direct_gather(&mut decoder, page, &dict, &splits, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(K);
        let mut out = vec![0i64; max_selected];
        let _ = run_multi_call_direct_gather_checked(&mut decoder, &pages[0], &dict, &splits, &mut out);
        group.bench_function(format!("direct_gather_checked/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_multi_call_direct_gather_checked(&mut decoder, page, &dict, &splits, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    group.finish();
}

fn run_full_matrix(c: &mut Criterion) {
    let cells = build_cells();
    for (cell_index, cell) in cells.iter().enumerate() {
        bench_cell(c, cell, cell_index);
    }
    const SURVIVALS: [(f64, u32); 3] = [(1.0 / 64.0, 64), (1.0 / 16.0, 16), (1.0 / 4.0, 4)];
    for (i, &(survival, denom)) in SURVIVALS.iter().enumerate() {
        bench_multi_call_cell(c, survival, denom, cells.len() + i);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    run_full_matrix(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
