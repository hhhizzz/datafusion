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

//! R1 (isolation on pure-RLE fixtures) Criterion bench for experiment
//! `arrow-rle-selected-fill-v21`. See
//! `codex/experiments/arrow-rle-selected-fill-v21.md` -- in particular its "Census-informed
//! R1/R2 concretization" section -- for the frozen design this file implements.
//!
//! Four arms decode the same pure-RLE dictionary-index page and are timed separately:
//!
//! - `selected_fill` (arm A): `RleDecoder::get_batch_with_dict_selected` (popcount + `fill`).
//! - `selected_fill_fast` (arm A'): `RleDecoder::get_batch_with_dict_selected_fast`, identical
//!   except for `PackedSelection`'s internal word loader (`bits_u64` vs. `bits_u64_fast`).
//! - `decode_all_indices_compact` (arm C): stock `RleDecoder::get_batch::<i32>`, chunked, then
//!   a selection-bitmap walk gathering `dict[idx]`.
//! - `materialize_then_filter` (arm D): stock `RleDecoder::get_batch_with_dict::<i64>`,
//!   chunked, fully materializing every value, then a selection-bitmap walk copying survivors.
//!
//! `PackedSelection`/`get_batch_with_dict_selected`/`get_batch_with_dict_selected_fast` are a
//! small additive patch on top of otherwise-unmodified upstream Arrow (pinned commit
//! `ed92960c8a85eda657fce3525c905616ccc5a983`), reachable only under the `parquet` crate's
//! `experimental` Cargo feature (same pattern `benchmarks/paper-select-fourarm` already uses
//! for `RleDecoder`). This bench is R1's primary correctness oracle for that brand-new,
//! never-locally-compiled entry point: every page's 4 arms are cross-checked digest-for-digest
//! during untimed setup (see `verify_digests`) before any timed measurement of it occurs.
//!
//! ## Cell grid and count
//!
//! 6 run lengths (`L`) x 4 bit widths (`k`) x (3 random-shape survivals + 1 dense control) + 1
//! single clustered guard cell = 6*4*4 + 1 = **97 cells**, each yielding 4 timed
//! `bench_function`s (388 total). This is a deliberate recomputation from the frozen grid's
//! *definition* (run length is explicitly "the primary axis", swept across all 6 values and
//! crossed with every other axis; the frozen text's own illustrative arithmetic sketch, "4
//! k-values x 4 + 1 = 17", omits the run-length axis entirely and undercounts by 80 cells --
//! the frozen contract itself invites recomputing this rather than trusting that sketch). Only
//! the clustered shape is a single pinned-point guard cell (`L=64, k=8, s=1/16`, not crossed
//! with the rest of the grid), matching how a "guard" cell was used in the structurally
//! analogous `arrow-paper-select-fourarm-v18` bench (its `writer_real` shape was one pinned
//! cell, while all 4 of its `k` values were fully crossed with its primary shapes).
//!
//! ## Clustered mask target run length
//!
//! The frozen contract's concretization section specifies the clustered guard cell directly as
//! "2-state Markov chain, mean selected-run length ~= 64"; `CLUSTERED_TARGET_MEAN_RUN` below is
//! that already-frozen number, not a free choice made here.
//!
//! ## No BMI2/CPU-feature gating
//!
//! Unlike `paper-select-fourarm`, this experiment is explicitly out of scope for BMI2/PEXT (see
//! the experiment doc's "Out of scope" section: "pure popcount + fill; that is the point"), so
//! `criterion_benchmark` below runs unconditionally on every target, no feature detection.
//!
//! ## `cargo bench`'s injected `--bench` argv
//!
//! This harness never reads `std::env::args()` itself (only Criterion's own `criterion_main!`
//! macro does, and it already handles `cargo bench`'s unconditionally-injected bare `--bench`
//! correctly -- that is one of Criterion's own supported flags), so the argv-parsing caveat
//! that bit a prior, non-Criterion `harness = false` program in this workspace
//! (`benchmarks/v19-static-census`, which reads `std::env::args()` directly for a dataset-root
//! override) does not apply here.

use std::hint;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, criterion_group, criterion_main};
use parquet::encodings::rle::{PackedSelection, RleDecoder};

// kernel.rs is shared across this crate's bench binaries (rle_fill/rle_fill_r15
// /bitpacked_direct_gather); not every helper it exports is used by every binary.
#[allow(dead_code)]
#[path = "rle_fill/kernel.rs"]
mod kernel;

/// Single top-level seed constant the whole fixture matrix is reproducible from ("COFFEE" +
/// this experiment's freeze date, 2026-08-07 -- an arbitrary but fixed mnemonic, unrelated to
/// any other experiment's seed constant).
const TOP_SEED: u64 = 0xC0FF_EE15_2026_0807;

/// Sentinel `derive_seed` index for a cell's dictionary seed, guaranteed distinct from every
/// real page index (`0..PAGES_PER_CELL`).
const DICT_SEED_INDEX: u64 = u64::MAX;

const PAGES_PER_CELL: usize = 32;

/// Frozen total page value count (2^17), fixed uniformly across every cell so only run length
/// varies per-run granularity while total per-page work stays constant.
const N_TOTAL: usize = 131_072;

/// Production RLE batch granularity used by arms C and D: matches
/// `RLE_DECODER_INDEX_BUFFER_SIZE`, `RleDecoder`'s own internal scratch-buffer size for the
/// equivalent stock decode calls (`parquet/src/encodings/rle.rs:339` at the pinned commit) --
/// not an arbitrary choice.
const RLE_CHUNK: usize = 1024;

/// Target mean selected-run length for the clustered guard cell's Markov chain, frozen by the
/// experiment contract's concretization section ("mean selected-run length ~= 64").
const CLUSTERED_TARGET_MEAN_RUN: f64 = 64.0;

const RUN_LENGTHS: [usize; 6] = [8, 16, 64, 512, 4096, 65536];
const BIT_WIDTHS: [u8; 4] = [2, 8, 12, 16];
const SURVIVALS: [(f64, u32); 3] = [(1.0 / 64.0, 64), (1.0 / 16.0, 16), (1.0 / 4.0, 4)];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    Random,
    Clustered,
    Dense,
}

impl Shape {
    fn label(self) -> &'static str {
        match self {
            Shape::Random => "random",
            Shape::Clustered => "clustered",
            Shape::Dense => "dense",
        }
    }
}

struct CellSpec {
    l: usize,
    k: u8,
    shape: Shape,
    survival: f64,
    denom: u32,
}

/// See the module doc comment's "Cell grid and count" section for the recomputed total (97)
/// and why it differs from the frozen contract's own illustrative arithmetic.
fn build_cells() -> Vec<CellSpec> {
    let mut cells = Vec::with_capacity(97);
    for &l in &RUN_LENGTHS {
        for &k in &BIT_WIDTHS {
            for &(survival, denom) in &SURVIVALS {
                cells.push(CellSpec { l, k, shape: Shape::Random, survival, denom });
            }
            cells.push(CellSpec { l, k, shape: Shape::Dense, survival: 1.0, denom: 1 });
        }
    }
    cells.push(CellSpec {
        l: 64,
        k: 8,
        shape: Shape::Clustered,
        survival: 1.0 / 16.0,
        denom: 16,
    });
    debug_assert_eq!(cells.len(), 97);
    cells
}

// -----------------------------------------------------------------------------------------
// Fixture: one page is one pure-RLE-encoded dictionary-index stream plus its page-level
// selection mask, in two equivalent representations.
// -----------------------------------------------------------------------------------------

struct Page {
    /// Encoded pure-RLE dictionary-index stream for this page (no leading bit-width byte;
    /// `RleDecoder` is constructed directly with a known `bit_width`, per `RleDecoder::new`).
    buffer: Bytes,
    /// Page-level selection mask spanning all `N_TOTAL` logical positions, byte-packed
    /// LSB-first to match `PackedSelection`'s exact bit convention -- consumed by arms A/A'.
    mask_bytes: Vec<u8>,
    /// The same mask as `mask_bytes`, as `u64` words (LSB-first within word), for the fast
    /// trailing-zeros bit-walk arms C/D use and for the untimed digest check. Never passed to
    /// `PackedSelection`, which only ever sees `mask_bytes`.
    mask_words: Vec<u64>,
}

/// Builds one page: `N_TOTAL / l` consecutive pure-RLE runs (independently random dictionary
/// indices), followed by a page-level selection mask generated independently of the run
/// boundaries (a "random" or "clustered" mask does not need to -- and by construction does
/// not -- align to run boundaries at all).
fn build_page(l: usize, k: u8, shape: Shape, survival: f64, seed: u64) -> Page {
    let mut rng = kernel::Xorshift64Star::new(seed);
    let buffer = kernel::build_rle_page(N_TOTAL, l, k, &mut rng);
    let mask_words = match shape {
        Shape::Random => kernel::generate_random_mask(N_TOTAL, survival, &mut rng),
        Shape::Clustered => {
            kernel::generate_clustered_mask(N_TOTAL, survival, CLUSTERED_TARGET_MEAN_RUN, &mut rng)
        }
        Shape::Dense => kernel::generate_dense_mask(N_TOTAL),
    };
    let mask_bytes = kernel::words_to_packed_bytes(&mask_words);
    Page { buffer: Bytes::from(buffer), mask_bytes, mask_words }
}

// -----------------------------------------------------------------------------------------
// Per-arm runners. Arms A/A' fill a fixed-size output slice in place (the real API's own
// shape) and return how many values were written; arms C/D append survivors to a growable
// `Vec` the caller clears first (matching this program's established push-and-clear idiom for
// chunked decode-then-gather arms).
// -----------------------------------------------------------------------------------------

/// Arm A (`selected_fill`): `RleDecoder::get_batch_with_dict_selected`. Not chunked by this
/// harness -- the method processes the whole page's selection in one call, looping over the
/// page's RLE runs internally.
fn run_arm_a(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("arm A: RleDecoder::set_data failed");
    let selection =
        PackedSelection::new(&page.mask_bytes, 0, N_TOTAL).expect("arm A: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected(dict, out, selection)
        .expect("arm A: get_batch_with_dict_selected failed");
    assert_eq!(consumed, N_TOTAL, "arm A: RleDecoder did not consume the whole page");
    written
}

/// Arm A' (`selected_fill_fast`): identical shape to [`run_arm_a`], calling
/// `get_batch_with_dict_selected_fast` instead. Differs only in `PackedSelection`'s internal
/// word loader (`bits_u64` vs. `bits_u64_fast`); see the Arrow-side diff's doc comments.
fn run_arm_a_fast(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("arm A': RleDecoder::set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, N_TOTAL)
        .expect("arm A': PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_fast(dict, out, selection)
        .expect("arm A': get_batch_with_dict_selected_fast failed");
    assert_eq!(consumed, N_TOTAL, "arm A': RleDecoder did not consume the whole page");
    written
}

/// Arm C (`decode_all_indices_compact`): stock `RleDecoder::get_batch::<i32>` in `RLE_CHUNK`
/// (1024)-value chunks -- the decoder's own internal scratch-buffer granularity for the
/// equivalent stock call, not an arbitrary choice (see `RLE_CHUNK`'s doc comment) -- then a
/// per-chunk selection-word walk gathering `dict[idx]` for set bits.
fn run_arm_c(
    decoder: &mut RleDecoder,
    page: &Page,
    dict: &[i64],
    idx_buf: &mut [i32; RLE_CHUNK],
    out: &mut Vec<i64>,
) {
    decoder.set_data(page.buffer.clone()).expect("arm C: RleDecoder::set_data failed");
    let mut processed = 0usize;
    while processed < N_TOTAL {
        let chunk_len = (N_TOTAL - processed).min(RLE_CHUNK);
        let got = decoder
            .get_batch::<i32>(&mut idx_buf[..chunk_len])
            .expect("arm C: RleDecoder::get_batch failed");
        assert_eq!(got, chunk_len, "arm C: RleDecoder produced fewer values than the page promised");

        let word_idx0 = processed / 64;
        let words_in_chunk = chunk_len.div_ceil(64);
        for wi in 0..words_in_chunk {
            let mut word = page.mask_words[word_idx0 + wi];
            let base = wi * 64;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                word &= word - 1; // clear lowest set bit
                let local = base + bit;
                if local < chunk_len {
                    out.push(dict[idx_buf[local] as usize]);
                }
            }
        }
        processed += chunk_len;
    }
}

/// Arm D (`materialize_then_filter`): stock `RleDecoder::get_batch_with_dict::<i64>` in
/// `RLE_CHUNK`-value chunks, fully materializing *every* chunk regardless of its selection
/// (never skipped, not even for a chunk with zero selected bits -- this is what makes it a
/// faithful "decode everything, then filter" baseline), then the same selection-word walk
/// copying survivors.
fn run_arm_d(
    decoder: &mut RleDecoder,
    page: &Page,
    dict: &[i64],
    val_buf: &mut [i64; RLE_CHUNK],
    out: &mut Vec<i64>,
) {
    decoder.set_data(page.buffer.clone()).expect("arm D: RleDecoder::set_data failed");
    let mut processed = 0usize;
    while processed < N_TOTAL {
        let chunk_len = (N_TOTAL - processed).min(RLE_CHUNK);
        let got = decoder
            .get_batch_with_dict::<i64>(dict, &mut val_buf[..chunk_len], chunk_len)
            .expect("arm D: RleDecoder::get_batch_with_dict failed");
        assert_eq!(got, chunk_len, "arm D: RleDecoder produced fewer values than the page promised");

        let word_idx0 = processed / 64;
        let words_in_chunk = chunk_len.div_ceil(64);
        for wi in 0..words_in_chunk {
            let mut word = page.mask_words[word_idx0 + wi];
            let base = wi * 64;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                word &= word - 1; // clear lowest set bit
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
// Cross-arm digest verification (untimed, at setup).
// -----------------------------------------------------------------------------------------

/// Compares the 4 named arms' digests (`named[0]`, `selected_fill`, is the reference) and
/// panics naming exactly which arm(s) diverged, the cell (`cell_desc`), and the page index, if
/// any differ. This is R1's primary correctness oracle for the brand-new, never-locally-compiled
/// `get_batch_with_dict_selected`/`_fast`/`PackedSelection` entry points.
fn verify_digests(cell_desc: &str, page_idx: usize, named: [(&str, u64); 4]) {
    let reference = named[0].1;
    let mut diverged: Vec<String> = Vec::new();
    for &(name, d) in named.iter().skip(1) {
        if d != reference {
            diverged.push(format!("{name} (digest {d:#018x})"));
        }
    }
    if !diverged.is_empty() {
        panic!(
            "rle_fill cell {cell_desc} page={page_idx}: cross-arm digest mismatch. Reference \
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
    let cell_base_seed = kernel::derive_seed(TOP_SEED, cell_index as u64);
    let dict_seed = kernel::derive_seed(cell_base_seed, DICT_SEED_INDEX);
    let dict = kernel::generate_dict(cell.k as u32, dict_seed);
    assert_eq!(dict.len(), 1usize << cell.k, "fixture invariant: dict.len() == 1<<k");

    let pages: Vec<Page> = (0..PAGES_PER_CELL)
        .map(|page_idx| {
            let seed = kernel::derive_seed(cell_base_seed, page_idx as u64);
            build_page(cell.l, cell.k, cell.shape, cell.survival, seed)
        })
        .collect();

    let max_selected = pages.iter().map(|p| kernel::popcount_words(&p.mask_words)).max().unwrap_or(0);

    let l = cell.l;
    let k = cell.k;
    let shape = cell.shape.label();
    let denom = cell.denom;
    let cell_desc = format!("L={l} k={k} shape={shape} s=1/{denom}");

    // --- untimed cross-arm correctness check (setup only, never inside a timed closure) ---
    {
        let mut decoder_a = RleDecoder::new(k);
        let mut decoder_a_fast = RleDecoder::new(k);
        let mut decoder_c = RleDecoder::new(k);
        let mut decoder_d = RleDecoder::new(k);
        let mut out_a = vec![0i64; max_selected];
        let mut out_a_fast = vec![0i64; max_selected];
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut val_buf = [0i64; RLE_CHUNK];
        let mut out_c = Vec::with_capacity(max_selected);
        let mut out_d = Vec::with_capacity(max_selected);

        for (page_idx, page) in pages.iter().enumerate() {
            let written_a = run_arm_a(&mut decoder_a, page, &dict, &mut out_a);
            let written_a_fast = run_arm_a_fast(&mut decoder_a_fast, page, &dict, &mut out_a_fast);
            out_c.clear();
            run_arm_c(&mut decoder_c, page, &dict, &mut idx_buf, &mut out_c);
            out_d.clear();
            run_arm_d(&mut decoder_d, page, &dict, &mut val_buf, &mut out_d);

            verify_digests(
                &cell_desc,
                page_idx,
                [
                    ("selected_fill", kernel::fnv1a64(&out_a[..written_a])),
                    ("selected_fill_fast", kernel::fnv1a64(&out_a_fast[..written_a_fast])),
                    ("decode_all_indices_compact", kernel::fnv1a64(&out_c)),
                    ("materialize_then_filter", kernel::fnv1a64(&out_d)),
                ],
            );
        }
    }

    // --- timed loop: one Criterion BenchmarkGroup for this cell ---
    let mut group = c.benchmark_group("rle_fill");
    group.sample_size(12);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs_f64(2.5));

    {
        let mut cursor = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        // Prime the decoder's lazily-allocated `index_buf` scratch (touched unconditionally at
        // the top of `get_batch_with_dict_selected`'s main loop -- and by the decode-all
        // short-circuit it takes for dense cells -- even though pure-RLE fixtures never take
        // the bit-packed branch that actually reads it) so the closure below never risks a
        // first-touch heap allocation during a measured sample.
        let _ = run_arm_a(&mut decoder, &pages[0], &dict, &mut out);
        group.bench_function(format!("selected_fill/L{l}/k{k}/{shape}/s{denom}"), |b| {
            b.iter(|| {
                let page = &pages[cursor];
                cursor = (cursor + 1) % pages.len();
                let written = run_arm_a(&mut decoder, page, &dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_arm_a_fast(&mut decoder, &pages[0], &dict, &mut out);
        group.bench_function(format!("selected_fill_fast/L{l}/k{k}/{shape}/s{denom}"), |b| {
            b.iter(|| {
                let page = &pages[cursor];
                cursor = (cursor + 1) % pages.len();
                let written = run_arm_a_fast(&mut decoder, page, &dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut out = Vec::with_capacity(max_selected);
        // Prime any lazily-allocated decoder scratch state (untimed); see the comment on the
        // arm-A prime call above. `get_batch::<i32>` itself touches no such state, but priming
        // defensively matches this program's established pattern. `out` is cleared on the
        // first real (timed) iteration regardless.
        run_arm_c(&mut decoder, &pages[0], &dict, &mut idx_buf, &mut out);
        group.bench_function(format!("decode_all_indices_compact/L{l}/k{k}/{shape}/s{denom}"), |b| {
            b.iter(|| {
                let page = &pages[cursor];
                cursor = (cursor + 1) % pages.len();
                out.clear();
                run_arm_c(&mut decoder, page, &dict, &mut idx_buf, &mut out);
                hint::black_box(out.as_slice());
            });
        });
    }
    {
        let mut cursor = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut val_buf = [0i64; RLE_CHUNK];
        let mut out = Vec::with_capacity(max_selected);
        // Prime the decoder's lazily-allocated `index_buf` scratch (untimed); see the comment
        // on the arm-A prime call above.
        run_arm_d(&mut decoder, &pages[0], &dict, &mut val_buf, &mut out);
        group.bench_function(format!("materialize_then_filter/L{l}/k{k}/{shape}/s{denom}"), |b| {
            b.iter(|| {
                let page = &pages[cursor];
                cursor = (cursor + 1) % pages.len();
                out.clear();
                run_arm_d(&mut decoder, page, &dict, &mut val_buf, &mut out);
                hint::black_box(out.as_slice());
            });
        });
    }
    group.finish();

    // `pages`, `dict`, and everything above are dropped here, before the next cell's fixtures
    // are generated (peak memory bounded to one cell's working set at a time).
}

fn run_full_matrix(c: &mut Criterion) {
    for (cell_index, cell) in build_cells().into_iter().enumerate() {
        bench_cell(c, &cell, cell_index);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    run_full_matrix(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
