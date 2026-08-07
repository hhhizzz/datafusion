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

//! R1.5 (short-run attribution / variant elimination) Criterion bench for
//! experiment `arrow-rle-selected-fill-v21`.
//!
//! R1 found arm A (`selected_fill`, `RleDecoder::get_batch_with_dict_selected`)
//! losing ~1.8x to the baseline at the decision cell (`L=16, s=1/16`), while
//! winning up to 2.1x at long run lengths. The working hypothesis: arm A's
//! RLE branch calls `PackedSelection::selected_count_range` once per RLE
//! run, and that method is a *stateless, random-access* reader -- it
//! re-derives an absolute byte offset/bit shift and reloads a word from
//! `data` on every call, even when the previous call already loaded
//! overlapping bytes. At `L=8` a page has 16,384 runs sharing selection
//! words 8-to-a-word; arm A pays that reload cost once per run, while arm
//! C's harness-level mask walk reads `page.mask_words` directly (no
//! reload/shift machinery at all) once per 64-bit chunk -- an 8x
//! amortization mismatch at `L=8`.
//!
//! Two new arms (both added to the Arrow-side patch, not this file) test
//! that hypothesis and one adjacent one:
//!
//! - `selected_fill_cursor` (arm E): `RleDecoder::get_batch_with_dict_selected_cursor`.
//!   Replaces `selected_count_range` with a stateful `RleSelectionCursor`
//!   that only reloads when its cached word is exhausted, and skips the
//!   RLE branch's unused `index_buf` touch.
//! - `selected_fill_cursor_unchecked` (arm F):
//!   `RleDecoder::get_batch_with_dict_selected_cursor_unchecked`. Arm E
//!   plus an unchecked (`get_unchecked`) dictionary lookup in the RLE
//!   branch, isolating whether the checked-access bounds check is a
//!   further contributor beyond the cursor fix alone.
//!
//! Reduced grid (this is an attribution stage, not a full re-run of R1):
//! `L ∈ {8, 16, 64, 512}` (spans the R1-observed crossover) x `k = 8` only
//! x (3 random-shape survivals + 1 dense control) = 16 cells x 5 arms (A,
//! E, F, C, D -- arm A' is not retested here, R1 already found it a clean
//! null result). Same `N_TOTAL`, page count, seed derivation, and Criterion
//! protocol as `rle_fill.rs`; deliberately reuses the *same* `kernel.rs`
//! fixture generator unchanged, so cells are directly comparable to R1's
//! own `k=8` rows.
//!
//! Outcome this stage answers: if arm E/F close the gap at the decision
//! cell (`L=16`), the loss was implementation overhead, not the count+fill
//! mechanism, and the unconditional route B (v21) may be revivable as-is.
//! If not, the gap is intrinsic to short runs and a run-length-conditional
//! admission (not attempted here) is the next open question.

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

/// Distinct from `rle_fill.rs`'s `TOP_SEED` so this stage's fixtures are
/// independently generated, not a subset reuse of R1's own page set (a
/// fresh, independent measurement is more informative here than reusing
/// R1's exact bytes would be, and this stage's per-cell page count/seed
/// derivation already differs from R1's cell-index-based derivation).
const TOP_SEED: u64 = 0xC0FF_EE15_2026_0807_u64 ^ 0x5215;

const DICT_SEED_INDEX: u64 = u64::MAX;
const PAGES_PER_CELL: usize = 32;
const N_TOTAL: usize = 131_072;
const RLE_CHUNK: usize = 1024;

/// Fixed at `k=8`: R1 found no material `k`-dependence within the tested
/// range (2/8/12/16 all showed the same crossover shape), so this
/// attribution stage does not re-sweep it.
const BIT_WIDTH: u8 = 8;

const RUN_LENGTHS: [usize; 4] = [8, 16, 64, 512];
const SURVIVALS: [(f64, u32); 3] = [(1.0 / 64.0, 64), (1.0 / 16.0, 16), (1.0 / 4.0, 4)];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    Random,
    Dense,
}

impl Shape {
    fn label(self) -> &'static str {
        match self {
            Shape::Random => "random",
            Shape::Dense => "dense",
        }
    }
}

struct CellSpec {
    l: usize,
    shape: Shape,
    survival: f64,
    denom: u32,
}

fn build_cells() -> Vec<CellSpec> {
    let mut cells = Vec::with_capacity(16);
    for &l in &RUN_LENGTHS {
        for &(survival, denom) in &SURVIVALS {
            cells.push(CellSpec { l, shape: Shape::Random, survival, denom });
        }
        cells.push(CellSpec { l, shape: Shape::Dense, survival: 1.0, denom: 1 });
    }
    debug_assert_eq!(cells.len(), 16);
    cells
}

struct Page {
    buffer: Bytes,
    mask_bytes: Vec<u8>,
    mask_words: Vec<u64>,
}

fn build_page(l: usize, shape: Shape, survival: f64, seed: u64) -> Page {
    let mut rng = kernel::Xorshift64Star::new(seed);
    let buffer = kernel::build_rle_page(N_TOTAL, l, BIT_WIDTH, &mut rng);
    let mask_words = match shape {
        Shape::Random => kernel::generate_random_mask(N_TOTAL, survival, &mut rng),
        Shape::Dense => kernel::generate_dense_mask(N_TOTAL),
    };
    let mask_bytes = kernel::words_to_packed_bytes(&mask_words);
    Page { buffer: Bytes::from(buffer), mask_bytes, mask_words }
}

// -----------------------------------------------------------------------------------------
// Per-arm runners.
// -----------------------------------------------------------------------------------------

fn run_arm_a(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("arm A: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, N_TOTAL).expect("arm A: PackedSelection::new failed");
    let (consumed, written) =
        decoder.get_batch_with_dict_selected(dict, out, selection).expect("arm A: get_batch_with_dict_selected failed");
    assert_eq!(consumed, N_TOTAL, "arm A: RleDecoder did not consume the whole page");
    written
}

fn run_arm_e(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("arm E: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, N_TOTAL).expect("arm E: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_cursor(dict, out, selection)
        .expect("arm E: get_batch_with_dict_selected_cursor failed");
    assert_eq!(consumed, N_TOTAL, "arm E: RleDecoder did not consume the whole page");
    written
}

fn run_arm_f(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("arm F: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, N_TOTAL).expect("arm F: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_cursor_unchecked(dict, out, selection)
        .expect("arm F: get_batch_with_dict_selected_cursor_unchecked failed");
    assert_eq!(consumed, N_TOTAL, "arm F: RleDecoder did not consume the whole page");
    written
}

fn run_arm_c(decoder: &mut RleDecoder, page: &Page, dict: &[i64], idx_buf: &mut [i32; RLE_CHUNK], out: &mut Vec<i64>) {
    decoder.set_data(page.buffer.clone()).expect("arm C: set_data failed");
    let mut processed = 0usize;
    while processed < N_TOTAL {
        let chunk_len = (N_TOTAL - processed).min(RLE_CHUNK);
        let got = decoder.get_batch::<i32>(&mut idx_buf[..chunk_len]).expect("arm C: get_batch failed");
        assert_eq!(got, chunk_len, "arm C: RleDecoder produced fewer values than the page promised");

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

fn run_arm_d(decoder: &mut RleDecoder, page: &Page, dict: &[i64], val_buf: &mut [i64; RLE_CHUNK], out: &mut Vec<i64>) {
    decoder.set_data(page.buffer.clone()).expect("arm D: set_data failed");
    let mut processed = 0usize;
    while processed < N_TOTAL {
        let chunk_len = (N_TOTAL - processed).min(RLE_CHUNK);
        let got = decoder
            .get_batch_with_dict::<i64>(dict, &mut val_buf[..chunk_len], chunk_len)
            .expect("arm D: get_batch_with_dict failed");
        assert_eq!(got, chunk_len, "arm D: RleDecoder produced fewer values than the page promised");

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
// Cross-arm digest verification (untimed, at setup).
// -----------------------------------------------------------------------------------------

fn verify_digests(cell_desc: &str, page_idx: usize, named: [(&str, u64); 5]) {
    let reference = named[0].1;
    let mut diverged: Vec<String> = Vec::new();
    for &(name, d) in named.iter().skip(1) {
        if d != reference {
            diverged.push(format!("{name} (digest {d:#018x})"));
        }
    }
    if !diverged.is_empty() {
        panic!(
            "rle_fill_r15 cell {cell_desc} page={page_idx}: cross-arm digest mismatch. Reference \
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
    let dict = kernel::generate_dict(BIT_WIDTH as u32, dict_seed);
    assert_eq!(dict.len(), 1usize << BIT_WIDTH, "fixture invariant: dict.len() == 1<<k");

    let pages: Vec<Page> = (0..PAGES_PER_CELL)
        .map(|page_idx| {
            let seed = kernel::derive_seed(cell_base_seed, page_idx as u64);
            build_page(cell.l, cell.shape, cell.survival, seed)
        })
        .collect();

    let max_selected = pages.iter().map(|p| kernel::popcount_words(&p.mask_words)).max().unwrap_or(0);

    let l = cell.l;
    let shape = cell.shape.label();
    let denom = cell.denom;
    let cell_desc = format!("L={l} k={BIT_WIDTH} shape={shape} s=1/{denom}");

    // --- untimed cross-arm correctness check ---
    {
        let mut decoder_a = RleDecoder::new(BIT_WIDTH);
        let mut decoder_e = RleDecoder::new(BIT_WIDTH);
        let mut decoder_f = RleDecoder::new(BIT_WIDTH);
        let mut decoder_c = RleDecoder::new(BIT_WIDTH);
        let mut decoder_d = RleDecoder::new(BIT_WIDTH);
        let mut out_a = vec![0i64; max_selected];
        let mut out_e = vec![0i64; max_selected];
        let mut out_f = vec![0i64; max_selected];
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut val_buf = [0i64; RLE_CHUNK];
        let mut out_c = Vec::with_capacity(max_selected);
        let mut out_d = Vec::with_capacity(max_selected);

        for (page_idx, page) in pages.iter().enumerate() {
            let written_a = run_arm_a(&mut decoder_a, page, &dict, &mut out_a);
            let written_e = run_arm_e(&mut decoder_e, page, &dict, &mut out_e);
            let written_f = run_arm_f(&mut decoder_f, page, &dict, &mut out_f);
            out_c.clear();
            run_arm_c(&mut decoder_c, page, &dict, &mut idx_buf, &mut out_c);
            out_d.clear();
            run_arm_d(&mut decoder_d, page, &dict, &mut val_buf, &mut out_d);

            verify_digests(
                &cell_desc,
                page_idx,
                [
                    ("selected_fill", kernel::fnv1a64(&out_a[..written_a])),
                    ("selected_fill_cursor", kernel::fnv1a64(&out_e[..written_e])),
                    ("selected_fill_cursor_unchecked", kernel::fnv1a64(&out_f[..written_f])),
                    ("decode_all_indices_compact", kernel::fnv1a64(&out_c)),
                    ("materialize_then_filter", kernel::fnv1a64(&out_d)),
                ],
            );
        }
    }

    // --- timed loop ---
    let mut group = c.benchmark_group("rle_fill_r15");
    group.sample_size(12);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs_f64(2.5));

    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut out = vec![0i64; max_selected];
        let _ = run_arm_a(&mut decoder, &pages[0], &dict, &mut out);
        group.bench_function(format!("selected_fill/L{l}/k{BIT_WIDTH}/{shape}/s{denom}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_arm_a(&mut decoder, page, &dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut out = vec![0i64; max_selected];
        let _ = run_arm_e(&mut decoder, &pages[0], &dict, &mut out);
        group.bench_function(format!("selected_fill_cursor/L{l}/k{BIT_WIDTH}/{shape}/s{denom}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_arm_e(&mut decoder, page, &dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut out = vec![0i64; max_selected];
        let _ = run_arm_f(&mut decoder, &pages[0], &dict, &mut out);
        group.bench_function(format!("selected_fill_cursor_unchecked/L{l}/k{BIT_WIDTH}/{shape}/s{denom}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_arm_f(&mut decoder, page, &dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut out = Vec::with_capacity(max_selected);
        run_arm_c(&mut decoder, &pages[0], &dict, &mut idx_buf, &mut out);
        group.bench_function(format!("decode_all_indices_compact/L{l}/k{BIT_WIDTH}/{shape}/s{denom}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                out.clear();
                run_arm_c(&mut decoder, page, &dict, &mut idx_buf, &mut out);
                hint::black_box(out.as_slice());
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut val_buf = [0i64; RLE_CHUNK];
        let mut out = Vec::with_capacity(max_selected);
        run_arm_d(&mut decoder, &pages[0], &dict, &mut val_buf, &mut out);
        group.bench_function(format!("materialize_then_filter/L{l}/k{BIT_WIDTH}/{shape}/s{denom}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                out.clear();
                run_arm_d(&mut decoder, page, &dict, &mut val_buf, &mut out);
                hint::black_box(out.as_slice());
            });
        });
    }
    group.finish();
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
