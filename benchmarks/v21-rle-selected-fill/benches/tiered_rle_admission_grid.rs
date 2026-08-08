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

//! Step 0 of experiment `arrow-selected-decode-reader-wiring-v26`: quantifies
//! `get_batch_with_dict_selected_direct_gather_tiered`'s known RLE-run-length
//! risk (its RLE branch is per-run cursor count+fill, the shape R1.5 measured
//! losing ~1.6x at short runs) and checks whether
//! `get_batch_with_dict_selected_direct_gather_tiered_admitted` (arrow-rs, same
//! file, added alongside `tiered`) fixes it without losing `tiered`'s existing
//! long-run wins. See the experiment doc for the frozen G-W0 rule.
//!
//! ## Arms
//!
//! - `tiered` (unfixed, object under risk assessment):
//!   `get_batch_with_dict_selected_direct_gather_tiered` -- v23's carried-forward
//!   candidate, RLE branch is unconditional per-run count+fill.
//! - `tiered_admitted` (the fix under test):
//!   `get_batch_with_dict_selected_direct_gather_tiered_admitted` -- identical
//!   bit-packed tiers; RLE branch takes count+fill only at
//!   `effective_run_len >= 4096`, otherwise batch-decodes via the production
//!   `get_batch_with_dict` (spanning multiple short runs per call) and filters
//!   once per batch.
//! - `cursor`: `get_batch_with_dict_selected_cursor`, R1.5's unconditional
//!   count+fill and the digest reference (named first in every comparison).
//! - `decode_all_indices_compact` (C): the neutrality/win baseline, same as
//!   every prior stage.
//!
//! ## Cells
//!
//! Reuses R1's own short-run and long-run cells exactly, for direct
//! comparability with R1/R1.5/v24's already-published numbers:
//! - Neutrality (risk): `L in {8,16,64} x s in {1/64,1/16,1/4}` (9 cells) plus
//!   an `L=16, s=1` dense control (10 total). `tiered` is predicted to reproduce
//!   R1's ~1.6x-class loss here (its RLE branch is the same shape R1.5's
//!   `cursor` arm was); `tiered_admitted` must land within `[0.97,1.03]` of C.
//! - Win retention: `L in {4096, 65536} x s in {1/64,1/16,1/4}` (6 cells).
//!   `tiered_admitted` must retain whatever win `tiered` already has here
//!   (both arms should land close together, since long runs take the
//!   unmodified count+fill branch in both).
//!
//! Untimed cross-arm FNV-1a-64 digest verification per page before any timed
//! measurement (`cursor` is the reference, matching every prior stage's
//! convention), Criterion `sample_size=12`, `warm_up_time=1s`,
//! `measurement_time=2.5s`, `PAGES_PER_CELL=32` round-robin.

use std::hint;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, criterion_group, criterion_main};
use parquet::encodings::rle::{PackedSelection, RleDecoder};

#[allow(dead_code)]
#[path = "rle_fill/kernel.rs"]
mod kernel;

const TOP_SEED: u64 = 0xC0FF_EE15_2026_0807_u64 ^ 0x2601;

const PAGES_PER_CELL: usize = 32;
const N_TOTAL: usize = 131_072;
const RLE_CHUNK: usize = 1024;
const BIT_WIDTH: u8 = 8;

struct Page {
    buffer: Bytes,
    mask_bytes: Vec<u8>,
    mask_words: Vec<u64>,
}

#[derive(Clone, Copy)]
struct Survival {
    p: f64,
    label: &'static str,
}

const R1_SURVIVALS: [Survival; 3] = [
    Survival { p: 1.0 / 64.0, label: "s64" },
    Survival { p: 1.0 / 16.0, label: "s16" },
    Survival { p: 1.0 / 4.0, label: "s4" },
];

fn build_uniform_page(l: usize, survival: f64, seed: u64) -> Page {
    let mut rng = kernel::Xorshift64Star::new(seed);
    let buffer = kernel::build_rle_page(N_TOTAL, l, BIT_WIDTH, &mut rng);
    let mask_words = kernel::generate_random_mask(N_TOTAL, survival, &mut rng);
    let mask_bytes = kernel::words_to_packed_bytes(&mask_words);
    Page { buffer: Bytes::from(buffer), mask_bytes, mask_words }
}

fn build_dense_page(l: usize, seed: u64) -> Page {
    let mut rng = kernel::Xorshift64Star::new(seed);
    let buffer = kernel::build_rle_page(N_TOTAL, l, BIT_WIDTH, &mut rng);
    let mask_words = kernel::generate_dense_mask(N_TOTAL);
    let mask_bytes = kernel::words_to_packed_bytes(&mask_words);
    Page { buffer: Bytes::from(buffer), mask_bytes, mask_words }
}

// -----------------------------------------------------------------------------------------
// Per-arm runners.
// -----------------------------------------------------------------------------------------

fn run_tiered<T: Default + Clone>(decoder: &mut RleDecoder, page: &Page, dict: &[T], out: &mut [T]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("tiered: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, N_TOTAL).expect("tiered: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_direct_gather_tiered(dict, out, selection)
        .expect("tiered: decode failed");
    assert_eq!(consumed, N_TOTAL, "tiered: RleDecoder did not consume the whole page");
    written
}

fn run_tiered_admitted<T: Default + Clone>(
    decoder: &mut RleDecoder,
    page: &Page,
    dict: &[T],
    out: &mut [T],
) -> usize {
    decoder.set_data(page.buffer.clone()).expect("tiered_admitted: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, N_TOTAL)
        .expect("tiered_admitted: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_direct_gather_tiered_admitted(dict, out, selection)
        .expect("tiered_admitted: decode failed");
    assert_eq!(consumed, N_TOTAL, "tiered_admitted: RleDecoder did not consume the whole page");
    written
}

fn run_cursor<T: Default + Clone>(decoder: &mut RleDecoder, page: &Page, dict: &[T], out: &mut [T]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("cursor: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, N_TOTAL).expect("cursor: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_cursor(dict, out, selection)
        .expect("cursor: get_batch_with_dict_selected_cursor failed");
    assert_eq!(consumed, N_TOTAL, "cursor: RleDecoder did not consume the whole page");
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
    while processed < N_TOTAL {
        let chunk_len = (N_TOTAL - processed).min(RLE_CHUNK);
        let got = decoder
            .get_batch::<i32>(&mut idx_buf[..chunk_len])
            .expect("decode_all_indices_compact: get_batch failed");
        assert_eq!(got, chunk_len, "decode_all_indices_compact: fewer values than promised");

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

fn digest_i64(values: &[i64]) -> u64 {
    kernel::fnv1a64(values)
}

fn verify_digests(cell_desc: &str, page_idx: usize, named: &[(&str, u64)]) {
    let reference = named[0].1;
    let mut diverged: Vec<String> = Vec::new();
    for &(name, d) in named.iter().skip(1) {
        if d != reference {
            diverged.push(format!("{name} (digest {d:#018x})"));
        }
    }
    if !diverged.is_empty() {
        panic!(
            "tiered_rle_admission_grid cell {cell_desc} page={page_idx}: cross-arm digest mismatch. \
             Reference ({}) digest = {reference:#018x}. Diverged: [{}]",
            named[0].0,
            diverged.join(", ")
        );
    }
}

// -----------------------------------------------------------------------------------------
// Cell core.
// -----------------------------------------------------------------------------------------

fn run_cell(c: &mut Criterion, cell_desc: &str, dict: &[i64], pages: &[Page]) {
    let max_selected = pages.iter().map(|p| kernel::popcount_words(&p.mask_words)).max().unwrap_or(0);

    // --- untimed cross-arm correctness ---
    {
        let mut dec_t = RleDecoder::new(BIT_WIDTH);
        let mut dec_ta = RleDecoder::new(BIT_WIDTH);
        let mut dec_e = RleDecoder::new(BIT_WIDTH);
        let mut dec_c = RleDecoder::new(BIT_WIDTH);
        let mut out_t = vec![0i64; max_selected.max(1)];
        let mut out_ta = vec![0i64; max_selected.max(1)];
        let mut out_e = vec![0i64; max_selected.max(1)];
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut out_c = Vec::with_capacity(max_selected);

        for (page_idx, page) in pages.iter().enumerate() {
            let wt = run_tiered(&mut dec_t, page, dict, &mut out_t);
            let wta = run_tiered_admitted(&mut dec_ta, page, dict, &mut out_ta);
            let we = run_cursor(&mut dec_e, page, dict, &mut out_e);
            out_c.clear();
            run_decode_all_indices_compact(&mut dec_c, page, dict, &mut idx_buf, &mut out_c);

            verify_digests(
                cell_desc,
                page_idx,
                &[
                    ("cursor", digest_i64(&out_e[..we])),
                    ("tiered", digest_i64(&out_t[..wt])),
                    ("tiered_admitted", digest_i64(&out_ta[..wta])),
                    ("decode_all_indices_compact", digest_i64(&out_c)),
                ],
            );
        }

        // selected == 0 edge: a zero mask must skip-and-write-nothing through both new/existing
        // selected methods, on this cell's first page.
        let zero_mask = vec![0u8; pages[0].mask_bytes.len()];
        let mut dec = RleDecoder::new(BIT_WIDTH);
        dec.set_data(pages[0].buffer.clone()).expect("zero-mask set_data (tiered)");
        let selection = PackedSelection::new(&zero_mask, 0, N_TOTAL).expect("zero mask");
        let (consumed, written) = dec
            .get_batch_with_dict_selected_direct_gather_tiered(dict, &mut out_t, selection)
            .expect("zero-mask tiered failed");
        assert_eq!((consumed, written), (N_TOTAL, 0), "{cell_desc}: tiered selected==0 edge violated");

        let mut dec2 = RleDecoder::new(BIT_WIDTH);
        dec2.set_data(pages[0].buffer.clone()).expect("zero-mask set_data (tiered_admitted)");
        let selection2 = PackedSelection::new(&zero_mask, 0, N_TOTAL).expect("zero mask");
        let (consumed2, written2) = dec2
            .get_batch_with_dict_selected_direct_gather_tiered_admitted(dict, &mut out_ta, selection2)
            .expect("zero-mask tiered_admitted failed");
        assert_eq!(
            (consumed2, written2),
            (N_TOTAL, 0),
            "{cell_desc}: tiered_admitted selected==0 edge violated"
        );
    }

    // --- timed loop ---
    let mut group = c.benchmark_group("tiered_rle_admission_grid");
    group.sample_size(12);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs_f64(2.5));

    {
        let mut pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut out = vec![0i64; max_selected.max(1)];
        let _ = run_tiered(&mut decoder, &pages[0], dict, &mut out);
        group.bench_function(format!("tiered/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[pos];
                pos = (pos + 1) % pages.len();
                let written = run_tiered(&mut decoder, page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut out = vec![0i64; max_selected.max(1)];
        let _ = run_tiered_admitted(&mut decoder, &pages[0], dict, &mut out);
        group.bench_function(format!("tiered_admitted/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[pos];
                pos = (pos + 1) % pages.len();
                let written = run_tiered_admitted(&mut decoder, page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut out = vec![0i64; max_selected.max(1)];
        let _ = run_cursor(&mut decoder, &pages[0], dict, &mut out);
        group.bench_function(format!("cursor/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[pos];
                pos = (pos + 1) % pages.len();
                let written = run_cursor(&mut decoder, page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut out = Vec::with_capacity(max_selected);
        run_decode_all_indices_compact(&mut decoder, &pages[0], dict, &mut idx_buf, &mut out);
        group.bench_function(format!("decode_all_indices_compact/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[pos];
                pos = (pos + 1) % pages.len();
                out.clear();
                run_decode_all_indices_compact(&mut decoder, page, dict, &mut idx_buf, &mut out);
                hint::black_box(out.as_slice());
            });
        });
    }
    group.finish();
}

// -----------------------------------------------------------------------------------------
// Top-level driver.
// -----------------------------------------------------------------------------------------

fn run_full_matrix(c: &mut Criterion) {
    let mut cell_index: u64 = 0;
    let mut next_cell_seed = || {
        let seed = kernel::derive_seed(TOP_SEED, cell_index);
        cell_index += 1;
        seed
    };

    let dict = kernel::generate_dict(BIT_WIDTH as u32, kernel::derive_seed(TOP_SEED, u64::MAX));
    assert_eq!(dict.len(), 1usize << BIT_WIDTH);

    // Neutrality / risk-quantification cells: R1's own short-run grid.
    for &l in &[8usize, 16, 64] {
        for surv in R1_SURVIVALS {
            let base = next_cell_seed();
            let pages: Vec<Page> = (0..PAGES_PER_CELL)
                .map(|i| build_uniform_page(l, surv.p, kernel::derive_seed(base, i as u64)))
                .collect();
            run_cell(c, &format!("L{l}/{}", surv.label), &dict, &pages);
        }
    }
    {
        let base = next_cell_seed();
        let pages: Vec<Page> = (0..PAGES_PER_CELL)
            .map(|i| build_dense_page(16, kernel::derive_seed(base, i as u64)))
            .collect();
        run_cell(c, "L16/dense", &dict, &pages);
    }

    // Win-retention cells: R1's own long-run grid.
    for &l in &[4096usize, 65536] {
        for surv in R1_SURVIVALS {
            let base = next_cell_seed();
            let pages: Vec<Page> = (0..PAGES_PER_CELL)
                .map(|i| build_uniform_page(l, surv.p, kernel::derive_seed(base, i as u64)))
                .collect();
            run_cell(c, &format!("L{l}/{}", surv.label), &dict, &pages);
        }
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    run_full_matrix(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
