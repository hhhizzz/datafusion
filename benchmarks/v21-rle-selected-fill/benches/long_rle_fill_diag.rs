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

//! DIAGNOSTIC ONLY, temporary (2026-08-08). Not part of `arrow-long-rle-count-fill-v24.md`'s
//! frozen grid -- investigates G-V24-1's 2026-08-08 finding (`admitted` measured 19-42% slower
//! than `decode_all_indices_compact` on every short-run grid cell, cause not diagnosed) by
//! comparing `admitted` against three ablation siblings
//! (`_diag_inline`/`_diag_unchecked`/`_diag_both`, arrow-rs `b997de5b8`) on a focused 3-cell
//! slice (`L in {8,16,64}`, `s=1/16` only -- one point per `L`, not the full survival x shape
//! grid). Delete this file once the investigation concludes.
//!
//! Reads as: whichever diagnostic arm's `C/arm` ratio jumps back toward 1.0 (from `admitted`'s
//! measured ~0.79-0.85, i.e. `arm/C` 1.19-1.42) identifies the dominant cause. If none move
//! materially, both checked hypotheses are ruled out and the cause is structural (needs
//! disassembly or a more invasive redesign, not a targeted flag flip).

use std::hint;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, criterion_group, criterion_main};
use parquet::encodings::rle::{PackedSelection, RleDecoder};

#[allow(dead_code)]
#[path = "rle_fill/kernel.rs"]
mod kernel;

const TOP_SEED: u64 = 0xC0FF_EE15_2026_0807_u64 ^ 0xD1A6;
const PAGES_PER_CELL: usize = 32;
const N_TOTAL: usize = 131_072;
const RLE_CHUNK: usize = 1024;
const BIT_WIDTH: u8 = 8;
const SURVIVAL: f64 = 1.0 / 16.0;

struct Page {
    buffer: Bytes,
    mask_bytes: Vec<u8>,
    mask_words: Vec<u64>,
}

fn build_page(l: usize, seed: u64) -> Page {
    let mut rng = kernel::Xorshift64Star::new(seed);
    let buffer = kernel::build_rle_page(N_TOTAL, l, BIT_WIDTH, &mut rng);
    let mask_words = kernel::generate_random_mask(N_TOTAL, SURVIVAL, &mut rng);
    let mask_bytes = kernel::words_to_packed_bytes(&mask_words);
    Page { buffer: Bytes::from(buffer), mask_bytes, mask_words }
}

macro_rules! def_runner {
    ($name:ident, $method:ident) => {
        fn $name(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
            decoder.set_data(page.buffer.clone()).expect(concat!(stringify!($name), ": set_data failed"));
            let selection = PackedSelection::new(&page.mask_bytes, 0, N_TOTAL)
                .expect(concat!(stringify!($name), ": PackedSelection::new failed"));
            let (consumed, written) = decoder
                .$method(dict, out, selection)
                .expect(concat!(stringify!($name), ": decode failed"));
            assert_eq!(consumed, N_TOTAL, concat!(stringify!($name), ": did not consume the whole page"));
            written
        }
    };
}

def_runner!(run_admitted, get_batch_with_dict_selected_admitted);
def_runner!(run_diag_inline, get_batch_with_dict_selected_admitted_diag_inline);
def_runner!(run_diag_unchecked, get_batch_with_dict_selected_admitted_diag_unchecked);
def_runner!(run_diag_both, get_batch_with_dict_selected_admitted_diag_both);

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
            "long_rle_fill_diag cell {cell_desc} page={page_idx}: cross-arm digest mismatch. \
             Reference ({}) digest = {reference:#018x}. Diverged: [{}]",
            named[0].0,
            diverged.join(", ")
        );
    }
}

fn run_cell(c: &mut Criterion, l: usize) {
    let dict = kernel::generate_dict(BIT_WIDTH as u32, kernel::derive_seed(TOP_SEED, u64::MAX));
    assert_eq!(dict.len(), 1usize << BIT_WIDTH);

    let base = kernel::derive_seed(TOP_SEED, l as u64);
    let pages: Vec<Page> =
        (0..PAGES_PER_CELL).map(|i| build_page(l, kernel::derive_seed(base, i as u64))).collect();
    let max_selected = pages.iter().map(|p| kernel::popcount_words(&p.mask_words)).max().unwrap_or(0);
    let cell_desc = format!("L{l}/s16");

    // --- untimed cross-arm correctness check ---
    {
        let mut dec_a = RleDecoder::new(BIT_WIDTH);
        let mut dec_i = RleDecoder::new(BIT_WIDTH);
        let mut dec_u = RleDecoder::new(BIT_WIDTH);
        let mut dec_b = RleDecoder::new(BIT_WIDTH);
        let mut dec_c = RleDecoder::new(BIT_WIDTH);
        let mut out_a = vec![0i64; max_selected];
        let mut out_i = vec![0i64; max_selected];
        let mut out_u = vec![0i64; max_selected];
        let mut out_b = vec![0i64; max_selected];
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut out_c = Vec::with_capacity(max_selected);

        for (page_idx, page) in pages.iter().enumerate() {
            let wa = run_admitted(&mut dec_a, page, &dict, &mut out_a);
            let wi = run_diag_inline(&mut dec_i, page, &dict, &mut out_i);
            let wu = run_diag_unchecked(&mut dec_u, page, &dict, &mut out_u);
            let wb = run_diag_both(&mut dec_b, page, &dict, &mut out_b);
            out_c.clear();
            run_decode_all_indices_compact(&mut dec_c, page, &dict, &mut idx_buf, &mut out_c);

            verify_digests(
                &cell_desc,
                page_idx,
                &[
                    ("admitted", kernel::fnv1a64(&out_a[..wa])),
                    ("diag_inline", kernel::fnv1a64(&out_i[..wi])),
                    ("diag_unchecked", kernel::fnv1a64(&out_u[..wu])),
                    ("diag_both", kernel::fnv1a64(&out_b[..wb])),
                    ("decode_all_indices_compact", kernel::fnv1a64(&out_c)),
                ],
            );
        }
    }

    // --- timed loop ---
    let mut group = c.benchmark_group("long_rle_fill_diag");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs_f64(3.0));

    macro_rules! time_arm {
        ($label:literal, $runner:ident) => {{
            let mut pos = 0usize;
            let mut decoder = RleDecoder::new(BIT_WIDTH);
            let mut out = vec![0i64; max_selected];
            let _ = $runner(&mut decoder, &pages[0], &dict, &mut out);
            group.bench_function(format!("{}/{cell_desc}", $label), |b| {
                b.iter(|| {
                    let page = &pages[pos];
                    pos = (pos + 1) % pages.len();
                    let written = $runner(&mut decoder, page, &dict, &mut out);
                    hint::black_box(&out[..written]);
                });
            });
        }};
    }

    time_arm!("admitted", run_admitted);
    time_arm!("diag_inline", run_diag_inline);
    time_arm!("diag_unchecked", run_diag_unchecked);
    time_arm!("diag_both", run_diag_both);
    {
        let mut pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut out = Vec::with_capacity(max_selected);
        run_decode_all_indices_compact(&mut decoder, &pages[0], &dict, &mut idx_buf, &mut out);
        group.bench_function(format!("decode_all_indices_compact/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[pos];
                pos = (pos + 1) % pages.len();
                out.clear();
                run_decode_all_indices_compact(&mut decoder, page, &dict, &mut idx_buf, &mut out);
                hint::black_box(out.as_slice());
            });
        });
    }
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    for &l in &[8usize, 16, 64] {
        run_cell(c, l);
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
