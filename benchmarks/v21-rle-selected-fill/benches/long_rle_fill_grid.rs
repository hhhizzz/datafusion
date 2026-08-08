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

//! Synthetic-grid Criterion bench for experiment `arrow-long-rle-count-fill-v24` (see that doc
//! for the frozen V1 admission, proof obligations, and v22-informed amendments this file
//! implements). Covers obligations 1-4 and amendments 2/4; the replay-fixture family (RLE
//! stratum + the amendment-3 BYTE_ARRAY bench) is a separate bench target.
//!
//! ## Arms
//!
//! - `admitted` (object under test): `RleDecoder::get_batch_with_dict_selected_admitted`
//!   (arrow-rs `ab416dd92`) -- count+fill only when `effective_run_len >= 4096`, arm-C-shaped
//!   deferred decode-then-filter otherwise.
//! - `cursor`: `get_batch_with_dict_selected_cursor`, the R1.5-admitted *unconditional*
//!   count+fill -- shows the short-run crossover the admission exists to avoid, and doubles as
//!   the digest reference (named first in every comparison).
//! - `decode_all_indices_compact` (C): the neutrality/win baseline for obligations 1 and 3.
//! - `materialize_then_filter` (D): context arm, unchanged from every prior stage.
//!
//! ## Cell groups
//!
//! 1. **Neutrality** (obligation 1; INT64, k=8): `L in {8,16,64} x s in {1/64,1/16,1/4}` plus a
//!    dense `L=16, s=1` control -- R1's short-run cells, where `admitted` must sit within noise
//!    of C (every run takes branch-false) while `cursor` reproduces its known ~1.6x loss.
//! 2. **Win retention** (obligation 3; INT64, k=8): `L in {4096, 65536} x s in {1/64,1/16,1/4}`
//!    -- every run admitted; compare against R1's measured band.
//! 3. **Real-density validation** (amendment 4; INT64, k=8): `L in {4096, 65536} x s in
//!    {1/400, 0.30, 0.50, 0.70}` -- the densities v22 measured real (ClickBench median 0.23%,
//!    TPC-DS median 56.33%) that R1's grid never touched. Gate: if the L=4096 cell loses at any
//!    of these, V1's threshold rises (admission correctness, not tuning).
//! 4. **INT32 parity** (amendment 2): `{L=16, s=1/16}`, `{L=4096, s=1/16}`, `{L=65536, s=1/16}`,
//!    `{L=4096, s=0.50}` with an `i32` dictionary and output -- expected ~= INT64 (same
//!    fixed-width fill mechanism), measured rather than assumed.
//! 5. **Mixed pages + run continuation** (obligation 2; INT64, k=8): one cell whose pages cycle
//!    `[8192-value RLE run, 192 x 16-value RLE runs, 20 x 256-value bit-packed runs]` (16,384
//!    values/cycle, 8 cycles/page) -- threshold crossed both ways plus bit-packed interleaving
//!    within one stream; and two multi-call cells over a page that is a *single* 131,072-value
//!    RLE run, consumed in uniform slices of 64 (every call sub-threshold: the long run must
//!    route branch-false per-call, per the review directive baked into `effective_run_len`) and
//!    of 8192 (every call admitted).
//!
//! Obligation 4's edges ride along: the dense control exercises the `selected == len`
//! short-circuit; every cell's digest check covers partial selections; `selected == 0` is
//! covered by a dedicated zero-mask digest-only check in every group-1 cell (no timing -- a
//! skip-only path has nothing to time).
//!
//! Untimed cross-arm FNV-1a-64 digest verification per page before any timed measurement
//! (`cursor` is the reference), Criterion `sample_size=12`, `warm_up_time=1s`,
//! `measurement_time=2.5s`, `PAGES_PER_CELL=32` round-robin -- all unchanged from every prior
//! stage. Two full rounds as separate Jobs, direction-agreement rule as before.

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

/// Distinct from every other bench file's seed constant in this crate.
const TOP_SEED: u64 = 0xC0FF_EE15_2026_0807_u64 ^ 0x2401;

const PAGES_PER_CELL: usize = 32;
const N_TOTAL: usize = 131_072;
const RLE_CHUNK: usize = 1024;
const BIT_WIDTH: u8 = 8;

// -----------------------------------------------------------------------------------------
// Fixture.
// -----------------------------------------------------------------------------------------

struct Page {
    buffer: Bytes,
    mask_bytes: Vec<u8>,
    mask_words: Vec<u64>,
}

/// Survival specification: iid Bernoulli at `p`, labeled `label` in bench ids.
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
const REAL_DENSITIES: [Survival; 4] = [
    Survival { p: 1.0 / 400.0, label: "d400" },
    Survival { p: 0.30, label: "p30" },
    Survival { p: 0.50, label: "p50" },
    Survival { p: 0.70, label: "p70" },
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

/// Cycle: one admitted-length RLE run, then 192 sub-threshold 16-value RLE runs, then 20
/// 256-value bit-packed runs. 8192 + 3072 + 5120 = 16,384 values per cycle, 8 cycles = N_TOTAL.
fn build_mixed_page(survival: f64, seed: u64) -> Page {
    const CYCLE: usize = 16_384;
    assert_eq!(N_TOTAL % CYCLE, 0);
    let mut rng = kernel::Xorshift64Star::new(seed);
    let max_value = (1u64 << BIT_WIDTH) - 1;
    let mut buffer = Vec::new();
    for _ in 0..(N_TOTAL / CYCLE) {
        kernel::write_rle_run(&mut buffer, 8192, rng.next_u64() & max_value, BIT_WIDTH);
        for _ in 0..192 {
            kernel::write_rle_run(&mut buffer, 16, rng.next_u64() & max_value, BIT_WIDTH);
        }
        for _ in 0..20 {
            let values = kernel::generate_bitpacked_values(256, BIT_WIDTH, &mut rng);
            kernel::write_bit_packed_run(&mut buffer, &values, BIT_WIDTH);
        }
    }
    let mask_words = kernel::generate_random_mask(N_TOTAL, survival, &mut rng);
    let mask_bytes = kernel::words_to_packed_bytes(&mask_words);
    Page { buffer: Bytes::from(buffer), mask_bytes, mask_words }
}

/// One single RLE run spanning the whole page, for the multi-call run-continuation cells.
fn build_single_run_page(survival: f64, seed: u64) -> Page {
    let mut rng = kernel::Xorshift64Star::new(seed);
    let max_value = (1u64 << BIT_WIDTH) - 1;
    let mut buffer = Vec::new();
    kernel::write_rle_run(&mut buffer, N_TOTAL as u64, rng.next_u64() & max_value, BIT_WIDTH);
    let mask_words = kernel::generate_random_mask(N_TOTAL, survival, &mut rng);
    let mask_bytes = kernel::words_to_packed_bytes(&mask_words);
    Page { buffer: Bytes::from(buffer), mask_bytes, mask_words }
}

// -----------------------------------------------------------------------------------------
// Per-arm runners, generic over the value type (INT64 cells use i64, parity cells i32).
// -----------------------------------------------------------------------------------------

fn run_admitted<T: Default + Clone>(
    decoder: &mut RleDecoder,
    page: &Page,
    dict: &[T],
    out: &mut [T],
) -> usize {
    decoder.set_data(page.buffer.clone()).expect("admitted: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, N_TOTAL)
        .expect("admitted: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_admitted(dict, out, selection)
        .expect("admitted: get_batch_with_dict_selected_admitted failed");
    assert_eq!(consumed, N_TOTAL, "admitted: RleDecoder did not consume the whole page");
    written
}

fn run_cursor<T: Default + Clone>(
    decoder: &mut RleDecoder,
    page: &Page,
    dict: &[T],
    out: &mut [T],
) -> usize {
    decoder.set_data(page.buffer.clone()).expect("cursor: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, N_TOTAL)
        .expect("cursor: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_cursor(dict, out, selection)
        .expect("cursor: get_batch_with_dict_selected_cursor failed");
    assert_eq!(consumed, N_TOTAL, "cursor: RleDecoder did not consume the whole page");
    written
}

fn run_decode_all_indices_compact<T: Copy>(
    decoder: &mut RleDecoder,
    page: &Page,
    dict: &[T],
    idx_buf: &mut [i32; RLE_CHUNK],
    out: &mut Vec<T>,
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

fn run_materialize_then_filter<T: Default + Clone + Copy>(
    decoder: &mut RleDecoder,
    page: &Page,
    dict: &[T],
    val_buf: &mut [T],
    out: &mut Vec<T>,
) {
    decoder.set_data(page.buffer.clone()).expect("materialize_then_filter: set_data failed");
    let mut processed = 0usize;
    while processed < N_TOTAL {
        let chunk_len = (N_TOTAL - processed).min(RLE_CHUNK);
        let got = decoder
            .get_batch_with_dict::<T>(dict, &mut val_buf[..chunk_len], chunk_len)
            .expect("materialize_then_filter: get_batch_with_dict failed");
        assert_eq!(got, chunk_len, "materialize_then_filter: fewer values than promised");

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

/// Multi-call variants: the whole page consumed in uniform `slice_len`-value calls.
fn run_multi_call_admitted<T: Default + Clone>(
    decoder: &mut RleDecoder,
    page: &Page,
    dict: &[T],
    slice_len: usize,
    out: &mut [T],
) -> usize {
    decoder.set_data(page.buffer.clone()).expect("mc admitted: set_data failed");
    let mut start = 0usize;
    let mut written = 0usize;
    while start < N_TOTAL {
        let len = slice_len.min(N_TOTAL - start);
        let selection = PackedSelection::new(&page.mask_bytes, start, len)
            .expect("mc admitted: PackedSelection::new failed");
        let (consumed, w) = decoder
            .get_batch_with_dict_selected_admitted(dict, &mut out[written..], selection)
            .expect("mc admitted: get_batch_with_dict_selected_admitted failed");
        assert_eq!(consumed, len, "mc admitted: did not consume the whole slice");
        written += w;
        start += len;
    }
    written
}

fn run_multi_call_cursor<T: Default + Clone>(
    decoder: &mut RleDecoder,
    page: &Page,
    dict: &[T],
    slice_len: usize,
    out: &mut [T],
) -> usize {
    decoder.set_data(page.buffer.clone()).expect("mc cursor: set_data failed");
    let mut start = 0usize;
    let mut written = 0usize;
    while start < N_TOTAL {
        let len = slice_len.min(N_TOTAL - start);
        let selection = PackedSelection::new(&page.mask_bytes, start, len)
            .expect("mc cursor: PackedSelection::new failed");
        let (consumed, w) = decoder
            .get_batch_with_dict_selected_cursor(dict, &mut out[written..], selection)
            .expect("mc cursor: get_batch_with_dict_selected_cursor failed");
        assert_eq!(consumed, len, "mc cursor: did not consume the whole slice");
        written += w;
        start += len;
    }
    written
}

// -----------------------------------------------------------------------------------------
// Digest helpers. `fnv1a64` hashes i64 streams; i32 outputs are widened (untimed setup only).
// -----------------------------------------------------------------------------------------

fn digest_i64(values: &[i64]) -> u64 {
    kernel::fnv1a64(values)
}

fn digest_i32(values: &[i32]) -> u64 {
    let widened: Vec<i64> = values.iter().map(|&v| v as i64).collect();
    kernel::fnv1a64(&widened)
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
            "long_rle_fill_grid cell {cell_desc} page={page_idx}: cross-arm digest mismatch. \
             Reference ({}) digest = {reference:#018x}. Diverged: [{}]",
            named[0].0,
            diverged.join(", ")
        );
    }
}

// -----------------------------------------------------------------------------------------
// Cell cores.
// -----------------------------------------------------------------------------------------

/// Single-call 4-arm cell over prebuilt pages, INT64 values.
fn run_cell_i64(c: &mut Criterion, cell_desc: &str, dict: &[i64], pages: &[Page]) {
    let max_selected = pages.iter().map(|p| kernel::popcount_words(&p.mask_words)).max().unwrap_or(0);

    // --- untimed cross-arm correctness (plus the selected==0 edge, digest-only) ---
    {
        let mut dec_a = RleDecoder::new(BIT_WIDTH);
        let mut dec_e = RleDecoder::new(BIT_WIDTH);
        let mut dec_c = RleDecoder::new(BIT_WIDTH);
        let mut dec_d = RleDecoder::new(BIT_WIDTH);
        let mut out_a = vec![0i64; max_selected.max(1)];
        let mut out_e = vec![0i64; max_selected.max(1)];
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut val_buf = [0i64; RLE_CHUNK];
        let mut out_c = Vec::with_capacity(max_selected);
        let mut out_d = Vec::with_capacity(max_selected);

        for (page_idx, page) in pages.iter().enumerate() {
            let wa = run_admitted(&mut dec_a, page, dict, &mut out_a);
            let we = run_cursor(&mut dec_e, page, dict, &mut out_e);
            out_c.clear();
            run_decode_all_indices_compact(&mut dec_c, page, dict, &mut idx_buf, &mut out_c);
            out_d.clear();
            run_materialize_then_filter(&mut dec_d, page, dict, &mut val_buf, &mut out_d);

            verify_digests(
                cell_desc,
                page_idx,
                &[
                    ("cursor", digest_i64(&out_e[..we])),
                    ("admitted", digest_i64(&out_a[..wa])),
                    ("decode_all_indices_compact", digest_i64(&out_c)),
                    ("materialize_then_filter", digest_i64(&out_d)),
                ],
            );
        }

        // selected == 0 edge (obligation 4): a zero mask must skip-and-write-nothing through
        // both selected methods, on this cell's first page.
        let zero_mask = vec![0u8; pages[0].mask_bytes.len()];
        let selection = PackedSelection::new(&zero_mask, 0, N_TOTAL).expect("zero mask");
        let mut dec = RleDecoder::new(BIT_WIDTH);
        dec.set_data(pages[0].buffer.clone()).expect("zero-mask set_data");
        let (consumed, written) = dec
            .get_batch_with_dict_selected_admitted(dict, &mut out_a, selection)
            .expect("zero-mask admitted failed");
        assert_eq!((consumed, written), (N_TOTAL, 0), "{cell_desc}: selected==0 edge violated");
    }

    // --- timed loop ---
    let mut group = c.benchmark_group("long_rle_fill_grid");
    group.sample_size(12);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs_f64(2.5));

    {
        let mut pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut out = vec![0i64; max_selected.max(1)];
        let _ = run_admitted(&mut decoder, &pages[0], dict, &mut out);
        group.bench_function(format!("admitted/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[pos];
                pos = (pos + 1) % pages.len();
                let written = run_admitted(&mut decoder, page, dict, &mut out);
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
    {
        let mut pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut val_buf = [0i64; RLE_CHUNK];
        let mut out = Vec::with_capacity(max_selected);
        run_materialize_then_filter(&mut decoder, &pages[0], dict, &mut val_buf, &mut out);
        group.bench_function(format!("materialize_then_filter/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[pos];
                pos = (pos + 1) % pages.len();
                out.clear();
                run_materialize_then_filter(&mut decoder, page, dict, &mut val_buf, &mut out);
                hint::black_box(out.as_slice());
            });
        });
    }
    group.finish();
}

/// INT32 parity cell (amendment 2): identical structure, i32 dictionary and output.
fn run_cell_i32(c: &mut Criterion, cell_desc: &str, dict: &[i32], pages: &[Page]) {
    let max_selected = pages.iter().map(|p| kernel::popcount_words(&p.mask_words)).max().unwrap_or(0);

    {
        let mut dec_a = RleDecoder::new(BIT_WIDTH);
        let mut dec_e = RleDecoder::new(BIT_WIDTH);
        let mut dec_c = RleDecoder::new(BIT_WIDTH);
        let mut dec_d = RleDecoder::new(BIT_WIDTH);
        let mut out_a = vec![0i32; max_selected.max(1)];
        let mut out_e = vec![0i32; max_selected.max(1)];
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut val_buf = [0i32; RLE_CHUNK];
        let mut out_c = Vec::with_capacity(max_selected);
        let mut out_d = Vec::with_capacity(max_selected);

        for (page_idx, page) in pages.iter().enumerate() {
            let wa = run_admitted(&mut dec_a, page, dict, &mut out_a);
            let we = run_cursor(&mut dec_e, page, dict, &mut out_e);
            out_c.clear();
            run_decode_all_indices_compact(&mut dec_c, page, dict, &mut idx_buf, &mut out_c);
            out_d.clear();
            run_materialize_then_filter(&mut dec_d, page, dict, &mut val_buf, &mut out_d);

            verify_digests(
                cell_desc,
                page_idx,
                &[
                    ("cursor", digest_i32(&out_e[..we])),
                    ("admitted", digest_i32(&out_a[..wa])),
                    ("decode_all_indices_compact", digest_i32(&out_c)),
                    ("materialize_then_filter", digest_i32(&out_d)),
                ],
            );
        }
    }

    let mut group = c.benchmark_group("long_rle_fill_grid");
    group.sample_size(12);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs_f64(2.5));

    {
        let mut pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut out = vec![0i32; max_selected.max(1)];
        let _ = run_admitted(&mut decoder, &pages[0], dict, &mut out);
        group.bench_function(format!("admitted/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[pos];
                pos = (pos + 1) % pages.len();
                let written = run_admitted(&mut decoder, page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut out = vec![0i32; max_selected.max(1)];
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
    {
        let mut pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut val_buf = [0i32; RLE_CHUNK];
        let mut out = Vec::with_capacity(max_selected);
        run_materialize_then_filter(&mut decoder, &pages[0], dict, &mut val_buf, &mut out);
        group.bench_function(format!("materialize_then_filter/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[pos];
                pos = (pos + 1) % pages.len();
                out.clear();
                run_materialize_then_filter(&mut decoder, page, dict, &mut val_buf, &mut out);
                hint::black_box(out.as_slice());
            });
        });
    }
    group.finish();
}

/// Multi-call run-continuation cell (obligation 2): single-run pages consumed in uniform
/// `slice_len` slices by the two selected methods; C/D consume the whole page single-call
/// (their chunking is internal and call-independent), giving an independent digest witness.
fn run_multi_call_cell(c: &mut Criterion, cell_desc: &str, slice_len: usize, dict: &[i64], pages: &[Page]) {
    let max_selected = pages.iter().map(|p| kernel::popcount_words(&p.mask_words)).max().unwrap_or(0);

    {
        let mut dec_a = RleDecoder::new(BIT_WIDTH);
        let mut dec_e = RleDecoder::new(BIT_WIDTH);
        let mut dec_c = RleDecoder::new(BIT_WIDTH);
        let mut dec_d = RleDecoder::new(BIT_WIDTH);
        let mut out_a = vec![0i64; max_selected.max(1)];
        let mut out_e = vec![0i64; max_selected.max(1)];
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut val_buf = [0i64; RLE_CHUNK];
        let mut out_c = Vec::with_capacity(max_selected);
        let mut out_d = Vec::with_capacity(max_selected);

        for (page_idx, page) in pages.iter().enumerate() {
            let wa = run_multi_call_admitted(&mut dec_a, page, dict, slice_len, &mut out_a);
            let we = run_multi_call_cursor(&mut dec_e, page, dict, slice_len, &mut out_e);
            out_c.clear();
            run_decode_all_indices_compact(&mut dec_c, page, dict, &mut idx_buf, &mut out_c);
            out_d.clear();
            run_materialize_then_filter(&mut dec_d, page, dict, &mut val_buf, &mut out_d);

            verify_digests(
                cell_desc,
                page_idx,
                &[
                    ("cursor", digest_i64(&out_e[..we])),
                    ("admitted", digest_i64(&out_a[..wa])),
                    ("decode_all_indices_compact", digest_i64(&out_c)),
                    ("materialize_then_filter", digest_i64(&out_d)),
                ],
            );
        }
    }

    let mut group = c.benchmark_group("long_rle_fill_grid_multi_call");
    group.sample_size(12);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs_f64(2.5));

    {
        let mut pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut out = vec![0i64; max_selected.max(1)];
        let _ = run_multi_call_admitted(&mut decoder, &pages[0], dict, slice_len, &mut out);
        group.bench_function(format!("admitted/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[pos];
                pos = (pos + 1) % pages.len();
                let written = run_multi_call_admitted(&mut decoder, page, dict, slice_len, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut pos = 0usize;
        let mut decoder = RleDecoder::new(BIT_WIDTH);
        let mut out = vec![0i64; max_selected.max(1)];
        let _ = run_multi_call_cursor(&mut decoder, &pages[0], dict, slice_len, &mut out);
        group.bench_function(format!("cursor/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[pos];
                pos = (pos + 1) % pages.len();
                let written = run_multi_call_cursor(&mut decoder, page, dict, slice_len, &mut out);
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

    let dict_i64 = kernel::generate_dict(BIT_WIDTH as u32, kernel::derive_seed(TOP_SEED, u64::MAX));
    assert_eq!(dict_i64.len(), 1usize << BIT_WIDTH);
    let dict_i32: Vec<i32> = {
        let mut rng = kernel::Xorshift64Star::new(kernel::derive_seed(TOP_SEED, u64::MAX - 1));
        (0..1usize << BIT_WIDTH).map(|_| rng.next_u64() as i32).collect()
    };

    // Group 1: neutrality (short runs) + dense control.
    for &l in &[8usize, 16, 64] {
        for surv in R1_SURVIVALS {
            let base = next_cell_seed();
            let pages: Vec<Page> = (0..PAGES_PER_CELL)
                .map(|i| build_uniform_page(l, surv.p, kernel::derive_seed(base, i as u64)))
                .collect();
            run_cell_i64(c, &format!("L{l}/{}", surv.label), &dict_i64, &pages);
        }
    }
    {
        let base = next_cell_seed();
        let pages: Vec<Page> = (0..PAGES_PER_CELL)
            .map(|i| build_dense_page(16, kernel::derive_seed(base, i as u64)))
            .collect();
        run_cell_i64(c, "L16/dense", &dict_i64, &pages);
    }

    // Groups 2+3: long runs at R1 survivals and at the v22-real densities.
    for &l in &[4096usize, 65536] {
        for surv in R1_SURVIVALS.iter().chain(REAL_DENSITIES.iter()) {
            let base = next_cell_seed();
            let pages: Vec<Page> = (0..PAGES_PER_CELL)
                .map(|i| build_uniform_page(l, surv.p, kernel::derive_seed(base, i as u64)))
                .collect();
            run_cell_i64(c, &format!("L{l}/{}", surv.label), &dict_i64, &pages);
        }
    }

    // Group 4: INT32 parity cells.
    for (l, surv) in [
        (16usize, R1_SURVIVALS[1]),
        (4096, R1_SURVIVALS[1]),
        (65536, R1_SURVIVALS[1]),
        (4096, REAL_DENSITIES[2]),
    ] {
        let base = next_cell_seed();
        let pages: Vec<Page> = (0..PAGES_PER_CELL)
            .map(|i| build_uniform_page(l, surv.p, kernel::derive_seed(base, i as u64)))
            .collect();
        run_cell_i32(c, &format!("i32/L{l}/{}", surv.label), &dict_i32, &pages);
    }

    // Group 5: mixed pages + multi-call run continuation.
    {
        let base = next_cell_seed();
        let pages: Vec<Page> = (0..PAGES_PER_CELL)
            .map(|i| build_mixed_page(1.0 / 16.0, kernel::derive_seed(base, i as u64)))
            .collect();
        run_cell_i64(c, "mixed/s16", &dict_i64, &pages);
    }
    for slice_len in [64usize, 8192] {
        let base = next_cell_seed();
        let pages: Vec<Page> = (0..PAGES_PER_CELL)
            .map(|i| build_single_run_page(1.0 / 16.0, kernel::derive_seed(base, i as u64)))
            .collect();
        run_multi_call_cell(
            c,
            &format!("singlerun/slice{slice_len}/s16"),
            slice_len,
            &dict_i64,
            &pages,
        );
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    run_full_matrix(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
