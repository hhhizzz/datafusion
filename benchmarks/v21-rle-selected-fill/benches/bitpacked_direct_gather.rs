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

//! Smoke-stage Criterion bench for experiment `arrow-bitpacked-direct-gather-gate-v23`. See
//! `codex/experiments/arrow-bitpacked-direct-gather-gate-v23.md` for the frozen gate this
//! feeds. This is deliberately a *focused* grid, not the full v18-style matrix: its purpose is
//! the first real-API compile-and-correctness signal for
//! `RleDecoder::get_batch_with_dict_selected_direct_gather`/`_checked` (added in commit
//! `177308a7e`), which until now have only been checked with simplified standalone-Rust mocks,
//! never the real `RleDecoder`/`BitReader`/`PackedSelection` types. The full grid (survivals x
//! shapes x the v22 replay fixtures) is a follow-up once this compiles and every digest matches
//! on real K8s hardware.
//!
//! ## Arms
//!
//! - `cursor` (reference): `RleDecoder::get_batch_with_dict_selected_cursor`, R1.5's already
//!   -admitted trusted baseline (arm E there). Named first in every digest comparison below.
//! - `direct_gather`: `RleDecoder::get_batch_with_dict_selected_direct_gather`, the
//!   preflight-then-unchecked object under test.
//! - `direct_gather_checked`: `RleDecoder::get_batch_with_dict_selected_direct_gather_checked`,
//!   its always-checked, always-direct-gather-algorithm sibling (kill gate 3's "checked
//!   treatment" sub-arm).
//! - `tiered`: `RleDecoder::get_batch_with_dict_selected_direct_gather_tiered` (2026-08-08
//!   addition, see the experiment doc's "Tiered-admission arm" section) -- splits
//!   `direct_gather`'s single preflight into two independent per-run checks so a run whose only
//!   failure is an undersized dictionary gets a checked-dictionary direct gather instead of the
//!   full decode-all-then-filter fallback. Group 4 (undersized dictionary) is this arm's own
//!   primary target: its cells exercise the new checked-dictionary path (`direct_gather`
//!   /`direct_gather_checked` do not distinguish "checked dict, unchecked read" from either of
//!   their own two paths). Groups 1's page-final runs and group 3's third multi-call chunk
//!   continue to incidentally exercise the shared fallback branch (see their own entries below).
//! - `decode_all_indices_compact` / `materialize_then_filter`: the same two independent,
//!   decoder-internals-agnostic witnesses `rle_fill.rs`/`rle_fill_r15.rs` already use (stock
//!   `get_batch`/`get_batch_with_dict` plus a harness-level selection-word walk) -- included in
//!   the single-call cell groups below for the same reason they were there: neither shares any
//!   code with *any* of the three `RleDecoder::get_batch_with_dict_selected_*` methods, so they
//!   catch a bug shared between the reference and the object under test that digest-only
//!   comparison of the three `selected_*` methods could not.
//!
//! ## Cell groups (14 cells total, far smaller than R1's 97 or even R1.5's 16)
//!
//! 1. **Pure bit-packed, single call, exact dictionary** (8 cells): `dict.len() == 1 << k` for
//!    `k` in the established grid (`{2, 8, 12, 16}`) crossed with run length `l` in `{8, 512}`
//!    (the shortest length this format allows, and the top of the confirmed real-world range --
//!    see the experiment doc's v22-informed amendments: "confirmed real bit-packed runs are all
//!    <= 512 values"). `dict.len() == 1 << k` satisfies `direct_gather_safe`'s dictionary-size
//!    condition throughout, so every run except (per run's own, independent safety check) a
//!    page's *last* run -- whose payload butts against the page buffer's own end, leaving no
//!    trailing byte slack for `max_safe_peek_bit_position`'s 8-byte margin -- takes the
//!    unchecked fast path; that last run correctly falls back instead, an intentional, documented
//!    case (see `get_batch_with_dict_selected_direct_gather`'s own doc comment: "a run ending
//!    within 8 bytes of the value-byte stream's end"). Both are exercised across this group's 8
//!    cells; the digest check does not care which one ran. Every `k` not a divisor of 8 (`12`)
//!    already forces most in-run values onto a non-byte-aligned bit position (only `k`'s own
//!    multiples-of-8 boundary re-aligns), which is this format's only real notion of "non-zero
//!    bit phase" within a single run -- run *starts* are always byte-aligned (each run's header
//!    +payload is byte-rounded as a whole), so no separate scenario is needed to reach it.
//! 2. **Mixed RLE + bit-packed, single call** (2 cells): one page alternating a 768-value RLE
//!    run with a 256-value bit-packed run (768+256=1024 divides `N_TOTAL` evenly), at `k` in
//!    `{2, 12}`. Exercises `reload()`'s existing (unmodified) run-kind dispatch feeding these
//!    new methods' `rle_left`/`bit_packed_left` branches alternately within one call, which R1
//!    /R1.5's pure-RLE fixtures and group 1's pure-bit-packed fixtures never do together.
//! 3. **Multi-call incremental consumption** (2 cells): one page holding a *single* bit-packed
//!    run spanning the entire page (`l == N_TOTAL`), decoded via three successive calls at
//!    arbitrary, non-64-aligned split points (`MULTI_CALL_SPLITS`), each with its own
//!    `PackedSelection` window (`PackedSelection::new(data, bit_offset, len)` with `bit_offset`
//!    advanced to the cumulative split point) rather than one call consuming the whole page --
//!    this is the shape a real batched caller (e.g. an 8192-row `RecordBatch` read against a
//!    <=512-value run) will actually use. The second call resumes `bit_reader.bit_position()`
//!    from a non-byte-aligned point left by `advance_by_bits` on the first -- verified (by hand,
//!    for both `k` values below) to stay comfortably inside `max_safe_peek_bit_position`'s
//!    margin, so it exercises the *fast* path from a non-byte-aligned start, the one scenario
//!    group 1 cannot reach (there, every run starts a fresh call at a byte-aligned position).
//!    The third call reaches this single-run page's own tail -- like group 1's last-run-in-a
//!    -page case, that trips the same documented tail-safety fallback instead, so it exercises
//!    *that* path's non-byte-aligned resume; still a useful check, just not this group's primary
//!    target. `k in {12, 16}`: one non-8-divisor, one byte-aligned control.
//! 4. **Undersized dictionary** (2 cells): `dict.len() < 1 << k` (`k=12` with `dict_len=3000`,
//!    `k=16` with `dict_len=40000`) -- the *common* real case, since a real Parquet writer picks
//!    the smallest `bit_width` with `dict.len() <= 1 << bit_width`, so `dict.len()` is an exact
//!    power of 2 only by coincidence. Forces `direct_gather`'s preflight to fail -- regardless of
//!    tail position, unlike groups 1/3's incidental fallback cases -- and take its decode-all
//!    -then-filter fallback, while `direct_gather_checked` -- which has no such preflight and
//!    always runs the direct-gather algorithm -- exercises its checked dictionary lookup against
//!    an undersized dictionary for the first time. Only the three `selected_*` arms are compared
//!    here (arms C/D are dictionary-size-agnostic and already covered by group 1).
//!
//! ## Fixture provenance
//!
//! `write_bit_packed_run`/`generate_bitpacked_values`/`generate_bitpacked_values_bounded` are
//! new additions to the shared `rle_fill/kernel.rs` (synced byte-for-byte to
//! `.scratch/v21-r1-devcheck` and locally pressure-tested there, including a hand-verified cross
//! -check against the Apache Parquet format specification's own worked bit-packing example --
//! see `.scratch/v21-r1-devcheck/src/tests.rs`). `write_rle_run`/`generate_dict`/mask generators
//! /`fnv1a64` are unchanged, already-admitted R1 helpers, reused as-is.

use std::hint;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, criterion_group, criterion_main};
use parquet::encodings::rle::{PackedSelection, RleDecoder};

// kernel.rs is shared across this crate's bench binaries; not every helper it exports is used
// by every binary (e.g. this file uses generate_random_mask but not generate_dense_mask).
#[allow(dead_code)]
#[path = "rle_fill/kernel.rs"]
mod kernel;

/// Distinct from `rle_fill.rs`/`rle_fill_r15.rs`'s seed constants, so this stage's fixtures are
/// independently generated, not a reused subset of either prior stage's exact bytes.
const TOP_SEED: u64 = 0xC0FF_EE15_2026_0807_u64 ^ 0x5223;

const PAGES_PER_CELL: usize = 32;

/// Frozen total page value count, matching R1/R1.5 (kept identical so the pure-RLE reload/skip
/// arithmetic these methods share with the trusted baseline is exercised at the same scale).
const N_TOTAL: usize = 131_072;

const RLE_CHUNK: usize = 1024;

/// Representative selection density for every smoke cell (R1.5's own "decision cell" density,
/// where the count+fill mechanism's win over decode-all-then-filter was smallest) -- sweeping
/// survival is the full grid's job, not this smoke stage's.
const SURVIVAL: f64 = 1.0 / 16.0;

/// Arbitrary, deliberately non-64-aligned, non-8-aligned split points for the multi-call cell
/// group: three chunks summing to `N_TOTAL`, each starting mid-run relative to the single
/// bit-packed run those fixtures use.
const MULTI_CALL_SPLITS: [usize; 3] = [41_111, 47_777, N_TOTAL - 41_111 - 47_777];

// -----------------------------------------------------------------------------------------
// Fixture: one page is one dictionary-index stream (bit-packed only, or mixed with RLE) plus
// its page-level selection mask, in the same two-representation shape R1/R1.5 use.
// -----------------------------------------------------------------------------------------

struct Page {
    buffer: Bytes,
    mask_bytes: Vec<u8>,
    mask_words: Vec<u64>,
}

fn build_pure_bitpacked_page(l: usize, k: u8, seed: u64) -> Page {
    assert_eq!(N_TOTAL % l, 0, "N_TOTAL must be a multiple of the run length l");
    let mut rng = kernel::Xorshift64Star::new(seed);
    let mut buffer = Vec::new();
    for _ in 0..(N_TOTAL / l) {
        let values = kernel::generate_bitpacked_values(l, k, &mut rng);
        kernel::write_bit_packed_run(&mut buffer, &values, k);
    }
    let mask_words = kernel::generate_random_mask(N_TOTAL, SURVIVAL, &mut rng);
    let mask_bytes = kernel::words_to_packed_bytes(&mask_words);
    Page { buffer: Bytes::from(buffer), mask_bytes, mask_words }
}

/// Alternating RLE(768) / bit-packed(256) runs, repeated to exactly fill `N_TOTAL`
/// (768+256=1024, and `N_TOTAL` is a multiple of 1024).
fn build_mixed_page(k: u8, seed: u64) -> Page {
    const RLE_LEN: u64 = 768;
    const BP_LEN: usize = 256;
    const CYCLE: usize = RLE_LEN as usize + BP_LEN;
    assert_eq!(N_TOTAL % CYCLE, 0, "N_TOTAL must be a multiple of the mixed-page cycle length");
    let mut rng = kernel::Xorshift64Star::new(seed);
    let max_value = (1u64 << k) - 1;
    let mut buffer = Vec::new();
    for _ in 0..(N_TOTAL / CYCLE) {
        let rle_value = rng.next_u64() & max_value;
        kernel::write_rle_run(&mut buffer, RLE_LEN, rle_value, k);
        let bp_values = kernel::generate_bitpacked_values(BP_LEN, k, &mut rng);
        kernel::write_bit_packed_run(&mut buffer, &bp_values, k);
    }
    let mask_words = kernel::generate_random_mask(N_TOTAL, SURVIVAL, &mut rng);
    let mask_bytes = kernel::words_to_packed_bytes(&mask_words);
    Page { buffer: Bytes::from(buffer), mask_bytes, mask_words }
}

/// One bit-packed run spanning the whole page, for the multi-call incremental cell group.
fn build_single_run_page(k: u8, seed: u64) -> Page {
    let mut rng = kernel::Xorshift64Star::new(seed);
    let values = kernel::generate_bitpacked_values(N_TOTAL, k, &mut rng);
    let mut buffer = Vec::new();
    kernel::write_bit_packed_run(&mut buffer, &values, k);
    let mask_words = kernel::generate_random_mask(N_TOTAL, SURVIVAL, &mut rng);
    let mask_bytes = kernel::words_to_packed_bytes(&mask_words);
    Page { buffer: Bytes::from(buffer), mask_bytes, mask_words }
}

/// Pure bit-packed page whose values are all `< dict_len < 1 << k`, for the undersized
/// -dictionary cell group.
fn build_pure_bitpacked_page_bounded(l: usize, k: u8, dict_len: usize, seed: u64) -> Page {
    assert_eq!(N_TOTAL % l, 0, "N_TOTAL must be a multiple of the run length l");
    let mut rng = kernel::Xorshift64Star::new(seed);
    let mut buffer = Vec::new();
    for _ in 0..(N_TOTAL / l) {
        let values = kernel::generate_bitpacked_values_bounded(l, k, dict_len, &mut rng);
        kernel::write_bit_packed_run(&mut buffer, &values, k);
    }
    let mask_words = kernel::generate_random_mask(N_TOTAL, SURVIVAL, &mut rng);
    let mask_bytes = kernel::words_to_packed_bytes(&mask_words);
    Page { buffer: Bytes::from(buffer), mask_bytes, mask_words }
}

// -----------------------------------------------------------------------------------------
// Per-arm runners: single call, consuming the whole page's selection in one shot (groups 1/2
// /4's shape -- matches how R1/R1.5 call every `get_batch_with_dict_selected_*` method).
// -----------------------------------------------------------------------------------------

fn run_cursor(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("cursor: set_data failed");
    let selection =
        PackedSelection::new(&page.mask_bytes, 0, N_TOTAL).expect("cursor: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_cursor(dict, out, selection)
        .expect("cursor: get_batch_with_dict_selected_cursor failed");
    assert_eq!(consumed, N_TOTAL, "cursor: RleDecoder did not consume the whole page");
    written
}

fn run_direct_gather(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("direct_gather: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, N_TOTAL)
        .expect("direct_gather: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_direct_gather(dict, out, selection)
        .expect("direct_gather: get_batch_with_dict_selected_direct_gather failed");
    assert_eq!(consumed, N_TOTAL, "direct_gather: RleDecoder did not consume the whole page");
    written
}

fn run_direct_gather_checked(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("direct_gather_checked: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, N_TOTAL)
        .expect("direct_gather_checked: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_direct_gather_checked(dict, out, selection)
        .expect("direct_gather_checked: get_batch_with_dict_selected_direct_gather_checked failed");
    assert_eq!(consumed, N_TOTAL, "direct_gather_checked: RleDecoder did not consume the whole page");
    written
}

fn run_tiered(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("tiered: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, N_TOTAL)
        .expect("tiered: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_direct_gather_tiered(dict, out, selection)
        .expect("tiered: get_batch_with_dict_selected_direct_gather_tiered failed");
    assert_eq!(consumed, N_TOTAL, "tiered: RleDecoder did not consume the whole page");
    written
}

/// Independent witness (arm C-equivalent): stock `RleDecoder::get_batch::<i32>` in `RLE_CHUNK`
/// -value chunks, then a harness-level selection-word walk gathering `dict[idx]`. Unchanged
/// from `rle_fill.rs`/`rle_fill_r15.rs`'s own `run_arm_c`.
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

/// Independent witness (arm D-equivalent): stock `RleDecoder::get_batch_with_dict::<i64>`,
/// fully materializing every chunk, then the same selection-word walk copying survivors.
/// Unchanged from `rle_fill.rs`/`rle_fill_r15.rs`'s own `run_arm_d`.
fn run_materialize_then_filter(
    decoder: &mut RleDecoder,
    page: &Page,
    dict: &[i64],
    val_buf: &mut [i64; RLE_CHUNK],
    out: &mut Vec<i64>,
) {
    decoder.set_data(page.buffer.clone()).expect("materialize_then_filter: set_data failed");
    let mut processed = 0usize;
    while processed < N_TOTAL {
        let chunk_len = (N_TOTAL - processed).min(RLE_CHUNK);
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
// Per-arm runners: multi-call incremental consumption (group 3's shape).
// -----------------------------------------------------------------------------------------

fn run_multi_call_cursor(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("multi-call cursor: set_data failed");
    let mut chunk_start = 0usize;
    let mut written = 0usize;
    for &chunk_len in &MULTI_CALL_SPLITS {
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

fn run_multi_call_direct_gather(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("multi-call direct_gather: set_data failed");
    let mut chunk_start = 0usize;
    let mut written = 0usize;
    for &chunk_len in &MULTI_CALL_SPLITS {
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
    out: &mut [i64],
) -> usize {
    decoder.set_data(page.buffer.clone()).expect("multi-call direct_gather_checked: set_data failed");
    let mut chunk_start = 0usize;
    let mut written = 0usize;
    for &chunk_len in &MULTI_CALL_SPLITS {
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

fn run_multi_call_tiered(decoder: &mut RleDecoder, page: &Page, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.buffer.clone()).expect("multi-call tiered: set_data failed");
    let mut chunk_start = 0usize;
    let mut written = 0usize;
    for &chunk_len in &MULTI_CALL_SPLITS {
        let selection = PackedSelection::new(&page.mask_bytes, chunk_start, chunk_len)
            .expect("multi-call tiered: PackedSelection::new failed");
        let (consumed, w) = decoder
            .get_batch_with_dict_selected_direct_gather_tiered(dict, &mut out[written..], selection)
            .expect("multi-call tiered: get_batch_with_dict_selected_direct_gather_tiered failed");
        assert_eq!(consumed, chunk_len, "multi-call tiered: did not consume the whole chunk");
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
// Group 1/2/4 (single-call, 5 arms): shared verify-then-time core. `pages`/`dict` are already
// built by the caller, since the three groups construct fixtures differently. Takes `k`
// (needed to construct correctly-widthed `RleDecoder`s -- it has no public bit_width getter to
// recover it from an existing instance).
// -----------------------------------------------------------------------------------------

fn run_single_call_cell(c: &mut Criterion, cell_desc: &str, k: u8, dict: &[i64], pages: &[Page]) {
    let max_selected = pages.iter().map(|p| kernel::popcount_words(&p.mask_words)).max().unwrap_or(0);

    // --- untimed cross-arm correctness check (setup only, never inside a timed closure) ---
    {
        let mut decoder_cursor = RleDecoder::new(k);
        let mut decoder_dg = RleDecoder::new(k);
        let mut decoder_dgc = RleDecoder::new(k);
        let mut decoder_tiered = RleDecoder::new(k);
        let mut decoder_c = RleDecoder::new(k);
        let mut decoder_d = RleDecoder::new(k);
        let mut out_cursor = vec![0i64; max_selected];
        let mut out_dg = vec![0i64; max_selected];
        let mut out_dgc = vec![0i64; max_selected];
        let mut out_tiered = vec![0i64; max_selected];
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut val_buf = [0i64; RLE_CHUNK];
        let mut out_c = Vec::with_capacity(max_selected);
        let mut out_d = Vec::with_capacity(max_selected);

        for (page_idx, page) in pages.iter().enumerate() {
            let written_cursor = run_cursor(&mut decoder_cursor, page, dict, &mut out_cursor);
            let written_dg = run_direct_gather(&mut decoder_dg, page, dict, &mut out_dg);
            let written_dgc = run_direct_gather_checked(&mut decoder_dgc, page, dict, &mut out_dgc);
            let written_tiered = run_tiered(&mut decoder_tiered, page, dict, &mut out_tiered);
            out_c.clear();
            run_decode_all_indices_compact(&mut decoder_c, page, dict, &mut idx_buf, &mut out_c);
            out_d.clear();
            run_materialize_then_filter(&mut decoder_d, page, dict, &mut val_buf, &mut out_d);

            verify_digests(
                "bitpacked_direct_gather",
                cell_desc,
                page_idx,
                &[
                    ("cursor", kernel::fnv1a64(&out_cursor[..written_cursor])),
                    ("direct_gather", kernel::fnv1a64(&out_dg[..written_dg])),
                    ("direct_gather_checked", kernel::fnv1a64(&out_dgc[..written_dgc])),
                    ("tiered", kernel::fnv1a64(&out_tiered[..written_tiered])),
                    ("decode_all_indices_compact", kernel::fnv1a64(&out_c)),
                    ("materialize_then_filter", kernel::fnv1a64(&out_d)),
                ],
            );
        }
    }

    // --- timed loop ---
    let mut group = c.benchmark_group("bitpacked_direct_gather");
    group.sample_size(12);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs_f64(2.5));

    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_cursor(&mut decoder, &pages[0], dict, &mut out);
        group.bench_function(format!("cursor/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_cursor(&mut decoder, page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_direct_gather(&mut decoder, &pages[0], dict, &mut out);
        group.bench_function(format!("direct_gather/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_direct_gather(&mut decoder, page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_direct_gather_checked(&mut decoder, &pages[0], dict, &mut out);
        group.bench_function(format!("direct_gather_checked/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_direct_gather_checked(&mut decoder, page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_tiered(&mut decoder, &pages[0], dict, &mut out);
        group.bench_function(format!("tiered/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_tiered(&mut decoder, page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut out = Vec::with_capacity(max_selected);
        run_decode_all_indices_compact(&mut decoder, &pages[0], dict, &mut idx_buf, &mut out);
        group.bench_function(format!("decode_all_indices_compact/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                out.clear();
                run_decode_all_indices_compact(&mut decoder, page, dict, &mut idx_buf, &mut out);
                hint::black_box(out.as_slice());
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut val_buf = [0i64; RLE_CHUNK];
        let mut out = Vec::with_capacity(max_selected);
        run_materialize_then_filter(&mut decoder, &pages[0], dict, &mut val_buf, &mut out);
        group.bench_function(format!("materialize_then_filter/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                out.clear();
                run_materialize_then_filter(&mut decoder, page, dict, &mut val_buf, &mut out);
                hint::black_box(out.as_slice());
            });
        });
    }
    group.finish();
}

/// Group 3's core: 4 arms (no C/D -- see the module doc comment's group 3 entry), multi-call.
fn run_multi_call_cell(c: &mut Criterion, cell_desc: &str, k: u8, dict: &[i64], pages: &[Page]) {
    let max_selected = pages.iter().map(|p| kernel::popcount_words(&p.mask_words)).max().unwrap_or(0);

    // --- untimed cross-arm correctness check ---
    {
        let mut decoder_cursor = RleDecoder::new(k);
        let mut decoder_dg = RleDecoder::new(k);
        let mut decoder_dgc = RleDecoder::new(k);
        let mut decoder_tiered = RleDecoder::new(k);
        let mut out_cursor = vec![0i64; max_selected];
        let mut out_dg = vec![0i64; max_selected];
        let mut out_dgc = vec![0i64; max_selected];
        let mut out_tiered = vec![0i64; max_selected];

        for (page_idx, page) in pages.iter().enumerate() {
            let written_cursor = run_multi_call_cursor(&mut decoder_cursor, page, dict, &mut out_cursor);
            let written_dg = run_multi_call_direct_gather(&mut decoder_dg, page, dict, &mut out_dg);
            let written_dgc =
                run_multi_call_direct_gather_checked(&mut decoder_dgc, page, dict, &mut out_dgc);
            let written_tiered = run_multi_call_tiered(&mut decoder_tiered, page, dict, &mut out_tiered);

            verify_digests(
                "bitpacked_direct_gather",
                cell_desc,
                page_idx,
                &[
                    ("cursor", kernel::fnv1a64(&out_cursor[..written_cursor])),
                    ("direct_gather", kernel::fnv1a64(&out_dg[..written_dg])),
                    ("direct_gather_checked", kernel::fnv1a64(&out_dgc[..written_dgc])),
                    ("tiered", kernel::fnv1a64(&out_tiered[..written_tiered])),
                ],
            );
        }
    }

    // --- timed loop ---
    let mut group = c.benchmark_group("bitpacked_direct_gather_multi_call");
    group.sample_size(12);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs_f64(2.5));

    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_multi_call_cursor(&mut decoder, &pages[0], dict, &mut out);
        group.bench_function(format!("cursor/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_multi_call_cursor(&mut decoder, page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_multi_call_direct_gather(&mut decoder, &pages[0], dict, &mut out);
        group.bench_function(format!("direct_gather/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_multi_call_direct_gather(&mut decoder, page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_multi_call_direct_gather_checked(&mut decoder, &pages[0], dict, &mut out);
        group.bench_function(format!("direct_gather_checked/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_multi_call_direct_gather_checked(&mut decoder, page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut cursor_pos = 0usize;
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_multi_call_tiered(&mut decoder, &pages[0], dict, &mut out);
        group.bench_function(format!("tiered/{cell_desc}"), |b| {
            b.iter(|| {
                let page = &pages[cursor_pos];
                cursor_pos = (cursor_pos + 1) % pages.len();
                let written = run_multi_call_tiered(&mut decoder, page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    group.finish();
}

// -----------------------------------------------------------------------------------------
// Cell enumeration and top-level driver.
// -----------------------------------------------------------------------------------------

const GROUP1_KS: [u8; 4] = [2, 8, 12, 16];
const GROUP1_LS: [usize; 2] = [8, 512];
const GROUP2_KS: [u8; 2] = [2, 12];
const GROUP3_KS: [u8; 2] = [12, 16];
/// (k, dict_len) pairs for the undersized-dictionary group -- both realistic (not exact powers
/// of 2, both comfortably below `1 << k`).
const GROUP4_CELLS: [(u8, usize); 2] = [(12, 3000), (16, 40_000)];

fn run_full_matrix(c: &mut Criterion) {
    let mut cell_index: u64 = 0;
    let mut next_seed = |salt: u64| {
        let seed = kernel::derive_seed(kernel::derive_seed(TOP_SEED, cell_index), salt);
        cell_index += 1;
        seed
    };

    // Group 1: pure bit-packed, single call, exact dictionary.
    for &k in &GROUP1_KS {
        for &l in &GROUP1_LS {
            let dict_seed = next_seed(0xD1C7);
            let dict = kernel::generate_dict(k as u32, dict_seed);
            assert_eq!(dict.len(), 1usize << k, "fixture invariant: dict.len() == 1<<k");
            let pages: Vec<Page> =
                (0..PAGES_PER_CELL).map(|_| build_pure_bitpacked_page(l, k, next_seed(0xFA6E))).collect();
            let cell_desc = format!("pure/L{l}/k{k}/s{}", SURVIVAL.recip() as u64);
            run_single_call_cell(c, &cell_desc, k, &dict, &pages);
        }
    }

    // Group 2: mixed RLE + bit-packed, single call.
    for &k in &GROUP2_KS {
        let dict_seed = next_seed(0xD1C7);
        let dict = kernel::generate_dict(k as u32, dict_seed);
        assert_eq!(dict.len(), 1usize << k, "fixture invariant: dict.len() == 1<<k");
        let pages: Vec<Page> =
            (0..PAGES_PER_CELL).map(|_| build_mixed_page(k, next_seed(0xFA6E))).collect();
        let cell_desc = format!("mixed/k{k}/s{}", SURVIVAL.recip() as u64);
        run_single_call_cell(c, &cell_desc, k, &dict, &pages);
    }

    // Group 3: multi-call incremental consumption.
    for &k in &GROUP3_KS {
        let dict_seed = next_seed(0xD1C7);
        let dict = kernel::generate_dict(k as u32, dict_seed);
        assert_eq!(dict.len(), 1usize << k, "fixture invariant: dict.len() == 1<<k");
        let pages: Vec<Page> =
            (0..PAGES_PER_CELL).map(|_| build_single_run_page(k, next_seed(0xFA6E))).collect();
        let cell_desc = format!("multicall/k{k}");
        run_multi_call_cell(c, &cell_desc, k, &dict, &pages);
    }

    // Group 4: undersized dictionary (forces direct_gather's fallback; exercises
    // direct_gather_checked's checked lookup against a real out-of-power-of-2 dictionary size).
    for &(k, dict_len) in &GROUP4_CELLS {
        let dict_seed = next_seed(0xD1C7);
        let full_dict = kernel::generate_dict(k as u32, dict_seed);
        assert_eq!(full_dict.len(), 1usize << k, "fixture invariant: dict.len() == 1<<k before truncation");
        let mut dict = full_dict;
        dict.truncate(dict_len);
        assert_eq!(dict.len(), dict_len, "fixture invariant: truncated dict.len() == dict_len");
        let pages: Vec<Page> = (0..PAGES_PER_CELL)
            .map(|_| build_pure_bitpacked_page_bounded(512, k, dict_len, next_seed(0xFA6E)))
            .collect();
        let cell_desc = format!("undersized/k{k}/dictlen{dict_len}");
        run_single_call_cell(c, &cell_desc, k, &dict, &pages);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    run_full_matrix(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
