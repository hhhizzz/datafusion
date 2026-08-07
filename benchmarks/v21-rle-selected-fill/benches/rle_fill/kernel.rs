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

//! Pure-logic kernel module for the `v21-rle-selected-fill` microbench (experiment
//! `arrow-rle-selected-fill-v21`, R1 stage).
//!
//! Dependency-free (`std` only) by design: this file is shared byte-for-byte between the
//! DataFusion-worktree bench crate (`benchmarks/v21-rle-selected-fill`) and the
//! zero-dependency scratch dev-check crate (`.scratch/v21-r1-devcheck`), because the
//! DataFusion worktree and arrow-rs trees can never be compiled locally under this program's
//! rules -- the scratch crate is the only way to exercise this logic before the real (K8s-only)
//! run. Do not add any `parquet`/`criterion`/`bytes` imports here.
//!
//! ## Scope
//!
//! This module only builds *fixtures* (encoded pure-RLE dictionary-index streams, selection
//! masks, dictionaries) plus the cross-arm correctness digest. Unlike a kernel module that
//! reimplements a decode algorithm under test, R1's timed arms A/A' call directly into the
//! already-patched `parquet::encodings::rle::{RleDecoder, PackedSelection}` (see the harness
//! file `benches/rle_fill.rs`, which is *not* dependency-free), so there is no decode kernel
//! to host here -- only what is needed to build inputs for it and check its output.
//!
//! ## Pure-RLE run byte layout
//!
//! Confirmed directly against the pinned arrow-rs commit
//! `ed92960c8a85eda657fce3525c905616ccc5a983` (`parquet/src/encodings/rle.rs`'s
//! `RleDecoder::reload`, and `parquet/src/util/bit_util.rs`'s `BitWriter::put_vlq_int` /
//! `BitReader::get_vlq_int` / `BitReader::get_aligned` / `read_num_bytes`). Each RLE run is:
//!
//! 1. a ULEB128 ("VLQ") varint header encoding `run_length << 1` -- 7 bits per byte,
//!    least-significant group first, the high bit (`0x80`) of each byte is the continuation
//!    flag (set on every byte but the last). The decoded header's low bit clear signals "this
//!    is an RLE run" (a bit-packed run's header has the low bit set instead); this is exactly
//!    what `RleDecoder::reload` branches on;
//! 2. followed by `ceil(bit_width / 8)` bytes holding the repeated value, little-endian,
//!    zero-padded/truncated to that width (`BitReader::get_aligned::<u64>` reads exactly this
//!    many bytes, byte-aligned, via `read_num_bytes`, which is explicitly documented as
//!    interpreting its input as "little-endian order").
//!
//! There is no overall stream length prefix and no inter-run padding: `RleDecoder` determines
//! its own end from the encoded `Bytes`' length and the run structure it walks (an empty
//! `get_vlq_int` read, once the buffer is exhausted, signals EOF).

use std::collections::HashSet;

// =========================================================================================
// PRNG: xorshift64* (Vigna; public domain) + splitmix64-style seed derivation.
// =========================================================================================

/// Xorshift64* pseudo-random generator (Vigna; public domain algorithm, fixed multiplier
/// `0x2545F4914F6CDD1D`). Requires a nonzero internal state, so a zero seed is remapped to a
/// fixed nonzero constant.
pub struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    /// Next 64-bit pseudo-random word.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform pseudo-random `f64` in `[0, 1)`, built from the top 53 bits of a `next_u64`
    /// draw (standard double-precision-mantissa technique).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let top53 = self.next_u64() >> 11;
        top53 as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Derives a deterministic child seed from `base_seed` and an integer `index` via one
/// splitmix64 mixing round (Steele/Lea/Vigna; public domain). Used to fan a single top-level
/// seed constant out into per-cell / per-page seeds reproducibly.
#[inline]
pub fn derive_seed(base_seed: u64, index: u64) -> u64 {
    let mut z = base_seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// =========================================================================================
// Dictionary generation.
// =========================================================================================

/// Generates `1 << bit_width` distinct pseudo-random `i64` dictionary values.
pub fn generate_dict(bit_width: u32, seed: u64) -> Vec<i64> {
    let len = 1usize << bit_width;
    let mut rng = Xorshift64Star::new(seed);
    let mut seen = HashSet::with_capacity(len);
    let mut dict = Vec::with_capacity(len);
    while dict.len() < len {
        let v = rng.next_u64() as i64;
        if seen.insert(v) {
            dict.push(v);
        }
    }
    dict
}

// =========================================================================================
// Pure-RLE run encoding.
// =========================================================================================

/// Appends `v` to `out` as an unsigned ULEB128/VLQ varint: 7 bits per byte, least-significant
/// group first, high bit of each byte set on every byte but the last (continuation flag).
pub fn write_uleb128(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
}

/// Byte width of the fixed-width repeated-value field for a given `bit_width`:
/// `ceil(bit_width / 8)`.
#[inline]
pub fn rle_value_byte_width(bit_width: u8) -> usize {
    (bit_width as usize).div_ceil(8)
}

/// Appends one pure-RLE run to `out`: a ULEB128 header `(run_length << 1)` (low bit clear
/// signals RLE) followed by `rle_value_byte_width(bit_width)` bytes holding `value`,
/// little-endian. `value` must fit in `bit_width` bits (`value < 1 << bit_width`); the only
/// caller in this module ([`build_rle_page`]) guarantees this by construction.
pub fn write_rle_run(out: &mut Vec<u8>, run_length: u64, value: u64, bit_width: u8) {
    write_uleb128(out, run_length << 1);
    let width = rle_value_byte_width(bit_width);
    let le = value.to_le_bytes();
    out.extend_from_slice(&le[..width]);
}

/// Builds one page's worth of pure-RLE-encoded dictionary indices: `n_total / run_length`
/// consecutive RLE runs, each `run_length` values long, each carrying an independently random
/// dictionary index drawn uniformly from `[0, 1 << bit_width)`. `n_total % run_length` must be
/// `0` (the R1 fixture grid guarantees this: every frozen run length is a power of two no
/// larger than `n_total`).
pub fn build_rle_page(
    n_total: usize,
    run_length: usize,
    bit_width: u8,
    rng: &mut Xorshift64Star,
) -> Vec<u8> {
    debug_assert_eq!(n_total % run_length, 0, "n_total must be a whole multiple of run_length");
    let num_runs = n_total / run_length;
    let index_mask = (1u64 << bit_width) - 1;
    let mut out = Vec::new();
    for _ in 0..num_runs {
        let index = rng.next_u64() & index_mask;
        write_rle_run(&mut out, run_length as u64, index, bit_width);
    }
    out
}

// =========================================================================================
// Selection-mask generation. Masks are packed as `ceil(n_values/64)` `u64` words: bit `v` of
// logical position `v` lives at `words[v/64]` bit `v%64` (LSB-first within the word); any
// bits at or beyond `n_values` in the final word are left zero.
// =========================================================================================

/// iid Bernoulli(`survival`) mask over `n_values` logical positions.
pub fn generate_random_mask(n_values: usize, survival: f64, rng: &mut Xorshift64Star) -> Vec<u64> {
    let mut words = vec![0u64; n_values.div_ceil(64)];
    for v in 0..n_values {
        if rng.next_f64() < survival {
            words[v / 64] |= 1u64 << (v % 64);
        }
    }
    words
}

/// Clustered mask via a 2-state Markov chain (state = "currently selected"), tuned so the
/// mean length of a selected run is `target_mean_run` and the overall density is `survival`.
///
/// Derivation: fixing `p_leave = 1 / target_mean_run` makes the "selected" sojourn length
/// geometric with mean `target_mean_run`. The chain's stationary "selected" probability is
/// `p_enter / (p_enter + p_leave)`; solving that equal to `survival` gives
/// `p_enter = survival / (target_mean_run * (1 - survival))`. The initial state is drawn from
/// the stationary distribution so the mask carries no warm-up transient at the start of the
/// page.
pub fn generate_clustered_mask(
    n_values: usize,
    survival: f64,
    target_mean_run: f64,
    rng: &mut Xorshift64Star,
) -> Vec<u64> {
    debug_assert!((0.0..1.0).contains(&survival), "clustered shape needs survival in [0, 1)");
    debug_assert!(target_mean_run >= 1.0);
    let p_leave = 1.0 / target_mean_run;
    let p_enter = survival / (target_mean_run * (1.0 - survival));

    let mut words = vec![0u64; n_values.div_ceil(64)];
    let mut selected = rng.next_f64() < survival;
    for v in 0..n_values {
        let flip_prob = if selected { p_leave } else { p_enter };
        if rng.next_f64() < flip_prob {
            selected = !selected;
        }
        if selected {
            words[v / 64] |= 1u64 << (v % 64);
        }
    }
    words
}

/// The dense "select everything" control mask: all `n_values` positions selected, with any
/// trailing bits in the final word (beyond `n_values`) left zero.
pub fn generate_dense_mask(n_values: usize) -> Vec<u64> {
    let num_words = n_values.div_ceil(64);
    let mut words = vec![u64::MAX; num_words];
    let tail = n_values % 64;
    if tail != 0 {
        words[num_words - 1] = (1u64 << tail) - 1;
    }
    words
}

/// Returns whether logical position `v` is set in a word-packed mask (see the module-level
/// convention comment above). Used by tests and by the harness's untimed digest check; the
/// harness's *timed* code paths use their own inlined trailing-zeros bit-walks for speed.
#[inline]
pub fn mask_bit(words: &[u64], v: usize) -> bool {
    (words[v / 64] >> (v % 64)) & 1 != 0
}

/// Total number of set bits across a word-packed mask.
pub fn popcount_words(words: &[u64]) -> usize {
    words.iter().map(|w| w.count_ones() as usize).sum()
}

/// Repacks a word-packed mask (see the module-level convention comment above) into the exact
/// byte layout `parquet::encodings::rle::PackedSelection` expects: bit `v` of logical position
/// `v` at `bytes[v/8]` bit `v%8` (LSB-first within the byte). A `u64` word's little-endian byte
/// representation already places bit `i` of the word at byte `i/8` bit `i%8` of that 8-byte
/// span, so this is exactly `words[i].to_le_bytes()` concatenated in order -- no bit shuffling
/// needed. Always returns `words.len() * 8` bytes (a whole number of words), which may be a
/// few bytes longer than `ceil(n_values/8)`; any such extra bytes are the mask's own zeroed
/// word-tail padding and are never observed by a caller that only queries logical positions
/// `< n_values`.
pub fn words_to_packed_bytes(words: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 8);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

// =========================================================================================
// FNV-1a 64 digest (cross-arm correctness check).
// =========================================================================================

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

/// FNV-1a 64 digest over an `i64` stream, in order (each value hashed as its 8 little-endian
/// bytes).
pub fn fnv1a64(values: &[i64]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &v in values {
        for byte in v.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

// =========================================================================================
// Bit-packed run encoding (experiment v23, `arrow-bitpacked-direct-gather-gate-v23`).
//
// Same run-header framing as the pure-RLE encoder above (see this module's doc comment): a
// bit-packed run's ULEB128 header is `(num_groups << 1) | 1` (low bit *set*, unlike an RLE
// run's clear low bit), followed by `num_groups * 8` values flat-packed at `bit_width` bits
// each, LSB-first: value `i` (0-indexed within the run) occupies bit range
// `[i*bit_width, i*bit_width+bit_width)` of the payload, flat bit `b` at byte `b/8` bit `b%8`.
// Confirmed against the same pinned commit as the RLE encoder above
// (`RleDecoder::reload`'s bit-packed arm and `BitReader::get_batch`).
// =========================================================================================

/// Appends one bit-packed run's header and payload to `out`. `values.len()` must be a multiple
/// of 8 (the hybrid format's own group-of-8 requirement); every `values[i]` must be `< 1 <<
/// bit_width` (not enforced here -- the only callers in this module mask by construction).
pub fn write_bit_packed_run(out: &mut Vec<u8>, values: &[u64], bit_width: u8) {
    assert_eq!(values.len() % 8, 0, "bit-packed runs group values in multiples of 8");
    let num_groups = (values.len() / 8) as u64;
    write_uleb128(out, (num_groups << 1) | 1);
    let k = bit_width as u64;
    let total_bits = values.len() as u64 * k;
    let start = out.len();
    out.resize(start + total_bits.div_ceil(8) as usize, 0u8);
    let mut bit_pos: u64 = 0;
    for &v in values {
        debug_assert!(k == 64 || v < (1u64 << k), "value does not fit in bit_width bits");
        let mut remaining = k;
        let mut src = v;
        let mut b = bit_pos;
        while remaining > 0 {
            let byte_idx = start + (b / 8) as usize;
            let bit_in_byte = (b % 8) as u32;
            let space = 8 - bit_in_byte;
            let take = remaining.min(space as u64) as u32;
            let chunk = (src & ((1u64 << take) - 1)) as u8;
            out[byte_idx] |= chunk << bit_in_byte;
            src >>= take;
            remaining -= take as u64;
            b += take as u64;
        }
        bit_pos += k;
    }
}

/// Generates `n_values` pseudo-random dictionary indices, each uniform in `[0, 1 << bit_width)`,
/// rounded up to a multiple of 8 with zero-padding (see [`write_bit_packed_run`]'s group-of-8
/// requirement) -- the padding values are never selected by any mask this module builds for
/// exactly `n_values` logical positions, since [`generate_random_mask`]/friends zero every bit
/// at or beyond `n_values`.
pub fn generate_bitpacked_values(n_values: usize, bit_width: u8, rng: &mut Xorshift64Star) -> Vec<u64> {
    let index_mask = if bit_width >= 64 { u64::MAX } else { (1u64 << bit_width) - 1 };
    let padded = n_values.div_ceil(8) * 8;
    (0..padded).map(|i| if i < n_values { rng.next_u64() & index_mask } else { 0 }).collect()
}

/// Like [`generate_bitpacked_values`], but every produced value is uniform in `[0, dict_len)`
/// rather than the run's full `[0, 1 << bit_width)` code space. Models a dictionary smaller
/// than its run's declared bit width -- the common case for a real Parquet writer, which picks
/// the smallest `bit_width` with `dict_len <= 1 << bit_width`, so `dict_len` is an exact power
/// of 2 only by coincidence. Requires `dict_len <= 1 << bit_width` (asserted) and `dict_len >=
/// 1`, so every produced value still fits in `bit_width` bits.
pub fn generate_bitpacked_values_bounded(
    n_values: usize,
    bit_width: u8,
    dict_len: usize,
    rng: &mut Xorshift64Star,
) -> Vec<u64> {
    assert!(dict_len >= 1, "dict_len must be at least 1");
    assert!(
        bit_width >= 64 || dict_len as u64 <= (1u64 << bit_width),
        "dict_len must fit within bit_width bits"
    );
    let padded = n_values.div_ceil(8) * 8;
    (0..padded).map(|i| if i < n_values { rng.next_u64() % dict_len as u64 } else { 0 }).collect()
}
