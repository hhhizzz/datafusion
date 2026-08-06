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

//! Pure-logic kernel module for the `paper-select-fourarm` microbench (experiment
//! `arrow-paper-select-fourarm-v18`).
//!
//! This file is intentionally dependency-free (only `std`): it is shared byte-for-byte
//! between the DataFusion-worktree bench crate (`benchmarks/paper-select-fourarm`) and the
//! zero-dependency scratch dev-check crate (`.scratch/paper-select-fourarm-devcheck`) used to
//! unit-test it locally. Do not add any `parquet`/`criterion`/`arrow` imports here.
//!
//! Contents:
//! - Arm A (`paper_pext`): [`select_run_bmi2`] (x86_64 + BMI2 only) and its portable
//!   ground-truth/fallback twin [`select_run_scalar`].
//! - Arm B (`sparse_direct`): [`sparse_direct`].
//! - Fixture generation: an inline xorshift64* PRNG, deterministic per-page seed derivation,
//!   dictionary generation, Parquet-shaped bit-packing / RLE-hybrid run encoding, and
//!   random/clustered/dense selection-mask generation.
//! - [`fnv1a64`]: the cross-arm correctness digest.
//!
//! ## Bit-packing layout
//!
//! Values are packed LSB-first into a little-endian byte stream, exactly matching Parquet's
//! `BitWriter`/`BitReader` (see `parquet/src/util/bit_util.rs` at the pinned arrow-rs commit
//! `ed92960c8a85eda657fce3525c905616ccc5a983`): value `i` (0-indexed within a run) occupies
//! flat bit range `[i*k, i*k+k)` of the run's payload, where flat bit `b` lives at byte
//! `b/8`, bit `b%8` (LSB-first within the byte). A run's packed payload byte length is always
//! exactly `n_values*k/8` (no partial-byte padding within the run itself), because callers
//! guarantee `n_values % 8 == 0`.

use std::collections::HashSet;

/// Maximum supported dictionary bit width. The paper sweeps `k` in `1..=16`; this bound sizes
/// the fixed-size stack buffers used by the BMI2 frame loop.
pub const MAX_K: usize = 16;

/// Upper bound on how many `u64` words the sparse path's compacted-bit accumulator can flush
/// for a single frame: at most `frame_values - 1 <= 63` selected values at up to `MAX_K` bits
/// each is `63 * 16 = 1008` bits, i.e. at most 16 flushed words plus one final partial word.
/// Rounded up generously.
pub const MAX_COMPACTED_WORDS: usize = 20;

// ---------------------------------------------------------------------------------------
// PRNG: inline xorshift64* (Vigna) + splitmix64-style deterministic seed derivation.
// ---------------------------------------------------------------------------------------

/// Inline xorshift64* PRNG (Vigna). Public-domain algorithm; fixed multiplier
/// `0x2545F4914F6CDD1D`. Requires a nonzero seed (zero is remapped to a fixed nonzero
/// constant so callers never have to special-case it).
pub struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// A pseudo-random `f64` uniform in `[0, 1)`, built from the top 53 bits of a
    /// `next_u64()` draw (standard double-precision-mantissa technique).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let top53 = self.next_u64() >> 11;
        (top53 as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

/// Deterministically derives a page-level (or cell-level) seed from a base seed and an
/// integer index, via one splitmix64 mixing step (Steele/Lea/Vigna; public domain). This is
/// the "splitmix-style seed derivation" used so the whole fixture matrix is reproducible from
/// a single top-level seed constant.
#[inline]
pub fn derive_seed(base_seed: u64, index: u64) -> u64 {
    let mut z = base_seed.wrapping_add(index.wrapping_mul(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------------------
// Dictionary generation.
// ---------------------------------------------------------------------------------------

/// Generates `1 << k` distinct `i64` dictionary values via [`Xorshift64Star`].
pub fn generate_dict(k: u32, seed: u64) -> Vec<i64> {
    let len = 1usize << k;
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

// ---------------------------------------------------------------------------------------
// Parquet-shaped bit-packing / RLE-hybrid bit-packed-run encoding.
// ---------------------------------------------------------------------------------------

/// Appends `v` to `buf` as an unsigned LEB128 / VLQ varint (matches Parquet's
/// `BitWriter::put_vlq_int`: 7 bits per byte, LSB-first, high bit of each byte is the
/// continuation flag).
pub fn write_uvarint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            buf.push(byte | 0x80);
        } else {
            buf.push(byte);
            break;
        }
    }
}

/// Bit-packs `values` (each masked to its low `k` bits) LSB-first into `out`, appending
/// exactly `values.len() * k / 8` bytes. Requires `values.len() * k` to be a multiple of 8
/// (guaranteed whenever `values.len() % 8 == 0`, which all callers here ensure).
pub fn pack_values_into(out: &mut Vec<u8>, values: &[u32], k: u32) {
    let total_bits = values.len() as u64 * k as u64;
    debug_assert_eq!(total_bits % 8, 0, "n_values * k must be a whole number of bytes");
    let start = out.len();
    out.resize(start + (total_bits / 8) as usize, 0u8);
    let mask = if k == 0 { 0 } else { (1u64 << k) - 1 };
    let mut pos: u64 = 0;
    for &v in values {
        let mut val = (v as u64) & mask;
        let mut remaining = k;
        let mut p = pos;
        while remaining > 0 {
            let byte_idx = start + (p / 8) as usize;
            let bit_in_byte = (p % 8) as u32;
            let space = 8 - bit_in_byte;
            let take = remaining.min(space);
            let bits = (val & ((1u64 << take) - 1)) as u8;
            out[byte_idx] |= bits << bit_in_byte;
            val >>= take;
            p += take as u64;
            remaining -= take;
        }
        pos += k as u64;
    }
}

/// Builds one Parquet-shaped "bit-packed run" of the RLE/bit-packing hybrid encoding: a
/// LEB128 bit-packed-run header `((n_values/8) << 1) | 1` followed by the packed payload.
/// `values.len()` must be a multiple of 8. Returns the byte offset within `out` where the
/// packed payload begins (right after the header) -- this is the `byte_offset` that
/// `select_run_*`/`sparse_direct` expect.
pub fn write_bitpacked_run(out: &mut Vec<u8>, values: &[u32], k: u32) -> usize {
    debug_assert_eq!(values.len() % 8, 0, "bit-packed runs group values in multiples of 8");
    let num_groups = (values.len() / 8) as u64;
    let indicator = (num_groups << 1) | 1;
    write_uvarint(out, indicator);
    let payload_offset = out.len();
    pack_values_into(out, values, k);
    payload_offset
}

// ---------------------------------------------------------------------------------------
// Selection-mask generation.
// ---------------------------------------------------------------------------------------

/// Generates an iid Bernoulli(`survival`) selection mask over `n_values` positions, packed
/// as `ceil(n_values/64)` words (bit `v` of `sel_words[v/64]` at position `v%64`; bits beyond
/// `n_values` in the final word are left zero).
pub fn generate_random_mask(n_values: usize, survival: f64, rng: &mut Xorshift64Star) -> Vec<u64> {
    let mut sel_words = vec![0u64; n_values.div_ceil(64)];
    for v in 0..n_values {
        if rng.next_f64() < survival {
            sel_words[v / 64] |= 1u64 << (v % 64);
        }
    }
    sel_words
}

/// Generates a clustered selection mask via a 2-state Markov chain (state = "currently
/// selected?"), tuned so the mean selected-run length is ~64 and the overall density is
/// `survival`. Derivation: fix `p_leave = 1/64` (mean run length in the "selected" state is
/// `1/p_leave = 64`, geometric-distribution mean); the stationary density of state "selected"
/// is `p_enter / (p_enter + p_leave)`, so solving for the stationary density to equal
/// `survival` gives `p_enter = survival / (64 * (1 - survival))`.
pub fn generate_clustered_mask(
    n_values: usize,
    survival: f64,
    rng: &mut Xorshift64Star,
) -> Vec<u64> {
    debug_assert!((0.0..1.0).contains(&survival), "clustered shape needs survival in [0, 1)");
    let mut sel_words = vec![0u64; n_values.div_ceil(64)];
    let p_leave = 1.0 / 64.0;
    let p_enter = survival / (64.0 * (1.0 - survival));
    // Seed the initial state from the stationary distribution so the mask doesn't have a
    // warm-up transient at the start of the page.
    let mut selected = rng.next_f64() < survival;
    for v in 0..n_values {
        let flip_prob = if selected { p_leave } else { p_enter };
        if rng.next_f64() < flip_prob {
            selected = !selected;
        }
        if selected {
            sel_words[v / 64] |= 1u64 << (v % 64);
        }
    }
    sel_words
}

/// Generates the "dense" control mask: every one of `n_values` positions selected, with any
/// trailing bits in the final word (beyond `n_values`) zeroed.
pub fn generate_dense_mask(n_values: usize) -> Vec<u64> {
    let num_words = n_values.div_ceil(64);
    let mut sel_words = vec![u64::MAX; num_words];
    let tail = n_values % 64;
    if tail != 0 {
        sel_words[num_words - 1] = (1u64 << tail) - 1;
    }
    sel_words
}

/// Extracts `len` consecutive logical bits starting at logical bit `start` from the flat
/// bit-array `words` (bit `i` at `words[i/64]`, bit `i%64`), and repacks them into a fresh
/// 0-based flat bit array of `ceil(len/64)` words (any trailing bits in the final word beyond
/// `len` are left zero). Used to slice a page-level selection mask into per-run masks when a
/// page is encoded as many separate bit-packed runs (the `writer_real` guard cell), since run
/// boundaries are generally not aligned to 64-value word boundaries.
pub fn extract_bit_range(words: &[u64], start: usize, len: usize) -> Vec<u64> {
    let mut out = vec![0u64; len.div_ceil(64)];
    for i in 0..len {
        let src_bit = start + i;
        let bit = (words[src_bit / 64] >> (src_bit % 64)) & 1;
        if bit != 0 {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------
// FNV-1a 64 digest (cross-arm correctness check).
// ---------------------------------------------------------------------------------------

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

/// FNV-1a 64 digest over the selected `i64` output stream, in order (each value hashed as its
/// 8 little-endian bytes).
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

// ---------------------------------------------------------------------------------------
// Shared low-level unaligned read helper.
// ---------------------------------------------------------------------------------------

/// Reads a `u64` from `payload` at byte offset `byte_pos`, little-endian, with no alignment
/// requirement.
///
/// # Safety
/// The caller must guarantee `byte_pos + 8 <= payload.len()`. All call sites in this module
/// rely on the fixture contract that `payload` carries at least 8 bytes of padding beyond the
/// run's last meaningful byte, so every unaligned 8-byte read this module performs (even ones
/// that "overshoot" into padding, e.g. inside a tail frame or right before a straddled value)
/// stays in-bounds.
#[inline]
unsafe fn unaligned_read_u64(payload: &[u8], byte_pos: usize) -> u64 {
    debug_assert!(byte_pos + 8 <= payload.len());
    // SAFETY: see function doc; caller-guaranteed padding keeps this in-bounds.
    unsafe { std::ptr::read_unaligned(payload.as_ptr().add(byte_pos) as *const u64) }
}

// ---------------------------------------------------------------------------------------
// Arm B: sparse_direct.
// ---------------------------------------------------------------------------------------

/// Arm B (`sparse_direct`): per selection word, walk set bits (`trailing_zeros` +
/// clear-lowest-set-bit), and for each selected absolute value index `v` compute
/// `bit_offset = byte_offset*8 + v*k`, do one unaligned `u64` load at `bit_offset/8` shifted
/// right by `bit_offset%8` and masked to `k` bits, then gather `dict[code]`. O(selected) work,
/// no per-frame structure.
///
/// `k` is derived from `dict.len()` (`dict.len() == 1 << k` is a caller guarantee, asserted
/// once here in debug builds).
pub fn sparse_direct(
    payload: &[u8],
    byte_offset: usize,
    n_values: usize,
    sel_words: &[u64],
    dict: &[i64],
    out: &mut Vec<i64>,
) {
    debug_assert_eq!(n_values % 8, 0);
    debug_assert!(dict.len().is_power_of_two());
    debug_assert_eq!(sel_words.len(), n_values.div_ceil(64));
    let k = dict.len().trailing_zeros();
    let mask = if k == 0 { 0 } else { (1u64 << k) - 1 };

    for (word_idx, &orig_word) in sel_words.iter().enumerate() {
        let mut word = orig_word;
        let base_v = word_idx * 64;
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            word &= word - 1; // clear lowest set bit
            let v = base_v + bit;
            if v >= n_values {
                continue; // defensive: guaranteed unreachable given the zeroed-tail contract
            }
            let bit_offset = byte_offset as u64 * 8 + (v as u64) * (k as u64);
            let byte_pos = (bit_offset / 8) as usize;
            let shift = (bit_offset % 8) as u32;
            // SAFETY: byte_pos+8 <= payload.len() because the fixture guarantees >=8 bytes
            // of padding beyond the run's last byte, and v < n_values bounds bit_offset to
            // within the run's own bits.
            let word_bits = unsafe { unaligned_read_u64(payload, byte_pos) };
            let code = (word_bits >> shift) & mask;
            // SAFETY: dict.len() == 1<<k (debug-asserted above) and code is a k-bit value.
            out.push(unsafe { *dict.get_unchecked(code as usize) });
        }
    }
}

// ---------------------------------------------------------------------------------------
// Arm A: select_run_scalar (portable ground-truth twin / non-x86_64 fallback).
// ---------------------------------------------------------------------------------------

/// Arm A's portable scalar twin: same API/semantics as [`select_run_bmi2`], implemented with
/// per-value shift/mask extraction driven by iterating set bits in each frame's selection
/// word -- no intrinsics. Used for correctness cross-checking, as the non-x86_64 fallback,
/// and as the ground truth the BMI2 path must match bit-for-bit. Not a timed arm on x86_64.
pub fn select_run_scalar(
    payload: &[u8],
    byte_offset: usize,
    n_values: usize,
    sel_words: &[u64],
    dict: &[i64],
    out: &mut Vec<i64>,
) {
    debug_assert_eq!(n_values % 8, 0);
    debug_assert!(dict.len().is_power_of_two());
    debug_assert_eq!(sel_words.len(), n_values.div_ceil(64));
    let k = dict.len().trailing_zeros();
    let mask = if k == 0 { 0 } else { (1u64 << k) - 1 };

    let frame_stride = 8usize * k as usize;
    let full_frames = n_values / 64;
    let tail = n_values % 64;

    let mut frame_byte = byte_offset;
    for frame_idx in 0..full_frames {
        let sel = sel_words[frame_idx];
        if sel != 0 {
            extract_selected_from_frame(payload, frame_byte, sel, k, mask, dict, out);
        }
        frame_byte += frame_stride;
    }
    if tail > 0 {
        // Already masked to valid bits by the caller.
        let sel = sel_words[full_frames];
        if sel != 0 {
            extract_selected_from_frame(payload, frame_byte, sel, k, mask, dict, out);
        }
    }
}

#[inline]
fn extract_selected_from_frame(
    payload: &[u8],
    frame_byte: usize,
    mut sel: u64,
    k: u32,
    mask: u64,
    dict: &[i64],
    out: &mut Vec<i64>,
) {
    while sel != 0 {
        let v = sel.trailing_zeros() as u64;
        sel &= sel - 1; // clear lowest set bit
        let bit_pos = v * k as u64;
        let byte_pos = frame_byte + (bit_pos / 8) as usize;
        let shift = (bit_pos % 8) as u32;
        // SAFETY: byte_pos+8 <= payload.len() given the fixture's padding guarantee; v is a
        // valid in-frame value index (bounded by the frame's selection word).
        let word = unsafe { unaligned_read_u64(payload, byte_pos) };
        let code = (word >> shift) & mask;
        // SAFETY: dict.len() == 1<<k (debug-asserted by the caller) and code is a k-bit value.
        out.push(unsafe { *dict.get_unchecked(code as usize) });
    }
}

// ---------------------------------------------------------------------------------------
// Arm A: select_run_bmi2 (x86_64 + BMI2 only).
// ---------------------------------------------------------------------------------------

/// Precomputes, for a given `k`, the "cyclic phase code-start masks" and matching starting
/// value-index for each of the (at most `k`) input words in a 64-value frame.
///
/// For word `w` (0-indexed within a frame), `phase_masks[w]` is a 64-bit mask over local bit
/// positions `p` such that `(w*64 + p) % k == 0`, i.e. the bit positions at which some value's
/// k-bit code *starts*. `j_lo[w] = ceil(w*64 / k)` is the index (0..63) of the first value
/// whose code starts in word `w` (the value at `phase_masks[w]`'s lowest set bit). Because a
/// full frame is exactly `k` words for 64 values, this mapping depends only on `w` and `k`, so
/// it is computed once per `select_run_bmi2` call and reused for every frame in the run.
#[cfg(target_arch = "x86_64")]
fn compute_phase_tables(k: u32) -> ([u64; MAX_K], [u64; MAX_K]) {
    let mut phase_masks = [0u64; MAX_K];
    let mut j_lo = [0u64; MAX_K];
    let k64 = k as u64;
    for w in 0..k as usize {
        let mut mask = 0u64;
        for p in 0..64u64 {
            if (w as u64 * 64 + p) % k64 == 0 {
                mask |= 1u64 << p;
            }
        }
        phase_masks[w] = mask;
        j_lo[w] = (w as u64 * 64).div_ceil(k64);
    }
    (phase_masks, j_lo)
}

/// Sequentially extracts `count` consecutive `k`-bit codes from `words` (flat LSB-first
/// bit-packing, straddling a word boundary exactly like the top-level run payload) and
/// gathers `dict[code]` into `out` for each, in order.
///
/// # Safety
/// `dict.len() == 1 << k` must hold (debug-asserted by callers upstream); every extracted
/// `code` is a `k`-bit value and therefore `< dict.len()`.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn extract_sequential_codes(words: &[u64], count: usize, k: u32, dict: &[i64], out: &mut Vec<i64>) {
    let mask = if k == 0 { 0 } else { (1u64 << k) - 1 };
    let mut bit_pos: u64 = 0;
    for _ in 0..count {
        let word_idx = (bit_pos / 64) as usize;
        let bit_off = bit_pos % 64;
        let lo = words[word_idx] >> bit_off;
        let code = if bit_off + k as u64 > 64 {
            let hi = words[word_idx + 1] << (64 - bit_off);
            (lo | hi) & mask
        } else {
            lo & mask
        };
        // SAFETY: see function doc.
        out.push(unsafe { *dict.get_unchecked(code as usize) });
        bit_pos += k as u64;
    }
}

/// Processes one frame (64 values, or a final tail of `n_values % 64` values) of the BMI2
/// kernel: `sel == 0` advances only (no output); `sel == full_mask` (every valid bit in this
/// frame set) takes the dense fast path (plain shift/mask unpack of every value); otherwise
/// takes the sparse PEXT path.
///
/// # Safety
/// Requires the `bmi2` target feature (uses `_pext_u64`/`_pdep_u64`). `frame_byte` plus the
/// bytes needed for this frame's words must have at least 8 bytes of padding beyond the run's
/// last meaningful byte (fixture guarantee). `dict.len() == 1 << k`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[inline]
unsafe fn process_frame_bmi2(
    payload: &[u8],
    frame_byte: usize,
    frame_values: usize,
    sel: u64,
    k: u32,
    phase_masks: &[u64; MAX_K],
    j_lo: &[u64; MAX_K],
    dict: &[i64],
    out: &mut Vec<i64>,
) {
    if sel == 0 {
        return;
    }
    debug_assert!(frame_values <= 64 && frame_values % 8 == 0 && frame_values > 0);

    let full_mask = if frame_values == 64 {
        u64::MAX
    } else {
        (1u64 << frame_values) - 1
    };
    let words_needed = ((frame_values as u64) * (k as u64)).div_ceil(64) as usize;

    let mut words = [0u64; MAX_K];
    for (w, slot) in words.iter_mut().enumerate().take(words_needed) {
        // SAFETY: fixture guarantees >=8 bytes of padding beyond the run's last byte, and
        // words_needed is exactly the number of words this frame's valid bits span (proved:
        // total valid bits in this frame = frame_values*k, which never exceeds
        // words_needed*64 by more than 63 bits, and the caller-guaranteed padding covers the
        // overshoot even for the very last frame of the very last run in a buffer).
        *slot = unsafe { unaligned_read_u64(payload, frame_byte + w * 8) };
    }

    if sel == full_mask {
        // Dense fast path: unpack every value with a plain shift/mask loop (no PEXT needed).
        // SAFETY: dict.len() == 1<<k upheld by caller; words[..words_needed] holds exactly
        // this frame's packed bits.
        unsafe { extract_sequential_codes(&words[..words_needed], frame_values, k, dict, out) };
        return;
    }

    // Sparse path: build per-word k-bit lane masks from selection bits via the precomputed
    // cyclic phase code-start masks, PEXT-compact each word, and stitch straddled values
    // across the word boundary via a carry mask threaded from word w-1 into word w.
    let k_ones: u128 = if k == 0 { 0 } else { (1u128 << k) - 1 };
    let mut carry_mask: u64 = 0;
    let mut acc: u128 = 0;
    let mut acc_bits: u32 = 0;
    let mut flush = [0u64; MAX_COMPACTED_WORDS];
    let mut flush_len = 0usize;

    for w in 0..words_needed {
        let jlo = j_lo[w];
        let shifted_sel = if jlo >= 64 { 0 } else { sel >> jlo };
        // No explicit `unsafe` block needed: `_pdep_u64` requires the `bmi2` target feature,
        // which this function's own `#[target_feature(enable = "bmi2")]` already guarantees
        // for its whole body (calling a same-feature intrinsic from a function that already
        // carries that target_feature is permitted without a nested unsafe block).
        let starts_subset = std::arch::x86_64::_pdep_u64(shifted_sel, phase_masks[w]);

        // Expand each selected code-start bit into a full k-bit lane via the multiply-smear
        // identity `x * ((1<<k)-1) == (x<<k) - x`, which turns an isolated bit at position p
        // into a contiguous k-bit run [p, p+k). Widened to u128 because a lane starting near
        // the top of the word overflows bit 63 into the next word's low bits (a straddling
        // value); that overflow becomes `carry_mask` for the next iteration.
        let smear: u128 = (starts_subset as u128) * k_ones;
        let mask_w = (smear as u64) | carry_mask;
        carry_mask = (smear >> 64) as u64;

        // No explicit `unsafe` block needed: see the `_pdep_u64` call above.
        let compacted = std::arch::x86_64::_pext_u64(words[w], mask_w);
        let bits_w = mask_w.count_ones();

        acc |= (compacted as u128) << acc_bits;
        acc_bits += bits_w;
        while acc_bits >= 64 {
            flush[flush_len] = acc as u64;
            flush_len += 1;
            acc >>= 64;
            acc_bits -= 64;
        }
    }
    if acc_bits > 0 {
        flush[flush_len] = acc as u64;
        flush_len += 1;
    }

    let m = sel.count_ones() as usize;
    // SAFETY: dict.len() == 1<<k upheld by caller; flush[..flush_len] holds exactly the m
    // compacted k-bit codes for this frame's selected values, in order.
    unsafe { extract_sequential_codes(&flush[..flush_len], m, k, dict, out) };
}

/// Arm A (`paper_pext`): faithfully paper-shaped PEXT kernel. Processes one bit-packed run in
/// 64-value frames (each frame = exactly `k` input `u64` words). BMI2 capability must be
/// decided once by the caller/harness outside any loop -- this function assumes the `bmi2`
/// target feature is available and uses `_pext_u64`/`_pdep_u64` unconditionally.
///
/// `k` is derived from `dict.len()` (`dict.len() == 1 << k` is a caller guarantee).
///
/// # Safety
/// The caller must guarantee the `bmi2` CPU feature is available (e.g. via
/// `std::arch::is_x86_64_feature_detected!("bmi2")` checked once at startup), that
/// `n_values % 8 == 0`, that `payload` has at least 8 bytes beyond the run's last byte, that
/// `sel_words.len() == n_values.div_ceil(64)` with bits beyond `n_values` zeroed, and that
/// `dict.len() == 1 << k` for some `k <= MAX_K`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
pub unsafe fn select_run_bmi2(
    payload: &[u8],
    byte_offset: usize,
    n_values: usize,
    sel_words: &[u64],
    dict: &[i64],
    out: &mut Vec<i64>,
) {
    debug_assert_eq!(n_values % 8, 0);
    debug_assert!(dict.len().is_power_of_two());
    debug_assert_eq!(sel_words.len(), n_values.div_ceil(64));
    let k = dict.len().trailing_zeros();
    debug_assert!(k as usize <= MAX_K);

    let (phase_masks, j_lo) = compute_phase_tables(k);
    let frame_stride = 8usize * k as usize;
    let full_frames = n_values / 64;
    let tail = n_values % 64;

    let mut frame_byte = byte_offset;
    for frame_idx in 0..full_frames {
        let sel = sel_words[frame_idx];
        // SAFETY: bmi2 feature enabled on this fn (target_feature); frame_byte in-bounds
        // per the run-level padding guarantee.
        unsafe {
            process_frame_bmi2(payload, frame_byte, 64, sel, k, &phase_masks, &j_lo, dict, out);
        }
        frame_byte += frame_stride;
    }
    if tail > 0 {
        let sel = sel_words[full_frames]; // already masked to valid bits by the caller
        unsafe {
            process_frame_bmi2(payload, frame_byte, tail, sel, k, &phase_masks, &j_lo, dict, out);
        }
    }
}
