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

//! Pure-logic kernel module for the `v19-static-census` tool (experiment
//! `arrow-selected-lane-incidence-census-v19`, static-structural-census component: facts 1-4
//! only -- required-vs-optional, encoding, RLE-run-vs-bit-packed-run composition/length
//! distribution, bit width distribution. Fact 5-6 live selection shape/strategy is a separate
//! component and nothing here touches it).
//!
//! This file is intentionally dependency-free (only `std`): it is shared byte-for-byte between
//! the DataFusion-worktree bin crate (`benchmarks/v19-static-census`) and the zero-dependency
//! scratch dev-check crate (`.scratch/v19-static-census-devcheck`) used to unit-test it locally,
//! mirroring the pattern `arrow-paper-select-fourarm-v18`'s `kernel.rs` established. Do not add
//! any `parquet`/`arrow`/`bytes`/`serde` imports here -- every function takes plain `&[u8]` (plus
//! small caller-supplied integers) and returns plain structured counts.
//!
//! ## What this module walks, and what it deliberately does not decode
//!
//! Two closely related but distinct byte streams share the same run-header framing:
//!
//! 1. **Dictionary-index streams** ([`walk_index_stream`]): the per-data-page RLE/bit-packing
//!    hybrid stream of dictionary indices for `Encoding::RLE_DICTIONARY` /
//!    `Encoding::PLAIN_DICTIONARY` pages. This walker only needs run *headers* and *lengths* --
//!    it never decodes an individual dictionary index value -- so it stays far simpler than a
//!    real `RleDecoder`.
//! 2. **Definition-level streams** ([`walk_rle_levels`]): the same run-header framing, used to
//!    recover an authoritative non-null count for `DataPageV1` optional columns (see below for
//!    why this is needed). Unlike (1), this walker *does* decode individual level values out of
//!    bit-packed runs, because it needs to know how many equal `max_def_level` (i.e. are
//!    non-null), not just how many there are.
//!
//! ## RLE/bit-packing hybrid run framing
//!
//! Confirmed against arrow-rs's own module doc comment and `RleDecoder::reload()` at pinned
//! commit `ed92960c8a85eda657fce3525c905616ccc5a983`
//! (`parquet/src/encodings/rle.rs:18-30,611-618`): a run stream is a sequence of runs, each
//! starting with a ULEB128 ("VLQ") "indicator" varint `hdr` (7 bits per byte, LSB-first, high
//! bit = continuation; see `parquet/src/util/bit_util.rs` `get_vlq_int`/`put_vlq_int`):
//!
//! - `hdr == 0`: **not** a zero-length RLE run. arrow-rs's own decoder treats this as an
//!   explicit end-of-stream sentinel (added by some writers, e.g. fastparquet, as trailing
//!   padding) and stops immediately without reading further bytes. This module does the same
//!   (see [`walk_index_stream`]/[`walk_rle_levels`] doc comments).
//! - `hdr & 1 == 0`: an **RLE run** of `hdr >> 1` repeats of one value, stored immediately after
//!   the header in `ceil(bit_width / 8)` bytes (little-endian, zero-padded to that byte width).
//! - `hdr & 1 == 1`: a **bit-packed run** of `(hdr >> 1) * 8` values, stored immediately after
//!   the header as `(hdr >> 1) * 8 * bit_width / 8` bytes of flat LSB-first bit-packed values
//!   (value `i`, 0-indexed within the run, occupies bit range `[i*bit_width, i*bit_width +
//!   bit_width)` of the payload, where flat bit `b` lives at byte `b/8`, bit `b%8`).
//!
//! Neither walker here ever needs an overall stream length prefix: callers already know exactly
//! where each stream ends (the caller-supplied slice *is* the stream), so both walkers simply
//! walk runs until the slice is exhausted, a zero-header sentinel is hit, or the stream turns
//! out to be structurally truncated (in which case an anomaly is reported, never a panic).
//!
//! ## Why definition levels need decoding, not just skipping, for `DataPageV1`
//!
//! `DataPageV2` carries an explicit `num_nulls` in its page header, so the non-null count for
//! the dictionary-index stream is always known for free. `DataPageV1` carries no such field;
//! arrow-rs's own `ColumnValueDecoder::set_data` doc comment
//! (`parquet/src/column/reader/decoder.rs:104-110`) says as much: "data encoded with
//! `Encoding::RLE` may not know its exact length, as the final run may be zero-padded... if
//! `num_values` is not provided..., subsequent calls... may yield more values than non-null
//! definition levels within the page". For a `DataPageV1` optional column, the only way to
//! learn the exact non-null count (short of literally walking the dictionary-index stream to
//! its natural end and trusting that, which is circular for a self-consistency check) is to
//! decode the definition-level stream and count entries equal to `max_def_level`. That is what
//! [`walk_rle_levels`] is for.

// ============================================================================================
// Bucket definitions (frozen by the v19 census contract).
// ============================================================================================

/// Run-length histogram bucket labels, in bucket-index order. Bucketed by the run's
/// value-count: `hdr >> 1` for an RLE run, `(hdr >> 1) * 8` for a bit-packed run.
pub const RUN_LENGTH_BUCKET_LABELS: [&str; 6] = ["1-8", "9-64", "65-512", "513-4096", "4097-65535", ">=65536"];

/// Returns the bucket index for a run of `n_values` values. `n_values` must be `>= 1`
/// (zero-length runs, a degenerate construct no known writer emits, are not bucketed by
/// callers -- see [`RunLengthHistogram::add`]).
pub fn run_length_bucket(n_values: u64) -> usize {
    debug_assert!(n_values >= 1, "zero-length runs must not reach run_length_bucket");
    match n_values {
        1..=8 => 0,
        9..=64 => 1,
        65..=512 => 2,
        513..=4096 => 3,
        4097..=65535 => 4,
        _ => 5,
    }
}

/// Bit-width histogram bucket labels, in bucket-index order.
pub const BITWIDTH_BUCKET_LABELS: [&str; 7] = ["1", "2", "3-4", "5-8", "9-12", "13-16", ">16"];

/// Returns the bucket index for dictionary bit width `k`. `k == 0` (the degenerate
/// zero-or-one-entry dictionary case, which needs no bits at all) is folded into the `"1"`
/// bucket, since the contract defines no `"0"` bucket -- this is a documented assumption, not
/// something confirmed by the frozen contract text.
pub fn bitwidth_bucket(k: u32) -> usize {
    match k {
        0 | 1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=8 => 3,
        9..=12 => 4,
        13..=16 => 5,
        _ => 6,
    }
}

/// Run-length histogram: both a count of *runs* and a sum of *values* per bucket. The v19
/// contract's "run-length distribution" is naturally a count of runs per bucket (frequency of
/// run lengths); the value-count side is tracked too since it is nearly free and independently
/// useful (e.g. "what share of values sit in long runs").
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct RunLengthHistogram {
    pub run_counts: [u64; 6],
    pub value_counts: [u64; 6],
}

impl RunLengthHistogram {
    /// Records one run of `n_values` values. A zero-length run (see the module doc comment on
    /// `hdr == 1` for bit-packed runs -- `num_groups = 0` is structurally parseable but
    /// degenerate) contributes nothing and is silently not bucketed, since it isn't meaningful
    /// under either the `1-8` or `>=65536` extremes.
    ///
    /// Uses saturating arithmetic throughout: a run header is caller-controlled (ultimately
    /// file-controlled) data, and a corrupt/adversarial header can claim an astronomically
    /// large run length while needing few or zero body bytes (e.g. at `bit_width == 0`, an RLE
    /// run's repeated-value body is always 0 bytes regardless of its claimed repeat count).
    /// Saturating at `u64::MAX` keeps this module's hard "never panics on malformed input"
    /// contract rather than overflow-panicking in debug builds.
    pub fn add(&mut self, n_values: u64) {
        if n_values == 0 {
            return;
        }
        let b = run_length_bucket(n_values);
        self.run_counts[b] = self.run_counts[b].saturating_add(1);
        self.value_counts[b] = self.value_counts[b].saturating_add(n_values);
    }

    /// Merges `other`'s bucket counts into `self` (saturating; see [`Self::add`]).
    pub fn merge(&mut self, other: &Self) {
        for i in 0..6 {
            self.run_counts[i] = self.run_counts[i].saturating_add(other.run_counts[i]);
            self.value_counts[i] = self.value_counts[i].saturating_add(other.value_counts[i]);
        }
    }
}

/// Bit-width histogram: sum of bit-packed-run *value-counts*, weighted per the contract ("a
/// bucket's count is the SUM of value-counts of all bit-packed runs whose column's bit width
/// falls in that bucket, not a count of runs"). Since bit width `k` is fixed per dictionary-index
/// stream (one leading byte per data page), callers add once per page with that page's total
/// bit-packed value count, rather than once per run.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct BitWidthHistogram {
    pub value_counts: [u64; 7],
}

impl BitWidthHistogram {
    /// Saturating; see [`RunLengthHistogram::add`] for why.
    pub fn add(&mut self, k: u32, n_values: u64) {
        if n_values == 0 {
            return;
        }
        let b = bitwidth_bucket(k);
        self.value_counts[b] = self.value_counts[b].saturating_add(n_values);
    }

    /// Saturating; see [`RunLengthHistogram::merge`].
    pub fn merge(&mut self, other: &Self) {
        for i in 0..7 {
            self.value_counts[i] = self.value_counts[i].saturating_add(other.value_counts[i]);
        }
    }
}

// ============================================================================================
// Shared bit-width arithmetic.
// ============================================================================================

/// `64 - x.leading_zeros()`: the minimum number of bits needed to represent the unsigned value
/// `x` (`0` for `x == 0`). Mirrors arrow-rs's `parquet::util::bit_util::num_required_bits`
/// (`parquet/src/util/bit_util.rs:279-282`, pinned commit) bit-for-bit; re-derived here (rather
/// than imported, since this module has no `parquet` dependency).
fn num_required_bits(x: u64) -> u32 {
    64 - x.leading_zeros()
}

/// The RLE/bit-packing hybrid dictionary-index bit width a Parquet writer would use for a
/// dictionary of `num_entries` distinct values: `num_required_bits(num_entries - 1)`, i.e. the
/// number of bits needed to represent indices `0..num_entries` (saturating to `0` when
/// `num_entries <= 1`). Mirrors arrow-rs's `DictEncoder::bit_width`
/// (`parquet/src/encodings/encoding/dict_encoder.rs:153-155`, pinned commit) bit-for-bit. Used
/// only to *cross-check* the self-describing bit-width byte actually stored at the start of
/// every `RLE_DICTIONARY` data page's value bytes -- never as a substitute for reading it.
pub fn dict_bit_width(num_entries: u64) -> u8 {
    num_required_bits(num_entries.saturating_sub(1)) as u8
}

/// The bit width used to encode a definition/repetition level stream whose maximum level is
/// `max_level`: `num_required_bits(max_level)` (no `-1`, since levels range over
/// `0..=max_level` inclusive and `max_level` itself is the largest representable value).
/// Mirrors arrow-rs's `parquet/src/column/reader.rs` (`parse_v1_level`'s `BIT_PACKED` arm) and
/// `parquet/src/column/reader/decoder.rs` (`DefinitionLevelDecoderImpl::new`,
/// `RepetitionLevelDecoderImpl::new`), both of which compute
/// `num_required_bits(max_level as u64)` at the pinned commit.
pub fn level_bit_width(max_level: i16) -> u32 {
    num_required_bits(max_level.max(0) as u64)
}

// ============================================================================================
// ULEB128 / VLQ varint reader.
// ============================================================================================

/// Reads a ULEB128 ("VLQ") varint starting at `stream[pos]`: 7 bits per byte, LSB-first, high
/// bit of each byte is the continuation flag. Mirrors Parquet's `BitReader::get_vlq_int` /
/// `BitWriter::put_vlq_int` framing (`parquet/src/util/bit_util.rs`, pinned commit). Returns
/// `(value, new_pos)` on success, or `None` if the byte stream ends before a terminating
/// (high-bit-clear) byte is found, or if more than 10 continuation bytes are read without
/// terminating (10 bytes * 7 bits = 70 bits comfortably covers any real `u64` header value;
/// anything longer is treated as malformed rather than risking a shift overflow). Never panics.
fn read_uvarint(stream: &[u8], pos: usize) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    let mut p = pos;
    loop {
        if p >= stream.len() || p - pos >= 10 {
            return None;
        }
        let byte = stream[p];
        p += 1;
        if shift < 64 {
            value |= ((byte & 0x7F) as u64) << shift;
        }
        if byte & 0x80 == 0 {
            return Some((value, p));
        }
        shift += 7;
    }
}

// ============================================================================================
// Dictionary-index RLE-hybrid stream walker (lane census: facts 2-4).
// ============================================================================================

/// Why a dictionary-index (or definition-level) stream stopped short of covering every value
/// the caller expected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkAnomaly {
    /// A run header's varint continuation chain ran off the end of the slice (or exceeded the
    /// 10-byte sanity cap) before terminating.
    TruncatedHeader { byte_offset: usize },
    /// An RLE run's header was read, but fewer than `ceil(bit_width/8)` bytes remained for its
    /// repeated value.
    TruncatedRleValue { byte_offset: usize, needed: usize, available: usize },
    /// A bit-packed run's header was read, but fewer bytes than its declared payload length
    /// remained.
    TruncatedBitPackedPayload { byte_offset: usize, needed: usize, available: usize },
}

impl WalkAnomaly {
    /// Human-readable description, independent of any `std::fmt::Display` impl so callers in
    /// the real crate can fold this into a larger anomaly message without extra ceremony.
    pub fn describe(&self) -> String {
        match self {
            WalkAnomaly::TruncatedHeader { byte_offset } => {
                format!("truncated run header varint at byte offset {byte_offset}")
            }
            WalkAnomaly::TruncatedRleValue { byte_offset, needed, available } => {
                format!(
                    "truncated RLE run repeated-value at byte offset {byte_offset}: needed {needed} bytes, {available} available"
                )
            }
            WalkAnomaly::TruncatedBitPackedPayload { byte_offset, needed, available } => {
                format!(
                    "truncated bit-packed run payload at byte offset {byte_offset}: needed {needed} bytes, {available} available"
                )
            }
        }
    }
}

/// Result of walking one dictionary-index RLE-hybrid stream (the bytes *after* the leading
/// bit-width byte has already been stripped by the caller).
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct IndexStreamWalk {
    /// Total values contributed by RLE runs (sum of `hdr >> 1` across all RLE runs walked).
    pub rle_values: u64,
    /// Total values contributed by bit-packed runs (sum of `(hdr >> 1) * 8` across all
    /// bit-packed runs walked).
    pub bitpacked_values: u64,
    pub rle_run_lengths: RunLengthHistogram,
    pub bitpacked_run_lengths: RunLengthHistogram,
    /// How far into the input slice the walk got before stopping (cleanly or otherwise).
    pub bytes_consumed: usize,
    /// Set if the walk stopped due to structural truncation rather than a clean end-of-stream.
    pub anomaly: Option<WalkAnomaly>,
}

impl IndexStreamWalk {
    /// Saturating add; see [`RunLengthHistogram::add`] for why this module uses saturating
    /// arithmetic throughout rather than trusting file-controlled run headers not to overflow
    /// a `u64` accumulator.
    pub fn total_values(&self) -> u64 {
        self.rle_values.saturating_add(self.bitpacked_values)
    }
}

/// Walks one dictionary-index RLE/bit-packing hybrid run stream (the bytes *after* the leading
/// bit-width byte, which the caller has already read and stripped) and buckets every run found.
///
/// Stops cleanly (`anomaly == None`) when:
/// - the slice is exactly exhausted (no bytes left to start a new header), or
/// - a header varint decodes to `0`, which arrow-rs's own `RleDecoder::reload()` treats as an
///   explicit end-of-stream sentinel (`parquet/src/encodings/rle.rs:629-632`, pinned commit:
///   "fastparquet adds padding to the end of pages... `if indicator_value == 0 { return
///   Ok(false); }`").
///
/// Reports a [`WalkAnomaly`] (and stops) when the stream is structurally truncated: a header
/// varint whose continuation chain runs off the end of the slice, or a run whose declared body
/// needs more bytes than remain.
///
/// Never panics on malformed input; never decodes individual dictionary index *values* -- only
/// run headers and lengths -- matching this tool's read-only, non-decoding design (it exists to
/// census run composition, not to reconstruct decoded values).
pub fn walk_index_stream(stream: &[u8], bit_width: u8) -> IndexStreamWalk {
    let mut result = IndexStreamWalk::default();
    let k = bit_width as u64;
    let rle_value_bytes = (bit_width as usize).div_ceil(8);
    let mut pos = 0usize;

    loop {
        if pos >= stream.len() {
            break;
        }
        let header_start = pos;
        let (hdr, new_pos) = match read_uvarint(stream, pos) {
            Some(v) => v,
            None => {
                result.anomaly = Some(WalkAnomaly::TruncatedHeader { byte_offset: header_start });
                break;
            }
        };
        if hdr == 0 {
            break;
        }
        pos = new_pos;

        if hdr & 1 == 1 {
            // Bit-packed run: (hdr >> 1) groups of 8 values each.
            let num_groups = hdr >> 1;
            let n_values = num_groups.saturating_mul(8);
            let payload_bits: u128 = (n_values as u128) * (k as u128);
            let payload_bytes_u128 = payload_bits.div_ceil(8);
            let available = stream.len().saturating_sub(pos);
            if payload_bytes_u128 > available as u128 {
                result.anomaly = Some(WalkAnomaly::TruncatedBitPackedPayload {
                    byte_offset: pos,
                    needed: payload_bytes_u128.min(usize::MAX as u128) as usize,
                    available,
                });
                break;
            }
            let payload_bytes = payload_bytes_u128 as usize;
            pos += payload_bytes;
            result.bitpacked_values = result.bitpacked_values.saturating_add(n_values);
            result.bitpacked_run_lengths.add(n_values);
        } else {
            // RLE run: hdr >> 1 repeats of one value, stored in `rle_value_bytes` bytes.
            let n_values = hdr >> 1;
            let available = stream.len().saturating_sub(pos);
            if rle_value_bytes > available {
                result.anomaly = Some(WalkAnomaly::TruncatedRleValue {
                    byte_offset: pos,
                    needed: rle_value_bytes,
                    available,
                });
                break;
            }
            pos += rle_value_bytes;
            result.rle_values = result.rle_values.saturating_add(n_values);
            result.rle_run_lengths.add(n_values);
        }
    }

    result.bytes_consumed = pos;
    result
}

// ============================================================================================
// Flat bit-packed value extraction (used only by the level-stream walker below).
// ============================================================================================

/// Extracts the `idx`-th (0-based) `bit_width`-bit value from `payload`, using the same
/// LSB-first flat bit-packing layout as every bit-packed run in this format (value `i` occupies
/// bit range `[i*bit_width, i*bit_width+bit_width)`, flat bit `b` at byte `b/8`, bit `b%8`).
/// Returns `None` if the needed bits run past `payload`'s end. `bit_width == 0` always yields
/// `Some(0)` (a zero-width field carries no information).
fn unpack_value_at(payload: &[u8], idx: usize, bit_width: u32) -> Option<u64> {
    if bit_width == 0 {
        return Some(0);
    }
    let start_bit = idx as u64 * bit_width as u64;
    let end_bit = start_bit + bit_width as u64;
    if end_bit.div_ceil(8) as usize > payload.len() {
        return None;
    }
    let mut value: u64 = 0;
    let mut got: u32 = 0;
    let mut bit = start_bit;
    while got < bit_width {
        let byte_idx = (bit / 8) as usize;
        let bit_in_byte = (bit % 8) as u32;
        let space = 8 - bit_in_byte;
        let take = (bit_width - got).min(space);
        let chunk = (payload[byte_idx] as u64 >> bit_in_byte) & ((1u64 << take) - 1);
        value |= chunk << got;
        got += take;
        bit += take as u64;
    }
    Some(value)
}

// ============================================================================================
// DataPageV1 level block framing (skip, for both rep and def levels).
// ============================================================================================

/// The two level encodings arrow-rs's own `parse_v1_level` accepts, plus a catch-all for
/// anything else (which arrow-rs itself rejects too -- see
/// `parquet/src/column/reader.rs:613` `_ => Err(general_err!("invalid level encoding: {}",
/// encoding))`, pinned commit). Kept local to this module (rather than importing
/// `parquet::basic::Encoding`) so this file stays dependency-free; callers translate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelEncodingKind {
    Rle,
    /// Deprecated flat bit-packing (no run structure at all), still spec-legal for V1 levels.
    BitPacked,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelBlockError {
    /// Not enough bytes for the 4-byte RLE length prefix, or the declared length overruns the
    /// buffer.
    TruncatedRleLengthPrefix,
    /// Not enough bytes for the (deprecated) flat `BIT_PACKED` level block's declared
    /// `ceil(num_values * bit_width / 8)` bytes.
    TruncatedBitPacked,
    /// A level encoding other than `RLE` or (deprecated) `BIT_PACKED`.
    UnsupportedLevelEncoding,
}

impl LevelBlockError {
    pub fn describe(&self) -> &'static str {
        match self {
            LevelBlockError::TruncatedRleLengthPrefix => {
                "level block RLE length prefix missing or overruns the page buffer"
            }
            LevelBlockError::TruncatedBitPacked => {
                "level block declared BIT_PACKED byte length overruns the page buffer"
            }
            LevelBlockError::UnsupportedLevelEncoding => {
                "level block uses an encoding other than RLE or BIT_PACKED"
            }
        }
    }
}

/// Mirrors arrow-rs's `parse_v1_level` (`parquet/src/column/reader.rs:588-614`, pinned commit)
/// byte-for-byte: for `Encoding::RLE`, a 4-byte little-endian `i32` length prefix followed by
/// that many bytes of RLE-hybrid-encoded level data; for the deprecated `Encoding::BIT_PACKED`,
/// `ceil(num_values * bit_width / 8)` raw packed bytes with **no** length prefix at all (a
/// flat, non-hybrid bit-packing -- distinct from the RLE/bit-pack hybrid format used everywhere
/// else in this module). Returns `(total_bytes_consumed, payload_slice)` on success, where
/// `payload_slice` is exactly the level data (for `Rle`: the hybrid run stream with the 4-byte
/// prefix already stripped; for `BitPacked`: the flat packed bytes themselves).
pub fn parse_v1_level_block(
    buf: &[u8],
    encoding: LevelEncodingKind,
    num_values: u64,
    bit_width: u32,
) -> Result<(usize, &[u8]), LevelBlockError> {
    match encoding {
        LevelEncodingKind::Rle => {
            if buf.len() < 4 {
                return Err(LevelBlockError::TruncatedRleLengthPrefix);
            }
            let data_size = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
            let end = 4usize.checked_add(data_size).ok_or(LevelBlockError::TruncatedRleLengthPrefix)?;
            if end > buf.len() {
                return Err(LevelBlockError::TruncatedRleLengthPrefix);
            }
            Ok((end, &buf[4..end]))
        }
        LevelEncodingKind::BitPacked => {
            let num_bytes_u128 = (num_values as u128 * bit_width as u128).div_ceil(8);
            if num_bytes_u128 > buf.len() as u128 {
                return Err(LevelBlockError::TruncatedBitPacked);
            }
            let num_bytes = num_bytes_u128 as usize;
            Ok((num_bytes, &buf[..num_bytes]))
        }
        LevelEncodingKind::Other => Err(LevelBlockError::UnsupportedLevelEncoding),
    }
}

// ============================================================================================
// Definition-level RLE-hybrid walker (non-null recount for DataPageV1 optional columns).
// ============================================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelWalkAnomaly {
    TruncatedHeader { byte_offset: usize },
    TruncatedRleValue { byte_offset: usize, needed: usize, available: usize },
    TruncatedBitPackedPayload { byte_offset: usize, needed: usize, available: usize },
}

impl LevelWalkAnomaly {
    pub fn describe(&self) -> String {
        match self {
            LevelWalkAnomaly::TruncatedHeader { byte_offset } => {
                format!("truncated level run header varint at byte offset {byte_offset}")
            }
            LevelWalkAnomaly::TruncatedRleValue { byte_offset, needed, available } => {
                format!(
                    "truncated level RLE run repeated-value at byte offset {byte_offset}: needed {needed} bytes, {available} available"
                )
            }
            LevelWalkAnomaly::TruncatedBitPackedPayload { byte_offset, needed, available } => {
                format!(
                    "truncated level bit-packed run payload at byte offset {byte_offset}: needed {needed} bytes, {available} available"
                )
            }
        }
    }
}

/// Result of walking a definition-level `RLE`-hybrid stream.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct LevelWalk {
    /// Total level entries walked (null + non-null).
    pub total_values: u64,
    /// Count of walked level entries equal to `max_level` (i.e. non-null).
    pub nonnull_values: u64,
    pub bytes_consumed: usize,
    pub anomaly: Option<LevelWalkAnomaly>,
}

/// Walks an `Encoding::RLE`-encoded definition-level stream (the payload slice returned by
/// [`parse_v1_level_block`] for the `Rle` case) and counts how many of its `total_values`
/// decoded level entries equal `max_level` (i.e. are non-null).
///
/// Uses the same run-header framing as [`walk_index_stream`], but two differences follow from
/// levels being small self-contained integers rather than dictionary codes:
/// - `bit_width` is supplied by the caller (via [`level_bit_width`]), not read from a leading
///   byte -- arrow-rs computes it the same way, externally, from `max_def_level` /
///   `max_rep_level` (`parquet/src/column/reader/decoder.rs`
///   `DefinitionLevelDecoderImpl::new`/`RepetitionLevelDecoderImpl::new`, pinned commit).
/// - Bit-packed runs must be individually unpacked (via [`unpack_value_at`]) and compared
///   against `max_level`, rather than just length-counted, since a run's value matters here.
///
/// Stop/anomaly conditions mirror [`walk_index_stream`] exactly (clean stop at slice
/// exhaustion or a zero-header sentinel; anomaly on structural truncation). Never panics.
pub fn walk_rle_levels(stream: &[u8], bit_width: u32, max_level: u64) -> LevelWalk {
    let mut result = LevelWalk::default();
    let rle_value_bytes = (bit_width as usize).div_ceil(8);
    let mut pos = 0usize;

    loop {
        if pos >= stream.len() {
            break;
        }
        let header_start = pos;
        let (hdr, new_pos) = match read_uvarint(stream, pos) {
            Some(v) => v,
            None => {
                result.anomaly = Some(LevelWalkAnomaly::TruncatedHeader { byte_offset: header_start });
                break;
            }
        };
        if hdr == 0 {
            break;
        }
        pos = new_pos;

        if hdr & 1 == 1 {
            let num_groups = hdr >> 1;
            let n_values = num_groups.saturating_mul(8);

            if bit_width == 0 {
                // Every value in a zero-width run is trivially 0 -- decode in O(1) rather than
                // looping `n_values` times. This also sidesteps a real hang risk: a zero-width
                // payload is always 0 bytes regardless of how large a (corrupt/adversarial)
                // header claims `n_values` to be, so without this special case `n_values` would
                // be unbounded by `available` and the per-value loop below could spin for an
                // enormous number of iterations on malformed input.
                if max_level == 0 {
                    result.nonnull_values = result.nonnull_values.saturating_add(n_values);
                }
                result.total_values = result.total_values.saturating_add(n_values);
                continue;
            }

            let payload_bits: u128 = (n_values as u128) * (bit_width as u128);
            let payload_bytes_u128 = payload_bits.div_ceil(8);
            let available = stream.len().saturating_sub(pos);
            if payload_bytes_u128 > available as u128 {
                result.anomaly = Some(LevelWalkAnomaly::TruncatedBitPackedPayload {
                    byte_offset: pos,
                    needed: payload_bytes_u128.min(usize::MAX as u128) as usize,
                    available,
                });
                break;
            }
            // `n_values` is bounded by `available * 8 / bit_width` here (bit_width >= 1, proven
            // by the check above), so this cast and the loop below are bounded by real input
            // size, not by an unbounded header claim.
            let payload_bytes = payload_bytes_u128 as usize;
            let payload = &stream[pos..pos + payload_bytes];
            for i in 0..n_values as usize {
                // Safe: `payload` is sized exactly for `n_values * bit_width` bits above, so
                // every index `0..n_values` is in-bounds; `unwrap_or` is defensive belt-and-
                // braces only, never expected to hit the `None` arm.
                let v = unpack_value_at(payload, i, bit_width).unwrap_or(u64::MAX);
                if v == max_level {
                    result.nonnull_values = result.nonnull_values.saturating_add(1);
                }
            }
            pos += payload_bytes;
            result.total_values = result.total_values.saturating_add(n_values);
        } else {
            let n_values = hdr >> 1;
            let available = stream.len().saturating_sub(pos);
            if rle_value_bytes > available {
                result.anomaly = Some(LevelWalkAnomaly::TruncatedRleValue {
                    byte_offset: pos,
                    needed: rle_value_bytes,
                    available,
                });
                break;
            }
            let mut repeated: u64 = 0;
            for (i, &b) in stream[pos..pos + rle_value_bytes].iter().enumerate() {
                repeated |= (b as u64) << (8 * i);
            }
            pos += rle_value_bytes;
            if repeated == max_level {
                result.nonnull_values = result.nonnull_values.saturating_add(n_values);
            }
            result.total_values = result.total_values.saturating_add(n_values);
        }
    }

    result.bytes_consumed = pos;
    result
}
