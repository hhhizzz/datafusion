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

//! `v19-static-census`: the **static structural census** component of experiment
//! `arrow-selected-lane-incidence-census-v19` (see
//! `codex/experiments/arrow-selected-lane-incidence-census-v19.md`). Answers facts 1-4 only
//! (required-vs-optional, encoding, RLE-run-vs-bit-packed-run composition/length distribution,
//! bit width distribution). Facts 5-6 (live selection shape/strategy) are out of scope for this
//! tool entirely -- nothing here touches selection or strategy-resolution code.
//!
//! This is a one-shot, read-only analysis program, not a real benchmark (`Cargo.toml` declares
//! it `[[bench]] harness = false` only so the guarded K8s runner's `cargo bench -p <package>
//! --bench <target>` invocation can reach it -- Criterion does not apply): it opens each target
//! dataset's Parquet files via stock, **unmodified** upstream `parquet` (composed at arrow-rs
//! commit
//! `ed92960c8a85eda657fce3525c905616ccc5a983`), walks every row group/column/page, and prints a
//! census. It never writes to, or modifies, any file it inspects.
//!
//! ## Public API surface used (and why no arrow-rs source modification, or `experimental`
//! Cargo feature, is needed)
//!
//! Stock `parquet::file::reader::{FileReader, RowGroupReader, SerializedFileReader}` and
//! `parquet::column::page::{Page, PageReader}` already expose everything needed to read
//! column-chunk metadata and raw (already-decompressed) page bytes. Confirmed at the pinned
//! commit that `column` and `file` are *unconditionally* `pub mod` in `parquet/src/lib.rs`
//! (lines 198, 206) -- unlike `compression`/`encodings`, which are only `pub` when the
//! `experimental` Cargo feature is enabled (`parquet/src/lib.rs` lines 147-166's `experimental!`
//! macro: without the feature, `experimental!(mod compression)` expands to a *private* `mod`,
//! unreachable outside the crate). This tool needs neither module: it decodes the RLE-hybrid
//! run *structure* itself from scratch (see `kernel.rs`), and page bytes already arrive fully
//! decompressed (see below), so `parquet::compression::create_codec` is never called -- indeed
//! it is not even reachable without `experimental`, which this crate does not enable.
//!
//! ## Byte-layout facts confirmed against the pinned commit (see `kernel.rs` for the RLE-hybrid
//! run framing itself; this module's own responsibility is locating each page's value-byte
//! stream within the raw page buffer)
//!
//! - **`Page::buf` is always fully decompressed.** `parquet/src/column/page.rs` line ~33's own
//!   doc comment: "these are 1-to-1 mapped from the equivalent Thrift definitions, except `buf`
//!   which used to store uncompressed bytes of the page." Traced concretely through
//!   `parquet/src/file/serialized_reader.rs`'s `decode_page` (~line 393-470): it always
//!   decompresses (when a codec is configured) before constructing any `Page` variant, and for
//!   `DataPageV2` specifically, copies the `def_levels_byte_len + rep_levels_byte_len`-byte
//!   level prefix through **unchanged** (never compressed on disk in the first place) while
//!   decompressing only the remainder -- confirming the census-runner's expectation that "levels
//!   are NOT compressed even when `is_compressed` is true for the data section in V2."
//! - **`DataPageV1` `buf` layout**: `[rep-level block if max_rep_level>0][def-level block if
//!   max_def_level>0][value bytes]`, confirmed field-for-field against
//!   `parquet/src/column/reader.rs` `read_new_page`'s `Page::DataPage` arm (~lines 448-503) and
//!   its `parse_v1_level` helper (~lines 588-614). Each level block present is either
//!   `Encoding::RLE` (a 4-byte little-endian `i32` length prefix, then that many bytes of
//!   RLE-hybrid-encoded level data) or the deprecated `Encoding::BIT_PACKED` (`ceil(num_values *
//!   bit_width / 8)` raw flat-packed bytes, **no** length prefix, not even the RLE-hybrid
//!   format). See `kernel.rs`'s `parse_v1_level_block` for the re-derived logic.
//! - **`DataPageV2` byte-length fields**: `def_levels_byte_len`/`rep_levels_byte_len` give the
//!   exact skip lengths directly; `read_new_page`'s `Page::DataPageV2` arm confirms the order is
//!   rep-levels-then-def-levels-then-values (`buf.slice((rep_levels_byte_len +
//!   def_levels_byte_len)..)` for the value decoder), and separately, that the true non-null
//!   value count fed to the index-stream decoder is `num_values - num_nulls`, not the page's raw
//!   `num_values` (see below).
//! - **Dictionary-index bit-width byte is self-describing and authoritative.**
//!   `parquet/src/encodings/decoding.rs` `DictDecoder::set_data` (~line 381): "First byte in
//!   `data` is bit width" -- `let bit_width = data.as_ref()[0];` -- read directly, not derived.
//!   This tool cross-checks it against the bit width a writer would have chosen for the
//!   preceding dictionary page's cardinality (`parquet/src/encodings/encoding/dict_encoder.rs`
//!   `DictEncoder::bit_width`, ~line 153: `num_required_bits(num_entries.saturating_sub(1))`),
//!   logging a disagreement as an anomaly rather than silently trusting either value alone (see
//!   `finish_value_stream` below), but always *uses* the on-disk byte for the actual walk.
//! - **`DataPageV1` optional columns have no explicit non-null count** (unlike `DataPageV2`,
//!   which always carries `num_nulls`). `parquet/src/column/reader/decoder.rs`'s
//!   `ColumnValueDecoder::set_data` doc comment (~lines 104-110) says so explicitly: "data
//!   encoded with `Encoding::RLE` may not know its exact length... subsequent calls... may yield
//!   more values than non-null definition levels within the page." This tool therefore decodes
//!   the definition-level stream itself (`kernel::walk_rle_levels`) to recover an authoritative
//!   non-null count for the page-level self-consistency check on `DataPageV1` optional columns,
//!   rather than naively comparing the walked dictionary-index total against the page's raw
//!   (nulls-inclusive) `num_values` -- see the `expected_nonnull` computation and its comment in
//!   `walk_v1_data_page` below for why that naive comparison would be wrong on every optional
//!   `DataPageV1` page with any nulls at all (the overwhelming majority of pages in both target
//!   datasets, per the parent census program's own prior finding that TPC-DS SF10 is effectively
//!   all-optional).
//!
//! ## Documented assumptions (things this file could *not* confirm from the pinned commit and
//! had to decide instead -- flagged here as the highest-risk parts of this component; also
//! repeated in the final task report)
//!
//! - The bit-width histogram's bucket labels (`1,2,3-4,5-8,9-12,13-16,>16`) have no `"0"` bucket,
//!   but a dictionary with cardinality `<= 1` legitimately needs `0` bits. This tool folds `k ==
//!   0` into the `"1"` bucket (see `kernel::bitwidth_bucket`) as the closest reasonable choice,
//!   not something the frozen contract specifies.
//! - The frozen lane table has no "other encodings" row (only `PLAIN (any)` and the two dict
//!   lanes), yet real columns can use `RLE` (for `BOOLEAN` values directly, not levels),
//!   `DELTA_BINARY_PACKED`, `BYTE_STREAM_SPLIT`, etc. This tool tallies those separately (see
//!   `other_encoding_values`) rather than folding them into `PLAIN (any)` or silently dropping
//!   them from the "share of decoded values" denominator, and reports them as a supplementary
//!   section beyond the frozen table.
//! - Per the census contract's explicit instruction for `PLAIN` pages ("still record its
//!   `num_values` toward the... denominator"), the `PLAIN (any)` lane counts a page's raw
//!   `num_values` (nulls included, for an optional `PLAIN` column), whereas the two dict lanes
//!   count only the *walked, non-null* value total. This is a real asymmetry inherent in the
//!   census contract's own treatment of `PLAIN` vs dictionary pages, not smoothed over here --
//!   flagged in the printed report so it is never mistaken for a bug.
//! - The frozen lane table's required/optional split (`max_def_level == 0 && max_rep_level ==
//!   0` vs `max_def_level > 0`) leaves the theoretical case `max_def_level == 0 && max_rep_level
//!   > 0` unclassified by either definition. This tool buckets that (believed-unreachable-in-
//!   practice, since Parquet's own repeated-field encoding conventionally implies
//!   `max_def_level > 0` too) case as optional and flags it once per affected column as a side
//!   fact, rather than silently dropping its values from the lane table.

#[path = "v19_static_census/kernel.rs"]
mod kernel;

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use parquet::basic::Encoding;
use parquet::column::page::Page;
use parquet::file::metadata::{ColumnChunkMetaData, RowGroupMetaData};
use parquet::file::reader::{FileReader, RowGroupReader, SerializedFileReader};

/// Default dataset root; overridable via the first CLI argument (`cargo bench -p
/// v19-static-census --bench v19_static_census -- <root>`, or a `--bench-arg` in the guarded
/// runner). Hardcoding this (rather than requiring a flag) keeps the common case -- the K8s
/// enclave, where the datasets always live here -- a zero-argument invocation, while still
/// allowing an override for local/alternate layouts -- simplest option that is still flexible.
const DEFAULT_DATASET_ROOT: &str = "/workspace/datasets/";

/// Cap on how many anomaly messages are printed per dataset (the full count is always reported
/// too, via `DatasetCensus::anomalies_total_count`).
const MAX_REPORTED_ANOMALIES: usize = 20;

const CLICKBENCH_DIR_NAME: &str = "clickbench-100m-single-v1";
const TPCDS_DIR_NAME: &str = "tpcds-sf10-v1";

fn main() {
    // `cargo bench` unconditionally injects a bare `--bench` into argv[1] for every
    // `[[bench]]` target it runs, even with `harness = false` (a legacy libtest-compat
    // convention baked into Cargo itself, independent of anything this crate declares) --
    // confirmed empirically by a real K8s run: without this filter, `args().nth(1)` picked up
    // the literal string "--bench" as a dataset-root override. Skip any leading argv entries
    // that look like flags (start with `-`) and take the first genuine positional argument, if
    // any, as the override -- this also makes a real override still reachable via
    // `--bench-arg <root>` in the guarded runner, since Cargo passes those through after its
    // own `--bench`.
    let root = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .unwrap_or_else(|| DEFAULT_DATASET_ROOT.to_string());
    let root = PathBuf::from(root);
    println!("v19-static-census (static structural census, v19 facts 1-4 only)");
    println!("dataset root: {}", root.display());
    println!();

    let mut any_ran = false;

    let clickbench_dir = root.join(CLICKBENCH_DIR_NAME);
    match discover_clickbench(&clickbench_dir) {
        Some(tables) if !tables.is_empty() => {
            any_ran = true;
            let census = run_census(CLICKBENCH_DIR_NAME, &tables);
            report(&census);
        }
        _ => {
            println!(
                "{CLICKBENCH_DIR_NAME}: not found under {} -- skipping (dataset may not be published yet).",
                clickbench_dir.display()
            );
            println!();
        }
    }

    let tpcds_dir = root.join(TPCDS_DIR_NAME);
    match discover_tpcds(&tpcds_dir) {
        Some(tables) if !tables.is_empty() => {
            any_ran = true;
            let census = run_census(TPCDS_DIR_NAME, &tables);
            report(&census);
        }
        _ => {
            println!(
                "{TPCDS_DIR_NAME}: not found under {} -- skipping (dataset may not be published yet).",
                tpcds_dir.display()
            );
            println!();
        }
    }

    if !any_ran {
        println!("Neither dataset was found under {}. Nothing to census.", root.display());
        std::process::exit(1);
    }
}

// ================================================================================================
// Dataset discovery. Both datasets are discovered by *listing the directory at runtime* rather
// than assuming a fixed file/subdirectory layout, per the task's explicit instruction (the
// manifest file name is not authoritative for on-disk layout, and TPC-DS's per-table layout is
// unspecified up front).
// ================================================================================================

/// Discovers the ClickBench single-file dataset. Prefers the official `hits.parquet` name; if
/// that exact file is absent, lists the directory and falls back to whatever `*.parquet` files
/// are actually present (printing a note either way, so a layout surprise is never silent).
/// Returns `None` if the directory does not exist at all.
fn discover_clickbench(dir: &Path) -> Option<Vec<(String, Vec<PathBuf>)>> {
    if !dir.is_dir() {
        return None;
    }
    let preferred = dir.join("hits.parquet");
    if preferred.is_file() {
        return Some(vec![("hits".to_string(), vec![preferred])]);
    }

    let mut parquet_files: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_parquet_file(p))
        .collect();
    parquet_files.sort();

    if parquet_files.is_empty() {
        println!("NOTE: {} exists but contains no hits.parquet and no other *.parquet files.", dir.display());
        return Some(Vec::new());
    }
    println!(
        "NOTE: {} does not contain the expected hits.parquet; found {} *.parquet file(s) instead: {:?}. \
         Treating each as its own table for this census (confirm this is expected).",
        dir.display(),
        parquet_files.len(),
        parquet_files.iter().map(|p| p.display().to_string()).collect::<Vec<_>>()
    );
    Some(
        parquet_files
            .into_iter()
            .map(|p| {
                let name = file_stem_or(&p, "hits");
                (name, vec![p])
            })
            .collect(),
    )
}

/// Discovers TPC-DS SF10's per-table layout by listing `dir`: a direct `<table>.parquet` file is
/// one table; a subdirectory is treated as one table whose files are every `*.parquet` found
/// recursively beneath it (covers both a flat one-file-per-table layout and a partitioned
/// directory-per-table layout without assuming which). Returns `None` if `dir` does not exist.
fn discover_tpcds(dir: &Path) -> Option<Vec<(String, Vec<PathBuf>)>> {
    if !dir.is_dir() {
        return None;
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();

    let mut tables = Vec::new();
    for entry in entries {
        if entry.is_file() {
            if is_parquet_file(&entry) {
                let name = file_stem_or(&entry, "unknown_table");
                tables.push((name, vec![entry]));
            }
            // Non-.parquet files directly under the dataset root (manifests, _SUCCESS markers,
            // etc.) are silently skipped -- they are not tables.
        } else if entry.is_dir() {
            let name = entry.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "unknown_table".to_string());
            let mut files = Vec::new();
            collect_parquet_files_recursive(&entry, &mut files);
            files.sort();
            if !files.is_empty() {
                tables.push((name, files));
            }
        }
    }
    tables.sort_by(|a, b| a.0.cmp(&b.0));
    println!("NOTE: discovered {} table(s) under {}: {:?}", tables.len(), dir.display(), tables.iter().map(|(n, _)| n).collect::<Vec<_>>());
    Some(tables)
}

fn collect_parquet_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            collect_parquet_files_recursive(&path, out);
        } else if is_parquet_file(&path) {
            out.push(path);
        }
    }
}

fn file_stem_or(path: &Path, fallback: &str) -> String {
    path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| fallback.to_string())
}

/// `true` iff `path`'s extension is exactly `parquet`. Compares via `Option<&str>` (rather than
/// `OsStr`'s own `PartialEq` impls directly) so this is unambiguously correct regardless of
/// exactly which `OsStr` comparison traits are implemented.
fn is_parquet_file(path: &Path) -> bool {
    path.extension().and_then(|ext| ext.to_str()) == Some("parquet")
}

// ================================================================================================
// Accumulator types.
// ================================================================================================

#[derive(Default, Clone)]
struct ColumnStats {
    table: String,
    path: String,
    max_def_level: i16,
    max_rep_level: i16,
    encodings: BTreeSet<String>,
    compression_codecs: BTreeSet<String>,
    /// Sum of page-level `num_values` (nulls included) across all data pages of this column.
    total_values: u64,
    total_pages: u64,
    dictionary_pages: u64,
    /// Sum of `num_nulls` from `DataPageV2` pages only (V1 carries no such field).
    total_nulls_v2_known: u64,
}

#[derive(Default, Clone)]
struct LaneTotals {
    required_bitpacked: u64,
    required_rle: u64,
    optional_bitpacked: u64,
    optional_rle: u64,
    plain: u64,
}

#[derive(Default)]
struct DatasetCensus {
    dataset: String,
    files_scanned: Vec<String>,
    total_row_groups: usize,
    total_rows_from_metadata: i64,
    /// Sum of page-level `num_values` across every data page of every column, any encoding
    /// (the "share of decoded values" denominator).
    total_decoded_values: u64,
    columns: BTreeMap<String, ColumnStats>,
    lanes: LaneTotals,
    other_encoding_values: BTreeMap<String, u64>,
    rle_run_hist: kernel::RunLengthHistogram,
    bitpacked_run_hist: kernel::RunLengthHistogram,
    bitwidth_hist: kernel::BitWidthHistogram,
    /// Side fact (v19 fact 1 also asks about nested/repeated columns; the lane table itself
    /// only needs required/optional/PLAIN, per the task's explicit instruction).
    columns_with_rep_gt_0: BTreeSet<String>,
    /// Columns hitting the `max_def_level == 0 && max_rep_level > 0` edge case that neither the
    /// contract's "required-flat" nor "optional" definition literally covers (bucketed as
    /// optional here; see the module doc comment).
    edge_case_columns: BTreeSet<String>,
    columns_with_legacy_bitpacked_levels: BTreeSet<String>,
    anomalies: Vec<String>,
    anomalies_total_count: u64,
}

impl DatasetCensus {
    fn push_anomaly(&mut self, msg: String) {
        self.anomalies_total_count += 1;
        if self.anomalies.len() < MAX_REPORTED_ANOMALIES {
            self.anomalies.push(msg);
        }
    }
}

// ================================================================================================
// Per-page walk outcome: computed by pure-ish helper functions (taking only page bytes and
// caller-supplied context, no `DatasetCensus` access) so the accumulation step in
// `process_column_chunk` never needs two overlapping mutable borrows of `census`.
// ================================================================================================

#[derive(Default)]
struct PageOutcome {
    /// The page's own `num_values` (nulls included where applicable).
    page_values: u64,
    encoding_debug: String,
    /// Populated for `RLE_DICTIONARY`/`PLAIN_DICTIONARY` pages: the walked run composition plus
    /// the bit width (`k`) it was walked with (needed by the bit-width histogram, which is
    /// weighted by value count per page, not per run -- see `kernel::BitWidthHistogram`).
    dict_walk: Option<(kernel::IndexStreamWalk, u32)>,
    /// Populated for `PLAIN` pages: the value credited to the `PLAIN (any)` lane (the page's raw
    /// `num_values`, nulls included -- see the module doc comment on why this differs from how
    /// the dict lanes are counted).
    plain_values: Option<u64>,
    /// Populated for any other encoding, tagged with its `{:?}` label.
    other_encoding: Option<String>,
    /// `DataPageV2` only: this page's `num_nulls`.
    nulls_v2: Option<u64>,
    anomalies: Vec<String>,
    /// Set when this page's definition-level stream used the deprecated flat `BIT_PACKED`
    /// encoding (not the RLE-hybrid format), meaning this tool could skip past it but could not
    /// recover an exact non-null count from it -- recorded once per column, not once per page.
    legacy_bitpacked_levels: bool,
}

fn is_dict_encoding(e: Encoding) -> bool {
    matches!(e, Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY)
}

#[allow(deprecated)]
fn level_encoding_kind(e: Encoding) -> kernel::LevelEncodingKind {
    match e {
        Encoding::RLE => kernel::LevelEncodingKind::Rle,
        Encoding::BIT_PACKED => kernel::LevelEncodingKind::BitPacked,
        _ => kernel::LevelEncodingKind::Other,
    }
}

/// Finishes processing a `DataPage`/`DataPageV2`'s value-byte stream (`value_bytes`, i.e. `buf`
/// with any level blocks already stripped by the caller), given the page's declared encoding.
///
/// `expected_nonnull`, when `Some`, is an authoritative non-null value count the walked
/// dictionary-index total is checked against exactly (used for required columns, `DataPageV2`
/// columns via their explicit `num_nulls`, and `DataPageV1` optional columns whose definition
/// levels this tool successfully decoded). When `None` (a `DataPageV1` optional column whose
/// definition levels use the legacy `BIT_PACKED` encoding, or whose level stream itself turned
/// out malformed), only a much weaker sanity bound applies: the walked total must not *exceed*
/// `num_values` (which is always a valid upper bound, regardless of nulls).
fn finish_value_stream(
    outcome: &mut PageOutcome,
    value_bytes: &[u8],
    num_values: u32,
    encoding: Encoding,
    current_dict_num_values: Option<u32>,
    expected_nonnull: Option<u64>,
) {
    match encoding {
        Encoding::PLAIN => {
            outcome.plain_values = Some(num_values as u64);
        }
        _ if is_dict_encoding(encoding) => {
            if value_bytes.is_empty() {
                outcome.anomalies.push(
                    "RLE_DICTIONARY/PLAIN_DICTIONARY page has an empty value-byte stream (no bit-width byte present)"
                        .to_string(),
                );
                return;
            }
            let k_byte = value_bytes[0];
            let stream = &value_bytes[1..];

            match current_dict_num_values {
                Some(d) => {
                    let k_dict = kernel::dict_bit_width(d as u64);
                    if k_dict != k_byte {
                        outcome.anomalies.push(format!(
                            "dictionary bit-width disagreement: on-disk byte says k={k_byte}, dictionary cardinality {d} implies k={k_dict} (using the on-disk byte as authoritative per DictDecoder::set_data)"
                        ));
                    }
                }
                None => {
                    outcome.anomalies.push(
                        "RLE_DICTIONARY/PLAIN_DICTIONARY data page observed with no preceding DictionaryPage in this column chunk's page stream"
                            .to_string(),
                    );
                }
            }

            if k_byte > 32 {
                outcome.anomalies.push(format!(
                    "bit-width byte {k_byte} exceeds Parquet's documented max of 32 (arrow-rs's own DictDecoder would reject this page); skipping the RLE-hybrid walk for it"
                ));
                return;
            }

            let walk = kernel::walk_index_stream(stream, k_byte);
            if let Some(anomaly) = &walk.anomaly {
                outcome.anomalies.push(format!("dictionary-index stream structurally truncated: {}", anomaly.describe()));
            }

            let walked_total = walk.total_values();
            match expected_nonnull {
                Some(expected) if walked_total != expected => {
                    outcome.anomalies.push(format!(
                        "run-length self-consistency check failed: walked {walked_total} values from the dictionary-index stream, expected exactly {expected} non-null values"
                    ));
                }
                None if walked_total > num_values as u64 => {
                    outcome.anomalies.push(format!(
                        "run-length sum impossible: walked {walked_total} values from the dictionary-index stream, which exceeds this page's own num_values ({num_values})"
                    ));
                }
                _ => {}
            }

            outcome.dict_walk = Some((walk, k_byte as u32));
        }
        other => {
            outcome.other_encoding = Some(format!("{other:?}"));
        }
    }
}

/// Walks one `DataPage` (V1): locates the value-byte stream by skipping any present level
/// blocks, then delegates to [`finish_value_stream`].
///
/// For non-dictionary encodings (`PLAIN` and everything else), the level-skip is not even
/// attempted: this tool never needs to locate the value bytes for those (see the census
/// contract's explicit "just count PLAIN under a separate bucket... do not attempt to walk it"
/// instruction, generalized here to every non-dictionary encoding equally), so skipping the
/// level-parsing dance entirely both simplifies this function and avoids a spurious anomaly on
/// a page this tool was never going to walk regardless.
#[allow(clippy::too_many_arguments)]
fn walk_v1_data_page(
    buf: &[u8],
    num_values: u32,
    encoding: Encoding,
    def_level_encoding: Encoding,
    rep_level_encoding: Encoding,
    max_def: i16,
    max_rep: i16,
    current_dict_num_values: Option<u32>,
) -> PageOutcome {
    let mut outcome =
        PageOutcome { page_values: num_values as u64, encoding_debug: format!("{encoding:?}"), ..Default::default() };

    if !is_dict_encoding(encoding) {
        finish_value_stream(&mut outcome, &[], num_values, encoding, None, None);
        return outcome;
    }

    let mut offset = 0usize;

    if max_rep > 0 {
        let kind = level_encoding_kind(rep_level_encoding);
        match kernel::parse_v1_level_block(&buf[offset..], kind, num_values as u64, kernel::level_bit_width(max_rep)) {
            Ok((consumed, _payload)) => offset += consumed,
            Err(e) => {
                outcome.anomalies.push(format!(
                    "failed to skip V1 repetition-level block ({rep_level_encoding:?} encoding): {}",
                    e.describe()
                ));
                return outcome;
            }
        }
    }

    let mut nonnull_from_levels: Option<u64> = None;
    if max_def > 0 {
        let kind = level_encoding_kind(def_level_encoding);
        match kernel::parse_v1_level_block(&buf[offset..], kind, num_values as u64, kernel::level_bit_width(max_def)) {
            Ok((consumed, payload)) => {
                offset += consumed;
                match kind {
                    kernel::LevelEncodingKind::Rle => {
                        let lw = kernel::walk_rle_levels(payload, kernel::level_bit_width(max_def), max_def as u64);
                        match lw.anomaly {
                            None => nonnull_from_levels = Some(lw.nonnull_values),
                            Some(a) => outcome
                                .anomalies
                                .push(format!("definition-level stream structurally malformed: {}", a.describe())),
                        }
                    }
                    kernel::LevelEncodingKind::BitPacked => {
                        outcome.legacy_bitpacked_levels = true;
                    }
                    kernel::LevelEncodingKind::Other => {
                        // parse_v1_level_block would already have returned Err for this case.
                        unreachable!("parse_v1_level_block succeeded with LevelEncodingKind::Other");
                    }
                }
            }
            Err(e) => {
                outcome.anomalies.push(format!(
                    "failed to skip V1 definition-level block ({def_level_encoding:?} encoding): {}",
                    e.describe()
                ));
                return outcome;
            }
        }
    }

    // A `max_def_level == 0` column can never carry a null at this position, so `num_values` IS
    // the exact expected non-null count regardless of `max_rep_level` (nullability is governed
    // entirely by definition level; this holds even in the `max_def_level == 0 && max_rep_level
    // > 0` edge case discussed in the module doc comment -- that edge case only affects which
    // *lane* a page's values are credited to, not this expectation).
    let expected_nonnull = if max_def == 0 { Some(num_values as u64) } else { nonnull_from_levels };

    finish_value_stream(&mut outcome, &buf[offset..], num_values, encoding, current_dict_num_values, expected_nonnull);
    outcome
}

/// Walks one `DataPageV2`: the level byte-lengths are given directly, so no parsing is needed to
/// locate the value-byte stream, and the exact non-null count (`num_values - num_nulls`) is
/// always known from the page header -- no level *decoding* is ever needed for `DataPageV2`.
fn walk_v2_data_page(
    buf: &[u8],
    num_values: u32,
    num_nulls: u32,
    encoding: Encoding,
    def_levels_byte_len: u32,
    rep_levels_byte_len: u32,
    current_dict_num_values: Option<u32>,
) -> PageOutcome {
    let mut outcome = PageOutcome {
        page_values: num_values as u64,
        encoding_debug: format!("{encoding:?}"),
        nulls_v2: Some(num_nulls as u64),
        ..Default::default()
    };

    let level_prefix = rep_levels_byte_len as u64 + def_levels_byte_len as u64;
    if level_prefix > buf.len() as u64 {
        outcome.anomalies.push(format!(
            "V2 rep+def level byte lengths ({level_prefix}) exceed this page's buffer length ({})",
            buf.len()
        ));
        return outcome;
    }
    let value_bytes = &buf[level_prefix as usize..];
    let expected_nonnull = num_values.saturating_sub(num_nulls) as u64;
    finish_value_stream(&mut outcome, value_bytes, num_values, encoding, current_dict_num_values, Some(expected_nonnull));
    outcome
}

// ================================================================================================
// Top-level walk.
// ================================================================================================

fn run_census(dataset: &str, tables: &[(String, Vec<PathBuf>)]) -> DatasetCensus {
    let mut census = DatasetCensus { dataset: dataset.to_string(), ..Default::default() };

    for (table, files) in tables {
        for file_path in files {
            census.files_scanned.push(file_path.display().to_string());

            let file = match File::open(file_path) {
                Ok(f) => f,
                Err(e) => {
                    census.push_anomaly(format!("failed to open {}: {e}", file_path.display()));
                    continue;
                }
            };
            let reader = match SerializedFileReader::new(file) {
                Ok(r) => r,
                Err(e) => {
                    census.push_anomaly(format!("failed to read Parquet footer metadata for {}: {e}", file_path.display()));
                    continue;
                }
            };

            for rg_idx in 0..reader.num_row_groups() {
                let row_group_reader = match reader.get_row_group(rg_idx) {
                    Ok(rgr) => rgr,
                    Err(e) => {
                        census.push_anomaly(format!("{}: failed to open row group {rg_idx}: {e}", file_path.display()));
                        continue;
                    }
                };
                census.total_row_groups += 1;
                let rg_meta = row_group_reader.metadata();
                census.total_rows_from_metadata += rg_meta.num_rows();

                for col_idx in 0..row_group_reader.num_columns() {
                    process_column_chunk(&mut census, table, file_path, rg_idx, col_idx, row_group_reader.as_ref(), rg_meta);
                }
            }
        }
    }

    census
}

#[allow(clippy::too_many_arguments)]
fn process_column_chunk(
    census: &mut DatasetCensus,
    table: &str,
    file_path: &Path,
    rg_idx: usize,
    col_idx: usize,
    row_group_reader: &dyn RowGroupReader,
    rg_meta: &RowGroupMetaData,
) {
    let col_chunk_meta: &ColumnChunkMetaData = rg_meta.column(col_idx);
    let col_descr = col_chunk_meta.column_descr();
    let max_def = col_descr.max_def_level();
    let max_rep = col_descr.max_rep_level();
    let column_path = col_descr.path().to_string();
    let key = format!("{table}::{column_path}");

    // Scoped tightly to end the `census.columns` borrow (`stats`) before any call to
    // `census.push_anomaly` below, which needs `&mut census` as a whole -- a mutable borrow of
    // one field (`census.columns`, via `stats`) held live across a call needing the whole
    // struct would be a borrow-checker conflict, so `mismatch` is captured as a plain owned
    // value here and only acted on after `stats`'s borrow has ended.
    let mismatch: Option<(i16, i16)> = {
        let stats = census.columns.entry(key.clone()).or_insert_with(|| ColumnStats {
            table: table.to_string(),
            path: column_path.clone(),
            max_def_level: max_def,
            max_rep_level: max_rep,
            ..Default::default()
        });
        stats.compression_codecs.insert(col_chunk_meta.compression().to_string());
        if stats.max_def_level != max_def || stats.max_rep_level != max_rep {
            Some((stats.max_def_level, stats.max_rep_level))
        } else {
            None
        }
    };
    if let Some((prior_def, prior_rep)) = mismatch {
        census.push_anomaly(format!(
            "{}: row group {rg_idx} col {column_path}: max_def_level/max_rep_level ({max_def}/{max_rep}) disagrees with a prior row group of the same column ({prior_def}/{prior_rep}); schema should be uniform across row groups",
            file_path.display()
        ));
    }

    if max_rep > 0 {
        census.columns_with_rep_gt_0.insert(key.clone());
    }
    if max_def == 0 && max_rep > 0 {
        census.edge_case_columns.insert(key.clone());
    }
    let is_required = max_def == 0 && max_rep == 0;

    let mut page_reader = match row_group_reader.get_column_page_reader(col_idx) {
        Ok(pr) => pr,
        Err(e) => {
            census.push_anomaly(format!(
                "{}: row group {rg_idx} col {column_path}: failed to open page reader: {e}",
                file_path.display()
            ));
            return;
        }
    };

    let mut current_dict_num_values: Option<u32> = None;
    let mut page_idx: usize = 0;

    loop {
        let page = match page_reader.next() {
            None => break,
            Some(Ok(p)) => p,
            Some(Err(e)) => {
                census.push_anomaly(format!(
                    "{}: row group {rg_idx} col {column_path} page {page_idx}: failed to read/decompress page: {e}",
                    file_path.display()
                ));
                break;
            }
        };

        let outcome = match page {
            Page::DictionaryPage { num_values, .. } => {
                current_dict_num_values = Some(num_values);
                if let Some(stats) = census.columns.get_mut(&key) {
                    stats.dictionary_pages += 1;
                }
                continue;
            }
            Page::DataPage { buf, num_values, encoding, def_level_encoding, rep_level_encoding, .. } => {
                page_idx += 1;
                walk_v1_data_page(&buf, num_values, encoding, def_level_encoding, rep_level_encoding, max_def, max_rep, current_dict_num_values)
            }
            Page::DataPageV2 { buf, num_values, encoding, num_nulls, def_levels_byte_len, rep_levels_byte_len, .. } => {
                page_idx += 1;
                walk_v2_data_page(&buf, num_values, num_nulls, encoding, def_levels_byte_len, rep_levels_byte_len, current_dict_num_values)
            }
        };

        apply_page_outcome(census, &key, file_path, rg_idx, page_idx, is_required, outcome);
    }
}

fn apply_page_outcome(
    census: &mut DatasetCensus,
    key: &str,
    file_path: &Path,
    rg_idx: usize,
    page_idx: usize,
    is_required: bool,
    outcome: PageOutcome,
) {
    {
        let stats = census.columns.get_mut(key).expect("column stats entry created before any page is processed");
        stats.total_pages += 1;
        stats.total_values += outcome.page_values;
        stats.encodings.insert(outcome.encoding_debug.clone());
        if let Some(n) = outcome.nulls_v2 {
            stats.total_nulls_v2_known += n;
        }
    }
    census.total_decoded_values += outcome.page_values;

    if let Some((walk, k)) = &outcome.dict_walk {
        if is_required {
            census.lanes.required_bitpacked += walk.bitpacked_values;
            census.lanes.required_rle += walk.rle_values;
        } else {
            census.lanes.optional_bitpacked += walk.bitpacked_values;
            census.lanes.optional_rle += walk.rle_values;
        }
        census.rle_run_hist.merge(&walk.rle_run_lengths);
        census.bitpacked_run_hist.merge(&walk.bitpacked_run_lengths);
        census.bitwidth_hist.add(*k, walk.bitpacked_values);
    }
    if let Some(v) = outcome.plain_values {
        census.lanes.plain += v;
    }
    if let Some(label) = outcome.other_encoding {
        *census.other_encoding_values.entry(label).or_insert(0) += outcome.page_values;
    }
    if outcome.legacy_bitpacked_levels && census.columns_with_legacy_bitpacked_levels.insert(key.to_string()) {
        census.push_anomaly(format!(
            "col {key}: at least one page uses the deprecated flat BIT_PACKED definition-level encoding; \
             this tool can skip past it but cannot recover an exact non-null count from it, so the \
             run-length self-consistency check for this column's optional pages is limited to an \
             upper-bound-only comparison (noted once here, not once per page)"
        ));
    }
    for msg in outcome.anomalies {
        census.push_anomaly(format!("{}: rg{rg_idx} col={key} page={page_idx}: {msg}", file_path.display()));
    }
}

// ================================================================================================
// Reporting: human-readable table, then a clearly delimited JSON blob.
// ================================================================================================

fn pct(numerator: u64, denominator: u64) -> String {
    if denominator == 0 {
        "N/A (zero decoded values)".to_string()
    } else {
        format!("{:.4}%", (numerator as f64 / denominator as f64) * 100.0)
    }
}

fn frac(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn report(census: &DatasetCensus) {
    println!("{}", "=".repeat(100));
    println!("DATASET: {}", census.dataset);
    println!("{}", "=".repeat(100));
    println!("Files scanned ({}):", census.files_scanned.len());
    for f in &census.files_scanned {
        println!("  {f}");
    }
    println!("Row groups processed:        {}", census.total_row_groups);
    println!("Rows (row-group metadata, summed): {}", census.total_rows_from_metadata);
    println!("Total decoded values (all columns, all pages, any encoding): {}", census.total_decoded_values);
    println!();

    println!("--- Per-column summary ({} columns) ---", census.columns.len());
    println!(
        "{:<24} {:<40} {:>8} {:>8} {:>24} {:>10} {:>14} {:>7} {:>6}",
        "table", "column", "max_def", "max_rep", "encodings", "compress", "total_values", "pages", "dictP"
    );
    for stats in census.columns.values() {
        println!(
            "{:<24} {:<40} {:>8} {:>8} {:>24} {:>10} {:>14} {:>7} {:>6}",
            truncate(&stats.table, 24),
            truncate(&stats.path, 40),
            stats.max_def_level,
            stats.max_rep_level,
            truncate(&stats.encodings.iter().cloned().collect::<Vec<_>>().join("|"), 24),
            truncate(&stats.compression_codecs.iter().cloned().collect::<Vec<_>>().join("|"), 10),
            stats.total_values,
            stats.total_pages,
            stats.dictionary_pages,
        );
    }
    println!();

    println!("--- Frozen v19 lane table (deliverable) ---");
    println!("{:<40} | {:>20} | {:>18}", "Lane", "Addressable values", "Share of decoded");
    let rows: [(&str, u64); 5] = [
        ("required-flat, dict, bit-packed", census.lanes.required_bitpacked),
        ("required-flat, dict, RLE run", census.lanes.required_rle),
        ("optional, dict, bit-packed", census.lanes.optional_bitpacked),
        ("optional, dict, RLE run", census.lanes.optional_rle),
        ("PLAIN (any)", census.lanes.plain),
    ];
    for (label, value) in rows {
        println!("{label:<40} | {value:>20} | {:>18}", pct(value, census.total_decoded_values));
    }
    println!(
        "{:<40} | {:>20} | {:>18}",
        "unreachable under default Auto", "N/A (needs the", "dynamic component)"
    );
    println!(
        "  (note: PLAIN (any) counts each page's raw num_values, nulls included per the census \
         contract's explicit instruction; the two dict lanes above count only the walked, \
         non-null value total -- see the module doc comment for why this asymmetry is inherent \
         to the contract, not a bug here.)"
    );
    println!();

    if !census.other_encoding_values.is_empty() {
        println!("--- Other (non-PLAIN, non-dictionary) encodings observed (not part of the frozen lane table) ---");
        for (label, value) in &census.other_encoding_values {
            println!("  {label:<24} {value:>16}  ({} of decoded values)", pct(*value, census.total_decoded_values));
        }
        println!();
    }

    println!("--- Side facts (v19 fact 1 also asks about nested/repeated; lane table itself only needs required/optional/PLAIN) ---");
    println!("Columns observed with max_rep_level > 0 (nested/repeated): {}", census.columns_with_rep_gt_0.len());
    for c in &census.columns_with_rep_gt_0 {
        println!("  {c}");
    }
    println!(
        "Columns hitting the max_def_level==0 && max_rep_level>0 edge case (bucketed as optional; see module doc comment): {}",
        census.edge_case_columns.len()
    );
    for c in &census.edge_case_columns {
        println!("  {c}");
    }
    println!();

    println!("--- Run-length distribution (bucketed by run value-count) ---");
    println!("{:<12} {:>12} {:>16}   {:>12} {:>16}", "bucket", "RLE runs", "RLE values", "BP runs", "BP values");
    for i in 0..6 {
        println!(
            "{:<12} {:>12} {:>16}   {:>12} {:>16}",
            kernel::RUN_LENGTH_BUCKET_LABELS[i],
            census.rle_run_hist.run_counts[i],
            census.rle_run_hist.value_counts[i],
            census.bitpacked_run_hist.run_counts[i],
            census.bitpacked_run_hist.value_counts[i],
        );
    }
    println!();

    println!("--- Bit-width distribution of bit-packed runs (weighted by run value-count) ---");
    println!("{:<8} {:>16}", "bucket", "value_count");
    for i in 0..7 {
        println!("{:<8} {:>16}", kernel::BITWIDTH_BUCKET_LABELS[i], census.bitwidth_hist.value_counts[i]);
    }
    println!();

    println!(
        "--- Anomalies (showing {} of {} total) ---",
        census.anomalies.len(),
        census.anomalies_total_count
    );
    if census.anomalies.is_empty() {
        println!("  (none)");
    }
    for (i, a) in census.anomalies.iter().enumerate() {
        println!("  {}. {a}", i + 1);
    }
    println!();

    println!("----- BEGIN JSON: {} -----", census.dataset);
    let json = census_to_json(census);
    println!("{}", serde_json::to_string_pretty(&json).unwrap_or_else(|e| format!("{{\"error\": \"failed to serialize: {e}\"}}")));
    println!("----- END JSON: {} -----", census.dataset);
    println!();
}

/// Truncates `s` to at most `max` *characters* (not bytes) for display purposes, appending `~`
/// when truncated. Operates on `chars()` rather than byte-slicing so it can never panic on a
/// non-ASCII table/column name by slicing through the middle of a multi-byte UTF-8 sequence.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('~');
        out
    }
}

fn census_to_json(census: &DatasetCensus) -> serde_json::Value {
    let columns: Vec<serde_json::Value> = census
        .columns
        .values()
        .map(|s| {
            serde_json::json!({
                "table": s.table,
                "path": s.path,
                "max_def_level": s.max_def_level,
                "max_rep_level": s.max_rep_level,
                "encodings": s.encodings.iter().collect::<Vec<_>>(),
                "compression_codecs": s.compression_codecs.iter().collect::<Vec<_>>(),
                "total_values": s.total_values,
                "total_pages": s.total_pages,
                "dictionary_pages": s.dictionary_pages,
                "total_nulls_v2_known": s.total_nulls_v2_known,
            })
        })
        .collect();

    let lane_table = serde_json::json!([
        {
            "lane": "required-flat, dict, bit-packed",
            "addressable_values": census.lanes.required_bitpacked,
            "share_of_decoded_values": frac(census.lanes.required_bitpacked, census.total_decoded_values),
        },
        {
            "lane": "required-flat, dict, RLE run",
            "addressable_values": census.lanes.required_rle,
            "share_of_decoded_values": frac(census.lanes.required_rle, census.total_decoded_values),
        },
        {
            "lane": "optional, dict, bit-packed",
            "addressable_values": census.lanes.optional_bitpacked,
            "share_of_decoded_values": frac(census.lanes.optional_bitpacked, census.total_decoded_values),
        },
        {
            "lane": "optional, dict, RLE run",
            "addressable_values": census.lanes.optional_rle,
            "share_of_decoded_values": frac(census.lanes.optional_rle, census.total_decoded_values),
        },
        {
            "lane": "PLAIN (any)",
            "addressable_values": census.lanes.plain,
            "share_of_decoded_values": frac(census.lanes.plain, census.total_decoded_values),
            "note": "counts raw page num_values (nulls included), unlike the dict lanes above -- see report header",
        },
        {
            "lane": "unreachable under default Auto",
            "addressable_values": "N/A (needs the dynamic component)",
            "share_of_decoded_values": "N/A (needs the dynamic component)",
        },
    ]);

    serde_json::json!({
        "dataset": census.dataset,
        "files_scanned": census.files_scanned,
        "total_row_groups": census.total_row_groups,
        "total_rows_from_metadata": census.total_rows_from_metadata,
        "total_decoded_values": census.total_decoded_values,
        "columns": columns,
        "lane_table": lane_table,
        "other_encodings": census.other_encoding_values,
        "run_length_distribution": {
            "bucket_labels": kernel::RUN_LENGTH_BUCKET_LABELS,
            "rle_runs": {
                "run_counts": census.rle_run_hist.run_counts,
                "value_counts": census.rle_run_hist.value_counts,
            },
            "bitpacked_runs": {
                "run_counts": census.bitpacked_run_hist.run_counts,
                "value_counts": census.bitpacked_run_hist.value_counts,
            },
        },
        "bitwidth_distribution": {
            "bucket_labels": kernel::BITWIDTH_BUCKET_LABELS,
            "value_counts": census.bitwidth_hist.value_counts,
        },
        "side_facts": {
            "columns_with_rep_level_gt_0": census.columns_with_rep_gt_0.iter().collect::<Vec<_>>(),
            "edge_case_def0_rep_gt_0_columns": census.edge_case_columns.iter().collect::<Vec<_>>(),
            "columns_with_legacy_bitpacked_levels": census.columns_with_legacy_bitpacked_levels.iter().collect::<Vec<_>>(),
        },
        "anomalies": {
            "total_count": census.anomalies_total_count,
            "shown": census.anomalies,
            "truncated": census.anomalies_total_count as usize > census.anomalies.len(),
        },
    })
}
