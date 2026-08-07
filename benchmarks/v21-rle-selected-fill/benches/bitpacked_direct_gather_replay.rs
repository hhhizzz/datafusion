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

//! Replay-fixture Criterion bench for experiment `arrow-bitpacked-direct-gather-gate-v23`. See
//! `codex/experiments/arrow-bitpacked-direct-gather-gate-v23.md` -- this is the "Workload
//! replays" fixture family the frozen contract requires and calls this gate's *primary*
//! weighting input, over `bitpacked_direct_gather_grid.rs`'s synthetic grid.
//!
//! ## Fixture provenance
//!
//! Real, byte-for-byte Parquet dictionary-index page bytes captured during v22's Stage A,
//! embedded at compile time (`include_bytes!`) from `fixtures/v22-replay/<id>/{dict,page1}.bin`
//! -- copies of `codex/references/bmi-selection-pushdown/v22-replay-fixtures/<id>/*`, needed
//! here because a fresh K8s Run's container clones only the pinned `datafusion`/`arrow-rs`
//! commits, not this workspace's untracked `codex/references/` tree (the same reason this
//! program's static-census tool embeds its touched-filter/fixture-request specs via
//! `include_str!`). Only the 4 fixtures PROVENANCE.md tags `run kind: bitpacked` are used here
//! (this gate's own scope, not v24's `rle` fixtures): `26` (ClickBench, `hits.WatchID`, INT64,
//! required), `48`/`49`/`51` (TPC-DS, `catalog_returns.cr_returned_date_sk`
//! /`cr_refunded_hdemo_sk`, `catalog_sales.cs_sold_time_sk`, INT64, optional). Fixture `26` also
//! has 3 more `page*.bin` files that PROVENANCE.md does not mention are non-dictionary-encoded
//! -- verified directly by this file's author (not assumed) that `page2.bin`/`page3.bin`
//! /`page4.bin` are `PLAIN`-encoded literal i64 arrays (each file's byte length equals
//! `num_values * 8` exactly, with no room for any RLE/bit-packing header), a dictionary
//! -encoding fallback partway through that column chunk; only `page1.bin` (`PLAIN_DICTIONARY`)
//! is a dictionary-index stream and is the only one this file uses.
//!
//! ## Page byte layout (verified against the real bytes, not assumed)
//!
//! `page1.bin` is `Page::DataPage.buf` (or, in v22's static-census-tool capture, its byte-exact
//! equivalent) -- the *raw* V1 page bytes, before any level-block splitting
//! (`parquet::column::reader::GenericColumnReader::read_new_page`, which this file's parsing
//! mirrors). For a required column (`max_def_level=0`, fixture `26`), there is no level block at
//! all. For an optional column (`max_def_level=1`, fixtures `48`/`49`/`51`, all `RLE`-encoded
//! definition levels), the page starts with a 4-byte little-endian `i32` length prefix followed
//! by that many bytes of RLE-encoded definition levels
//! (`parquet::column::reader::parse_v1_level`'s `Encoding::RLE` arm) -- this file skips that
//! block without decoding it (see Precision below), matching how `DictIndexDecoder::new` is fed
//! its `data` by the real reader. Either way, the byte immediately after any level block is the
//! dictionary-index stream's own 1-byte bit-width prefix (`parquet::arrow::decoder
//! ::dictionary_index::DictIndexDecoder::new`: `let bit_width = data[0]`), with the remaining
//! bytes being exactly what `RleDecoder::set_data` consumes. All four byte layouts (one required,
//! three optional) were walked byte-for-byte with an independent Python script before writing
//! this file's Rust parser, confirming each page's declared value count is consumed with zero
//! leftover bytes.
//!
//! ## Precision
//!
//! Required-column selections (fixture `26`) are exact per v22's own `precision: exact` marker.
//! Optional-column fixtures' `num_values` in `meta.json` is the *logical* (nulls-included) page
//! value count; the *physical* (dictionary-index) count this file actually needs is smaller
//! whenever the page has real nulls (confirmed empirically: fixture `49` physically decodes
//! 120,620 of a logical 122,880; fixture `51` decodes 122,228 of 122,880; fixture `48` happens to
//! decode the full 122,880, i.e. this particular captured page has none). This file never
//! hardcodes a physical count -- each fixture's real value is determined by an untimed counting
//! pass through the real `RleDecoder::get_batch` (see [`count_physical_values`]) before any
//! digest check or timed measurement, so a wrong hand-derivation here cannot silently corrupt a
//! result. Per the experiment doc's v22-informed amendment 4, this file does not attempt to
//! reconstruct which physical positions are real nulls (v22 never captured that): every
//! selection mask below is a *synthesized* approximation calibrated to a real captured query's
//! density and mean selected-run length (see Selection synthesis), not the real bit-for-bit
//! mask -- the checked-baseline arm (`cursor`) shares the same approximation as every treatment
//! arm, so the cross-arm comparison stays fair even though it is not the literal historical mask.
//!
//! ## Selection synthesis
//!
//! For each fixture, 1-2 real captured `(density, mean_run)` points are pulled from v22's own
//! `v22-captured-selections-{clickbench,tpcds}.jsonl` (matched by `file_fingerprint`
//! /`row_group_idx`/query, cross-referenced against `PROVENANCE.md`'s table; see that directory's
//! `PROVENANCE.md` for how to reproduce this pairing) and hardcoded as `DensityPoint` constants,
//! chosen to span each fixture's own real density range rather than picking one arbitrary query.
//! A mask is synthesized via `kernel::generate_clustered_mask(physical_n_values, density,
//! mean_run, &mut rng)` for every point uniformly (this generalizes correctly down to near-1
//! mean-run values too: a geometric chain with `p_leave = 1/mean_run` at `mean_run ~= 1` produces
//! selected runs that are almost always exactly 1 long, statistically equivalent to iid random
//! selection at that density -- so this file does not need a separate "random vs clustered"
//! branch the way the synthetic grid does).
//!
//! ## Undersized dictionaries are the norm here, not the exception
//!
//! `direct_gather`'s preflight requires `dict.len() >= 1 << bit_width`. Measured directly from
//! these real fixtures: fixture `26`'s dictionary is exactly `1 << 17` (131,072, a coincidental
//! power of 2) so `direct_gather` exercises its real unchecked fast path there; fixtures `48`
//! /`49`/`51`'s dictionaries (492/7,200/12,410) are all smaller than their own bit width's `1 <<
//! k` (512/8,192/16,384) -- the common real-writer case documented in
//! `bitpacked_direct_gather.rs`'s "undersized dictionary" cell group -- so `direct_gather` always
//! takes its decode-all-then-filter fallback on every TPC-DS fixture here, while
//! `direct_gather_checked` (no preflight, always the real algorithm) is the only treatment arm
//! whose TPC-DS numbers reflect the direct-gather kernel itself. Any reading of this file's
//! results must account for this asymmetry explicitly, not average over it.
//!
//! ## Cache residency caveat
//!
//! Unlike the synthetic grid's 32-independently-generated-pages-per-cell round robin (which
//! defeats single-page cache residency by design), each fixture here is exactly one real page,
//! decoded repeatedly across Criterion's sampled iterations -- results benefit from cache
//! residency the grid's cells do not. Treat these numbers as directional per-page latency on
//! real byte content, not a memory-bandwidth-realistic throughput estimate.
//!
//! ## Arms and protocol
//!
//! Same 5-arm set, same untimed cross-arm FNV-1a-64 digest verification before any timed
//! measurement, same Criterion parameters as `bitpacked_direct_gather_grid.rs` -- see that file's
//! module doc comment for what each arm does.

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
const TOP_SEED: u64 = 0xC0FF_EE15_2026_0807_u64 ^ 0x7E91;

const RLE_CHUNK: usize = 1024;

// -----------------------------------------------------------------------------------------
// Fixture catalog: raw embedded bytes plus the real captured density/mean-run points to
// synthesize masks at. See this file's module doc comment for full provenance and precision
// notes.
// -----------------------------------------------------------------------------------------

struct RealFixture {
    id: u32,
    workload: &'static str,
    label: &'static str,
    precision: &'static str,
    dict_bytes: &'static [u8],
    page_bytes: &'static [u8],
    has_def_level_block: bool,
    /// `(query label, density, mean selected-run length)`, pulled from v22's own captured
    /// selections for this fixture's real `(file_fingerprint, row_group_idx)`.
    density_points: &'static [(&'static str, f64, f64)],
}

const FIXTURES: &[RealFixture] = &[
    RealFixture {
        id: 26,
        workload: "clickbench",
        label: "hits.WatchID",
        precision: "exact",
        dict_bytes: include_bytes!("../fixtures/v22-replay/26/dict.bin"),
        page_bytes: include_bytes!("../fixtures/v22-replay/26/page1.bin"),
        has_def_level_block: false,
        density_points: &[("q1", 0.027808, 2.19), ("q27", 0.999576, 9382.69)],
    },
    RealFixture {
        id: 48,
        workload: "tpcds",
        label: "catalog_returns.cr_returned_date_sk",
        precision: "page_granular_approx",
        dict_bytes: include_bytes!("../fixtures/v22-replay/48/dict.bin"),
        page_bytes: include_bytes!("../fixtures/v22-replay/48/page1.bin"),
        has_def_level_block: true,
        density_points: &[("q49", 0.008846, 1.01), ("q77", 0.109334, 1.17)],
    },
    RealFixture {
        id: 49,
        workload: "tpcds",
        label: "catalog_returns.cr_refunded_hdemo_sk",
        precision: "page_granular_approx",
        dict_bytes: include_bytes!("../fixtures/v22-replay/49/dict.bin"),
        page_bytes: include_bytes!("../fixtures/v22-replay/49/page1.bin"),
        has_def_level_block: true,
        density_points: &[("q5", 0.051090, 1.08)],
    },
    RealFixture {
        id: 51,
        workload: "tpcds",
        label: "catalog_sales.cs_sold_time_sk",
        precision: "page_granular_approx",
        dict_bytes: include_bytes!("../fixtures/v22-replay/51/dict.bin"),
        page_bytes: include_bytes!("../fixtures/v22-replay/51/page1.bin"),
        has_def_level_block: true,
        density_points: &[("q14", 0.994914, 195.92)],
    },
];

// -----------------------------------------------------------------------------------------
// Byte-layout parsing. See the module doc comment's "Page byte layout" section.
// -----------------------------------------------------------------------------------------

fn parse_dict(bytes: &'static [u8]) -> Vec<i64> {
    assert_eq!(bytes.len() % 8, 0, "dict.bin length must be a whole number of i64 values");
    bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Returns `(bit_width, rle_payload)` -- `rle_payload` is exactly what
/// `RleDecoder::set_data` consumes, matching `DictIndexDecoder::new`'s own framing.
fn parse_value_stream(page_bytes: &'static [u8], has_def_level_block: bool) -> (u8, Bytes) {
    let value_stream: &'static [u8] = if has_def_level_block {
        let data_size = i32::from_le_bytes(page_bytes[0..4].try_into().unwrap()) as usize;
        &page_bytes[4 + data_size..]
    } else {
        page_bytes
    };
    let bit_width = value_stream[0];
    (bit_width, Bytes::from_static(&value_stream[1..]))
}

/// Determines the real physical (dictionary-index) value count by reading the stream to
/// exhaustion with the stock, already-trusted `RleDecoder::get_batch` -- never hand-derived or
/// hardcoded, so a mistaken byte-layout assumption elsewhere in this file cannot silently
/// corrupt a downstream result (see the module doc comment's Precision section).
fn count_physical_values(bit_width: u8, rle_data: &Bytes) -> usize {
    let mut decoder = RleDecoder::new(bit_width);
    decoder.set_data(rle_data.clone()).expect("count pass: set_data failed");
    let mut idx_buf = [0i32; RLE_CHUNK];
    let mut total = 0usize;
    loop {
        let got = decoder.get_batch::<i32>(&mut idx_buf).expect("count pass: get_batch failed");
        total += got;
        if got < idx_buf.len() {
            break;
        }
    }
    total
}

struct LoadedFixture {
    id: u32,
    workload: &'static str,
    label: &'static str,
    precision: &'static str,
    dict: Vec<i64>,
    bit_width: u8,
    rle_data: Bytes,
    physical_n_values: usize,
    density_points: &'static [(&'static str, f64, f64)],
}

fn load_fixture(f: &RealFixture) -> LoadedFixture {
    let dict = parse_dict(f.dict_bytes);
    let (bit_width, rle_data) = parse_value_stream(f.page_bytes, f.has_def_level_block);
    let physical_n_values = count_physical_values(bit_width, &rle_data);
    LoadedFixture {
        id: f.id,
        workload: f.workload,
        label: f.label,
        precision: f.precision,
        dict,
        bit_width,
        rle_data,
        physical_n_values,
        density_points: f.density_points,
    }
}

// -----------------------------------------------------------------------------------------
// Fixture: one real page's decode inputs plus one synthesized selection.
// -----------------------------------------------------------------------------------------

struct ReplayPage {
    rle_data: Bytes,
    n_values: usize,
    mask_bytes: Vec<u8>,
    mask_words: Vec<u64>,
}

fn build_replay_page(loaded: &LoadedFixture, density: f64, mean_run: f64, seed: u64) -> ReplayPage {
    let mut rng = kernel::Xorshift64Star::new(seed);
    let mask_words = kernel::generate_clustered_mask(loaded.physical_n_values, density, mean_run, &mut rng);
    let mask_bytes = kernel::words_to_packed_bytes(&mask_words);
    ReplayPage {
        rle_data: loaded.rle_data.clone(),
        n_values: loaded.physical_n_values,
        mask_bytes,
        mask_words,
    }
}

// -----------------------------------------------------------------------------------------
// Per-arm runners (same shape as bitpacked_direct_gather_grid.rs, operating on ReplayPage).
// -----------------------------------------------------------------------------------------

fn run_cursor(decoder: &mut RleDecoder, page: &ReplayPage, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.rle_data.clone()).expect("cursor: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, page.n_values)
        .expect("cursor: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_cursor(dict, out, selection)
        .expect("cursor: get_batch_with_dict_selected_cursor failed");
    assert_eq!(consumed, page.n_values, "cursor: RleDecoder did not consume the whole page");
    written
}

fn run_direct_gather(decoder: &mut RleDecoder, page: &ReplayPage, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.rle_data.clone()).expect("direct_gather: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, page.n_values)
        .expect("direct_gather: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_direct_gather(dict, out, selection)
        .expect("direct_gather: get_batch_with_dict_selected_direct_gather failed");
    assert_eq!(consumed, page.n_values, "direct_gather: RleDecoder did not consume the whole page");
    written
}

fn run_direct_gather_checked(decoder: &mut RleDecoder, page: &ReplayPage, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.rle_data.clone()).expect("direct_gather_checked: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, page.n_values)
        .expect("direct_gather_checked: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_direct_gather_checked(dict, out, selection)
        .expect("direct_gather_checked: get_batch_with_dict_selected_direct_gather_checked failed");
    assert_eq!(consumed, page.n_values, "direct_gather_checked: RleDecoder did not consume the whole page");
    written
}

fn run_tiered(decoder: &mut RleDecoder, page: &ReplayPage, dict: &[i64], out: &mut [i64]) -> usize {
    decoder.set_data(page.rle_data.clone()).expect("tiered: set_data failed");
    let selection = PackedSelection::new(&page.mask_bytes, 0, page.n_values)
        .expect("tiered: PackedSelection::new failed");
    let (consumed, written) = decoder
        .get_batch_with_dict_selected_direct_gather_tiered(dict, out, selection)
        .expect("tiered: get_batch_with_dict_selected_direct_gather_tiered failed");
    assert_eq!(consumed, page.n_values, "tiered: RleDecoder did not consume the whole page");
    written
}

fn run_decode_all_indices_compact(
    decoder: &mut RleDecoder,
    page: &ReplayPage,
    dict: &[i64],
    idx_buf: &mut [i32; RLE_CHUNK],
    out: &mut Vec<i64>,
) {
    decoder.set_data(page.rle_data.clone()).expect("decode_all_indices_compact: set_data failed");
    let mut processed = 0usize;
    while processed < page.n_values {
        let chunk_len = (page.n_values - processed).min(RLE_CHUNK);
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

fn run_materialize_then_filter(
    decoder: &mut RleDecoder,
    page: &ReplayPage,
    dict: &[i64],
    val_buf: &mut [i64; RLE_CHUNK],
    out: &mut Vec<i64>,
) {
    decoder.set_data(page.rle_data.clone()).expect("materialize_then_filter: set_data failed");
    let mut processed = 0usize;
    while processed < page.n_values {
        let chunk_len = (page.n_values - processed).min(RLE_CHUNK);
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
// Cross-arm digest verification (untimed, at setup).
// -----------------------------------------------------------------------------------------

fn verify_digests(cell_desc: &str, named: &[(&str, u64)]) {
    let reference = named[0].1;
    let mut diverged: Vec<String> = Vec::new();
    for &(name, d) in named.iter().skip(1) {
        if d != reference {
            diverged.push(format!("{name} (digest {d:#018x})"));
        }
    }
    if !diverged.is_empty() {
        panic!(
            "bitpacked_direct_gather_replay cell {cell_desc}: cross-arm digest mismatch. \
             Reference ({}) digest = {reference:#018x}. Diverged: [{}]",
            named[0].0,
            diverged.join(", ")
        );
    }
}

// -----------------------------------------------------------------------------------------
// Per-(fixture, density point) benchmarking.
// -----------------------------------------------------------------------------------------

fn bench_density_point(c: &mut Criterion, loaded: &LoadedFixture, query: &str, density: f64, mean_run: f64, seed: u64) {
    let page = build_replay_page(loaded, density, mean_run, seed);
    let max_selected = kernel::popcount_words(&page.mask_words);
    let dict = &loaded.dict;
    let k = loaded.bit_width;

    let cell_desc = format!(
        "{}/id{}/{}/{query}/density{:.4}",
        loaded.workload, loaded.id, loaded.label, density
    );

    // --- untimed cross-arm correctness check ---
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

        let written_cursor = run_cursor(&mut decoder_cursor, &page, dict, &mut out_cursor);
        let written_dg = run_direct_gather(&mut decoder_dg, &page, dict, &mut out_dg);
        let written_dgc = run_direct_gather_checked(&mut decoder_dgc, &page, dict, &mut out_dgc);
        let written_tiered = run_tiered(&mut decoder_tiered, &page, dict, &mut out_tiered);
        run_decode_all_indices_compact(&mut decoder_c, &page, dict, &mut idx_buf, &mut out_c);
        run_materialize_then_filter(&mut decoder_d, &page, dict, &mut val_buf, &mut out_d);

        verify_digests(
            &cell_desc,
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

    // --- timed loop: repeatedly decoding the same real page (see module doc comment's Cache
    // residency caveat) ---
    let mut group = c.benchmark_group("bitpacked_direct_gather_replay");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs_f64(3.0));

    {
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_cursor(&mut decoder, &page, dict, &mut out);
        group.bench_function(format!("cursor/{cell_desc}"), |b| {
            b.iter(|| {
                let written = run_cursor(&mut decoder, &page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_direct_gather(&mut decoder, &page, dict, &mut out);
        group.bench_function(format!("direct_gather/{cell_desc}"), |b| {
            b.iter(|| {
                let written = run_direct_gather(&mut decoder, &page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_direct_gather_checked(&mut decoder, &page, dict, &mut out);
        group.bench_function(format!("direct_gather_checked/{cell_desc}"), |b| {
            b.iter(|| {
                let written = run_direct_gather_checked(&mut decoder, &page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut decoder = RleDecoder::new(k);
        let mut out = vec![0i64; max_selected];
        let _ = run_tiered(&mut decoder, &page, dict, &mut out);
        group.bench_function(format!("tiered/{cell_desc}"), |b| {
            b.iter(|| {
                let written = run_tiered(&mut decoder, &page, dict, &mut out);
                hint::black_box(&out[..written]);
            });
        });
    }
    {
        let mut decoder = RleDecoder::new(k);
        let mut idx_buf = [0i32; RLE_CHUNK];
        let mut out = Vec::with_capacity(max_selected);
        run_decode_all_indices_compact(&mut decoder, &page, dict, &mut idx_buf, &mut out);
        group.bench_function(format!("decode_all_indices_compact/{cell_desc}"), |b| {
            b.iter(|| {
                out.clear();
                run_decode_all_indices_compact(&mut decoder, &page, dict, &mut idx_buf, &mut out);
                hint::black_box(out.as_slice());
            });
        });
    }
    {
        let mut decoder = RleDecoder::new(k);
        let mut val_buf = [0i64; RLE_CHUNK];
        let mut out = Vec::with_capacity(max_selected);
        run_materialize_then_filter(&mut decoder, &page, dict, &mut val_buf, &mut out);
        group.bench_function(format!("materialize_then_filter/{cell_desc}"), |b| {
            b.iter(|| {
                out.clear();
                run_materialize_then_filter(&mut decoder, &page, dict, &mut val_buf, &mut out);
                hint::black_box(out.as_slice());
            });
        });
    }
    group.finish();

    eprintln!(
        "bitpacked_direct_gather_replay: {cell_desc}: precision={} physical_n_values={} \
         dict.len()={} bit_width={} direct_gather_safe_dict_condition={}",
        loaded.precision,
        loaded.physical_n_values,
        loaded.dict.len(),
        k,
        loaded.dict.len() as u64 >= (1u64 << k)
    );
}

fn run_full_matrix(c: &mut Criterion) {
    for (fixture_idx, f) in FIXTURES.iter().enumerate() {
        let loaded = load_fixture(f);
        for (point_idx, &(query, density, mean_run)) in loaded.density_points.iter().enumerate() {
            let seed = kernel::derive_seed(
                kernel::derive_seed(TOP_SEED, fixture_idx as u64),
                point_idx as u64,
            );
            bench_density_point(c, &loaded, query, density, mean_run, seed);
        }
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    run_full_matrix(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
