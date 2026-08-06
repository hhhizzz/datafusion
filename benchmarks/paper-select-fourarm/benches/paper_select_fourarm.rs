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

//! Paper-Select four-arm paper-scale microbench (experiment `arrow-paper-select-fourarm-v18`).
//!
//! See `codex/experiments/arrow-paper-select-fourarm-v18.md` for the frozen design this file
//! implements. This bench is meaningless without BMI2 (arm A's whole point is measuring a
//! faithfully paper-shaped PEXT kernel) and is x86_64-only by contract; `criterion_benchmark`
//! aborts immediately rather than silently reporting a scalar-only "passing" run.

use criterion::{Criterion, criterion_group, criterion_main};

#[path = "paper_select_fourarm/kernel.rs"]
mod kernel;

/// Entry point registered with Criterion. Gates on BMI2 availability *before* building or
/// benchmarking anything, then (on x86_64 with BMI2) delegates to [`x86_impl::run_full_matrix`].
fn criterion_benchmark(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    {
        // BMI2 capability is decided exactly once here, outside of any loop; every arm-A call
        // site downstream unconditionally assumes it (see kernel::select_run_bmi2's safety
        // contract).
        if !std::arch::is_x86_feature_detected!("bmi2") {
            eprintln!(
                "paper-select-fourarm-bench: BMI2 not detected on this x86_64 host. This bench \
                 exists to measure a faithfully paper-shaped BMI2 PEXT kernel (arm A, \
                 `paper_pext`); running it without BMI2 would silently fall back to a \
                 meaningless scalar measurement. Refusing to run. Aborting."
            );
            std::process::exit(1);
        }
        x86_impl::run_full_matrix(c);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = c;
        eprintln!(
            "paper-select-fourarm-bench: this bench is x86_64-only by contract (it exists to \
             measure a BMI2 PEXT kernel, which only exists on x86_64). Refusing to silently \
             run a scalar-only measurement and report it as a passing bench run. Aborting."
        );
        std::process::exit(1);
    }
}

/// All of the substantive harness logic lives behind `target_arch = "x86_64"`: several of the
/// helpers below call `kernel::select_run_bmi2`, which is itself only compiled for x86_64
/// (see kernel.rs), so referencing it from code that could be compiled for other targets would
/// fail to build there. Gating the whole module keeps that boundary simple and matches the
/// contract's "must never silently run scalar-only" intent -- on non-x86_64 targets nothing
/// past the abort in `criterion_benchmark` above is even compiled.
#[cfg(target_arch = "x86_64")]
mod x86_impl {
    use std::hint;
    use std::time::Duration;

    use bytes::Bytes;
    use criterion::Criterion;
    use parquet::encodings::rle::RleDecoder;

    use crate::kernel;

    /// Single top-level seed constant the whole fixture matrix is reproducible from.
    const TOP_SEED: u64 = 0x9E3779B97F4A7C15;
    const PAGES_PER_CELL: usize = 32;
    /// Production RLE batch granularity used by arms C and D.
    const RLE_CHUNK: usize = 1024;
    /// Real-world Parquet writer bit-packed-run cap: 63 groups of 8 values keeps the LEB128
    /// run-header encodable in a single byte (see arrow-rs parquet/src/encodings/rle.rs
    /// `test_long_run`, and the format doc comment atop the same file).
    const WRITER_REAL_RUN_LEN: usize = 504;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Shape {
        Random,
        Clustered,
        Dense,
        WriterReal,
    }

    impl Shape {
        fn label(self) -> &'static str {
            match self {
                Shape::Random => "random",
                Shape::Clustered => "clustered",
                Shape::Dense => "dense",
                Shape::WriterReal => "writer_real",
            }
        }
    }

    struct CellSpec {
        k: u32,
        shape: Shape,
        survival: f64,
        denom: u32,
    }

    /// 4 k-values x (3 survivals x 2 shapes + 1 dense) + 1 guard = 29 cells.
    fn build_cells() -> Vec<CellSpec> {
        const KS: [u32; 4] = [2, 5, 8, 12];
        const SURVIVALS: [(f64, u32); 3] = [(1.0 / 64.0, 64), (1.0 / 16.0, 16), (1.0 / 4.0, 4)];

        let mut cells = Vec::with_capacity(29);
        for &k in &KS {
            for shape in [Shape::Random, Shape::Clustered] {
                for &(survival, denom) in &SURVIVALS {
                    cells.push(CellSpec { k, shape, survival, denom });
                }
            }
            cells.push(CellSpec { k, shape: Shape::Dense, survival: 1.0, denom: 1 });
        }
        cells.push(CellSpec {
            k: 8,
            shape: Shape::WriterReal,
            survival: 1.0 / 16.0,
            denom: 16,
        });
        debug_assert_eq!(cells.len(), 29);
        cells
    }

    /// `n = floor(2^23 / k)` rounded down to a multiple of 512 (long-run page value count).
    fn long_run_n_values(k: u32) -> usize {
        let raw = (1usize << 23) / k as usize;
        raw - raw % 512
    }

    // -----------------------------------------------------------------------------------
    // Long-run (non-guard) pages: one bit-packed hybrid run per page.
    // -----------------------------------------------------------------------------------

    struct Page {
        /// LEB128 bit-packed-run header + packed payload + 8 zero pad bytes.
        buffer: Bytes,
        /// Byte offset of the packed payload (right after the header) within `buffer`.
        payload_offset: usize,
        n_values: usize,
        sel_words: Vec<u64>,
    }

    fn build_page(k: u32, n_values: usize, shape: Shape, survival: f64, seed: u64) -> Page {
        debug_assert_ne!(shape, Shape::WriterReal, "writer_real pages use build_writer_real_page");
        let mut rng = kernel::Xorshift64Star::new(seed);
        let mask = (1u64 << k) - 1;
        let values: Vec<u32> = (0..n_values).map(|_| (rng.next_u64() & mask) as u32).collect();

        let mut buffer = Vec::new();
        let payload_offset = kernel::write_bitpacked_run(&mut buffer, &values, k);
        buffer.extend_from_slice(&[0u8; 8]);

        let sel_words = match shape {
            Shape::Random => kernel::generate_random_mask(n_values, survival, &mut rng),
            Shape::Clustered => kernel::generate_clustered_mask(n_values, survival, &mut rng),
            Shape::Dense => kernel::generate_dense_mask(n_values),
            Shape::WriterReal => unreachable!(),
        };

        Page {
            buffer: Bytes::from(buffer),
            payload_offset,
            n_values,
            sel_words,
        }
    }

    // -----------------------------------------------------------------------------------
    // writer_real guard cell: many 504-value runs packed back-to-back in one buffer.
    // -----------------------------------------------------------------------------------

    struct RunInfo {
        byte_offset: usize,
        /// This run's 504-value slice of the page-level mask, repacked 0-based (8 words, last
        /// word only 56 valid bits, rest zeroed) -- run boundaries are not 64-aligned, so this
        /// cannot be a plain slice of the page-level `sel_words`.
        sel_words: Vec<u64>,
    }

    struct WriterRealPage {
        buffer: Bytes,
        /// Total value count across all runs (same order of magnitude as the k=8 long-run
        /// twin's page: `(long_run_n_values(8) / 504) * 504`, i.e. as many whole 504-value
        /// runs as fit -- see the comment in `build_writer_real_page`).
        n_values: usize,
        /// Flat page-level mask across all `n_values`, used by arms C/D (which decode the
        /// whole multi-run stream as usual via RleDecoder's transparent run-boundary handling).
        sel_words: Vec<u64>,
        /// Per-run byte offsets and repacked masks, used by arms A/B (which have no notion of
        /// the bit-packed-run headers and must be invoked once per run).
        runs: Vec<RunInfo>,
    }

    fn build_writer_real_page(k: u32, survival: f64, seed: u64) -> WriterRealPage {
        // "Same total payload as the long-run k=8 cells": floor-divide the k=8 long-run twin's
        // value count into as many whole 504-value runs as fit. 1_048_576 / 504 = 2080 runs
        // (2080 * 504 = 1_048_320 values, ~99.98% of the twin's 1_048_576 -- close enough to be
        // the same order of magnitude/packed working set for the G4 entry-tax comparison; noted
        // in the experiment report since the contract does not pin an exact run count here).
        let twin_n_values = long_run_n_values(8);
        let num_runs = twin_n_values / WRITER_REAL_RUN_LEN;
        let n_values = num_runs * WRITER_REAL_RUN_LEN;

        let mut rng = kernel::Xorshift64Star::new(seed);
        let mask = (1u64 << k) - 1;

        let mut buffer = Vec::new();
        let mut run_offsets = Vec::with_capacity(num_runs);
        for _ in 0..num_runs {
            let values: Vec<u32> = (0..WRITER_REAL_RUN_LEN).map(|_| (rng.next_u64() & mask) as u32).collect();
            run_offsets.push(kernel::write_bitpacked_run(&mut buffer, &values, k));
        }
        buffer.extend_from_slice(&[0u8; 8]);

        let sel_words = kernel::generate_random_mask(n_values, survival, &mut rng);
        let runs = run_offsets
            .into_iter()
            .enumerate()
            .map(|(run_idx, byte_offset)| RunInfo {
                byte_offset,
                sel_words: kernel::extract_bit_range(&sel_words, run_idx * WRITER_REAL_RUN_LEN, WRITER_REAL_RUN_LEN),
            })
            .collect();

        WriterRealPage {
            buffer: Bytes::from(buffer),
            n_values,
            sel_words,
            runs,
        }
    }

    // -----------------------------------------------------------------------------------
    // Per-arm runners. Each appends this page's (or, for A/B on writer_real, this page's
    // per-run) selected values to `out` in order; callers `out.clear()` first.
    // -----------------------------------------------------------------------------------

    fn run_arm_a_page(page: &Page, dict: &[i64], out: &mut Vec<i64>) {
        // SAFETY: BMI2 checked once at process startup in `criterion_benchmark` before
        // `run_full_matrix` (and therefore any arm-A call) ever runs.
        unsafe {
            kernel::select_run_bmi2(&page.buffer, page.payload_offset, page.n_values, &page.sel_words, dict, out);
        }
    }

    fn run_arm_a_writer_real(page: &WriterRealPage, dict: &[i64], out: &mut Vec<i64>) {
        for run in &page.runs {
            // SAFETY: see run_arm_a_page.
            unsafe {
                kernel::select_run_bmi2(&page.buffer, run.byte_offset, WRITER_REAL_RUN_LEN, &run.sel_words, dict, out);
            }
        }
    }

    fn run_arm_b_page(page: &Page, dict: &[i64], out: &mut Vec<i64>) {
        kernel::sparse_direct(&page.buffer, page.payload_offset, page.n_values, &page.sel_words, dict, out);
    }

    fn run_arm_b_writer_real(page: &WriterRealPage, dict: &[i64], out: &mut Vec<i64>) {
        for run in &page.runs {
            kernel::sparse_direct(&page.buffer, run.byte_offset, WRITER_REAL_RUN_LEN, &run.sel_words, dict, out);
        }
    }

    /// Arm C (`decode_all_indices_compact`): production `RleDecoder::get_batch::<i32>` in
    /// 1024-value chunks, then a per-selection-word set-bit walk gathering `dict[idx]`.
    /// `decoder` is reused across calls (its own internal 1024-`i32` scratch buffer is
    /// allocated lazily once and never reallocated); `idx_buf` is the caller's reused chunk
    /// buffer.
    fn run_arm_c(
        decoder: &mut RleDecoder,
        buffer: &Bytes,
        n_values: usize,
        sel_words: &[u64],
        dict: &[i64],
        idx_buf: &mut [i32; RLE_CHUNK],
        out: &mut Vec<i64>,
    ) {
        decoder.set_data(buffer.clone()).expect("arm C: RleDecoder::set_data failed");
        let mut processed = 0usize;
        while processed < n_values {
            let chunk_len = (n_values - processed).min(RLE_CHUNK);
            let got = decoder
                .get_batch::<i32>(&mut idx_buf[..chunk_len])
                .expect("arm C: RleDecoder::get_batch failed");
            assert_eq!(got, chunk_len, "arm C: RleDecoder produced fewer values than the run promised");

            let word_idx0 = processed / 64;
            let words_in_chunk = chunk_len.div_ceil(64);
            for wi in 0..words_in_chunk {
                let mut word = sel_words[word_idx0 + wi];
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

    /// Arm D (`materialize_then_filter`): production `RleDecoder::get_batch_with_dict::<i64>`
    /// in 1024-value chunks, fully materializing *every* chunk (never skipped, even when a
    /// chunk's selection is empty -- that is what makes this a faithful "decode all, then
    /// filter" baseline rather than an optimized one), then a set-bit walk copying survivors.
    fn run_arm_d(
        decoder: &mut RleDecoder,
        buffer: &Bytes,
        n_values: usize,
        sel_words: &[u64],
        dict: &[i64],
        val_buf: &mut [i64; RLE_CHUNK],
        out: &mut Vec<i64>,
    ) {
        decoder.set_data(buffer.clone()).expect("arm D: RleDecoder::set_data failed");
        let mut processed = 0usize;
        while processed < n_values {
            let chunk_len = (n_values - processed).min(RLE_CHUNK);
            let got = decoder
                .get_batch_with_dict::<i64>(dict, &mut val_buf[..chunk_len], chunk_len)
                .expect("arm D: RleDecoder::get_batch_with_dict failed");
            assert_eq!(got, chunk_len, "arm D: RleDecoder produced fewer values than the run promised");

            let word_idx0 = processed / 64;
            let words_in_chunk = chunk_len.div_ceil(64);
            for wi in 0..words_in_chunk {
                let mut word = sel_words[word_idx0 + wi];
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

    // -----------------------------------------------------------------------------------
    // Cross-arm digest verification (untimed, at setup).
    // -----------------------------------------------------------------------------------

    /// Compares 5 digests (the 4 named arms, plus arm A's scalar twin as an extra ground-truth
    /// cross-check -- kernel.rs documents that the BMI2 kernel must match the scalar twin
    /// bit-for-bit) against the first (`paper_pext`) and panics naming exactly which arm(s)
    /// diverged, the cell, and the page index, if any differ.
    #[allow(clippy::too_many_arguments)]
    fn verify_digests(
        k: u32,
        shape: &str,
        denom: u32,
        page_idx: usize,
        paper_pext: u64,
        select_run_scalar: u64,
        sparse_direct: u64,
        decode_all_indices_compact: u64,
        materialize_then_filter: u64,
    ) {
        let named: [(&str, u64); 5] = [
            ("paper_pext", paper_pext),
            ("select_run_scalar (ground-truth twin)", select_run_scalar),
            ("sparse_direct", sparse_direct),
            ("decode_all_indices_compact", decode_all_indices_compact),
            ("materialize_then_filter", materialize_then_filter),
        ];
        let reference = named[0].1;
        let mut diverged: Vec<String> = Vec::new();
        for &(name, d) in named.iter().skip(1) {
            if d != reference {
                diverged.push(format!("{name} (digest {d:#018x})"));
            }
        }
        if !diverged.is_empty() {
            panic!(
                "paper4 cell k={k} shape={shape} s=1/{denom} page={page_idx}: cross-arm digest \
                 mismatch. Reference (paper_pext) digest = {reference:#018x}. Diverged: \
                 [{}]",
                diverged.join(", ")
            );
        }
    }

    // -----------------------------------------------------------------------------------
    // Per-cell benchmarking.
    // -----------------------------------------------------------------------------------

    fn bench_cell(c: &mut Criterion, cell: &CellSpec, cell_index: usize) {
        let dict_seed = TOP_SEED.wrapping_add(cell_index as u64);
        let dict = kernel::generate_dict(cell.k, dict_seed);
        assert_eq!(dict.len(), 1usize << cell.k, "fixture invariant: dict.len() == 1<<k");

        let cell_base_seed = kernel::derive_seed(TOP_SEED, cell_index as u64);
        let n_values = long_run_n_values(cell.k);
        let pages: Vec<Page> = (0..PAGES_PER_CELL)
            .map(|page_idx| {
                let seed = kernel::derive_seed(cell_base_seed, page_idx as u64);
                build_page(cell.k, n_values, cell.shape, cell.survival, seed)
            })
            .collect();

        let max_selected = pages
            .iter()
            .map(|p| p.sel_words.iter().map(|w| w.count_ones() as usize).sum::<usize>())
            .max()
            .unwrap_or(0);

        // --- untimed cross-arm correctness check (setup only, never inside a timed closure) ---
        {
            let mut decoder_c = RleDecoder::new(cell.k as u8);
            let mut decoder_d = RleDecoder::new(cell.k as u8);
            let mut idx_buf = [0i32; RLE_CHUNK];
            let mut val_buf = [0i64; RLE_CHUNK];
            let mut out_a = Vec::with_capacity(max_selected);
            let mut out_scalar = Vec::with_capacity(max_selected);
            let mut out_b = Vec::with_capacity(max_selected);
            let mut out_c = Vec::with_capacity(max_selected);
            let mut out_d = Vec::with_capacity(max_selected);

            for (page_idx, page) in pages.iter().enumerate() {
                out_a.clear();
                run_arm_a_page(page, &dict, &mut out_a);
                out_scalar.clear();
                kernel::select_run_scalar(&page.buffer, page.payload_offset, page.n_values, &page.sel_words, &dict, &mut out_scalar);
                out_b.clear();
                run_arm_b_page(page, &dict, &mut out_b);
                out_c.clear();
                run_arm_c(&mut decoder_c, &page.buffer, page.n_values, &page.sel_words, &dict, &mut idx_buf, &mut out_c);
                out_d.clear();
                run_arm_d(&mut decoder_d, &page.buffer, page.n_values, &page.sel_words, &dict, &mut val_buf, &mut out_d);

                verify_digests(
                    cell.k,
                    cell.shape.label(),
                    cell.denom,
                    page_idx,
                    kernel::fnv1a64(&out_a),
                    kernel::fnv1a64(&out_scalar),
                    kernel::fnv1a64(&out_b),
                    kernel::fnv1a64(&out_c),
                    kernel::fnv1a64(&out_d),
                );
            }
        }

        // --- timed loop: one Criterion BenchmarkGroup for this cell ---
        let mut group = c.benchmark_group("paper4");
        group.sample_size(12);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs_f64(2.5));

        let k = cell.k;
        let shape = cell.shape.label();
        let denom = cell.denom;

        {
            let mut cursor = 0usize;
            let mut out = Vec::with_capacity(max_selected);
            group.bench_function(format!("paper_pext/k{k}/{shape}/s{denom}"), |b| {
                b.iter(|| {
                    let page = &pages[cursor];
                    cursor = (cursor + 1) % pages.len();
                    out.clear();
                    run_arm_a_page(page, &dict, &mut out);
                    hint::black_box(out.as_slice());
                });
            });
        }
        {
            let mut cursor = 0usize;
            let mut out = Vec::with_capacity(max_selected);
            group.bench_function(format!("sparse_direct/k{k}/{shape}/s{denom}"), |b| {
                b.iter(|| {
                    let page = &pages[cursor];
                    cursor = (cursor + 1) % pages.len();
                    out.clear();
                    run_arm_b_page(page, &dict, &mut out);
                    hint::black_box(out.as_slice());
                });
            });
        }
        {
            let mut cursor = 0usize;
            let mut out = Vec::with_capacity(max_selected);
            let mut decoder = RleDecoder::new(k as u8);
            let mut idx_buf = [0i32; RLE_CHUNK];
            // Prime any lazily-allocated decoder scratch state (untimed) so the closure below
            // never risks a first-touch heap allocation during a measured sample; `out` is
            // cleared on the first real iteration regardless.
            run_arm_c(&mut decoder, &pages[0].buffer, pages[0].n_values, &pages[0].sel_words, &dict, &mut idx_buf, &mut out);
            group.bench_function(format!("decode_all_indices_compact/k{k}/{shape}/s{denom}"), |b| {
                b.iter(|| {
                    let page = &pages[cursor];
                    cursor = (cursor + 1) % pages.len();
                    out.clear();
                    run_arm_c(&mut decoder, &page.buffer, page.n_values, &page.sel_words, &dict, &mut idx_buf, &mut out);
                    hint::black_box(out.as_slice());
                });
            });
        }
        {
            let mut cursor = 0usize;
            let mut out = Vec::with_capacity(max_selected);
            let mut decoder = RleDecoder::new(k as u8);
            let mut val_buf = [0i64; RLE_CHUNK];
            // Prime the decoder's lazily-allocated 1024-i64 scratch buffer (untimed); see the
            // comment on the arm C prime call above.
            run_arm_d(&mut decoder, &pages[0].buffer, pages[0].n_values, &pages[0].sel_words, &dict, &mut val_buf, &mut out);
            group.bench_function(format!("materialize_then_filter/k{k}/{shape}/s{denom}"), |b| {
                b.iter(|| {
                    let page = &pages[cursor];
                    cursor = (cursor + 1) % pages.len();
                    out.clear();
                    run_arm_d(&mut decoder, &page.buffer, page.n_values, &page.sel_words, &dict, &mut val_buf, &mut out);
                    hint::black_box(out.as_slice());
                });
            });
        }
        group.finish();

        // `pages`, `dict`, and everything above are dropped here, before the next cell's
        // fixtures are generated (peak memory bounded to one cell's working set at a time).
    }

    fn bench_writer_real_cell(c: &mut Criterion, cell: &CellSpec, cell_index: usize) {
        let dict_seed = TOP_SEED.wrapping_add(cell_index as u64);
        let dict = kernel::generate_dict(cell.k, dict_seed);
        assert_eq!(dict.len(), 1usize << cell.k, "fixture invariant: dict.len() == 1<<k");

        let cell_base_seed = kernel::derive_seed(TOP_SEED, cell_index as u64);
        let pages: Vec<WriterRealPage> = (0..PAGES_PER_CELL)
            .map(|page_idx| {
                let seed = kernel::derive_seed(cell_base_seed, page_idx as u64);
                build_writer_real_page(cell.k, cell.survival, seed)
            })
            .collect();

        let max_selected = pages
            .iter()
            .map(|p| p.sel_words.iter().map(|w| w.count_ones() as usize).sum::<usize>())
            .max()
            .unwrap_or(0);

        // --- untimed cross-arm correctness check ---
        {
            let mut decoder_c = RleDecoder::new(cell.k as u8);
            let mut decoder_d = RleDecoder::new(cell.k as u8);
            let mut idx_buf = [0i32; RLE_CHUNK];
            let mut val_buf = [0i64; RLE_CHUNK];
            let mut out_a = Vec::with_capacity(max_selected);
            let mut out_scalar = Vec::with_capacity(max_selected);
            let mut out_b = Vec::with_capacity(max_selected);
            let mut out_c = Vec::with_capacity(max_selected);
            let mut out_d = Vec::with_capacity(max_selected);

            for (page_idx, page) in pages.iter().enumerate() {
                out_a.clear();
                run_arm_a_writer_real(page, &dict, &mut out_a);
                out_scalar.clear();
                for run in &page.runs {
                    kernel::select_run_scalar(&page.buffer, run.byte_offset, WRITER_REAL_RUN_LEN, &run.sel_words, &dict, &mut out_scalar);
                }
                out_b.clear();
                run_arm_b_writer_real(page, &dict, &mut out_b);
                out_c.clear();
                run_arm_c(&mut decoder_c, &page.buffer, page.n_values, &page.sel_words, &dict, &mut idx_buf, &mut out_c);
                out_d.clear();
                run_arm_d(&mut decoder_d, &page.buffer, page.n_values, &page.sel_words, &dict, &mut val_buf, &mut out_d);

                verify_digests(
                    cell.k,
                    cell.shape.label(),
                    cell.denom,
                    page_idx,
                    kernel::fnv1a64(&out_a),
                    kernel::fnv1a64(&out_scalar),
                    kernel::fnv1a64(&out_b),
                    kernel::fnv1a64(&out_c),
                    kernel::fnv1a64(&out_d),
                );
            }
        }

        // --- timed loop ---
        let mut group = c.benchmark_group("paper4");
        group.sample_size(12);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs_f64(2.5));

        let k = cell.k;
        let shape = cell.shape.label();
        let denom = cell.denom;

        {
            let mut cursor = 0usize;
            let mut out = Vec::with_capacity(max_selected);
            group.bench_function(format!("paper_pext/k{k}/{shape}/s{denom}"), |b| {
                b.iter(|| {
                    let page = &pages[cursor];
                    cursor = (cursor + 1) % pages.len();
                    out.clear();
                    run_arm_a_writer_real(page, &dict, &mut out);
                    hint::black_box(out.as_slice());
                });
            });
        }
        {
            let mut cursor = 0usize;
            let mut out = Vec::with_capacity(max_selected);
            group.bench_function(format!("sparse_direct/k{k}/{shape}/s{denom}"), |b| {
                b.iter(|| {
                    let page = &pages[cursor];
                    cursor = (cursor + 1) % pages.len();
                    out.clear();
                    run_arm_b_writer_real(page, &dict, &mut out);
                    hint::black_box(out.as_slice());
                });
            });
        }
        {
            let mut cursor = 0usize;
            let mut out = Vec::with_capacity(max_selected);
            let mut decoder = RleDecoder::new(k as u8);
            let mut idx_buf = [0i32; RLE_CHUNK];
            // Prime any lazily-allocated decoder scratch state (untimed); see the comment in
            // bench_cell's arm C block.
            run_arm_c(&mut decoder, &pages[0].buffer, pages[0].n_values, &pages[0].sel_words, &dict, &mut idx_buf, &mut out);
            group.bench_function(format!("decode_all_indices_compact/k{k}/{shape}/s{denom}"), |b| {
                b.iter(|| {
                    let page = &pages[cursor];
                    cursor = (cursor + 1) % pages.len();
                    out.clear();
                    run_arm_c(&mut decoder, &page.buffer, page.n_values, &page.sel_words, &dict, &mut idx_buf, &mut out);
                    hint::black_box(out.as_slice());
                });
            });
        }
        {
            let mut cursor = 0usize;
            let mut out = Vec::with_capacity(max_selected);
            let mut decoder = RleDecoder::new(k as u8);
            let mut val_buf = [0i64; RLE_CHUNK];
            // Prime the decoder's lazily-allocated 1024-i64 scratch buffer (untimed); see the
            // comment in bench_cell's arm C block.
            run_arm_d(&mut decoder, &pages[0].buffer, pages[0].n_values, &pages[0].sel_words, &dict, &mut val_buf, &mut out);
            group.bench_function(format!("materialize_then_filter/k{k}/{shape}/s{denom}"), |b| {
                b.iter(|| {
                    let page = &pages[cursor];
                    cursor = (cursor + 1) % pages.len();
                    out.clear();
                    run_arm_d(&mut decoder, &page.buffer, page.n_values, &page.sel_words, &dict, &mut val_buf, &mut out);
                    hint::black_box(out.as_slice());
                });
            });
        }
        group.finish();
    }

    pub(super) fn run_full_matrix(c: &mut Criterion) {
        for (cell_index, cell) in build_cells().into_iter().enumerate() {
            if cell.shape == Shape::WriterReal {
                bench_writer_real_cell(c, &cell, cell_index);
            } else {
                bench_cell(c, &cell, cell_index);
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
