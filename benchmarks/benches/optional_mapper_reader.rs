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

use std::hint;
use std::sync::Arc;

use arrow::array::{BooleanBufferBuilder, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::{
    ParquetRecordBatchReaderBuilder, RowSelection, RowSelectionPolicy,
};
use parquet::file::properties::WriterProperties;

const LOGICAL_ROWS: usize = 8192;
const BATCH_SIZE: usize = 8192;
const DICTIONARY_BIT_WIDTH: usize = 5;
const SELECT_PERCENTS: &[usize] = &[25, 90];

struct BenchInput {
    parquet_data: Bytes,
    selection: RowSelection,
    expected_rows: usize,
}

/// Production-shaped full-reader transfer benchmark for the optional mapper.
///
/// Every source-bound backend commit runs the same nullable, dictionary-encoded
/// Int64 payload through the public Mask reader. The timed loop includes reader
/// construction and complete batch consumption, but not Parquet fixture or
/// selection construction.
fn criterion_benchmark(c: &mut Criterion) {
    let parquet_data = build_nullable_dictionary_parquet();

    for &select_percent in SELECT_PERCENTS {
        let (selection, expected_rows) = build_percent_selection(select_percent);
        let input = BenchInput {
            parquet_data: parquet_data.clone(),
            selection,
            expected_rows,
        };

        // Fail before Criterion starts if a source-bound backend changes output
        // cardinality. Every timed invocation repeats the same guard below.
        assert_eq!(run_read(&input), input.expected_rows);

        let case = format!(
            "nullable-dict-i64-bw{DICTIONARY_BIT_WIDTH:02}-n04-sel{select_percent:03}"
        );
        c.bench_with_input(
            BenchmarkId::new("optional_mapper_reader", case),
            &input,
            |b, input| {
                b.iter(|| {
                    let total = run_read(input);
                    assert_eq!(total, input.expected_rows);
                    hint::black_box(total);
                });
            },
        );
    }
}

fn build_nullable_dictionary_parquet() -> Bytes {
    let dictionary_len = 1usize << DICTIONARY_BIT_WIDTH;
    let values = Int64Array::from_iter((0..LOGICAL_ROWS).map(|idx| {
        // One null every 25 logical rows yields 328 / 8192 = 4.0039% nulls.
        // The odd multiplier permutes the 32 dictionary entries and avoids
        // long equal-value RLE runs in the packed index stream.
        (idx % 25 != 0)
            .then_some(((idx.wrapping_mul(4051)) & (dictionary_len - 1)) as i64)
    }));
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        true,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
    let props = WriterProperties::builder()
        .set_dictionary_enabled(true)
        .set_dictionary_page_size_limit(8 * 1024 * 1024)
        .set_data_page_row_count_limit(LOGICAL_ROWS)
        .build();
    let mut writer = ArrowWriter::try_new(Vec::new(), schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    Bytes::from(writer.into_inner().unwrap())
}

fn build_percent_selection(select_percent: usize) -> (RowSelection, usize) {
    let mut mask = BooleanBufferBuilder::new(LOGICAL_ROWS);
    let mut selected_rows = 0;
    for idx in 0..LOGICAL_ROWS {
        // Deterministically spread selected rows while preserving an exact
        // percentage in each complete 100-row period.
        let selected = (idx.wrapping_mul(37) % 100) < select_percent;
        selected_rows += usize::from(selected);
        mask.append(selected);
    }
    (
        RowSelection::from_boolean_buffer(mask.finish()),
        selected_rows,
    )
}

fn run_read(input: &BenchInput) -> usize {
    let reader = ParquetRecordBatchReaderBuilder::try_new(input.parquet_data.clone())
        .unwrap()
        .with_batch_size(BATCH_SIZE)
        .with_row_selection(input.selection.clone())
        .with_row_selection_policy(RowSelectionPolicy::Mask)
        .build()
        .unwrap();

    reader.map(|batch| batch.unwrap().num_rows()).sum::<usize>()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
