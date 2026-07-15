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

//! [`EarlyStoppingStream`] terminates a Parquet file scan when a dynamic
//! filter narrows after the scan has already started.

use std::pin::Pin;
use std::task::{Context, Poll};

use arrow::array::RecordBatch;
use datafusion_common::Result;
use datafusion_physical_plan::metrics::PruningMetrics;
use datafusion_pruning::FilePruner;
use futures::{Stream, StreamExt, ready};

/// Wraps an inner RecordBatchStream and a [`FilePruner`]
///
/// This can terminate the scan early when some dynamic filters is updated after
/// the scan starts, so we discover after the scan starts that the file can be
/// pruned (can't have matching rows).
pub(super) struct EarlyStoppingStream<S> {
    /// Has the stream finished processing? All subsequent polls will return
    /// None
    done: bool,
    file_pruner: FilePruner,
    files_ranges_pruned_statistics: PruningMetrics,
    /// The inner stream
    inner: Option<S>,
}

impl<S> EarlyStoppingStream<S> {
    pub(super) fn new(
        stream: S,
        file_pruner: FilePruner,
        files_ranges_pruned_statistics: PruningMetrics,
    ) -> Self {
        Self {
            done: false,
            inner: Some(stream),
            file_pruner,
            files_ranges_pruned_statistics,
        }
    }
}

impl<S> EarlyStoppingStream<S>
where
    S: Stream<Item = Result<RecordBatch>> + Unpin,
{
    fn check_prune(&mut self, input: Result<RecordBatch>) -> Result<Option<RecordBatch>> {
        let batch = input?;

        // Since dynamic filters may have been updated, see if we can stop
        // reading this stream entirely.
        if self.file_pruner.should_prune()? {
            self.files_ranges_pruned_statistics.add_pruned(1);
            // Previously this file range has been counted as matched
            self.files_ranges_pruned_statistics.subtract_matched(1);
            self.finish();
            Ok(None)
        } else {
            // Return the adapted batch
            Ok(Some(batch))
        }
    }

    fn finish(&mut self) {
        self.done = true;
        self.inner = None;
    }
}

impl<S> Stream for EarlyStoppingStream<S>
where
    S: Stream<Item = Result<RecordBatch>> + Unpin,
{
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        let next = ready!(
            self.inner
                .as_mut()
                .expect("unfinished stream must retain its inner stream")
                .poll_next_unpin(cx)
        );
        match next {
            None => {
                // input done
                self.finish();
                Poll::Ready(None)
            }
            Some(input_batch) => {
                let output = self.check_prune(input_batch);
                Poll::Ready(output.transpose())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::stats::Precision;
    use datafusion_common::{ColumnStatistics, ScalarValue, Statistics};
    use datafusion_datasource::PartitionedFile;
    use datafusion_physical_expr::expressions::Literal;
    use datafusion_physical_plan::metrics::Count;
    use futures::StreamExt;

    use super::*;

    struct DropObservedStream {
        items: VecDeque<Result<RecordBatch>>,
        dropped: Arc<AtomicBool>,
    }

    impl DropObservedStream {
        fn new(
            items: impl IntoIterator<Item = Result<RecordBatch>>,
        ) -> (Self, Arc<AtomicBool>) {
            let dropped = Arc::new(AtomicBool::new(false));
            (
                Self {
                    items: items.into_iter().collect(),
                    dropped: Arc::clone(&dropped),
                },
                dropped,
            )
        }
    }

    impl Stream for DropObservedStream {
        type Item = Result<RecordBatch>;

        fn poll_next(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.items.pop_front())
        }
    }

    impl Drop for DropObservedStream {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    fn always_pruning_file_pruner(schema: &Arc<Schema>) -> FilePruner {
        let file = PartitionedFile::new("early-stop.parquet", 0).with_statistics(
            Arc::new(Statistics {
                num_rows: Precision::Exact(1),
                total_byte_size: Precision::Absent,
                column_statistics: vec![ColumnStatistics {
                    null_count: Precision::Exact(0),
                    max_value: Precision::Exact(ScalarValue::Int32(Some(1))),
                    min_value: Precision::Exact(ScalarValue::Int32(Some(1))),
                    sum_value: Precision::Absent,
                    distinct_count: Precision::Exact(1),
                    byte_size: Precision::Absent,
                }],
            }),
        );
        let predicate = Arc::new(Literal::new(ScalarValue::Boolean(Some(false))));
        FilePruner::try_new(predicate, schema, &file, Count::new()).unwrap()
    }

    fn test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]))
    }

    fn pruning_metrics() -> PruningMetrics {
        let metrics = PruningMetrics::new();
        metrics.add_matched(1);
        metrics
    }

    #[tokio::test]
    async fn prune_transition_drops_inner_before_returning_eof() {
        let schema = test_schema();
        let batch = RecordBatch::new_empty(Arc::clone(&schema));
        let (inner, dropped) = DropObservedStream::new([Ok(batch)]);
        let mut stream = EarlyStoppingStream::new(
            inner,
            always_pruning_file_pruner(&schema),
            pruning_metrics(),
        );

        assert!(stream.next().await.is_none());
        assert!(
            dropped.load(Ordering::SeqCst),
            "inner must be dropped before prune EOF is returned"
        );
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn ordinary_eof_drops_inner_and_remains_fused() {
        let schema = test_schema();
        let (inner, dropped) = DropObservedStream::new([]);
        let mut stream = EarlyStoppingStream::new(
            inner,
            always_pruning_file_pruner(&schema),
            pruning_metrics(),
        );

        assert!(stream.next().await.is_none());
        assert!(
            dropped.load(Ordering::SeqCst),
            "inner must be dropped before ordinary EOF is returned"
        );
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn item_error_preserves_inner_for_later_poll() {
        let schema = test_schema();
        let batch = RecordBatch::new_empty(Arc::clone(&schema));
        let error = datafusion_common::DataFusionError::Execution(
            "scripted inner error".to_string(),
        );
        let (inner, dropped) = DropObservedStream::new([Err(error), Ok(batch)]);
        let mut stream = EarlyStoppingStream::new(
            inner,
            always_pruning_file_pruner(&schema),
            pruning_metrics(),
        );

        let error = stream.next().await.unwrap().unwrap_err();
        assert!(error.to_string().contains("scripted inner error"));
        assert!(!dropped.load(Ordering::SeqCst));

        assert!(stream.next().await.is_none());
        assert!(dropped.load(Ordering::SeqCst));
        assert!(stream.next().await.is_none());
    }
}
