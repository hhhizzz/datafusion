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

use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, new_null_array};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::{Result, internal_err};
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_expr::{EmitTo, GroupsAccumulator};
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;

use crate::PhysicalExpr;
use crate::aggregates::group_values::{GroupByMetrics, GroupValues, new_group_values};
use crate::aggregates::order::GroupOrdering;
use crate::aggregates::row_hash::create_group_accumulator;
use crate::aggregates::{
    AggregateExec, PhysicalGroupBy, aggregate_expressions, evaluate_group_by,
};

/// Marker for raw rows -> partial state aggregation.
pub(in crate::aggregates) struct PartialMarker;
/// Marker for raw rows -> partial state conversion without aggregation.
pub(in crate::aggregates) struct PartialSkipMarker;
/// Marker for partial state -> final value aggregation.
pub(in crate::aggregates) struct FinalMarker;

/// Grouped hash table shared by the partial and final paths.
///
/// While building, it consumes input batches and updates group / accumulator
/// state. While outputting, it incrementally drains that state into output
/// batches.
///
/// # Logical and Physical Model
///
/// Logically, this is a hash table that maps { group keys -> accumulator states }
/// For example, `AVG(v) GROUP BY k` stores one entry per `k`, where each
/// entry owns the `sum(v)` and `count(v)` state needed to compute the final
/// average.
///
/// Physically, the group keys and accumulators are backed by [`GroupValues`] and
/// [`GroupsAccumulator`]. Both use columnar storage so aggregation can stay
/// vectorized.
///
/// # Marker Type
/// `AggrMode` selects the aggregate semantics.
///
/// e.g. `AggregateHashTable::<PartialMarker>::new(...)` creates an aggregate hash table
/// for the partial hash aggregate stage, the input schema is raw rows and output
/// schema is intermediate states.
///
/// It is a zero-sized compile-time marker, so each stage keeps its update logic
/// in a separate impl block, to make the behavior difference explicit.
pub(in crate::aggregates) struct AggregateHashTable<AggrMode> {
    /// Grouping and accumulator-specific timing metrics.
    pub(super) group_by_metrics: GroupByMetrics,

    /// Raw input schema, used to evaluate expressions and synthesize empty
    /// grouping-set rows.
    pub(super) input_schema: SchemaRef,

    /// Output schema: group columns followed by aggregate state or final values.
    pub(super) output_schema: SchemaRef,

    /// Maximum rows per emitted output batch, from config `batch_size`.
    pub(super) batch_size: usize,

    /// Lifecycle-specific state: building stage / outputting stage.
    pub(super) state: AggregateHashTableState,

    pub(super) _mode: PhantomData<AggrMode>,
}

/// Methods shared by all aggregate hash table modes.
impl<AggrMode> AggregateHashTable<AggrMode> {
    pub(super) fn new_with_filters(
        agg: &AggregateExec,
        partition: usize,
        output_schema: SchemaRef,
        batch_size: usize,
        filters: Vec<Option<Arc<dyn PhysicalExpr>>>,
    ) -> Result<Self> {
        if batch_size == 0 {
            return internal_err!("AggregateHashTable requires config batch_size >= 1");
        }

        let input_schema = agg.input().schema();
        let aggregate_arguments = aggregate_expressions(
            &agg.aggr_expr,
            &agg.mode,
            agg.group_by.num_group_exprs(),
        )?;
        let accumulators: Vec<_> = agg
            .aggr_expr
            .iter()
            .zip(aggregate_arguments)
            .zip(filters)
            .map(|((agg_expr, arguments), filter)| {
                let accumulator = create_group_accumulator(agg_expr)?;
                Ok(HashAggregateAccumulator::new(
                    Arc::clone(agg_expr),
                    arguments,
                    filter,
                    accumulator,
                ))
            })
            .collect::<Result<_>>()?;

        let group_schema = agg.group_by.group_schema(&input_schema)?;
        let group_values = new_group_values(group_schema, &GroupOrdering::None)?;

        Ok(Self {
            group_by_metrics: GroupByMetrics::new(&agg.metrics, partition),
            input_schema,
            output_schema,
            batch_size,
            state: AggregateHashTableState::Building(AggregateHashTableBuffer {
                group_by: Arc::clone(&agg.group_by),
                group_values,
                batch_group_indices: Default::default(),
                accumulators,
            }),
            _mode: PhantomData,
        })
    }

    /// See comments in [`EvaluatedAggregateBatch`]
    pub(super) fn evaluate_batch(
        &self,
        batch: &RecordBatch,
    ) -> Result<EvaluatedAggregateBatch> {
        let state = self.state.building();
        let timer = self.group_by_metrics.time_calculating_group_ids.timer();
        // outer vec: one per each grouping set
        // inner vec: all group by exprs for the current grouping set
        let grouping_set_args = evaluate_group_by(&state.group_by, batch)?;
        drop(timer);

        let timer = self.group_by_metrics.aggregate_arguments_time.timer();
        // The evaluated args for each accumulator
        let accumulator_args = self
            .state
            .building()
            .accumulators
            .iter()
            .map(|acc| acc.evaluate_acc_args(batch))
            .collect::<Result<Vec<_>>>()?;
        drop(timer);

        Ok(EvaluatedAggregateBatch {
            grouping_set_args,
            accumulator_args,
        })
    }

    pub(in crate::aggregates) fn memory_size(&self) -> usize {
        match &self.state {
            AggregateHashTableState::Building(state)
            | AggregateHashTableState::Outputting(state) => {
                let acc = state
                    .accumulators
                    .iter()
                    .map(|acc| acc.accumulator.size())
                    .sum::<usize>();

                acc + state.group_values.size()
                    + state.batch_group_indices.allocated_size()
            }
            AggregateHashTableState::OutputtingMaterialized(output) => {
                output.memory_size()
            }
            AggregateHashTableState::Done => 0,
        }
    }

    /// Returns the number of distinct groups accumulated so far.
    pub(in crate::aggregates) fn building_group_count(&self) -> usize {
        self.state.building().group_values.len()
    }

    pub(in crate::aggregates) fn is_building(&self) -> bool {
        matches!(self.state, AggregateHashTableState::Building(_))
    }

    pub(in crate::aggregates) fn is_done(&self) -> bool {
        matches!(self.state, AggregateHashTableState::Done)
    }

    pub(super) fn start_outputting(&mut self) {
        let AggregateHashTableState::Building(mut state) =
            std::mem::replace(&mut self.state, AggregateHashTableState::Done)
        else {
            unreachable!("hash aggregate table is not building")
        };

        state.batch_group_indices = Vec::new();
        self.state = AggregateHashTableState::Outputting(state);
    }
}

pub(super) fn emit_to_for_batch_size(batch_size: usize, group_count: usize) -> EmitTo {
    debug_assert!(batch_size > 0);
    if group_count <= batch_size {
        EmitTo::All
    } else {
        EmitTo::FirstBlock(batch_size)
    }
}

/// State and argument information for a single Aggregate
///
/// For example, for `SELECT COUNT(x), SUM(y WHERE z > 10) ...`  there would be two
/// `HashAggregateAccumulator`, one each for `COUNT(x)` and `SUM(y WHERE z > 10)`
pub(super) struct HashAggregateAccumulator {
    /// Aggregate expression used to create a fresh accumulator for related
    /// hash tables, such as the partial-skip table.
    aggregate_expr: Arc<AggregateFunctionExpr>,

    /// Arguments to pass to this accumulator.
    ///
    /// Example: `CORR(x, y)` stores two expressions here, while `SUM(x)` stores one.
    arguments: Vec<Arc<dyn PhysicalExpr>>,

    /// Optional `FILTER` expression for this accumulator.
    ///
    /// Example: `SUM(x) FILTER (WHERE x > 10)` stores the `x > 10` predicate.
    filter: Option<Arc<dyn PhysicalExpr>>,

    /// Accumulator state for all groups for one aggregate expression.
    accumulator: Box<dyn GroupsAccumulator>,
}

/// Evaluated aggregate arguments and filter for one input batch.
///
/// For example, `AVG(x + 1) FILTER (WHERE x > 0)` evaluates both `x + 1`
/// and `x > 0`.
///
/// These arrays can be passed directly to [`GroupsAccumulator`].
pub(super) struct EvaluatedAccumulatorArgs {
    /// Evaluated argument arrays. Some aggregate functions take multiple arguments.
    pub(super) arguments: Vec<ArrayRef>,
    /// Evaluated filter array, `Some` if the aggregate has a `FILTER` expression.
    pub(super) filter: Option<ArrayRef>,
}

/// Evaluated all group by keys and accumulator args.
///
/// e.g., `select k+1, sum(v*v) from t group by (k+1)`, this function evaluates
/// `k+1`, `v*v`
pub(super) struct EvaluatedAggregateBatch {
    /// One entry per grouping set; each entry contains all evaluated group key
    /// arrays for the current input batch.
    pub(super) grouping_set_args: Vec<Vec<ArrayRef>>,

    /// Evaluated arguments and filters, one entry per aggregate expression.
    pub(super) accumulator_args: Vec<EvaluatedAccumulatorArgs>,
}

/// Buffer for the aggregate hash table's group keys and accumulator states.
///
/// It accumulates input during aggregation and emits final results during the
/// outputting stage.
///
/// [`GroupValues`] stores the physical group-key layout, while
/// [`GroupsAccumulator`] stores per-group aggregate state.
pub(super) struct AggregateHashTableBuffer {
    /// GROUP BY expressions evaluated for each input batch.
    pub(super) group_by: Arc<PhysicalGroupBy>,

    /// Interned group keys. Accumulator state is stored separately by group index.
    pub(super) group_values: Box<dyn GroupValues>,

    /// Group index for each row in the current input batch.
    ///
    /// Each value indexes into `group_values`, and the same index is used by every
    /// accumulator to update that group's aggregate state.
    pub(super) batch_group_indices: Vec<usize>,

    /// One item per aggregate expression.
    ///
    /// Example: `COUNT(x), SUM(y)` creates two items. Each item owns the input
    /// expressions, optional filter, and accumulator state for all groups.
    pub(super) accumulators: Vec<HashAggregateAccumulator>,
}

pub(super) enum AggregateHashTableState {
    /// Accumulating input rows into group keys and aggregate state.
    Building(AggregateHashTableBuffer),
    /// Emitting results directly from group keys and aggregate state.
    Outputting(AggregateHashTableBuffer),
    /// Materialize output results once, then incrementally emit slices.
    OutputtingMaterialized(MaterializedAggregateOutput),
    Done,
}

/// Fully evaluated aggregate output and the next row offset to emit.
pub(super) struct MaterializedAggregateOutput {
    batch: RecordBatch,
    offset: usize,
}

impl MaterializedAggregateOutput {
    pub(super) fn new(batch: RecordBatch) -> Self {
        Self { batch, offset: 0 }
    }

    pub(super) fn next_batch(&mut self, batch_size: usize) -> Option<RecordBatch> {
        debug_assert!(batch_size > 0);
        if self.is_exhausted() {
            return None;
        }

        let length = batch_size.min(self.batch.num_rows() - self.offset);
        let batch = self.batch.slice(self.offset, length);
        self.offset += length;
        Some(batch)
    }

    pub(super) fn is_exhausted(&self) -> bool {
        self.offset >= self.batch.num_rows()
    }

    pub(super) fn memory_size(&self) -> usize {
        self.batch.get_array_memory_size()
    }
}

impl HashAggregateAccumulator {
    fn new(
        aggregate_expr: Arc<AggregateFunctionExpr>,
        arguments: Vec<Arc<dyn PhysicalExpr>>,
        filter: Option<Arc<dyn PhysicalExpr>>,
        accumulator: Box<dyn GroupsAccumulator>,
    ) -> Self {
        Self {
            aggregate_expr,
            arguments,
            filter,
            accumulator,
        }
    }

    /// Construct a new accumulator with the same definition, but with empty internal
    /// state buffers (empty [`GroupsAccumulator`]).
    pub(super) fn empty_like(&self) -> Result<Self> {
        let accumulator = create_group_accumulator(&self.aggregate_expr)?;
        Ok(Self::new(
            Arc::clone(&self.aggregate_expr),
            self.arguments.clone(),
            self.filter.clone(),
            accumulator,
        ))
    }

    /// Evaluate aggregate arguments and filter for one input batch.
    ///
    /// For example, `AVG(x + 1) FILTER (WHERE x > 0)` evaluates both `x + 1`
    /// and `x > 0`.
    ///
    /// These arrays can be passed directly to [`GroupsAccumulator`] next.
    fn evaluate_acc_args(&self, batch: &RecordBatch) -> Result<EvaluatedAccumulatorArgs> {
        let arguments = self
            .arguments
            .iter()
            .map(|expr| {
                expr.evaluate(batch)
                    .and_then(|value| value.into_array(batch.num_rows()))
            })
            .collect::<Result<_>>()?;

        let filter = self
            .filter
            .as_ref()
            .map(|filter| {
                filter
                    .evaluate(batch)
                    .and_then(|value| value.into_array(batch.num_rows()))
            })
            .transpose()?;

        Ok(EvaluatedAccumulatorArgs { arguments, filter })
    }

    pub(super) fn update_batch(
        &mut self,
        values: &EvaluatedAccumulatorArgs,
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        let filter = values.filter.as_ref().map(|filter| filter.as_boolean());
        self.accumulator.update_batch(
            &values.arguments,
            group_indices,
            filter,
            total_num_groups,
        )
    }

    pub(super) fn merge_batch(
        &mut self,
        values: &EvaluatedAccumulatorArgs,
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        debug_assert!(values.filter.is_none());
        self.accumulator
            .merge_batch(&values.arguments, group_indices, total_num_groups)
    }

    /// Evaluating final aggregate results according to `EmitTo`, and reset inner
    /// states. (e.g. after `evaluate(EmitTo::All)`, it returns all accumulated groups
    /// , and clear the inner buffers)
    pub(super) fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        self.accumulator.evaluate(emit_to)
    }

    /// Evaluating partial aggregate results according to `EmitTo`, and reset inner
    /// states. (e.g. after `state(EmitTo::All)`, it returns all accumulated groups
    /// , and clear the inner buffers)
    pub(super) fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        self.accumulator.state(emit_to)
    }

    pub(super) fn supports_convert_to_state(&self) -> bool {
        self.accumulator.supports_convert_to_state()
    }

    pub(super) fn convert_to_state(
        &mut self,
        values: &EvaluatedAccumulatorArgs,
    ) -> Result<Vec<ArrayRef>> {
        let opt_filter = values.filter.as_ref().map(|filter| filter.as_boolean());
        self.accumulator
            .convert_to_state(&values.arguments, opt_filter)
    }

    pub(super) fn null_arguments(
        &self,
        input_schema: &SchemaRef,
    ) -> Result<Vec<ArrayRef>> {
        self.arguments
            .iter()
            .map(|expr| {
                let data_type = expr.data_type(input_schema)?;
                Ok(new_null_array(&data_type, 1))
            })
            .collect()
    }
}

impl AggregateHashTableState {
    pub(super) fn building(&self) -> &AggregateHashTableBuffer {
        let Self::Building(state) = self else {
            unreachable!("hash aggregate table is not building")
        };
        state
    }

    pub(super) fn building_mut(&mut self) -> &mut AggregateHashTableBuffer {
        let Self::Building(state) = self else {
            unreachable!("hash aggregate table is not building")
        };
        state
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use arrow::array::{Array, ArrayRef, BooleanArray, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::col;

    use crate::aggregates::PhysicalGroupBy;
    use crate::metrics::ExecutionPlanMetricsSet;

    use super::*;
    use datafusion_functions_aggregate::count::count_udaf;

    #[test]
    fn emit_to_for_batch_size_uses_first_block_for_partial_batches() {
        assert_eq!(emit_to_for_batch_size(4, 4), EmitTo::All);
        assert_eq!(emit_to_for_batch_size(4, 5), EmitTo::FirstBlock(4));
    }

    #[test]
    fn final_output_evaluates_in_output_sized_chunks() -> Result<()> {
        let (mut table, emit_calls) = final_output_test_table(2)?;

        let first = table.next_output_batch()?.unwrap();
        assert_eq!(int32_values(&first, 0), vec![10, 20]);
        assert_eq!(int32_values(&first, 1), vec![100, 200]);
        assert_eq!(recorded_emit_to(&emit_calls), vec![EmitTo::FirstBlock(2)]);

        let second = table.next_output_batch()?.unwrap();
        assert_eq!(int32_values(&second, 0), vec![30, 40]);
        assert_eq!(int32_values(&second, 1), vec![300, 400]);
        assert_eq!(
            recorded_emit_to(&emit_calls),
            vec![EmitTo::FirstBlock(2), EmitTo::FirstBlock(2)]
        );

        let third = table.next_output_batch()?.unwrap();
        assert_eq!(int32_values(&third, 0), vec![50]);
        assert_eq!(int32_values(&third, 1), vec![500]);
        assert_eq!(
            recorded_emit_to(&emit_calls),
            vec![EmitTo::FirstBlock(2), EmitTo::FirstBlock(2), EmitTo::All]
        );

        assert!(table.next_output_batch()?.is_none());
        assert!(table.is_done());

        Ok(())
    }

    fn final_output_test_table(
        batch_size: usize,
    ) -> Result<(AggregateHashTable<FinalMarker>, Arc<Mutex<Vec<EmitTo>>>)> {
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("group_col", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));
        let group_schema = Arc::new(Schema::new(vec![Field::new(
            "group_col",
            DataType::Int32,
            false,
        )]));
        let output_schema = Arc::new(Schema::new(vec![
            Field::new("group_col", DataType::Int32, false),
            Field::new("count", DataType::Int32, false),
        ]));

        let mut group_values = new_group_values(group_schema, &GroupOrdering::None)?;
        let group_array = Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50]));
        let mut batch_group_indices = Vec::new();
        group_values.intern(&[group_array], &mut batch_group_indices)?;

        let aggregate_expr = Arc::new(
            AggregateExprBuilder::new(count_udaf(), vec![col("value", &input_schema)?])
                .schema(Arc::clone(&input_schema))
                .alias("COUNT(value)")
                .build()?,
        );

        let emit_calls = Arc::new(Mutex::new(Vec::new()));
        let accumulator = Box::new(RecordingGroupsAccumulator {
            values: vec![100, 200, 300, 400, 500],
            emit_calls: Arc::clone(&emit_calls),
        });

        let table = AggregateHashTable {
            group_by_metrics: GroupByMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
            input_schema: Arc::clone(&input_schema),
            output_schema,
            batch_size,
            state: AggregateHashTableState::Outputting(AggregateHashTableBuffer {
                group_by: Arc::new(PhysicalGroupBy::default()),
                group_values,
                batch_group_indices,
                accumulators: vec![HashAggregateAccumulator::new(
                    aggregate_expr,
                    vec![],
                    None,
                    accumulator,
                )],
            }),
            _mode: PhantomData,
        };

        Ok((table, emit_calls))
    }

    struct RecordingGroupsAccumulator {
        values: Vec<i32>,
        emit_calls: Arc<Mutex<Vec<EmitTo>>>,
    }

    impl RecordingGroupsAccumulator {
        fn take_values(&mut self, emit_to: EmitTo) -> ArrayRef {
            let len = match emit_to {
                EmitTo::All => self.values.len(),
                EmitTo::First(n) | EmitTo::FirstBlock(n) => n.min(self.values.len()),
            };
            let values = match emit_to {
                EmitTo::All => std::mem::take(&mut self.values),
                EmitTo::First(_) | EmitTo::FirstBlock(_) => {
                    self.values.drain(0..len).collect()
                }
            };
            Arc::new(Int32Array::from(values))
        }
    }

    impl GroupsAccumulator for RecordingGroupsAccumulator {
        fn update_batch(
            &mut self,
            _values: &[ArrayRef],
            _group_indices: &[usize],
            _opt_filter: Option<&BooleanArray>,
            _total_num_groups: usize,
        ) -> Result<()> {
            Ok(())
        }

        fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
            self.emit_calls.lock().unwrap().push(emit_to);
            Ok(self.take_values(emit_to))
        }

        fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
            Ok(vec![self.take_values(emit_to)])
        }

        fn merge_batch(
            &mut self,
            _values: &[ArrayRef],
            _group_indices: &[usize],
            _total_num_groups: usize,
        ) -> Result<()> {
            Ok(())
        }

        fn size(&self) -> usize {
            self.values.len() * size_of::<i32>()
        }
    }

    fn recorded_emit_to(emit_calls: &Arc<Mutex<Vec<EmitTo>>>) -> Vec<EmitTo> {
        emit_calls.lock().unwrap().clone()
    }

    fn int32_values(batch: &RecordBatch, column: usize) -> Vec<i32> {
        let array = batch
            .column(column)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        (0..array.len()).map(|idx| array.value(idx)).collect()
    }
}
