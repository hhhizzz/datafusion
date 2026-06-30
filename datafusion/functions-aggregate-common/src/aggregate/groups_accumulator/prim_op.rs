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

use std::mem::size_of;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, PrimitiveArray};
use arrow::buffer::NullBuffer;
use arrow::compute;
use arrow::datatypes::ArrowPrimitiveType;
use arrow::datatypes::DataType;
use datafusion_common::{DataFusionError, Result, internal_datafusion_err};
use datafusion_expr_common::groups_accumulator::{EmitTo, GroupsAccumulator};

use super::accumulate::NullState;

/// An accumulator that implements a single operation over
/// [`ArrowPrimitiveType`] where the accumulated state is the same as
/// the input type (such as `Sum`)
///
/// F: The function to apply to two elements. The first argument is
/// the existing value and should be updated with the second value
/// (e.g. [`BitAndAssign`] style).
///
/// [`BitAndAssign`]: std::ops::BitAndAssign
#[derive(Debug)]
pub struct PrimitiveGroupsAccumulator<T, F>
where
    T: ArrowPrimitiveType + Send,
    F: Fn(&mut T::Native, T::Native) + Send + Sync + 'static,
{
    /// values per group, stored as the native type
    values: Vec<T::Native>,

    /// The output type (needed for Decimal precision and scale)
    data_type: DataType,

    /// The starting value for new groups
    starting_value: T::Native,

    /// Track nulls in the input / filters
    null_state: NullState,

    /// Materialized values retained while emitting `FirstBlock`s.
    first_block_values: Option<PrimitiveArray<T>>,

    /// Function that computes the primitive result
    prim_fn: F,

    /// Number of leading groups already emitted by `EmitTo::FirstBlock`.
    first_block_emit_offset: usize,
}

impl<T, F> PrimitiveGroupsAccumulator<T, F>
where
    T: ArrowPrimitiveType + Send,
    F: Fn(&mut T::Native, T::Native) + Send + Sync + 'static,
{
    pub fn new(data_type: &DataType, prim_fn: F) -> Self {
        Self {
            values: vec![],
            data_type: data_type.clone(),
            null_state: NullState::new(),
            first_block_values: None,
            starting_value: T::default_value(),
            prim_fn,
            first_block_emit_offset: 0,
        }
    }

    /// Set the starting values for new groups
    pub fn with_starting_value(mut self, starting_value: T::Native) -> Self {
        self.starting_value = starting_value;
        self
    }

    fn compact_first_block_state(&mut self) {
        if let Some(values) = self.first_block_values.take() {
            let offset = self.first_block_emit_offset;
            let remaining_len = values.len() - offset;
            self.values.clear();
            if remaining_len > 0 {
                let remaining = values.slice(offset, remaining_len);
                self.values.extend_from_slice(remaining.values());
                self.null_state
                    .reset_from_nulls(remaining.len(), remaining.nulls());
            } else {
                self.null_state.reset_from_nulls(0, None);
            }
            self.first_block_emit_offset = 0;
            return;
        }

        let n = self.first_block_emit_offset;
        if n == 0 {
            return;
        }

        let _ = EmitTo::First(n).take_needed(&mut self.values);
        self.null_state.compact_first_block_state();
        self.first_block_emit_offset = 0;
    }

    fn materialize_first_block_values(&mut self) -> &PrimitiveArray<T> {
        if self.first_block_values.is_none() {
            let nulls = self.null_state.build(EmitTo::All);
            self.first_block_values = Some(
                PrimitiveArray::<T>::new(std::mem::take(&mut self.values).into(), nulls)
                    .with_data_type(self.data_type.clone()),
            );
        }
        self.first_block_values.as_ref().unwrap()
    }

    fn take_array(&mut self, emit_to: EmitTo) -> PrimitiveArray<T> {
        match emit_to {
            EmitTo::All => match self.first_block_values.take() {
                Some(values) => {
                    let offset = self.first_block_emit_offset;
                    let output = values.slice(offset, values.len() - offset);
                    self.first_block_emit_offset = 0;
                    output
                }
                None => {
                    let nulls = self.null_state.build(EmitTo::All);
                    PrimitiveArray::<T>::new(
                        std::mem::take(&mut self.values).into(),
                        nulls,
                    )
                    .with_data_type(self.data_type.clone())
                }
            },
            EmitTo::First(n) => {
                self.compact_first_block_state();
                let nulls = self.null_state.build(EmitTo::First(n));
                PrimitiveArray::<T>::new(
                    EmitTo::First(n).take_needed(&mut self.values).into(),
                    nulls,
                )
                .with_data_type(self.data_type.clone())
            }
            EmitTo::FirstBlock(n) => {
                let start = self.first_block_emit_offset;
                let (output, end, values_len) = {
                    let values = self.materialize_first_block_values();
                    let end = start + n;
                    (values.slice(start, n), end, values.len())
                };

                if end == values_len {
                    self.first_block_values = None;
                    self.first_block_emit_offset = 0;
                } else {
                    self.first_block_emit_offset = end;
                }

                output
            }
        }
    }
}

impl<T, F> GroupsAccumulator for PrimitiveGroupsAccumulator<T, F>
where
    T: ArrowPrimitiveType + Send,
    F: Fn(&mut T::Native, T::Native) + Send + Sync + 'static,
{
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        assert_eq!(values.len(), 1, "single argument to update_batch");
        let values = values[0].as_primitive::<T>();

        self.compact_first_block_state();

        // update values
        self.values.resize(total_num_groups, self.starting_value);

        // NullState dispatches / handles tracking nulls and groups that saw no values
        self.null_state.accumulate(
            group_indices,
            values,
            opt_filter,
            total_num_groups,
            |group_index, new_value| {
                // SAFETY: group_index is guaranteed to be in bounds
                let value = unsafe { self.values.get_unchecked_mut(group_index) };
                (self.prim_fn)(value, new_value);
            },
        );

        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        Ok(Arc::new(self.take_array(emit_to)))
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        self.evaluate(emit_to).map(|arr| vec![arr])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        // update / merge are the same
        self.update_batch(values, group_indices, None, total_num_groups)
    }

    /// Converts an input batch directly to a state batch
    ///
    /// The state is:
    /// - self.prim_fn for all non null, non filtered values
    /// - null otherwise
    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let values = values[0].as_primitive::<T>().clone();

        // Initializing state with starting values
        let initial_state =
            PrimitiveArray::<T>::from_value(self.starting_value, values.len());

        // Recalculating values in case there is filter
        let values = match opt_filter {
            None => values,
            Some(filter) => {
                let (filter_values, filter_nulls) = filter.clone().into_parts();
                // Calculating filter mask as a result of bitand of filter, and converting it to null buffer
                let filter_bool = match filter_nulls {
                    Some(filter_nulls) => filter_nulls.inner() & &filter_values,
                    None => filter_values,
                };
                let filter_nulls = NullBuffer::from(filter_bool);

                // Rebuilding input values with a new nulls mask, which is equal to
                // the union of original nulls and filter mask
                let (dt, values_buf, original_nulls) = values.into_parts();
                let nulls_buf =
                    NullBuffer::union(original_nulls.as_ref(), Some(&filter_nulls));
                PrimitiveArray::<T>::new(values_buf, nulls_buf).with_data_type(dt)
            }
        };

        let state_values = compute::binary_mut(initial_state, &values, |mut x, y| {
            (self.prim_fn)(&mut x, y);
            x
        });
        let state_values = state_values
            .map_err(|_| {
                internal_datafusion_err!(
                    "initial_values underlying buffer must not be shared"
                )
            })?
            .map_err(DataFusionError::from)?
            .with_data_type(self.data_type.clone());

        Ok(vec![Arc::new(state_values)])
    }

    fn supports_convert_to_state(&self) -> bool {
        true
    }

    fn size(&self) -> usize {
        let first_block_values_size = self
            .first_block_values
            .as_ref()
            .map(|values| values.get_array_memory_size())
            .unwrap_or_default();
        self.values.capacity() * size_of::<T::Native>()
            + self.null_state.size()
            + first_block_values_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::Int64Type;

    #[test]
    fn primitive_groups_emit_first_block_materializes_values_once() -> Result<()> {
        let mut accumulator = PrimitiveGroupsAccumulator::<Int64Type, _>::new(
            &DataType::Int64,
            |current, new| *current += new,
        );
        let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6]));
        accumulator.update_batch(&[values], &[0, 1, 2, 3, 4, 5], None, 6)?;

        let emitted = accumulator.evaluate(EmitTo::FirstBlock(3))?;
        let emitted = emitted.as_primitive::<Int64Type>();
        assert_eq!(emitted.values(), &[1, 2, 3]);
        assert_eq!(
            accumulator.values.len(),
            0,
            "FirstBlock should materialize primitive state once instead of keeping the mutable values vec for repeated chunk copies"
        );

        let remaining = accumulator.evaluate(EmitTo::FirstBlock(3))?;
        let remaining = remaining.as_primitive::<Int64Type>();
        assert_eq!(remaining.values(), &[4, 5, 6]);
        assert!(accumulator.values.is_empty());

        Ok(())
    }

    #[test]
    fn primitive_groups_first_block_compacts_before_update() -> Result<()> {
        let mut accumulator = PrimitiveGroupsAccumulator::<Int64Type, _>::new(
            &DataType::Int64,
            |current, new| *current += new,
        );
        let values: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            None,
            Some(6),
        ]));
        accumulator.update_batch(&[values], &[0, 1, 2, 3, 4, 5], None, 6)?;

        let emitted = accumulator.evaluate(EmitTo::FirstBlock(3))?;
        let emitted = emitted.as_primitive::<Int64Type>();
        assert_eq!(emitted.values(), &[1, 2, 3]);

        let values: ArrayRef = Arc::new(Int64Array::from(vec![10, 5, 20]));
        accumulator.update_batch(&[values], &[0, 1, 3], None, 4)?;

        let emitted = accumulator.evaluate(EmitTo::All)?;
        assert_eq!(
            emitted
                .as_primitive::<Int64Type>()
                .iter()
                .collect::<Vec<_>>(),
            vec![Some(14), Some(5), Some(6), Some(20)]
        );

        Ok(())
    }
}
