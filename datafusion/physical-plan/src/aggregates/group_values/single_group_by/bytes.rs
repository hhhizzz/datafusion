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

use crate::aggregates::group_values::GroupValues;

use arrow::array::{Array, ArrayRef, OffsetSizeTrait};
use datafusion_common::Result;
use datafusion_expr::EmitTo;
use datafusion_physical_expr_common::binary_map::{ArrowBytesMap, OutputType};

/// A [`GroupValues`] storing single column of Utf8/LargeUtf8/Binary/LargeBinary values
///
/// This specialization is significantly faster than using the more general
/// purpose `Row`s format
pub struct GroupValuesBytes<O: OffsetSizeTrait> {
    /// Map string/binary values to group index
    map: ArrowBytesMap<O, usize>,
    /// The total number of groups so far (used to assign group_index)
    num_groups: usize,
    /// Materialized group values retained while emitting `FirstBlock`s.
    first_block_values: Option<ArrayRef>,
    /// Number of leading groups already emitted by `EmitTo::FirstBlock`.
    first_block_emit_offset: usize,
}

impl<O: OffsetSizeTrait> GroupValuesBytes<O> {
    pub fn new(output_type: OutputType) -> Self {
        Self {
            map: ArrowBytesMap::new(output_type),
            num_groups: 0,
            first_block_values: None,
            first_block_emit_offset: 0,
        }
    }

    fn compact_first_block_state(&mut self) -> Result<()> {
        let Some(values) = self.first_block_values.take() else {
            return Ok(());
        };

        let offset = self.first_block_emit_offset;
        let remaining_len = values.len() - offset;
        self.map.take();
        self.num_groups = 0;
        self.first_block_emit_offset = 0;

        if remaining_len > 0 {
            let remaining_values = values.slice(offset, remaining_len);
            let mut group_indexes = vec![];
            self.intern(&[remaining_values], &mut group_indexes)?;

            // Verify that the group indexes were assigned in the correct order
            assert_eq!(0, group_indexes[0]);
        }

        Ok(())
    }

    fn first_block_values_size(&self) -> usize {
        self.first_block_values
            .as_ref()
            .map(|values| values.get_array_memory_size())
            .unwrap_or_default()
    }
}

impl<O: OffsetSizeTrait> GroupValues for GroupValuesBytes<O> {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), 1);

        self.compact_first_block_state()?;

        // look up / add entries in the table
        let arr = &cols[0];

        groups.clear();
        self.map.insert_if_new(
            arr,
            // called for each new group
            |_value| {
                // assign new group index on each insert
                let group_idx = self.num_groups;
                self.num_groups += 1;
                group_idx
            },
            // called for each group
            |group_idx| {
                groups.push(group_idx);
            },
        );

        // ensure we assigned a group to for each row
        assert_eq!(groups.len(), arr.len());
        Ok(())
    }

    fn size(&self) -> usize {
        self.map.size() + self.first_block_values_size() + size_of::<Self>()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.num_groups - self.first_block_emit_offset
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let group_values = match emit_to {
            EmitTo::All => {
                if let Some(values) = self.first_block_values.take() {
                    let values = values.slice(self.first_block_emit_offset, self.len());
                    self.num_groups = 0;
                    self.first_block_emit_offset = 0;
                    values
                } else {
                    // Reset the map to default, and convert it into a single array
                    let map_contents = self.map.take().into_state();
                    self.num_groups -= map_contents.len();
                    map_contents
                }
            }
            EmitTo::First(n) if n == self.len() => {
                self.compact_first_block_state()?;
                let map_contents = self.map.take().into_state();
                self.num_groups -= map_contents.len();
                map_contents
            }
            EmitTo::First(n) => {
                self.compact_first_block_state()?;
                // Reset the map to default, and convert it into a single array
                let map_contents = self.map.take().into_state();
                // if we only wanted to take the first n, insert the rest back
                // into the map we could potentially avoid this reallocation, at
                // the expense of much more complex code.
                // see https://github.com/apache/datafusion/issues/9195
                let emit_group_values = map_contents.slice(0, n);
                let remaining_group_values =
                    map_contents.slice(n, map_contents.len() - n);

                self.num_groups = 0;
                let mut group_indexes = vec![];
                self.intern(&[remaining_group_values], &mut group_indexes)?;

                // Verify that the group indexes were assigned in the correct order
                assert_eq!(0, group_indexes[0]);

                emit_group_values
            }
            EmitTo::FirstBlock(n) => {
                if self.first_block_values.is_none() {
                    let map_contents = self.map.take().into_state();
                    self.first_block_values = Some(map_contents);
                }

                let values = self.first_block_values.as_ref().unwrap();
                let emit_group_values = values.slice(self.first_block_emit_offset, n);
                self.first_block_emit_offset += n;

                if self.first_block_emit_offset == self.num_groups {
                    self.first_block_values = None;
                    self.first_block_emit_offset = 0;
                    self.num_groups = 0;
                }

                emit_group_values
            }
        };

        Ok(vec![group_values])
    }

    fn clear_shrink(&mut self, _num_rows: usize) {
        // in theory we could potentially avoid this reallocation and clear the
        // contents of the maps, but for now we just reset the map from the beginning
        self.map.take();
        self.num_groups = 0;
        self.first_block_values = None;
        self.first_block_emit_offset = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, StringArray};
    use std::sync::Arc;

    #[test]
    fn bytes_group_values_first_block_does_not_rebuild_remaining_state() -> Result<()> {
        let mut group_values = GroupValuesBytes::<i32>::new(OutputType::Utf8);
        let values: ArrayRef =
            Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e", "f"]));
        group_values.intern(&[values], &mut vec![])?;

        let emitted = group_values.emit(EmitTo::FirstBlock(3))?;
        assert_eq!(emitted[0].as_string::<i32>().value(0), "a");
        assert_eq!(emitted[0].as_string::<i32>().value(2), "c");
        assert_eq!(group_values.len(), 3);
        assert_eq!(
            group_values.num_groups, 6,
            "FirstBlock should advance a logical cursor without rebuilding bytes state"
        );

        let remaining = group_values.emit(EmitTo::FirstBlock(3))?;
        assert_eq!(remaining[0].as_string::<i32>().value(0), "d");
        assert_eq!(remaining[0].as_string::<i32>().value(2), "f");
        assert!(group_values.is_empty());

        Ok(())
    }

    #[test]
    fn bytes_group_values_compacts_before_intern_after_first_block() -> Result<()> {
        let mut group_values = GroupValuesBytes::<i32>::new(OutputType::Utf8);
        let values: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c", "d"]));
        group_values.intern(&[values], &mut vec![])?;

        let emitted = group_values.emit(EmitTo::FirstBlock(2))?;
        assert_eq!(emitted[0].as_string::<i32>().value(0), "a");
        assert_eq!(emitted[0].as_string::<i32>().value(1), "b");

        let values: ArrayRef = Arc::new(StringArray::from(vec!["c", "e"]));
        let mut groups = vec![];
        group_values.intern(&[values], &mut groups)?;
        assert_eq!(groups, vec![0, 2]);

        let emitted = group_values.emit(EmitTo::All)?;
        let values = emitted[0].as_string::<i32>();
        assert_eq!(values.len(), 3);
        assert_eq!(values.value(0), "c");
        assert_eq!(values.value(1), "d");
        assert_eq!(values.value(2), "e");
        assert!(group_values.is_empty());

        Ok(())
    }
}
