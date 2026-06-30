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

use crate::aggregates::group_values::GroupValues;
use arrow::array::types::{IntervalDayTime, IntervalMonthDayNano};
use arrow::array::{
    ArrayRef, ArrowNativeTypeOp, ArrowPrimitiveType, NullBufferBuilder, PrimitiveArray,
    cast::AsArray,
};
use arrow::datatypes::{DataType, i256};
use datafusion_common::Result;
use datafusion_common::hash_utils::RandomState;
use datafusion_common::utils::split_vec_min_alloc;
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_expr::EmitTo;
use half::f16;
use hashbrown::hash_table::HashTable;
#[cfg(not(feature = "force_hash_collisions"))]
use std::hash::BuildHasher;
use std::mem::size_of;
use std::sync::Arc;

/// A trait to allow hashing of floating point numbers
pub trait HashValue {
    fn hash(&self, state: &RandomState) -> u64;

    /// Return a canonical representative whose bit pattern is identical for
    /// all values that should be grouped together. Default is the identity;
    /// floats override this to fold `-0.0` into `+0.0` so the bit-equal
    /// `is_eq` check used during insertion treats them as the same group.
    /// NaN payload bits are preserved.
    #[inline]
    fn canonicalize(self) -> Self
    where
        Self: Sized,
    {
        self
    }
}

macro_rules! hash_integer {
    ($($t:ty),+) => {
        $(impl HashValue for $t {
            #[cfg(not(feature = "force_hash_collisions"))]
            fn hash(&self, state: &RandomState) -> u64 {
                state.hash_one(self)
            }

            #[cfg(feature = "force_hash_collisions")]
            fn hash(&self, _state: &RandomState) -> u64 {
                0
            }
        })+
    };
}
hash_integer!(i8, i16, i32, i64, i128, i256);
hash_integer!(u8, u16, u32, u64);
hash_integer!(IntervalDayTime, IntervalMonthDayNano);

macro_rules! hash_float {
    ($($t:ty),+) => {
        $(impl HashValue for $t {
            #[cfg(not(feature = "force_hash_collisions"))]
            fn hash(&self, state: &RandomState) -> u64 {
                state.hash_one(self.canonicalize().to_bits())
            }

            #[cfg(feature = "force_hash_collisions")]
            fn hash(&self, _state: &RandomState) -> u64 {
                0
            }

            #[inline]
            fn canonicalize(self) -> Self {
                let bits = self.to_bits();
                let bits = if bits << 1 == 0 { 0 } else { bits };
                Self::from_bits(bits)
            }
        })+
    };
}

hash_float!(f16, f32, f64);

/// A [`GroupValues`] storing a single column of primitive values
///
/// This specialization is significantly faster than using the more general
/// purpose `Row`s format
pub struct GroupValuesPrimitive<T: ArrowPrimitiveType> {
    /// The data type of the output array
    data_type: DataType,
    /// Stores the `(group_index, hash)` based on the hash of its value
    ///
    /// We also store `hash` is for reducing cost of rehashing. Such cost
    /// is obvious in high cardinality group by situation.
    /// More details can see:
    /// <https://github.com/apache/datafusion/issues/15961>
    map: HashTable<(usize, u64)>,
    /// The group index of the null value if any
    null_group: Option<usize>,
    /// The values for each group index
    values: Vec<T::Native>,
    /// Number of leading groups already emitted by `EmitTo::FirstBlock`.
    first_block_emit_offset: usize,
    /// The random state used to generate hashes
    random_state: RandomState,
}

impl<T: ArrowPrimitiveType> GroupValuesPrimitive<T> {
    pub fn new(data_type: DataType) -> Self {
        assert!(PrimitiveArray::<T>::is_compatible(&data_type));
        Self {
            data_type,
            map: HashTable::with_capacity(128),
            values: Vec::with_capacity(128),
            null_group: None,
            first_block_emit_offset: 0,
            random_state: crate::aggregates::AGGREGATION_HASH_SEED,
        }
    }
}

impl<T: ArrowPrimitiveType> GroupValuesPrimitive<T>
where
    T::Native: HashValue,
{
    fn rebuild_map(&mut self) {
        self.map.clear();
        let state = &self.random_state;
        for (group_idx, key) in self.values.iter().copied().enumerate() {
            if self.null_group == Some(group_idx) {
                continue;
            }
            let hash = key.hash(state);
            self.map.insert_unique(hash, (group_idx, hash), |&(_, h)| h);
        }
    }

    fn compact_first_block_remainder(&mut self) {
        let offset = self.first_block_emit_offset;
        if offset == 0 {
            return;
        }

        if offset >= self.values.len() {
            self.values.clear();
            self.map.clear();
            self.null_group = None;
        } else {
            self.values.drain(0..offset);
            self.null_group = self
                .null_group
                .and_then(|group_idx| group_idx.checked_sub(offset));
            self.rebuild_map();
        }
        self.first_block_emit_offset = 0;
    }

    fn build_primitive(
        values: Vec<T::Native>,
        null_idx: Option<usize>,
    ) -> PrimitiveArray<T> {
        let nulls = null_idx.map(|null_idx| {
            let mut buffer = NullBufferBuilder::new(values.len());
            buffer.append_n_non_nulls(null_idx);
            buffer.append_null();
            buffer.append_n_non_nulls(values.len() - null_idx - 1);
            // NOTE: The inner builder must be constructed as there is at least one null
            buffer.finish().unwrap()
        });
        PrimitiveArray::<T>::new(values.into(), nulls)
    }

    fn take_first_block(&mut self, n: usize) -> PrimitiveArray<T> {
        self.map.clear();
        let start = self.first_block_emit_offset;
        let end = (start + n).min(self.values.len());
        let null_group = self
            .null_group
            .filter(|group_idx| (start..end).contains(group_idx))
            .map(|group_idx| group_idx - start);
        let array = Self::build_primitive(self.values[start..end].to_vec(), null_group);

        if end == self.values.len() {
            self.values.clear();
            self.null_group = None;
            self.first_block_emit_offset = 0;
        } else {
            self.first_block_emit_offset = end;
        }

        array
    }
}

impl<T: ArrowPrimitiveType> GroupValues for GroupValuesPrimitive<T>
where
    T::Native: HashValue,
{
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), 1);
        self.compact_first_block_remainder();
        groups.clear();

        for v in cols[0].as_primitive::<T>() {
            let group_id = match v {
                None => *self.null_group.get_or_insert_with(|| {
                    let group_id = self.values.len();
                    self.values.push(Default::default());
                    group_id
                }),
                Some(key) => {
                    // Fold equivalence-class duplicates (e.g. `-0.0` → `+0.0`)
                    // so the bit-equal `is_eq` matches and the stored value is
                    // the canonical representative.
                    let key = key.canonicalize();
                    let state = &self.random_state;
                    let hash = key.hash(state);
                    let insert = self.map.entry(
                        hash,
                        |&(g, h)| unsafe {
                            hash == h && self.values.get_unchecked(g).is_eq(key)
                        },
                        |&(_, h)| h,
                    );

                    match insert {
                        hashbrown::hash_table::Entry::Occupied(o) => o.get().0,
                        hashbrown::hash_table::Entry::Vacant(v) => {
                            let g = self.values.len();
                            v.insert((g, hash));
                            self.values.push(key);
                            g
                        }
                    }
                }
            };
            groups.push(group_id)
        }
        Ok(())
    }

    fn size(&self) -> usize {
        self.map.capacity() * size_of::<(usize, u64)>() + self.values.allocated_size()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.values.len() - self.first_block_emit_offset
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let array: PrimitiveArray<T> = match emit_to {
            EmitTo::All => {
                self.map.clear();
                let offset = self.first_block_emit_offset;
                if offset == 0 {
                    Self::build_primitive(
                        std::mem::take(&mut self.values),
                        self.null_group.take(),
                    )
                } else {
                    let null_group = self
                        .null_group
                        .and_then(|group_idx| group_idx.checked_sub(offset));
                    let array =
                        Self::build_primitive(self.values[offset..].to_vec(), null_group);
                    self.values.clear();
                    self.null_group = None;
                    self.first_block_emit_offset = 0;
                    array
                }
            }
            EmitTo::First(n) => {
                self.compact_first_block_remainder();
                self.map.retain(|entry| {
                    // Decrement group index by n
                    let group_idx = entry.0;
                    match group_idx.checked_sub(n) {
                        // Group index was >= n, shift value down
                        Some(sub) => {
                            entry.0 = sub;
                            true
                        }
                        // Group index was < n, so remove from table
                        None => false,
                    }
                });
                let null_group = match &mut self.null_group {
                    Some(v) if *v >= n => {
                        *v -= n;
                        None
                    }
                    Some(_) => self.null_group.take(),
                    None => None,
                };
                Self::build_primitive(
                    split_vec_min_alloc(&mut self.values, n),
                    null_group,
                )
            }
            EmitTo::FirstBlock(n) => self.take_first_block(n),
        };

        Ok(vec![Arc::new(array.with_data_type(self.data_type.clone()))])
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        self.values.clear();
        self.first_block_emit_offset = 0;
        self.null_group = None;
        self.values.shrink_to(num_rows);
        self.map.clear();
        self.map.shrink_to(num_rows, |_| 0); // hasher does not matter since the map is cleared
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::types::Int32Type;
    use arrow::array::{ArrayRef, Int32Array};
    use arrow::datatypes::DataType;
    use datafusion_expr::EmitTo;
    use std::sync::Arc;

    /// Mirror of the `EmitTo::take_needed` regression test, applied to the
    /// concrete `GroupValuesPrimitive` accumulator.
    ///
    /// When `n` is small, the old `split_off(n) + swap` pattern used inside
    /// `emit(EmitTo::First(n))` left `self.values` with a small fresh allocation
    /// and returned the emitted prefix carrying the original large backing.
    ///
    /// With `split_vec_min_alloc` and `n * 2 <= len`, the drain branch is taken:
    /// the emitted prefix gets a compact allocation and `self.values` retains the
    /// original large one.
    #[test]
    fn emit_first_small_n_allocates_minimally() -> Result<()> {
        let mut gv = GroupValuesPrimitive::<Int32Type>::new(DataType::Int32);

        // Intern 20 distinct values; `new()` pre-allocates capacity 128 for `values`.
        let arr: ArrayRef = Arc::new(Int32Array::from_iter_values(0..20i32));
        let mut groups = vec![];
        gv.intern(&[arr], &mut groups)?;
        let capacity_before = gv.values.capacity(); // 128

        // n=4, n*2=8 <= len=20 -> drain branch
        let emitted = gv.emit(EmitTo::First(4))?;

        assert_eq!(emitted[0].len(), 4);

        // `self.values` must retain its original large allocation.
        // Old split_off+swap left it with a fresh small allocation (~16).
        assert_eq!(
            gv.values.capacity(),
            capacity_before,
            "self.values capacity {} should equal original {} after small First(n) emit",
            gv.values.capacity(),
            capacity_before,
        );

        Ok(())
    }

    #[test]
    fn primitive_group_values_emit_first_block() -> Result<()> {
        let mut gv = GroupValuesPrimitive::<Int32Type>::new(DataType::Int32);
        let arr: ArrayRef = Arc::new(Int32Array::from_iter_values(0..6i32));
        let mut groups = vec![];

        gv.intern(&[arr], &mut groups)?;

        let emitted = gv.emit(EmitTo::FirstBlock(4))?;
        let emitted = emitted[0].as_primitive::<Int32Type>();
        assert_eq!(emitted.values(), &[0, 1, 2, 3]);

        let remaining = gv.emit(EmitTo::All)?;
        let remaining = remaining[0].as_primitive::<Int32Type>();
        assert_eq!(remaining.values(), &[4, 5]);

        Ok(())
    }

    #[test]
    fn primitive_group_values_first_block_does_not_shift_remaining_state() -> Result<()> {
        let mut gv = GroupValuesPrimitive::<Int32Type>::new(DataType::Int32);
        let arr: ArrayRef = Arc::new(Int32Array::from_iter_values(0..6i32));
        let mut groups = vec![];

        gv.intern(&[arr], &mut groups)?;
        let original_capacity = gv.values.capacity();

        let emitted = gv.emit(EmitTo::FirstBlock(4))?;
        assert_eq!(emitted[0].len(), 4);
        assert_eq!(gv.values.len(), 6);
        assert_eq!(gv.values.capacity(), original_capacity);
        assert_eq!(gv.len(), 2);

        let remaining = gv.emit(EmitTo::FirstBlock(2))?;
        let remaining = remaining[0].as_primitive::<Int32Type>();
        assert_eq!(remaining.values(), &[4, 5]);
        assert!(gv.is_empty());
        assert!(gv.values.is_empty());

        Ok(())
    }
}
