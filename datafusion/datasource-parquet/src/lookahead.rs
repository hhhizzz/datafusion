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

use datafusion_execution::memory_pool::{MemoryPool, MemoryReservation};
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

pub(crate) const MAX_IN_FLIGHT_RANGES: usize = 24;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "consumed by the lookahead decoder task")
)]
pub(crate) const MAX_RANGES_PER_FILE_FETCH: usize = 4;
pub(crate) const MAX_SPECULATIVE_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "permits are consumed by the lookahead decoder task"
    )
)]
pub(crate) struct ParquetLookaheadCoordinator {
    pub(crate) range_permits: Arc<Semaphore>,
    byte_permits: Arc<Semaphore>,
}

impl ParquetLookaheadCoordinator {
    pub(crate) fn new() -> Self {
        Self {
            range_permits: Arc::new(Semaphore::new(MAX_IN_FLIGHT_RANGES)),
            byte_permits: Arc::new(Semaphore::new(MAX_SPECULATIVE_BYTES)),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LookaheadScanContext {
    pub(crate) coordinator: Arc<ParquetLookaheadCoordinator>,
    pub(crate) memory_pool: Arc<dyn MemoryPool>,
}

#[derive(Debug, Clone)]
pub(crate) struct LookaheadFileContext {
    pub(crate) coordinator: Arc<ParquetLookaheadCoordinator>,
    pub(crate) reservation: Arc<MemoryReservation>,
}

impl LookaheadFileContext {
    pub(crate) fn new(
        coordinator: Arc<ParquetLookaheadCoordinator>,
        reservation: Arc<MemoryReservation>,
    ) -> Self {
        Self {
            coordinator,
            reservation,
        }
    }

    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "consumed by the lookahead decoder task")
    )]
    pub(crate) fn try_reserve(&self, bytes: usize) -> Option<SpeculativeLease> {
        let permit_count = u32::try_from(bytes).ok()?;
        let byte_permit = Arc::clone(&self.coordinator.byte_permits)
            .try_acquire_many_owned(permit_count)
            .ok()?;

        self.reservation.try_grow(bytes).ok()?;

        Some(SpeculativeLease {
            reservation: Arc::clone(&self.reservation),
            bytes,
            _byte_permit: byte_permit,
        })
    }
}

#[derive(Debug)]
pub(crate) struct SpeculativeLease {
    reservation: Arc<MemoryReservation>,
    bytes: usize,
    _byte_permit: OwnedSemaphorePermit,
}

impl Drop for SpeculativeLease {
    fn drop(&mut self) {
        self.reservation.shrink(self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::config::ConfigOptions;
    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryConsumer, MemoryPool, UnboundedMemoryPool,
    };
    use std::process::Command;
    use std::sync::Arc;

    const ENV_TEST_CHILD: &str = "DATAFUSION_ROW_GROUP_LOOKAHEAD_ENV_TEST_CHILD";
    const ROW_GROUP_LOOKAHEAD_ENV: &str =
        "DATAFUSION_EXECUTION_PARQUET_ROW_GROUP_LOOKAHEAD";

    fn context_with_pool(
        pool: Arc<dyn MemoryPool>,
    ) -> (Arc<ParquetLookaheadCoordinator>, LookaheadFileContext) {
        let coordinator = Arc::new(ParquetLookaheadCoordinator::new());
        let reservation = Arc::new(MemoryConsumer::new("lookahead-test").register(&pool));
        let context = LookaheadFileContext::new(Arc::clone(&coordinator), reservation);
        (coordinator, context)
    }

    #[test]
    fn row_group_lookahead_defaults_to_false() {
        assert!(
            !ConfigOptions::default()
                .execution
                .parquet
                .row_group_lookahead
        );
    }

    #[test]
    fn row_group_lookahead_parses_from_env() {
        let output = Command::new(std::env::current_exe().unwrap())
            .arg("row_group_lookahead_parses_from_env_child")
            .arg("--nocapture")
            .env(ENV_TEST_CHILD, "1")
            .env(ROW_GROUP_LOOKAHEAD_ENV, "true")
            .output()
            .expect("failed to run isolated environment test");

        assert!(
            output.status.success(),
            "isolated environment test failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn row_group_lookahead_parses_from_env_child() {
        if std::env::var_os(ENV_TEST_CHILD).is_none() {
            return;
        }

        let options = ConfigOptions::from_env().unwrap();

        assert!(options.execution.parquet.row_group_lookahead);
    }

    #[test]
    fn coordinator_has_exact_fixed_budgets() {
        let coordinator = ParquetLookaheadCoordinator::new();

        assert_eq!(MAX_IN_FLIGHT_RANGES, 24);
        assert_eq!(MAX_RANGES_PER_FILE_FETCH, 4);
        assert_eq!(MAX_SPECULATIVE_BYTES, 256 * 1024 * 1024);
        assert_eq!(
            coordinator.range_permits.available_permits(),
            MAX_IN_FLIGHT_RANGES
        );
        assert_eq!(
            coordinator.byte_permits.available_permits(),
            MAX_SPECULATIVE_BYTES
        );
    }

    #[test]
    fn successful_lease_reserves_bytes_in_both_budgets() {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let (coordinator, context) = context_with_pool(Arc::clone(&pool));

        let lease = context.try_reserve(1024).expect("reservation should fit");

        assert_eq!(context.reservation.size(), 1024);
        assert_eq!(pool.reserved(), 1024);
        assert_eq!(
            coordinator.byte_permits.available_permits(),
            MAX_SPECULATIVE_BYTES - 1024
        );
        drop(lease);
    }

    #[test]
    fn byte_cap_denial_returns_none_without_growing_memory() {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let (coordinator, context) = context_with_pool(Arc::clone(&pool));
        let lease = context
            .try_reserve(MAX_SPECULATIVE_BYTES)
            .expect("exact byte budget should fit");

        assert!(context.try_reserve(1).is_none());
        assert_eq!(context.reservation.size(), MAX_SPECULATIVE_BYTES);
        assert_eq!(pool.reserved(), MAX_SPECULATIVE_BYTES);
        assert_eq!(coordinator.byte_permits.available_permits(), 0);

        drop(lease);
        assert_eq!(
            coordinator.byte_permits.available_permits(),
            MAX_SPECULATIVE_BYTES
        );
    }

    #[test]
    fn memory_pool_denial_returns_none_and_releases_byte_permits() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(7));
        let (coordinator, context) = context_with_pool(Arc::clone(&pool));

        assert!(context.try_reserve(8).is_none());

        assert_eq!(context.reservation.size(), 0);
        assert_eq!(pool.reserved(), 0);
        assert_eq!(
            coordinator.byte_permits.available_permits(),
            MAX_SPECULATIVE_BYTES
        );
    }

    #[test]
    fn dropping_leases_releases_each_reservation_exactly_once() {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let (coordinator, context) = context_with_pool(Arc::clone(&pool));
        let first = context.try_reserve(5).unwrap();
        let second = context.try_reserve(7).unwrap();

        drop(first);
        assert_eq!(context.reservation.size(), 7);
        assert_eq!(pool.reserved(), 7);
        assert_eq!(
            coordinator.byte_permits.available_permits(),
            MAX_SPECULATIVE_BYTES - 7
        );

        drop(second);
        assert_eq!(context.reservation.size(), 0);
        assert_eq!(pool.reserved(), 0);
        assert_eq!(
            coordinator.byte_permits.available_permits(),
            MAX_SPECULATIVE_BYTES
        );
    }
}
