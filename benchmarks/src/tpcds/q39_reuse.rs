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

use std::sync::Arc;

use datafusion::datasource::{MemTable, TableProvider};
use datafusion::error::Result;
use datafusion::prelude::SessionContext;

pub const Q39_REUSE_TABLE: &str = "q39_reuse_inv";

const INV_SQL: &str = r#"
SELECT w_warehouse_name, w_warehouse_sk, i_item_sk, d_moy,
       stdev, mean, CASE mean WHEN 0 THEN NULL ELSE stdev / mean END cov
FROM (
    SELECT w_warehouse_name, w_warehouse_sk, i_item_sk, d_moy,
           stddev_samp(inv_quantity_on_hand) stdev,
           avg(inv_quantity_on_hand) mean
    FROM inventory, item, warehouse, date_dim
    WHERE inv_item_sk = i_item_sk
      AND inv_warehouse_sk = w_warehouse_sk
      AND inv_date_sk = d_date_sk
      AND d_year = 1998
      AND d_moy IN (4, 5)
    GROUP BY w_warehouse_name, w_warehouse_sk, i_item_sk, d_moy
) foo
WHERE CASE mean WHEN 0 THEN 0 ELSE stdev / mean END > 1
"#;

pub struct Q39ReuseStats {
    pub rows: usize,
    pub batches: usize,
    pub estimated_bytes: usize,
}

pub struct MaterializedInv {
    pub table: Arc<dyn TableProvider>,
    pub stats: Q39ReuseStats,
}

pub async fn materialize(ctx: &SessionContext) -> Result<MaterializedInv> {
    let df = ctx.sql(INV_SQL).await?;
    let schema = Arc::new(df.schema().as_arrow().clone());
    let partitions = df.collect_partitioned().await?;
    let stats = Q39ReuseStats {
        rows: partitions
            .iter()
            .flatten()
            .map(|batch| batch.num_rows())
            .sum(),
        batches: partitions.iter().flatten().count(),
        estimated_bytes: partitions
            .iter()
            .flatten()
            .map(|batch| batch.get_array_memory_size())
            .sum(),
    };
    let table = MemTable::try_new(schema, partitions)?;

    Ok(MaterializedInv {
        table: Arc::new(table),
        stats,
    })
}

pub fn consumer_sql() -> [&'static str; 2] {
    [
        r#"
SELECT inv1.w_warehouse_sk, inv1.i_item_sk, inv1.d_moy, inv1.mean, inv1.cov,
       inv2.w_warehouse_sk, inv2.i_item_sk, inv2.d_moy, inv2.mean, inv2.cov
FROM q39_reuse_inv inv1, q39_reuse_inv inv2
WHERE inv1.i_item_sk = inv2.i_item_sk
  AND inv1.w_warehouse_sk = inv2.w_warehouse_sk
  AND inv1.d_moy = 4
  AND inv2.d_moy = 4 + 1
ORDER BY inv1.w_warehouse_sk, inv1.i_item_sk, inv1.d_moy, inv1.mean, inv1.cov,
         inv2.d_moy, inv2.mean, inv2.cov
"#,
        r#"
SELECT inv1.w_warehouse_sk, inv1.i_item_sk, inv1.d_moy, inv1.mean, inv1.cov,
       inv2.w_warehouse_sk, inv2.i_item_sk, inv2.d_moy, inv2.mean, inv2.cov
FROM q39_reuse_inv inv1, q39_reuse_inv inv2
WHERE inv1.i_item_sk = inv2.i_item_sk
  AND inv1.w_warehouse_sk = inv2.w_warehouse_sk
  AND inv1.d_moy = 4
  AND inv2.d_moy = 4 + 1
  AND inv1.cov > 1.5
ORDER BY inv1.w_warehouse_sk, inv1.i_item_sk, inv1.d_moy, inv1.mean, inv1.cov,
         inv2.d_moy, inv2.mean, inv2.cov
"#,
    ]
}

#[cfg(test)]
mod tests {
    use super::{Q39_REUSE_TABLE, consumer_sql, materialize};
    use arrow::array::Int64Array;
    use arrow::datatypes::SchemaRef;
    use arrow::util::pretty::pretty_format_batches;
    use datafusion::error::Result;
    use datafusion::prelude::SessionContext;

    const CANONICAL_Q39_SQL: &str =
        include_str!("../../../datafusion/core/tests/tpc-ds/39.sql");

    #[tokio::test]
    async fn materialized_consumers_match_canonical_q39_results() -> Result<()> {
        let ctx = SessionContext::new();
        register_q39_fixture(&ctx).await?;

        let canonical_sql: Vec<_> = CANONICAL_Q39_SQL
            .split(';')
            .map(str::trim)
            .filter(|sql| !sql.is_empty())
            .collect();
        assert_eq!(canonical_sql.len(), 2);

        let mut canonical_results = Vec::with_capacity(canonical_sql.len());
        for sql in &canonical_sql {
            canonical_results.push(formatted_result(&ctx, sql).await?);
        }

        let materialized = materialize(&ctx).await?;
        assert_eq!(materialized.stats.rows, 4);
        assert!(materialized.stats.batches > 0);
        assert!(materialized.stats.estimated_bytes > 0);
        ctx.register_table(Q39_REUSE_TABLE, materialized.table)?;

        let mut consumer_results = Vec::with_capacity(2);
        for sql in consumer_sql() {
            consumer_results.push(formatted_result(&ctx, sql).await?);
        }

        assert_eq!(canonical_results.len(), consumer_results.len());
        for (canonical, consumer) in canonical_results.iter().zip(&consumer_results) {
            assert_eq!(canonical.0, consumer.0);
            assert_eq!(canonical.1, consumer.1);
        }
        assert_eq!(canonical_results[0].2, vec![10, 20]);
        assert_eq!(canonical_results[1].2, vec![10]);

        ctx.deregister_table(Q39_REUSE_TABLE)?;
        assert!(ctx.table(Q39_REUSE_TABLE).await.is_err());

        Ok(())
    }

    async fn formatted_result(
        ctx: &SessionContext,
        sql: &str,
    ) -> Result<(SchemaRef, String, Vec<i64>)> {
        let batches = ctx.sql(sql).await?.collect().await?;
        let schema = batches
            .first()
            .expect("q39 fixture produces a result batch")
            .schema();
        let item_sks = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("q39 item key is Int64")
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        Ok((
            schema,
            pretty_format_batches(&batches)?.to_string(),
            item_sks,
        ))
    }

    async fn register_q39_fixture(ctx: &SessionContext) -> Result<()> {
        for sql in [
            "CREATE TABLE inventory (\
             inv_item_sk BIGINT, inv_warehouse_sk BIGINT, inv_date_sk BIGINT, \
             inv_quantity_on_hand BIGINT) AS VALUES \
             (10, 1, 1, 0), (10, 1, 2, 0), (10, 1, 3, 3), \
             (10, 1, 4, 0), (10, 1, 5, 0), (10, 1, 6, 3), \
             (20, 1, 7, 0), (20, 1, 8, 2), \
             (20, 1, 9, 0), (20, 1, 10, 2)",
            "CREATE TABLE item (i_item_sk BIGINT) AS VALUES (10), (20)",
            "CREATE TABLE warehouse (w_warehouse_sk BIGINT, w_warehouse_name VARCHAR) \
             AS VALUES (1, 'warehouse 1')",
            "CREATE TABLE date_dim (d_date_sk BIGINT, d_year BIGINT, d_moy BIGINT) \
             AS VALUES \
             (1, 1998, 4), (2, 1998, 4), (3, 1998, 4), \
             (4, 1998, 5), (5, 1998, 5), (6, 1998, 5), \
             (7, 1998, 4), (8, 1998, 4), (9, 1998, 5), (10, 1998, 5)",
        ] {
            ctx.sql(sql).await?.collect().await?;
        }

        Ok(())
    }
}
