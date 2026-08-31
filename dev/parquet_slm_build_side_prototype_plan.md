# DataFusion Parquet SLM 机制原型与性能验证计划

日期：2026-08-31

状态：**PLAN ONLY — 未授权实现**

计划分支：`test/parquet-slm-plan-20260831`（DataFusion 仓库，plan-only）

## 1. 要回答的问题

在 DataFusion 当前 columnar HashJoin 与 Parquet reader 中，把 **build-side、仅在 join 后使用的宽 payload** 从首次 scan 延迟到 Inner HashJoin 之后读取，是否能在结果完全等价的前提下：

1. 降低首次 Parquet 读取/解码字节；
2. 降低 HashJoin build-side 常驻 Arrow buffer 与 `concat_batches` 峰值；
3. 在计入 row-handle 传播、排序去重、二次读取和 scatter 后，获得可分辨的端到端收益。

本计划先回答机制问题，不先实现通用 SLM optimizer。

## 2. 固定基线与证据边界

- DataFusion baseline：`e4cf35cbca92e6fda855d5eac4104b47e0e600e9`。
- Arrow/Parquet dependency：`782e5a685501a9db6cc8e9a3b7cbff894940c47a`（59.2.0）。
- 分析依据：本地 `codex/references/slm-datafusion-feasibility-20260831.md`；本计划自包含执行约束与验收门。
- 所有编译、测试、benchmark、profiling 必须通过 `codex/scripts/dfssh`；不得本地、裸 SSH 或 Kubernetes 执行。
- 第一阶段是 warm-cache、单机、单文件、单 partition 的机制证据；不能据此声称冷磁盘、对象存储或分布式收益。
- 所有性能数字必须同时记录 DataFusion/Arrow SHA、fixture 几何、session options、host load 和 evidence 目录。

## 3. 已冻结的先行 WIP

在用户发出 plan-only 指令前，曾创建并运行：

- DataFusion WIP branch：`prototype/parquet-slm-build-20260831`；
- commits：`e5d4345bfc`、`fcf88b8514`、`06a89dbb71`；
- dfssh baseline smoke evidence：
  - `codex/logs/dfssh/20260831-093253-slm-baseline-smoke-20260831/`
  - `codex/logs/dfssh/20260831-093708-slm-baseline-contract-smoke2-20260831/`
- worktree 当前还有未提交的 forced-arm WIP。

这些内容保持冻结，不属于本计划分支。未经用户再次明确批准，不继续修改、运行、清理、合并或提交该 WIP。

## 4. Phase A 范围

### 4.1 纳入范围

- 一个 immutable 本地 Parquet build file；
- 一个单 partition MemTable probe；
- `Inner HashJoin`，Parquet 固定为 left/build side；
- `PartitionMode::CollectLeft`；
- payload 是 top-level、非空、output-only 的 base column；
- payload 不参与 scan predicate、pre-join filter、join key、`JoinFilter`、sort 或 repartition expression；
- 读取时使用 `Utf8View`；
- build key 唯一，probe 全部命中，可控制 distinct selected rows 与 duplicate fanout；
- baseline 与 forced 均使用同一 runtime、metadata cache、reader options 和 fixture。

### 4.2 明确不做

- 不修改 production `ExecutionPlan`、optimizer rule 或公开 interface；
- 不实现 `ParquetMaterializeExec`；
- 不实现自动成本模型；
- 不支持 multi-file、outer/semi/anti join、nested payload、schema evolution、mutable source；
- 不支持 spill、bounded backpressure、cancellation 或 proto/FFI；
- 不跑 JOB；当前 dfssh 没有 JOB runner；
- 不把两阶段 harness 描述为“已把 SLM 集成进一条 SQL physical plan”。

## 5. 必须先成立的正确性 contract

| Contract | Phase A 处理 | 未来 production 要求 |
|---|---|---|
| 同一 snapshot | 单个测试进程持有 immutable temp file | ObjectStore version/ETag 或 If-Match，不能仅按 path 二读 |
| row handle 真实可传播 | 使用 Parquet `file_row_index()` 产生普通 `Int64` physical column | 多文件使用 query-local `file_id + row_number` typed token |
| deferred 属性无前置引用 | plan-shape assertion + 固定 SQL | optimizer 做完整 attribute-use closure 分析 |
| schema semantics 一致 | fixture schema 完全一致，二读复用 `ParquetSource` 与 table options | 复用 per-file schema adapter、missing-column/cast/INT96/encryption |
| access plan 唯一 | fresh `PartitionedFile + ParquetRowSelection` | 与原 access plan 求交，不得同时附加两个 access extensions |
| selection 覆盖完整文件 | `RowSelection::from_consecutive_ranges(..., total_rows)` | 校验头尾 skip、row group/page offsets 与 file range |
| duplicates/order | unique fetch 一次，按 output ordinal scatter 多次 | null handle、fanout、ordering、partitioning 全覆盖 |
| exact output | baseline/forced schema与逐行结果完全一致 | 全套 SQL differential tests，不允许静默 fallback |

## 6. 计划中的 test source branch

只有用户批准本计划后，才创建新的 DataFusion source branch；建议名称：

`test/parquet-slm-mechanism-20260831`

不要继续使用已冻结的 `prototype/parquet-slm-build-20260831` 作为正式实验身份。新分支从固定 baseline `e4cf35cbca` 创建干净 worktree。

计划采用三笔可独立识别的 commit：

1. `bench: add Parquet SLM baseline harness`
   - 只含 fixture、baseline plan、oracle、metrics；
   - 不含 forced code；
   - 作为 `main/off` 测量身份。
2. `bench: add forced Parquet late-fetch arm`
   - 增加 row-handle join、selection fetch、scatter；
   - 同一 commit 可运行 `--arm baseline` 与 `--arm forced`；
   - baseline mode 作为 `candidate/off`。
3. `bench: pin SLM plan and measurement contracts`
   - 只修 plan assertions、metrics 与可重复性；
   - 不改变数据/算法语义；
   - 若改变语义，必须重新建立三臂 SHA。

## 7. 计划修改的文件

仅限：

- `datafusion/core/Cargo.toml`
  - 注册一个 `harness = false`、`required-features = ["parquet"]` 的 throwaway bench target。
- `datafusion/core/benches/parquet_slm_build_side_prototype.rs`
  - 文件头必须写明 `THROWAWAY` 和要回答的问题；
  - 包含 fixture、baseline/forced、oracle、metrics、manual balanced runner。

不得修改其他 production 文件。若 public API 不足以完成两阶段 prototype，应停止并回报缺口，而不是扩大 patch。

## 8. 两条执行路径

### 8.1 Baseline / eager

```text
ParquetSource(build_key, payload)
  -> HashJoinExec(left/build, CollectLeft)
  -> output(probe_id, payload)
```

### 8.2 Forced / two-stage SLM mechanism

```text
Stage 1:
ParquetSource(build_key, file_row_index AS row_handle)
  -> HashJoinExec(left/build, CollectLeft)
  -> output(probe_id, row_handle)

Stage 2:
sort + dedup row_handle
  -> fresh ParquetRowSelection(payload only)
  -> read unique payload rows in file order
  -> take/scatter back to original join order and duplicates
  -> output(probe_id, payload)
```

Stage 1、Stage 2、sort/dedup、plan creation 和 scatter 全部计入 forced wall time。

## 9. 必须编码的 plan-shape assertions

两臂共同：

- 恰好一个 `HashJoinExec`；
- `join_type = Inner`；
- `partition_mode = CollectLeft`；
- left subtree 是 Parquet，right subtree 是 MemTable；
- 禁止 perfect-hash fast path；
- `target_partitions = 1`、`join_reordering = false`、`repartition_joins = false`；
- `hash_join_single_partition_threshold` 与 rows threshold 显式固定，防止大 fixture 变成 `Partitioned`。

Baseline：

- left schema 精确包含 `build_key + payload`；
- payload 类型为 `Utf8View`；
- 不含 row handle。

Forced Stage 1：

- `file_row_index()` 必须位于 Parquet-only derived table 内；不能放在 join 顶层；
- left schema 精确包含 `build_key + row_handle`；
- 不含 payload；
- scan plan 必须显示 Parquet `RowNumber` virtual column。

Forced Stage 2：

- projection 只包含 payload；
- `second_read_rows == unique_handles`；
- `requested_handles == join_output_rows`；
- selection 总长度等于 file row count；
- 最终 schema 与 baseline 完全相同。

## 10. Fixture contract

### 10.1 Smoke

- build rows：16,384；
- payload：64B；
- row group：4,096 rows；
- page：512 rows；
- selected：128；
- probe/output：512；
- 目的：编译、plan、oracle 和 metrics，不用于性能结论。

### 10.2 Formal

- build rows：1,048,576；
- payload：512B、每行不同的 Utf8；
- 16 × 65,536-row row groups；
- 1,024 rows/page；
- `UNCOMPRESSED`；
- dictionary disabled；
- page statistics enabled；
- offset index enabled且逐 column 断言存在；
- DataFusion read schema 强制/断言 `Utf8View`。

### 10.3 预注册 cases

| Case | Distinct selected | Duplicate factor | Pattern | 预期用途 |
|---|---:|---:|---|---|
| `clustered_1pct_d1` | 10,486 | 1 | contiguous | locality 有利样例 |
| `random_64_d1` | 64 | 1 | uniform random | 稀疏随机有利样例 |
| `random_64_d16` | 64 | 16 | uniform random | dedup/fanout 样例 |
| `random_10pct_d1` | 104,858 | 1 | uniform random | 负对照，预计触及绝大多数 pages |

注意：`random 0.01%` 并不天然稀疏。若 100+ selected rows 均匀落在 1,024 个 pages，触页率需按 `1-(1-1/pages)^N` 计算并报告，不能只看 row selectivity。

## 11. Oracle 与正确性测试

每个 case 在 timing 前执行一次 baseline/forced preflight：

1. schema、field type、nullability 完全相等；
2. row count 完全相等；
3. 以 `probe_id` 排序后逐行比较 payload bytes；
4. 输出稳定 ordered digest 供日志核对；
5. `requested == probe_rows`；
6. `unique == selected_rows`；
7. `second_read_rows == unique`；
8. row handles 全部在 `[0, build_rows)`；
9. duplicates 只二读一次，但 scatter 后出现正确次数；
10. 任一 assertion 失败则不进入 timing。

## 12. 必收 metrics

每个 trial 同时输出：

- `wall_ns`；
- `first_read_bytes`；
- `second_read_bytes`；
- `total_bytes_scanned`；
- `build_mem_used`；
- `build_time`；
- `requested_handles`；
- `unique_handles`；
- `second_read_rows`；
- `row_groups/pages/ranges touched`（若现有 metrics 不足，先标记 unavailable，不扩大 production patch）；
- `sort_dedup_ns`；
- `second_fetch_ns`；
- `scatter_ns`；
- output rows 与 digest。

需要从 causality 上区分：

```text
baseline total = first scan + join + output take
forced total   = first scan + join + sort/dedup + second fetch + scatter
```

## 13. 三臂与运行顺序

三臂身份：

| Arm | DataFusion SHA | Mode | 目的 |
|---|---|---|---|
| `main/off` | baseline-harness commit | baseline | upstream 等价控制 |
| `candidate/off` | forced commit | baseline | 新代码存在但功能关闭的零开销检查 |
| `candidate/force` | forced commit | forced | 机制效果 |

所有 arm 使用同一个 Arrow SHA `782e5a...`。

每个独立 run 内：

- preflight 不计时；
- measured operation 前做同臂 priming；
- paired cross-check 使用交替 `AA/BB`：
  - round 0：baseline, baseline, forced, forced；
  - round 1：forced, forced, baseline, baseline；
- formal 至少 4 个 measured rounds；
- case/arm 的独立 dfssh runs 也使用 ABBA 顺序；
- 记录 `load_before/load_after`；高负载或明显竞争 run 作废重跑。

## 14. dfssh 计划命令（批准后才执行）

```bash
codex/scripts/dfssh doctor

codex/scripts/dfssh bench cargo \
  --slot a \
  --repo datafusion \
  --datafusion <baseline-harness-sha> \
  --arrow 782e5a685501a9db6cc8e9a3b7cbff894940c47a \
  -p datafusion \
  --bench parquet_slm_build_side_prototype \
  --features parquet \
  --label slm-main-off-r1 \
  -- --arm baseline <fixture-args>

codex/scripts/dfssh bench cargo \
  --slot b \
  --repo datafusion \
  --datafusion <forced-sha> \
  --arrow 782e5a685501a9db6cc8e9a3b7cbff894940c47a \
  -p datafusion \
  --bench parquet_slm_build_side_prototype \
  --features parquet \
  --label slm-candidate-force-r1 \
  -- --arm forced <fixture-args>
```

这是 custom `harness = false` binary，不传 Criterion 的 `--bench`、`--exact` 或 `--noplot` 到 binary。

## 15. 性能解释规则

- 先计算同臂 spread；效果小于 spread 时结论为不可分辨；
- per-case 只在 baseline/forced 多轮分布分离时下结论；
- candidate/off 若稳定偏离 main/off，先解释 harness/commit drift，不解读 forced；
- 只报告 warm-cache CPU/decode/build-memory 结果；
- `bytes_scanned` 是 reader 指标，不等于物理磁盘 bytes；
- 不把单轮最佳值或论文百分比当作当前实现收益；
- suite 总量若被单个 case 主导，按该 case 报告。

## 16. Go / No-Go gate

### Go：进入 production design

必须全部满足：

1. 所有 correctness preflight 零差异；
2. candidate/off 与 main/off 无可分辨回归；
3. 至少两个 favorable cases 中，forced wall-time 分布与 baseline 分离，且 gain ≥ 5%；
4. gain 能由 bytes/decode/build-memory 指标解释，而不是顺序或 cache 偏差；
5. `random_10pct_d1` 负对照可以变慢，但必须能由 page coverage 与 second-fetch cost 解释；
6. 结果在另一个 dfssh slot 或交叉顺序复现。

### Conditional Go：只进入 memory-pressure follow-up

- wall time 未赢，但 `build_mem_used` 明显下降且在预注册 memory-limit case 中减少 spill/OOM；
- 必须新增单独 preregistration，不能事后挑内存阈值。

### No-Go

- favorable cases 仍无可分辨收益；或
- two-stage fetch/scatter 吞掉全部 scan/build savings；或
- correctness 需要修改通用 production 接口才能在 Phase A 成立；或
- 收益只来自不公平 cache、metadata 或 plan-shape 差异。

No-Go 时保留报告与测试 branch，停止开发；不继续做自动 planner。

## 17. 通过后才讨论的 production seam

若 Phase A 为 Go，另写设计，不直接把 benchmark code搬入 production。候选 deep module：

```text
ParquetMaterializeExec
  input: standard RecordBatch stream carrying typed row handles
  output: original schema with deferred payload restored
  hides: grouping, selection, I/O, dedup, scatter, memory admission, metrics
```

Production design 必须补：

- snapshot-pinned reader；
- compact multi-file token；
- bounded streaming/backpressure/memory reservation；
- cancellation/error semantics；
- schema adapter/access-plan integration；
- optimizer eligibility 与 cost uncertainty fallback；
- outer join NULL handles；
- proto/FFI/EXPLAIN/metrics；
- full differential test matrix。

在出现第二种 production row-fetch source 前，不抽象通用 `RowFetchSource` interface。

## 18. 启动条件

本计划提交后保持 plan-only。只有用户明确回复批准执行本计划，才允许：

1. 创建新的 DataFusion test source branch；
2. 编写 benchmark harness；
3. 运行 dfssh smoke/benchmark；
4. 根据预注册 gate 给出 Go/No-Go。

不得把“可以开始”扩展成 production 集成、PR 创建或外部发布授权。
