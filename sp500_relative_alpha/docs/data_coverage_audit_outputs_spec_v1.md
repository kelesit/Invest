# Data Coverage Audit Outputs Spec v1

**日期**: 2026-04-15  
**状态**: superseded for current daily v1 / retained for historical 1-minute path  
**作用**: 定义 `data_coverage_audit_protocol_v1` 所要求输出物的逻辑 schema、字段语义、主键与一致性校验规则

## 0. 这份文档的边界

注意：

- 本文件是原 `1-minute OHLCV` coverage audit outputs 的 schema
- 当前 v1 已改为 daily `OHLCV`
- 因此本文件不再是当前 round1 的主线 outputs spec

这份文档定义的是：

- 审计输出物**必须表达什么信息**
- 每个输出物的最小字段集合
- 哪些列构成主键
- 哪些枚举值是允许的
- 不同输出物之间应满足哪些一致性关系

这份文档**不**定义：

- 具体文件格式
- 具体存储路径
- 具体实现语言
- 具体数据库或 dataframe 框架

也就是说：

- 这是**逻辑 schema**
- 不是 L4 的工程落地规范

---

## 1. Shared Metadata

所有 audit 输出物都必须能追溯到同一次审计运行。

逻辑上，每个输出物都必须能关联以下共享元信息：

- `audit_run_id`
  - 本次 coverage audit 的唯一标识
- `protocol_version`
  - 当前应为 `data_coverage_audit_protocol_v1`
- `outputs_spec_version`
  - 当前应为 `data_coverage_audit_outputs_spec_v1`
- `preregistration_id`
  - 当前 round 1 应为 `SP500RA-V1-R1`
- `data_snapshot_id`
  - 对应原始数据冻结点的标识
- `generated_at_utc`
  - 输出物生成时间

允许的表达方式：

- 逐行列出
- 或作为文件级元信息保存

但要求是：

- 审计重放时必须能无歧义恢复这些字段

---

## 2. Output Objects

### 2.1 `universe_proxy_manifest`

作用：

- 冻结本轮 proxy universe 的 constituent 名单与身份映射

逻辑主键：

- `(audit_run_id, symbol_role, symbol)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `audit_run_id` | string | 审计运行 ID |
| `symbol_role` | enum | `constituent` 或 `benchmark` |
| `symbol` | string | 显示 symbol，例如 `AAPL` |
| `security_id` | string | 内部连续 security identity |
| `instrument_type` | string | 证券类型说明 |
| `include_flag` | bool | 是否纳入本轮 audit universe |
| `exclude_reason` | string/null | 若不纳入，记录原因 |
| `universe_note` | string/null | 额外说明 |

约束：

- `SPY` 必须出现且 `symbol_role = benchmark`
- 所有纳入 constituent 必须满足 `include_flag = true`
- 若 `include_flag = false`，则 `exclude_reason` 不得为空

### 2.2 `trading_calendar_manifest`

作用：

- 冻结本轮 audit 使用的交易日历与理论 session 长度

逻辑主键：

- `(audit_run_id, trading_date)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `audit_run_id` | string | 审计运行 ID |
| `trading_date` | date | 交易日期 |
| `market_status` | enum | `full_day`, `half_day`, `market_closed` |
| `session_open_et` | string/null | 例如 `09:30:00` |
| `session_close_et` | string/null | 例如 `16:00:00` 或 `13:00:00` |
| `expected_regular_minutes` | int | `390`, `210`, 或 `0` |
| `calendar_note` | string/null | 额外说明 |

约束：

- `full_day -> expected_regular_minutes = 390`
- `half_day -> expected_regular_minutes = 210`
- `market_closed -> expected_regular_minutes = 0`

### 2.3 `symbol_day_audit_table`

作用：

- 给出每个 `symbol-day` 的完整状态、失败原因与基础结构统计

逻辑主键：

- `(audit_run_id, symbol, trading_date)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `audit_run_id` | string | 审计运行 ID |
| `symbol` | string | 证券 symbol |
| `security_id` | string | 内部连续 security identity |
| `trading_date` | date | 交易日期 |
| `market_status` | enum | `full_day`, `half_day`, `market_closed` |
| `expected_regular_minutes` | int | 理论分钟数 |
| `observed_regular_minutes` | int | 实际 regular-session 分钟数 |
| `symbol_day_status` | enum | `full_valid`, `partial`, `missing`, `not_expected` |
| `duplicate_timestamp_count` | int | 重复时间戳数量 |
| `outside_session_bar_count` | int | session 外 bar 数量 |
| `invalid_price_count` | int | 非法价格条数 |
| `invalid_hilo_count` | int | `high/low` 结构非法条数 |
| `first_bar_ts_et` | string/null | 当日首个 regular-session bar 时间 |
| `last_bar_ts_et` | string/null | 当日最后一个 regular-session bar 时间 |
| `failure_code_primary` | enum/null | 主失败原因 |
| `failure_code_secondary` | enum/null | 次失败原因 |
| `adjustment_support_flag` | bool | 该 symbol-day 是否具备调整口径支持 |

允许的 `failure_code_*` 值：

- `NONE`
- `MISSING_ALL`
- `MISSING_PARTIAL`
- `DUPLICATE_TIMESTAMP`
- `OUTSIDE_SESSION`
- `INVALID_PRICE`
- `INVALID_HILO`
- `ADJUSTMENT_UNSUPPORTED`
- `MARKET_CLOSED`

约束：

- `market_status = market_closed -> symbol_day_status = not_expected`
- `symbol_day_status = full_valid -> observed_regular_minutes = expected_regular_minutes`
- `symbol_day_status = missing -> observed_regular_minutes = 0`
- `symbol_day_status = full_valid -> failure_code_primary in {NONE, null}`

说明：

- `outside_session_bar_count` 是信息列
- raw source 中存在盘前 / 盘后 bars 不自动构成失败
- `OUTSIDE_SESSION` 只有在 session 外记录无法被 cleanly 过滤、从而污染 canonical regular-session rows 时才应作为 failure code

### 2.4 `daily_breadth_summary`

作用：

- 汇总每天横截面有多宽，以及质量状态如何分布

逻辑主键：

- `(audit_run_id, trading_date)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `audit_run_id` | string | 审计运行 ID |
| `trading_date` | date | 交易日期 |
| `market_status` | enum | `full_day`, `half_day`, `market_closed` |
| `n_constituents_expected` | int | universe 中应统计的 constituent 数 |
| `n_full_valid` | int | `full_valid` constituent 数 |
| `n_partial` | int | `partial` constituent 数 |
| `n_missing` | int | `missing` constituent 数 |
| `breadth` | float | `n_full_valid / n_constituents_expected` |

约束：

- `breadth = n_full_valid / n_constituents_expected`
- `n_full_valid + n_partial + n_missing = n_constituents_expected`
- `market_status = market_closed` 的日期不应进入 breadth 统计主序列

### 2.5 `symbol_coverage_summary`

作用：

- 汇总每个 constituent symbol 在整个候选样本区间内的覆盖连续性

逻辑主键：

- `(audit_run_id, symbol)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `audit_run_id` | string | 审计运行 ID |
| `symbol` | string | constituent symbol |
| `security_id` | string | 内部连续 security identity |
| `n_expected_days` | int | 期望交易日数量 |
| `n_full_valid_days` | int | `full_valid` 天数 |
| `n_partial_days` | int | `partial` 天数 |
| `n_missing_days` | int | `missing` 天数 |
| `coverage_ratio` | float | `n_full_valid_days / n_expected_days` |
| `max_consecutive_missing_days` | int | 最长连续 `missing` 天数 |
| `max_consecutive_nonfull_days` | int | 最长连续非 `full_valid` 天数 |
| `coverage_pass_95` | bool | 是否达到 `coverage_i >= 95%` |

约束：

- `n_full_valid_days + n_partial_days + n_missing_days = n_expected_days`
- `coverage_ratio = n_full_valid_days / n_expected_days`

### 2.6 `spy_coverage_summary`

作用：

- 作为 benchmark anchor 的单独连续性审计摘要

逻辑主键：

- `(audit_run_id, benchmark_symbol)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `audit_run_id` | string | 审计运行 ID |
| `benchmark_symbol` | string | 当前应为 `SPY` |
| `n_expected_days` | int | 候选区间内应有交易日数 |
| `n_full_valid_days` | int | `SPY` 的 `full_valid` 天数 |
| `n_partial_days` | int | `SPY` 的 `partial` 天数 |
| `n_missing_days` | int | `SPY` 的 `missing` 天数 |
| `coverage_ratio` | float | `coverage_SPY` |
| `max_consecutive_missing_days` | int | 最长连续缺失 |
| `holdout_missing_day_count` | int | 候选 holdout 区间内缺失天数 |
| `benchmark_gate_pass` | bool | 是否通过 benchmark gate |

约束：

- `benchmark_symbol` 当前必须等于 `SPY`
- `coverage_ratio = n_full_valid_days / n_expected_days`

### 2.7 `adjustment_support_summary`

作用：

- 单独汇总 split-consistent adjustment 支持情况

逻辑主键：

- `(audit_run_id, symbol)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `audit_run_id` | string | 审计运行 ID |
| `symbol` | string | constituent 或 `SPY` |
| `security_id` | string | 内部连续 security identity |
| `adjustment_support_pass` | bool | 是否具备可验证 adjustment 支持 |
| `first_supported_date` | date/null | 首个受支持日期 |
| `last_supported_date` | date/null | 最后受支持日期 |
| `adjustment_note` | string/null | 失败原因或说明 |

约束：

- 若 `adjustment_support_pass = false`，则 `adjustment_note` 不得为空

### 2.8 `sample_boundary_decision`

作用：

- 记录最终的 `GO / NO-GO` 决策与冻结边界

逻辑主键：

- `(audit_run_id)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `audit_run_id` | string | 审计运行 ID |
| `decision` | enum | `GO` 或 `NO_GO` |
| `candidate_raw_start` | date/null | 候选冻结样本起点 |
| `candidate_raw_end` | date/null | 候选冻结样本终点 |
| `research_period_start` | date/null | 研究期起点 |
| `research_period_end` | date/null | 研究期终点 |
| `final_holdout_start` | date/null | final holdout 起点 |
| `final_holdout_end` | date/null | final holdout 终点 |
| `benchmark_gate_pass` | bool | 第 5.1 节是否通过 |
| `breadth_gate_pass` | bool | 第 5.2 节是否通过 |
| `symbol_continuity_gate_pass` | bool | 第 5.3 节是否通过 |
| `adjustment_gate_pass` | bool | 第 5.4 节是否通过 |
| `span_gate_pass` | bool | 第 5.5 节是否通过 |
| `decision_reason` | string | 决策解释 |

约束：

- `decision = GO` 时，所有 gate pass 必须为 `true`
- `decision = GO` 时，所有 boundary 日期不得为空
- `decision = NO_GO` 时，boundary 日期允许为空

---

## 3. Enumerations

允许的核心枚举值如下。

### 3.1 `market_status`

- `full_day`
- `half_day`
- `market_closed`

### 3.2 `symbol_day_status`

- `full_valid`
- `partial`
- `missing`
- `not_expected`

### 3.3 `symbol_role`

- `constituent`
- `benchmark`

### 3.4 `decision`

- `GO`
- `NO_GO`

---

## 4. Cross-Output Consistency Rules

单个表有主键还不够，输出物之间还必须彼此对得上。

### 4.1 Universe consistency

- `daily_breadth_summary.n_constituents_expected`
  - 必须等于 `universe_proxy_manifest` 中 `symbol_role = constituent and include_flag = true` 的数量

### 4.2 Breadth consistency

对任意 `trading_date`：

- `daily_breadth_summary.n_full_valid`
  - 必须等于 `symbol_day_audit_table` 中当日 constituent `full_valid` 的数量
- `daily_breadth_summary.n_partial`
  - 必须等于当日 constituent `partial` 的数量
- `daily_breadth_summary.n_missing`
  - 必须等于当日 constituent `missing` 的数量

### 4.3 Coverage consistency

对任意 symbol：

- `symbol_coverage_summary.n_full_valid_days`
  - 必须等于 `symbol_day_audit_table` 中该 symbol 的 `full_valid` 天数
- `symbol_coverage_summary.n_partial_days`
  - 必须等于 `partial` 天数
- `symbol_coverage_summary.n_missing_days`
  - 必须等于 `missing` 天数

### 4.4 Benchmark consistency

- `spy_coverage_summary` 必须只对应 `SPY`
- `SPY` 的 daily 状态必须能从 `symbol_day_audit_table` 中完整重算

### 4.5 Adjustment consistency

- 若某 symbol 在 `adjustment_support_summary.adjustment_support_pass = false`
  - 则该 symbol 不得在 `sample_boundary_decision` 的 `GO` 区间内被视为 fully supported

---

## 5. Minimal Validation Queries

一轮 audit 结束后，至少必须能回答下列问题：

- 当前 universe 一共有多少 constituent？
- 哪些交易日是 half day？
- 每个交易日有多少 constituent 是 `full_valid`？
- 哪些 symbol 覆盖率低于 `95%`？
- `SPY` 是否在候选 holdout 内存在缺失？
- adjustment support 是不是对所有纳入 symbol 都通过？
- 最终结论是 `GO` 还是 `NO_GO`，为什么？

如果这些问题不能直接由输出物回答：

- 说明 schema 设计仍不完整

---

## 6. Relationship to Audit Protocol

这份文档是：

- [data_coverage_audit_protocol_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/data_coverage_audit_protocol_v1.md)

第 `7` 节 Required Audit Outputs 的 schema 展开版。

关系是：

- protocol 定义“必须产出哪些对象，以及为什么”
- 本文定义“每个对象至少要包含哪些字段与一致性约束”
