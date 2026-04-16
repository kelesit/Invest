# Audit Data Contract v1

**日期**: 2026-04-15  
**状态**: superseded for current daily v1 / retained for historical 1-minute path  
**作用**: 把 `system_spec_v1`、`data_coverage_audit_protocol_v1` 和 `data_coverage_audit_outputs_spec_v1` 映射成一组可执行的数据对象契约，定义 audit 上游必须具备哪些 canonical objects，以及这些对象之间如何变换

## 0. 这份文档做什么

注意：

- 本文件是原 `1-minute OHLCV` audit/data pipeline 的逻辑契约
- 当前 v1 已改为 daily `OHLCV`
- 因此本文件不再是当前 round1 的主线数据契约

前面三份文档分别回答了：

- 研究定义是什么
- 数据覆盖审计必须检查什么
- 审计输出物至少要长什么样

但还缺一层：

> 为了真正执行 audit，系统内部必须先有哪些 canonical data objects，  
> 每个对象的主键、字段、语义和变换边界是什么。

这份文档就是这一层。

它不是训练代码，也不是供应商映射代码。  
它定义的是：

- audit 执行前必须具备的输入合同
- audit 过程中允许出现的中间对象
- 最终 audit outputs 应该由哪些对象生成

---

## 1. Object Graph

当前 v1 的 audit 数据链路固定为：

`security master`
-> `exchange calendar`
-> `split reference`
-> `raw minute bars`
-> `normalized session minute bars`
-> `daily bar aggregates`
-> `symbol-day quality facts`
-> `audit outputs`

其中：

- 前四个是 audit 的上游输入对象
- 中间三个是 audit 执行时的 canonical 中间对象
- 最后一个是 [data_coverage_audit_outputs_spec_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/data_coverage_audit_outputs_spec_v1.md) 中定义的输出物集合

---

## 2. Input Contracts

### 2.1 `security_master_contract`

作用：

- 给每个 symbol 一个稳定的内部身份，并标记其在本轮 audit 里的角色

逻辑主键：

- `(security_id)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `security_id` | string | 内部稳定 security identity |
| `symbol` | string | 显示 symbol |
| `symbol_role` | enum | `constituent` / `benchmark` |
| `instrument_type` | string | 证券类型 |
| `include_flag` | bool | 是否纳入本轮 audit |
| `security_status_note` | string/null | 备注说明 |

约束：

- `SPY` 必须存在，且 `symbol_role = benchmark`
- constituent universe 只来自 `include_flag = true` 的普通股对象

### 2.2 `exchange_calendar_contract`

作用：

- 作为 session 规范化与预期 minute 数的唯一日历来源

逻辑主键：

- `(trading_date)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `trading_date` | date | 交易日期 |
| `market_status` | enum | `full_day`, `half_day`, `market_closed` |
| `session_open_et` | time/null | `09:30:00` 等 |
| `session_close_et` | time/null | `16:00:00` / `13:00:00` 等 |
| `expected_regular_minutes` | int | `390`, `210`, `0` |

约束：

- 必须与 `system_spec_v1` 的 regular-session 规则一致
- 所有 minute normalization 都以此对象为准，而不是以供应商原始标签为准

### 2.3 `split_reference_contract`

作用：

- 支持把原始分钟 OHLCV 映射到 split-consistent OHLCV

逻辑主键：

- `(security_id, ex_date)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `security_id` | string | 内部稳定 security identity |
| `ex_date` | date | 拆股生效日 |
| `split_factor` | float | 当次拆股因子 |
| `cum_split_factor` | float | 相对于某个参考点的累计因子 |
| `support_flag` | bool | 该 security 是否具备可验证的拆股支持 |
| `support_note` | string/null | 失败原因或说明 |

约束：

- 该对象只负责 split，不负责 dividend / total-return
- 若 `support_flag = false`，则该 security 不可被视为 fully adjustment-supported

### 2.4 `raw_minute_bar_contract`

作用：

- 承载供应商提供的未经本项目 session 规范化的分钟 bar

逻辑主键：

- `(security_id, raw_ts_source)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `security_id` | string | 内部稳定 security identity |
| `symbol` | string | 显示 symbol |
| `raw_ts_source` | timestamp | 源数据时间戳 |
| `source_tz` | string | 源时区 |
| `source_ts_convention` | enum | `bar_start`, `bar_end`, `unknown` |
| `open_raw` | float | 原始 open |
| `high_raw` | float | 原始 high |
| `low_raw` | float | 原始 low |
| `close_raw` | float | 原始 close |
| `volume_raw` | float | 原始 volume |
| `vendor_note` | string/null | 供应商附加说明 |

约束：

- 不允许在这个对象里提前丢弃 session 外 bars
- 不允许在这个对象里提前做 split adjustment
- 这个对象必须保留足够信息，使 session normalization 可重放

---

## 3. Canonical Intermediate Objects

### 3.1 `normalized_session_minute_bars`

作用：

- 把原始 minute bars 统一到本项目冻结的 regular-session 语义

逻辑主键：

- `(security_id, trading_date, minute_index)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `security_id` | string | 内部稳定 security identity |
| `symbol` | string | 显示 symbol |
| `trading_date` | date | ET 交易日期 |
| `minute_index` | int | 当日 session 内分钟序号，从 `0` 开始 |
| `bar_start_et` | timestamp | 该分钟起点 |
| `bar_end_et` | timestamp | 该分钟终点 |
| `open_raw_norm` | float | 规范化后的原始 open |
| `high_raw_norm` | float | 规范化后的原始 high |
| `low_raw_norm` | float | 规范化后的原始 low |
| `close_raw_norm` | float | 规范化后的原始 close |
| `volume_raw_norm` | float | 规范化后的原始 volume |
| `inside_regular_session_flag` | bool | 是否属于 canonical regular session |
| `duplicate_source_count` | int | 合并到该 canonical minute 的源记录数 |

约束：

- `minute_index` 对 full day 取值范围必须是 `0..389`
- `minute_index` 对 half day 取值范围必须是 `0..209`
- 不允许存在 session 外 canonical row
- 若供应商使用 `bar_end` 时间戳，则必须在本对象中统一转换到 canonical minute interval

### 3.2 `adjusted_session_minute_bars`

作用：

- 在 canonical regular-session minute bars 上施加 split-consistent adjustment

逻辑主键：

- `(security_id, trading_date, minute_index)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `security_id` | string | 内部稳定 security identity |
| `trading_date` | date | ET 交易日期 |
| `minute_index` | int | session minute 序号 |
| `open_adj` | float | split-adjusted open |
| `high_adj` | float | split-adjusted high |
| `low_adj` | float | split-adjusted low |
| `close_adj` | float | split-adjusted close |
| `volume_adj` | float | inverse-split-adjusted volume |
| `adjustment_support_flag` | bool | 该 minute 是否具备 adjustment 支持 |
| `applied_cum_split_factor` | float | 应用的累计 split 因子 |

约束：

- `adjustment_support_flag = false` 的行，不得参与 downstream full-valid 判定
- volume adjustment 必须与 price adjustment 方向一致，以保持 split-consistency

### 3.3 `daily_bar_aggregates`

作用：

- 从 adjusted minute bars 聚合出日频 OHLCV 与日频派生量

逻辑主键：

- `(security_id, trading_date)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `security_id` | string | 内部稳定 security identity |
| `symbol` | string | 显示 symbol |
| `trading_date` | date | ET 交易日期 |
| `market_status` | enum | `full_day`, `half_day` |
| `n_minutes_present` | int | 实际 canonical minute 数 |
| `open_d` | float | 当日 open |
| `high_d` | float | 当日 high |
| `low_d` | float | 当日 low |
| `close_d` | float | 当日 close |
| `volume_d` | float | 当日 volume |
| `bar_dollar_volume_d` | float | `sum(close_adj_1m * volume_adj_1m)` |
| `adjustment_support_flag` | bool | 当日是否 fully adjustment-supported |

约束：

- 这里只由 adjusted minute bars 聚合
- `bar_dollar_volume_d` 是项目级 aggregate，不得被误称为 trade-level exact dollar volume

### 3.4 `symbol_day_quality_facts`

作用：

- 把 minute 层完整性与 adjustment 支持压缩成 per symbol-day 的质量事实表

逻辑主键：

- `(security_id, trading_date)`

最小字段：

| Field | Type | Meaning |
|---|---|---|
| `security_id` | string | 内部稳定 security identity |
| `symbol` | string | 显示 symbol |
| `trading_date` | date | ET 交易日期 |
| `market_status` | enum | `full_day`, `half_day`, `market_closed` |
| `expected_regular_minutes` | int | 理论分钟数 |
| `observed_regular_minutes` | int | 实际 canonical minute 数 |
| `duplicate_timestamp_count` | int | 重复时间戳计数 |
| `outside_session_bar_count` | int | session 外 bar 计数 |
| `invalid_price_count` | int | 非法价格计数 |
| `invalid_hilo_count` | int | 非法 high/low 计数 |
| `adjustment_support_flag` | bool | 当日 adjustment 是否支持 |
| `symbol_day_status` | enum | `full_valid`, `partial`, `missing`, `not_expected` |
| `failure_code_primary` | enum/null | 主失败原因 |
| `failure_code_secondary` | enum/null | 次失败原因 |

约束：

- 这是生成 `symbol_day_audit_table` 的 canonical upstream
- `symbol_day_status` 的判定必须完全由本对象字段机械推出

---

## 4. Transformation Contracts

### 4.1 Raw -> Normalized session

这一步只允许做：

- 时区标准化
- bar start / end convention 对齐
- session 内外判定
- canonical minute indexing

这一步不允许做：

- split adjustment
- 缺失 minute 补齐
- 删除异常后伪装为完整

### 4.2 Normalized session -> Adjusted session

这一步只允许做：

- 基于 `split_reference_contract` 应用 split-consistent adjustment

这一步不允许做：

- dividend / total-return adjustment
- 人工修改价格异常

### 4.3 Adjusted minute -> Daily aggregates

这一步只允许做：

- deterministic aggregation
  - daily OHLCV
  - `bar_dollar_volume_d`
  - minute count

### 4.4 Daily / minute facts -> Symbol-day quality

这一步只允许做：

- 把结构完整性、session 合法性、adjustment 支持整合为 per symbol-day 的质量判定

### 4.5 Quality facts -> Audit outputs

这一步必须产出：

- `symbol_day_audit_table`
- `daily_breadth_summary`
- `symbol_coverage_summary`
- `spy_coverage_summary`
- `adjustment_support_summary`
- `sample_boundary_decision`

其 schema 以 [data_coverage_audit_outputs_spec_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/data_coverage_audit_outputs_spec_v1.md) 为准。

---

## 5. Canonical Keys and Replay Invariants

为了保证 replay，当前 v1 明确冻结以下键与不变量。

### 5.1 Identity invariants

- 内部主键优先使用 `security_id`
- `symbol` 只作为展示字段，不作为长期稳定 identity

### 5.2 Time invariants

- 交易日期一律以 `America/New_York` 确定
- 所有 canonical minute rows 必须映射到唯一 `(trading_date, minute_index)`

### 5.3 Session invariants

- full day 只能有 `390` 个 canonical minutes
- half day 只能有 `210` 个 canonical minutes
- 不允许 padding
- 不允许把 session 外 row 偷渡到 canonical session rows

### 5.4 Adjustment invariants

- features 依赖的所有 minute/daily 价格必须来自 split-consistent OHLCV
- labels 依赖的 open price 必须来自 split-adjusted, price-only daily open

### 5.5 Audit invariants

- 若 `adjustment_support_flag = false`
  - 不得把该 symbol-day 视为 `full_valid`
- 若 `observed_regular_minutes = 0` 且当天应交易
  - `symbol_day_status` 必须为 `missing`

---

## 6. Minimal Execution Sequence

真正执行 audit 时，最低顺序必须是：

1. 冻结 `security_master_contract`
2. 冻结 `exchange_calendar_contract`
3. 冻结 `split_reference_contract`
4. 读取 `raw_minute_bar_contract`
5. 生成 `normalized_session_minute_bars`
6. 生成 `adjusted_session_minute_bars`
7. 生成 `daily_bar_aggregates`
8. 生成 `symbol_day_quality_facts`
9. 生成 audit outputs
10. 生成 `sample_boundary_decision`

不允许跳过其中的 quality-facts 层，直接从原始 bars 手工拼 summary。

---

## 7. Relationship to Other Docs

这份文档和其他文档的关系是：

- [system_spec_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/system_spec_v1.md)
  - 定义研究层输入空间与 session / adjustment 原则
- [data_coverage_audit_protocol_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/data_coverage_audit_protocol_v1.md)
  - 定义 audit 必须检查什么、何时 GO / NO-GO
- [data_coverage_audit_outputs_spec_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/data_coverage_audit_outputs_spec_v1.md)
  - 定义 audit 输出物至少要长什么样

而本文定义的是：

- 为了让这些要求真正可执行，系统内部必须先具备哪些 canonical data objects
