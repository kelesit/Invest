# Data Coverage Audit Protocol v1

**日期**: 2026-04-15  
**状态**: superseded for current daily v1 / retained for historical 1-minute path  
**作用**: 定义在冻结 `Sample Boundary` 之前，v1 必须完成的数据覆盖审计、通过标准、输出物与 stop/go 规则

## 0. 这份文档回答什么

它回答的不是：

- 最后该用哪个数据供应商
- 实现代码应该怎么写
- 模型应该怎么训练

它只回答一件事：

> 在当前 v1 研究定义下，什么样的数据覆盖与完整性检查通过后，  
> 我们才有资格把样本起止区间、研究期和 final holdout 一次性写死。

如果没有这份协议，就会出现一种很危险的做法：

- 先大概看看数据
- 感觉差不多就挑一个起始日期
- 跑完再根据结果调整样本边界

这会直接污染 OOS 信用。

---

## 1. Audit Scope

注意：

- 本文件是原 `1-minute OHLCV` 方案的 coverage audit 协议
- 当前 v1 已改为 daily `OHLCV`
- 因此本文件不再是当前 round1 的主线执行协议

原 1-minute 方案的数据覆盖审计范围为：

- 证券集合：
  - 当前 `S&P 500` 普通股 proxy constituents
  - `SPY`
- 原始数据粒度：
  - regular-session `1-minute OHLCV`
- 时间口径：
  - `America/New_York`
- session 协议：
  - 正常交易日：`09:30-16:00`
  - 半日市：`09:30-13:00`

当前 audit 不负责：

- PIT membership 重构
- quote / trade-level 审计
- 执行成本校准
- Alpha101 特征正确性验证

---

## 2. Canonical Audit Units

为了避免“看上去差不多”，审计必须先把对象定义清楚。

### 2.1 Trading day

一个 `trading day` 必须先被归类为：

- `full day`
- `half day`
- `market closed`

只有完成这一步，才知道当天理论上应该有多少分钟 bar。

### 2.2 Expected minute count

在当前 v1 协议下：

- `full day` 期望 `390` 个 regular-session 1-minute bars
- `half day` 期望 `210` 个 regular-session 1-minute bars

### 2.3 Symbol-day 状态

对每个证券 `i`、每个交易日 `t`，审计必须把该 `symbol-day` 归为四类之一：

- `full_valid`
  - 所有预期 minute bars 都存在，且字段结构合法
- `partial`
  - 存在部分 minute bars，但未达到完整 regular session
- `missing`
  - 该交易日没有可用 regular-session bars
- `not_expected`
  - 该日不应有交易，例如市场休市

### 2.4 Full-valid 的最低结构要求

一个 `symbol-day` 若要被视为 `full_valid`，至少必须满足：

- minute bar 数量等于该日的理论值
- 无重复时间戳
- `open/high/low/close > 0`
- `volume >= 0`
- 对每个 minute bar：
  - `low <= min(open, close, high)`
  - `high >= max(open, close, low)`

补充说明：

- 原始数据源可以包含盘前 / 盘后 bars
- 这本身**不构成 blocker**
- 真正的 blocker 是：
  - 无法把 regular session 无歧义切出来
  - 或 session 外 bars 污染了 canonical regular-session minute rows

当前协议明确：

- 不允许通过补齐伪造 bar 把 `partial` 修成 `full_valid`
- 不允许通过丢弃异常 minute 再把该日视作完整

---

## 3. Mandatory Audit Checks

### 3.1 Universe manifest check

必须先冻结本轮 proxy universe 名单：

- 当前 constituent ticker list
- 每个 symbol 的 security identity 映射
- 哪些 symbol 被纳入，哪些被排除，以及排除理由

如果连这一步都没冻结：

- 后续所有 coverage ratio 都不可解释

### 3.2 Exchange calendar / session check

必须先建立并冻结 audit 使用的交易日历：

- 哪些日子是 `full day`
- 哪些日子是 `half day`
- 哪些日子是 `market closed`

并要求：

- 所有后续 minute-count 检查都以这份日历为准

### 3.3 SPY continuity check

由于 label 是相对 `SPY` 构造的，因此 `SPY` 不是普通一只证券，而是 benchmark anchor。

必须检查：

- `SPY` 在候选样本区间内的 regular-session minute data 是否连续
- `SPY` 是否存在缺失交易日
- `SPY` 的 split-adjusted open 序列是否可用来构造全部 label horizon

如果 `SPY` 本身不连续：

- 当前样本边界不得冻结

### 3.4 Symbol-day structural validity check

对每个 constituent symbol-day，必须统计：

- `full_valid` 天数
- `partial` 天数
- `missing` 天数
- 结构错误的具体原因

例如：

- 缺分钟
- 重复时间戳
- 非法价格
- 非法 high/low 关系
- session 外 bar 混入

### 3.5 Adjustment support check

当前 v1 使用：

- features：split-consistent OHLCV
- labels：split-adjusted, price-only open-to-open returns

因此必须检查：

- 候选样本区间内是否具备足够的拆股调整支持
- 拆股前后价格与成交量口径能否保持连续

如果只能拿到未调整价格而无法可靠重建 split-consistent OHLCV：

- 当前 v1 不得启动正式实验

### 3.6 Derived-sample sufficiency check

冻结样本边界时，不能只看“原始 bar 有没有”，还必须看：

- 训练窗是否足够长
- 研究期 OOS 块是否足够多
- final holdout 是否仍然完整

当前 v1 最低要求：

- 最小训练窗：`4` 年
- 单个测试块：`6` 个月
- final holdout：最后 `2` 年
- 研究期内至少应有 `4` 个 OOS 测试块

因此 raw sample 的最低理论长度必须至少支持：

- `4` 年训练
- `2` 年研究期 OOS
- `2` 年 final holdout

也就是：

- 至少约 `8` 年的可用交易日跨度

若达不到：

- 不能硬跑当前 v1
- 必须回到 preregistration 修改 protocol，而不是偷偷缩短

---

## 4. Aggregated Breadth Metrics

### 4.1 Daily breadth

对每个交易日 `t`，定义：

`breadth_t = (# of constituents with full_valid symbol-day at t) / (# of proxy constituents)`

这个量衡量的是：

- 当天横截面到底有多宽

### 4.2 Symbol coverage ratio

对每个证券 `i`，在候选样本区间内定义：

`coverage_i = (# of full_valid trading days for i) / (# of expected trading days)`

这个量衡量的是：

- 单只股票在整个样本内到底覆盖得有多连续

### 4.3 Benchmark coverage ratio

对 `SPY` 定义：

`coverage_SPY = (# of full_valid trading days for SPY) / (# of expected trading days)`

这是 benchmark 锚点的连续性指标。

---

## 5. Stop / Go Thresholds

当前 v1 冻结 `Sample Boundary` 前，必须同时满足以下门槛。

### 5.1 SPY benchmark gate

- `coverage_SPY >= 99.5%`
- final holdout 区间内 `SPY` 不允许存在缺失交易日
- 不允许存在连续 `2` 个及以上应交易日的 `SPY missing`

### 5.2 Cross-sectional breadth gate

在候选冻结样本区间内：

- `median(breadth_t) >= 90%`
- `5th percentile(breadth_t) >= 80%`

解释：

- 我们不是要求每天都完美
- 但也不允许横截面宽度长期塌缩

### 5.3 Symbol continuity gate

在候选冻结样本区间内：

- 至少 `80%` 的 constituent symbols 满足 `coverage_i >= 95%`

解释：

- v1 用的是当前 S&P 500 回填历史，本来就已带 proxy 偏差
- 如果再叠加大量个股分钟数据长期断裂，proxy 证据会进一步失真

### 5.4 Adjustment gate

- 候选冻结样本区间内，所有纳入 symbol 和 `SPY` 都必须具备可验证的 split-consistent price/volume adjustment support

### 5.5 Span sufficiency gate

- 候选冻结样本区间必须支持：
  - `4` 年最小训练窗
  - 至少 `4` 个研究期 OOS 测试块
  - 最后 `2` 年 final holdout

若以上任一门槛未通过：

- 结论必须是 `NO-GO`
- 不允许因为“先跑跑看”而继续

---

## 6. Sample Boundary Freeze Rule

样本边界的冻结步骤必须是机械的。

### 6.1 Candidate end date

候选样本终点定义为：

- 数据冻结时点之前，最后一个满足 benchmark continuity 与 adjustment support 的交易日

### 6.2 Candidate start date

候选样本起点必须定义为：

- 从终点向前回扫后，最早一个使整个冻结区间同时满足第 `5` 节所有门槛的日期

这意味着：

- 起点不是“想取多早就多早”
- 而是由覆盖质量反推出来的最早可接受日期

### 6.3 Holdout placement

一旦原始冻结样本 `[T_start, T_end]` 被确定：

- `final holdout` = 最后 `2` 年
- `research period` = holdout 之前的全部样本

随后再检查：

- 研究期是否仍能支持 `4` 年最小训练窗与至少 `4` 个 OOS 测试块

若不能：

- 该候选样本边界作废

### 6.4 No manual cherry-picking

明确禁止：

- 因为结果更好看而把起点后移
- 因为想多拿样本而把起点前移到低质区间
- 因为 holdout 不好而重新定义终点

任何此类修改都属于：

- 事后重定义样本
- 需要新一轮 preregistration

---

## 7. Required Audit Outputs

在 `Sample Boundary` 冻结前，审计至少必须产出以下对象：

这些输出物的逻辑 schema 以：

- [data_coverage_audit_outputs_spec_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/data_coverage_audit_outputs_spec_v1.md)

为准。

- `universe_proxy_manifest`
  - 本轮 constituent symbol 列表与映射说明
- `trading_calendar_manifest`
  - full day / half day / market closed 标记
- `symbol_day_audit_table`
  - 每个 symbol-day 的状态与失败原因
- `daily_breadth_summary`
  - 每日 breadth 序列与分位数摘要
- `symbol_coverage_summary`
  - 每个 symbol 的 `coverage_i`
- `spy_coverage_summary`
  - `SPY` 的连续性与缺失摘要
- `adjustment_support_summary`
  - split-consistent adjustment 支持情况摘要
- `sample_boundary_decision`
  - 最终 `T_start / T_end / research period / holdout` 的冻结决定及理由

没有这些输出物，就不应声称 audit 已完成。

---

## 8. Pass / Fail Interpretation

### 8.1 GO

只有当第 `5` 节全部通过，且第 `7` 节输出物完整时，才可宣布：

- `GO: sample boundary may be frozen`

### 8.2 NO-GO

以下任一情形成立，必须宣布：

- `NO-GO: current v1 sample boundary cannot be frozen`

典型情形包括：

- `SPY` continuity 不达标
- 横截面 breadth 长期不足
- split-adjustment support 不足
- 理论样本跨度不足以支撑当前 protocol

### 8.3 NO-GO 之后允许做什么

允许：

- 诊断数据问题
- 更换数据源
- 重新做 coverage audit
- 在新一轮 preregistration 中修改 protocol

不允许：

- 绕过 audit 直接开始训练
- 口头承认数据有问题，但仍继续跑 primary cells

---

## 9. Relationship to Registration

这份文档与：

- [preregistration_round1_sp500_proxy_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/preregistration_round1_sp500_proxy_v1.md)

的关系是：

- preregistration 定义“实验是什么”
- 本文定义“什么条件下才有资格冻结样本边界并启动实验”

因此：

- prereg 先行
- audit 再冻结 sample boundary
- sample boundary 冻结后，round 1 才能从 `Draft` 进入 `Frozen`
