# Alpha101 Boosting System Spec v1

**日期**: 2026-04-15  
**状态**: draft / canonical design doc for phase 0  
**范围**: 只定义实验目标、Universe、数据管道、调整策略、blocker  
**不包含**: 特征细节、标签细节、CPCV 细节、模型调参、组合优化细节

## 0. 这份文档的地位

这不是“实现说明”，而是**正式研究定义**。  
从这一版开始：

- `equity_alpha101/` 里的已有代码只算 prototype
- 只有和本 spec 一致的实现，才算正式方案
- 如果实现和 spec 冲突，以 spec 为准

本阶段不追求“尽快跑出结果”，只追求先把研究对象定义正确。

---

## 1. 研究目标与不做什么

### 1.1 研究目标

回答一个明确问题：

> 在一个**严格定义的美股日频股票截面任务**上，  
> Alpha101 风格的公式化量价因子子集，能否在 XGBoost / CatBoost 的非线性组合下，产生稳定的样本外横截面预测力？

这个问题有 4 个关键词：

- **严格定义**
  - Universe、数据口径、session、调整规则、评估协议都必须先定
- **美股日频股票截面**
  - 研究对象是股票横截面，不是单股票时序
- **Alpha101 风格因子子集**
  - 当前阶段不是完整 Alpha101 复现，而是严格字段下可实现的子集
- **样本外预测力**
  - 最终目标不是 in-sample 拟合，而是 honest OOS performance

### 1.2 本项目当前不做什么

当前 phase 明确不做：

- 不做“完整 Alpha101 复现”的宣传
- 不做近似 VWAP
- 不做 yfinance / Wikipedia / 临时 symbol scrape 驱动的正式研究
- 不做“先把模型跑起来再说”
- 不做未定义 universe 的全市场训练
- 不做 reference data 缺失时的静默降级

### 1.3 当前 phase 的成功标准

当前 phase 不以收益率为成功标准，而以**定义闭环**为成功标准：

1. Universe 被正式定义
2. Databento 数据口径被正式定义
3. 原始数据到 strict daily panel 的构建规则被正式定义
4. corporate actions / PIT reference 的约束被诚实写清
5. 后续 feature / label / CPCV 都能在这个定义上继续展开

---

## 2. Universe 与市场数据定义

### 2.1 Universe 是实验定义的核心

本项目接受下面这个原则：

> 如果 universe 没有被先定义清楚，后面的因子、标签、模型、组合都没有研究意义。

因此 universe 不是实现细节，而是一级研究对象。

### 2.2 理论上正确的 universe 定义

一个理论上正确的股票截面 universe，至少要定义：

- 市场范围：哪一个市场 / venue scope
- 证券类型：保留什么，排除什么
- 点时点原则：某日只能看到当日可见证券集合
- 纳入规则：价格、流动性、状态过滤
- 重构频率：daily / weekly / monthly
- benchmark 关系：研究对象与 benchmark 是否同一市场口径

### 2.3 本项目 v1 的正式目标 universe

**目标 universe（theoretical target）**：

- 市场：美国股票
- Phase 1 聚焦：**point-in-time S&P 500 common equities**
- benchmark 关系：**相对 `SPY` 定义大盘股横截面任务**
- 交易时段：**regular session only**
  - `09:30 - 16:00 America/New_York`
- Universe 类型：**point-in-time daily eligible universe**

选择 `S&P 500 common equities` 作为第一阶段正式目标的原因：

1. claim 明显收窄，更接近大盘股指数增强的真实任务
2. `SPY` 作为 benchmark 语义清晰
3. 大盘股流动性更高，更适合短持有期量价信号研究
4. 与项目中已有的 S&P 500 研究脉络更一致
5. 比“Nasdaq venue slice”更接近策略层真正关心的股票池定义

### 2.4 为什么当前还没有冻结单一 canonical raw dataset

#### `XNAS.ITCH`

问题：

- 数据质量高，也有 `trades`
- 但它只覆盖 Nasdaq venue scope
- 因此不能单独代表完整 `S&P 500` universe

结论：

- 可以作为 S&P 500 中 Nasdaq 成员的 prototype raw layer
- **不能单独作为本实验 canonical universe input layer**

#### `XNYS.PILLAR`

问题：

- 数据质量没问题，也有 `trades`
- 但它只覆盖 NYSE venue scope
- 也不能单独代表完整 `S&P 500` universe

结论：

- 可以作为 S&P 500 中 NYSE 成员的 prototype raw layer
- **不能单独作为本实验 canonical universe input layer**

#### `EQUS.SUMMARY`

问题：

- 有官方 `ohlcv-1d`
- 但没有 `trades`
- 因此不能从真实成交精确构造日内 `VWAP`

结论：

- 可以作为参考 daily bar / benchmark 对齐层
- **不能作为严格 Alpha101 输入层的唯一来源**

#### `EQUS.MINI`

问题：

- 有 `trades`
- 覆盖范围比单 venue 更接近真实大盘股样本
- 但它仍是 Databento 的产品 universe，不等于正式的 `S&P 500` 成员定义

结论：

- 可作为 prototype S&P 500 slice 的工程入口
- 当前不作为 v1 canonical universe engine

### 2.5 v1 universe inclusion / exclusion 规则

#### 2.5.1 理论目标规则

目标上应保留：

- point-in-time S&P 500 members
- common stocks
- active regular-way securities

目标上应排除：

- ETF
- ADR
- preferred shares
- warrants / rights
- SPAC units
- closed-end funds
- 其他非普通股证券

#### 2.5.2 当前实现 blocker

这套规则**当前不能完全严格落地**，因为：

- 需要 PIT security master / security type / classification
- 当前 Databento 账号对 `security_master` 返回 `403`

所以截至 **2026-04-15**，我们可以定义目标 universe，但还**不能声称已严格构建出该 universe**。

### 2.6 v1 流动性与价格过滤目标

正式研究阶段建议采用：

- `close >= $5`
- `ADV20 >= $10M`

解释：

- `$5` 是为了剔除低价股的极端 microstructure 噪音
- `$10M` 是为了让 Alpha101 这类短持有期价量因子更接近可交易环境

但这两个阈值的**计算前提**是：

- universe constituents 已经先确定
- `ADV20` 的 daily dollar volume 是严格计算的

因此 v1 只把这两个规则写成目标规则，不把它们误说成已经 fully operational。

### 2.7 Universe 重构频率

v1 正式目标：

- **monthly reconstitution**
- 但资格判断使用过去 20 交易日数据

原因：

- 比 daily universe churn 更稳
- 又不会像季度重构那样过于迟钝

### 2.8 当前诚实结论

截至 2026-04-15，本项目对 universe 的状态应表述为：

- **目标 universe 已定义**
- **PIT 严格实现仍被 reference data 权限卡住**

因此当前不能把任何 symbol list 叫作“正式 universe”。

---

## 3. 数据管道定义

### 3.1 数据管道的设计原则

本项目的数据管道必须满足：

1. 原始数据可追溯
2. daily panel 的每个字段有明确统计定义
3. 不使用 silent approximation
4. raw data 与 derived panel 分层存储

### 3.2 数据层级

v1 数据管道分 3 层：

#### Layer A: raw universe and market data

来源：

- PIT universe membership / security identity data
- Databento market data covering the chosen S&P 500 sample slice

需要的 schema：

- constituent membership / security identity source
- `trades` and/or `ohlcv-1m`
- `definition`

作用：

- universe membership source 用于回答“某天谁在 S&P 500 里”
- `trades` / `ohlcv-1m` 用于构造日频价量字段
- `definition` 用于 instrument metadata / status / symbol mapping

#### Layer B: exact daily panel

由 `trades` 精确聚合得到：

- `open`
- `high`
- `low`
- `close`
- `volume`
- `vwap`
- `dollar_volume`
- `trade_count`

定义：

- `open/high/low/close` 来自 regular session 成交序列
- `volume` 为 regular session 成交量总和
- `vwap = sum(price * size) / sum(size)`
- `dollar_volume = vwap * volume` 或等价的成交额总和

#### Layer C: research-ready panel

在 exact daily panel 之上派生：

- `returns`
- `adv{d}`
- 后续特征输入矩阵

这一层还不是本阶段实现重点，但其输入定义必须由 Layer B 保证。

### 3.3 session 定义

v1 canonical session：

- 时区：`America/New_York`
- 开始：`09:30`
- 结束：`16:00`
- extended hours：**全部排除**

原因：

- Alpha101 研究对象是日频股票 alpha
- 先锁 regular session 可最大化可比性
- 盘前盘后会引入额外流动性异质性和 venue 结构问题

### 3.4 daily bar 的正式定义

#### 理论正确

daily bar 应由真实 `trades` 聚合，而不是 vendor 预计算字段直接替代研究定义。

#### v1 正式定义

对每个 `symbol × session_date`：

- `open` = 当日 regular session 第一笔成交价
- `high` = 当日 regular session 最高成交价
- `low` = 当日 regular session 最低成交价
- `close` = 当日 regular session 最后一笔成交价
- `volume` = 当日 regular session 成交量求和
- `vwap` = `sum(price * size) / sum(size)`

### 3.5 symbol mapping 与 instrument lifecycle

v1 要求：

- 原始下载以 `raw_symbol` 为输入
- 实际存储时保留 Databento 的 `instrument_id`
- daily panel 主键使用 `(date, ticker)`，但必须保留 dataset / source metadata

后续需要进一步定义：

- ticker rename 如何处理
- symbol mapping 历史变更如何回溯

这一块当前还没有 fully designed，但必须留在数据层，不允许拖到模型层解决。

### 3.6 canonical 存储结构

建议正式结构：

```text
data/equity_alpha101/
  raw/
    market_data/
    universe_membership/
  exact/
    daily_panel/
  research/
    universe/
    labels/
    features/
```

本阶段只要求锁定结构，不要求全部实现。

### 3.7 当前账号下的已知能力边界

截至 2026-04-15，通过 live probe 已确认：

- 可访问：
  - `XNAS.ITCH trades`
  - `XNYS.PILLAR` / `XNAS.BASIC` / `EQUS.MINI` trade-like market data
  - `EQUS.SUMMARY ohlcv-1d`
- 不可访问：
  - `security_master`
  - `corporate_actions`
  - `adjustment_factors`

### 3.8 当前诚实结论

v1 数据管道当前可以正式推进到：

- user-supplied S&P 500 slice 的 strict market-data panel
- exact / minute-bar regular-session daily panel

但还不能正式推进到：

- PIT `S&P 500` constituent engine
- adjusted research-ready master panel

---

## 4. 调整策略与 blocker

### 4.1 理论上正确的调整策略

理论上，正式研究应同时维护两套价格体系：

#### raw panel

用于：

- 执行语义
- 原始成交层核对

#### adjusted panel

用于：

- return 计算
- 跨 split / dividend 的特征可比性
- 长样本历史研究

调整所需数据：

- corporate actions
- adjustment factors
- point-in-time 生效时点

### 4.2 当前状态

截至 2026-04-15：

- `corporate_actions`：403
- `adjustment_factors`：403

因此当前项目**不能声称拥有正式 adjusted panel**。

### 4.3 v1 正式策略

v1 采用下面这个硬规则：

> 在 reference data 未打通前，项目只允许构建 **raw exact panel**，  
> 不允许把 raw panel 冒充 adjusted research master。

这意味着：

- 可以继续推进数据层
- 可以继续定义后续标签层接口
- 但不能把长期 OOS 结果包装成 canonical 研究结论

### 4.4 关键 blocker 列表

#### Blocker A: PIT security master 缺失

影响：

- common stock filter 不可严格实现
- ETF / ADR / preferred / fund 排除规则不可严格实现
- `cap` 不可用
- `IndClass` 不可用

状态：

- 当前 Databento 账号 403

#### Blocker B: corporate actions / adjustment factors 缺失

影响：

- adjusted returns 不可严格构造
- Alpha101 中大量依赖 `close/returns/delay/delta` 的公式长期可比性受限

状态：

- 当前 Databento 账号 403

#### Blocker C: 请求预算有限

影响：

- 大样本历史请求会遇到 `402 account_insufficient_funds`
- 原型级验证可以做
- 正式全样本数据管道需要预算计划

#### Blocker D: benchmark 尚未冻结

说明：

- universe 已改为 `S&P 500`
- 因此 benchmark 目标方向已经明显收敛到 `SPY`
- 但 benchmark panel 的正式输入合同仍需在标签阶段单独冻结

当前状态：

- 目标方向已收敛为 `SPY`
- 但 benchmark panel 构造尚未 fully operational

### 4.5 v1 的项目结论

截至本 spec：

- 可以正式继续推进的只有两件事：
  1. 定义并实现 S&P 500 prototype slice 的 strict market-data pipeline
  2. 定义并实现 exact / minute regular-session daily panel

- 明确**不应该继续直接推进**的有三件事：
  1. 完整 Alpha101 因子库
  2. 正式标签研究
  3. 正式 OOS 绩效比较

原因不是代码不够，而是上游定义和 reference data 还没闭环。

---

## 5. 下一步边界

在本 spec 定稿后，下一阶段只允许做：

### Step 1

实现并固化：

- prototype S&P 500 slice 的原始下载规范
- exact daily panel 构建规范
- 原始层 / exact 层存储结构

### Step 2

在数据层定稿后，再单独写：

- `feature_library_spec_v1.md`

那时才讨论：

- Alpha101 子集范围
- 字段 contract
- 特征版本管理

在此之前，不进入标签 / CPCV / 模型 / 组合层。
