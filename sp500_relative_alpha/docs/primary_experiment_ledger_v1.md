# Primary Experiment Ledger v1

**日期**: 2026-04-15  
**状态**: frozen / primary ledger for SP500RA-V1-R1  
**作用**: 定义 v1 的 primary hypothesis family、研究信用边界、以及什么变化会被视为“新增检验”

## 0. 为什么需要这份账本

`system_spec_v1.md` 冻结了研究定义，  
但要真正执行 anti-snooping 纪律，还需要一份更机械的账本：

- 哪些 cell 属于本轮 primary family
- 哪些比较允许在研究期内进行
- 哪些改动会消耗新的检验预算
- 什么条件下才允许查看 final holdout

没有这份账本，就会出现一种常见自欺：

- 名义上说“只是同一个实验”
- 实际上已经偷偷改了 horizon、model、feature family、cost gate 或 sample protocol

---

## 1. v1 的 primary family

当前 v1 的 primary family 明确定义为：

- universe：采用 [system_spec_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/system_spec_v1.md) 中冻结的 `S&P 500 proxy universe`
- raw input：daily `OHLCV` + `SPY`
- data snapshot：`local_equity_daily_20260415_v1`
- feature family：当前 allowlist 中全部 `52` 个严格可构造的 Alpha101 特征
  - 以 [alpha101_feature_admissibility_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/alpha101_feature_admissibility_v1.md) 为准
- label family：`benchmark-relative price excess return`
- execution alignment：`t` 收盘后出信号，`t+1` 开盘执行
- horizon set：`H in {5, 10, 15, ..., 60}`
- model family：`XGBoost` 与 `CatBoost`
- OOS protocol：expanding walk-forward + `60` trading-day purge gap + final holdout
- main metric：`OOS mean cross-sectional Rank IC`
- primary inference：HAC / Newey-West robust `t-stat` + block bootstrap CI
- economic gate：`top-25`, `equal-weight`, `long-only`, one-way `10 bps`

因此，v1 primary family 的大小为：

- `12 horizons x 2 model families = 24 primary cells`

---

## 2. 24 个 primary cells

| Cell ID | Horizon | Model | Feature Family | Status |
|---|---|---|---|---|
| `H05_XGB` | `5` | `XGBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H10_XGB` | `10` | `XGBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H15_XGB` | `15` | `XGBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H20_XGB` | `20` | `XGBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H25_XGB` | `25` | `XGBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H30_XGB` | `30` | `XGBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H35_XGB` | `35` | `XGBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H40_XGB` | `40` | `XGBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H45_XGB` | `45` | `XGBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H50_XGB` | `50` | `XGBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H55_XGB` | `55` | `XGBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H60_XGB` | `60` | `XGBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H05_CAT` | `5` | `CatBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H10_CAT` | `10` | `CatBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H15_CAT` | `15` | `CatBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H20_CAT` | `20` | `CatBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H25_CAT` | `25` | `CatBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H30_CAT` | `30` | `CatBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H35_CAT` | `35` | `CatBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H40_CAT` | `40` | `CatBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H45_CAT` | `45` | `CatBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H50_CAT` | `50` | `CatBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H55_CAT` | `55` | `CatBoost` | Fixed Alpha101 allowlist (`52`) | Primary |
| `H60_CAT` | `60` | `CatBoost` | Fixed Alpha101 allowlist (`52`) | Primary |

---

## 3. 哪些改动不算新增检验

以下变化，在不改变研究问题含义的前提下，可视为**同一 primary cell 内部的实现细化**：

- 同一模型家族内部、预注册范围内的常规超参训练
- 数值稳定性修复
- 缺失值处理的工程修复
- rolling operator 的实现优化
- 结果缓存、并行化、加速
- 日内到日频聚合的工程实现，只要不改变定义

前提是：

- 不改变输入空间
- 不改变标签定义
- 不改变 OOS 切分
- 不改变主 metric
- 不因为看了 OOS 结果才临时追加这些修改

---

## 4. 哪些改动算新增检验

以下任一变化，都必须被记为**新增 hypothesis / 新研究信用消耗**：

- 新增或替换 horizon
- 把 `skip-1w` label family 纳入比较
- 新增模型族
  - 例如 `LightGBM`、`linear model`、`MLP`
- 改变 feature family
  - 例如从 `52` 个 allowlist 中事后筛出新子集作为 primary
- 改变 cost model
  - 例如从线性 `10 bps` 改为 spread + impact
- 改变组合协议
  - 例如 `top-25` 改为 `top-10`、`top-50`、long-short、risk-neutral
- 改变主 metric
  - 例如把 `Rank IC` 换成 quantile spread 或 portfolio return
- 改变 OOS protocol
  - 例如测试块长度、purge 长度、holdout 位置
- 改变价格 / 标签口径
  - 例如从 split-adjusted price-only 改成 total return

这些变化不是“不允许”，而是：

- 必须被单独记录
- 必须承认研究信用被再次消耗
- 不能再伪装成原来的同一组结论

---

## 5. holdout 查看资格

只有满足以下条件的候选，才有资格进入 final holdout：

- 属于本文件定义的 `24` 个 primary cells 之一
- 研究期 aggregate `MeanRankIC > 0`
- 研究期主统计检验通过 `BH-FDR q=10%`
- 至少 `60%` 的 OOS 测试块 `MeanRankIC` 为正
- 简单预注册组合在研究期保留正的成本后 active return

如果某个结果不满足这些条件：

- 不允许以“我只是想看看”为理由提前查看 holdout

---

## 6. holdout 之后的纪律

一旦查看 final holdout：

- 不允许新增 horizon
- 不允许新增模型族
- 不允许新增 feature family
- 不允许依据 holdout 结果反向修改候选集合

如果发生上述修改，则：

- 当前 holdout 自动失效
- 必须声明这是新的研究轮次

---

## 7. 研究日志模板

正式实验开始后，每次定义层变更都要按以下格式记账：

| Date | Change ID | Layer | Change Summary | Reason | Pre/Post Holdout | Budget Impact |
|---|---|---|---|---|---|---|
| `YYYY-MM-DD` | `CHG-001` | `L2b` | `Example only` | `Example only` | `Pre-holdout` | `New hypothesis` |

字段解释：

- `Layer`
  - `L0-L4` 中受影响的层
- `Pre/Post Holdout`
  - 变更发生在查看 holdout 之前还是之后
- `Budget Impact`
  - `No new hypothesis`
  - `New hypothesis`
  - `Invalidates holdout`

---

## 8. 当前明确不纳入 primary family 的内容

以下内容可以未来研究，但当前**不属于** v1 primary family：

- `skip-1w` label family
- beta-adjusted residual label
- VWAP / trade-print / intraday 扩展 Alpha101
- PIT S&P 500 重构
- 更复杂的 benchmark / counterfactual family
- 更复杂的交易成本模型
- 更复杂的组合优化器

这些内容的地位是：

- 明确 deferred
- 不是默许存在的自由改动空间
