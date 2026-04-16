# Preregistration Round 1: S&P 500 Proxy Relative Alpha v1

**日期**: 2026-04-15  
**状态**: frozen / registered experiment instance  
**作用**: 作为本项目第一轮正式实验的注册实例，冻结研究定义、正式数据快照、样本边界与 holdout discipline

## 0. 当前地位

这份文档不是模板，而是：

- 当前项目第一轮实验的**注册实例**

当前状态：

- **Frozen**
- 研究定义已冻结
- 正式数据快照已冻结
- 样本边界、研究期、pre-holdout purge、final holdout 已写死
- final holdout 尚未查看
- primary experiment 尚未执行

---

## 1. Experiment Header

- Experiment ID：`SP500RA-V1-R1`
- Registration Date：`2026-04-15`
- Owner：`hsy`
- Status：`Frozen`

状态解释：

- `Draft`
  - 研究定义已写下，但尚未完成样本边界冻结
- `Frozen`
  - 样本边界、研究期、holdout 边界全部写死，且 holdout 尚未查看
- `Executed`
  - 正式按注册执行完 primary family
- `Invalidated`
  - 关键定义被事后修改，或 holdout 纪律被破坏

---

## 2. Claim

- Empirical claim：
  - 在以 `point-in-time S&P 500 common equities` 为理论目标、但在 v1 中以“当前 S&P 500 成分股回填历史”的 proxy universe 近似实现的横截面内，仅使用 daily `OHLCV` 及其确定性聚合物构造的严格可重算量价特征，能够对未来 `5-60` 个交易日的相对 `SPY` 价格超额收益形成稳定、可重复、样本外有效的排序预测。
- Methodological claim：
  - 若在实验开始前冻结 universe、label、OOS protocol、multiple testing protocol 与 holdout discipline，则得到的结论比“边看 OOS 边改定义”的研究流程更可信。
- Null-result claim：
  - 若该系统在严格协议下失败，则可接受的知识结论是：在本项目定义的 proxy universe、label、输入空间与评估协议下，纯量价 Alpha101 子集对未来相对 `SPY` 的价格超额收益不具备足够稳定、经济上可用的 OOS 预测力。
- Falsification form：
  - 若 `24` 个 primary cells 在研究期内无一通过 `BH-FDR q=10%`，或研究期候选进入 holdout 后 `MeanRankIC <= 0`，或成本后 active return <= `0`，则 empirical claim 被否定或显著收窄。

---

## 3. Frozen Definitions

- Universe：
  - 理论目标：`point-in-time S&P 500 common equities`
  - v1 proxy：`backfilled current S&P 500 common-equity constituents`
- Universe caveat：
  - 明确接受以下偏差：
  - survivorship bias
  - membership look-ahead bias
  - deleted / merged names undercoverage
  - 因此结论只能解释为 `proxy evidence`
- Raw input：
  - daily `OHLCV` for constituents + `SPY`
- Daily bar protocol：
  - one daily OHLCV row per symbol per trading day
  - no intraday bar completeness requirement in daily v1
- Price / volume adjustment：
  - features：split-consistent OHLCV
  - `open/high/low/close`：split-adjusted
  - raw `volume`：inverse-split-adjusted shares volume
  - Alpha101 internal `V`：`((high + low + close) / 3) * adjusted_shares_volume`
  - labels：split-adjusted, price-only `open-to-open`
- Feature family：
  - strictly constructible Alpha101 subset
  - allowlist size：`52`
  - blocker size：`49`
  - formal source：
    - [alpha101_feature_admissibility_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/alpha101_feature_admissibility_v1.md)
- Label：
  - `benchmark-relative price excess return`
  - `y(i,t;H) = r_open_to_open(i, t+1 -> t+1+H) - r_open_to_open(SPY, t+1 -> t+1+H)`
- Execution alignment：
  - signal at `t` regular-session close
  - execute at `t+1` regular-session open
- Deferred label families：
  - `skip-1w`
  - beta-adjusted residual labels
- Horizon set：
  - `H in {5, 10, 15, ..., 60}`
- Model family：
  - `XGBoost`
  - `CatBoost`
- Main metric：
  - `OOS mean cross-sectional Rank IC`
- Primary inference：
  - HAC / Newey-West robust `t-stat`
  - block bootstrap confidence interval
- Economic gate：
  - `top-25 equal-weight long-only`
- Cost model：
  - one-way `10 bps` linear turnover cost

---

## 4. Sample Boundary

当前状态：

- **Frozen**

正式数据快照：

- Snapshot ID：
  - `local_equity_daily_20260415_v1`
- Frozen cache path：
  - `/Users/hsy/Work/Invest/data/equity`
- Snapshot manifest：
  - [snapshot_manifest.csv](/Users/hsy/Work/Invest/sp500_relative_alpha/artifacts/data_snapshots/local_equity_daily_20260415_v1/snapshot_manifest.csv)
- Snapshot summary：
  - [snapshot_summary.json](/Users/hsy/Work/Invest/sp500_relative_alpha/artifacts/data_snapshots/local_equity_daily_20260415_v1/snapshot_summary.json)
- Daily coverage audit summary：
  - [daily_coverage_audit_summary.json](/Users/hsy/Work/Invest/sp500_relative_alpha/artifacts/data_snapshots/local_equity_daily_20260415_v1/daily_coverage_audit_summary.json)
- Snapshot manifest SHA256：
  - `7c410f9b2075e38a76d14017e556fc50a324e1b471966d8d0c7d3ea977753bf0`

正式样本边界：

- Raw sample start：
  - `2015-01-02`
- Raw sample end：
  - `2026-03-31`
- First feature signal date：
  - `2015-12-31`
- Last labelable signal date：
  - `2025-12-31`
- Research period start：
  - `2015-12-31`
- Research period end：
  - `2023-10-02`
- Pre-holdout purge start：
  - `2023-10-03`
- Pre-holdout purge end：
  - `2023-12-27`
- Final holdout start：
  - `2023-12-28`
- Final holdout end：
  - `2025-12-31`
- Minimum training window：
  - `4 years`
- Test block length：
  - `6 months`
- Purge gap：
  - `60 trading days`

冻结规则：

- 以上日期已经由当前正式数据快照的 daily coverage audit 写死
- 后续 primary experiment 必须使用该 snapshot manifest 所记录的文件集合与 hash
- 若任一 parquet 文件内容改变，必须生成新的 snapshot ID，并登记为新数据快照
- 若因为结果表现而修改样本边界、holdout 起止或数据快照，则当前注册自动 `Invalidated`
- final holdout 不得在研究期候选筛选前提前查看

### 4.1 Frozen Daily Coverage Audit Evidence

以下结果来自当前正式数据快照：

- Audit module：
  - [daily_coverage_audit.py](/Users/hsy/Work/Invest/sp500_relative_alpha/daily_coverage_audit.py)
- Audit status：
  - `Frozen evidence for SP500RA-V1-R1`

审计摘要：

- Decision：
  - `GO`
- Raw sample start：
  - `2015-01-02`
- Raw sample end：
  - `2026-03-31`
- First feature signal date：
  - `2015-12-31`
- Last labelable signal date：
  - `2025-12-31`
- Research period start：
  - `2015-12-31`
- Research period end：
  - `2023-10-02`
- Pre-holdout purge start：
  - `2023-10-03`
- Pre-holdout purge end：
  - `2023-12-27`
- Final holdout start：
  - `2023-12-28`
- Final holdout end：
  - `2025-12-31`

支持性统计：

- `SPY` calendar days：
  - `2827`
- labelable signal days：
  - `2515`
- research signal days：
  - `1951`
- final holdout signal days：
  - `504`
- research OOS folds supported：
  - `5`
- minimum signal-sample breadth ratio：
  - `0.9225`
- median signal-sample breadth ratio：
  - `0.9761`
- benchmark coverage ratio：
  - `1.0000`
- constituents：
  - `503`
- constituents passing `95%` coverage：
  - `466`
- constituent coverage-pass ratio：
  - `0.9264`

解释：

- `raw_sample_end` 晚于 `last_labelable_signal_date` 是刻意设计，不是错误
- `H_max = 60` 的标签需要未来 `t+1+H` open，因此最后约 `61` 个 raw trading days 只能用于补全标签，不能作为信号日
- 本节记录的是输入覆盖证据，不包含任何模型预测、Rank IC、组合收益或 alpha 表现

---

## 5. Primary Family

- Primary ledger reference：
  - [primary_experiment_ledger_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/primary_experiment_ledger_v1.md)
- Eligible primary cells：
  - `24`
- Cell definition：
  - `12 horizons x 2 model families`
- Multiple testing protocol：
  - `Benjamini-Hochberg FDR`
- FDR level：
  - `q = 10%`

不属于本轮 primary family 的内容：

- `skip-1w` labels
- alternative portfolio constructors
- non-linear cost models
- PIT S&P 500 reconstruction
- VWAP-enabled Alpha101 extensions

---

## 6. Success Criteria

- Research candidate gate：
  - research aggregate `MeanRankIC > 0`
  - primary test passes `BH-FDR q=10%`
  - at least `60%` of OOS test folds have positive `MeanRankIC`
  - simple preregistered portfolio has positive cost-adjusted active return vs `SPY`
- Final pass gate：
  - holdout `MeanRankIC > 0`
  - holdout cost-adjusted active return > `0`
  - result direction does not flip
  - holdout result does not collapse to near-zero noise

---

## 7. Failure / Inconclusive Criteria

- Failure：
  - no primary cell passes `BH-FDR q=10%`
  - research candidate fails holdout sign
  - research candidate fails holdout cost gate
  - result is obviously driven by a narrow time slice
- Inconclusive：
  - some weak positives exist, but fail stability or final confirmation
  - results do not justify promotion to “research-usable candidate”

---

## 8. Explicitly Deferred

- `skip-1w` labels
- beta-adjusted residual labels
- PIT membership reconstruction
- VWAP / trade-print / intraday Alpha101 expansion
- non-linear cost models
- alternative portfolio constructors
- richer benchmark / counterfactual family
- portfolio optimization beyond preregistered `top-25 equal-weight long-only`

---

## 9. Sign-off

- Definition frozen by：
  - `hsy`
- Freeze timestamp：
  - `2026-04-15`
- Holdout not yet viewed：
  - `Yes`
- Notes：
  - 研究定义、数据快照、样本边界已全部冻结。
  - primary experiment 尚未执行，final holdout 尚未查看。
