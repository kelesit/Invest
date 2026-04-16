# Preregistration Template v1

**日期**: 2026-04-15  
**状态**: draft / experiment registration template  
**作用**: 在正式跑研究前，把 v1 的关键定义、样本边界、比较对象和成功条件写成一份可签字的实验注册表

## 0. 使用规则

这份模板不是为了“显得严谨”，而是为了在看结果之前先写清楚：

- 我们打算跑什么
- 为什么这算同一个研究问题
- 什么结果算支持
- 什么结果算失败

填写原则：

- 未填写项不得默认视为“以后再说”
- 若某项会影响结论解释，必须在开跑前写明
- 若开跑后修改关键项，必须登记为新轮次

---

## 1. Experiment Header

- Experiment ID：
- Registration Date：
- Owner：
- Status：
  - `Draft`
  - `Frozen`
  - `Executed`
  - `Invalidated`

---

## 2. Claim

- Empirical claim：
- Methodological claim：
- Null-result claim：
- Falsification form：

---

## 3. Frozen Definitions

- Universe：
- Universe caveat：
  - `current S&P 500 backfilled history proxy`
- Raw input：
  - `daily OHLCV + SPY`
- Feature family：
  - `strictly constructible Alpha101 subset`
- Label：
  - `benchmark-relative price excess return`
- Execution alignment：
  - `signal at t close, execute at t+1 open`
- Horizon set：
  - `5, 10, 15, ..., 60`
- Model family：
  - `XGBoost / CatBoost`
- Main metric：
  - `OOS mean cross-sectional Rank IC`
- Economic gate：
  - `top-25 equal-weight long-only`
- Cost model：
  - `one-way 10 bps linear turnover cost`

---

## 4. Sample Boundary

- Raw sample start：
- Raw sample end：
- Research period start：
- Research period end：
- Final holdout start：
- Final holdout end：
- Minimum training window：
  - `4 years`
- Test block length：
  - `6 months`
- Purge gap：
  - `60 trading days`

---

## 5. Primary Family

- Primary ledger reference：
  - [primary_experiment_ledger_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/primary_experiment_ledger_v1.md)
- Eligible primary cells：
  - `24`
- Multiple testing protocol：
  - `Benjamini-Hochberg FDR`
- FDR level：
  - `q = 10%`

---

## 6. Success Criteria

- Research candidate gate：
  - `MeanRankIC > 0`
  - passes `BH-FDR q=10%`
  - at least `60%` of OOS test folds positive
  - positive cost-adjusted active return vs `SPY`
- Final pass gate：
  - holdout `MeanRankIC > 0`
  - holdout cost-adjusted active return > `0`
  - direction does not flip

---

## 7. Failure / Inconclusive Criteria

- Failure：
  - no primary cell passes `BH-FDR q=10%`
  - research winner fails holdout sign or cost gate
- Inconclusive：
  - weak positive signals without stability
  - results depend on narrow time slices

---

## 8. Explicitly Deferred

- `skip-1w` labels
- beta-adjusted residual labels
- PIT membership reconstruction
- VWAP / trade-print Alpha101 expansion
- non-linear cost models
- alternative portfolio constructors

---

## 9. Sign-off

- Definition frozen by：
- Freeze timestamp：
- Holdout not yet viewed：
  - `Yes / No`
- Notes：
