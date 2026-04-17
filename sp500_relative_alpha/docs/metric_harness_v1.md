# Metric Harness v1

**日期**: 2026-04-16  
**状态**: implementation-facing protocol  
**作用**: 把 v1 主 metric `OOS mean cross-sectional Rank IC` 映射为可测试代码对象

## 1. Scope

本 harness 负责：

- 对每个 `signal_date, horizon` 计算 cross-sectional Rank IC
- 在 fold OOS test block 内生成 daily Rank IC panel
- 聚合 fold-level 与 horizon-level `MeanRankIC`
- 计算 Newey-West / HAC robust `t-stat`
- 计算 deterministic moving-block bootstrap confidence interval

本 harness 不负责：

- 模型训练
- 特征选择
- multiple-testing / BH-FDR 判定
- 组合收益 / economic gate
- final holdout 查看

## 2. Daily Rank IC

对每个 `signal_date = t` 与 `horizon = H`：

`RankIC_t = SpearmanCorr(score(i,t), y(i,t;H))`

实现规则：

- 先对当日横截面 `score` 做 average-rank
- 再对当日横截面 label 做 average-rank
- 对两个 rank 向量计算 Pearson correlation
- 若横截面有效样本数低于 `min_cross_section_size`，输出 `NaN`
- 若 score 或 label 横截面常数，输出 `NaN`

实现：

- [metrics.py](/Users/hsy/Work/Invest/sp500_relative_alpha/metrics.py)

## 3. OOS Panel

输入：

- research dataset
- frozen walk-forward folds
- 预测分数列，默认列名 `score`

输出：

- `fold_id`
- `horizon`
- `signal_date`
- `rank_ic`
- `n_obs`

只使用每个 fold 的 `test` block。

## 4. HAC / Newey-West Inference

对 daily `RankIC_t` 序列计算：

- `mean_rank_ic`
- `hac_se`
- `hac_t_stat`
- one-sided / two-sided normal-approx p-value

默认 lag 规则：

`lag = floor(4 * (T / 100)^(2/9))`

并截断到 `[0, T-1]`。

说明：

- 这是实现层默认规则
- 它不改变主 metric
- 若之后决定 horizon-specific lag，例如直接用 `H` 或 `H/5`，必须登记为评估协议变更

## 5. Block Bootstrap CI

默认 bootstrap 规则：

- moving-block bootstrap
- deterministic seed：`20260416`
- default block length：`round(sqrt(T))`
- default iterations：`1000`
- default CI：`95%`

说明：

- bootstrap CI 用于辅助判断噪音带
- primary multiple-testing 判定仍以后续 ledger harness 统一处理

## 6. Current Discipline

当前只允许在以下输入上验证 metric harness：

- synthetic / toy samples
- dummy predictions
- plumbing tests

在 model training harness 和 multiple-testing harness 完成前，不应把真实 Alpha101/XGBoost/CatBoost 分数接进来计算正式 OOS Rank IC。
在当前阶段，即使 modeling harness 已经具备 dummy / baseline predictor plumbing，也仍不应接入真实 `XGBoost / CatBoost` primary experiment。
