# Multiple-Testing Ledger Execution v1

**日期**: 2026-04-16  
**状态**: implementation-facing protocol  
**作用**: 把 frozen primary ledger 中的 `24` 个 primary cells 与 `BH-FDR q=10%` 候选门槛机械化

## 1. Scope

本 harness 负责：

- 生成 frozen `12 horizons x 2 model families = 24` primary cell registry
- 对完整 primary family 执行 Benjamini-Hochberg FDR
- 计算每个 cell 的 fold stability gate
- 合并 economic gate 输入
- 输出 `research-usable candidate` 判定

本 harness 不负责：

- 训练 `XGBoost / CatBoost`
- 计算 Rank IC
- 计算 portfolio active return
- 查看 final holdout

## 2. Primary Registry

当前 registry 固定为：

- horizons：`5, 10, 15, ..., 60`
- model families：`XGBoost`, `CatBoost`
- feature family：`Alpha101_OHLCV_allowlist_52`

Cell ID 规则：

- `H05_XGB`
- `H10_XGB`
- ...
- `H60_XGB`
- `H05_CAT`
- ...
- `H60_CAT`

实现：

- [multiple_testing.py](/Users/hsy/Work/Invest/sp500_relative_alpha/multiple_testing.py)

## 3. BH-FDR Rule

输入必须包含完整 `24` 个 primary cells。

对于每个 cell：

- 使用 research-period aggregate `MeanRankIC`
- 使用 one-sided p-value，当前默认列名 `p_value_one_sided`

Benjamini-Hochberg 控制：

- `q = 10%`
- 分母固定为完整 primary family 的 `24`
- p-value 为 `NaN` 的 cell 不通过，但仍占 primary family 位置

这防止一种常见作弊：

- 只把看起来好的 horizon / model 交给 FDR
- 把失败或未跑通的 cell 从分母里消失掉

## 4. Research Candidate Gates

一个 cell 只有同时满足以下条件，才可标记为：

`research_usable_candidate = True`

条件：

- `MeanRankIC > 0`
- `BH-FDR q=10%` passed
- 至少 `60%` 的 OOS folds `MeanRankIC > 0`
- economic gate 可评估
- `cost_adjusted_active_return > 0`

如果没有 economic summary：

- 可以标记 `statistical_candidate = True`
- 但不能标记 `research_usable_candidate = True`
- `candidate_status = pending_economic_gate`

这是刻意设计：统计门槛通过不等于可以看 final holdout。

## 5. Candidate Status

输出状态包括：

- `research_usable_candidate`
- `failed_mean_rank_ic`
- `failed_fdr`
- `failed_fold_stability`
- `pending_economic_gate`
- `failed_economic_gate`

这些状态的优先级固定，不能根据结果事后重排。

## 6. Discipline

在以下内容完成前，不允许执行 primary experiment：

- economic gate harness
- end-to-end synthetic plumbing test
- final pre-run checklist

本 harness 当前只证明：

- primary family 分母固定
- FDR 判定可重放
- candidate gate 不会因为缺少 economic gate 而提前放行
