# Evaluation Harness v1

**日期**: 2026-04-16  
**状态**: implementation-facing protocol  
**作用**: 把已冻结的 research-period OOS、purge、final holdout 隔离规则映射为可测试的代码对象

## 1. Scope

本 harness 只负责：

- 构造 `feature + label` research dataset
- 构造 research-period expanding walk-forward folds
- 校验 fold 内 label window 不重叠
- 校验 research period 与 final holdout 的 label window 隔离

本 harness 不负责：

- 模型训练
- Rank IC 计算
- 组合收益计算
- holdout 结果查看

## 2. Dataset grain

样本表粒度冻结为：

`symbol, signal_date, horizon`

其中：

- `signal_date`：`t` 日收盘后形成信号
- `entry_date`：`t+1` 日开盘执行
- `exit_date`：`t+1+H` 日开盘退出
- `alpha_*`：在 `signal_date` 当天收盘及以前可构造的特征
- label：`benchmark_relative_open_to_open_return`

实现：

- [research_dataset.py](/Users/hsy/Work/Invest/sp500_relative_alpha/research_dataset.py)

## 3. Fold geometry

研究期内部使用：

- expanding walk-forward
- minimum train：`1008` trading days
- pre-test purge gap：`60` trading days
- test block：`126` trading days
- no separate post-test embargo

单个 fold 的结构：

`train -> gap -> test`

其中：

- `train` 为历史累计信号日
- `gap` 不进入训练、不进入测试
- `test` 为下一个 OOS block

下一个 fold 的 test start 按：

`previous_test_start + test_block_days + purge_gap_days`

推进。这与 frozen daily coverage audit 的 fold-count 逻辑保持一致。

实现：

- [folds.py](/Users/hsy/Work/Invest/sp500_relative_alpha/folds.py)

## 4. Leakage tests

每个 fold 必须满足：

`max(train.exit_date) < min(test.entry_date)`

这比简单比较 `signal_date` 更严格，因为它检查的是 label 实际消耗的未来价格路径。

final holdout 隔离必须满足：

`max(research.exit_date) < min(final_holdout.entry_date)`

因此：

- research period 末端与 final holdout 之间的 pre-holdout purge 不是装饰
- 它是为了防止 `H=60` 的 research label 尾部污染 holdout 起点

## 5. Round 1 smoke evidence

在 frozen snapshot 的 `AAPL, AMZN, GOOGL, MSFT, NVDA, SPY` 子集上：

- fold count：`5`
- first fold：
  - train：`2015-12-31 -> 2020-01-02`
  - gap：`2020-01-03 -> 2020-03-30`
  - test：`2020-03-31 -> 2020-09-28`
- last fold：
  - train：`2015-12-31 -> 2022-12-14`
  - gap：`2022-12-15 -> 2023-03-14`
  - test：`2023-03-15 -> 2023-09-13`

解释：

- research period 结束于 `2023-10-02`
- `2023-09-14 -> 2023-10-02` 不足一个完整 `126` trading-day test block
- 因此不会被强行拼成一个不符合注册规则的半块 OOS

## 6. Discipline

在以下内容实现并通过测试前，不允许执行 primary experiment：

- model training harness
- multiple-testing ledger execution
- baseline / economic gate harness

本文件当前只证明：

- 样本表能对齐
- OOS folds 能生成
- label window purge/isolation 规则能被机械验证
