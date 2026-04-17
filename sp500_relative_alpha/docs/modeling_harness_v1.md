# Modeling Harness v1

**日期**: 2026-04-16  
**状态**: implementation-facing plumbing protocol  
**作用**: 冻结 walk-forward model interface，先验证训练/预测管道的时间纪律，再接入真实 `XGBoost / CatBoost`

## 1. Scope

本 harness 负责：

- 对每个 fold 单独实例化 predictor
- 在 fold `train` block 上调用 `fit`
- 在 fold `test` block 上调用 `predict`
- 输出 OOS prediction frame，供 metric harness 消费

本 harness 当前不负责：

- `XGBoost` 训练
- `CatBoost` 训练
- 超参搜索
- feature selection
- multiple testing / BH-FDR
- economic gate

## 2. Interface Discipline

`fit(train_samples, feature_columns, label_column)`：

- 可以看到 train features
- 可以看到 train label
- 只允许使用当前 fold 的 train block

`predict(inference_samples, feature_columns)`：

- 只能看到 test metadata：
  - `signal_date`
  - `symbol`
  - `horizon`
- 只能看到 `alpha_*` feature columns
- 不能看到：
  - label column
  - `asset_open_to_open_return`
  - `benchmark_open_to_open_return`
  - `entry_date`
  - `exit_date`

这是为了保证模型接口本身不提供未来信息通道。

实现：

- [modeling.py](/Users/hsy/Work/Invest/sp500_relative_alpha/modeling.py)

## 3. Current Predictors

当前只实现 plumbing / baseline predictors：

- `ConstantPredictor`
  - 输出常数分数
  - 用于测试 metric 对 constant score 的处理
- `FeaturePassthroughPredictor`
  - 把某个 `alpha_*` 列直接作为 score
  - 用于 synthetic plumbing tests
- `SymbolMeanLabelPredictor`
  - 使用 train period 内每个 symbol 的平均 label
  - 只作为简单 baseline 接口测试，不是 primary model

这些都不是 v1 的 primary model family。

## 4. Output Contract

输出 prediction frame 包含：

- `fold_id`
- 原始 test sample columns
- `score`

注意：

- predictor 的 `predict` 阶段看不到 label
- harness 输出时会把 label 拼回 prediction frame
- 这样 metric harness 可以评估 OOS score，但模型本身没有 label leakage 通道

## 5. Discipline

在接入真实 `XGBoost / CatBoost` 前，必须先完成：

- economic gate harness
- end-to-end synthetic plumbing test

当前仍不允许执行 primary experiment。
