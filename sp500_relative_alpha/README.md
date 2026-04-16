# S&P 500 Relative Alpha

这是一个**完全独立于现有 `equity/`、`equity_alpha101/` 和旧 research 分支**的新项目目录。

它的目标不是“尽快跑一个模型”，而是先把一个严肃的大盘股相对选股系统的研究定义冻结下来，再决定实现。

## 当前状态

- 状态：`phase 1 / preregistration frozen, data snapshot frozen`
- 正式数据快照：`local_equity_daily_20260415_v1`
- 样本边界、研究期、pre-holdout purge、final holdout 已冻结
- 当前允许进入 L4 实现：daily feature builder、label generator、purged walk-forward evaluation harness
- 当前仍不允许：在评估 harness 与 preregistration 一致性测试完成前执行 primary experiment
- 任何查看 alpha OOS 结果后的定义修改，都会使当前注册失效
- 当前**不允许**把旧代码、旧数据源、旧目录结构当作约束条件反推研究定义

## 当前 v1 方向

- Universe：以 `point-in-time S&P 500 common equities` 为理论目标，
  但 v1 简化实现先使用**当前 S&P 500 成分股回填历史**的 proxy universe
- Raw material：daily `OHLCV` for constituents + `SPY`
- Feature library：所有能够由 daily `OHLCV` 及其确定性聚合物**严格构造**的 `Alpha101` 特征
- Label：相对 `SPY` 的 future `open-to-open` 价格超额收益，多 horizon：`5, 10, 15, ..., 60` 个交易日
- Model families：`XGBoost` 与 `CatBoost`
- Evaluation：尽可能严格的 OOS 协议

注意：

- 这是一个**日频截面研究任务**
- 主线时间协议是：`t` 日 regular-session 收盘后形成信号，`t+1` 日开盘执行
- 这里的 label 当前叫 **benchmark-relative price excess return**
- 只有在显式做了 `beta` 调整后，才应升级叫作 residual return
- 价格口径在 v1 采用 **split-adjusted, price-only**
- raw daily `volume` 是 inverse-split-adjusted shares volume；Alpha101 内部 `V` 冻结为 `((H + L + C) / 3) * adjusted_shares_volume`
- 因为原材料先冻结为 daily `OHLCV`，任何无法由 daily `OHLCV` 及其确定性聚合物严格推出的 Alpha101 特征都不进入 v1
- 当前成分股回填历史会引入 survivorship / membership look-ahead bias，因此 v1 的结论只能被解释为 proxy 证据，不是严格 PIT 证据

## 目录结构

- [docs/research_stack_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/research_stack_v1.md)
  新系统的顶层研究分层框架
- [docs/system_spec_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/system_spec_v1.md)
  当前项目的正式定义草案
- [docs/alpha101_feature_admissibility_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/alpha101_feature_admissibility_v1.md)
  Alpha101 特征的 v1 准入清单与 blocker 说明
- [docs/primary_experiment_ledger_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/primary_experiment_ledger_v1.md)
  v1 的 primary hypothesis family、holdout discipline 与研究信用边界
- [docs/evaluation_harness_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/evaluation_harness_v1.md)
  research dataset、purged expanding walk-forward folds 与 label-window leakage tests
- [docs/preregistration_template_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/preregistration_template_v1.md)
  正式实验前填写的预注册模板
- [docs/preregistration_round1_sp500_proxy_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/preregistration_round1_sp500_proxy_v1.md)
  第一轮实验的注册实例，冻结当前已确定定义并显式标注未冻结样本边界
- [docs/data_coverage_audit_protocol_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/data_coverage_audit_protocol_v1.md)
  原 `1-minute` 路径的数据覆盖审计协议，当前 daily v1 已不再以它为主线
- [docs/data_coverage_audit_outputs_spec_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/data_coverage_audit_outputs_spec_v1.md)
  原 `1-minute` 路径审计输出物的逻辑 schema，当前 daily v1 已不再以它为主线
- [docs/audit_data_contract_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/audit_data_contract_v1.md)
  原 `1-minute` 路径的 canonical 输入对象、中间对象与变换契约
- [daily_coverage_audit.py](/Users/hsy/Work/Invest/sp500_relative_alpha/daily_coverage_audit.py)
  当前 daily v1 主线的数据覆盖审计与样本边界候选冻结工具
- [data_snapshot.py](/Users/hsy/Work/Invest/sp500_relative_alpha/data_snapshot.py)
  正式数据快照 manifest、hash 指纹与快照校验工具
- [daily_data_loader.py](/Users/hsy/Work/Invest/sp500_relative_alpha/daily_data_loader.py)
  snapshot-locked daily OHLCV loader；加载前强制校验冻结 manifest hash
- [labels.py](/Users/hsy/Work/Invest/sp500_relative_alpha/labels.py)
  benchmark-relative `open-to-open` label generator
- [alpha101_ops.py](/Users/hsy/Work/Invest/sp500_relative_alpha/alpha101_ops.py)
  Alpha101 底层算子库与 canonical input matrix builder
- [alpha101_features.py](/Users/hsy/Work/Invest/sp500_relative_alpha/alpha101_features.py)
  已实现全部 `52` 个 daily `OHLCV` allowlist Alpha101 feature formulas
- [research_dataset.py](/Users/hsy/Work/Invest/sp500_relative_alpha/research_dataset.py)
  feature-label research sample builder；一行为 `symbol, signal_date, horizon`
- [folds.py](/Users/hsy/Work/Invest/sp500_relative_alpha/folds.py)
  purged expanding walk-forward fold generator 与 label-window leakage validator
- [artifacts/data_snapshots/local_equity_daily_20260415_v1/snapshot_summary.json](/Users/hsy/Work/Invest/sp500_relative_alpha/artifacts/data_snapshots/local_equity_daily_20260415_v1/snapshot_summary.json)
  Round 1 冻结数据快照摘要

## 工作原则

1. 先冻结研究 claim，再冻结研究对象、标签、输入、评估。
2. 先定义什么观察会让我们承认失败，再讨论如何实现。
3. 小 universe 可以接受，但 claim 必须收窄，不能偷换成“全市场结论”。
4. null result 也是结果。
5. 旧实现可以参考历史教训，但**不是**这个目录的上游依赖。

## 下一步

1. 实现 Rank IC metric harness
2. 实现 model training harness 的最小接口，但先只接 baseline / dummy model 做 plumbing test
3. 接入 multiple-testing ledger execution
4. 在 metric、training、ledger 自测完成前，不跑 primary experiment
