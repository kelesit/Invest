# 项目：Alpha101 + XGBoost / CatBoost

## 1. 目标 & 结束条件

目标：
1. 建一条独立于 `equity/` 主线的并行实验线
2. 用 Alpha101 风格公式因子库作为特征基准
3. 比较 XGBoost 与 CatBoost 在纯量价截面任务上的 OOS 表现
4. 判断这条“公式因子工厂 + booster”路线值不值得进入主生产线

结束条件：
1. 得到一份可信的 OOS 对比：`equity/` 手工特征线 vs Alpha101 boosting 线
2. 如果 Alpha101 boosting 在同口径评估下没有稳定增量，则这条线转入“已死分支”
3. 如果它有稳定增量，则进入下一阶段：CPCV + 更完整 factor library + benchmark 明确化

## 2. 主问题 & 假设树

- [ ] Alpha101 风格特征库是否比当前手工 25 特征提供更高的 OOS IC / ICIR？
- [ ] XGBoost 是否比 CatBoost 更适合当前这种全数值、低缺失、截面 rank 后的特征矩阵？
- [ ] Kaggle 式 tabular boosting 经验能否迁移到股票横截面任务，还是只在竞赛标签上有效？

## 3. 工作信念（项目局部）

- Alpha101 的价值首先是“宽特征工厂”，不一定是单个公式本身多强
- 在纯量价任务里，模型增量更可能来自非线性交互，而不是单个新信号
- 如果 benchmark 都说不清楚，这个项目很容易滑向“看起来很像在做 ML，实际上在漂移”
- 如果拿不到 PIT `cap` / `IndClass` / corporate actions，就不能把结果叫成“完整 Alpha101 复现”

## 4. 下一步

1. 先补 `exp_001`，明确：
   - 对标哪一个 Kaggle competition
   - 为什么它和我们的任务可比
   - 主指标到底是 IC、ICIR、top-bottom spread，还是组合收益
2. 先完成系统级定义文档：
   - [system_spec_v1.md](/Users/hsy/Work/Invest/research/proj_alpha101_boosting/system_spec_v1.md)
   - 在这份 spec 定稿前，不把当前实现当成正式方案
3. 先固定 strict 数据口径：
   - 如何覆盖 `S&P 500` 成员，而不是只覆盖 Nasdaq slice
   - regular session 还是 extended hours
   - raw return、relative return 还是 residual return
4. 在 strict panel 上跑第一版 XGBoost baseline
5. 在同一切分上跑 CatBoost

## 5. 已死的分支

暂无
