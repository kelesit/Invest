# 股票截面多因子选股 — 项目进展

**日期**: 2026-04-08
**阶段**: 项目五（股票截面多因子），特征研究 + 信号诊断阶段

---

## 一、项目目标

用纯量价动量特征 + LightGBM 在 S&P 500 中预测未来 10 天 beta-adjusted 残差收益率，选出 top 10 股票等权持有，目标是 SPY 指数增强。

## 二、已完成的工作

### 2.1 基础 Pipeline（v0-v1）

- 接入 S&P 500 全部 ~500 只成分股的日线 OHLCV（2015-2026），SPY 作为 benchmark
- 标签: 未来 10 天 beta-adjusted 残差收益率（rolling 252 天 beta）
- Purged Time-Series CV（5 折，purge gap = 10 天，embargo = 5 天）
- LightGBM 回归 + early stopping
- 回测引擎: Top-N long-only，10 天调仓，10bps 交易成本

### 2.2 特征体系演进（v1 → v2）

**v1**（39 个特征）模型完全不学习（early stopping 在 iteration 1 停止）。问题诊断:
- 特征高度冗余（多窗口动量相关性 0.9+）
- 双重 rank 变换（特征 rank + 标签 rank）压缩了信息
- 模型预测全部集中在 ~0.5，无区分力

**v2**（25 个特征，5 组）彻底重构:

| 组别 | 数量 | 假说 |
|------|------|------|
| 基础动量 | 5 | 不同窗口的过去收益，谁最近更强 |
| Skip 动量 | 3 | 去掉近期反转噪音，可延续的趋势 |
| 路径质量 | 6 | 效率比、涨跌比、收益集中度、回撤 |
| 风险调整动量 | 4 | return/vol、slope/RMSE |
| 量价交互 | 7 | 异常成交量、量价确认/背离 |

关键设计决策:
- 标签用原始残差收益（不 rank），保留收益幅度信息
- 特征只做一次 cross-sectional rank，不做 sector neutralize
- 不加入行业等非量价信息，确保评估的是纯量价因子的预测力

### 2.3 逐步调优（v2.1 → v2.4）

| 版本 | 改动 | Mean IC | 关键发现 |
|------|------|---------|----------|
| v2.0 | 基础 18 特征 | ~0.005 | 模型能学习了，但信号弱 |
| v2.1 | +7 量价交互特征 | ~0.011 | Top 20/30 开始跑赢 SPY |
| v2.2 | 去掉 sector 特征 | 0.014 | 纯量价因子本身有效 |
| v2.3 | min_child_samples 100→20 | 0.018 | 模型不再过度保守 |
| v2.4 | train_days 500→750 | 0.028 | 更多 fold 能训练出有意义的树 |

排除的尝试:
- 市场环境特征（波动率/离散度/动量）作为模型输入 → IC 从 +0.018 下降到 -0.007，模型过拟合时序噪音
- 标签时间尺度扫描（5/10/20/40/60d）→ 10d 的 IC/ICIR 最优

### 2.4 深度诊断（v2.5）

**IC vs 市场条件分析:**
- SPY 20 日波动率与 IC 的相关性 = +0.616（p < 0.001）
- 高波动时 Mean IC = +0.061，低波动时 Mean IC = -0.004
- 截面离散度也正相关（+0.134），SPY 动量无显著关系
- 结论: 信号是 regime-dependent，高波动/高离散度时有效

**Per-fold 拆解:**

| Fold | 日期 | Mean IC | Q5-Q1 Resid | Q5-Q1 Raw | Top-10 Excess |
|------|------|---------|-------------|-----------|---------------|
| 1 | 2018-07 → 2018-10 | +0.016 | +0.0040 | +0.0034 | 正 |
| 2 | 2020-05 → 2020-08 | +0.098 | +0.0282 | +0.0282 | 强正 |
| 3 | 2022-03 → 2022-06 | +0.081 | NaN | NaN | 负（-18%） |
| 4 | 2024-02 → 2024-05 | -0.010 | 负 | 负 | 负 |
| 5 | 2025-12 → 2026-03 | -0.027 | 负 | 负 | 负 |

- IC > 0: 3/5 窗口
- Q5-Q1 residual > 0: 2/5 窗口
- Residual/Raw 同方向: 5/5（排除了 beta 帮忙的假说）

**Fold 3 根因诊断:**
- Best iteration = 1（只有 1 棵树），预测值仅 28 个唯一值
- 原因: validation set（训练集末尾 10%）恰逢 2021 年底 regime 突变
- Val loss 从第 1 轮起单调上升 → early stopping 判断"不学比学好"
- 暴露了 Purged CV 与 Early Stopping 的结构性矛盾

## 三、当前状态评估

**有什么:**
- 纯量价动量在 S&P 500 截面上有可检测的 alpha 信号
- 信号在高波动/高离散度环境下有效（IC ≈ +0.06）
- Residual 和 raw return 排序方向一致，说明不是 beta 伪装

**没有什么:**
- 信号不稳定，5 个 OOS 窗口中只有 2 个 Q5-Q1 > 0
- 低波动环境（2024-2026）信号完全失效
- Validation set 有结构性缺陷，Fold 3 的评估不可信
- Long-only 组合缺少 beta/行业暴露控制

**结论: 纯量价动量有继续研究的价值，但还不能作为成熟的 SPY 增强模型。**

## 四、已知缺陷（待修复）

1. **Validation set 构造**: 取训练集末尾 10%，无 purge gap，受 regime shift 影响导致 early stopping 不可靠。需要在 train/val 之间加 purge gap，或改用固定轮数。

2. **组合暴露控制**: 当前 long-only Top-N 组合没有约束 beta/行业暴露。即使信号排序正确，组合仍可能因暴露不当跑输 SPY（Fold 3 就是这种情况）。

3. **Regime 分析的统计严谨性**: IC vs 市场条件分析基于 rolling window，存在自相关，p-value 偏乐观。不能直接当作 trading rule。

。

## 六、代码结构

```
equity/
├── data.py          # 数据获取: S&P 500 成分股 + SPY
├── features.py      # 特征工程: 5 组 25 个量价特征
├── labels.py        # 标签构造: beta-adjusted 残差收益
├── model.py         # LightGBM + Purged Time-Series CV
├── backtest.py      # Top-N long-only 回测引擎
├── analysis.py      # IC/分层/权益曲线/feature importance
└── insight_from_gpt.md  # GPT 特征设计建议

notebooks/
└── equity_momentum.ipynb  # 主研究 notebook（13 个 section）

output/
├── equity_ic_analysis.png
├── equity_quantile_analysis.png
├── equity_curve.png
├── equity_feature_importance.png
├── equity_ic_vs_regime.png
└── equity_per_fold_diagnostic.png
```
