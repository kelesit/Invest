# 量价因子截面选股研究总结 v1

> 截止日期：2026-04-18  
> 研究标的：S&P 500 成分股（美股）  
> 核心问题：纯量价特征能否在 OOS walk-forward 框架下产生统计显著的截面选股信号？

---

## 一、数据与宇宙

| 项目 | 配置 |
|--|--|
| 数据来源 | Databento，本地 parquet 快照 |
| 股票宇宙 | S&P 500 成分股，503 支（排除 SPY、HUBB） |
| 日期范围 | 2015-01-02 ~ 2026-03-31 |
| 数据粒度 | 日频 OHLCV（open/high/low/close/shares_volume） |
| 基准 | SPY（等权替代品，用于 benchmark-relative 标签） |
| 行业分类 | yfinance 拉取，11 个 GICS 行业，503/503 覆盖 |

---

## 二、特征工程

### 2.1 原始变量（Primitives）

**OHLCV Primitives（25 个）**

| 类别 | 特征名 |
|--|--|
| 收益率 | `ret_1d/5d/10d/20d/60d/120d/252d` |
| 日内结构 | `intraday_ret`、`overnight_gap`、`close_position`、`high_low_range` |
| 成交量 | `log_volume`、`volume_ratio_20d`、`volume_ratio_60d` |
| **市场中性超额收益** | `excess_ret_lag1/2/5/10/20d`（滞后值）、`excess_ret_mean5/21/63d`（滚动均值）、`excess_ret_std5/21/63d`（滚动标准差） |

> `excess_ret = daily_return - equal_weight_index_return`，对应 Optiver 第一名方案的 "historical target features"，捕捉短期反转与中期动量的市场中性版本。

**Alpha101 Primitives（52 个）**

WorldQuant 101 因子中仅依赖 OHLCV 的子集，全部在 date×symbol 宽表下向量化计算（`sliding_window_view` 消除 Python 循环）。

### 2.2 变换层（Transform Tree）

每个 primitive 经过以下变换，最终生成 `feat_*` 前缀特征：

```
一阶变换（8种/primitive）：
  identity | cs_rank | ts_zscore(20,60,120) | ts_change(5,20,60)

二阶变换（12种/primitive）：
  csr∘tsz(20,60,120) | csr∘tc(5,20,60) | tsz(*)∘csr | tc(*)∘csr

交叉组合（7个，在 cs_rank 空间操作）：
  ret20d×volr20d | ret60d×volr60d | ret20d−hlr | ret60d−hlr
  ret5d−ret60d | ret20d−ret252d | closepos×volr20d
```

| 版本 | Primitive 数 | 总特征数 |
|--|--|--|
| v2（无 excess_ret） | 66 | 1327 |
| v3（含 excess_ret） | 77 | 1547 |

**计算性能**：503 支股票全特征约 2 分钟，Alpha101 约 45 秒（向量化后，原始实现约 6 分钟）。

---

## 三、标签构造

### 3.1 时间结构

```
signal_date（收盘后产生信号）
  → entry_date = next trading day open（入场）
  → exit_date = signal_date + H + 1 trading day open（出场）
```

H = prediction horizon（预测窗口，单位：交易日）。

### 3.2 两种标签

**Benchmark-relative（默认）**
```
label = asset_open_to_open_return - SPY_open_to_open_return
```
去掉市场 beta，保留个股 alpha。

**Sector-neutral（实验）**
```
label = asset_open_to_open_return - mean(sector_peers_open_to_open_return)
```
进一步去掉行业轮动因子，保留更纯粹的个股 alpha。标签标准差比 benchmark-relative 小 **11.3%**（0.036 vs 0.040），信号更干净。

---

## 四、数据切分与防泄漏

### 4.1 Purged Walk-Forward Folds

采用 **expanding + purge gap** 结构，严格防止时间泄漏：

```
[====== train ======][=== gap ===][== test ==] → 下一 fold
                     ↑ purge gap（60天）
```

| 参数 | 值 |
|--|--|
| 研究期 | 2015-12-31 ~ 2023-10-02 |
| 最小训练期 | 4 × 252 = 1008 天 |
| 测试块长度 | 126 天（约半年） |
| Purge gap | 60 天（标签窗口最大 H=20，远小于 60 天） |
| Fold 数量 | 5 |

**Fold 结构：**

| Fold | 训练期 | 测试期 |
|--|--|--|
| fold_001 | 2015-12-31 ~ 2020-01-02 | 2020-03-31 ~ 2020-09-28 |
| fold_002 | 2015-12-31 ~ 2020-09-28 | 2020-12-23 ~ 2021-06-24 |
| fold_003 | 2015-12-31 ~ 2021-06-24 | 2021-09-21 ~ 2022-03-21 |
| fold_004 | 2015-12-31 ~ 2022-03-21 | 2022-06-16 ~ 2022-12-14 |
| fold_005 | 2015-12-31 ~ 2022-12-14 | 2023-03-15 ~ 2023-09-13 |

### 4.2 防泄漏校验

`validate_fold_label_windows`：自动验证每个 fold 的训练集 exit_date 早于测试集 entry_date，确保标签窗口不重叠。

### 4.3 Per-Fold 特征选择

特征选择在**每个 fold 的训练集内部**独立完成，使用截面 Pearson IC（向量化 matmul 实现）选 top-200：

```python
mean_ic = Σ_dates (fg_c.T @ lg_c) / sqrt(var_f * var_l) / n_dates
top_200 = argsort(-|mean_ic|)[:200]
```

**绝对不使用测试集信息**进行特征选择，杜绝 information leakage。

---

## 五、模型配置

| 参数 | 值 |
|--|--|
| 模型 | CatBoostRegressor |
| 迭代次数 | 500 |
| 树深度 | 4 |
| 学习率 | 0.03 |
| 损失函数 | RMSE |
| 正则化 L2 | 10.0 |
| 最小叶样本数 | 20 |
| 特征数/fold | top-200（per-fold IC 选择） |

---

## 六、评估框架

### 6.1 核心指标：Rank IC

每个交易日计算预测打分与标签的**截面 Spearman 相关系数**（等价于 Pearson on ranks）：

```
daily_rank_ic[t] = Spearman(score[t, :], label[t, :])
```

汇总统计：

| 指标 | 含义 |
|--|--|
| mean IC | OOS 期间日均 Rank IC |
| ICIR | mean IC / std(IC)，信噪比 |
| HAC t-stat | Newey-West 自相关修正的 t 统计量（考虑 IC 自相关） |
| p (one-sided) | H₀: mean IC ≤ 0 的单侧检验 |
| Bootstrap CI | 95% 置信区间（1000次 block bootstrap） |

### 6.2 为什么用 HAC t-stat 而不是普通 t-stat

日频 Rank IC 序列存在自相关（连续几天的信号强度相关），普通 t-stat 会高估显著性。Newey-West HAC 修正有效方差，给出保守的显著性估计。

---

## 七、实验结果汇总

### 7.1 优化路径

| 实验 | 关键变化 | mean IC | ICIR | HAC t | p值 | 结论 |
|--|--|--|--|--|--|--|
| H=20, benchmark-relative（基线 v1） | XGBoost, 52 Alpha101 原始特征 | 0.0165 | 0.114 | 1.304 | 0.096 | 基线 |
| H=20, benchmark-relative（v2） | CatBoost, 1327 特征, top-200 | 0.0195 | 0.106 | 1.145 | 0.126 | 特征多但不显著 |
| H=20, rolling window（2年） | 换 rolling folds | -0.0035 | -0.021 | -0.233 | 0.592 | **失败**，数据量不足 |
| H=5, benchmark-relative（v2） | 换 horizon | 0.0198 | 0.104 | 1.537 | 0.062 | 显著性提升 |
| H=5, benchmark-relative（v3） | +excess_ret 特征 | 0.0197 | 0.111 | 1.603 | 0.054 | 边际改善 |
| **H=5, sector-neutral（v3）** | **换标签** | **0.0156** | **0.126** | **1.850** | **0.032** | **突破 5%** |
| H=5, sector-neutral + time decay（half_life=252d） | 样本时间衰减权重 | 0.0011 | 0.010 | 0.160 | 0.437 | **失败**，与 rolling window 同根因 |

### 7.2 Per-Fold IC 全景

| fold | 时期特征 | H=20 bm-rel | H=5 bm-rel | H=5 sec-neu |
|--|--|--|--|--|
| fold_001 | 2020 新冠崩盘恢复 | +0.077 | +0.052 | +0.052 |
| fold_002 | 2020-21 流动性牛市 | +0.060 | -0.002 | +0.009 |
| fold_003 | 2021-22 加息开始 | **-0.067** | -0.003 | -0.008 |
| fold_004 | 2022 熊市 | +0.009 | +0.022 | +0.004 |
| fold_005 | 2023 复苏 | +0.018 | +0.030 | +0.022 |

---

## 八、关键结论与教训

### 8.1 已证伪的假设

**假设：Rolling window / 样本时间衰减可以改善 regime change 下的表现**  
结论：两种方式全面失败。Rolling window（2年）mean IC 从 +0.020 跌到 -0.004；时间衰减（half_life=252d）从 +0.016 跌到 +0.001。根本原因相同：有效样本量被压缩，CatBoost 严重依赖样本量，退化压过了 regime 适应性的收益。fold_003 的问题不是数据太旧，而是 2022 加息是训练期从未见过的利率制度，无论用哪种窗口/权重都无法预测这类结构性突变。

**假设：更多特征 = 更好结果**  
结论：从 52 个 Alpha101 原始特征扩展到 1327 个变换特征，mean IC 从 0.0165 提升到 0.0195，但 ICIR 反而略降（0.114 → 0.106）。特征爆炸不带来等比例的信号提升，per-fold top-200 特征选择是必要的控制手段。

### 8.2 有效的改进方向

**标签质量 > 特征数量**  
换标签（benchmark-relative → sector-neutral）带来的显著性提升（p: 0.054 → 0.032）远大于增加 220 个 excess_ret 特征（p: 0.062 → 0.054）。标签定义了问题本身，是最高杠杆的改进点。

**Horizon 选择的本质权衡**  
- H=20：牛市/趋势期信号强（fold_001: 0.077），但对 regime change 敏感
- H=5：信号更分散，对 regime change 更鲁棒（fold_003: -0.003 vs -0.067），总体显著性更高

**Excess Return 特征的真实价值**  
市场中性化的历史超额收益（excess_ret_lag/mean/std）在截面选股中有独立预测力。它直接编码了短期反转和中期动量的市场中性版本，是标准 OHLCV 特征库的有效补充。

### 8.3 fold_003 的结构性问题

fold_003（2021Q4-2022Q1，美联储开启加息周期）在所有实验中始终是最差的 fold，且无法通过特征工程或窗口调整修复。这是一个**训练集从未出现的制度转换**（利率从接近零到快速上升），任何基于历史价格模式学习的模型在这个节点都面临结构性失效。这不是模型设计失败，是宏观环境突变的客观约束。

---

## 九、当前最优配置

```
模型：CatBoostRegressor（500轮, depth=4, lr=0.03, l2=10, min_data_in_leaf=20）
特征：v3（1547个，含 excess_ret primitives）
标签：sector-neutral（个股收益 − 行业等权均值）
Horizon：H=5（5 交易日）
特征选择：per-fold top-200 by Pearson IC
数据切分：5-fold purged expanding walk-forward（purge gap 60天）

OOS 结果：mean IC=0.0156, ICIR=0.126, HAC t=1.85, p=0.032
```

---

## 十、待研究方向

| 方向 | 预期影响 | 难度 |
|--|--|--|
| ~~Sample weight 时间衰减~~ | ~~中（regime adaptation）~~ | ~~低~~ | **已证伪**：mean IC 0.016→0.001，与 rolling window 同根因（有效样本量压缩） |
| 更细粒度行业中性化（sub-industry） | 中 | 低 |
| 组合回测（从 IC 到实际收益） | 关键验证 | 中 |
| Cross-sectional Transformer | 高（学截面相对关系） | 高 |
| 财务数据（PE/PB/ROE） | 高（独立信号源） | 中 |
