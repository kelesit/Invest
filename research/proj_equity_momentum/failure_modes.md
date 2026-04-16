# 旧 notebook 失败模式完整清单

**对象**: `notebooks/equity_momentum.ipynb` v2.5 + `equity/` 模块
**诊断日期**: 2026-04-14
**结论先行**: 这不是"信号弱但有救"的状态,而是**多个独立故障叠加形成的系统性问题**。在动新代码之前必须把这些故障逐个隔离,否则新框架只是把旧错误换个外壳重复一遍。

---

## 严重度 1: 会让任何结论失效的故障

### F1. 25 个特征里 21 个是负 IC——"momentum 库"实际上是 reversal 库

**证据**: cell 11 输出(单特征 IC 表)

```
ret_concentration_20d   +0.0104   ←
efficiency_60d          +0.0073   ← 仅 4 个特征 IC > 0
skip20_ret_120d         +0.0034   ←
abnormal_vol_20d        +0.0004   ←
ret_120d                -0.0004
abnormal_vol_5d         -0.0010
... (大量负的)
ret_5d                  -0.0155
ret_10d                 -0.0134
ret_20d                 -0.0172
ret_60d                 -0.0179
```

**意义**: 你设计的"基础动量族"(`ret_5d/10d/20d/60d/120d`)在 S&P 500 + 10 天 forward residual 这个特定组合下,**全部表现为反转,不是动量**。这不是 noise——这是统计上明确的负向关系。

**这是为什么**:
- 经典动量(Jegadeesh-Titman 1993)的 horizon 是 12 个月减 1 个月,即 J=12, K=1。10 天 forward 在 J-T 框架里属于"短期反转"区间,本来就该看到反转
- S&P 500 是大盘高效市场,残余动量本来就比小盘弱
- 你做的 beta-adjusted residual 进一步剥离了动量(因为高动量股往往 high beta,beta 调整后头部被压低)

**这意味着**: 你的"momentum 模型"其实是在让 LightGBM **逆向学习**这些负 IC 特征——靠 4 个正 IC 特征(都是路径质量类,不是方向类)拼凑出 mean IC ≈ 0.025。这是个**被特征污染严重稀释的弱反转信号**伪装成动量。

**修复方向**(三选一,不能并存):
- (a) 把 horizon 改回 60-120 天,让 momentum 真正出现
- (b) 接受 horizon 是 10 天,把整个特征库重新设计为 reversal 友好
- (c) 把那 21 个负 IC 特征整体反向(`feature → 1 - feature`),让符号对齐——但这是事后符号优化,要在另一段历史上验证

**优先验证**: 单独训一个只用那 4 个正 IC 特征的模型,看 mean IC 是否高于现在的 0.025。如果是,说明那 21 个负 IC 特征是噪声/拖累,不是信息源。

---

### F2. 预测值分布病态——大部分股票预测几乎相同,只有头尾 outliers

**证据**: cell 13 的 prediction 描述

```
count    147489
mean      0.000714
std       0.002174
25%       0.000312
50%       0.000776
75%       0.001229
min      -0.024204
max       0.101783
```

**问题**:
- 中位数 0.0008,IQR 仅 0.0009,但 max = 0.1018(中位数的 130 倍)
- 75% 的预测压在 0.0003 到 0.0012 这个极窄区间
- 模型对绝大多数股票实际上"没有意见",只对少数极端样本有非平凡的预测

**带来的连锁后果**:
- **Q5/Q1 看起来辉煌但是假象**:cell 17 的分位回测显示 Q1=-38%, Q5=+49%, Q5-Q1=87% 年化。但中间三档不是单调的——Q2=+27%, Q3=+4%, Q4=-5%。这是 **U/V 形,不是单调递增**。意思是模型只在分布尾部区分得开,中间完全没区分力。Q5-Q1 spread 漂亮只是因为两端的 outliers 自我对齐
- **qcut 在 Fold 3 直接崩**(cell 33):Fold 3 只有 28 个 unique 预测值,5 桶分不出来,Q5-Q1 = NaN
- **Top-N 选股 ≈ 选 outliers**:Top-10 永远是那几个 outlier 预测的股票,不是稳定排序的头部

**根因**: 学习率 0.05 + max_depth 6 + min_child_samples 20 + 大量负 IC / 噪声特征,模型早期收敛到"对大多数样本输出训练集均值",只在极端特征组合下偏离。这不是单一超参数问题,是**特征质量不够支撑这个模型容量**。

---

### F3. `nlargest` 在 ties 上按字母序选股——隐形 bug

**证据**: `equity/backtest.py:63`

```python
top_tickers = day_preds.nlargest(top_n).index.tolist()
```

**问题**: `pd.Series.nlargest` 在并列时按 index 顺序返回。Index 是 ticker 字母序,所以并列时永远选出 A、AAPL、ABBV、ABT...

**与 F2 的复合**: Fold 3 只有 28 个 unique 预测值 + 500 只股票 → 平均每个值 ~18 只股票并列 → Top-10 实际上是"并列最高分的前 10 只字母序股票"。

**Fold 3 的 Top-10 = -23.48%(cell 29)是这个 bug 的产物,不是模型预测错的产物**。我之前判断说"信号正确但组合构造失败",这个判断**是错的**。Fold 3 是模型完全没工作 + 字母序随机选股 + 2022 年 3 月加息暴跌的三重叠加。

**修复**: 在 nlargest 后加随机打散,或者直接用 prediction + 微小噪音排序(`day_preds + np.random.normal(0, 1e-10, ...)`)消除 ties。但这是表层修复——根本上要解决 F2 的 ties 来源。

---

## 严重度 2: 方法论层面的故障

### F4. Early stopping 的结构性失效(已知,但理解不够深)

**证据**: cell 32 的 train/val/test 还原 + 各 fold 的 best_iteration

| Fold | best_iter | pred_nunique |
|------|-----------|--------------|
| 1 | 43 | 20795 |
| 2 | 8 | 1320 |
| 3 | **1** | **28** |
| 4 | **2** | 117 |
| 5 | 24 | 4239 |

**根本问题不是 val 没加 purge**(那只是表层),而是 **early stopping 的整个前提在时序非平稳数据上不成立**:

Early stopping 假设 `min(val_loss) ≈ min(test_loss)`。这要求 val 和 test 同分布。在金融时序上,val 是 train 末尾的一个时间片段,test 是 val 之后的另一个时间片段——它们之间的相似度 = 那两段时间是不是同 regime,完全不可控。

Fold 3 的具体过程(cell 32 输出):
```
[1]  train l2: 0.00342  val l2: 0.00274
[2]  train l2: 0.00341  val l2: 0.00274007
...
[10] train l2: 0.00337  val l2: 0.00274299  ← val 单调上升
```

train_label_mean = +0.00106, val_label_mean = -0.00046——**符号不同**。任何能把训练集 fit 得更好的树都会让 val 更差,因为 val 的"对的方向"是反的。Early stopping 当然说"0/1 棵树最好"。

**修复方向**:
- (a) 完全移除 early stopping,用固定 num_iterations(通过 nested CV 在 train 内部选定)
- (b) 改用多 val 集成(把 train 切多个时间段,每段轮流当 val,平均 best_iter)
- (c) 完全不用 LightGBM 的 early stopping,改用线性模型/简单非参方法,避开复杂模型对 val 的依赖

### F5. 标签 horizon 是 in-sample IC 优化出来的

**证据**: progress_report v2.4

> 标签时间尺度扫描(5/10/20/40/60d) → 10d 的 IC/ICIR 最优

**问题**: 你扫了 5 个 horizon,在 OOS metric 上挑了最好的——这是 **multiple testing without correction**。即使每个 horizon 的真 IC 都是 0,扫 5 个挑最好,也会得到正的"最佳 IC"(数量级是 σ × √(2 log 5) ≈ 1.8σ)。

10d 之所以"看起来最好",可能是真的有效,也可能是 5 个 horizon 里 in-sample 抽到的 lucky one。诚实做法:
- 预先(在看数据前)选定一个理论上合理的 horizon,锁死
- 或者 ensemble 多个 horizon,对每个的预测求平均

### F6. 标签的 β 估计和应用不一致

**证据**: `equity/labels.py:54-105`

```python
beta = cov_daily / var_daily   # 用日度 return 估 β
residual = stock_fwd_10d - beta × spy_fwd_10d  # 应用到 10 天 forward
```

**问题**:
1. **β 时间尺度错配**:日度 β 不等于 10 天 β,只在收益严格 IID 时相等。实际收益有自相关,这是个理论近似(没在文档/代码里标出来,违反"必须区分理论与近似"的原则)
2. **β 估计噪声**:252 天滚动 β 本身有标准差,小盘股/高 vol 股的 β 更不稳定。这部分噪声直接进了标签
3. **只去 market,不去 sector/size/style**:残差仍包含巨量系统性暴露。这是 momentum 模型在 2022-03 应该死得更难看的原因之一(纯 momentum 偏 high-beta tech,2022 加息正好打这个暴露)

### F7. 5 折太少,任何"稳定性"判断都不可信

**证据**: cell 28 / 29

5 个 OOS 窗口,每个 60 天,合计 300 天测试。
- IC > 0 的窗口: 3/5
- Q5-Q1 spread > 0: 2/5
- Top-10 跑赢 SPY: 2/5

这个样本量做不出任何关于"信号是否稳定"的统计判断。3/5 vs 2/5 在 binomial(5, 0.5) 下完全不显著。"信号在高 vol 有效"这个判断基于 281 个日度 IC 数据点(更稳),但仍是 in-sample 发现。

**修复方向**: rolling-window CV 而不是固定 5 折。用 100+ 个滚动窗口计算 IC 分布,而不是 5 个固定段。这正是 CPCV 的设计动机。

---

## 严重度 3: 结构性问题(不是 bug 但限制了上限)

### F8. Top-10 太集中,即使有信号也被 idiosyncratic 噪声淹没

**理论**: 信号驱动的组合 alpha SNR ∝ √N。Mean IC = 0.025 对应的单股信号比 ~0.025,Top-10 的组合层信号比 ~0.025 × √10 ≈ 0.08,Top-50 ≈ 0.18。

**经验**: 学界用 Top decile(50-100 只)而不是 Top 10 是有原因的。你的"Top 10 vs Top 20 同方向所以信号稳"这个判断在 N=10 量级下没意义,因为两边都被噪声主导。

### F9. 把回归任务当排序任务训练,但用 MSE 损失

**证据**: `equity/model.py:21` `metric: "mse"`

**问题**: 你的 use case 是 cross-sectional ranking(选 top N),但训练目标是 MSE。MSE 对极端值高度敏感,而你的 label(残差收益)有重尾。模型会被几个极端样本主导,这正是 F2 的预测分布病态的另一个推手。

**正确做法**: 用 ranking loss(LambdaRank, pairwise loss)或者把 label 先做 Winsorization(把极端 1% 削平)再训。或者最简单——把 label 也 cross-sectional rank 后训练(但这又会回到 v1 的问题:模型预测全部集中在 0.5)。这是个不好平衡的工程问题,但当前的"raw residual + MSE"是已知次优。

### F10. 没有 sector / style neutralization 的组合层防线

**证据**: `equity/backtest.py:63` 直接 `nlargest(top_n)`,无暴露约束

**问题**: 即使信号正确,无约束 long-only Top-N 会自然集中到当时表现好的 sector / style。Fold 3 之所以即使没有 F3 bug 也可能赔钱,是因为纯 momentum 在 2022 年 3 月该买的"已经涨过的股票"恰好就是接下来跌得最狠的高 beta tech。

**解决路径**: 组合构造层加 sector neutralization(每个 sector 不超过 SPY 权重 ± k%)和 beta neutralization。这是后期组合优化模块的任务,当前不做也行,但要承认:**信号正确不等于 strategy 赚钱**,Top-N 直接持仓掩盖了组合层的失败。

---

## 故障的相互作用——为什么单修一个没用

```
F1(特征族错配 horizon) ─┐
                        ├─→ F2(预测分布病态) ─┐
F9(MSE 损失重尾)  ──────┘                    ├─→ Q5-Q1 = U 形
                                              │   Q5-Q1 spread 看起来高但假象
F4(early stopping 失效) ────→ Fold 3,4 退化 ─┤
                                              ├─→ F3(nlargest ties→字母序)
                                              │   Top-N 实际上在选噪声
                                              │
F7(5 折太少) ────────────→ 任何稳定性判断不可信
                                              │
F6(β 噪声) ──────────→ label 噪声放大 ───────┘
F10(无 sector neutralization) ─→ 即使信号对组合也容易死
```

**关键插入点**: F1 是上游,所有下游故障都被它放大。如果不解决 F1(特征/horizon 错配),修 F2-F10 任何一个都只是 cosmetic。但 F1 的修复需要重新设计特征体系,不是改个超参数。

---

## 我之前诊断里说错的地方(自我修正)

在这份完整诊断之前,我说过"Fold 3 是组合构造的失败,不是信号的失败,因为 IC=+0.079 但 Top-10 = -18%"。

**这个判断是错的**。看完 cell 33 我才意识到:

- Fold 3 的 best_iter = 1,只有 1 棵树,只有 28 个 unique 预测值
- IC=+0.079 是在 28 个值上算的,大量 ties,Spearman 用平均秩 → 极少数非并列样本可以驱动出虚高相关
- Top-10 在 28 个 unique 值上等价于"并列最高分里按字母序前 10 只"
- Top-10 = -23% 是 F3(nlargest ties bug)+ 2022 加息暴跌的产物,不是模型有效预测后的组合失败

正确的描述: **Fold 3 是模型彻底失效,IC 和 Top-N 都是 artifact**,不能用作"信号 vs 组合"的诊断对象。

---

## 修复的优先顺序(下次会话起点)

1. **先解决 F1**: 单独训只用 4 个正 IC 特征的模型,看 mean IC 是否反而提升。这个实验决定整个特征体系是否需要推倒重来
2. **接着解决 F2 + F3**: 任何修复都需要先消除 ties 病态。看是 model capacity 不够(特征贫瘠)还是 metric 选错(MSE)
3. **F4(early stopping)单独修**: 改成固定 iterations 或者 nested CV。这一步独立于上面两个,可以并行
4. **F5(horizon 选择)做敏感性分析**: 不要再 in-sample 挑,而是显式 ensemble 多个 horizon
5. **F6(β 噪声)的修复在更后面**: 现在更紧迫的是上游问题
6. **F7-F10 在新框架(eqxs/ + CPCV harness)里直接绕过**: 不在旧 notebook 里修

---

## 给 eqxs/ 新框架的设计约束(从这些故障反推)

- harness 必须**默认报告预测值的唯一值数量、IQR、tail 比例**——F2 那种病态分布不能再隐身
- harness 必须**强制要求 tie-breaking 策略显式声明**(随机 / 字母序 / 二级特征)——F3 那种隐形 bug 不能再发生
- harness 必须**报告每个 fold 的 best_iter / 模型容量指标**——F4 那种 1 棵树的退化要被立刻看到
- harness 必须**默认报告单特征 IC 表 + 模型 IC 表**,后者必须显著高于前者最大值才算"模型加了价值"——F1 那种"模型 IC ≈ 最强单特征 IC"的情况要被自动 flag
- harness 必须**支持 rolling-window CV 而不只是 fixed K-fold**——F7
- harness **不应**默认开 early stopping,如果开就必须报告 val/test regime distance——F4
