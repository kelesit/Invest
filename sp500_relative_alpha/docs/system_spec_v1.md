# S&P 500 Relative Alpha System Spec v1

**日期**: 2026-04-15  
**状态**: draft / canonical design doc for phase 0  
**阶段边界**: 只定义 `L0-L3`，不讨论实现

## 0. 这份文档的地位

这份文档是这个新目录的正式起点。

从现在开始：

- `equity/`、`equity_alpha101/` 和其他旧实验目录都不是这里的上游依赖
- 旧代码只能作为历史教训，不能作为研究定义的约束
- 本目录在 `L0-L3` 冻结前，不接受“先按现有条件做近似”这种倒推

这份 spec 的首要目标不是产出代码，而是冻结问题定义。

---

## 1. L0: 研究声明（Epistemic Claim）

### 1.1 empirical claim

本项目当前想检验的经验性主张是：

> 在一个以 `point-in-time S&P 500` 为理论目标、  
> 但在 v1 中以“当前 S&P 500 成分股回填历史”的 proxy universe 实现的横截面内，  
> 使用严格 point-in-time 的纯量价输入，  
> 可以对未来一段短中期持有期内、相对 `SPY` 的价格超额收益构造出稳定、可重复、样本外有效的预测排序。

注意：

- 这是一个**收窄后的 claim**
- 这是一个**proxy 版本的 claim**
- 它不是“全美股都存在这种规律”
- 它也不是“任何 universe 上都存在这种规律”
- 它也不是“严格 PIT S&P 500 上已经被证明成立”

### 1.2 methodological claim

本项目还隐含一个方法论主张：

> 如果严格冻结 universe、label、OOS 协议和 anti-snooping 纪律，  
> 那么得到的结论会比常见“边试边改、频繁偷看 OOS”的流程更可信。

### 1.3 null-result claim

如果系统在严格协议下失败，我们愿意接受如下结论也具有知识价值：

> 在本项目定义的 universe、输入约束、label 和评估协议下，  
> 纯量价特征对未来相对 `SPY` 的价格超额收益，不具备足够稳定和经济上有意义的 OOS 预测力。

### 1.4 可证伪形式

以下观察将否定 `1.1` 的经验性主张：

- 主 metric 在严格 OOS 下不能持续超过噪音带
- 结果只在少数时间片有效，跨时间不稳定
- 结果无法稳定超过预注册 baseline
- 经济结果在合理成本后消失
- 不同实现路径无法复现同方向结论

### 1.5 可重复性要求

v1 的最低可重复性要求：

- 跨多个时间切分成立
- 在相同原始数据与相同协议下可被独立重放
- 不要求一上来就跨 universe 成立

但更高标准下，未来应继续检查：

- 跨不同 large-cap sub-universe 是否仍成立
- 跨实现者是否仍成立

---

## 2. L1: 研究问题的精确形式

当前 v1 研究问题可以表述为：

> 在一个以 `S&P 500 common equities` 为目标、  
> 但在 v1 中用当前成分股回填历史来近似的 proxy universe 内，  
> 在严格的日频决策协议与未来信息隔离约束下，  
> 仅使用决策时点之前可得的 daily `OHLCV` 纯量价输入，  
> 构造对未来持有期内相对 `SPY` 价格超额收益的横截面预测分数，  
> 并在严格 OOS、purging、embargo、多重检验与最终留出检验下，  
> 判断这种预测是否达到预注册的统计与经济成功标准。

当前已经冻结到足以约束 v1 研究定义的部分包括：

- 主 metric：`OOS mean cross-sectional Rank IC`
- 经济 gate 的主线组合：`top-25`, `equal-weight`, `long-only`
- v1 成本模型：线性换手成本，单边 `10 bps`

当前仍保留为后续 protocol supplement 再细化的部分包括：

- 是否为不同 horizon 使用不同采样频率
- 是否在 v1 之后补充更强的 benchmark / counterfactual family

### 2.1 当前冻结的 v1 简化目标

为了让第一版以“简单但诚实”为主，当前额外冻结以下设计选择：

- 原始数据粒度：daily
- 原始字段：`OHLCV`
- 研究任务：日频横截面任务，而不是 minute-level 交易任务
- 特征库：所有能够由 daily `OHLCV` 及其确定性聚合物严格构造的 `Alpha101` 特征
- 模型族：`XGBoost` 与 `CatBoost`
- 标签族：相对 `SPY` 的多 horizon `open-to-open` 价格超额收益，`H in {5, 10, 15, ..., 60}`
- 主线时间协议：`t` 日收盘后形成信号，`t+1` 日开盘执行
- Alpha101 volume 口径：公式内部 `V = ((H + L + C) / 3) * adjusted_shares_volume`

这意味着：

- v1 不研究 intraday / quote / order book / tick-sequence alpha
- v1 不研究任何无法由 daily `OHLCV` 及其确定性聚合物严格推出的 Alpha101 特征
- v1 的核心目标不是“找单一最优 cell”，而是找出在严格 OOS 下**可用的** feature-label-model 组合

---

## 3. L2a: Universe 定义

### 3.1 理论目标

v1 universe 的正式目标是：

`point-in-time S&P 500 common equities`

### 3.1.1 v1 实际 proxy universe

为了让第一版以简单为主，v1 当前接受如下 proxy：

`backfilled current S&P 500 common-equity constituents`

也就是：

- 先取当前时点的 S&P 500 普通股成分名单
- 将这份名单回填到历史样本期
- 不额外做 PIT membership 重构

### 3.1.2 这个 proxy 的已知问题

这不是一个中性近似，而是会引入明确偏差：

- survivorship bias
- membership look-ahead bias
- 对退市、被剔除、并购标的覆盖不足
- 对历史真实指数换手与成分迁移刻画不足
- 结果可能高估稳定性与可投资性

因此：

- v1 结果只能解释为**proxy 证据**
- 不能把 v1 结果表述为“严格 PIT S&P 500 已被验证”

### 3.2 这句话的含义

它至少包含 5 层约束：

- `point-in-time`
  - 任意历史时点都应能重构当时真实指数成员
- `S&P 500`
  - 研究对象不是全美股，而是大盘股指数成分
- `common equities`
  - 排除 ETF、preferred、warrant、right、fund shares、其他非普通股证券
- `eligible`
  - 必须可正常进入研究与持仓候选
- `time-varying`
  - 成员资格随时间变化，必须被记录，而不是静态回填

### 3.3 survivorship 规则

正式原则：

- 退市、并购、剔除的历史成员必须保留在历史样本里
- 不允许用今天的成分股名单回填过去
- 任何成员变更都必须带时间戳

### 3.4 identity model

研究对象主键不能是显示 ticker 本身。

正式要求：

- 一个证券必须由可跨 ticker 变化保持连续的 security identity 跟踪
- ticker 只是一层展示映射

### 3.5 流动性与可交易性

v1 当前冻结为：

- 不额外施加流动性过滤
- 不额外施加价格过滤

理由：

- S&P 500 本身已经是高流动性大盘股集合
- v1 以简单为主，不再叠加第二层可交易性筛选

代价：

- 研究对象完全等同于该 proxy universe
- 不再区分“指数成员”与“最终可交易集合”

### 3.6 代表性声明

v1 结论只主张：

- 对美国大盘股指数成分内的横截面有效性作判断

不主张：

- 对全美股有效
- 对中小盘有效
- 对其他市场自动有效

---

## 4. L2b: Label 定义

### 4.1 当前 v1 label 名称

当前正式名称：

`benchmark-relative price excess return`

而不是：

`residual return`

### 4.2 数学定义

若持有期为 `H` 个交易日，则初版 label 可写为：

`y(i,t;H) = r_open_to_open(i, t+1 -> t+1+H) - r_open_to_open(SPY, t+1 -> t+1+H)`

其中：

- `r_open_to_open(i, t+1 -> t+1+H)` 是证券 `i` 从 `t+1` 日开盘到 `t+1+H` 日开盘的未来收益
- `r_open_to_open(SPY, t+1 -> t+1+H)` 是 `SPY` 在同区间的未来收益

### 4.2.1 执行对齐

当前主线协议正式冻结为：

- 信号形成：`t` 日 regular-session 收盘后
- 执行建仓：`t+1` 日 regular-session 开盘
- 标签起点：`t+1` 日 regular-session 开盘

示例：

- 若 `H = 5`，则收益区间为 `t+1 open -> t+6 open`

这一定义的目的，是保证 label 与真实可交易起点严格对齐。

### 4.3 为什么现在不叫 residual

因为严格的 residual 至少要求：

- 有显式市场暴露模型
- `beta` 估计只使用 `t` 以前数据
- 必要时还要考虑行业 / 风格暴露

因此：

- `v1` 先做 benchmark-relative price excess return
- `v2` 再考虑 beta-adjusted residual return

### 4.4 label 的经济意义

它对应一个非常清楚的现实问题：

> 在这个 S&P 500 proxy universe 里，哪些股票会在未来持有期内跑赢 `SPY`？

这是可解释的，也是可交易的。

### 4.5 当前未冻结项

当前仍未完全冻结的 label 细节只剩：

- 是否引入额外 skip period 变体
- 是否为不同 horizon 使用不同采样频率

主线 label 与 execution 的对齐方式已经冻结。

### 4.5.1 关于“1 周 skip”提议

这里需要明确一个逻辑关系：

- 当前已经冻结的主线协议是：`t` 收盘后出信号，`t+1` 开盘执行
- 在这个协议下，主线 label 必须从 `t+1 open` 开始

因此，“额外再跳过 1 周再开始算收益”不能与当前主线 label 同时成立；  
它会变成**另一套不同的 label family**。

若未来要研究这个变体，其形式应为：

`y_skip1w(i,t;H) = r_open_to_open(i, t+6 -> t+6+H) - r_open_to_open(SPY, t+6 -> t+6+H)`

其中这里把“1 周”解释为 `5` 个交易日。

当前决定：

- `skip-1w` 只记录为备选 label family
- 不作为 v1 主线 label 的一部分

### 4.6 当前 v1 的 horizon 集

当前正式冻结的预注册 horizon 集为：

`H in {5, 10, 15, ..., 60}`

原则：

- 这些 horizon 必须被视为一组**并行检验**
- 不能在跑完之后只挑“最好看”的一个，当作唯一结论
- 如果后续只保留少数 horizon 进入下一阶段，必须经过额外 OOS 或最终 holdout 再确认

### 4.7 关于标签重叠

这里要区分两件事：

- 不同 horizon 的标签，属于**不同研究问题 / 并行检验**
- 同一 horizon 下，如果按天形成样本，则相邻日期的标签区间会**彼此重叠**

以 `H = 5` 为例：

- `y(i,t;5)` 使用的是 `t+1 open -> t+6 open`
- `y(i,t+1;5)` 使用的是 `t+2 open -> t+7 open`

这两个区间共享了：

- `t+2 open -> t+6 open`

因此影响是：

- 相邻样本并非独立
- `RankIC_t` 的时间序列会出现自相关
- 训练集和测试集如果时间上离得太近，可能共享未来收益区间
- 有效样本量会小于“日期数”

所以“不同 label 是不同测试”这句话只对“不同 horizon”成立；  
对“同一 horizon 的相邻日期样本”并不成立，因为它们共享未来收益片段。

---

## 5. L2c: 输入空间定义

### 5.1 输入边界

v1 只研究纯量价输入。

允许的理论原始数据类型包括：

- daily `OHLCV`

当前不引入：

- 财务报表
- 分析师预期
- 新闻 / 公告 NLP
- 受限 IP 数据
- quote / order book / tick-sequence 数据

### 5.1.1 为什么原材料改为 daily OHLCV

当前从 `1-minute OHLCV` 改为 daily `OHLCV` 的理由是：

- `1-minute` 全 universe 长样本成本过高
- daily 数据成本低得多，更适合第一版快速完成完整研究闭环
- daily OHLCV 已足以支撑 Alpha101 中不依赖 `VWAP / cap / industry` 的大部分特征
- 与 `5-60` 交易日的标签尺度匹配

正式解释：

- v1 仍然是一个日频截面预测任务
- 放弃日内信息是成本约束下的主动收窄
- 因此 v1 不能声称研究了任何 intraday alpha

### 5.1.2 daily bar 口径

当前 v1 使用 daily OHLCV，因此不再审计分钟 bar 完整性。

daily bar 的最低要求是：

- 每个交易日每个 symbol 至多一条 daily OHLCV
- `open/high/low/close > 0`
- `volume >= 0`
- `low <= min(open, close, high)`
- `high >= max(open, close, low)`
- 价格口径必须满足 split-adjusted
- volume 必须与 split-adjusted price 形成 split-consistent 口径

如果数据源的 daily bar 是否包含 auction、是否来自 consolidated tape、是否有 corporate-action back-adjustment 存在差异，必须在数据源映射时记录。

### 5.2 point-in-time 要求

任一 `t` 时刻的 feature 值，只能依赖 `t` 之前真实可见的信息。

正式解释：

- 如果执行在 `t+1` 开盘，则 `t` 日收盘后才能确认的数据可以进入 feature
- 如果执行在 `t` 收盘，则 `t` 收盘成交本身是否可用必须单独定义

### 5.2.1 价格调整口径

v1 当前正式选择：

- `features` 使用 **split-consistent** 的 OHLCV 序列：
  - `open/high/low/close` 使用 split-adjusted prices
  - raw `volume` 使用与 split-adjusted price 一致的 inverse-split-adjusted shares volume
  - Alpha101 公式内部的 `V` 使用 `((high + low + close) / 3) * adjusted_shares_volume`
- `labels` 使用 **split-adjusted, price-only** 的 `open-to-open` 收益
- 不做 dividend / total-return 调整

理由：

- 不做 split adjustment，会让拆股/并股事件污染量价特征与收益标签
- split-consistent 的 price/volume 口径，是量价特征连续性的最低必要条件
- 用 typical-price dollar-volume proxy 作为 Alpha101 `V`，可以让 `V` 与 `ADV` 的单位一致
- total-return 更完整，但会把 v1 对企业分红数据的依赖显著提高

代价：

- v1 的 label 研究的是**价格超额收益**，不是总回报超额收益
- 分红事件会给部分样本带来额外噪音

因此 v1 的正式表述必须是：

- `benchmark-relative price excess return`

而不是：

- `total return alpha`

### 5.2.2 Alpha101 准入清单

v1 的 Alpha101 准入与 blocker，不在本节口头枚举，而是单独冻结在：

- [alpha101_feature_admissibility_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/alpha101_feature_admissibility_v1.md)

该文档是 v1 feature allowlist 的正式依据。

### 5.3 允许的变换族

v1 暂时只接受可独立复算、可解释、可 replay 的量价变换。

例如：

- rolling return / volatility
- turnover / volume-profile 类派生
- cross-sectional ranks
- 所有能够由 daily `OHLCV` 及其确定性聚合物严格构造的 Alpha101 风格公式化算子组合

### 5.3.1 Alpha101 子集的 v1 限制

当前 v1 的特征库不是“完整 Alpha101”，而是：

`strictly daily-OHLCV-constructible Alpha101 subset`

因此以下特征不进入 v1：

- 任何需要 `VWAP`、逐笔成交金额、成交笔数、或日内路径但无法由 daily `OHLCV` 严格推出的特征
- 任何依赖 `cap` 的特征
- 任何依赖 `industry classification / industry neutralization` 的特征

这不是暂时忘记，而是正式范围收窄。

### 5.4 禁止项

明确禁止：

- 任何显式或隐式使用未来信息的 feature
- 任何无法被第三方从同样原始数据重算的 feature
- 任何研究者主观临时拼凑、事后难以 replay 的特征定义

---

## 6. L2d: 评估协议定义

### 6.1 OOS 原则

正式要求：

- 必须有研究期内部的严格 OOS
- 必须有最终不可触碰的 final holdout
- final holdout 在系统规格冻结前不得作为调参依据

### 6.1.1 v1 的正式 OOS 切分方案

v1 当前冻结为：

- 研究期内部使用 **expanding walk-forward OOS**
- 最小训练窗：`4` 年
- 单个测试块长度：`6` 个月
- 最终 `final holdout`：样本最后 `2` 年
- `final holdout` 按 **labelable signal dates** 计算，而不是按 raw bar dates 计算

每个研究期 fold 的结构为：

- `train`
- `gap`
- `test`

其中：

- `train` 为历史累计样本
- `gap` 为 purge gap
- `test` 为下一个 `6` 个月测试块

选择这套协议的原因：

- expanding window 更贴近真实研究与部署流程
- `6` 个月测试块能提供多个 OOS 切片
- 最后 `2` 年 holdout 足够长，不只是偶然的一小段行情

### 6.1.1.1 raw sample dates vs labelable signal dates

daily v1 必须区分四类日期：

- `raw_sample_start / raw_sample_end`
  - 数据文件里实际存在的 daily OHLCV 起止日期
- `first_feature_signal_date`
  - 满足最大 feature lookback 后，第一天理论上可以形成完整特征的信号日
- `last_labelable_signal_date`
  - 对 `H_max = 60` 仍能完整计算未来 `t+1 open -> t+1+H open` 标签的最后信号日
- `final_holdout_start / final_holdout_end`
  - 在 labelable signal date 空间里切出的最终留出区间

因此：

- `raw_sample_end` 不等于 `last_labelable_signal_date`
- 若 `raw_sample_end = T_raw`，则 `last_labelable_signal_date` 必须至少向前留出 `H_max + 1` 个交易日
- 这是为了确保最后一个信号日的 `t+1+H` open 已经真实存在，而不是被未来数据补出来

### 6.1.1.2 final holdout 前的 purge

final holdout 不是普通的连续切片。  
在 `research period` 和 `final holdout` 之间，v1 同样保留：

- `pre-holdout purge gap = 60 trading days`

这段日期：

- 不进入研究期训练
- 不进入研究期 OOS 测试
- 不进入 final holdout

原因：

- research period 末端的长 horizon label 会与 holdout 起点附近的价格路径重叠
- 如果不 purge，最终留出区也会被研究期标签尾部污染

### 6.1.2 final holdout 规则

当前正式冻结：

- `final holdout` 完全不参与研究期模型比较与候选筛选
- 只有在研究期协议、候选组合、统计检验规则都冻结后，才允许查看 holdout 结果
- 一旦根据 holdout 结果修改系统定义，该 holdout 自动作废

### 6.2 时间结构约束

评估协议必须显式处理：

- label 重叠
- feature lookback 重叠
- 调仓频率与持有期耦合
- purge
- embargo

### 6.2.1 v1 的 purge / embargo 选择

由于 v1 同时研究：

- `H in {5, 10, 15, ..., 60}`

而主线标签按天形成样本，因此：

- 研究期内部统一使用 `H_max = 60` 个交易日作为 purge gap

正式定义：

- `purge gap = 60 trading days`

解释：

- 任一测试块开始前，向前留出 `60` 个交易日空档
- 这段空档既不进入训练，也不进入测试

### 6.2.2 为什么 v1 不单独再加 post-test embargo

理论上，在更一般的时间序列 CV 中：

- 若训练样本既取测试块之前，也取测试块之后，则还需要单独的 post-test embargo

但 v1 当前使用的是：

- **forward-only expanding walk-forward**

也就是训练集只取测试块之前的历史数据。

因此当前选择是：

- 使用 `60` 个交易日的 pre-test purge gap
- **不单独设置 post-test embargo**

这不是否认 embargo 的理论必要性，而是因为在当前 forward-only 结构下，它已经不再是额外独立步骤。

### 6.3 metric 架构

统计层与经济层必须分开。

正式原则：

- 主 metric 先衡量横截面排序能力
- 推断统计量单独定义，不能和 effect size 混为一谈
- 组合收益只作为经济 gate，不作为 v1 的主 metric

### 6.3.1 主 metric

v1 的主 metric 正式定义为：

`OOS mean cross-sectional Rank IC`

对每个调仓日 `t`，在当日 eligible universe `U_t` 上计算：

`RankIC_t = SpearmanCorr(score(i,t), y(i,t;H))`

其中：

- `score(i,t)` 是模型在 `t` 时刻对证券 `i` 给出的预测分数
- `y(i,t;H)` 是证券 `i` 在未来持有期 `H` 内相对 `SPY` 的价格超额收益 label
- `U_t` 是当日 v1 proxy 协议下的 eligible universe

然后在 OOS 调仓日序列上取时间平均：

`MeanRankIC = (1 / T) * sum_t RankIC_t`

### 6.3.2 为什么主 metric 选 Rank IC

原因不是“大家常用”，而是它最贴合当前 claim。

本项目当前 claim 是：

> 模型能否在 `S&P 500` 横截面里，把未来相对 `SPY` 表现更强的股票排得更靠前。

因此：

- 这是一个排序问题，而不是点预测精度问题
- 我们首先关心横截面顺序是否正确
- `Rank IC` 对量纲与极端值更稳健
- 它比直接看组合收益更少掺杂权重映射、成本模型和组合约束噪音

### 6.3.3 主推断统计量

主 metric 之外，必须单独冻结主推断统计量。

v1 当前冻结：

- `MeanRankIC` 的 HAC / Newey-West 稳健 `t-stat`
- `MeanRankIC` 的 block bootstrap 置信区间

原因：

- `RankIC_t` 时间序列可能存在自相关
- 标签重叠、持有期重叠、调仓频率设置都会影响独立性假设
- 因此不能用朴素 IID 假设下的标准误

### 6.3.4 经济 gate metric

虽然主 metric 不是组合收益，但经济层必须单独设 gate。

v1 的经济 gate 当前冻结为一个预注册、简单、可复现的 long-only 组合评估。

### 6.3.4.1 主线组合协议

当前正式冻结：

- 组合形式：`top-25`, `equal-weight`, `long-only`
- 每次调仓后，将当期预测分数最高的 `25` 只股票重置为等权
- 每只目标权重：
  - `1 / 25 = 4%`

因此：

- 当期进入 `top-25` 的股票，目标权重为 `4%`
- 其他股票目标权重为 `0`

选择 `top-25` 的原因：

- 对约 `500` 只股票的 universe 来说，`25` 只约等于前 `5%`
- 足够集中，能体现排序信号
- 又不会像 `top-5` 或 `top-10` 那样过度暴露于个股事件风险

### 6.3.4.2 经济 gate 必须评估的量

- top-k long-only 组合相对 `SPY` 的价格超额收益
- 成本后 active return
- information ratio
- 换手
- 回撤

正式原则：

- 统计上有预测力，不自动等于经济上值得交易
- 如果 `Rank IC` 为正，但简单预注册组合在合理成本后没有实质内容，则 empirical claim 需要收窄

### 6.3.4.3 换手定义

在第 `t` 次调仓时，定义：

- `w_pretrade(i,t)`：调仓前、经历价格漂移后的实际权重
- `w_target(i,t)`：调仓后目标权重

则正常调仓时的组合换手定义为：

`turnover_t = (1 / 2) * sum_i |w_target(i,t) - w_pretrade(i,t)|`

解释：

- 绝对权重变动衡量总交易量
- 乘 `1/2` 是为了避免把买入和卖出重复记成双倍

### 6.3.4.4 初次建仓

对第一次从现金进入组合的建仓日，当前单独定义：

- `turnover_0 = 100%`

理由：

- 从空仓建到满仓，经济上更符合“组合规模全部进入市场”的直觉
- 这比机械套用 `(1/2) * sum |Δw| = 50%` 更适合作为成本计算口径

### 6.3.4.5 成本模型

v1 当前正式冻结为线性换手成本模型：

`cost_t = c * turnover_t`

其中：

- 主线单边成本假设：`c = 10 bps = 0.10% = 0.001`

解释：

- 若某次调仓 `turnover_t = 20%`
- 则该次调仓成本为：
  - `0.001 * 0.20 = 0.0002 = 2 bps`

### 6.3.4.6 为什么 v1 使用简化成本模型

当前不直接上更复杂的冲击 / spread / auction execution 模型，原因是：

- 真实执行协议尚未细化到足以支撑更复杂的成本校准
- 当前输入数据不包含 quote / order book 层信息
- 复杂成本模型在 v1 阶段更容易形成伪精确

因此 v1 的成本模型目标不是精确复刻实盘，而是：

- 用一个透明、保守、可解释的成本门槛，判断信号在基本交易摩擦下是否仍有内容

### 6.3.5 次级统计 metric

除主 metric 外，可以记录但不能偷换为主 metric 的统计量包括：

- `ICIR`
- OOS hit rate
- quantile spread
- calendar subperiod stability

这些指标用于补充解释，不用于事后替代主 metric。

### 6.3.6 明确不作为主 metric 的指标

以下指标可以记录，但不应作为 v1 主 metric：

- `MSE`
- `RMSE`
- `R^2`
- 单一组合收益率曲线
- 单次 top-k 组合的最终累计收益

原因：

- `MSE / R^2` 更适合点预测误差，不贴合横截面排序 claim
- 纯组合收益会过度混入权重映射、成本假设和组合构建选择
- 单条收益曲线极容易被 path dependence 和少数股票偶然驱动

### 6.4 benchmark / counterfactual

v1 当前先冻结最简经济基线为：

- 被动持有 `SPY`

原因：

- 当前阶段优先回答“相对基准指数是否有增量”
- 更复杂的随机 / 线性 / naive 统计基线，留到下一轮评估协议细化时再冻结

因此本节当前不再扩展其他 baseline。

### 6.5 多重检验纪律

必须维护明确的 hypothesis ledger：

- 每新增一个 feature family
- 每新增一个 horizon
- 每新增一个模型族
- 每新增一个重要筛选规则

都要记作新的检验消耗。

对 v1 而言，最低限度应明确承认：

- `12` 个 horizon 已经构成一组并行检验
- `XGBoost` 与 `CatBoost` 是两条并行模型线

由于 v1 当前冻结为：

- 使用一份固定的、全部严格可构造的 Alpha101 特征库
- 不把 feature subset selection 当作 primary 比较轴

所以 v1 当前 primary family 的规模为：

- `12 horizons x 2 models = 24 primary cells`

### 6.5.1 v1 的正式 multiple testing 协议

当前冻结：

- 对 `24` 个 primary cells 的主统计检验结果进行 **Benjamini-Hochberg FDR** 控制
- 控制水平：`q = 10%`

解释：

- v1 是 discovery-oriented 第一版
- 目标是找到“研究上可用的候选组合”
- 后面还有 final holdout 作为第二道确认门

因此当前不使用更保守的 family-wise error 控制。

因此 v1 的正式输出不应是“最好看的单一结果”，而应是：

- 完整 horizon x model 结果面板
- 其中通过研究门槛的候选组合清单

### 6.5.2 holdout 阶段的纪律

研究期通过 FDR 的组合：

- 才有资格进入 final holdout

holdout 阶段禁止：

- 新增 horizon
- 新增模型族
- 新增 feature family
- 因为 holdout 表现而重定义候选集合

### 6.6 再现性要求

任何正式结论必须能够被：

- 同一人未来重放
- 另一实现者独立重放

如果只能得到“差不多”的结论，而不能得到同方向、同性质的结论，就不应视为稳固发现。

---

## 7. L2e: 成功与证伪条件

### 7.1 当前冻结的是“分层门槛”

虽然 v1 的主线成本模型与组合协议已经冻结，但当前仍不写过细的收益数值门槛。

但当前已经可以冻结：

- 研究候选门槛
- 最终通过门槛
- 证伪条件

### 7.2 成功的必要结构

系统若要被宣布“工作”，至少必须经过两层门槛。

### 7.2.1 第一层：研究期候选门槛

一个组合若要进入 final holdout，至少必须同时满足：

- 研究期 aggregate `MeanRankIC > 0`
- 研究期主统计检验在 `BH-FDR q=10%` 下通过
- 至少 `60%` 的 OOS 测试块 `MeanRankIC` 为正
- 相对被动 `SPY` 的简单预注册组合，在研究期 aggregate 后保留正的成本后 active return

满足以上条件的组合，称为：

- `research-usable candidate`

### 7.2.2 第二层：最终通过门槛

一个 `research-usable candidate` 若要被宣布为 v1 的“最终通过组合”，还必须满足：

- final holdout 上 `MeanRankIC > 0`
- final holdout 上相对 `SPY` 的成本后 active return > 0
- holdout 结果方向不翻转，且没有退化到近零噪音

### 7.3 证伪的必要结构

以下任一情况成立，就应判定经验性 claim 失败：

- `24` 个 primary cells 在研究期里没有一个通过 `BH-FDR q=10%`
- 某组合研究期通过，但 final holdout 上 `MeanRankIC <= 0`
- 某组合研究期通过，但 final holdout 上成本后 active return <= 0
- 结果明显只依赖单一切分或少数时间片，稳定性不足

### 7.4 中间状态

如果结果既不成功也不彻底失败：

- 不能自动默认“继续调参”
- 必须先回到 `L0-L2` 检查 claim、label、universe 是否定义得过宽或过弱
- 并明确把结果归类为 `inconclusive`

### 7.5 “可用的组合”在 v1 中是什么意思

本项目当前最终目标，不是直接宣布“生产可上线模型”，而是识别：

`feature set x label horizon x model family`

中的哪些组合，可以被诚实地称为“研究上可用”。

一个组合若要被称为“可用”，至少应满足：

- 特征全部可由预注册输入空间严格构造
- 研究期 aggregate `MeanRankIC > 0`
- 研究期主统计检验在 `BH-FDR q=10%` 下通过
- 至少 `60%` 的 OOS 测试块方向为正
- 相对被动 `SPY` 基线保留实质增量
- 不完全依赖单一时间片或单一少数股票
- 简单预注册组合在合理成本后没有被完全抹平

这一定义仍然是研究层面的“可用”，不是生产部署层面的“可用”。

---

## 8. L3: 系统级约束

### 8.1 anti-snooping

正式原则：

- OOS 查看次数必须被记账
- final holdout 的查看权必须被严格限制
- 一旦依据 final holdout 修改系统，该 holdout 自动失效

### 8.2 预注册

跑正式实验前，至少要预注册：

- universe
- label
- 持有期
- 调仓协议
- baseline
- 主 metric
- 主推断统计量
- 经济 gate metric
- 成功 / 证伪条件

### 8.3 平稳性假设

本项目不能默认市场平稳。

必须显式承认：

- 若结果依赖少数 regime，claim 就应被收窄
- 若方法不建模 regime，就必须接受其外推能力受限

### 8.4 样本预算

样本与 OOS 信用都属于有限预算。

正式要求：

- 研究阶段和最终检验阶段必须分账
- 超参搜索不能无限透支最终检验区的解释力

### 8.5 replay

正式研究结论必须满足：

- 一年后仍可重放
- 他人可按定义重放
- 关键中间对象可追溯

### 8.6 信息隔离

研究者在流程上必须尽量减少如下污染：

- 偶然瞥见未来数据
- 先看结果再改定义
- 选择性记录成功实验、忽略失败实验

---

## 9. L4 边界

当前已经允许进入一个很窄的 L4：

- daily coverage audit
- sample-boundary freeze tooling
- 输入契约与 artifact schema 的可执行化

但在 sample boundary 冻结前，仍然禁止：

- 正式训练 `XGBoost / CatBoost`
- 生成或查看 alpha OOS 结果
- 根据任何预测表现修改 universe、label、feature family、OOS protocol 或成功门槛

原因：

- coverage audit 只看输入数据是否足以支持协议，不看预测收益
- 它不会消耗 alpha 研究信用
- 但模型训练与 OOS 结果会直接污染后续定义选择

---

## 10. 下一步

当前主线 `Universe / Label / Evaluation` 已经在本 draft 中基本冻结。  
下一步只做三件事：

1. 跑 daily coverage audit
2. 基于输入覆盖结果提出 sample boundary candidate
3. 人工确认后把 [preregistration_round1_sp500_proxy_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/preregistration_round1_sp500_proxy_v1.md) 从 `Draft` 提升为 `Frozen`

在此之前，不开始正式训练代码。
