# Alpha101 Feature Admissibility v1

**日期**: 2026-04-15  
**状态**: draft / formal allowlist for v1  
**作用**: 定义在当前 `S&P500 + daily OHLCV` v1 系统中，哪些 Alpha101 特征允许进入，哪些必须被阻断

## 0. 这份文档回答什么

它回答的不是：

- 哪些 Alpha101 特征更强
- 哪些 Alpha101 特征更适合 `XGBoost` 或 `CatBoost`

它只回答一件事：

> 在当前 v1 输入空间下，哪些 Alpha101 特征是**严格可构造**的？

这里的“严格可构造”指：

- 只依赖当前 v1 允许的原始输入
- 所有中间量都可由这些输入以确定性方式复算
- 不偷偷引入 `VWAP / cap / industry classification / trade-level notional` 之类额外信息

### 0.1 Appendix A 的机械分类规则

为避免靠印象分类，v1 对原论文 Appendix A 的 `101` 个公式采用如下机械规则：

- 若公式在任意深度显式出现 `vwap`，则记为 `VWAP-blocked`
- 若公式在任意深度显式出现 `IndNeutralize / indneutralize / IndClass`，则记为 `industry-blocked`
- 若公式在任意深度显式出现 `cap`，则记为 `cap-blocked`
- 若公式不含上述 blocker，但显式使用 `adv{d}`，则记为 `Tier B`
- 若公式既不含 blocker，也不含 `adv{d}`，则记为 `Tier A`

这一定义是：

- 可复核的
- 可重放的
- 独立于任何旧实现代码的

---

## 1. v1 的 canonical input grammar

当前 v1 允许的原始输入是：

- daily `OHLCV`
- `SPY` 同口径序列

当前冻结的 canonical inputs 是：

- `open_t`
- `high_t`
- `low_t`
- `close_t`
- raw `shares_volume_t`
- `typical_price_t = (high_t + low_t + close_t) / 3`
- `alpha_volume_t = typical_price_t * shares_volume_t`
- `returns_t = close_t / close_{t-1} - 1`

其中价格与成交量的调整口径为：

- `open/high/low/close`: split-adjusted
- raw `shares_volume`: inverse-split-adjusted

这样做的目的，是让拆股前后的量价关系保持连续。

重要约定：

- raw daily OHLCV 里的 `volume` 字段代表成交股数
- 但 Alpha101 公式内部的 `volume / V` 在当前 daily v1 中统一映射为 `alpha_volume_t`
- 也就是说，所有 Alpha101 volume-family 特征使用的是 daily dollar-volume proxy，而不是 raw shares volume
- 这样 `volume` 与 `adv{d}` 的单位保持一致

---

## 2. 关于 `adv{d}` 的项目级定义

原论文把：

- `adv{d}`

定义为：

- `average daily dollar volume for the past d days`

但当前 v1 只有 daily OHLCV，没有逐笔成交金额，也没有 daily `VWAP`。  
因此如果直接按原论文语义去理解 `adv{d}`，它并不是原样可得。

为了让一部分 Alpha101 特征仍然可以在当前输入空间下进入 v1，当前项目级 canonical 定义冻结为：

- `typical_price_t = (high_t + low_t + close_t) / 3`
- `alpha_volume_t = typical_price_t * shares_volume_t`
- `adv{d}_t = mean(alpha_volume_{t-d+1:t})`

这一定义的地位是：

- 它是**严格可复算**的
- 它是**当前 v1 输入 grammar 下的项目级 aggregate**
- 它**不是**原论文 trade-level daily dollar volume 或 true daily VWAP dollar volume 的精确等价物
- 它比 `close_t * shares_volume_t` 更少依赖收盘价单点代表全天成交价格的假设
- 它也避免了把 raw shares volume 与 dollar-volume `adv{d}` 放进同一个比值或阈值比较中的单位错误

因此，凡是依赖 `adv{d}` 的 Alpha101 特征，在 v1 中都属于：

- **project-admissible**

而不是：

- **paper-input-exact**

这不是缺点掩盖，而是必须明说的口径收窄。

---

## 3. 不构成 blocker 的东西

以下因素本身不构成 v1 的 admissibility blocker：

- 原论文里的 `delay-0 / delay-1` 交易意图
- 窗口长度是小数

解释：

- 在当前项目里，Alpha101 单个公式不是直接拿来原样交易，而是作为 ML feature
- 因此只要该特征在 `t` 日收盘后可完全观测，它就可以作为 `t` 的 feature 进入模型
- 对于小数窗口，原论文 Appendix A 已说明应取 `floor(d)`

这意味着：

- 像 `Alpha#53`、`Alpha#54` 这类原文里与 close/open 时点关系很紧的公式，不因为“delay-0”身份而自动被禁止
- 真正的 blocker 是输入字段不可得，而不是原始 alpha 的交易时机设定

---

## 4. v1 的三层准入结论

### 4.1 Tier A: 直接 admissible

这些特征只依赖：

- split-consistent `OHLCV`
- `returns`

不依赖项目级 `adv{d}` 定义，也不依赖 `VWAP / cap / industry`

准入清单：

`001, 002, 003, 004, 006, 008, 009, 010, 012, 013, 014, 015, 016, 018, 019, 020, 022, 023, 024, 026, 029, 030, 033, 034, 035, 037, 038, 040, 044, 045, 046, 049, 051, 052, 053, 054, 055, 060, 101`

数量：

- `39`

### 4.2 Tier B: admissible under canonical `adv{d}`

这些特征不依赖 `VWAP / cap / industry`，  
但依赖项目级定义的：

- `adv{d}`

准入清单：

`007, 017, 021, 028, 031, 039, 043, 068, 085, 088, 092, 095, 099`

数量：

- `13`

### 4.3 v1 的正式 allowlist

v1 的正式 Alpha101 allowlist = `Tier A + Tier B`

注意：

- 从 `1-minute OHLCV` 改成 daily `OHLCV` 后，**编号级 allowlist 不变**
- 变化的是 Tier B 的 `ADV` 口径：
  - 原 minute 方案可以用 `sum(close_1m * volume_1m)` 做更细的项目级 dollar-volume aggregate
  - daily 方案使用 `((high_d + low_d + close_d) / 3) * shares_volume_d`
- 因此 Tier B 在 daily v1 中仍然是 project-admissible，但比 minute 方案更不是 paper-input-exact

完整准入清单：

`001, 002, 003, 004, 006, 007, 008, 009, 010, 012, 013, 014, 015, 016, 017, 018, 019, 020, 021, 022, 023, 024, 026, 028, 029, 030, 031, 033, 034, 035, 037, 038, 039, 040, 043, 044, 045, 046, 049, 051, 052, 053, 054, 055, 060, 068, 085, 088, 092, 095, 099, 101`

总数：

- `52`

---

## 5. Blocked features and reasons

### 5.1 Blocker A: `VWAP`

以下 Alpha101 特征依赖 `VWAP`，因此在当前只有 daily `OHLCV` 的 v1 中被阻断：

`005, 011, 025, 027, 032, 036, 041, 042, 047, 050, 057, 058, 059, 061, 062, 063, 064, 065, 066, 067, 069, 070, 071, 072, 073, 074, 075, 076, 077, 078, 079, 081, 083, 084, 086, 087, 089, 091, 093, 094, 096, 097, 098`

数量：

- `43`

原因：

- current v1 input 不含 daily `VWAP`
- 仅凭 daily `OHLCV` 无法严格恢复日内成交价格分布
- 因而无法严格恢复原论文语义下的 daily `VWAP`

### 5.2 Blocker B: `industry neutralization`

以下特征依赖：

- `IndNeutralize`
- `IndClass`

因此被阻断：

`048, 058, 059, 063, 067, 069, 070, 076, 079, 080, 082, 087, 089, 090, 091, 093, 097, 100`

数量：

- `18`

原因：

- 当前 v1 不引入 PIT industry classification
- 因而无法进行严格、可 replay 的行业中性化

### 5.3 Blocker C: `cap`

以下特征依赖 market cap：

`056`

数量：

- `1`

原因：

- 当前 v1 不引入 PIT market cap / shares outstanding 输入

### 5.4 unique blocked set

在合并 `VWAP / industry / cap` 三类 blocker 后，  
v1 的 unique blocked set 为：

`005, 011, 025, 027, 032, 036, 041, 042, 047, 048, 050, 056, 057, 058, 059, 061, 062, 063, 064, 065, 066, 067, 069, 070, 071, 072, 073, 074, 075, 076, 077, 078, 079, 080, 081, 082, 083, 084, 086, 087, 089, 090, 091, 093, 094, 096, 097, 098, 100`

总数：

- `49`

---

## 6. 特别说明

### 6.1 `Alpha#53` 与 `Alpha#54`

这两个特征在原论文语境下常被视为与当日 close 附近执行关系更紧的表达。  
但在当前项目里，它们作为：

- `t` 日收盘后可观测 feature

进入模型，因此：

- 不因原始交易意图而被阻断

它们之所以保留，是因为它们只依赖：

- `open/high/low/close`

### 6.2 如果未来拿到 `VWAP`

一旦 future v2/v3 拿到：

- daily `VWAP`
- minute-level `VWAP`
- 或 trade-level prints

则 `VWAP` blocker 列表里的特征可以重新审议。

### 6.3 如果未来拿到 PIT reference

一旦 future v2/v3 拿到：

- PIT industry classification
- PIT market cap / shares outstanding

则 `industry` 和 `cap` blocker 列表里的特征可以重新审议。

---

## 7. 实现纪律

在 v1 中，feature implementation 必须遵守：

- 只实现本 allowlist 中的 Alpha101 特征
- 不允许实现时临时把 blocked features 偷渡进来
- 不允许把 `adv{d}` 的项目级定义说成“与原论文完全同义”
- 如果将来修改 allowlist，必须同步修改本文件

---

## 8. Source Notes

本清单依据下列原始定义整理：

- Zura Kakushadze, *101 Formulaic Alphas*, Appendix A, input definitions and formulas
  - [arXiv / paper mirror PDF](https://www.linkojones.com/pdfs/SSRN-id2701346.pdf)

其中原论文明确给出：

- `returns = daily close-to-close returns`
- `open, close, high, low, volume = standard definitions`
- `vwap = daily volume-weighted average price`
- `cap = market cap`
- `adv{d} = average daily dollar volume`

当前版本按 Appendix A 的 `1..101` 公式逐条分类后得到：

- allowlist：`52`
- unique blocked：`49`

当前 v1 allowlist 的工作，是把这些原始定义映射到本项目自己的 daily `OHLCV` input grammar 下，并诚实标记哪些地方已经不再与原论文输入完全等价。
