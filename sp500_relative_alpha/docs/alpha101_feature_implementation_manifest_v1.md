# Alpha101 Feature Implementation Manifest v1

**日期**: 2026-04-15  
**状态**: draft / implementation-facing manifest  
**作用**: 把 v1 allowlist 中的 Alpha101 特征，逐编号映射到实现所需的 canonical inputs、`adv{d}` 依赖和数值稳定性风险

## 0. 使用方式

这份文档是：

- `alpha101_feature_admissibility_v1.md` 的实现层补充

它回答：

- 每个可实现 Alpha 编号需要哪些 canonical inputs
- 是否依赖项目级 `adv{d}`
- 实现时最需要小心的数值稳定性问题是什么

它不回答：

- 这些特征是否最终应该进入模型
- 哪些特征在 OOS 中最强

## 1. Canonical input legend

- `O` = daily open
- `H` = daily high
- `L` = daily low
- `C` = daily close
- `SHV` = raw daily shares volume, inverse-split-adjusted
- `TP` = daily typical price, `(H + L + C) / 3`
- `V` = Alpha101 canonical volume, `TP * SHV`
- `R` = daily close-to-close returns
- `ADV` = project-level `adv{d}` aggregate defined as rolling mean of Alpha101 canonical `V` in [alpha101_feature_admissibility_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/alpha101_feature_admissibility_v1.md)

Note:

- Manifest rows keep the canonical Alpha101 symbol `V`
- In daily v1, `V` does **not** mean raw shares volume
- This keeps `V`, `V/ADV`, and `ADV < V` formulas unit-consistent under the chosen project grammar

## 2. Risk flag legend

- `L` = low
  - 普通 rolling / rank / correlation，无明显分母奇点
- `M` = medium
  - 包含分段条件、长 lookback、嵌套算子、或对极端值较敏感
- `H` = high
  - 明显分母接近零风险，或深层嵌套导致更高的数值脆弱性

## 3. Manifest

| Alpha | Tier | Inputs | Uses `ADV` | Risk | Notes |
|---|---|---|---|---|---|
| `001` | A | `C, R` | No | `M` | Piecewise on `R<0`, then `signedpower` + `ts_argmax` |
| `002` | A | `O, C, V` | No | `M` | Uses `log(V)` and `(C-O)/O` |
| `003` | A | `O, V` | No | `L` | Simple correlation of `rank(O)` and `rank(V)` |
| `004` | A | `L` | No | `L` | Pure low-price rank/ts-rank |
| `006` | A | `O, V` | No | `L` | Simple `correlation(O, V)` |
| `007` | B | `C, V, ADV` | Yes | `M` | Has `ADV< V` gate and sign logic |
| `008` | A | `O, R` | No | `L` | Sum-open times sum-returns difference |
| `009` | A | `C` | No | `M` | Piecewise over rolling min/max of `delta(C,1)` |
| `010` | A | `C` | No | `M` | Same family as `009`, wrapped in cross-sectional rank |
| `012` | A | `C, V` | No | `L` | `sign(delta(V,1)) * delta(C,1)` |
| `013` | A | `C, V` | No | `L` | Covariance of cross-sectional ranks |
| `014` | A | `R, O, V` | No | `L` | `delta(R,3)` times `correlation(O,V)` |
| `015` | A | `H, V` | No | `L` | Sum of ranked high-volume correlations |
| `016` | A | `H, V` | No | `L` | Covariance of `rank(H)` and `rank(V)` |
| `017` | B | `C, V, ADV` | Yes | `M` | Uses `volume/adv20` and second difference of close |
| `018` | A | `O, C` | No | `L` | `stddev(abs(C-O),5)` plus `correlation(C,O)` |
| `019` | A | `C, R` | No | `M` | Includes long `sum(R,250)` component |
| `020` | A | `O, H, L, C` | No | `L` | Pure gap-style open vs delayed H/C/L |
| `021` | B | `C, V, ADV` | Yes | `M` | Multi-branch rule with `V/ADV` gate |
| `022` | A | `H, V, C` | No | `L` | `delta(correlation(H,V,5),5)` times `rank(stddev(C,20))` |
| `023` | A | `H` | No | `L` | Threshold on `H` vs average high |
| `024` | A | `C` | No | `M` | Long `100d` lookback with branch on moving-average drift |
| `026` | A | `H, V` | No | `L` | `ts_max(correlation(ts_rank(V), ts_rank(H)))` |
| `028` | B | `H, L, C, ADV` | Yes | `L` | Correlation of `ADV` with low, then scale |
| `029` | A | `C, R` | No | `H` | Deeply nested rank/log/product construction |
| `030` | A | `C, V` | No | `L` | Sign chain on delayed closes times volume ratio |
| `031` | B | `C, L, ADV` | Yes | `M` | Deep rank/decay stack plus `correlation(ADV, L)` |
| `033` | A | `O, C` | No | `L` | Pure open/close ratio rank |
| `034` | A | `C, R` | No | `M` | Ratio of short-window return stddevs |
| `035` | A | `H, L, C, V, R` | No | `L` | Product of ts-ranks of `V`, range proxy, and `R` |
| `037` | A | `O, C` | No | `M` | Uses `correlation(delay(O-C), C, 200)` |
| `038` | A | `O, C` | No | `L` | `ts_rank(C,10)` times `rank(C/O)` |
| `039` | B | `C, V, R, ADV` | Yes | `M` | Uses `V/ADV` and long `sum(R,250)` |
| `040` | A | `H, V` | No | `L` | `rank(stddev(H,10)) * correlation(H,V,10)` |
| `043` | B | `C, V, ADV` | Yes | `L` | Pure `V/ADV` ts-rank times `delta(C,7)` ts-rank |
| `044` | A | `H, V` | No | `L` | `correlation(H, rank(V), 5)` |
| `045` | A | `C, V` | No | `L` | Close-volume correlation and moving-close correlation |
| `046` | A | `C` | No | `M` | Piecewise slope comparison on delayed close |
| `049` | A | `C` | No | `M` | Piecewise threshold rule on delayed close slopes |
| `051` | A | `C` | No | `M` | Same family as `049`, different threshold |
| `052` | A | `L, V, R` | No | `M` | Combines `ts_min(L,5)`, long return spread, and `ts_rank(V,5)` |
| `053` | A | `H, L, C` | No | `H` | Division by `(C-L)` can blow up near zero |
| `054` | A | `O, H, L, C` | No | `H` | Ratio uses `(L-H)` in denominator |
| `055` | A | `H, L, C, V` | No | `M` | Range normalization with `ts_max(H)-ts_min(L)` denominator |
| `060` | A | `H, L, C, V` | No | `H` | Uses intraday location over `(H-L)` denominator |
| `068` | B | `H, L, C, ADV` | Yes | `L` | `correlation(rank(H), rank(ADV15))` vs delta of weighted `C/L` |
| `085` | B | `H, L, C, V, ADV` | Yes | `L` | Two correlations, one with `ADV30`, one with `V` |
| `088` | B | `O, H, L, C, ADV` | Yes | `M` | Min of decay/rank stack and `correlation(ts_rank(C), ts_rank(ADV60))` |
| `092` | B | `O, H, L, C, ADV` | Yes | `M` | Boolean branch plus `correlation(rank(L), rank(ADV30))` |
| `095` | B | `O, H, L, ADV` | Yes | `M` | Threshold compares open-location with powered ranked correlation |
| `099` | B | `H, L, V, ADV` | Yes | `L` | Compare `correlation(sum((H+L)/2), sum(ADV60))` vs `correlation(L,V)` |
| `101` | A | `O, H, L, C` | No | `L` | Denominator has explicit `+0.001` stabilizer |

## 4. Notes on implementation order

建议实现顺序：

1. 先做不依赖 `ADV` 的 Tier A
2. 再引入项目级 `ADV`
3. 最后实现高风险分母类：
   - `053`
   - `054`
   - `055`
   - `060`
4. 最后接入依赖项目级 `ADV` 的 Tier B 公式：
   - `007, 017, 021, 028, 031, 039, 043, 068, 085, 088, 092, 095, 099`

这样可以更容易定位数值问题到底来自：

- 基础 rolling / rank operator
- `ADV` 项目级 aggregate
- 还是分母接近零的公式

### 4.1 Current implementation status

当前已经实现：

- canonical input matrix builder：
  - [alpha101_ops.py](/Users/hsy/Work/Invest/sp500_relative_alpha/alpha101_ops.py)
- bottom operator layer：
  - `rank`
  - `delay`
  - `delta`
  - `ts_mean / ts_sum / ts_product`
  - `ts_stddev`
  - `ts_min / ts_max`
  - `ts_rank`
  - `ts_argmax / ts_argmin`
  - `correlation / covariance`
  - `decay_linear`
  - `scale`
  - `signedpower`
  - `safe_divide`
- first formula batch：
  - [alpha101_features.py](/Users/hsy/Work/Invest/sp500_relative_alpha/alpha101_features.py)
  - `003, 004, 006, 012, 013, 016, 020, 033, 101`
- second formula batch：
  - [alpha101_features.py](/Users/hsy/Work/Invest/sp500_relative_alpha/alpha101_features.py)
  - `008, 014, 015, 018, 022, 023, 030, 038, 040, 044, 045`
- third formula batch：
  - [alpha101_features.py](/Users/hsy/Work/Invest/sp500_relative_alpha/alpha101_features.py)
  - `001, 002, 009, 010, 019, 024, 034, 046, 049, 051, 052`
- fourth formula batch：
  - [alpha101_features.py](/Users/hsy/Work/Invest/sp500_relative_alpha/alpha101_features.py)
  - `026, 029, 035, 037, 053, 054, 055, 060`
- Tier B / `ADV` formula batch：
  - [alpha101_features.py](/Users/hsy/Work/Invest/sp500_relative_alpha/alpha101_features.py)
  - `007, 017, 021, 028, 031, 039, 043, 068, 085, 088, 092, 095, 099`

实现覆盖：

- 当前 daily `OHLCV` allowlist 中的 `52 / 52` 个 Alpha101 公式已接入
- 原始论文里需要 `VWAP`、intraday return、industry neutralization 或其他不可由 daily `OHLCV` 严格推出的公式仍保持 blocked，不进入 v1 allowlist

纪律：

- 在逐编号接入公式前，不生成任何 alpha 预测效果、Rank IC 或 OOS 结果
- 每个公式必须有独立单元测试，先验证输入依赖、shape、lookback、NaN 行为和数值稳定性，再进入评估 harness

### 4.2 Frozen operator semantics

当前底层算子的项目级语义冻结为：

- fractional window：
  - `floor(d)`
- rolling operators：
  - 必须满足完整窗口才输出非空值
- cross-sectional `rank`：
  - 每个日期横截面 percentile rank
  - ties 使用 average rank
- `ts_rank`：
  - rolling window 内最新观测值的 percentile rank
  - ties 使用 average rank
- `ts_argmax / ts_argmin`：
  - rolling window 内极值位置
  - `1` 表示窗口内最旧观测，`d` 表示最新观测
- `ts_stddev / covariance`：
  - 使用 population 口径 `ddof=0`
- `decay_linear`：
  - 线性权重随时间递增，最新观测权重最大
- `correlation / covariance`：
  - 逐 symbol time-series rolling 计算，不做 cross-sectional 混合

## 5. Relationship to allowlist

这份 manifest 只覆盖：

- 当前 allowlist 中的 `52` 个 Alpha101 特征

blocked features 仍以：

- [alpha101_feature_admissibility_v1.md](/Users/hsy/Work/Invest/sp500_relative_alpha/docs/alpha101_feature_admissibility_v1.md)

为准。
