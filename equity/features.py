"""Feature engineering for equity cross-sectional momentum.

~25 features organized in 5 groups, each answering a distinct question:

1. Basic momentum — 谁最近更强？ (5 features)
2. Skip momentum — 去掉近期噪音后谁更强？ (3 features)
3. Path quality — 动量是稳态趋势还是靠几根大阳线硬拉？ (6 features)
4. Risk-adjusted momentum — 单位风险下谁更强？ (4 features)
5. Volume-price interaction — 成交量确认还是背离了价格信号？ (7 features)

Design principles:
- Each feature maps to a testable hypothesis about future returns
- Minimal redundancy between features (no ret_5d + ma_dev_5d + rsi_5 synonyms)
- Raw values → single cross-sectional rank at the end, no double transformation
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Basic momentum — 不同时间窗口的过去收益
# ---------------------------------------------------------------------------

def _basic_momentum(close: pd.Series) -> pd.DataFrame:
    """Multi-window returns: who has been stronger recently?"""
    return pd.DataFrame({
        f"ret_{w}d": close.pct_change(w)
        for w in [5, 10, 20, 60, 120]
    }, index=close.index)


# ---------------------------------------------------------------------------
# 2. Skip momentum — 去掉近期反转噪音，保留趋势延续部分
#
# 预测未来 10 天时，最近几天的价格包含短期反转成分，
# skip 掉后更接近"可延续的动量"。
# Reference: Jegadeesh & Titman (1993) skip most recent month.
# ---------------------------------------------------------------------------

def _skip_momentum(close: pd.Series) -> pd.DataFrame:
    """Returns with recent days removed to isolate trend continuation."""
    close_5 = close.shift(5)
    close_20 = close.shift(20)
    return pd.DataFrame({
        "skip5_ret_20d": close_5 / close.shift(20) - 1,    # T-20 → T-5
        "skip5_ret_60d": close_5 / close.shift(60) - 1,    # T-60 → T-5
        "skip20_ret_120d": close_20 / close.shift(120) - 1, # T-120 → T-20
    }, index=close.index)


# ---------------------------------------------------------------------------
# 3. Path quality — 同样涨 8%，平滑上行 vs 暴涨暴跌完全不同
#
# 路径特征是 LightGBM 能发挥优势的地方：
# 它能学到 "强动量 + 平滑路径" vs "强动量 + 高波动" 的区别。
# ---------------------------------------------------------------------------

def _path_quality(close: pd.Series, high: pd.Series) -> pd.DataFrame:
    """Path characteristics: how the return was achieved matters."""
    daily_ret = close.pct_change()

    out = {}

    # Efficiency ratio: |net move| / sum(|daily moves|)
    # 接近 1 = 单边平滑行情，接近 0 = 来回震荡
    for w in [20, 60]:
        net_move = (close / close.shift(w) - 1).abs()
        total_move = daily_ret.abs().rolling(w).sum()
        out[f"efficiency_{w}d"] = net_move / total_move.replace(0, np.nan)

    # Up-day ratio: 过去 20 天有多少天是涨的
    # 区分"广泛上涨"和"靠单日大阳线拉动"
    out["up_ratio_20d"] = (daily_ret > 0).astype(float).rolling(20).mean()

    # Return concentration: 前 3 大日收益占总正收益的比例
    # 高 = 收益靠少数几天，低 = 收益均匀分布
    def _top3_concentration(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        pos = arr[arr > 0]
        if len(pos) == 0:
            return np.nan
        pos.sort()
        return pos[-3:].sum() / pos.sum()

    out["ret_concentration_20d"] = daily_ret.rolling(20).apply(
        _top3_concentration, raw=True
    )

    # Max drawdown over 20 days
    # 正在回撤的股票 vs 处于高位的股票，未来行为不同
    rolling_max = high.rolling(20).max()
    out["max_dd_20d"] = close / rolling_max - 1  # <= 0

    # Distance to 60-day high
    rolling_max_60 = high.rolling(60).max()
    out["dist_high_60d"] = close / rolling_max_60 - 1

    return pd.DataFrame(out, index=close.index)


# ---------------------------------------------------------------------------
# 4. Risk-adjusted momentum — 单位风险的强度
#
# 预测残差收益时，原始涨幅不够，需要看"波动率标准化后的强度"。
# 区分"安静地强"和"高噪音地乱冲"。
# ---------------------------------------------------------------------------

def _risk_adjusted_momentum(close: pd.Series) -> pd.DataFrame:
    """Momentum strength per unit of risk."""
    daily_ret = close.pct_change()
    out = {}

    # Return / volatility (类似 rolling Sharpe)
    for w in [20, 60]:
        ret = close.pct_change(w)
        vol = daily_ret.rolling(w).std() * np.sqrt(w)
        out[f"ret_vol_{w}d"] = ret / vol.replace(0, np.nan)

    # Linear regression slope / RMSE
    # 价格对时间回归的斜率 = 趋势方向与强度
    # RMSE = 围绕趋势线的噪音
    # slope / RMSE = 趋势的信噪比
    for w in [20, 60]:
        x = np.arange(w, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()

        def _slope_over_rmse(window: np.ndarray) -> float:
            if np.isnan(window).any():
                return np.nan
            y = window
            y_mean = y.mean()
            slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
            predicted = y_mean + slope * (x - x_mean)
            rmse = np.sqrt(((y - predicted) ** 2).mean())
            if rmse == 0:
                return np.nan
            # Normalize slope by mean price level to make it comparable
            return (slope / y_mean) / (rmse / y_mean)

        out[f"slope_rmse_{w}d"] = close.rolling(w).apply(
            _slope_over_rmse, raw=True
        )

    return pd.DataFrame(out, index=close.index)


# ---------------------------------------------------------------------------
# 5. Volume-price interaction — 成交量确认还是背离了价格信号？
#
# 成交量是独立于价格的信息源。关键不是成交量本身的高低，
# 而是成交量和价格变动的关系：
# - 放量下跌后反转概率 vs 缩量下跌后反转概率完全不同
# - 涨的时候放量 = 资金流入确认，涨的时候缩量 = 动量耗竭
# 全部向量化实现，无 rolling().apply()。
# ---------------------------------------------------------------------------

def _volume_price_interaction(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Volume-price interaction features."""
    daily_ret = close.pct_change()
    out = {}

    # --- Abnormal volume: 近期成交量相对历史水平 ---
    # 放量 = 信息到达，市场对该股票的关注度突然升高
    vol_5 = volume.rolling(5).mean()
    vol_20 = volume.rolling(20).mean()
    vol_60 = volume.rolling(60).mean()
    out["abnormal_vol_5d"] = vol_5 / vol_60.replace(0, np.nan)
    out["abnormal_vol_20d"] = vol_20 / vol_60.replace(0, np.nan)

    # --- Volume-weighted return: 高成交量日的收益权重更大 ---
    # 放量日的价格变动包含更多信息（更多参与者达成共识）
    # 如果放量日普遍是涨的 → 资金在流入
    # 如果放量日普遍是跌的 → 资金在流出
    for w in [20, 60]:
        vw_ret = (daily_ret * volume).rolling(w).sum() / volume.rolling(w).sum().replace(0, np.nan)
        out[f"vwap_ret_{w}d"] = vw_ret

    # --- Up-volume ratio: 上涨日的成交量 / 下跌日的成交量 ---
    # > 1 = 涨的时候比跌的时候更活跃，买方力量占优
    # < 1 = 跌的时候成交量更大，卖方主导
    up_vol = (volume * (daily_ret > 0)).rolling(20).sum()
    down_vol = (volume * (daily_ret <= 0)).rolling(20).sum()
    out["up_vol_ratio_20d"] = up_vol / down_vol.replace(0, np.nan)

    # --- Reversal × volume interaction: 放量反转信号 ---
    # 核心假说：放量下跌 → 恐慌性抛售 → 反转概率更高
    #           缩量下跌 → 阴跌 → 可能继续跌
    # ret_5d 本身 IC = -0.015（反转信号），乘以异常成交量后
    # 应该能区分"值得反转的下跌"和"不值得的下跌"
    out["reversal_vol_5d"] = close.pct_change(5) * (vol_5 / vol_60.replace(0, np.nan))

    # --- Volume-price correlation: 量价同向还是背离 ---
    # 正相关 = 涨放量跌缩量 = 趋势健康
    # 负相关 = 涨缩量跌放量 = 趋势不健康，可能反转
    out["vol_price_corr_20d"] = daily_ret.rolling(20).corr(volume)

    return pd.DataFrame(out, index=close.index)


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for a single stock.

    Args:
        df: DataFrame with columns [open, high, low, close, volume], indexed by date.

    Returns:
        DataFrame with ~18 feature columns, same date index.
    """
    close = df["close"]
    high = df["high"]
    volume = df["volume"]

    parts = [
        _basic_momentum(close),
        _skip_momentum(close),
        _path_quality(close, high),
        _risk_adjusted_momentum(close),
        _volume_price_interaction(close, volume),
    ]

    return pd.concat(parts, axis=1)


def compute_all_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute features for all stocks in the panel.

    Args:
        panel: MultiIndex (date, ticker) DataFrame with OHLCV columns.

    Returns:
        MultiIndex (date, ticker) DataFrame with feature columns.
    """
    results = []
    tickers = panel.index.get_level_values("ticker").unique()

    for ticker in tickers:
        df = panel.xs(ticker, level="ticker")
        feats = compute_features(df)
        feats["ticker"] = ticker
        feats.index.name = "date"
        results.append(feats.reset_index())

    all_feats = pd.concat(results, ignore_index=True)
    all_feats = all_feats.set_index(["date", "ticker"]).sort_index()
    return all_feats


def cross_sectional_rank(features: pd.DataFrame) -> pd.DataFrame:
    """Transform all features to cross-sectional percentile ranks [0, 1].

    For each date, ranks all stocks on each feature. This removes scale
    differences across time and limits outlier impact. For tree models,
    rank preserves all ordinal information (identical split points).

    Args:
        features: MultiIndex (date, ticker) DataFrame with feature columns.

    Returns:
        Same shape DataFrame with values in [0, 1].
    """
    return features.groupby(level="date").rank(pct=True)


def get_feature_names() -> list[str]:
    """Return the list of feature column names."""
    dummy = pd.DataFrame({
        "open": np.random.randn(300).cumsum() + 100,
        "high": np.random.randn(300).cumsum() + 101,
        "low": np.random.randn(300).cumsum() + 99,
        "close": np.random.randn(300).cumsum() + 100,
        "volume": np.abs(np.random.randn(300)) * 1e6,
    })
    # Ensure high >= close >= low for realistic data
    dummy["high"] = dummy[["open", "high", "close"]].max(axis=1)
    dummy["low"] = dummy[["open", "low", "close"]].min(axis=1)
    return compute_features(dummy).columns.tolist()
