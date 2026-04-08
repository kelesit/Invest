"""Feature engineering for equity cross-sectional momentum.

~18 features organized in 4 groups, each answering a distinct question:

1. Basic momentum — 谁最近更强？ (5 features)
2. Skip momentum — 去掉近期噪音后谁更强？ (3 features)
3. Path quality — 动量是稳态趋势还是靠几根大阳线硬拉？ (6 features)
4. Risk-adjusted momentum — 单位风险下谁更强？ (4 features)

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

    parts = [
        _basic_momentum(close),
        _skip_momentum(close),
        _path_quality(close, high),
        _risk_adjusted_momentum(close),
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
