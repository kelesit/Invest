"""Feature engineering for equity momentum strategy.

~50 price/volume features organized in 5 categories:
1. Price momentum (returns, MA deviation, 52w high/low)
2. Volume features (volume ratio, price-volume divergence)
3. Volatility features (realized vol, ATR, intraday range)
4. Technical indicators (RSI, MACD, Bollinger %B)
5. Cross-sectional rank transformation
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Price momentum features
# ---------------------------------------------------------------------------

def _returns(close: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Multi-window log returns."""
    out = {}
    for w in windows:
        out[f"ret_{w}d"] = close.pct_change(w)
    return pd.DataFrame(out, index=close.index)


def _ma_deviation(close: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Close / SMA ratio — measures deviation from moving average."""
    out = {}
    for w in windows:
        sma = close.rolling(w).mean()
        out[f"ma_dev_{w}d"] = close / sma - 1
    return pd.DataFrame(out, index=close.index)


def _high_low_distance(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """Distance from 52-week (252-day) high and low."""
    high_252 = high.rolling(252).max()
    low_252 = low.rolling(252).min()
    return pd.DataFrame({
        "dist_52w_high": close / high_252 - 1,  # <= 0, how far below high
        "dist_52w_low": close / low_252 - 1,    # >= 0, how far above low
    }, index=close.index)


def _momentum_acceleration(close: pd.Series) -> pd.DataFrame:
    """Momentum acceleration: short-term minus long-term momentum."""
    ret_5 = close.pct_change(5)
    ret_20 = close.pct_change(20)
    ret_60 = close.pct_change(60)
    ret_120 = close.pct_change(120)
    return pd.DataFrame({
        "mom_accel_5_20": ret_5 - ret_20,
        "mom_accel_20_60": ret_20 - ret_60,
        "mom_accel_60_120": ret_60 - ret_120,
    }, index=close.index)


# ---------------------------------------------------------------------------
# 2. Volume features
# ---------------------------------------------------------------------------

def _volume_ratio(volume: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Volume relative to its moving average."""
    out = {}
    for w in windows:
        vol_ma = volume.rolling(w).mean()
        out[f"vol_ratio_{w}d"] = volume / vol_ma
    return pd.DataFrame(out, index=volume.index)


def _volume_trend(volume: pd.Series) -> pd.DataFrame:
    """Volume trend: short MA / long MA of volume."""
    vol_5 = volume.rolling(5).mean()
    vol_20 = volume.rolling(20).mean()
    vol_60 = volume.rolling(60).mean()
    return pd.DataFrame({
        "vol_trend_5_20": vol_5 / vol_20,
        "vol_trend_20_60": vol_20 / vol_60,
    }, index=volume.index)


def _price_volume_divergence(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Price-volume divergence: sign agreement between price change and volume change."""
    price_chg = close.pct_change(5)
    vol_chg = volume.rolling(5).mean() / volume.rolling(20).mean() - 1
    return pd.DataFrame({
        "pv_divergence": price_chg * vol_chg,
    }, index=close.index)


# ---------------------------------------------------------------------------
# 3. Volatility features
# ---------------------------------------------------------------------------

def _realized_vol(close: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Realized volatility (std of daily returns) over multiple windows."""
    daily_ret = close.pct_change()
    out = {}
    for w in windows:
        out[f"rvol_{w}d"] = daily_ret.rolling(w).std()
    return pd.DataFrame(out, index=close.index)


def _vol_change(close: pd.Series) -> pd.DataFrame:
    """Volatility regime change: short-term vol / long-term vol."""
    daily_ret = close.pct_change()
    vol_5 = daily_ret.rolling(5).std()
    vol_20 = daily_ret.rolling(20).std()
    vol_60 = daily_ret.rolling(60).std()
    return pd.DataFrame({
        "vol_change_5_20": vol_5 / vol_20,
        "vol_change_5_60": vol_5 / vol_60,
    }, index=close.index)


def _intraday_range(high: pd.Series, low: pd.Series, close: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Average intraday range (high-low)/close over windows."""
    daily_range = (high - low) / close
    out = {}
    for w in windows:
        out[f"intraday_range_{w}d"] = daily_range.rolling(w).mean()
    return pd.DataFrame(out, index=close.index)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Average True Range normalized by close."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    out = {}
    for w in windows:
        out[f"atr_{w}d"] = tr.rolling(w).mean() / close
    return pd.DataFrame(out, index=close.index)


# ---------------------------------------------------------------------------
# 4. Technical indicator features
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Relative Strength Index."""
    delta = close.diff()
    out = {}
    for w in windows:
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        rs = gain / loss.replace(0, np.nan)
        out[f"rsi_{w}"] = 100 - 100 / (1 + rs)
    return pd.DataFrame(out, index=close.index)


def _macd_hist(close: pd.Series) -> pd.DataFrame:
    """Normalized MACD histogram."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    # Normalize by rolling std for cross-stock comparability
    hist_std = hist.rolling(60).std().replace(0, np.nan)
    return pd.DataFrame({
        "macd_hist_norm": hist / hist_std,
    }, index=close.index)


def _bollinger_pctb(close: pd.Series, window: int = 20) -> pd.DataFrame:
    """Bollinger Band %B: position within bands [0=lower, 1=upper]."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    pctb = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"bb_pctb": pctb}, index=close.index)


def _breakout_ratio(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.DataFrame:
    """Fraction of recent days where close was near high vs low of the window."""
    rolling_high = high.rolling(window).max()
    rolling_low = low.rolling(window).min()
    rng = (rolling_high - rolling_low).replace(0, np.nan)
    position = (close - rolling_low) / rng
    return pd.DataFrame({"channel_position": position}, index=close.index)


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

RETURN_WINDOWS = [5, 10, 20, 60, 120]
MA_WINDOWS = [5, 10, 20, 60, 120]
VOL_RATIO_WINDOWS = [5, 10, 20]
RVOL_WINDOWS = [5, 10, 20, 60]
RANGE_WINDOWS = [5, 20]
ATR_WINDOWS = [14, 20]
RSI_WINDOWS = [5, 14]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all time-series features for a single stock.

    Args:
        df: DataFrame with columns [open, high, low, close, volume], indexed by date.

    Returns:
        DataFrame with ~50 feature columns, same date index.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    parts = [
        # Price momentum (~15)
        _returns(close, RETURN_WINDOWS),
        _ma_deviation(close, MA_WINDOWS),
        _high_low_distance(close, high, low),
        _momentum_acceleration(close),

        # Volume (~8)
        _volume_ratio(volume, VOL_RATIO_WINDOWS),
        _volume_trend(volume),
        _price_volume_divergence(close, volume),

        # Volatility (~10)
        _realized_vol(close, RVOL_WINDOWS),
        _vol_change(close),
        _intraday_range(high, low, close, RANGE_WINDOWS),
        _atr(high, low, close, ATR_WINDOWS),

        # Technical (~6)
        _rsi(close, RSI_WINDOWS),
        _macd_hist(close),
        _bollinger_pctb(close),
        _breakout_ratio(high, low, close),
    ]

    features = pd.concat(parts, axis=1)
    return features


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
    differences between features and limits the impact of outliers.

    Args:
        features: MultiIndex (date, ticker) DataFrame with feature columns.

    Returns:
        Same shape DataFrame with values in [0, 1].
    """
    return features.groupby(level="date").rank(pct=True)


def get_feature_names() -> list[str]:
    """Return the list of feature column names (for reference)."""
    # Generate from a dummy stock to get exact column names
    dummy = pd.DataFrame({
        "open": np.ones(300),
        "high": np.ones(300) * 1.01,
        "low": np.ones(300) * 0.99,
        "close": np.ones(300),
        "volume": np.ones(300) * 1e6,
    })
    return compute_features(dummy).columns.tolist()
