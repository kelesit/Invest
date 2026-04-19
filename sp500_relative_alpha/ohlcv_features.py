from __future__ import annotations

import numpy as np
import pandas as pd


class OhlcvFeatureError(RuntimeError):
    """Raised when OHLCV primitive matrices cannot be built."""


RETURN_WINDOWS = (1, 5, 10, 20, 60, 120, 252)
EXCESS_RET_LAGS = (1, 2, 5, 10, 20)
EXCESS_RET_WINDOWS = (5, 21, 63)

_REQUIRED_COLUMNS = {"symbol", "date", "open", "high", "low", "close", "shares_volume"}


def compute_ohlcv_feature_matrices(
    daily_bars: pd.DataFrame,
    benchmark_symbol: str = "SPY",
) -> dict[str, pd.DataFrame]:
    """Build OHLCV primitive feature matrices (date × symbol).

    Returns a dict keyed by primitive name, e.g. "ret_1d", "log_volume".
    The benchmark symbol is excluded from the output universe but its calendar
    is used to align all matrices — matching the Alpha101 convention.
    """
    _validate(daily_bars, benchmark_symbol)

    benchmark_calendar = pd.DatetimeIndex(
        daily_bars.loc[daily_bars["symbol"] == benchmark_symbol, "date"]
        .drop_duplicates()
        .sort_values()
    )
    universe = daily_bars.loc[daily_bars["symbol"] != benchmark_symbol]

    close = _pivot(universe, "close", benchmark_calendar)
    open_ = _pivot(universe, "open", benchmark_calendar)
    high = _pivot(universe, "high", benchmark_calendar)
    low = _pivot(universe, "low", benchmark_calendar)
    volume = _pivot(universe, "shares_volume", benchmark_calendar)

    matrices: dict[str, pd.DataFrame] = {}

    for w in RETURN_WINDOWS:
        matrices[f"ret_{w}d"] = close.pct_change(w)

    matrices["intraday_ret"] = _safe_divide(close - open_, open_)
    matrices["overnight_gap"] = _safe_divide(open_ - close.shift(1), close.shift(1))
    matrices["close_position"] = _safe_divide(close - low, high - low)
    matrices["high_low_range"] = _safe_divide(high - low, close)

    matrices["log_volume"] = _safe_log(volume)
    matrices["volume_ratio_20d"] = _safe_divide(volume, volume.rolling(20, min_periods=20).mean())
    matrices["volume_ratio_60d"] = _safe_divide(volume, volume.rolling(60, min_periods=60).mean())

    # Market-neutral excess return primitives (equal-weight index neutralization).
    # excess_ret captures reversal and momentum after stripping the market factor,
    # equivalent to the "historical target features" in the Optiver 1st-place solution.
    daily_ret = close.pct_change(1)
    index_ret = daily_ret.mean(axis=1)  # equal-weight cross-sectional mean per date
    excess_ret = daily_ret.sub(index_ret, axis=0)

    for lag in EXCESS_RET_LAGS:
        matrices[f"excess_ret_lag{lag}d"] = excess_ret.shift(lag)
    for w in EXCESS_RET_WINDOWS:
        matrices[f"excess_ret_mean{w}d"] = excess_ret.rolling(w, min_periods=w).mean()
        matrices[f"excess_ret_std{w}d"] = excess_ret.rolling(w, min_periods=w).std(ddof=1)

    return matrices


def _pivot(df: pd.DataFrame, col: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    return (
        df.pivot(index="date", columns="symbol", values=col)
        .sort_index()
        .sort_index(axis=1)
        .reindex(calendar)
    )


def _safe_divide(num: pd.DataFrame, denom: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(invalid="ignore", divide="ignore"):
        result = num / denom.replace(0.0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _safe_log(x: pd.DataFrame) -> pd.DataFrame:
    arr = x.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan)
    mask = arr > 0
    out[mask] = np.log(arr[mask])
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _validate(daily_bars: pd.DataFrame, benchmark_symbol: str) -> None:
    missing = sorted(_REQUIRED_COLUMNS - set(daily_bars.columns))
    if missing:
        raise OhlcvFeatureError(f"daily_bars is missing columns: {missing}")
    if benchmark_symbol not in set(daily_bars["symbol"]):
        raise OhlcvFeatureError(f"benchmark_symbol={benchmark_symbol!r} not found in daily_bars")
