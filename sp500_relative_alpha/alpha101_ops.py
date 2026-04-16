from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np
import pandas as pd


class Alpha101OperatorError(RuntimeError):
    """Raised when Alpha101 operator inputs violate the frozen grammar."""


@dataclass(frozen=True)
class Alpha101InputMatrices:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    returns: pd.DataFrame
    shares_volume: pd.DataFrame
    typical_price: pd.DataFrame

    def adv(self, window: float) -> pd.DataFrame:
        """Project-level adv{d}: rolling mean of Alpha101 canonical dollar-volume V."""

        return ts_mean(self.volume, window)


def build_alpha101_input_matrices(
    daily_bars: pd.DataFrame,
    benchmark_symbol: str = "SPY",
    include_benchmark: bool = False,
) -> Alpha101InputMatrices:
    """Build date x symbol matrices for Alpha101 formulas.

    In this project, Alpha101 canonical `volume` means `alpha_volume`, i.e.
    typical-price dollar-volume proxy, not raw shares volume.
    """

    required = {
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "shares_volume",
        "typical_price",
        "alpha_volume",
        "close_to_close_return",
    }
    missing = sorted(required - set(daily_bars.columns))
    if missing:
        raise Alpha101OperatorError(f"daily_bars is missing columns required for Alpha101 inputs: {missing}")

    universe = daily_bars.copy()
    benchmark_calendar = pd.DatetimeIndex(
        universe.loc[universe["symbol"] == benchmark_symbol, "date"].drop_duplicates().sort_values()
    )
    if len(benchmark_calendar) == 0:
        raise Alpha101OperatorError(f"benchmark_symbol={benchmark_symbol!r} is missing from daily_bars")

    if not include_benchmark:
        universe = universe.loc[universe["symbol"] != benchmark_symbol].copy()
    if universe.empty:
        raise Alpha101OperatorError("Alpha101 input universe is empty after benchmark filtering")

    return Alpha101InputMatrices(
        open=_pivot(universe, "open", benchmark_calendar),
        high=_pivot(universe, "high", benchmark_calendar),
        low=_pivot(universe, "low", benchmark_calendar),
        close=_pivot(universe, "close", benchmark_calendar),
        volume=_pivot(universe, "alpha_volume", benchmark_calendar),
        returns=_pivot(universe, "close_to_close_return", benchmark_calendar),
        shares_volume=_pivot(universe, "shares_volume", benchmark_calendar),
        typical_price=_pivot(universe, "typical_price", benchmark_calendar),
    )


def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank by date."""

    return x.rank(axis=1, pct=True)


def delay(x: pd.DataFrame, period: float) -> pd.DataFrame:
    return x.shift(_window(period))


def delta(x: pd.DataFrame, period: float) -> pd.DataFrame:
    return x - delay(x, period)


def ts_mean(x: pd.DataFrame, window: float) -> pd.DataFrame:
    width = _window(window)
    return x.rolling(width, min_periods=width).mean()


def ts_sum(x: pd.DataFrame, window: float) -> pd.DataFrame:
    width = _window(window)
    return x.rolling(width, min_periods=width).sum()


def ts_product(x: pd.DataFrame, window: float) -> pd.DataFrame:
    width = _window(window)
    return x.rolling(width, min_periods=width).apply(np.prod, raw=True)


def ts_stddev(x: pd.DataFrame, window: float) -> pd.DataFrame:
    width = _window(window)
    return x.rolling(width, min_periods=width).std(ddof=0)


def ts_min(x: pd.DataFrame, window: float) -> pd.DataFrame:
    width = _window(window)
    return x.rolling(width, min_periods=width).min()


def ts_max(x: pd.DataFrame, window: float) -> pd.DataFrame:
    width = _window(window)
    return x.rolling(width, min_periods=width).max()


def ts_rank(x: pd.DataFrame, window: float) -> pd.DataFrame:
    width = _window(window)
    return x.rolling(width, min_periods=width).apply(_last_rank_pct, raw=True)


def ts_argmax(x: pd.DataFrame, window: float) -> pd.DataFrame:
    """1-based position of the max inside the rolling window, oldest observation = 1."""

    width = _window(window)
    return x.rolling(width, min_periods=width).apply(_argmax_1based, raw=True)


def ts_argmin(x: pd.DataFrame, window: float) -> pd.DataFrame:
    """1-based position of the min inside the rolling window, oldest observation = 1."""

    width = _window(window)
    return x.rolling(width, min_periods=width).apply(_argmin_1based, raw=True)


def correlation(x: pd.DataFrame, y: pd.DataFrame, window: float) -> pd.DataFrame:
    width = _window(window)
    x_aligned, y_aligned = x.align(y, join="outer", axis=None)
    return x_aligned.rolling(width, min_periods=width).corr(y_aligned)


def covariance(x: pd.DataFrame, y: pd.DataFrame, window: float) -> pd.DataFrame:
    width = _window(window)
    x_aligned, y_aligned = x.align(y, join="outer", axis=None)
    return x_aligned.rolling(width, min_periods=width).cov(y_aligned, ddof=0)


def decay_linear(x: pd.DataFrame, window: float) -> pd.DataFrame:
    width = _window(window)
    weights = np.arange(1, width + 1, dtype=float)
    weights = weights / weights.sum()
    return x.rolling(width, min_periods=width).apply(lambda values: float(np.dot(values, weights)), raw=True)


def scale(x: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    denominator = x.abs().sum(axis=1).replace(0.0, np.nan)
    return x.mul(a / denominator, axis=0)


def signedpower(x: pd.DataFrame, exponent: float) -> pd.DataFrame:
    return np.sign(x) * np.power(np.abs(x), exponent)


def safe_divide(numerator: pd.DataFrame | pd.Series | float, denominator: pd.DataFrame | pd.Series | float):
    denominator_safe = (
        denominator.replace(0.0, np.nan) if isinstance(denominator, (pd.DataFrame, pd.Series)) else denominator
    )
    if not isinstance(denominator_safe, (pd.DataFrame, pd.Series)) and denominator_safe == 0:
        denominator_safe = np.nan
    return numerator / denominator_safe


def _pivot(daily_bars: pd.DataFrame, value_column: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    if daily_bars[["symbol", "date"]].duplicated().any():
        duplicates = daily_bars.loc[daily_bars[["symbol", "date"]].duplicated(), ["symbol", "date"]].head(10)
        raise Alpha101OperatorError(f"duplicate symbol-date rows detected: {duplicates.to_dict('records')}")
    return (
        daily_bars.pivot(index="date", columns="symbol", values=value_column)
        .sort_index()
        .sort_index(axis=1)
        .reindex(calendar)
    )


def _window(window: float) -> int:
    width = int(floor(float(window)))
    if width <= 0:
        raise Alpha101OperatorError(f"window must floor to a positive integer, got {window!r}")
    return width


def _last_rank_pct(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    last = values[-1]
    n_less = float(np.sum(values < last))
    n_equal = float(np.sum(values == last))
    average_rank = n_less + (n_equal + 1.0) / 2.0
    return average_rank / float(len(values))


def _argmax_1based(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    return float(np.argmax(values) + 1)


def _argmin_1based(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    return float(np.argmin(values) + 1)
