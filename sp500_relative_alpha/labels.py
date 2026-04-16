from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_HORIZONS = tuple(range(5, 65, 5))
ROUND1_FIRST_FEATURE_SIGNAL_DATE = pd.Timestamp("2015-12-31")
ROUND1_LAST_LABELABLE_SIGNAL_DATE = pd.Timestamp("2025-12-31")


class LabelGenerationError(RuntimeError):
    """Raised when labels cannot be generated under the frozen alignment rules."""


@dataclass(frozen=True)
class OpenToOpenLabelConfig:
    benchmark_symbol: str = "SPY"
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    include_benchmark: bool = False
    drop_missing_labels: bool = True
    min_signal_date: pd.Timestamp | None = None
    max_signal_date: pd.Timestamp | None = None


def build_round1_benchmark_relative_open_to_open_labels(
    daily_bars: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Generate Round 1 labels inside the frozen common signal-date window."""

    return build_benchmark_relative_open_to_open_labels(
        daily_bars,
        OpenToOpenLabelConfig(
            horizons=tuple(horizons),
            min_signal_date=ROUND1_FIRST_FEATURE_SIGNAL_DATE,
            max_signal_date=ROUND1_LAST_LABELABLE_SIGNAL_DATE,
        ),
    )


def build_benchmark_relative_open_to_open_labels(
    daily_bars: pd.DataFrame,
    config: OpenToOpenLabelConfig | None = None,
) -> pd.DataFrame:
    """Generate labels using t close signal, t+1 open entry, and t+1+H open exit."""

    label_config = config or OpenToOpenLabelConfig()
    _validate_label_inputs(daily_bars, label_config)

    horizons = tuple(int(horizon) for horizon in label_config.horizons)
    if any(horizon <= 0 for horizon in horizons):
        raise LabelGenerationError(f"horizons must be positive integers: {horizons}")

    benchmark_calendar = pd.DatetimeIndex(
        daily_bars.loc[daily_bars["symbol"] == label_config.benchmark_symbol, "date"]
        .drop_duplicates()
        .sort_values()
    )
    open_matrix = (
        daily_bars.pivot(index="date", columns="symbol", values="open")
        .sort_index()
        .sort_index(axis=1)
        .reindex(benchmark_calendar)
    )
    benchmark_open = open_matrix[label_config.benchmark_symbol]
    labels: list[pd.DataFrame] = []

    for horizon in horizons:
        entry_open = open_matrix.shift(-1)
        exit_open = open_matrix.shift(-(horizon + 1))
        asset_return = exit_open / entry_open - 1.0
        benchmark_return = benchmark_open.shift(-(horizon + 1)) / benchmark_open.shift(-1) - 1.0
        relative_return = asset_return.sub(benchmark_return, axis=0)

        if not label_config.include_benchmark and label_config.benchmark_symbol in relative_return.columns:
            relative_return = relative_return.drop(columns=[label_config.benchmark_symbol])
            asset_return = asset_return.drop(columns=[label_config.benchmark_symbol])

        horizon_labels = _stack_horizon_labels(
            relative_return=relative_return,
            asset_return=asset_return[relative_return.columns],
            benchmark_return=benchmark_return,
            open_dates=pd.DatetimeIndex(open_matrix.index),
            horizon=horizon,
        )
        labels.append(horizon_labels)

    result = pd.concat(labels, ignore_index=True)
    if label_config.drop_missing_labels:
        result = result.dropna(
            subset=[
                "asset_open_to_open_return",
                "benchmark_open_to_open_return",
                "benchmark_relative_open_to_open_return",
                "entry_date",
                "exit_date",
            ]
        ).reset_index(drop=True)

    result = _filter_signal_window(result, label_config)
    return result.sort_values(["horizon", "signal_date", "symbol"], ignore_index=True)


def _stack_horizon_labels(
    relative_return: pd.DataFrame,
    asset_return: pd.DataFrame,
    benchmark_return: pd.Series,
    open_dates: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    signal_dates = pd.Series(open_dates, index=open_dates)
    entry_dates = pd.Series(open_dates, index=open_dates).shift(-1)
    exit_dates = pd.Series(open_dates, index=open_dates).shift(-(horizon + 1))

    relative_long = _melt_by_signal_date(
        relative_return,
        value_name="benchmark_relative_open_to_open_return",
    )
    asset_long = _melt_by_signal_date(
        asset_return,
        value_name="asset_open_to_open_return",
    )
    frame = relative_long.merge(asset_long, how="left", on=["signal_date", "symbol"])
    frame["horizon"] = horizon
    frame["entry_date"] = frame["signal_date"].map(entry_dates)
    frame["exit_date"] = frame["signal_date"].map(exit_dates)
    frame["benchmark_open_to_open_return"] = frame["signal_date"].map(benchmark_return)
    frame["signal_date"] = frame["signal_date"].map(signal_dates)

    return frame[
        [
            "symbol",
            "signal_date",
            "horizon",
            "entry_date",
            "exit_date",
            "asset_open_to_open_return",
            "benchmark_open_to_open_return",
            "benchmark_relative_open_to_open_return",
        ]
    ]


def _melt_by_signal_date(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    return (
        frame.rename_axis(index="signal_date", columns="symbol")
        .reset_index()
        .melt(
            id_vars="signal_date",
            var_name="symbol",
            value_name=value_name,
        )
    )


def _validate_label_inputs(daily_bars: pd.DataFrame, config: OpenToOpenLabelConfig) -> None:
    required = {"symbol", "date", "open"}
    missing = sorted(required - set(daily_bars.columns))
    if missing:
        raise LabelGenerationError(f"daily_bars is missing required columns: {missing}")

    if daily_bars[["symbol", "date"]].duplicated().any():
        duplicates = daily_bars.loc[daily_bars[["symbol", "date"]].duplicated(), ["symbol", "date"]].head(10)
        raise LabelGenerationError(f"duplicate symbol-date rows detected: {duplicates.to_dict('records')}")

    if config.benchmark_symbol not in set(daily_bars["symbol"]):
        raise LabelGenerationError(f"benchmark_symbol={config.benchmark_symbol!r} is missing from daily_bars")

    if not np.isfinite(pd.to_numeric(daily_bars["open"], errors="coerce")).all():
        raise LabelGenerationError("daily_bars contains non-finite open prices")
    if (pd.to_numeric(daily_bars["open"], errors="coerce") <= 0).any():
        raise LabelGenerationError("daily_bars contains non-positive open prices")


def _filter_signal_window(labels: pd.DataFrame, config: OpenToOpenLabelConfig) -> pd.DataFrame:
    result = labels
    if config.min_signal_date is not None:
        result = result.loc[result["signal_date"] >= pd.Timestamp(config.min_signal_date)]
    if config.max_signal_date is not None:
        result = result.loc[result["signal_date"] <= pd.Timestamp(config.max_signal_date)]
    return result.reset_index(drop=True)
