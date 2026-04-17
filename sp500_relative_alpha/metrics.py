from __future__ import annotations

from dataclasses import dataclass, replace
from math import erfc, sqrt
from typing import Iterable

import numpy as np
import pandas as pd

from .folds import WalkForwardFold, fold_period_mask


class MetricComputationError(RuntimeError):
    """Raised when registered metrics cannot be computed from the sample table."""


@dataclass(frozen=True)
class RankICConfig:
    score_column: str = "score"
    label_column: str = "benchmark_relative_open_to_open_return"
    min_cross_section_size: int = 5
    hac_lag: int | None = None
    bootstrap_iterations: int = 1000
    bootstrap_block_length: int | None = None
    bootstrap_seed: int = 20260416
    ci_level: float = 0.95


def compute_daily_rank_ic(samples: pd.DataFrame, config: RankICConfig | None = None) -> pd.DataFrame:
    """Compute cross-sectional Rank IC for each signal date and horizon."""

    metric_config = config or RankICConfig()
    _validate_rank_ic_inputs(samples, metric_config)
    frame = samples.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="raise").astype(int)

    rows: list[dict[str, object]] = []
    for (horizon, signal_date), group in frame.groupby(["horizon", "signal_date"], sort=True):
        clean = group[["symbol", metric_config.score_column, metric_config.label_column]].dropna()
        n_obs = len(clean)
        rank_ic = np.nan
        if n_obs >= metric_config.min_cross_section_size:
            rank_ic = _spearman_rank_ic(clean[metric_config.score_column], clean[metric_config.label_column])
        rows.append(
            {
                "horizon": int(horizon),
                "signal_date": pd.Timestamp(signal_date),
                "rank_ic": rank_ic,
                "n_obs": int(n_obs),
            }
        )

    return pd.DataFrame(rows).sort_values(["horizon", "signal_date"], ignore_index=True)


def build_oos_rank_ic_panel(
    samples: pd.DataFrame,
    folds: Iterable[WalkForwardFold],
    config: RankICConfig | None = None,
) -> pd.DataFrame:
    """Compute daily Rank IC only on each fold's OOS test block."""

    metric_config = config or RankICConfig()
    _validate_rank_ic_inputs(samples, metric_config)
    frames: list[pd.DataFrame] = []
    for fold in folds:
        test_samples = samples.loc[fold_period_mask(samples, fold, "test")].copy()
        if test_samples.empty:
            continue
        daily = compute_daily_rank_ic(test_samples, metric_config)
        daily.insert(0, "fold_id", fold.fold_id)
        frames.append(daily)

    if not frames:
        return pd.DataFrame(columns=["fold_id", "horizon", "signal_date", "rank_ic", "n_obs"])
    return pd.concat(frames, ignore_index=True).sort_values(["horizon", "fold_id", "signal_date"], ignore_index=True)


def summarize_rank_ic_series(
    rank_ic: pd.Series,
    config: RankICConfig | None = None,
) -> dict[str, float | int]:
    """Summarize a daily Rank IC series with HAC and block-bootstrap inference."""

    metric_config = config or RankICConfig()
    values = pd.to_numeric(rank_ic, errors="coerce").dropna().to_numpy(dtype=float)
    n_dates = int(len(values))
    if n_dates == 0:
        return {
            "n_dates": 0,
            "mean_rank_ic": np.nan,
            "positive_rate": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "hac_lag": 0,
            "hac_se": np.nan,
            "hac_t_stat": np.nan,
            "p_value_one_sided": np.nan,
            "p_value_two_sided": np.nan,
            "bootstrap_block_length": 0,
            "bootstrap_ci_lower": np.nan,
            "bootstrap_ci_upper": np.nan,
        }

    mean_rank_ic = float(np.mean(values))
    ic_std = float(np.std(values, ddof=1)) if n_dates > 1 else np.nan
    hac_lag, hac_se, hac_t_stat = _newey_west_t_stat(values, metric_config.hac_lag)
    ci_block_length, ci_lower, ci_upper = _moving_block_bootstrap_ci(values, metric_config)
    return {
        "n_dates": n_dates,
        "mean_rank_ic": mean_rank_ic,
        "positive_rate": float(np.mean(values > 0.0)),
        "ic_std": ic_std,
        "icir": float(mean_rank_ic / ic_std) if np.isfinite(ic_std) and ic_std > 0 else np.nan,
        "hac_lag": int(hac_lag),
        "hac_se": hac_se,
        "hac_t_stat": hac_t_stat,
        "p_value_one_sided": _normal_sf(hac_t_stat),
        "p_value_two_sided": min(1.0, 2.0 * _normal_sf(abs(hac_t_stat))) if np.isfinite(hac_t_stat) else np.nan,
        "bootstrap_block_length": int(ci_block_length),
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
    }


def summarize_oos_rank_ic_panel(
    rank_ic_panel: pd.DataFrame,
    config: RankICConfig | None = None,
    group_columns: tuple[str, ...] = ("horizon",),
) -> pd.DataFrame:
    """Aggregate an OOS daily Rank IC panel by horizon or fold/horizon."""

    metric_config = config or RankICConfig()
    required = {"rank_ic", *group_columns}
    missing = sorted(required - set(rank_ic_panel.columns))
    if missing:
        raise MetricComputationError(f"rank_ic_panel is missing required columns: {missing}")

    rows: list[dict[str, object]] = []
    for keys, group in rank_ic_panel.groupby(list(group_columns), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        summary = summarize_rank_ic_series(group["rank_ic"], metric_config)
        rows.append({**dict(zip(group_columns, keys, strict=True)), **summary})
    if not rows:
        return pd.DataFrame(columns=[*group_columns])
    return pd.DataFrame(rows).sort_values(list(group_columns), ignore_index=True)


def evaluate_oos_rank_ic(
    samples: pd.DataFrame,
    folds: Iterable[WalkForwardFold],
    config: RankICConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return daily OOS IC, fold-level summaries, and aggregate horizon summaries."""

    metric_config = config or RankICConfig()
    panel = build_oos_rank_ic_panel(samples, folds, metric_config)
    fold_summary = summarize_oos_rank_ic_panel(
        panel,
        replace(metric_config, bootstrap_seed=metric_config.bootstrap_seed + 1),
        group_columns=("horizon", "fold_id"),
    )
    horizon_summary = summarize_oos_rank_ic_panel(
        panel,
        replace(metric_config, bootstrap_seed=metric_config.bootstrap_seed + 2),
        group_columns=("horizon",),
    )
    return panel, fold_summary, horizon_summary


def _validate_rank_ic_inputs(samples: pd.DataFrame, config: RankICConfig) -> None:
    required = {"signal_date", "symbol", "horizon", config.score_column, config.label_column}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise MetricComputationError(f"samples are missing required columns: {missing}")
    if config.min_cross_section_size < 2:
        raise MetricComputationError("min_cross_section_size must be >= 2")
    if config.bootstrap_iterations < 0:
        raise MetricComputationError("bootstrap_iterations must be non-negative")
    if not 0.0 < config.ci_level < 1.0:
        raise MetricComputationError("ci_level must be between 0 and 1")


def _spearman_rank_ic(score: pd.Series, label: pd.Series) -> float:
    score_rank = score.rank(method="average")
    label_rank = label.rank(method="average")
    score_values = score_rank.to_numpy(dtype=float)
    label_values = label_rank.to_numpy(dtype=float)
    if np.nanstd(score_values) == 0.0 or np.nanstd(label_values) == 0.0:
        return np.nan
    return float(np.corrcoef(score_values, label_values)[0, 1])


def _newey_west_t_stat(values: np.ndarray, requested_lag: int | None) -> tuple[int, float, float]:
    n_obs = len(values)
    if n_obs == 0:
        return 0, np.nan, np.nan
    lag = _default_hac_lag(n_obs) if requested_lag is None else int(requested_lag)
    lag = max(0, min(lag, n_obs - 1))
    centered = values - np.mean(values)
    gamma0 = float(np.dot(centered, centered) / n_obs)
    long_run_variance = gamma0
    for lag_i in range(1, lag + 1):
        weight = 1.0 - lag_i / (lag + 1.0)
        gamma_i = float(np.dot(centered[lag_i:], centered[:-lag_i]) / n_obs)
        long_run_variance += 2.0 * weight * gamma_i
    if long_run_variance <= 0.0 or not np.isfinite(long_run_variance):
        return lag, np.nan, np.nan
    hac_se = float(np.sqrt(long_run_variance / n_obs))
    if hac_se <= 0.0 or not np.isfinite(hac_se):
        return lag, np.nan, np.nan
    return lag, hac_se, float(np.mean(values) / hac_se)


def _default_hac_lag(n_obs: int) -> int:
    if n_obs <= 1:
        return 0
    return max(1, int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))


def _moving_block_bootstrap_ci(values: np.ndarray, config: RankICConfig) -> tuple[int, float, float]:
    n_obs = len(values)
    if n_obs == 0 or config.bootstrap_iterations == 0:
        return 0, np.nan, np.nan
    block_length = (
        max(1, int(round(np.sqrt(n_obs))))
        if config.bootstrap_block_length is None
        else int(config.bootstrap_block_length)
    )
    block_length = max(1, min(block_length, n_obs))
    starts = np.arange(0, n_obs - block_length + 1)
    rng = np.random.default_rng(config.bootstrap_seed)
    means = np.empty(config.bootstrap_iterations, dtype=float)
    for i in range(config.bootstrap_iterations):
        draw: list[float] = []
        while len(draw) < n_obs:
            start = int(rng.choice(starts))
            draw.extend(values[start : start + block_length])
        means[i] = float(np.mean(draw[:n_obs]))
    alpha = 1.0 - config.ci_level
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return block_length, float(lower), float(upper)


def _normal_sf(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(0.5 * erfc(value / sqrt(2.0)))
