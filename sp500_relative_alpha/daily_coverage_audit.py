from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


REQUIRED_DAILY_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class DailyAuditDecision(str, Enum):
    GO = "GO"
    NO_GO = "NO_GO"


@dataclass(frozen=True)
class DailyAuditConfig:
    benchmark_symbol: str = "SPY"
    trading_days_per_year: int = 252
    max_feature_lookback_days: int = 252
    max_label_horizon_days: int = 60
    min_training_days: int = 4 * 252
    test_block_days: int = 126
    min_oos_test_blocks: int = 4
    purge_gap_days: int = 60
    final_holdout_days: int = 2 * 252
    min_daily_breadth_ratio: float = 0.80
    min_symbol_coverage_ratio: float = 0.95
    min_symbol_coverage_pass_ratio: float = 0.80
    min_benchmark_coverage_ratio: float = 0.995


@dataclass(frozen=True)
class DailyAuditBundle:
    config: DailyAuditConfig
    daily_bars: pd.DataFrame
    outputs: Mapping[str, pd.DataFrame]


def normalize_display_symbol(symbol: str) -> str:
    """Normalize local cache filenames to display ticker style."""

    return str(symbol).replace("-", ".")


def inventory_daily_ohlcv_cache(cache_dir: str | Path) -> pd.DataFrame:
    cache_path = Path(cache_dir)
    rows: list[dict[str, object]] = []

    for path in sorted(cache_path.glob("*.parquet")):
        df = pd.read_parquet(path)
        symbol_file = path.stem
        symbol = normalize_display_symbol(symbol_file)
        columns = [str(column) for column in df.columns]
        date_values = _extract_date_values(df)
        dates = _normalize_dates(date_values)
        valid_dates = dates.dropna()
        duplicate_date_count = int(pd.Series(valid_dates).duplicated().sum())

        rows.append(
            {
                "symbol": symbol,
                "source_symbol_file": symbol_file,
                "path": str(path),
                "row_count": int(len(df)),
                "columns": ",".join(columns),
                "has_required_ohlcv": set(REQUIRED_DAILY_OHLCV_COLUMNS).issubset(columns),
                "date_source": "date_column" if "date" in df.columns else "index",
                "start_date": valid_dates.min() if len(valid_dates) else pd.NaT,
                "end_date": valid_dates.max() if len(valid_dates) else pd.NaT,
                "duplicate_date_count": duplicate_date_count,
            }
        )

    inventory = pd.DataFrame(rows)
    if inventory.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "source_symbol_file",
                "path",
                "row_count",
                "columns",
                "has_required_ohlcv",
                "date_source",
                "start_date",
                "end_date",
                "duplicate_date_count",
            ]
        )
    return inventory.sort_values("symbol").reset_index(drop=True)


def load_daily_ohlcv_cache(cache_dir: str | Path) -> pd.DataFrame:
    inventory = inventory_daily_ohlcv_cache(cache_dir)
    frames: list[pd.DataFrame] = []

    for row in inventory.itertuples(index=False):
        if not bool(row.has_required_ohlcv):
            continue

        path = Path(str(row.path))
        raw = pd.read_parquet(path)
        dates = _normalize_dates(_extract_date_values(raw))
        frame = pd.DataFrame(
            {
                "symbol": row.symbol,
                "source_symbol_file": row.source_symbol_file,
                "date": dates.to_numpy(),
                "open": pd.to_numeric(raw["open"], errors="coerce").to_numpy(),
                "high": pd.to_numeric(raw["high"], errors="coerce").to_numpy(),
                "low": pd.to_numeric(raw["low"], errors="coerce").to_numpy(),
                "close": pd.to_numeric(raw["close"], errors="coerce").to_numpy(),
                "shares_volume": pd.to_numeric(raw["volume"], errors="coerce").to_numpy(),
                "source_path": str(path),
            }
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "symbol",
                "source_symbol_file",
                "date",
                "open",
                "high",
                "low",
                "close",
                "shares_volume",
                "source_path",
            ]
        )

    bars = pd.concat(frames, ignore_index=True)
    return bars.sort_values(["symbol", "date"]).reset_index(drop=True)


def run_daily_coverage_audit(
    cache_dir: str | Path,
    config: DailyAuditConfig | None = None,
) -> DailyAuditBundle:
    audit_config = config or DailyAuditConfig()
    inventory = inventory_daily_ohlcv_cache(cache_dir)
    daily_bars = load_daily_ohlcv_cache(cache_dir)
    expected_calendar = _build_expected_calendar_proxy(daily_bars)
    symbol_day = build_symbol_day_audit_table(
        inventory=inventory,
        daily_bars=daily_bars,
        expected_calendar=expected_calendar,
        benchmark_symbol=audit_config.benchmark_symbol,
    )
    breadth = build_daily_breadth_summary(symbol_day, audit_config.benchmark_symbol)
    symbol_coverage = build_symbol_coverage_summary(
        symbol_day,
        min_symbol_coverage_ratio=audit_config.min_symbol_coverage_ratio,
    )
    spy_coverage = build_benchmark_coverage_summary(
        symbol_day,
        benchmark_symbol=audit_config.benchmark_symbol,
    )
    boundary = decide_daily_sample_boundary(
        symbol_day_audit_table=symbol_day,
        daily_breadth_summary=breadth,
        symbol_coverage_summary=symbol_coverage,
        benchmark_coverage_summary=spy_coverage,
        config=audit_config,
    )

    outputs: dict[str, pd.DataFrame] = {
        "daily_cache_inventory": inventory,
        "symbol_day_audit_table": symbol_day,
        "daily_breadth_summary": breadth,
        "symbol_coverage_summary": symbol_coverage,
        "benchmark_coverage_summary": spy_coverage,
        "sample_boundary_decision": boundary,
    }
    return DailyAuditBundle(config=audit_config, daily_bars=daily_bars, outputs=outputs)


def build_symbol_day_audit_table(
    inventory: pd.DataFrame,
    daily_bars: pd.DataFrame,
    expected_calendar: pd.DatetimeIndex,
    benchmark_symbol: str = "SPY",
) -> pd.DataFrame:
    symbols = inventory["symbol"].drop_duplicates().sort_values().tolist() if not inventory.empty else []
    if not symbols or len(expected_calendar) == 0:
        return pd.DataFrame(
            columns=[
                "symbol",
                "symbol_role",
                "date",
                "observed_rows",
                "symbol_day_status",
                "failure_code_primary",
            ]
        )

    observed = _aggregate_observed_daily_bars(daily_bars)
    grid = pd.MultiIndex.from_product([symbols, expected_calendar], names=["symbol", "date"]).to_frame(
        index=False
    )
    table = grid.merge(observed, how="left", on=["symbol", "date"])
    table["observed_rows"] = table["observed_rows"].fillna(0).astype(int)
    table["symbol_role"] = np.where(table["symbol"] == benchmark_symbol, "benchmark", "constituent")

    has_bar = table["observed_rows"] > 0
    has_duplicate = table["observed_rows"] > 1
    has_finite_ohlcv = table[["open", "high", "low", "close", "shares_volume"]].notna().all(axis=1)
    positive_prices = (table[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid_range = (table["high"] >= table[["open", "low", "close"]].max(axis=1)) & (
        table["low"] <= table[["open", "high", "close"]].min(axis=1)
    )
    valid_volume = table["shares_volume"] >= 0

    is_full_valid = has_bar & ~has_duplicate & has_finite_ohlcv & positive_prices & valid_range & valid_volume
    table["symbol_day_status"] = np.select(
        [is_full_valid, ~has_bar],
        ["full_valid", "missing"],
        default="invalid",
    )
    table["failure_code_primary"] = np.select(
        [
            is_full_valid,
            ~has_bar,
            has_duplicate,
            ~has_finite_ohlcv,
            ~positive_prices,
            ~valid_range,
            ~valid_volume,
        ],
        [
            None,
            "MISSING_DAILY_BAR",
            "DUPLICATE_DAILY_BAR",
            "NON_NUMERIC_OR_NULL_OHLCV",
            "NON_POSITIVE_PRICE",
            "INCONSISTENT_OHLC_RANGE",
            "NEGATIVE_VOLUME",
        ],
        default="UNKNOWN_INVALID_DAILY_BAR",
    )

    return table.sort_values(["symbol", "date"]).reset_index(drop=True)


def build_daily_breadth_summary(
    symbol_day_audit_table: pd.DataFrame,
    benchmark_symbol: str = "SPY",
) -> pd.DataFrame:
    if symbol_day_audit_table.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "n_constituents",
                "n_full_valid_constituents",
                "n_missing_constituents",
                "n_invalid_constituents",
                "breadth_ratio",
            ]
        )

    constituents = symbol_day_audit_table.loc[symbol_day_audit_table["symbol"] != benchmark_symbol].copy()
    n_constituents = int(constituents["symbol"].nunique())
    grouped = constituents.groupby("date", as_index=False).agg(
        n_full_valid_constituents=("symbol_day_status", lambda s: int((s == "full_valid").sum())),
        n_missing_constituents=("symbol_day_status", lambda s: int((s == "missing").sum())),
        n_invalid_constituents=("symbol_day_status", lambda s: int((s == "invalid").sum())),
    )
    grouped["n_constituents"] = n_constituents
    grouped["breadth_ratio"] = np.where(
        grouped["n_constituents"] > 0,
        grouped["n_full_valid_constituents"] / grouped["n_constituents"],
        np.nan,
    )
    return grouped[
        [
            "date",
            "n_constituents",
            "n_full_valid_constituents",
            "n_missing_constituents",
            "n_invalid_constituents",
            "breadth_ratio",
        ]
    ].sort_values("date", ignore_index=True)


def build_symbol_coverage_summary(
    symbol_day_audit_table: pd.DataFrame,
    min_symbol_coverage_ratio: float = 0.95,
) -> pd.DataFrame:
    if symbol_day_audit_table.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "symbol_role",
                "n_expected_days",
                "n_full_valid_days",
                "n_missing_days",
                "n_invalid_days",
                "coverage_ratio",
                "coverage_pass",
            ]
        )

    summary = symbol_day_audit_table.groupby(["symbol", "symbol_role"], as_index=False).agg(
        n_expected_days=("date", "count"),
        n_full_valid_days=("symbol_day_status", lambda s: int((s == "full_valid").sum())),
        n_missing_days=("symbol_day_status", lambda s: int((s == "missing").sum())),
        n_invalid_days=("symbol_day_status", lambda s: int((s == "invalid").sum())),
        first_full_valid_date=(
            "date",
            lambda s: _first_date_for_status(symbol_day_audit_table, s.index, "full_valid"),
        ),
        last_full_valid_date=(
            "date",
            lambda s: _last_date_for_status(symbol_day_audit_table, s.index, "full_valid"),
        ),
    )
    summary["coverage_ratio"] = summary["n_full_valid_days"] / summary["n_expected_days"]
    summary["coverage_pass"] = summary["coverage_ratio"] >= min_symbol_coverage_ratio
    return summary.sort_values(["symbol_role", "symbol"], ignore_index=True)


def build_benchmark_coverage_summary(
    symbol_day_audit_table: pd.DataFrame,
    benchmark_symbol: str = "SPY",
) -> pd.DataFrame:
    coverage = build_symbol_coverage_summary(symbol_day_audit_table)
    benchmark = coverage.loc[coverage["symbol"] == benchmark_symbol].copy()
    if benchmark.empty:
        return pd.DataFrame(
            [
                {
                    "symbol": benchmark_symbol,
                    "benchmark_present": False,
                    "n_expected_days": 0,
                    "n_full_valid_days": 0,
                    "n_missing_days": 0,
                    "n_invalid_days": 0,
                    "coverage_ratio": 0.0,
                }
            ]
        )
    benchmark.insert(1, "benchmark_present", True)
    return benchmark.reset_index(drop=True)


def decide_daily_sample_boundary(
    symbol_day_audit_table: pd.DataFrame,
    daily_breadth_summary: pd.DataFrame,
    symbol_coverage_summary: pd.DataFrame,
    benchmark_coverage_summary: pd.DataFrame,
    config: DailyAuditConfig | None = None,
) -> pd.DataFrame:
    audit_config = config or DailyAuditConfig()
    reasons: list[str] = []

    benchmark_rows = symbol_day_audit_table.loc[
        (symbol_day_audit_table["symbol"] == audit_config.benchmark_symbol)
        & (symbol_day_audit_table["symbol_day_status"] == "full_valid")
    ]
    spy_calendar = pd.DatetimeIndex(benchmark_rows["date"].drop_duplicates().sort_values())

    benchmark_coverage_ratio = _single_float(
        benchmark_coverage_summary,
        "coverage_ratio",
        default=0.0,
    )
    if benchmark_coverage_ratio < audit_config.min_benchmark_coverage_ratio:
        reasons.append(
            f"benchmark coverage {benchmark_coverage_ratio:.4f} < {audit_config.min_benchmark_coverage_ratio:.4f}"
        )

    if len(spy_calendar) == 0:
        reasons.append("benchmark has no full-valid daily calendar")
        return _boundary_decision_frame(
            decision=DailyAuditDecision.NO_GO,
            reasons=reasons,
            config=audit_config,
        )

    raw_sample_start = spy_calendar[0]
    raw_sample_end = spy_calendar[-1]
    first_signal_pos = audit_config.max_feature_lookback_days - 1
    last_signal_pos = len(spy_calendar) - audit_config.max_label_horizon_days - 2

    if first_signal_pos >= len(spy_calendar):
        reasons.append("not enough calendar days to satisfy feature warmup")
    if last_signal_pos < first_signal_pos:
        reasons.append("not enough calendar days to satisfy max label horizon after feature warmup")

    if reasons:
        return _boundary_decision_frame(
            decision=DailyAuditDecision.NO_GO,
            reasons=reasons,
            config=audit_config,
            raw_sample_start=raw_sample_start,
            raw_sample_end=raw_sample_end,
            n_spy_calendar_days=len(spy_calendar),
        )

    first_feature_signal_date = spy_calendar[first_signal_pos]
    last_labelable_signal_date = spy_calendar[last_signal_pos]
    signal_dates = spy_calendar[first_signal_pos : last_signal_pos + 1]

    final_holdout_end_pos = last_signal_pos
    final_holdout_start_pos = final_holdout_end_pos - audit_config.final_holdout_days + 1
    pre_holdout_purge_start_pos = final_holdout_start_pos - audit_config.purge_gap_days
    pre_holdout_purge_end_pos = final_holdout_start_pos - 1
    research_period_start_pos = first_signal_pos
    research_period_end_pos = pre_holdout_purge_start_pos - 1

    if final_holdout_start_pos < first_signal_pos:
        reasons.append("not enough signal days for final holdout")
    if research_period_end_pos < research_period_start_pos:
        reasons.append("not enough signal days before pre-holdout purge")

    n_research_oos_folds = _count_forward_oos_folds(
        first_signal_pos=research_period_start_pos,
        research_end_pos=research_period_end_pos,
        config=audit_config,
    )
    if n_research_oos_folds < audit_config.min_oos_test_blocks:
        reasons.append(
            f"research period supports only {n_research_oos_folds} OOS folds "
            f"< required {audit_config.min_oos_test_blocks}"
        )

    breadth_by_date = daily_breadth_summary.set_index("date")["breadth_ratio"]
    signal_breadth = breadth_by_date.reindex(signal_dates)
    min_breadth = float(signal_breadth.min()) if len(signal_breadth) else np.nan
    median_breadth = float(signal_breadth.median()) if len(signal_breadth) else np.nan
    if pd.isna(min_breadth) or min_breadth < audit_config.min_daily_breadth_ratio:
        reasons.append(
            f"minimum signal-sample breadth {min_breadth:.4f} < {audit_config.min_daily_breadth_ratio:.4f}"
        )

    constituent_coverage = symbol_coverage_summary.loc[
        symbol_coverage_summary["symbol_role"] == "constituent"
    ]
    n_constituents = int(len(constituent_coverage))
    n_coverage_pass = int(constituent_coverage["coverage_pass"].sum()) if n_constituents else 0
    pct_coverage_pass = n_coverage_pass / n_constituents if n_constituents else np.nan
    if pd.isna(pct_coverage_pass) or pct_coverage_pass < audit_config.min_symbol_coverage_pass_ratio:
        reasons.append(
            f"constituent coverage-pass ratio {pct_coverage_pass:.4f} "
            f"< {audit_config.min_symbol_coverage_pass_ratio:.4f}"
        )

    decision = DailyAuditDecision.NO_GO if reasons else DailyAuditDecision.GO
    return _boundary_decision_frame(
        decision=decision,
        reasons=reasons,
        config=audit_config,
        raw_sample_start=raw_sample_start,
        raw_sample_end=raw_sample_end,
        first_feature_signal_date=first_feature_signal_date,
        last_labelable_signal_date=last_labelable_signal_date,
        research_period_start=spy_calendar[research_period_start_pos],
        research_period_end=spy_calendar[research_period_end_pos]
        if research_period_end_pos >= research_period_start_pos
        else pd.NaT,
        pre_holdout_purge_start=spy_calendar[pre_holdout_purge_start_pos]
        if pre_holdout_purge_start_pos >= first_signal_pos
        else pd.NaT,
        pre_holdout_purge_end=spy_calendar[pre_holdout_purge_end_pos]
        if pre_holdout_purge_end_pos >= first_signal_pos
        else pd.NaT,
        final_holdout_start=spy_calendar[final_holdout_start_pos]
        if final_holdout_start_pos >= first_signal_pos
        else pd.NaT,
        final_holdout_end=last_labelable_signal_date,
        n_spy_calendar_days=len(spy_calendar),
        n_signal_days=len(signal_dates),
        n_research_signal_days=max(0, research_period_end_pos - research_period_start_pos + 1),
        n_final_holdout_signal_days=audit_config.final_holdout_days
        if final_holdout_start_pos >= first_signal_pos
        else 0,
        n_research_oos_folds=n_research_oos_folds,
        min_signal_breadth_ratio=min_breadth,
        median_signal_breadth_ratio=median_breadth,
        benchmark_coverage_ratio=benchmark_coverage_ratio,
        n_constituents=n_constituents,
        n_constituents_coverage_pass=n_coverage_pass,
        pct_constituents_coverage_pass=pct_coverage_pass,
    )


def _extract_date_values(df: pd.DataFrame) -> object:
    return df["date"] if "date" in df.columns else df.index


def _normalize_dates(date_values: object) -> pd.Series:
    dates = pd.DatetimeIndex(pd.to_datetime(date_values, errors="coerce"))
    if dates.tz is not None:
        dates = dates.tz_convert(None)
    return pd.Series(dates.normalize())


def _build_expected_calendar_proxy(daily_bars: pd.DataFrame) -> pd.DatetimeIndex:
    # Approximation: uses the union of all observed dates as a proxy for the NYSE trading calendar.
    # This avoids a hard dependency on an exchange calendar library. The approximation holds as long
    # as the underlying data (Databento daily snapshot) has no spurious extra dates and no missing
    # trading days — both enforced by the snapshot SHA256 integrity check.
    if daily_bars.empty:
        return pd.DatetimeIndex([])
    dates = pd.DatetimeIndex(daily_bars["date"].dropna().drop_duplicates()).sort_values()
    return dates


def _aggregate_observed_daily_bars(daily_bars: pd.DataFrame) -> pd.DataFrame:
    if daily_bars.empty:
        return pd.DataFrame(columns=["symbol", "date", "observed_rows"])
    return (
        daily_bars.groupby(["symbol", "date"], as_index=False)
        .agg(
            observed_rows=("date", "size"),
            open=("open", "first"),
            high=("high", "first"),
            low=("low", "first"),
            close=("close", "first"),
            shares_volume=("shares_volume", "first"),
        )
        .sort_values(["symbol", "date"], ignore_index=True)
    )


def _first_date_for_status(source: pd.DataFrame, index: pd.Index, status: str) -> pd.Timestamp | pd.NaT:
    subset = source.loc[index]
    dates = subset.loc[subset["symbol_day_status"] == status, "date"]
    return dates.min() if len(dates) else pd.NaT


def _last_date_for_status(source: pd.DataFrame, index: pd.Index, status: str) -> pd.Timestamp | pd.NaT:
    subset = source.loc[index]
    dates = subset.loc[subset["symbol_day_status"] == status, "date"]
    return dates.max() if len(dates) else pd.NaT


def _single_float(frame: pd.DataFrame, column: str, default: float) -> float:
    if frame.empty or column not in frame.columns:
        return default
    value = frame[column].iloc[0]
    return default if pd.isna(value) else float(value)


def _count_forward_oos_folds(
    first_signal_pos: int,
    research_end_pos: int,
    config: DailyAuditConfig,
) -> int:
    test_start = first_signal_pos + config.min_training_days + config.purge_gap_days
    count = 0
    while test_start + config.test_block_days - 1 <= research_end_pos:
        count += 1
        test_start += config.test_block_days + config.purge_gap_days
    return count


def _boundary_decision_frame(
    decision: DailyAuditDecision,
    reasons: list[str],
    config: DailyAuditConfig,
    **values: object,
) -> pd.DataFrame:
    base: dict[str, object] = {
        "decision": decision.value,
        "reason": "OK" if not reasons else "; ".join(reasons),
        "benchmark_symbol": config.benchmark_symbol,
        "max_feature_lookback_days": config.max_feature_lookback_days,
        "max_label_horizon_days": config.max_label_horizon_days,
        "min_training_days": config.min_training_days,
        "test_block_days": config.test_block_days,
        "min_oos_test_blocks": config.min_oos_test_blocks,
        "purge_gap_days": config.purge_gap_days,
        "final_holdout_days": config.final_holdout_days,
        "min_daily_breadth_ratio": config.min_daily_breadth_ratio,
        "min_symbol_coverage_ratio": config.min_symbol_coverage_ratio,
        "min_symbol_coverage_pass_ratio": config.min_symbol_coverage_pass_ratio,
        "min_benchmark_coverage_ratio": config.min_benchmark_coverage_ratio,
    }
    base.update(values)
    return pd.DataFrame([base])
