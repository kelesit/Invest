from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable

import numpy as np
import pandas as pd

ET_TZ = "America/New_York"
PROTOCOL_VERSION = "data_coverage_audit_protocol_v1"
OUTPUTS_SPEC_VERSION = "data_coverage_audit_outputs_spec_v1"

SECURITY_MASTER_COLUMNS = (
    "security_id",
    "symbol",
    "symbol_role",
    "instrument_type",
    "include_flag",
)
CALENDAR_COLUMNS = (
    "trading_date",
    "market_status",
    "session_open_et",
    "session_close_et",
    "expected_regular_minutes",
)
SPLIT_REFERENCE_COLUMNS = (
    "security_id",
    "ex_date",
    "split_factor",
    "cum_split_factor",
    "support_flag",
)
RAW_MINUTE_BAR_COLUMNS = (
    "security_id",
    "symbol",
    "raw_ts_source",
    "source_tz",
    "source_ts_convention",
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
    "volume_raw",
)


class MarketStatus(StrEnum):
    FULL_DAY = "full_day"
    HALF_DAY = "half_day"
    MARKET_CLOSED = "market_closed"


class SymbolRole(StrEnum):
    CONSTITUENT = "constituent"
    BENCHMARK = "benchmark"


class SymbolDayStatus(StrEnum):
    FULL_VALID = "full_valid"
    PARTIAL = "partial"
    MISSING = "missing"
    NOT_EXPECTED = "not_expected"


class FailureCode(StrEnum):
    NONE = "NONE"
    MISSING_ALL = "MISSING_ALL"
    MISSING_PARTIAL = "MISSING_PARTIAL"
    DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"
    OUTSIDE_SESSION = "OUTSIDE_SESSION"
    INVALID_PRICE = "INVALID_PRICE"
    INVALID_HILO = "INVALID_HILO"
    ADJUSTMENT_UNSUPPORTED = "ADJUSTMENT_UNSUPPORTED"
    MARKET_CLOSED = "MARKET_CLOSED"


class AuditDecision(StrEnum):
    GO = "GO"
    NO_GO = "NO_GO"


@dataclass(frozen=True)
class AuditThresholds:
    spy_coverage_min: float = 0.995
    breadth_median_min: float = 0.90
    breadth_p05_min: float = 0.80
    symbol_coverage_min: float = 0.95
    symbol_coverage_share_min: float = 0.80
    min_train_years: int = 4
    min_research_oos_years: int = 2
    holdout_years: int = 2

    @property
    def min_total_years(self) -> int:
        return self.min_train_years + self.min_research_oos_years + self.holdout_years


@dataclass(frozen=True)
class AuditRunContext:
    audit_run_id: str
    data_snapshot_id: str = "dev-snapshot"
    preregistration_id: str = "SP500RA-V1-R1"
    protocol_version: str = PROTOCOL_VERSION
    outputs_spec_version: str = OUTPUTS_SPEC_VERSION
    generated_at_utc: pd.Timestamp = field(
        default_factory=lambda: pd.Timestamp.now("UTC").floor("s")
    )

    def shared_metadata(self) -> dict[str, object]:
        return {
            "audit_run_id": self.audit_run_id,
            "protocol_version": self.protocol_version,
            "outputs_spec_version": self.outputs_spec_version,
            "preregistration_id": self.preregistration_id,
            "data_snapshot_id": self.data_snapshot_id,
            "generated_at_utc": self.generated_at_utc,
        }


@dataclass
class AuditBundle:
    normalized_session_minute_bars: pd.DataFrame
    adjusted_session_minute_bars: pd.DataFrame
    daily_bar_aggregates: pd.DataFrame
    symbol_day_quality_facts: pd.DataFrame
    outputs: dict[str, pd.DataFrame]


def _require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> pd.DataFrame:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    return df.copy()


def _normalize_date_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = pd.to_datetime(df[column]).dt.tz_localize(None).dt.normalize()
    return df


def _prepare_security_master(security_master: pd.DataFrame) -> pd.DataFrame:
    master = _require_columns(security_master, SECURITY_MASTER_COLUMNS, "security_master")
    master["include_flag"] = master["include_flag"].astype(bool)
    if not (
        (master["symbol"] == "SPY")
        & (master["symbol_role"] == SymbolRole.BENCHMARK.value)
        & master["include_flag"]
    ).any():
        raise ValueError("security_master must include SPY as the benchmark with include_flag=True.")
    return master.sort_values(["symbol_role", "symbol"]).reset_index(drop=True)


def _prepare_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    prepared = _require_columns(calendar, CALENDAR_COLUMNS, "calendar")
    prepared = _normalize_date_column(prepared, "trading_date")
    prepared["market_status"] = prepared["market_status"].astype(str)
    prepared["expected_regular_minutes"] = prepared["expected_regular_minutes"].astype(int)
    return prepared.sort_values("trading_date").reset_index(drop=True)


def _prepare_split_reference(split_reference: pd.DataFrame | None) -> pd.DataFrame:
    if split_reference is None or split_reference.empty:
        return pd.DataFrame(columns=SPLIT_REFERENCE_COLUMNS)
    prepared = _require_columns(
        split_reference,
        SPLIT_REFERENCE_COLUMNS,
        "split_reference",
    )
    prepared = _normalize_date_column(prepared, "ex_date")
    prepared["support_flag"] = prepared["support_flag"].astype(bool)
    prepared["split_factor"] = prepared["split_factor"].astype(float)
    prepared["cum_split_factor"] = prepared["cum_split_factor"].astype(float)
    return prepared.sort_values(["security_id", "ex_date"]).reset_index(drop=True)


def _convert_raw_timestamps_to_et(raw_bars: pd.DataFrame) -> pd.Series:
    source_ts = pd.to_datetime(raw_bars["raw_ts_source"], errors="raise")

    if getattr(source_ts.dt, "tz", None) is not None:
        return source_ts.dt.tz_convert(ET_TZ)

    converted = pd.Series(index=raw_bars.index, dtype=f"datetime64[ns, {ET_TZ}]")
    for source_tz, index in raw_bars.groupby("source_tz").groups.items():
        if pd.isna(source_tz):
            raise ValueError("raw_minute_bars contains naive timestamps with missing source_tz.")
        localized = source_ts.loc[index].dt.tz_localize(str(source_tz)).dt.tz_convert(ET_TZ)
        converted.loc[index] = localized
    return converted


def _annotate_raw_minute_bars(raw_bars: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    bars = _require_columns(raw_bars, RAW_MINUTE_BAR_COLUMNS, "raw_minute_bars")
    bars = bars.copy()
    bars["ts_et"] = _convert_raw_timestamps_to_et(bars)

    convention = bars["source_ts_convention"].astype(str)
    if (convention == "unknown").any():
        raise ValueError("raw_minute_bars contains source_ts_convention='unknown'.")

    start_mask = convention == "bar_start"
    end_mask = convention == "bar_end"
    bars["bar_start_et"] = pd.Series(
        pd.NaT,
        index=bars.index,
        dtype=f"datetime64[ns, {ET_TZ}]",
    )
    bars["bar_end_et"] = pd.Series(
        pd.NaT,
        index=bars.index,
        dtype=f"datetime64[ns, {ET_TZ}]",
    )
    bars.loc[start_mask, "bar_start_et"] = bars.loc[start_mask, "ts_et"].dt.floor("min")
    bars.loc[start_mask, "bar_end_et"] = (
        bars.loc[start_mask, "bar_start_et"] + pd.Timedelta(minutes=1)
    )
    bars.loc[end_mask, "bar_end_et"] = bars.loc[end_mask, "ts_et"].dt.floor("min")
    bars.loc[end_mask, "bar_start_et"] = (
        bars.loc[end_mask, "bar_end_et"] - pd.Timedelta(minutes=1)
    )
    bars["trading_date"] = bars["bar_start_et"].dt.tz_localize(None).dt.normalize()

    annotated = bars.merge(calendar, on="trading_date", how="left", validate="m:1")
    if annotated["market_status"].isna().any():
        missing_dates = sorted(annotated.loc[annotated["market_status"].isna(), "trading_date"].unique())
        raise ValueError(f"calendar is missing trading_date rows for raw minute bars: {missing_dates}")

    annotated["session_open_ts_et"] = pd.Series(
        pd.NaT,
        index=annotated.index,
        dtype=f"datetime64[ns, {ET_TZ}]",
    )
    annotated["session_close_ts_et"] = pd.Series(
        pd.NaT,
        index=annotated.index,
        dtype=f"datetime64[ns, {ET_TZ}]",
    )
    open_mask = annotated["market_status"] != MarketStatus.MARKET_CLOSED.value
    session_open_delta = pd.to_timedelta(annotated.loc[open_mask, "session_open_et"])
    session_close_delta = pd.to_timedelta(annotated.loc[open_mask, "session_close_et"])
    session_base = pd.to_datetime(annotated.loc[open_mask, "trading_date"]).dt.tz_localize(ET_TZ)
    annotated.loc[open_mask, "session_open_ts_et"] = session_base + session_open_delta
    annotated.loc[open_mask, "session_close_ts_et"] = session_base + session_close_delta

    annotated["inside_regular_session_flag"] = False
    annotated.loc[open_mask, "inside_regular_session_flag"] = (
        (annotated.loc[open_mask, "bar_start_et"] >= annotated.loc[open_mask, "session_open_ts_et"])
        & (annotated.loc[open_mask, "bar_end_et"] <= annotated.loc[open_mask, "session_close_ts_et"])
    )

    minute_delta = (
        annotated.loc[open_mask, "bar_start_et"] - annotated.loc[open_mask, "session_open_ts_et"]
    )
    annotated["minute_index"] = pd.NA
    annotated.loc[open_mask, "minute_index"] = (
        minute_delta // pd.Timedelta(minutes=1)
    ).astype("Int64")
    return annotated.sort_values(["security_id", "bar_start_et", "raw_ts_source"]).reset_index(drop=True)


def normalize_raw_minute_bars(
    raw_bars: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    annotated = _annotate_raw_minute_bars(raw_bars, calendar)
    inside = annotated.loc[annotated["inside_regular_session_flag"]].copy()
    if inside.empty:
        return pd.DataFrame(
            columns=[
                "security_id",
                "symbol",
                "trading_date",
                "minute_index",
                "market_status",
                "expected_regular_minutes",
                "bar_start_et",
                "bar_end_et",
                "open_raw_norm",
                "high_raw_norm",
                "low_raw_norm",
                "close_raw_norm",
                "volume_raw_norm",
                "inside_regular_session_flag",
                "duplicate_source_count",
            ]
        )

    inside = inside.sort_values(["security_id", "trading_date", "minute_index", "bar_start_et"])
    grouped = (
        inside.groupby(
            [
                "security_id",
                "symbol",
                "trading_date",
                "minute_index",
                "market_status",
                "expected_regular_minutes",
            ],
            as_index=False,
        )
        .agg(
            bar_start_et=("bar_start_et", "min"),
            bar_end_et=("bar_end_et", "max"),
            open_raw_norm=("open_raw", "first"),
            high_raw_norm=("high_raw", "max"),
            low_raw_norm=("low_raw", "min"),
            close_raw_norm=("close_raw", "last"),
            volume_raw_norm=("volume_raw", "sum"),
            duplicate_source_count=("raw_ts_source", "size"),
        )
    )
    grouped["inside_regular_session_flag"] = True
    grouped["minute_index"] = grouped["minute_index"].astype(int)
    return grouped.sort_values(["security_id", "trading_date", "minute_index"]).reset_index(drop=True)


def _effective_split_factors(
    normalized_bars: pd.DataFrame,
    split_reference: pd.DataFrame,
) -> pd.DataFrame:
    pairs = normalized_bars[["security_id", "trading_date"]].drop_duplicates()
    if pairs.empty:
        return pd.DataFrame(
            columns=["security_id", "trading_date", "applied_cum_split_factor", "adjustment_support_flag"]
        )

    if split_reference.empty:
        factors = pairs.copy()
        factors["applied_cum_split_factor"] = 1.0
        factors["adjustment_support_flag"] = True
        return factors

    rows: list[dict[str, object]] = []
    for security_id, security_pairs in pairs.groupby("security_id"):
        events = split_reference.loc[split_reference["security_id"] == security_id].copy()
        support_ok = bool(events["support_flag"].all()) if not events.empty else True
        for trading_date in security_pairs["trading_date"].sort_values():
            if not support_ok:
                applied = np.nan
            else:
                future_splits = events.loc[events["ex_date"] > trading_date, "split_factor"]
                applied = float(future_splits.prod()) if not future_splits.empty else 1.0
            rows.append(
                {
                    "security_id": security_id,
                    "trading_date": trading_date,
                    "applied_cum_split_factor": applied,
                    "adjustment_support_flag": support_ok,
                }
            )
    return pd.DataFrame(rows)


def apply_split_adjustment(
    normalized_bars: pd.DataFrame,
    split_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    normalized = normalized_bars.copy()
    if normalized.empty:
        return pd.DataFrame(
            columns=[
                "security_id",
                "trading_date",
                "minute_index",
                "open_adj",
                "high_adj",
                "low_adj",
                "close_adj",
                "volume_adj",
                "adjustment_support_flag",
                "applied_cum_split_factor",
            ]
        )

    split_reference = _prepare_split_reference(split_reference)
    factors = _effective_split_factors(normalized, split_reference)
    adjusted = normalized.merge(factors, on=["security_id", "trading_date"], how="left")
    adjusted["adjustment_support_flag"] = adjusted["adjustment_support_flag"].fillna(False)
    factor = adjusted["applied_cum_split_factor"].fillna(np.nan)
    adjusted["open_adj"] = adjusted["open_raw_norm"] / factor
    adjusted["high_adj"] = adjusted["high_raw_norm"] / factor
    adjusted["low_adj"] = adjusted["low_raw_norm"] / factor
    adjusted["close_adj"] = adjusted["close_raw_norm"] / factor
    adjusted["volume_adj"] = adjusted["volume_raw_norm"] * factor
    return adjusted.sort_values(["security_id", "trading_date", "minute_index"]).reset_index(drop=True)


def aggregate_daily_bars(
    adjusted_bars: pd.DataFrame,
    normalized_bars: pd.DataFrame,
) -> pd.DataFrame:
    if adjusted_bars.empty:
        return pd.DataFrame(
            columns=[
                "security_id",
                "symbol",
                "trading_date",
                "market_status",
                "n_minutes_present",
                "open_d",
                "high_d",
                "low_d",
                "close_d",
                "volume_d",
                "bar_dollar_volume_d",
                "adjustment_support_flag",
            ]
        )

    enriched = adjusted_bars.copy()
    daily = (
        enriched.groupby(
            ["security_id", "symbol", "trading_date", "market_status"],
            as_index=False,
        )
        .agg(
            n_minutes_present=("minute_index", "size"),
            open_d=("open_adj", "first"),
            high_d=("high_adj", "max"),
            low_d=("low_adj", "min"),
            close_d=("close_adj", "last"),
            volume_d=("volume_adj", "sum"),
            bar_dollar_volume_d=("close_adj", lambda s: 0.0),
            adjustment_support_flag=("adjustment_support_flag", "all"),
        )
    )

    dollar_volume = (
        enriched.assign(_bar_dollar_volume=enriched["close_adj"] * enriched["volume_adj"])
        .groupby(["security_id", "trading_date"], as_index=False)["_bar_dollar_volume"]
        .sum()
        .rename(columns={"_bar_dollar_volume": "bar_dollar_volume_d"})
    )
    daily = daily.drop(columns=["bar_dollar_volume_d"]).merge(
        dollar_volume,
        on=["security_id", "trading_date"],
        how="left",
        validate="1:1",
    )
    return daily.sort_values(["security_id", "trading_date"]).reset_index(drop=True)


def build_symbol_day_quality_facts(
    security_master: pd.DataFrame,
    calendar: pd.DataFrame,
    raw_bars: pd.DataFrame,
    normalized_bars: pd.DataFrame,
    adjusted_bars: pd.DataFrame,
) -> pd.DataFrame:
    master = _prepare_security_master(security_master)
    cal = _prepare_calendar(calendar)
    annotated = _annotate_raw_minute_bars(raw_bars, cal)
    normalized = normalized_bars.copy()
    adjusted = adjusted_bars.copy()

    included = master.loc[master["include_flag"], ["security_id", "symbol", "symbol_role"]].copy()
    base = included.assign(_key=1).merge(
        cal.assign(_key=1),
        on="_key",
        how="outer",
    ).drop(columns="_key")

    observed_minutes = (
        normalized.groupby(["security_id", "trading_date"], as_index=False)
        .agg(
            observed_regular_minutes=("minute_index", "size"),
            first_bar_ts_et=("bar_start_et", "min"),
            last_bar_ts_et=("bar_end_et", "max"),
            duplicate_timestamp_count=("duplicate_source_count", lambda s: int((s - 1).clip(lower=0).sum())),
        )
    )

    outside_counts = (
        annotated.loc[
            (annotated["market_status"] != MarketStatus.MARKET_CLOSED.value)
            & (~annotated["inside_regular_session_flag"])
        ]
        .groupby(["security_id", "trading_date"], as_index=False)
        .size()
        .rename(columns={"size": "outside_session_bar_count"})
    )

    inside_annotated = annotated.loc[
        (annotated["market_status"] != MarketStatus.MARKET_CLOSED.value)
        & (annotated["inside_regular_session_flag"])
    ].copy()
    invalid_price_mask = (
        (inside_annotated["open_raw"] <= 0)
        | (inside_annotated["high_raw"] <= 0)
        | (inside_annotated["low_raw"] <= 0)
        | (inside_annotated["close_raw"] <= 0)
        | (inside_annotated["volume_raw"] < 0)
    )
    invalid_hilo_mask = (
        (inside_annotated["low_raw"] > inside_annotated[["open_raw", "close_raw", "high_raw"]].min(axis=1))
        | (inside_annotated["high_raw"] < inside_annotated[["open_raw", "close_raw", "low_raw"]].max(axis=1))
    )
    invalid_counts = (
        inside_annotated.assign(
            invalid_price=invalid_price_mask.astype(int),
            invalid_hilo=invalid_hilo_mask.astype(int),
        )
        .groupby(["security_id", "trading_date"], as_index=False)
        .agg(
            invalid_price_count=("invalid_price", "sum"),
            invalid_hilo_count=("invalid_hilo", "sum"),
        )
    )

    adjustment_support = (
        adjusted.groupby(["security_id", "trading_date"], as_index=False)
        .agg(adjustment_support_flag=("adjustment_support_flag", "all"))
    )

    facts = (
        base.merge(observed_minutes, on=["security_id", "trading_date"], how="left")
        .merge(outside_counts, on=["security_id", "trading_date"], how="left")
        .merge(invalid_counts, on=["security_id", "trading_date"], how="left")
        .merge(adjustment_support, on=["security_id", "trading_date"], how="left")
    )

    fill_zero = [
        "observed_regular_minutes",
        "duplicate_timestamp_count",
        "outside_session_bar_count",
        "invalid_price_count",
        "invalid_hilo_count",
    ]
    for column in fill_zero:
        facts[column] = facts[column].fillna(0).astype(int)
    facts["adjustment_support_flag"] = facts["adjustment_support_flag"].fillna(
        facts["market_status"] == MarketStatus.MARKET_CLOSED.value
    )

    statuses: list[str] = []
    primary_codes: list[str | None] = []
    secondary_codes: list[str | None] = []
    for row in facts.itertuples(index=False):
        reasons: list[str] = []
        if row.market_status == MarketStatus.MARKET_CLOSED.value:
            statuses.append(SymbolDayStatus.NOT_EXPECTED.value)
            primary_codes.append(FailureCode.MARKET_CLOSED.value)
            secondary_codes.append(None)
            continue

        if row.observed_regular_minutes == 0:
            statuses.append(SymbolDayStatus.MISSING.value)
            primary_codes.append(FailureCode.MISSING_ALL.value)
            secondary_codes.append(None)
            continue

        if not bool(row.adjustment_support_flag):
            reasons.append(FailureCode.ADJUSTMENT_UNSUPPORTED.value)
        if row.observed_regular_minutes < row.expected_regular_minutes:
            reasons.append(FailureCode.MISSING_PARTIAL.value)
        if row.duplicate_timestamp_count > 0:
            reasons.append(FailureCode.DUPLICATE_TIMESTAMP.value)
        if row.invalid_price_count > 0:
            reasons.append(FailureCode.INVALID_PRICE.value)
        if row.invalid_hilo_count > 0:
            reasons.append(FailureCode.INVALID_HILO.value)

        if reasons:
            statuses.append(SymbolDayStatus.PARTIAL.value)
            primary_codes.append(reasons[0])
            secondary_codes.append(reasons[1] if len(reasons) > 1 else None)
        else:
            statuses.append(SymbolDayStatus.FULL_VALID.value)
            primary_codes.append(FailureCode.NONE.value)
            secondary_codes.append(None)

    facts["symbol_day_status"] = statuses
    facts["failure_code_primary"] = primary_codes
    facts["failure_code_secondary"] = secondary_codes
    return facts.sort_values(["security_id", "trading_date"]).reset_index(drop=True)


def build_daily_breadth_summary(
    security_master: pd.DataFrame,
    quality_facts: pd.DataFrame,
) -> pd.DataFrame:
    master = _prepare_security_master(security_master)
    constituents = set(
        master.loc[
            master["include_flag"] & (master["symbol_role"] == SymbolRole.CONSTITUENT.value),
            "security_id",
        ]
    )
    facts = quality_facts.loc[
        quality_facts["security_id"].isin(constituents)
        & (quality_facts["market_status"] != MarketStatus.MARKET_CLOSED.value)
    ].copy()

    breadth = (
        facts.groupby(["trading_date", "market_status"], as_index=False)
        .agg(
            n_constituents_expected=("security_id", "size"),
            n_full_valid=("symbol_day_status", lambda s: int((s == SymbolDayStatus.FULL_VALID.value).sum())),
            n_partial=("symbol_day_status", lambda s: int((s == SymbolDayStatus.PARTIAL.value).sum())),
            n_missing=("symbol_day_status", lambda s: int((s == SymbolDayStatus.MISSING.value).sum())),
        )
    )
    breadth["breadth"] = breadth["n_full_valid"] / breadth["n_constituents_expected"]
    return breadth.sort_values("trading_date").reset_index(drop=True)


def build_symbol_coverage_summary(
    security_master: pd.DataFrame,
    quality_facts: pd.DataFrame,
) -> pd.DataFrame:
    master = _prepare_security_master(security_master)
    constituents = master.loc[
        master["include_flag"] & (master["symbol_role"] == SymbolRole.CONSTITUENT.value),
        ["security_id", "symbol"],
    ].copy()
    facts = quality_facts.loc[
        quality_facts["security_id"].isin(constituents["security_id"])
        & (quality_facts["market_status"] != MarketStatus.MARKET_CLOSED.value)
    ].copy()

    def _max_consecutive(mask: pd.Series) -> int:
        max_run = 0
        current = 0
        for flag in mask.tolist():
            if flag:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return max_run

    rows: list[dict[str, object]] = []
    for security_id, group in facts.groupby("security_id"):
        group = group.sort_values("trading_date")
        n_expected_days = len(group)
        n_full_valid_days = int((group["symbol_day_status"] == SymbolDayStatus.FULL_VALID.value).sum())
        n_partial_days = int((group["symbol_day_status"] == SymbolDayStatus.PARTIAL.value).sum())
        n_missing_days = int((group["symbol_day_status"] == SymbolDayStatus.MISSING.value).sum())
        coverage_ratio = n_full_valid_days / n_expected_days if n_expected_days else 0.0
        rows.append(
            {
                "security_id": security_id,
                "symbol": group["symbol"].iloc[0],
                "n_expected_days": n_expected_days,
                "n_full_valid_days": n_full_valid_days,
                "n_partial_days": n_partial_days,
                "n_missing_days": n_missing_days,
                "coverage_ratio": coverage_ratio,
                "max_consecutive_missing_days": _max_consecutive(
                    group["symbol_day_status"] == SymbolDayStatus.MISSING.value
                ),
                "max_consecutive_nonfull_days": _max_consecutive(
                    group["symbol_day_status"] != SymbolDayStatus.FULL_VALID.value
                ),
                "coverage_pass_95": coverage_ratio >= 0.95,
            }
        )
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def build_adjustment_support_summary(
    security_master: pd.DataFrame,
    quality_facts: pd.DataFrame,
) -> pd.DataFrame:
    master = _prepare_security_master(security_master)
    included = master.loc[master["include_flag"], ["security_id", "symbol"]]
    facts = quality_facts.loc[
        quality_facts["security_id"].isin(included["security_id"])
        & (quality_facts["market_status"] != MarketStatus.MARKET_CLOSED.value)
    ].copy()

    rows: list[dict[str, object]] = []
    for security_id, group in facts.groupby("security_id"):
        supported_dates = group.loc[group["adjustment_support_flag"], "trading_date"].sort_values()
        pass_flag = bool(group["adjustment_support_flag"].all())
        rows.append(
            {
                "security_id": security_id,
                "symbol": group["symbol"].iloc[0],
                "adjustment_support_pass": pass_flag,
                "first_supported_date": supported_dates.iloc[0] if not supported_dates.empty else pd.NaT,
                "last_supported_date": supported_dates.iloc[-1] if not supported_dates.empty else pd.NaT,
                "adjustment_note": None if pass_flag else "at least one trading day lacks split support",
            }
        )
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def _first_open_date_on_or_after(open_dates: list[pd.Timestamp], cutoff: pd.Timestamp) -> pd.Timestamp | None:
    for date in open_dates:
        if date >= cutoff:
            return date
    return None


def _previous_open_date(open_dates: list[pd.Timestamp], current: pd.Timestamp) -> pd.Timestamp | None:
    previous = [date for date in open_dates if date < current]
    if not previous:
        return None
    return previous[-1]


def decide_sample_boundary(
    security_master: pd.DataFrame,
    calendar: pd.DataFrame,
    quality_facts: pd.DataFrame,
    *,
    context: AuditRunContext | None = None,
    thresholds: AuditThresholds | None = None,
) -> pd.DataFrame:
    context = context or AuditRunContext(audit_run_id="dev-audit")
    thresholds = thresholds or AuditThresholds()
    master = _prepare_security_master(security_master)
    cal = _prepare_calendar(calendar)

    include_ids = set(master.loc[master["include_flag"], "security_id"])
    benchmark_id = master.loc[
        master["include_flag"] & (master["symbol_role"] == SymbolRole.BENCHMARK.value),
        "security_id",
    ].iloc[0]
    constituent_ids = set(
        master.loc[
            master["include_flag"] & (master["symbol_role"] == SymbolRole.CONSTITUENT.value),
            "security_id",
        ]
    )

    open_dates = sorted(
        cal.loc[cal["market_status"] != MarketStatus.MARKET_CLOSED.value, "trading_date"].tolist()
    )
    benchmark_facts = quality_facts.loc[
        (quality_facts["security_id"] == benchmark_id)
        & (quality_facts["market_status"] != MarketStatus.MARKET_CLOSED.value)
    ].copy()

    candidate_end_series = benchmark_facts.loc[
        (benchmark_facts["symbol_day_status"] == SymbolDayStatus.FULL_VALID.value)
        & benchmark_facts["adjustment_support_flag"],
        "trading_date",
    ]
    if candidate_end_series.empty:
        return _decision_frame(
            context,
            decision=AuditDecision.NO_GO,
            decision_reason="benchmark continuity never reaches a full_valid, adjustment-supported end date",
            benchmark_gate_pass=False,
            breadth_gate_pass=False,
            symbol_continuity_gate_pass=False,
            adjustment_gate_pass=False,
            span_gate_pass=False,
        )

    candidate_end = candidate_end_series.max()
    latest_possible_start = candidate_end - pd.DateOffset(years=thresholds.min_total_years)
    start_candidates = [date for date in open_dates if date <= latest_possible_start]
    if not start_candidates:
        return _decision_frame(
            context,
            decision=AuditDecision.NO_GO,
            decision_reason="raw sample span is shorter than the preregistered minimum total span",
            benchmark_gate_pass=False,
            breadth_gate_pass=False,
            symbol_continuity_gate_pass=False,
            adjustment_gate_pass=False,
            span_gate_pass=False,
        )

    last_gate_result: dict[str, bool] | None = None
    last_reason = "no candidate interval satisfied every audit gate"

    for candidate_start in start_candidates:
        interval_dates = [date for date in open_dates if candidate_start <= date <= candidate_end]
        if not interval_dates:
            continue

        holdout_start = _first_open_date_on_or_after(
            interval_dates,
            candidate_end - pd.DateOffset(years=thresholds.holdout_years),
        )
        if holdout_start is None:
            last_gate_result = {
                "benchmark": False,
                "breadth": False,
                "symbol": False,
                "adjustment": False,
                "span": False,
            }
            last_reason = "cannot locate a final holdout start inside the candidate interval"
            continue

        research_end = _previous_open_date(interval_dates, holdout_start)
        research_start = candidate_start
        span_gate_pass = (
            candidate_end >= candidate_start + pd.DateOffset(years=thresholds.min_total_years)
            and research_end is not None
            and research_end >= research_start + pd.DateOffset(
                years=thresholds.min_train_years + thresholds.min_research_oos_years
            )
        )

        interval_slice = quality_facts.loc[
            quality_facts["security_id"].isin(include_ids)
            & quality_facts["trading_date"].isin(interval_dates)
            & (quality_facts["market_status"] != MarketStatus.MARKET_CLOSED.value)
        ].copy()

        bench_slice = interval_slice.loc[interval_slice["security_id"] == benchmark_id].sort_values(
            "trading_date"
        )
        benchmark_coverage = float(
            (bench_slice["symbol_day_status"] == SymbolDayStatus.FULL_VALID.value).mean()
        )
        benchmark_missing = bench_slice["symbol_day_status"] == SymbolDayStatus.MISSING.value
        max_consecutive_missing = _max_consecutive_run(benchmark_missing.tolist())
        holdout_dates = [date for date in interval_dates if date >= holdout_start]
        holdout_missing_day_count = int(
            bench_slice.loc[bench_slice["trading_date"].isin(holdout_dates), "symbol_day_status"]
            .eq(SymbolDayStatus.MISSING.value)
            .sum()
        )
        benchmark_gate_pass = (
            benchmark_coverage >= thresholds.spy_coverage_min
            and max_consecutive_missing < 2
            and holdout_missing_day_count == 0
        )

        breadth_rows = []
        for trading_date, group in interval_slice.loc[
            interval_slice["security_id"].isin(constituent_ids)
        ].groupby("trading_date"):
            expected = len(group)
            full_valid = int((group["symbol_day_status"] == SymbolDayStatus.FULL_VALID.value).sum())
            breadth_rows.append(full_valid / expected if expected else 0.0)
        if breadth_rows:
            breadth_series = pd.Series(breadth_rows, dtype=float)
            breadth_gate_pass = (
                breadth_series.median() >= thresholds.breadth_median_min
                and breadth_series.quantile(0.05) >= thresholds.breadth_p05_min
            )
        else:
            breadth_gate_pass = False

        coverage_rows = []
        adjustment_ok = True
        for security_id, group in interval_slice.loc[
            interval_slice["security_id"].isin(constituent_ids)
        ].groupby("security_id"):
            coverage_ratio = float(
                (group["symbol_day_status"] == SymbolDayStatus.FULL_VALID.value).mean()
            )
            coverage_rows.append(coverage_ratio)
            if not bool(group["adjustment_support_flag"].all()):
                adjustment_ok = False
        symbol_continuity_gate_pass = False
        if coverage_rows:
            coverage_array = np.array(coverage_rows, dtype=float)
            symbol_continuity_gate_pass = bool(
                (coverage_array >= thresholds.symbol_coverage_min).mean()
                >= thresholds.symbol_coverage_share_min
            )

        benchmark_adjustment_ok = bool(bench_slice["adjustment_support_flag"].all())
        adjustment_gate_pass = adjustment_ok and benchmark_adjustment_ok

        gate_result = {
            "benchmark": benchmark_gate_pass,
            "breadth": breadth_gate_pass,
            "symbol": symbol_continuity_gate_pass,
            "adjustment": adjustment_gate_pass,
            "span": span_gate_pass,
        }
        if all(gate_result.values()):
            return _decision_frame(
                context,
                decision=AuditDecision.GO,
                candidate_raw_start=candidate_start,
                candidate_raw_end=candidate_end,
                research_period_start=research_start,
                research_period_end=research_end,
                final_holdout_start=holdout_start,
                final_holdout_end=candidate_end,
                decision_reason="candidate interval satisfies every preregistered audit gate",
                benchmark_gate_pass=benchmark_gate_pass,
                breadth_gate_pass=breadth_gate_pass,
                symbol_continuity_gate_pass=symbol_continuity_gate_pass,
                adjustment_gate_pass=adjustment_gate_pass,
                span_gate_pass=span_gate_pass,
            )

        last_gate_result = gate_result
        failing = [name for name, passed in gate_result.items() if not passed]
        last_reason = f"candidate interval failed gates: {', '.join(failing)}"

    return _decision_frame(
        context,
        decision=AuditDecision.NO_GO,
        decision_reason=last_reason,
        benchmark_gate_pass=bool(last_gate_result and last_gate_result["benchmark"]),
        breadth_gate_pass=bool(last_gate_result and last_gate_result["breadth"]),
        symbol_continuity_gate_pass=bool(last_gate_result and last_gate_result["symbol"]),
        adjustment_gate_pass=bool(last_gate_result and last_gate_result["adjustment"]),
        span_gate_pass=bool(last_gate_result and last_gate_result["span"]),
    )


def _decision_frame(
    context: AuditRunContext,
    *,
    decision: AuditDecision,
    decision_reason: str,
    benchmark_gate_pass: bool,
    breadth_gate_pass: bool,
    symbol_continuity_gate_pass: bool,
    adjustment_gate_pass: bool,
    span_gate_pass: bool,
    candidate_raw_start: pd.Timestamp | None = None,
    candidate_raw_end: pd.Timestamp | None = None,
    research_period_start: pd.Timestamp | None = None,
    research_period_end: pd.Timestamp | None = None,
    final_holdout_start: pd.Timestamp | None = None,
    final_holdout_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    row = {
        **context.shared_metadata(),
        "decision": decision.value,
        "candidate_raw_start": candidate_raw_start,
        "candidate_raw_end": candidate_raw_end,
        "research_period_start": research_period_start,
        "research_period_end": research_period_end,
        "final_holdout_start": final_holdout_start,
        "final_holdout_end": final_holdout_end,
        "benchmark_gate_pass": benchmark_gate_pass,
        "breadth_gate_pass": breadth_gate_pass,
        "symbol_continuity_gate_pass": symbol_continuity_gate_pass,
        "adjustment_gate_pass": adjustment_gate_pass,
        "span_gate_pass": span_gate_pass,
        "decision_reason": decision_reason,
    }
    return pd.DataFrame([row])


def _max_consecutive_run(flags: list[bool]) -> int:
    max_run = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def build_spy_coverage_summary(
    security_master: pd.DataFrame,
    quality_facts: pd.DataFrame,
    sample_boundary_decision: pd.DataFrame,
) -> pd.DataFrame:
    master = _prepare_security_master(security_master)
    benchmark_id = master.loc[
        master["include_flag"] & (master["symbol_role"] == SymbolRole.BENCHMARK.value),
        "security_id",
    ].iloc[0]
    benchmark_symbol = master.loc[master["security_id"] == benchmark_id, "symbol"].iloc[0]
    benchmark = quality_facts.loc[
        (quality_facts["security_id"] == benchmark_id)
        & (quality_facts["market_status"] != MarketStatus.MARKET_CLOSED.value)
    ].sort_values("trading_date")
    n_expected_days = len(benchmark)
    n_full_valid_days = int((benchmark["symbol_day_status"] == SymbolDayStatus.FULL_VALID.value).sum())
    n_partial_days = int((benchmark["symbol_day_status"] == SymbolDayStatus.PARTIAL.value).sum())
    n_missing_days = int((benchmark["symbol_day_status"] == SymbolDayStatus.MISSING.value).sum())
    coverage_ratio = n_full_valid_days / n_expected_days if n_expected_days else 0.0

    holdout_start = sample_boundary_decision["final_holdout_start"].iloc[0]
    holdout_end = sample_boundary_decision["final_holdout_end"].iloc[0]
    holdout_missing_day_count = 0
    if pd.notna(holdout_start) and pd.notna(holdout_end):
        holdout_missing_day_count = int(
            benchmark.loc[
                benchmark["trading_date"].between(holdout_start, holdout_end),
                "symbol_day_status",
            ]
            .eq(SymbolDayStatus.MISSING.value)
            .sum()
        )

    row = {
        "benchmark_symbol": benchmark_symbol,
        "n_expected_days": n_expected_days,
        "n_full_valid_days": n_full_valid_days,
        "n_partial_days": n_partial_days,
        "n_missing_days": n_missing_days,
        "coverage_ratio": coverage_ratio,
        "max_consecutive_missing_days": _max_consecutive_run(
            benchmark["symbol_day_status"].eq(SymbolDayStatus.MISSING.value).tolist()
        ),
        "holdout_missing_day_count": holdout_missing_day_count,
        "benchmark_gate_pass": bool(sample_boundary_decision["benchmark_gate_pass"].iloc[0]),
    }
    return pd.DataFrame([row])


def _attach_shared_metadata(
    df: pd.DataFrame,
    context: AuditRunContext,
) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        for key, value in context.shared_metadata().items():
            df[key] = pd.Series(dtype="object")
        return df
    metadata = context.shared_metadata()
    enriched = df.copy()
    for key, value in metadata.items():
        enriched.insert(len(enriched.columns), key, value)
    return enriched


def run_coverage_audit(
    security_master: pd.DataFrame,
    calendar: pd.DataFrame,
    split_reference: pd.DataFrame | None,
    raw_bars: pd.DataFrame,
    *,
    context: AuditRunContext | None = None,
    thresholds: AuditThresholds | None = None,
) -> AuditBundle:
    context = context or AuditRunContext(audit_run_id="dev-audit")
    thresholds = thresholds or AuditThresholds()

    master = _prepare_security_master(security_master)
    cal = _prepare_calendar(calendar)
    normalized = normalize_raw_minute_bars(raw_bars, cal)
    adjusted = apply_split_adjustment(normalized, split_reference)
    daily = aggregate_daily_bars(adjusted, normalized)
    quality = build_symbol_day_quality_facts(master, cal, raw_bars, normalized, adjusted)
    daily_breadth = build_daily_breadth_summary(master, quality)
    symbol_coverage = build_symbol_coverage_summary(master, quality)
    adjustment_support = build_adjustment_support_summary(master, quality)
    decision = decide_sample_boundary(
        master,
        cal,
        quality,
        context=context,
        thresholds=thresholds,
    )
    spy_coverage = build_spy_coverage_summary(master, quality, decision)

    universe_proxy_manifest = master[
        ["security_id", "symbol_role", "symbol", "instrument_type", "include_flag"]
    ].copy()
    universe_proxy_manifest["exclude_reason"] = np.where(
        universe_proxy_manifest["include_flag"],
        None,
        "excluded from current audit universe",
    )
    universe_proxy_manifest["universe_note"] = None

    trading_calendar_manifest = cal.copy()
    symbol_day_audit_table = quality[
        [
            "security_id",
            "symbol",
            "trading_date",
            "market_status",
            "expected_regular_minutes",
            "observed_regular_minutes",
            "symbol_day_status",
            "duplicate_timestamp_count",
            "outside_session_bar_count",
            "invalid_price_count",
            "invalid_hilo_count",
            "first_bar_ts_et",
            "last_bar_ts_et",
            "failure_code_primary",
            "failure_code_secondary",
            "adjustment_support_flag",
        ]
    ].copy()

    outputs = {
        "universe_proxy_manifest": _attach_shared_metadata(universe_proxy_manifest, context),
        "trading_calendar_manifest": _attach_shared_metadata(trading_calendar_manifest, context),
        "symbol_day_audit_table": _attach_shared_metadata(symbol_day_audit_table, context),
        "daily_breadth_summary": _attach_shared_metadata(daily_breadth, context),
        "symbol_coverage_summary": _attach_shared_metadata(symbol_coverage, context),
        "spy_coverage_summary": _attach_shared_metadata(spy_coverage, context),
        "adjustment_support_summary": _attach_shared_metadata(adjustment_support, context),
        "sample_boundary_decision": decision.copy(),
    }
    return AuditBundle(
        normalized_session_minute_bars=normalized,
        adjusted_session_minute_bars=adjusted,
        daily_bar_aggregates=daily,
        symbol_day_quality_facts=quality,
        outputs=outputs,
    )
