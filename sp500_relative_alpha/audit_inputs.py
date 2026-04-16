from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_DAILY_COLUMNS = ("open", "high", "low", "close", "volume")


def normalize_display_symbol(symbol: str) -> str:
    """Normalize cache filenames to display symbols.

    Local cache files use `BRK-B.parquet` style filenames while the research
    docs use display symbols like `BRK.B`. We normalize only the display layer;
    this does not claim that ticker is a stable security identity.
    """

    return symbol.replace("-", ".")


@dataclass(frozen=True)
class LocalEquityCacheDiagnostic:
    inventory: pd.DataFrame
    security_master: pd.DataFrame
    summary: pd.DataFrame


def inventory_local_equity_cache(cache_dir: str | Path) -> pd.DataFrame:
    cache_path = Path(cache_dir)
    paths = sorted(cache_path.glob("*.parquet"))
    rows: list[dict[str, object]] = []

    for path in paths:
        df = pd.read_parquet(path)
        symbol_file = path.stem
        symbol = normalize_display_symbol(symbol_file)
        columns = [str(column) for column in df.columns]
        index = df.index
        is_datetime_index = isinstance(index, pd.DatetimeIndex)
        has_intraday_time = bool(is_datetime_index and index.normalize().nunique() != len(index))
        inferred_freq = pd.infer_freq(index[: min(len(index), 20)]) if is_datetime_index and len(index) >= 3 else None
        rows.append(
            {
                "symbol": symbol,
                "source_symbol_file": symbol_file,
                "path": str(path),
                "row_count": int(len(df)),
                "columns": ",".join(columns),
                "has_required_ohlcv": set(REQUIRED_DAILY_COLUMNS).issubset(columns),
                "index_type": type(index).__name__,
                "index_name": index.name,
                "start_ts": index.min() if len(index) else pd.NaT,
                "end_ts": index.max() if len(index) else pd.NaT,
                "is_datetime_index": is_datetime_index,
                "has_intraday_time": has_intraday_time,
                "frequency_guess": inferred_freq,
                "is_minute_candidate": bool(has_intraday_time),
            }
        )

    inventory = pd.DataFrame(rows)
    if inventory.empty:
        return inventory
    return inventory.sort_values("symbol").reset_index(drop=True)


def build_security_master_from_inventory(
    inventory: pd.DataFrame,
    benchmark_symbol: str = "SPY",
) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame(
            columns=[
                "security_id",
                "symbol",
                "symbol_role",
                "instrument_type",
                "include_flag",
                "security_status_note",
            ]
        )

    rows = []
    for symbol in inventory["symbol"]:
        symbol_role = "benchmark" if symbol == benchmark_symbol else "constituent"
        rows.append(
            {
                "security_id": f"cache::{symbol}",
                "symbol": symbol,
                "symbol_role": symbol_role,
                "instrument_type": "unknown_from_local_cache" if symbol == benchmark_symbol else "common_stock_proxy",
                "include_flag": True,
                "security_status_note": "derived from local parquet cache filename only",
            }
        )
    return pd.DataFrame(rows).sort_values(["symbol_role", "symbol"]).reset_index(drop=True)


def diagnose_local_equity_cache_for_round1(
    cache_dir: str | Path,
    benchmark_symbol: str = "SPY",
) -> LocalEquityCacheDiagnostic:
    inventory = inventory_local_equity_cache(cache_dir)
    security_master = build_security_master_from_inventory(inventory, benchmark_symbol=benchmark_symbol)

    file_count = int(len(inventory))
    benchmark_present = bool((inventory["symbol"] == benchmark_symbol).any()) if not inventory.empty else False
    all_have_ohlcv = bool(inventory["has_required_ohlcv"].all()) if not inventory.empty else False
    minute_candidate_count = int(inventory["is_minute_candidate"].sum()) if not inventory.empty else 0
    earliest_ts = inventory["start_ts"].min() if not inventory.empty else pd.NaT
    latest_ts = inventory["end_ts"].max() if not inventory.empty else pd.NaT

    if file_count == 0:
        verdict = "NO_GO"
        reason = "local cache directory is empty"
    elif not benchmark_present:
        verdict = "NO_GO"
        reason = "local cache is missing SPY, so benchmark anchor is absent"
    elif not all_have_ohlcv:
        verdict = "NO_GO"
        reason = "one or more parquet files do not contain the required OHLCV columns"
    elif minute_candidate_count == 0:
        verdict = "GO_FOR_DAILY_AUDIT"
        reason = (
            "local cache appears to be daily-only; this matches the current round1 daily OHLCV plan "
            "and should proceed to daily coverage audit"
        )
    else:
        verdict = "PARTIAL_GO"
        reason = "local cache includes intraday candidates, but current round1 uses daily OHLCV"

    summary = pd.DataFrame(
        [
            {
                "cache_dir": str(Path(cache_dir)),
                "file_count": file_count,
                "benchmark_symbol": benchmark_symbol,
                "benchmark_present": benchmark_present,
                "all_have_required_ohlcv": all_have_ohlcv,
                "minute_candidate_count": minute_candidate_count,
                "earliest_ts": earliest_ts,
                "latest_ts": latest_ts,
                "verdict": verdict,
                "reason": reason,
            }
        ]
    )
    return LocalEquityCacheDiagnostic(
        inventory=inventory,
        security_master=security_master,
        summary=summary,
    )
