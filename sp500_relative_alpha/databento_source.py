from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import dotenv_values

import databento as db


def load_databento_key(env_path: str | Path = ".env") -> str:
    key = dotenv_values(env_path).get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError(f"DATABENTO_API_KEY is missing in {env_path!s}.")
    return key


def create_historical_client(env_path: str | Path = ".env") -> db.Historical:
    return db.Historical(load_databento_key(env_path))


def build_security_master_from_symbols(
    symbols: Iterable[str],
    benchmark_symbol: str = "SPY",
) -> pd.DataFrame:
    unique_symbols = sorted(set(symbols))
    if benchmark_symbol not in unique_symbols:
        unique_symbols.append(benchmark_symbol)
        unique_symbols = sorted(set(unique_symbols))

    rows = []
    for symbol in unique_symbols:
        rows.append(
            {
                "security_id": f"raw_symbol::{symbol}",
                "symbol": symbol,
                "symbol_role": "benchmark" if symbol == benchmark_symbol else "constituent",
                "instrument_type": "unknown_from_databento_symbol",
                "include_flag": True,
                "security_status_note": "security identity is tied to raw_symbol in v1 smoke path",
            }
        )
    return pd.DataFrame(rows).sort_values(["symbol_role", "symbol"]).reset_index(drop=True)


def fetch_databento_ohlcv_1m_contract(
    symbols: Iterable[str],
    start: str,
    end: str,
    *,
    dataset: str = "EQUS.MINI",
    stype_in: str = "raw_symbol",
    env_path: str | Path = ".env",
    client: db.Historical | None = None,
) -> pd.DataFrame:
    """Fetch Databento 1-minute OHLCV bars and map them into the raw-minute contract.

    This is intentionally a thin source adapter:
    - it does not filter regular session
    - it does not apply split adjustment
    - it does not reinterpret timestamps beyond recording the known convention
    """

    client = client or create_historical_client(env_path)
    symbol_list = sorted(set(symbols))
    store = client.timeseries.get_range(
        dataset=dataset,
        schema="ohlcv-1m",
        symbols=symbol_list,
        start=start,
        end=end,
        stype_in=stype_in,
    )
    df = store.to_df()
    if df.empty:
        return pd.DataFrame(
            columns=[
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
                "vendor_note",
            ]
        )

    if "symbol" not in df.columns:
        raise ValueError("Databento ohlcv-1m response is missing the symbol column.")

    frame = df.reset_index().rename(
        columns={
            "ts_event": "raw_ts_source",
            "open": "open_raw",
            "high": "high_raw",
            "low": "low_raw",
            "close": "close_raw",
            "volume": "volume_raw",
        }
    )
    frame["raw_ts_source"] = pd.to_datetime(frame["raw_ts_source"], utc=True)
    frame["security_id"] = frame["symbol"].map(lambda symbol: f"raw_symbol::{symbol}")
    frame["source_tz"] = "UTC"
    frame["source_ts_convention"] = "bar_start"
    frame["vendor_note"] = f"databento dataset={dataset} schema=ohlcv-1m"

    return frame[
        [
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
            "vendor_note",
        ]
    ].sort_values(["symbol", "raw_ts_source"]).reset_index(drop=True)
