from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .daily_coverage_audit import REQUIRED_DAILY_OHLCV_COLUMNS
from .data_snapshot import verify_daily_data_snapshot_manifest


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
ROUND1_SNAPSHOT_ID = "local_equity_daily_20260415_v1"
ROUND1_SNAPSHOT_MANIFEST = (
    PACKAGE_ROOT
    / "artifacts"
    / "data_snapshots"
    / ROUND1_SNAPSHOT_ID
    / "snapshot_manifest.csv"
)
ROUND1_CACHE_DIR = PROJECT_ROOT / "data" / "equity"


class DailyDataLoadError(RuntimeError):
    """Raised when frozen daily data cannot be loaded without violating the contract."""


@dataclass(frozen=True)
class SnapshotDailyDataConfig:
    manifest_path: Path = ROUND1_SNAPSHOT_MANIFEST
    cache_dir: Path = ROUND1_CACHE_DIR
    verify_snapshot: bool = True
    symbols: tuple[str, ...] | None = None


def load_round1_daily_ohlcv(
    symbols: Iterable[str] | None = None,
    verify_snapshot: bool = True,
) -> pd.DataFrame:
    """Load the frozen Round 1 daily OHLCV snapshot."""

    normalized_symbols = tuple(symbols) if symbols is not None else None
    return load_snapshot_daily_ohlcv(
        SnapshotDailyDataConfig(
            symbols=normalized_symbols,
            verify_snapshot=verify_snapshot,
        )
    )


def load_snapshot_daily_ohlcv(config: SnapshotDailyDataConfig) -> pd.DataFrame:
    manifest_path = Path(config.manifest_path)
    cache_dir = Path(config.cache_dir)
    if not manifest_path.exists():
        raise DailyDataLoadError(f"snapshot manifest does not exist: {manifest_path}")
    if not cache_dir.exists():
        raise DailyDataLoadError(f"cache_dir does not exist: {cache_dir}")

    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise DailyDataLoadError(f"snapshot manifest is empty: {manifest_path}")

    if config.verify_snapshot:
        verification = verify_daily_data_snapshot_manifest(manifest_path, cache_dir=cache_dir)
        if not bool(verification["sha256_match"].all()):
            mismatches = verification.loc[~verification["sha256_match"]]
            preview = mismatches[["symbol", "relative_path", "exists"]].head(10).to_dict("records")
            raise DailyDataLoadError(
                "frozen data snapshot verification failed; refusing to load daily data. "
                f"mismatch_count={len(mismatches)}, preview={preview}"
            )

    requested_symbols = set(config.symbols) if config.symbols is not None else None
    if requested_symbols is not None:
        known_symbols = set(manifest["symbol"])
        unknown = sorted(requested_symbols - known_symbols)
        if unknown:
            raise DailyDataLoadError(f"requested symbols are not in snapshot manifest: {unknown}")
        manifest = manifest.loc[manifest["symbol"].isin(requested_symbols)].copy()

    frames: list[pd.DataFrame] = []
    for row in manifest.sort_values("symbol").itertuples(index=False):
        path = cache_dir / str(row.relative_path)
        raw = pd.read_parquet(path)
        _validate_raw_daily_frame(raw, symbol=str(row.symbol), path=path)
        dates = _normalize_dates(raw["date"] if "date" in raw.columns else raw.index)
        frame = pd.DataFrame(
            {
                "snapshot_id": str(row.snapshot_id),
                "symbol": str(row.symbol),
                "date": dates.to_numpy(),
                "open": pd.to_numeric(raw["open"], errors="coerce").to_numpy(),
                "high": pd.to_numeric(raw["high"], errors="coerce").to_numpy(),
                "low": pd.to_numeric(raw["low"], errors="coerce").to_numpy(),
                "close": pd.to_numeric(raw["close"], errors="coerce").to_numpy(),
                "shares_volume": pd.to_numeric(raw["volume"], errors="coerce").to_numpy(),
            }
        )
        frames.append(frame)

    if not frames:
        raise DailyDataLoadError("requested symbol filter produced no rows")

    daily = pd.concat(frames, ignore_index=True)
    daily = daily.sort_values(["symbol", "date"]).reset_index(drop=True)
    daily["typical_price"] = (daily["high"] + daily["low"] + daily["close"]) / 3.0
    daily["alpha_volume"] = daily["typical_price"] * daily["shares_volume"]
    daily["close_to_close_return"] = daily.groupby("symbol", sort=False)["close"].pct_change()
    validate_canonical_daily_ohlcv(daily)
    return daily


def validate_canonical_daily_ohlcv(daily: pd.DataFrame) -> None:
    required = {
        "snapshot_id",
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
    missing = sorted(required - set(daily.columns))
    if missing:
        raise DailyDataLoadError(f"canonical daily OHLCV is missing columns: {missing}")

    if daily[["symbol", "date"]].duplicated().any():
        duplicates = daily.loc[daily[["symbol", "date"]].duplicated(), ["symbol", "date"]].head(10)
        raise DailyDataLoadError(f"duplicate symbol-date rows detected: {duplicates.to_dict('records')}")

    if daily["date"].isna().any():
        raise DailyDataLoadError("canonical daily OHLCV contains null dates")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "shares_volume",
        "typical_price",
        "alpha_volume",
    ]
    if not np.isfinite(daily[numeric_columns].to_numpy()).all():
        raise DailyDataLoadError("canonical daily OHLCV contains non-finite OHLCV values")

    if (daily[["open", "high", "low", "close"]] <= 0).any().any():
        raise DailyDataLoadError("canonical daily OHLCV contains non-positive prices")
    if (daily["shares_volume"] < 0).any():
        raise DailyDataLoadError("canonical daily OHLCV contains negative shares_volume")

    invalid_range = (daily["high"] < daily[["open", "low", "close"]].max(axis=1)) | (
        daily["low"] > daily[["open", "high", "close"]].min(axis=1)
    )
    if invalid_range.any():
        preview = daily.loc[invalid_range, ["symbol", "date", "open", "high", "low", "close"]].head(10)
        raise DailyDataLoadError(f"inconsistent OHLC ranges detected: {preview.to_dict('records')}")

    expected_alpha_volume = ((daily["high"] + daily["low"] + daily["close"]) / 3.0) * daily[
        "shares_volume"
    ]
    if not np.allclose(daily["alpha_volume"], expected_alpha_volume, equal_nan=False):
        raise DailyDataLoadError("alpha_volume does not match frozen typical-price dollar-volume definition")


def _validate_raw_daily_frame(raw: pd.DataFrame, symbol: str, path: Path) -> None:
    missing = sorted(set(REQUIRED_DAILY_OHLCV_COLUMNS) - set(raw.columns))
    if missing:
        raise DailyDataLoadError(f"{symbol} at {path} is missing raw OHLCV columns: {missing}")


def _normalize_dates(date_values: object) -> pd.Series:
    dates = pd.DatetimeIndex(pd.to_datetime(date_values, errors="coerce"))
    if dates.tz is not None:
        dates = dates.tz_convert(None)
    return pd.Series(dates.normalize())
