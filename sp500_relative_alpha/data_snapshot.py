from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .daily_coverage_audit import (
    REQUIRED_DAILY_OHLCV_COLUMNS,
    inventory_daily_ohlcv_cache,
)


@dataclass(frozen=True)
class DailyDataSnapshot:
    snapshot_id: str
    cache_dir: Path
    created_at_utc: str
    manifest: pd.DataFrame
    summary: dict[str, object]


def build_daily_data_snapshot(
    cache_dir: str | Path,
    snapshot_id: str,
    created_at_utc: str | None = None,
) -> DailyDataSnapshot:
    cache_path = Path(cache_dir).resolve()
    created_at = created_at_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    inventory = inventory_daily_ohlcv_cache(cache_path)

    manifest_rows: list[dict[str, object]] = []
    for row in inventory.itertuples(index=False):
        path = Path(str(row.path)).resolve()
        manifest_rows.append(
            {
                "snapshot_id": snapshot_id,
                "symbol": row.symbol,
                "source_symbol_file": row.source_symbol_file,
                "relative_path": str(path.relative_to(cache_path)),
                "absolute_path_at_freeze": str(path),
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "row_count": int(row.row_count),
                "columns": row.columns,
                "has_required_ohlcv": bool(row.has_required_ohlcv),
                "start_date": _format_date(row.start_date),
                "end_date": _format_date(row.end_date),
                "duplicate_date_count": int(row.duplicate_date_count),
            }
        )

    manifest = pd.DataFrame(manifest_rows).sort_values("symbol").reset_index(drop=True)
    summary = summarize_daily_snapshot(
        snapshot_id=snapshot_id,
        cache_dir=cache_path,
        created_at_utc=created_at,
        manifest=manifest,
    )
    return DailyDataSnapshot(
        snapshot_id=snapshot_id,
        cache_dir=cache_path,
        created_at_utc=created_at,
        manifest=manifest,
        summary=summary,
    )


def materialize_daily_data_snapshot(
    snapshot: DailyDataSnapshot,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_path / "snapshot_manifest.csv"
    snapshot.manifest.to_csv(manifest_path, index=False)

    summary = dict(snapshot.summary)
    summary["snapshot_manifest_sha256"] = sha256_file(manifest_path)
    summary_path = output_path / "snapshot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "snapshot_manifest": manifest_path,
        "snapshot_summary": summary_path,
    }


def verify_daily_data_snapshot_manifest(
    manifest_path: str | Path,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    cache_path = Path(cache_dir).resolve() if cache_dir is not None else None
    rows: list[dict[str, object]] = []

    for row in manifest.itertuples(index=False):
        path = (
            cache_path / str(row.relative_path)
            if cache_path is not None
            else Path(str(row.absolute_path_at_freeze))
        )
        exists = path.exists()
        actual_sha256 = sha256_file(path) if exists else None
        rows.append(
            {
                "snapshot_id": row.snapshot_id,
                "symbol": row.symbol,
                "relative_path": row.relative_path,
                "path_checked": str(path),
                "exists": exists,
                "expected_sha256": row.sha256,
                "actual_sha256": actual_sha256,
                "sha256_match": bool(exists and actual_sha256 == row.sha256),
            }
        )

    return pd.DataFrame(rows)


def summarize_daily_snapshot(
    snapshot_id: str,
    cache_dir: Path,
    created_at_utc: str,
    manifest: pd.DataFrame,
) -> dict[str, object]:
    if manifest.empty:
        return {
            "snapshot_id": snapshot_id,
            "created_at_utc": created_at_utc,
            "cache_dir": str(cache_dir),
            "file_count": 0,
            "benchmark_present": False,
            "required_columns": list(REQUIRED_DAILY_OHLCV_COLUMNS),
            "all_files_have_required_ohlcv": False,
        }

    return {
        "snapshot_id": snapshot_id,
        "created_at_utc": created_at_utc,
        "cache_dir": str(cache_dir),
        "file_count": int(len(manifest)),
        "total_size_bytes": int(manifest["file_size_bytes"].sum()),
        "benchmark_present": bool((manifest["symbol"] == "SPY").any()),
        "required_columns": list(REQUIRED_DAILY_OHLCV_COLUMNS),
        "all_files_have_required_ohlcv": bool(manifest["has_required_ohlcv"].all()),
        "raw_sample_start": str(pd.to_datetime(manifest["start_date"]).min().date()),
        "raw_sample_end": str(pd.to_datetime(manifest["end_date"]).max().date()),
        "max_duplicate_date_count": int(manifest["duplicate_date_count"].max()),
    }


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_date(value: object) -> str | None:
    if pd.isna(value):
        return None
    return str(pd.Timestamp(value).date())
