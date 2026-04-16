from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sp500_relative_alpha.data_snapshot import (
    build_daily_data_snapshot,
    materialize_daily_data_snapshot,
    verify_daily_data_snapshot_manifest,
)


class DailyDataSnapshotTests(unittest.TestCase):
    def test_build_snapshot_records_file_hashes_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            df = pd.DataFrame(
                {
                    "open": [1.0, 2.0],
                    "high": [1.1, 2.1],
                    "low": [0.9, 1.9],
                    "close": [1.05, 2.05],
                    "volume": [100, 200],
                },
                index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            )
            df.index.name = "date"
            df.to_parquet(cache_dir / "AAPL.parquet")
            df.to_parquet(cache_dir / "SPY.parquet")

            snapshot = build_daily_data_snapshot(
                cache_dir,
                snapshot_id="unit_snapshot",
                created_at_utc="2026-04-15T00:00:00+00:00",
            )

        self.assertEqual(snapshot.summary["file_count"], 2)
        self.assertTrue(snapshot.summary["benchmark_present"])
        self.assertTrue(snapshot.summary["all_files_have_required_ohlcv"])
        self.assertEqual(snapshot.summary["raw_sample_start"], "2024-01-02")
        self.assertEqual(snapshot.summary["raw_sample_end"], "2024-01-03")
        self.assertEqual(set(snapshot.manifest["symbol"]), {"AAPL", "SPY"})
        self.assertTrue(snapshot.manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all())

    def test_materialize_snapshot_writes_manifest_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            df = pd.DataFrame(
                {
                    "open": [1.0],
                    "high": [1.1],
                    "low": [0.9],
                    "close": [1.05],
                    "volume": [100],
                },
                index=pd.to_datetime(["2024-01-02"]),
            )
            df.index.name = "date"
            df.to_parquet(cache_dir / "SPY.parquet")

            snapshot = build_daily_data_snapshot(
                cache_dir,
                snapshot_id="unit_snapshot",
                created_at_utc="2026-04-15T00:00:00+00:00",
            )
            paths = materialize_daily_data_snapshot(snapshot, root / "snapshot")

            manifest = pd.read_csv(paths["snapshot_manifest"])
            summary = json.loads(paths["snapshot_summary"].read_text(encoding="utf-8"))

        self.assertEqual(len(manifest), 1)
        self.assertEqual(summary["snapshot_id"], "unit_snapshot")
        self.assertRegex(summary["snapshot_manifest_sha256"], r"^[0-9a-f]{64}$")

    def test_verify_snapshot_manifest_detects_current_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            df = pd.DataFrame(
                {
                    "open": [1.0],
                    "high": [1.1],
                    "low": [0.9],
                    "close": [1.05],
                    "volume": [100],
                },
                index=pd.to_datetime(["2024-01-02"]),
            )
            df.index.name = "date"
            df.to_parquet(cache_dir / "SPY.parquet")

            snapshot = build_daily_data_snapshot(
                cache_dir,
                snapshot_id="unit_snapshot",
                created_at_utc="2026-04-15T00:00:00+00:00",
            )
            paths = materialize_daily_data_snapshot(snapshot, root / "snapshot")
            verification = verify_daily_data_snapshot_manifest(
                paths["snapshot_manifest"],
                cache_dir=cache_dir,
            )

        self.assertEqual(len(verification), 1)
        self.assertTrue(verification["exists"].all())
        self.assertTrue(verification["sha256_match"].all())


if __name__ == "__main__":
    unittest.main()
