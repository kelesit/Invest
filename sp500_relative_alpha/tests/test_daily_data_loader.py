from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sp500_relative_alpha.daily_data_loader import (
    DailyDataLoadError,
    SnapshotDailyDataConfig,
    load_snapshot_daily_ohlcv,
)
from sp500_relative_alpha.data_snapshot import (
    build_daily_data_snapshot,
    materialize_daily_data_snapshot,
)


def _write_daily_file(cache_dir: Path, symbol: str, closes: list[float]) -> None:
    dates = pd.bdate_range("2024-01-02", periods=len(closes))
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [close + 1.0 for close in closes],
            "low": [close - 1.0 for close in closes],
            "close": closes,
            "volume": [1000 + i for i in range(len(closes))],
        },
        index=dates,
    )
    df.index.name = "date"
    df.to_parquet(cache_dir / f"{symbol}.parquet")


class SnapshotDailyDataLoaderTests(unittest.TestCase):
    def test_loader_verifies_snapshot_and_builds_canonical_volume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            _write_daily_file(cache_dir, "AAPL", [10.0, 11.0, 12.0])
            _write_daily_file(cache_dir, "SPY", [20.0, 21.0, 22.0])

            snapshot = build_daily_data_snapshot(
                cache_dir,
                snapshot_id="unit_snapshot",
                created_at_utc="2026-04-15T00:00:00+00:00",
            )
            paths = materialize_daily_data_snapshot(snapshot, root / "snapshot")

            daily = load_snapshot_daily_ohlcv(
                SnapshotDailyDataConfig(
                    manifest_path=paths["snapshot_manifest"],
                    cache_dir=cache_dir,
                )
            )

        aapl_day1 = daily.loc[(daily["symbol"] == "AAPL") & (daily["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
        expected_typical = (11.0 + 9.0 + 10.0) / 3.0
        self.assertEqual(len(daily), 6)
        self.assertAlmostEqual(aapl_day1["typical_price"], expected_typical)
        self.assertAlmostEqual(aapl_day1["alpha_volume"], expected_typical * 1000)
        self.assertTrue(pd.isna(aapl_day1["close_to_close_return"]))

        aapl_day2 = daily.loc[(daily["symbol"] == "AAPL") & (daily["date"] == pd.Timestamp("2024-01-03"))].iloc[0]
        self.assertAlmostEqual(aapl_day2["close_to_close_return"], 0.1)

    def test_loader_refuses_to_load_when_snapshot_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            _write_daily_file(cache_dir, "SPY", [20.0, 21.0])

            snapshot = build_daily_data_snapshot(
                cache_dir,
                snapshot_id="unit_snapshot",
                created_at_utc="2026-04-15T00:00:00+00:00",
            )
            paths = materialize_daily_data_snapshot(snapshot, root / "snapshot")
            _write_daily_file(cache_dir, "SPY", [20.0, 999.0])

            with self.assertRaises(DailyDataLoadError):
                load_snapshot_daily_ohlcv(
                    SnapshotDailyDataConfig(
                        manifest_path=paths["snapshot_manifest"],
                        cache_dir=cache_dir,
                    )
                )

    def test_loader_rejects_unknown_symbol_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            _write_daily_file(cache_dir, "SPY", [20.0, 21.0])

            snapshot = build_daily_data_snapshot(
                cache_dir,
                snapshot_id="unit_snapshot",
                created_at_utc="2026-04-15T00:00:00+00:00",
            )
            paths = materialize_daily_data_snapshot(snapshot, root / "snapshot")

            with self.assertRaises(DailyDataLoadError):
                load_snapshot_daily_ohlcv(
                    SnapshotDailyDataConfig(
                        manifest_path=paths["snapshot_manifest"],
                        cache_dir=cache_dir,
                        symbols=("AAPL",),
                    )
                )


if __name__ == "__main__":
    unittest.main()
