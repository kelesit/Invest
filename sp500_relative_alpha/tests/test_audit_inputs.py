from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sp500_relative_alpha.audit_inputs import (
    build_security_master_from_inventory,
    diagnose_local_equity_cache_for_round1,
    inventory_local_equity_cache,
    normalize_display_symbol,
)


class LocalEquityCacheInputTests(unittest.TestCase):
    def test_normalize_display_symbol_replaces_dash_with_dot(self) -> None:
        self.assertEqual(normalize_display_symbol("BRK-B"), "BRK.B")
        self.assertEqual(normalize_display_symbol("BF-B"), "BF.B")
        self.assertEqual(normalize_display_symbol("AAPL"), "AAPL")

    def test_inventory_detects_daily_cache_as_non_minute_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            df = pd.DataFrame(
                {
                    "open": [1.0, 2.0, 3.0],
                    "high": [1.1, 2.1, 3.1],
                    "low": [0.9, 1.9, 2.9],
                    "close": [1.05, 2.05, 3.05],
                    "volume": [100, 120, 90],
                },
                index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            )
            df.index.name = "date"
            df.to_parquet(cache_dir / "AAPL.parquet")
            df.to_parquet(cache_dir / "SPY.parquet")

            inventory = inventory_local_equity_cache(cache_dir)

        self.assertEqual(set(inventory["symbol"]), {"AAPL", "SPY"})
        self.assertTrue(inventory["has_required_ohlcv"].all())
        self.assertEqual(int(inventory["is_minute_candidate"].sum()), 0)

    def test_security_master_marks_spy_as_benchmark(self) -> None:
        inventory = pd.DataFrame(
            [
                {"symbol": "AAPL"},
                {"symbol": "SPY"},
                {"symbol": "MSFT"},
            ]
        )
        master = build_security_master_from_inventory(inventory)
        spy_role = master.loc[master["symbol"] == "SPY", "symbol_role"].iloc[0]
        self.assertEqual(spy_role, "benchmark")

    def test_round1_diagnostic_accepts_daily_only_cache_for_daily_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            df = pd.DataFrame(
                {
                    "open": [1.0, 2.0, 3.0],
                    "high": [1.1, 2.1, 3.1],
                    "low": [0.9, 1.9, 2.9],
                    "close": [1.05, 2.05, 3.05],
                    "volume": [100, 120, 90],
                },
                index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            )
            df.index.name = "date"
            df.to_parquet(cache_dir / "AAPL.parquet")
            df.to_parquet(cache_dir / "SPY.parquet")

            diagnostic = diagnose_local_equity_cache_for_round1(cache_dir)

        summary = diagnostic.summary.iloc[0]
        self.assertEqual(summary["verdict"], "GO_FOR_DAILY_AUDIT")
        self.assertIn("daily-only", summary["reason"])


if __name__ == "__main__":
    unittest.main()
