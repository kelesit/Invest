from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sp500_relative_alpha.daily_coverage_audit import (
    DailyAuditConfig,
    DailyAuditDecision,
    run_daily_coverage_audit,
)


def _write_symbol_daily_file(
    cache_dir: Path,
    symbol: str,
    dates: pd.DatetimeIndex,
    *,
    invalid_date: pd.Timestamp | None = None,
) -> None:
    df = pd.DataFrame(
        {
            "open": [100.0 + i for i in range(len(dates))],
            "high": [101.0 + i for i in range(len(dates))],
            "low": [99.0 + i for i in range(len(dates))],
            "close": [100.5 + i for i in range(len(dates))],
            "volume": [1_000_000 + i for i in range(len(dates))],
        },
        index=dates,
    )
    df.index.name = "date"
    if invalid_date is not None:
        df.loc[invalid_date, "high"] = df.loc[invalid_date, "low"] - 1.0
    df.to_parquet(cache_dir / f"{symbol}.parquet")


class DailyCoverageAuditTests(unittest.TestCase):
    def test_clean_daily_cache_can_freeze_candidate_boundaries(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=30)
        config = DailyAuditConfig(
            max_feature_lookback_days=3,
            max_label_horizon_days=2,
            min_training_days=4,
            test_block_days=3,
            min_oos_test_blocks=2,
            purge_gap_days=1,
            final_holdout_days=4,
            min_daily_breadth_ratio=0.80,
            min_symbol_coverage_pass_ratio=0.80,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            for symbol in ["AAPL", "MSFT", "SPY"]:
                _write_symbol_daily_file(cache_dir, symbol, dates)

            bundle = run_daily_coverage_audit(cache_dir, config=config)

        decision = bundle.outputs["sample_boundary_decision"].iloc[0]
        self.assertEqual(decision["decision"], DailyAuditDecision.GO.value)
        self.assertEqual(pd.Timestamp(decision["first_feature_signal_date"]), dates[2])
        self.assertEqual(pd.Timestamp(decision["last_labelable_signal_date"]), dates[-4])
        self.assertEqual(pd.Timestamp(decision["final_holdout_start"]), dates[-7])
        self.assertEqual(pd.Timestamp(decision["final_holdout_end"]), dates[-4])
        self.assertGreaterEqual(int(decision["n_research_oos_folds"]), 2)

    def test_daily_audit_marks_missing_and_invalid_symbol_days(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5)
        invalid_date = dates[1]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            _write_symbol_daily_file(cache_dir, "AAPL", dates, invalid_date=invalid_date)
            _write_symbol_daily_file(cache_dir, "SPY", dates)
            _write_symbol_daily_file(cache_dir, "MSFT", dates.delete(2))

            bundle = run_daily_coverage_audit(
                cache_dir,
                config=DailyAuditConfig(
                    max_feature_lookback_days=1,
                    max_label_horizon_days=1,
                    min_training_days=1,
                    test_block_days=1,
                    min_oos_test_blocks=1,
                    purge_gap_days=0,
                    final_holdout_days=1,
                    min_daily_breadth_ratio=0.0,
                    min_symbol_coverage_pass_ratio=0.0,
                    min_benchmark_coverage_ratio=0.0,
                ),
            )

        symbol_day = bundle.outputs["symbol_day_audit_table"]
        aapl_invalid = symbol_day.loc[
            (symbol_day["symbol"] == "AAPL") & (symbol_day["date"] == invalid_date),
            "failure_code_primary",
        ].iloc[0]
        msft_missing = symbol_day.loc[
            (symbol_day["symbol"] == "MSFT") & (symbol_day["date"] == dates[2]),
            "failure_code_primary",
        ].iloc[0]

        self.assertEqual(aapl_invalid, "INCONSISTENT_OHLC_RANGE")
        self.assertEqual(msft_missing, "MISSING_DAILY_BAR")

    def test_low_breadth_blocks_go_decision(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=20)
        config = DailyAuditConfig(
            max_feature_lookback_days=1,
            max_label_horizon_days=1,
            min_training_days=2,
            test_block_days=2,
            min_oos_test_blocks=1,
            purge_gap_days=0,
            final_holdout_days=2,
            min_daily_breadth_ratio=0.75,
            min_symbol_coverage_pass_ratio=0.0,
            min_benchmark_coverage_ratio=0.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            _write_symbol_daily_file(cache_dir, "SPY", dates)
            _write_symbol_daily_file(cache_dir, "AAPL", dates)
            _write_symbol_daily_file(cache_dir, "MSFT", dates[:2])

            bundle = run_daily_coverage_audit(cache_dir, config=config)

        decision = bundle.outputs["sample_boundary_decision"].iloc[0]
        self.assertEqual(decision["decision"], DailyAuditDecision.NO_GO.value)
        self.assertIn("minimum signal-sample breadth", decision["reason"])


if __name__ == "__main__":
    unittest.main()
