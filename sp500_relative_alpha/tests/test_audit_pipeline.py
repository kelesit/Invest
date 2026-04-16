from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sp500_relative_alpha.audit_artifacts import (
    ARTIFACT_NAMES,
    AuditArtifactContext,
    materialize_output_bundle,
)
from sp500_relative_alpha.audit_pipeline import (
    AuditDecision,
    AuditRunContext,
    decide_sample_boundary,
    run_coverage_audit,
)


def _full_day_minutes(trading_date: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(f"{trading_date} 09:30:00", tz="America/New_York")
    return [start + pd.Timedelta(minutes=i) for i in range(390)]


def _make_full_day_rows(security_id: str, symbol: str, trading_date: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, ts_et in enumerate(_full_day_minutes(trading_date)):
        ts_utc = ts_et.tz_convert("UTC")
        base = 100.0 + index / 100.0
        rows.append(
            {
                "security_id": security_id,
                "symbol": symbol,
                "raw_ts_source": ts_utc,
                "source_tz": "UTC",
                "source_ts_convention": "bar_start",
                "open_raw": base,
                "high_raw": base + 0.2,
                "low_raw": base - 0.2,
                "close_raw": base + 0.1,
                "volume_raw": 100 + index,
            }
        )
    return rows


class CoverageAuditPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.security_master = pd.DataFrame(
            [
                {
                    "security_id": "sec_aapl",
                    "symbol": "AAPL",
                    "symbol_role": "constituent",
                    "instrument_type": "common_stock",
                    "include_flag": True,
                },
                {
                    "security_id": "sec_msft",
                    "symbol": "MSFT",
                    "symbol_role": "constituent",
                    "instrument_type": "common_stock",
                    "include_flag": True,
                },
                {
                    "security_id": "sec_spy",
                    "symbol": "SPY",
                    "symbol_role": "benchmark",
                    "instrument_type": "etf",
                    "include_flag": True,
                },
            ]
        )
        self.calendar = pd.DataFrame(
            [
                {
                    "trading_date": "2024-01-02",
                    "market_status": "full_day",
                    "session_open_et": "09:30:00",
                    "session_close_et": "16:00:00",
                    "expected_regular_minutes": 390,
                },
                {
                    "trading_date": "2024-01-03",
                    "market_status": "full_day",
                    "session_open_et": "09:30:00",
                    "session_close_et": "16:00:00",
                    "expected_regular_minutes": 390,
                },
            ]
        )

    def test_pipeline_builds_quality_outputs_and_short_span_is_no_go(self) -> None:
        rows: list[dict[str, object]] = []
        for trading_date in ["2024-01-02", "2024-01-03"]:
            rows.extend(_make_full_day_rows("sec_aapl", "AAPL", trading_date))
            rows.extend(_make_full_day_rows("sec_spy", "SPY", trading_date))

        msft_day1 = _make_full_day_rows("sec_msft", "MSFT", "2024-01-02")
        msft_day2 = _make_full_day_rows("sec_msft", "MSFT", "2024-01-03")[:-1]
        rows.extend(msft_day1)
        rows.extend(msft_day2)

        raw_bars = pd.DataFrame(rows)
        bundle = run_coverage_audit(
            self.security_master,
            self.calendar,
            split_reference=None,
            raw_bars=raw_bars,
            context=AuditRunContext(audit_run_id="unit-short-span"),
        )

        normalized = bundle.normalized_session_minute_bars
        self.assertEqual(
            len(
                normalized.loc[
                    (normalized["symbol"] == "AAPL")
                    & (normalized["trading_date"] == pd.Timestamp("2024-01-02"))
                ]
            ),
            390,
        )

        symbol_day = bundle.outputs["symbol_day_audit_table"]
        msft_day2_status = symbol_day.loc[
            (symbol_day["symbol"] == "MSFT")
            & (symbol_day["trading_date"] == pd.Timestamp("2024-01-03")),
            "symbol_day_status",
        ].iloc[0]
        msft_day2_reason = symbol_day.loc[
            (symbol_day["symbol"] == "MSFT")
            & (symbol_day["trading_date"] == pd.Timestamp("2024-01-03")),
            "failure_code_primary",
        ].iloc[0]
        self.assertEqual(msft_day2_status, "partial")
        self.assertEqual(msft_day2_reason, "MISSING_PARTIAL")

        breadth = bundle.outputs["daily_breadth_summary"]
        day2_breadth = breadth.loc[
            breadth["trading_date"] == pd.Timestamp("2024-01-03"),
            "breadth",
        ].iloc[0]
        self.assertEqual(day2_breadth, 0.5)

        decision = bundle.outputs["sample_boundary_decision"]
        self.assertEqual(decision["decision"].iloc[0], AuditDecision.NO_GO.value)

    def test_decide_sample_boundary_returns_go_for_clean_multi_year_quality_facts(self) -> None:
        open_dates = pd.bdate_range("2014-01-02", "2022-01-04")
        calendar = pd.DataFrame(
            {
                "trading_date": open_dates,
                "market_status": "full_day",
                "session_open_et": "09:30:00",
                "session_close_et": "16:00:00",
                "expected_regular_minutes": 390,
            }
        )

        rows = []
        for security_id, symbol, role in [
            ("sec_aapl", "AAPL", "constituent"),
            ("sec_msft", "MSFT", "constituent"),
            ("sec_spy", "SPY", "benchmark"),
        ]:
            for trading_date in open_dates:
                rows.append(
                    {
                        "security_id": security_id,
                        "symbol": symbol,
                        "symbol_role": role,
                        "trading_date": pd.Timestamp(trading_date),
                        "market_status": "full_day",
                        "expected_regular_minutes": 390,
                        "observed_regular_minutes": 390,
                        "duplicate_timestamp_count": 0,
                        "outside_session_bar_count": 0,
                        "invalid_price_count": 0,
                        "invalid_hilo_count": 0,
                        "first_bar_ts_et": pd.Timestamp(trading_date).tz_localize("America/New_York")
                        + pd.Timedelta(hours=9, minutes=30),
                        "last_bar_ts_et": pd.Timestamp(trading_date).tz_localize("America/New_York")
                        + pd.Timedelta(hours=16),
                        "adjustment_support_flag": True,
                        "symbol_day_status": "full_valid",
                        "failure_code_primary": "NONE",
                        "failure_code_secondary": None,
                    }
                )
        quality_facts = pd.DataFrame(rows)

        decision = decide_sample_boundary(
            self.security_master,
            calendar,
            quality_facts,
            context=AuditRunContext(audit_run_id="unit-go"),
        )

        self.assertEqual(decision["decision"].iloc[0], AuditDecision.GO.value)
        self.assertFalse(pd.isna(decision["candidate_raw_start"].iloc[0]))
        self.assertFalse(pd.isna(decision["final_holdout_start"].iloc[0]))

    def test_artifact_materialization_writes_expected_files(self) -> None:
        rows = []
        for trading_date in ["2024-01-02", "2024-01-03"]:
            rows.extend(_make_full_day_rows("sec_aapl", "AAPL", trading_date))
            rows.extend(_make_full_day_rows("sec_msft", "MSFT", trading_date))
            rows.extend(_make_full_day_rows("sec_spy", "SPY", trading_date))
        bundle = run_coverage_audit(
            self.security_master,
            self.calendar,
            split_reference=None,
            raw_bars=pd.DataFrame(rows),
            context=AuditRunContext(audit_run_id="unit-materialize"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            context = AuditArtifactContext(
                base_dir=Path(tmpdir),
                audit_run_id="unit-materialize",
                data_snapshot_id="snapshot-001",
            )
            paths = materialize_output_bundle(bundle.outputs, context)

            for name in ARTIFACT_NAMES:
                self.assertIn(name, paths)
                self.assertTrue(paths[name].exists())
            self.assertTrue(paths["manifest"].exists())


if __name__ == "__main__":
    unittest.main()
