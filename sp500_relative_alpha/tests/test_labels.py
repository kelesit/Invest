from __future__ import annotations

import unittest

import pandas as pd

from sp500_relative_alpha.labels import (
    LabelGenerationError,
    OpenToOpenLabelConfig,
    build_benchmark_relative_open_to_open_labels,
    build_round1_benchmark_relative_open_to_open_labels,
)


def _make_daily_bars() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=5)
    rows: list[dict[str, object]] = []
    for symbol, opens in {
        "AAPL": [10.0, 11.0, 12.0, 15.0, 18.0],
        "MSFT": [20.0, 20.0, 22.0, 22.0, 24.0],
        "SPY": [100.0, 101.0, 103.0, 104.0, 106.0],
    }.items():
        for date, open_price in zip(dates, opens, strict=True):
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": open_price,
                    "high": open_price + 1.0,
                    "low": open_price - 1.0,
                    "close": open_price + 0.5,
                    "shares_volume": 1000,
                }
            )
    return pd.DataFrame(rows)


class LabelGenerationTests(unittest.TestCase):
    def test_open_to_open_label_uses_t_plus_1_entry_and_t_plus_1_plus_h_exit(self) -> None:
        labels = build_benchmark_relative_open_to_open_labels(
            _make_daily_bars(),
            OpenToOpenLabelConfig(horizons=(1,)),
        )

        row = labels.loc[
            (labels["symbol"] == "AAPL")
            & (labels["signal_date"] == pd.Timestamp("2024-01-02"))
            & (labels["horizon"] == 1)
        ].iloc[0]

        asset_return = 12.0 / 11.0 - 1.0
        benchmark_return = 103.0 / 101.0 - 1.0
        self.assertEqual(row["entry_date"], pd.Timestamp("2024-01-03"))
        self.assertEqual(row["exit_date"], pd.Timestamp("2024-01-04"))
        self.assertAlmostEqual(row["asset_open_to_open_return"], asset_return)
        self.assertAlmostEqual(row["benchmark_open_to_open_return"], benchmark_return)
        self.assertAlmostEqual(row["benchmark_relative_open_to_open_return"], asset_return - benchmark_return)

    def test_label_generator_excludes_benchmark_by_default(self) -> None:
        labels = build_benchmark_relative_open_to_open_labels(
            _make_daily_bars(),
            OpenToOpenLabelConfig(horizons=(1,)),
        )

        self.assertNotIn("SPY", set(labels["symbol"]))

    def test_label_generator_drops_unlabelable_tail_dates(self) -> None:
        labels = build_benchmark_relative_open_to_open_labels(
            _make_daily_bars(),
            OpenToOpenLabelConfig(horizons=(2,)),
        )

        self.assertEqual(labels["signal_date"].max(), pd.Timestamp("2024-01-03"))

    def test_label_generator_rejects_missing_benchmark(self) -> None:
        bars = _make_daily_bars().loc[lambda df: df["symbol"] != "SPY"]

        with self.assertRaises(LabelGenerationError):
            build_benchmark_relative_open_to_open_labels(
                bars,
                OpenToOpenLabelConfig(horizons=(1,)),
            )

    def test_label_generator_uses_benchmark_calendar_not_union_calendar(self) -> None:
        bars = _make_daily_bars()
        extra = bars.loc[(bars["symbol"] == "AAPL") & (bars["date"] == pd.Timestamp("2024-01-02"))].copy()
        extra["date"] = pd.Timestamp("2024-01-06")
        extra["open"] = 999.0
        bars = pd.concat([bars, extra], ignore_index=True)

        labels = build_benchmark_relative_open_to_open_labels(
            bars,
            OpenToOpenLabelConfig(horizons=(1,)),
        )

        self.assertNotIn(pd.Timestamp("2024-01-06"), set(labels["signal_date"]))

    def test_label_generator_can_apply_frozen_signal_window(self) -> None:
        labels = build_benchmark_relative_open_to_open_labels(
            _make_daily_bars(),
            OpenToOpenLabelConfig(
                horizons=(1,),
                min_signal_date=pd.Timestamp("2024-01-03"),
                max_signal_date=pd.Timestamp("2024-01-04"),
            ),
        )

        self.assertEqual(labels["signal_date"].min(), pd.Timestamp("2024-01-03"))
        self.assertEqual(labels["signal_date"].max(), pd.Timestamp("2024-01-04"))

    def test_round1_helper_applies_frozen_window(self) -> None:
        labels = build_round1_benchmark_relative_open_to_open_labels(
            _make_daily_bars(),
            horizons=(1,),
        )

        self.assertGreaterEqual(labels["signal_date"].min(), pd.Timestamp("2015-12-31"))
        self.assertLessEqual(labels["signal_date"].max(), pd.Timestamp("2025-12-31"))


if __name__ == "__main__":
    unittest.main()
