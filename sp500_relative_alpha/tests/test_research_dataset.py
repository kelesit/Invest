from __future__ import annotations

import unittest

import pandas as pd

from sp500_relative_alpha.alpha101_features import compute_alpha101_feature_matrices
from sp500_relative_alpha.alpha101_ops import build_alpha101_input_matrices
from sp500_relative_alpha.labels import OpenToOpenLabelConfig, build_benchmark_relative_open_to_open_labels
from sp500_relative_alpha.research_dataset import (
    ResearchDatasetError,
    build_research_dataset,
)


def _make_daily_bars() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=12)
    rows: list[dict[str, object]] = []
    for symbol, base in {"AAPL": 10.0, "MSFT": 20.0, "SPY": 100.0}.items():
        for i, date in enumerate(dates):
            open_price = base + 0.2 * i
            close = open_price + 0.1
            high = close + 0.5
            low = open_price - 0.5
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "shares_volume": 1000.0 + i,
                    "typical_price": (high + low + close) / 3.0,
                    "alpha_volume": ((high + low + close) / 3.0) * (1000.0 + i),
                    "close_to_close_return": pd.NA,
                }
            )
    bars = pd.DataFrame(rows).sort_values(["symbol", "date"], ignore_index=True)
    bars["close_to_close_return"] = bars.groupby("symbol", sort=False)["close"].pct_change()
    return bars


class ResearchDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bars = _make_daily_bars()
        self.inputs = build_alpha101_input_matrices(self.bars)
        self.features = compute_alpha101_feature_matrices(self.inputs, alpha_ids=("033", "101"))
        self.labels = build_benchmark_relative_open_to_open_labels(
            self.bars,
            OpenToOpenLabelConfig(horizons=(1, 2)),
        )

    def test_build_research_dataset_aligns_features_and_labels(self) -> None:
        dataset = build_research_dataset(self.features, self.labels)

        self.assertEqual(len(dataset), len(self.labels))
        self.assertEqual(dataset.iloc[0]["signal_date"], self.labels.iloc[0]["signal_date"])
        self.assertIn("alpha_033", dataset.columns)
        self.assertIn("alpha_101", dataset.columns)
        self.assertIn("benchmark_relative_open_to_open_return", dataset.columns)
        self.assertNotIn("SPY", set(dataset["symbol"]))
        self.assertTrue((dataset["signal_date"] < dataset["entry_date"]).all())
        self.assertTrue((dataset["entry_date"] <= dataset["exit_date"]).all())

    def test_features_are_repeated_across_horizons_for_same_symbol_date(self) -> None:
        dataset = build_research_dataset(self.features, self.labels)
        sample = dataset.loc[
            (dataset["symbol"] == "AAPL")
            & (dataset["signal_date"] == dataset["signal_date"].min())
        ].sort_values("horizon")

        self.assertEqual(set(sample["horizon"]), {1, 2})
        self.assertEqual(sample["alpha_101"].nunique(dropna=False), 1)

    def test_rejects_duplicate_label_rows(self) -> None:
        duplicate = pd.concat([self.labels, self.labels.iloc[[0]]], ignore_index=True)

        with self.assertRaises(ResearchDatasetError):
            build_research_dataset(self.features, duplicate)

    def test_rejects_invalid_label_alignment(self) -> None:
        bad_labels = self.labels.copy()
        bad_labels.loc[0, "entry_date"] = bad_labels.loc[0, "signal_date"]

        with self.assertRaises(ResearchDatasetError):
            build_research_dataset(self.features, bad_labels)

    def test_rejects_empty_feature_family(self) -> None:
        with self.assertRaises(ResearchDatasetError):
            build_research_dataset({}, self.labels)


if __name__ == "__main__":
    unittest.main()
