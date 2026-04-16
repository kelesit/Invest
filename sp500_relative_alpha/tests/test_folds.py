from __future__ import annotations

import unittest

import pandas as pd

from sp500_relative_alpha.folds import (
    FoldGenerationError,
    WalkForwardFoldConfig,
    build_purged_expanding_walk_forward_folds,
    fold_period_mask,
    validate_final_holdout_label_isolation,
    validate_fold_label_windows,
)


def _make_labels(signal_dates: pd.DatetimeIndex, horizon: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in ("AAPL", "MSFT"):
        for i, signal_date in enumerate(signal_dates):
            if i + horizon + 1 >= len(signal_dates):
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "signal_date": signal_date,
                    "horizon": horizon,
                    "entry_date": signal_dates[i + 1],
                    "exit_date": signal_dates[i + horizon + 1],
                    "benchmark_relative_open_to_open_return": 0.01,
                }
            )
    return pd.DataFrame(rows)


class FoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.signal_dates = pd.bdate_range("2020-01-01", periods=120)
        self.config = WalkForwardFoldConfig(
            research_start=self.signal_dates[0],
            research_end=self.signal_dates[79],
            pre_holdout_purge_start=self.signal_dates[80],
            pre_holdout_purge_end=self.signal_dates[84],
            final_holdout_start=self.signal_dates[85],
            final_holdout_end=self.signal_dates[119],
            min_training_days=20,
            test_block_days=10,
            purge_gap_days=5,
        )

    def test_expanding_walk_forward_uses_train_gap_test_geometry(self) -> None:
        folds = build_purged_expanding_walk_forward_folds(self.signal_dates, self.config)

        self.assertEqual(len(folds), 4)
        first = folds[0]
        self.assertEqual(first.fold_id, "fold_001")
        self.assertEqual(first.train_start, self.signal_dates[0])
        self.assertEqual(first.train_end, self.signal_dates[19])
        self.assertEqual(first.gap_start, self.signal_dates[20])
        self.assertEqual(first.gap_end, self.signal_dates[24])
        self.assertEqual(first.test_start, self.signal_dates[25])
        self.assertEqual(first.test_end, self.signal_dates[34])
        self.assertEqual(first.n_train_signal_dates, 20)
        self.assertEqual(first.n_gap_signal_dates, 5)
        self.assertEqual(first.n_test_signal_dates, 10)

        second = folds[1]
        self.assertEqual(second.train_end, self.signal_dates[34])
        self.assertEqual(second.gap_start, self.signal_dates[35])
        self.assertEqual(second.test_start, self.signal_dates[40])

    def test_fold_period_mask_selects_expected_rows(self) -> None:
        folds = build_purged_expanding_walk_forward_folds(self.signal_dates, self.config)
        frame = pd.DataFrame({"signal_date": self.signal_dates})

        self.assertEqual(int(fold_period_mask(frame, folds[0], "train").sum()), 20)
        self.assertEqual(int(fold_period_mask(frame, folds[0], "gap").sum()), 5)
        self.assertEqual(int(fold_period_mask(frame, folds[0], "test").sum()), 10)

    def test_label_window_validation_accepts_sufficient_purge(self) -> None:
        folds = build_purged_expanding_walk_forward_folds(self.signal_dates, self.config)
        labels = _make_labels(self.signal_dates, horizon=5)

        validate_fold_label_windows(labels, folds)

    def test_label_window_validation_rejects_insufficient_purge(self) -> None:
        bad_config = WalkForwardFoldConfig(
            research_start=self.signal_dates[0],
            research_end=self.signal_dates[79],
            pre_holdout_purge_start=self.signal_dates[80],
            pre_holdout_purge_end=self.signal_dates[84],
            final_holdout_start=self.signal_dates[85],
            final_holdout_end=self.signal_dates[119],
            min_training_days=20,
            test_block_days=10,
            purge_gap_days=2,
        )
        folds = build_purged_expanding_walk_forward_folds(self.signal_dates, bad_config)
        labels = _make_labels(self.signal_dates, horizon=5)

        with self.assertRaises(FoldGenerationError):
            validate_fold_label_windows(labels, folds)

    def test_final_holdout_isolation_accepts_pre_holdout_purge(self) -> None:
        labels = _make_labels(self.signal_dates, horizon=5)

        validate_final_holdout_label_isolation(labels, self.config)

    def test_final_holdout_isolation_rejects_overlapping_label_windows(self) -> None:
        bad_config = WalkForwardFoldConfig(
            research_start=self.signal_dates[0],
            research_end=self.signal_dates[79],
            pre_holdout_purge_start=self.signal_dates[80],
            pre_holdout_purge_end=self.signal_dates[81],
            final_holdout_start=self.signal_dates[82],
            final_holdout_end=self.signal_dates[119],
            min_training_days=20,
            test_block_days=10,
            purge_gap_days=2,
        )
        labels = _make_labels(self.signal_dates, horizon=5)

        with self.assertRaises(FoldGenerationError):
            validate_final_holdout_label_isolation(labels, bad_config)

    def test_rejects_too_short_research_calendar(self) -> None:
        short_config = WalkForwardFoldConfig(
            research_start=self.signal_dates[0],
            research_end=self.signal_dates[10],
            final_holdout_start=self.signal_dates[11],
            final_holdout_end=self.signal_dates[119],
            min_training_days=20,
            test_block_days=10,
            purge_gap_days=5,
        )

        with self.assertRaises(FoldGenerationError):
            build_purged_expanding_walk_forward_folds(self.signal_dates, short_config)


if __name__ == "__main__":
    unittest.main()
