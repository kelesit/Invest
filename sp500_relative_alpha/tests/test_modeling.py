from __future__ import annotations

import unittest

import pandas as pd

from sp500_relative_alpha.folds import WalkForwardFoldConfig, build_purged_expanding_walk_forward_folds
from sp500_relative_alpha.metrics import RankICConfig, evaluate_oos_rank_ic
from sp500_relative_alpha.modeling import (
    ConstantPredictor,
    FeaturePassthroughPredictor,
    ModelHarnessError,
    SymbolMeanLabelPredictor,
    run_walk_forward_predictions,
)


def _make_samples() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=30)
    symbols = ("A", "B", "C", "D", "E")
    rows: list[dict[str, object]] = []
    for horizon in (5, 10):
        for date_i, signal_date in enumerate(dates):
            for rank_i, symbol in enumerate(symbols):
                rows.append(
                    {
                        "signal_date": signal_date,
                        "symbol": symbol,
                        "horizon": horizon,
                        "entry_date": signal_date + pd.offsets.BDay(1),
                        "exit_date": signal_date + pd.offsets.BDay(horizon + 1),
                        "benchmark_relative_open_to_open_return": float(rank_i),
                        "asset_open_to_open_return": float(rank_i) / 100.0,
                        "benchmark_open_to_open_return": 0.0,
                        "alpha_good": float(rank_i),
                        "alpha_noise": float(date_i),
                    }
                )
    return pd.DataFrame(rows)


def _make_folds(samples: pd.DataFrame):
    dates = pd.DatetimeIndex(sorted(samples["signal_date"].unique()))
    config = WalkForwardFoldConfig(
        research_start=dates[0],
        research_end=dates[-1],
        final_holdout_start=dates[-1] + pd.Timedelta(days=1),
        final_holdout_end=dates[-1] + pd.Timedelta(days=30),
        min_training_days=5,
        purge_gap_days=2,
        test_block_days=5,
    )
    return build_purged_expanding_walk_forward_folds(dates, config)


class ModelingHarnessTests(unittest.TestCase):
    def test_feature_passthrough_predictions_integrate_with_rank_ic_metric(self) -> None:
        samples = _make_samples()
        folds = _make_folds(samples)

        predictions = run_walk_forward_predictions(
            samples,
            folds,
            lambda: FeaturePassthroughPredictor("alpha_good"),
        )
        _, _, horizon_summary = evaluate_oos_rank_ic(
            predictions,
            folds,
            RankICConfig(bootstrap_iterations=0),
        )

        self.assertEqual(set(predictions["fold_id"]), {fold.fold_id for fold in folds})
        self.assertIn("score", predictions.columns)
        self.assertAlmostEqual(horizon_summary["mean_rank_ic"].min(), 1.0)

    def test_predictor_does_not_receive_label_or_future_return_columns_at_prediction_time(self) -> None:
        samples = _make_samples()
        folds = _make_folds(samples)
        seen_columns: list[set[str]] = []

        class ColumnProbePredictor:
            def fit(self, train_samples, feature_columns, label_column):
                if label_column not in train_samples.columns:
                    raise AssertionError("label column should be available during fit")
                return self

            def predict(self, inference_samples, feature_columns):
                seen_columns.append(set(inference_samples.columns))
                return inference_samples["alpha_good"]

        run_walk_forward_predictions(samples, folds, lambda: ColumnProbePredictor())

        self.assertTrue(seen_columns)
        forbidden = {
            "benchmark_relative_open_to_open_return",
            "asset_open_to_open_return",
            "benchmark_open_to_open_return",
            "entry_date",
            "exit_date",
        }
        for columns in seen_columns:
            self.assertTrue(forbidden.isdisjoint(columns))
            self.assertIn("alpha_good", columns)
            self.assertIn("signal_date", columns)

    def test_constant_predictor_returns_oos_rows_without_training_dependency(self) -> None:
        samples = _make_samples()
        folds = _make_folds(samples)

        predictions = run_walk_forward_predictions(samples, folds, lambda: ConstantPredictor(3.0))

        self.assertTrue((predictions["score"] == 3.0).all())
        self.assertGreater(len(predictions), 0)

    def test_symbol_mean_label_predictor_uses_train_labels_only(self) -> None:
        samples = _make_samples()
        folds = _make_folds(samples)

        predictions = run_walk_forward_predictions(samples, folds, lambda: SymbolMeanLabelPredictor())

        self.assertFalse(predictions["score"].isna().any())
        expected_scores = dict(zip(("A", "B", "C", "D", "E"), range(5), strict=True))
        first_fold = predictions.loc[predictions["fold_id"] == folds[0].fold_id]
        for symbol, expected in expected_scores.items():
            self.assertAlmostEqual(first_fold.loc[first_fold["symbol"] == symbol, "score"].iloc[0], expected)

    def test_rejects_samples_that_already_contain_score_column(self) -> None:
        samples = _make_samples()
        samples["score"] = 0.0

        with self.assertRaises(ModelHarnessError):
            run_walk_forward_predictions(samples, _make_folds(samples), lambda: ConstantPredictor())

    def test_rejects_duplicate_sample_rows(self) -> None:
        samples = pd.concat([_make_samples(), _make_samples().iloc[[0]]], ignore_index=True)

        with self.assertRaises(ModelHarnessError):
            run_walk_forward_predictions(samples, _make_folds(samples), lambda: ConstantPredictor())

    def test_rejects_missing_feature_columns(self) -> None:
        samples = _make_samples().drop(columns=["alpha_good", "alpha_noise"])

        with self.assertRaises(ModelHarnessError):
            run_walk_forward_predictions(samples, _make_folds(samples), lambda: ConstantPredictor())

    def test_rejects_wrong_score_length(self) -> None:
        samples = _make_samples()
        folds = _make_folds(samples)

        class BadLengthPredictor:
            def fit(self, train_samples, feature_columns, label_column):
                return self

            def predict(self, inference_samples, feature_columns):
                return pd.Series([1.0])

        with self.assertRaises(ModelHarnessError):
            run_walk_forward_predictions(samples, folds, lambda: BadLengthPredictor())

    def test_rejects_missing_passthrough_feature(self) -> None:
        samples = _make_samples()
        folds = _make_folds(samples)

        with self.assertRaises(ModelHarnessError):
            run_walk_forward_predictions(samples, folds, lambda: FeaturePassthroughPredictor("alpha_missing"))


if __name__ == "__main__":
    unittest.main()
