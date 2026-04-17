from __future__ import annotations

import unittest

import pandas as pd

from sp500_relative_alpha.modeling import ModelHarnessError
from sp500_relative_alpha.xgboost_models import XGBoostRegressorConfig, XGBoostRegressorPredictor


def _make_samples() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(30):
        rows.append(
            {
                "symbol": f"S{i % 5}",
                "signal_date": pd.Timestamp("2024-01-02") + pd.offsets.BDay(i),
                "horizon": 20,
                "alpha_001": float(i % 5),
                "alpha_002": float(i // 5),
                "benchmark_relative_open_to_open_return": float(i % 5) * 0.01,
            }
        )
    return pd.DataFrame(rows)


class XGBoostModelTests(unittest.TestCase):
    def test_xgboost_regressor_predictor_fits_and_predicts(self) -> None:
        samples = _make_samples()
        predictor = XGBoostRegressorPredictor(
            XGBoostRegressorConfig(
                n_estimators=5,
                max_depth=2,
                n_jobs=1,
                random_state=1,
            )
        )

        predictor.fit(samples.iloc[:20], ("alpha_001", "alpha_002"), "benchmark_relative_open_to_open_return")
        scores = predictor.predict(samples.iloc[20:], ("alpha_001", "alpha_002"))

        self.assertEqual(len(scores), len(samples.iloc[20:]))
        self.assertFalse(scores.isna().any())

    def test_xgboost_regressor_rejects_empty_feature_columns(self) -> None:
        samples = _make_samples()

        with self.assertRaises(ModelHarnessError):
            XGBoostRegressorPredictor().fit(samples, (), "benchmark_relative_open_to_open_return")

    def test_xgboost_regressor_rejects_all_missing_labels(self) -> None:
        samples = _make_samples()
        samples["benchmark_relative_open_to_open_return"] = pd.NA

        with self.assertRaises(ModelHarnessError):
            XGBoostRegressorPredictor().fit(samples, ("alpha_001",), "benchmark_relative_open_to_open_return")

    def test_xgboost_regressor_treats_infinite_features_as_missing(self) -> None:
        samples = _make_samples()
        samples.loc[0, "alpha_001"] = float("inf")
        predictor = XGBoostRegressorPredictor(XGBoostRegressorConfig(n_estimators=3, n_jobs=1))

        predictor.fit(samples.iloc[:20], ("alpha_001", "alpha_002"), "benchmark_relative_open_to_open_return")
        scores = predictor.predict(samples.iloc[20:], ("alpha_001", "alpha_002"))

        self.assertEqual(len(scores), len(samples.iloc[20:]))


if __name__ == "__main__":
    unittest.main()
