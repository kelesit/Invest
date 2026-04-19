from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .modeling import ModelHarnessError


@dataclass(frozen=True)
class CatBoostRegressorConfig:
    iterations: int = 500
    depth: int = 4
    learning_rate: float = 0.03
    l2_leaf_reg: float = 10.0
    min_data_in_leaf: int = 20
    random_seed: int = 20260417
    thread_count: int = -1
    verbose: int = 0
    # Exponential time decay: weight = exp(-log(2) / half_life * days_ago).
    # None = uniform weights (original behaviour).
    sample_weight_half_life_days: int | None = None


class CatBoostRegressorPredictor:
    """CatBoost regression predictor implementing WalkForwardPredictor.

    CatBoost handles NaN natively — no imputation needed before fit/predict.
    """

    def __init__(self, config: CatBoostRegressorConfig | None = None) -> None:
        self.config = config or CatBoostRegressorConfig()
        self._model = None

    def fit(
        self,
        train_samples: pd.DataFrame,
        feature_columns: tuple[str, ...],
        label_column: str,
    ) -> CatBoostRegressorPredictor:
        if not feature_columns:
            raise ModelHarnessError("CatBoostRegressorPredictor requires at least one feature column")
        if label_column not in train_samples.columns:
            raise ModelHarnessError(f"train_samples is missing label_column: {label_column}")

        y = pd.to_numeric(train_samples[label_column], errors="coerce")
        valid = y.notna()
        if not valid.any():
            raise ModelHarnessError("CatBoostRegressorPredictor has no non-null training labels")

        x_train = _feature_frame(train_samples.loc[valid], feature_columns)
        y_train = y.loc[valid].to_numpy(dtype=float)
        sample_weight = _compute_time_weights(
            train_samples.loc[valid], self.config.sample_weight_half_life_days
        )

        self._model = self._build_model()
        self._model.fit(x_train, y_train, sample_weight=sample_weight)
        return self

    def predict(
        self,
        inference_samples: pd.DataFrame,
        feature_columns: tuple[str, ...],
    ) -> pd.Series:
        if self._model is None:
            raise ModelHarnessError("CatBoostRegressorPredictor predict called before fit")
        x_test = _feature_frame(inference_samples, feature_columns)
        scores = self._model.predict(x_test)
        return pd.Series(scores, index=inference_samples.index, dtype=float)

    def _build_model(self):
        try:
            from catboost import CatBoostRegressor
        except Exception as exc:
            raise ModelHarnessError("catboost is not installed or cannot be imported") from exc

        return CatBoostRegressor(
            iterations=self.config.iterations,
            depth=self.config.depth,
            learning_rate=self.config.learning_rate,
            l2_leaf_reg=self.config.l2_leaf_reg,
            min_data_in_leaf=self.config.min_data_in_leaf,
            random_seed=self.config.random_seed,
            thread_count=self.config.thread_count,
            verbose=self.config.verbose,
            allow_writing_files=False,
        )


def _feature_frame(samples: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    missing = sorted(set(feature_columns) - set(samples.columns))
    if missing:
        raise ModelHarnessError(f"samples are missing feature columns: {missing[:5]}")
    return samples.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")


def _compute_time_weights(
    samples: pd.DataFrame,
    half_life_days: int | None,
) -> np.ndarray | None:
    """Return per-sample exponential decay weights, or None for uniform weights.

    weight_i = exp(-log(2) / half_life * days_ago_i)

    days_ago is computed from signal_date relative to the most recent date in
    the training set, so the newest sample always gets weight 1.0.
    """
    if half_life_days is None:
        return None
    if "signal_date" not in samples.columns:
        return None

    dates = pd.to_datetime(samples["signal_date"])
    days_ago = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    weights = np.exp(-np.log(2) / half_life_days * days_ago)
    return weights
