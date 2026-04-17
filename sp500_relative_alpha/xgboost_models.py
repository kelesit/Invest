from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .modeling import ModelHarnessError


@dataclass(frozen=True)
class XGBoostRegressorConfig:
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 10.0
    reg_alpha: float = 0.0
    min_child_weight: float = 20.0
    random_state: int = 20260416
    n_jobs: int = 4
    tree_method: str = "hist"


class XGBoostRegressorPredictor:
    """Conservative XGBoost regression predictor for single-cell sanity runs."""

    def __init__(self, config: XGBoostRegressorConfig | None = None) -> None:
        self.config = config or XGBoostRegressorConfig()
        self._model = None

    def fit(
        self,
        train_samples: pd.DataFrame,
        feature_columns: tuple[str, ...],
        label_column: str,
    ) -> XGBoostRegressorPredictor:
        if not feature_columns:
            raise ModelHarnessError("XGBoostRegressorPredictor requires at least one feature column")
        if label_column not in train_samples.columns:
            raise ModelHarnessError(f"train_samples is missing label_column: {label_column}")

        y = pd.to_numeric(train_samples[label_column], errors="coerce")
        valid = y.notna()
        if not valid.any():
            raise ModelHarnessError("XGBoostRegressorPredictor has no non-null training labels")

        x_train = _feature_frame(train_samples.loc[valid], feature_columns)
        y_train = y.loc[valid].to_numpy(dtype=float)
        self._model = self._build_model()
        self._model.fit(x_train, y_train)
        return self

    def predict(self, inference_samples: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.Series:
        if self._model is None:
            raise ModelHarnessError("XGBoostRegressorPredictor predict called before fit")
        x_test = _feature_frame(inference_samples, feature_columns)
        scores = self._model.predict(x_test)
        return pd.Series(scores, index=inference_samples.index, dtype=float)

    def _build_model(self):
        try:
            from xgboost import XGBRegressor
        except Exception as exc:  # pragma: no cover - dependency availability is environment-specific.
            raise ModelHarnessError("xgboost is not installed or cannot be imported") from exc

        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_lambda=self.config.reg_lambda,
            reg_alpha=self.config.reg_alpha,
            min_child_weight=self.config.min_child_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            tree_method=self.config.tree_method,
            missing=np.nan,
            verbosity=0,
        )


def _feature_frame(samples: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    missing = sorted(set(feature_columns) - set(samples.columns))
    if missing:
        raise ModelHarnessError(f"samples are missing feature columns: {missing[:5]}")
    frame = samples.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan)
