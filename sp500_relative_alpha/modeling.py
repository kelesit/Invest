from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .folds import WalkForwardFold, fold_period_mask


class ModelHarnessError(RuntimeError):
    """Raised when fold-level model plumbing violates the evaluation contract."""


class WalkForwardPredictor(Protocol):
    """Minimal interface consumed by the walk-forward prediction harness."""

    def fit(self, train_samples: pd.DataFrame, feature_columns: tuple[str, ...], label_column: str) -> WalkForwardPredictor:
        ...

    def predict(self, inference_samples: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.Series:
        ...


PredictorFactory = Callable[[], WalkForwardPredictor]


@dataclass(frozen=True)
class WalkForwardPredictionConfig:
    label_column: str = "benchmark_relative_open_to_open_return"
    score_column: str = "score"
    feature_prefix: str = "alpha_"
    metadata_columns: tuple[str, ...] = ("signal_date", "symbol", "horizon")
    n_top_features: int | None = None  # per-fold IC selection; None = use all


class ConstantPredictor:
    """Deterministic plumbing predictor that emits one constant score."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def fit(self, train_samples: pd.DataFrame, feature_columns: tuple[str, ...], label_column: str) -> ConstantPredictor:
        _ = train_samples, feature_columns, label_column
        return self

    def predict(self, inference_samples: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.Series:
        _ = feature_columns
        return pd.Series(self.value, index=inference_samples.index, dtype=float)


class FeaturePassthroughPredictor:
    """Deterministic predictor that uses one pre-existing feature as the score."""

    def __init__(self, feature_column: str) -> None:
        self.feature_column = feature_column

    def fit(self, train_samples: pd.DataFrame, feature_columns: tuple[str, ...], label_column: str) -> FeaturePassthroughPredictor:
        _ = train_samples, label_column
        if self.feature_column not in feature_columns:
            raise ModelHarnessError(f"feature_column is not available for training: {self.feature_column}")
        return self

    def predict(self, inference_samples: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.Series:
        if self.feature_column not in feature_columns:
            raise ModelHarnessError(f"feature_column is not available for prediction: {self.feature_column}")
        if self.feature_column not in inference_samples.columns:
            raise ModelHarnessError(f"inference_samples is missing feature_column: {self.feature_column}")
        return pd.to_numeric(inference_samples[self.feature_column], errors="coerce")


class SymbolMeanLabelPredictor:
    """Simple baseline that predicts each symbol's training-period mean label."""

    def __init__(self) -> None:
        self._symbol_means: pd.Series | None = None
        self._global_mean: float | None = None

    def fit(self, train_samples: pd.DataFrame, feature_columns: tuple[str, ...], label_column: str) -> SymbolMeanLabelPredictor:
        _ = feature_columns
        if label_column not in train_samples.columns:
            raise ModelHarnessError(f"train_samples is missing label_column: {label_column}")
        labels = pd.to_numeric(train_samples[label_column], errors="coerce")
        self._symbol_means = labels.groupby(train_samples["symbol"]).mean()
        self._global_mean = float(labels.mean()) if labels.notna().any() else 0.0
        return self

    def predict(self, inference_samples: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.Series:
        _ = feature_columns
        if self._symbol_means is None or self._global_mean is None:
            raise ModelHarnessError("predict called before fit")
        scores = inference_samples["symbol"].map(self._symbol_means).fillna(self._global_mean)
        return pd.to_numeric(scores, errors="coerce")


def run_walk_forward_predictions(
    samples: pd.DataFrame,
    folds: tuple[WalkForwardFold, ...],
    predictor_factory: PredictorFactory,
    config: WalkForwardPredictionConfig | None = None,
) -> pd.DataFrame:
    """Fit one predictor per fold and return OOS test predictions.

    The predictor sees labels during `fit` and sees only metadata + features
    during `predict`. The returned frame restores labels so the metric harness
    can evaluate OOS predictions without giving labels to the predictor.
    """

    prediction_config = config or WalkForwardPredictionConfig()
    _validate_prediction_inputs(samples, folds, prediction_config)
    feature_columns = _feature_columns(samples, prediction_config.feature_prefix)
    frames: list[pd.DataFrame] = []

    for fold in folds:
        train_samples = samples.loc[fold_period_mask(samples, fold, "train")].copy()
        test_samples = samples.loc[fold_period_mask(samples, fold, "test")].copy()
        if train_samples.empty or test_samples.empty:
            raise ModelHarnessError(f"{fold.fold_id} has empty train or test samples")
        _validate_fold_order(train_samples, test_samples, fold.fold_id)

        if prediction_config.n_top_features is not None:
            fold_features = _select_top_features_by_ic(
                train_samples, feature_columns,
                prediction_config.label_column,
                prediction_config.n_top_features,
            )
        else:
            fold_features = feature_columns

        predictor = predictor_factory()
        predictor.fit(train_samples, fold_features, prediction_config.label_column)
        inference_samples = _build_inference_samples(test_samples, fold_features, prediction_config.metadata_columns)
        scores = predictor.predict(inference_samples, fold_features)
        score_series = _coerce_score_series(scores, test_samples.index, fold.fold_id, prediction_config.score_column)

        predictions = test_samples.copy()
        predictions.insert(0, "fold_id", fold.fold_id)
        predictions[prediction_config.score_column] = score_series.to_numpy(dtype=float)
        frames.append(predictions)

    if not frames:
        return pd.DataFrame(columns=["fold_id", *samples.columns, prediction_config.score_column])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["horizon", "fold_id", "signal_date", "symbol"],
        ignore_index=True,
    )


def _select_top_features_by_ic(
    train: pd.DataFrame,
    feature_cols: tuple[str, ...],
    label_col: str,
    n_top: int,
) -> tuple[str, ...]:
    """Return the top n_top features ranked by |mean cross-sectional Pearson IC| on training data.

    Loops over date groups (O(n_dates) Python iters), but each iter is a fully vectorized
    matmul over all features. Feature NaN is replaced with cross-sectional mean (contributes
    zero after centering, neutral for correlation).
    """
    n_top = min(n_top, len(feature_cols))
    train = train[train[label_col].notna()].sort_values(["signal_date", "symbol"])

    dates = train["signal_date"].to_numpy()
    feat_arr = train[list(feature_cols)].to_numpy(dtype=np.float64)
    label_arr = train[label_col].to_numpy(dtype=np.float64)

    _, date_starts = np.unique(dates, return_index=True)
    date_ends = np.append(date_starts[1:], len(dates))

    n_feat = len(feature_cols)
    ic_sum = np.zeros(n_feat)
    ic_cnt = np.zeros(n_feat, dtype=np.int32)

    for s, e in zip(date_starts, date_ends):
        lg = label_arr[s:e]
        fg = feat_arr[s:e]

        if (~np.isfinite(lg)).all() or (e - s) < 5:
            continue

        fv = np.isfinite(fg)
        col_cnt = fv.sum(axis=0).astype(np.float64)
        col_mean = np.where(col_cnt > 0, np.where(fv, fg, 0.0).sum(axis=0) / np.maximum(col_cnt, 1), 0.0)
        fg_c = np.where(fv, fg - col_mean, 0.0)    # NaN → 0 after centering

        lg_c = lg - lg.mean()

        num   = fg_c.T @ lg_c                       # (n_feat,)
        var_f = (fg_c ** 2).sum(axis=0)
        var_l = float((lg_c ** 2).sum())

        if var_l <= 0:
            continue

        with np.errstate(invalid="ignore", divide="ignore"):
            ic = np.where(var_f > 0, num / np.sqrt(var_f * var_l), np.nan)

        has = np.isfinite(ic)
        ic_sum += np.where(has, ic, 0.0)
        ic_cnt += has.astype(np.int32)

    mean_ic = ic_sum / np.maximum(ic_cnt, 1)
    top_idx = np.argsort(-np.abs(mean_ic))[:n_top]
    return tuple(np.array(list(feature_cols))[top_idx])


def _validate_prediction_inputs(
    samples: pd.DataFrame,
    folds: tuple[WalkForwardFold, ...],
    config: WalkForwardPredictionConfig,
) -> None:
    required = {"signal_date", "symbol", "horizon", config.label_column, *config.metadata_columns}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ModelHarnessError(f"samples are missing required columns: {missing}")
    if not folds:
        raise ModelHarnessError("folds is empty")
    if config.score_column in samples.columns:
        raise ModelHarnessError(f"samples already contain score_column: {config.score_column}")
    if samples[["signal_date", "symbol", "horizon"]].duplicated().any():
        duplicates = samples.loc[
            samples[["signal_date", "symbol", "horizon"]].duplicated(),
            ["signal_date", "symbol", "horizon"],
        ].head(5)
        raise ModelHarnessError(f"duplicate sample rows detected: {duplicates.to_dict('records')}")
    if not _feature_columns(samples, config.feature_prefix):
        raise ModelHarnessError(f"samples contain no feature columns with prefix {config.feature_prefix!r}")


def _validate_fold_order(train_samples: pd.DataFrame, test_samples: pd.DataFrame, fold_id: str) -> None:
    max_train_date = pd.to_datetime(train_samples["signal_date"]).max()
    min_test_date = pd.to_datetime(test_samples["signal_date"]).min()
    if max_train_date >= min_test_date:
        raise ModelHarnessError(
            f"{fold_id} is not forward-only: max_train_date={max_train_date.date()}, "
            f"min_test_date={min_test_date.date()}"
        )


def _build_inference_samples(
    test_samples: pd.DataFrame,
    feature_columns: tuple[str, ...],
    metadata_columns: tuple[str, ...],
) -> pd.DataFrame:
    columns = [*metadata_columns, *feature_columns]
    return test_samples.loc[:, columns].copy()


def _coerce_score_series(
    scores: pd.Series,
    expected_index: pd.Index,
    fold_id: str,
    score_column: str,
) -> pd.Series:
    if not isinstance(scores, pd.Series):
        raise ModelHarnessError(f"{fold_id} predictor returned non-Series scores for {score_column}")
    if len(scores) != len(expected_index):
        raise ModelHarnessError(
            f"{fold_id} predictor returned wrong score length: got={len(scores)}, expected={len(expected_index)}"
        )
    if not scores.index.equals(expected_index):
        scores = pd.Series(scores.to_numpy(), index=expected_index)
    return pd.to_numeric(scores, errors="coerce")


def _feature_columns(samples: pd.DataFrame, feature_prefix: str) -> tuple[str, ...]:
    return tuple(sorted(column for column in samples.columns if column.startswith(feature_prefix)))
