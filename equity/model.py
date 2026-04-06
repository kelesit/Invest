"""LightGBM model training with Purged Time-Series Cross-Validation.

Purged CV prevents information leakage in time-series ML:
- Data is split chronologically (no random shuffle)
- A 'purge gap' between train and test removes samples whose labels
  overlap with the test period (forward-looking label leakage)
- An 'embargo' adds extra buffer after purge for safety

Reference: Marcos López de Prado, "Advances in Financial Machine Learning", Ch. 7
"""

import numpy as np
import pandas as pd
import lightgbm as lgb


# Conservative hyperparameters — prefer underfitting to overfitting
DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "min_child_samples": 100,
    "verbose": -1,
}


def purged_time_series_cv(
    dates: pd.DatetimeIndex,
    n_splits: int = 5,
    train_days: int = 500,
    test_days: int = 60,
    purge_days: int = 10,
    embargo_days: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate train/test index splits with purge gap and embargo.

    Timeline for each fold:
        [...train_days...][purge_days][embargo_days][...test_days...]

    Args:
        dates: Sorted unique trading dates.
        n_splits: Number of CV folds.
        train_days: Training window size in trading days.
        test_days: Test window size in trading days.
        purge_days: Gap between train end and test start (= label horizon).
        embargo_days: Extra buffer after purge.

    Returns:
        List of (train_indices, test_indices) tuples, where indices refer
        to positions in the dates array.
    """
    unique_dates = dates.unique().sort_values()
    n_dates = len(unique_dates)

    total_per_fold = train_days + purge_days + embargo_days + test_days
    if total_per_fold > n_dates:
        raise ValueError(
            f"Not enough dates ({n_dates}) for even 1 fold "
            f"(need {total_per_fold} = {train_days}+{purge_days}+{embargo_days}+{test_days})"
        )

    # Space folds evenly across available dates
    available_for_splits = n_dates - total_per_fold
    step = max(1, available_for_splits // max(1, n_splits - 1)) if n_splits > 1 else 0

    splits = []
    for i in range(n_splits):
        train_start = i * step
        train_end = train_start + train_days
        test_start = train_end + purge_days + embargo_days
        test_end = test_start + test_days

        if test_end > n_dates:
            break

        train_dates = unique_dates[train_start:train_end]
        test_dates = unique_dates[test_start:test_end]

        splits.append((train_dates, test_dates))

    return splits


def train_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> np.ndarray:
    """Train LightGBM and return predictions on test set.

    Uses 10% of training data as validation for early stopping.
    """
    params = params or DEFAULT_PARAMS

    # Split last 10% of training as validation for early stopping
    n = len(X_train)
    val_size = max(1, n // 10)
    X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
    y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_val, label=y_val)

    callbacks = [lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        callbacks=callbacks,
    )

    predictions = model.predict(X_test)
    return predictions, model


def run_cv_pipeline(
    features: pd.DataFrame,
    labels: pd.Series,
    params: dict | None = None,
    n_splits: int = 5,
    train_days: int = 500,
    test_days: int = 60,
    purge_days: int = 10,
    embargo_days: int = 5,
) -> tuple[pd.DataFrame, list]:
    """Run full purged CV pipeline: split → train → predict for each fold.

    Args:
        features: MultiIndex (date, ticker) DataFrame with feature columns.
        labels: MultiIndex (date, ticker) Series of labels.
        params: LightGBM parameters.
        n_splits, train_days, test_days, purge_days, embargo_days: CV config.

    Returns:
        (predictions_df, models_list)
        predictions_df: DataFrame with columns [date, ticker, prediction, actual]
            containing out-of-sample predictions from all folds.
        models_list: List of trained LightGBM models (one per fold).
    """
    # Align features and labels — drop rows where either is NaN
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    # Drop rows with any NaN in features or labels
    valid_mask = features.notna().all(axis=1) & labels.notna()
    features = features.loc[valid_mask]
    labels = labels.loc[valid_mask]

    dates = features.index.get_level_values("date")
    unique_dates = dates.unique().sort_values()

    splits = purged_time_series_cv(
        unique_dates,
        n_splits=n_splits,
        train_days=train_days,
        test_days=test_days,
        purge_days=purge_days,
        embargo_days=embargo_days,
    )

    all_predictions = []
    all_models = []

    for fold_i, (train_dates, test_dates) in enumerate(splits):
        # Select samples by date
        train_mask = dates.isin(train_dates)
        test_mask = dates.isin(test_dates)

        X_train = features.loc[train_mask]
        y_train = labels.loc[train_mask]
        X_test = features.loc[test_mask]
        y_test = labels.loc[test_mask]

        print(
            f"Fold {fold_i + 1}/{len(splits)}: "
            f"train {train_dates[0].date()}→{train_dates[-1].date()} ({len(X_train)} samples), "
            f"test {test_dates[0].date()}→{test_dates[-1].date()} ({len(X_test)} samples)"
        )

        preds, model = train_and_predict(X_train, y_train, X_test, params)
        all_models.append(model)

        fold_df = pd.DataFrame({
            "prediction": preds,
            "actual": y_test.values,
        }, index=X_test.index)
        all_predictions.append(fold_df)

    predictions_df = pd.concat(all_predictions)
    return predictions_df, all_models
