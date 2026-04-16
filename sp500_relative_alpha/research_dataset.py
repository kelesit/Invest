from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .alpha101_features import compute_alpha101_feature_matrices, stack_alpha101_feature_matrices
from .alpha101_ops import build_alpha101_input_matrices
from .labels import (
    DEFAULT_HORIZONS,
    build_round1_benchmark_relative_open_to_open_labels,
)


class ResearchDatasetError(RuntimeError):
    """Raised when feature and label samples cannot be aligned honestly."""


@dataclass(frozen=True)
class ResearchDatasetConfig:
    label_column: str = "benchmark_relative_open_to_open_return"
    drop_rows_without_features: bool = False


def build_round1_research_dataset(
    daily_bars: pd.DataFrame,
    *,
    alpha_ids: Iterable[str] | None = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    config: ResearchDatasetConfig | None = None,
) -> pd.DataFrame:
    """Build the Round 1 feature-label sample table without evaluating performance."""

    inputs = build_alpha101_input_matrices(daily_bars)
    feature_matrices = compute_alpha101_feature_matrices(inputs, alpha_ids=alpha_ids)
    labels = build_round1_benchmark_relative_open_to_open_labels(daily_bars, horizons=horizons)
    return build_research_dataset(feature_matrices, labels, config=config)


def build_research_dataset(
    feature_matrices: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    config: ResearchDatasetConfig | None = None,
) -> pd.DataFrame:
    """Align Alpha101 feature matrices with horizon-specific labels.

    Output grain:

    - one row = one `symbol, signal_date, horizon`
    - feature columns are repeated across horizons for the same symbol-date
    - labels retain their entry/exit dates so downstream folds can audit leakage
    """

    dataset_config = config or ResearchDatasetConfig()
    features = stack_alpha101_feature_matrices(feature_matrices).rename(columns={"date": "signal_date"})
    _validate_features(features)
    clean_labels = _prepare_labels(labels, dataset_config.label_column)

    merged = clean_labels.merge(
        features,
        how="left",
        on=["signal_date", "symbol"],
        validate="many_to_one",
    )
    feature_columns = _feature_columns(merged)
    if dataset_config.drop_rows_without_features:
        merged = merged.dropna(subset=feature_columns, how="all").reset_index(drop=True)

    ordered_columns = [
        "signal_date",
        "symbol",
        "horizon",
        "entry_date",
        "exit_date",
        dataset_config.label_column,
        "asset_open_to_open_return",
        "benchmark_open_to_open_return",
        *feature_columns,
    ]
    optional_ordered_columns = [column for column in ordered_columns if column in merged.columns]
    remainder = [column for column in merged.columns if column not in optional_ordered_columns]
    result = merged[optional_ordered_columns + remainder]

    return result.sort_values(["horizon", "signal_date", "symbol"], ignore_index=True)


def _validate_features(features: pd.DataFrame) -> None:
    required = {"signal_date", "symbol"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ResearchDatasetError(f"features are missing required columns: {missing}")
    feature_columns = _feature_columns(features)
    if not feature_columns:
        raise ResearchDatasetError("features contain no alpha_* columns")
    if features[["signal_date", "symbol"]].duplicated().any():
        duplicates = features.loc[features[["signal_date", "symbol"]].duplicated(), ["signal_date", "symbol"]].head(5)
        raise ResearchDatasetError(f"duplicate feature rows detected: {duplicates.to_dict('records')}")
    features["signal_date"] = pd.to_datetime(features["signal_date"])


def _prepare_labels(labels: pd.DataFrame, label_column: str) -> pd.DataFrame:
    required = {
        "signal_date",
        "symbol",
        "horizon",
        "entry_date",
        "exit_date",
        label_column,
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ResearchDatasetError(f"labels are missing required columns: {missing}")
    if labels[["signal_date", "symbol", "horizon"]].duplicated().any():
        duplicates = labels.loc[
            labels[["signal_date", "symbol", "horizon"]].duplicated(),
            ["signal_date", "symbol", "horizon"],
        ].head(5)
        raise ResearchDatasetError(f"duplicate label rows detected: {duplicates.to_dict('records')}")

    clean = labels.copy()
    for column in ("signal_date", "entry_date", "exit_date"):
        clean[column] = pd.to_datetime(clean[column])
    bad_alignment = clean.loc[~((clean["signal_date"] < clean["entry_date"]) & (clean["entry_date"] <= clean["exit_date"]))]
    if not bad_alignment.empty:
        preview = bad_alignment[["signal_date", "entry_date", "exit_date"]].head(5).to_dict("records")
        raise ResearchDatasetError(f"label date alignment is invalid; preview={preview}")
    return clean


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith("alpha_"))
