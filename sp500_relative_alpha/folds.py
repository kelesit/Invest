from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd


ROUND1_RESEARCH_PERIOD_START = pd.Timestamp("2015-12-31")
ROUND1_RESEARCH_PERIOD_END = pd.Timestamp("2023-10-02")
ROUND1_PRE_HOLDOUT_PURGE_START = pd.Timestamp("2023-10-03")
ROUND1_PRE_HOLDOUT_PURGE_END = pd.Timestamp("2023-12-27")
ROUND1_FINAL_HOLDOUT_START = pd.Timestamp("2023-12-28")
ROUND1_FINAL_HOLDOUT_END = pd.Timestamp("2025-12-31")
ROUND1_MIN_TRAINING_DAYS = 4 * 252
ROUND1_TEST_BLOCK_DAYS = 126
ROUND1_PURGE_GAP_DAYS = 60

FoldPeriod = Literal["train", "gap", "test"]


class FoldGenerationError(RuntimeError):
    """Raised when walk-forward folds cannot satisfy the frozen time protocol."""


@dataclass(frozen=True)
class WalkForwardFoldConfig:
    research_start: pd.Timestamp = ROUND1_RESEARCH_PERIOD_START
    research_end: pd.Timestamp = ROUND1_RESEARCH_PERIOD_END
    pre_holdout_purge_start: pd.Timestamp = ROUND1_PRE_HOLDOUT_PURGE_START
    pre_holdout_purge_end: pd.Timestamp = ROUND1_PRE_HOLDOUT_PURGE_END
    final_holdout_start: pd.Timestamp = ROUND1_FINAL_HOLDOUT_START
    final_holdout_end: pd.Timestamp = ROUND1_FINAL_HOLDOUT_END
    min_training_days: int = ROUND1_MIN_TRAINING_DAYS
    test_block_days: int = ROUND1_TEST_BLOCK_DAYS
    purge_gap_days: int = ROUND1_PURGE_GAP_DAYS
    rolling_window_days: int | None = None  # None = expanding; int = rolling


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    gap_start: pd.Timestamp
    gap_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train_signal_dates: int
    n_gap_signal_dates: int
    n_test_signal_dates: int


def build_round1_rolling_walk_forward_folds(
    signal_dates: Iterable[object],
    rolling_window_days: int = 2 * 252,
) -> tuple[WalkForwardFold, ...]:
    """Build Round 1 walk-forward folds with a rolling training window."""
    return build_purged_expanding_walk_forward_folds(
        signal_dates,
        WalkForwardFoldConfig(rolling_window_days=rolling_window_days),
    )


def build_round1_walk_forward_folds(signal_dates: Iterable[object]) -> tuple[WalkForwardFold, ...]:
    """Build the frozen Round 1 expanding walk-forward folds."""

    return build_purged_expanding_walk_forward_folds(signal_dates, WalkForwardFoldConfig())


def build_purged_expanding_walk_forward_folds(
    signal_dates: Iterable[object],
    config: WalkForwardFoldConfig | None = None,
) -> tuple[WalkForwardFold, ...]:
    """Build train/gap/test folds over labelable signal dates.

    The fold protocol is forward-only:

    - train is the expanding historical block
    - gap is the pre-test purge region
    - test is the next OOS block

    The next fold starts after the previous test block plus a fresh purge gap,
    matching the frozen daily coverage audit's fold-count logic.
    """

    fold_config = config or WalkForwardFoldConfig()
    _validate_config(fold_config)
    calendar = _normalize_signal_dates(signal_dates)
    research_calendar = calendar[
        (calendar >= pd.Timestamp(fold_config.research_start)) & (calendar <= pd.Timestamp(fold_config.research_end))
    ]
    if research_calendar.empty:
        raise FoldGenerationError("no signal dates fall inside the configured research period")

    required_days = fold_config.min_training_days + fold_config.purge_gap_days + fold_config.test_block_days
    if len(research_calendar) < required_days:
        raise FoldGenerationError(
            "not enough research signal dates for one fold: "
            f"n={len(research_calendar)}, required={required_days}"
        )

    folds: list[WalkForwardFold] = []
    test_start_pos = fold_config.min_training_days + fold_config.purge_gap_days
    while test_start_pos + fold_config.test_block_days <= len(research_calendar):
        train_end_pos = test_start_pos - fold_config.purge_gap_days - 1
        gap_start_pos = train_end_pos + 1
        gap_end_pos = test_start_pos - 1
        test_end_pos = test_start_pos + fold_config.test_block_days - 1

        if fold_config.rolling_window_days is not None:
            train_start_pos = max(0, train_end_pos - fold_config.rolling_window_days + 1)
        else:
            train_start_pos = 0

        folds.append(
            WalkForwardFold(
                fold_id=f"fold_{len(folds) + 1:03d}",
                train_start=research_calendar[train_start_pos],
                train_end=research_calendar[train_end_pos],
                gap_start=research_calendar[gap_start_pos],
                gap_end=research_calendar[gap_end_pos],
                test_start=research_calendar[test_start_pos],
                test_end=research_calendar[test_end_pos],
                n_train_signal_dates=train_end_pos - train_start_pos + 1,
                n_gap_signal_dates=fold_config.purge_gap_days,
                n_test_signal_dates=fold_config.test_block_days,
            )
        )
        test_start_pos += fold_config.test_block_days + fold_config.purge_gap_days

    if not folds:
        raise FoldGenerationError("fold generation produced no OOS folds")
    return tuple(folds)


def fold_period_mask(frame: pd.DataFrame, fold: WalkForwardFold, period: FoldPeriod, date_column: str = "signal_date") -> pd.Series:
    """Return a boolean mask for one fold period."""

    if date_column not in frame.columns:
        raise FoldGenerationError(f"frame is missing date column: {date_column}")
    dates = pd.to_datetime(frame[date_column])
    if period == "train":
        return (dates >= fold.train_start) & (dates <= fold.train_end)
    if period == "gap":
        return (dates >= fold.gap_start) & (dates <= fold.gap_end)
    if period == "test":
        return (dates >= fold.test_start) & (dates <= fold.test_end)
    raise FoldGenerationError(f"unknown fold period: {period}")


def validate_fold_label_windows(
    labels: pd.DataFrame,
    folds: Iterable[WalkForwardFold],
    *,
    signal_date_column: str = "signal_date",
    entry_date_column: str = "entry_date",
    exit_date_column: str = "exit_date",
) -> None:
    """Validate that each fold's training labels cannot overlap its test labels."""

    required = {signal_date_column, entry_date_column, exit_date_column}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise FoldGenerationError(f"labels are missing required columns: {missing}")

    label_dates = labels.copy()
    for column in (signal_date_column, entry_date_column, exit_date_column):
        label_dates[column] = pd.to_datetime(label_dates[column])

    bad_alignment = label_dates.loc[
        ~(
            (label_dates[signal_date_column] < label_dates[entry_date_column])
            & (label_dates[entry_date_column] <= label_dates[exit_date_column])
        )
    ]
    if not bad_alignment.empty:
        preview = bad_alignment[[signal_date_column, entry_date_column, exit_date_column]].head(5).to_dict("records")
        raise FoldGenerationError(f"label date alignment is invalid; preview={preview}")

    for fold in folds:
        train = label_dates.loc[fold_period_mask(label_dates, fold, "train", signal_date_column)]
        test = label_dates.loc[fold_period_mask(label_dates, fold, "test", signal_date_column)]
        if train.empty or test.empty:
            raise FoldGenerationError(f"{fold.fold_id} has empty train or test labels")

        max_train_exit = train[exit_date_column].max()
        min_test_entry = test[entry_date_column].min()
        if max_train_exit >= min_test_entry:
            raise FoldGenerationError(
                f"{fold.fold_id} label windows overlap: "
                f"max_train_exit={max_train_exit.date()}, min_test_entry={min_test_entry.date()}"
            )


def validate_final_holdout_label_isolation(
    labels: pd.DataFrame,
    config: WalkForwardFoldConfig | None = None,
    *,
    signal_date_column: str = "signal_date",
    entry_date_column: str = "entry_date",
    exit_date_column: str = "exit_date",
) -> None:
    """Validate that research-period labels do not overlap final holdout labels."""

    fold_config = config or WalkForwardFoldConfig()
    required = {signal_date_column, entry_date_column, exit_date_column}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise FoldGenerationError(f"labels are missing required columns: {missing}")

    label_dates = labels.copy()
    for column in (signal_date_column, entry_date_column, exit_date_column):
        label_dates[column] = pd.to_datetime(label_dates[column])

    research = label_dates.loc[
        (label_dates[signal_date_column] >= pd.Timestamp(fold_config.research_start))
        & (label_dates[signal_date_column] <= pd.Timestamp(fold_config.research_end))
    ]
    holdout = label_dates.loc[
        (label_dates[signal_date_column] >= pd.Timestamp(fold_config.final_holdout_start))
        & (label_dates[signal_date_column] <= pd.Timestamp(fold_config.final_holdout_end))
    ]
    if research.empty or holdout.empty:
        raise FoldGenerationError("cannot validate holdout isolation with empty research or holdout labels")

    max_research_exit = research[exit_date_column].max()
    min_holdout_entry = holdout[entry_date_column].min()
    if max_research_exit >= min_holdout_entry:
        raise FoldGenerationError(
            "research labels overlap final holdout labels: "
            f"max_research_exit={max_research_exit.date()}, min_holdout_entry={min_holdout_entry.date()}"
        )


def _normalize_signal_dates(signal_dates: Iterable[object]) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(pd.to_datetime(list(signal_dates))).dropna().drop_duplicates().sort_values()
    if calendar.empty:
        raise FoldGenerationError("signal_dates is empty")
    return calendar


def _validate_config(config: WalkForwardFoldConfig) -> None:
    if config.min_training_days <= 0:
        raise FoldGenerationError("min_training_days must be positive")
    if config.test_block_days <= 0:
        raise FoldGenerationError("test_block_days must be positive")
    if config.purge_gap_days < 0:
        raise FoldGenerationError("purge_gap_days must be non-negative")
    if pd.Timestamp(config.research_start) > pd.Timestamp(config.research_end):
        raise FoldGenerationError("research_start must be <= research_end")
    if pd.Timestamp(config.research_end) >= pd.Timestamp(config.final_holdout_start):
        raise FoldGenerationError("research_end must be before final_holdout_start")
