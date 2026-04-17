from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from sp500_relative_alpha.folds import WalkForwardFoldConfig, build_purged_expanding_walk_forward_folds
from sp500_relative_alpha.metrics import (
    MetricComputationError,
    RankICConfig,
    build_oos_rank_ic_panel,
    compute_daily_rank_ic,
    evaluate_oos_rank_ic,
    summarize_oos_rank_ic_panel,
    summarize_rank_ic_series,
)


def _make_metric_samples() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=30)
    symbols = ("A", "B", "C", "D", "E")
    rows: list[dict[str, object]] = []
    for horizon in (5, 10):
        for date_i, date in enumerate(dates):
            for rank_i, symbol in enumerate(symbols):
                label = float(rank_i)
                score = float(rank_i) if horizon == 5 else float(len(symbols) - rank_i)
                rows.append(
                    {
                        "signal_date": date,
                        "symbol": symbol,
                        "horizon": horizon,
                        "score": score + 0.01 * date_i,
                        "benchmark_relative_open_to_open_return": label,
                    }
                )
    return pd.DataFrame(rows)


class RankICMetricTests(unittest.TestCase):
    def test_compute_daily_rank_ic_matches_spearman_ordering(self) -> None:
        daily = compute_daily_rank_ic(_make_metric_samples())

        first_h5 = daily.loc[(daily["horizon"] == 5) & (daily["signal_date"] == pd.Timestamp("2024-01-02"))].iloc[0]
        first_h10 = daily.loc[(daily["horizon"] == 10) & (daily["signal_date"] == pd.Timestamp("2024-01-02"))].iloc[0]
        self.assertAlmostEqual(first_h5["rank_ic"], 1.0)
        self.assertAlmostEqual(first_h10["rank_ic"], -1.0)
        self.assertEqual(first_h5["n_obs"], 5)

    def test_compute_daily_rank_ic_returns_nan_for_constant_scores(self) -> None:
        samples = _make_metric_samples()
        samples.loc[samples["horizon"] == 5, "score"] = 1.0

        daily = compute_daily_rank_ic(samples)

        self.assertTrue(daily.loc[daily["horizon"] == 5, "rank_ic"].isna().all())

    def test_compute_daily_rank_ic_respects_min_cross_section_size(self) -> None:
        samples = _make_metric_samples().loc[lambda df: df["symbol"].isin(["A", "B", "C"])]

        daily = compute_daily_rank_ic(samples, RankICConfig(min_cross_section_size=4))

        self.assertTrue(daily["rank_ic"].isna().all())
        self.assertTrue((daily["n_obs"] == 3).all())

    def test_summarize_rank_ic_series_reports_hac_and_bootstrap_fields(self) -> None:
        summary = summarize_rank_ic_series(
            pd.Series([0.02, 0.03, 0.01, 0.04, 0.05, 0.01]),
            RankICConfig(bootstrap_iterations=200, bootstrap_seed=7),
        )

        self.assertEqual(summary["n_dates"], 6)
        self.assertGreater(summary["mean_rank_ic"], 0)
        self.assertEqual(summary["positive_rate"], 1.0)
        self.assertGreater(summary["hac_t_stat"], 0)
        self.assertGreaterEqual(summary["bootstrap_ci_upper"], summary["bootstrap_ci_lower"])

    def test_bootstrap_summary_is_deterministic(self) -> None:
        config = RankICConfig(bootstrap_iterations=100, bootstrap_seed=123)
        values = pd.Series([0.01, 0.02, -0.01, 0.03, 0.02, 0.04])

        first = summarize_rank_ic_series(values, config)
        second = summarize_rank_ic_series(values, config)

        self.assertEqual(first["bootstrap_ci_lower"], second["bootstrap_ci_lower"])
        self.assertEqual(first["bootstrap_ci_upper"], second["bootstrap_ci_upper"])

    def test_build_oos_rank_ic_panel_uses_only_fold_test_blocks(self) -> None:
        samples = _make_metric_samples()
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
        folds = build_purged_expanding_walk_forward_folds(dates, config)

        panel = build_oos_rank_ic_panel(samples, folds)

        self.assertEqual(set(panel["fold_id"]), {fold.fold_id for fold in folds})
        for fold in folds:
            subset = panel.loc[panel["fold_id"] == fold.fold_id]
            self.assertGreaterEqual(subset["signal_date"].min(), fold.test_start)
            self.assertLessEqual(subset["signal_date"].max(), fold.test_end)

    def test_evaluate_oos_rank_ic_returns_panel_fold_and_horizon_summaries(self) -> None:
        samples = _make_metric_samples()
        dates = pd.DatetimeIndex(sorted(samples["signal_date"].unique()))
        fold_config = WalkForwardFoldConfig(
            research_start=dates[0],
            research_end=dates[-1],
            final_holdout_start=dates[-1] + pd.Timedelta(days=1),
            final_holdout_end=dates[-1] + pd.Timedelta(days=30),
            min_training_days=5,
            purge_gap_days=2,
            test_block_days=5,
        )
        folds = build_purged_expanding_walk_forward_folds(dates, fold_config)

        panel, fold_summary, horizon_summary = evaluate_oos_rank_ic(
            samples,
            folds,
            RankICConfig(bootstrap_iterations=50),
        )

        self.assertFalse(panel.empty)
        self.assertEqual(set(horizon_summary["horizon"]), {5, 10})
        self.assertEqual(set(fold_summary["horizon"]), {5, 10})
        h5_mean = horizon_summary.loc[horizon_summary["horizon"] == 5, "mean_rank_ic"].iloc[0]
        h10_mean = horizon_summary.loc[horizon_summary["horizon"] == 10, "mean_rank_ic"].iloc[0]
        self.assertAlmostEqual(h5_mean, 1.0)
        self.assertAlmostEqual(h10_mean, -1.0)

    def test_summarize_oos_rank_ic_panel_rejects_missing_columns(self) -> None:
        with self.assertRaises(MetricComputationError):
            summarize_oos_rank_ic_panel(pd.DataFrame({"rank_ic": [0.1]}))

    def test_compute_daily_rank_ic_rejects_missing_score_column(self) -> None:
        samples = _make_metric_samples().drop(columns=["score"])

        with self.assertRaises(MetricComputationError):
            compute_daily_rank_ic(samples)

    def test_summary_of_empty_series_is_nan_not_zero(self) -> None:
        summary = summarize_rank_ic_series(pd.Series([np.nan, np.nan]))

        self.assertEqual(summary["n_dates"], 0)
        self.assertTrue(np.isnan(summary["mean_rank_ic"]))


if __name__ == "__main__":
    unittest.main()
