from __future__ import annotations

import unittest

import pandas as pd

from sp500_relative_alpha.multiple_testing import (
    MultipleTestingError,
    benjamini_hochberg,
    build_primary_cell_registry,
    evaluate_primary_hypothesis_family,
)


def _make_horizon_summary() -> pd.DataFrame:
    registry = build_primary_cell_registry()
    rows: list[dict[str, object]] = []
    for row in registry.itertuples(index=False):
        mean_rank_ic = 0.01 if row.cell_id in {"H05_XGB", "H10_XGB", "H15_XGB"} else -0.01
        p_value = {"H05_XGB": 0.001, "H10_XGB": 0.002, "H15_XGB": 0.003}.get(row.cell_id, 0.90)
        rows.append(
            {
                "horizon": row.horizon,
                "model_family": row.model_family,
                "mean_rank_ic": mean_rank_ic,
                "p_value_one_sided": p_value,
            }
        )
    return pd.DataFrame(rows)


def _make_fold_summary() -> pd.DataFrame:
    registry = build_primary_cell_registry()
    rows: list[dict[str, object]] = []
    for row in registry.itertuples(index=False):
        for fold_i in range(5):
            if row.cell_id == "H05_XGB":
                fold_mean = 0.01
            elif row.cell_id == "H10_XGB":
                fold_mean = 0.01 if fold_i < 2 else -0.01
            elif row.cell_id == "H15_XGB":
                fold_mean = 0.01
            else:
                fold_mean = -0.01
            rows.append(
                {
                    "horizon": row.horizon,
                    "model_family": row.model_family,
                    "fold_id": f"fold_{fold_i + 1:03d}",
                    "mean_rank_ic": fold_mean,
                }
            )
    return pd.DataFrame(rows)


def _make_economic_summary() -> pd.DataFrame:
    registry = build_primary_cell_registry()
    rows: list[dict[str, object]] = []
    for row in registry.itertuples(index=False):
        active_return = 0.02 if row.cell_id == "H05_XGB" else -0.01
        rows.append(
            {
                "horizon": row.horizon,
                "model_family": row.model_family,
                "cost_adjusted_active_return": active_return,
            }
        )
    return pd.DataFrame(rows)


class MultipleTestingTests(unittest.TestCase):
    def test_primary_cell_registry_matches_registered_24_cells(self) -> None:
        registry = build_primary_cell_registry()

        self.assertEqual(len(registry), 24)
        self.assertEqual(registry.iloc[0]["cell_id"], "H05_XGB")
        self.assertEqual(registry.iloc[11]["cell_id"], "H60_XGB")
        self.assertEqual(registry.iloc[12]["cell_id"], "H05_CAT")
        self.assertEqual(registry.iloc[-1]["cell_id"], "H60_CAT")

    def test_benjamini_hochberg_rejects_using_full_family_denominator(self) -> None:
        results = pd.DataFrame({"cell_id": ["a", "b", "c", "d"], "p_value": [0.001, 0.008, 0.03, 0.20]})

        adjusted = benjamini_hochberg(results, p_value_column="p_value", q=0.10)

        self.assertEqual(adjusted["bh_reject"].tolist(), [True, True, True, False])
        self.assertEqual(adjusted.loc[0, "bh_rank"], 1)
        self.assertAlmostEqual(adjusted.loc[2, "bh_threshold"], 0.075)
        self.assertLessEqual(adjusted.loc[0, "bh_adjusted_p"], adjusted.loc[1, "bh_adjusted_p"])

    def test_benjamini_hochberg_keeps_nan_p_values_unrejected(self) -> None:
        results = pd.DataFrame({"cell_id": ["a", "b"], "p_value": [0.01, None]})

        adjusted = benjamini_hochberg(results, p_value_column="p_value", q=0.10)

        self.assertEqual(adjusted["bh_reject"].tolist(), [True, False])
        self.assertTrue(pd.isna(adjusted.loc[1, "bh_adjusted_p"]))

    def test_evaluate_primary_hypothesis_family_marks_research_candidate_only_after_all_gates(self) -> None:
        result = evaluate_primary_hypothesis_family(
            _make_horizon_summary(),
            _make_fold_summary(),
            _make_economic_summary(),
        )

        h05 = result.loc[result["cell_id"] == "H05_XGB"].iloc[0]
        h10 = result.loc[result["cell_id"] == "H10_XGB"].iloc[0]
        h15 = result.loc[result["cell_id"] == "H15_XGB"].iloc[0]
        self.assertTrue(bool(h05["research_usable_candidate"]))
        self.assertEqual(h05["candidate_status"], "research_usable_candidate")
        self.assertFalse(bool(h10["research_usable_candidate"]))
        self.assertEqual(h10["candidate_status"], "failed_fold_stability")
        self.assertFalse(bool(h15["research_usable_candidate"]))
        self.assertEqual(h15["candidate_status"], "failed_economic_gate")

    def test_without_economic_summary_statistical_candidate_cannot_enter_holdout(self) -> None:
        result = evaluate_primary_hypothesis_family(_make_horizon_summary(), _make_fold_summary())

        h05 = result.loc[result["cell_id"] == "H05_XGB"].iloc[0]
        self.assertTrue(bool(h05["statistical_candidate"]))
        self.assertFalse(bool(h05["research_usable_candidate"]))
        self.assertFalse(bool(h05["economic_gate_evaluable"]))
        self.assertEqual(h05["candidate_status"], "pending_economic_gate")

    def test_rejects_incomplete_primary_metric_summary(self) -> None:
        summary = _make_horizon_summary().iloc[:-1]

        with self.assertRaises(MultipleTestingError):
            evaluate_primary_hypothesis_family(summary, _make_fold_summary())

    def test_rejects_unknown_model_family(self) -> None:
        summary = _make_horizon_summary()
        summary.loc[0, "model_family"] = "LightGBM"

        with self.assertRaises(MultipleTestingError):
            evaluate_primary_hypothesis_family(summary, _make_fold_summary())

    def test_rejects_invalid_p_values(self) -> None:
        results = pd.DataFrame({"cell_id": ["a"], "p_value": [1.2]})

        with self.assertRaises(MultipleTestingError):
            benjamini_hochberg(results, p_value_column="p_value")


if __name__ == "__main__":
    unittest.main()
