from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .labels import DEFAULT_HORIZONS


PRIMARY_MODEL_FAMILIES = (("XGB", "XGBoost"), ("CAT", "CatBoost"))
PRIMARY_FDR_Q = 0.10
PRIMARY_MIN_POSITIVE_FOLD_RATE = 0.60


class MultipleTestingError(RuntimeError):
    """Raised when primary-family inference inputs violate the registered ledger."""


@dataclass(frozen=True)
class PrimaryLedgerConfig:
    fdr_q: float = PRIMARY_FDR_Q
    min_positive_fold_rate: float = PRIMARY_MIN_POSITIVE_FOLD_RATE
    p_value_column: str = "p_value_one_sided"
    mean_rank_ic_column: str = "mean_rank_ic"
    economic_return_column: str = "cost_adjusted_active_return"


def build_primary_cell_registry() -> pd.DataFrame:
    """Return the frozen `12 horizons x 2 model families` primary registry."""

    rows: list[dict[str, object]] = []
    for model_code, model_family in PRIMARY_MODEL_FAMILIES:
        for horizon in DEFAULT_HORIZONS:
            rows.append(
                {
                    "cell_id": f"H{horizon:02d}_{model_code}",
                    "horizon": int(horizon),
                    "model_code": model_code,
                    "model_family": model_family,
                    "feature_family": "Alpha101_OHLCV_allowlist_52",
                    "status": "Primary",
                }
            )
    return pd.DataFrame(rows)


def benjamini_hochberg(
    results: pd.DataFrame,
    *,
    p_value_column: str = "p_value",
    q: float = PRIMARY_FDR_Q,
) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR control while preserving input rows."""

    _validate_q(q)
    if p_value_column not in results.columns:
        raise MultipleTestingError(f"results is missing p_value_column: {p_value_column}")

    frame = results.copy()
    p_values = pd.to_numeric(frame[p_value_column], errors="coerce")
    finite_mask = p_values.notna()
    invalid = finite_mask & ~p_values.between(0.0, 1.0)
    if invalid.any():
        preview = frame.loc[invalid, [p_value_column]].head(5).to_dict("records")
        raise MultipleTestingError(f"p-values must be in [0, 1]; preview={preview}")

    m_total = len(frame)
    frame["bh_rank"] = pd.NA
    frame["bh_threshold"] = np.nan
    frame["bh_adjusted_p"] = np.nan
    frame["bh_reject"] = False
    if m_total == 0 or not finite_mask.any():
        return frame

    ordered = frame.loc[finite_mask].assign(_p_value=p_values.loc[finite_mask]).sort_values(
        ["_p_value"],
        kind="mergesort",
    )
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    thresholds = q * ranks / float(m_total)
    passed = ordered["_p_value"].to_numpy(dtype=float) <= thresholds
    cutoff = float(ordered["_p_value"].to_numpy(dtype=float)[passed].max()) if passed.any() else np.nan

    adjusted_sorted = ordered["_p_value"].to_numpy(dtype=float) * float(m_total) / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    frame.loc[ordered.index, "bh_rank"] = ranks.astype(int)
    frame.loc[ordered.index, "bh_threshold"] = thresholds
    frame.loc[ordered.index, "bh_adjusted_p"] = adjusted_sorted
    if np.isfinite(cutoff):
        frame.loc[finite_mask, "bh_reject"] = p_values.loc[finite_mask] <= cutoff
    return frame.drop(columns=["_p_value"], errors="ignore")


def evaluate_primary_hypothesis_family(
    horizon_summary: pd.DataFrame,
    fold_summary: pd.DataFrame,
    economic_summary: pd.DataFrame | None = None,
    config: PrimaryLedgerConfig | None = None,
) -> pd.DataFrame:
    """Evaluate the frozen primary-family gates without looking at holdout data."""

    ledger_config = config or PrimaryLedgerConfig()
    _validate_q(ledger_config.fdr_q)
    if not 0.0 <= ledger_config.min_positive_fold_rate <= 1.0:
        raise MultipleTestingError("min_positive_fold_rate must be in [0, 1]")

    registry = build_primary_cell_registry()
    metric = _prepare_metric_summary(horizon_summary, registry, ledger_config)
    fold = _prepare_fold_summary(fold_summary, registry, ledger_config)
    economic = _prepare_economic_summary(economic_summary, registry, ledger_config)

    fdr_frame = benjamini_hochberg(
        metric,
        p_value_column=ledger_config.p_value_column,
        q=ledger_config.fdr_q,
    )
    result = registry.merge(fdr_frame.drop(columns=["horizon", "model_family"], errors="ignore"), on="cell_id", how="left")
    result = result.merge(fold, on="cell_id", how="left")
    result = result.merge(economic, on="cell_id", how="left")

    result["positive_mean_rank_ic"] = pd.to_numeric(result[ledger_config.mean_rank_ic_column], errors="coerce") > 0.0
    result["fdr_passed"] = result["bh_reject"].fillna(False).astype(bool)
    result["fold_stability_passed"] = result["fold_positive_rate"] >= ledger_config.min_positive_fold_rate
    result["economic_gate_evaluable"] = result["economic_gate_evaluable"].fillna(False).astype(bool)
    result["economic_gate_passed"] = result["economic_gate_passed"].fillna(False).astype(bool)
    result["statistical_candidate"] = (
        result["positive_mean_rank_ic"] & result["fdr_passed"] & result["fold_stability_passed"]
    )
    result["research_usable_candidate"] = result["statistical_candidate"] & result["economic_gate_passed"]
    result["candidate_status"] = [_candidate_status(row) for row in result.to_dict("records")]
    return result.sort_values(["model_code", "horizon"], ignore_index=True)


def _prepare_metric_summary(
    horizon_summary: pd.DataFrame,
    registry: pd.DataFrame,
    config: PrimaryLedgerConfig,
) -> pd.DataFrame:
    required = {"horizon", "model_family", config.mean_rank_ic_column, config.p_value_column}
    _require_columns(horizon_summary, required, "horizon_summary")
    frame = _with_cell_ids(horizon_summary)
    _require_exact_primary_cells(frame, registry, "horizon_summary")
    return frame[["cell_id", config.mean_rank_ic_column, config.p_value_column]].copy()


def _prepare_fold_summary(
    fold_summary: pd.DataFrame,
    registry: pd.DataFrame,
    config: PrimaryLedgerConfig,
) -> pd.DataFrame:
    required = {"horizon", "model_family", "fold_id", config.mean_rank_ic_column}
    _require_columns(fold_summary, required, "fold_summary")
    frame = _with_cell_ids(fold_summary, allow_duplicate_cells=True)
    _require_subset_primary_cells(frame, registry, "fold_summary")
    grouped = frame.groupby("cell_id", sort=False)[config.mean_rank_ic_column]
    result = grouped.agg(
        n_oos_folds="count",
        n_positive_oos_folds=lambda values: int((pd.to_numeric(values, errors="coerce") > 0.0).sum()),
    ).reset_index()
    result["fold_positive_rate"] = result["n_positive_oos_folds"] / result["n_oos_folds"]
    missing = sorted(set(registry["cell_id"]) - set(result["cell_id"]))
    if missing:
        raise MultipleTestingError(f"fold_summary is missing primary cells: {missing[:5]}")
    return result


def _prepare_economic_summary(
    economic_summary: pd.DataFrame | None,
    registry: pd.DataFrame,
    config: PrimaryLedgerConfig,
) -> pd.DataFrame:
    if economic_summary is None:
        return pd.DataFrame(
            {
                "cell_id": registry["cell_id"],
                config.economic_return_column: np.nan,
                "economic_gate_evaluable": False,
                "economic_gate_passed": False,
            }
        )

    required = {"horizon", "model_family", config.economic_return_column}
    _require_columns(economic_summary, required, "economic_summary")
    frame = _with_cell_ids(economic_summary)
    _require_exact_primary_cells(frame, registry, "economic_summary")
    result = frame[["cell_id", config.economic_return_column]].copy()
    returns = pd.to_numeric(result[config.economic_return_column], errors="coerce")
    result["economic_gate_evaluable"] = returns.notna()
    result["economic_gate_passed"] = returns > 0.0
    return result


def _with_cell_ids(frame: pd.DataFrame, *, allow_duplicate_cells: bool = False) -> pd.DataFrame:
    result = frame.copy()
    result["horizon"] = pd.to_numeric(result["horizon"], errors="raise").astype(int)
    result["model_family"] = result["model_family"].astype(str)
    model_to_code = {model_family: model_code for model_code, model_family in PRIMARY_MODEL_FAMILIES}
    result["model_code"] = result["model_family"].map(model_to_code)
    unknown = sorted(result.loc[result["model_code"].isna(), "model_family"].unique())
    if unknown:
        raise MultipleTestingError(f"unknown model_family values: {unknown}")
    result["cell_id"] = result.apply(lambda row: f"H{int(row['horizon']):02d}_{row['model_code']}", axis=1)
    if not allow_duplicate_cells and result["cell_id"].duplicated().any():
        duplicates = result.loc[result["cell_id"].duplicated(), ["cell_id", "horizon", "model_family"]].head(5)
        raise MultipleTestingError(f"duplicate primary cell rows detected: {duplicates.to_dict('records')}")
    return result


def _require_exact_primary_cells(frame: pd.DataFrame, registry: pd.DataFrame, name: str) -> None:
    observed = set(frame["cell_id"])
    expected = set(registry["cell_id"])
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise MultipleTestingError(f"{name} must contain exactly the 24 primary cells; missing={missing[:5]}, extra={extra[:5]}")


def _require_subset_primary_cells(frame: pd.DataFrame, registry: pd.DataFrame, name: str) -> None:
    extra = sorted(set(frame["cell_id"]) - set(registry["cell_id"]))
    if extra:
        raise MultipleTestingError(f"{name} contains non-primary cells: {extra[:5]}")


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise MultipleTestingError(f"{name} is missing required columns: {missing}")


def _validate_q(q: float) -> None:
    if not 0.0 < q < 1.0:
        raise MultipleTestingError("FDR q must be between 0 and 1")


def _candidate_status(row: dict[str, object]) -> str:
    if bool(row["research_usable_candidate"]):
        return "research_usable_candidate"
    if not bool(row["positive_mean_rank_ic"]):
        return "failed_mean_rank_ic"
    if not bool(row["fdr_passed"]):
        return "failed_fdr"
    if not bool(row["fold_stability_passed"]):
        return "failed_fold_stability"
    if not bool(row["economic_gate_evaluable"]):
        return "pending_economic_gate"
    if not bool(row["economic_gate_passed"]):
        return "failed_economic_gate"
    return "failed_unknown"
