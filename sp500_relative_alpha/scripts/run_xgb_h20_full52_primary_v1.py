"""Run XGBoost primary experiment: H=20, all 52 allowlisted Alpha101 features.

This is a primary-family cell (H20_XGB) under the frozen SP500RA-V1-R1 preregistration.
Results are saved to artifacts/model_runs/xgb_h20_full52_primary_v1/.

Usage:
    uv run python -m sp500_relative_alpha.scripts.run_xgb_h20_full52_primary_v1
    # or from project root:
    uv run python sp500_relative_alpha/scripts/run_xgb_h20_full52_primary_v1.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from sp500_relative_alpha.daily_data_loader import ROUND1_SNAPSHOT_MANIFEST, load_round1_daily_ohlcv
from sp500_relative_alpha.folds import build_round1_walk_forward_folds, validate_fold_label_windows
from sp500_relative_alpha.metrics import RankICConfig, evaluate_oos_rank_ic
from sp500_relative_alpha.modeling import WalkForwardPredictionConfig, run_walk_forward_predictions
from sp500_relative_alpha.research_dataset import build_round1_research_dataset
from sp500_relative_alpha.xgboost_models import XGBoostRegressorConfig, XGBoostRegressorPredictor

HORIZON = 20
RUN_ID = "xgb_h20_full52_primary_v1"
MODEL_FAMILY = "XGBoost"
CELL_ID = f"H{HORIZON:02d}_XGB"

# HUBB has one OHLC range-inconsistent row (2021-05-05) in the frozen snapshot.
# Excluded here as in the sanity run. Documented in the snapshot; do not modify
# the snapshot to preserve preregistration integrity.
EXCLUDED_SYMBOLS: tuple[str, ...] = ("HUBB",)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "sp500_relative_alpha" / "artifacts" / "model_runs" / RUN_ID


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load symbols from manifest, exclude known bad symbols ---
    manifest = pd.read_csv(ROUND1_SNAPSHOT_MANIFEST)
    all_symbols = manifest["symbol"].tolist()
    load_symbols = [s for s in all_symbols if s not in EXCLUDED_SYMBOLS]
    print(f"Loading {len(load_symbols)} symbols (excluded: {list(EXCLUDED_SYMBOLS)})...")
    daily_bars = load_round1_daily_ohlcv(symbols=load_symbols)
    print(f"  rows={len(daily_bars):,}  symbols={daily_bars['symbol'].nunique()}")

    # --- 2. Build research dataset for H=20 only ---
    print(f"Building research dataset for horizon={HORIZON}...")
    dataset = build_round1_research_dataset(daily_bars, horizons=[HORIZON])
    h_samples = dataset[dataset["horizon"] == HORIZON].copy().reset_index(drop=True)
    print(f"  dataset rows={len(h_samples):,}  symbols={h_samples['symbol'].nunique()}")

    # --- 3. Build folds ---
    signal_dates = h_samples["signal_date"].drop_duplicates()
    folds = build_round1_walk_forward_folds(signal_dates)
    print(f"  folds={len(folds)}")

    # --- 4. Validate label windows (no-leakage check) ---
    validate_fold_label_windows(h_samples, folds)
    print("  label window validation passed")

    # --- 5. Walk-forward predictions ---
    xgb_config = XGBoostRegressorConfig()
    prediction_config = WalkForwardPredictionConfig()
    print("Running walk-forward predictions...")
    predictions = run_walk_forward_predictions(
        h_samples,
        folds,
        lambda: XGBoostRegressorPredictor(xgb_config),
        config=prediction_config,
    )
    print(f"  prediction rows={len(predictions):,}")

    # --- 6. Compute OOS Rank IC ---
    print("Computing OOS Rank IC...")
    metric_config = RankICConfig()
    panel, fold_summary, horizon_summary = evaluate_oos_rank_ic(predictions, folds, metric_config)

    # --- 7. Save artifacts ---
    print(f"Saving to {OUTPUT_DIR}...")
    panel.to_csv(OUTPUT_DIR / "daily_rank_ic.csv", index=False)
    fold_summary.to_csv(OUTPUT_DIR / "fold_rank_ic_summary.csv", index=False)
    horizon_summary.to_csv(OUTPUT_DIR / "horizon_rank_ic_summary.csv", index=False)

    slim_cols = [
        "fold_id", "signal_date", "symbol", "horizon",
        prediction_config.score_column,
        prediction_config.label_column,
    ]
    predictions[slim_cols].to_parquet(OUTPUT_DIR / "oos_predictions_slim.parquet", index=False)

    feature_columns = sorted(c for c in h_samples.columns if c.startswith("alpha_"))
    summary = {
        "run_id": RUN_ID,
        "cell_id": CELL_ID,
        "status": "primary_family_research_period_only",
        "preregistration_id": "SP500RA-V1-R1",
        "model_family": MODEL_FAMILY,
        "horizon": HORIZON,
        "final_holdout_viewed": False,
        "excluded_symbols": list(EXCLUDED_SYMBOLS),
        "bars_rows": len(daily_bars),
        "raw_symbols_loaded": len(load_symbols),
        "dataset_rows": len(h_samples),
        "dataset_symbols": int(h_samples["symbol"].nunique()),
        "feature_count": len(feature_columns),
        "alpha_ids": [c.replace("alpha_", "") for c in feature_columns],
        "fold_count": len(folds),
        "prediction_rows": len(predictions),
        "xgboost_config": asdict(xgb_config),
        "horizon_summary": horizon_summary.to_dict("records"),
        "fold_summary": fold_summary.to_dict("records"),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # --- 8. Print result ---
    row = horizon_summary[horizon_summary["horizon"] == HORIZON].iloc[0]
    print(f"\n=== {CELL_ID} results ===")
    print(f"  mean Rank IC : {row['mean_rank_ic']:.4f}")
    print(f"  positive rate: {row['positive_rate']:.2%}")
    print(f"  ICIR         : {row['icir']:.3f}")
    print(f"  HAC t-stat   : {row['hac_t_stat']:.3f}")
    print(f"  p (one-sided): {row['p_value_one_sided']:.4f}")
    print(f"  bootstrap CI : [{row['bootstrap_ci_lower']:.4f}, {row['bootstrap_ci_upper']:.4f}]")
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
