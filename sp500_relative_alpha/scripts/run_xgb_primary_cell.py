"""Run XGBoost primary experiment for a given horizon with all 52 allowlisted Alpha101 features.

Usage:
    uv run python sp500_relative_alpha/scripts/run_xgb_primary_cell.py --horizon 5
    uv run python sp500_relative_alpha/scripts/run_xgb_primary_cell.py --horizon 20
"""

from __future__ import annotations

import argparse
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

EXCLUDED_SYMBOLS: tuple[str, ...] = ("HUBB",)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main(horizon: int) -> None:
    run_id = f"xgb_h{horizon:02d}_full52_primary_v1"
    cell_id = f"H{horizon:02d}_XGB"
    output_dir = PROJECT_ROOT / "sp500_relative_alpha" / "artifacts" / "model_runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {cell_id} | horizon={horizon} | run_id={run_id} ===")

    # 1. Load data
    manifest = pd.read_csv(ROUND1_SNAPSHOT_MANIFEST)
    load_symbols = [s for s in manifest["symbol"].tolist() if s not in EXCLUDED_SYMBOLS]
    print(f"Loading {len(load_symbols)} symbols...")
    daily_bars = load_round1_daily_ohlcv(symbols=load_symbols)
    print(f"  rows={len(daily_bars):,}  symbols={daily_bars['symbol'].nunique()}")

    # 2. Research dataset for this horizon only
    print(f"Building research dataset (horizon={horizon})...")
    dataset = build_round1_research_dataset(daily_bars, horizons=[horizon])
    h_samples = dataset[dataset["horizon"] == horizon].copy().reset_index(drop=True)
    print(f"  rows={len(h_samples):,}  symbols={h_samples['symbol'].nunique()}")

    # 3. Folds + label validation
    folds = build_round1_walk_forward_folds(h_samples["signal_date"].drop_duplicates())
    validate_fold_label_windows(h_samples, folds)
    print(f"  folds={len(folds)}  label validation passed")

    # 4. Walk-forward predictions
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

    # 5. OOS Rank IC
    print("Computing OOS Rank IC...")
    panel, fold_summary, horizon_summary = evaluate_oos_rank_ic(predictions, folds, RankICConfig())

    # 6. Save artifacts
    panel.to_csv(output_dir / "daily_rank_ic.csv", index=False)
    fold_summary.to_csv(output_dir / "fold_rank_ic_summary.csv", index=False)
    horizon_summary.to_csv(output_dir / "horizon_rank_ic_summary.csv", index=False)

    slim_cols = [
        "fold_id", "signal_date", "symbol", "horizon",
        prediction_config.score_column,
        prediction_config.label_column,
    ]
    predictions[slim_cols].to_parquet(output_dir / "oos_predictions_slim.parquet", index=False)

    feature_columns = sorted(c for c in h_samples.columns if c.startswith("alpha_"))
    summary = {
        "run_id": run_id,
        "cell_id": cell_id,
        "status": "primary_family_research_period_only",
        "preregistration_id": "SP500RA-V1-R1",
        "model_family": "XGBoost",
        "horizon": horizon,
        "final_holdout_viewed": False,
        "excluded_symbols": list(EXCLUDED_SYMBOLS),
        "bars_rows": len(daily_bars),
        "raw_symbols_loaded": len(load_symbols),
        "dataset_rows": len(h_samples),
        "dataset_symbols": int(h_samples["symbol"].nunique()),
        "feature_count": len(feature_columns),
        "fold_count": len(folds),
        "prediction_rows": len(predictions),
        "xgboost_config": asdict(xgb_config),
        "horizon_summary": horizon_summary.to_dict("records"),
        "fold_summary": fold_summary.to_dict("records"),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # 7. Print result
    row = horizon_summary[horizon_summary["horizon"] == horizon].iloc[0]
    print(f"\n=== {cell_id} results ===")
    print(f"  mean Rank IC : {row['mean_rank_ic']:.4f}")
    print(f"  positive rate: {row['positive_rate']:.2%}")
    print(f"  ICIR         : {row['icir']:.3f}")
    print(f"  HAC t-stat   : {row['hac_t_stat']:.3f}")
    print(f"  p (one-sided): {row['p_value_one_sided']:.4f}")
    print(f"  bootstrap CI : [{row['bootstrap_ci_lower']:.4f}, {row['bootstrap_ci_upper']:.4f}]")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True, help="Prediction horizon in trading days (e.g. 5, 10, 20)")
    args = parser.parse_args()
    main(args.horizon)
