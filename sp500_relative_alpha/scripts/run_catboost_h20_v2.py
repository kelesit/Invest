"""CatBoost v2 experiment: H=20, OHLCV primitives + Alpha101, full transform tree.

Feature pipeline:
  - 15 OHLCV primitives (ret_Nd, intraday structure, volume)
  - 51 Alpha101 outputs
  - Transforms: identity + cs_rank + ts_zscore(20,60,120) + ts_change(5,20,60)
    + second-order combinations
  - Selected cross-combinations (momentum × volume, risk-adjusted momentum,
    multi-horizon momentum spread)
  - All features prefixed with "feat_"

Usage:
    uv run python -m sp500_relative_alpha.scripts.run_catboost_h20_v2
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from sp500_relative_alpha.catboost_models import CatBoostRegressorConfig, CatBoostRegressorPredictor
from sp500_relative_alpha.daily_data_loader import ROUND1_SNAPSHOT_MANIFEST, load_round1_daily_ohlcv
from sp500_relative_alpha.feature_transforms import CrossCombinationSpec, TransformConfig
from sp500_relative_alpha.folds import build_round1_walk_forward_folds, validate_fold_label_windows
from sp500_relative_alpha.metrics import RankICConfig, evaluate_oos_rank_ic
from sp500_relative_alpha.modeling import WalkForwardPredictionConfig, run_walk_forward_predictions
from sp500_relative_alpha.research_dataset import build_v2_research_dataset

HORIZON = 20
RUN_ID = "catboost_h20_v2"
MODEL_FAMILY = "CatBoost"
CELL_ID = f"H{HORIZON:02d}_CB_V2"

EXCLUDED_SYMBOLS: tuple[str, ...] = ("HUBB",)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "sp500_relative_alpha" / "artifacts" / "model_runs" / RUN_ID

# Cross-combinations: operate on cs_rank versions (bounded [0,1], numerically safe)
CROSS_COMBINATIONS = (
    # momentum × volume: price signal confirmed by volume
    CrossCombinationSpec("ret_20d", "volume_ratio_20d", "mul", "ret20d_x_volr20d"),
    CrossCombinationSpec("ret_60d", "volume_ratio_60d", "mul", "ret60d_x_volr60d"),
    # risk-adjusted momentum: ret / volatility (both in cs_rank space)
    CrossCombinationSpec("ret_20d", "high_low_range", "sub", "ret20d_sub_hlr"),
    CrossCombinationSpec("ret_60d", "high_low_range", "sub", "ret60d_sub_hlr"),
    # multi-horizon momentum spread: short minus long
    CrossCombinationSpec("ret_5d", "ret_60d", "sub", "ret5d_sub_ret60d"),
    CrossCombinationSpec("ret_20d", "ret_252d", "sub", "ret20d_sub_ret252d"),
    # price position × volume
    CrossCombinationSpec("close_position", "volume_ratio_20d", "mul", "closepos_x_volr20d"),
)

TRANSFORM_CONFIG = TransformConfig(
    keep_identity=True,
    apply_cs_rank=True,
    ts_zscore_windows=(20, 60, 120),
    ts_change_windows=(5, 20, 60),
    apply_second_order=True,
    cross_combinations=CROSS_COMBINATIONS,
    output_prefix="feat_",
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load data ---
    manifest = pd.read_csv(ROUND1_SNAPSHOT_MANIFEST)
    all_symbols = manifest["symbol"].tolist()
    load_symbols = [s for s in all_symbols if s not in EXCLUDED_SYMBOLS]
    print(f"Loading {len(load_symbols)} symbols...")
    daily_bars = load_round1_daily_ohlcv(symbols=load_symbols)
    print(f"  rows={len(daily_bars):,}  symbols={daily_bars['symbol'].nunique()}")

    # --- 2. Build v2 dataset ---
    print("Building v2 feature dataset...")
    dataset = build_v2_research_dataset(
        daily_bars,
        horizons=[HORIZON],
        transform_config=TRANSFORM_CONFIG,
    )
    h_samples = dataset[dataset["horizon"] == HORIZON].copy().reset_index(drop=True)
    feature_cols = sorted(c for c in h_samples.columns if c.startswith("feat_"))
    print(f"  rows={len(h_samples):,}  symbols={h_samples['symbol'].nunique()}  features={len(feature_cols)}")

    # --- 3. Folds ---
    signal_dates = h_samples["signal_date"].drop_duplicates()
    folds = build_round1_walk_forward_folds(signal_dates)
    validate_fold_label_windows(h_samples, folds)
    print(f"  folds={len(folds)}, label window validation passed")

    # --- 4. Walk-forward predictions ---
    cb_config = CatBoostRegressorConfig()
    prediction_config = WalkForwardPredictionConfig(feature_prefix="feat_", n_top_features=200)
    print("Running walk-forward predictions (CatBoost)...")
    predictions = run_walk_forward_predictions(
        h_samples,
        folds,
        lambda: CatBoostRegressorPredictor(cb_config),
        config=prediction_config,
    )
    print(f"  prediction rows={len(predictions):,}")

    # --- 5. Rank IC ---
    print("Computing OOS Rank IC...")
    panel, fold_summary, horizon_summary = evaluate_oos_rank_ic(predictions, folds, RankICConfig())

    # --- 6. Save ---
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

    summary = {
        "run_id": RUN_ID,
        "cell_id": CELL_ID,
        "model_family": MODEL_FAMILY,
        "horizon": HORIZON,
        "excluded_symbols": list(EXCLUDED_SYMBOLS),
        "bars_rows": len(daily_bars),
        "raw_symbols_loaded": len(load_symbols),
        "dataset_rows": len(h_samples),
        "dataset_symbols": int(h_samples["symbol"].nunique()),
        "feature_count": len(feature_cols),
        "fold_count": len(folds),
        "prediction_rows": len(predictions),
        "transform_config": {
            "ts_zscore_windows": list(TRANSFORM_CONFIG.ts_zscore_windows),
            "ts_change_windows": list(TRANSFORM_CONFIG.ts_change_windows),
            "apply_second_order": TRANSFORM_CONFIG.apply_second_order,
            "cross_combinations": [
                {"left": s.left, "right": s.right, "op": s.op, "name": s.name}
                for s in CROSS_COMBINATIONS
            ],
        },
        "catboost_config": asdict(cb_config),
        "horizon_summary": horizon_summary.to_dict("records"),
        "fold_summary": fold_summary.to_dict("records"),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

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
