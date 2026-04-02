"""参数扫描：对 lookback 参数做完整扫描，输出热力图。

用法: uv run python tools/param_scan.py

观察方法：
- 如果大部分参数都赚钱（热力图大面积绿色）→ 信号本身有效
- 如果只有个别参数赚钱（热力图只有零星绿色）→ 过拟合
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cta.data_loader import load_multiple
from cta.signals import momentum
from cta.position_sizing import volatility_sized_position
from cta.backtest import run_backtest
from cta.analysis import calc_metrics

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path("data/raw")
LOOKBACKS = list(range(5, 125, 5))  # 5, 10, 15, ..., 120

PRODUCTS = {
    "ES": {"point_value": 50.0, "commission": 2.5, "slippage_points": 0.25},
    "CL": {"point_value": 1000.0, "commission": 2.5, "slippage_points": 0.02},
    "GC": {"point_value": 100.0, "commission": 2.5, "slippage_points": 0.10},
    "ZN": {"point_value": 1000.0, "commission": 2.5, "slippage_points": 0.01},
}

CAPITAL = 1_000_000
RISK_FRACTION = 0.01 / len(PRODUCTS)

# 时间段
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"


def run_single(daily, params, lookback, period=None):
    """对单品种、单 lookback 跑回测，返回夏普比率。"""
    if period == "train":
        daily = daily[daily.index <= TRAIN_END]
    elif period == "test":
        daily = daily[daily.index >= TEST_START]

    if len(daily) < lookback + 30:
        return np.nan

    signal = momentum(daily["close"], lookback=lookback)
    position = volatility_sized_position(
        signal=signal,
        high=daily["high"],
        low=daily["low"],
        close=daily["close"],
        capital=CAPITAL,
        risk_fraction=RISK_FRACTION,
        point_value=params["point_value"],
        atr_period=20,
    )
    result = run_backtest(
        daily_price=daily,
        position=position,
        point_value=params["point_value"],
        commission_per_contract=params["commission"],
        slippage_points=params["slippage_points"],
        initial_capital=CAPITAL,
    )
    metrics = calc_metrics(result["equity"])
    return metrics["sharpe_ratio"]


def main():
    print("加载数据...")
    all_data = load_multiple(DATA_DIR)

    for period_name, period in [("full (2019-2026)", None), ("train (2019-2023)", "train"), ("test (2024-2026)", "test")]:
        print(f"\n{'=' * 50}")
        print(f"  参数扫描 — {period_name}")
        print(f"{'=' * 50}")

        results = {}
        for product, params in PRODUCTS.items():
            if product not in all_data:
                continue
            daily = all_data[product]
            sharpes = []
            for lb in LOOKBACKS:
                s = run_single(daily, params, lb, period=period)
                sharpes.append(s)
            results[product] = sharpes
            positive = sum(1 for s in sharpes if not np.isnan(s) and s > 0)
            print(f"  {product}: {positive}/{len(sharpes)} 个参数夏普 > 0")

        # 热力图
        df = pd.DataFrame(results, index=LOOKBACKS)
        df.index.name = "lookback"

        fig, ax = plt.subplots(figsize=(14, 5))
        sns.heatmap(
            df.T,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            xticklabels=True,
        )
        suffix = period or "full"
        ax.set_title(f"Sharpe by Lookback — {period_name}")
        ax.set_xlabel("Lookback (days)")
        ax.set_ylabel("Product")
        plt.tight_layout()
        filename = f"output/param_scan_{suffix}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  热力图已保存: {filename}")


if __name__ == "__main__":
    main()
