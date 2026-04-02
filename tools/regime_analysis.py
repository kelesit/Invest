"""市场环境分析：各信号在不同市场状态下的表现差异。

用法: uv run python tools/regime_analysis.py

市场环境划分：
    1. 趋势 vs 震荡 — 用效率比率 (Efficiency Ratio) 衡量
    2. 高波动 vs 低波动 — 用滚动波动率的中位数分位

产出:
    output/regime_signal_performance.png — 各信号在不同环境下的夏普对比
    output/regime_timeline.png          — 市场环境时间线
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cta.data_loader import load_multiple
from cta.signals import SIGNAL_REGISTRY, generate_signal
from cta.position_sizing import volatility_sized_position
from cta.backtest import run_backtest
from cta.analysis import calc_metrics

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

PRODUCTS = {
    "ES": {"point_value": 50.0, "commission": 2.5, "slippage_points": 0.25},
    "CL": {"point_value": 1000.0, "commission": 2.5, "slippage_points": 0.02},
    "GC": {"point_value": 100.0, "commission": 2.5, "slippage_points": 0.10},
    "ZN": {"point_value": 1000.0, "commission": 2.5, "slippage_points": 0.01},
}

CAPITAL = 1_000_000
RISK_FRACTION = 0.01

# 环境划分参数
ER_WINDOW = 60       # 效率比率窗口
ER_THRESHOLD = 0.3   # 效率比率阈值：> 0.3 = 趋势, <= 0.3 = 震荡
VOL_WINDOW = 60      # 波动率计算窗口
VOL_LOOKBACK = 252   # 波动率分位数的回看窗口


def calc_efficiency_ratio(close: pd.Series, window: int = 60) -> pd.Series:
    """效率比率 (Kaufman's Efficiency Ratio)。

    ER = |净变动| / 路径总长度
    - ER → 1: 价格沿一个方向持续运动（强趋势）
    - ER → 0: 价格来回波动但没有净位移（震荡）
    """
    net_change = (close - close.shift(window)).abs()
    daily_changes = close.diff().abs()
    path_length = daily_changes.rolling(window).sum()
    er = net_change / path_length.replace(0, np.nan)
    return er.fillna(0)


def classify_regime(daily: pd.DataFrame) -> pd.DataFrame:
    """对单品种划分市场环境，返回包含环境标签的 DataFrame。"""
    close = daily["close"]

    # 趋势 vs 震荡
    er = calc_efficiency_ratio(close, ER_WINDOW)
    is_trending = er > ER_THRESHOLD

    # 高波动 vs 低波动
    daily_returns = close.pct_change()
    rolling_vol = daily_returns.rolling(VOL_WINDOW).std() * np.sqrt(252)
    vol_median = rolling_vol.rolling(VOL_LOOKBACK, min_periods=120).median()
    is_high_vol = rolling_vol > vol_median

    # 组合成 4 种环境
    regime = pd.Series("", index=daily.index)
    regime[is_trending & is_high_vol] = "Trend+HighVol"
    regime[is_trending & ~is_high_vol] = "Trend+LowVol"
    regime[~is_trending & is_high_vol] = "Chop+HighVol"
    regime[~is_trending & ~is_high_vol] = "Chop+LowVol"

    # 前面的 NaN 期间标记为空
    regime[er.isna() | rolling_vol.isna() | vol_median.isna()] = ""

    return pd.DataFrame({
        "efficiency_ratio": er,
        "rolling_vol": rolling_vol,
        "regime": regime,
    }, index=daily.index)


def main():
    print("加载数据...")
    all_data = load_multiple(DATA_DIR)
    products = [p for p in PRODUCTS if p in all_data]
    signal_names = list(SIGNAL_REGISTRY.keys())
    n_products = len(products)

    # ============================================================
    # 1. 划分环境 & 计算各信号每日收益
    # ============================================================
    print("划分市场环境并计算各信号收益...")

    # 收集所有品种的环境标签（取主品种 ES 的环境）
    regimes = {}
    signal_daily_pnl = {s: {} for s in signal_names}

    for product in products:
        daily = all_data[product]
        params = PRODUCTS[product]
        regimes[product] = classify_regime(daily)

        for sig_name in signal_names:
            signal = generate_signal(daily, sig_name)
            position = volatility_sized_position(
                signal=signal,
                high=daily["high"],
                low=daily["low"],
                close=daily["close"],
                capital=CAPITAL,
                risk_fraction=RISK_FRACTION / n_products,
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
            signal_daily_pnl[sig_name][product] = result["net_pnl"]

    # 汇总组合级别每日 PnL
    portfolio_pnl = {}
    for sig_name in signal_names:
        pnl_df = pd.DataFrame(signal_daily_pnl[sig_name]).fillna(0)
        portfolio_pnl[sig_name] = pnl_df.sum(axis=1)

    # 用所有品种的环境标签的众数作为"市场环境"
    regime_dfs = pd.DataFrame({p: regimes[p]["regime"] for p in products})
    # 简化：用 ES 的环境（主市场）
    market_regime = regimes["ES"]["regime"] if "ES" in regimes else regimes[products[0]]["regime"]

    # ============================================================
    # 2. 按环境计算各信号年化夏普
    # ============================================================
    print("按环境计算绩效...")

    regime_labels = ["Trend+HighVol", "Trend+LowVol", "Chop+HighVol", "Chop+LowVol"]
    results = []

    for sig_name in signal_names:
        pnl = portfolio_pnl[sig_name]
        for regime_label in regime_labels:
            mask = market_regime == regime_label
            regime_pnl = pnl[mask]
            if len(regime_pnl) < 30:
                continue
            # 年化夏普
            daily_ret = regime_pnl / CAPITAL
            sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
            results.append({
                "signal": sig_name,
                "regime": regime_label,
                "sharpe": sharpe,
                "avg_daily_pnl": regime_pnl.mean(),
                "days": len(regime_pnl),
            })

    results_df = pd.DataFrame(results)

    # ============================================================
    # 3. 绘制环境分析图
    # ============================================================
    print("绘制环境分析图...")

    # 3a. 信号×环境 夏普热力图
    pivot = results_df.pivot(index="signal", columns="regime", values="sharpe")
    pivot = pivot.reindex(columns=regime_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Sharpe by Signal × Market Regime (ES regime classification)")
    ax.set_ylabel("Signal")
    ax.set_xlabel("Market Regime")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "regime_signal_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: output/regime_signal_performance.png")

    # 3b. 环境时间线
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Market Regime Timeline (ES)", fontsize=14)

    es_data = all_data["ES"] if "ES" in all_data else all_data[products[0]]
    es_regime = regimes["ES"] if "ES" in regimes else regimes[products[0]]

    # 价格
    axes[0].plot(es_data["close"], linewidth=0.8)
    axes[0].set_ylabel("Close Price")
    axes[0].grid(True, alpha=0.3)

    # 效率比率
    axes[1].plot(es_regime["efficiency_ratio"], linewidth=0.8, color="blue")
    axes[1].axhline(ER_THRESHOLD, color="red", linestyle="--", alpha=0.7, label=f"ER threshold={ER_THRESHOLD}")
    axes[1].set_ylabel("Efficiency Ratio")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 环境色带
    regime_colors = {
        "Trend+HighVol": "green",
        "Trend+LowVol": "lightgreen",
        "Chop+HighVol": "red",
        "Chop+LowVol": "lightyellow",
    }
    regime_series = es_regime["regime"]
    for label, color in regime_colors.items():
        mask = regime_series == label
        if mask.any():
            axes[2].fill_between(
                regime_series.index, 0, 1, where=mask,
                color=color, alpha=0.7, label=label,
            )
    axes[2].set_ylabel("Regime")
    axes[2].set_yticks([])
    axes[2].legend(loc="upper left", fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "regime_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: output/regime_timeline.png")

    # 打印统计
    print(f"\n{'=' * 60}")
    print("  各环境天数统计 (ES)")
    print(f"{'=' * 60}")
    for label in regime_labels:
        days = (market_regime == label).sum()
        pct = days / (market_regime != "").sum() * 100
        print(f"  {label:<12} {days:>5} 天 ({pct:>5.1f}%)")

    print(f"\n{'=' * 60}")
    print("  各环境下表现最好的信号")
    print(f"{'=' * 60}")
    for label in regime_labels:
        subset = results_df[results_df["regime"] == label].sort_values("sharpe", ascending=False)
        if not subset.empty:
            best = subset.iloc[0]
            print(f"  {label:<12} → {best['signal']:<20} (Sharpe={best['sharpe']:.2f})")


if __name__ == "__main__":
    main()
