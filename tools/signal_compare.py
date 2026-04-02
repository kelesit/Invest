"""信号对比仪表盘：一键跑完所有信号 × 所有品种，输出对比图表。

用法: uv run python tools/signal_compare.py

产出:
    output/signal_sharpe_heatmap.png  — 信号×品种 夏普热力图
    output/signal_correlation.png     — 信号间收益相关性矩阵
    output/signal_equity_overlay.png  — 各信号组合权益曲线叠加
    output/signal_summary.csv         — 所有指标汇总表
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


def run_signal_product(daily, params, signal_name, n_products):
    """对单品种、单信号跑回测，返回 (metrics_dict, net_pnl_series)。"""
    signal = generate_signal(daily, signal_name)
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
    metrics = calc_metrics(result["equity"])
    return metrics, result["net_pnl"]


def main():
    print("加载数据...")
    all_data = load_multiple(DATA_DIR)
    products = [p for p in PRODUCTS if p in all_data]
    signal_names = list(SIGNAL_REGISTRY.keys())
    n_products = len(products)

    print(f"品种: {products}")
    print(f"信号: {signal_names}")
    print()

    # ============================================================
    # 1. 跑所有信号 × 品种组合
    # ============================================================
    sharpe_matrix = {}  # {signal: {product: sharpe}}
    all_metrics = []  # 汇总表数据
    portfolio_pnl = {}  # {signal: portfolio daily pnl}
    signal_daily_returns = {}  # {signal: portfolio daily returns} 用于相关性

    for sig_name in signal_names:
        print(f"  {sig_name}...")
        sharpe_matrix[sig_name] = {}
        sig_pnl = {}

        for product in products:
            daily = all_data[product]
            params = PRODUCTS[product]
            try:
                metrics, net_pnl = run_signal_product(daily, params, sig_name, n_products)
                sharpe_matrix[sig_name][product] = metrics["sharpe_ratio"]
                sig_pnl[product] = net_pnl

                all_metrics.append({
                    "signal": sig_name,
                    "product": product,
                    **metrics,
                })
            except Exception as e:
                print(f"    {product} 失败: {e}")
                sharpe_matrix[sig_name][product] = np.nan

        # 组合
        if sig_pnl:
            pnl_df = pd.DataFrame(sig_pnl).fillna(0)
            portfolio_daily_pnl = pnl_df.sum(axis=1)
            portfolio_pnl[sig_name] = portfolio_daily_pnl

            portfolio_equity = CAPITAL + portfolio_daily_pnl.cumsum()
            portfolio_metrics = calc_metrics(portfolio_equity)

            all_metrics.append({
                "signal": sig_name,
                "product": "PORTFOLIO",
                **portfolio_metrics,
            })
            sharpe_matrix[sig_name]["PORTFOLIO"] = portfolio_metrics["sharpe_ratio"]

            # 每日收益率用于相关性计算
            signal_daily_returns[sig_name] = portfolio_equity.pct_change().dropna()

    # ============================================================
    # 2. 夏普热力图
    # ============================================================
    print("\n绘制夏普热力图...")
    sharpe_df = pd.DataFrame(sharpe_matrix).T
    sharpe_df.index.name = "signal"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sharpe_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Sharpe Ratio: Signal × Product")
    ax.set_ylabel("Signal")
    ax.set_xlabel("Product")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "signal_sharpe_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: output/signal_sharpe_heatmap.png")

    # ============================================================
    # 3. 信号相关性矩阵
    # ============================================================
    print("绘制信号相关性矩阵...")
    returns_df = pd.DataFrame(signal_daily_returns)
    corr = returns_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        mask=mask,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Signal Correlation (Portfolio Daily Returns)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "signal_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: output/signal_correlation.png")

    # ============================================================
    # 4. 权益曲线叠加图
    # ============================================================
    print("绘制权益曲线叠加图...")
    fig, ax = plt.subplots(figsize=(14, 7))
    for sig_name, pnl in portfolio_pnl.items():
        equity = CAPITAL + pnl.cumsum()
        ax.plot(equity, label=sig_name, linewidth=1)
    ax.set_title("Portfolio Equity by Signal")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "signal_equity_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: output/signal_equity_overlay.png")

    # ============================================================
    # 5. 汇总表
    # ============================================================
    print("保存汇总表...")
    summary_df = pd.DataFrame(all_metrics)
    summary_df = summary_df.round(4)
    summary_df.to_csv(OUTPUT_DIR / "signal_summary.csv", index=False)
    print(f"  已保存: output/signal_summary.csv")

    # 打印组合级别的排名
    print(f"\n{'=' * 60}")
    print("  组合级别排名（按夏普）")
    print(f"{'=' * 60}")
    portfolio_rows = summary_df[summary_df["product"] == "PORTFOLIO"].sort_values(
        "sharpe_ratio", ascending=False
    )
    for _, row in portfolio_rows.iterrows():
        print(
            f"  {row['signal']:<22} "
            f"Sharpe={row['sharpe_ratio']:>6.2f}  "
            f"Return={row['annual_return']:>7.2%}  "
            f"MaxDD={row['max_drawdown']:>7.2%}  "
            f"Calmar={row['calmar_ratio']:>5.2f}"
        )


if __name__ == "__main__":
    main()
