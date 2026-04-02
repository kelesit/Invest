"""风控对比研究：对比不同风控机制对组合表现的影响。

用法: uv run python tools/risk_compare.py

对比方案：
    1. 基线 — 无风控（等权 + ATR sizing）
    2. 追踪止损 — 3×ATR 止损
    3. 波动率择时 — 单品种层面 vol scaling
    4. 组合波动率目标 — 组合层面 vol targeting
    5. 逆波动率权重 — 替代等权
    6. 风险平价 — 考虑相关性的权重分配
    7. 全套风控 — 风险平价 + 波动率目标 + 止损

产出:
    output/risk_equity_compare.png   — 各方案权益曲线
    output/risk_drawdown_compare.png — 回撤对比
    output/risk_summary.csv          — 绩效汇总
    output/risk_correlation.png      — 品种间动态相关性
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cta.data_loader import load_multiple
from cta.signals import generate_signal
from cta.position_sizing import volatility_sized_position
from cta.backtest import run_backtest
from cta.analysis import calc_metrics
from cta.risk import trailing_stop, vol_scale, vol_target_portfolio
from cta.portfolio import (
    inverse_vol_weights,
    risk_parity_weights,
    calc_dynamic_correlation,
)

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
SIGNAL_NAME = "ma_crossover"  # 用项目二中表现最好的信号


def run_baseline(all_data, products):
    """方案1：基线 — 等权 + ATR sizing，无风控。"""
    n = len(products)
    all_pnl = {}
    for product in products:
        daily = all_data[product]
        params = PRODUCTS[product]
        signal = generate_signal(daily, SIGNAL_NAME)
        position = volatility_sized_position(
            signal=signal, high=daily["high"], low=daily["low"], close=daily["close"],
            capital=CAPITAL, risk_fraction=RISK_FRACTION / n,
            point_value=params["point_value"], atr_period=20,
        )
        result = run_backtest(
            daily_price=daily, position=position,
            point_value=params["point_value"],
            commission_per_contract=params["commission"],
            slippage_points=params["slippage_points"],
            initial_capital=CAPITAL,
        )
        all_pnl[product] = result["net_pnl"]
    return _combine_pnl(all_pnl)


def run_with_trailing_stop(all_data, products):
    """方案2：追踪止损 — 3×ATR。"""
    n = len(products)
    all_pnl = {}
    for product in products:
        daily = all_data[product]
        params = PRODUCTS[product]
        signal = generate_signal(daily, SIGNAL_NAME)
        position = volatility_sized_position(
            signal=signal, high=daily["high"], low=daily["low"], close=daily["close"],
            capital=CAPITAL, risk_fraction=RISK_FRACTION / n,
            point_value=params["point_value"], atr_period=20,
        )
        position = trailing_stop(
            position, daily["close"], atr_mult=3.0, atr_period=20,
            high=daily["high"], low=daily["low"],
        )
        result = run_backtest(
            daily_price=daily, position=position,
            point_value=params["point_value"],
            commission_per_contract=params["commission"],
            slippage_points=params["slippage_points"],
            initial_capital=CAPITAL,
        )
        all_pnl[product] = result["net_pnl"]
    return _combine_pnl(all_pnl)


def run_with_vol_scaling(all_data, products):
    """方案3：单品种波动率择时。"""
    n = len(products)
    all_pnl = {}
    for product in products:
        daily = all_data[product]
        params = PRODUCTS[product]
        signal = generate_signal(daily, SIGNAL_NAME)
        position = volatility_sized_position(
            signal=signal, high=daily["high"], low=daily["low"], close=daily["close"],
            capital=CAPITAL, risk_fraction=RISK_FRACTION / n,
            point_value=params["point_value"], atr_period=20,
        )
        position = vol_scale(position, daily["close"], target_vol=0.15, vol_window=60)
        result = run_backtest(
            daily_price=daily, position=position,
            point_value=params["point_value"],
            commission_per_contract=params["commission"],
            slippage_points=params["slippage_points"],
            initial_capital=CAPITAL,
        )
        all_pnl[product] = result["net_pnl"]
    return _combine_pnl(all_pnl)


def run_with_vol_target(all_data, products):
    """方案4：组合层面波动率目标。"""
    # 先跑一遍基线拿到组合 PnL
    n = len(products)
    all_pnl = {}
    all_positions = {}
    all_results = {}
    for product in products:
        daily = all_data[product]
        params = PRODUCTS[product]
        signal = generate_signal(daily, SIGNAL_NAME)
        position = volatility_sized_position(
            signal=signal, high=daily["high"], low=daily["low"], close=daily["close"],
            capital=CAPITAL, risk_fraction=RISK_FRACTION / n,
            point_value=params["point_value"], atr_period=20,
        )
        all_positions[product] = position
        result = run_backtest(
            daily_price=daily, position=position,
            point_value=params["point_value"],
            commission_per_contract=params["commission"],
            slippage_points=params["slippage_points"],
            initial_capital=CAPITAL,
        )
        all_pnl[product] = result["net_pnl"]

    # 计算组合波动率缩放系数
    portfolio_pnl = pd.DataFrame(all_pnl).fillna(0).sum(axis=1)
    scale = vol_target_portfolio(portfolio_pnl, CAPITAL, target_vol=0.10, vol_window=60)

    # 重新用缩放后的仓位跑回测
    scaled_pnl = {}
    for product in products:
        daily = all_data[product]
        params = PRODUCTS[product]
        scaled_pos = all_positions[product] * scale.reindex(all_positions[product].index).fillna(1.0)
        result = run_backtest(
            daily_price=daily, position=scaled_pos,
            point_value=params["point_value"],
            commission_per_contract=params["commission"],
            slippage_points=params["slippage_points"],
            initial_capital=CAPITAL,
        )
        scaled_pnl[product] = result["net_pnl"]
    return _combine_pnl(scaled_pnl)


def run_with_inv_vol_weights(all_data, products):
    """方案5：逆波动率权重。"""
    # 计算每个品种的日收益率
    product_returns = {}
    for product in products:
        daily = all_data[product]
        product_returns[product] = daily["close"].pct_change()

    weights = inverse_vol_weights(product_returns, vol_window=60)

    all_pnl = {}
    for product in products:
        daily = all_data[product]
        params = PRODUCTS[product]
        signal = generate_signal(daily, SIGNAL_NAME)

        # 用动态权重替代等权
        w = weights[product].reindex(daily.index).fillna(1.0 / len(products))
        position = volatility_sized_position(
            signal=signal, high=daily["high"], low=daily["low"], close=daily["close"],
            capital=CAPITAL, risk_fraction=RISK_FRACTION * w,
            point_value=params["point_value"], atr_period=20,
        )
        result = run_backtest(
            daily_price=daily, position=position,
            point_value=params["point_value"],
            commission_per_contract=params["commission"],
            slippage_points=params["slippage_points"],
            initial_capital=CAPITAL,
        )
        all_pnl[product] = result["net_pnl"]
    return _combine_pnl(all_pnl)


def run_with_risk_parity(all_data, products):
    """方案6：风险平价权重。"""
    product_returns = {}
    for product in products:
        daily = all_data[product]
        product_returns[product] = daily["close"].pct_change()

    weights = risk_parity_weights(product_returns, corr_window=120, vol_window=60)

    all_pnl = {}
    for product in products:
        daily = all_data[product]
        params = PRODUCTS[product]
        signal = generate_signal(daily, SIGNAL_NAME)

        w = weights[product].reindex(daily.index).fillna(1.0 / len(products))
        position = volatility_sized_position(
            signal=signal, high=daily["high"], low=daily["low"], close=daily["close"],
            capital=CAPITAL, risk_fraction=RISK_FRACTION * w,
            point_value=params["point_value"], atr_period=20,
        )
        result = run_backtest(
            daily_price=daily, position=position,
            point_value=params["point_value"],
            commission_per_contract=params["commission"],
            slippage_points=params["slippage_points"],
            initial_capital=CAPITAL,
        )
        all_pnl[product] = result["net_pnl"]
    return _combine_pnl(all_pnl)


def run_full_risk_management(all_data, products):
    """方案7：全套风控 — 风险平价 + 波动率目标 + 止损。"""
    product_returns = {}
    for product in products:
        daily = all_data[product]
        product_returns[product] = daily["close"].pct_change()

    weights = risk_parity_weights(product_returns, corr_window=120, vol_window=60)

    # 第一轮：用风险平价权重 + 止损
    all_pnl = {}
    all_positions = {}
    for product in products:
        daily = all_data[product]
        params = PRODUCTS[product]
        signal = generate_signal(daily, SIGNAL_NAME)

        w = weights[product].reindex(daily.index).fillna(1.0 / len(products))
        position = volatility_sized_position(
            signal=signal, high=daily["high"], low=daily["low"], close=daily["close"],
            capital=CAPITAL, risk_fraction=RISK_FRACTION * w,
            point_value=params["point_value"], atr_period=20,
        )
        position = trailing_stop(
            position, daily["close"], atr_mult=3.0, atr_period=20,
            high=daily["high"], low=daily["low"],
        )
        all_positions[product] = position

        result = run_backtest(
            daily_price=daily, position=position,
            point_value=params["point_value"],
            commission_per_contract=params["commission"],
            slippage_points=params["slippage_points"],
            initial_capital=CAPITAL,
        )
        all_pnl[product] = result["net_pnl"]

    # 第二轮：组合波动率目标
    portfolio_pnl = pd.DataFrame(all_pnl).fillna(0).sum(axis=1)
    scale = vol_target_portfolio(portfolio_pnl, CAPITAL, target_vol=0.10, vol_window=60)

    scaled_pnl = {}
    for product in products:
        daily = all_data[product]
        params = PRODUCTS[product]
        scaled_pos = all_positions[product] * scale.reindex(all_positions[product].index).fillna(1.0)
        result = run_backtest(
            daily_price=daily, position=scaled_pos,
            point_value=params["point_value"],
            commission_per_contract=params["commission"],
            slippage_points=params["slippage_points"],
            initial_capital=CAPITAL,
        )
        scaled_pnl[product] = result["net_pnl"]
    return _combine_pnl(scaled_pnl)


def _combine_pnl(all_pnl):
    """合并各品种 PnL，返回 (组合权益曲线, 组合日 PnL)。"""
    pnl_df = pd.DataFrame(all_pnl).fillna(0)
    portfolio_pnl = pnl_df.sum(axis=1)
    portfolio_equity = CAPITAL + portfolio_pnl.cumsum()
    return portfolio_equity, portfolio_pnl


def main():
    print("加载数据...")
    all_data = load_multiple(DATA_DIR)
    products = [p for p in PRODUCTS if p in all_data]
    print(f"品种: {products}, 信号: {SIGNAL_NAME}\n")

    # ============================================================
    # 1. 跑所有方案
    # ============================================================
    scenarios = {
        "1. Baseline (no risk mgmt)": run_baseline,
        "2. Trailing Stop (3×ATR)": run_with_trailing_stop,
        "3. Vol Scaling (single)": run_with_vol_scaling,
        "4. Vol Target (portfolio)": run_with_vol_target,
        "5. Inverse Vol Weights": run_with_inv_vol_weights,
        "6. Risk Parity": run_with_risk_parity,
        "7. Full (RP+VolTarget+Stop)": run_full_risk_management,
    }

    results = {}
    all_metrics = []
    for name, fn in scenarios.items():
        print(f"  {name}...")
        equity, pnl = fn(all_data, products)
        results[name] = {"equity": equity, "pnl": pnl}
        metrics = calc_metrics(equity)
        metrics["method"] = name
        all_metrics.append(metrics)

    # ============================================================
    # 2. 权益曲线对比
    # ============================================================
    print("\n绘制对比图...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    for name, data in results.items():
        lw = 2.0 if "Full" in name or "Baseline" in name else 1.0
        alpha = 1.0 if "Full" in name or "Baseline" in name else 0.7
        axes[0].plot(data["equity"], label=name, linewidth=lw, alpha=alpha)

    axes[0].set_title(f"Risk Management Comparison (signal: {SIGNAL_NAME})")
    axes[0].set_ylabel("Equity")
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # 回撤
    for name, data in results.items():
        eq = data["equity"]
        dd = (eq - eq.cummax()) / eq.cummax()
        lw = 1.5 if "Full" in name or "Baseline" in name else 0.8
        axes[1].plot(dd, label=name, linewidth=lw)
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(loc="lower left", fontsize=6)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "risk_equity_compare.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: output/risk_equity_compare.png")

    # ============================================================
    # 3. 品种间动态相关性
    # ============================================================
    print("计算品种间动态相关性...")
    product_returns = {p: all_data[p]["close"].pct_change() for p in products}
    corr_data = calc_dynamic_correlation(product_returns, window=60)

    fig, ax = plt.subplots(figsize=(14, 5))
    for pair in corr_data["pair"].unique():
        pair_data = corr_data[corr_data["pair"] == pair]
        ax.plot(pair_data["date"], pair_data["correlation"], label=pair, linewidth=0.8, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Rolling 60-day Correlation Between Products")
    ax.set_ylabel("Correlation")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "risk_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: output/risk_correlation.png")

    # ============================================================
    # 4. 汇总表
    # ============================================================
    summary_df = pd.DataFrame(all_metrics)
    cols = ["method", "sharpe_ratio", "annual_return", "annual_volatility",
            "max_drawdown", "calmar_ratio", "win_rate", "profit_factor"]
    summary_df = summary_df[cols].round(4)
    summary_df.to_csv(OUTPUT_DIR / "risk_summary.csv", index=False)
    print(f"  已保存: output/risk_summary.csv")

    # 打印排名
    print(f"\n{'=' * 75}")
    print("  风控方案对比（按夏普排序）")
    print(f"{'=' * 75}")
    summary_df = summary_df.sort_values("sharpe_ratio", ascending=False)
    for _, row in summary_df.iterrows():
        print(
            f"  {row['method']:<35} "
            f"Sharpe={row['sharpe_ratio']:>5.2f}  "
            f"Return={row['annual_return']:>7.2%}  "
            f"MaxDD={row['max_drawdown']:>7.2%}  "
            f"Vol={row['annual_volatility']:>6.2%}  "
            f"Calmar={row['calmar_ratio']:>5.2f}"
        )


if __name__ == "__main__":
    main()
