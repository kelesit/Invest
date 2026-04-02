"""信号组合研究：对比不同信号组合方式的效果。

用法: uv run python tools/signal_combination.py

组合方式:
    1. 等权平均 — 所有信号简单平均
    2. 逆波动率加权 — 历史波动率低的信号权重高
    3. 低相关性加权 — 与其他信号相关性低的权重高

产出:
    output/combination_equity.png    — 各组合方式权益曲线对比
    output/combination_summary.csv   — 绩效汇总
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
WEIGHT_LOOKBACK = 252  # 计算权重的回看窗口


def run_combined_signal(all_data, products, combined_signal_by_product):
    """用组合信号跑多品种回测，返回组合权益曲线。"""
    n_products = len(products)
    all_pnl = {}

    for product in products:
        daily = all_data[product]
        params = PRODUCTS[product]
        signal = combined_signal_by_product[product]

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
        all_pnl[product] = result["net_pnl"]

    pnl_df = pd.DataFrame(all_pnl).fillna(0)
    portfolio_pnl = pnl_df.sum(axis=1)
    portfolio_equity = CAPITAL + portfolio_pnl.cumsum()
    return portfolio_equity, portfolio_pnl


def main():
    print("加载数据...")
    all_data = load_multiple(DATA_DIR)
    products = [p for p in PRODUCTS if p in all_data]
    signal_names = list(SIGNAL_REGISTRY.keys())
    n_signals = len(signal_names)

    # ============================================================
    # 1. 生成所有单信号的信号序列
    # ============================================================
    print("生成所有信号...")
    # {product: {signal_name: signal_series}}
    all_signals = {p: {} for p in products}
    # {signal_name: portfolio_daily_pnl} 用于计算权重
    signal_portfolio_pnl = {}

    for sig_name in signal_names:
        sig_pnl = {}
        for product in products:
            daily = all_data[product]
            signal = generate_signal(daily, sig_name)
            all_signals[product][sig_name] = signal

            params = PRODUCTS[product]
            position = volatility_sized_position(
                signal=signal,
                high=daily["high"],
                low=daily["low"],
                close=daily["close"],
                capital=CAPITAL,
                risk_fraction=RISK_FRACTION / len(products),
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
            sig_pnl[product] = result["net_pnl"]

        pnl_df = pd.DataFrame(sig_pnl).fillna(0)
        signal_portfolio_pnl[sig_name] = pnl_df.sum(axis=1)

    # ============================================================
    # 2. 构建组合信号
    # ============================================================
    print("构建组合信号...")
    combinations = {}

    # --- 2a. 等权平均 ---
    equal_weight_signals = {}
    for product in products:
        sig_stack = pd.concat(
            [all_signals[product][s] for s in signal_names], axis=1
        )
        equal_weight_signals[product] = sig_stack.mean(axis=1).clip(-1, 1)
    combinations["Equal Weight"] = equal_weight_signals

    # --- 2b. 逆波动率加权 ---
    # 权重 = 1/σ，σ = 滚动窗口内的日度 PnL 标准差
    pnl_df_all = pd.DataFrame(signal_portfolio_pnl)
    rolling_vol = pnl_df_all.rolling(WEIGHT_LOOKBACK, min_periods=63).std()
    inv_vol_weights = (1.0 / rolling_vol.replace(0, np.nan))
    # 归一化使权重之和 = 1
    inv_vol_weights = inv_vol_weights.div(inv_vol_weights.sum(axis=1), axis=0).fillna(1.0 / n_signals)

    inv_vol_signals = {}
    for product in products:
        sig_stack = pd.concat(
            [all_signals[product][s] for s in signal_names], axis=1
        )
        sig_stack.columns = signal_names
        # 用当天的权重乘以各信号
        weighted = sig_stack.multiply(inv_vol_weights.reindex(sig_stack.index).fillna(1.0 / n_signals))
        inv_vol_signals[product] = weighted.sum(axis=1).clip(-1, 1)
    combinations["Inverse Vol"] = inv_vol_signals

    # --- 2c. 低相关性加权 ---
    # 权重 = 1 / 平均相关性（与其他信号的平均相关性越低，权重越高）
    rolling_corr_weights = {}
    common_idx = pnl_df_all.dropna().index

    # 用滚动窗口计算相关性矩阵，取每个信号与其他信号的平均相关性
    def calc_corr_weights(window_pnl):
        corr = window_pnl.corr()
        # 平均相关性（排除自身）
        n = len(corr)
        avg_corr = (corr.sum(axis=1) - 1) / (n - 1)
        # 逆相关性作为权重
        inv_corr = 1.0 / avg_corr.clip(0.1)  # 避免除以零或负数
        return inv_corr / inv_corr.sum()

    # 简化：用整体相关性（非滚动），避免计算量过大
    overall_corr_weights = calc_corr_weights(pnl_df_all.dropna())
    print(f"  低相关性权重: {dict(zip(signal_names, overall_corr_weights.round(3)))}")

    low_corr_signals = {}
    for product in products:
        sig_stack = pd.concat(
            [all_signals[product][s] for s in signal_names], axis=1
        )
        sig_stack.columns = signal_names
        weighted = sig_stack.multiply(overall_corr_weights)
        low_corr_signals[product] = weighted.sum(axis=1).clip(-1, 1)
    combinations["Low Correlation"] = low_corr_signals

    # ============================================================
    # 3. 跑回测并对比
    # ============================================================
    print("\n跑组合回测...")
    all_equity = {}
    all_results = []

    # 单信号中最好的作为 baseline
    best_single_name = None
    best_single_sharpe = -999
    for sig_name in signal_names:
        pnl = signal_portfolio_pnl[sig_name]
        equity = CAPITAL + pnl.cumsum()
        metrics = calc_metrics(equity)
        if metrics["sharpe_ratio"] > best_single_sharpe:
            best_single_sharpe = metrics["sharpe_ratio"]
            best_single_name = sig_name
        all_results.append({"method": f"Single: {sig_name}", **metrics})

    # 最佳单信号
    best_equity = CAPITAL + signal_portfolio_pnl[best_single_name].cumsum()
    all_equity[f"Best Single ({best_single_name})"] = best_equity

    # 组合方式
    for combo_name, combo_signals in combinations.items():
        equity, pnl = run_combined_signal(all_data, products, combo_signals)
        all_equity[combo_name] = equity
        metrics = calc_metrics(equity)
        all_results.append({"method": combo_name, **metrics})
        print(f"  {combo_name}: Sharpe={metrics['sharpe_ratio']:.2f}")

    # ============================================================
    # 4. 绘图
    # ============================================================
    print("\n绘制对比图...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # 权益曲线
    for name, equity in all_equity.items():
        lw = 2 if "Single" not in name else 1.5
        ls = "-" if "Single" not in name else "--"
        axes[0].plot(equity, label=name, linewidth=lw, linestyle=ls)
    axes[0].set_title("Signal Combination Comparison")
    axes[0].set_ylabel("Equity")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # 回撤对比
    for name, equity in all_equity.items():
        if "Single" in name:
            continue
        dd = (equity - equity.cummax()) / equity.cummax()
        axes[1].plot(dd, label=name, linewidth=1)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combination_equity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: output/combination_equity.png")

    # 汇总表
    summary_df = pd.DataFrame(all_results).round(4)
    summary_df.to_csv(OUTPUT_DIR / "combination_summary.csv", index=False)
    print(f"  已保存: output/combination_summary.csv")

    # 打印对比
    print(f"\n{'=' * 70}")
    print("  组合方式对比")
    print(f"{'=' * 70}")
    combo_rows = summary_df[~summary_df["method"].str.startswith("Single")].sort_values(
        "sharpe_ratio", ascending=False
    )
    # 加上最佳单信号
    best_row = summary_df[summary_df["method"] == f"Single: {best_single_name}"].iloc[0]
    print(f"  {'Best Single (' + best_single_name + ')':<25} "
          f"Sharpe={best_row['sharpe_ratio']:>6.2f}  "
          f"Return={best_row['annual_return']:>7.2%}  "
          f"MaxDD={best_row['max_drawdown']:>7.2%}  "
          f"Calmar={best_row['calmar_ratio']:>5.2f}")
    print(f"  {'─' * 65}")
    for _, row in combo_rows.iterrows():
        print(f"  {row['method']:<25} "
              f"Sharpe={row['sharpe_ratio']:>6.2f}  "
              f"Return={row['annual_return']:>7.2%}  "
              f"MaxDD={row['max_drawdown']:>7.2%}  "
              f"Calmar={row['calmar_ratio']:>5.2f}")


if __name__ == "__main__":
    main()
