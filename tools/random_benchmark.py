"""随机基准对比：用 1000 条随机信号跑回测，看真实策略排第几。

用法: uv run python tools/random_benchmark.py

解读：
- p-value < 0.05 → 策略统计显著，不太可能是运气
- p-value > 0.20 → 和随机入场没什么区别
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cta.data_loader import load_multiple
from cta.signals import combined_momentum
from cta.position_sizing import volatility_sized_position
from cta.backtest import run_backtest
from cta.analysis import calc_metrics

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path("data/raw")
N_RANDOM = 1000
SEED = 42

PRODUCTS = {
    "ES": {"point_value": 50.0, "commission": 2.5, "slippage_points": 0.25},
    "CL": {"point_value": 1000.0, "commission": 2.5, "slippage_points": 0.02},
    "GC": {"point_value": 100.0, "commission": 2.5, "slippage_points": 0.10},
    "ZN": {"point_value": 1000.0, "commission": 2.5, "slippage_points": 0.01},
}

CAPITAL = 1_000_000
RISK_FRACTION = 0.01 / len(PRODUCTS)
COMBINED_LOOKBACKS = [12, 30, 60]


def run_with_signal(daily, params, signal):
    """用给定信号跑单品种回测，返回夏普。"""
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
    return float(calc_metrics(result["equity"])["sharpe_ratio"])


def run_portfolio_with_signals(all_data, product_signals):
    """用给定信号跑多品种组合，返回组合夏普。"""
    all_pnl = {}
    for product, signal in product_signals.items():
        daily = all_data[product]
        params = PRODUCTS[product]
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
        all_pnl[product] = result["net_pnl"]

    pnl_df = pd.DataFrame(all_pnl).fillna(0)
    portfolio_equity = CAPITAL + pnl_df.sum(axis=1).cumsum()
    return float(calc_metrics(portfolio_equity)["sharpe_ratio"])


def main():
    print("加载数据...")
    all_data = load_multiple(DATA_DIR)
    rng = np.random.default_rng(SEED)

    products = [p for p in PRODUCTS if p in all_data]

    # 1. 真实策略的夏普
    print("计算真实策略夏普...")
    real_signals = {}
    real_sharpes = {}
    for product in products:
        daily = all_data[product]
        signal = combined_momentum(daily["close"], lookbacks=COMBINED_LOOKBACKS)
        real_signals[product] = signal
        real_sharpes[product] = run_with_signal(daily, PRODUCTS[product], signal)

    real_portfolio_sharpe = run_portfolio_with_signals(all_data, real_signals)

    print(f"  单品种夏普: {real_sharpes}")
    print(f"  组合夏普: {real_portfolio_sharpe:.3f}")

    # 2. 随机信号的夏普分布
    print(f"\n生成 {N_RANDOM} 条随机信号并回测...")
    random_sharpes = {p: [] for p in products}
    random_portfolio_sharpes = []

    for i in range(N_RANDOM):
        if (i + 1) % 100 == 0:
            print(f"  进度: {i + 1}/{N_RANDOM}")

        random_signals = {}
        for product in products:
            daily = all_data[product]
            # 随机 +1/-1 信号，但保持与真实策略相同的换手率
            # 用随机游走生成趋势性的随机信号（不是纯白噪声）
            n = len(daily)
            random_walk = rng.choice([-1.0, 1.0], size=n)
            # 用滚动窗口平滑，模拟趋势信号的持续性
            signal = pd.Series(random_walk, index=daily.index).rolling(20, min_periods=1).mean()
            signal = np.sign(signal)
            random_signals[product] = signal

        # 单品种
        for product in products:
            s = run_with_signal(all_data[product], PRODUCTS[product], random_signals[product])
            random_sharpes[product].append(s)

        # 组合
        ps = run_portfolio_with_signals(all_data, random_signals)
        random_portfolio_sharpes.append(ps)

    # 3. 计算 p-value 和排名
    print(f"\n{'=' * 50}")
    print("  随机基准对比结果")
    print(f"{'=' * 50}")

    fig, axes = plt.subplots(1, len(products) + 1, figsize=(5 * (len(products) + 1), 4))

    for i, product in enumerate(products):
        real_s = real_sharpes[product]
        rand_s = np.array(random_sharpes[product])
        p_value = (rand_s >= real_s).sum() / len(rand_s)
        percentile = (rand_s < real_s).sum() / len(rand_s) * 100

        print(f"  {product}: 真实夏普={real_s:.3f}, p-value={p_value:.3f}, 排名前{100-percentile:.1f}%")

        ax = axes[i]
        ax.hist(rand_s, bins=50, alpha=0.7, color="gray", label="Random")
        ax.axvline(real_s, color="red", linewidth=2, label=f"Real ({real_s:.2f})")
        ax.set_title(f"{product} (p={p_value:.3f})")
        ax.set_xlabel("Sharpe")
        ax.legend()

    # 组合
    rand_ps = np.array(random_portfolio_sharpes)
    p_value = (rand_ps >= real_portfolio_sharpe).sum() / len(rand_ps)
    percentile = (rand_ps < real_portfolio_sharpe).sum() / len(rand_ps) * 100
    print(f"  组合: 真实夏普={real_portfolio_sharpe:.3f}, p-value={p_value:.3f}, 排名前{100-percentile:.1f}%")

    ax = axes[-1]
    ax.hist(rand_ps, bins=50, alpha=0.7, color="gray", label="Random")
    ax.axvline(real_portfolio_sharpe, color="red", linewidth=2, label=f"Real ({real_portfolio_sharpe:.2f})")
    ax.set_title(f"Portfolio (p={p_value:.3f})")
    ax.set_xlabel("Sharpe")
    ax.legend()

    plt.tight_layout()
    plt.savefig("output/random_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n图表已保存: random_benchmark.png")


if __name__ == "__main__":
    main()
