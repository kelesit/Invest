"""CTA 最小可用系统 — 入口文件。

用法: uv run python main.py
"""

from pathlib import Path

import pandas as pd

from cta.data_loader import load_multiple
from cta.signals import SIGNAL_REGISTRY, generate_signal
from cta.position_sizing import volatility_sized_position
from cta.backtest import run_backtest
from cta.analysis import calc_metrics, print_metrics, plot_backtest, plot_monthly_returns
from cta.risk import trailing_stop, vol_scale, vol_target_portfolio

# ============================================================
# 配置
# ============================================================

# 品种配置：每个品种的合约参数
PRODUCTS = {
    "ES": {"point_value": 50.0, "commission": 2.5, "slippage_points": 0.25},   # E-mini S&P 500
    "CL": {"point_value": 1000.0, "commission": 2.5, "slippage_points": 0.02}, # Crude Oil
    "GC": {"point_value": 100.0, "commission": 2.5, "slippage_points": 0.10},  # Gold
    "ZN": {"point_value": 1000.0, "commission": 2.5, "slippage_points": 0.01}, # 10Y Treasury
}

CONFIG = {
    # 数据
    "data_dir": Path("data/raw"),

    # 运行哪些品种（从 PRODUCTS 中选）
    "products": ["ES", "CL", "GC", "ZN"],

    # 策略参数（可用信号见 SIGNAL_REGISTRY）
    "signal_fn": "combined_momentum",

    # 仓位管理
    "initial_capital": 1_000_000,
    "risk_fraction": 0.01,
    "atr_period": 20,

    # 风控（设为 None 关闭）
    "trailing_stop_atr": 3.0,       # 追踪止损 ATR 倍数，None=不止损
    "vol_target": 0.10,             # 组合波动率目标，None=不做 vol targeting
}


def main():
    cfg = CONFIG

    # 加载所有品种数据
    print("加载数据...")
    all_data = load_multiple(cfg["data_dir"])
    print(f"  可用品种: {list(all_data.keys())}")

    products = cfg["products"]
    print(f"  本次运行: {products}")
    print(f"  信号: {cfg['signal_fn']}")
    stop_label = "关闭" if cfg["trailing_stop_atr"] is None else f"{cfg['trailing_stop_atr']}×ATR"
    vol_label = "关闭" if cfg["vol_target"] is None else f"{cfg['vol_target']:.0%}"
    print(f"  止损: {stop_label}")
    print(f"  波动率目标: {vol_label}")
    print()

    # 收集每个品种的每日净收益，用于组合
    all_net_pnl = {}
    all_positions = {}

    for product in products:
        if product not in all_data:
            print(f"  {product}: 无数据，跳过")
            continue

        daily = all_data[product]
        params = PRODUCTS[product]
        print(f"{'=' * 50}")
        print(f"  {product}")
        print(f"  {daily.index.min().date()} ~ {daily.index.max().date()}, {len(daily)} 个交易日")
        print(f"{'=' * 50}")

        # 生成信号
        signal = generate_signal(daily, cfg["signal_fn"])

        # 计算仓位
        position = volatility_sized_position(
            signal=signal,
            high=daily["high"],
            low=daily["low"],
            close=daily["close"],
            capital=cfg["initial_capital"],
            risk_fraction=cfg["risk_fraction"] / len(products),
            point_value=params["point_value"],
            atr_period=cfg["atr_period"],
        )

        # 追踪止损
        if cfg["trailing_stop_atr"] is not None:
            position = trailing_stop(
                position, daily["close"],
                atr_mult=cfg["trailing_stop_atr"],
                atr_period=cfg["atr_period"],
                high=daily["high"], low=daily["low"],
            )

        all_positions[product] = position

        # 回测
        result = run_backtest(
            daily_price=daily,
            position=position,
            point_value=params["point_value"],
            commission_per_contract=params["commission"],
            slippage_points=params["slippage_points"],
            initial_capital=cfg["initial_capital"],
        )

        # 单品种绩效
        metrics = calc_metrics(result["equity"])
        print_metrics(metrics)
        plot_backtest(result, title=f"{product} CTA — {cfg['signal_fn']}", save_prefix=product)
        print()

        # 收集净收益
        all_net_pnl[product] = result["net_pnl"]

    # ============================================================
    # 组合波动率目标（如果开启）
    # ============================================================
    if cfg["vol_target"] is not None and len(all_net_pnl) > 1:
        portfolio_pnl_raw = pd.DataFrame(all_net_pnl).fillna(0).sum(axis=1)
        scale = vol_target_portfolio(
            portfolio_pnl_raw, cfg["initial_capital"],
            target_vol=cfg["vol_target"], vol_window=60,
        )

        # 用缩放后的仓位重新跑回测
        all_net_pnl = {}
        for product in all_positions:
            daily = all_data[product]
            params = PRODUCTS[product]
            scaled_pos = all_positions[product] * scale.reindex(all_positions[product].index).fillna(1.0)
            result = run_backtest(
                daily_price=daily, position=scaled_pos,
                point_value=params["point_value"],
                commission_per_contract=params["commission"],
                slippage_points=params["slippage_points"],
                initial_capital=cfg["initial_capital"],
            )
            all_net_pnl[product] = result["net_pnl"]

    # ============================================================
    # 多品种组合
    # ============================================================
    if len(all_net_pnl) > 1:
        print(f"\n{'#' * 50}")
        print(f"  组合绩效（{len(all_net_pnl)} 个品种）")
        print(f"{'#' * 50}")

        # 合并所有品种的每日净收益，缺失填0
        pnl_df = pd.DataFrame(all_net_pnl)
        pnl_df = pnl_df.fillna(0)

        # 组合每日总收益
        portfolio_pnl = pnl_df.sum(axis=1)
        portfolio_equity = cfg["initial_capital"] + portfolio_pnl.cumsum()

        # 构造组合 result 用于绘图
        portfolio_result = pd.DataFrame({
            "net_pnl": portfolio_pnl,
            "equity": portfolio_equity,
            "position": pnl_df.apply(lambda x: (x != 0).astype(int)).sum(axis=1),
        })

        metrics = calc_metrics(portfolio_equity)
        print_metrics(metrics)
        plot_backtest(portfolio_result, title=f"Portfolio CTA ({len(all_net_pnl)} products)", save_prefix="portfolio")
        plot_monthly_returns(portfolio_equity, save_prefix="portfolio")

        # 品种贡献分解
        print("\n  品种收益贡献:")
        for product in pnl_df.columns:
            total = pnl_df[product].sum()
            pct = total / (portfolio_equity.iloc[-1] - cfg["initial_capital"]) * 100
            print(f"    {product:<4}  ${total:>12,.0f}  ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
