"""绩效分析与可视化。

输入: 权益曲线 (backtest 输出)
输出: 绩效指标 + 图表
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def calc_metrics(equity: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """计算核心绩效指标。

    参数:
        equity: 每日权益曲线
        risk_free_rate: 年化无风险利率

    返回: 指标字典
    """
    daily_returns = equity.pct_change().dropna()

    # 年化收益率
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_days = len(equity)
    annual_return = (1 + total_return) ** (252 / n_days) - 1

    # 年化波动率
    annual_vol = daily_returns.std() * np.sqrt(252)

    # 夏普比率
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0

    # 最大回撤
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown = drawdown.min()

    # Calmar 比率
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # 胜率（按日）
    win_rate = (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else 0.0

    # 盈亏比
    avg_win = daily_returns[daily_returns > 0].mean() if (daily_returns > 0).any() else 0.0
    avg_loss = abs(daily_returns[daily_returns < 0].mean()) if (daily_returns < 0).any() else 1.0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trading_days": n_days,
    }


def print_metrics(metrics: dict) -> None:
    """打印绩效指标。"""
    print("\n" + "=" * 40)
    print("  绩效报告")
    print("=" * 40)
    labels = {
        "total_return": "总收益",
        "annual_return": "年化收益",
        "annual_volatility": "年化波动率",
        "sharpe_ratio": "夏普比率",
        "max_drawdown": "最大回撤",
        "calmar_ratio": "Calmar比率",
        "win_rate": "日胜率",
        "profit_factor": "盈亏比",
        "trading_days": "交易天数",
    }
    fmt = {
        "total_return": lambda v: f"{v:.2%}",
        "annual_return": lambda v: f"{v:.2%}",
        "annual_volatility": lambda v: f"{v:.2%}",
        "sharpe_ratio": lambda v: f"{v:.2f}",
        "max_drawdown": lambda v: f"{v:.2%}",
        "calmar_ratio": lambda v: f"{v:.2f}",
        "win_rate": lambda v: f"{v:.2%}",
        "profit_factor": lambda v: f"{v:.2f}",
        "trading_days": lambda v: f"{v}",
    }
    for key, label in labels.items():
        value = fmt[key](metrics[key])
        print(f"  {label:<10} {value:>10}")
    print("=" * 40)


def plot_backtest(result: pd.DataFrame, title: str = "CTA Backtest", save_prefix: str = "backtest") -> None:
    """绘制回测结果图表。

    包含：权益曲线、回撤、持仓变化。
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14)

    # 权益曲线
    axes[0].plot(result["equity"], linewidth=1)
    axes[0].set_ylabel("Equity")
    axes[0].grid(True, alpha=0.3)

    # 回撤
    cummax = result["equity"].cummax()
    drawdown = (result["equity"] - cummax) / cummax
    axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.5, color="red")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.3)

    # 持仓
    axes[2].plot(result["position"], linewidth=0.8, color="green")
    axes[2].set_ylabel("Position")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = OUTPUT_DIR / f"{save_prefix}_result.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存: {filename}")


def plot_monthly_returns(equity: pd.Series, save_prefix: str = "backtest") -> None:
    """绘制月度收益热力图。"""
    daily_returns = equity.pct_change().dropna()
    monthly = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = monthly_df.pivot(index="year", columns="month", values="return")

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Monthly Returns")
    plt.tight_layout()
    filename = OUTPUT_DIR / f"{save_prefix}_monthly.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存: {filename}")
