"""Performance analysis and visualization for equity strategy."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def summary_metrics(returns: pd.Series, annual_factor: int = 252) -> dict:
    """Calculate standard performance metrics from a return series."""
    total_ret = (1 + returns).prod() - 1
    n_days = len(returns)
    annual_ret = (1 + total_ret) ** (annual_factor / n_days) - 1
    annual_vol = returns.std() * np.sqrt(annual_factor)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

    # Max drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (returns > 0).mean()

    return {
        "total_return": total_ret,
        "annual_return": annual_ret,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "trading_days": n_days,
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print performance metrics."""
    print(f"  Annual Return:     {metrics['annual_return']:>8.2%}")
    print(f"  Annual Volatility: {metrics['annual_volatility']:>8.2%}")
    print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>8.2f}")
    print(f"  Max Drawdown:      {metrics['max_drawdown']:>8.2%}")
    print(f"  Calmar Ratio:      {metrics['calmar_ratio']:>8.2f}")
    print(f"  Win Rate:          {metrics['win_rate']:>8.2%}")
    print(f"  Trading Days:      {metrics['trading_days']:>8d}")


# ---------------------------------------------------------------------------
# IC Analysis
# ---------------------------------------------------------------------------

def calc_ic_series(predictions: pd.DataFrame) -> pd.Series:
    """Calculate daily Information Coefficient (rank correlation).

    IC = Spearman correlation between prediction and actual return,
    computed cross-sectionally for each date.

    Args:
        predictions: MultiIndex (date, ticker) with columns [prediction, actual].

    Returns:
        Series indexed by date with daily IC values.
    """
    def _daily_ic(group):
        if len(group) < 10:
            return np.nan
        corr, _ = stats.spearmanr(group["prediction"], group["actual"])
        return corr

    ic = predictions.groupby(level="date").apply(_daily_ic)
    ic.name = "IC"
    return ic


# ---------------------------------------------------------------------------
# Quantile Analysis
# ---------------------------------------------------------------------------

def quantile_analysis(
    predictions: pd.DataFrame,
    n_groups: int = 5,
) -> pd.DataFrame:
    """Stratified backtest: group stocks by prediction quintile, compute average return.

    If the model works, top quintile should have highest return and
    bottom quintile the lowest (monotonic increase from Q1 to Q5).

    Args:
        predictions: MultiIndex (date, ticker) with columns [prediction, actual].
        n_groups: Number of quantile groups.

    Returns:
        DataFrame with mean actual return per quantile per date.
    """
    def _assign_quantile(group):
        group = group.copy()
        group["quantile"] = pd.qcut(
            group["prediction"], n_groups, labels=range(1, n_groups + 1), duplicates="drop"
        )
        return group

    with_q = predictions.groupby(level="date").apply(_assign_quantile)
    # Reset multi-level index from groupby
    if with_q.index.nlevels > 2:
        with_q = with_q.droplevel(0)

    quantile_returns = with_q.groupby(["date", "quantile"])["actual"].mean()
    quantile_returns = quantile_returns.unstack("quantile")
    return quantile_returns


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_equity_curve(
    backtest_results: pd.DataFrame,
    spy_returns: pd.Series,
    title: str = "Strategy vs SPY",
) -> plt.Figure:
    """Plot cumulative return of strategy vs SPY benchmark."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})

    # --- Equity curve ---
    ax = axes[0]
    strat_cum = backtest_results["cumulative_return"]

    # Align SPY to same dates
    spy_aligned = spy_returns.reindex(strat_cum.index).fillna(0)
    spy_cum = (1 + spy_aligned).cumprod()

    ax.plot(strat_cum.index, strat_cum.values, label="Strategy", linewidth=1.5)
    ax.plot(spy_cum.index, spy_cum.values, label="SPY", linewidth=1.5, alpha=0.7)
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Drawdown ---
    ax = axes[1]
    peak = strat_cum.cummax()
    dd = (strat_cum - peak) / peak
    ax.fill_between(dd.index, dd.values, 0, alpha=0.5, color="red")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)

    # --- Turnover ---
    ax = axes[2]
    ax.bar(backtest_results.index, backtest_results["turnover"], width=2, alpha=0.6)
    ax.set_ylabel("Turnover")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ic_analysis(ic_series: pd.Series, title: str = "IC Analysis") -> plt.Figure:
    """Plot IC time series and distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # IC time series with rolling mean
    ax = axes[0]
    ax.bar(ic_series.index, ic_series.values, width=2, alpha=0.4, color="steelblue")
    ic_rolling = ic_series.rolling(20).mean()
    ax.plot(ic_rolling.index, ic_rolling.values, color="red", linewidth=1.5, label="20-day MA")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title(f"{title} — Time Series")
    ax.set_ylabel("IC (Spearman)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # IC distribution
    ax = axes[1]
    ax.hist(ic_series.dropna(), bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    mean_ic = ic_series.mean()
    ax.axvline(mean_ic, color="red", linewidth=2, label=f"Mean IC: {mean_ic:.4f}")
    ic_ir = mean_ic / ic_series.std() if ic_series.std() > 0 else 0
    ax.set_title(f"{title} — Distribution (ICIR: {ic_ir:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_quantile_returns(quantile_returns: pd.DataFrame, title: str = "Quantile Analysis") -> plt.Figure:
    """Plot cumulative returns by prediction quintile."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cumulative returns per quintile
    ax = axes[0]
    cum_ret = (1 + quantile_returns).cumprod()
    for col in cum_ret.columns:
        ax.plot(cum_ret.index, cum_ret[col].values, label=f"Q{col}", linewidth=1.2)
    ax.set_title(f"{title} — Cumulative Returns")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Average return per quintile (bar chart — should be monotonically increasing)
    ax = axes[1]
    mean_rets = quantile_returns.mean() * 252  # annualized
    colors = sns.color_palette("RdYlGn", len(mean_rets))
    ax.bar(mean_rets.index, mean_rets.values, color=colors, edgecolor="white")
    ax.set_title(f"{title} — Annualized Mean Return by Quintile")
    ax.set_xlabel("Quintile (1=worst, 5=best)")
    ax.set_ylabel("Annualized Return")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_feature_importance(models: list, feature_names: list[str], top_n: int = 20) -> plt.Figure:
    """Plot average feature importance across CV folds."""
    importances = np.zeros(len(feature_names))
    for model in models:
        importances += model.feature_importance(importance_type="gain")
    importances /= len(models)

    # Sort and take top N
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_values = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_names)), top_values[::-1], color="steelblue", edgecolor="white")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Average Gain Importance")
    ax.set_title(f"Top {top_n} Feature Importance (averaged over {len(models)} folds)")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig
