"""Cross-sectional equity backtest engine.

Implements a simple top-N long-only strategy with periodic rebalancing
and transaction cost modeling.
"""

import numpy as np
import pandas as pd


def backtest_topN(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    top_n: int = 30,
    rebalance_days: int = 10,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Backtest a long-only top-N strategy based on model predictions.

    Every rebalance_days trading days, select the top_n stocks by prediction
    score, hold equal-weight positions until next rebalance.

    Args:
        predictions: DataFrame with columns [prediction, actual],
            MultiIndex (date, ticker). From model.run_cv_pipeline output.
        returns: MultiIndex (date, ticker) Series or DataFrame with daily
            stock returns (not residual — raw returns for realistic P&L).
        top_n: Number of stocks to hold.
        rebalance_days: Rebalance frequency in trading days.
        cost_bps: One-way transaction cost in basis points.

    Returns:
        DataFrame indexed by date with columns:
        - portfolio_return: daily portfolio return (after costs)
        - turnover: fraction of portfolio traded
        - n_holdings: number of stocks held
    """
    # Get prediction dates
    pred_dates = predictions.index.get_level_values("date").unique().sort_values()

    # Daily returns for all stocks (unstacked: date × ticker)
    if isinstance(returns, pd.Series):
        returns_wide = returns.unstack("ticker")
    else:
        returns_wide = returns["daily_ret"].unstack("ticker") if "daily_ret" in returns.columns else returns.unstack("ticker")

    # Only use dates where we have both predictions and returns
    common_dates = pred_dates.intersection(returns_wide.index)

    # Determine rebalance dates
    rebalance_dates = common_dates[::rebalance_days]

    # Track portfolio
    current_weights = pd.Series(dtype=float)  # ticker → weight
    daily_results = []

    for date in common_dates:
        if date in rebalance_dates:
            # Get predictions for this date
            if date in predictions.index.get_level_values("date"):
                day_preds = predictions.xs(date, level="date")["prediction"]
                # Select top N
                top_tickers = day_preds.nlargest(top_n).index.tolist()
                new_weights = pd.Series(1.0 / len(top_tickers), index=top_tickers)
            else:
                new_weights = current_weights

            # Calculate turnover
            all_tickers = set(current_weights.index) | set(new_weights.index)
            old_w = current_weights.reindex(all_tickers, fill_value=0)
            new_w = new_weights.reindex(all_tickers, fill_value=0)
            turnover = (new_w - old_w).abs().sum() / 2  # one-way

            current_weights = new_weights
        else:
            turnover = 0.0

        # Daily return = sum of weight × stock return
        day_returns = returns_wide.loc[date] if date in returns_wide.index else pd.Series(dtype=float)
        held_tickers = current_weights.index.intersection(day_returns.index)

        if len(held_tickers) > 0:
            port_ret = (current_weights[held_tickers] * day_returns[held_tickers]).sum()
        else:
            port_ret = 0.0

        # Transaction cost
        cost = turnover * cost_bps / 10000 * 2  # × 2 for round-trip approximation

        daily_results.append({
            "date": date,
            "portfolio_return": port_ret - cost,
            "gross_return": port_ret,
            "turnover": turnover,
            "cost": cost,
            "n_holdings": len(current_weights),
        })

    results = pd.DataFrame(daily_results).set_index("date")
    results["cumulative_return"] = (1 + results["portfolio_return"]).cumprod()
    return results


def compute_daily_returns(panel: pd.DataFrame) -> pd.Series:
    """Compute daily returns from panel data for backtest use.

    Args:
        panel: MultiIndex (date, ticker) with 'close' column.

    Returns:
        Series with MultiIndex (date, ticker) of daily returns.
    """
    returns = panel.groupby(level="ticker")["close"].pct_change()
    returns.name = "daily_ret"
    return returns
