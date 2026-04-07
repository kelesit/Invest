"""Label construction for equity cross-sectional prediction.

Main label: forward N-day beta-adjusted residual return.

Theory:
  residual = r_stock - β × r_market
  where β is estimated via rolling OLS regression of stock returns on
  market returns over a trailing window.

Previous version assumed β=1 for all stocks, which systematically
biases labels for stocks with β far from 1 (utilities ~0.5, tech ~1.3).
"""

import numpy as np
import pandas as pd


def compute_forward_returns(close: pd.Series, periods: int = 10) -> pd.Series:
    """Compute forward N-day return.

    For date T, this is the return from T close to T+periods close.
    Using shift(-periods) so the label at T reflects the FUTURE return.
    """
    return close.pct_change(periods).shift(-periods)


def estimate_rolling_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252,
    min_periods: int = 126,
) -> pd.Series:
    """Estimate rolling β via covariance / variance.

    β = Cov(r_stock, r_market) / Var(r_market)

    This is equivalent to OLS slope but much faster to compute.

    Args:
        stock_returns: Daily stock returns.
        market_returns: Daily market (SPY) returns, aligned to same dates.
        window: Rolling window in trading days (252 = ~1 year).
        min_periods: Minimum observations required for a valid estimate.

    Returns:
        Series of rolling β values, same index as stock_returns.
    """
    cov = stock_returns.rolling(window, min_periods=min_periods).cov(market_returns)
    var = market_returns.rolling(window, min_periods=min_periods).var()
    beta = cov / var.replace(0, np.nan)
    return beta


def make_labels(
    panel: pd.DataFrame,
    spy: pd.DataFrame,
    periods: int = 10,
    beta_window: int = 252,
) -> pd.Series:
    """Construct beta-adjusted residual return labels for all stocks.

    For each stock on each date:
      label = forward_return_stock - β × forward_return_SPY

    Args:
        panel: MultiIndex (date, ticker) with at least a 'close' column.
        spy: SPY DataFrame with 'close' column, indexed by date.
        periods: Forward return horizon in trading days.
        beta_window: Rolling window for β estimation.

    Returns:
        Series with MultiIndex (date, ticker), named 'label'.
    """
    # Market daily and forward returns
    spy_daily_ret = spy["close"].pct_change()
    spy_fwd = compute_forward_returns(spy["close"], periods)

    tickers = panel.index.get_level_values("ticker").unique()
    results = []

    for ticker in tickers:
        stock_close = panel.xs(ticker, level="ticker")["close"]
        stock_daily_ret = stock_close.pct_change()

        # Estimate rolling β from daily returns
        aligned_market = spy_daily_ret.reindex(stock_daily_ret.index)
        beta = estimate_rolling_beta(
            stock_daily_ret, aligned_market, window=beta_window
        )

        # Forward returns
        stock_fwd = compute_forward_returns(stock_close, periods)
        aligned_spy_fwd = spy_fwd.reindex(stock_fwd.index)

        # Beta-adjusted residual
        residual = stock_fwd - beta * aligned_spy_fwd

        residual_df = residual.to_frame("label")
        residual_df["ticker"] = ticker
        residual_df.index.name = "date"
        results.append(residual_df.reset_index())

    labels = pd.concat(results, ignore_index=True)
    labels = labels.set_index(["date", "ticker"]).sort_index()
    return labels["label"]


def rank_labels(labels: pd.Series) -> pd.Series:
    """Transform labels to cross-sectional percentile ranks [0, 1].

    For each date, ranks all stocks' labels. This removes extreme values
    and normalizes the distribution, making the regression target much
    easier for the model to learn.

    Args:
        labels: Series with MultiIndex (date, ticker).

    Returns:
        Same shape Series with values in [0, 1].
    """
    return labels.groupby(level="date").rank(pct=True)
