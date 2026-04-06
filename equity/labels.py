"""Label construction for equity cross-sectional prediction.

Main label: forward N-day residual return = stock return - market (SPY) return.

Theory vs approximation:
- Correct: residual return = r_i - beta_i * r_m (beta-adjusted)
- Approximation used here: r_i - r_SPY (assumes beta = 1 for all stocks)
- Error source: stocks with beta far from 1 (e.g., utilities ~0.5, tech ~1.3)
  will have systematically biased labels. This is acceptable for a first version
  because the cross-sectional rank transformation in features partially absorbs
  this bias, and beta estimation itself introduces noise.
"""

import pandas as pd


def compute_forward_returns(close: pd.Series, periods: int = 10) -> pd.Series:
    """Compute forward N-day return.

    For date T, this is the return from T+1 close to T+periods close.
    Using shift(-periods) so the label at T reflects the FUTURE return.

    Args:
        close: Daily close prices indexed by date.
        periods: Number of forward days.

    Returns:
        Series of forward returns, NaN for the last `periods` dates.
    """
    return close.pct_change(periods).shift(-periods)


def make_labels(
    panel: pd.DataFrame,
    spy: pd.DataFrame,
    periods: int = 10,
) -> pd.Series:
    """Construct residual return labels for all stocks.

    Args:
        panel: MultiIndex (date, ticker) with at least a 'close' column.
        spy: SPY DataFrame with 'close' column, indexed by date.
        periods: Forward return horizon in trading days.

    Returns:
        Series with MultiIndex (date, ticker), named 'label'.
    """
    # SPY forward return — one value per date
    spy_fwd = compute_forward_returns(spy["close"], periods)
    spy_fwd.name = "spy_fwd_ret"

    # Per-stock forward return
    tickers = panel.index.get_level_values("ticker").unique()
    results = []

    for ticker in tickers:
        stock_close = panel.xs(ticker, level="ticker")["close"]
        stock_fwd = compute_forward_returns(stock_close, periods)

        # Residual = stock - market
        aligned_spy = spy_fwd.reindex(stock_fwd.index)
        residual = stock_fwd - aligned_spy

        residual_df = residual.to_frame("label")
        residual_df["ticker"] = ticker
        residual_df.index.name = "date"
        results.append(residual_df.reset_index())

    labels = pd.concat(results, ignore_index=True)
    labels = labels.set_index(["date", "ticker"]).sort_index()
    return labels["label"]
