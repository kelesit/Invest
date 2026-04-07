"""S&P 500 stock data download and management via yfinance."""

from io import StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf


# Hardcoded fallback removed — we scrape the live list from Wikipedia.
# If Wikipedia changes its table layout, this function will need updating.


def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            html = response.read().decode("utf-8")
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to fetch S&P 500 constituents from Wikipedia: {exc}") from exc

    tables = pd.read_html(StringIO(html))
    tickers = tables[0]["Symbol"].str.strip().tolist()
    # yfinance uses dots not hyphens (BRK.B not BRK-B)
    tickers = [t.replace(".", "-") for t in tickers]
    return sorted(tickers)


def download_stock_data(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str = "2026-04-01",
    cache_dir: str | Path = "data/equity",
) -> None:
    """Download daily OHLCV for each ticker and save as parquet.

    Skips tickers that already have a cached file.
    Also downloads SPY as market benchmark.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_tickers = list(set(tickers + ["SPY"]))
    to_download = [t for t in all_tickers if not (cache_dir / f"{t}.parquet").exists()]

    if not to_download:
        print(f"All {len(all_tickers)} tickers already cached.")
        return

    print(f"Downloading {len(to_download)} tickers ({len(all_tickers) - len(to_download)} cached)...")

    # yfinance batch download — much faster than one-by-one
    data = yf.download(
        to_download,
        start=start,
        end=end,
        auto_adjust=True,
        threads=True,
        progress=True,
    )

    if data.empty:
        print("WARNING: yfinance returned empty data.")
        return

    # yf.download returns MultiIndex columns: (field, ticker) for multiple tickers
    # For single ticker it returns flat columns
    if len(to_download) == 1:
        ticker = to_download[0]
        data.columns = data.columns.str.lower()
        data.index.name = "date"
        data = data[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        data.to_parquet(cache_dir / f"{ticker}.parquet")
        print(f"  Saved {ticker}: {len(data)} days")
    else:
        saved, failed = 0, 0
        for ticker in to_download:
            try:
                df = data.xs(ticker, level=1, axis=1).copy()
                df.columns = df.columns.str.lower()
                df.index.name = "date"
                df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
                if len(df) < 100:
                    print(f"  SKIP {ticker}: only {len(df)} days")
                    failed += 1
                    continue
                df.to_parquet(cache_dir / f"{ticker}.parquet")
                saved += 1
            except (KeyError, ValueError):
                failed += 1
        print(f"Saved {saved} tickers, skipped {failed}.")


def load_universe(
    cache_dir: str | Path = "data/equity",
    min_history_days: int = 252,
) -> pd.DataFrame:
    """Load all cached stock data into a single panel DataFrame.

    Returns:
        DataFrame with MultiIndex (date, ticker) and columns [open, high, low, close, volume].
        Only includes stocks with at least min_history_days of data.
    """
    cache_dir = Path(cache_dir)
    parquet_files = sorted(cache_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {cache_dir}. Run download_stock_data first.")

    frames = []
    for f in parquet_files:
        ticker = f.stem
        if ticker == "SPY":
            continue  # SPY is benchmark, not in universe
        df = pd.read_parquet(f)
        if len(df) < min_history_days:
            continue
        df["ticker"] = ticker
        frames.append(df)

    panel = pd.concat(frames)
    panel = panel.reset_index().set_index(["date", "ticker"]).sort_index()
    return panel


def load_spy(cache_dir: str | Path = "data/equity") -> pd.DataFrame:
    """Load SPY benchmark data."""
    path = Path(cache_dir) / "SPY.parquet"
    if not path.exists():
        raise FileNotFoundError(f"SPY data not found at {path}. Run download_stock_data first.")
    return pd.read_parquet(path)
