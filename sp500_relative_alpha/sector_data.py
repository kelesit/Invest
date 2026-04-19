from __future__ import annotations

from pathlib import Path

import pandas as pd

SECTOR_CACHE_PATH = Path(__file__).parent / "artifacts" / "sector_classification.csv"

GICS_SECTORS = (
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
)


class SectorDataError(RuntimeError):
    pass


def load_sector_classification(
    symbols: list[str],
    cache_path: Path = SECTOR_CACHE_PATH,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame with columns [symbol, sector, industry].

    Fetches from yfinance on first call and caches to disk. Pass refresh=True
    to re-fetch even if the cache exists.
    """
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path)
        # top up any symbols missing from the cache
        missing = sorted(set(symbols) - set(cached["symbol"]))
        if missing:
            print(f"Fetching {len(missing)} symbols missing from cache...")
            new_rows = _fetch_from_yfinance(missing)
            cached = pd.concat([cached, new_rows], ignore_index=True)
            cached.to_csv(cache_path, index=False)
        return cached[cached["symbol"].isin(symbols)].reset_index(drop=True)

    print(f"Fetching sector data for {len(symbols)} symbols via yfinance...")
    df = _fetch_from_yfinance(symbols)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Saved to {cache_path}")
    return df


def build_sector_map(symbols: list[str], **kwargs) -> dict[str, str]:
    """Return {symbol: sector} dict, with 'Unknown' for any missing entries."""
    df = load_sector_classification(symbols, **kwargs)
    result = df.set_index("symbol")["sector"].fillna("Unknown").to_dict()
    for sym in symbols:
        result.setdefault(sym, "Unknown")
    return result


def _fetch_from_yfinance(symbols: list[str]) -> pd.DataFrame:
    import yfinance as yf

    rows = []
    for i, sym in enumerate(symbols):
        try:
            info = yf.Ticker(sym).info
            rows.append({
                "symbol": sym,
                "sector": info.get("sector") or None,
                "industry": info.get("industry") or None,
            })
        except Exception as exc:
            rows.append({"symbol": sym, "sector": None, "industry": None})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(symbols)} done")

    df = pd.DataFrame(rows)
    n_missing = df["sector"].isna().sum()
    if n_missing:
        print(f"  {n_missing} symbols returned no sector (set to None)")
    return df
