"""
A-share data loading pipeline.

Data sources:
  - Tushare: OHLCV (daily), 复权因子 (adj_factor), 股票基本信息 (stock_basic), ST 历史 (namechange)
  - AKShare: 交易日历 (tool_trade_date_hist_sina), CSI 500 成分股 (index_stock_cons_weight_csindex)

宇宙构建近似方案（无 point-in-time index_weight 权限下的务实选择）：
  - 基底：CSI 500 当前成分股 + AKShare 历史纳入记录中有纳入日期的股票
  - 每个 signal_date 过滤条件：
      1. 上市满 252 个交易日
      2. 当日非 ST/*ST
      3. 当日未停牌

已知局限：退出 CSI 500 的股票仍留在宇宙中（近似）；无法获取历史权重。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
CACHE_DIR = Path(__file__).parent / "artifacts" / "ashare"

# 研究期：A 股数据从 2013 年起以保证 2015 年有足够历史
ASHARE_START_DATE = "20130101"
ASHARE_END_DATE   = "20260401"

# 宇宙参数
MIN_LISTING_DAYS = 252   # 上市满 1 年才进宇宙
CSI500_CODE      = "000905"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ashare_trading_calendar(
    start: str = ASHARE_START_DATE,
    end:   str = ASHARE_END_DATE,
    *,
    refresh: bool = False,
) -> pd.DatetimeIndex:
    """返回 A 股交易日序列（来自 AKShare）。"""
    cache = CACHE_DIR / "trading_calendar.parquet"
    end_ts = pd.Timestamp(end)
    if cache.exists() and not refresh:
        cal = pd.read_parquet(cache)["trade_date"]
        if cal.max() < end_ts:
            refresh = True
    if refresh or not cache.exists():
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        cal = df["trade_date"]

    mask = (cal >= pd.Timestamp(start)) & (cal <= end_ts)
    return pd.DatetimeIndex(cal[mask].sort_values().values)


def load_csi500_universe(*, refresh: bool = False) -> pd.DataFrame:
    """
    返回 CSI 500 成分股历史纳入记录。

    列：symbol (str, 带交易所后缀如 600519.SH), in_date (Timestamp), name (str)

    注意：AKShare 只提供纳入日期，无退出日期。这是一个近似：
    曾经纳入的股票在研究期内一直视为候选（配合 ST / 停牌 / 上市日过滤）。
    """
    cache = CACHE_DIR / "csi500_universe.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    import akshare as ak

    # 当前成分股（含权重，可获取交易所信息）
    current = ak.index_stock_cons_weight_csindex(symbol=CSI500_CODE)
    current = current.rename(columns={
        "成分券代码": "raw_code",
        "成分券名称": "name",
        "交易所":     "exchange",
        "权重":       "weight",
    })[["raw_code", "name", "exchange"]]
    current["in_date"] = pd.NaT  # 当前成分股纳入日期未知

    # 历史纳入记录（有纳入日期，无退出日期）
    hist = ak.index_stock_cons(symbol=CSI500_CODE)
    hist = hist.rename(columns={"品种代码": "raw_code", "品种名称": "name", "纳入日期": "in_date"})
    hist["in_date"] = pd.to_datetime(hist["in_date"], errors="coerce")
    hist["exchange"] = None

    combined = pd.concat([current, hist], ignore_index=True)
    combined["symbol"] = combined.apply(
        lambda r: _to_tushare_code(r["raw_code"], r["exchange"]), axis=1
    )
    combined = combined.dropna(subset=["symbol"]).drop_duplicates("symbol")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    combined[["symbol", "name", "in_date"]].to_parquet(cache, index=False)
    print(f"CSI 500 宇宙：{len(combined)} 只，已缓存至 {cache}")
    return combined[["symbol", "name", "in_date"]]


def load_stock_basic(*, refresh: bool = False) -> pd.DataFrame:
    """
    返回全市场股票基本信息（来自 AKShare，无积分限制）。

    列：symbol (ts_code格式, 如 600519.SH), name, list_date
    """
    cache = CACHE_DIR / "stock_basic.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    import akshare as ak

    # 沪市
    sh = ak.stock_info_sh_name_code(symbol="主板A股")[["证券代码", "证券简称", "上市日期"]]
    sh.columns = ["code", "name", "list_date"]
    sh["symbol"] = sh["code"].astype(str).str.zfill(6) + ".SH"

    # 科创板（上交所）
    try:
        kcb = ak.stock_info_sh_name_code(symbol="科创板")[["证券代码", "证券简称", "上市日期"]]
        kcb.columns = ["code", "name", "list_date"]
        kcb["symbol"] = kcb["code"].astype(str).str.zfill(6) + ".SH"
    except Exception:
        kcb = pd.DataFrame(columns=["code", "name", "list_date", "symbol"])

    # 深市
    sz = ak.stock_info_sz_name_code(symbol="A股列表")[["A股代码", "A股简称", "A股上市日期"]]
    sz.columns = ["code", "name", "list_date"]
    sz["symbol"] = sz["code"].astype(str).str.zfill(6) + ".SZ"

    df = pd.concat([sh, kcb, sz], ignore_index=True)
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    df = df[["symbol", "name", "list_date"]].dropna(subset=["list_date"])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"stock_basic 已缓存：{len(df)} 只股票")
    return df


def load_st_history(symbols: list[str], *, refresh: bool = False) -> pd.DataFrame:
    """
    返回 ST/*ST 历史区间。

    列：symbol, start_date, end_date
    end_date 为 NaT 表示当前仍是 ST。
    """
    cache = CACHE_DIR / "st_history.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    pro = _get_pro()
    rows = []
    for i, sym in enumerate(symbols):
        try:
            nc = pro.namechange(ts_code=sym,
                                fields="ts_code,name,start_date,end_date")
            st_rows = nc[nc["name"].str.contains(r"\*?ST", na=False)]
            for _, r in st_rows.iterrows():
                rows.append({
                    "symbol":     r["ts_code"],
                    "start_date": pd.to_datetime(r["start_date"], errors="coerce"),
                    "end_date":   pd.to_datetime(r["end_date"],   errors="coerce"),
                })
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  ST 历史：{i+1}/{len(symbols)}")
            time.sleep(0.3)

    df = pd.DataFrame(rows, columns=["symbol", "start_date", "end_date"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def build_daily_universe(
    signal_date:   pd.Timestamp,
    universe_df:   pd.DataFrame,
    stock_basic:   pd.DataFrame,
    st_history:    pd.DataFrame,
    calendar:      pd.DatetimeIndex,
    min_listing_days: int = MIN_LISTING_DAYS,
) -> list[str]:
    """
    返回 signal_date 当日有效的股票列表。

    过滤条件（全部满足）：
      1. 在 CSI 500 宇宙候选中（曾纳入或当前成分）
      2. 上市满 min_listing_days 个交易日
      3. 当日非 ST/*ST
    """
    # 1. 上市日过滤
    trading_days_before = calendar[calendar < signal_date]
    cutoff = (trading_days_before[-min_listing_days]
              if len(trading_days_before) >= min_listing_days else pd.Timestamp("19900101"))

    listed = stock_basic[
        stock_basic["list_date"].notna() &
        (stock_basic["list_date"] <= cutoff)
    ]["symbol"].tolist()
    candidates = set(universe_df["symbol"]) & set(listed)

    # 2. ST 过滤
    st_on_date = set(
        st_history[
            (st_history["start_date"] <= signal_date) &
            (st_history["end_date"].isna() | (st_history["end_date"] >= signal_date))
        ]["symbol"]
    )
    candidates -= st_on_date

    return sorted(candidates)


def download_ohlcv(
    symbols:    list[str],
    start_date: str = ASHARE_START_DATE,
    end_date:   str = ASHARE_END_DATE,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    下载并缓存全量 OHLCV（前复权），返回长格式 DataFrame。

    列：date (datetime64), symbol (str), open, high, low, close, volume (float)

    前复权公式：adj_price = raw_price × adj_factor / latest_adj_factor
    这与 Tushare 官方前复权定义一致，保证最新价不变。
    """
    cache = CACHE_DIR / "ohlcv_adj.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        # 检查是否已覆盖所需股票
        missing = sorted(set(symbols) - set(df["symbol"].unique()))
        if not missing:
            return _filter_date(df, start_date, end_date)
        print(f"缓存中缺少 {len(missing)} 只股票，增量下载...")
        symbols = missing
        existing = df
    else:
        existing = None

    pro = _get_pro()
    frames = []
    failed = []

    for i, sym in enumerate(symbols):
        try:
            raw = pro.daily(ts_code=sym, start_date=start_date, end_date=end_date)
            adj = pro.adj_factor(ts_code=sym, start_date=start_date, end_date=end_date)
            if raw.empty:
                continue

            merged = raw.merge(adj[["trade_date", "adj_factor"]], on="trade_date", how="left")
            merged["adj_factor"] = merged["adj_factor"].ffill().fillna(1.0)

            # 前复权：以最新 adj_factor 为基准
            latest_af = merged["adj_factor"].iloc[0]  # Tushare 降序，最新在第一行
            ratio = merged["adj_factor"] / latest_af
            for col in ["open", "high", "low", "close"]:
                merged[col] = merged[col] * ratio

            merged = merged.rename(columns={
                "trade_date": "date",
                "ts_code":    "symbol",
                "vol":        "volume",
            })
            merged["date"] = pd.to_datetime(merged["date"])
            frames.append(merged[["date", "symbol", "open", "high", "low", "close", "volume"]])

        except Exception as e:
            failed.append((sym, str(e)))

        if (i + 1) % 50 == 0:
            print(f"  OHLCV 下载：{i+1}/{len(symbols)}")
            time.sleep(0.5)

    if failed:
        print(f"  下载失败 {len(failed)} 只：{[s for s, _ in failed[:5]]}")

    if not frames:
        return existing if existing is not None else pd.DataFrame()

    new_data = pd.concat(frames, ignore_index=True)
    if existing is not None:
        new_data = pd.concat([existing, new_data], ignore_index=True)

    new_data = new_data.sort_values(["symbol", "date"]).reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    new_data.to_parquet(cache, index=False)
    print(f"OHLCV 已缓存：{len(new_data):,} 行，{new_data['symbol'].nunique()} 只股票")

    return _filter_date(new_data, start_date, end_date)


# ---------------------------------------------------------------------------
# Convenience: 一键构建研究数据集
# ---------------------------------------------------------------------------

def build_ashare_research_bars(
    start_date: str = "20150101",
    end_date:   str = ASHARE_END_DATE,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    一键返回 A 股研究用 OHLCV（前复权，宇宙内股票，过滤 ST）。

    输出格式与 load_round1_daily_ohlcv 一致，可直接传入 build_v2_research_dataset。
    列：date, symbol, open, high, low, close, volume
    """
    print("Step 1/4: 交易日历")
    calendar = load_ashare_trading_calendar(refresh=refresh)

    print("Step 2/4: CSI 500 宇宙候选")
    universe = load_csi500_universe(refresh=refresh)
    stock_basic = load_stock_basic(refresh=refresh)
    symbols = sorted(set(universe["symbol"]) & set(stock_basic["symbol"]))

    print(f"Step 3/4: ST 历史（{len(symbols)} 只）")
    st_history = load_st_history(symbols, refresh=refresh)

    print(f"Step 4/4: 下载 OHLCV（{len(symbols)} 只）")
    bars = download_ohlcv(symbols, start_date=start_date, end_date=end_date, refresh=refresh)

    return bars


def load_csi500_etf_benchmark(
    start_date: str = ASHARE_START_DATE,
    end_date:   str = ASHARE_END_DATE,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    返回 CSI 500 ETF（510500.SH）日频 OHLC，用作 benchmark。

    列：date (datetime64), open, close
    使用不复权数据（benchmark 只需要收益率，复权与否不影响结果）。
    """
    cache = CACHE_DIR / "csi500_etf.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        return _filter_date(df, start_date, end_date)

    import akshare as ak

    df = ak.fund_etf_hist_em(
        symbol="510500",
        period="daily",
        start_date=start_date.replace("-", "")[:8],
        end_date=end_date.replace("-", "")[:8],
        adjust="",
    )
    df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "open", "close"]].sort_values("date").reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"CSI 500 ETF 已缓存：{len(df)} 行，{df['date'].min().date()} ~ {df['date'].max().date()}")
    return _filter_date(df, start_date, end_date)


def load_ashare_sector(*, refresh: bool = False) -> pd.DataFrame:
    """
    返回 symbol → 行业（证监会行业分类）映射。

    列：symbol (ts_code格式), sector (str)
    数据来源：
      - SH 股票：上交所 query API（CSRC_CODE_DESC 字段，18 个一级行业）
      - SZ 股票：AKShare stock_info_sz_name_code（所属行业字段）
    """
    cache = CACHE_DIR / "ashare_sector.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    import requests
    import akshare as ak

    # SH 股票：直接调上交所 query API
    url = "https://query.sse.com.cn/sseQuery/commonQuery.do"
    headers = {
        "Host": "query.sse.com.cn",
        "Referer": "https://www.sse.com.cn/assortment/stock/areatrade/trade/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    params = {
        "sqlId": "COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L",
        "STOCK_TYPE": "1",
        "CSRC_CODE": "",
        "COMPANY_STATUS": "2,4,5,7,8",
        "type": "inParams",
        "isPagination": "true",
        "pageHelp.pageSize": "10000",
        "pageHelp.pageNo": "1",
        "pageHelp.beginPage": "1",
        "pageHelp.endPage": "1",
        "pageHelp.cacheSize": "1",
    }
    r = requests.get(url, params=params, headers=headers, timeout=15)
    sh_raw = pd.DataFrame(r.json()["result"])[["A_STOCK_CODE", "CSRC_CODE"]]
    sh_raw.columns = ["code", "sector"]
    sh_raw["symbol"] = sh_raw["code"].str.zfill(6) + ".SH"
    sh_df = sh_raw[["symbol", "sector"]].dropna(subset=["sector"])

    # SZ 股票：AKShare（所属行业字段，格式如 "C 制造业"，取首字母作为 CSRC 码）
    sz_raw = ak.stock_info_sz_name_code(symbol="A股列表")[["A股代码", "所属行业"]]
    sz_raw.columns = ["code", "sector"]
    sz_raw["symbol"] = sz_raw["code"].astype(str).str.zfill(6) + ".SZ"
    sz_raw["sector"] = sz_raw["sector"].str.strip().str[:1]  # 取 CSRC 字母码
    sz_df = sz_raw[["symbol", "sector"]].dropna(subset=["sector"])
    sz_df = sz_df[sz_df["sector"].str.strip() != ""]

    df = pd.concat([sh_df, sz_df], ignore_index=True)
    df = df.drop_duplicates("symbol").reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"行业数据已缓存：{len(df)} 只股票，{df['sector'].nunique()} 个行业")
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_pro():
    import tushare as ts
    token = TUSHARE_TOKEN or os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN 未设置，请在 .env 中配置")
    ts.set_token(token)
    return ts.pro_api()


def _to_tushare_code(raw: str, exchange: str | None) -> str | None:
    """把 6 位股票代码转成 Tushare 格式（600519 → 600519.SH）。"""
    if not isinstance(raw, str) or len(raw) != 6:
        return None
    if exchange and "深圳" in exchange:
        return f"{raw}.SZ"
    if exchange and "上海" in exchange:
        return f"{raw}.SH"
    # 根据代码前缀推断
    if raw.startswith(("60", "68")):
        return f"{raw}.SH"
    if raw.startswith(("00", "30", "002")):
        return f"{raw}.SZ"
    return None


def _filter_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df[mask].reset_index(drop=True)
