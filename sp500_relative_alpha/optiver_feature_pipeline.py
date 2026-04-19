"""
SP500 横截面选股 Feature Pipeline
思路来源：Optiver Trading at the Close 1st Place Solution + Alpha101

特征分五层：
  Layer 1 - Base Features       : OHLCV 直接派生
  Layer 2 - Rolling Features    : 多周期时序统计
  Layer 3 - Group Features      : 时间分段特征（对应 seconds_in_bucket_group）
  Layer 4 - Cross-sectional     : 横截面 rank / zscore
  Layer 5 - Alpha101            : WorldQuant 101因子（~25个，仅需OHLCV）
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 0. 配置
# ============================================================

# 时间分段：对应比赛里 seconds_in_bucket_group，按经济含义划分
# 对应动量研究的标准周期：周/月/季/年
TIME_GROUPS = {
    "week":    (1,   5),   # 过去 1~5   天（短期，捕捉反转）
    "month":   (5,   21),  # 过去 5~21  天（月频动量）
    "quarter": (21,  63),  # 过去 21~63 天（季度动量，主力区间）
    "year":    (63, 252),  # 过去 63~252天（年度动量，跳过近月避免反转）
}

# 做时间分段特征和横截面特征的基础列
BASE_COLS = [
    "daily_return", "amplitude", "close_position",
    "upper_shadow", "lower_shadow", "gap_return",
    "intraday_return", "volume_ratio",
]

# 做横截面特征的列（在 BASE_COLS 基础上加入动量/波动率）
CS_COLS = BASE_COLS + ["mom_5", "mom_20", "mom_60", "vol_5", "vol_20", "pv_corr_10"]


# ============================================================
# 1. 数据加载
# ============================================================

def load_sp500_data(
    data_dir: str = "equity",
    start_date: str = "2015-01-01",
    end_date:   str = "2025-01-01",
    min_coverage: float = 0.8,
):
    """
    从本地 equity/ 目录读取各股票的 parquet 文件，合并成长表。
    排除 SPY（指数本身不参与选股）。
    """
    from pathlib import Path

    files = sorted(Path(data_dir).glob("*.parquet"))
    print(f"发现 {len(files)} 个 parquet 文件，加载中...")

    frames = []
    for f in files:
        ticker = f.stem           # 文件名去掉 .parquet 即 ticker
        if ticker == "SPY":       # 指数不参与选股
            continue
        tmp = pd.read_parquet(f)
        tmp.index = pd.to_datetime(tmp.index)
        tmp.index.name = "date"
        tmp = tmp.loc[start_date:end_date].copy()
        if tmp.empty:
            continue
        tmp["stock_id"] = ticker
        tmp = tmp.reset_index()   # date → 列
        frames.append(tmp)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["close"]).sort_values(["stock_id", "date"]).reset_index(drop=True)

    # 过滤覆盖率不足的股票
    total_days = df["date"].nunique()
    valid_ids  = df.groupby("stock_id")["date"].count()
    valid_ids  = valid_ids[valid_ids >= total_days * min_coverage].index
    df = df[df["stock_id"].isin(valid_ids)].reset_index(drop=True)

    print(f"有效股票: {df['stock_id'].nunique()} 支  |  日期: {df['date'].min().date()} ~ {df['date'].max().date()}  |  总行数: {len(df):,}")
    return df


# ============================================================
# 2-5. 特征构造（Layer 1~4）
# 全程在 wide format（日期 × 股票）下计算，最后一次性转长表。
# 比逐列 groupby.transform 快 20~50 倍。
# ============================================================

def build_features(df: pd.DataFrame, windows: tuple = (5, 10, 20, 60, 120)) -> pd.DataFrame:
    import time

    # ── Pivot to wide ──────────────────────────────────────────
    t0 = time.time()
    close  = df.pivot(index="date", columns="stock_id", values="close")
    open_  = df.pivot(index="date", columns="stock_id", values="open")
    high   = df.pivot(index="date", columns="stock_id", values="high")
    low    = df.pivot(index="date", columns="stock_id", values="low")
    volume = df.pivot(index="date", columns="stock_id", values="volume")
    n_dates, n_stocks = close.shape
    print(f"  Pivot done ({time.time()-t0:.1f}s)  [{n_dates} dates × {n_stocks} stocks]")

    # ── Layer 1: Base series ───────────────────────────────────
    t0 = time.time()
    prev_close = close.shift(1)
    hl         = (high - low).clip(lower=1e-8)
    log_ret    = np.log(close / prev_close)
    dollar_vol = close * volume
    vol_ma20   = volume.rolling(20, min_periods=10).mean()
    adv20      = dollar_vol.rolling(20, min_periods=10).mean()
    vwap       = (high + low + close) / 3

    # upper/lower shadow 用 numpy 做 element-wise max/min（比 pd.DataFrame.max(axis=1) 快）
    oc_max = pd.DataFrame(np.maximum(open_.values, close.values), index=close.index, columns=close.columns)
    oc_min = pd.DataFrame(np.minimum(open_.values, close.values), index=close.index, columns=close.columns)

    base = {
        "daily_return":    close / prev_close - 1,
        "log_return":      log_ret,
        "gap_return":      open_ / prev_close - 1,
        "intraday_return": close / open_ - 1,
        "amplitude":       hl / prev_close,
        "close_position":  (close - low) / hl,
        "open_position":   (open_ - low) / hl,
        "upper_shadow":    (high - oc_max) / hl,
        "lower_shadow":    (oc_min - low)  / hl,
        "volume_ratio":    volume / vol_ma20.clip(lower=1),
    }
    ROLL_COLS = [
        "daily_return", "amplitude", "close_position",
        "upper_shadow", "lower_shadow", "gap_return",
        "intraday_return", "volume_ratio",
    ]
    CS_BASE = ROLL_COLS  # 横截面特征用的基础列
    print(f"  Layer 1 done ({time.time()-t0:.1f}s)")

    # base series 本身也作为特征输出
    features: dict[str, pd.DataFrame] = {k: v for k, v in base.items()}

    # ── Layer 2: Rolling features ──────────────────────────────
    t0 = time.time()
    for col in ROLL_COLS:
        w_df = base[col]
        for w in windows:
            roll = w_df.rolling(w, min_periods=w // 2)
            features[f"{col}_mean_{w}"] = roll.mean()
            features[f"{col}_std_{w}"]  = roll.std()

    for lag in [5, 10, 20, 60, 120, 252]:
        features[f"mom_{lag}"] = close / close.shift(lag) - 1

    for w in [5, 20, 60]:
        features[f"vol_{w}"] = log_ret.rolling(w, min_periods=w // 2).std()

    features["vol_ratio_5_20"]  = features["vol_5"]  / features["vol_20"].clip(lower=1e-8)
    features["vol_ratio_20_60"] = features["vol_20"] / features["vol_60"].clip(lower=1e-8)

    for w in [5, 20, 60]:
        ma = close.rolling(w, min_periods=w // 2).mean()
        features[f"price_ma{w}_ratio"] = close / ma.clip(lower=1e-8)

    features["ma_cross_5_20"]  = features["price_ma5_ratio"]  / features["price_ma20_ratio"]
    features["ma_cross_20_60"] = features["price_ma20_ratio"] / features["price_ma60_ratio"]

    for w in [10, 20]:
        features[f"pv_corr_{w}"] = log_ret.rolling(w, min_periods=w // 2).corr(base["volume_ratio"])

    # Historical target features（对应第一名明确提到的特征类别）
    # 用过去已实现的市场中性超额收益作为特征，等价于在线学习中的 revealed_targets
    daily_ret  = base["daily_return"]
    index_ret  = daily_ret.mean(axis=1)                    # 等权指数日收益
    excess_ret = daily_ret.sub(index_ret, axis=0)          # 每支股票的日超额收益
    for lag in [1, 2, 3, 5, 10, 20]:
        features[f"excess_ret_lag_{lag}"] = excess_ret.shift(lag)
    for w in [5, 21, 63]:
        features[f"excess_ret_mean_{w}"] = excess_ret.rolling(w, min_periods=w // 2).mean()
        features[f"excess_ret_std_{w}"]  = excess_ret.rolling(w, min_periods=w // 2).std()

    print(f"  Layer 2 done ({time.time()-t0:.1f}s)")

    # ── Layer 3: 时间分段特征 ──────────────────────────────────
    t0 = time.time()
    for col in ROLL_COLS:
        w_df = base[col]
        for seg, (start, end) in TIME_GROUPS.items():
            w = end - start
            group_mean  = w_df.shift(start).rolling(w, min_periods=1).mean()
            group_first = w_df.shift(end)
            features[f"{col}_{seg}_mean_ratio"]  = w_df / group_mean.clip(lower=1e-8)
            features[f"{col}_{seg}_first_ratio"] = w_df / group_first.replace(0, np.nan)
    print(f"  Layer 3 done ({time.time()-t0:.1f}s)")

    # ── Layer 4: 横截面特征 ────────────────────────────────────
    # wide format 下 axis=1 操作即横截面，无需 groupby
    t0 = time.time()
    cs_targets = CS_BASE + ["mom_5", "mom_20", "mom_60", "vol_5", "vol_20", "pv_corr_10"]
    for col in cs_targets:
        w_df = base.get(col) if col in base else features.get(col)
        if w_df is None:
            continue
        cs_mean = w_df.mean(axis=1)                       # (n_dates,) Series
        cs_std  = w_df.std(axis=1).clip(lower=1e-8)
        features[f"{col}_cs_rank"]       = w_df.rank(axis=1, pct=True, na_option="keep")
        features[f"{col}_cs_zscore"]     = w_df.sub(cs_mean, axis=0).div(cs_std, axis=0)
        features[f"{col}_cs_mean_ratio"] = w_df.div(cs_mean.clip(lower=1e-8), axis=0)

    # 对比值特征（group ratio）也做横截面 rank
    # 对应第一名的组合拳：先算组内比值，再算跨股票排名
    for col in ROLL_COLS:
        for seg in TIME_GROUPS:
            for suffix in ("mean_ratio", "first_ratio"):
                fname = f"{col}_{seg}_{suffix}"
                if fname in features:
                    features[f"{fname}_cs_rank"] = (
                        features[fname].rank(axis=1, pct=True, na_option="keep")
                    )
    print(f"  Layer 4 done ({time.time()-t0:.1f}s)")

    # ── Assemble: wide → long (一次性) ────────────────────────
    t0 = time.time()
    print(f"  Assembling {len(features)} features → long format...")
    dates_arr  = np.repeat(close.index.values,  n_stocks)
    stocks_arr = np.tile(close.columns.values, n_dates)

    result = {"date": dates_arr, "stock_id": stocks_arr}
    for name, wide in features.items():
        result[name] = wide.values.ravel()

    feat_df = pd.DataFrame(result)
    feat_df["date"] = pd.to_datetime(feat_df["date"])

    # 把 vwap / adv20 也带进去（Alpha101 需要）
    vwap_arr  = vwap.values.ravel()
    adv20_arr = adv20.values.ravel()

    base_df = df[["date", "stock_id", "open", "high", "low", "close", "volume"]].copy()
    feat_df["vwap"]  = vwap_arr
    feat_df["adv20"] = adv20_arr

    out = base_df.merge(feat_df, on=["date", "stock_id"], how="left")
    print(f"  Assemble done ({time.time()-t0:.1f}s)")
    return out


# ============================================================
# 6. Alpha101 因子（~25个，仅需 OHLCV）
# ============================================================

def build_alpha101(df: pd.DataFrame) -> pd.DataFrame:
    """Wide format 计算，_ts_rank 用 rolling.rank(pct=True) 替代慢的 apply。"""
    import time
    t0 = time.time()

    close  = df.pivot(index="date", columns="stock_id", values="close")
    open_  = df.pivot(index="date", columns="stock_id", values="open")
    high   = df.pivot(index="date", columns="stock_id", values="high")
    low    = df.pivot(index="date", columns="stock_id", values="low")
    volume = df.pivot(index="date", columns="stock_id", values="volume")
    vwap   = df.pivot(index="date", columns="stock_id", values="vwap")
    adv20  = df.pivot(index="date", columns="stock_id", values="adv20")
    ret    = close.pct_change()

    def _rank(x):   return x.rank(axis=1, pct=True, na_option="keep")
    def _delay(x,d): return x.shift(d)
    def _delta(x,d): return x.diff(d)
    def _ts_sum(x,d): return x.rolling(d, min_periods=d//2).sum()
    def _ts_min(x,d): return x.rolling(d, min_periods=d//2).min()
    def _ts_max(x,d): return x.rolling(d, min_periods=d//2).max()
    def _stddev(x,d): return x.rolling(d, min_periods=d//2).std()
    def _corr(x,y,d): return x.rolling(d, min_periods=d//2).corr(y)
    def _cov(x,y,d):  return x.rolling(d, min_periods=d//2).cov(y)
    def _scale(x):    return x.div(x.abs().sum(axis=1).clip(lower=1e-8), axis=0)

    # rolling.rank(pct=True) 是 Cython 实现，比 rolling.apply 快很多
    def _ts_rank(x, d):
        return x.rolling(d, min_periods=d//2).rank(pct=True)

    alphas = {}

    alphas["alpha002"] = -_corr(
        _rank(np.log(volume + 1e-8).diff(2)),
        _rank((close - open_) / open_.clip(lower=1e-8)), 6)

    alphas["alpha003"] = -_corr(_rank(open_), _rank(volume), 10)
    alphas["alpha006"] = -_corr(open_, volume, 10)

    alphas["alpha011"] = (
        _rank(_ts_max(vwap - close, 3)) + _rank(_ts_min(vwap - close, 3))
    ) * _rank(_delta(volume, 3))

    alphas["alpha012"] = np.sign(_delta(volume, 1)) * (-_delta(close, 1))
    alphas["alpha013"] = -_rank(_cov(_rank(close), _rank(volume), 5))
    alphas["alpha014"] = -_rank(_delta(ret, 3)) * _corr(open_, volume, 10)
    alphas["alpha015"] = -_ts_sum(_rank(_corr(_rank(high), _rank(volume), 3)), 3)
    alphas["alpha016"] = -_rank(_cov(_rank(high), _rank(volume), 5))

    alphas["alpha018"] = -_rank(
        _stddev((close - open_).abs(), 5) + (close - open_) + _corr(close, open_, 10))

    alphas["alpha020"] = (
        -_rank(open_ - _delay(high, 1))
        * _rank(open_ - _delay(close, 1))
        * _rank(open_ - _delay(low, 1)))

    alphas["alpha025"] = _rank(-ret * adv20 * vwap * (high - close))

    alphas["alpha026"] = -_ts_max(
        _corr(_ts_rank(volume, 5), _ts_rank(high, 5), 5), 3)

    alphas["alpha028"] = _scale(_corr(adv20, low, 5) + (high + low) / 2 - close)
    alphas["alpha033"] = _rank(-1 + open_ / close.clip(lower=1e-8))

    alphas["alpha034"] = (
        _rank(1 - _rank(_stddev(ret, 2) / _stddev(ret, 5).clip(lower=1e-8)))
        + _rank(_delta(close, 1)))

    alphas["alpha035"] = (
        _ts_rank(volume, 32)
        * (1 - _ts_rank(close + high - low, 16))
        * (1 - _ts_rank(ret, 32)))

    alphas["alpha040"] = -_rank(_stddev(high, 10)) * _corr(high, volume, 10)
    alphas["alpha041"] = np.sqrt((high * low).clip(lower=0)) - vwap
    alphas["alpha043"] = (volume / adv20.clip(lower=1)) * (-_delta(close, 7))
    alphas["alpha044"] = -_corr(high, _rank(volume), 5)

    alphas["alpha052"] = (
        (_ts_min(low, 5) - _delay(_ts_min(low, 5), 5))
        * _rank((_ts_sum(ret, 240) - _ts_sum(ret, 20)) / 220)
        * _ts_rank(volume, 5))

    body = ((close - low - high + close) / (close - low + 1e-8)).clip(-10, 10)
    alphas["alpha053"] = -_delta(body, 9)

    hl_w  = (high - low).clip(lower=1e-8)
    body2 = ((close - low) - (high - close)) / hl_w * volume
    alphas["alpha060"] = -(2 * _scale(_rank(body2)) - _scale(_rank(_ts_rank(close, 10))))

    alphas["alpha101"] = (close - open_) / (high - low + 1e-8)

    # wide → long (一次性 ravel，避免反复 stack+merge)
    n_dates, n_stocks = close.shape
    dates_arr  = np.repeat(close.index.values, n_stocks)
    stocks_arr = np.tile(close.columns.values, n_dates)

    alpha_data = {"date": dates_arr, "stock_id": stocks_arr}
    for name, wide in alphas.items():
        alpha_data[name] = wide.values.ravel()

    alpha_df = pd.DataFrame(alpha_data)
    alpha_df["date"] = pd.to_datetime(alpha_df["date"])

    df = df.merge(alpha_df, on=["date", "stock_id"], how="left")
    print(f"  Alpha101 done ({time.time()-t0:.1f}s)")
    return df


# ============================================================
# 7. Target 构造
# ============================================================

def build_target(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """target = 股票未来N日收益 - 等权指数未来N日收益（市场中性超额收益）"""
    close_w = df.pivot(index="date", columns="stock_id", values="close")

    stock_fwd = close_w.shift(-forward_days) / close_w - 1      # (n_dates, n_stocks)
    index_fwd = stock_fwd.mean(axis=1)                           # 等权指数
    excess    = stock_fwd.sub(index_fwd, axis=0)                 # 超额收益

    n_dates, n_stocks = close_w.shape
    target_df = pd.DataFrame({
        "date":     np.repeat(close_w.index.values, n_stocks),
        "stock_id": np.tile(close_w.columns.values, n_dates),
        "target":   excess.values.ravel(),
    })
    target_df["date"] = pd.to_datetime(target_df["date"])

    df = df.merge(target_df, on=["date", "stock_id"], how="left")
    return df


# ============================================================
# 8. 特征筛选（CatBoost Feature Importance）
# ============================================================

def select_features(df: pd.DataFrame, top_n: int = 150) -> tuple[list, pd.Series]:
    from catboost import CatBoostRegressor

    exclude = {
        "date", "stock_id", "target",
        "open", "high", "low", "close", "volume",
        "prev_close", "vwap", "adv20", "dollar_volume",
    }
    feature_cols = [c for c in df.columns if c not in exclude]

    df_clean = df.dropna(subset=["target"])

    # 前 70% 做训练，只跑一轮快速筛选
    split_date = df_clean["date"].quantile(0.7)
    train = df_clean[df_clean["date"] <= split_date]

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        verbose=100,
        random_seed=42,
        allow_writing_files=False,
    )
    model.fit(
        train[feature_cols].fillna(0),
        train["target"],
    )

    importance = pd.Series(model.get_feature_importance(), index=feature_cols).sort_values(ascending=False)
    top_features = importance.head(top_n).index.tolist()

    print(f"\nTop 15 特征:\n{importance.head(15).to_string()}")
    return top_features, importance


# ============================================================
# 9. Walk-forward 验证
# ============================================================

def walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: list,
    n_folds: int = 3,
    forward_days: int = 5,
) -> list[dict]:
    """
    时序 walk-forward：每次用更多历史数据训练，向前预测一段时间。
    报告每个 fold 的 MAE 和 IC（rank correlation）。
    """
    from catboost import CatBoostRegressor

    df_clean = df.dropna(subset=["target"]).copy()
    dates = sorted(df_clean["date"].unique())
    n = len(dates)

    # 前 60% 作为初始训练集，剩余 40% 分成 n_folds 个验证窗口
    base_train_size = int(n * 0.6)
    val_size = (n - base_train_size) // n_folds

    results = []
    for i in range(n_folds):
        train_end = dates[base_train_size + i * val_size - 1]
        val_start = dates[base_train_size + i * val_size]
        val_end   = dates[min(base_train_size + (i + 1) * val_size - 1, n - 1)]

        train = df_clean[df_clean["date"] <= train_end]
        val   = df_clean[(df_clean["date"] >= val_start) & (df_clean["date"] <= val_end)]

        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="MAE",
            verbose=0,
            random_seed=42,
            allow_writing_files=False,
        )
        model.fit(
            train[feature_cols].fillna(0), train["target"],
            eval_set=(val[feature_cols].fillna(0), val["target"]),
            early_stopping_rounds=50,
        )

        pred = model.predict(val[feature_cols].fillna(0))
        mae  = np.abs(pred - val["target"].values).mean()

        # IC：每天横截面预测值与实际值的 rank 相关，再取均值
        val = val.copy()
        val["pred"] = pred
        daily_ic = val.groupby("date").apply(
            lambda x: x["pred"].corr(x["target"], method="spearman")
        )
        ic_mean = daily_ic.mean()
        ic_std  = daily_ic.std()

        print(
            f"Fold {i+1} | {val_start.strftime('%Y-%m-%d')} ~ {val_end.strftime('%Y-%m-%d')} | "
            f"MAE={mae:.4f}  IC={ic_mean:.4f}±{ic_std:.4f}"
        )
        results.append({"mae": mae, "ic_mean": ic_mean, "ic_std": ic_std, "model": model})

    avg_mae = np.mean([r["mae"]     for r in results])
    avg_ic  = np.mean([r["ic_mean"] for r in results])
    print(f"\n平均 MAE={avg_mae:.4f}  平均 IC={avg_ic:.4f}")
    return results


# ============================================================
# 10. 完整 Pipeline
# ============================================================

def run_pipeline(
    start_date: str = "2015-01-01",
    end_date:   str = "2025-01-01",
    forward_days: int = 5,
    top_features: int = 150,
):
    steps = [
        ("加载数据",       lambda d: load_sp500_data("equity", start_date, end_date)),
        ("特征构造 L1-4",  build_features),
        ("Alpha101 L5",   build_alpha101),
        ("Target",        lambda d: build_target(d, forward_days)),
    ]

    df = None
    for name, fn in steps:
        print(f"\n{'='*55}\nStep: {name}\n{'='*55}")
        df = fn(df) if df is not None else fn(None)
        feat_cols = [c for c in df.columns if c not in
                     {"date","stock_id","open","high","low","close","volume",
                      "prev_close","vwap","adv20","dollar_volume","target"}]
        print(f"  当前特征数: {len(feat_cols)}  |  总行数: {len(df):,}")

    print(f"\n{'='*55}\nStep: 特征筛选 (top {top_features})\n{'='*55}")
    top_feats, importance = select_features(df, top_n=top_features)

    print(f"\n{'='*55}\nStep: Walk-forward 验证\n{'='*55}")
    results = walk_forward_validation(df, top_feats, forward_days=forward_days)

    return df, top_feats, importance, results


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    df, features, importance, results = run_pipeline(
        start_date="2015-01-01",
        end_date="2025-01-01",
        forward_days=20,
        top_features=150,
    )
