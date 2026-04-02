"""数据加载与连续合约构建。

职责：
- 加载 Databento dbn 文件为统一格式的 DataFrame
- 对连续合约数据做多种价差调整（Panama / 比例调整）
- 输出包含多种表示的 DataFrame，供不同类型的信号按需使用

输出列说明：
- open/high/low/close: 比例调整后的价格（用于收益率、动量、波动率）
- panama_close: Panama 调整后的价格（用于均线、通道突破等价格形态信号）
- volume: 成交量（未调整）
"""

from pathlib import Path

import databento as db
import numpy as np
import pandas as pd


# ============================================================
# 基础加载
# ============================================================

def _load_raw(file_path: str | Path) -> pd.DataFrame:
    """加载 Databento 连续合约 dbn 文件，返回含 instrument_id 的原始 DataFrame。"""
    store = db.DBNStore.from_file(file_path)
    df = store.to_df()
    df = df[["instrument_id", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["close"])
    df = df[~df.index.duplicated(keep="last")]
    return df


def _find_rolls(df: pd.DataFrame) -> pd.Series:
    """通过 instrument_id 变化识别换月日，返回布尔 Series。"""
    is_roll = df["instrument_id"] != df["instrument_id"].shift(1)
    is_roll.iloc[0] = False
    return is_roll


# ============================================================
# 调整方法
# ============================================================

def _ratio_adjust(df: pd.DataFrame) -> pd.DataFrame:
    """比例调整法（乘法）。

    换月时计算比例系数 = 前一天 close / 当天 open，
    从后往前累乘，使最近价格保持真实值，历史价格按比例缩放。
    保留百分比收益率的正确性。
    """
    is_roll = _find_rolls(df)
    prev_close = df["close"].shift(1)
    roll_ratio = prev_close / df["open"]  # 旧合约close / 新合约open

    # 从后往前累乘
    factors = np.ones(len(df))
    cumulative_factor = 1.0

    for i in range(len(df) - 1, 0, -1):
        if is_roll.iloc[i]:
            cumulative_factor *= roll_ratio.iloc[i]
        factors[i - 1] = cumulative_factor

    if len(df) > 0:
        factors[0] = cumulative_factor

    adjusted = df[["open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close"]:
        adjusted[col] = adjusted[col] * factors

    return adjusted


def _panama_adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Panama 调整法（加法）。

    换月时计算价差 = 当天 open - 前一天 close，
    从后往前累加，使最近价格保持真实值，历史价格做偏移。
    保留绝对价差关系，适合均线等价格形态信号。
    """
    is_roll = _find_rolls(df)
    overnight_gap = df["open"] - df["close"].shift(1)

    adjustments = np.zeros(len(df))
    cumulative_adj = 0.0

    for i in range(len(df) - 1, 0, -1):
        if is_roll.iloc[i]:
            cumulative_adj += overnight_gap.iloc[i]
        adjustments[i - 1] = cumulative_adj

    if len(df) > 0:
        adjustments[0] = cumulative_adj

    adjusted = df[["open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close"]:
        adjusted[col] = adjusted[col] - adjustments

    return adjusted


# ============================================================
# 主入口
# ============================================================

def load_continuous(file_path: str | Path) -> pd.DataFrame:
    """加载 Databento 连续合约日线数据，同时输出两种调整。

    返回 DataFrame 包含:
    - open, high, low, close: 比例调整（默认，用于收益率计算）
    - panama_open, panama_high, panama_low, panama_close: Panama 调整（用于价格形态信号）
    - volume: 成交量
    """
    raw = _load_raw(file_path)

    ratio = _ratio_adjust(raw)
    panama = _panama_adjust(raw)

    result = ratio[["open", "high", "low", "close", "volume"]].copy()
    result["panama_open"] = panama["open"]
    result["panama_high"] = panama["high"]
    result["panama_low"] = panama["low"]
    result["panama_close"] = panama["close"]

    return result


def load_multiple(data_dir: str | Path, pattern: str = "*-continuous-*.dbn.zst") -> dict[str, pd.DataFrame]:
    """批量加载多个品种的连续合约数据。

    返回: {品种名: DataFrame} 字典
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
    products = {}
    for f in files:
        name = f.name.split("-")[0]
        products[name] = load_continuous(f)
    return products
