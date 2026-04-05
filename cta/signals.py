"""信号生成模块。

每个信号函数的接口统一：
    输入: 价格序列 (pd.Series) + 参数
    输出: 信号序列 (pd.Series, 值域 [-1, +1])

值的含义：
    +1 = 强烈做多, -1 = 强烈做空, 0 = 无方向
    中间值（如 0.3）表示弱信号/低置信度
"""

import numpy as np
import pandas as pd


# ============================================================
# 动量类信号
# ============================================================

def momentum(close: pd.Series, lookback: int = 20) -> pd.Series:
    """时间序列动量：过去 N 天收益率的符号。

    - 过去 lookback 天收益 > 0 → +1 (做多)
    - 过去 lookback 天收益 < 0 → -1 (做空)
    - 否则 → 0
    """
    returns = close.pct_change(lookback)
    signal = pd.Series(0.0, index=close.index)
    signal[returns > 0] = 1.0
    signal[returns < 0] = -1.0
    return signal


def combined_momentum(close: pd.Series, lookbacks: list[int] = [12, 30, 60]) -> pd.Series:
    """多周期动量组合：对多个 lookback 的动量信号等权平均。

    输出范围 [-1, +1]，值的绝对值反映趋势一致性：
    - +1.0: 所有周期都看多（强趋势）
    - +0.33: 两个看多一个看空（弱趋势）
    - 0.0: 多空信号完全抵消
    """
    signals = pd.concat([momentum(close, lb) for lb in lookbacks], axis=1)
    return signals.mean(axis=1)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.Series:
    """MACD 信号：快慢 EMA 之差与信号线的关系。

    MACD 线 = EMA(fast) - EMA(slow)
    信号线 = EMA(MACD, signal_period)
    柱状图 = MACD - 信号线

    输出：柱状图归一化到 [-1, +1]（用滚动标准差归一化）。
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    # 用滚动标准差归一化，然后 clip 到 [-1, +1]
    rolling_std = histogram.rolling(63, min_periods=20).std()
    normalized = histogram / rolling_std.replace(0, np.nan)
    normalized = normalized.clip(-2, 2) / 2  # [-2σ, +2σ] → [-1, +1]
    return normalized.fillna(0.0)


# ============================================================
# 均线类信号
# ============================================================

def ma_crossover(close: pd.Series, fast: int = 10, slow: int = 50) -> pd.Series:
    """均线交叉（SMA）：快线在慢线上方做多，反之做空。

    - fast SMA > slow SMA → +1 (做多)
    - fast SMA < slow SMA → -1 (做空)
    """
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    signal = pd.Series(0.0, index=close.index)
    signal[ma_fast > ma_slow] = 1.0
    signal[ma_fast < ma_slow] = -1.0
    return signal


def ema_crossover(close: pd.Series, fast: int = 10, slow: int = 50) -> pd.Series:
    """均线交叉（EMA）：与 SMA 版本类似但对近期价格更敏感。

    输出连续值：用快慢 EMA 之差除以慢 EMA 归一化，
    值越大表示趋势越强。
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    diff = ema_fast - ema_slow

    # 用滚动标准差归一化
    rolling_std = diff.rolling(63, min_periods=20).std()
    normalized = diff / rolling_std.replace(0, np.nan)
    normalized = normalized.clip(-2, 2) / 2
    return normalized.fillna(0.0)


# ============================================================
# 通道突破类信号
# ============================================================

def donchian_breakout(close: pd.Series, period: int = 20) -> pd.Series:
    """Donchian 通道突破：价格突破 N 日最高/最低价。

    - close >= N 日最高 → +1（向上突破）
    - close <= N 日最低 → -1（向下突破）
    - 中间位置 → 按在通道中的相对位置线性映射到 [-1, +1]
    """
    upper = close.rolling(period).max()
    lower = close.rolling(period).min()
    width = upper - lower

    # 相对位置：0 = 通道底部, 1 = 通道顶部
    position = (close - lower) / width.replace(0, np.nan)
    # 映射到 [-1, +1]
    signal = (position * 2 - 1).fillna(0.0)
    return signal


def bollinger_breakout(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """布林带突破：价格相对布林带的位置。

    布林带 = SMA ± num_std × 标准差
    输出：价格在带中的相对位置，归一化到 [-1, +1]。
    突破上轨 > +1 会被 clip。
    """
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = upper - lower

    position = (close - lower) / width.replace(0, np.nan)
    signal = (position * 2 - 1).clip(-1, 1).fillna(0.0)
    return signal


def keltner_breakout(
    high: pd.Series, low: pd.Series, close: pd.Series,
    ema_period: int = 20, atr_period: int = 20, atr_mult: float = 2.0,
) -> pd.Series:
    """Keltner 通道突破：基于 ATR 的通道。

    中轨 = EMA(close)
    上轨 = 中轨 + atr_mult × ATR
    下轨 = 中轨 - atr_mult × ATR

    输出：价格在通道中的相对位置，归一化到 [-1, +1]。
    """
    middle = close.ewm(span=ema_period, adjust=False).mean()

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    upper = middle + atr_mult * atr
    lower = middle - atr_mult * atr
    width = upper - lower

    position = (close - lower) / width.replace(0, np.nan)
    signal = (position * 2 - 1).clip(-1, 1).fillna(0.0)
    return signal


# ============================================================
# 回归类信号
# ============================================================

def linear_regression_slope(close: pd.Series, period: int = 30) -> pd.Series:
    """线性回归斜率：滚动窗口内价格对时间的回归斜率。

    斜率为正 → 上升趋势, 斜率为负 → 下降趋势。
    用 R² 加权：R² 高表示趋势线性度好（置信度高）。
    输出归一化到 [-1, +1]。
    """
    def _calc_slope_r2(window):
        n = len(window)
        if n < period:
            return 0.0
        x = np.arange(n, dtype=float)
        y = window.values
        x_mean = x.mean()
        y_mean = y.mean()
        ss_xy = ((x - x_mean) * (y - y_mean)).sum()
        ss_xx = ((x - x_mean) ** 2).sum()
        ss_yy = ((y - y_mean) ** 2).sum()
        if ss_xx == 0 or ss_yy == 0:
            return 0.0
        slope = ss_xy / ss_xx
        r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)
        # 斜率归一化：除以 y 的标准差使跨品种可比
        y_std = np.sqrt(ss_yy / (n - 1))
        normalized_slope = slope / y_std if y_std > 0 else 0.0
        # 乘以 R² 作为置信度
        return normalized_slope * r_squared

    raw = close.rolling(period).apply(_calc_slope_r2, raw=False)
    # 最终归一化到 [-1, +1]
    rolling_std = raw.rolling(252, min_periods=63).std()
    normalized = raw / rolling_std.replace(0, np.nan)
    normalized = normalized.clip(-2, 2) / 2
    return normalized.fillna(0.0)


# ============================================================
# 趋势强度信号
# ============================================================

def adx_dmi(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ADX/DMI 信号：方向运动指标。

    +DI > -DI → 做多方向
    +DI < -DI → 做空方向
    ADX 值作为置信度权重（ADX 高=趋势强）。

    输出：方向 × ADX 归一化到 [-1, +1]。
    """
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=close.index)
    minus_dm = pd.Series(0.0, index=close.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    # Smoothed (Wilder's smoothing = EMA with alpha=1/period)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)

    # DX and ADX
    di_diff = (plus_di - minus_di).abs()
    di_sum = plus_di + minus_di
    dx = 100 * di_diff / di_sum.replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    # 方向：+DI > -DI → 正, 否则 → 负
    direction = np.sign(plus_di - minus_di)

    # ADX 归一化到 [0, 1]：通常 ADX 在 0-60 之间，25+ 算有趋势
    adx_normalized = (adx / 50).clip(0, 1)

    signal = (direction * adx_normalized).fillna(0.0)
    return signal


# ============================================================
# Carry 因子（非价格趋势类，用期限结构数据）
# ============================================================

def carry(front_close: pd.Series, back_close: pd.Series, smooth: int = 5) -> pd.Series:
    """Carry 因子：近月-远月价差。

    经典定义：
        carry = (F_near - F_far) / F_near
        正值 = backwardation（现货溢价）→ 做多有利
        负值 = contango（期货溢价）→ 做空有利

    本项目实现：
        计算原始 carry 值后，用滚动标准差归一化到 [-1, +1]。
        smooth 参数对原始 carry 做短期平滑，减少日间噪音。

    参数:
        front_close: 主力合约（c.0）收盘价（未调整）
        back_close: 次主力合约（c.1）收盘价（未调整）
        smooth: 平滑窗口（天）
    """
    raw_carry = (front_close - back_close) / front_close.replace(0, np.nan)

    # 短期平滑
    if smooth > 1:
        raw_carry = raw_carry.rolling(smooth, min_periods=1).mean()

    # 归一化到 [-1, +1]
    rolling_std = raw_carry.rolling(252, min_periods=63).std()
    normalized = raw_carry / rolling_std.replace(0, np.nan)
    normalized = normalized.clip(-2, 2) / 2

    return normalized.fillna(0.0)


# ============================================================
# 信号注册表
# ============================================================

# 每个信号的注册信息：函数、默认参数、使用哪种价格
# price_type: "close" = 比例调整, "panama" = Panama 调整, "ohlc" = 需要 OHLC
SIGNAL_REGISTRY = {
    "momentum": {
        "fn": momentum,
        "params": {"lookback": 20},
        "price_type": "close",
    },
    "combined_momentum": {
        "fn": combined_momentum,
        "params": {"lookbacks": [12, 30, 60]},
        "price_type": "close",
    },
    "macd": {
        "fn": macd,
        "params": {"fast": 12, "slow": 26, "signal_period": 9},
        "price_type": "close",
    },
    "ma_crossover": {
        "fn": ma_crossover,
        "params": {"fast": 10, "slow": 50},
        "price_type": "panama",
    },
    "ema_crossover": {
        "fn": ema_crossover,
        "params": {"fast": 10, "slow": 50},
        "price_type": "panama",
    },
    "donchian": {
        "fn": donchian_breakout,
        "params": {"period": 20},
        "price_type": "panama",
    },
    "bollinger": {
        "fn": bollinger_breakout,
        "params": {"period": 20, "num_std": 2.0},
        "price_type": "panama",
    },
    "keltner": {
        "fn": keltner_breakout,
        "params": {"ema_period": 20, "atr_period": 20, "atr_mult": 2.0},
        "price_type": "ohlc",
    },
    "linreg": {
        "fn": linear_regression_slope,
        "params": {"period": 30},
        "price_type": "close",
    },
    "adx": {
        "fn": adx_dmi,
        "params": {"period": 14},
        "price_type": "ohlc",
    },
}


def generate_signal(daily: pd.DataFrame, signal_name: str, **override_params) -> pd.Series:
    """通用信号生成入口。

    参数:
        daily: 日线 DataFrame（包含 close, panama_close, high, low 等列）
        signal_name: 信号名称（SIGNAL_REGISTRY 中的 key）
        **override_params: 覆盖默认参数

    返回: 信号序列 [-1, +1]
    """
    if signal_name not in SIGNAL_REGISTRY:
        raise ValueError(f"未知信号: {signal_name}，可用: {list(SIGNAL_REGISTRY.keys())}")

    entry = SIGNAL_REGISTRY[signal_name]
    fn = entry["fn"]
    params = {**entry["params"], **override_params}
    price_type = entry["price_type"]

    if price_type == "close":
        return fn(daily["close"], **params)
    elif price_type == "panama":
        return fn(daily["panama_close"], **params)
    elif price_type == "ohlc":
        # OHLC 信号需要 high, low, close（用比例调整）
        return fn(high=daily["high"], low=daily["low"], close=daily["close"], **params)
    else:
        raise ValueError(f"未知 price_type: {price_type}")
