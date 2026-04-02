"""信号生成模块。

每个信号函数的接口统一：
    输入: 价格序列 (pd.Series, close prices) + 参数
    输出: 信号序列 (pd.Series, 值为 +1 / -1 / 0)
"""

import pandas as pd


def momentum(close: pd.Series, lookback: int = 20) -> pd.Series:
    """时间序列动量：过去 N 天收益率的符号。

    - 过去 lookback 天收益 > 0 → +1 (做多)
    - 过去 lookback 天收益 < 0 → -1 (做空)
    - 否则 → 0
    """
    returns = close.pct_change(lookback)
    signal = pd.Series(0, index=close.index, dtype=float)
    signal[returns > 0] = 1.0
    signal[returns < 0] = -1.0
    return signal


def ma_crossover(close: pd.Series, fast: int = 10, slow: int = 50) -> pd.Series:
    """均线交叉：快线在慢线上方做多，反之做空。

    - fast MA > slow MA → +1 (做多)
    - fast MA < slow MA → -1 (做空)
    """
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    signal = pd.Series(0, index=close.index, dtype=float)
    signal[ma_fast > ma_slow] = 1.0
    signal[ma_fast < ma_slow] = -1.0
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
