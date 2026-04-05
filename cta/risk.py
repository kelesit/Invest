"""风控模块。

三种风控机制：
    1. 止损 (trailing stop) — 从最高点回撤超过阈值时平仓
    2. 波动率择时 (vol scaling) — 高波动时缩小仓位，低波动时放大
    3. 组合波动率目标 (vol targeting) — 动态调整整体仓位使组合波动率维持在目标水平
"""

import numpy as np
import pandas as pd

from cta.position_sizing import calc_atr


def trailing_stop(
    position: pd.Series,
    close: pd.Series,
    atr_mult: float = 3.0,
    atr_period: int = 20,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
) -> pd.Series:
    """追踪止损：持仓期间价格从极值回撤超过 atr_mult × ATR 时平仓。

    - 多头：价格从持仓期最高点回落 > atr_mult × ATR → 平仓
    - 空头：价格从持仓期最低点反弹 > atr_mult × ATR → 平仓
    - 平仓后保持空仓，直到原始信号方向翻转

    参数:
        position: 原始仓位序列
        close: 收盘价
        atr_mult: ATR 倍数（越大越宽松）
        atr_period: ATR 计算窗口
        high, low: 最高/最低价（用于 ATR 计算），如不提供则用 close 近似
    """
    if high is None:
        high = close
    if low is None:
        low = close

    atr = calc_atr(high, low, close, period=atr_period)
    result = position.copy()

    # 追踪止损状态
    stopped = False
    stopped_direction = 0  # 被止损时的方向
    tracking_high = -np.inf
    tracking_low = np.inf

    for i in range(len(position)):
        pos = position.iloc[i]
        price = close.iloc[i]
        current_atr = atr.iloc[i]

        if np.isnan(current_atr) or current_atr == 0:
            continue

        # 如果被止损了，等信号方向翻转才恢复
        if stopped:
            if np.sign(pos) != stopped_direction and pos != 0:
                stopped = False
                tracking_high = price if pos > 0 else -np.inf
                tracking_low = price if pos < 0 else np.inf
            else:
                result.iloc[i] = 0
                continue

        if pos > 0:  # 多头
            tracking_high = max(tracking_high, price)
            tracking_low = np.inf
            if tracking_high - price > atr_mult * current_atr:
                result.iloc[i] = 0
                stopped = True
                stopped_direction = 1
                tracking_high = -np.inf
        elif pos < 0:  # 空头
            tracking_low = min(tracking_low, price)
            tracking_high = -np.inf
            if price - tracking_low > atr_mult * current_atr:
                result.iloc[i] = 0
                stopped = True
                stopped_direction = -1
                tracking_low = np.inf
        else:  # 空仓
            tracking_high = -np.inf
            tracking_low = np.inf
            stopped = False

    return result


def vol_scale(
    position: pd.Series,
    close: pd.Series,
    target_vol: float = 0.15,
    vol_window: int = 60,
    vol_cap: float = 2.0,
) -> pd.Series:
    """波动率择时：根据当前波动率与目标的比值调整仓位。

    当实际波动率 > 目标波动率时缩小仓位，反之放大。

    公式: 调整后仓位 = 原始仓位 × (目标波动率 / 实际波动率)

    参数:
        position: 原始仓位序列
        close: 收盘价
        target_vol: 目标年化波动率（默认 15%）
        vol_window: 波动率计算窗口
        vol_cap: 缩放倍数上限（防止低波动时仓位过大）
    """
    daily_returns = close.pct_change()
    realized_vol = daily_returns.rolling(vol_window, min_periods=20).std() * np.sqrt(252)

    scale_factor = target_vol / realized_vol.replace(0, np.nan)
    scale_factor = scale_factor.clip(0, vol_cap).fillna(1.0)

    # 用前一天的缩放系数，避免前视偏差
    scale_factor = scale_factor.shift(1).fillna(1.0)

    return position * scale_factor


def vol_target_portfolio(
    portfolio_pnl: pd.Series,
    capital: float,
    target_vol: float = 0.10,
    vol_window: int = 60,
    scale_cap: float = 2.0,
) -> pd.Series:
    """组合层面波动率目标：动态调整整体仓位使组合波动率维持在目标水平。

    与 vol_scale 的区别：vol_scale 在单品种层面调整，这个在组合层面调整。

    返回: 缩放系数序列（乘到每个品种的仓位上）
    """
    portfolio_returns = portfolio_pnl / capital
    realized_vol = portfolio_returns.rolling(vol_window, min_periods=20).std() * np.sqrt(252)

    scale_factor = target_vol / realized_vol.replace(0, np.nan)
    scale_factor = scale_factor.clip(0, scale_cap).fillna(1.0)

    # 用前一天的缩放系数（避免前视偏差）
    return scale_factor.shift(1).fillna(1.0)
