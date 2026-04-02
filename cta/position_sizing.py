"""仓位计算模块。

核心逻辑：波动率标准化（ATR-based），使每个品种对组合的风险贡献大致相等。
"""

import pandas as pd


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """计算 Average True Range (ATR)。"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def volatility_sized_position(
    signal: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    capital: float,
    risk_fraction: float = 0.01,
    point_value: float = 50.0,
    atr_period: int = 20,
) -> pd.Series:
    """根据波动率标准化计算持仓手数。

    公式: 持仓手数 = (capital * risk_fraction) / (ATR * point_value)
    然后乘以信号方向 (+1/-1)。

    参数:
        signal: 信号序列 (+1/-1/0)
        high, low, close: 价格序列
        capital: 组合总资金
        risk_fraction: 每个品种的目标风险占比（默认 1%）
        point_value: 合约乘数（ES = 50 美元/点）
        atr_period: ATR 计算窗口

    返回: 持仓手数序列（浮点数，实际交易时取整）
    """
    atr = calc_atr(high, low, close, period=atr_period)
    dollar_risk_per_contract = atr * point_value
    # 避免除以零
    dollar_risk_per_contract = dollar_risk_per_contract.replace(0, float("nan"))
    target_risk = capital * risk_fraction
    num_contracts = target_risk / dollar_risk_per_contract
    position = signal * num_contracts
    return position
