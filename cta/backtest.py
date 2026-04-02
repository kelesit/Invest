"""向量化回测引擎。

输入: 价格数据 + 持仓序列
输出: 每日权益曲线
"""

import pandas as pd


def run_backtest(
    daily_price: pd.DataFrame,
    position: pd.Series,
    point_value: float = 50.0,
    commission_per_contract: float = 2.5,
    slippage_points: float = 0.25,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """运行单品种向量化回测。

    参数:
        daily_price: 日线 DataFrame，需包含 close 列
        position: 持仓手数序列（正=多头，负=空头）
        point_value: 合约乘数
        commission_per_contract: 每手单边手续费
        slippage_points: 滑点（点数）
        initial_capital: 初始资金

    返回: DataFrame，包含 pnl, commission, slippage, net_pnl, equity 列
    """
    # 对齐索引
    common_idx = daily_price.index.intersection(position.index)
    price = daily_price.loc[common_idx]
    pos = position.loc[common_idx]

    # 每日价格变动
    price_change = price["close"].diff()

    # 持仓收益：昨日持仓 × 今日价格变动 × 合约乘数
    prev_pos = pos.shift(1).fillna(0)
    pnl = prev_pos * price_change * point_value

    # 换手（持仓变动的绝对值）
    turnover = pos.diff().abs().fillna(pos.abs())

    # 手续费：换手 × 每手单边手续费
    commission = turnover * commission_per_contract

    # 滑点：换手 × 滑点点数 × 合约乘数
    slippage = turnover * slippage_points * point_value

    # 净收益（热身期 NaN 填 0）
    net_pnl = (pnl - commission - slippage).fillna(0)

    # 权益曲线
    equity = initial_capital + net_pnl.cumsum()

    result = pd.DataFrame({
        "position": pos,
        "price_change": price_change,
        "pnl": pnl,
        "commission": commission,
        "slippage": slippage,
        "net_pnl": net_pnl,
        "equity": equity,
    })
    return result
