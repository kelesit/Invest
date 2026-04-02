"""组合优化模块。

三种权重分配方式：
    1. 等权 (equal weight) — 每个品种分配相同资金
    2. 风险平价 (risk parity) — 每个品种贡献相同风险
    3. 动态风险平价 — 用滚动相关性和波动率实时调整权重
"""

import numpy as np
import pandas as pd


def equal_weight(n_products: int) -> dict[str, float]:
    """等权分配：每个品种分 1/N。"""
    return 1.0 / n_products


def inverse_vol_weights(
    product_returns: dict[str, pd.Series],
    vol_window: int = 60,
) -> pd.DataFrame:
    """逆波动率权重：波动率低的品种权重高。

    权重_i = (1/σ_i) / Σ(1/σ_j)

    这是风险平价在"不考虑相关性"时的简化版本。
    当各品种两两不相关时，逆波动率权重 = 风险平价权重。

    返回: DataFrame，index=日期，columns=品种名，值=权重（每行和为1）
    """
    # 计算滚动波动率
    vol_df = pd.DataFrame({
        name: ret.rolling(vol_window, min_periods=20).std() * np.sqrt(252)
        for name, ret in product_returns.items()
    })

    # 逆波动率
    inv_vol = 1.0 / vol_df.replace(0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    # 前视偏差修正：用前一天的权重
    weights = weights.shift(1)

    # NaN 填等权
    n = len(product_returns)
    weights = weights.fillna(1.0 / n)

    return weights


def risk_parity_weights(
    product_returns: dict[str, pd.Series],
    corr_window: int = 120,
    vol_window: int = 60,
) -> pd.DataFrame:
    """风险平价权重（考虑相关性）。

    目标：每个品种对组合总波动率的边际贡献相等。

    组合波动率 σ_p = sqrt(w' Σ w)
    品种 i 的风险贡献 RC_i = w_i × (Σw)_i / σ_p

    用迭代法求解使 RC_1 = RC_2 = ... = RC_n 的权重。

    返回: DataFrame，index=日期，columns=品种名，值=权重（每行和为1）
    """
    returns_df = pd.DataFrame(product_returns)
    products = returns_df.columns.tolist()
    n = len(products)

    # 计算滚动协方差矩阵
    rolling_weights = []
    dates = returns_df.index

    # 需要足够的历史数据
    min_periods = max(corr_window, vol_window)

    for i in range(len(dates)):
        if i < min_periods:
            rolling_weights.append(np.ones(n) / n)
            continue

        window_returns = returns_df.iloc[max(0, i - corr_window):i]
        cov_matrix = window_returns.cov().values * 252  # 年化

        # 检查协方差矩阵是否有效
        if np.any(np.isnan(cov_matrix)) or np.any(np.diag(cov_matrix) <= 0):
            rolling_weights.append(np.ones(n) / n)
            continue

        w = _solve_risk_parity(cov_matrix)
        rolling_weights.append(w)

    weights = pd.DataFrame(rolling_weights, index=dates, columns=products)

    # 前视偏差修正
    weights = weights.shift(1).fillna(1.0 / n)

    return weights


def _solve_risk_parity(cov_matrix: np.ndarray, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """求解风险平价权重（Newton-Raphson 迭代法的简化版本）。

    使用 Spinu (2013) 的公式化简：
    w_i ∝ 1 / (Σw)_i

    从等权开始迭代，通常 10-20 次收敛。
    """
    n = len(cov_matrix)
    w = np.ones(n) / n

    for _ in range(max_iter):
        sigma_w = cov_matrix @ w
        # 风险贡献
        rc = w * sigma_w
        total_risk = rc.sum()

        if total_risk <= 0:
            return np.ones(n) / n

        # 目标：每个 RC 相等 = total_risk / n
        target_rc = total_risk / n

        # 更新：w_i_new ∝ target_rc / sigma_w_i
        w_new = target_rc / np.maximum(sigma_w, 1e-10)
        w_new = w_new / w_new.sum()

        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w = w_new

    return w


def calc_dynamic_correlation(
    product_returns: dict[str, pd.Series],
    window: int = 60,
) -> pd.DataFrame:
    """计算品种间滚动相关性矩阵（用于分析和可视化）。

    返回: 长格式 DataFrame (date, pair, correlation)
    """
    returns_df = pd.DataFrame(product_returns)
    products = returns_df.columns.tolist()

    records = []
    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            p1, p2 = products[i], products[j]
            rolling_corr = returns_df[p1].rolling(window).corr(returns_df[p2])
            for date, corr_val in rolling_corr.items():
                if not np.isnan(corr_val):
                    records.append({
                        "date": date,
                        "pair": f"{p1}-{p2}",
                        "correlation": corr_val,
                    })

    return pd.DataFrame(records)
