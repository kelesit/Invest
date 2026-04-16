from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from .alpha101_ops import (
    Alpha101InputMatrices,
    Alpha101OperatorError,
    correlation,
    covariance,
    decay_linear,
    delay,
    delta,
    rank,
    safe_divide,
    scale,
    signedpower,
    ts_argmax,
    ts_max,
    ts_product,
    ts_rank,
    ts_mean,
    ts_min,
    ts_stddev,
    ts_sum,
)


AlphaFunction = Callable[[Alpha101InputMatrices], pd.DataFrame]

FIRST_BATCH_ALPHA_IDS = ("003", "004", "006", "012", "013", "016", "020", "033", "101")
SECOND_BATCH_ALPHA_IDS = ("008", "014", "015", "018", "022", "023", "030", "038", "040", "044", "045")
THIRD_BATCH_ALPHA_IDS = ("001", "002", "009", "010", "019", "024", "034", "046", "049", "051", "052")
FOURTH_BATCH_ALPHA_IDS = ("026", "029", "035", "037", "053", "054", "055", "060")
TIER_B_ALPHA_IDS = ("007", "017", "021", "028", "031", "039", "043", "068", "085", "088", "092", "095", "099")
IMPLEMENTED_ALPHA_IDS = (
    "001",
    "002",
    "003",
    "004",
    "006",
    "007",
    "008",
    "009",
    "010",
    "012",
    "013",
    "014",
    "015",
    "016",
    "017",
    "018",
    "019",
    "020",
    "021",
    "022",
    "023",
    "024",
    "026",
    "028",
    "029",
    "030",
    "031",
    "033",
    "034",
    "035",
    "037",
    "038",
    "039",
    "040",
    "043",
    "044",
    "045",
    "046",
    "049",
    "051",
    "052",
    "053",
    "054",
    "055",
    "060",
    "068",
    "085",
    "088",
    "092",
    "095",
    "099",
    "101",
)


def alpha001(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    base = inputs.close.where(inputs.returns >= 0, ts_stddev(inputs.returns, 20))
    return rank(ts_argmax(signedpower(base, 2.0), 5)) - 0.5


def alpha002(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    log_volume = _safe_log(inputs.volume)
    intraday_return = safe_divide(inputs.close - inputs.open, inputs.open)
    return -1.0 * correlation(rank(delta(log_volume, 2)), rank(intraday_return), 6)


def alpha003(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return -1.0 * correlation(rank(inputs.open), rank(inputs.volume), 10)


def alpha004(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return -1.0 * ts_rank(rank(inputs.low), 9)


def alpha006(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return -1.0 * correlation(inputs.open, inputs.volume, 10)


def alpha007(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    close_delta_7 = delta(inputs.close, 7)
    adv20 = inputs.adv(20)
    true_branch = (-1.0 * ts_rank(close_delta_7.abs(), 60)) * np.sign(close_delta_7)
    result = pd.DataFrame(-1.0, index=inputs.close.index, columns=inputs.close.columns)
    return true_branch.where(adv20 < inputs.volume, result).where(adv20.notna() & close_delta_7.notna())


def alpha008(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    product = ts_sum(inputs.open, 5) * ts_sum(inputs.returns, 5)
    return -1.0 * rank(product - delay(product, 10))


def alpha009(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    close_delta = delta(inputs.close, 1)
    return _signed_delta_when_rolling_extremes_agree(close_delta, 5)


def alpha010(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    close_delta = delta(inputs.close, 1)
    return rank(_signed_delta_when_rolling_extremes_agree(close_delta, 4))


def alpha012(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return np.sign(delta(inputs.volume, 1)) * (-1.0 * delta(inputs.close, 1))


def alpha013(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return -1.0 * rank(covariance(rank(inputs.close), rank(inputs.volume), 5))


def alpha014(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return (-1.0 * rank(delta(inputs.returns, 3))) * correlation(inputs.open, inputs.volume, 10)


def alpha015(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return -1.0 * ts_sum(rank(correlation(rank(inputs.high), rank(inputs.volume), 3)), 3)


def alpha016(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return -1.0 * rank(covariance(rank(inputs.high), rank(inputs.volume), 5))


def alpha017(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return (
        (-1.0 * rank(ts_rank(inputs.close, 10)))
        * rank(delta(delta(inputs.close, 1), 1))
        * rank(ts_rank(safe_divide(inputs.volume, inputs.adv(20)), 5))
    )


def alpha018(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    open_close_spread = inputs.close - inputs.open
    signal = ts_stddev(open_close_spread.abs(), 5) + open_close_spread + correlation(
        inputs.close,
        inputs.open,
        10,
    )
    return -1.0 * rank(signal)


def alpha019(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    direction = np.sign((inputs.close - delay(inputs.close, 7)) + delta(inputs.close, 7))
    return (-1.0 * direction) * (1.0 + rank(1.0 + ts_sum(inputs.returns, 250)))


def alpha020(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    open_ = inputs.open
    return (
        (-1.0 * rank(open_ - delay(inputs.high, 1)))
        * rank(open_ - delay(inputs.close, 1))
        * rank(open_ - delay(inputs.low, 1))
    )


def alpha021(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    close_mean_8 = ts_mean(inputs.close, 8)
    close_mean_2 = ts_mean(inputs.close, 2)
    close_std_8 = ts_stddev(inputs.close, 8)
    volume_ratio = safe_divide(inputs.volume, inputs.adv(20))

    result = pd.DataFrame(-1.0, index=inputs.close.index, columns=inputs.close.columns)
    result = result.where(~(volume_ratio >= 1.0), 1.0)
    result = result.where(~(close_mean_2 < (close_mean_8 - close_std_8)), 1.0)
    result = result.where(~((close_mean_8 + close_std_8) < close_mean_2), -1.0)
    valid = close_mean_8.notna() & close_mean_2.notna() & close_std_8.notna() & volume_ratio.notna()
    return result.where(valid)


def alpha022(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return -1.0 * (delta(correlation(inputs.high, inputs.volume, 5), 5) * rank(ts_stddev(inputs.close, 20)))


def alpha023(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    signal = -1.0 * delta(inputs.high, 2)
    return signal.where(ts_mean(inputs.high, 20) < inputs.high, 0.0)


def alpha024(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    close_mean_100 = ts_mean(inputs.close, 100)
    condition = safe_divide(delta(close_mean_100, 100), delay(inputs.close, 100)) <= 0.05
    true_branch = -1.0 * (inputs.close - ts_min(inputs.close, 100))
    false_branch = -1.0 * delta(inputs.close, 3)
    valid = close_mean_100.notna() & delay(close_mean_100, 100).notna() & delay(inputs.close, 100).notna()
    return true_branch.where(condition, false_branch).where(valid)


def alpha026(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return -1.0 * ts_max(correlation(ts_rank(inputs.volume, 5), ts_rank(inputs.high, 5), 5), 3)


def alpha028(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return scale((correlation(inputs.adv(20), inputs.low, 5) + ((inputs.high + inputs.low) / 2.0)) - inputs.close)


def alpha029(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    nested = rank(
        rank(
            -1.0
            * rank(
                delta(
                    inputs.close - 1.0,
                    5,
                )
            )
        )
    )
    first_term = ts_min(
        ts_product(
            rank(
                rank(
                    _safe_log(
                        ts_sum(
                            ts_min(nested, 2),
                            1,
                        )
                    ).pipe(scale)
                )
            ),
            1,
        ),
        5,
    )
    second_term = ts_rank(delay(-1.0 * inputs.returns, 6), 5)
    return first_term + second_term


def alpha030(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    sign_chain = (
        np.sign(inputs.close - delay(inputs.close, 1))
        + np.sign(delay(inputs.close, 1) - delay(inputs.close, 2))
        + np.sign(delay(inputs.close, 2) - delay(inputs.close, 3))
    )
    return safe_divide((1.0 - rank(sign_chain)) * ts_sum(inputs.volume, 5), ts_sum(inputs.volume, 20))


def alpha031(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    decay_component = decay_linear((-1.0 * rank(rank(delta(inputs.close, 10)))), 10)
    return (
        rank(rank(rank(decay_component)))
        + rank(-1.0 * delta(inputs.close, 3))
        + np.sign(scale(correlation(inputs.adv(20), inputs.low, 12)))
    )


def alpha033(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return rank(-1.0 * (1.0 - safe_divide(inputs.open, inputs.close)))


def alpha034(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    volatility_ratio = safe_divide(ts_stddev(inputs.returns, 2), ts_stddev(inputs.returns, 5))
    return rank((1.0 - rank(volatility_ratio)) + (1.0 - rank(delta(inputs.close, 1))))


def alpha035(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    price_range_signal = (inputs.close + inputs.high) - inputs.low
    return ts_rank(inputs.volume, 32) * (1.0 - ts_rank(price_range_signal, 16)) * (
        1.0 - ts_rank(inputs.returns, 32)
    )


def alpha037(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return rank(correlation(delay(inputs.open - inputs.close, 1), inputs.close, 200)) + rank(
        inputs.open - inputs.close
    )


def alpha038(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return (-1.0 * rank(ts_rank(inputs.close, 10))) * rank(safe_divide(inputs.close, inputs.open))


def alpha039(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    volume_ratio_decay = decay_linear(safe_divide(inputs.volume, inputs.adv(20)), 9)
    return (-1.0 * rank(delta(inputs.close, 7) * (1.0 - rank(volume_ratio_decay)))) * (
        1.0 + rank(ts_sum(inputs.returns, 250))
    )


def alpha040(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return (-1.0 * rank(ts_stddev(inputs.high, 10))) * correlation(inputs.high, inputs.volume, 10)


def alpha043(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return ts_rank(safe_divide(inputs.volume, inputs.adv(20)), 20) * ts_rank(-1.0 * delta(inputs.close, 7), 8)


def alpha044(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return -1.0 * correlation(inputs.high, rank(inputs.volume), 5)


def alpha045(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return -1.0 * (
        rank(ts_sum(delay(inputs.close, 5), 20) / 20.0)
        * correlation(inputs.close, inputs.volume, 2)
        * rank(correlation(ts_sum(inputs.close, 5), ts_sum(inputs.close, 20), 2))
    )


def alpha046(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    slope_diff = _close_slope_diff(inputs.close)
    return _threshold_slope_delta_alpha(inputs.close, slope_diff, lower_threshold=0.0, upper_threshold=0.25)


def alpha049(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    slope_diff = _close_slope_diff(inputs.close)
    return _negative_delta_unless_slope_too_negative(inputs.close, slope_diff, threshold=-0.1)


def alpha051(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    slope_diff = _close_slope_diff(inputs.close)
    return _negative_delta_unless_slope_too_negative(inputs.close, slope_diff, threshold=-0.05)


def alpha052(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return (
        (-1.0 * delta(ts_min(inputs.low, 5), 5))
        * rank((ts_sum(inputs.returns, 240) - ts_sum(inputs.returns, 20)) / 220.0)
        * ts_rank(inputs.volume, 5)
    )


def alpha053(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    location = safe_divide((inputs.close - inputs.low) - (inputs.high - inputs.close), inputs.close - inputs.low)
    return -1.0 * delta(location, 9)


def alpha054(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    numerator = -1.0 * (inputs.low - inputs.close) * np.power(inputs.open, 5)
    denominator = (inputs.low - inputs.high) * np.power(inputs.close, 5)
    return safe_divide(numerator, denominator)


def alpha055(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    range_position = safe_divide(
        inputs.close - ts_min(inputs.low, 12),
        ts_max(inputs.high, 12) - ts_min(inputs.low, 12),
    )
    return -1.0 * correlation(rank(range_position), rank(inputs.volume), 6)


def alpha060(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    intraday_location = safe_divide((inputs.close - inputs.low) - (inputs.high - inputs.close), inputs.high - inputs.low)
    return -1.0 * ((2.0 * scale(rank(intraday_location))) - scale(rank(ts_argmax(inputs.close, 10))))


def alpha068(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    left = ts_rank(correlation(rank(inputs.high), rank(inputs.adv(15)), 8.91644), 13.9333)
    right = rank(delta((inputs.close * 0.518371) + (inputs.low * (1.0 - 0.518371)), 1.06157))
    return (-1.0 * (left < right).astype(float)).where(left.notna() & right.notna())


def alpha085(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    left = rank(correlation((inputs.high * 0.876703) + (inputs.close * (1.0 - 0.876703)), inputs.adv(30), 9.61331))
    right = rank(correlation(ts_rank((inputs.high + inputs.low) / 2.0, 3.70596), ts_rank(inputs.volume, 10.1595), 7.11408))
    return _safe_power(left, right)


def alpha088(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    first = rank(
        decay_linear(
            (rank(inputs.open) + rank(inputs.low)) - (rank(inputs.high) + rank(inputs.close)),
            8.06882,
        )
    )
    second = ts_rank(
        decay_linear(
            correlation(ts_rank(inputs.close, 8.44728), ts_rank(inputs.adv(60), 20.6966), 8.01266),
            6.65053,
        ),
        2.61957,
    )
    return _elementwise_min(first, second)


def alpha092(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    left_side = ((inputs.high + inputs.low) / 2.0) + inputs.close
    right_side = inputs.low + inputs.open
    boolean_signal = (left_side < right_side).astype(float).where(left_side.notna() & right_side.notna())
    first = ts_rank(decay_linear(boolean_signal, 14.7221), 18.8683)
    second = ts_rank(
        decay_linear(correlation(rank(inputs.low), rank(inputs.adv(30)), 7.58555), 6.94024),
        6.80584,
    )
    return _elementwise_min(first, second)


def alpha095(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    left = rank(inputs.open - ts_min(inputs.open, 12.4105))
    corr = correlation(ts_sum((inputs.high + inputs.low) / 2.0, 19.1351), ts_sum(inputs.adv(40), 19.1351), 12.8742)
    right = ts_rank(np.power(rank(corr), 5.0), 11.7584)
    return (left < right).astype(float).where(left.notna() & right.notna())


def alpha099(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    left = rank(correlation(ts_sum((inputs.high + inputs.low) / 2.0, 19.8975), ts_sum(inputs.adv(60), 19.8975), 8.8136))
    right = rank(correlation(inputs.low, inputs.volume, 6.28259))
    return (-1.0 * (left < right).astype(float)).where(left.notna() & right.notna())


def alpha101(inputs: Alpha101InputMatrices) -> pd.DataFrame:
    return safe_divide(inputs.close - inputs.open, (inputs.high - inputs.low) + 0.001)


ALPHA101_FUNCTIONS: dict[str, AlphaFunction] = {
    "001": alpha001,
    "002": alpha002,
    "003": alpha003,
    "004": alpha004,
    "006": alpha006,
    "007": alpha007,
    "008": alpha008,
    "009": alpha009,
    "010": alpha010,
    "012": alpha012,
    "013": alpha013,
    "014": alpha014,
    "015": alpha015,
    "016": alpha016,
    "017": alpha017,
    "018": alpha018,
    "019": alpha019,
    "020": alpha020,
    "021": alpha021,
    "022": alpha022,
    "023": alpha023,
    "024": alpha024,
    "026": alpha026,
    "028": alpha028,
    "029": alpha029,
    "030": alpha030,
    "031": alpha031,
    "033": alpha033,
    "034": alpha034,
    "035": alpha035,
    "037": alpha037,
    "038": alpha038,
    "039": alpha039,
    "040": alpha040,
    "043": alpha043,
    "044": alpha044,
    "045": alpha045,
    "046": alpha046,
    "049": alpha049,
    "051": alpha051,
    "052": alpha052,
    "053": alpha053,
    "054": alpha054,
    "055": alpha055,
    "060": alpha060,
    "068": alpha068,
    "085": alpha085,
    "088": alpha088,
    "092": alpha092,
    "095": alpha095,
    "099": alpha099,
    "101": alpha101,
}


def compute_alpha101_feature_matrices(
    inputs: Alpha101InputMatrices,
    alpha_ids: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    ids = tuple(alpha_ids) if alpha_ids is not None else tuple(ALPHA101_FUNCTIONS)
    unknown = sorted(set(ids) - set(ALPHA101_FUNCTIONS))
    if unknown:
        raise Alpha101OperatorError(f"Alpha101 feature ids are not implemented: {unknown}")

    matrices: dict[str, pd.DataFrame] = {}
    for alpha_id in ids:
        matrix = ALPHA101_FUNCTIONS[alpha_id](inputs)
        matrices[alpha_id] = _validate_feature_matrix(alpha_id, matrix, inputs.close)
    return matrices


def stack_alpha101_feature_matrices(feature_matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for alpha_id, matrix in sorted(feature_matrices.items()):
        frame = (
            matrix.rename_axis(index="date", columns="symbol")
            .reset_index()
            .melt(id_vars="date", var_name="symbol", value_name=f"alpha_{alpha_id}")
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["date", "symbol"])

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, how="outer", on=["date", "symbol"])
    return result.sort_values(["date", "symbol"], ignore_index=True)


def _validate_feature_matrix(
    alpha_id: str,
    matrix: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    if not matrix.index.equals(reference.index):
        raise Alpha101OperatorError(f"alpha_{alpha_id} index does not match input calendar")
    if not matrix.columns.equals(reference.columns):
        raise Alpha101OperatorError(f"alpha_{alpha_id} columns do not match input universe")
    return matrix.astype(float)


def _safe_log(x: pd.DataFrame) -> pd.DataFrame:
    values = x.to_numpy(dtype=float)
    logged = np.full_like(values, np.nan, dtype=float)
    positive = values > 0
    logged[positive] = np.log(values[positive])
    return pd.DataFrame(logged, index=x.index, columns=x.columns)


def _safe_power(base: pd.DataFrame, exponent: pd.DataFrame) -> pd.DataFrame:
    base_aligned, exponent_aligned = base.align(exponent, join="outer", axis=None)
    values = np.power(base_aligned.to_numpy(dtype=float), exponent_aligned.to_numpy(dtype=float))
    return pd.DataFrame(values, index=base_aligned.index, columns=base_aligned.columns)


def _elementwise_min(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    left_aligned, right_aligned = left.align(right, join="outer", axis=None)
    return pd.DataFrame(
        np.minimum(left_aligned.to_numpy(dtype=float), right_aligned.to_numpy(dtype=float)),
        index=left_aligned.index,
        columns=left_aligned.columns,
    )


def _signed_delta_when_rolling_extremes_agree(close_delta: pd.DataFrame, window: float) -> pd.DataFrame:
    rolling_min = ts_min(close_delta, window)
    rolling_max = ts_max(close_delta, window)
    same_positive = rolling_min > 0
    same_negative = rolling_max < 0
    valid = rolling_min.notna() & rolling_max.notna() & close_delta.notna()
    return close_delta.where(same_positive | same_negative, -1.0 * close_delta).where(valid)


def _close_slope_diff(close: pd.DataFrame) -> pd.DataFrame:
    return ((delay(close, 20) - delay(close, 10)) / 10.0) - ((delay(close, 10) - close) / 10.0)


def _threshold_slope_delta_alpha(
    close: pd.DataFrame,
    slope_diff: pd.DataFrame,
    lower_threshold: float,
    upper_threshold: float,
) -> pd.DataFrame:
    true_branch = pd.DataFrame(-1.0, index=close.index, columns=close.columns)
    middle_branch = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    false_branch = -1.0 * delta(close, 1)
    result = false_branch.where(slope_diff >= lower_threshold, middle_branch)
    result = result.where(slope_diff <= upper_threshold, true_branch)
    return result.where(slope_diff.notna())


def _negative_delta_unless_slope_too_negative(
    close: pd.DataFrame,
    slope_diff: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    true_branch = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    false_branch = -1.0 * delta(close, 1)
    return false_branch.where(slope_diff >= threshold, true_branch).where(slope_diff.notna())
