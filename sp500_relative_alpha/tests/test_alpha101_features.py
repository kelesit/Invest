from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from sp500_relative_alpha.alpha101_features import (
    ALPHA101_FUNCTIONS,
    FIRST_BATCH_ALPHA_IDS,
    FOURTH_BATCH_ALPHA_IDS,
    IMPLEMENTED_ALPHA_IDS,
    SECOND_BATCH_ALPHA_IDS,
    THIRD_BATCH_ALPHA_IDS,
    TIER_B_ALPHA_IDS,
    alpha001,
    alpha002,
    alpha003,
    alpha004,
    alpha006,
    alpha007,
    alpha008,
    alpha009,
    alpha010,
    alpha012,
    alpha013,
    alpha014,
    alpha015,
    alpha016,
    alpha017,
    alpha018,
    alpha019,
    alpha020,
    alpha021,
    alpha022,
    alpha023,
    alpha024,
    alpha026,
    alpha028,
    alpha029,
    alpha030,
    alpha031,
    alpha033,
    alpha034,
    alpha035,
    alpha037,
    alpha038,
    alpha039,
    alpha040,
    alpha043,
    alpha044,
    alpha045,
    alpha046,
    alpha049,
    alpha051,
    alpha052,
    alpha053,
    alpha054,
    alpha055,
    alpha060,
    alpha068,
    alpha085,
    alpha088,
    alpha092,
    alpha095,
    alpha099,
    alpha101,
    compute_alpha101_feature_matrices,
    stack_alpha101_feature_matrices,
)
from sp500_relative_alpha.alpha101_ops import (
    build_alpha101_input_matrices,
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
    ts_mean,
    ts_min,
    ts_product,
    ts_rank,
    ts_stddev,
    ts_sum,
)


def _make_daily_bars(n_days: int = 280) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rows: list[dict[str, object]] = []
    symbol_params = {
        "AAPL": (20.0, 0.020, 1500.0, 0.0),
        "AMZN": (19.8, 0.015, 1510.0, 21.0),
        "GOOGL": (20.1, 0.000, 1490.0, 28.0),
        "MSFT": (20.2, -0.005, 1520.0, 7.0),
        "NVDA": (20.4, 0.010, 1480.0, 14.0),
        "SPY": (100.0, 0.015, 5000.0, 3.0),
    }
    for symbol, (base, slope, volume_base, phase) in symbol_params.items():
        for i, date in enumerate(dates):
            open_price = base + slope * i + 3.0 * np.sin((i + phase) / 5.0) + 1.2 * np.cos((i + phase) / 11.0)
            close = open_price + 0.35 * np.cos((i + phase) / 2.0) + 0.15 * np.sin((i + phase) / 9.0)
            high = max(open_price, close) + 0.5 + 0.2 * np.sin((i + phase) / 13.0)
            low = min(open_price, close) - 0.4 - 0.2 * np.cos((i + phase) / 17.0)
            shares_volume = (
                volume_base
                + 5.0 * i
                + 350.0 * np.cos((i + phase) / 6.0)
                + 250.0 * np.sin((i + phase) / 17.0)
            )
            typical_price = (high + low + close) / 3.0
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "shares_volume": shares_volume,
                    "typical_price": typical_price,
                    "alpha_volume": typical_price * shares_volume,
                    "close_to_close_return": np.nan,
                }
            )
    bars = pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)
    bars["close_to_close_return"] = bars.groupby("symbol", sort=False)["close"].pct_change()
    return bars


def _elementwise_min(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    left_aligned, right_aligned = left.align(right, join="outer", axis=None)
    return pd.DataFrame(
        np.minimum(left_aligned.to_numpy(dtype=float), right_aligned.to_numpy(dtype=float)),
        index=left_aligned.index,
        columns=left_aligned.columns,
    )


class Alpha101FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = build_alpha101_input_matrices(_make_daily_bars())

    def assertFrameEqual(self, left: pd.DataFrame, right: pd.DataFrame) -> None:  # noqa: N802
        pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)

    def test_registry_is_exactly_implemented_ids(self) -> None:
        self.assertEqual(tuple(ALPHA101_FUNCTIONS), IMPLEMENTED_ALPHA_IDS)
        self.assertTrue(set(FIRST_BATCH_ALPHA_IDS).issubset(ALPHA101_FUNCTIONS))
        self.assertTrue(set(SECOND_BATCH_ALPHA_IDS).issubset(ALPHA101_FUNCTIONS))
        self.assertTrue(set(THIRD_BATCH_ALPHA_IDS).issubset(ALPHA101_FUNCTIONS))
        self.assertTrue(set(FOURTH_BATCH_ALPHA_IDS).issubset(ALPHA101_FUNCTIONS))
        self.assertTrue(set(TIER_B_ALPHA_IDS).issubset(ALPHA101_FUNCTIONS))

    def test_alpha001_formula(self) -> None:
        base = self.inputs.close.where(self.inputs.returns >= 0, ts_stddev(self.inputs.returns, 20))
        expected = rank(ts_argmax(signedpower(base, 2.0), 5)) - 0.5
        self.assertFrameEqual(alpha001(self.inputs), expected)

    def test_alpha002_formula(self) -> None:
        volume_values = self.inputs.volume.to_numpy(dtype=float)
        log_values = np.full_like(volume_values, np.nan, dtype=float)
        positive = volume_values > 0
        log_values[positive] = np.log(volume_values[positive])
        log_volume = pd.DataFrame(
            log_values,
            index=self.inputs.volume.index,
            columns=self.inputs.volume.columns,
        )
        intraday_return = safe_divide(self.inputs.close - self.inputs.open, self.inputs.open)
        expected = -1.0 * correlation(rank(delta(log_volume, 2)), rank(intraday_return), 6)
        self.assertFrameEqual(alpha002(self.inputs), expected)

    def test_alpha003_formula(self) -> None:
        expected = -1.0 * correlation(rank(self.inputs.open), rank(self.inputs.volume), 10)
        self.assertFrameEqual(alpha003(self.inputs), expected)

    def test_alpha004_formula(self) -> None:
        expected = -1.0 * ts_rank(rank(self.inputs.low), 9)
        self.assertFrameEqual(alpha004(self.inputs), expected)

    def test_alpha006_formula(self) -> None:
        expected = -1.0 * correlation(self.inputs.open, self.inputs.volume, 10)
        self.assertFrameEqual(alpha006(self.inputs), expected)

    def test_alpha007_formula(self) -> None:
        close_delta_7 = delta(self.inputs.close, 7)
        adv20 = self.inputs.adv(20)
        true_branch = (-1.0 * ts_rank(close_delta_7.abs(), 60)) * np.sign(close_delta_7)
        expected = true_branch.where(adv20 < self.inputs.volume, -1.0).where(adv20.notna() & close_delta_7.notna())
        self.assertFrameEqual(alpha007(self.inputs), expected)

    def test_alpha008_formula(self) -> None:
        product = ts_sum(self.inputs.open, 5) * ts_sum(self.inputs.returns, 5)
        expected = -1.0 * rank(product - delay(product, 10))
        self.assertFrameEqual(alpha008(self.inputs), expected)

    def test_alpha009_formula(self) -> None:
        close_delta = delta(self.inputs.close, 1)
        rolling_min = ts_min(close_delta, 5)
        rolling_max = ts_max(close_delta, 5)
        expected = close_delta.where((rolling_min > 0) | (rolling_max < 0), -1.0 * close_delta).where(
            rolling_min.notna() & rolling_max.notna() & close_delta.notna()
        )
        self.assertFrameEqual(alpha009(self.inputs), expected)

    def test_alpha010_formula(self) -> None:
        close_delta = delta(self.inputs.close, 1)
        rolling_min = ts_min(close_delta, 4)
        rolling_max = ts_max(close_delta, 4)
        signed_delta = close_delta.where((rolling_min > 0) | (rolling_max < 0), -1.0 * close_delta).where(
            rolling_min.notna() & rolling_max.notna() & close_delta.notna()
        )
        self.assertFrameEqual(alpha010(self.inputs), rank(signed_delta))

    def test_alpha012_formula(self) -> None:
        expected = np.sign(delta(self.inputs.volume, 1)) * (-1.0 * delta(self.inputs.close, 1))
        self.assertFrameEqual(alpha012(self.inputs), expected)

    def test_alpha013_formula(self) -> None:
        expected = -1.0 * rank(covariance(rank(self.inputs.close), rank(self.inputs.volume), 5))
        self.assertFrameEqual(alpha013(self.inputs), expected)

    def test_alpha014_formula(self) -> None:
        expected = (-1.0 * rank(delta(self.inputs.returns, 3))) * correlation(self.inputs.open, self.inputs.volume, 10)
        self.assertFrameEqual(alpha014(self.inputs), expected)

    def test_alpha015_formula(self) -> None:
        expected = -1.0 * ts_sum(rank(correlation(rank(self.inputs.high), rank(self.inputs.volume), 3)), 3)
        self.assertFrameEqual(alpha015(self.inputs), expected)

    def test_alpha016_formula(self) -> None:
        expected = -1.0 * rank(covariance(rank(self.inputs.high), rank(self.inputs.volume), 5))
        self.assertFrameEqual(alpha016(self.inputs), expected)

    def test_alpha017_formula(self) -> None:
        expected = (
            (-1.0 * rank(ts_rank(self.inputs.close, 10)))
            * rank(delta(delta(self.inputs.close, 1), 1))
            * rank(ts_rank(safe_divide(self.inputs.volume, self.inputs.adv(20)), 5))
        )
        self.assertFrameEqual(alpha017(self.inputs), expected)

    def test_alpha018_formula(self) -> None:
        spread = self.inputs.close - self.inputs.open
        expected = -1.0 * rank(ts_stddev(spread.abs(), 5) + spread + correlation(self.inputs.close, self.inputs.open, 10))
        self.assertFrameEqual(alpha018(self.inputs), expected)

    def test_alpha019_formula(self) -> None:
        direction = np.sign((self.inputs.close - delay(self.inputs.close, 7)) + delta(self.inputs.close, 7))
        expected = (-1.0 * direction) * (1.0 + rank(1.0 + ts_sum(self.inputs.returns, 250)))
        self.assertFrameEqual(alpha019(self.inputs), expected)

    def test_alpha020_formula(self) -> None:
        open_ = self.inputs.open
        expected = (
            (-1.0 * rank(open_ - delay(self.inputs.high, 1)))
            * rank(open_ - delay(self.inputs.close, 1))
            * rank(open_ - delay(self.inputs.low, 1))
        )
        self.assertFrameEqual(alpha020(self.inputs), expected)

    def test_alpha021_formula(self) -> None:
        close_mean_8 = ts_mean(self.inputs.close, 8)
        close_mean_2 = ts_mean(self.inputs.close, 2)
        close_std_8 = ts_stddev(self.inputs.close, 8)
        volume_ratio = safe_divide(self.inputs.volume, self.inputs.adv(20))
        expected = pd.DataFrame(-1.0, index=self.inputs.close.index, columns=self.inputs.close.columns)
        expected = expected.where(~(volume_ratio >= 1.0), 1.0)
        expected = expected.where(~(close_mean_2 < (close_mean_8 - close_std_8)), 1.0)
        expected = expected.where(~((close_mean_8 + close_std_8) < close_mean_2), -1.0)
        valid = close_mean_8.notna() & close_mean_2.notna() & close_std_8.notna() & volume_ratio.notna()
        self.assertFrameEqual(alpha021(self.inputs), expected.where(valid))

    def test_alpha022_formula(self) -> None:
        expected = -1.0 * (
            delta(correlation(self.inputs.high, self.inputs.volume, 5), 5) * rank(ts_stddev(self.inputs.close, 20))
        )
        self.assertFrameEqual(alpha022(self.inputs), expected)

    def test_alpha023_formula(self) -> None:
        expected = (-1.0 * delta(self.inputs.high, 2)).where(ts_mean(self.inputs.high, 20) < self.inputs.high, 0.0)
        self.assertFrameEqual(alpha023(self.inputs), expected)

    def test_alpha024_formula(self) -> None:
        close_mean_100 = ts_mean(self.inputs.close, 100)
        condition = safe_divide(delta(close_mean_100, 100), delay(self.inputs.close, 100)) <= 0.05
        true_branch = -1.0 * (self.inputs.close - ts_min(self.inputs.close, 100))
        false_branch = -1.0 * delta(self.inputs.close, 3)
        valid = close_mean_100.notna() & delay(close_mean_100, 100).notna() & delay(self.inputs.close, 100).notna()
        self.assertFrameEqual(alpha024(self.inputs), true_branch.where(condition, false_branch).where(valid))

    def test_alpha026_formula(self) -> None:
        expected = -1.0 * ts_max(correlation(ts_rank(self.inputs.volume, 5), ts_rank(self.inputs.high, 5), 5), 3)
        self.assertFrameEqual(alpha026(self.inputs), expected)

    def test_alpha028_formula(self) -> None:
        expected = scale((correlation(self.inputs.adv(20), self.inputs.low, 5) + ((self.inputs.high + self.inputs.low) / 2.0)) - self.inputs.close)
        self.assertFrameEqual(alpha028(self.inputs), expected)

    def test_alpha029_formula(self) -> None:
        nested = rank(rank(-1.0 * rank(delta(self.inputs.close - 1.0, 5))))
        log_input = ts_sum(ts_min(nested, 2), 1)
        log_values = log_input.to_numpy(dtype=float)
        safe_logged_values = np.full_like(log_values, np.nan, dtype=float)
        positive = log_values > 0
        safe_logged_values[positive] = np.log(log_values[positive])
        safe_logged = pd.DataFrame(safe_logged_values, index=log_input.index, columns=log_input.columns)
        expected = ts_min(ts_product(rank(rank(scale(safe_logged))), 1), 5) + ts_rank(
            delay(-1.0 * self.inputs.returns, 6),
            5,
        )
        self.assertFrameEqual(alpha029(self.inputs), expected)

    def test_alpha030_formula(self) -> None:
        sign_chain = (
            np.sign(self.inputs.close - delay(self.inputs.close, 1))
            + np.sign(delay(self.inputs.close, 1) - delay(self.inputs.close, 2))
            + np.sign(delay(self.inputs.close, 2) - delay(self.inputs.close, 3))
        )
        expected = safe_divide((1.0 - rank(sign_chain)) * ts_sum(self.inputs.volume, 5), ts_sum(self.inputs.volume, 20))
        self.assertFrameEqual(alpha030(self.inputs), expected)

    def test_alpha031_formula(self) -> None:
        decay_component = decay_linear((-1.0 * rank(rank(delta(self.inputs.close, 10)))), 10)
        expected = (
            rank(rank(rank(decay_component)))
            + rank(-1.0 * delta(self.inputs.close, 3))
            + np.sign(scale(correlation(self.inputs.adv(20), self.inputs.low, 12)))
        )
        self.assertFrameEqual(alpha031(self.inputs), expected)

    def test_alpha033_formula(self) -> None:
        expected = rank(-1.0 * (1.0 - safe_divide(self.inputs.open, self.inputs.close)))
        self.assertFrameEqual(alpha033(self.inputs), expected)

    def test_alpha034_formula(self) -> None:
        volatility_ratio = safe_divide(ts_stddev(self.inputs.returns, 2), ts_stddev(self.inputs.returns, 5))
        expected = rank((1.0 - rank(volatility_ratio)) + (1.0 - rank(delta(self.inputs.close, 1))))
        self.assertFrameEqual(alpha034(self.inputs), expected)

    def test_alpha035_formula(self) -> None:
        price_range_signal = (self.inputs.close + self.inputs.high) - self.inputs.low
        expected = ts_rank(self.inputs.volume, 32) * (1.0 - ts_rank(price_range_signal, 16)) * (
            1.0 - ts_rank(self.inputs.returns, 32)
        )
        self.assertFrameEqual(alpha035(self.inputs), expected)

    def test_alpha037_formula(self) -> None:
        expected = rank(correlation(delay(self.inputs.open - self.inputs.close, 1), self.inputs.close, 200)) + rank(
            self.inputs.open - self.inputs.close
        )
        self.assertFrameEqual(alpha037(self.inputs), expected)

    def test_alpha038_formula(self) -> None:
        expected = (-1.0 * rank(ts_rank(self.inputs.close, 10))) * rank(safe_divide(self.inputs.close, self.inputs.open))
        self.assertFrameEqual(alpha038(self.inputs), expected)

    def test_alpha039_formula(self) -> None:
        volume_ratio_decay = decay_linear(safe_divide(self.inputs.volume, self.inputs.adv(20)), 9)
        expected = (-1.0 * rank(delta(self.inputs.close, 7) * (1.0 - rank(volume_ratio_decay)))) * (
            1.0 + rank(ts_sum(self.inputs.returns, 250))
        )
        self.assertFrameEqual(alpha039(self.inputs), expected)

    def test_alpha040_formula(self) -> None:
        expected = (-1.0 * rank(ts_stddev(self.inputs.high, 10))) * correlation(self.inputs.high, self.inputs.volume, 10)
        self.assertFrameEqual(alpha040(self.inputs), expected)

    def test_alpha043_formula(self) -> None:
        expected = ts_rank(safe_divide(self.inputs.volume, self.inputs.adv(20)), 20) * ts_rank(
            -1.0 * delta(self.inputs.close, 7),
            8,
        )
        self.assertFrameEqual(alpha043(self.inputs), expected)

    def test_alpha044_formula(self) -> None:
        expected = -1.0 * correlation(self.inputs.high, rank(self.inputs.volume), 5)
        self.assertFrameEqual(alpha044(self.inputs), expected)

    def test_alpha045_formula(self) -> None:
        expected = -1.0 * (
            rank(ts_sum(delay(self.inputs.close, 5), 20) / 20.0)
            * correlation(self.inputs.close, self.inputs.volume, 2)
            * rank(correlation(ts_sum(self.inputs.close, 5), ts_sum(self.inputs.close, 20), 2))
        )
        self.assertFrameEqual(alpha045(self.inputs), expected)

    def test_alpha046_formula(self) -> None:
        slope_diff = ((delay(self.inputs.close, 20) - delay(self.inputs.close, 10)) / 10.0) - (
            (delay(self.inputs.close, 10) - self.inputs.close) / 10.0
        )
        expected = (-1.0 * delta(self.inputs.close, 1)).where(slope_diff >= 0.0, 1.0)
        expected = expected.where(slope_diff <= 0.25, -1.0).where(slope_diff.notna())
        self.assertFrameEqual(alpha046(self.inputs), expected)

    def test_alpha049_formula(self) -> None:
        slope_diff = ((delay(self.inputs.close, 20) - delay(self.inputs.close, 10)) / 10.0) - (
            (delay(self.inputs.close, 10) - self.inputs.close) / 10.0
        )
        expected = (-1.0 * delta(self.inputs.close, 1)).where(slope_diff >= -0.1, 1.0).where(slope_diff.notna())
        self.assertFrameEqual(alpha049(self.inputs), expected)

    def test_alpha051_formula(self) -> None:
        slope_diff = ((delay(self.inputs.close, 20) - delay(self.inputs.close, 10)) / 10.0) - (
            (delay(self.inputs.close, 10) - self.inputs.close) / 10.0
        )
        expected = (-1.0 * delta(self.inputs.close, 1)).where(slope_diff >= -0.05, 1.0).where(slope_diff.notna())
        self.assertFrameEqual(alpha051(self.inputs), expected)

    def test_alpha052_formula(self) -> None:
        expected = (
            (-1.0 * delta(ts_min(self.inputs.low, 5), 5))
            * rank((ts_sum(self.inputs.returns, 240) - ts_sum(self.inputs.returns, 20)) / 220.0)
            * ts_rank(self.inputs.volume, 5)
        )
        self.assertFrameEqual(alpha052(self.inputs), expected)

    def test_alpha053_formula(self) -> None:
        location = safe_divide(
            (self.inputs.close - self.inputs.low) - (self.inputs.high - self.inputs.close),
            self.inputs.close - self.inputs.low,
        )
        self.assertFrameEqual(alpha053(self.inputs), -1.0 * delta(location, 9))

    def test_alpha054_formula(self) -> None:
        numerator = -1.0 * (self.inputs.low - self.inputs.close) * np.power(self.inputs.open, 5)
        denominator = (self.inputs.low - self.inputs.high) * np.power(self.inputs.close, 5)
        self.assertFrameEqual(alpha054(self.inputs), safe_divide(numerator, denominator))

    def test_alpha055_formula(self) -> None:
        range_position = safe_divide(
            self.inputs.close - ts_min(self.inputs.low, 12),
            ts_max(self.inputs.high, 12) - ts_min(self.inputs.low, 12),
        )
        self.assertFrameEqual(alpha055(self.inputs), -1.0 * correlation(rank(range_position), rank(self.inputs.volume), 6))

    def test_alpha060_formula(self) -> None:
        intraday_location = safe_divide(
            (self.inputs.close - self.inputs.low) - (self.inputs.high - self.inputs.close),
            self.inputs.high - self.inputs.low,
        )
        expected = -1.0 * ((2.0 * scale(rank(intraday_location))) - scale(rank(ts_argmax(self.inputs.close, 10))))
        self.assertFrameEqual(alpha060(self.inputs), expected)

    def test_alpha068_formula(self) -> None:
        left = ts_rank(correlation(rank(self.inputs.high), rank(self.inputs.adv(15)), 8.91644), 13.9333)
        right = rank(delta((self.inputs.close * 0.518371) + (self.inputs.low * (1.0 - 0.518371)), 1.06157))
        expected = (-1.0 * (left < right).astype(float)).where(left.notna() & right.notna())
        self.assertFrameEqual(alpha068(self.inputs), expected)

    def test_alpha085_formula(self) -> None:
        left = rank(
            correlation(
                (self.inputs.high * 0.876703) + (self.inputs.close * (1.0 - 0.876703)),
                self.inputs.adv(30),
                9.61331,
            )
        )
        right = rank(
            correlation(
                ts_rank((self.inputs.high + self.inputs.low) / 2.0, 3.70596),
                ts_rank(self.inputs.volume, 10.1595),
                7.11408,
            )
        )
        expected = pd.DataFrame(
            np.power(left.to_numpy(dtype=float), right.to_numpy(dtype=float)),
            index=left.index,
            columns=left.columns,
        )
        self.assertFrameEqual(alpha085(self.inputs), expected)

    def test_alpha088_formula(self) -> None:
        first = rank(
            decay_linear(
                (rank(self.inputs.open) + rank(self.inputs.low)) - (rank(self.inputs.high) + rank(self.inputs.close)),
                8.06882,
            )
        )
        second = ts_rank(
            decay_linear(
                correlation(ts_rank(self.inputs.close, 8.44728), ts_rank(self.inputs.adv(60), 20.6966), 8.01266),
                6.65053,
            ),
            2.61957,
        )
        self.assertFrameEqual(alpha088(self.inputs), _elementwise_min(first, second))

    def test_alpha092_formula(self) -> None:
        left_side = ((self.inputs.high + self.inputs.low) / 2.0) + self.inputs.close
        right_side = self.inputs.low + self.inputs.open
        boolean_signal = (left_side < right_side).astype(float).where(left_side.notna() & right_side.notna())
        first = ts_rank(decay_linear(boolean_signal, 14.7221), 18.8683)
        second = ts_rank(
            decay_linear(correlation(rank(self.inputs.low), rank(self.inputs.adv(30)), 7.58555), 6.94024),
            6.80584,
        )
        self.assertFrameEqual(alpha092(self.inputs), _elementwise_min(first, second))

    def test_alpha095_formula(self) -> None:
        left = rank(self.inputs.open - ts_min(self.inputs.open, 12.4105))
        corr = correlation(
            ts_sum((self.inputs.high + self.inputs.low) / 2.0, 19.1351),
            ts_sum(self.inputs.adv(40), 19.1351),
            12.8742,
        )
        right = ts_rank(np.power(rank(corr), 5.0), 11.7584)
        expected = (left < right).astype(float).where(left.notna() & right.notna())
        self.assertFrameEqual(alpha095(self.inputs), expected)

    def test_alpha099_formula(self) -> None:
        left = rank(
            correlation(
                ts_sum((self.inputs.high + self.inputs.low) / 2.0, 19.8975),
                ts_sum(self.inputs.adv(60), 19.8975),
                8.8136,
            )
        )
        right = rank(correlation(self.inputs.low, self.inputs.volume, 6.28259))
        expected = (-1.0 * (left < right).astype(float)).where(left.notna() & right.notna())
        self.assertFrameEqual(alpha099(self.inputs), expected)

    def test_alpha101_formula(self) -> None:
        expected = safe_divide(self.inputs.close - self.inputs.open, (self.inputs.high - self.inputs.low) + 0.001)
        self.assertFrameEqual(alpha101(self.inputs), expected)

    def test_compute_feature_matrices_preserves_calendar_and_excludes_benchmark(self) -> None:
        matrices = compute_alpha101_feature_matrices(self.inputs)

        self.assertEqual(tuple(matrices), IMPLEMENTED_ALPHA_IDS)
        for matrix in matrices.values():
            self.assertEqual(matrix.shape, self.inputs.close.shape)
            self.assertEqual(list(matrix.columns), ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"])

    def test_second_batch_has_non_null_values_after_warmup(self) -> None:
        matrices = compute_alpha101_feature_matrices(self.inputs, alpha_ids=SECOND_BATCH_ALPHA_IDS)

        for alpha_id, matrix in matrices.items():
            self.assertTrue(matrix.notna().any().any(), msg=f"alpha_{alpha_id} should not be all NaN")

    def test_third_batch_has_non_null_values_after_warmup(self) -> None:
        matrices = compute_alpha101_feature_matrices(self.inputs, alpha_ids=THIRD_BATCH_ALPHA_IDS)

        for alpha_id, matrix in matrices.items():
            self.assertTrue(matrix.notna().any().any(), msg=f"alpha_{alpha_id} should not be all NaN")

    def test_fourth_batch_has_non_null_values_after_warmup(self) -> None:
        matrices = compute_alpha101_feature_matrices(self.inputs, alpha_ids=FOURTH_BATCH_ALPHA_IDS)

        for alpha_id, matrix in matrices.items():
            self.assertTrue(matrix.notna().any().any(), msg=f"alpha_{alpha_id} should not be all NaN")

    def test_tier_b_has_non_null_values_after_warmup(self) -> None:
        matrices = compute_alpha101_feature_matrices(self.inputs, alpha_ids=TIER_B_ALPHA_IDS)

        for alpha_id, matrix in matrices.items():
            self.assertTrue(matrix.notna().any().any(), msg=f"alpha_{alpha_id} should not be all NaN")

    def test_stack_feature_matrices_returns_one_row_per_date_symbol(self) -> None:
        matrices = compute_alpha101_feature_matrices(self.inputs, alpha_ids=("033", "101"))
        stacked = stack_alpha101_feature_matrices(matrices)

        self.assertEqual(len(stacked), len(self.inputs.close.index) * len(self.inputs.close.columns))
        self.assertEqual(set(stacked.columns), {"date", "symbol", "alpha_033", "alpha_101"})
        self.assertNotIn("SPY", set(stacked["symbol"]))


if __name__ == "__main__":
    unittest.main()
