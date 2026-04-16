from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from sp500_relative_alpha.alpha101_ops import (
    Alpha101OperatorError,
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
    ts_argmin,
    ts_max,
    ts_mean,
    ts_min,
    ts_product,
    ts_rank,
    ts_stddev,
    ts_sum,
)


def _wide_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [4.0, 3.0, 2.0, 1.0],
            "C": [1.0, 1.0, 2.0, 2.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )


def _daily_bars() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=4)
    rows: list[dict[str, object]] = []
    for symbol, opens in {
        "AAPL": [10.0, 11.0, 12.0, 13.0],
        "MSFT": [20.0, 21.0, 22.0, 23.0],
        "SPY": [100.0, 101.0, 102.0, 103.0],
    }.items():
        for date, open_price in zip(dates, opens, strict=True):
            high = open_price + 1.0
            low = open_price - 1.0
            close = open_price + 0.5
            shares_volume = 1000.0
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
    return pd.DataFrame(rows)


class Alpha101OperatorTests(unittest.TestCase):
    def test_rank_is_cross_sectional_percentile_rank(self) -> None:
        ranked = rank(_wide_frame())

        first_day = ranked.loc[pd.Timestamp("2024-01-02")]
        self.assertAlmostEqual(first_day["A"], 0.5)
        self.assertAlmostEqual(first_day["B"], 1.0)
        self.assertAlmostEqual(first_day["C"], 0.5)

    def test_delay_and_delta_are_time_series_by_symbol(self) -> None:
        x = _wide_frame()[["A"]]

        delayed = delay(x, 2)
        differenced = delta(x, 2)

        self.assertTrue(pd.isna(delayed["A"].iloc[0]))
        self.assertEqual(delayed["A"].iloc[2], 1.0)
        self.assertEqual(differenced["A"].iloc[2], 2.0)

    def test_basic_rolling_operators_use_full_windows(self) -> None:
        x = _wide_frame()[["A"]]

        self.assertTrue(pd.isna(ts_sum(x, 3)["A"].iloc[1]))
        self.assertEqual(ts_sum(x, 3)["A"].iloc[2], 6.0)
        self.assertEqual(ts_mean(x, 3)["A"].iloc[2], 2.0)
        self.assertEqual(ts_product(x, 3)["A"].iloc[2], 6.0)
        self.assertAlmostEqual(ts_stddev(x, 3)["A"].iloc[2], np.std([1.0, 2.0, 3.0]))
        self.assertEqual(ts_min(x, 3)["A"].iloc[2], 1.0)
        self.assertEqual(ts_max(x, 3)["A"].iloc[2], 3.0)

    def test_time_series_rank_and_arg_operators_are_window_local(self) -> None:
        x = pd.DataFrame(
            {"A": [3.0, 1.0, 2.0, 4.0]},
            index=pd.bdate_range("2024-01-02", periods=4),
        )

        self.assertAlmostEqual(ts_rank(x, 3)["A"].iloc[2], 2.0 / 3.0)
        self.assertEqual(ts_argmax(x, 3)["A"].iloc[2], 1.0)
        self.assertEqual(ts_argmin(x, 3)["A"].iloc[2], 2.0)

    def test_correlation_and_covariance_are_columnwise(self) -> None:
        x = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [1.0, 2.0, 4.0]})
        y = pd.DataFrame({"A": [2.0, 4.0, 6.0], "B": [4.0, 2.0, 1.0]})

        corr = correlation(x, y, 3)
        cov = covariance(x, y, 3)

        self.assertAlmostEqual(corr["A"].iloc[-1], 1.0)
        self.assertLess(corr["B"].iloc[-1], 0.0)
        self.assertAlmostEqual(cov["A"].iloc[-1], np.cov([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], bias=True)[0, 1])

    def test_decay_linear_weights_recent_values_more(self) -> None:
        x = pd.DataFrame({"A": [1.0, 2.0, 10.0]})

        decayed = decay_linear(x, 3)

        expected = (1.0 * 1.0 + 2.0 * 2.0 + 10.0 * 3.0) / 6.0
        self.assertAlmostEqual(decayed["A"].iloc[-1], expected)

    def test_scale_signedpower_and_safe_divide(self) -> None:
        x = pd.DataFrame({"A": [-2.0, 0.0], "B": [1.0, 0.0]})

        scaled = scale(x)
        powered = signedpower(x, 2)
        divided = safe_divide(pd.DataFrame({"A": [1.0]}), pd.DataFrame({"A": [0.0]}))

        self.assertAlmostEqual(scaled["A"].iloc[0], -2.0 / 3.0)
        self.assertAlmostEqual(scaled["B"].iloc[0], 1.0 / 3.0)
        self.assertTrue(pd.isna(scaled["A"].iloc[1]))
        self.assertEqual(powered["A"].iloc[0], -4.0)
        self.assertTrue(pd.isna(divided["A"].iloc[0]))

    def test_build_alpha101_input_matrices_excludes_benchmark_and_uses_alpha_volume(self) -> None:
        inputs = build_alpha101_input_matrices(_daily_bars())

        self.assertEqual(list(inputs.open.columns), ["AAPL", "MSFT"])
        self.assertNotIn("SPY", set(inputs.open.columns))
        first_aapl_volume = inputs.volume.loc[pd.Timestamp("2024-01-02"), "AAPL"]
        expected_volume = ((11.0 + 9.0 + 10.5) / 3.0) * 1000.0
        second_expected_volume = ((12.0 + 10.0 + 11.5) / 3.0) * 1000.0
        self.assertAlmostEqual(first_aapl_volume, expected_volume)
        self.assertAlmostEqual(
            inputs.adv(2).loc[pd.Timestamp("2024-01-03"), "AAPL"],
            (expected_volume + second_expected_volume) / 2.0,
        )

    def test_build_alpha101_input_matrices_requires_benchmark_calendar(self) -> None:
        bars = _daily_bars().loc[lambda df: df["symbol"] != "SPY"]

        with self.assertRaises(Alpha101OperatorError):
            build_alpha101_input_matrices(bars)


if __name__ == "__main__":
    unittest.main()
