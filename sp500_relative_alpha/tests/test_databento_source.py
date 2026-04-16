from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sp500_relative_alpha.databento_source import (
    build_security_master_from_symbols,
    fetch_databento_ohlcv_1m_contract,
    load_databento_key,
)


class _FakeStore:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_df(self) -> pd.DataFrame:
        return self._df.copy()


class _FakeTimeseries:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.calls: list[dict[str, object]] = []

    def get_range(self, **kwargs: object) -> _FakeStore:
        self.calls.append(kwargs)
        return _FakeStore(self._df)


class _FakeHistorical:
    def __init__(self, df: pd.DataFrame) -> None:
        self.timeseries = _FakeTimeseries(df)


class DatabentoSourceTests(unittest.TestCase):
    def test_load_databento_key_reads_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("DATABENTO_API_KEY=test-key\n", encoding="utf-8")
            self.assertEqual(load_databento_key(env_path), "test-key")

    def test_security_master_builder_marks_benchmark(self) -> None:
        master = build_security_master_from_symbols(["AAPL", "XOM"], benchmark_symbol="SPY")
        self.assertEqual(len(master), 3)
        spy_role = master.loc[master["symbol"] == "SPY", "symbol_role"].iloc[0]
        self.assertEqual(spy_role, "benchmark")

    def test_fetch_maps_databento_frame_to_raw_contract(self) -> None:
        index = pd.to_datetime(
            ["2024-01-02T14:30:00Z", "2024-01-02T14:31:00Z"],
            utc=True,
        )
        index.name = "ts_event"
        frame = pd.DataFrame(
            {
                "open": [100.0, 200.0],
                "high": [101.0, 201.0],
                "low": [99.0, 199.0],
                "close": [100.5, 200.5],
                "volume": [10, 20],
                "symbol": ["AAPL", "SPY"],
            },
            index=index,
        )
        client = _FakeHistorical(frame)

        result = fetch_databento_ohlcv_1m_contract(
            symbols=["AAPL", "SPY"],
            start="2024-01-02",
            end="2024-01-03",
            client=client,  # type: ignore[arg-type]
        )

        self.assertEqual(
            list(result.columns),
            [
                "security_id",
                "symbol",
                "raw_ts_source",
                "source_tz",
                "source_ts_convention",
                "open_raw",
                "high_raw",
                "low_raw",
                "close_raw",
                "volume_raw",
                "vendor_note",
            ],
        )
        self.assertEqual(result["security_id"].tolist(), ["raw_symbol::AAPL", "raw_symbol::SPY"])
        self.assertTrue((result["source_ts_convention"] == "bar_start").all())
        self.assertTrue((result["source_tz"] == "UTC").all())
        self.assertEqual(client.timeseries.calls[0]["dataset"], "EQUS.MINI")
        self.assertEqual(client.timeseries.calls[0]["schema"], "ohlcv-1m")


if __name__ == "__main__":
    unittest.main()
