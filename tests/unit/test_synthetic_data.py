"""Tests for the seeded synthetic OHLCV generator."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas.testing as pdt
import pytest

from quant_platform.core.types import OHLCV_COLUMNS, Timeframe
from quant_platform.data.synthetic import SyntheticDataConfig, generate_ohlcv
from quant_platform.data.validation import validate_ohlcv

UTC = timezone.utc


def _default_config(**overrides: object) -> SyntheticDataConfig:
    defaults: dict[str, object] = {
        "start": datetime(2024, 1, 1, tzinfo=UTC),
        "periods": 500,
        "timeframe": Timeframe.M15,
        "seed": 42,
    }
    defaults.update(overrides)
    return SyntheticDataConfig(**defaults)  # type: ignore[arg-type]


class TestDeterminism:
    def test_same_seed_produces_identical_output(self) -> None:
        config = _default_config(seed=123)
        df1 = generate_ohlcv(config)
        df2 = generate_ohlcv(config)
        pdt.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_output(self) -> None:
        df1 = generate_ohlcv(_default_config(seed=1))
        df2 = generate_ohlcv(_default_config(seed=2))
        assert not df1["close"].equals(df2["close"])


class TestShapeAndSchema:
    def test_row_count_matches_periods(self) -> None:
        df = generate_ohlcv(_default_config(periods=250))
        assert len(df) == 250

    def test_has_canonical_schema(self) -> None:
        df = generate_ohlcv(_default_config())
        assert list(df.columns) == list(OHLCV_COLUMNS)

    def test_open_times_are_correctly_spaced_and_utc(self) -> None:
        df = generate_ohlcv(_default_config(periods=10, timeframe=Timeframe.M15))
        deltas = df["open_time"].diff().dropna().unique()
        assert len(deltas) == 1
        assert deltas[0] == Timeframe.M15.duration
        assert str(df["open_time"].dt.tz) == "UTC"

    @pytest.mark.parametrize("timeframe", list(Timeframe))
    def test_all_timeframes_produce_correctly_spaced_bars(self, timeframe: Timeframe) -> None:
        df = generate_ohlcv(_default_config(periods=20, timeframe=timeframe))
        deltas = df["open_time"].diff().dropna().unique()
        assert len(deltas) == 1
        assert deltas[0] == timeframe.duration


class TestOHLCInvariants:
    def test_generated_data_passes_quality_validation(self) -> None:
        df = generate_ohlcv(_default_config(periods=1000, annualized_volatility=0.6))
        report = validate_ohlcv(df, symbol="SYNTH", timeframe=Timeframe.M15)
        assert report.is_valid, report.summary()

    def test_high_is_always_the_max_and_low_the_min(self) -> None:
        df = generate_ohlcv(_default_config(periods=500, annualized_volatility=0.8, seed=7))
        assert (df["high"] >= df[["open", "close", "low"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close", "high"]].min(axis=1)).all()

    def test_prices_are_always_positive_even_under_high_volatility(self) -> None:
        df = generate_ohlcv(_default_config(periods=2000, annualized_volatility=3.0, seed=99))
        assert (df[["open", "high", "low", "close"]] > 0).all().all()


class TestConfigValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("periods", 0),
            ("periods", -5),
            ("initial_price", 0.0),
            ("initial_price", -10.0),
            ("annualized_volatility", -0.1),
            ("sub_steps", 0),
            ("base_volume", 0.0),
        ],
    )
    def test_rejects_invalid_field(self, field: str, value: object) -> None:
        with pytest.raises(ValueError):
            _default_config(**{field: value})
