"""Tests for `historical.models`: the canonical raw-historical-bar schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.exceptions import SchemaError, TimezoneError
from quant_platform.historical.models import (
    RAW_HISTORICAL_COLUMNS,
    coerce_historical_dtypes,
    schema_fingerprint,
    spread_points_to_price,
    validate_historical_schema,
)


def _raw_frame(n: int = 3) -> pd.DataFrame:
    open_time = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": np.full(n, 2000.12, dtype=np.float64),
            "high": np.full(n, 2001.34, dtype=np.float64),
            "low": np.full(n, 1999.56, dtype=np.float64),
            "close": np.full(n, 2000.78, dtype=np.float64),
            "tick_volume": np.full(n, 120, dtype=np.int64),
            "real_volume": np.zeros(n, dtype=np.int64),
            "spread": np.full(n, 18, dtype=np.int64),
        }
    )


class TestValidateHistoricalSchema:
    def test_accepts_well_formed_frame(self) -> None:
        validate_historical_schema(_raw_frame(), context="test")  # must not raise

    def test_rejects_missing_column(self) -> None:
        df = _raw_frame().drop(columns=["spread"])
        with pytest.raises(SchemaError, match="missing required historical columns"):
            validate_historical_schema(df, context="test")

    def test_rejects_naive_open_time(self) -> None:
        df = _raw_frame()
        df["open_time"] = df["open_time"].dt.tz_localize(None)
        with pytest.raises(TimezoneError, match="timezone-naive"):
            validate_historical_schema(df, context="test")

    def test_rejects_wrong_float_dtype(self) -> None:
        df = _raw_frame()
        df["open"] = df["open"].astype(np.int64)
        with pytest.raises(SchemaError, match="must be float64"):
            validate_historical_schema(df, context="test")

    def test_rejects_wrong_int_dtype(self) -> None:
        df = _raw_frame()
        df["tick_volume"] = df["tick_volume"].astype(np.float64)
        with pytest.raises(SchemaError, match="must be an integer dtype"):
            validate_historical_schema(df, context="test")

    def test_accepts_empty_frame_without_touching_dtypes(self) -> None:
        df = _raw_frame().iloc[0:0]
        validate_historical_schema(df, context="test")  # must not raise


class TestCoerceHistoricalDtypes:
    def test_casts_unsigned_and_narrow_int_types_to_int64(self) -> None:
        df = _raw_frame()
        df["tick_volume"] = df["tick_volume"].astype(np.uint64)
        df["spread"] = df["spread"].astype(np.int32)
        out = coerce_historical_dtypes(df)
        assert out["tick_volume"].dtype == np.int64
        assert out["spread"].dtype == np.int64

    def test_orders_columns_per_schema(self) -> None:
        df = _raw_frame()[list(reversed(RAW_HISTORICAL_COLUMNS))]
        out = coerce_historical_dtypes(df)
        assert list(out.columns) == list(RAW_HISTORICAL_COLUMNS)

    def test_normalizes_open_time_to_a_single_canonical_resolution(self) -> None:
        # Regression test for a real bug found via the end-to-end pipeline
        # integration test: `pd.to_datetime(..., unit="s")` (what the MT5
        # adapter uses on integer-second epoch values) produces
        # `datetime64[s, UTC]`, and writing that to Parquet then reading it
        # back silently produced `datetime64[ms, UTC]` instead -- same
        # data, different dtype string, which made
        # `historical.models.schema_fingerprint` falsely report schema
        # drift on every single raw-snapshot round trip. Every column must
        # come out at the one canonical resolution regardless of its
        # input resolution.
        df = _raw_frame()
        df["open_time"] = df["open_time"].astype("datetime64[s, UTC]")
        out = coerce_historical_dtypes(df)
        assert out["open_time"].dtype == pd.DatetimeTZDtype(unit="ns", tz="UTC")


class TestSchemaFingerprintResolutionIndependence:
    def test_fingerprint_is_identical_across_different_datetime_storage_resolutions(self) -> None:
        # Even WITHOUT `coerce_historical_dtypes` normalizing first,
        # `schema_fingerprint` itself must not be fooled by a
        # representation-only difference -- defense in depth for the same
        # bug covered above.
        df_ns = _raw_frame()
        df_ms = df_ns.copy()
        df_ms["open_time"] = df_ms["open_time"].astype("datetime64[ms, UTC]")
        assert df_ns["open_time"].dtype != df_ms["open_time"].dtype  # premise: genuinely different dtypes
        assert schema_fingerprint(df_ns) == schema_fingerprint(df_ms)

    def test_fingerprint_still_differs_for_a_genuinely_different_timezone(self) -> None:
        df_utc = _raw_frame()
        df_other = df_utc.copy()
        df_other["open_time"] = df_other["open_time"].dt.tz_convert("America/New_York")
        assert schema_fingerprint(df_utc) != schema_fingerprint(df_other)

    def test_fingerprint_still_differs_for_naive_vs_aware(self) -> None:
        df_aware = _raw_frame()
        df_naive = df_aware.copy()
        df_naive["open_time"] = df_naive["open_time"].dt.tz_localize(None)
        assert schema_fingerprint(df_aware) != schema_fingerprint(df_naive)


class TestSpreadPointsToPrice:
    def test_hand_computed_conversion(self) -> None:
        # 18 points at a 0.01 point size is a 0.18 price-unit spread -- hand-computed.
        assert spread_points_to_price(18, point_size=0.01) == pytest.approx(0.18)

    def test_rejects_non_positive_point_size(self) -> None:
        with pytest.raises(ValueError, match="point_size must be positive"):
            spread_points_to_price(18, point_size=0.0)

    def test_series_conversion(self) -> None:
        out = spread_points_to_price(pd.Series([10, 20, 30]), point_size=0.01)
        assert list(out) == pytest.approx([0.1, 0.2, 0.3])


class TestFloatPrecision:
    """Justifies the `historical.models` module docstring's claim that
    float64 loses no precision for realistic XAUUSD price/decimal scales,
    including a full Parquet round-trip (not just an in-memory check)."""

    def test_realistic_prices_round_trip_exactly_through_parquet(self, tmp_path) -> None:
        prices = [1987.653, 2384.021, 2001.10, 1999.999, 2500.005]
        df = pd.DataFrame({"price": pd.Series(prices, dtype=np.float64)})
        path = tmp_path / "roundtrip.parquet"
        df.to_parquet(path)
        reloaded = pd.read_parquet(path)
        assert list(reloaded["price"]) == prices

    def test_arithmetic_on_realistic_prices_matches_hand_computed_value(self) -> None:
        # 2384.021 - 1987.653 = 396.368, computed by hand.
        a = np.float64(2384.021)
        b = np.float64(1987.653)
        assert float(a - b) == pytest.approx(396.368, abs=1e-9)
