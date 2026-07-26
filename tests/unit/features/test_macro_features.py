from __future__ import annotations

import pandas as pd
from tests.unit.features.conftest import make_synthetic_ohlcv

from quant_platform.core.types import Timeframe
from quant_platform.features.engine import FeatureEngine
from quant_platform.features.macro.macro_features import MacroSourceConfig, register_macro_features
from quant_platform.features.registry import FeatureRegistry


def _engine_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.rename(columns={"tick_volume": "volume"})[["open_time", "open", "high", "low", "close", "volume"]]


class TestMacroReleaseTiming:
    def test_january_value_not_visible_before_february_release(self) -> None:
        """The Section 6 required proof, exercised through the full
        registered-feature path (not just `as_of_join_external` directly)."""
        registry = FeatureRegistry()
        register_macro_features(registry, base_timeframe=Timeframe.M1, config=MacroSourceConfig(source_name="cpi"))
        macro_df = pd.DataFrame(
            {
                "observation_time": [pd.Timestamp("2024-01-01T00:00:00Z")],
                "release_time": [pd.Timestamp("2024-02-10T00:00:00Z")],
                "value": [3.1],
            }
        )
        base_df = _engine_df(make_synthetic_ohlcv(3, freq_minutes=1440, start="2024-01-15T00:00:00", seed=1))
        engine = FeatureEngine(registry)
        result = engine.compute(
            base_df=base_df, symbol="X", timeframe=Timeframe.D1, feature_names=("macro_cpi_level", "macro_cpi_is_stale"),
            macro_data={"cpi": macro_df},
        )
        # every base row is in January, strictly before the February release
        assert result.features["macro_cpi_level"].isna().all()
        assert result.features["macro_cpi_is_stale"].all()

    def test_value_visible_the_instant_after_release(self) -> None:
        registry = FeatureRegistry()
        register_macro_features(registry, base_timeframe=Timeframe.D1, config=MacroSourceConfig(source_name="cpi"))
        macro_df = pd.DataFrame(
            {
                "observation_time": [pd.Timestamp("2024-01-01T00:00:00Z")],
                "release_time": [pd.Timestamp("2024-01-16T00:00:00Z")],
                "value": [3.1],
            }
        )
        base_df = _engine_df(make_synthetic_ohlcv(3, freq_minutes=1440, start="2024-01-16T00:00:00", seed=1))
        engine = FeatureEngine(registry)
        result = engine.compute(
            base_df=base_df, symbol="X", timeframe=Timeframe.D1, feature_names=("macro_cpi_level",),
            macro_data={"cpi": macro_df},
        )
        assert result.features["macro_cpi_level"].iloc[0] == 3.1

    def test_change_reflects_prior_release_not_future_one(self) -> None:
        registry = FeatureRegistry()
        register_macro_features(
            registry, base_timeframe=Timeframe.D1, config=MacroSourceConfig(source_name="rate", change_lookback=1)
        )
        macro_df = pd.DataFrame(
            {
                "observation_time": pd.date_range("2024-01-01", periods=3, freq="MS", tz="UTC"),
                "release_time": pd.date_range("2024-01-05", periods=3, freq="MS", tz="UTC"),
                "value": [5.0, 5.25, 5.5],
            }
        )
        base_df = _engine_df(
            make_synthetic_ohlcv(1, freq_minutes=1440, start=str(pd.Timestamp("2024-03-10")), seed=1)
        )
        engine = FeatureEngine(registry)
        result = engine.compute(
            base_df=base_df, symbol="X", timeframe=Timeframe.D1, feature_names=("macro_rate_change_1",),
            macro_data={"rate": macro_df},
        )
        # As of March 10, only the Jan/Feb/Mar releases (Jan5/Feb5/Mar5) are visible;
        # the most recent is March's release (value 5.5), one release before it is Feb (5.25)
        assert result.features["macro_rate_change_1"].iloc[0] == 5.5 - 5.25

    def test_release_age_grows_with_time(self) -> None:
        registry = FeatureRegistry()
        register_macro_features(registry, base_timeframe=Timeframe.D1, config=MacroSourceConfig(source_name="cpi"))
        macro_df = pd.DataFrame(
            {"observation_time": [pd.Timestamp("2024-01-01T00:00:00Z")], "release_time": [pd.Timestamp("2024-01-01T00:00:00Z")], "value": [1.0]}
        )
        base_df = _engine_df(make_synthetic_ohlcv(5, freq_minutes=1440, start="2024-01-02T00:00:00", seed=1))
        engine = FeatureEngine(registry)
        result = engine.compute(
            base_df=base_df, symbol="X", timeframe=Timeframe.D1, feature_names=("macro_cpi_release_age_days",),
            macro_data={"cpi": macro_df},
        )
        age = result.features["macro_cpi_release_age_days"]
        assert age.is_monotonic_increasing

    def test_no_macro_data_at_all_marks_every_row_stale(self) -> None:
        registry = FeatureRegistry()
        register_macro_features(registry, base_timeframe=Timeframe.D1, config=MacroSourceConfig(source_name="cpi"))
        empty_macro_df = pd.DataFrame(
            {
                "observation_time": pd.Series([], dtype="datetime64[ns, UTC]"),
                "release_time": pd.Series([], dtype="datetime64[ns, UTC]"),
                "value": pd.Series([], dtype="float64"),
            }
        )
        base_df = _engine_df(make_synthetic_ohlcv(3, freq_minutes=1440, seed=1))
        engine = FeatureEngine(registry)
        result = engine.compute(
            base_df=base_df, symbol="X", timeframe=Timeframe.D1, feature_names=("macro_cpi_is_stale",),
            macro_data={"cpi": empty_macro_df},
        )
        assert result.features["macro_cpi_is_stale"].all()
