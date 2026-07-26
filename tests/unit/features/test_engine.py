from __future__ import annotations

import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv

from quant_platform.core.exceptions import (
    FeatureComputationError,
    LabelLeakageError,
    PointInTimeViolationError,
    TimezoneError,
)
from quant_platform.core.types import Timeframe
from quant_platform.features.engine import FeatureEngine
from quant_platform.features.interfaces import FeatureDefinition
from quant_platform.features.models import FeatureCategory, FeatureSpec
from quant_platform.features.registry import FeatureRegistry


def _engine_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.rename(columns={"tick_volume": "volume"})[["open_time", "open", "high", "low", "close", "volume"]]


def _spec(name: str, **overrides) -> FeatureSpec:
    base = {
        "name": name, "version": "1", "description": "d", "category": FeatureCategory.PRICE, "required_inputs": ("close",),
        "source_symbols": (), "source_timeframe": Timeframe.M1, "output_dtype": "float64", "lookback_bars": 0, "warmup_bars": 0,
    }
    base.update(overrides)
    return FeatureSpec(**base)


class TestLabelLeakageGuard:
    def test_rejects_label_prefixed_column(self) -> None:
        registry = FeatureRegistry()
        registry.register(FeatureDefinition(spec=_spec("f"), compute=lambda ctx: ctx.base_df["close"]))
        base_df = _engine_df(make_synthetic_ohlcv(50, seed=1))
        base_df["label_future_return"] = 0.0
        engine = FeatureEngine(registry)
        with pytest.raises(LabelLeakageError):
            engine.compute(base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("f",))

    def test_rejects_target_prefixed_column(self) -> None:
        registry = FeatureRegistry()
        registry.register(FeatureDefinition(spec=_spec("f"), compute=lambda ctx: ctx.base_df["close"]))
        base_df = _engine_df(make_synthetic_ohlcv(50, seed=1))
        base_df["target_direction"] = 0.0
        engine = FeatureEngine(registry)
        with pytest.raises(LabelLeakageError):
            engine.compute(base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("f",))

    def test_ordinary_extra_columns_are_fine(self) -> None:
        registry = FeatureRegistry()
        registry.register(FeatureDefinition(spec=_spec("f"), compute=lambda ctx: ctx.base_df["close"]))
        base_df = _engine_df(make_synthetic_ohlcv(50, seed=1))
        base_df["spread"] = 1.0
        engine = FeatureEngine(registry)
        result = engine.compute(base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("f",))
        assert len(result.features) == 50


class TestInputValidation:
    def test_naive_open_time_rejected(self) -> None:
        """Adversarial self-audit (Section 20 'timezone mismatches'): a
        naive base_df must never be silently assumed to be UTC -- that is
        exactly the ambiguity class that can desynchronize a cross-
        timeframe clock and leak future data (see `historical.timezones`'s
        module docstring)."""
        registry = FeatureRegistry()
        registry.register(FeatureDefinition(spec=_spec("f"), compute=lambda ctx: ctx.base_df["close"]))
        naive_base_df = pd.DataFrame({
            "open_time": pd.date_range("2024-01-01", periods=10, freq="1min"),  # no tz
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
        })
        engine = FeatureEngine(registry)
        with pytest.raises(TimezoneError):
            engine.compute(base_df=naive_base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("f",))

    def test_unsorted_base_df_rejected(self) -> None:
        registry = FeatureRegistry()
        registry.register(FeatureDefinition(spec=_spec("f"), compute=lambda ctx: ctx.base_df["close"]))
        base_df = _engine_df(make_synthetic_ohlcv(50, seed=1))
        shuffled = base_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
        engine = FeatureEngine(registry)
        with pytest.raises(PointInTimeViolationError):
            engine.compute(base_df=shuffled, symbol="X", timeframe=Timeframe.M1, feature_names=("f",))

    def test_duplicate_open_time_rejected(self) -> None:
        registry = FeatureRegistry()
        registry.register(FeatureDefinition(spec=_spec("f"), compute=lambda ctx: ctx.base_df["close"]))
        base_df = _engine_df(make_synthetic_ohlcv(50, seed=1))
        duplicated = pd.concat([base_df, base_df.iloc[[0]]]).reset_index(drop=True)
        engine = FeatureEngine(registry)
        with pytest.raises(PointInTimeViolationError):
            engine.compute(base_df=duplicated, symbol="X", timeframe=Timeframe.M1, feature_names=("f",))

    def test_missing_open_time_column_rejected(self) -> None:
        registry = FeatureRegistry()
        registry.register(FeatureDefinition(spec=_spec("f"), compute=lambda ctx: ctx.base_df["close"]))
        engine = FeatureEngine(registry)
        with pytest.raises(FeatureComputationError):
            engine.compute(base_df=pd.DataFrame({"close": [1.0]}), symbol="X", timeframe=Timeframe.M1, feature_names=("f",))

    def test_feature_returning_wrong_length_raises(self) -> None:
        registry = FeatureRegistry()
        registry.register(FeatureDefinition(spec=_spec("bad"), compute=lambda _ctx: pd.Series([1.0, 2.0])))
        base_df = _engine_df(make_synthetic_ohlcv(50, seed=1))
        engine = FeatureEngine(registry)
        with pytest.raises(FeatureComputationError):
            engine.compute(base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("bad",))


class TestDependencyExecution:
    def test_dependent_feature_can_read_dependency_output(self) -> None:
        registry = FeatureRegistry()
        registry.register(FeatureDefinition(spec=_spec("base_feat"), compute=lambda ctx: ctx.base_df["close"]))
        registry.register(
            FeatureDefinition(
                spec=_spec("doubled", feature_dependencies=("base_feat",)),
                compute=lambda ctx: ctx.require_feature("base_feat") * 2,
            )
        )
        base_df = _engine_df(make_synthetic_ohlcv(10, seed=1))
        engine = FeatureEngine(registry)
        result = engine.compute(base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("doubled",))
        assert (result.features["doubled"].to_numpy() == base_df["close"].to_numpy() * 2).all()

    def test_only_explicitly_requested_features_appear_in_output(self) -> None:
        """A dependency pulled in purely to satisfy the DAG must not leak
        into the caller-visible output unless it was itself requested."""
        registry = FeatureRegistry()
        registry.register(FeatureDefinition(spec=_spec("base_feat"), compute=lambda ctx: ctx.base_df["close"]))
        registry.register(
            FeatureDefinition(
                spec=_spec("doubled", feature_dependencies=("base_feat",)),
                compute=lambda ctx: ctx.require_feature("base_feat") * 2,
            )
        )
        base_df = _engine_df(make_synthetic_ohlcv(10, seed=1))
        engine = FeatureEngine(registry)
        result = engine.compute(base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("doubled",))
        assert list(result.features.columns) == ["doubled"]


class TestAvailabilityDelay:
    def test_delay_shifts_feature_forward_by_whole_bars(self) -> None:
        import pandas as pd

        registry = FeatureRegistry()
        spec = _spec("delayed", availability_delay=pd.Timedelta(minutes=3))
        registry.register(FeatureDefinition(spec=spec, compute=lambda ctx: ctx.base_df["close"]))
        base_df = _engine_df(make_synthetic_ohlcv(20, seed=1))
        engine = FeatureEngine(registry)
        result = engine.compute(base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("delayed",))
        # First 3 rows must be null (nothing to shift from); row i afterwards equals close[i-3]
        assert result.features["delayed"].iloc[:3].isna().all()
        assert result.features["delayed"].iloc[5] == base_df["close"].iloc[2]


class TestWarmupTracking:
    def test_max_warmup_reflects_largest_lookback(self) -> None:
        registry = FeatureRegistry()
        registry.register(FeatureDefinition(spec=_spec("short", warmup_bars=2), compute=lambda ctx: ctx.base_df["close"]))
        registry.register(FeatureDefinition(spec=_spec("long", warmup_bars=20), compute=lambda ctx: ctx.base_df["close"]))
        base_df = _engine_df(make_synthetic_ohlcv(50, seed=1))
        engine = FeatureEngine(registry)
        result = engine.compute(base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("short", "long"))
        assert result.warmup_bars == 20
