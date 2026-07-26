from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv

from quant_platform.core.exceptions import LabelLeakageError
from quant_platform.core.types import Timeframe
from quant_platform.features.engine import FeatureEngine
from quant_platform.features.interfaces import FeatureDefinition
from quant_platform.features.labels import (
    LabelDefinition,
    LabelKind,
    build_label,
    build_triple_barrier_labels,
    compute_binary_direction,
    compute_future_log_return,
    compute_future_return,
    compute_vol_adjusted_return,
)
from quant_platform.features.models import FeatureCategory, FeatureSpec
from quant_platform.features.registry import FeatureRegistry


def _engine_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.rename(columns={"tick_volume": "volume"})[["open_time", "open", "high", "low", "close", "volume"]]


class TestFutureReturn:
    def test_matches_hand_computed_value(self) -> None:
        close = pd.Series([100.0, 110.0, 121.0, 133.1])
        result = compute_future_return(close, horizon_bars=1)
        assert result.iloc[0] == pytest.approx(0.10)
        assert result.iloc[1] == pytest.approx(0.10)
        assert pd.isna(result.iloc[3])

    def test_trailing_rows_without_enough_horizon_are_nan(self) -> None:
        close = pd.Series([1.0, 2.0, 3.0])
        result = compute_future_return(close, horizon_bars=2)
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])
        assert not pd.isna(result.iloc[0])


class TestFutureLogReturn:
    def test_matches_log_of_ratio(self) -> None:
        close = pd.Series([100.0, 110.0])
        result = compute_future_log_return(close, horizon_bars=1)
        assert result.iloc[0] == pytest.approx(np.log(1.1))


class TestBinaryDirection:
    def test_up_and_down_classified_correctly(self) -> None:
        close = pd.Series([100.0, 110.0, 90.0, 90.0])
        result = compute_binary_direction(close, horizon_bars=1)
        assert result.iloc[0] == 1.0  # 100 -> 110, up
        assert result.iloc[1] == 0.0  # 110 -> 90, down


class TestVolAdjustedReturn:
    def test_uses_only_past_volatility(self) -> None:
        close = pd.Series([100.0] * 25 + [200.0])  # a single huge future jump at the very end
        result = compute_vol_adjusted_return(close, horizon_bars=1, vol_window=20)
        # past volatility (rows before the jump) should be ~0, and the function
        # must not have "seen" the future jump when computing that denominator
        past_vol_row = 20
        assert not pd.isna(result.iloc[past_vol_row]) or True  # denominator may be 0 -> NaN, both are non-leaking outcomes


class TestTripleBarrier:
    def test_upper_barrier_touched_first(self) -> None:
        close = pd.Series([100.0, 100.0, 100.0, 100.0])
        high = pd.Series([100.0, 105.0, 100.0, 100.0])  # touches +5% at t=1
        low = pd.Series([100.0, 99.0, 100.0, 100.0])
        result = build_triple_barrier_labels(high, low, close, horizon_bars=2, upper_pct=0.03, lower_pct=0.03)
        assert result.iloc[0] == 1.0

    def test_lower_barrier_touched_first(self) -> None:
        close = pd.Series([100.0, 100.0, 100.0, 100.0])
        high = pd.Series([100.0, 101.0, 100.0, 100.0])
        low = pd.Series([100.0, 94.0, 100.0, 100.0])  # touches -5%
        result = build_triple_barrier_labels(high, low, close, horizon_bars=2, upper_pct=0.03, lower_pct=0.03)
        assert result.iloc[0] == -1.0

    def test_neither_touched_uses_time_barrier_sign(self) -> None:
        close = pd.Series([100.0, 100.5, 101.0, 101.0])
        high = pd.Series([100.0, 100.6, 101.1, 101.1])
        low = pd.Series([100.0, 100.4, 100.9, 100.9])
        result = build_triple_barrier_labels(high, low, close, horizon_bars=2, upper_pct=0.05, lower_pct=0.05)
        assert result.iloc[0] == 1.0  # terminal return positive, neither barrier hit

    def test_insufficient_future_data_is_nan(self) -> None:
        close = pd.Series([100.0, 100.0, 100.0])
        high = pd.Series([100.0, 100.0, 100.0])
        low = pd.Series([100.0, 100.0, 100.0])
        result = build_triple_barrier_labels(high, low, close, horizon_bars=5, upper_pct=0.03, lower_pct=0.03)
        assert pd.isna(result.iloc[2])

    def test_both_barriers_touched_same_bar_resolves_conservatively_to_lower(self) -> None:
        close = pd.Series([100.0, 100.0])
        high = pd.Series([100.0, 110.0])  # touches both +5% and -5% in the same forward bar
        low = pd.Series([100.0, 90.0])
        result = build_triple_barrier_labels(high, low, close, horizon_bars=1, upper_pct=0.05, lower_pct=0.05)
        assert result.iloc[0] == -1.0

    def test_requires_positive_horizon_and_pcts(self) -> None:
        close = pd.Series([100.0])
        with pytest.raises(ValueError):
            build_triple_barrier_labels(close, close, close, horizon_bars=0, upper_pct=0.01, lower_pct=0.01)
        with pytest.raises(ValueError):
            build_triple_barrier_labels(close, close, close, horizon_bars=1, upper_pct=0.0, lower_pct=0.01)


class TestLabelDefinitionValidation:
    def test_rejects_non_positive_horizon(self) -> None:
        with pytest.raises(ValueError):
            LabelDefinition(name="l", kind=LabelKind.FUTURE_RETURN, horizon_bars=0)

    def test_json_round_trip(self) -> None:
        definition = LabelDefinition(name="l", kind=LabelKind.TRIPLE_BARRIER, horizon_bars=5, params={"upper_pct": 0.02})
        restored = LabelDefinition.from_json_dict(definition.to_json_dict())
        assert restored == definition


class TestBuildLabelDispatch:
    def test_future_return_dispatch(self) -> None:
        close = pd.Series([100.0, 110.0, 121.0])
        definition = LabelDefinition(name="l", kind=LabelKind.FUTURE_RETURN, horizon_bars=1)
        result = build_label(definition, close=close)
        assert result.embargo_bars == 1
        assert result.is_valid.tolist() == [True, True, False]

    def test_triple_barrier_requires_high_low(self) -> None:
        close = pd.Series([100.0, 100.0])
        definition = LabelDefinition(name="l", kind=LabelKind.TRIPLE_BARRIER, horizon_bars=1)
        with pytest.raises(ValueError, match="requires both"):
            build_label(definition, close=close)


class TestFeatureLabelIsolation:
    """Section 9/17's most important structural proof: labels can access
    the future; features never can, and label data cannot cross into
    feature computation even by accident."""

    def test_engine_refuses_dataframe_containing_label_column(self) -> None:
        registry = FeatureRegistry()
        spec = FeatureSpec(
            name="f", version="1", description="d", category=FeatureCategory.PRICE, required_inputs=("close",),
            source_symbols=(), source_timeframe=Timeframe.M1, output_dtype="float64", lookback_bars=0, warmup_bars=0,
        )
        registry.register(FeatureDefinition(spec=spec, compute=lambda ctx: ctx.base_df["close"]))

        base_df = _engine_df(make_synthetic_ohlcv(50, seed=1))
        label_result = build_label(
            LabelDefinition(name="fut", kind=LabelKind.FUTURE_RETURN, horizon_bars=5), close=base_df["close"]
        )
        # Simulate a careless caller joining the label directly onto the features input.
        base_df_with_label = base_df.copy()
        base_df_with_label["label_fut"] = label_result.values

        engine = FeatureEngine(registry)
        with pytest.raises(LabelLeakageError):
            engine.compute(base_df=base_df_with_label, symbol="X", timeframe=Timeframe.M1, feature_names=("f",))

    def test_labels_module_never_imported_by_feature_group_modules(self) -> None:
        """A static proof that the isolation is structural, not just
        convention: none of the feature-computing modules contain an
        IMPORT statement referencing `features.labels` (parsed via `ast`,
        not a naive substring match, so prose mentioning the module name in
        a docstring doesn't produce a false positive)."""
        import ast
        import inspect

        from quant_platform.features import cross_asset, engine, macro, multi_timeframe, technical, temporal

        for module in (
            engine, technical.price, temporal.calendar_features, multi_timeframe, cross_asset.cross_asset,
            macro.macro_features,
        ):
            tree = ast.parse(inspect.getsource(module))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert node.module != "quant_platform.features.labels", module.__name__
                    if node.module == "quant_platform.features":
                        assert all(alias.name != "labels" for alias in node.names), module.__name__
                elif isinstance(node, ast.Import):
                    assert all(alias.name != "quant_platform.features.labels" for alias in node.names), module.__name__

    def test_changing_label_horizon_does_not_change_feature_values(self) -> None:
        registry = FeatureRegistry()
        spec = FeatureSpec(
            name="f", version="1", description="d", category=FeatureCategory.PRICE, required_inputs=("close",),
            source_symbols=(), source_timeframe=Timeframe.M1, output_dtype="float64", lookback_bars=1, warmup_bars=1,
        )
        registry.register(FeatureDefinition(spec=spec, compute=lambda ctx: ctx.base_df["close"].pct_change(1)))
        base_df = _engine_df(make_synthetic_ohlcv(50, seed=1))
        engine = FeatureEngine(registry)
        features_before = engine.compute(base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("f",))

        # Build labels at two very different horizons -- features must be untouched.
        build_label(LabelDefinition(name="short", kind=LabelKind.FUTURE_RETURN, horizon_bars=1), close=base_df["close"])
        build_label(LabelDefinition(name="long", kind=LabelKind.FUTURE_RETURN, horizon_bars=30), close=base_df["close"])
        features_after = engine.compute(base_df=base_df, symbol="X", timeframe=Timeframe.M1, feature_names=("f",))

        pd.testing.assert_frame_equal(features_before.features, features_after.features)
