from __future__ import annotations

import pytest
from pydantic import ValidationError

from quant_platform.config.feature_schemas import (
    LabelConfig,
    MultiTimeframeFeatureConfig,
    PreprocessingConfig,
    ResearchDatasetConfig,
    SplitConfig,
    TechnicalFeatureConfig,
)


def _minimal_config(**overrides) -> dict:
    base = {
        "symbol": "XAUUSD", "base_timeframe": "M1", "start": "2024-01-01T00:00:00Z", "end": "2024-06-01T00:00:00Z",
        "historical_storage_root": "./data", "research_storage_root": "./research_data",
        "label": {"name": "fut", "kind": "future_return", "horizon_bars": 5},
        "split": {"strategy": "chronological", "train_fraction": 0.7, "validation_fraction": 0.15},
    }
    base.update(overrides)
    return base


class TestResearchDatasetConfig:
    def test_minimal_config_validates(self) -> None:
        config = ResearchDatasetConfig.model_validate(_minimal_config())
        assert config.symbol == "XAUUSD"
        assert config.build_base_timeframe().value == "M1"

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            ResearchDatasetConfig.model_validate(_minimal_config(unexpected_field=True))

    def test_rejects_invalid_timeframe(self) -> None:
        with pytest.raises(ValidationError):
            ResearchDatasetConfig.model_validate(_minimal_config(base_timeframe="M2"))


class TestSplitConfig:
    def test_chronological_requires_fractions(self) -> None:
        with pytest.raises(ValidationError, match="train_fraction"):
            SplitConfig(strategy="chronological")

    def test_walk_forward_requires_n_splits_and_test_size(self) -> None:
        with pytest.raises(ValidationError, match="n_splits"):
            SplitConfig(strategy="expanding_walk_forward")

    def test_valid_chronological_builds_params(self) -> None:
        config = SplitConfig(strategy="chronological", train_fraction=0.7, validation_fraction=0.15, purge_bars=5)
        params = config.build_params()
        assert params["train_fraction"] == 0.7
        assert params["purge_bars"] == 5

    def test_valid_rolling_walk_forward_includes_max_train_size(self) -> None:
        config = SplitConfig(strategy="rolling_walk_forward", n_splits=3, test_size=100, max_train_size=500)
        params = config.build_params()
        assert params["max_train_size"] == 500


class TestLabelConfig:
    def test_builds_label_definition(self) -> None:
        config = LabelConfig(name="fut", kind="triple_barrier", horizon_bars=10, params={"upper_pct": 0.02})
        definition = config.build()
        assert definition.horizon_bars == 10
        assert definition.params["upper_pct"] == 0.02

    def test_rejects_non_positive_horizon(self) -> None:
        with pytest.raises(ValidationError):
            LabelConfig(name="fut", kind="future_return", horizon_bars=0)


class TestTechnicalFeatureConfig:
    def test_defaults_build_windows(self) -> None:
        windows = TechnicalFeatureConfig().build()
        assert windows.return_windows == (1, 5, 15)

    def test_rejects_non_positive_window(self) -> None:
        with pytest.raises(ValidationError):
            TechnicalFeatureConfig(volatility_window=1)


class TestPreprocessingConfig:
    def test_builds_transform_kind_mapping(self) -> None:
        config = PreprocessingConfig(transforms={"close": "standard_scale"})
        built = config.build()
        assert built["close"].value == "standard_scale"

    def test_rejects_unknown_transform_kind(self) -> None:
        with pytest.raises(ValidationError):
            PreprocessingConfig(transforms={"close": "not_a_real_transform"})


class TestMultiTimeframeFeatureConfig:
    def test_requires_at_least_one_higher_timeframe(self) -> None:
        with pytest.raises(ValidationError):
            MultiTimeframeFeatureConfig(higher_timeframes=[])

    def test_builds_timeframes(self) -> None:
        config = MultiTimeframeFeatureConfig(higher_timeframes=["H1", "H4"])
        timeframes = config.build_timeframes()
        assert [t.value for t in timeframes] == ["H1", "H4"]


class TestExampleConfigFile:
    def test_bundled_xauusd_example_config_validates(self) -> None:
        import json
        from pathlib import Path

        example_path = Path(__file__).resolve().parents[2] / "examples" / "xauusd_research_dataset.example.json"
        raw = json.loads(example_path.read_text())
        config = ResearchDatasetConfig.model_validate(raw)
        assert config.symbol == "XAUUSD"
        assert config.multi_timeframe is not None
        assert len(config.cross_assets) == 2
        assert len(config.macro_sources) == 1
