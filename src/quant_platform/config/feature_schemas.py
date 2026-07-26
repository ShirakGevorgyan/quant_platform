"""Pydantic configuration schemas for the feature engineering / research
dataset platform (Milestone 3). Same conventions as
`config.historical_schemas` -- frozen, `extra="forbid"`, a `.build()`
factory per schema turning validated config into the runtime object it
describes -- extending, not replacing, the existing `config` subpackage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_platform.core.types import Timeframe
from quant_platform.features.labels import LabelDefinition, LabelKind
from quant_platform.features.multi_timeframe import MultiTimeframeWindows
from quant_platform.features.normalization import TransformKind
from quant_platform.features.technical.price import TechnicalWindows
from quant_platform.features.validation import ValidationThresholds

_TIMEFRAME_CHOICES = ("M1", "M5", "M15", "M30", "H1", "H4", "H12", "D1")


class TechnicalFeatureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    return_windows: list[int] = Field(default_factory=lambda: [1, 5, 15])
    momentum_windows: list[int] = Field(default_factory=lambda: [10, 20])
    volatility_window: int = Field(default=20, gt=1)
    zscore_window: int = Field(default=20, gt=1)
    ma_distance_windows: list[int] = Field(default_factory=lambda: [20, 50])
    high_low_distance_window: int = Field(default=20, gt=1)
    atr_window: int = Field(default=14, gt=1)
    volume_window: int = Field(default=20, gt=1)
    include_spread: bool = False
    spread_window: int = Field(default=20, gt=1)

    def build(self) -> TechnicalWindows:
        return TechnicalWindows(
            return_windows=tuple(self.return_windows), momentum_windows=tuple(self.momentum_windows),
            volatility_window=self.volatility_window, zscore_window=self.zscore_window,
            ma_distance_windows=tuple(self.ma_distance_windows),
            high_low_distance_window=self.high_low_distance_window, atr_window=self.atr_window,
            volume_window=self.volume_window, include_spread=self.include_spread,
            spread_window=self.spread_window,
        )


class TemporalFeatureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    use_session_calendar: bool = False


class MultiTimeframeFeatureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    higher_timeframes: list[Literal["M5", "M15", "M30", "H1", "H4", "H12", "D1"]] = Field(min_length=1)
    return_window: int = Field(default=3, gt=0)
    volatility_window: int = Field(default=10, gt=0)
    trend_window: int = Field(default=10, gt=0)

    def build_windows(self) -> MultiTimeframeWindows:
        return MultiTimeframeWindows(
            return_window=self.return_window, volatility_window=self.volatility_window,
            trend_window=self.trend_window,
        )

    def build_timeframes(self) -> tuple[Timeframe, ...]:
        return tuple(Timeframe(v) for v in self.higher_timeframes)


class CrossAssetInstrumentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    timeframe: Literal["M1", "M5", "M15", "M30", "H1", "H4", "H12", "D1"] = "M1"
    return_window: int = Field(default=5, gt=0)
    momentum_window: int = Field(default=10, gt=0)
    volatility_window: int = Field(default=20, gt=0)
    correlation_window: int = Field(default=30, gt=1)

    def build_timeframe(self) -> Timeframe:
        return Timeframe(self.timeframe)


class MacroSourceFeatureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_name: str = Field(min_length=1)
    data_path: Path
    """CSV file with columns `observation_time,release_time,value` (ISO8601
    UTC timestamps) -- macro/external data is not part of the Milestone 2
    historical OHLCV pipeline, so it is supplied directly rather than
    loaded via `DatasetLoader`."""
    change_lookback: int = Field(default=1, gt=0)
    tolerance_days: float | None = Field(default=None, gt=0.0)


class LabelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["future_return", "future_log_return", "binary_direction", "vol_adjusted_return", "triple_barrier"]
    horizon_bars: int = Field(gt=0)
    params: dict[str, float] = Field(default_factory=dict)

    def build(self) -> LabelDefinition:
        return LabelDefinition(
            name=self.name, kind=LabelKind(self.kind), horizon_bars=self.horizon_bars,
            params=dict(self.params),
        )


class SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["chronological", "expanding_walk_forward", "rolling_walk_forward"]
    train_fraction: float | None = Field(default=None, gt=0.0, lt=1.0)
    validation_fraction: float | None = Field(default=None, gt=0.0, lt=1.0)
    purge_bars: int = Field(default=0, ge=0)
    embargo_bars: int = Field(default=0, ge=0)
    gap_bars: int = Field(default=0, ge=0)
    n_splits: int | None = Field(default=None, gt=0)
    test_size: int | None = Field(default=None, gt=0)
    max_train_size: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_required_fields(self) -> SplitConfig:
        if self.strategy == "chronological":
            if self.train_fraction is None or self.validation_fraction is None:
                raise ValueError("train_fraction and validation_fraction are required when strategy='chronological'")
        else:
            if self.n_splits is None or self.test_size is None:
                raise ValueError("n_splits and test_size are required for a walk-forward strategy")
        return self

    def build_params(self) -> dict[str, object]:
        if self.strategy == "chronological":
            return {
                "train_fraction": self.train_fraction, "validation_fraction": self.validation_fraction,
                "purge_bars": self.purge_bars, "embargo_bars": self.embargo_bars, "gap_bars": self.gap_bars,
            }
        params: dict[str, object] = {
            "n_splits": self.n_splits, "test_size": self.test_size, "label_horizon": self.purge_bars,
            "embargo": self.embargo_bars,
        }
        if self.strategy == "rolling_walk_forward":
            params["max_train_size"] = self.max_train_size
        return params


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    transforms: dict[str, Literal["standard_scale", "robust_scale", "winsorize", "signed_log1p"]] = Field(
        default_factory=dict
    )

    def build(self) -> dict[str, TransformKind]:
        return {name: TransformKind(kind) for name, kind in self.transforms.items()}


class ValidationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_missing_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    constant_std_epsilon: float = Field(default=1e-12, ge=0.0)
    outlier_zscore_threshold: float = Field(default=8.0, gt=0.0)
    max_stale_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    target_leakage_correlation_threshold: float = Field(default=0.999, gt=0.0, le=1.0)

    def build(self) -> ValidationThresholds:
        return ValidationThresholds(
            max_missing_fraction=self.max_missing_fraction, constant_std_epsilon=self.constant_std_epsilon,
            outlier_zscore_threshold=self.outlier_zscore_threshold, max_stale_fraction=self.max_stale_fraction,
            target_leakage_correlation_threshold=self.target_leakage_correlation_threshold,
        )


class ResearchDatasetConfig(BaseModel):
    """The top-level config for one research dataset build -- everything
    `quant_platform.feature_cli` needs, all in one validated object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    base_timeframe: Literal["M1", "M5", "M15", "M30", "H1", "H4", "H12", "D1"] = "M1"
    start: str
    end: str
    dataset_version: str | None = None
    historical_storage_root: Path
    research_storage_root: Path
    technical: TechnicalFeatureConfig = Field(default_factory=TechnicalFeatureConfig)
    temporal: TemporalFeatureConfig = Field(default_factory=TemporalFeatureConfig)
    multi_timeframe: MultiTimeframeFeatureConfig | None = None
    cross_assets: list[CrossAssetInstrumentConfig] = Field(default_factory=list)
    macro_sources: list[MacroSourceFeatureConfig] = Field(default_factory=list)
    label: LabelConfig
    split: SplitConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    drop_unlabeled_rows: bool = True
    allow_critical_validation_issues: bool = False

    def build_base_timeframe(self) -> Timeframe:
        return Timeframe(self.base_timeframe)


__all__ = [
    "CrossAssetInstrumentConfig",
    "LabelConfig",
    "MacroSourceFeatureConfig",
    "MultiTimeframeFeatureConfig",
    "PreprocessingConfig",
    "ResearchDatasetConfig",
    "SplitConfig",
    "TechnicalFeatureConfig",
    "TemporalFeatureConfig",
    "ValidationConfig",
]
