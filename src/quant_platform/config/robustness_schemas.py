"""Pydantic configuration schema for Milestone 6's robustness/promotion
framework. Same conventions as `config.backtesting_schemas`: frozen,
`extra="forbid"`, a `.build()` factory turning validated config into the
runtime `RobustnessSpec` it describes.

WHY THIS CONFIG DERIVES `dataset_content_id`/`split_plan_fingerprint`/
`instrument_identity`/`bar_interval` FROM THE SOURCE BACKTEST, NEVER
FROM A HUMAN-TYPED FIELD
--------------------------------------------------------------------------
Exactly `BacktestConfig`'s own precedent one layer down: a human only
provides `source_backtest_id`; `RobustnessConfig.build()` takes the
ALREADY-LOADED source `BacktestSpec` and derives every identity-bearing
field from it directly -- never re-typed, where it could silently drift
from the backtest actually being analyzed (see `robustness.source.
verify_and_load_source_backtest`'s identical cross-check at RUN time,
which would catch -- but only after wasted work -- a config file that
tried to claim a mismatched identity anyway).

SCOPE: this schema exposes the bootstrap/threshold/minimum-sample knobs
operators most commonly tune. `stress_scenarios`/`regime_definitions`/
`perturbations`/`promotion_policy` use this package's own `DEFAULT_*`
constants and are not yet independently configurable from this JSON
config format -- an operator who needs a custom stress matrix, regime
set, perturbation grid, or promotion policy constructs a `RobustnessSpec`
directly via the Python API (see `robustness.specs`)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.backtesting.specs import BacktestSpec
from quant_platform.robustness.models import (
    BootstrapMethodKind,
    MultipleTestingCorrectionKind,
    ReturnSeriesKind,
)
from quant_platform.robustness.specs import (
    DEFAULT_PROMOTION_GATES,
    DEFAULT_REGIME_DEFINITIONS,
    DEFAULT_STRESS_SCENARIOS,
    ROBUSTNESS_SPEC_SCHEMA_VERSION,
    BootstrapSpec,
    PromotionPolicySpec,
    RobustnessSpec,
    StabilityThresholds,
)


class RobustnessConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ml_artifacts_root: Path
    historical_storage_root: Path
    research_storage_root: Path
    source_backtest_id: str

    return_series_kind: Literal["bar_net", "bar_gross", "trade_net", "trade_gross", "stitched_bar_net", "per_fold_bar_net", "benchmark_relative"] = "stitched_bar_net"
    bootstrap_method: Literal["iid", "moving_block", "stationary", "fold_level"] = "stationary"
    bootstrap_repetitions: int = Field(default=1000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    block_length: int | None = Field(default=10, ge=1)
    seed: int = Field(default=0, ge=0)
    multiple_testing_correction: Literal["bonferroni", "holm", "benjamini_hochberg"] = "benjamini_hochberg"
    strategy_family_id: str | None = None
    minimum_fold_count: int = Field(default=3, ge=1)
    minimum_trade_count: int = Field(default=30, ge=1)
    minimum_effective_sample_size: int = Field(default=30, ge=1)
    minimum_profitable_fold_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    maximum_single_fold_profit_concentration: float = Field(default=0.6, ge=0.0, le=1.0)
    maximum_single_trade_profit_concentration: float = Field(default=0.4, ge=0.0, le=1.0)
    maximum_single_direction_profit_concentration: float = Field(default=0.7, ge=0.0, le=1.0)

    def build(self, *, source_backtest_spec: BacktestSpec) -> RobustnessSpec:
        block_length = self.block_length if BootstrapMethodKind(self.bootstrap_method) in (BootstrapMethodKind.MOVING_BLOCK, BootstrapMethodKind.STATIONARY) else None
        return RobustnessSpec(
            schema_version=ROBUSTNESS_SPEC_SCHEMA_VERSION, source_backtest_id=self.source_backtest_id,
            dataset_content_id=source_backtest_spec.dataset_content_id, split_plan_fingerprint=source_backtest_spec.split_plan_fingerprint,
            instrument_identity=source_backtest_spec.instrument_identity, bar_interval=source_backtest_spec.bar_interval,
            return_series_kind=ReturnSeriesKind(self.return_series_kind),
            bootstrap_spec=BootstrapSpec(
                method=BootstrapMethodKind(self.bootstrap_method), repetitions=self.bootstrap_repetitions, confidence_level=self.confidence_level,
                block_length=block_length,
            ),
            seed=self.seed, multiple_testing_correction=MultipleTestingCorrectionKind(self.multiple_testing_correction),
            strategy_family_id=self.strategy_family_id, minimum_fold_count=self.minimum_fold_count, minimum_trade_count=self.minimum_trade_count,
            minimum_effective_sample_size=self.minimum_effective_sample_size,
            stability_thresholds=StabilityThresholds(
                minimum_profitable_fold_fraction=self.minimum_profitable_fold_fraction,
                maximum_single_fold_profit_concentration=self.maximum_single_fold_profit_concentration,
                maximum_single_trade_profit_concentration=self.maximum_single_trade_profit_concentration,
                maximum_single_direction_profit_concentration=self.maximum_single_direction_profit_concentration,
            ),
            stress_scenarios=DEFAULT_STRESS_SCENARIOS, regime_definitions=DEFAULT_REGIME_DEFINITIONS,
            promotion_policy=PromotionPolicySpec(gates=DEFAULT_PROMOTION_GATES),
        )


__all__ = ["RobustnessConfig"]
