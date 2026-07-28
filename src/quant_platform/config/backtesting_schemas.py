"""Pydantic configuration schemas for the leakage-safe financial
evaluation / backtesting framework (Milestone 5). Same conventions as
`config.calibration_schemas`: frozen, `extra="forbid"`, a `.build()`
factory per schema turning validated config into the runtime object it
describes.

WHY THIS CONFIG REFERENCES AN EXISTING `source_calibration_id`, NEVER A
FRESH EXPERIMENT/CALIBRATION CONFIG FILE
--------------------------------------------------------------------------
Exactly `CalibrationConfig`'s own precedent, one layer up: `BacktestConfig`
evaluates an ALREADY-COMPLETED calibration's outputs -- there is no new
experiment or calibration to build here. `BacktestConfig.build()` takes an
already-loaded `CalibrationManifest`/`ExperimentSpec` and derives
`BacktestSpec`'s identity-relevant `source_experiment_id`/
`dataset_content_id`/`split_plan_fingerprint`/`instrument_identity`/
`bar_interval` fields directly from them -- never re-typed by a human into
this config file, where they could silently drift from the actual bound
calibration/experiment (see `backtesting.runner.resolve_backtest_inputs`'s
identical cross-check at RUN time)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.backtesting.models import (
    CommissionModelKind,
    CompoundingPolicyKind,
    DecisionTimestampPolicyKind,
    EntryPolicyKind,
    ExitPolicyKind,
    FinalTradePolicyKind,
    FinancingModelKind,
    OverlapPolicyKind,
    PositionMode,
    PriceBasisKind,
    ReturnCalculationPolicyKind,
    SignalMappingPolicyKind,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.specs import (
    DEFAULT_COST_SENSITIVITY_SCENARIOS,
    BacktestSpec,
    CommissionSpec,
    CostSensitivityScenario,
    EntrySpec,
    ExitSpec,
    FinancingSpec,
    SignalMappingSpec,
    SlippageSpec,
    SpreadSpec,
)
from quant_platform.calibration.manifests import CalibrationManifest
from quant_platform.calibration.models import DeterminismPolicy
from quant_platform.core.types import Timeframe
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.fingerprints import fingerprint_json


class BacktestSignalMappingConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[
        "directional_long_flat", "directional_long_short", "probability_bands", "abstention_aware",
        "confidence_floor", "uncertainty_ceiling", "combined_confidence_uncertainty",
    ]
    probability_band_long_min: float | None = Field(default=None, ge=0.0, le=1.0)
    probability_band_short_max: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_floor: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty_ceiling: float | None = Field(default=None, ge=0.0, le=1.0)

    def build(self) -> SignalMappingSpec:
        return SignalMappingSpec(
            kind=SignalMappingPolicyKind(self.kind), probability_band_long_min=self.probability_band_long_min,
            probability_band_short_max=self.probability_band_short_max, confidence_floor=self.confidence_floor,
            uncertainty_ceiling=self.uncertainty_ceiling,
        )


class BacktestEntryConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["next_bar_open", "next_bar_mid", "next_bar_side_aware", "delayed_bar"] = "next_bar_open"
    delay_bars: int = Field(default=1, ge=0)
    allow_same_bar_close: bool = False

    def build(self) -> EntrySpec:
        return EntrySpec(kind=EntryPolicyKind(self.kind), delay_bars=self.delay_bars, allow_same_bar_close=self.allow_same_bar_close)


class BacktestExitConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["fixed_horizon", "next_bar_close", "end_of_fold", "opposite_signal"]
    holding_period_bars: int | None = Field(default=None, ge=1)
    final_trade_policy: Literal["discard_incomplete", "force_close_at_final_price", "mark_incomplete_exclude"] = "mark_incomplete_exclude"

    def build(self) -> ExitSpec:
        return ExitSpec(kind=ExitPolicyKind(self.kind), holding_period_bars=self.holding_period_bars, final_trade_policy=FinalTradePolicyKind(self.final_trade_policy))


class BacktestSpreadConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["zero", "fixed_price_units", "fixed_basis_points", "bid_ask_observed"] = "zero"
    price_units: float | None = Field(default=None, ge=0.0)
    basis_points: float | None = Field(default=None, ge=0.0)

    def build(self) -> SpreadSpec:
        return SpreadSpec(kind=SpreadModelKind(self.kind), price_units=self.price_units, basis_points=self.basis_points)


class BacktestCommissionConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["zero", "per_side_basis_points", "fixed_per_trade"] = "zero"
    per_side_basis_points: float | None = Field(default=None, ge=0.0)
    fixed_per_trade: float | None = Field(default=None, ge=0.0)

    def build(self) -> CommissionSpec:
        return CommissionSpec(kind=CommissionModelKind(self.kind), per_side_basis_points=self.per_side_basis_points, fixed_per_trade=self.fixed_per_trade)


class BacktestSlippageConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["zero", "fixed_basis_points", "fixed_price_units"] = "zero"
    basis_points: float | None = Field(default=None, ge=0.0)
    price_units: float | None = Field(default=None, ge=0.0)

    def build(self) -> SlippageSpec:
        return SlippageSpec(kind=SlippageModelKind(self.kind), basis_points=self.basis_points, price_units=self.price_units)


class BacktestFinancingConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["none", "fixed_daily_basis_points"] = "none"
    daily_basis_points: float | None = None

    def build(self) -> FinancingSpec:
        return FinancingSpec(kind=FinancingModelKind(self.kind), daily_basis_points=self.daily_basis_points)


class BacktestCostSensitivityScenarioConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    spread_multiplier: float = Field(default=1.0, ge=0.0)
    slippage_multiplier: float = Field(default=1.0, ge=0.0)
    commission_multiplier: float = Field(default=1.0, ge=0.0)

    def build(self) -> CostSensitivityScenario:
        return CostSensitivityScenario(
            name=self.name, spread_multiplier=self.spread_multiplier, slippage_multiplier=self.slippage_multiplier,
            commission_multiplier=self.commission_multiplier,
        )


class BacktestConfig(BaseModel):
    """The top-level config for one backtest run -- everything
    `quant_platform.ml_cli`'s `run-backtest`/`resume-backtest` commands
    need, all in one validated object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ml_artifacts_root: Path
    historical_storage_root: Path
    """Where raw OHLCV market bars are read from (`historical.loader.
    DatasetLoader`) -- Section 6's market-data contract is independent of
    the ML artifact store, exactly `feature_cli`'s identical separation of
    `historical_storage_root` from `research_storage_root`/
    `ml_artifacts_root`."""
    research_storage_root: Path
    source_calibration_id: str = Field(min_length=64, max_length=64)
    decision_timestamp_policy: Literal["after_bar_close", "before_next_bar_open", "externally_timestamped"] = "after_bar_close"
    signal_mapping: BacktestSignalMappingConfigSchema
    position_mode: Literal["long_flat", "long_short"]
    entry: BacktestEntryConfigSchema = Field(default_factory=BacktestEntryConfigSchema)
    exit: BacktestExitConfigSchema
    overlap_policy: Literal["ignore", "close_and_reverse", "close_only", "queue", "independent_overlapping"] = "ignore"
    price_basis: Literal["close", "mid", "bid_ask"] = "close"
    spread: BacktestSpreadConfigSchema = Field(default_factory=BacktestSpreadConfigSchema)
    commission: BacktestCommissionConfigSchema = Field(default_factory=BacktestCommissionConfigSchema)
    slippage: BacktestSlippageConfigSchema = Field(default_factory=BacktestSlippageConfigSchema)
    financing: BacktestFinancingConfigSchema = Field(default_factory=BacktestFinancingConfigSchema)
    return_calculation_policy: Literal["simple", "log"] = "simple"
    compounding_policy: Literal["non_compounded", "compounded"] = "non_compounded"
    initial_notional: float = Field(gt=0.0)
    respect_calibration_abstention: bool = True
    exposure_cap: float = Field(default=1.0, gt=0.0)
    annual_risk_free_rate: float = 0.0
    minimum_trades_for_valid_fold: int = Field(default=1, ge=0)
    timestamp_column: str = "open_time"
    seed: int = Field(default=0, ge=0)
    determinism_policy: Literal["strict", "warn"] = "strict"
    cost_sensitivity_scenarios: list[BacktestCostSensitivityScenarioConfigSchema] | None = None
    """`None` (the default) uses `backtesting.specs.
    DEFAULT_COST_SENSITIVITY_SCENARIOS` -- a caller only overrides this to
    declare a DIFFERENT bounded, pre-declared scenario set (Section 20),
    never to hand-pick scenarios after seeing results."""

    def build(self, *, calibration_manifest: CalibrationManifest, experiment_spec: ExperimentSpec) -> BacktestSpec:
        scenarios = (
            DEFAULT_COST_SENSITIVITY_SCENARIOS if self.cost_sensitivity_scenarios is None
            else tuple(s.build() for s in self.cost_sensitivity_scenarios)
        )
        return BacktestSpec(
            schema_version=1, source_calibration_id=self.source_calibration_id, source_experiment_id=calibration_manifest.source_experiment_id,
            source_execution_id=calibration_manifest.source_experiment_id, dataset_content_id=experiment_spec.dataset_binding.content_id,
            split_plan_fingerprint=fingerprint_json(experiment_spec.split_binding.to_json_dict()),
            instrument_identity=experiment_spec.dataset_binding.symbol, market_timezone="UTC",
            bar_interval=Timeframe(experiment_spec.dataset_binding.base_timeframe), decision_timestamp_policy=DecisionTimestampPolicyKind(self.decision_timestamp_policy),
            signal_mapping=self.signal_mapping.build(), position_mode=PositionMode(self.position_mode), entry_spec=self.entry.build(),
            exit_spec=self.exit.build(), overlap_policy=OverlapPolicyKind(self.overlap_policy), price_basis=PriceBasisKind(self.price_basis),
            spread_spec=self.spread.build(), commission_spec=self.commission.build(), slippage_spec=self.slippage.build(),
            financing_spec=self.financing.build(), return_calculation_policy=ReturnCalculationPolicyKind(self.return_calculation_policy),
            compounding_policy=CompoundingPolicyKind(self.compounding_policy), initial_notional=self.initial_notional,
            determinism_policy=DeterminismPolicy(self.determinism_policy), respect_calibration_abstention=self.respect_calibration_abstention,
            exposure_cap=self.exposure_cap, annual_risk_free_rate=self.annual_risk_free_rate,
            minimum_trades_for_valid_fold=self.minimum_trades_for_valid_fold, timestamp_column=self.timestamp_column, seed=self.seed,
            cost_sensitivity_scenarios=scenarios,
        )


__all__ = [
    "BacktestCommissionConfigSchema",
    "BacktestConfig",
    "BacktestCostSensitivityScenarioConfigSchema",
    "BacktestEntryConfigSchema",
    "BacktestExitConfigSchema",
    "BacktestFinancingConfigSchema",
    "BacktestSignalMappingConfigSchema",
    "BacktestSlippageConfigSchema",
    "BacktestSpreadConfigSchema",
]
