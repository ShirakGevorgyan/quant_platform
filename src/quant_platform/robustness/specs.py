"""Immutable specifications for `quant_platform.robustness` (Milestone 6,
Section 2). Every spec here is a frozen, slotted dataclass with an
explicit `__post_init__` validator -- nothing is an arbitrary,
unvalidated dict at a public boundary. `RobustnessSpec` is the top-level,
content-addressed specification; every other spec in this module is
embedded within it and therefore participates in its identity, exactly
`backtesting.specs.BacktestSpec`'s own convention."""

from __future__ import annotations

import math
from dataclasses import dataclass

from quant_platform.core.exceptions import RobustnessValidationError
from quant_platform.core.types import Timeframe
from quant_platform.ml.fingerprints import fingerprint_json, is_valid_sha256_hex
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version
from quant_platform.robustness.models import (
    BootstrapMethodKind,
    MultipleTestingCorrectionKind,
    PerturbationAxisKind,
    RegimeDimensionKind,
    ReturnSeriesKind,
    StressAxisKind,
)

ROBUSTNESS_SPEC_SCHEMA_VERSION = 1


def _finite(value: float, *, field_name: str) -> None:
    if not math.isfinite(value):
        raise RobustnessValidationError(f"{field_name} must be finite, got {value!r}")


def _positive(value: float, *, field_name: str) -> None:
    _finite(value, field_name=field_name)
    if value <= 0.0:
        raise RobustnessValidationError(f"{field_name} must be > 0, got {value!r}")


def _non_negative(value: float, *, field_name: str) -> None:
    _finite(value, field_name=field_name)
    if value < 0.0:
        raise RobustnessValidationError(f"{field_name} must be >= 0, got {value!r}")


def _unit_interval(value: float, *, field_name: str) -> None:
    _finite(value, field_name=field_name)
    if not (0.0 < value < 1.0):
        raise RobustnessValidationError(f"{field_name} must be strictly between 0 and 1, got {value!r}")


# --------------------------------------------------------------------------
# Bootstrap (Section 5)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BootstrapSpec:
    """Section 5: one declared, deterministic bootstrap configuration.
    `block_length` is REQUIRED for `MOVING_BLOCK`/`STATIONARY` (the
    dependence-aware methods) and must be `None` for `IID`/`FOLD_LEVEL`
    (which have no block-length concept).

    Closure-audit note (performance/resource, Section 13): `repetitions`
    has a LOWER bound (`>= 100`, too few for a stable CI below that) but
    deliberately no UPPER bound -- an operator can request an arbitrarily
    expensive run (e.g. `repetitions=10_000_000`). This is a disclosed
    operational consideration, not a correctness defect: `bootstrap.
    BootstrapEstimate` never persists raw per-repetition samples (only
    the point estimate, CI bounds, and repetition/failure counts), so
    memory growth is bounded by `repetitions` alone (each resample is
    discarded after its statistic is computed), not amplified by
    persistence. No hard cap is imposed here since the "right" ceiling is
    an operational/deployment decision, not something this audit should
    invent unilaterally."""

    method: BootstrapMethodKind
    repetitions: int
    confidence_level: float
    block_length: int | None = None

    def __post_init__(self) -> None:
        if self.repetitions < 100:
            raise RobustnessValidationError(f"BootstrapSpec.repetitions must be >= 100 (too few for a stable CI), got {self.repetitions}")
        _unit_interval(self.confidence_level, field_name="BootstrapSpec.confidence_level")
        if self.method in (BootstrapMethodKind.MOVING_BLOCK, BootstrapMethodKind.STATIONARY):
            if self.block_length is None or self.block_length < 1:
                raise RobustnessValidationError(f"BootstrapSpec.block_length must be a positive int when method={self.method.value!r}")
        elif self.block_length is not None:
            raise RobustnessValidationError(f"BootstrapSpec.block_length must be None when method={self.method.value!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "method": self.method.value, "repetitions": self.repetitions, "confidence_level": self.confidence_level,
            "block_length": self.block_length,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BootstrapSpec:
        return cls(
            method=BootstrapMethodKind(raw["method"]), repetitions=int(str(raw["repetitions"])),
            confidence_level=float(str(raw["confidence_level"])),
            block_length=(None if raw.get("block_length") is None else int(str(raw["block_length"]))),
        )


# --------------------------------------------------------------------------
# Stability thresholds (Section 8)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StabilityThresholds:
    minimum_profitable_fold_fraction: float
    maximum_single_fold_profit_concentration: float
    maximum_single_trade_profit_concentration: float
    maximum_single_direction_profit_concentration: float

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_profitable_fold_fraction", self.minimum_profitable_fold_fraction),
            ("maximum_single_fold_profit_concentration", self.maximum_single_fold_profit_concentration),
            ("maximum_single_trade_profit_concentration", self.maximum_single_trade_profit_concentration),
            ("maximum_single_direction_profit_concentration", self.maximum_single_direction_profit_concentration),
        ):
            _unit_interval(value, field_name=f"StabilityThresholds.{name}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "minimum_profitable_fold_fraction": self.minimum_profitable_fold_fraction,
            "maximum_single_fold_profit_concentration": self.maximum_single_fold_profit_concentration,
            "maximum_single_trade_profit_concentration": self.maximum_single_trade_profit_concentration,
            "maximum_single_direction_profit_concentration": self.maximum_single_direction_profit_concentration,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> StabilityThresholds:
        return cls(
            minimum_profitable_fold_fraction=float(str(raw["minimum_profitable_fold_fraction"])),
            maximum_single_fold_profit_concentration=float(str(raw["maximum_single_fold_profit_concentration"])),
            maximum_single_trade_profit_concentration=float(str(raw["maximum_single_trade_profit_concentration"])),
            maximum_single_direction_profit_concentration=float(str(raw["maximum_single_direction_profit_concentration"])),
        )


# --------------------------------------------------------------------------
# Stress scenarios (Section 10) -- ONE declared matrix, each entry tagged
# with its own StressAxisKind; Section 2 names "cost/latency/spread/
# slippage stress" as four concepts, but Section 10's own minimum required
# scenario list is inherently a single mixed matrix (each scenario usually
# perturbs exactly one axis, "combined_adverse" perturbs several at once)
# -- so a single declared tuple, filterable by axis, is the faithful,
# non-redundant representation of what Section 10 actually asks for.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StressScenarioSpec:
    name: str
    axis: StressAxisKind
    spread_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    commission_multiplier: float = 1.0
    financing_multiplier: float = 1.0
    additional_latency_bars: int = 0
    named_profile: str | None = None
    """Section 10: an identifier such as `rollover_spread_expansion` --
    a CONFIGURATION identity this run declares it is stress-testing
    against, never a claim about actual broker behavior."""

    def __post_init__(self) -> None:
        if not self.name:
            raise RobustnessValidationError("StressScenarioSpec.name must not be empty")
        for field_name, value in (
            ("spread_multiplier", self.spread_multiplier), ("slippage_multiplier", self.slippage_multiplier),
            ("commission_multiplier", self.commission_multiplier), ("financing_multiplier", self.financing_multiplier),
        ):
            _non_negative(value, field_name=f"StressScenarioSpec.{field_name}")
        if self.additional_latency_bars < 0:
            raise RobustnessValidationError(f"StressScenarioSpec.additional_latency_bars must be >= 0, got {self.additional_latency_bars}")
        if self.axis is StressAxisKind.NAMED_PROFILE and not self.named_profile:
            raise RobustnessValidationError("StressScenarioSpec.named_profile is required when axis=named_profile")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "axis": self.axis.value, "spread_multiplier": self.spread_multiplier,
            "slippage_multiplier": self.slippage_multiplier, "commission_multiplier": self.commission_multiplier,
            "financing_multiplier": self.financing_multiplier, "additional_latency_bars": self.additional_latency_bars,
            "named_profile": self.named_profile,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> StressScenarioSpec:
        return cls(
            name=str(raw["name"]), axis=StressAxisKind(raw["axis"]), spread_multiplier=float(str(raw.get("spread_multiplier", 1.0))),
            slippage_multiplier=float(str(raw.get("slippage_multiplier", 1.0))), commission_multiplier=float(str(raw.get("commission_multiplier", 1.0))),
            financing_multiplier=float(str(raw.get("financing_multiplier", 1.0))), additional_latency_bars=int(str(raw.get("additional_latency_bars", 0))),
            named_profile=(None if raw.get("named_profile") is None else str(raw["named_profile"])),
        )


DEFAULT_STRESS_SCENARIOS: tuple[StressScenarioSpec, ...] = (
    StressScenarioSpec(name="zero_cost", axis=StressAxisKind.ZERO_COST, spread_multiplier=0.0, slippage_multiplier=0.0, commission_multiplier=0.0, financing_multiplier=0.0),
    StressScenarioSpec(name="base_cost", axis=StressAxisKind.BASE_COST),
    StressScenarioSpec(name="1.5x_spread", axis=StressAxisKind.SPREAD_MULTIPLIER, spread_multiplier=1.5),
    StressScenarioSpec(name="2x_spread", axis=StressAxisKind.SPREAD_MULTIPLIER, spread_multiplier=2.0),
    StressScenarioSpec(name="3x_spread", axis=StressAxisKind.SPREAD_MULTIPLIER, spread_multiplier=3.0),
    StressScenarioSpec(name="2x_slippage", axis=StressAxisKind.SLIPPAGE_MULTIPLIER, slippage_multiplier=2.0),
    StressScenarioSpec(name="3x_slippage", axis=StressAxisKind.SLIPPAGE_MULTIPLIER, slippage_multiplier=3.0),
    StressScenarioSpec(name="increased_commission", axis=StressAxisKind.COMMISSION_MULTIPLIER, commission_multiplier=2.0),
    StressScenarioSpec(name="increased_financing", axis=StressAxisKind.FINANCING_MULTIPLIER, financing_multiplier=2.0),
    StressScenarioSpec(name="1bar_latency", axis=StressAxisKind.ADDITIONAL_LATENCY_BARS, additional_latency_bars=1),
    StressScenarioSpec(name="2bar_latency", axis=StressAxisKind.ADDITIONAL_LATENCY_BARS, additional_latency_bars=2),
    StressScenarioSpec(
        name="combined_adverse", axis=StressAxisKind.COMBINED_ADVERSE, spread_multiplier=2.0, slippage_multiplier=2.0,
        commission_multiplier=1.5, financing_multiplier=1.5, additional_latency_bars=1,
    ),
)


# --------------------------------------------------------------------------
# Regime definitions (Section 11)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RegimeDefinitionSpec:
    dimension: RegimeDimensionKind
    minimum_regime_samples: int
    n_quantiles: int | None = None
    trailing_window_bars: int | None = None
    """Section 11: the trailing, strictly-past-information window used to
    classify a bar's regime -- required for every dimension that is not
    purely calendar-derived (`SESSION`/`DAY_OF_WEEK`/`HOUR_OF_DAY` need
    no window; every other dimension does, to remain leakage-safe)."""

    _CALENDAR_DIMENSIONS = frozenset({RegimeDimensionKind.SESSION, RegimeDimensionKind.DAY_OF_WEEK, RegimeDimensionKind.HOUR_OF_DAY})
    _QUANTILE_DIMENSIONS = frozenset({
        RegimeDimensionKind.VOLATILITY_QUANTILE, RegimeDimensionKind.LIQUIDITY_QUANTILE, RegimeDimensionKind.SPREAD_QUANTILE,
    })

    def __post_init__(self) -> None:
        if self.minimum_regime_samples < 1:
            raise RobustnessValidationError(f"RegimeDefinitionSpec.minimum_regime_samples must be >= 1, got {self.minimum_regime_samples}")
        if self.dimension not in self._CALENDAR_DIMENSIONS:
            if self.trailing_window_bars is None or self.trailing_window_bars < 2:
                raise RobustnessValidationError(
                    f"RegimeDefinitionSpec.trailing_window_bars must be >= 2 (leakage-safe trailing window required) "
                    f"for dimension={self.dimension.value!r}"
                )
        elif self.trailing_window_bars is not None:
            raise RobustnessValidationError(f"RegimeDefinitionSpec.trailing_window_bars must be None for calendar dimension={self.dimension.value!r}")
        if self.dimension in self._QUANTILE_DIMENSIONS:
            if self.n_quantiles is None or self.n_quantiles < 2:
                raise RobustnessValidationError(f"RegimeDefinitionSpec.n_quantiles must be >= 2 for dimension={self.dimension.value!r}")
        elif self.n_quantiles is not None:
            raise RobustnessValidationError(f"RegimeDefinitionSpec.n_quantiles must be None for dimension={self.dimension.value!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension.value, "minimum_regime_samples": self.minimum_regime_samples,
            "n_quantiles": self.n_quantiles, "trailing_window_bars": self.trailing_window_bars,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RegimeDefinitionSpec:
        return cls(
            dimension=RegimeDimensionKind(raw["dimension"]), minimum_regime_samples=int(str(raw["minimum_regime_samples"])),
            n_quantiles=(None if raw.get("n_quantiles") is None else int(str(raw["n_quantiles"]))),
            trailing_window_bars=(None if raw.get("trailing_window_bars") is None else int(str(raw["trailing_window_bars"]))),
        )


DEFAULT_REGIME_DEFINITIONS: tuple[RegimeDefinitionSpec, ...] = (
    RegimeDefinitionSpec(dimension=RegimeDimensionKind.VOLATILITY_QUANTILE, minimum_regime_samples=10, n_quantiles=3, trailing_window_bars=20),
    RegimeDefinitionSpec(dimension=RegimeDimensionKind.TREND_DIRECTION, minimum_regime_samples=10, trailing_window_bars=20),
    RegimeDefinitionSpec(dimension=RegimeDimensionKind.PRICE_REGIME, minimum_regime_samples=10, trailing_window_bars=50),
    RegimeDefinitionSpec(dimension=RegimeDimensionKind.DAY_OF_WEEK, minimum_regime_samples=10),
    RegimeDefinitionSpec(dimension=RegimeDimensionKind.HOUR_OF_DAY, minimum_regime_samples=10),
)


# --------------------------------------------------------------------------
# Parameter perturbations (Section 9)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PerturbationSpec:
    """One declared axis of controlled perturbation around the ALREADY-
    SELECTED operating point. `relative_deltas` are dimensionless
    fractions applied multiplicatively to the source spec's own value for
    this axis (e.g. `(-0.2, -0.1, 0.0, 0.1, 0.2)`); `0.0` (the unperturbed
    point) need not be included explicitly -- it is always implicitly
    evaluated as the baseline every perturbed point is compared against.

    Closure-audit note (performance/resource, Section 13): `RobustnessSpec
    .perturbations` (the set of AXES) is inherently bounded by
    `PerturbationAxisKind`'s own small closed enum (uniqueness-by-axis
    enforced) -- but `relative_deltas` WITHIN one axis has no count limit;
    each declared point costs one full re-simulation in `sensitivity.py`.
    Same disclosed operational consideration as `BootstrapSpec.
    repetitions`'s own note -- no hard cap invented here."""

    axis: PerturbationAxisKind
    relative_deltas: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.relative_deltas:
            raise RobustnessValidationError(f"PerturbationSpec.relative_deltas must not be empty for axis={self.axis.value!r}")
        for delta in self.relative_deltas:
            _finite(delta, field_name=f"PerturbationSpec.relative_deltas[{self.axis.value}]")
        if len(set(self.relative_deltas)) != len(self.relative_deltas):
            raise RobustnessValidationError(f"PerturbationSpec.relative_deltas must not contain duplicates for axis={self.axis.value!r}")

    def to_json_dict(self) -> dict[str, object]:
        # MUST preserve declared order: `sensitivity._compute_axis_sensitivity`
        # builds `AxisSensitivityResult.points` by enumerating `relative_deltas`
        # positionally, and `verification.verify_robustness` reloads a
        # RobustnessSpec from its persisted JSON and RE-RUNS sensitivity
        # analysis from it, comparing the recomputed SensitivityReport against
        # the one persisted during the forward pass (built from the spec's
        # ORIGINAL, not-yet-round-tripped order) -- sorting here silently
        # reordered `points` between the two passes and produced a spurious
        # `sensitivity_report_mismatch` (release-audit regression, caught by
        # the full acceptance-workflow test failing after this method was
        # first changed to sort). Canonicalization for `robustness_id`
        # purposes belongs ONLY in `to_identity_payload`, which never touches
        # the durable/round-tripped representation.
        return {"axis": self.axis.value, "relative_deltas": list(self.relative_deltas)}

    def to_identity_payload(self) -> dict[str, object]:
        """`relative_deltas` is semantically an unordered grid of test
        points (duplicate-free, per `__post_init__`) -- declared order must
        not affect `robustness_id`, even though it MUST be preserved by
        `to_json_dict` for round-trip/recomputation fidelity (see there)."""
        return {"axis": self.axis.value, "relative_deltas": sorted(self.relative_deltas)}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PerturbationSpec:
        return cls(axis=PerturbationAxisKind(raw["axis"]), relative_deltas=tuple(float(str(d)) for d in as_json_list(raw["relative_deltas"], field_name="relative_deltas")))


DEFAULT_PERTURBATIONS: tuple[PerturbationSpec, ...] = (
    PerturbationSpec(axis=PerturbationAxisKind.PROBABILITY_THRESHOLD, relative_deltas=(-0.1, -0.05, 0.05, 0.1)),
    PerturbationSpec(axis=PerturbationAxisKind.HOLDING_PERIOD_BARS, relative_deltas=(-0.2, -0.1, 0.1, 0.2)),
    PerturbationSpec(axis=PerturbationAxisKind.EXPOSURE_CAP, relative_deltas=(-0.2, -0.1, 0.1, 0.2)),
)


# --------------------------------------------------------------------------
# Promotion policy (Section 13) -- gate SPECS here; measured values and
# pass/fail/skip outcomes are computed and persisted by `promotion.py`.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PromotionGateSpec:
    name: str
    mandatory: bool
    minimum_value: float | None = None
    maximum_value: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise RobustnessValidationError("PromotionGateSpec.name must not be empty")
        if self.minimum_value is None and self.maximum_value is None:
            raise RobustnessValidationError(f"PromotionGateSpec {self.name!r}: at least one of minimum_value/maximum_value must be set")
        if self.minimum_value is not None:
            _finite(self.minimum_value, field_name=f"PromotionGateSpec[{self.name}].minimum_value")
        if self.maximum_value is not None:
            _finite(self.maximum_value, field_name=f"PromotionGateSpec[{self.name}].maximum_value")

    def to_json_dict(self) -> dict[str, object]:
        return {"name": self.name, "mandatory": self.mandatory, "minimum_value": self.minimum_value, "maximum_value": self.maximum_value}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PromotionGateSpec:
        return cls(
            name=str(raw["name"]), mandatory=bool(raw["mandatory"]),
            minimum_value=(None if raw.get("minimum_value") is None else float(str(raw["minimum_value"]))),
            maximum_value=(None if raw.get("maximum_value") is None else float(str(raw["maximum_value"]))),
        )


@dataclass(frozen=True, slots=True)
class PromotionPolicySpec:
    gates: tuple[PromotionGateSpec, ...]

    def __post_init__(self) -> None:
        if not self.gates:
            raise RobustnessValidationError("PromotionPolicySpec.gates must not be empty")
        if len({g.name for g in self.gates}) != len(self.gates):
            raise RobustnessValidationError("PromotionPolicySpec.gates names must be unique")

    def to_json_dict(self) -> dict[str, object]:
        # MUST preserve declared order -- see PerturbationSpec.to_json_dict's
        # identical note: promotion/selection evaluation code that consumes a
        # RELOADED (persisted-then-decoded) policy iterates `gates` positionally,
        # so sorting here (rather than only in `to_identity_payload`) silently
        # changed recomputed-vs-persisted comparison order for any pipeline that
        # round-trips a policy through JSON -- a release-audit regression.
        return {"gates": [g.to_json_dict() for g in self.gates]}

    def to_identity_payload(self) -> dict[str, object]:
        """`gates` is semantically an unordered set (uniqueness enforced by
        name in `__post_init__`) -- declared order must not affect identity,
        even though it MUST be preserved by `to_json_dict` (see there)."""
        return {"gates": [g.to_json_dict() for g in sorted(self.gates, key=lambda g: g.name)]}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PromotionPolicySpec:
        return cls(gates=tuple(PromotionGateSpec.from_json_dict(as_json_dict(g, field_name="gates[]")) for g in as_json_list(raw["gates"], field_name="gates")))


DEFAULT_PROMOTION_GATES: tuple[PromotionGateSpec, ...] = (
    PromotionGateSpec(name="verified_source_backtest", mandatory=True, minimum_value=1.0),
    PromotionGateSpec(name="minimum_outer_folds", mandatory=True, minimum_value=3.0),
    PromotionGateSpec(name="minimum_trades", mandatory=True, minimum_value=30.0),
    PromotionGateSpec(name="minimum_effective_sample_size", mandatory=True, minimum_value=30.0),
    PromotionGateSpec(name="profitable_fold_fraction", mandatory=True, minimum_value=0.5),
    PromotionGateSpec(name="worst_fold_return", mandatory=False, minimum_value=-0.5),
    PromotionGateSpec(name="maximum_drawdown", mandatory=True, maximum_value=0.5),
    PromotionGateSpec(name="bootstrap_lower_bound_total_net_return", mandatory=True, minimum_value=0.0),
    PromotionGateSpec(name="probability_of_loss", mandatory=True, maximum_value=0.5),
    PromotionGateSpec(name="stressed_cost_total_net_return", mandatory=True, minimum_value=0.0),
    PromotionGateSpec(name="latency_stress_total_net_return", mandatory=False, minimum_value=0.0),
    PromotionGateSpec(name="no_extreme_fold_concentration", mandatory=True, minimum_value=1.0),
    PromotionGateSpec(name="no_parameter_cliff", mandatory=False, minimum_value=1.0),
    PromotionGateSpec(name="no_critical_verification_findings", mandatory=True, minimum_value=1.0),
)


# --------------------------------------------------------------------------
# The top-level spec (Section 2)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RobustnessSpec:
    """The top-level, content-addressed specification for one
    statistical-robustness analysis of one already-verified, already-
    COMPLETED backtest. `robustness_id` (see `compute_robustness_
    identity`) is a pure function of every field below except
    `schema_version`."""

    schema_version: int
    source_backtest_id: str
    dataset_content_id: str
    split_plan_fingerprint: str
    instrument_identity: str
    bar_interval: Timeframe
    return_series_kind: ReturnSeriesKind
    bootstrap_spec: BootstrapSpec
    seed: int
    multiple_testing_correction: MultipleTestingCorrectionKind
    strategy_family_id: str | None
    minimum_fold_count: int
    minimum_trade_count: int
    minimum_effective_sample_size: int
    stability_thresholds: StabilityThresholds
    stress_scenarios: tuple[StressScenarioSpec, ...]
    regime_definitions: tuple[RegimeDefinitionSpec, ...]
    promotion_policy: PromotionPolicySpec
    """Closure-audit finding, fixed in `RobustnessRunner._execute_pipeline`
    (release-audit regression): this field was validated, participated
    in `robustness_id`, and was durably persisted, but `RobustnessRunner`
    previously called `promotion.evaluate_promotion` with its own
    hardcoded `promotion.DEFAULT_PROMOTION_POLICY`, never this field --
    an operator-declared CUSTOM `promotion_policy` (a fully supported,
    validated Python-API construction) was silently ignored by the
    actual promotion decision. Confirmed safe to wire directly:
    `evaluate_promotion`'s gate loop already fails closed with a clear
    `PromotionError` (not a raw `KeyError`) for any gate name absent from
    `promotion._GATE_MEASURERS`, and every current caller (`config.
    robustness_schemas.RobustnessConfig`, every test, the acceptance
    workflow) already declares `promotion_policy=PromotionPolicySpec(
    gates=DEFAULT_PROMOTION_GATES)`, byte-identical in content to
    `DEFAULT_PROMOTION_POLICY` -- so this fix changes behavior for NO
    currently-exercised path, only for a future custom-policy caller."""
    perturbations: tuple[PerturbationSpec, ...] = DEFAULT_PERTURBATIONS

    def __post_init__(self) -> None:
        for field_name, value in (("source_backtest_id", self.source_backtest_id), ("dataset_content_id", self.dataset_content_id), ("split_plan_fingerprint", self.split_plan_fingerprint)):
            if not is_valid_sha256_hex(value):
                raise RobustnessValidationError(f"RobustnessSpec.{field_name} must be a 64-character lowercase hex SHA-256 digest, got {value!r}")
        if self.strategy_family_id is not None and not is_valid_sha256_hex(self.strategy_family_id):
            raise RobustnessValidationError(f"RobustnessSpec.strategy_family_id must be a valid sha256 hex id or None, got {self.strategy_family_id!r}")
        if not self.instrument_identity:
            raise RobustnessValidationError("RobustnessSpec.instrument_identity must not be empty")
        if self.minimum_fold_count < 1:
            raise RobustnessValidationError(f"RobustnessSpec.minimum_fold_count must be >= 1, got {self.minimum_fold_count}")
        if self.minimum_trade_count < 1:
            raise RobustnessValidationError(f"RobustnessSpec.minimum_trade_count must be >= 1, got {self.minimum_trade_count}")
        if self.minimum_effective_sample_size < 1:
            raise RobustnessValidationError(f"RobustnessSpec.minimum_effective_sample_size must be >= 1, got {self.minimum_effective_sample_size}")
        if self.seed < 0:
            raise RobustnessValidationError(f"RobustnessSpec.seed must be >= 0, got {self.seed}")
        if not self.stress_scenarios:
            raise RobustnessValidationError("RobustnessSpec.stress_scenarios must not be empty")
        if len({s.name for s in self.stress_scenarios}) != len(self.stress_scenarios):
            raise RobustnessValidationError("RobustnessSpec.stress_scenarios names must be unique")
        if not any(s.name == "base_cost" for s in self.stress_scenarios):
            raise RobustnessValidationError("RobustnessSpec.stress_scenarios must always include a scenario named 'base_cost' as the reference point")
        if not self.regime_definitions:
            raise RobustnessValidationError("RobustnessSpec.regime_definitions must not be empty")
        if len({r.dimension for r in self.regime_definitions}) != len(self.regime_definitions):
            raise RobustnessValidationError("RobustnessSpec.regime_definitions dimensions must be unique")
        if len({p.axis for p in self.perturbations}) != len(self.perturbations):
            raise RobustnessValidationError("RobustnessSpec.perturbations axes must be unique")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "source_backtest_id": self.source_backtest_id,
            "dataset_content_id": self.dataset_content_id, "split_plan_fingerprint": self.split_plan_fingerprint,
            "instrument_identity": self.instrument_identity, "bar_interval": self.bar_interval.value,
            "return_series_kind": self.return_series_kind.value, "bootstrap_spec": self.bootstrap_spec.to_json_dict(),
            "seed": self.seed, "multiple_testing_correction": self.multiple_testing_correction.value,
            "strategy_family_id": self.strategy_family_id, "minimum_fold_count": self.minimum_fold_count,
            "minimum_trade_count": self.minimum_trade_count, "minimum_effective_sample_size": self.minimum_effective_sample_size,
            "stability_thresholds": self.stability_thresholds.to_json_dict(),
            # MUST preserve caller-declared order: `sensitivity.py`/`stress.py`/
            # `regimes.py` each build their result tuples by iterating
            # `spec.stress_scenarios`/`perturbations`/`regime_definitions`
            # POSITIONALLY, and `verification.verify_robustness` reloads a
            # RobustnessSpec from its persisted JSON (via `from_json_dict`) to
            # independently recompute those same reports for comparison against
            # what the forward pass persisted. An earlier version of this method
            # SORTED these fields (to make `robustness_id` order-independent --
            # see `to_identity_payload` below, where that canonicalization
            # correctly belongs instead): because the forward pass computes
            # results from the spec in its ORIGINAL construction order while a
            # reloaded spec would then be reordered alphabetically, the
            # recomputed report's result tuples came back in a different order
            # than the persisted ones -- three CRITICAL, spurious
            # "recomputed result does not match the persisted artifact" findings
            # (`stress_report_mismatch`, `sensitivity_report_mismatch`,
            # `regime_report_mismatch`) on every real run, caught by the
            # acceptance-workflow integration test failing (release-audit
            # regression: introduced and then caught within the same audit
            # pass). The durable/round-tripped representation must always be
            # order-preserving; only `to_identity_payload` may canonicalize.
            "stress_scenarios": [s.to_json_dict() for s in self.stress_scenarios],
            "regime_definitions": [r.to_json_dict() for r in self.regime_definitions],
            "promotion_policy": self.promotion_policy.to_json_dict(),
            "perturbations": [p.to_json_dict() for p in self.perturbations],
        }

    def to_identity_payload(self) -> dict[str, object]:
        """Unlike `to_json_dict` (which MUST preserve caller-declared order
        for round-trip/recomputation fidelity -- see there), `robustness_id`
        must NOT depend on the declaration order of fields that are
        semantically unordered sets (each already validated unique by its own
        sub-key in `__post_init__`): `stress_scenarios` (by name),
        `regime_definitions` (by dimension), `perturbations` (by axis, and
        each axis's own `relative_deltas` by value), and `promotion_policy`'s
        `gates` (by name). This mirrors `multiple_testing.StrategyFamily.
        to_identity_payload`'s identical, already-established pattern of a
        SEPARATE, canonicalized identity view distinct from the durable
        `to_json_dict` representation."""
        return {
            "source_backtest_id": self.source_backtest_id,
            "dataset_content_id": self.dataset_content_id, "split_plan_fingerprint": self.split_plan_fingerprint,
            "instrument_identity": self.instrument_identity, "bar_interval": self.bar_interval.value,
            "return_series_kind": self.return_series_kind.value, "bootstrap_spec": self.bootstrap_spec.to_json_dict(),
            "seed": self.seed, "multiple_testing_correction": self.multiple_testing_correction.value,
            "strategy_family_id": self.strategy_family_id, "minimum_fold_count": self.minimum_fold_count,
            "minimum_trade_count": self.minimum_trade_count, "minimum_effective_sample_size": self.minimum_effective_sample_size,
            "stability_thresholds": self.stability_thresholds.to_json_dict(),
            "stress_scenarios": [s.to_json_dict() for s in sorted(self.stress_scenarios, key=lambda s: s.name)],
            "regime_definitions": [r.to_json_dict() for r in sorted(self.regime_definitions, key=lambda r: r.dimension.value)],
            "promotion_policy": self.promotion_policy.to_identity_payload(),
            "perturbations": [p.to_identity_payload() for p in sorted(self.perturbations, key=lambda p: p.axis.value)],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RobustnessSpec:
        require_schema_version(raw, supported=ROBUSTNESS_SPEC_SCHEMA_VERSION, context="RobustnessSpec")
        return cls(
            schema_version=ROBUSTNESS_SPEC_SCHEMA_VERSION, source_backtest_id=str(raw["source_backtest_id"]),
            dataset_content_id=str(raw["dataset_content_id"]), split_plan_fingerprint=str(raw["split_plan_fingerprint"]),
            instrument_identity=str(raw["instrument_identity"]), bar_interval=Timeframe(raw["bar_interval"]),
            return_series_kind=ReturnSeriesKind(raw["return_series_kind"]),
            bootstrap_spec=BootstrapSpec.from_json_dict(as_json_dict(raw["bootstrap_spec"], field_name="bootstrap_spec")),
            seed=int(str(raw["seed"])), multiple_testing_correction=MultipleTestingCorrectionKind(raw["multiple_testing_correction"]),
            strategy_family_id=(None if raw.get("strategy_family_id") is None else str(raw["strategy_family_id"])),
            minimum_fold_count=int(str(raw["minimum_fold_count"])), minimum_trade_count=int(str(raw["minimum_trade_count"])),
            minimum_effective_sample_size=int(str(raw["minimum_effective_sample_size"])),
            stability_thresholds=StabilityThresholds.from_json_dict(as_json_dict(raw["stability_thresholds"], field_name="stability_thresholds")),
            stress_scenarios=tuple(StressScenarioSpec.from_json_dict(as_json_dict(s, field_name="stress_scenarios[]")) for s in as_json_list(raw["stress_scenarios"], field_name="stress_scenarios")),
            regime_definitions=tuple(RegimeDefinitionSpec.from_json_dict(as_json_dict(r, field_name="regime_definitions[]")) for r in as_json_list(raw["regime_definitions"], field_name="regime_definitions")),
            promotion_policy=PromotionPolicySpec.from_json_dict(as_json_dict(raw["promotion_policy"], field_name="promotion_policy")),
            # Release-audit finding (Section 1): `raw.get("perturbations") or
            # [...]` previously treated an explicitly-persisted EMPTY list
            # (a legal, meaningful RobustnessSpec.perturbations=() -- "run
            # with zero sensitivity axes") the same as a WHOLLY ABSENT key,
            # silently substituting DEFAULT_PERTURBATIONS in both cases and
            # breaking JSON round-trip fidelity (and, transitively,
            # robustness_id) for that legal value. `in raw` distinguishes
            # "key omitted" (apply the default, for backward compatibility
            # with payloads written before this field existed) from "key
            # present" (use exactly what was persisted, including `[]`).
            perturbations=tuple(
                PerturbationSpec.from_json_dict(as_json_dict(p, field_name="perturbations[]"))
                for p in as_json_list(
                    raw["perturbations"] if "perturbations" in raw else [p.to_json_dict() for p in DEFAULT_PERTURBATIONS],
                    field_name="perturbations",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class RobustnessIdentity:
    schema_version: int
    robustness_id: str

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "robustness_id": self.robustness_id}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RobustnessIdentity:
        require_schema_version(raw, supported=1, context="RobustnessIdentity")
        return cls(schema_version=1, robustness_id=str(raw["robustness_id"]))


def compute_robustness_identity(spec: RobustnessSpec) -> RobustnessIdentity:
    """Pure function: `RobustnessSpec` -> `RobustnessIdentity`. Mirrors
    `backtesting.specs.compute_backtest_identity` exactly."""
    payload = dict(spec.to_identity_payload())
    payload["identity_schema_version"] = 1
    robustness_id = fingerprint_json(payload)
    return RobustnessIdentity(schema_version=1, robustness_id=robustness_id)


__all__ = [
    "DEFAULT_PERTURBATIONS",
    "DEFAULT_PROMOTION_GATES",
    "DEFAULT_REGIME_DEFINITIONS",
    "DEFAULT_STRESS_SCENARIOS",
    "ROBUSTNESS_SPEC_SCHEMA_VERSION",
    "BootstrapSpec",
    "PerturbationSpec",
    "PromotionGateSpec",
    "PromotionPolicySpec",
    "RegimeDefinitionSpec",
    "RobustnessIdentity",
    "RobustnessSpec",
    "StabilityThresholds",
    "StressScenarioSpec",
    "compute_robustness_identity",
]
