"""Immutable specifications for the financial evaluation / backtesting
framework (Milestone 5, Section 4). Every spec here is a frozen, slotted
dataclass with an explicit `__post_init__` validator -- nothing is an
arbitrary, unvalidated dict at a public boundary. `BacktestSpec` is the
top-level, content-addressed specification; every other spec in this
module is embedded within it and therefore participates in its identity.

`DeterminismPolicy` is imported directly from `quant_platform.calibration.
models` -- reused unchanged, never redefined a second time."""

from __future__ import annotations

import math
from dataclasses import dataclass

from quant_platform.backtesting.models import (
    SUPPORTED_DECISION_TIMESTAMP_POLICIES,
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
from quant_platform.calibration.models import DeterminismPolicy
from quant_platform.core.exceptions import BacktestValidationError
from quant_platform.core.types import Timeframe
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

BACKTEST_SPEC_SCHEMA_VERSION = 1


def _finite(value: float, *, field_name: str) -> None:
    if not math.isfinite(value):
        raise BacktestValidationError(f"{field_name} must be finite, got {value!r}")


def _non_negative(value: float, *, field_name: str) -> None:
    _finite(value, field_name=field_name)
    if value < 0.0:
        raise BacktestValidationError(f"{field_name} must be >= 0, got {value!r}")


def _unit_interval(value: float, *, field_name: str) -> None:
    _finite(value, field_name=field_name)
    if not (0.0 <= value <= 1.0):
        raise BacktestValidationError(f"{field_name} must be in [0, 1], got {value!r}")


# --------------------------------------------------------------------------
# Cost model specs (Section 12/13)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SpreadSpec:
    kind: SpreadModelKind
    price_units: float | None = None
    basis_points: float | None = None

    def __post_init__(self) -> None:
        if self.kind is SpreadModelKind.ZERO or self.kind is SpreadModelKind.BID_ASK_OBSERVED:
            if self.price_units is not None or self.basis_points is not None:
                raise BacktestValidationError(f"SpreadSpec: price_units/basis_points must be None when kind={self.kind.value!r}")
        elif self.kind is SpreadModelKind.FIXED_PRICE_UNITS:
            if self.price_units is None:
                raise BacktestValidationError("SpreadSpec.price_units is required when kind=fixed_price_units")
            _non_negative(self.price_units, field_name="SpreadSpec.price_units")
            if self.basis_points is not None:
                raise BacktestValidationError("SpreadSpec.basis_points must be None when kind=fixed_price_units")
        elif self.kind is SpreadModelKind.FIXED_BASIS_POINTS:
            if self.basis_points is None:
                raise BacktestValidationError("SpreadSpec.basis_points is required when kind=fixed_basis_points")
            _non_negative(self.basis_points, field_name="SpreadSpec.basis_points")
            if self.price_units is not None:
                raise BacktestValidationError("SpreadSpec.price_units must be None when kind=fixed_basis_points")

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "price_units": self.price_units, "basis_points": self.basis_points}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SpreadSpec:
        return cls(
            kind=SpreadModelKind(raw["kind"]),
            price_units=(None if raw.get("price_units") is None else float(str(raw["price_units"]))),
            basis_points=(None if raw.get("basis_points") is None else float(str(raw["basis_points"]))),
        )


@dataclass(frozen=True, slots=True)
class CommissionSpec:
    kind: CommissionModelKind
    per_side_basis_points: float | None = None
    fixed_per_trade: float | None = None

    def __post_init__(self) -> None:
        if self.kind is CommissionModelKind.ZERO:
            if self.per_side_basis_points is not None or self.fixed_per_trade is not None:
                raise BacktestValidationError("CommissionSpec: fields must be None when kind=zero")
        elif self.kind is CommissionModelKind.PER_SIDE_BASIS_POINTS:
            if self.per_side_basis_points is None:
                raise BacktestValidationError("CommissionSpec.per_side_basis_points is required when kind=per_side_basis_points")
            _non_negative(self.per_side_basis_points, field_name="CommissionSpec.per_side_basis_points")
            if self.fixed_per_trade is not None:
                raise BacktestValidationError("CommissionSpec.fixed_per_trade must be None when kind=per_side_basis_points")
        elif self.kind is CommissionModelKind.FIXED_PER_TRADE:
            if self.fixed_per_trade is None:
                raise BacktestValidationError("CommissionSpec.fixed_per_trade is required when kind=fixed_per_trade")
            _non_negative(self.fixed_per_trade, field_name="CommissionSpec.fixed_per_trade")
            if self.per_side_basis_points is not None:
                raise BacktestValidationError("CommissionSpec.per_side_basis_points must be None when kind=fixed_per_trade")

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "per_side_basis_points": self.per_side_basis_points, "fixed_per_trade": self.fixed_per_trade}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CommissionSpec:
        return cls(
            kind=CommissionModelKind(raw["kind"]),
            per_side_basis_points=(None if raw.get("per_side_basis_points") is None else float(str(raw["per_side_basis_points"]))),
            fixed_per_trade=(None if raw.get("fixed_per_trade") is None else float(str(raw["fixed_per_trade"]))),
        )


@dataclass(frozen=True, slots=True)
class SlippageSpec:
    """Section 13: deterministic slippage only in this milestone -- no
    seeded-stochastic model is implemented (documented limitation)."""

    kind: SlippageModelKind
    basis_points: float | None = None
    price_units: float | None = None

    def __post_init__(self) -> None:
        if self.kind is SlippageModelKind.ZERO:
            if self.basis_points is not None or self.price_units is not None:
                raise BacktestValidationError("SlippageSpec: fields must be None when kind=zero")
        elif self.kind is SlippageModelKind.FIXED_BASIS_POINTS:
            if self.basis_points is None:
                raise BacktestValidationError("SlippageSpec.basis_points is required when kind=fixed_basis_points")
            _non_negative(self.basis_points, field_name="SlippageSpec.basis_points")
            if self.price_units is not None:
                raise BacktestValidationError("SlippageSpec.price_units must be None when kind=fixed_basis_points")
        elif self.kind is SlippageModelKind.FIXED_PRICE_UNITS:
            if self.price_units is None:
                raise BacktestValidationError("SlippageSpec.price_units is required when kind=fixed_price_units")
            _non_negative(self.price_units, field_name="SlippageSpec.price_units")
            if self.basis_points is not None:
                raise BacktestValidationError("SlippageSpec.basis_points must be None when kind=fixed_price_units")

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "basis_points": self.basis_points, "price_units": self.price_units}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SlippageSpec:
        return cls(
            kind=SlippageModelKind(raw["kind"]),
            basis_points=(None if raw.get("basis_points") is None else float(str(raw["basis_points"]))),
            price_units=(None if raw.get("price_units") is None else float(str(raw["price_units"]))),
        )


@dataclass(frozen=True, slots=True)
class FinancingSpec:
    kind: FinancingModelKind
    daily_basis_points: float | None = None

    def __post_init__(self) -> None:
        if self.kind is FinancingModelKind.NONE:
            if self.daily_basis_points is not None:
                raise BacktestValidationError("FinancingSpec.daily_basis_points must be None when kind=none")
        elif self.kind is FinancingModelKind.FIXED_DAILY_BASIS_POINTS:
            if self.daily_basis_points is None:
                raise BacktestValidationError("FinancingSpec.daily_basis_points is required when kind=fixed_daily_basis_points")
            _finite(self.daily_basis_points, field_name="FinancingSpec.daily_basis_points")

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "daily_basis_points": self.daily_basis_points}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FinancingSpec:
        return cls(
            kind=FinancingModelKind(raw["kind"]),
            daily_basis_points=(None if raw.get("daily_basis_points") is None else float(str(raw["daily_basis_points"]))),
        )


@dataclass(frozen=True, slots=True)
class CostSensitivityScenario:
    """Section 20: a pre-declared, named cost-scenario multiplier --
    transparency, never post-hoc selection."""

    name: str
    spread_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    commission_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise BacktestValidationError("CostSensitivityScenario.name must not be empty")
        for field_name, value in (
            ("spread_multiplier", self.spread_multiplier), ("slippage_multiplier", self.slippage_multiplier),
            ("commission_multiplier", self.commission_multiplier),
        ):
            _non_negative(value, field_name=f"CostSensitivityScenario.{field_name}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "spread_multiplier": self.spread_multiplier,
            "slippage_multiplier": self.slippage_multiplier, "commission_multiplier": self.commission_multiplier,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CostSensitivityScenario:
        return cls(
            name=str(raw["name"]), spread_multiplier=float(str(raw.get("spread_multiplier", 1.0))),
            slippage_multiplier=float(str(raw.get("slippage_multiplier", 1.0))),
            commission_multiplier=float(str(raw.get("commission_multiplier", 1.0))),
        )


DEFAULT_COST_SENSITIVITY_SCENARIOS: tuple[CostSensitivityScenario, ...] = (
    CostSensitivityScenario(name="zero_cost", spread_multiplier=0.0, slippage_multiplier=0.0, commission_multiplier=0.0),
    CostSensitivityScenario(name="base_cost"),
    CostSensitivityScenario(name="1.5x_spread", spread_multiplier=1.5),
    CostSensitivityScenario(name="2x_spread", spread_multiplier=2.0),
    CostSensitivityScenario(name="increased_slippage", slippage_multiplier=2.0),
)


# --------------------------------------------------------------------------
# Signal mapping / entry / exit specs (Sections 8/10/11)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SignalMappingSpec:
    kind: SignalMappingPolicyKind
    probability_band_long_min: float | None = None
    probability_band_short_max: float | None = None
    confidence_floor: float | None = None
    uncertainty_ceiling: float | None = None

    def __post_init__(self) -> None:
        needs_bands = self.kind is SignalMappingPolicyKind.PROBABILITY_BANDS
        if needs_bands:
            if self.probability_band_long_min is None or self.probability_band_short_max is None:
                raise BacktestValidationError(
                    "SignalMappingSpec.probability_band_long_min/probability_band_short_max are both required "
                    "when kind=probability_bands"
                )
            _unit_interval(self.probability_band_long_min, field_name="SignalMappingSpec.probability_band_long_min")
            _unit_interval(self.probability_band_short_max, field_name="SignalMappingSpec.probability_band_short_max")
            if self.probability_band_short_max >= self.probability_band_long_min:
                raise BacktestValidationError(
                    f"SignalMappingSpec.probability_band_short_max ({self.probability_band_short_max}) must be < "
                    f"probability_band_long_min ({self.probability_band_long_min})"
                )
        elif self.probability_band_long_min is not None or self.probability_band_short_max is not None:
            raise BacktestValidationError("SignalMappingSpec probability bands must be None unless kind=probability_bands")

        needs_confidence = self.kind in (SignalMappingPolicyKind.CONFIDENCE_FLOOR, SignalMappingPolicyKind.COMBINED_CONFIDENCE_UNCERTAINTY)
        if needs_confidence:
            if self.confidence_floor is None:
                raise BacktestValidationError(f"SignalMappingSpec.confidence_floor is required when kind={self.kind.value!r}")
            _unit_interval(self.confidence_floor, field_name="SignalMappingSpec.confidence_floor")
        elif self.confidence_floor is not None:
            raise BacktestValidationError("SignalMappingSpec.confidence_floor must be None unless a confidence-filtering kind is used")

        needs_uncertainty = self.kind in (SignalMappingPolicyKind.UNCERTAINTY_CEILING, SignalMappingPolicyKind.COMBINED_CONFIDENCE_UNCERTAINTY)
        if needs_uncertainty:
            if self.uncertainty_ceiling is None:
                raise BacktestValidationError(f"SignalMappingSpec.uncertainty_ceiling is required when kind={self.kind.value!r}")
            _unit_interval(self.uncertainty_ceiling, field_name="SignalMappingSpec.uncertainty_ceiling")
        elif self.uncertainty_ceiling is not None:
            raise BacktestValidationError("SignalMappingSpec.uncertainty_ceiling must be None unless an uncertainty-filtering kind is used")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value, "probability_band_long_min": self.probability_band_long_min,
            "probability_band_short_max": self.probability_band_short_max, "confidence_floor": self.confidence_floor,
            "uncertainty_ceiling": self.uncertainty_ceiling,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SignalMappingSpec:
        def _opt(key: str) -> float | None:
            return None if raw.get(key) is None else float(str(raw[key]))

        return cls(
            kind=SignalMappingPolicyKind(raw["kind"]), probability_band_long_min=_opt("probability_band_long_min"),
            probability_band_short_max=_opt("probability_band_short_max"), confidence_floor=_opt("confidence_floor"),
            uncertainty_ceiling=_opt("uncertainty_ceiling"),
        )


@dataclass(frozen=True, slots=True)
class EntrySpec:
    """Section 7/10: the earliest executable fill defaults to bar t+1
    open. `delay_bars` >= 1 is REQUIRED unless `allow_same_bar_close` is
    explicitly set -- Section 1's "do not allow bar t close entry by
    default" is enforced structurally here, not merely documented."""

    kind: EntryPolicyKind
    delay_bars: int = 1
    allow_same_bar_close: bool = False

    def __post_init__(self) -> None:
        if self.delay_bars < 0:
            raise BacktestValidationError(f"EntrySpec.delay_bars must be >= 0, got {self.delay_bars}")
        if self.delay_bars == 0 and not self.allow_same_bar_close:
            raise BacktestValidationError(
                "EntrySpec.delay_bars=0 requires allow_same_bar_close=True -- entry at the SAME bar the "
                "prediction was generated from is a look-ahead risk unless explicitly, deliberately permitted "
                "(Section 1/7: 'do not allow bar t close entry by default')"
            )
        if self.kind is EntryPolicyKind.DELAYED_BAR and self.delay_bars < 1:
            raise BacktestValidationError("EntrySpec.delay_bars must be >= 1 when kind=delayed_bar")

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "delay_bars": self.delay_bars, "allow_same_bar_close": self.allow_same_bar_close}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> EntrySpec:
        return cls(
            kind=EntryPolicyKind(raw["kind"]), delay_bars=int(str(raw.get("delay_bars", 1))),
            allow_same_bar_close=bool(raw.get("allow_same_bar_close", False)),
        )


@dataclass(frozen=True, slots=True)
class ExitSpec:
    kind: ExitPolicyKind
    holding_period_bars: int | None = None
    final_trade_policy: FinalTradePolicyKind = FinalTradePolicyKind.MARK_INCOMPLETE_EXCLUDE

    def __post_init__(self) -> None:
        if self.kind is ExitPolicyKind.FIXED_HORIZON:
            if self.holding_period_bars is None:
                raise BacktestValidationError("ExitSpec.holding_period_bars is required when kind=fixed_horizon")
            if self.holding_period_bars < 1:
                raise BacktestValidationError(f"ExitSpec.holding_period_bars must be >= 1, got {self.holding_period_bars}")
        elif self.holding_period_bars is not None:
            raise BacktestValidationError("ExitSpec.holding_period_bars must be None unless kind=fixed_horizon")

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "holding_period_bars": self.holding_period_bars, "final_trade_policy": self.final_trade_policy.value}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExitSpec:
        return cls(
            kind=ExitPolicyKind(raw["kind"]),
            holding_period_bars=(None if raw.get("holding_period_bars") is None else int(str(raw["holding_period_bars"]))),
            final_trade_policy=FinalTradePolicyKind(raw.get("final_trade_policy", "mark_incomplete_exclude")),
        )


# --------------------------------------------------------------------------
# Top-level BacktestSpec (Section 4)
# --------------------------------------------------------------------------
_SUPPORTED_BAR_INTERVALS: tuple[Timeframe, ...] = tuple(Timeframe)


@dataclass(frozen=True, slots=True)
class BacktestSpec:
    """The top-level, content-addressed specification for one backtest
    run. `backtest_id` (see `compute_backtest_identity`) is a pure
    function of every field below except `schema_version` -- two
    independently constructed `BacktestSpec`s with identical field values
    always produce the same `backtest_id`, regardless of process,
    machine, or wall-clock time."""

    schema_version: int
    source_calibration_id: str
    source_experiment_id: str
    source_execution_id: str
    dataset_content_id: str
    split_plan_fingerprint: str
    instrument_identity: str
    market_timezone: str
    bar_interval: Timeframe
    decision_timestamp_policy: DecisionTimestampPolicyKind
    signal_mapping: SignalMappingSpec
    position_mode: PositionMode
    entry_spec: EntrySpec
    exit_spec: ExitSpec
    overlap_policy: OverlapPolicyKind
    price_basis: PriceBasisKind
    spread_spec: SpreadSpec
    commission_spec: CommissionSpec
    slippage_spec: SlippageSpec
    financing_spec: FinancingSpec
    return_calculation_policy: ReturnCalculationPolicyKind
    compounding_policy: CompoundingPolicyKind
    initial_notional: float
    determinism_policy: DeterminismPolicy
    respect_calibration_abstention: bool = True
    exposure_cap: float = 1.0
    annual_risk_free_rate: float = 0.0
    minimum_trades_for_valid_fold: int = 1
    timestamp_column: str = "open_time"
    seed: int = 0
    cost_sensitivity_scenarios: tuple[CostSensitivityScenario, ...] = DEFAULT_COST_SENSITIVITY_SCENARIOS

    def __post_init__(self) -> None:
        for field_name, value in (
            ("source_calibration_id", self.source_calibration_id), ("source_experiment_id", self.source_experiment_id),
            ("source_execution_id", self.source_execution_id), ("dataset_content_id", self.dataset_content_id),
            ("split_plan_fingerprint", self.split_plan_fingerprint),
        ):
            if not is_valid_sha256_hex(value):
                raise BacktestValidationError(f"BacktestSpec.{field_name} must be a 64-character lowercase hex SHA-256 digest, got {value!r}")
        if self.source_execution_id != self.source_experiment_id:
            raise BacktestValidationError(
                f"BacktestSpec.source_execution_id ({self.source_execution_id!r}) must equal source_experiment_id "
                f"({self.source_experiment_id!r}) -- this platform's ExecutionManifest is keyed by experiment_id "
                "directly (no separate execution identity is ever minted); the two fields are kept structurally "
                "distinct here only so a caller-visible field always names 'which verified EXECUTION run' this "
                "backtest is bound to, matching Section 4's explicit identity requirement."
            )
        if not self.instrument_identity:
            raise BacktestValidationError("BacktestSpec.instrument_identity must not be empty")
        if not self.market_timezone:
            raise BacktestValidationError("BacktestSpec.market_timezone must not be empty")
        if self.market_timezone != "UTC":
            raise BacktestValidationError(
                f"BacktestSpec.market_timezone must be 'UTC' (this platform's fixed, enforced timezone convention "
                f"-- see historical.timezones.require_utc), got {self.market_timezone!r}"
            )
        if self.bar_interval not in _SUPPORTED_BAR_INTERVALS:
            raise BacktestValidationError(f"BacktestSpec.bar_interval {self.bar_interval!r} is not a supported Timeframe")
        if not self.timestamp_column:
            raise BacktestValidationError("BacktestSpec.timestamp_column must not be empty")

        # Milestone 5.1, Section 5: `decision_timestamp_policy` must not be
        # decorative -- exactly one policy (AFTER_BAR_CLOSE) is actually
        # implementable from this platform's current source artifacts (see
        # `DecisionTimestampPolicyKind`'s own docstring for why the other
        # two are rejected here, fail-closed, rather than silently
        # accepted and ignored).
        if self.decision_timestamp_policy not in SUPPORTED_DECISION_TIMESTAMP_POLICIES:
            raise BacktestValidationError(
                f"BacktestSpec.decision_timestamp_policy={self.decision_timestamp_policy.value!r} is not "
                f"implementable from this platform's current source artifacts (only "
                f"{sorted(p.value for p in SUPPORTED_DECISION_TIMESTAMP_POLICIES)} supported) -- rejected at "
                "construction rather than silently treated as equivalent to a supported policy",
                context={"decision_timestamp_policy": self.decision_timestamp_policy.value},
            )
        if self.decision_timestamp_policy is DecisionTimestampPolicyKind.AFTER_BAR_CLOSE and self.entry_spec.delay_bars == 0:
            raise BacktestValidationError(
                "BacktestSpec: decision_timestamp_policy=after_bar_close requires entry_spec.delay_bars >= 1 -- "
                "a prediction derived from bar N is only knowable at bar N's CLOSE, so a fill targeting bar N "
                "itself (delay_bars=0) would necessarily use information from before it existed, an IMPOSSIBLE "
                "fill regardless of entry_spec.allow_same_bar_close (none of this platform's EntryPolicyKind "
                "conventions price a fill at exactly the signal bar's own close -- they all price off the TARGET "
                "bar's own open/mid/side-aware value, and the target bar IS the signal bar when delay_bars=0)",
                context={"delay_bars": self.entry_spec.delay_bars, "allow_same_bar_close": self.entry_spec.allow_same_bar_close},
            )

        if self.position_mode is PositionMode.LONG_FLAT and self.signal_mapping.kind is SignalMappingPolicyKind.DIRECTIONAL_LONG_SHORT:
            raise BacktestValidationError(
                "BacktestSpec: signal_mapping.kind=directional_long_short is incompatible with position_mode=long_flat"
            )
        if self.position_mode is PositionMode.LONG_SHORT and self.signal_mapping.kind is SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT:
            raise BacktestValidationError(
                "BacktestSpec: signal_mapping.kind=directional_long_flat is incompatible with position_mode=long_short "
                "(use directional_long_short, or set position_mode=long_flat)"
            )

        if self.price_basis is PriceBasisKind.BID_ASK and self.spread_spec.kind not in (SpreadModelKind.BID_ASK_OBSERVED,):
            raise BacktestValidationError(
                "BacktestSpec: price_basis=bid_ask requires spread_spec.kind=bid_ask_observed (an explicit bid/ask "
                "spread model) -- never silently synthesize bid/ask from a different spread model"
            )
        if self.price_basis is not PriceBasisKind.BID_ASK and self.spread_spec.kind is SpreadModelKind.BID_ASK_OBSERVED:
            raise BacktestValidationError("BacktestSpec: spread_spec.kind=bid_ask_observed requires price_basis=bid_ask")

        if self.initial_notional <= 0.0:
            raise BacktestValidationError(f"BacktestSpec.initial_notional must be > 0, got {self.initial_notional!r}")
        _finite(self.initial_notional, field_name="BacktestSpec.initial_notional")
        if self.exposure_cap <= 0.0:
            raise BacktestValidationError(f"BacktestSpec.exposure_cap must be > 0, got {self.exposure_cap!r}")
        _finite(self.exposure_cap, field_name="BacktestSpec.exposure_cap")
        _finite(self.annual_risk_free_rate, field_name="BacktestSpec.annual_risk_free_rate")
        if self.minimum_trades_for_valid_fold < 0:
            raise BacktestValidationError(f"BacktestSpec.minimum_trades_for_valid_fold must be >= 0, got {self.minimum_trades_for_valid_fold}")
        if self.seed < 0:
            raise BacktestValidationError(f"BacktestSpec.seed must be >= 0, got {self.seed}")

        if not self.cost_sensitivity_scenarios:
            raise BacktestValidationError("BacktestSpec.cost_sensitivity_scenarios must not be empty")
        if len({s.name for s in self.cost_sensitivity_scenarios}) != len(self.cost_sensitivity_scenarios):
            raise BacktestValidationError("BacktestSpec.cost_sensitivity_scenarios names must be unique")
        if not any(s.name == "base_cost" for s in self.cost_sensitivity_scenarios):
            raise BacktestValidationError(
                "BacktestSpec.cost_sensitivity_scenarios must always include a scenario named 'base_cost' "
                "(the actually-configured spread/commission/slippage, unmultiplied) as the reference point"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "source_calibration_id": self.source_calibration_id,
            "source_experiment_id": self.source_experiment_id, "source_execution_id": self.source_execution_id,
            "dataset_content_id": self.dataset_content_id, "split_plan_fingerprint": self.split_plan_fingerprint,
            "instrument_identity": self.instrument_identity, "market_timezone": self.market_timezone,
            "bar_interval": self.bar_interval.value, "decision_timestamp_policy": self.decision_timestamp_policy.value,
            "signal_mapping": self.signal_mapping.to_json_dict(), "position_mode": self.position_mode.value,
            "entry_spec": self.entry_spec.to_json_dict(), "exit_spec": self.exit_spec.to_json_dict(),
            "overlap_policy": self.overlap_policy.value, "price_basis": self.price_basis.value,
            "spread_spec": self.spread_spec.to_json_dict(), "commission_spec": self.commission_spec.to_json_dict(),
            "slippage_spec": self.slippage_spec.to_json_dict(), "financing_spec": self.financing_spec.to_json_dict(),
            "return_calculation_policy": self.return_calculation_policy.value, "compounding_policy": self.compounding_policy.value,
            "initial_notional": self.initial_notional, "determinism_policy": self.determinism_policy.value,
            "respect_calibration_abstention": self.respect_calibration_abstention, "exposure_cap": self.exposure_cap,
            "annual_risk_free_rate": self.annual_risk_free_rate, "minimum_trades_for_valid_fold": self.minimum_trades_for_valid_fold,
            "timestamp_column": self.timestamp_column, "seed": self.seed,
            "cost_sensitivity_scenarios": [s.to_json_dict() for s in self.cost_sensitivity_scenarios],
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = self.to_json_dict()
        del payload["schema_version"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BacktestSpec:
        require_schema_version(raw, supported=BACKTEST_SPEC_SCHEMA_VERSION, context="BacktestSpec")
        return cls(
            schema_version=BACKTEST_SPEC_SCHEMA_VERSION, source_calibration_id=str(raw["source_calibration_id"]),
            source_experiment_id=str(raw["source_experiment_id"]), source_execution_id=str(raw["source_execution_id"]),
            dataset_content_id=str(raw["dataset_content_id"]), split_plan_fingerprint=str(raw["split_plan_fingerprint"]),
            instrument_identity=str(raw["instrument_identity"]), market_timezone=str(raw["market_timezone"]),
            bar_interval=Timeframe(raw["bar_interval"]), decision_timestamp_policy=DecisionTimestampPolicyKind(raw["decision_timestamp_policy"]),
            signal_mapping=SignalMappingSpec.from_json_dict(as_json_dict(raw["signal_mapping"], field_name="signal_mapping")),
            position_mode=PositionMode(raw["position_mode"]),
            entry_spec=EntrySpec.from_json_dict(as_json_dict(raw["entry_spec"], field_name="entry_spec")),
            exit_spec=ExitSpec.from_json_dict(as_json_dict(raw["exit_spec"], field_name="exit_spec")),
            overlap_policy=OverlapPolicyKind(raw["overlap_policy"]), price_basis=PriceBasisKind(raw["price_basis"]),
            spread_spec=SpreadSpec.from_json_dict(as_json_dict(raw["spread_spec"], field_name="spread_spec")),
            commission_spec=CommissionSpec.from_json_dict(as_json_dict(raw["commission_spec"], field_name="commission_spec")),
            slippage_spec=SlippageSpec.from_json_dict(as_json_dict(raw["slippage_spec"], field_name="slippage_spec")),
            financing_spec=FinancingSpec.from_json_dict(as_json_dict(raw["financing_spec"], field_name="financing_spec")),
            return_calculation_policy=ReturnCalculationPolicyKind(raw["return_calculation_policy"]),
            compounding_policy=CompoundingPolicyKind(raw["compounding_policy"]), initial_notional=float(str(raw["initial_notional"])),
            determinism_policy=DeterminismPolicy(raw["determinism_policy"]),
            respect_calibration_abstention=bool(raw.get("respect_calibration_abstention", True)),
            exposure_cap=float(str(raw.get("exposure_cap", 1.0))), annual_risk_free_rate=float(str(raw.get("annual_risk_free_rate", 0.0))),
            minimum_trades_for_valid_fold=int(str(raw.get("minimum_trades_for_valid_fold", 1))),
            timestamp_column=str(raw.get("timestamp_column", "open_time")), seed=int(str(raw.get("seed", 0))),
            cost_sensitivity_scenarios=tuple(
                CostSensitivityScenario.from_json_dict(as_json_dict(s, field_name="cost_sensitivity_scenarios[]"))
                for s in as_json_list(raw.get("cost_sensitivity_scenarios") or [s.to_json_dict() for s in DEFAULT_COST_SENSITIVITY_SCENARIOS], field_name="cost_sensitivity_scenarios")
            ),
        )


@dataclass(frozen=True, slots=True)
class BacktestIdentity:
    schema_version: int
    backtest_id: str

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "backtest_id": self.backtest_id}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BacktestIdentity:
        require_schema_version(raw, supported=1, context="BacktestIdentity")
        return cls(schema_version=1, backtest_id=str(raw["backtest_id"]))


def compute_backtest_identity(spec: BacktestSpec) -> BacktestIdentity:
    """Pure function: `BacktestSpec` -> `BacktestIdentity`. Mirrors
    `calibration.specs.compute_calibration_identity` exactly."""
    from quant_platform.ml.fingerprints import fingerprint_json

    payload = dict(spec.to_identity_payload())
    payload["identity_schema_version"] = 1
    backtest_id = fingerprint_json(payload)
    return BacktestIdentity(schema_version=1, backtest_id=backtest_id)


__all__ = [
    "BACKTEST_SPEC_SCHEMA_VERSION",
    "DEFAULT_COST_SENSITIVITY_SCENARIOS",
    "BacktestIdentity",
    "BacktestSpec",
    "CommissionSpec",
    "CostSensitivityScenario",
    "EntrySpec",
    "ExitSpec",
    "FinancingSpec",
    "SignalMappingSpec",
    "SlippageSpec",
    "SpreadSpec",
    "compute_backtest_identity",
]
