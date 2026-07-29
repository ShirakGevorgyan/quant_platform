"""Pydantic configuration schema for Milestone 7's paper-trading/shadow-
execution engine (Section 29). Same conventions as `config.robustness_
schemas`/`config.backtesting_schemas`: every model is frozen, `extra=
"forbid"` (no unknown field can ever be silently accepted -- this is also
what makes "reject broker credentials"/"reject endpoint URLs intended for
order transmission" true BY CONSTRUCTION: no such field is ever defined
anywhere in this module, and none can be smuggled in through `extra`),
every float that must be finite declares `allow_inf_nan=False` explicitly
(pydantic's bare numeric constraints like `gt=0.0` do NOT reject `inf` on
their own -- only `nan`, since every ordering comparison against `nan` is
`False`), and every leaf schema has a `.build()` factory turning validated
config into the immutable runtime spec it describes.

REUSES `config.backtesting_schemas`' own spread/slippage/commission/
financing sub-schemas directly (`BacktestSpreadConfigSchema`, etc.) rather
than re-declaring the same `kind`/parameter fields a second time --
`specs.FinancingPolicySpec` wraps exactly two independent `backtesting.
specs.FinancingSpec` instances (long/short), so `long_financing`/
`short_financing` below are simply two separate `BacktestFinancingConfig
Schema` instances.

IDENTITY FIELDS ARE OPERATOR-TYPED, NOT DERIVED: unlike `RobustnessConfig`
(which derives `dataset_content_id`/etc. from an ALREADY-LOADED source
`BacktestSpec` at `.build()` time, since that source object exists before
a `RobustnessSpec` is ever built), there is no single upstream object here
to derive `strategy_candidate_identity`/`model_artifact_identity`/
`calibration_artifact_identity`/`feature_spec_identity` from without
already knowing them -- `eligibility.verify_paper_trading_eligibility`
needs a fully-formed `PaperTradingSpec` to check them AGAINST, not the
other way around. All six identity fields are therefore operator-typed
sha256-hex strings, exactly like `RobustnessConfig.source_backtest_id`
one layer down; this config's job is to capture the operator's CLAIM of
which artifacts a session is for, and Section 4's eligibility chain
(`create_paper_session`, fail-closed) is what independently verifies that
claim before any session is ever created -- a mismatched or fabricated
identity here is rejected at that point, never silently trusted.

`session_mode` is a `Literal` of exactly the three legal `SessionMode`
values (`replay_paper`/`forward_paper`/`shadow_observation`) -- there is
no `LIVE` value anywhere in this schema for a "reject LIVE-like mode
strings" check to even need to catch; pydantic's own `Literal` validation
already makes any other string, including anything live-sounding, a
structural parse error."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_platform.config.backtesting_schemas import (
    BacktestCommissionConfigSchema,
    BacktestFinancingConfigSchema,
    BacktestSlippageConfigSchema,
    BacktestSpreadConfigSchema,
)
from quant_platform.core.types import Timeframe
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.paper_trading.models import (
    PAPER_SESSION_SPEC_SCHEMA_VERSION,
    BarAmbiguityPolicyKind,
    ClockMode,
    MarketEventMode,
    MarkFieldKind,
    PartialFillPolicyKind,
    SessionMode,
)
from quant_platform.paper_trading.specs import (
    ExecutionPolicySpec,
    FillPolicySpec,
    FinancingPolicySpec,
    InstrumentSpec,
    LatencyPolicySpec,
    LiquidityPolicySpec,
    OrderPolicySpec,
    PaperTradingSpec,
    PositionPolicySpec,
    RiskLimitsSpec,
    SessionBoundaryPolicySpec,
    StartingPositionSpec,
)

_SHA256_HEX_PATTERN = r"^[0-9a-f]{64}$"
_TIMEFRAME_LITERAL = Literal["M1", "M5", "M15", "M30", "H1", "H4", "H12", "D1"]


def _sha256_hex_field(*, required: bool = True) -> object:
    if required:
        return Field(pattern=_SHA256_HEX_PATTERN)
    return Field(default=None, pattern=_SHA256_HEX_PATTERN)


class PaperTradingInstrumentConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    base_currency: str | None = None
    quote_currency: str = Field(min_length=1)
    contract_multiplier: float = Field(gt=0.0, allow_inf_nan=False)
    tick_size: float = Field(gt=0.0, allow_inf_nan=False)
    tick_value: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    quantity_step: float = Field(gt=0.0, allow_inf_nan=False)
    minimum_quantity: float = Field(gt=0.0, allow_inf_nan=False)
    maximum_quantity: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    price_precision: int = Field(ge=0)
    quantity_precision: int = Field(ge=0)
    margin_mode: str = Field(min_length=1)
    account_currency: str = Field(min_length=1)
    financing_convention: str = Field(min_length=1)
    trading_timezone: str = Field(min_length=1)
    session_calendar_identity: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_quantity_bounds(self) -> PaperTradingInstrumentConfigSchema:
        if self.maximum_quantity is not None and self.maximum_quantity < self.minimum_quantity:
            raise ValueError("instrument.maximum_quantity must be >= instrument.minimum_quantity")
        return self

    def build(self) -> InstrumentSpec:
        return InstrumentSpec(
            symbol=self.symbol, base_currency=self.base_currency, quote_currency=self.quote_currency, contract_multiplier=self.contract_multiplier,
            tick_size=self.tick_size, tick_value=self.tick_value, quantity_step=self.quantity_step, minimum_quantity=self.minimum_quantity,
            maximum_quantity=self.maximum_quantity, price_precision=self.price_precision, quantity_precision=self.quantity_precision,
            margin_mode=self.margin_mode, account_currency=self.account_currency, financing_convention=self.financing_convention,
            trading_timezone=self.trading_timezone, session_calendar_identity=self.session_calendar_identity,
        )


class PaperTradingStartingPositionConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_symbol: str = Field(min_length=1)
    signed_quantity: float = Field(allow_inf_nan=False)
    average_entry_price: float = Field(gt=0.0, allow_inf_nan=False)

    def build(self) -> StartingPositionSpec:
        return StartingPositionSpec(instrument_symbol=self.instrument_symbol, signed_quantity=self.signed_quantity, average_entry_price=self.average_entry_price)


class PaperTradingOrderPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    close_before_reverse: bool = True
    cooldown_bars: int = Field(default=0, ge=0)
    maximum_orders_per_event: int = Field(default=5, ge=1)
    maximum_order_rate_per_window: int = Field(default=20, ge=1)
    order_rate_window_events: int = Field(default=100, ge=1)

    def build(self) -> OrderPolicySpec:
        return OrderPolicySpec(
            close_before_reverse=self.close_before_reverse, cooldown_bars=self.cooldown_bars, maximum_orders_per_event=self.maximum_orders_per_event,
            maximum_order_rate_per_window=self.maximum_order_rate_per_window, order_rate_window_events=self.order_rate_window_events,
        )


class PaperTradingExecutionPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bar_ambiguity_policy: Literal["worst_case", "best_case", "stop_first", "target_first", "unsupported_and_reject"] = "worst_case"
    mark_field: Literal["close"] = "close"

    def build(self) -> ExecutionPolicySpec:
        return ExecutionPolicySpec(bar_ambiguity_policy=BarAmbiguityPolicyKind(self.bar_ambiguity_policy), mark_field=MarkFieldKind(self.mark_field))


class PaperTradingFillPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    partial_fill_policy: Literal["full_fill_only", "deterministic_partial"] = "full_fill_only"

    def build(self) -> FillPolicySpec:
        return FillPolicySpec(partial_fill_policy=PartialFillPolicyKind(self.partial_fill_policy))


class PaperTradingLatencyPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_to_submit_ms: int = Field(default=0, ge=0)
    submit_to_accept_ms: int = Field(default=0, ge=0)
    accept_to_fill_eligible_ms: int = Field(default=0, ge=0)

    def build(self) -> LatencyPolicySpec:
        return LatencyPolicySpec(decision_to_submit_ms=self.decision_to_submit_ms, submit_to_accept_ms=self.submit_to_accept_ms, accept_to_fill_eligible_ms=self.accept_to_fill_eligible_ms)


class PaperTradingLiquidityPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trust_disclosed_size: bool = False

    def build(self) -> LiquidityPolicySpec:
        return LiquidityPolicySpec(trust_disclosed_size=self.trust_disclosed_size)


class PaperTradingPositionPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    single_instrument_only: bool = True
    reduce_only_default: bool = False

    def build(self) -> PositionPolicySpec:
        return PositionPolicySpec(single_instrument_only=self.single_instrument_only, reduce_only_default=self.reduce_only_default)


class PaperTradingRiskLimitsConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    maximum_signed_position: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    maximum_absolute_position: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    maximum_gross_exposure: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    maximum_order_quantity: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    maximum_order_notional: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    maximum_turnover: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    maximum_daily_loss: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    maximum_drawdown_fraction: float | None = Field(default=None, gt=0.0, le=1.0, allow_inf_nan=False)
    maximum_realized_loss: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    maximum_unrealized_loss: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    maximum_rejected_order_count: int | None = Field(default=None, ge=1)
    maximum_consecutive_execution_failures: int | None = Field(default=None, ge=1)
    maximum_stale_data_seconds: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    maximum_reconciliation_discrepancy: float = Field(default=1e-6, ge=0.0, allow_inf_nan=False)

    def build(self) -> RiskLimitsSpec:
        return RiskLimitsSpec(
            maximum_signed_position=self.maximum_signed_position, maximum_absolute_position=self.maximum_absolute_position,
            maximum_gross_exposure=self.maximum_gross_exposure, maximum_order_quantity=self.maximum_order_quantity,
            maximum_order_notional=self.maximum_order_notional, maximum_turnover=self.maximum_turnover, maximum_daily_loss=self.maximum_daily_loss,
            maximum_drawdown_fraction=self.maximum_drawdown_fraction, maximum_realized_loss=self.maximum_realized_loss,
            maximum_unrealized_loss=self.maximum_unrealized_loss, maximum_rejected_order_count=self.maximum_rejected_order_count,
            maximum_consecutive_execution_failures=self.maximum_consecutive_execution_failures, maximum_stale_data_seconds=self.maximum_stale_data_seconds,
            maximum_reconciliation_discrepancy=self.maximum_reconciliation_discrepancy,
        )


class PaperTradingSessionBoundaryPolicyConfigSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    force_close_open_orders_at_end_of_stream: bool = True
    force_close_open_positions_at_end_of_stream: bool = False
    allow_unresolved_working_orders_at_completion: bool = False

    def build(self) -> SessionBoundaryPolicySpec:
        return SessionBoundaryPolicySpec(
            force_close_open_orders_at_end_of_stream=self.force_close_open_orders_at_end_of_stream,
            force_close_open_positions_at_end_of_stream=self.force_close_open_positions_at_end_of_stream,
            allow_unresolved_working_orders_at_completion=self.allow_unresolved_working_orders_at_completion,
        )


class PaperTradingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ml_artifacts_root: Path
    """Also where `paper_trading.manifests.PaperSessionManifestStore`/
    `persistence.PaperSessionEventStore` persist session data (a
    `paper_sessions/` subdirectory of this same root) -- mirrors
    `RobustnessConfig.ml_artifacts_root`'s identical dual role one layer
    down, never a hand-typed, driftable second path."""
    historical_storage_root: Path
    research_storage_root: Path

    verified_robustness_id: str = _sha256_hex_field()  # type: ignore[assignment]
    verified_promotion_decision_id: str = _sha256_hex_field()  # type: ignore[assignment]
    strategy_candidate_identity: str = _sha256_hex_field()  # type: ignore[assignment]
    model_artifact_identity: str = _sha256_hex_field()  # type: ignore[assignment]
    calibration_artifact_identity: str | None = _sha256_hex_field(required=False)  # type: ignore[assignment]
    feature_spec_identity: str = _sha256_hex_field()  # type: ignore[assignment]

    instrument: PaperTradingInstrumentConfigSchema
    price_precision: int = Field(ge=0)
    quantity_precision: int = Field(ge=0)
    session_mode: Literal["replay_paper", "forward_paper", "shadow_observation"] = "replay_paper"
    market_event_mode: Literal["quote", "bar"] = "bar"
    bar_interval: _TIMEFRAME_LITERAL | None = "H1"
    clock_mode: Literal["replay", "forward", "manual_test"] = "replay"
    starting_cash: float = Field(gt=0.0, allow_inf_nan=False)
    starting_positions: list[PaperTradingStartingPositionConfigSchema] = Field(default_factory=list)
    """An omitted field and an explicitly empty list (`[]`) are treated
    identically -- both build zero starting positions -- exactly matching
    `PaperTradingSpec.from_json_dict`'s own tolerant handling of this same
    field one layer down; there is no domain behavior that distinguishes
    the two here, so no distinction is manufactured at this layer either."""

    order_policy: PaperTradingOrderPolicyConfigSchema = Field(default_factory=PaperTradingOrderPolicyConfigSchema)
    execution_policy: PaperTradingExecutionPolicyConfigSchema = Field(default_factory=PaperTradingExecutionPolicyConfigSchema)
    fill_policy: PaperTradingFillPolicyConfigSchema = Field(default_factory=PaperTradingFillPolicyConfigSchema)
    spread: BacktestSpreadConfigSchema = Field(default_factory=BacktestSpreadConfigSchema)
    slippage: BacktestSlippageConfigSchema = Field(default_factory=BacktestSlippageConfigSchema)
    commission: BacktestCommissionConfigSchema = Field(default_factory=BacktestCommissionConfigSchema)
    long_financing: BacktestFinancingConfigSchema = Field(default_factory=BacktestFinancingConfigSchema)
    short_financing: BacktestFinancingConfigSchema = Field(default_factory=BacktestFinancingConfigSchema)
    latency_policy: PaperTradingLatencyPolicyConfigSchema = Field(default_factory=PaperTradingLatencyPolicyConfigSchema)
    liquidity_policy: PaperTradingLiquidityPolicyConfigSchema = Field(default_factory=PaperTradingLiquidityPolicyConfigSchema)
    position_policy: PaperTradingPositionPolicyConfigSchema = Field(default_factory=PaperTradingPositionPolicyConfigSchema)
    risk_limits: PaperTradingRiskLimitsConfigSchema = Field(default_factory=PaperTradingRiskLimitsConfigSchema)
    session_boundary_policy: PaperTradingSessionBoundaryPolicyConfigSchema = Field(default_factory=PaperTradingSessionBoundaryPolicyConfigSchema)
    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_identities_and_bar_interval(self) -> PaperTradingConfig:
        for field_name, value in (
            ("verified_robustness_id", self.verified_robustness_id), ("verified_promotion_decision_id", self.verified_promotion_decision_id),
            ("strategy_candidate_identity", self.strategy_candidate_identity), ("model_artifact_identity", self.model_artifact_identity),
            ("feature_spec_identity", self.feature_spec_identity),
        ):
            if not is_valid_sha256_hex(value):
                raise ValueError(f"{field_name} must be a 64-character lowercase hex SHA-256 digest, got {value!r}")
        if self.calibration_artifact_identity is not None and not is_valid_sha256_hex(self.calibration_artifact_identity):
            raise ValueError(f"calibration_artifact_identity must be a valid sha256 hex digest or omitted, got {self.calibration_artifact_identity!r}")
        if self.market_event_mode == "bar" and self.bar_interval is None:
            raise ValueError("bar_interval is required when market_event_mode='bar'")
        if self.market_event_mode == "quote" and self.bar_interval is not None:
            raise ValueError("bar_interval must be omitted (null) when market_event_mode='quote'")
        return self

    def build(self) -> PaperTradingSpec:
        return PaperTradingSpec(
            schema_version=PAPER_SESSION_SPEC_SCHEMA_VERSION, verified_robustness_id=self.verified_robustness_id,
            verified_promotion_decision_id=self.verified_promotion_decision_id, strategy_candidate_identity=self.strategy_candidate_identity,
            model_artifact_identity=self.model_artifact_identity, calibration_artifact_identity=self.calibration_artifact_identity,
            feature_spec_identity=self.feature_spec_identity, instrument=self.instrument.build(), price_precision=self.price_precision,
            quantity_precision=self.quantity_precision, session_mode=SessionMode(self.session_mode), market_event_mode=MarketEventMode(self.market_event_mode),
            bar_interval=(None if self.bar_interval is None else Timeframe(self.bar_interval)), clock_mode=ClockMode(self.clock_mode),
            starting_cash=self.starting_cash, starting_positions=tuple(p.build() for p in self.starting_positions), order_policy=self.order_policy.build(),
            execution_policy=self.execution_policy.build(), fill_policy=self.fill_policy.build(), spread_policy=self.spread.build(),
            slippage_policy=self.slippage.build(), commission_policy=self.commission.build(),
            financing_policy=FinancingPolicySpec(long_financing=self.long_financing.build(), short_financing=self.short_financing.build()),
            latency_policy=self.latency_policy.build(), liquidity_policy=self.liquidity_policy.build(), position_policy=self.position_policy.build(),
            risk_limits=self.risk_limits.build(), session_boundary_policy=self.session_boundary_policy.build(), seed=self.seed,
        )


__all__ = [
    "PaperTradingConfig",
    "PaperTradingExecutionPolicyConfigSchema",
    "PaperTradingFillPolicyConfigSchema",
    "PaperTradingInstrumentConfigSchema",
    "PaperTradingLatencyPolicyConfigSchema",
    "PaperTradingLiquidityPolicyConfigSchema",
    "PaperTradingOrderPolicyConfigSchema",
    "PaperTradingPositionPolicyConfigSchema",
    "PaperTradingRiskLimitsConfigSchema",
    "PaperTradingSessionBoundaryPolicyConfigSchema",
    "PaperTradingStartingPositionConfigSchema",
]
