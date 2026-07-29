"""Immutable specifications for `quant_platform.paper_trading` (Milestone
7, Section 3/14). Every spec here is a frozen, slotted dataclass with an
explicit `__post_init__` validator. `PaperTradingSpec` is the top-level,
content-addressed specification; every other spec in this module is
embedded within it and therefore participates in its identity.

DURABLE ORDER VERSUS IDENTITY CANONICALIZATION (release-audit lesson
carried forward from Milestone 6's own regression): `to_json_dict()`
methods in this module ALWAYS preserve caller-declared order -- this is
the durable, round-tripped representation, and downstream code may
iterate these fields positionally. Canonicalization of genuinely
UNORDERED fields (sorting so declaration order does not affect identity)
happens ONLY inside `to_identity_payload()`, never inside `to_json_dict()`
itself. See `robustness.specs.RobustnessSpec`'s identical, hard-won
convention for the regression this avoids repeating."""

from __future__ import annotations

import math
from dataclasses import dataclass

from quant_platform.backtesting.specs import CommissionSpec, FinancingSpec, SlippageSpec, SpreadSpec
from quant_platform.core.exceptions import PaperTradingSpecError
from quant_platform.core.types import Timeframe
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.paper_trading.identity import compute_content_id
from quant_platform.paper_trading.models import (
    PAPER_SESSION_SPEC_SCHEMA_VERSION,
    BarAmbiguityPolicyKind,
    ClockMode,
    MarketEventMode,
    MarkFieldKind,
    PartialFillPolicyKind,
    SessionMode,
)


def _finite(value: float, *, field_name: str) -> None:
    if not math.isfinite(value):
        raise PaperTradingSpecError(f"{field_name} must be finite, got {value!r}")


def _positive(value: float, *, field_name: str) -> None:
    _finite(value, field_name=field_name)
    if value <= 0.0:
        raise PaperTradingSpecError(f"{field_name} must be > 0, got {value!r}")


def _non_negative(value: float, *, field_name: str) -> None:
    _finite(value, field_name=field_name)
    if value < 0.0:
        raise PaperTradingSpecError(f"{field_name} must be >= 0, got {value!r}")


def _non_negative_int(value: int, *, field_name: str) -> None:
    if value < 0:
        raise PaperTradingSpecError(f"{field_name} must be >= 0, got {value!r}")


def _positive_int(value: int, *, field_name: str) -> None:
    if value < 1:
        raise PaperTradingSpecError(f"{field_name} must be >= 1, got {value!r}")


_LIVE_LIKE_TOKENS: frozenset[str] = frozenset({"live", "real", "production_trading", "broker_live"})


def _reject_live_like(value: str, *, field_name: str) -> None:
    if value.strip().lower() in _LIVE_LIKE_TOKENS:
        raise PaperTradingSpecError(f"{field_name}={value!r} looks like a LIVE-trading mode string -- this package has no LIVE mode, ever")


# --------------------------------------------------------------------------
# Instrument contract (Section 14) -- broker-neutral, XAUUSD-future-proof,
# no MT5-specific or hardcoded values.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """A broker-neutral description of one tradable instrument, generic
    enough to represent a future XAUUSD/MT5 contract without depending on
    MT5 at all -- see `docs/paper_trading_and_shadow_execution.md` for a
    worked hypothetical-XAUUSD example. Broker-specific values (tick
    size/value, margin mode, session calendar) must later come from the
    real MT5 symbol specification and may differ by broker; nothing here
    is a claim about any real broker's actual terms."""

    symbol: str
    base_currency: str | None
    """`None` for instruments without a meaningful base/quote pair split
    (e.g. a CFD-style contract on a commodity price) -- quote_currency
    alone is then the account's P&L currency."""
    quote_currency: str
    contract_multiplier: float
    tick_size: float
    tick_value: float | None
    """`None` if not yet known -- P&L must then be computed from price
    delta * contract_multiplier * quantity directly, never a fabricated
    tick_value."""
    quantity_step: float
    minimum_quantity: float
    maximum_quantity: float | None
    price_precision: int
    quantity_precision: int
    margin_mode: str
    account_currency: str
    financing_convention: str
    trading_timezone: str
    session_calendar_identity: str

    def __post_init__(self) -> None:
        if not self.symbol:
            raise PaperTradingSpecError("InstrumentSpec.symbol must not be empty")
        if not self.quote_currency:
            raise PaperTradingSpecError("InstrumentSpec.quote_currency must not be empty")
        _positive(self.contract_multiplier, field_name="InstrumentSpec.contract_multiplier")
        _positive(self.tick_size, field_name="InstrumentSpec.tick_size")
        if self.tick_value is not None:
            _positive(self.tick_value, field_name="InstrumentSpec.tick_value")
        _positive(self.quantity_step, field_name="InstrumentSpec.quantity_step")
        _positive(self.minimum_quantity, field_name="InstrumentSpec.minimum_quantity")
        if self.maximum_quantity is not None:
            _positive(self.maximum_quantity, field_name="InstrumentSpec.maximum_quantity")
            if self.maximum_quantity < self.minimum_quantity:
                raise PaperTradingSpecError("InstrumentSpec.maximum_quantity must be >= minimum_quantity")
        if self.minimum_quantity % self.quantity_step > 1e-9 and (self.quantity_step - (self.minimum_quantity % self.quantity_step)) > 1e-9:
            raise PaperTradingSpecError("InstrumentSpec.minimum_quantity must be an exact multiple of quantity_step")
        _non_negative_int(self.price_precision, field_name="InstrumentSpec.price_precision")
        _non_negative_int(self.quantity_precision, field_name="InstrumentSpec.quantity_precision")
        if not self.margin_mode:
            raise PaperTradingSpecError("InstrumentSpec.margin_mode must not be empty")
        if not self.account_currency:
            raise PaperTradingSpecError("InstrumentSpec.account_currency must not be empty")
        if not self.financing_convention:
            raise PaperTradingSpecError("InstrumentSpec.financing_convention must not be empty")
        if not self.trading_timezone:
            raise PaperTradingSpecError("InstrumentSpec.trading_timezone must not be empty")
        if not self.session_calendar_identity:
            raise PaperTradingSpecError("InstrumentSpec.session_calendar_identity must not be empty")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol, "base_currency": self.base_currency, "quote_currency": self.quote_currency,
            "contract_multiplier": self.contract_multiplier, "tick_size": self.tick_size, "tick_value": self.tick_value,
            "quantity_step": self.quantity_step, "minimum_quantity": self.minimum_quantity, "maximum_quantity": self.maximum_quantity,
            "price_precision": self.price_precision, "quantity_precision": self.quantity_precision, "margin_mode": self.margin_mode,
            "account_currency": self.account_currency, "financing_convention": self.financing_convention,
            "trading_timezone": self.trading_timezone, "session_calendar_identity": self.session_calendar_identity,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> InstrumentSpec:
        return cls(
            symbol=str(raw["symbol"]), base_currency=(None if raw.get("base_currency") is None else str(raw["base_currency"])),
            quote_currency=str(raw["quote_currency"]), contract_multiplier=float(str(raw["contract_multiplier"])),
            tick_size=float(str(raw["tick_size"])), tick_value=(None if raw.get("tick_value") is None else float(str(raw["tick_value"]))),
            quantity_step=float(str(raw["quantity_step"])), minimum_quantity=float(str(raw["minimum_quantity"])),
            maximum_quantity=(None if raw.get("maximum_quantity") is None else float(str(raw["maximum_quantity"]))),
            price_precision=int(str(raw["price_precision"])), quantity_precision=int(str(raw["quantity_precision"])),
            margin_mode=str(raw["margin_mode"]), account_currency=str(raw["account_currency"]),
            financing_convention=str(raw["financing_convention"]), trading_timezone=str(raw["trading_timezone"]),
            session_calendar_identity=str(raw["session_calendar_identity"]),
        )


# --------------------------------------------------------------------------
# Latency policy (Section 10.4) -- deterministic, applied ARITHMETICALLY to
# event time, never via real sleep/wall-clock (required for byte-identical
# replay).
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LatencyPolicySpec:
    decision_to_submit_ms: int
    submit_to_accept_ms: int
    accept_to_fill_eligible_ms: int

    def __post_init__(self) -> None:
        for name, value in (
            ("decision_to_submit_ms", self.decision_to_submit_ms), ("submit_to_accept_ms", self.submit_to_accept_ms),
            ("accept_to_fill_eligible_ms", self.accept_to_fill_eligible_ms),
        ):
            _non_negative_int(value, field_name=f"LatencyPolicySpec.{name}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "decision_to_submit_ms": self.decision_to_submit_ms, "submit_to_accept_ms": self.submit_to_accept_ms,
            "accept_to_fill_eligible_ms": self.accept_to_fill_eligible_ms,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LatencyPolicySpec:
        return cls(
            decision_to_submit_ms=int(str(raw["decision_to_submit_ms"])), submit_to_accept_ms=int(str(raw["submit_to_accept_ms"])),
            accept_to_fill_eligible_ms=int(str(raw["accept_to_fill_eligible_ms"])),
        )


# --------------------------------------------------------------------------
# Order policy (Section 9)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OrderPolicySpec:
    close_before_reverse: bool
    """`True`: a reversal (e.g. long -> short) is submitted as two orders
    (close, then open) -- never as one atomically-signed target-delta
    order. `False`: submitted as a single atomic target-delta order.
    Explicitly configured, never silently assumed either way."""
    cooldown_bars: int
    """Minimum bars/events between two orders for the SAME instrument --
    `0` disables the cooldown."""
    maximum_orders_per_event: int
    maximum_order_rate_per_window: int
    order_rate_window_events: int
    """`maximum_order_rate_per_window` orders per this many consecutive
    events, sliding -- both must be set together and are meaningless
    alone."""

    def __post_init__(self) -> None:
        _non_negative_int(self.cooldown_bars, field_name="OrderPolicySpec.cooldown_bars")
        _positive_int(self.maximum_orders_per_event, field_name="OrderPolicySpec.maximum_orders_per_event")
        _positive_int(self.maximum_order_rate_per_window, field_name="OrderPolicySpec.maximum_order_rate_per_window")
        _positive_int(self.order_rate_window_events, field_name="OrderPolicySpec.order_rate_window_events")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "close_before_reverse": self.close_before_reverse, "cooldown_bars": self.cooldown_bars,
            "maximum_orders_per_event": self.maximum_orders_per_event, "maximum_order_rate_per_window": self.maximum_order_rate_per_window,
            "order_rate_window_events": self.order_rate_window_events,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> OrderPolicySpec:
        return cls(
            close_before_reverse=bool(raw["close_before_reverse"]), cooldown_bars=int(str(raw["cooldown_bars"])),
            maximum_orders_per_event=int(str(raw["maximum_orders_per_event"])),
            maximum_order_rate_per_window=int(str(raw["maximum_order_rate_per_window"])), order_rate_window_events=int(str(raw["order_rate_window_events"])),
        )


# --------------------------------------------------------------------------
# Execution / fill / liquidity policy (Section 10-11)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ExecutionPolicySpec:
    bar_ambiguity_policy: BarAmbiguityPolicyKind
    mark_field: MarkFieldKind

    def to_json_dict(self) -> dict[str, object]:
        return {"bar_ambiguity_policy": self.bar_ambiguity_policy.value, "mark_field": self.mark_field.value}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExecutionPolicySpec:
        return cls(bar_ambiguity_policy=BarAmbiguityPolicyKind(raw["bar_ambiguity_policy"]), mark_field=MarkFieldKind(raw["mark_field"]))


DEFAULT_EXECUTION_POLICY = ExecutionPolicySpec(bar_ambiguity_policy=BarAmbiguityPolicyKind.WORST_CASE, mark_field=MarkFieldKind.CLOSE)


# --------------------------------------------------------------------------
# Financing policy (Section 16) -- long/short ASYMMETRIC, unlike
# `backtesting.specs.FinancingSpec` alone (one symmetric rate). Reuses
# that exact, already-validated per-side spec/formula for EACH direction
# rather than duplicating the formula -- the asymmetry is purely a
# policy-level choice of which side's spec applies, never a second
# implementation of the daily-rate math itself.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FinancingPolicySpec:
    long_financing: FinancingSpec
    """Applied while the simulated position is LONG. A negative
    `daily_basis_points` represents a credit (cash increases), a positive
    one a cost (cash decreases) -- see `costs.compute_financing_cash_delta`."""
    short_financing: FinancingSpec
    """Applied while the simulated position is SHORT. Configuring both
    sides identically recovers a single symmetric rate."""

    def to_json_dict(self) -> dict[str, object]:
        return {"long_financing": self.long_financing.to_json_dict(), "short_financing": self.short_financing.to_json_dict()}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FinancingPolicySpec:
        from quant_platform.ml.persistence import as_json_dict

        return cls(
            long_financing=FinancingSpec.from_json_dict(as_json_dict(raw["long_financing"], field_name="long_financing")),
            short_financing=FinancingSpec.from_json_dict(as_json_dict(raw["short_financing"], field_name="short_financing")),
        )


@dataclass(frozen=True, slots=True)
class FillPolicySpec:
    partial_fill_policy: PartialFillPolicyKind

    def to_json_dict(self) -> dict[str, object]:
        return {"partial_fill_policy": self.partial_fill_policy.value}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FillPolicySpec:
        return cls(partial_fill_policy=PartialFillPolicyKind(raw["partial_fill_policy"]))


@dataclass(frozen=True, slots=True)
class LiquidityPolicySpec:
    trust_disclosed_size: bool
    """Whether to honor a `QuoteEvent`'s own disclosed bid/ask size for
    `DETERMINISTIC_PARTIAL` fills. When `False`, or when an event
    discloses no size, liquidity is NEVER invented -- the fill falls back
    to `FULL_FILL_ONLY` behavior for that event (Section 10.8)."""

    def to_json_dict(self) -> dict[str, object]:
        return {"trust_disclosed_size": self.trust_disclosed_size}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LiquidityPolicySpec:
        return cls(trust_disclosed_size=bool(raw["trust_disclosed_size"]))


# --------------------------------------------------------------------------
# Position policy (Section 13) -- fail-closed single-instrument-per-session
# constraint, explicit per Section 13's own instruction.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PositionPolicySpec:
    single_instrument_only: bool
    reduce_only_default: bool

    def to_json_dict(self) -> dict[str, object]:
        return {"single_instrument_only": self.single_instrument_only, "reduce_only_default": self.reduce_only_default}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PositionPolicySpec:
        return cls(single_instrument_only=bool(raw["single_instrument_only"]), reduce_only_default=bool(raw["reduce_only_default"]))


DEFAULT_POSITION_POLICY = PositionPolicySpec(single_instrument_only=True, reduce_only_default=False)


# --------------------------------------------------------------------------
# Risk limits (Section 17) -- every field optional (`None` = not enforced).
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RiskLimitsSpec:
    maximum_signed_position: float | None
    maximum_absolute_position: float | None
    maximum_gross_exposure: float | None
    maximum_order_quantity: float | None
    maximum_order_notional: float | None
    maximum_turnover: float | None
    maximum_daily_loss: float | None
    maximum_drawdown_fraction: float | None
    maximum_realized_loss: float | None
    maximum_unrealized_loss: float | None
    maximum_rejected_order_count: int | None
    maximum_consecutive_execution_failures: int | None
    maximum_stale_data_seconds: float | None
    maximum_reconciliation_discrepancy: float

    def __post_init__(self) -> None:
        for name, value in (
            ("maximum_signed_position", self.maximum_signed_position), ("maximum_absolute_position", self.maximum_absolute_position),
            ("maximum_gross_exposure", self.maximum_gross_exposure), ("maximum_order_quantity", self.maximum_order_quantity),
            ("maximum_order_notional", self.maximum_order_notional), ("maximum_turnover", self.maximum_turnover),
            ("maximum_daily_loss", self.maximum_daily_loss), ("maximum_realized_loss", self.maximum_realized_loss),
            ("maximum_unrealized_loss", self.maximum_unrealized_loss), ("maximum_stale_data_seconds", self.maximum_stale_data_seconds),
        ):
            if value is not None:
                _positive(value, field_name=f"RiskLimitsSpec.{name}")
        if self.maximum_drawdown_fraction is not None:
            _finite(self.maximum_drawdown_fraction, field_name="RiskLimitsSpec.maximum_drawdown_fraction")
            if not (0.0 < self.maximum_drawdown_fraction <= 1.0):
                raise PaperTradingSpecError(f"RiskLimitsSpec.maximum_drawdown_fraction must be in (0, 1], got {self.maximum_drawdown_fraction!r}")
        if self.maximum_rejected_order_count is not None:
            _positive_int(self.maximum_rejected_order_count, field_name="RiskLimitsSpec.maximum_rejected_order_count")
        if self.maximum_consecutive_execution_failures is not None:
            _positive_int(self.maximum_consecutive_execution_failures, field_name="RiskLimitsSpec.maximum_consecutive_execution_failures")
        _non_negative(self.maximum_reconciliation_discrepancy, field_name="RiskLimitsSpec.maximum_reconciliation_discrepancy")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "maximum_signed_position": self.maximum_signed_position, "maximum_absolute_position": self.maximum_absolute_position,
            "maximum_gross_exposure": self.maximum_gross_exposure, "maximum_order_quantity": self.maximum_order_quantity,
            "maximum_order_notional": self.maximum_order_notional, "maximum_turnover": self.maximum_turnover,
            "maximum_daily_loss": self.maximum_daily_loss, "maximum_drawdown_fraction": self.maximum_drawdown_fraction,
            "maximum_realized_loss": self.maximum_realized_loss, "maximum_unrealized_loss": self.maximum_unrealized_loss,
            "maximum_rejected_order_count": self.maximum_rejected_order_count,
            "maximum_consecutive_execution_failures": self.maximum_consecutive_execution_failures,
            "maximum_stale_data_seconds": self.maximum_stale_data_seconds,
            "maximum_reconciliation_discrepancy": self.maximum_reconciliation_discrepancy,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RiskLimitsSpec:
        def _opt_f(key: str) -> float | None:
            v = raw.get(key)
            return None if v is None else float(str(v))

        def _opt_i(key: str) -> int | None:
            v = raw.get(key)
            return None if v is None else int(str(v))

        return cls(
            maximum_signed_position=_opt_f("maximum_signed_position"), maximum_absolute_position=_opt_f("maximum_absolute_position"),
            maximum_gross_exposure=_opt_f("maximum_gross_exposure"), maximum_order_quantity=_opt_f("maximum_order_quantity"),
            maximum_order_notional=_opt_f("maximum_order_notional"), maximum_turnover=_opt_f("maximum_turnover"),
            maximum_daily_loss=_opt_f("maximum_daily_loss"), maximum_drawdown_fraction=_opt_f("maximum_drawdown_fraction"),
            maximum_realized_loss=_opt_f("maximum_realized_loss"), maximum_unrealized_loss=_opt_f("maximum_unrealized_loss"),
            maximum_rejected_order_count=_opt_i("maximum_rejected_order_count"),
            maximum_consecutive_execution_failures=_opt_i("maximum_consecutive_execution_failures"),
            maximum_stale_data_seconds=_opt_f("maximum_stale_data_seconds"),
            maximum_reconciliation_discrepancy=float(str(raw.get("maximum_reconciliation_discrepancy", 1e-6))),
        )


DEFAULT_RISK_LIMITS = RiskLimitsSpec(
    maximum_signed_position=None, maximum_absolute_position=None, maximum_gross_exposure=None, maximum_order_quantity=None,
    maximum_order_notional=None, maximum_turnover=None, maximum_daily_loss=None, maximum_drawdown_fraction=None,
    maximum_realized_loss=None, maximum_unrealized_loss=None, maximum_rejected_order_count=None,
    maximum_consecutive_execution_failures=None, maximum_stale_data_seconds=None, maximum_reconciliation_discrepancy=1e-6,
)


# --------------------------------------------------------------------------
# Session boundary / force-close policy (Section 20)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SessionBoundaryPolicySpec:
    force_close_open_orders_at_end_of_stream: bool
    force_close_open_positions_at_end_of_stream: bool
    allow_unresolved_working_orders_at_completion: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "force_close_open_orders_at_end_of_stream": self.force_close_open_orders_at_end_of_stream,
            "force_close_open_positions_at_end_of_stream": self.force_close_open_positions_at_end_of_stream,
            "allow_unresolved_working_orders_at_completion": self.allow_unresolved_working_orders_at_completion,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SessionBoundaryPolicySpec:
        return cls(
            force_close_open_orders_at_end_of_stream=bool(raw["force_close_open_orders_at_end_of_stream"]),
            force_close_open_positions_at_end_of_stream=bool(raw["force_close_open_positions_at_end_of_stream"]),
            allow_unresolved_working_orders_at_completion=bool(raw["allow_unresolved_working_orders_at_completion"]),
        )


DEFAULT_SESSION_BOUNDARY_POLICY = SessionBoundaryPolicySpec(
    force_close_open_orders_at_end_of_stream=True, force_close_open_positions_at_end_of_stream=False,
    allow_unresolved_working_orders_at_completion=False,
)


# --------------------------------------------------------------------------
# The top-level spec (Section 3)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StartingPositionSpec:
    instrument_symbol: str
    signed_quantity: float
    average_entry_price: float

    def __post_init__(self) -> None:
        if not self.instrument_symbol:
            raise PaperTradingSpecError("StartingPositionSpec.instrument_symbol must not be empty")
        _finite(self.signed_quantity, field_name="StartingPositionSpec.signed_quantity")
        if self.signed_quantity == 0.0:
            raise PaperTradingSpecError("StartingPositionSpec.signed_quantity must not be zero -- omit a flat starting position entirely")
        _positive(self.average_entry_price, field_name="StartingPositionSpec.average_entry_price")

    def to_json_dict(self) -> dict[str, object]:
        return {"instrument_symbol": self.instrument_symbol, "signed_quantity": self.signed_quantity, "average_entry_price": self.average_entry_price}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> StartingPositionSpec:
        return cls(instrument_symbol=str(raw["instrument_symbol"]), signed_quantity=float(str(raw["signed_quantity"])), average_entry_price=float(str(raw["average_entry_price"])))


@dataclass(frozen=True, slots=True)
class PaperTradingSpec:
    """The top-level, content-addressed specification for one paper-
    trading (or shadow-observation) session. `paper_session_spec_id` (see
    `compute_paper_session_spec_id`) is a pure function of every field
    below except `schema_version`."""

    schema_version: int
    verified_robustness_id: str
    verified_promotion_decision_id: str
    strategy_candidate_identity: str
    """The source backtest's own `backtest_id` -- the strategy candidate
    the verified promotion decision was issued for."""
    model_artifact_identity: str
    calibration_artifact_identity: str | None
    feature_spec_identity: str
    instrument: InstrumentSpec
    price_precision: int
    quantity_precision: int
    session_mode: SessionMode
    market_event_mode: MarketEventMode
    bar_interval: Timeframe | None
    """Required when `market_event_mode is BAR`, must be `None` for
    `QUOTE`."""
    clock_mode: ClockMode
    starting_cash: float
    starting_positions: tuple[StartingPositionSpec, ...]
    order_policy: OrderPolicySpec
    execution_policy: ExecutionPolicySpec
    fill_policy: FillPolicySpec
    spread_policy: SpreadSpec
    slippage_policy: SlippageSpec
    commission_policy: CommissionSpec
    financing_policy: FinancingPolicySpec
    latency_policy: LatencyPolicySpec
    liquidity_policy: LiquidityPolicySpec
    position_policy: PositionPolicySpec
    risk_limits: RiskLimitsSpec
    session_boundary_policy: SessionBoundaryPolicySpec
    seed: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("verified_robustness_id", self.verified_robustness_id), ("verified_promotion_decision_id", self.verified_promotion_decision_id),
            ("strategy_candidate_identity", self.strategy_candidate_identity), ("model_artifact_identity", self.model_artifact_identity),
            ("feature_spec_identity", self.feature_spec_identity),
        ):
            if not is_valid_sha256_hex(value):
                raise PaperTradingSpecError(f"PaperTradingSpec.{field_name} must be a 64-character lowercase hex SHA-256 digest, got {value!r}")
        if self.calibration_artifact_identity is not None and not is_valid_sha256_hex(self.calibration_artifact_identity):
            raise PaperTradingSpecError(f"PaperTradingSpec.calibration_artifact_identity must be a valid sha256 hex id or None, got {self.calibration_artifact_identity!r}")
        _reject_live_like(self.session_mode.value, field_name="PaperTradingSpec.session_mode")
        if self.market_event_mode is MarketEventMode.BAR and self.bar_interval is None:
            raise PaperTradingSpecError("PaperTradingSpec.bar_interval is required when market_event_mode=bar")
        if self.market_event_mode is MarketEventMode.QUOTE and self.bar_interval is not None:
            raise PaperTradingSpecError("PaperTradingSpec.bar_interval must be None when market_event_mode=quote")
        _non_negative_int(self.price_precision, field_name="PaperTradingSpec.price_precision")
        _non_negative_int(self.quantity_precision, field_name="PaperTradingSpec.quantity_precision")
        _non_negative(self.starting_cash, field_name="PaperTradingSpec.starting_cash")
        symbols = [p.instrument_symbol for p in self.starting_positions]
        if len(set(symbols)) != len(symbols):
            raise PaperTradingSpecError("PaperTradingSpec.starting_positions must not repeat an instrument_symbol")
        if self.position_policy.single_instrument_only:
            for p in self.starting_positions:
                if p.instrument_symbol != self.instrument.symbol:
                    raise PaperTradingSpecError(
                        f"PaperTradingSpec.position_policy.single_instrument_only=True but starting_positions references "
                        f"{p.instrument_symbol!r}, not the session's own instrument {self.instrument.symbol!r}"
                    )
            if len(self.starting_positions) > 1:
                raise PaperTradingSpecError("PaperTradingSpec.position_policy.single_instrument_only=True permits at most one starting position")
        if self.seed < 0:
            raise PaperTradingSpecError(f"PaperTradingSpec.seed must be >= 0, got {self.seed}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "verified_robustness_id": self.verified_robustness_id,
            "verified_promotion_decision_id": self.verified_promotion_decision_id, "strategy_candidate_identity": self.strategy_candidate_identity,
            "model_artifact_identity": self.model_artifact_identity, "calibration_artifact_identity": self.calibration_artifact_identity,
            "feature_spec_identity": self.feature_spec_identity, "instrument": self.instrument.to_json_dict(),
            "price_precision": self.price_precision, "quantity_precision": self.quantity_precision, "session_mode": self.session_mode.value,
            "market_event_mode": self.market_event_mode.value, "bar_interval": (None if self.bar_interval is None else self.bar_interval.value),
            "clock_mode": self.clock_mode.value, "starting_cash": self.starting_cash,
            "starting_positions": [p.to_json_dict() for p in self.starting_positions], "order_policy": self.order_policy.to_json_dict(),
            "execution_policy": self.execution_policy.to_json_dict(), "fill_policy": self.fill_policy.to_json_dict(),
            "spread_policy": self.spread_policy.to_json_dict(), "slippage_policy": self.slippage_policy.to_json_dict(),
            "commission_policy": self.commission_policy.to_json_dict(), "financing_policy": self.financing_policy.to_json_dict(),
            "latency_policy": self.latency_policy.to_json_dict(), "liquidity_policy": self.liquidity_policy.to_json_dict(),
            "position_policy": self.position_policy.to_json_dict(), "risk_limits": self.risk_limits.to_json_dict(),
            "session_boundary_policy": self.session_boundary_policy.to_json_dict(), "seed": self.seed,
        }

    def to_identity_payload(self) -> dict[str, object]:
        """Unlike `to_json_dict` (order-preserving, durable), `starting_
        positions` is canonicalized here by `instrument_symbol` since it
        is semantically an unordered set (uniqueness already enforced in
        `__post_init__`) -- everything else in this spec is either a
        scalar or an already-order-insensitive nested object, so no
        further canonicalization is needed."""
        payload = self.to_json_dict()
        del payload["schema_version"]
        payload["starting_positions"] = [p.to_json_dict() for p in sorted(self.starting_positions, key=lambda p: p.instrument_symbol)]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PaperTradingSpec:
        from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

        require_schema_version(raw, supported=PAPER_SESSION_SPEC_SCHEMA_VERSION, context="PaperTradingSpec")
        bar_interval_raw = raw.get("bar_interval")
        return cls(
            schema_version=PAPER_SESSION_SPEC_SCHEMA_VERSION, verified_robustness_id=str(raw["verified_robustness_id"]),
            verified_promotion_decision_id=str(raw["verified_promotion_decision_id"]),
            strategy_candidate_identity=str(raw["strategy_candidate_identity"]), model_artifact_identity=str(raw["model_artifact_identity"]),
            calibration_artifact_identity=(None if raw.get("calibration_artifact_identity") is None else str(raw["calibration_artifact_identity"])),
            feature_spec_identity=str(raw["feature_spec_identity"]),
            instrument=InstrumentSpec.from_json_dict(as_json_dict(raw["instrument"], field_name="instrument")),
            price_precision=int(str(raw["price_precision"])), quantity_precision=int(str(raw["quantity_precision"])),
            session_mode=SessionMode(raw["session_mode"]), market_event_mode=MarketEventMode(raw["market_event_mode"]),
            bar_interval=(None if bar_interval_raw is None else Timeframe(bar_interval_raw)), clock_mode=ClockMode(raw["clock_mode"]),
            starting_cash=float(str(raw["starting_cash"])),
            starting_positions=tuple(
                StartingPositionSpec.from_json_dict(as_json_dict(p, field_name="starting_positions[]"))
                for p in as_json_list(raw.get("starting_positions") if "starting_positions" in raw else [], field_name="starting_positions")
            ),
            order_policy=OrderPolicySpec.from_json_dict(as_json_dict(raw["order_policy"], field_name="order_policy")),
            execution_policy=ExecutionPolicySpec.from_json_dict(as_json_dict(raw["execution_policy"], field_name="execution_policy")),
            fill_policy=FillPolicySpec.from_json_dict(as_json_dict(raw["fill_policy"], field_name="fill_policy")),
            spread_policy=SpreadSpec.from_json_dict(as_json_dict(raw["spread_policy"], field_name="spread_policy")),
            slippage_policy=SlippageSpec.from_json_dict(as_json_dict(raw["slippage_policy"], field_name="slippage_policy")),
            commission_policy=CommissionSpec.from_json_dict(as_json_dict(raw["commission_policy"], field_name="commission_policy")),
            financing_policy=FinancingPolicySpec.from_json_dict(as_json_dict(raw["financing_policy"], field_name="financing_policy")),
            latency_policy=LatencyPolicySpec.from_json_dict(as_json_dict(raw["latency_policy"], field_name="latency_policy")),
            liquidity_policy=LiquidityPolicySpec.from_json_dict(as_json_dict(raw["liquidity_policy"], field_name="liquidity_policy")),
            position_policy=PositionPolicySpec.from_json_dict(as_json_dict(raw["position_policy"], field_name="position_policy")),
            risk_limits=RiskLimitsSpec.from_json_dict(as_json_dict(raw["risk_limits"], field_name="risk_limits")),
            session_boundary_policy=SessionBoundaryPolicySpec.from_json_dict(as_json_dict(raw["session_boundary_policy"], field_name="session_boundary_policy")),
            seed=int(str(raw["seed"])),
        )


@dataclass(frozen=True, slots=True)
class PaperSessionSpecIdentity:
    schema_version: int
    paper_session_spec_id: str

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "paper_session_spec_id": self.paper_session_spec_id}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PaperSessionSpecIdentity:
        from quant_platform.ml.persistence import require_schema_version

        require_schema_version(raw, supported=1, context="PaperSessionSpecIdentity")
        return cls(schema_version=1, paper_session_spec_id=str(raw["paper_session_spec_id"]))


def compute_paper_session_spec_id(spec: PaperTradingSpec) -> PaperSessionSpecIdentity:
    """Pure function: `PaperTradingSpec` -> `PaperSessionSpecIdentity`.
    Mirrors `robustness.specs.compute_robustness_identity` exactly, using
    the package-wide `identity.compute_content_id` building block shared
    with every other content-addressed id in this package."""
    paper_session_spec_id = compute_content_id("paper_session_spec", spec.to_identity_payload())
    return PaperSessionSpecIdentity(schema_version=1, paper_session_spec_id=paper_session_spec_id)


__all__ = [
    "DEFAULT_EXECUTION_POLICY",
    "DEFAULT_POSITION_POLICY",
    "DEFAULT_RISK_LIMITS",
    "DEFAULT_SESSION_BOUNDARY_POLICY",
    "ExecutionPolicySpec",
    "FillPolicySpec",
    "FinancingPolicySpec",
    "InstrumentSpec",
    "LatencyPolicySpec",
    "LiquidityPolicySpec",
    "OrderPolicySpec",
    "PaperSessionSpecIdentity",
    "PaperTradingSpec",
    "PositionPolicySpec",
    "RiskLimitsSpec",
    "SessionBoundaryPolicySpec",
    "StartingPositionSpec",
    "compute_paper_session_spec_id",
]
