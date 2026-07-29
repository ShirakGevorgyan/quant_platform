"""Shared vocabulary for `quant_platform.paper_trading` (Milestone 7) --
stage machines, enums, and small cross-cutting value types every other
module in this package imports. Mirrors `robustness.models`'s role one
layer up: a single place every enum/transition table lives, so no two
modules independently redefine the same closed set of names.

WHAT THIS PACKAGE DOES AND DOES NOT DO
--------------------------------------------------------------------------
`paper_trading` deterministically SIMULATES the trading lifecycle
(decisions, orders, fills, positions, accounting) for a strategy already
independently verified `ELIGIBLE_FOR_PAPER_TRADING` by Milestone 6. It
NEVER transmits an order to a broker, exchange, MT5 terminal, or any
live-trading API -- there is no network client, no broker credential
field, and no `LIVE` session mode anywhere in this package. Simulated
fills are not broker fills; paper-trading eligibility is not
live-trading approval.

`backtesting.models.PositionDirection` (`FLAT`/`LONG`/`SHORT`) is reused
directly for position state -- semantically identical, no need for a
second definition."""

from __future__ import annotations

from enum import Enum


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------
class SessionMode(Enum):
    """Section 0.8/19. Exactly three modes -- deliberately no `LIVE`
    member anywhere in this enum; it cannot be constructed because it
    does not exist."""

    REPLAY_PAPER = "replay_paper"
    """Deterministic replay of a bounded, pre-validated market-event
    source (Section 32's replay reader) -- fully reproducible, used for
    acceptance testing and offline analysis."""
    FORWARD_PAPER = "forward_paper"
    """Driven by an externally supplied, live-arriving (but never
    broker-connected) event stream -- the engine reacts to whatever
    events it is handed, in the order handed, never polling or inventing
    events itself."""
    SHADOW_OBSERVATION = "shadow_observation"
    """Decisions and hypothetical orders/fills are produced and
    persisted, but never applied to the simulated account -- see
    `shadow.py`."""


class ClockMode(Enum):
    REPLAY = "replay"
    FORWARD = "forward"
    MANUAL_TEST = "manual_test"


class MarketEventMode(Enum):
    """Which primary tick source a session consumes. `QUOTE` sessions
    have real bid/ask (spread is read directly from the quote); `BAR`
    sessions approximate spread/execution from OHLC via the SAME
    `backtesting.specs.SpreadSpec`/`SlippageSpec` policies backtesting
    itself uses."""

    QUOTE = "quote"
    BAR = "bar"


PAPER_SESSION_SPEC_SCHEMA_VERSION = 1


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderTypeKind(Enum):
    """Section 8: `STOP_LIMIT`/`MARKET_ON_CLOSE` are deliberately NOT
    members -- Section 8's own instruction permits omitting an order type
    "only if fully supportable"/"only if semantics are exact"; this
    milestone implements exactly the three types whose fill semantics are
    unambiguous in both QUOTE and BAR mode (see `execution.py`'s module
    docstring for the exact rules) and defers the other two rather than
    claim partial support for either."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class TimeInForceKind(Enum):
    """`FOK` IS implemented (Section 8 permits it "only if exact
    semantics are implemented" -- it is exact here: a marketable FOK
    order that cannot be filled for its FULL quantity against the
    currently available liquidity assumption is cancelled outright,
    never partially filled and never left working)."""

    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderState(Enum):
    """Section 8's required lifecycle, in `_LEGAL_ORDER_TRANSITIONS`
    order (see `orders.py`)."""

    CREATED = "created"
    VALIDATED = "validated"
    REJECTED = "rejected"
    ACCEPTED = "accepted"
    WORKING = "working"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_ORDER_STATES: frozenset[OrderState] = frozenset({OrderState.REJECTED, OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED})


class PositionIntentKind(Enum):
    """Deterministically derived by `order_policy.py` from the signed
    delta between current and target position -- never caller-supplied
    as a free-form flag."""

    OPEN = "open"
    INCREASE = "increase"
    REDUCE = "reduce"
    CLOSE = "close"
    REVERSE = "reverse"


class RejectReasonKind(Enum):
    """Closed, typed vocabulary for every order-validation/risk rejection
    -- never a free-text-only reason."""

    NON_POSITIVE_QUANTITY = "non_positive_quantity"
    QUANTITY_BELOW_MINIMUM = "quantity_below_minimum"
    QUANTITY_NOT_QUANTIZED = "quantity_not_quantized"
    PRICE_NOT_QUANTIZED = "price_not_quantized"
    MISSING_REQUIRED_PRICE = "missing_required_price"
    UNSUPPORTED_ORDER_TYPE = "unsupported_order_type"
    UNSUPPORTED_TIME_IN_FORCE = "unsupported_time_in_force"
    DUPLICATE_CLIENT_ORDER_ID = "duplicate_client_order_id"
    SESSION_NOT_ACCEPTING_ORDERS = "session_not_accepting_orders"
    TRADING_HALTED = "trading_halted"
    EXPOSURE_LIMIT_EXCEEDED = "exposure_limit_exceeded"
    ORDER_NOTIONAL_LIMIT_EXCEEDED = "order_notional_limit_exceeded"
    ORDER_QUANTITY_LIMIT_EXCEEDED = "order_quantity_limit_exceeded"
    ORDER_RATE_LIMIT_EXCEEDED = "order_rate_limit_exceeded"
    MAX_ORDERS_PER_EVENT_EXCEEDED = "max_orders_per_event_exceeded"
    STALE_MARKET_DATA = "stale_market_data"
    FOK_NOT_FULLY_FILLABLE = "fok_not_fully_fillable"
    IOC_NOT_IMMEDIATELY_FILLABLE = "ioc_not_immediately_fillable"
    REDUCE_ONLY_WOULD_INCREASE = "reduce_only_would_increase"
    RISK_HALT_ACTIVE = "risk_halt_active"


# --------------------------------------------------------------------------
# Risk / kill switch
# --------------------------------------------------------------------------
class RiskActionKind(Enum):
    ALLOW = "allow"
    REJECT_ORDER = "reject_order"
    CANCEL_OPEN_ORDERS = "cancel_open_orders"
    HALT_NEW_ORDERS = "halt_new_orders"
    FLATTEN_SIMULATED_POSITIONS = "flatten_simulated_positions"
    TERMINATE_SESSION = "terminate_session"


class KillSwitchState(Enum):
    ACTIVE = "active"
    HALTING = "halting"
    HALTED = "halted"
    FLATTENING = "flattening"
    TERMINATED = "terminated"


_LEGAL_KILL_SWITCH_TRANSITIONS: dict[KillSwitchState, frozenset[KillSwitchState]] = {
    KillSwitchState.ACTIVE: frozenset({KillSwitchState.HALTING}),
    KillSwitchState.HALTING: frozenset({KillSwitchState.HALTED, KillSwitchState.FLATTENING}),
    KillSwitchState.FLATTENING: frozenset({KillSwitchState.HALTED}),
    KillSwitchState.HALTED: frozenset({KillSwitchState.TERMINATED}),
    KillSwitchState.TERMINATED: frozenset(),
}
"""Deliberately has NO transition back to `ACTIVE` -- Section 18's own
instruction: "Never silently auto-resume after a safety halt." Resuming
trading after a halt requires a brand new session, never an in-place
kill-switch state change."""


def is_legal_kill_switch_transition(current: KillSwitchState, target: KillSwitchState) -> bool:
    return target in _LEGAL_KILL_SWITCH_TRANSITIONS[current]


class RiskTriggerKind(Enum):
    OPERATOR_REQUEST = "operator_request"
    LOSS_LIMIT = "loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    STALE_DATA = "stale_data"
    SEQUENCE_CORRUPTION = "sequence_corruption"
    RECONCILIATION_FAILURE = "reconciliation_failure"
    REPEATED_STRATEGY_ERRORS = "repeated_strategy_errors"
    REPEATED_EXECUTION_ERRORS = "repeated_execution_errors"
    ARTIFACT_CORRUPTION = "artifact_corruption"
    VERIFICATION_FAILURE = "verification_failure"


class ComparisonOperatorKind(Enum):
    LESS_THAN_OR_EQUAL = "le"
    GREATER_THAN_OR_EQUAL = "ge"
    LESS_THAN = "lt"
    GREATER_THAN = "gt"


# --------------------------------------------------------------------------
# Event-sourced ledger (Section 21)
# --------------------------------------------------------------------------
class LedgerEntryKind(Enum):
    """The closed discriminator for every entry `persistence.
    PaperSessionEventStore` will ever append. Matches Section 21's
    required 16-item list exactly, EXCEPT: `ORDER_STATE_EVENT` covers
    "order created/validated/accepted/rejected/partially filled/filled/
    cancelled" as ONE ledger-entry kind whose payload is `orders.
    OrderStateEvent.to_json_dict()` (its own `to_state` field already
    distinguishes all 7 transitions -- persisting 7 near-identical
    wrapper kinds would duplicate that information, not add any), and
    `FILL` is an ADDED 17th kind: Section 21's own "order ... filled"
    item names the ORDER's state transition, but reconstructing account
    state (Section 25's reconciliation) needs the `Fill` record itself
    (price/quantity/costs) as its own ledger entry too -- a necessary
    completion of that item, not a scope addition."""

    MARKET_EVENT_ACCEPTED = "market_event_accepted"
    STRATEGY_DECISION = "strategy_decision"
    ORDER_STATE_EVENT = "order_state_event"
    FILL = "fill"
    MARK_APPLIED = "mark_applied"
    FINANCING_APPLIED = "financing_applied"
    RISK_DECISION = "risk_decision"
    HALT_TRIGGERED = "halt_triggered"
    ACCOUNT_SNAPSHOT = "account_snapshot"
    RECONCILIATION_RESULT = "reconciliation_result"
    SESSION_TRANSITION = "session_transition"
    SHADOW_OBSERVATION = "shadow_observation"


# --------------------------------------------------------------------------
# Market events (Section 5)
# --------------------------------------------------------------------------
class MarketEventKind(Enum):
    """Discriminator tag used both for content-addressed identity
    namespacing (`identity.compute_content_id`) and for the `"kind"` field
    every persisted market-event ledger entry carries, so `events.
    market_event_from_json_dict` knows which dataclass to reconstruct.
    Deliberately excludes `CorporateOrInstrumentAdjustmentEvent` (Section
    5's own "only if generically supported" -- deferred, no generic
    corporate-action model exists anywhere in this repository to reuse)."""

    QUOTE = "quote"
    BAR = "bar"
    SESSION_OPEN = "session_open"
    SESSION_CLOSE = "session_close"
    TRADING_HALT = "trading_halt"
    TRADING_RESUME = "trading_resume"
    FINANCING = "financing"
    END_OF_STREAM = "end_of_stream"


class MarketEventQualityFlagKind(Enum):
    """Closed vocabulary for `QuoteEvent`/`BarEvent` quality disclosure --
    never a free-text field, so downstream risk/reporting code can match
    on it exactly."""

    STALE = "stale"
    WIDE_SPREAD = "wide_spread"
    SUSPECT_PRICE = "suspect_price"
    NORMALIZED_FROM_SOURCE = "normalized_from_source"
    """Section 5: "Replay ingestion may validate and prepare a source
    sequence before a session begins, but must disclose any
    normalization." Set by `replay.py` on any event it reordered,
    deduplicated, or otherwise altered from the raw source file."""


# --------------------------------------------------------------------------
# Execution policy
# --------------------------------------------------------------------------
class BarAmbiguityPolicyKind(Enum):
    """Section 10's bar-mode ambiguity policy: when a position has BOTH a
    working stop order and a working limit order whose trigger/fill
    conditions could both be satisfied within the same bar's [low, high]
    range, and the true intrabar path is unknown, this policy declares
    which is honored. Default is `WORST_CASE` (see `DEFAULT_EXECUTION_
    POLICY` in `specs.py`) -- the financially conservative assumption for
    the position, documented, never silently optimistic."""

    WORST_CASE = "worst_case"
    BEST_CASE = "best_case"
    STOP_FIRST = "stop_first"
    TARGET_FIRST = "target_first"
    UNSUPPORTED_AND_REJECT = "unsupported_and_reject"


class PartialFillPolicyKind(Enum):
    FULL_FILL_ONLY = "full_fill_only"
    """Fail-closed default: an order either fills for its complete
    remaining quantity or does not fill at all this event -- no
    liquidity size is ever invented."""
    DETERMINISTIC_PARTIAL = "deterministic_partial"
    """Only meaningful when the consumed market event carries an actual
    bid/ask SIZE (`QuoteEvent.bid_size`/`ask_size`) -- fills up to the
    disclosed size, deterministically, remainder stays `WORKING` (or is
    cancelled per time-in-force). Falls back to `FULL_FILL_ONLY`
    behavior for any event with no size information, never fabricating
    one (Section 10.8)."""


class MarkFieldKind(Enum):
    """BAR-mode mark-to-market field selection (Section 15)."""

    CLOSE = "close"


class SlippageDirectionKind(Enum):
    """Never used as a caller-facing config value -- internal tag on a
    cost line item distinguishing which side of the trade the slippage
    line refers to, for reporting clarity only."""

    ADVERSE = "adverse"


# --------------------------------------------------------------------------
# Session manifest stage machine (Section 20)
# --------------------------------------------------------------------------
class PaperSessionStage(Enum):
    CREATED = "created"
    ELIGIBILITY_VERIFIED = "eligibility_verified"
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    HALTING = "halting"
    HALTED = "halted"
    END_OF_STREAM = "end_of_stream"
    RECONCILING = "reconciling"
    VERIFIED = "verified"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


TERMINAL_PAPER_SESSION_STAGES: frozenset[PaperSessionStage] = frozenset({PaperSessionStage.COMPLETED, PaperSessionStage.FAILED, PaperSessionStage.TERMINATED})

_PAPER_SESSION_STAGE_ORDER: tuple[PaperSessionStage, ...] = (
    PaperSessionStage.CREATED, PaperSessionStage.ELIGIBILITY_VERIFIED, PaperSessionStage.INITIALIZED, PaperSessionStage.RUNNING,
    PaperSessionStage.END_OF_STREAM, PaperSessionStage.RECONCILING, PaperSessionStage.VERIFIED, PaperSessionStage.COMPLETED,
)
"""The MAIN linear happy-path spine (mirrors `robustness._ROBUSTNESS_
STAGE_ORDER`). `RUNNING` additionally self-loops (processing many
events) and can branch to `PAUSED`/`HALTING` -- see
`_LEGAL_PAPER_SESSION_TRANSITIONS` for the complete, explicit table;
this tuple is used only by `resume.py`'s linear-verification walk over
the spine stages that produce ledger-checkpointable progress."""

_PAPER_SESSION_STAGE_INDEX: dict[PaperSessionStage, int] = {stage: i for i, stage in enumerate(_PAPER_SESSION_STAGE_ORDER)}

_LEGAL_PAPER_SESSION_TRANSITIONS: dict[PaperSessionStage, frozenset[PaperSessionStage]] = {
    PaperSessionStage.CREATED: frozenset({PaperSessionStage.ELIGIBILITY_VERIFIED, PaperSessionStage.FAILED}),
    PaperSessionStage.ELIGIBILITY_VERIFIED: frozenset({PaperSessionStage.INITIALIZED, PaperSessionStage.FAILED}),
    PaperSessionStage.INITIALIZED: frozenset({PaperSessionStage.RUNNING, PaperSessionStage.FAILED}),
    PaperSessionStage.RUNNING: frozenset({
        PaperSessionStage.RUNNING, PaperSessionStage.PAUSED, PaperSessionStage.HALTING, PaperSessionStage.END_OF_STREAM, PaperSessionStage.FAILED,
    }),
    PaperSessionStage.PAUSED: frozenset({PaperSessionStage.RUNNING, PaperSessionStage.HALTING, PaperSessionStage.FAILED}),
    PaperSessionStage.HALTING: frozenset({PaperSessionStage.HALTED, PaperSessionStage.FAILED}),
    # A HALTED session may still be reconciled/verified/reported (evidence is
    # preserved, per Section 18) before being administratively TERMINATED --
    # it may NEVER transition back to RUNNING (no silent auto-resume).
    PaperSessionStage.HALTED: frozenset({PaperSessionStage.TERMINATED, PaperSessionStage.RECONCILING, PaperSessionStage.FAILED}),
    PaperSessionStage.END_OF_STREAM: frozenset({PaperSessionStage.RECONCILING, PaperSessionStage.FAILED}),
    PaperSessionStage.RECONCILING: frozenset({PaperSessionStage.VERIFIED, PaperSessionStage.FAILED}),
    PaperSessionStage.VERIFIED: frozenset({PaperSessionStage.COMPLETED, PaperSessionStage.FAILED}),
    PaperSessionStage.COMPLETED: frozenset(),
    PaperSessionStage.FAILED: frozenset(),
    PaperSessionStage.TERMINATED: frozenset(),
}


def is_legal_paper_session_transition(current: PaperSessionStage, target: PaperSessionStage) -> bool:
    if target in _LEGAL_PAPER_SESSION_TRANSITIONS[current]:
        return True
    # Same "rewind on detected corruption" allowance as
    # `robustness.models.is_legal_robustness_transition` -- resume must be
    # able to demote a manifest to the last independently-verified spine
    # stage, exactly the release-audit-fixed defect class from Milestone 6.
    if current in TERMINAL_PAPER_SESSION_STAGES or target in TERMINAL_PAPER_SESSION_STAGES:
        return False
    if current not in _PAPER_SESSION_STAGE_INDEX or target not in _PAPER_SESSION_STAGE_INDEX:
        return False
    return _PAPER_SESSION_STAGE_INDEX[target] < _PAPER_SESSION_STAGE_INDEX[current]


def is_terminal_paper_session_stage(stage: PaperSessionStage) -> bool:
    return stage in TERMINAL_PAPER_SESSION_STAGES


__all__ = [
    "PAPER_SESSION_SPEC_SCHEMA_VERSION",
    "TERMINAL_ORDER_STATES",
    "TERMINAL_PAPER_SESSION_STAGES",
    "BarAmbiguityPolicyKind",
    "ClockMode",
    "ComparisonOperatorKind",
    "KillSwitchState",
    "LedgerEntryKind",
    "MarkFieldKind",
    "MarketEventKind",
    "MarketEventMode",
    "MarketEventQualityFlagKind",
    "OrderSide",
    "OrderState",
    "OrderTypeKind",
    "PaperSessionStage",
    "PartialFillPolicyKind",
    "PositionIntentKind",
    "RejectReasonKind",
    "RiskActionKind",
    "RiskTriggerKind",
    "SessionMode",
    "SlippageDirectionKind",
    "TimeInForceKind",
    "is_legal_kill_switch_transition",
    "is_legal_paper_session_transition",
    "is_terminal_paper_session_stage",
]
