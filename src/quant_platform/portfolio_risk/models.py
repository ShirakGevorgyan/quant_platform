"""Shared vocabulary for `quant_platform.portfolio_risk` (Milestone 9) --
enums and small cross-cutting closed vocabularies every other module in
this package imports. Mirrors `execution_gateway.models`'s identical role
one layer up, and `paper_trading.models`'s identical role two layers up.

`OrderSide` is deliberately NOT redefined here -- it is byte-for-byte the
same closed vocabulary (`BUY`/`SELL`) `paper_trading.models` and
`execution_gateway.models` already define; importing it directly avoids a
third, parallel definition of the exact same concept.

`RiskDecisionKind` HAS EXACTLY THREE MEMBERS ON PURPOSE (`APPROVED`,
`DENIED`, `HALTED`) -- there is no `UNKNOWN`/pending value anywhere in
this enum, so no such value can ever be constructed, stored, or compared
against. An evaluation that cannot be completed safely (missing,
incoherent, or stale required inputs) must resolve to `DENIED` or
`HALTED`, never to a third "we don't know" state that a downstream
consumer might mistake for permission to proceed."""

from __future__ import annotations

from enum import Enum

# Re-exported for `portfolio_risk` callers so they never need to reach
# into `paper_trading.models` directly for this closed vocabulary.
from quant_platform.paper_trading.models import OrderSide

__all__ = [
    "PORTFOLIO_RISK_AUTHORIZATION_SCHEMA_VERSION",
    "PORTFOLIO_RISK_SPEC_SCHEMA_VERSION",
    "OrderSide",
    "RiskAuthorizationStatus",
    "RiskCheckSeverity",
    "RiskDecisionKind",
    "RiskDenialReason",
    "is_legal_risk_authorization_status_transition",
    "is_terminal_risk_authorization_status",
]

PORTFOLIO_RISK_SPEC_SCHEMA_VERSION = 1
PORTFOLIO_RISK_AUTHORIZATION_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# Decision kind -- no UNKNOWN/pending value exists (module docstring).
# --------------------------------------------------------------------------
class RiskDecisionKind(Enum):
    APPROVED = "approved"
    DENIED = "denied"
    HALTED = "halted"


# --------------------------------------------------------------------------
# Denial reason -- closed vocabulary, one member per required policy
# limit plus the structural, fail-closed reasons this package's own
# "unknown/incomplete state must fail closed" rule can produce.
# --------------------------------------------------------------------------
class RiskDenialReason(Enum):
    ORDER_NOTIONAL_LIMIT_EXCEEDED = "order_notional_limit_exceeded"
    POSITION_NOTIONAL_LIMIT_EXCEEDED = "position_notional_limit_exceeded"
    INSTRUMENT_GROSS_EXPOSURE_LIMIT_EXCEEDED = "instrument_gross_exposure_limit_exceeded"
    STRATEGY_GROSS_EXPOSURE_LIMIT_EXCEEDED = "strategy_gross_exposure_limit_exceeded"
    PORTFOLIO_GROSS_EXPOSURE_LIMIT_EXCEEDED = "portfolio_gross_exposure_limit_exceeded"
    PORTFOLIO_NET_EXPOSURE_LIMIT_EXCEEDED = "portfolio_net_exposure_limit_exceeded"
    CONCENTRATION_LIMIT_EXCEEDED = "concentration_limit_exceeded"
    LEVERAGE_LIMIT_EXCEEDED = "leverage_limit_exceeded"
    DAILY_REALIZED_LOSS_LIMIT_EXCEEDED = "daily_realized_loss_limit_exceeded"
    TOTAL_LOSS_LIMIT_EXCEEDED = "total_loss_limit_exceeded"
    DRAWDOWN_LIMIT_EXCEEDED = "drawdown_limit_exceeded"
    CONSECUTIVE_LOSSES_LIMIT_EXCEEDED = "consecutive_losses_limit_exceeded"
    CASH_BUFFER_BREACHED = "cash_buffer_breached"
    STALE_PRICE = "stale_price"
    STALE_PORTFOLIO_SNAPSHOT = "stale_portfolio_snapshot"
    PORTFOLIO_HALTED = "portfolio_halted"
    INCOHERENT_EVALUATION_STATE = "incoherent_evaluation_state"
    """The fail-closed catch-all: required inputs were missing, mutually
    inconsistent, or otherwise could not be safely evaluated. Never used
    to mean "approved by default" -- the presence of this reason always
    means the decision was DENIED or HALTED, never APPROVED."""


# --------------------------------------------------------------------------
# Check severity -- mirrors the ordered-severity pattern
# `paper_trading.risk._ACTION_SEVERITY`/`most_severe_action` already
# establish, adapted to this package's own decision vocabulary: a check's
# severity is what a (future) evaluator maxes over to pick the overall
# `RiskDecisionKind`.
# --------------------------------------------------------------------------
class RiskCheckSeverity(Enum):
    INFO = "info"
    """The check was evaluated and passed."""
    WARNING = "warning"
    """The check failed but does not, by itself, deny or halt."""
    DENY = "deny"
    """The check failed severely enough to deny this one order."""
    HALT = "halt"
    """The check failed severely enough to halt the whole portfolio."""


_CHECK_SEVERITY_ORDER: dict[RiskCheckSeverity, int] = {
    RiskCheckSeverity.INFO: 0,
    RiskCheckSeverity.WARNING: 1,
    RiskCheckSeverity.DENY: 2,
    RiskCheckSeverity.HALT: 3,
}


def most_severe_check_severity(severities: tuple[RiskCheckSeverity, ...]) -> RiskCheckSeverity:
    """Pure ordering helper -- mirrors `paper_trading.risk.
    most_severe_action` exactly. A future evaluator uses this to fold
    many `RiskCheckResult`s into one `RiskDecisionKind`; Phase 1 defines
    the ordering only, no evaluator calls it yet."""
    if not severities:
        return RiskCheckSeverity.INFO
    return max(severities, key=lambda s: _CHECK_SEVERITY_ORDER[s])


# --------------------------------------------------------------------------
# Authorization status -- event-sourced exactly like `ExecutionOrderState`/
# `KillSwitchState` (Milestone 8/7's identical pattern). `RiskAuthorization`
# itself is immutable and content-addressed, so its lifecycle status is
# NEVER a field stored on the object -- it is derived by replaying a
# durable sequence of status-transition events (Phase 3's own
# `state_machine.RiskAuthorizationStatusEvent`, mirroring
# `execution_gateway.state_machine.ExecutionOrderStateEvent` exactly)
# against this closed vocabulary and transition table.
#
# RESERVED and INVALIDATED were added in Phase 3 (Phase 1 defined only
# ISSUED/CONSUMED/EXPIRED/REVOKED, explicitly anticipating this
# extension -- see Phase 1's own module docstring note that "the full
# status state machine is deferred to a later phase"). This is a
# purely-additive change to an already-committed enum: no existing
# member was removed or renumbered, and every Phase 1/2 test that
# constructs a `RiskAuthorizationStatus` value continues to pass
# unchanged.
# --------------------------------------------------------------------------
class RiskAuthorizationStatus(Enum):
    ISSUED = "issued"
    RESERVED = "reserved"
    CONSUMED = "consumed"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"
    REVOKED = "revoked"


_TERMINAL_RISK_AUTHORIZATION_STATUSES: frozenset[RiskAuthorizationStatus] = frozenset({
    RiskAuthorizationStatus.CONSUMED, RiskAuthorizationStatus.EXPIRED, RiskAuthorizationStatus.INVALIDATED,
    RiskAuthorizationStatus.REVOKED,
})

_LEGAL_RISK_AUTHORIZATION_STATUS_TRANSITIONS: dict[RiskAuthorizationStatus, frozenset[RiskAuthorizationStatus]] = {
    # ISSUED: a caller may reserve it for economic use, or it may become
    # terminal without ever being reserved (expired while sitting unused,
    # invalidated by newer portfolio state, or explicitly revoked).
    RiskAuthorizationStatus.ISSUED: frozenset({
        RiskAuthorizationStatus.RESERVED, RiskAuthorizationStatus.EXPIRED, RiskAuthorizationStatus.INVALIDATED,
        RiskAuthorizationStatus.REVOKED,
    }),
    # RESERVED: a caller has durably recorded intent to consume it before
    # dispatch (Section "RESERVE AND CONSUME SEMANTICS") -- from here it
    # either completes (CONSUMED) or still terminates without completing
    # (expired/invalidated/revoked while reserved). There is NO edge back
    # to ISSUED -- a reservation can never be silently released for reuse
    # by a different economic submit; it must reach a terminal state.
    RiskAuthorizationStatus.RESERVED: frozenset({
        RiskAuthorizationStatus.CONSUMED, RiskAuthorizationStatus.EXPIRED, RiskAuthorizationStatus.INVALIDATED,
        RiskAuthorizationStatus.REVOKED,
    }),
    # Single-use by construction: every non-terminal state's legal exits
    # are exclusively terminal states -- there is no path back to ISSUED
    # or RESERVED from anywhere, exactly mirroring `KillSwitchState`'s own
    # "never silently resume" property (no edge returns to `ACTIVE`).
    RiskAuthorizationStatus.CONSUMED: frozenset(),
    RiskAuthorizationStatus.EXPIRED: frozenset(),
    RiskAuthorizationStatus.INVALIDATED: frozenset(),
    RiskAuthorizationStatus.REVOKED: frozenset(),
}


def is_legal_risk_authorization_status_transition(current: RiskAuthorizationStatus, target: RiskAuthorizationStatus) -> bool:
    return target in _LEGAL_RISK_AUTHORIZATION_STATUS_TRANSITIONS[current]


def is_terminal_risk_authorization_status(status: RiskAuthorizationStatus) -> bool:
    return status in _TERMINAL_RISK_AUTHORIZATION_STATUSES
