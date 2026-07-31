"""Authorization-use validation for `quant_platform.portfolio_risk`
(Milestone 9, Phase 3). `validate_authorization_use` is the single,
pure, stateless function `lifecycle.py`'s reserve/consume transactions
call before ever appending a ledger entry -- it never mutates anything
and never trusts a caller assertion; every binding field is checked by
RECOMPUTING what `authorization.risk_authorization_id` would be for the
attempted use and comparing (`authorization.
verify_risk_authorization_binding`, Phase 1), never by comparing
individual fields one at a time (which could miss a newly-added field).

EXACT BINDING FOR PHASE 3 (a deliberate design decision, not an
oversight): quantity and price are both required to match EXACTLY what
was evaluated -- no partial-use or bounded-slippage semantics exist in
this milestone. There is no strong repository-level precedent motivating
bounded semantics yet (no other milestone's authorization-shaped object
allows partial/slippage-tolerant reuse either), and exact binding is the
simpler, more conservative, and more easily independently-verifiable
choice -- see `docs/portfolio_risk_architecture.md`'s "Authorization
validation" section for the full rationale."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from quant_platform.portfolio_risk.authorization import RiskAuthorization, verify_risk_authorization_binding
from quant_platform.portfolio_risk.models import (
    RiskAuthorizationStatus,
    RiskDecisionKind,
    is_legal_risk_authorization_status_transition,
)

__all__ = ["AuthorizationRejectionReason", "AuthorizationUseValidation", "validate_authorization_use"]


class AuthorizationRejectionReason(Enum):
    BINDING_MISMATCH = "binding_mismatch"
    """The attempted intent/session/portfolio/snapshot/policy/quantity/
    price does not reproduce `authorization.risk_authorization_id` --
    covers every one of "cross-intent"/"cross-session"/"cross-portfolio"/
    "cross-snapshot"/"cross-policy" mismatch and "quantity/price
    changed" in one check."""
    EXPIRED = "expired"
    STATUS_DOES_NOT_PERMIT_USE = "status_does_not_permit_use"
    """The authorization's current lifecycle status (already terminal, or
    not yet reached the state this use requires) makes the attempted
    transition illegal."""
    CONFLICTING_CONSUMPTION = "conflicting_consumption"
    """The authorization is already at the target status, but under a
    DIFFERENT `consumption_identity` -- a genuine second economic use,
    fail-closed rejected (as opposed to an EXACT retry of the same
    `consumption_identity`, which is idempotently approved)."""


@dataclass(frozen=True, slots=True)
class AuthorizationUseValidation:
    approved: bool
    rejection_reason: AuthorizationRejectionReason | None
    detail: str
    is_exact_retry: bool
    """`True` only when `approved` and this use is an idempotent repeat
    of an already-recorded identical transition (same target status,
    same `consumption_identity`) -- `lifecycle.py` uses this to decide
    whether a NEW ledger entry needs to be appended at all."""

    def __post_init__(self) -> None:
        if self.approved and self.rejection_reason is not None:
            raise ValueError("AuthorizationUseValidation.rejection_reason must be None when approved=True")
        if not self.approved and self.rejection_reason is None:
            raise ValueError("AuthorizationUseValidation.rejection_reason is required when approved=False")
        if self.is_exact_retry and not self.approved:
            raise ValueError("AuthorizationUseValidation.is_exact_retry requires approved=True")


def validate_authorization_use(
    *, authorization: RiskAuthorization, current_status: RiskAuthorizationStatus, bound_consumption_identity: str | None,
    target_status: RiskAuthorizationStatus, execution_intent_id: str, execution_session_id: str, portfolio_id: str,
    portfolio_snapshot_id: str, price_snapshot_id: str, risk_policy_id: str, quantity: Decimal, price: Decimal, consumption_identity: str,
    expiry_time: datetime | None, evaluation_time: datetime,
) -> AuthorizationUseValidation:
    binding_matches = verify_risk_authorization_binding(
        authorization, execution_intent_id=execution_intent_id, execution_session_id=execution_session_id, portfolio_id=portfolio_id,
        portfolio_snapshot_id=portfolio_snapshot_id, price_snapshot_id=price_snapshot_id, risk_policy_id=risk_policy_id,
        risk_decision_id=authorization.risk_decision_id, decision_kind=RiskDecisionKind.APPROVED, evaluated_quantity=quantity,
        evaluated_price=price,
    )
    if not binding_matches:
        return AuthorizationUseValidation(
            approved=False, rejection_reason=AuthorizationRejectionReason.BINDING_MISMATCH,
            detail="the attempted intent/session/portfolio/snapshot/policy/quantity/price does not reproduce this authorization's own id",
            is_exact_retry=False,
        )

    if expiry_time is not None and evaluation_time > expiry_time:
        return AuthorizationUseValidation(
            approved=False, rejection_reason=AuthorizationRejectionReason.EXPIRED,
            detail=f"evaluation_time {evaluation_time!r} is beyond expiry_time {expiry_time!r}", is_exact_retry=False,
        )

    if current_status is target_status:
        if bound_consumption_identity == consumption_identity:
            return AuthorizationUseValidation(
                approved=True, rejection_reason=None, detail=f"exact retry of an already-recorded {target_status.value} use",
                is_exact_retry=True,
            )
        return AuthorizationUseValidation(
            approved=False, rejection_reason=AuthorizationRejectionReason.CONFLICTING_CONSUMPTION,
            detail=f"already {target_status.value} under a different consumption_identity", is_exact_retry=False,
        )

    if not is_legal_risk_authorization_status_transition(current_status, target_status):
        return AuthorizationUseValidation(
            approved=False, rejection_reason=AuthorizationRejectionReason.STATUS_DOES_NOT_PERMIT_USE,
            detail=f"cannot transition from {current_status.value!r} to {target_status.value!r}", is_exact_retry=False,
        )

    if bound_consumption_identity is not None and bound_consumption_identity != consumption_identity:
        # A NEW transition (current_status is NOT target_status, e.g. the
        # primary RESERVED -> CONSUMED step) into a consumption-identity-
        # carrying state must still honor whatever identity was bound by
        # an EARLIER transition (the RESERVE) -- otherwise this branch
        # would silently let a first-time CONSUME attempt use a
        # completely different economic identity than the one it was
        # reserved under, since it is not a same-target "retry" and would
        # otherwise fall straight through to unconditional approval
        # below (a real defect found and fixed during this phase's own
        # adversarial concurrency testing: two threads reserving under
        # "use-1" then consuming under "use-1" vs "use-2" both succeeded
        # -- the single-economic-use invariant was not actually enforced
        # on this path at all).
        return AuthorizationUseValidation(
            approved=False, rejection_reason=AuthorizationRejectionReason.CONFLICTING_CONSUMPTION,
            detail=f"bound to consumption_identity {bound_consumption_identity!r}, cannot {target_status.value} under {consumption_identity!r}",
            is_exact_retry=False,
        )

    return AuthorizationUseValidation(approved=True, rejection_reason=None, detail=f"new {target_status.value} transition", is_exact_retry=False)
