"""Pure `RiskAuthorization` issuance for `quant_platform.portfolio_risk`
(Milestone 9, Phase 3). `issue_risk_authorization` is the ONLY function
in this package that constructs a `RiskAuthorization` -- it enforces,
structurally, that only an `APPROVED` `RiskDecision` may ever produce
one (Required Invariant #1/#2), and that the authorization binds to
every field the request/decision actually agreed on (never a
caller-asserted value), by recomputing both the request's and the
decision's own content identity before trusting anything they claim.

No `utc_now()`, `uuid4()`, `random`, temp path, or process-specific
identity source participates anywhere in this module -- `event_time` and
`authorization_sequence` are always explicit, caller-supplied
parameters."""

from __future__ import annotations

from datetime import datetime

from quant_platform.core.exceptions import RiskAuthorizationIdentityError, RiskDenialError
from quant_platform.portfolio_risk.authorization import RiskAuthorization, create_risk_authorization
from quant_platform.portfolio_risk.decisions import (
    RISK_DECISION_KIND,
    RISK_EVALUATION_REQUEST_KIND,
    RiskDecision,
    RiskEvaluationRequest,
)
from quant_platform.portfolio_risk.identity import compute_content_id
from quant_platform.portfolio_risk.models import RiskDecisionKind

__all__ = ["issue_risk_authorization"]


def issue_risk_authorization(
    *, request: RiskEvaluationRequest, decision: RiskDecision, authorization_sequence: int, event_time: datetime,
) -> RiskAuthorization:
    if decision.kind is not RiskDecisionKind.APPROVED:
        raise RiskDenialError(
            f"issue_risk_authorization: cannot issue an authorization for a {decision.kind.value!r} decision "
            f"(risk_decision_id={decision.risk_decision_id!r}) -- only APPROVED decisions may produce a usable authorization"
        )

    recomputed_request_id = compute_content_id(RISK_EVALUATION_REQUEST_KIND, request.to_identity_payload())
    if recomputed_request_id != request.risk_evaluation_request_id:
        raise RiskAuthorizationIdentityError(
            "issue_risk_authorization: request identity does not reproduce its own content -- forged/tampered RiskEvaluationRequest"
        )
    recomputed_decision_id = compute_content_id(RISK_DECISION_KIND, decision.to_identity_payload())
    if recomputed_decision_id != decision.risk_decision_id:
        raise RiskAuthorizationIdentityError("issue_risk_authorization: decision identity does not reproduce its own content -- forged/tampered RiskDecision")
    if decision.risk_evaluation_request_id != request.risk_evaluation_request_id:
        raise RiskAuthorizationIdentityError(
            f"issue_risk_authorization: decision.risk_evaluation_request_id={decision.risk_evaluation_request_id!r} does not match "
            f"request.risk_evaluation_request_id={request.risk_evaluation_request_id!r}"
        )
    for field_name, decision_value, request_value in (
        ("portfolio_snapshot_id", decision.portfolio_snapshot_id, request.portfolio_snapshot_id),
        ("price_snapshot_id", decision.price_snapshot_id, request.price_snapshot_id),
        ("risk_policy_id", decision.risk_policy_id, request.risk_policy_id),
    ):
        if decision_value != request_value:
            raise RiskAuthorizationIdentityError(f"issue_risk_authorization: decision.{field_name}={decision_value!r} does not match request.{field_name}={request_value!r}")

    return create_risk_authorization(
        execution_intent_id=request.execution_intent_id, execution_session_id=request.execution_session_id, portfolio_id=request.portfolio_id,
        portfolio_snapshot_id=decision.portfolio_snapshot_id, price_snapshot_id=decision.price_snapshot_id, risk_policy_id=decision.risk_policy_id,
        risk_decision_id=decision.risk_decision_id, decision_kind=decision.kind, evaluated_quantity=decision.evaluated_quantity,
        evaluated_price=decision.evaluated_price, authorization_sequence=authorization_sequence, event_time=event_time,
    )
