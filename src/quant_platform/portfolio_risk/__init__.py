"""`quant_platform.portfolio_risk` -- Milestone 9: a broker-neutral,
deterministic, TEST-ONLY portfolio risk and capital management engine
that evaluates execution intents before dispatch and produces immutable,
auditable risk authorizations.

THIS IS NOT LIVE TRADING. THIS IS NOT MT5 INTEGRATION. This package never
opens a network connection, never imports a broker SDK, never defines a
credential field, and never claims profitability, broker readiness, or
operational live-trading readiness. It contains no float arithmetic for
financial values (`Decimal` throughout) and no wall-clock-dependent
economic decision (every timestamp affecting an economic outcome is
caller-supplied, never `datetime.now()`/`utc_now()`).

PHASE 1 SCOPE ONLY: this milestone currently delivers the domain
foundation -- exceptions, config schemas, enums, content-addressed
policy/spec identity, portfolio/price snapshot models, and risk-decision/
risk-authorization models -- and nothing else. It does NOT yet implement
an evaluator, a durable ledger, crash recovery, a CLI, or
execution-gateway enforcement. An `ExecutionIntent` is NOT yet blocked
from dispatch by this package in this phase; see
`docs/portfolio_risk_architecture.md` for the full scope statement and
`docs/milestone9_phase1_delivery_report.md` for the Phase 1 delivery
report.

FAIL-CLOSED BY CONSTRUCTION: `RiskDecisionKind` has exactly three members
-- `APPROVED`, `DENIED`, `HALTED` -- there is no `UNKNOWN`/pending value
to even construct. Every model in this package that cannot be
structurally validated raises rather than silently defaulting to an
approval-shaped outcome; there is no bypass flag anywhere in this
package that disables a risk check.

DEPENDENCY DIRECTION IS STRICTLY ONE-WAY: this package depends on
`quant_platform.paper_trading` (for shared content-identity
infrastructure only, exactly as `quant_platform.execution_gateway`
already does) and `quant_platform.core`/`quant_platform.ml`. It does NOT
depend on `quant_platform.execution_gateway` -- the intended future
consumption direction is the reverse: `execution_gateway`'s dispatch gate
will depend on `portfolio_risk` to check for a valid, matching
`RiskAuthorization` before ever calling the adapter. Cross-package
binding uses plain, sha256-validated id strings (`execution_intent_id`,
`execution_session_id`, ...) rather than direct object references, so no
import cycle is possible between the two packages."""

from __future__ import annotations
