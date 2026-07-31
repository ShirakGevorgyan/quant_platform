# Portfolio Risk and Capital Management Engine (Milestone 9) -- Phase 1 Architecture Skeleton

## Status: Phase 1 domain foundation only

This document is a SKELETON, matching the scope of Milestone 9 Phase 1.
It fully describes everything Phase 1 actually delivers -- exceptions,
config schemas, enums, content-addressed policy/spec identity, portfolio/
price snapshot models, and risk-decision/risk-authorization models -- and
explicitly marks every later-phase concept (evaluator, durable ledger,
crash recovery, CLI, execution-gateway enforcement, acceptance workflow)
as NOT YET IMPLEMENTED rather than silently omitting it. Sections for
those later phases are present as headings with a one-line status note,
so this document's own structure does not need to be reshuffled as each
phase lands -- only expanded.

## Scope and explicit safety boundary (verbatim safety statements)

This milestone delivers TEST-ONLY infrastructure. The following are
explicitly, structurally true of everything in this package, in this
phase and every phase after it:

- No MT5 implementation exists anywhere in this package.
- No FxPro implementation exists anywhere in this package.
- No real broker adapter exists anywhere in this package.
- No network connection is ever opened by any code in this package.
- No broker SDK is imported anywhere in this package.
- No credential field, broker-login field, or credential-shaped
  configuration exists anywhere in this package's domain objects or
  configuration schema (proven structurally: `config.
  portfolio_risk_schemas` uses `extra="forbid"` throughout, so no such
  field could even be smuggled in through an unrecognized key -- and no
  such field is ever declared in the first place).
- This package never claims profitability, broker readiness, broker
  compatibility, or operational live-trading readiness. Passing every
  test in this package proves none of those things and authorizes no
  real-money execution.
- No `Decimal`/`float` field in this package's domain objects models a
  real-money balance belonging to any real account -- every monetary
  figure is either caller-supplied test data or (in a later phase) a
  derived figure from that same test data.
- Every quantity, price, monetary value, rate, and threshold in this
  package is `Decimal` -- there is no float arithmetic for a financial
  value anywhere in this package (proven structurally by `tests/unit/
  portfolio_risk/test_portfolio_risk_safety.py::
  TestNoFloatArithmeticForFinancialValues`, an AST scan of every module
  for a bare `float(...)` call).
- No economic or staleness decision in this package depends on an
  internal wall-clock read -- every timestamp that can affect identity,
  validation, or (in a later phase) a risk decision is caller-supplied,
  never `datetime.now()`/`utcnow()`/`pd.Timestamp.now()` (proven
  structurally by the same safety-scan file's
  `TestNoWallClockDependentEconomicDecisions`).
- `RiskDecisionKind` has exactly three members -- `APPROVED`, `DENIED`,
  `HALTED`. There is no `UNKNOWN`/pending value anywhere in this enum, so
  no such value can ever be constructed, stored, or compared against.
  Unknown or incomplete risk state must resolve to `DENIED` or `HALTED`,
  never a third state a downstream consumer might mistake for permission
  to proceed.
- There is no bypass flag anywhere in this package that disables a risk
  check. No field on `PortfolioRiskPolicy`, no config schema field, and
  no function parameter exists whose purpose is "skip risk evaluation" or
  "force-approve".
- Milestone 10 has not been started.

## PHASE 1 EXPLICITLY DOES NOT YET IMPLEMENT

Stated plainly, so no later reader mistakes Phase 1's domain models for a
working risk engine:

- **No evaluator.** Nothing in this package yet computes a `RiskDecision`
  from a real `PortfolioSnapshot`/`PriceSnapshot`/`PortfolioRiskPolicy`
  triple. `decisions.py` defines the SHAPE of a decision and validates
  its internal coherence once constructed; it does not construct one from
  real state.
- **No durable ledger.** There is no append-only, hash-chained event
  store for this package's own objects (unlike `execution_gateway.
  persistence`). `RiskAuthorizationStatus`'s event-sourced legal-
  transition table exists in `models.py` specifically so a later phase
  can add one safely, but no such store exists yet.
- **No crash recovery, no reconciliation, no independent verification
  pass.** The corresponding exception classes exist in `core.exceptions`
  (mirroring `execution_gateway`'s identical exception shape) so a later
  phase's functions have a home to raise into, but no such functions
  exist yet.
- **No CLI.** No `ml_cli.py` command exists for this package yet.
- **No execution-gateway enforcement.** `execution_gateway.paper_bridge.
  ExecutionIntent.risk_authorization_id` is NOT checked against a real
  `portfolio_risk.authorization.RiskAuthorization` anywhere yet -- see
  "Relationship to `execution_gateway`" below for the precise current
  state of that field and what remains to wire up.
- **No acceptance workflow.** No end-to-end test chains a real paper
  session through this package's own risk evaluation; only isolated
  domain-model unit tests exist in this phase.

## Package architecture and dependency direction

`quant_platform.portfolio_risk` is a new top-level package, structurally
parallel to `quant_platform.execution_gateway` (both depend on
`quant_platform.paper_trading`/`quant_platform.core`/`quant_platform.ml`;
neither currently depends on the other).

**Dependency direction chosen:** `portfolio_risk` depends downward on
`paper_trading` (for `paper_trading.identity.compute_content_id` and
`paper_trading.models.OrderSide` reuse only -- the same shared low-level
infrastructure `execution_gateway` itself already reuses rather than
duplicating) and on `core`/`ml` (exceptions, JSON/timestamp
serialization, fingerprinting). It does **not** depend on
`execution_gateway`.

The intended future consumption direction is the reverse: a later
phase's `execution_gateway` dispatch gate will depend on `portfolio_risk`
to check for a valid, matching `RiskAuthorization` before ever calling
the adapter -- mirroring exactly how `execution_gateway` itself depends
on `paper_trading` today, never the other way around. This keeps the
overall dependency graph one-way and acyclic:

```
paper_trading  --->  execution_gateway
paper_trading  --->  portfolio_risk
                      portfolio_risk  --->  execution_gateway   (future phase, not yet wired)
```

Cross-package binding uses plain, sha256-validated id strings
(`execution_intent_id`, `execution_session_id`, ...) rather than direct
object references -- `RiskEvaluationRequest`/`RiskAuthorization` never
import an `execution_gateway` type, they only carry its content-addressed
id as a `str` field. This is what makes the dependency direction safe: no
import cycle is possible even once `execution_gateway` starts depending
on `portfolio_risk`, because `portfolio_risk` never needs to import
anything back.

**Why not the other direction (`execution_gateway` types imported
directly into `portfolio_risk`)?** Importing `execution_gateway.
paper_bridge.ExecutionIntent` directly into `portfolio_risk` would create
exactly the cycle the future dispatch-gate consumption direction would
then close (`execution_gateway` -> `portfolio_risk` -> `execution_gateway`).
Binding by id string avoids this without sacrificing anything Phase 1
needs: every invariant this phase validates (identity, coherence,
mismatch detection) is expressible purely in terms of id strings and
caller-supplied economic values.

### Relationship to `execution_gateway` -- a genuine finding from Phase 1's own inspection

`execution_gateway.paper_bridge.ExecutionIntent` already has a
`risk_authorization_id: str` field (sha256-validated), added during
Milestone 8. Today it is populated with `authorization.
execution_authorization_id` -- the id of Milestone 8's OWN internal
`ExecutionAuthorization` ("may this paper order be bridged from paper
trading into the execution gateway at all"), a different, earlier-stage
concept from this milestone's `portfolio_risk.authorization.
RiskAuthorization` ("may this specific quantity/price be dispatched
against this specific portfolio state").

**This is not a defect** -- Milestone 8 could not have populated this
field with a real portfolio-risk authorization id, because this package
did not exist yet. It is a genuine integration point a later phase must
resolve deliberately: either `ExecutionIntent` gains a second, distinct
field for the portfolio-risk authorization id, or `risk_authorization_id`
is repurposed once this package's evaluator exists, with `execution_
authorization_id` renamed/relocated to avoid the now-confusing overlap.
Phase 1 does not modify `execution_gateway` at all (per this phase's own
explicit scope boundary), so this is recorded here as a known
architectural note for whichever future phase implements enforcement,
not fixed now.

### Module layout

- `__init__.py` -- package docstring: full safety scope statement,
  Phase 1 boundary, dependency direction.
- `identity.py` -- `compute_content_id` (re-exported from `paper_trading.
  identity`), `decimal_to_json`/`decimal_from_float`/`parse_decimal`
  (Decimal<->JSON primitives, mirroring `execution_gateway.identity`
  exactly).
- `models.py` -- `OrderSide` (re-exported from `paper_trading.models`),
  `RiskDecisionKind`, `RiskDenialReason`, `RiskCheckSeverity` (+
  `most_severe_check_severity`), `RiskAuthorizationStatus` (+ its legal-
  transition table and helpers), schema version constants.
- `specs.py` -- `PortfolioRiskPolicy` (every required limit field),
  `PortfolioRiskSpec` (thin, portfolio-agnostic, content-addressed
  wrapper), `compute_portfolio_risk_spec_id`/
  `verify_portfolio_risk_spec_identity`.
- `snapshots.py` -- `PriceSnapshot`, `PositionSnapshot`,
  `PortfolioSnapshot`, `ExposureSnapshot` (+ `compute_*_exposure`
  derivation functions and `is_price_stale`/`is_portfolio_snapshot_stale`
  staleness predicates).
- `decisions.py` -- `RiskCheckResult`, `RiskEvaluationRequest`,
  `RiskDecision` (+ their `create_*` factories).
- `allocation.py` -- `CapitalAllocation`, `PositionSizeProposal` (+ their
  `create_*` factories).
- `authorization.py` -- `RiskAuthorization`, `create_risk_authorization`,
  `verify_risk_authorization_binding`.
- `config/portfolio_risk_schemas.py` (in `quant_platform.config`, mirroring
  every other milestone's config-schema placement) -- Pydantic
  `PortfolioRiskPolicyConfigSchema`/`PortfolioRiskConfigSchema`, both
  frozen and `extra="forbid"`.

## Identity and determinism

Every content-addressed object in this package follows the identical
placeholder-then-recompute pattern established by `paper_trading`/
`execution_gateway`: a `create_*` factory builds a provisional instance
with its own id field set to `"0" * 64`, computes the real id via
`compute_content_id(kind, provisional.to_identity_payload())`, and
reconstructs the final instance with that id. `to_identity_payload()`
always excludes only the object's own id field (and, for `PortfolioRiskSpec`,
`schema_version`) -- every other field, INCLUDING caller-supplied
timestamps, participates in identity, because those timestamps are never
sourced from an internal wall-clock read (see the safety boundary
above), so including them does not reintroduce the non-determinism class
Milestone 8's own defects #11/#12 (`adapter_id` and `source_event_time`
wall-clock capture leaking into `execution_gateway` identity) were found
and fixed for.

No `uuid4`, no random value, no current wall-clock time, no object memory
address, no temp path, and no `PYTHONHASHSEED`-dependent ordering
participates in any identity computation in this package.
`PortfolioSnapshot.to_identity_payload()` sorts `positions` by
`(instrument_id, strategy_id)` before hashing -- the one genuinely
unordered collection Phase 1 defines -- so declaration order never
affects identity, mirroring `execution_gateway.specs.
DummyBrokerScenarioSpec.to_identity_payload()`'s identical sort-by-key
convention exactly.

`RiskAuthorization.risk_authorization_id` binds to every field this
milestone's specification requires it to bind to (`execution_intent_id`,
`execution_session_id`, `portfolio_id`, `portfolio_snapshot_id`,
`price_snapshot_id`, `risk_policy_id`, `risk_decision_id`/`decision_kind`,
`evaluated_quantity`, `evaluated_price`, `authorization_sequence`,
`event_time`). Because ALL of these participate in identity, an
authorization can never be valid for a different intent, session,
portfolio, portfolio snapshot, price snapshot, policy, decision,
quantity, or price -- not via a separate runtime check, but structurally,
by construction. `verify_risk_authorization_binding` makes the
recompute-and-compare pattern a future dispatch gate will use explicit,
mirroring `execution_gateway.specs.verify_execution_gateway_spec_identity`'s
identical shape.

## Domain models and invariants

### `PortfolioRiskPolicy`/`PortfolioRiskSpec`

Every limit field is `Decimal | None` (or `int | None` for the two age
bounds and the consecutive-losses count) -- `None` means "not
configured", mirroring `paper_trading.specs.RiskLimitsSpec`'s identical
convention exactly. `allow_reduce_only_during_halt` is the one mandatory,
always-explicit field. `PortfolioRiskSpec` deliberately does NOT bind to
a specific `portfolio_id` -- one policy is reusable across many
portfolios; `RiskAuthorization` binds `portfolio_id` and `risk_policy_id`
independently.

### `PriceSnapshot`

`bid <= ask`, both positive; `reference_price` is always explicitly
caller-supplied, never auto-derived as a midpoint or otherwise. No live
lookup, no internal `utc_now()` call.

### `PositionSnapshot`/`PortfolioSnapshot`

A flat/closed position is not represented at all (Phase 1's own
documented simplification) -- `PortfolioSnapshot.positions` holds only
non-flat positions, keyed implicitly by `(instrument_id, strategy_id)`; a
duplicate identity is rejected at construction. Every position's
`unrealized_pnl` must exactly reconcile with `signed_quantity *
(mark_price - average_entry_price) * contract_multiplier`. The
portfolio's own `equity` must exactly reconcile with `cash + sum(position
market values)` (a fully cash-settled model, no separate liabilities/
accrued-costs line in Phase 1, the same documented simplification
`paper_trading.portfolio`/`execution_gateway.dispatcher` already carry).
`unrealized_pnl` at the portfolio level must equal the sum of every open
position's own `unrealized_pnl`; `realized_pnl` is NOT cross-validated
against currently-open positions (realized pnl accrues from closed
positions too, which are not retained in this point-in-time snapshot).
`drawdown_fraction` is a derived PROPERTY, never a stored field --
`peak_equity` must never be less than `equity`, enforced at construction.

### `ExposureSnapshot`

Always DERIVED via `compute_portfolio_exposure`/`compute_instrument_exposure`/
`compute_strategy_exposure`, never stored on `PortfolioSnapshot` itself --
the identical "never independently trust what can be derived" principle
this milestone's own specification requires for drawdown, applied
symmetrically to exposure. `gross_exposure` is always `>= 0`;
`abs(net_exposure) <= gross_exposure` is enforced at construction.

### `RiskCheckResult`/`RiskEvaluationRequest`/`RiskDecision`

A `RiskCheckResult`'s `severity`/`denial_reason` must cohere with its own
`passed` flag exactly like `paper_trading.risk.RiskCheckResult`'s
identical invariant. A `RiskDecision` with `kind=APPROVED` can carry no
denial reasons and no DENY/HALT-severity check result; a `DENIED`/
`HALTED` decision must carry at least one denial reason, and that set
must be a superset of every DENY/HALT-severity check's own reason --
`HALTED` additionally requires at least one HALT-severity check.
Nothing in Phase 1 yet CONSTRUCTS a `RiskDecision` from real state; these
invariants only constrain what a well-formed one may look like once a
later phase's evaluator exists.

### `CapitalAllocation`/`PositionSizeProposal`

Small, standalone, content-addressed value objects. `utilized_capital`
may never exceed `allocated_capital`. Neither is consumed by anything in
this phase.

## Exceptions

`PortfolioRiskError` (base) plus seventeen subclasses, one per required
category (policy validation, spec identity, snapshot validation, stale
snapshot, stale price, exposure calculation, position sizing, risk
evaluation, risk denial, authorization identity, authorization mismatch,
authorization reuse, portfolio halt, reconciliation, verification,
persistence, recovery) -- see `core.exceptions`'s own "Portfolio risk and
capital management engine (Milestone 9)" section for the full docstring
of each. No common infrastructure is duplicated: `QuantPlatformError`
remains the single root every package's exceptions ultimately derive
from.

## Known limitations (Phase 1 scope, not defects)

- No evaluator, ledger, recovery, CLI, or execution-gateway enforcement
  exists yet -- see "PHASE 1 EXPLICITLY DOES NOT YET IMPLEMENT" above.
- `execution_gateway.paper_bridge.ExecutionIntent.risk_authorization_id`
  currently holds Milestone 8's own `ExecutionAuthorization` id, not a
  `portfolio_risk.authorization.RiskAuthorization` id -- see
  "Relationship to `execution_gateway`" above.
- `RiskAuthorizationStatus`'s event-sourced legal-transition table is
  defined but nothing constructs a transition event yet -- no durable
  authorization ledger exists in this phase.
- `RiskDecision`/`RiskCheckResult` internal-coherence invariants are
  validated at construction, but nothing in this phase constructs one
  from real portfolio/price state -- that is a later phase's evaluator.

## Future phases (not implemented, not started)

- Phase 2: the evaluator itself -- pure functions computing a
  `RiskDecision` from a `RiskEvaluationRequest` against a
  `PortfolioSnapshot`/`PriceSnapshot`/`PortfolioRiskPolicy` triple, one
  check per required policy limit, `most_severe_check_severity` folding
  many `RiskCheckResult`s into one `RiskDecisionKind`.
- A durable, append-only, hash-chained ledger for this package's own
  objects, mirroring `execution_gateway.persistence` exactly.
- `RiskAuthorization` single-use/idempotent-use enforcement, replaying
  `RiskAuthorizationStatus` transition events against the legal-
  transition table Phase 1 already defines.
- `execution_gateway` dispatch-gate integration: refusing to dispatch any
  `ExecutionIntent` without a valid, matching, unconsumed
  `RiskAuthorization` -- and resolving the `risk_authorization_id`
  field-overlap noted above.
- Crash recovery, reconciliation, and independent verification passes,
  mirroring `execution_gateway`'s identical three-part safety net.
- A CLI surface on the shared `ml_cli.py` parser.
- A real, end-to-end acceptance workflow chaining a genuine paper/
  execution session through real risk evaluation.
