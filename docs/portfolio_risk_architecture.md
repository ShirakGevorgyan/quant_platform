# Portfolio Risk and Capital Management Engine (Milestone 9) -- Architecture

## Status: Phase 1 (domain foundation) + Phase 2 (deterministic exposure calculation and pre-trade risk evaluation) + Phase 3 (immutable authorization lifecycle, append-only ledger, persistence, replay, verification) + Phase 4 (execution gateway integration) delivered

This document covers everything Phase 1 through Phase 4 actually deliver
-- exceptions, config schemas, enums, content-addressed policy/spec
identity, portfolio/price snapshot models, risk-decision/risk-
authorization models (Phase 1); the pure exposure/projection/policy-
check/sizing/evaluation layer that turns those models into a real
`RiskDecision` (Phase 2); the durable, append-only, hash-chained risk
ledger, `RiskAuthorization` issuance/reservation/consumption lifecycle,
recovery, reconciliation, independent verification, deterministic replay,
and reporting that turn a `RiskDecision` into an auditable, single-use
authorization (Phase 3); and the mandatory, fail-closed integration with
`quant_platform.execution_gateway` (Milestone 8) that makes a
`RiskAuthorization` an actual PRECONDITION for dispatch, not merely a
durable record alongside it (Phase 4). It explicitly marks every later-
phase concept (CLI expansion, real broker execution) as NOT YET
IMPLEMENTED rather than silently omitting it.

**Phase 3 baseline**: Phase 2 was committed at
`4aac98c` ("Add deterministic portfolio risk evaluation engine"). Phase 3
was built on top of that commit and itself committed at `7ef860c` ("Add
durable portfolio risk authorization lifecycle"). Phase 4 (this section)
is built on top of `7ef860c` and is, per its own governing instructions,
NOT committed -- see `docs/milestone9_phase4_delivery_report.md` for the
exact `git status` at hand-off.

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

## PHASE 3 EXPLICITLY DOES NOT YET IMPLEMENT

Stated plainly, so no later reader mistakes Phase 3's ledger/lifecycle
for a deployed, enforcing risk gate:

- **No execution-gateway enforcement.** `execution_gateway.paper_bridge.
  ExecutionIntent.risk_authorization_id` is still NOT checked against a
  real `portfolio_risk.authorization.RiskAuthorization` anywhere, and
  nothing in `portfolio_risk` is called from `execution_gateway`, nor
  does `portfolio_risk` import anything from `execution_gateway` --
  verified by grepping both directions (see "Relationship to
  `execution_gateway`" below and "SEMANTIC COLLISION DECISION" for the
  precise current state and required future migration, unchanged and
  still deferred by Phase 3).
- **No broker calls, no dummy-broker changes, no live trading, no
  network access, no credentials.** Phase 3 stays entirely within
  `portfolio_risk`'s own durable ledger and lifecycle logic; a "reserved"
  or "consumed" authorization never contacts any broker -- the CALLER
  (a future execution-gateway integration) is responsible for actually
  dispatching, and for recording that dispatch's outcome back via
  `reserve_authorization`/`consume_authorization`.
- **No CLI.** No `ml_cli.py` command exists for this package yet.
- **No session manifest / explicit stage machine.** Phase 3 deliberately
  does NOT introduce a `PortfolioRiskSessionManifest` -- see "Why no
  session manifest" below for the explicit architectural rationale.
- **No pre-existing halt state or losing-streak count is derived by this
  package for `evaluate_risk`.** Phase 3's ledger durably records
  `RiskEvaluationRequest`/`RiskDecision`/lifecycle events, but nothing in
  Phase 3 wires that history back into `evaluate_risk`'s own
  `portfolio_halted`/`consecutive_losses` parameters -- those remain
  explicit, caller-supplied inputs, unchanged from Phase 2. A later phase
  could derive them from the Phase 3 ledger, but Phase 3 itself does not.
- **No acceptance workflow.** No end-to-end test chains a real paper
  session through this package's own risk evaluation and authorization
  lifecycle; only isolated unit/integration tests exist through Phase 3.

## PHASE 2 EXPLICITLY DOES NOT YET IMPLEMENT (superseded in part by Phase 3)

Stated plainly for historical context; items resolved by Phase 3 are
marked so a reader does not need to cross-reference forward:

- **No durable ledger.** *(Resolved by Phase 3 -- see "Append-only risk
  ledger design" below.)* There was no append-only, hash-chained event
  store for this package's own objects through Phase 2.
- **No `RiskAuthorization` is constructed by Phase 2.** *(Still true of
  `evaluate_risk` itself -- Phase 3's `issuance.issue_risk_authorization`
  is a SEPARATE, later step a caller invokes with an `APPROVED`
  `RiskDecision`, not something `evaluate_risk` does internally.)*
  `evaluate_risk` returns a `RiskDecision` (and, when approved, a
  `PositionSizeProposal`/`CapitalAllocation`) -- it does NOT construct a
  `RiskAuthorization`.
- **No crash recovery, no reconciliation, no independent verification
  pass.** *(Resolved by Phase 3 -- see "Recovery", "Reconciliation", and
  "Independent verification" below.)*
- **No execution-gateway enforcement.** *(Still true -- see "PHASE 3
  EXPLICITLY DOES NOT YET IMPLEMENT" above.)*

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
- `exposure.py` (Phase 2) -- `compute_long_gross_exposure`/
  `compute_short_gross_exposure`/`compute_concentration_fraction`/
  `compute_leverage`/`compute_available_cash`/`compute_daily_loss`/
  `compute_total_loss`; re-exports `compute_portfolio_exposure`/
  `compute_instrument_exposure`/`compute_strategy_exposure` from
  `snapshots.py` rather than redefining them.
- `valuation.py` (Phase 2) -- `project_fill_price`, `classify_trade_risk`/
  `TradeRiskClassification`, `project_position`/`PositionProjection`,
  `project_portfolio`/`PortfolioProjection`.
- `checks.py` (Phase 2) -- the 18 required `check_*` functions, each
  returning exactly one `RiskCheckResult`, plus `CHECK_ORDER` (the
  canonical, fixed evaluation/identity order) and
  `NOT_CONFIGURED_LIMIT_SENTINEL`.
- `sizing.py` (Phase 2) -- `max_quantity_by_*` constraint functions,
  `quantize_quantity_down`, `compute_maximum_allowed_quantity`.
- `evaluator.py` (Phase 2) -- `evaluate_risk`/`EvaluationOutcome`, the one
  orchestration function tying every module above together into a final
  `RiskDecision`.
- `ledger.py` (Phase 3) -- `RiskLedgerEntry` (+ `create_risk_ledger_entry`),
  `RiskLedgerEntryKind` (12-member enum), `verify_risk_ledger_chain_
  integrity`, `compute_risk_ledger_physical_digest`/`compute_risk_ledger_
  semantic_digest`, `portfolio_risk_lock`, `PortfolioRiskLedgerStore`,
  `append_ledger_entry`. The append-only, hash-chained, portfolio-
  partitioned durable store -- see "Append-only risk ledger design"
  below.
- `state_machine.py` (Phase 3) -- `RiskAuthorizationStatusEvent` (+
  `create_risk_authorization_status_event`), `resolve_risk_authorization_
  status`, `consumption_identity_for`. Pure, event-sourced status
  reconstruction, mirroring `execution_gateway.state_machine.
  ExecutionOrderStateEvent`/`resolve_execution_order_state` exactly.
- `issuance.py` (Phase 3) -- `issue_risk_authorization`, the single pure
  function turning an `APPROVED` `RiskDecision` into a `RiskAuthorization`.
- `validation.py` (Phase 3) -- `AuthorizationRejectionReason`,
  `AuthorizationUseValidation`, `validate_authorization_use` -- the
  single, pure, stateless function every reserve/consume transaction
  calls before ever appending a ledger entry.
- `idempotency.py` (Phase 3) -- seven durable index builders (`build_
  decision_to_authorization_index`, `build_authorization_payload_index`,
  `build_status_events_index`, `build_authorization_status_index`,
  `build_authorization_consumption_index`, `build_execution_intent_
  index`, `build_consumption_identity_index`), every one reconstructed BY
  SCANNING THE LEDGER, never an in-memory-only set.
- `lifecycle.py` (Phase 3) -- the only module that APPENDS to a risk
  ledger: `record_risk_evaluation_request`/`record_risk_decision`/
  `record_authorization_issuance`, `reserve_authorization`/
  `consume_authorization`, `expire_authorization`/`invalidate_
  authorization`/`revoke_authorization`.
- `recovery.py` (Phase 3) -- `RecoveryAction`, `recover_portfolio_risk_
  session` -- reconstructs solely from durable ledger evidence and
  classifies every authorization's own recovery disposition.
- `reconciliation.py` (Phase 3) -- `RiskReconciliationIssue`/
  `PortfolioRiskReconciliationReport`, `reconcile_portfolio_risk_session`
  -- structured, non-raising issue reporting; raises only on genuine
  structural ledger-reconstruction failure.
- `verification.py` (Phase 3) -- `verify_portfolio_risk_session` --
  independent re-verification against the raw ledger, never a cached
  report/persisted status/in-memory set/caller assertion. See
  "Independent verification honesty classification" below.
- `replay.py` (Phase 3) -- `PortfolioRiskReplayResult`, `compute_replay_
  result`/`assert_replay_deterministic` -- the comparison primitive tests
  use to prove replaying the same operation sequence into a fresh,
  independent store produces byte-identical outcomes.
- `reports.py` (Phase 3) -- `PortfolioRiskSessionReport`, `generate_
  portfolio_risk_session_report` -- deterministic, ledger-derived report
  sections, recomputed fresh on every call.

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
may never exceed `allocated_capital`. In Phase 1, neither was consumed by
anything; Phase 2's `evaluate_risk` now constructs both, but ONLY when
the decision is `APPROVED` (see "Evaluator orchestration" below).

## SEMANTIC COLLISION DECISION

`execution_gateway.paper_bridge.ExecutionIntent.risk_authorization_id`
(Milestone 8) is populated with what this document now names, explicitly
and permanently, the **`execution_bridge_authorization_id`** concept --
`ExecutionAuthorization.execution_authorization_id`, the id proving a
paper order was cleared to bridge from paper trading into the execution
gateway at all. This is DISTINCT from this milestone's own
**`portfolio_risk_authorization_id`** concept -- `RiskAuthorization.
risk_authorization_id` (Phase 1), the id proving a specific quantity/
price was cleared to dispatch against a specific, evaluated portfolio
state.

Phase 2 does NOT perform the cross-milestone rename (renaming
`ExecutionIntent.risk_authorization_id` or repurposing its current
value) -- neither compilation nor any test in this milestone strictly
requires it, and `execution_gateway` remains entirely unmodified in this
phase, per this phase's own explicit scope boundary. Phase 2's own
`RiskEvaluationRequest.risk_policy_id`/`RiskDecision`/`RiskAuthorization`
never read or write `ExecutionIntent.risk_authorization_id` -- Phase 2
operates exclusively on Milestone 9's own domain (`RiskEvaluationRequest`
in, `RiskDecision`/`EvaluationOutcome` out).

**The required future migration**, recorded here for whichever phase
implements execution-gateway enforcement: `ExecutionIntent` must gain a
SECOND, distinct field -- e.g. `portfolio_risk_authorization_id: str |
None` -- so a session can hold BOTH the existing
`execution_bridge_authorization_id` (renamed from today's
`risk_authorization_id`) AND the new `portfolio_risk_authorization_id`
side by side, without collapsing two genuinely different concepts into
one field name. Renaming the existing field outright (rather than adding
a second one) would be a breaking change to already-committed Milestone 8
identity payloads and is NOT recommended without a dedicated migration
plan.

## Lifecycle state machine (Phase 3)

`RiskAuthorizationStatus` has exactly SIX members (extended from Phase
1's original four -- `ISSUED`, `CONSUMED`, `EXPIRED`, `REVOKED` -- by
adding `RESERVED` and `INVALIDATED`, the smallest state set expressing
Phase 3's own required semantics):

```
ISSUED --> RESERVED --> CONSUMED         (terminal)
   |            |
   |            +----> EXPIRED           (terminal)
   |            +----> INVALIDATED       (terminal)
   |            +----> REVOKED           (terminal)
   +----> EXPIRED / INVALIDATED / REVOKED (terminal, directly from ISSUED)
```

`CONSUMED`, `EXPIRED`, `INVALIDATED`, `REVOKED` are all terminal -- no
legal transition exits any of them (enforced by `models.
is_legal_risk_authorization_status_transition` and independently
re-checked by `state_machine.RiskAuthorizationStatusEvent.__post_init__`
AND `resolve_risk_authorization_status`, the same defense-in-depth
double-check `execution_gateway.state_machine.ExecutionOrderStateEvent`
already uses). There is no ambiguous `UNKNOWN`/pending approval state
anywhere in this enum, mirroring `RiskDecisionKind`'s identical
three-member, no-`UNKNOWN` discipline from Phase 1.

Status is ALWAYS reconstructed by replaying an authorization's own
`RiskAuthorizationStatusEvent`s from an implicit initial state (`ISSUED`)
via `resolve_risk_authorization_status` -- never inferred from
in-memory-only state, never cached across calls. `resolve_risk_
authorization_status` additionally tracks the `consumption_identity`
bound at the authorization's own `RESERVED` transition and requires any
subsequent `CONSUMED` transition to carry the SAME identity, raising
`PortfolioRiskRecoveryError` otherwise -- a structural, replay-level
defense against a coherently re-chained ledger tamper (see "Defects
found and fixed" below, defect #5).

## Authorization issuance rules (Phase 3)

`issuance.issue_risk_authorization(*, request, decision, authorization_
sequence, event_time) -> RiskAuthorization` is a pure function:

- Only an `APPROVED` `RiskDecision` produces a `RiskAuthorization` --
  `DENIED`/`HALTED` raise `RiskEvaluationError` immediately, never
  producing a usable object.
- Every bound identity (`execution_intent_id`, `execution_session_id`,
  `portfolio_id`, `portfolio_snapshot_id`, `price_snapshot_id`,
  `risk_policy_id`) is verified to match between `request` and
  `decision` before issuance -- a mismatch raises rather than silently
  picking one side.
- `event_time` and `authorization_sequence` are always caller-supplied
  (never `utc_now()`/`uuid4()`/`random`/a temp path/a process-specific
  identity source) -- the same Phase 1 discipline `RiskAuthorization`
  itself already enforces, continued unchanged into Phase 3's own
  issuance function.
- The resulting `risk_authorization_id` is Phase 1's existing content-
  addressed id -- Phase 3 adds no second identity scheme; issuance is
  purely "construct the object Phase 1 already defined, from a decision
  that is durably APPROVED."

## Reservation, consumption, and idempotency semantics (Phase 3)

`validation.validate_authorization_use` is the single, pure, stateless
gate every reserve/consume transaction passes through BEFORE any ledger
entry is appended (`lifecycle._record_economic_use_transition`). Given
`current_status`, `bound_consumption_identity` (the identity bound by an
earlier `RESERVED` transition, or `None`), a `target_status`, and the
attempted binding/`consumption_identity`, it resolves to exactly one of:

1. **`BINDING_MISMATCH`** -- the attempted intent/session/portfolio/
   snapshot/policy/quantity/price does not reproduce the authorization's
   own `risk_authorization_id` (checked via `authorization.verify_risk_
   authorization_binding`'s single recompute-and-compare, catching every
   cross-intent/cross-session/cross-portfolio/cross-snapshot/cross-policy
   mismatch and any quantity/price change in one check).
2. **`EXPIRED`** -- `evaluation_time` is beyond an optional, caller-
   supplied `expiry_time`.
3. **An exact retry, idempotently approved** -- `current_status is
   target_status` AND the attempted `consumption_identity` matches the
   ALREADY-bound one: no new ledger entry is appended;
   `lifecycle.py` returns the prior event unchanged.
4. **`CONFLICTING_CONSUMPTION`** -- either `current_status is
   target_status` under a DIFFERENT `consumption_identity` (a same-
   target retry with a different identity), OR a NEW transition (e.g.
   the primary `RESERVED -> CONSUMED` step) whose `consumption_identity`
   does not match whatever identity was bound by an earlier `RESERVED`
   transition -- **both are fail-closed rejections; see defect #7 in
   "Defects found and fixed" below for why the second case required a
   dedicated fix.**
5. **`STATUS_DOES_NOT_PERMIT_USE`** -- the transition is not legal per
   the state machine above (e.g. attempting to consume an `ISSUED`,
   never-reserved authorization, or reserving an already-terminal one).
6. **A genuinely new, approved transition** -- everything else.

Every rejection is durably recorded as a `RISK_AUTHORIZATION_USE_
REJECTED` ledger entry BEFORE the caller's exception is raised -- a
rejected attempt is never silently invisible in the ledger's own audit
trail (see "Concurrency behavior" below for how this guarantee is
preserved even under a genuine race).

**Exact binding, not bounded slippage** (a deliberate Phase 3 decision):
quantity and price must match EXACTLY what was evaluated. There is no
repository-level precedent for bounded/partial-use semantics on any
other milestone's authorization-shaped object, and exact binding is
simpler and more easily independently verifiable -- a later phase could
introduce bounded semantics deliberately, but Phase 3 does not.

## Append-only risk ledger design (Phase 3)

`PortfolioRiskLedgerStore` persists to
`{storage_root}/portfolio_risk_ledgers/{portfolio_id}/events.jsonl` --
one file per portfolio (see "Ledger partitioning" below), mirroring
`execution_gateway.persistence`'s identical `ExecutionLedgerEntry`/
`ExecutionSessionEventStore` shape and on-disk conventions (canonical
JSON, one entry per line, `os.fsync` after every append, a `.lock` file
protecting the append itself).

**Two distinct hashes per entry** (mirroring Milestone 8's identical
pattern exactly): `entry_hash` is a SELF-validating hash of `payload`
ALONE, checked in `RiskLedgerEntry.__post_init__` on every load -- a
payload tampered directly in the JSONL file fails to even CONSTRUCT,
regardless of whether that entry kind happens to have its own downstream
domain check. `entry_id` is a hash of the WHOLE entry envelope, used as
the chain link (`previous_entry_hash` is literally the PRIOR entry's own
`entry_id`).

**Physical vs. semantic integrity, verified independently:**

- `verify_risk_ledger_chain_integrity`/`compute_risk_ledger_physical_
  digest` prove PHYSICAL storage integrity alone -- contiguous
  `entry_sequence`, correct `previous_entry_hash` linkage. They say
  NOTHING about whether the ECONOMIC content inside each payload is
  coherent.
- `compute_risk_ledger_semantic_digest` hashes ECONOMIC content only:
  `entry_sequence` + `entry_kind` + `payload`, excluding `entry_id`/
  `entry_hash`/`previous_entry_hash`/`recorded_time`/`event_time` at the
  ENTRY level (operational/physical bookkeeping only -- any economically
  meaningful timestamp already lives INSIDE a domain object's own JSON,
  nested in `payload`, and participates via that object's own already-
  established identity rules).

**Coherent-re-chaining defense**: an attacker who removes/reorders
entries and recomputes the physical chain to still validate is NOT
caught by `verify_risk_ledger_chain_integrity` -- it is caught by
DOMAIN-level replay instead (`state_machine.resolve_risk_authorization_
status` detecting a `from_state` discontinuity, or the consumption-
identity consistency check described above). This is the primary defense
against a sophisticated ledger tamper, and it is exercised by dedicated
tests in `test_portfolio_risk_reconciliation.py`/`test_portfolio_risk_
verification.py` (`TestReconstructionFailureIsCritical`/
`TestCoherentReChainingCaughtByReplayNotChainIntegrity`).

**Twelve ledger entry kinds**: `RISK_EVALUATION_REQUESTED`, `RISK_
DECISION_RECORDED`, `RISK_AUTHORIZATION_ISSUED`, `RISK_AUTHORIZATION_
RESERVED`, `RISK_AUTHORIZATION_CONSUMED`, `RISK_AUTHORIZATION_EXPIRED`,
`RISK_AUTHORIZATION_INVALIDATED`, `RISK_AUTHORIZATION_REVOKED`, `RISK_
AUTHORIZATION_USE_REJECTED`, `RECOVERY_STARTED`, `RECOVERY_COMPLETED`,
`VERIFICATION_COMPLETED`.

**Append invariants**: an identical append (same `entry_id` at the same
`entry_sequence`) is idempotently absorbed (a no-op, not an error); a
conflicting payload at the same `entry_sequence` is rejected
(`RiskAuthorizationReuseError`); a sequence gap is rejected
(`PortfolioRiskPersistenceError`); a wrong `previous_entry_hash` is
rejected; every append is protected by `portfolio_risk_lock` (below).

### Ledger partitioning

Partitioned by `portfolio_id` -- NOT by `execution_session_id` like
Milestone 8's `ExecutionLedgerEntry` -- a deliberate architectural choice:
a single portfolio persists across many execution sessions over its
lifetime, and this package's own domain concept is fundamentally
per-portfolio risk management, not per-session. `lifecycle.py` always
partitions by `authorization.portfolio_id` -- the authorization's own
TRUE, content-addressed owner -- never by a caller-CLAIMED portfolio id,
which may differ from the true one in a cross-portfolio attack attempt
(a `BINDING_MISMATCH` rejection is itself recorded in the TRUE owner's
own ledger, exactly where an attempted misuse against that portfolio
belongs).

## Durable idempotency indexes (Phase 3)

`idempotency.py`'s seven index builders are ALWAYS reconstructed by
scanning the ledger's own raw entries -- never an in-memory-only set.
`build_authorization_status_index`/`build_authorization_consumption_
index` seed their key universe from `build_authorization_payload_index`
(every `RISK_AUTHORIZATION_ISSUED` entry), NOT from the status-events
index alone -- an authorization issued but never subsequently touched has
an EMPTY event list, which correctly resolves to the implicit `ISSUED`
state; seeding from events alone would silently omit such an
authorization from the index entirely (see defect #1 below).

## Recovery (Phase 3)

`recovery.recover_portfolio_risk_session(*, portfolio_id, store,
recovery_time) -> list[RecoveryAction]` reconstructs SOLELY from durable
ledger evidence and classifies every authorization into a closed
vocabulary:

- `issued_only` -- never reserved; safe to reserve fresh.
- `reserved_unresolved_blocked` -- reserved with no terminal outcome
  recorded; recovery NEVER authorizes blind reuse of this authorization
  -- it remains blocked until a caller either records the exact same
  `consumption_identity` (idempotently accepted) or explicitly
  expires/invalidates/revokes it. Recovery itself never appends a
  lifecycle-transition entry -- only `RECOVERY_STARTED`/`RECOVERY_
  COMPLETED` bookkeeping.
- `terminal_consumed`/`terminal_expired`/`terminal_invalidated`/
  `terminal_revoked` -- already resolved, no action possible or needed.

Recovery does NOT pretend to resolve external execution ambiguity --
there is no execution integration in this milestone, so "was the order
actually placed with the broker" is not a question recovery can or does
answer. It answers only "what does the ledger's own durable evidence
say about this authorization's lifecycle state," which a future
execution-gateway integration would combine with its OWN broker-side
recovery evidence.

## Reconciliation (Phase 3)

`reconciliation.reconcile_portfolio_risk_session(*, portfolio_id,
ledger) -> PortfolioRiskReconciliationReport` NEVER raises for an
ordinary mismatch -- it returns a structured `RiskReconciliationIssue`
(severity `INFO`/`WARNING`/`BLOCKING`/`CRITICAL`) for each: an
authorization bound to a non-`APPROVED` decision, an orphan lifecycle
event (references an authorization never issued), cross-portfolio
contamination (an authorization's own declared `portfolio_id` doesn't
match the ledger it's recorded in), an approved decision never issued
into an authorization (`WARNING` -- may be a legitimate caller choice),
and an unresolved `RESERVED` authorization (`INFO` -- recovery's own job
to classify further). Only genuine STRUCTURAL corruption -- the ledger
cannot be reconstructed at all (a raised `PortfolioRiskPersistenceError`/
`PortfolioRiskRecoveryError` from the underlying idempotency-index
builders) -- surfaces as a single `internal_reconstruction_failed`
`CRITICAL` issue, never an uncaught exception from reconciliation itself.

## Independent verification honesty classification (Phase 3)

`verification.verify_portfolio_risk_session` never trusts a cached
report, a persisted final status, an in-memory idempotency set, or a
caller assertion -- every check independently reconstructs from the
ledger's own raw entries. Documented explicitly, per the governing
instruction, as a 3-tier honesty classification (verbatim from
`verification.py`'s own module docstring):

- **STRUCTURALLY INDEPENDENT** (this module's entire scope): ledger
  physical chain integrity, portfolio ownership, idempotency-index
  reconstruction (itself replaying every authorization's lifecycle via
  `state_machine.resolve_risk_authorization_status`, catching a
  coherently re-chained tamper), APPROVED-only issuance, forged-identity
  detection (recomputing each authorization's own content id from its
  ledger-recorded payload), single-use-identity coherence, and orphan-
  event detection. None of these require trusting a cached report, a
  persisted status, an in-memory set, or a caller assertion -- every one
  is a pure recomputation from the ledger's own raw entries.
- **NOT INDEPENDENTLY RE-VERIFIED** (an honest, explicit limitation, not
  an oversight): this module does NOT re-run Phase 2's evaluator
  (`evaluator.evaluate_risk`) against the original `PortfolioSnapshot`/
  `PriceSnapshot`/`PortfolioRiskPolicy` to confirm a recorded
  `RiskDecision`'s own 18 checks were computed correctly in the first
  place -- the ledger only durably stores the DECISION's own already-
  serialized JSON, not the full snapshot/policy inputs it was computed
  from. Re-deriving that would require this module to also durably store
  (or have access to) the original snapshots/policy, which is out of
  Phase 3's own scope (no snapshot/policy artifact store exists in this
  milestone). This module verifies the decision's OWN INTERNAL coherence
  (already guaranteed at construction by Phase 1/2's own `RiskDecision.
  __post_init__`) and its BINDING into an authorization -- never whether
  the decision's 18 checks were the economically correct ones for the
  real portfolio state.

`record: bool = True` distinguishes a durably-auditable verification
(the default, appending a `VERIFICATION_COMPLETED` entry -- used by
`reports.py`) from a side-effect-free measurement (`record=False` --
used by `replay.py`'s own comparison utility, where an unconditional
append would make repeated comparisons non-idempotent; see defect #3
below).

## Deterministic replay (Phase 3)

This package has no session RUNNER to replay wholesale (unlike
`execution_gateway.replay`, which re-runs a whole execution session end
to end) -- Phase 3's own "session" is simply every ledger entry recorded
for one `portfolio_id`, built by individual, discrete calls to
`lifecycle.py`'s transaction functions. `replay.compute_replay_result`/
`assert_replay_deterministic` provide the comparison primitive tests use
to prove that replaying the SAME sequence of operations into a FRESH,
independent `PortfolioRiskLedgerStore` -- a different temp directory, a
different filesystem-root shape, a different `verification_time` label,
a separate OS process with a different `PYTHONHASHSEED` -- produces an
identical `PortfolioRiskReplayResult` (semantic digest, authorization id
set, verification critical-issue count, reconciliation outcome).
`canonical_json_bytes`'s `sort_keys=True` encoding makes hash-seed
independence structural, not merely observed -- `set`/`frozenset`
iteration order never reaches a digest; every digest is computed from a
list built in a fixed, deterministic order.

## Concurrency behavior (Phase 3)

`PortfolioRiskLedgerStore`'s own lock (`portfolio_risk_lock`, wrapping
`historical.locking.DatasetLock` via `ml.concurrency.experiment_lock`)
FAILS FAST rather than blocking/retrying on contention (`historical.
locking.DatasetLock`'s own documented design choice) -- a losing caller
sees `PortfolioRiskLockError` and is expected to retry.

`lifecycle._record_economic_use_transition`/`_record_administrative_
transition` each wrap their own read-validate-append cycle in a bounded
(20-attempt) internal retry loop: if two callers both read state BEFORE
either has written (a genuine race), both can pass validation against
the SAME stale view and then race the final `append_ledger_entry` call
itself; the loser re-reads fresh state and re-validates, resolving to
either an idempotent exact-retry absorption or a properly AUDITED
`RISK_AUTHORIZATION_USE_REJECTED` rejection -- never a bare, unrecorded
storage-layer exception (see defect #6 below).

Required scenarios, all covered by dedicated real-`threading.Thread`
tests in `test_portfolio_risk_ledger_concurrency.py`: two threads
attempting the same exact reservation (both succeed, exactly one ledger
entry results); two threads attempting conflicting reservation payloads
(exactly one wins, the loser is rejected AND audited); duplicate exact
consumption (absorbed idempotently); conflicting second consumption
(one wins, one audited-rejected); an expiry race against a reservation
(both orderings are individually legal -- the ledger always resolves to
exactly one coherent final state, never corruption); a sequence-append
race among many threads issuing distinct, non-conflicting authorizations
concurrently (no ledger corruption, contiguous gapless sequence). In
every case: one economic use, an exact duplicate absorbed idempotently,
a genuine conflict rejected and durably audited, and the ledger's own
physical chain integrity holds afterward.

## Why no session manifest (Phase 3)

Phase 3 deliberately does NOT introduce a `PortfolioRiskSessionManifest`
or an explicit stage machine (`CREATED`/`EVALUATED`/`AUTHORIZED`/
`RESERVED`/`CONSUMED`/`VERIFYING`/`COMPLETED`/`FAILED`/`TERMINATED`).
The authorization ledger itself already IS the durable state: every
question a manifest's "current stage" field would answer (`has this been
evaluated? issued? reserved? consumed? verified?`) is already answerable
-- more honestly -- by replaying the ledger via `idempotency.py`'s index
builders and `verification.verify_portfolio_risk_session`, which
independently reconstruct the truth rather than trusting a separately-
persisted status field that could drift from the ledger's own evidence.
Introducing a manifest would create exactly the two-sources-of-truth
problem this milestone's own "ledger-derived vs persisted status" and
"report cannot override ledger truth" requirements are designed to avoid
(see the dedicated `TestReportCannotOverrideLedgerTruth` test in
`test_portfolio_risk_reconciliation.py`). A future phase could still
introduce one if a cross-portfolio, cross-authorization SESSION concept
emerges that genuinely cannot be expressed as "the ledger for one
portfolio_id" -- Phase 3's own domain does not need one.

## Defects found and fixed during Phase 3's own development

Eight real, confirmed defects were found (mostly via manual sanity
scripts BEFORE writing the full test suite, and via the adversarial
concurrency tests themselves) and fixed at root cause, each with a
regression test:

1. **`build_authorization_status_index`/`build_authorization_consumption_
   index` silently omitted issued-but-never-touched authorizations** --
   both seeded their key universe from status EVENTS alone, excluding any
   authorization with zero subsequent transitions. Fixed by seeding from
   `build_authorization_payload_index`'s keys instead.
2. **`RiskLedgerEntry` had no self-validating hash distinct from its
   chain-linking id** -- a payload tampered in the JSONL file (with
   `entry_id` left unchanged) was only caught opportunistically by
   whichever downstream domain check happened to exist for that entry
   kind. Fixed by adding the `entry_hash`/`entry_id` two-hash design (see
   "Append-only risk ledger design" above).
3. **`verify_portfolio_risk_session`'s unconditional `VERIFICATION_
   COMPLETED` append made `replay.compute_replay_result` non-idempotent**
   -- two independently-built identical scenarios produced different
   semantic digests because an earlier internal verification call had
   mutated one store but not the other. Fixed by adding `record: bool =
   True`; `replay.py` calls with `record=False`.
4. **`create_risk_ledger_entry` computed `entry_hash` before payload
   validation ran** -- a raw `Decimal` in a payload produced a bare,
   untyped `TypeError` from `json.dumps` instead of the intended
   `PortfolioRiskPersistenceError`. Fixed by extracting `_validate_
   payload_shape` and calling it before any hashing.
5. **`resolve_risk_authorization_status` did not check `consumption_
   identity` consistency between a `RESERVED` and its later `CONSUMED`
   transition** -- a directly-tampered ledger entry (bypassing
   `lifecycle.py`'s own write-time gate entirely) could coherently claim
   an authorization was consumed under a DIFFERENT economic identity than
   it was reserved under, and replay would not catch it. Fixed by
   tracking the bound identity through replay and raising
   `PortfolioRiskRecoveryError` on a mismatch -- see "Coherent-re-
   chaining defense" above.
6. **A race-losing thread's rejection was a bare, unaudited storage-layer
   exception** -- two callers reading state before either had written
   could both pass validation against the same stale view, then race the
   final append; the loser got a raw `RiskAuthorizationReuseError` from
   `PortfolioRiskLedgerStore.append` with NO ledger entry recorded for
   its own attempt at all, violating `lifecycle.py`'s own stated
   "never silently invisible" audit-trail invariant. Fixed by wrapping
   `_record_economic_use_transition`/`_record_administrative_transition`'s
   read-validate-append cycle in a bounded retry loop that re-validates
   against fresh state after losing an append race -- see "Concurrency
   behavior" above.
7. **`validate_authorization_use` never checked `consumption_identity`
   consistency for a NEW transition** -- the most serious defect found
   this phase. The `CONFLICTING_CONSUMPTION` check only fired when
   `current_status is target_status` (a same-target retry); the primary
   `RESERVED -> CONSUMED` step is a NEW transition and fell straight
   through to unconditional approval WITHOUT ever comparing
   `consumption_identity` against the identity bound at `RESERVED` --
   meaning the single-economic-use invariant was not actually enforced on
   the main consume path at all prior to this fix. Found via the
   adversarial concurrency tests (a race between consuming under the
   reserved identity and a different one both "succeeded"). Fixed by
   adding an explicit check: any NEW transition into a consumption-
   identity-carrying state must match a previously-bound identity, if one
   exists.
8. **A Windows-specific lock-release race raised an uncaught
   `PermissionError`** -- `historical.locking.DatasetLock.release`'s bare
   `Path.unlink(missing_ok=True)` (shared infrastructure, already
   documented as having a known Windows stale-lock-reclaim limitation on
   the ACQUIRE side) can raise a sharing-violation `PermissionError` on
   the RELEASE side under genuine thread contention, propagating
   uncaught out of `ml.concurrency.experiment_lock`'s `finally` block --
   a type `portfolio_risk_lock`'s existing `except ExperimentLockError`
   translation did not catch. Fixed LOCALLY (not by modifying the shared
   `historical.locking` module, which is outside Phase 3's own scope) by
   also catching `OSError` in `ledger.portfolio_risk_lock` and
   translating it into the same, already-retryable `PortfolioRiskLockError`.

Defects 1-4 were found via manual sanity scripts written between modules,
before the corresponding test file existed. Defects 5-8 were found via
the dedicated adversarial-audit and concurrency test files themselves
(discovering defects 5-8 is, in fact, the entire reason those test files
exist). No defect was ever classified as an acceptable limitation to
avoid fixing it.

## Exposure accounting equations (Phase 2)

Every figure below is DERIVED from a `PortfolioSnapshot`'s own
already-validated fields via `exposure.py` -- nothing is independently
stored or trusted.

- `long_gross_exposure = sum(abs(market_value) for p in positions if p.signed_quantity > 0)`
- `short_gross_exposure = sum(abs(market_value) for p in positions if p.signed_quantity < 0)`
- `portfolio_gross_exposure = long_gross_exposure + short_gross_exposure` (via `snapshots.compute_portfolio_exposure`, Phase 1)
- `portfolio_net_exposure = sum(market_value for p in positions)` (signed; Phase 1)
- `concentration_fraction = max(per-instrument gross exposure, aggregated across strategies) / portfolio_gross_exposure`, `0` when flat
- `leverage = portfolio_gross_exposure / equity` -- **raises `ExposureCalculationError` when `equity <= 0`** (undefined, never a sentinel `0`/`Infinity`); `checks.check_leverage_limit` converts this into an automatic HALT-severity failure regardless of whether a numeric leverage limit is configured
- `available_cash = cash - minimum_cash_buffer` (may be negative -- never clamped, so a breach is never hidden)
- `daily_loss = max(0, daily_start_equity - equity)` -- the day's TOTAL (realized + unrealized) equity decline
- `total_loss = max(0, -(realized_pnl + unrealized_pnl))` -- total pnl loss since inception

**Documented accounting-model decision for `max_daily_realized_loss`**:
`PortfolioSnapshot` (Phase 1) carries `realized_pnl` (cumulative SINCE
INCEPTION) and `daily_start_equity` (a day-scoped EQUITY baseline), but
no separate "realized pnl at the start of today" baseline. Phase 2
therefore checks `max_daily_realized_loss` against `daily_loss` as
defined above (the day's TOTAL equity decline, not isolated to the
realized component) -- an explicit, documented Known Limitation (see
below), never a silent approximation. `max_total_loss` is checked against
the genuinely inception-scoped `total_loss`, which mirrors `paper_trading.
risk.evaluate_continuous_risk`'s own `-(realized_pnl + unrealized_pnl)`
formula exactly (that function calls the equivalent figure
"`daily_loss`" only because `paper_trading.portfolio.PortfolioState` has
no day-boundary concept at all).

## Post-trade projection rules (Phase 2)

`valuation.project_position`/`project_portfolio` mirror `paper_trading.
accounting.apply_fill_to_position`'s SAME weighted-average-cost /
realize-on-reduce logic, Decimal-native, adapted to `PositionSnapshot`'s
lighter shape. GROSS-PRICE-ONLY: a BUY is assumed filled at `price.ask`,
a SELL at `price.bid` (mirroring `execution_gateway.dummy_broker`'s own
MARKET-order convention) -- **no commission, spread, or slippage cost is
modeled anywhere in Phase 2**, the same documented simplification
Milestones 7/8 already carry for their own fills. `PortfolioSnapshot.cash`
moves by exactly `quantity * fill_price * contract_multiplier`.

Every required scenario is supported by the SAME small function:
opening a new long/short position, adding to an existing position,
reducing, fully closing, and crossing through zero in either direction.
`classify_trade_risk` distinguishes:

- **INCREASING** -- opens from flat, adds in the same direction, OR
  crosses through zero into a new direction (even if the resulting
  magnitude is SMALLER -- a direction change always counts as
  risk-increasing, never risk-reducing).
- **REDUCING** -- strictly decreases magnitude without changing
  direction, including a full close.
- **NEUTRAL** -- degenerate (signed quantity unchanged); unreachable in
  practice given `RiskEvaluationRequest.quantity > 0` is already a Phase 1
  invariant, but never assumed unreachable by the code itself.

The projection never mutates the original `PortfolioSnapshot` (trivially
guaranteed -- both it and `PositionSnapshot` are frozen dataclasses); it
always returns a brand-new, independently `__post_init__`-validated
object.

## Policy checks (Phase 2) -- the 18 required checks

`checks.py` implements exactly 18 checks, run and appended in the FIXED
`CHECK_ORDER` sequence (never dict/set iteration) so `RiskDecision.
check_results` -- and therefore `RiskDecision.risk_decision_id` -- is
deterministic:

| # | `check_identity` | Severity when breached | Denial reason |
|---|---|---|---|
| 1 | `order_notional_limit` | DENY | `ORDER_NOTIONAL_LIMIT_EXCEEDED` |
| 2 | `position_notional_limit` | DENY | `POSITION_NOTIONAL_LIMIT_EXCEEDED` |
| 3 | `instrument_gross_exposure_limit` | DENY | `INSTRUMENT_GROSS_EXPOSURE_LIMIT_EXCEEDED` |
| 4 | `strategy_gross_exposure_limit` | DENY | `STRATEGY_GROSS_EXPOSURE_LIMIT_EXCEEDED` |
| 5 | `portfolio_gross_exposure_limit` | DENY | `PORTFOLIO_GROSS_EXPOSURE_LIMIT_EXCEEDED` |
| 6 | `portfolio_net_exposure_limit` | DENY | `PORTFOLIO_NET_EXPOSURE_LIMIT_EXCEEDED` |
| 7 | `concentration_fraction_limit` | DENY | `CONCENTRATION_LIMIT_EXCEEDED` |
| 8 | `leverage_limit` | DENY (HALT if leverage undefined) | `LEVERAGE_LIMIT_EXCEEDED` |
| 9 | `minimum_cash_buffer` | DENY | `CASH_BUFFER_BREACHED` |
| 10 | `daily_realized_loss_limit` | HALT | `DAILY_REALIZED_LOSS_LIMIT_EXCEEDED` |
| 11 | `total_loss_limit` | HALT | `TOTAL_LOSS_LIMIT_EXCEEDED` |
| 12 | `drawdown_limit` | HALT | `DRAWDOWN_LIMIT_EXCEEDED` |
| 13 | `consecutive_losses_limit` | HALT | `CONSECUTIVE_LOSSES_LIMIT_EXCEEDED` |
| 14 | `stale_price` | DENY | `STALE_PRICE` |
| 15 | `stale_portfolio_snapshot` | DENY | `STALE_PORTFOLIO_SNAPSHOT` |
| 16 | `portfolio_halted` | HALT | `PORTFOLIO_HALTED` |
| 17 | `reduce_only_validity` | DENY | `INCOHERENT_EVALUATION_STATE` |
| 18 | `missing_or_inconsistent_valuation_data` | DENY | `INCOHERENT_EVALUATION_STATE` |

**Severity rationale**: checks 1-9 are evaluated against the
POST-TRADE PROJECTED state, so a risk-REDUCING trade naturally improves
its own measured values and is never blocked by these checks without any
special-casing -- DENY (this one order) is the correct remedy. Checks
10-13 reflect ACCOUNT-WIDE health, independent of any single order --
HALT (portfolio-wide), mirroring `paper_trading.risk`'s own mapping of
loss/drawdown triggers to `FLATTEN_SIMULATED_POSITIONS`/
`HALT_NEW_ORDERS` (more severe than a single `REJECT_ORDER`). Checks 17
and 18 reuse `RiskDenialReason.INCOHERENT_EVALUATION_STATE` (Phase 1's own
documented "fail-closed catch-all") rather than requiring two new enum
members -- both genuinely represent "the request was not a coherent,
evaluable state," not a distinct economic limit breach.

Every check ALWAYS produces exactly one `RiskCheckResult`, including
passing ones -- `checks.NOT_CONFIGURED_LIMIT_SENTINEL` (a very large,
finite `Decimal`) is reported as `limit_value` when a `PortfolioRiskPolicy`
field is `None`, since `RiskCheckResult.limit_value` is a required,
FINITE field (Phase 1) with no `Decimal("Infinity")` sentinel available.
`RiskCheckSeverity.WARNING` is defined by Phase 1 but never produced by
any of the 18 checks in Phase 2 -- every check's failure mode is either
DENY (order-specific) or HALT (account-wide); `WARNING` is reserved for a
future phase that wants a soft, non-blocking signal.

## APPROVED / DENIED / HALTED aggregation rules (Phase 2)

`evaluator._aggregate_decision_kind` is the single function that turns 18
`RiskCheckResult`s into one `RiskDecisionKind`:

1. **Completeness is verified FIRST, unconditionally**: `check_results`
   must contain EXACTLY the 18 identities `CHECK_ORDER` requires --
   raises `RiskEvaluationError` otherwise. An `APPROVED` decision is
   therefore structurally impossible unless every required check
   actually ran, regardless of what the checks that DID run concluded.
2. `most_severe_check_severity` (Phase 1) folds every check's severity
   via `max`; `HALT` present anywhere -> `HALTED`; else `DENY` present
   anywhere -> `DENIED`; else -> `APPROVED`.
3. `denial_reasons` is the SORTED (deterministic, never set-iteration-
   order) set of every failing check's own `denial_reason`.

`evaluate_risk` itself is fail-closed at a layer BEFORE aggregation:
`_verify_request_bindings` recomputes every referenced object's OWN
identity (via Phase 1's `verify_portfolio_snapshot_identity`/
`verify_price_snapshot_identity`/`compute_portfolio_risk_spec_id`,
Phase 2's own additions to `snapshots.py`) and raises `RiskEvaluationError`
-- never returns a decision -- on a forged/tampered snapshot, a
cross-instrument price, a cross-portfolio mismatch, or a cross-policy
mismatch. These are CALLER/INTEGRATION defects, not economic conditions a
`RiskDecision` should ever represent. No `except Exception`/bare `except`
exists anywhere in `evaluator.py` -- a genuine programming defect
propagates as a real exception, visible to tests and callers, never
silently turned into an approval OR a denial.

## Position-sizing rules (Phase 2)

`sizing.py` implements ONLY policy-limit-based sizing (no volatility/
Kelly/ATR/model-based capital allocation, per this phase's own scope
boundary). Each `max_quantity_by_*` function returns the additional
quantity ONE specific limit still permits (`None` when unconstrained);
`compute_maximum_allowed_quantity` combines every applicable constraint
with the caller's own `requested_quantity` via `min` (so the result can
NEVER exceed what was requested), then quantizes the result DOWN
(conservative, never up) to an explicit, caller-supplied `quantity_step`
(a non-positive step is rejected). `max_quantity_by_cash_buffer` only
constrains a BUY -- a SELL always increases cash under this phase's
gross-price-only convention and is never buffer-constrained, giving
short orders symmetric, side-aware treatment rather than an accidental
asymmetry.

## Evaluator orchestration (Phase 2)

`evaluate_risk(request, portfolio, price, spec, evaluation_time,
portfolio_halted, consecutive_losses, contract_multiplier,
decision_sequence) -> EvaluationOutcome` is the one function tying every
module above together:

1. Verify request bindings (forged-identity/cross-reference defense,
   above) -- raises, never denies, on failure.
2. Project the portfolio forward by the requested trade
   (`valuation.project_portfolio`).
3. Compute every exposure figure against the PROJECTED state
   (`exposure.py`).
4. Run all 18 checks, in `CHECK_ORDER`, against those projected figures.
5. Aggregate to `RiskDecisionKind` (above); construct the `RiskDecision`
   via Phase 1's `create_risk_decision`.
6. ONLY when `APPROVED`: construct a `PositionSizeProposal` (the
   confirmed request) and a `CapitalAllocation` (`allocated_capital =
   policy.max_strategy_gross_exposure` when configured, else the
   projected strategy exposure itself; `utilized_capital` = the projected
   strategy exposure -- `utilized_capital <= allocated_capital` holds
   automatically whenever the decision is `APPROVED`, since check #4
   already confirmed it when a limit was configured).

`portfolio_halted`/`consecutive_losses`/`contract_multiplier` are
explicit, MANDATORY, caller-supplied parameters to `evaluate_risk` --
NOT fields on `RiskEvaluationRequest` (a Phase 1 model Phase 2 does not
modify). Phase 2 has no durable ledger yet to derive a pre-existing halt
state or a losing-streak count from real history; a later phase's
persistence layer would supply these. `evaluation_time` is likewise
always caller-supplied -- `evaluate_risk` never reads the wall clock, and
`checks.check_portfolio_halted` does NOT itself consult `policy.
allow_reduce_only_during_halt` (a later phase's dispatch gate is
responsible for consulting that flag before actually blocking a
reduce-only order during a halt -- see Known Limitations).

## Exceptions

`PortfolioRiskError` (base) plus eighteen subclasses, one per required
category (policy validation, spec identity, snapshot validation, stale
snapshot, stale price, exposure calculation, position sizing, risk
evaluation, risk denial, authorization identity, authorization mismatch,
authorization reuse, portfolio halt, reconciliation, verification,
persistence, recovery, and Phase 3's own `PortfolioRiskLockError` for
ledger-lock contention/filesystem races) -- see `core.exceptions`'s own
"Portfolio risk and capital management engine (Milestone 9)" section for
the full docstring of each. No common infrastructure is duplicated:
`QuantPlatformError` remains the single root every package's exceptions
ultimately derive from. `PortfolioRiskLockError` is raised by `ledger.
portfolio_risk_lock` both for a genuine lock-acquisition failure
(`ExperimentLockError`, fail-fast contention) AND for a known Windows-
specific release-side filesystem race (see defect #8 below) -- both are
caller-retryable, so both resolve to the same exception type. Phase 4
adds one further sibling under `execution_gateway`'s OWN exception
hierarchy (not this package's) -- see "Execution gateway integration
(Phase 4)" below.

## Execution gateway integration (Phase 4)

Phase 4 is the first real integration between Milestone 8
(`execution_gateway`) and Milestone 9 (`portfolio_risk`), realizing the
dependency direction documented since Phase 1 ("Package architecture and
dependency direction" above): `execution_gateway` now depends on
`portfolio_risk`; `portfolio_risk` still never imports `execution_gateway`
(verified in both directions by the delivery report's own grep).

### Primary guarantee

**No `ExecutionIntent` may reach `dispatcher.dispatch_command` without
first passing through a mandatory, fail-closed portfolio-risk gate.**
Implemented entirely in a NEW module, `execution_gateway.
portfolio_risk_gate`, and wired into `runner.py`'s `_run_intents_and_
events` (the ONLY code path that bridges a paper order into a NEW
`SubmitOrderCommand` and dispatches it) between intent construction and
`dispatch_command`. Verified structurally, not just by testing: exactly
two call sites of `dispatch_command` exist anywhere in `execution_gateway`
-- the gated one in `_run_intents_and_events`, and `authorize_cancel_or_
reduce_only_submit` (used ONLY for `CancelOrderCommand`/`ReplaceOrderCommand`
in this codebase, which do not correspond to a NEW execution intent at
all, and are therefore deliberately NOT gated -- see that function's own
docstring for the explicit scoping rationale).

### Flow

```
ExecutionIntent -> [authorize_portfolio_risk_dispatch] -> RiskAuthorization
                 -> [reserve_portfolio_risk_dispatch]   -> RESERVED
                 -> dispatcher.dispatch_command (runner.py's own call, unchanged)
                 -> [consume_portfolio_risk_dispatch]   -> CONSUMED  (ONLY on COMMAND_DISPATCH_SUCCEEDED)
```

`authorize_portfolio_risk_dispatch` builds a `RiskEvaluationRequest` from
the intent's own fields, records it, calls Phase 2's `evaluate_risk`
UNMODIFIED, records the resulting `RiskDecision`, and -- only if
`APPROVED` -- issues a `RiskAuthorization` via Phase 3's `issuance.
issue_risk_authorization` UNMODIFIED. A DENIED/HALTED decision (or any
identity/binding failure) raises `ExecutionPortfolioRiskAuthorizationError`
(a new `execution_gateway`-domain exception, mirroring `ExecutionHaltError`'s
identical role for a kill-switch refusal) -- `runner.py` durably records
this via the previously-unused `EXECUTION_INTENT_REJECTED` ledger entry
kind before the exception propagates uncaught (the SAME "no try/except
around this call" precedent `authorize_dispatch`'s own kill-switch
refusal already established). `reserve_portfolio_risk_dispatch`/
`consume_portfolio_risk_dispatch` are thin, fail-closed wrappers around
Phase 3's `lifecycle.reserve_authorization`/`consume_authorization`,
UNMODIFIED. **No portfolio-risk evaluator logic and no dispatch-
transaction logic was rewritten anywhere in this phase** -- Phase 4 is
integration, not a rewrite, exactly as scoped.

**Consumption is tied to `COMMAND_DISPATCH_SUCCEEDED` ONLY.** A capability
rejection (`COMMAND_REJECTED`), an ambiguous adapter exception
(`COMMAND_MARKED_UNKNOWN`), or a synchronous broker refusal
(`COMMAND_DISPATCH_REJECTED`) all leave the authorization RESERVED, never
consumed and never auto-invalidated -- `recover_portfolio_risk_dispatch_
gate` (below) is what later resolves that ambiguity, using BOTH
packages' own durable evidence, never a guess.

`consumption_identity` is always `intent.execution_intent_id` -- already
a deterministic, unique-per-intent, content-addressed id. Reusing it
(rather than minting a third id) makes an exact retry of the same
intent's dispatch attempt idempotent by construction (Phase 3's own
`validate_authorization_use` semantics), and gives recovery a direct,
unambiguous way to find the corresponding execution-gateway order for
any RESERVED-but-unresolved authorization.

### Semantic migration: `execution_bridge_authorization_id` / `portfolio_risk_authorization_id`

Resolves the collision documented since Phase 1 ("SEMANTIC COLLISION
DECISION" above). `ExecutionIntent.risk_authorization_id` (which actually
held Milestone 8's OWN bridge-eligibility proof,
`ExecutionAuthorization.execution_authorization_id`) is RENAMED to
`execution_bridge_authorization_id` -- same meaning, same value, same
sha256 validation, still participates in `execution_intent_id`'s
identity (unchanged behavior). A NEW field, `portfolio_risk_
authorization_id: str | None`, holds THIS milestone's own concept --
`RiskAuthorization.risk_authorization_id` -- always `None` at bridge-
construction time (the bridge has no portfolio-risk context) and bound
afterward via `bind_portfolio_risk_authorization` (a NEW helper,
`dataclasses.replace` under the hood).

**Deliberately excluded from `to_identity_payload()`**: binding an
authorization to an already-minted intent must never retroactively
change that intent's own `execution_intent_id` (which every downstream
id -- `client_order_id`, every command's own identity, `broker_order_id`,
every fill id -- already cascades from) -- a circular, self-contradictory
result otherwise, since a `RiskAuthorization` must bind to an ALREADY-
COMPUTED `execution_intent_id` (Phase 1's own required binding field),
but embedding the authorization id INTO that same intent's identity
would require the id before the intent exists. Resolved by excluding the
field from identity entirely; `bind_portfolio_risk_authorization`
structurally asserts `execution_intent_id` is unchanged after binding.

**`identity_version` bumped from 1 to 2** for every newly-constructed
intent -- exactly the purpose this pre-existing field was reserved for.
`ExecutionIntent.from_json_dict` reads `identity_version` to decide how
to parse: `1` (or absent) triggers `_migrate_execution_intent_payload_
v1_to_v2`, a deterministic, pure migration helper that maps the OLD
`risk_authorization_id` key onto `execution_bridge_authorization_id`
verbatim (never reinterpreted) and sets `portfolio_risk_authorization_id
= None` (pre-migration data predates this milestone's own authorization
concept entirely -- there is nothing to backfill, and claiming otherwise
would be exactly the "silently reinterpret historical data" this
migration must never do). The reconstructed object KEEPS `identity_
version=1` and its ORIGINAL `execution_intent_id` -- old data replays to
the exact same id it always did; only the in-memory field NAMES change,
never the persisted economic identity. `identity_version >= 2` payloads
are read directly, no migration. A payload missing BOTH `execution_
bridge_authorization_id` and `risk_authorization_id` is genuinely corrupt
(neither schema shape) and raises a plain `KeyError`, exactly like any
other missing required field on this class already does.

Blast radius of the rename: `paper_bridge.py` (5 sites, all updated),
`ExecutionLedgerEntryKind` (one new member, `PORTFOLIO_RISK_AUTHORIZATION_
BOUND`, an audit-trail convenience entry recorded by `runner.py` right
after a successful bind -- the `portfolio_risk` ledger's own `RISK_
AUTHORIZATION_ISSUED` entry remains the authoritative source, this is
never a second source of truth), and 3 pre-existing unit test files that
constructed `ExecutionIntent` directly with the old field name (all
updated, all still passing).

### `PortfolioRiskGatewayContext` -- deliberately NOT threaded through `ExecutionGatewaySpec`

`portfolio_id`, `portfolio_snapshot`, `price_snapshot`, `risk_spec`,
`portfolio_halted`, `consecutive_losses` live in a NEW, separate context
object (`portfolio_risk_gate.PortfolioRiskGatewayContext`), passed
alongside `RunnerEnvironment` -- deliberately NOT added as fields on
`ExecutionGatewaySpec`/`ExecutionGatewayConfigSchema`. Adding them there
would change `execution_gateway_spec_id`'s own identity computation for
EVERY existing spec, a far more invasive, identity-breaking change than
Phase 4's own "integration, not rewrite" scope calls for. `portfolio_
snapshot`/`price_snapshot`/`risk_spec` are FIXED for the lifetime of one
`run_execution_session` call -- Phase 2's `evaluate_risk` is stateless
and caller-supplied-everything by design (no portfolio-risk evaluator
rewrite in this phase), so this context does not evolve the snapshot as
fills accrue mid-session (see Known Limitations). `price_snapshot.
instrument_id` matching every intent's own `instrument_id` is safe to
assume because `ExecutionGatewaySpec` already scopes one whole execution
session to exactly one instrument (Milestone 8's own pre-existing
invariant, Check 8 in `paper_bridge.execution_intent_from_paper_order`).

`RunnerEnvironment` gains one new REQUIRED field, `portfolio_risk_context`
-- never optional, since the gate is mandatory. `replay.replay_execution_
session` gains a matching required parameter, threaded straight through
(mirrors how `paper_bridge_environment` is already handled). The CLI
(`ml_cli.py`) wires a MINIMAL, always-present DEFAULT context
automatically -- deliberately no new CLI flags or config-schema fields
(out of Phase 4's own scope: "no CLI expansion"). Every limit on the
default `PortfolioRiskPolicy` is `None` ("not configured", Phase 1's own
pre-existing convention -- NOT a bypass flag; no field anywhere in this
package exists whose purpose is "skip risk evaluation"). The gate still
runs for REAL, through the genuine `evaluate_risk` pipeline, on every
CLI-driven dispatch -- it simply always approves until an operator
supplies real policy configuration (a known, honestly-documented Known
Limitation, not a silently-glossed-over gap).

### Recovery

`recover_portfolio_risk_dispatch_gate` (new, `portfolio_risk_gate.py`)
runs at the SAME `ADAPTER_INITIALIZED` runner stage as `execution_
gateway`'s OWN `recover_unknown_orders`, immediately after it (order
matters: portfolio-risk recovery's own cross-reference reads execution-
gateway order state that `recover_unknown_orders` must have already
resolved as much as it can). It calls Phase 3's `recovery.recover_
portfolio_risk_session` UNMODIFIED, then for every authorization it
classifies `reserved_unresolved_blocked`, cross-references the
corresponding execution-gateway order (found via `consumption_identity`
== `execution_intent_id`, exact and unambiguous):

- Order confirmed `DISPATCHED`/`ACKNOWLEDGED`/`PARTIALLY_FILLED`/`FILLED`
  -> the dispatch DID succeed even though the `consume` call never
  durably completed (a crash between dispatch success and consume) ->
  **`consumed_now`** (the missed consume is durably completed).
- Order confirmed `CREATED`/`VALIDATED`/`REJECTED`/`CANCELLED`/`EXPIRED`
  (never went live, or was refused before ever reaching the broker) ->
  **`invalidated_now`** (released so the authorization does not stay
  blocked forever).
- Order still `UNKNOWN` even after execution_gateway's own recovery, or
  no matching `SubmitOrderCommand` was ever created at all (the crash
  window between reservation and the first dispatch-transaction ledger
  write) -> **`remains_blocked`** -- never blindly reused, never guessed.

This directly resolves the required "crash before dispatch"/"crash after
reservation"/"crash after dispatch" scenarios: since `reserve_portfolio_
risk_dispatch` is the last durable step before `dispatch_command`, "crash
before dispatch" and "crash after reservation" describe the SAME
recovery window (no command exists yet -> `remains_blocked`, safe,
inspectable, no double-use possible); "crash after dispatch" is the
`consumed_now` case above.

### Cross-milestone verification

`portfolio_risk_gate.verify_execution_portfolio_risk_integration`
combines each package's OWN independent verification (`execution_gateway.
verification.verify_execution_session`, `portfolio_risk.verification.
verify_portfolio_risk_session` with `record=False` -- exactly like
`replay.py`'s own comparison utility, so this function is itself side-
effect-free and repeatable -- NEITHER modified nor re-implemented here)
with cross-ledger checks NEITHER package alone can perform, since neither
has visibility into the other's own ledger: every accepted `ExecutionIntent`
has a matching `RiskAuthorization` bound to the correct `portfolio_id`
(`dispatched_intent_without_risk_authorization`/`authorization_binding_
mismatch` if not); every `PORTFOLIO_RISK_AUTHORIZATION_BOUND` audit entry
agrees with the portfolio-risk ledger's own `execution_intent_id` index
(`execution_ledger_authorization_binding_mismatch` if not); and an
authorization is `CONSUMED` if and only if its intent's command resolved
to `COMMAND_DISPATCH_SUCCEEDED` (`consumed_authorization_without_
successful_dispatch`/`successful_dispatch_without_consumed_authorization`
if not -- the "single economic execution" cross-check).

### Concurrency

Two real, confirmed defects were found via this phase's own adversarial
concurrency testing and fixed at root cause (see "Defects found and
fixed" below, #9 and #10). After both fixes: two threads authorizing
DIFFERENT intents concurrently both succeed (an internal bounded retry
loop, mirroring Phase 3's own `_MAX_APPEND_RACE_RETRIES` convention,
resolves a losing ledger-append race by recomputing fresh sequence
numbers); two threads reserving the SAME intent concurrently are
idempotently absorbed to exactly one ledger entry; `PortfolioRiskLockError`
(fail-fast lock contention, a documented, expected, RETRYABLE
infrastructure condition) always propagates UNCHANGED rather than being
misclassified as a business-level `ExecutionPortfolioRiskAuthorizationError`
denial.

A KNOWN, PRE-EXISTING, OUT-OF-SCOPE flake source remains: the shared
`historical.locking.DatasetLock` primitive both packages' locks wrap has
its own documented stale-lock-reclaim race (the exact same underlying
fragility already responsible for Phase 3's own "defect #8") that can,
extremely rarely (empirically roughly 1-in-200-to-300 attempts under a
synthetic, maximally-tight `threading.Barrier` stress pattern far
tighter than any realistic caller would ever produce), cause a genuine,
momentary loss of mutual exclusion. Confirmed via direct reproduction and
root-cause investigation to live entirely inside shared, pre-existing
locking infrastructure used by multiple milestones -- fixing it would
require rewriting that primitive's own lock-acquisition protocol,
explicitly outside Phase 4's own scope. See Known Limitations.

### Defects found and fixed during Phase 4's own development

Continuing the numbering from Phase 3's own "Defects found and fixed"
section (defects 1-8):

9. **`authorize_portfolio_risk_dispatch` could leak a raw, confusing
   `RiskAuthorizationReuseError` under a genuine concurrent race** --
   this function makes THREE separate appends to the SAME shared per-
   portfolio ledger (evaluation request, decision, authorization
   issuance). Two concurrent calls for two DIFFERENT intents could each
   pass `portfolio_risk_lock` for their own FIRST append, then race a
   LATER one, since the three-append sequence is not atomic as a whole
   -- unlike Phase 3's own single-append reserve/consume transactions
   (already race-safe). The losing call surfaced a bare, misleading
   ledger-sequence-conflict exception instead of succeeding. Fixed by
   wrapping the whole evaluate-and-record sequence in a bounded retry
   loop that recomputes fresh sequence numbers on a losing race -- safe
   because every step is a pure function of its inputs.
10. **`PortfolioRiskLockError` (transient lock contention) was
    misclassified as a business-level denial** -- `authorize_portfolio_
    risk_dispatch`/`reserve_portfolio_risk_dispatch`/`consume_portfolio_
    risk_dispatch` each had a broad `except PortfolioRiskError as exc:
    raise ExecutionPortfolioRiskAuthorizationError(...)` clause that also
    caught `PortfolioRiskLockError` (a `PortfolioRiskError` subclass),
    wrapping a purely transient, RETRYABLE infrastructure condition into
    an exception whose name implies "this dispatch was refused" -- a
    caller catching the wrapped type and treating it as final (never
    retry this intent) would be WRONG when the underlying cause was only
    lock contention. Found via this phase's own concurrency test
    development (a `_retry_on_lock` test helper stopped seeing the
    `PortfolioRiskLockError` it was specifically watching for). Fixed by
    adding an explicit `except PortfolioRiskLockError: raise` before each
    broader `except PortfolioRiskError` clause, letting lock errors
    propagate unchanged.

## Known limitations (Phase 1 through Phase 4 scope, not defects)

- **No CLI expansion, no real broker execution** -- Phase 4 wires the
  gate into every REAL dispatch path but adds no new CLI flags/commands
  and no live-trading code; the CLI's own default portfolio-risk context
  is minimal/unconfigured (see "PortfolioRiskGatewayContext" above).
- `execution_gateway.paper_bridge.ExecutionIntent`'s semantic collision
  (documented since Phase 1) is now RESOLVED by Phase 4's own field
  rename/split -- see "Semantic migration" above. This bullet is kept
  here, struck through in spirit, purely so a reader scanning historical
  limitations understands the collision no longer exists as of Phase 4.
- **Phase 4's `PortfolioRiskGatewayContext` does not evolve the
  portfolio/price snapshot mid-session** -- `portfolio_snapshot`/
  `price_snapshot`/`risk_spec` are fixed for one whole `run_execution_
  session` call. A session that bridges MULTIPLE orders evaluates every
  one against the SAME starting snapshot, never one updated by an
  earlier order's own fill within that same call. This mirrors Phase 2's
  own stateless `evaluate_risk` contract exactly (no portfolio-risk
  evaluator rewrite in this phase) -- a future phase wanting live,
  fill-aware re-evaluation would need to construct a NEW context between
  orders itself; nothing in Phase 4 does this automatically.
- **A known, pre-existing, out-of-scope shared-infrastructure flake
  source** -- `historical.locking.DatasetLock`'s own documented stale-
  lock-reclaim race can, extremely rarely under maximally-tight
  concurrent contention, cause a genuine, momentary loss of mutual
  exclusion. See "Concurrency" above for the full, honest write-up;
  confirmed not to be a Phase 4 defect, confirmed out of scope to fix.
- **`evaluate_risk` still does not construct a `RiskAuthorization`
  itself** -- `issuance.issue_risk_authorization` is a separate function a
  caller invokes with an `APPROVED` `RiskDecision`; Phase 3 does not wire
  `evaluate_risk` to call it automatically. This is a deliberate
  separation (evaluation and issuance are different economic events,
  potentially at different times), not an oversight.
- **Phase 3's ledger durably records `portfolio_halted`/`consecutive_
  losses` inputs (as part of the `RiskEvaluationRequest`/`RiskDecision`
  payloads it stores) but nothing derives `evaluate_risk`'s own
  `portfolio_halted`/`consecutive_losses` parameters FROM that recorded
  history** -- they remain explicit, caller-supplied inputs, unchanged
  from Phase 2. A later phase could add this derivation on top of the
  Phase 3 ledger; Phase 3 itself does not.
- **No session manifest / explicit stage machine** -- a deliberate
  architectural choice, not a gap; see "Why no session manifest" above
  for the full rationale.
- **Recovery does not resolve external (broker-side) execution
  ambiguity** -- there is no execution integration in this milestone, so
  recovery answers only what the risk ledger's own durable evidence says,
  never whether an order was actually placed. See "Recovery" above.
- **`verification.verify_portfolio_risk_session` does not re-run Phase
  2's evaluator against the original snapshots/policy** -- it verifies a
  recorded `RiskDecision`'s internal coherence and binding, never whether
  its 18 checks were the economically correct ones for the real
  portfolio state (no snapshot/policy artifact store exists to re-derive
  that from). See "Independent verification honesty classification"
  above for the full, explicit 3-tier breakdown.
- **`max_daily_realized_loss` measures the day's TOTAL (realized +
  unrealized) equity decline, not isolated realized pnl** -- `Portfolio
  Snapshot` carries no realized-pnl-at-day-start baseline to isolate the
  realized component. See "Exposure accounting equations" above for the
  full, explicit accounting-model rationale.
- **`checks.check_portfolio_halted` does not consult `policy.
  allow_reduce_only_during_halt`.** It reports the halt state as-is
  (HALT severity, unconditionally, whenever `portfolio_halted=True`); a
  later phase's dispatch gate is responsible for consulting the flag
  before actually refusing a reduce-only order during a halt. Phase 2's
  own `TestReduceOnlyDuringHalt` test documents this exact boundary.
  `PortfolioRiskPolicy.allow_reduce_only_during_halt` is therefore
  defined (Phase 1) and validated but not yet consumed by any check
  (Phase 2).
- **`RiskCheckSeverity.WARNING` is defined (Phase 1) but never produced**
  by any of Phase 2's 18 checks -- every failure mode is DENY or HALT.
  Reserved for a future phase that wants a non-blocking, soft signal.
- **No cost modeling anywhere in `valuation.py`'s post-trade
  projection** -- gross-price-only (fills at `ask`/`bid`, no commission/
  spread/slippage), the same documented simplification Milestones 7/8
  already carry.
- **`evaluate_risk`'s `portfolio_halted`/`consecutive_losses` are
  caller-supplied, not derived** -- Phase 2 has no durable history to
  derive either from; a later phase's persistence/ledger layer would
  supply them from real recorded events.

## Future phases (not implemented, not started)

- **`execution_gateway` dispatch-gate integration is DONE (Phase 4)** --
  removed from this list; see "Execution gateway integration (Phase 4)"
  above.
- Evolving `PortfolioRiskGatewayContext`'s portfolio/price snapshot
  mid-session as fills accrue, instead of one fixed snapshot per
  `run_execution_session` call -- see this phase's own Known Limitations.
- New CLI flags/config-schema fields letting an operator supply a REAL
  `PortfolioRiskPolicy` to the execution-gateway CLI, replacing today's
  minimal always-unconfigured default context. Explicitly out of Phase
  4's own scope ("no CLI expansion").
- Deriving `evaluate_risk`'s `portfolio_halted`/`consecutive_losses`
  parameters from the portfolio-risk ledger's own recorded history,
  instead of a caller-supplied parameter.
- Wiring `checks.check_portfolio_halted` (or a successor) to consult
  `policy.allow_reduce_only_during_halt` now that a real dispatch gate
  exists to act on the distinction.
- A real, end-to-end acceptance workflow chaining a genuine paper/
  execution session through real risk evaluation AND the full
  authorization lifecycle -- Phase 4 extends `tests/integration/
  test_execution_gateway_acceptance.py` with a working (always-approving)
  portfolio-risk context so the EXISTING acceptance workflow continues to
  pass, but does not add new acceptance scenarios exercising a
  DENYING/HALTING policy end-to-end through the real bridge.
- Real cost-component modeling (spread/slippage/commission) inside
  `valuation.py`'s post-trade projection, reusing this platform's
  existing `backtesting.costs`/`paper_trading.costs` formulas rather than
  the current gross-price-only fills.
- Re-running Phase 2's evaluator against durably-stored original
  snapshots/policy as part of independent verification, once an
  artifact store for those exists -- see "Independent verification
  honesty classification" above for the precise gap this would close.
- A root-cause fix to `historical.locking.DatasetLock`'s own stale-lock-
  reclaim race (shared infrastructure, out of scope for this milestone).
- Milestone 10 has not been started.
