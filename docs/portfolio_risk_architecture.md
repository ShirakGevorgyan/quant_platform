# Portfolio Risk and Capital Management Engine (Milestone 9) -- Architecture

## Status: Phase 1 (domain foundation) + Phase 2 (deterministic exposure calculation and pre-trade risk evaluation) delivered

This document covers everything Phase 1 AND Phase 2 actually deliver --
exceptions, config schemas, enums, content-addressed policy/spec
identity, portfolio/price snapshot models, risk-decision/risk-
authorization models (Phase 1), and the pure exposure/projection/policy-
check/sizing/evaluation layer that turns those models into a real
`RiskDecision` (Phase 2). It explicitly marks every later-phase concept
(durable ledger, crash recovery, CLI, execution-gateway enforcement,
acceptance workflow) as NOT YET IMPLEMENTED rather than silently omitting
it. Sections for those later phases are present as headings with a
one-line status note, so this document's own structure does not need to
be reshuffled as each phase lands -- only expanded.

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

## PHASE 2 EXPLICITLY DOES NOT YET IMPLEMENT

Stated plainly, so no later reader mistakes Phase 2's evaluator for a
deployed, enforcing risk gate:

- **No durable ledger.** There is still no append-only, hash-chained
  event store for this package's own objects (unlike `execution_gateway.
  persistence`). `RiskAuthorizationStatus`'s event-sourced legal-
  transition table (Phase 1) exists specifically so a later phase can add
  one safely, but no such store exists yet -- `evaluate_risk` never
  persists a `RiskDecision`/`RiskAuthorization` anywhere; it only returns
  one in memory.
- **No `RiskAuthorization` is constructed by Phase 2.** `evaluate_risk`
  returns a `RiskDecision` (and, when approved, a `PositionSizeProposal`/
  `CapitalAllocation`) -- it does NOT construct a `RiskAuthorization`
  (Phase 1's model for that). A later phase's persistence layer is
  expected to do that once a decision is durably recorded.
- **No crash recovery, no reconciliation, no independent verification
  pass.** The corresponding exception classes exist in `core.exceptions`
  (mirroring `execution_gateway`'s identical exception shape) so a later
  phase's functions have a home to raise into, but no such functions
  exist yet.
- **No CLI.** No `ml_cli.py` command exists for this package yet.
- **No execution-gateway enforcement.** `execution_gateway.paper_bridge.
  ExecutionIntent.risk_authorization_id` is NOT checked against a real
  `portfolio_risk.authorization.RiskAuthorization` anywhere yet, and
  `evaluate_risk` is not called from anywhere in `execution_gateway` --
  see "Relationship to `execution_gateway`" below and "SEMANTIC COLLISION
  DECISION" for the precise current state of that field and the required
  future migration.
- **No pre-existing halt state or losing-streak count is derived by this
  package.** `evaluate_risk`'s `portfolio_halted`/`consecutive_losses`
  parameters are explicit, mandatory, CALLER-supplied inputs -- Phase 2
  has no ledger to derive them from real history yet (see "Evaluator
  orchestration" below).
- **No acceptance workflow.** No end-to-end test chains a real paper
  session through this package's own risk evaluation; only isolated
  domain-model unit tests exist through Phase 2.

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

## Known limitations (Phase 1 + Phase 2 scope, not defects)

- No ledger, recovery, CLI, or execution-gateway enforcement exists yet
  -- see "PHASE 2 EXPLICITLY DOES NOT YET IMPLEMENT" above.
- `execution_gateway.paper_bridge.ExecutionIntent.risk_authorization_id`
  currently holds Milestone 8's own `execution_bridge_authorization_id`
  concept, not this milestone's `portfolio_risk_authorization_id` concept
  -- see "SEMANTIC COLLISION DECISION" above for the exact required
  future migration.
- `RiskAuthorizationStatus`'s event-sourced legal-transition table is
  defined but nothing constructs a transition event yet -- no durable
  authorization ledger exists yet, and `evaluate_risk` does not construct
  a `RiskAuthorization` at all (only a `RiskDecision`).
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

- A durable, append-only, hash-chained ledger for this package's own
  objects, mirroring `execution_gateway.persistence` exactly -- including
  persisting `RiskDecision`/`RiskAuthorization` and deriving
  `portfolio_halted`/`consecutive_losses` from real recorded history
  instead of a caller-supplied parameter.
- `RiskAuthorization` construction (from an `APPROVED` `RiskDecision`) and
  single-use/idempotent-use enforcement, replaying `RiskAuthorizationStatus`
  transition events against the legal-transition table Phase 1 already
  defines.
- `execution_gateway` dispatch-gate integration: refusing to dispatch any
  `ExecutionIntent` without a valid, matching, unconsumed
  `RiskAuthorization` -- and performing the `ExecutionIntent` field
  migration described in "SEMANTIC COLLISION DECISION" above.
- Wiring `checks.check_portfolio_halted` (or a successor) to consult
  `policy.allow_reduce_only_during_halt` once a real dispatch gate exists
  to act on the distinction.
- Crash recovery, reconciliation, and independent verification passes,
  mirroring `execution_gateway`'s identical three-part safety net.
- A CLI surface on the shared `ml_cli.py` parser.
- A real, end-to-end acceptance workflow chaining a genuine paper/
  execution session through real risk evaluation.
- Real cost-component modeling (spread/slippage/commission) inside
  `valuation.py`'s post-trade projection, reusing this platform's
  existing `backtesting.costs`/`paper_trading.costs` formulas rather than
  the current gross-price-only fills.
