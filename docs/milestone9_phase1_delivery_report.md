# Milestone 9 Phase 1 Delivery Report — Portfolio Risk and Capital Management Engine (Domain Foundation)

## 1. Baseline HEAD

Approved baseline commit for this phase's work: `9bd5cef73ee42074c9d53a1c308a88bbef1f7120`
("Add broker-neutral deterministic execution gateway" — Milestone 8).

**Note on this baseline:** at the start of this session, Milestone 8's own
work was still uncommitted (HEAD was at the Milestone 7 commit,
`a908aae3a4e872525648e4a8245a19959b4fe76d`). This was flagged to the user
before any Milestone 9 work began; per explicit user instruction, Milestone
8 was committed first (as the single commit above, containing exactly the
files already reviewed and gate-confirmed in that milestone's own delivery
report — nothing new was added or changed in that commit beyond what
Milestone 8 had already produced). HEAD has not moved since. No commit,
amend, stage, or push has occurred at any point during this Milestone 9
Phase 1 work itself — every file below remains an uncommitted, unstaged
working-tree change.

## 2. Files added and modified

**Modified (1 file):**
- `src/quant_platform/core/exceptions.py` — appended `PortfolioRiskError`
  base class plus 17 subclasses (policy validation, spec identity,
  snapshot validation, stale snapshot, stale price, exposure calculation,
  position sizing, risk evaluation, risk denial, authorization identity,
  authorization mismatch, authorization reuse, portfolio halt,
  reconciliation, verification, persistence, recovery). Pure addition —
  no existing class was modified or removed. +136 lines.

**Added — new package `src/quant_platform/portfolio_risk/` (8 files):**
- `__init__.py` — package docstring: full safety scope, Phase 1 boundary,
  dependency direction.
- `identity.py` — `compute_content_id` (re-exported from
  `paper_trading.identity`), `decimal_to_json`/`decimal_from_float`/
  `parse_decimal`.
- `models.py` — `OrderSide` (re-exported), `RiskDecisionKind`,
  `RiskDenialReason`, `RiskCheckSeverity` (+ `most_severe_check_severity`),
  `RiskAuthorizationStatus` (+ legal-transition table and helpers).
- `specs.py` — `PortfolioRiskPolicy`, `PortfolioRiskSpec`,
  `compute_portfolio_risk_spec_id`, `verify_portfolio_risk_spec_identity`.
- `snapshots.py` — `PriceSnapshot`, `PositionSnapshot`, `PortfolioSnapshot`,
  `ExposureSnapshot`, `compute_*_exposure`, `is_price_stale`,
  `is_portfolio_snapshot_stale`.
- `decisions.py` — `RiskCheckResult`, `RiskEvaluationRequest`,
  `RiskDecision` (+ `create_*` factories).
- `allocation.py` — `CapitalAllocation`, `PositionSizeProposal` (+
  `create_*` factories).
- `authorization.py` — `RiskAuthorization`, `create_risk_authorization`,
  `verify_risk_authorization_binding`.

**Added — config schema (1 file):**
- `src/quant_platform/config/portfolio_risk_schemas.py` —
  `PortfolioRiskPolicyConfigSchema`, `PortfolioRiskConfigSchema` (Pydantic,
  frozen, `extra="forbid"`).

**Added — tests, `tests/unit/portfolio_risk/` (9 files, 313 tests):**
- `test_portfolio_risk_identity.py` (12 tests)
- `test_portfolio_risk_models.py` (9 tests)
- `test_portfolio_risk_specs.py` (46 tests)
- `test_portfolio_risk_snapshots.py` (66 tests)
- `test_portfolio_risk_decisions.py` (35 tests)
- `test_portfolio_risk_allocation.py` (18 tests)
- `test_portfolio_risk_authorization.py` (29 tests)
- `test_portfolio_risk_config_schemas.py` (13 tests)
- `test_portfolio_risk_safety.py` (85 tests — AST-based safety scan +
  non-vacuity proof)

Every test file basename is prefixed `test_portfolio_risk_*`, deliberately
chosen to be globally unique across the entire `tests/` tree from the
start (no `__init__.py` exists anywhere under `tests/`, so pytest's
default import mode requires globally unique basenames — this was a real,
confirmed defect discovered and fixed the hard way during Milestone 8;
Phase 1 applies that lesson proactively rather than repeating it).
Confirmed via a scripted whole-tree scan: zero basename collisions.

**Added — documentation (2 files):**
- `docs/portfolio_risk_architecture.md` — architecture skeleton with every
  required safety statement, explicit "PHASE 1 EXPLICITLY DOES NOT YET
  IMPLEMENT" section, dependency-direction rationale, and a documented
  genuine finding about `execution_gateway.paper_bridge.ExecutionIntent.
  risk_authorization_id` (Section 5 below).
- `docs/milestone9_phase1_delivery_report.md` — this report.

**Total: 1 modified, 20 added, 1683 lines of new package source, 1702
lines of new test source.**

## 3. Dependency direction chosen and why

`portfolio_risk` depends downward on `paper_trading` (content-identity
infrastructure reuse only — `compute_content_id`, `OrderSide`, the same
low-level primitives `execution_gateway` itself already reuses rather than
duplicating) and on `core`/`ml` (exceptions, JSON/timestamp serialization,
fingerprinting). It does **not** depend on `execution_gateway`.

The intended future consumption direction is the reverse: a later phase's
`execution_gateway` dispatch gate will depend on `portfolio_risk` to check
for a valid `RiskAuthorization` before calling the adapter — mirroring how
`execution_gateway` already depends on `paper_trading`, never the other
way around. This keeps the whole dependency graph one-way and acyclic:

```
paper_trading  --->  execution_gateway
paper_trading  --->  portfolio_risk
                      portfolio_risk  --->  execution_gateway   (future phase)
```

Cross-package binding uses plain, sha256-validated id strings rather than
direct object references — `RiskEvaluationRequest`/`RiskAuthorization`
never import an `execution_gateway` type, only its content-addressed id as
a `str` field. This is what makes the eventual `execution_gateway ->
portfolio_risk` dependency safe to add later without ever creating a
cycle: `portfolio_risk` never needs to import anything back.

This decision was reached by first inspecting `execution_gateway.
paper_bridge.ExecutionIntent`, which surfaced a genuine, load-bearing
finding — see Section 5.

## 4. Domain models and invariants

See `docs/portfolio_risk_architecture.md` for the full, detailed
description of every model. Summary:

- **`PortfolioRiskPolicy`** — all 16 required limit fields, each
  `Decimal | None` (or `int | None` for the two age bounds and the
  consecutive-losses count); `None` = not configured, mirroring
  `paper_trading.specs.RiskLimitsSpec`'s identical convention.
  `allow_reduce_only_during_halt` is the one mandatory field.
- **`PortfolioRiskSpec`** — thin, content-addressed, deliberately
  portfolio-agnostic wrapper around one `PortfolioRiskPolicy`.
- **`PriceSnapshot`** — `bid <= ask`, both positive; `reference_price`
  always explicitly supplied, never auto-derived.
- **`PositionSnapshot`** — quantity always a positive magnitude with
  direction in `side`; `unrealized_pnl` must exactly reconcile with
  `signed_quantity * (mark_price - average_entry_price) *
  contract_multiplier`.
- **`PortfolioSnapshot`** — `equity` must exactly reconcile with
  `cash + sum(position market values)`; portfolio-level `unrealized_pnl`
  must equal the sum of every open position's own; `realized_pnl` is
  independently trusted (not derivable from currently-open positions
  alone, since closed positions' realized pnl isn't retained in this
  point-in-time view); duplicate `(instrument_id, strategy_id)` positions
  rejected; `peak_equity >= equity` enforced; `drawdown_fraction` is a
  derived property, never a stored field.
- **`ExposureSnapshot`** — always derived via `compute_portfolio_exposure`/
  `compute_instrument_exposure`/`compute_strategy_exposure`, never stored
  on `PortfolioSnapshot`; `gross_exposure >= 0`;
  `abs(net_exposure) <= gross_exposure`.
- **`RiskCheckResult`/`RiskEvaluationRequest`/`RiskDecision`** —
  `RiskDecision.kind=APPROVED` structurally forbids denial reasons and
  DENY/HALT-severity checks; `DENIED`/`HALTED` require at least one
  matching denial reason; `HALTED` additionally requires a HALT-severity
  check. No evaluator constructs one from real state yet — these
  invariants only constrain what a well-formed `RiskDecision` may look
  like.
- **`CapitalAllocation`/`PositionSizeProposal`** — small, standalone,
  content-addressed value objects; `utilized_capital` may never exceed
  `allocated_capital`.
- **`RiskAuthorization`** — binds every required field
  (`execution_intent_id`, `execution_session_id`, `portfolio_id`,
  `portfolio_snapshot_id`, `price_snapshot_id`, `risk_policy_id`,
  `risk_decision_id`/`decision_kind`, `evaluated_quantity`,
  `evaluated_price`, `authorization_sequence`, caller-supplied
  `event_time`) into its own content-addressed identity, so it can never
  be valid for a different intent/session/portfolio/snapshot/policy/
  decision/quantity/price by construction.

## 5. A genuine finding from this phase's own inspection

`execution_gateway.paper_bridge.ExecutionIntent` already has a
`risk_authorization_id: str` field (sha256-validated), added in Milestone
8. Today it is populated with `authorization.execution_authorization_id`
— Milestone 8's own internal `ExecutionAuthorization` ("may this paper
order be bridged into the execution gateway at all"), a different,
earlier-stage concept from this milestone's `RiskAuthorization` ("may this
specific quantity/price be dispatched against this specific portfolio
state").

**This is not a defect** — Milestone 8 could not have populated this field
with a real portfolio-risk authorization id, because this package did not
exist yet. It is a genuine integration point a later phase must resolve
deliberately (either `ExecutionIntent` gains a second, distinct field, or
`risk_authorization_id` is repurposed and Milestone 8's own
`execution_authorization_id` concept is relocated). Per this phase's own
explicit scope boundary ("do not modify execution-gateway dispatch
behavior yet"), `execution_gateway` was not touched. This is recorded in
`docs/portfolio_risk_architecture.md`'s own "Relationship to
`execution_gateway`" section for whichever future phase implements
enforcement.

## 6. Identity rules

Every content-addressed object follows the established placeholder-then-
recompute pattern (`create_*` factory builds with `"0" * 64`, computes the
real id via `compute_content_id`, reconstructs). `to_identity_payload()`
excludes only the object's own id field (and, for `PortfolioRiskSpec`,
`schema_version`) — every other field, including caller-supplied
timestamps, participates in identity, since those timestamps are never
sourced from an internal wall-clock read. `PortfolioSnapshot.
to_identity_payload()` sorts `positions` by `(instrument_id, strategy_id)`
before hashing — the one genuinely unordered collection this phase
defines. No `uuid4`, random value, wall-clock time, memory address, temp
path, or `PYTHONHASHSEED`-dependent ordering participates in any identity
computation anywhere in this package (proven both by direct source
inspection and by `test_portfolio_risk_specs.py::
TestPortfolioRiskSpecIdentity::test_identity_stable_under_pythonhashseed`,
a genuine cross-process test with three different `PYTHONHASHSEED` values).

## 7. Tests added and exact results

**313 tests**, `tests/unit/portfolio_risk/`, all passing:

| Gate | Result |
|---|---|
| `git diff --check` | Clean; exit 0 |
| `ruff check .` (full repository) | All checks passed; exit 0 |
| `mypy src` (full repository) | Success: no issues found in 258 source files; exit 0 |
| `pytest tests/unit/portfolio_risk/ -W error -q` (focused Phase 1 tests) | 313 passed in 2.68s, 0 warnings, exit 0 |
| `pytest tests/unit -q` (full fast-unit regression check) | 4726 passed, 1 skipped, 1 failed in 223.74s — see below |

The one full-unit-regression failure
(`tests/unit/execution/test_execution_property_based.py::
test_resume_plan_is_a_pure_function_of_its_inputs`) is a pre-existing
Hypothesis `DeadlineExceeded`/`FlakyFailure` (250.12ms vs. a 200ms
deadline on one run, 56.90ms on the immediate retry) in a Milestone 5
backtesting-execution property test this phase never touched. Re-run in
isolation immediately afterward: `1 passed in 28.07s`. This is system-
timing flakiness under whole-suite load, not a regression introduced by
this phase's changes — confirmed by the fact that the only file this phase
modified (`core/exceptions.py`) is a pure, additive append with zero
changes to any existing class, and every other file this phase touches is
new and outside `tests/unit/execution/`'s own import graph. A full-suite
regression run was not strictly required by this phase's own gate list
("git diff --check", "ruff check on changed files", "mypy on changed
source files", "focused Phase 1 unit tests with warnings as errors"); it
was run anyway as additional diligence given `core/exceptions.py` is
shared infrastructure.

No test count above was invented — every number is copied directly from
an actual command's own output.

## 8. Any genuine defects found in existing code

None found in already-committed code during this phase. (Section 5's
finding is an incomplete integration point Milestone 8 could not have
completed — since this package did not exist yet — not a defect in
Milestone 8's own code as written.)

## 9. Known limitations

See `docs/portfolio_risk_architecture.md`'s own "Known limitations"
section. Summary: no evaluator, ledger, recovery, CLI, or
execution-gateway enforcement exists yet (explicitly out of Phase 1's own
scope); `ExecutionIntent.risk_authorization_id`'s current meaning overlap
with Milestone 8 (Section 5); `RiskAuthorizationStatus`'s transition table
is defined but nothing constructs a transition event yet, since no durable
authorization ledger exists in this phase.

## 10. Exact git status

```
 M src/quant_platform/core/exceptions.py
?? docs/portfolio_risk_architecture.md
?? src/quant_platform/config/portfolio_risk_schemas.py
?? src/quant_platform/portfolio_risk/
?? tests/unit/portfolio_risk/
```

`git diff --stat`: 1 file changed, 136 insertions(+) (the exceptions.py
append only — everything else is untracked/new).

`git diff --cached --stat`: empty.

`git log -1`: `9bd5cef73ee42074c9d53a1c308a88bbef1f7120` — "Add
broker-neutral deterministic execution gateway" (unchanged from Section 1).

## 11. Explicit confirmation

- **Nothing staged.** `git diff --cached` is empty.
- **Nothing committed** in this Milestone 9 Phase 1 work. (Milestone 8
  itself was committed at the very start of this session, at the user's
  own explicit instruction, as the corrective action to the baseline
  discrepancy noted in Section 1 — not as part of, and prior to, any
  Milestone 9 work.)
- **Nothing pushed.** No `git push` was run at any point.
- **Milestone 10 has not been started.** No file, module, or reference to
  Milestone 10 exists anywhere in this repository.
- This milestone remains TEST-ONLY: no MT5 integration, no FxPro
  integration, no real broker adapter, no network connectivity, no
  credentials or credential-shaped configuration, no float arithmetic for
  a financial value, and no wall-clock-dependent economic decision exists
  anywhere in `quant_platform.portfolio_risk` — each proven structurally
  by `tests/unit/portfolio_risk/test_portfolio_risk_safety.py`'s AST-based
  scan (85 tests, including a non-vacuity proof that every detector
  actually fires against a deliberately bad snippet). No bypass flag
  exists anywhere in this package that disables a risk check — none was
  added, and `PortfolioRiskPolicy`/`PortfolioRiskConfigSchema` have no
  such field to even construct.

**STOP after Phase 1** — per the governing instructions, no evaluator,
ledger, recovery, CLI command, execution-gateway enforcement, or
acceptance workflow was implemented in this phase.
