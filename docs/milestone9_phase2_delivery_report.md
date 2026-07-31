# Milestone 9 Phase 2 Delivery Report — Deterministic Exposure Calculation and Pre-Trade Risk Evaluation Engine

## 1. Phase 1 commit hash used as baseline

`15384d51f38f6cf9cf209a8657afc384cac5979f` — "Add portfolio risk domain
foundation". HEAD has not moved since (no commit occurred during Phase 2
work). All Phase 2 work below is uncommitted, unstaged working-tree
changes on top of this commit.

## 2. Files added and modified

**Modified (1 file, purely additive):**
- `src/quant_platform/portfolio_risk/snapshots.py` — added
  `verify_portfolio_snapshot_identity`/`verify_price_snapshot_identity`
  (+33 lines; no existing field, function, or behavior changed). These
  are Phase-1-shaped, recompute-and-compare identity checks
  (`verify_portfolio_risk_spec_identity`'s own established pattern,
  applied to the two snapshot types) needed by Phase 2's evaluator to
  detect a forged/tampered snapshot before trusting it. Justified as a
  small, additive, non-breaking, well-precedented extension of an
  already-committed file rather than a new abstraction; documented here
  per the governing instruction to record any touch of already-committed
  code.

**Added — 5 new modules in `src/quant_platform/portfolio_risk/`:**
- `exposure.py` — derived exposure/account metrics beyond Phase 1's own
  `ExposureSnapshot` (long/short gross split, concentration, leverage,
  available cash, daily/total loss).
- `valuation.py` — deterministic post-trade projection
  (`project_position`/`project_portfolio`/`classify_trade_risk`).
- `checks.py` — the 18 required policy checks + canonical `CHECK_ORDER`.
- `sizing.py` — policy-limit-based position-sizing helpers.
- `evaluator.py` — `evaluate_risk`/`EvaluationOutcome`, the orchestration
  layer tying everything above into a final `RiskDecision`.

**Added — 5 new test files in `tests/unit/portfolio_risk/`:**
- `test_portfolio_risk_exposure.py` (22 tests)
- `test_portfolio_risk_valuation.py` (25 tests)
- `test_portfolio_risk_checks.py` (29 tests)
- `test_portfolio_risk_sizing.py` (30 tests)
- `test_portfolio_risk_evaluator.py` (34 tests)

Every new test file basename is prefixed `test_portfolio_risk_*`,
confirmed globally unique across the whole `tests/` tree (same discipline
established in Phase 1 after Milestone 8's own basename-collision
defect).

**Modified — documentation:**
- `docs/portfolio_risk_architecture.md` — updated from a Phase 1 skeleton
  to cover Phase 2 in full: exposure accounting equations, post-trade
  projection rules, the 18 policy checks (with a severity-rationale
  table), APPROVED/DENIED/HALTED aggregation rules, position-sizing
  rules, evaluator orchestration, a new explicit "SEMANTIC COLLISION
  DECISION" section (Section 2's own requirement), and an updated Known
  Limitations/Future Phases section.

**Added:**
- `docs/milestone9_phase2_delivery_report.md` — this report.

**Total: 2 modified (374 insertions, 53 deletions — almost entirely the
architecture doc rewrite plus the small snapshots.py addition), 10 added
files, 140 new tests.**

## 3. Exposure accounting equations

Every figure is DERIVED from `PortfolioSnapshot`'s own already-validated
fields (`exposure.py`) — nothing is independently stored or trusted:

- `long_gross_exposure = Σ abs(market_value)` over positions with
  `signed_quantity > 0`
- `short_gross_exposure = Σ abs(market_value)` over positions with
  `signed_quantity < 0`
- `portfolio_gross_exposure = long_gross_exposure + short_gross_exposure`
  (via Phase 1's `compute_portfolio_exposure`)
- `portfolio_net_exposure = Σ market_value` (signed; Phase 1)
- `concentration_fraction = max(per-instrument gross exposure, aggregated
  across strategies) / portfolio_gross_exposure`, `0` when flat
- `leverage = portfolio_gross_exposure / equity` — **raises
  `ExposureCalculationError` when `equity <= 0`** (undefined, never a
  sentinel); the leverage CHECK converts this into an automatic
  HALT-severity failure regardless of configured limit
- `available_cash = cash - minimum_cash_buffer` (never clamped —
  can go negative to make a breach visible, not hidden)
- `daily_loss = max(0, daily_start_equity - equity)` — the day's TOTAL
  (realized + unrealized) equity decline
- `total_loss = max(0, -(realized_pnl + unrealized_pnl))` — total pnl
  loss since inception

**Documented accounting-model decision**: `PortfolioSnapshot` (Phase 1)
carries `realized_pnl` (cumulative since inception) and
`daily_start_equity` (a day-scoped equity baseline) but no separate
"realized pnl at start of day" baseline. `max_daily_realized_loss` is
therefore checked against `daily_loss` (the day's TOTAL equity decline,
not isolated to the realized component) — an explicit, documented Known
Limitation (Section 11), never a silent approximation.
`max_total_loss` is checked against the genuinely inception-scoped
`total_loss`, mirroring `paper_trading.risk.evaluate_continuous_risk`'s
own `-(realized_pnl + unrealized_pnl)` formula exactly.

## 4. Post-trade projection rules

`valuation.project_position`/`project_portfolio` mirror `paper_trading.
accounting.apply_fill_to_position`'s weighted-average-cost /
realize-on-reduce logic, Decimal-native, adapted to `PositionSnapshot`'s
lighter shape. GROSS-PRICE-ONLY (explicitly documented, not silently
invented): a BUY fills at `price.ask`, a SELL at `price.bid` (mirroring
`execution_gateway.dummy_broker`'s own MARKET-order convention); **no
commission, spread, or slippage cost is modeled**. `cash` moves by
exactly `quantity * fill_price * contract_multiplier`.

One function handles every required scenario via a single, uniform
signed-quantity delta: opening a new long/short position, adding,
reducing, fully closing, and crossing through zero in either direction.
`classify_trade_risk` distinguishes INCREASING (opens from flat, adds
same-direction, OR crosses through zero — a direction change is always
increasing, even if the resulting magnitude is smaller), REDUCING
(strictly decreases magnitude, same direction, including full close), and
NEUTRAL (degenerate, unreachable given `quantity > 0` is already a Phase 1
invariant, but never assumed unreachable in code). The projection never
mutates the original `PortfolioSnapshot` — both it and `PositionSnapshot`
are frozen dataclasses, so a new, independently `__post_init__`-validated
object is always returned.

## 5. Exact policy checks implemented

All 18 required checks, in the fixed, canonical `CHECK_ORDER`:

| # | `check_identity` | Severity when breached |
|---|---|---|
| 1 | `order_notional_limit` | DENY |
| 2 | `position_notional_limit` | DENY |
| 3 | `instrument_gross_exposure_limit` | DENY |
| 4 | `strategy_gross_exposure_limit` | DENY |
| 5 | `portfolio_gross_exposure_limit` | DENY |
| 6 | `portfolio_net_exposure_limit` | DENY |
| 7 | `concentration_fraction_limit` | DENY |
| 8 | `leverage_limit` | DENY (HALT if leverage undefined) |
| 9 | `minimum_cash_buffer` | DENY |
| 10 | `daily_realized_loss_limit` | HALT |
| 11 | `total_loss_limit` | HALT |
| 12 | `drawdown_limit` | HALT |
| 13 | `consecutive_losses_limit` | HALT |
| 14 | `stale_price` | DENY |
| 15 | `stale_portfolio_snapshot` | DENY |
| 16 | `portfolio_halted` | HALT |
| 17 | `reduce_only_validity` | DENY |
| 18 | `missing_or_inconsistent_valuation_data` | DENY |

Checks 1–9 are evaluated against the POST-TRADE PROJECTED state, so a
risk-reducing trade naturally improves its own measured values and is
never blocked without special-casing. Checks 10–13 reflect account-wide
health, mirroring `paper_trading.risk`'s own loss/drawdown → portfolio-
wide-action mapping. Checks 17/18 reuse Phase 1's
`RiskDenialReason.INCOHERENT_EVALUATION_STATE` rather than requiring two
new enum members — no Phase 1 enum modification was needed; every one of
the other 16 checks maps 1:1 onto an existing Phase 1 `RiskDenialReason`
member. Every check ALWAYS produces exactly one `RiskCheckResult`,
including passing ones — confirmed by
`test_portfolio_risk_checks.py::TestAllChecksAlwaysProduceAResult`.

## 6. APPROVED / DENIED / HALTED aggregation rules

`evaluator._aggregate_decision_kind`:

1. **Completeness verified first, unconditionally**: `check_results`
   must contain EXACTLY the 18 identities `CHECK_ORDER` requires —
   raises `RiskEvaluationError` otherwise. `APPROVED` is structurally
   impossible unless every required check ran.
2. `most_severe_check_severity` (Phase 1) folds every severity via `max`
   — HALT anywhere → `HALTED`; else DENY anywhere → `DENIED`; else →
   `APPROVED`.
3. `denial_reasons` is the SORTED set of every failing check's reason
   (deterministic, never raw set-iteration order).

`evaluate_risk` fails closed BEFORE aggregation too:
`_verify_request_bindings` recomputes every referenced object's own
identity and raises `RiskEvaluationError` — never returns a decision —
on a forged/tampered snapshot, cross-instrument price, cross-portfolio,
or cross-policy mismatch. No `except Exception`/bare `except` exists
anywhere in `evaluator.py`.

## 7. Position-sizing rules

Policy-limit-based only (no volatility/Kelly/ATR/model-based sizing, per
this phase's own scope boundary). Each `max_quantity_by_*` function
returns the additional quantity ONE limit still permits (`None` when
unconstrained); `compute_maximum_allowed_quantity` combines every
applicable constraint with the caller's `requested_quantity` via `min`
(result can never exceed what was requested), then quantizes DOWN
(conservative) to a caller-supplied, mandatory `quantity_step` (a
non-positive step is rejected). `max_quantity_by_cash_buffer` only
constrains a BUY — a SELL always increases cash under this phase's
gross-price-only convention.

## 8. Determinism and identity rules

`RiskDecision.risk_decision_id` is stable for identical economic inputs
because: (a) `evaluate_risk` always builds `check_results` in the fixed
`CHECK_ORDER`, never dict/set iteration; (b) every timestamp is
caller-supplied (`evaluation_time`), never read from the wall clock; (c)
`PortfolioSnapshot.to_identity_payload()` (Phase 1) sorts `positions` by
`(instrument_id, strategy_id)`, so construction order of the INPUT
snapshot cannot change the projected portfolio's own id either. Verified
directly: identical inputs produce byte-identical serialized JSON and
identical ids; shuffled position construction order produces the SAME
portfolio snapshot id and therefore the same decision id; two completely
independently-constructed but economically-identical scenarios produce
the same decision id; a real wall-clock delay between two calls with the
same `evaluation_time` does not change the id; the id is stable across
three separate subprocesses with `PYTHONHASHSEED` set to `0`/`1`/`42`;
changing `quantity` (an economic field) changes the id.

## 9. Tests added and exact results

**140 new Phase 2 tests** (across `test_portfolio_risk_exposure.py`,
`test_portfolio_risk_valuation.py`, `test_portfolio_risk_checks.py`,
`test_portfolio_risk_sizing.py`, `test_portfolio_risk_evaluator.py`),
plus **45 additional safety-scan test instances** (Phase 1's
`test_portfolio_risk_safety.py` fixture auto-discovers every `*.py` file
in the package via `glob` — the 5 new Phase 2 modules were automatically
scanned with zero test-file changes, and all passed: no forbidden
imports, no `float(...)` call, no wall-clock read, no credential-shaped
identifier, anywhere in the 5 new modules).

**Total package suite: 498 tests** (313 from Phase 1 + 140 new Phase 2
tests + 45 safety-scan growth), all passing.

| Gate | Result |
|---|---|
| `git diff --check` | Clean; exit 0 |
| `ruff check .` (full repository) | All checks passed; exit 0 |
| `mypy src` (full repository) | Success: no issues found in 263 source files; exit 0 |
| `pytest tests/unit/portfolio_risk/ -W error -q` (focused Phase 1 + Phase 2) | 498 passed in 5.18s, 0 warnings, exit 0 |
| Targeted regression for the one modified file (`snapshots.py`) | Covered by the same 498-test run above — includes all of Phase 1's own `test_portfolio_risk_snapshots.py` (66 tests), unaffected by the purely-additive change |

Per the governing instruction, the multi-hour full-repository suite was
NOT run this phase: no cross-package regression appeared, collection did
not fail, and no shared infrastructure was modified beyond the
purely-additive `snapshots.py` change (no `core.exceptions` addition was
even needed — every Phase 2 failure mode maps onto one of Phase 1's
already-defined 17 exception classes).

No test count above was invented — every number is copied directly from
an actual command's own output.

## 10. Genuine defects found and fixed

None found in already-committed code (Phase 1 or Milestone 8) during
Phase 2 work. All test failures encountered during development were
Phase-2-test-authoring bugs (e.g. an unrealized-pnl mismatch in a
hand-constructed test fixture, an unrealistic realized/unrealized pnl
combination against a flat position), fixed in the test files themselves
before being counted in the totals above — never a loosened assertion.

## 11. Known limitations

See `docs/portfolio_risk_architecture.md`'s own "Known limitations"
section for full detail. Summary:

- No ledger, recovery, CLI, or execution-gateway enforcement exists yet.
- `max_daily_realized_loss` measures the day's TOTAL equity decline, not
  isolated realized pnl — `PortfolioSnapshot` carries no realized-pnl-at-
  day-start baseline (Section 3 above).
- `checks.check_portfolio_halted` does not consult `policy.
  allow_reduce_only_during_halt` — it reports the halt state
  unconditionally; a later phase's dispatch gate must consult the flag
  before actually blocking a reduce-only order during a halt.
- `RiskCheckSeverity.WARNING` (Phase 1) is never produced by any of the
  18 Phase 2 checks — every failure mode is DENY or HALT.
- No cost modeling in `valuation.py`'s projection (gross-price-only).
- `evaluate_risk`'s `portfolio_halted`/`consecutive_losses` are
  caller-supplied, not derived from any durable history (none exists
  yet).
- The `execution_bridge_authorization_id` /
  `portfolio_risk_authorization_id` semantic collision (Section 2's own
  required documentation, detailed in the architecture doc's "SEMANTIC
  COLLISION DECISION" section) remains unresolved by design — the
  cross-milestone rename was explicitly deferred, not performed.

## 12. Exact git status

```
 M docs/portfolio_risk_architecture.md
 M src/quant_platform/portfolio_risk/snapshots.py
?? docs/milestone9_phase2_delivery_report.md
?? src/quant_platform/portfolio_risk/checks.py
?? src/quant_platform/portfolio_risk/evaluator.py
?? src/quant_platform/portfolio_risk/exposure.py
?? src/quant_platform/portfolio_risk/sizing.py
?? src/quant_platform/portfolio_risk/valuation.py
?? tests/unit/portfolio_risk/test_portfolio_risk_checks.py
?? tests/unit/portfolio_risk/test_portfolio_risk_evaluator.py
?? tests/unit/portfolio_risk/test_portfolio_risk_exposure.py
?? tests/unit/portfolio_risk/test_portfolio_risk_sizing.py
?? tests/unit/portfolio_risk/test_portfolio_risk_valuation.py
```

`git diff --stat`: 2 files changed, 374 insertions(+), 53 deletions(-)
(the two tracked-file modifications only — every other path above is a
new, untracked file, which `git diff` without `--stat` on untracked paths
does not include).

`git diff --cached --stat`: empty.

`git log -1`: `15384d51f38f6cf9cf209a8657afc384cac5979f` — unchanged from
Section 1.

## 13. Confirmation

- **Phase 2 work is not staged.** `git diff --cached` is empty.
- **Phase 2 work is not committed.** HEAD remains
  `15384d51f38f6cf9cf209a8657afc384cac5979f`, the Phase 1 commit.
- **Nothing was pushed.** No `git push` was run at any point.
- **`execution_gateway` was not integrated or modified.** Zero `import`/
  `from` statements in any `portfolio_risk` file reference
  `execution_gateway` (confirmed by direct grep, excluding prose/
  docstring mentions); zero `import`/`from` statements in any
  `execution_gateway` file reference `portfolio_risk`. No file under
  `src/quant_platform/execution_gateway/` appears in `git status` or
  `git diff` at any point in this phase's work.
- **Milestone 10 was not started.** No file, module, or reference to
  Milestone 10 exists anywhere in this repository.
- This milestone remains TEST-ONLY, side-effect-free, and fail-closed:
  no MT5/FxPro/real-broker/network/credential code, no float arithmetic
  for a financial value, no wall-clock-dependent economic decision, no
  approval-by-default, and no bypass flag exists anywhere in
  `quant_platform.portfolio_risk` — each proven structurally by the
  safety-scan suite (now covering all 12 source modules, 130 test
  instances) automatically extended to cover Phase 2's 5 new modules
  with zero test-file changes required.

**STOP after Phase 2** — per the governing instructions, no ledger,
persistence, recovery, CLI, `RiskAuthorization` consumption, or
execution-gateway integration was implemented in this phase.
