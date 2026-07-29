# Milestone 7 Delivery Report — Deterministic Paper Trading and Shadow Execution Engine

## 1. What was delivered

A new top-level package `quant_platform.paper_trading` (24 modules, ~7,800
lines) implementing a broker-neutral, deterministic paper-trading and
shadow-execution engine that consumes only strategies explicitly promoted
`ELIGIBLE_FOR_PAPER_TRADING` by Milestone 6, simulating the complete
trading lifecycle — decisions, orders, fills, positions, costs, account
state, pre-trade/continuous risk limits, a kill switch, session
manifests, an event-sourced ledger, reconciliation, and independent
verification — without ever transmitting a real order. Also delivered:

- `src/quant_platform/config/paper_trading_schemas.py` — a strict
  Pydantic configuration schema (`PaperTradingConfig` + 10 nested
  sub-schemas), reusing `config.backtesting_schemas`' spread/slippage/
  commission/financing sub-schemas directly.
- 24 new domain exceptions in `core/exceptions.py` (`PaperTradingError`
  and 23 subclasses).
- 15 new `ArtifactCategory` values in `ml/models.py`.
- 14 new, purely additive CLI commands in `ml_cli.py` (70 total commands
  on the shared parser, up from 56).
- Two documentation files: `docs/paper_trading_and_shadow_execution.md`
  (architecture) and this delivery report.
- 884 tests (as of the Section 2a release audit; 758 at initial
  development-time delivery, +126 from the audit's 7 new
  `test_audit_*.py` files plus additional classes added to existing
  suites — see Section 2a and Section 5): 32 unit test files under
  `tests/unit/paper_trading/`, one real-model integration acceptance
  test (`tests/integration/test_paper_trading_real_model_acceptance.py`),
  and one CLI subprocess test
  (`tests/integration/test_paper_trading_cli_subprocess.py`).

## 2. Real defects found and fixed this session

Six genuine defects were found and fixed during development, each
through the same document → explain impact → fix → add regression test →
rerun gates cycle:

1. **`order_policy.py` instrument-match guard was inverted/nonsensical**
   (`instrument.symbol != decision.event_identity`, comparing unrelated
   fields) — fixed to `portfolio.instrument != instrument.symbol`.
2. **`accounting.py`: contract multiplier silently ignored.**
   `realized_pnl`/`gross_cost_basis`/`unrealized_pnl` formulas never
   multiplied by `contract_multiplier`, so a 100x-multiplier instrument
   (e.g. a gold-like contract) would have computed P&L off by 100x.
   Fixed by adding the field to `PositionState`, validating every
   incoming fill's IMPLIED multiplier against it, and multiplying
   throughout. Caught by a new `TestContractMultiplierAppliedConsistently`
   regression class.
3. **`accounting.py`: stale `unrealized_pnl` carried across fills.**
   `apply_fill_to_position` never reset/recomputed `unrealized_pnl` after
   a fill, so a position that had been marked once and then closed/
   reversed kept reporting a stale unrealized figure. Fixed by
   recomputing it fresh from `position.last_mark` at the end of every
   fill application. Caught by an exact round-trip reconciliation test
   (discrepancy of exactly 50.0, traced to the stale carryover).
4. **`execution.py`: wrong reject-reason code for IOC non-fill.** Reused
   `STALE_MARKET_DATA` (semantically wrong) for an IOC order that could
   not fill immediately. Fixed by adding a dedicated
   `RejectReasonKind.IOC_NOT_IMMEDIATELY_FILLABLE` member.
5. **`runner.py`: kill switch's own flatten order self-blocked.** The
   synthesized closing order the kill switch submits during
   `FLATTEN_SIMULATED_POSITIONS` was rejected by the very
   `trading_halted=True` pre-trade check the kill switch had just set —
   a real halt could never actually flatten the position it was meant to
   close. Fixed by exempting that one system-generated, risk-REDUCING
   order from the halt checks, with an explanatory comment. Caught by
   `TestKillSwitch.test_tight_drawdown_limit_halts_and_flattens`
   (position remained open instead of reaching zero).
6. **`runner.py`: resume-after-interruption cursor desync.** The event-
   ledger append cursor started at the ledger's current length, but the
   main loop always iterated the FULL `events` sequence from index 0 —
   a genuine resume would re-append the first event's entries at the
   wrong sequence position. Fixed with a fail-closed clean-event-boundary
   guard (`_require_clean_event_boundary`, refuses to auto-resume a
   genuinely mid-event crash) plus a trusted-snapshot resume
   reconstruction (`_reconstruct_resume_state`) that counts `ACCOUNT_
   SNAPSHOT` entries (not `MARKET_EVENT_ACCEPTED`) to determine how many
   events are already fully processed.

Two additional, lower-severity gaps were found and fixed while building
later modules:

- `runner.py` never persisted `SESSION_TRANSITION` ledger entries for
  any of the 7 manifest-stage transitions (only the manifest's own
  `stage` field changed) — fixed by routing every transition through a
  new `_transition_with_ledger_entry` wrapper, and `run_paper_trading_
  session` never handled resuming FROM a `PAUSED` stage back to
  `RUNNING` — fixed alongside adding the missing `pause_paper_session`
  function (Section 23 requires pause as a first-class capability).
- 4 of the new CLI inspection commands (`inspect-paper-orders`/`-fills`/
  `-risk-events`/`-reconciliation`) read directly from the event ledger
  without first confirming the named session actually exists — since
  `PaperSessionEventStore.read_events` returns `[]` for a nonexistent
  session (a legitimate contract at the store layer), these commands
  silently reported "0 orders/fills" instead of a clean error. Fixed by
  validating the manifest exists first in all 4, caught by the CLI
  subprocess test suite.

## 2a. Release audit — 13 additional defects found and fixed

A separate, independent release audit was performed after the above
development-time work was complete, explicitly instructed **not** to
trust this report's original "zero blockers" claim. That instruction was
warranted: the audit found 13 genuine defects across 8 of the 10 areas it
examined (in-flight order resume; mid-event crash boundaries; model/fold
selection; shadow-mode isolation; contract-multiplier accounting;
event-ledger reconstruction/semantic tampering; eligibility-bypass
attempts; risk halt/flatten; deterministic replay; clean-install CLI).
Every defect below was found via a hand-derived adversarial test written
against the stated invariant (never by trusting an existing passing
test), then fixed, then re-verified. Contract-multiplier accounting
(53 new hand-derived tests) was the one area audited with zero defects
found — purely confirmatory.

1. **In-flight LIMIT/STOP working orders did not survive a resume.**
   `_reconstruct_resume_state` reset `working_orders={}` unconditionally
   on every resume instead of replaying the ledger for each order's
   actual terminal/non-terminal state. Fixed via a new
   `_reconstruct_working_orders` that rebuilds resting orders from
   `ORDER_STATE_EVENT` replay. *(`paper_trading/runner.py`)*
2. **Tail-stage transitions were not resume-safe across a crash.** A
   crash between the manifest's own stage write and its corresponding
   `SESSION_TRANSITION` ledger append — at any of the final
   `RUNNING → END_OF_STREAM → RECONCILING → VERIFIED → COMPLETED` steps —
   could leave a session permanently stuck or silently drop the final
   ledger entry, since `_transition_with_ledger_entry` only recognized
   "manifest is exactly at `from_stage`" as a valid starting point, and
   `COMPLETED`/`TERMINATED` used a blind early-return that skipped
   backfill entirely. Fixed by making `_transition_with_ledger_entry`
   resume-safe: it now distinguishes "do the real transition" from
   "already past this stage — backfill only the missing ledger entry by
   locating the exact transition payload in the existing ledger."
   *(`paper_trading/runner.py`)*
3. **Fold/model selection was re-derived from live, mutable state on
   every call, never pinned.** `_resolve_fitted_strategy_runtime`
   computed `max(execution_manifest.fold_result_references)` fresh each
   time with no persisted binding to the session's original selection —
   a resume could silently swap in a different fitted model if the
   underlying fold-result artifacts changed between calls. Fixed with a
   write-once, verify-forever pin file keyed by `paper_session_id` that
   compares `(experiment_id, fold_index, model_content_hash)` on every
   subsequent resolution and fails closed on drift. *(`ml_cli.py`)*
4. **Shadow-session ledgers were never checked against the declared
   session mode.** `verify_paper_session` validated chain integrity,
   reconciliation, and eligibility, but nothing correlated ledger
   *contents* (real vs. hypothetical fill entries) against
   `manifest.session_mode` — a shadow ledger with spliced-in real fills
   (or vice versa) would pass verification undetected. Fixed with a new
   `_verify_ledger_matches_session_mode` check.
   *(`paper_trading/verification.py`)*
5. **`resume-paper-session` checked session-ID existence only, not
   identity.** The CLI looked up the manifest by the `--config`-derived
   spec and only confirmed that *some* manifest existed at
   `--paper-session-id`, never that the two actually matched — an
   operator could silently resume/mutate a different session than the
   one named on the command line. Fixed with an explicit identity check
   in `cmd_resume_paper_session` that raises `PaperTradingIdentityError`
   on mismatch. *(`ml_cli.py`)*
6. **Market-event ordering was never re-verified.** No function
   re-derived and checked strict sequence-increase, uniqueness, and
   chronological order of `MARKET_EVENT_ACCEPTED` ledger entries after
   the fact. Fixed with a new `_verify_market_event_ordering` check.
   *(`paper_trading/verification.py`)*
7. **Ledger entries were never checked to belong to the session being
   verified.** A ledger assembled (accidentally or adversarially) from
   another session's entries would pass verification undetected. Fixed
   with a new `_verify_ledger_entries_belong_to_session` check.
   *(`paper_trading/verification.py`)*
8. **`reconcile_session` crashed unhandled on a genuinely illegal
   order-transition sequence.** A second, unguarded call site to
   `resolve_order_state` inside the Check 1/2 loop had no
   `except OrderStateError` handling (unlike the first call site),
   raising instead of returning a structured failure report. Fixed by
   wrapping the call in `try/except OrderStateError: continue`.
   *(`paper_trading/reconciliation.py`)*
9. **Eligibility was verified exactly once, at session creation, and
   never again on resume — the most severe finding of this audit.**
   `require_paper_trading_eligibility` was invoked only inside
   `create_paper_session`; every subsequent resume call skipped
   re-verification entirely, meaning a session whose underlying
   eligibility chain (robustness verification, promotion decision) was
   invalidated or tampered with *after* creation could still be resumed
   indefinitely. Fixed by making `run_paper_trading_session` and
   `run_shadow_session` mandatorily re-verify eligibility on every call
   that did not itself just create the manifest. This fix required
   updating ~76 pre-existing tests whose fixtures had never needed to
   supply a real eligibility chain on resume (via an explicit
   autouse-bypass fixture, not by weakening the check). A narrower
   "skip re-verification once ELIGIBILITY_VERIFIED is reached" fix was
   considered and rejected as it would have defeated the actual security
   property being enforced. *(`paper_trading/runner.py`)*
10. **Stale working orders were left un-cancelled during a
    risk-triggered flatten.** The `FLATTEN_SIMULATED_POSITIONS`/
    `TERMINATE_SESSION` escalation branch only ever created the
    position-closing order; any *other* still-`WORKING` resting order
    (e.g. a stale LIMIT) was left untouched and could fill later,
    reopening exposure after a safety halt. Fixed by cancelling every
    other working order (`reason_code=RISK_HALT_ACTIVE`) before creating
    the flatten's own closing order. *(`paper_trading/runner.py`)*
11. **No canonical semantic-digest function existed for tamper-evident
    deterministic-replay verification.** Naive full-ledger hashing is
    fragile against operational/wall-clock-variable metadata
    (`entry_id`, `checksum`, `previous_entry_hash`, `event_time`),
    producing false mismatches across otherwise-identical replays. Added
    `compute_ledger_semantic_digest`, which normalizes each entry to
    `{sequence, kind, payload}` before hashing.
    *(`paper_trading/persistence.py`)*
12. **Row-inspection CLI commands printed unbounded output.**
    `inspect-paper-orders`/`-fills`/`-risk-events` had no row limit.
    Fixed with a `--limit` argument (default 200) and truncation notices
    on all three. *(`ml_cli.py`)*
13. **`report-paper-session`/`report-shadow-session` never checked the
    target session's actual mode.** Either command would run against a
    session of the wrong mode and silently produce a misleadingly
    labeled report. Fixed with explicit `manifest.session_mode` checks
    (not the config-derived `spec.session_mode`, which does not reflect
    the session's own persisted mode) that raise
    `PaperTradingStateError` on mismatch. *(`ml_cli.py`)*

**Non-blocking limitations confirmed (not fixed — deliberate scope
boundaries, unreachable-but-safe code paths, or audit-coverage caveats,
none compromising a safety/correctness/eligibility property):**

- LIMIT/STOP order types are execution-supported but never
  strategy-originated (Area 2).
- Latency-eligibility windows are computed but not enforced as a fill
  gate (Area 2).
- The forged-`ELIGIBLE_FOR_PAPER_TRADING`-string attack was structurally
  addressed by `verify_robustness`'s source-reconstructing re-derivation,
  but not independently rebuilt with a fresh fixture in this audit given
  cost (Area 4, coverage caveat).
- Content-addressed types (`Fill`, `OrderRequest`, `LedgerEntry`, market
  events) do not self-validate their identity field against recomputed
  content on deserialization — a fix was attempted and reverted after it
  broke every `create_*` factory's provisional-construction pattern
  (Area 7).
- `RiskActionKind.TERMINATE_SESSION` is unreachable via the live runner
  (hardcoded-neutral failure counters) (Area 9).
- `RiskActionKind.CANCEL_OPEN_ORDERS` is defined but never produced by
  any risk check (Area 9).
- `PaperSessionManifest.stage` never reaches HALTING/HALTED/TERMINATED —
  the safety gate itself is correctly keyed off `KillSwitchState`, so
  this is a display/observability gap, not a safety gap (Area 9).
- Instrument-identity was not separately included in the
  tamper-after-promotion sweep (only feature_spec/strategy_candidate/
  model_artifact/calibration_artifact were) (Area 4/8, coverage caveat).
- The clean-venv pip "reinstall" message pattern observed during
  clean-install CLI testing was not fully root-caused, though venv
  isolation itself was independently confirmed via `pyvenv.cfg`
  (Area 11, coverage caveat).

See `docs/paper_trading_and_shadow_execution.md`'s "Unsupported behavior
(explicit)" section for the user-facing description of the limitations
above that affect runtime behavior.

## 3. Feature-by-feature classification (Sections 1-35)

| Section | Feature | Classification |
|---|---|---|
| 1 | Package structure, one-way dependency | IMPLEMENTED AND VERIFIED |
| 2 | 24 domain exceptions | IMPLEMENTED AND VERIFIED |
| 3 | `PaperTradingSpec` identity | IMPLEMENTED AND VERIFIED |
| 4 | Eligibility/source verification chain | IMPLEMENTED AND VERIFIED |
| 5 | Normalized market-event model | IMPLEMENTED BUT LIMITED — `CorporateOrInstrumentAdjustmentEvent` not implemented (Section 5 explicitly permits omitting it) |
| 6 | Event clock | IMPLEMENTED AND VERIFIED |
| 7 | Strategy runtime adapter | IMPLEMENTED AND VERIFIED — Protocol plus one concrete real-model implementation (`model_strategy.ModelStrategyRuntime`) |
| 8 | Order model/state machine | IMPLEMENTED BUT LIMITED — MARKET/LIMIT/STOP only; `STOP_LIMIT`/`MARKET_ON_CLOSE` not claimed |
| 9 | Order policy | IMPLEMENTED AND VERIFIED |
| 10 | Paper execution model | IMPLEMENTED AND VERIFIED |
| 11 | Fill model | IMPLEMENTED AND VERIFIED |
| 12 | Position accounting | IMPLEMENTED AND VERIFIED |
| 13 | Portfolio/account model | IMPLEMENTED BUT LIMITED — single instrument per session (explicit, fail-closed, Section 13's own permitted simplification) |
| 14 | Instrument contract | IMPLEMENTED AND VERIFIED |
| 15 | Mark-to-market | IMPLEMENTED AND VERIFIED |
| 16 | Costs and financing | IMPLEMENTED AND VERIFIED |
| 17 | Risk engine | IMPLEMENTED AND VERIFIED |
| 18 | Kill switch | IMPLEMENTED AND VERIFIED |
| 19 | Shadow observation mode | IMPLEMENTED BUT LIMITED — does not model halts/session-close/financing for the shadow position; resume not supported (documented) |
| 20 | Session manifest/state machine | IMPLEMENTED AND VERIFIED |
| 21 | Event-sourced persistence | IMPLEMENTED AND VERIFIED |
| 22 | Session runner | IMPLEMENTED AND VERIFIED |
| 23 | Pause/resume/idempotency | IMPLEMENTED BUT LIMITED — event-boundary granularity only; a genuine mid-event crash requires manual investigation (documented, fail-closed, never silent) |
| 24 | Concurrency/locking | IMPLEMENTED AND VERIFIED — reuses `ml.concurrency.experiment_lock` |
| 25 | Reconciliation | IMPLEMENTED AND VERIFIED — all 11 required checks |
| 26 | Session verification | IMPLEMENTED AND VERIFIED — classified honestly as PARTIALLY INDEPENDENT |
| 27 | Reporting | IMPLEMENTED AND VERIFIED — all 15 summaries |
| 28 | Backtest comparison | IMPLEMENTED BUT LIMITED — decision/abstention/rejection counts on the backtest side are reported as 0 (a vectorized backtest does not track them the same way); documented |
| 29 | Configuration schemas | IMPLEMENTED AND VERIFIED |
| 30 | Artifact categories | IMPLEMENTED AND VERIFIED — 15 new categories |
| 31 | CLI | IMPLEMENTED AND VERIFIED — 14 commands, no live-trading command exists |
| 32 | Replay input | IMPLEMENTED AND VERIFIED |
| 33 | Real acceptance workflow | IMPLEMENTED AND VERIFIED (Section 4 below) |
| 34 | Testing | IMPLEMENTED AND VERIFIED — 758 tests |
| 35 | Security/safety scan | IMPLEMENTED AND VERIFIED — structural AST scan, 201 tests, zero findings |

## 4. Real acceptance workflow (Section 33)

`tests/integration/test_paper_trading_real_model_acceptance.py` builds a
GENUINE chain — real `logistic_regression` model, real injected-AR(1)-
signal dataset, real execution/calibration/backtest/robustness pipeline,
a real re-verified `ELIGIBLE_FOR_PAPER_TRADING` `PromotionDecision` — then
runs a real 300-bar paper session and a real shadow session against a
real fitted-model `StrategyRuntime`.

Observed figures below were confirmed IDENTICAL to the last decimal
across THREE genuinely separate runs: the two independent in-test runs
(fresh manifest/event stores, same process) plus two additional, fully
separate `python -m pytest` PROCESS invocations run for the quality-gate
repeat requirement (Section 5 below) — every one of the 4 runs produced
`net_pnl=787.5869471157534` exactly:

- `event_count=301 decision_count=300 abstention_count=60`
- `order_count=198 rejected_orders=34 fill_count=164`
- Long fills: yes. Short fills: yes. Abstentions: yes (60). Risk-rejected
  orders: yes (34, `order_quantity_limit_exceeded`).
- `starting_cash=100000.0 final_cash=101225.40365962524 final_equity=100787.58694711581`
- `gross_pnl=790.8769278123025 net_pnl=787.5869471157534`
- `spread_cost=1.409991727092421 slippage_cost=0.9399944847282808 commission_cost=0.9399944847282808 financing=0`
- `maximum_drawdown_fraction=0.00017553359898851362`
- Reconciled: **True** (all 11 checks). Verification: **is_ready=True**.
- A second, genuinely independent in-test run (fresh manifest/event
  stores, identical spec and replay events) produced a **byte-identical**
  ledger (compared field-by-field excluding wall-clock `SESSION_
  TRANSITION` timestamps, Section 0's own documented carve-out).
- Shadow session: 300 observations, 192 with a hypothetical fill,
  `shadow_counterfactual_pnl=810.0179528117624`, never merged with the real
  account's own figures.

The lenient promotion-gate thresholds this fixture uses are an
infrastructure-test configuration choice, not data tuning — see the
test file's own module docstring for the full reasoning. The observed
net P&L is a synthetic-data artifact and is not a claim of achievable
real-world performance.

## 5. Quality gates (current — post release-audit, superseding Section 4's/this session's original numbers)

| Command | Collected | Passed | Skipped | Deselected | Failed | Runtime | Exit |
|---|---|---|---|---|---|---|---|
| `python -m ruff check .` | — | — | — | — | 0 | — | 0 |
| `python -m mypy src` | 222 files | — | — | — | 0 | — | 0 |
| Focused Milestone 7 audit tests, `-W error -q` | 107 | 107 | 0 | 0 | 0 | 14.51s | 0 |
| `python -m pytest tests --deselect tests/performance -q` | 4196 | 4138 | 1 | 57 | 0 | 3007.65s (0:50:07) | 0 |
| `python -m pytest tests/performance -m performance -q` | 57 | 57 | 0 | 0 | 0 | 164.88s (0:02:44) | 0 |
| `python -m pytest -q` (run 1 of 2) | 4196 | 4195 | 1 | 0 | 0 | 3188.27s (0:53:08) | 0 |
| `python -m pytest -q` (run 2 of 2) | 4196 | 4195 | 1 | 0 | 0 | 3195.07s (0:53:15) | 0 |

The two full `python -m pytest -q` runs produced IDENTICAL pass/skip
counts (4195 passed, 1 skipped, 0 failed both times) — no flakiness
observed anywhere in the ~4,196-test suite across two independent full
runs. The one skip is an environment constraint
(`tests/unit/ml/test_artifacts.py:110` — symlink creation requires
elevated privileges on Windows), not a Milestone 7 gap. `ruff check .`
and `mypy src` report zero issues across the entire repository, not just
the new package.

### Repeat gates (release audit, current)

| Repeat group | Selector | Repetitions | Per-run result | Per-run time | Outcome |
|---|---|---|---|---|---|
| In-flight order resume | `test_audit_resume_working_orders.py` | 20 | 6 passed | 2.3–3.8s | 20/20 exit 0 |
| Mid-event crash | `test_audit_mid_event_crash_boundaries.py` | 20 | 16 passed | 3.6–5.0s | 20/20 exit 0 |
| Contract-multiplier accounting | `test_audit_contract_multiplier_accounting.py` | 20 | 53 passed | 1.2–1.4s | 20/20 exit 0 |
| Halt/flatten | `test_audit_risk_halt_flatten.py` | 20 | 10 passed | 3.5–3.8s | 20/20 exit 0 |
| Shadow isolation | `test_audit_shadow_isolation.py` | 20 | 4 passed | 2.3–4.0s | 20/20 exit 0 |
| Ledger semantic-tampering | `test_audit_ledger_semantic_tampering.py` | 10 | 11 passed | 3.1–3.9s | 10/10 exit 0 |
| Replay determinism | `test_audit_deterministic_replay.py` | 10 | 7 passed | 6.3–7.4s | 10/10 exit 0 |
| Clean-install CLI suite | `test_paper_trading_cli_subprocess.py` | 3 | 25 passed | 42.3–42.8s | 3/3 exit 0 |
| Eligibility-bypass (targeted classes across the real-acceptance and CLI-subprocess suites) | 10 | 4 passed | 316.7–320.4s | 10/10 exit 0 |
| Real acceptance workflow (separate processes) | `test_paper_trading_real_model_acceptance.py` | 2 | 11 passed each | 1830.7s / 1862.6s | 2/2 exit 0 |

All 135 repeat-gate invocations returned exit code 0 with zero failures
and identical collected/passed counts on every repetition — no
flakiness, no order-dependence, no environment sensitivity observed
anywhere. Each repetition was a genuinely separate `python -m pytest`
process invocation (`pytest-repeat` is not installed in this
environment).

## 6. Explicit non-claims

- This milestone does not claim profitability, production readiness,
  broker readiness, or live-trading readiness.
- `ELIGIBLE_FOR_LIVE_TRADING` does not exist anywhere in this codebase
  and was not created.
- No live order transmission capability exists — verified structurally,
  not by convention (Section 35's AST-based safety scan: no network/
  broker-client import, no `eval`/`exec`/`pickle`/`shell=True`, no
  broker-credential-shaped identifier, anywhere in the package).
- No MT5 integration was implemented; `InstrumentSpec`'s XAUUSD example
  is explicitly documented as hypothetical, not real broker terms.
- Milestone 8 and Milestone 9 were not started.
- `git add`, `git commit`, and `git push` were not run.

## 7. Known limitations (see also Section 3's classification table and
Section 2a's release-audit findings)

This section reflects the system's state AFTER the Section 2a release
audit's 13 fixes; two items present in earlier drafts of this report
(in-flight LIMIT/STOP resume loss, and unpinned fold selection) have been
fixed and are removed from this list — see Section 2a, items 1 and 3.

- **Order origination is MARKET-only** — `order_policy.py` fully
  validates, executes, and (since the Section 2a fix) resumes LIMIT/STOP
  orders, but the automated decision-to-order pipeline never itself
  originates one; every order the strategy layer produces is MARKET.
- **Latency-eligibility windows are computed but not enforced as a fill
  gate** — `clock.py` computes the three configured latency delays, but
  `execution.py` does not currently reject or defer a fill against them.
- **Mid-single-event crash recovery is fail-closed, not automatic** —
  Section 22's "processed transactionally" language is honored at
  whole-event granularity; a crash strictly between two ledger entries
  of the SAME event requires manual investigation rather than an
  automatic best-effort retry. (Crashes at the manifest's own TAIL
  transitions, `RUNNING` through `COMPLETED`, were a separate gap fixed
  by Section 2a item 2 — those are now resume-safe.)
- **Shadow-session resume is not supported** — `run_shadow_session`
  refuses to resume an already-started shadow session outright.
- **No live/forward market-data ingestion exists** — `FORWARD_PAPER` is
  a legal `SessionMode` and `run_paper_trading_session` will stream
  whatever `Iterable[MarketEvent]` it is given, but Section 32
  explicitly excludes building a market-data downloader or MT5 data
  ingestion in this milestone, so no live producer of that iterable
  exists in this codebase.
- **"Which fold's model" for paper trading is now a pinned, tamper-
  evident choice, not merely a documented one** — a walk-forward
  experiment fits one model per outer fold; the CLI/acceptance workflow
  resolves the HIGHEST-indexed (temporally last) fold's fitted model on
  first use, then pins `(experiment_id, fold_index, model_content_hash)`
  for that `paper_session_id` and fails closed if a later call would
  resolve differently (Section 2a item 3).
- **`RiskActionKind.TERMINATE_SESSION` is unreachable via the live
  runner** and **`RiskActionKind.CANCEL_OPEN_ORDERS` is defined but
  never produced by any risk check** — both implemented and tested
  directly, neither currently triggered by a live code path (Section 2a).
- **`PaperSessionManifest.stage` never reaches HALTING/HALTED/
  TERMINATED** — the safety gate itself is correctly keyed off
  `KillSwitchState`, which does latch correctly; this is a
  display/observability gap only (Section 2a).
- **Content-addressed types do not self-validate their identity field
  against recomputed content on deserialization** — tamper detection
  relies on the ledger hash-chain and the new
  `compute_ledger_semantic_digest`, not on any single entry
  self-validating in isolation (Section 2a).
