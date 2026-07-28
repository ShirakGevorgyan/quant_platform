# Milestone 6 Delivery Report — Leakage-Safe Statistical Robustness, Strategy Selection, and Promotion-Gate Framework

This report distinguishes **implemented and verified** (structural, and
proven by a passing test), **implemented but limited** (works, with a
documented scope boundary), **simplified** (a deliberate, documented
approximation), **deferred** (a real, explicitly out-of-scope gap), and
**unsupported and fail-closed** (deliberately refuses rather than
fabricates). It does **not** claim production readiness, commit
readiness, or profitability — see the explicit non-claims section.

## 1. What was delivered

- New top-level package `src/quant_platform/robustness/` — 19 modules
  (18 `.py` files + `__init__.py`), 5,678 lines: `models`, `specs`,
  `source`, `series`, `bootstrap`, `multiple_testing`, `stability`,
  `resimulation`, `sensitivity`, `stress`, `regimes`, `selection`,
  `promotion`, `manifests`, `resume`, `runner`, `verification`,
  `reporting`.
- `src/quant_platform/config/robustness_schemas.py` — the CLI's pydantic
  config schema (`RobustnessConfig`, 101 lines).
- 10 new CLI commands wired into the existing `ml_cli.py`
  (`create-robustness-spec`, `run-robustness`, `resume-robustness`,
  `inspect-robustness`, `report-robustness`, `verify-robustness`,
  `compare-robustness`, `inspect-promotion-decision`, `inspect-strategy-
  family`, `compare-strategy-candidates`) — additive only (+369 lines, 0
  deletions to that file).
- 15 new domain exceptions in `core/exceptions.py` (+113 lines, 0
  deletions) and 16 new `ArtifactCategory` values in `ml/models.py` (+62
  lines, 0 deletions) — every modification to a pre-existing, shared
  APPLICATION file this milestone touched was **purely additive**
  (`git diff --stat` confirms 0 deletions on `exceptions.py`,
  `ml/models.py`, and `ml_cli.py`). The ONE exception is a pre-existing
  TEST file, `tests/unit/test_ml_cli.py` (+10/-3 lines) — its own
  exact-command-set assertion needed the 10 new command names added; see
  "Real defects and pre-existing issues found and fixed" below.
- `docs/statistical_robustness_and_promotion.md` (400 lines) and this
  report.
- Test suite: 12 new test files, 2,039 lines, **145 new tests** (131
  unit across 10 files in `tests/unit/robustness/`, 13 CLI-subprocess
  integration tests, 1 real-model acceptance workflow test) — all
  passing, plus 1 pre-existing test file updated (7 net lines).

## 2. Real defects and pre-existing issues found and fixed this session

1. **`bootstrap._stationary_resample` degenerate-rotation bug** (a real
   functional defect in this milestone's own new code, found via a
   direct CI-width comparison against `moving_block`, not by reasoning
   alone): an earlier version randomized only the FIRST block's start
   position and then let it drift forward contiguously across every
   subsequent block, degenerating the "stationary bootstrap" into a pure
   circular rotation of the original series. Order-independent
   statistics like total return were then numerically IDENTICAL across
   every "resample," collapsing the reported CI to a near-zero width
   that looked plausible but was silently wrong. Fixed by re-randomizing
   both the start position and the geometric block length for EVERY
   block. Regression-covered by
   `tests/unit/robustness/test_bootstrap.py::
   TestBlockBootstrapBoundaryBehavior::
   test_stationary_bootstrap_is_not_degenerate_unlike_the_regression_this_session_fixed`.
2. **Test-basename collision** (a pytest infrastructure issue, not an
   application defect): `tests/unit/robustness/test_stability.py`
   collided with the pre-existing `tests/unit/optimization/
   test_stability.py` — this repository's test tree has no `__init__.py`
   files, so pytest resolves module identity by basename alone across
   the whole tree. Discovered during the first full-suite quality-gate
   run (a collection error, not a test failure). Fixed by renaming to
   `test_fold_stability.py`.
3. **Pre-existing exact-command-set test needed updating** (found during
   the full-suite quality-gate run, not a defect in prior milestones'
   work — the same situation Milestone 5's own delivery report already
   documents for its own 8 commands): `tests/unit/test_ml_cli.py::
   TestBuildParser`'s assertion enumerates every registered CLI command
   name exactly; it did not yet know about this milestone's 10 new
   robustness commands. Updated (+10/-3 lines) to include them and
   renamed from `test_all_forty_six_commands_registered` to
   `test_all_fifty_six_commands_registered`.

No other functional defects were found in this milestone's own new
code **during development** (i.e., as of this report's original
writing). A subsequent, independent release audit of this same
milestone found and fixed 6 further real defects — see "Release-audit
addendum" (section 7 below) for the complete, itemized list; this
sentence is retained for the historical record of what was known at
delivery time, not as a current claim.

Every modification to a pre-existing APPLICATION file remains purely
additive; the two items above are a test-tree naming collision and a
pre-existing test's own assertion needing extension, not
application-behavior regressions.

## 3. Feature-by-feature classification (Sections 2-15)

### Section 2 — `RobustnessSpec` identity: **implemented and verified**
Deterministic `robustness_id` via `compute_robustness_identity`,
order-independent, schema-version-independent. Verified in
`test_specs_and_identity.py` (16 tests, including a parametrized
"any identity-relevant field change flips the id" sweep).

### Section 3 — Verified backtest input contract: **implemented and verified**
`verify_and_load_source_backtest` never trusts `stage=COMPLETED` alone —
re-runs `backtesting.verification.verify_backtest`'s full raw-source
reconstruction and cross-checks 4 declared source identities. Exercised
live by the Section 17 acceptance workflow against a real backtest.

### Section 4 — Return-series contract: **implemented and verified**
7 distinct kinds, never mixed. AR(1) effective-sample-size correction for
bar-sampled series is a **simplified**, honestly-labeled approximation
(documented in the technical reference), not a rigorous long-memory
correction.

### Section 5 — Dependence-aware bootstrap: **implemented and verified**
All 4 methods implemented; the stationary-bootstrap regression above was
found and fixed here. Default `STANDARD_STATISTICS` covers total return/
mean return/max drawdown/hit rate/profit factor/expectancy —
**implemented but limited**: Sharpe/Sortino/cost-drag/benchmark-relative
CIs are fully supported via `compute_bootstrap_report`'s
`statistics_by_name` override and/or a `BENCHMARK_RELATIVE`-kind series,
but are not part of the bare default set, since `STANDARD_STATISTICS` is
a module-level constant with no access to any one backtest's own
`periods_per_year`. A dedicated bootstrappable "cost drag" return series
kind does not exist (would require an 8th `ReturnSeriesKind` and a new
`series.py` builder) — **deferred**.

### Section 6 — Downside/loss-probability analysis: **implemented and verified**
Every probability reported with explicit "resampling-based estimate"
framing. Probabilities requiring a caller-supplied series
(underperforms-flat/-long, cost-stressed-unprofitable) are `None`, never
`0.0`, when that series was not supplied.

### Section 7 — Multiple-testing and selection-bias control: **implemented and verified**
Bonferroni/Holm/BH verified against hand-computed textbook reference
values. PSR verified against an exact closed-form reference value
(computed independently via `math.erf`, never by calling the module's
own `_standard_normal_cdf`). DSR/MinTRL fail-closed paths verified;
DSR's `sigma_SR` is always the caller's real, observed family Sharpes,
never assumed. MinTRL's positive-case numeric value is checked for
shape/finiteness/lower-bound only, not an independently-rederived exact
formula (rederiving it in the test would risk a correlated error with
the implementation) — **implemented but limited**, documented in the
test file itself.

### Section 8 — Fold stability and concentration risk: **implemented and verified**
`ConcentrationReport`'s ratio formula and `FoldStabilityReport`'s
profitable-fraction/worst/median-return/direction-consistency fields all
verified against hand-computed reference values.

### Section 9 — Parameter/decision sensitivity: **implemented and verified**
Structural applicability, monotonicity-violation counting, cliff
detection, and the parameter-sensitivity-score formula all verified
against hand-computed reference values. `ABSTENTION_THRESHOLD` is
**unsupported and fail-closed** at this layer by deliberate design (a
calibration-time concept, not reachable by backtest-level
re-simulation) — always reported as an explicit whole-axis skip, never
silently ignored.

### Section 10 — Cost, latency, and execution stress: **implemented and verified**
Exact cost-scaling arithmetic and the break-even bracket search verified
against hand-computed reference values (via a monkeypatched
`resimulate_stitched_outcome`, isolating the bracket-selection logic from
the expensive real re-simulation pipeline). Named XAUUSD stress profiles
are **implemented but limited**: illustrative configuration identities,
explicitly documented as not claims about real broker behavior.

### Section 11 — Regime robustness: **implemented and verified**
Leakage prevention proven directly (bars before a trailing window are
absent from the classification mapping, never defaulted); expanding-
quantile discrimination proven with a non-monotonic hand-computed case.
`SPREAD_QUANTILE` is **unsupported and fail-closed**: this platform's
backtesting-layer OHLCV schema has no spread column — reported as an
explicit `unavailable` dimension, never fabricated from a cost-model
constant. Future external-data regimes (session/rollover/macro-event/
risk-on-off/real-yield/DXY) are **deferred** per Section 11's own
instruction not to implement them ahead of the underlying data existing.

### Section 12 — Champion/challenger selection: **implemented and verified**
No-candidate-disappears and deterministic-ranking-under-reordering both
proven directly. "Stressed Sharpe" (Section 12's literal wording) is
**implemented but limited**: substituted with `worst_stress_scenario_
net_return` (the minimum total net return across evaluated, non-zero-
cost stress scenarios), since a true Sharpe ratio is not computable from
`StressReport`'s already-persisted scalar summary outcomes without
re-simulating a second time — documented as a deliberate substitution in
`selection.py`'s own module docstring, not a silent reinterpretation.

### Section 13 — Paper-trading promotion gates: **implemented and verified**
All 4 decision branches individually proven reachable, including the
REJECTED-beats-MANUAL_REVIEW_REQUIRED precedence case. Fail-closed
behavior for a skipped mandatory gate and an unknown gate name both
proven directly. `ELIGIBLE_FOR_LIVE_TRADING` proven absent from the enum
itself (not merely "never produced by policy").

### Section 14 — Manifest, persistence, resume, verification: **implemented and verified**
Semantic-tampering detection proven directly: a manifest that CLAIMS a
later stage than its own artifacts support is correctly demoted by
`verify_completed_robustness_stages`. Independent re-verification
(`verify_robustness`) exercised live against 2 real candidates in the
Section 17 acceptance workflow, both passing (`is_ready=True`).

### Section 15 — CLI: **implemented and verified**
All 10 commands registered, `--help` enumerates all of them, every
command's fail-closed error path (missing config, malformed JSON,
unknown id/content-hash) proven via real `subprocess.run` OS process
launches — no traceback, correct exit codes, `ERROR:` prefix. `run-
robustness`/`resume-robustness` are **implemented but limited** in test
coverage specifically: proven correct via 145 in-process tests including
the full real-model acceptance workflow, but not re-exercised a second
time as a subprocess (would roughly double this suite's wall-clock cost
for no additional CLI-boundary coverage beyond what `create-robustness-
spec`'s identical argument-parsing/dispatch/config-loading path already
proves) — documented explicitly in `test_robustness_cli_subprocess.py`'s
own module docstring.

## 4. Real acceptance workflow (Section 17)

`tests/integration/test_robustness_real_model_acceptance.py` — a real
`logistic_regression` model (not a constant test model) against a
synthetic dataset with genuine injected AR(1) predictive structure,
carried through a real execution run, a real calibration run, and **two**
distinct completed backtests (differing in holding period: 10 vs. 20
bars) from the same calibration. A `StrategyFamily` links both
candidates. Each candidate ran the FULL robustness pipeline
(`RobustnessRunner`) to `VERIFIED`/`COMPLETED`, including its own
standalone selection eligibility check, promotion evaluation, and
independent re-verification (`verify_robustness`, `is_ready=True` for
both). A cross-candidate `SelectionReport` was then computed over both
REAL candidates' evidence.

Result (this run, seed-deterministic): both candidates were eligible;
`bd0ac624443b...` (the 20-bar-holding-period candidate) was selected as
champion. **This outcome is not claimed as evidence of profitability or
live-trading viability** — see Section 6 below. Total wall-clock time for
the full workflow (dataset build through both candidates' complete
robustness pipelines): 606.96s (~10m7s), reflecting the genuinely
expensive nature of independent re-verification recomputing sensitivity/
stress/regime analysis a second time per candidate, exactly as Section 14
requires ("do not trust persisted... outputs").

## 5. Quality gates (this session, in order)

- `python -m ruff check .`: **all checks passed**.
- `python -m mypy src` (196 source files, strict mode): **Success: no
  issues found in 196 source files**.
- `python -m pytest tests --deselect tests/performance -q`: **3084
  passed, 1 skipped (pre-existing Windows-privilege skip, unrelated to
  this milestone), 0 failed, 57 deselected, 992.04s**.
- `python -m pytest tests/performance -m performance -q`: **57 passed, 0
  failed, 160.81s**.
- `python -m pytest -q` (full suite, twice back-to-back): **run 1: 3141
  passed, 1 skipped, 0 failed, 1157.47s. Run 2: 3141 passed, 1 skipped, 0
  failed, 1196.43s — identical pass/skip/fail counts both runs.**
- New Milestone 6 test files (145 tests) run with `-W error -q`: **145
  passed, 0 failed, 0 warnings, 587.28s.**
- Repository safety scan (every new Milestone 6 file, plus the diffs of
  every modified shared file): no `pickle`/`eval(`/`exec(`/`shell=True`/
  `os.system`, no unsafe `yaml.load`, no bare `except:`, no `TODO`/
  `FIXME`/`HACK` markers, no `print(`/`pdb.`/`breakpoint(` calls, no
  hardcoded local absolute paths, no oversized binary artifacts.
- Two real issues discovered and fixed during this session's own quality
  gates, neither a defect in prior milestones' work:
  1. `tests/unit/robustness/test_stability.py` collided with the
     pre-existing `tests/unit/optimization/test_stability.py` (pytest's
     basename-only module resolution in a package without `__init__.py`
     files) — renamed to `test_fold_stability.py`.
  2. `tests/unit/test_ml_cli.py::TestBuildParser`'s exact-command-set
     assertion did not yet include the 10 new robustness commands —
     updated (mirrors Milestone 5's own identical, already-documented
     precedent of needing this same kind of update for its own 8 new
     commands).

## 6. Explicit non-claims

This framework is **not** claimed to be production-ready, commit-ready,
or evidence of profitability. Every promotion decision embeds a
structurally-required disclaimer
(`promotion.DISCLAIMER`) stating it is not financial advice and not a
guarantee of future profitability. `ELIGIBLE_FOR_PAPER_TRADING` is not,
and is never described as, approval for live trading with real capital —
`ELIGIBLE_FOR_LIVE_TRADING` does not exist as a constructible value
anywhere in this codebase. The Section 17 acceptance workflow's own
promotion outcome is synthetic-data infrastructure evidence only, not
market evidence about any real instrument. See `docs/statistical_
robustness_and_promotion.md`'s "Unsupported claims" section for the
complete, itemized list this report summarizes.

No `git add`, `git commit`, or `git push` was run as part of this
milestone. This report does not claim commit-readiness.

## 7. Release-audit addendum

An independent, multi-pass release audit of this milestone (conducted
after the delivery described above, against the same uncommitted
working tree) found and fixed 6 further real defects, each via a
document-first, smallest-correct-fix, regression-tested process:

1. **Order-dependence in `robustness_id`/`policy_identity`**:
   `RobustnessSpec.stress_scenarios`/`regime_definitions`/
   `perturbations`, `PromotionPolicySpec.gates`, `PerturbationSpec.
   relative_deltas`, `SelectionPolicy.gates` are semantically unordered
   sets, but were serialized in caller-supplied order — two specs
   describing identical content in different declaration order produced
   different identities. Fixed by sorting in dedicated `to_identity_
   payload()`/`to_identity_payload()`-equivalent methods (see item 3).
2. **Explicit empty `perturbations=()` silently replaced on JSON
   reload**: `RobustnessSpec.from_json_dict` used `raw.get(...) or
   [defaults]`, conflating "key absent" with "key present but empty".
3. **A regression in the fix for item 1**: sorting directly inside
   `to_json_dict()` (rather than only in a separate identity-payload
   view) broke round-trip fidelity with `verification.verify_robustness`
   's positional recomputation, causing spurious `stress_report_
   mismatch`/`sensitivity_report_mismatch`/`regime_report_mismatch`
   findings on every real run — caught by the full acceptance-workflow
   test failing. Corrected to match the codebase's own established
   `multiple_testing.StrategyFamily.to_identity_payload` precedent:
   `to_json_dict()` stays order-preserving; canonicalization lives only
   in a separate identity-payload method.
4. **Illegal resume rewind transition**: `RobustnessRunner._run_locked`
   attempts to rewind a manifest's stage to the last independently-
   verified stage when corruption is detected, but `is_legal_
   robustness_transition` only permitted forward-by-one (or to
   `FAILED`) — ANY rewind raised `RobustnessStateError`, leaving a
   corrupted run permanently stuck (resumable per `can_resume`, but
   crashing identically on every resume attempt). Fixed by legalizing a
   transition to any strictly-earlier non-terminal stage.
5. **`verify_robustness` never compared `source_verification_report`**:
   every other required artifact kind was loaded and compared against a
   freshly recomputed value; this one was only checked for presence.
   Tampering with the persisted artifact's content went undetected.
   Fixed by adding the missing load-and-compare pair.
6. **`RobustnessSpec.promotion_policy` silently ignored**: validated,
   hashed into `robustness_id`, and persisted, but `RobustnessRunner`/
   `verify_robustness` both called `evaluate_promotion` with the
   hardcoded `promotion.DEFAULT_PROMOTION_POLICY`, never the declared
   spec field. No currently-exercised path was affected (every caller
   already declares the default policy), but an operator-supplied
   custom policy would have been silently discarded. Fixed in both call
   sites consistently.

The audit additionally closed real test-coverage gaps (direct unit
tests for `source.py`, `series.py`'s every `ReturnSeriesKind`
construction path, `verification.py`'s full tamper matrix, and a
crash/resume matrix across every `RobustnessStage` boundary) and
disclosed, without code changes, two intentional-but-previously-
undocumented design facts: `BAR_NET`/`PER_FOLD_BAR_NET` currently
produce byte-identical series content (now documented and tested as
such), and `StrategyFamily` does not validate cross-candidate dataset/
instrument/split-plan/bar-interval homogeneity (left to operator
discipline). See the audit's own final closure report for the complete,
exact quality-gate output and remaining-issue classification.
