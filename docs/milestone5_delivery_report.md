# Milestone 5 Delivery Report — Leakage-Safe Financial Evaluation, Signal Simulation, Transaction-Cost Modeling, and Backtesting Framework

This report distinguishes **implemented guarantees** (structural, always
true), **tested behavior** (proven by an actual passing test), **assumptions**
(things taken as given, not independently re-verified by this milestone),
**simplified execution models** (deliberate, documented approximations),
and **unresolved limitations** (real gaps). It does **not** claim
production readiness or profitability — see the explicit non-claims at
the end.

## 1. What was delivered

- New top-level package `src/quant_platform/backtesting/` — 17 modules,
  ~5,010 lines: `models`, `specs`, `costs`, `signals`, `fills`,
  `positions`, `execution`, `trades`, `returns`, `drawdown`, `metrics`,
  `manifests`, `runner`, `resume`, `verification`, `reporting`,
  `__init__` (110-name public export surface).
- `src/quant_platform/config/backtesting_schemas.py` — the CLI's pydantic
  config schema (`BacktestConfig` + 8 sub-schemas).
- 8 new CLI commands wired into the existing `ml_cli.py`
  (`create-backtest-spec`, `run-backtest`, `resume-backtest`,
  `inspect-backtest`, `report-backtest`, `inspect-backtest-fold`,
  `verify-backtest`, `compare-backtests`) — additive only (+257 lines,
  0 deletions to that file).
- 12 new domain exceptions in `core/exceptions.py` (+104 lines, 0
  deletions) and 12 new `ArtifactCategory` values in `ml/models.py`
  (+53 lines, 0 deletions) — every modification to a pre-existing,
  shared file across this entire milestone was **purely additive**.
- `docs/financial_evaluation_backtesting.md` — the technical reference
  (~630 lines).
- Test suite: 5 new test files, ~1,755 lines, 88 tests — 2 unit
  (leakage-adversarial + reference-value regression), 2 integration
  (engine end-to-end/resume/corruption/concurrency + real-model
  acceptance), 1 performance.

## 2. Implemented guarantees (structural, hold by construction)

1. No bar-t-close entry by default — `EntrySpec.__post_init__` rejects
   `delay_bars=0` unless `allow_same_bar_close=True` is explicitly set.
2. `generate_signals` has no market-bar parameter anywhere in its call
   surface — structurally impossible for a future price to reach signal
   construction.
3. No position ever reads a bar beyond `fold_end_position` — every bar
   read in `execution.py` is bounds-checked before, never after, indexing.
4. No position ever crosses an outer-fold boundary — every fold starts
   flat; `FinalTradePolicyKind` explicitly governs a still-open position
   at fold end (discard / force-close / mark-incomplete-exclude).
5. Source calibration predictions are never trusted by filename/hash
   alone — `verify_and_load_predictions` independently re-derives the
   frozen calibrator and re-applies it to persisted raw probabilities,
   asserting reproduction within `1e-9`.
6. `BacktestSpec.source_execution_id` must equal `source_experiment_id`
   exactly, and the referenced `ExecutionManifest.stage` must be
   `COMPLETED` — checked at every run, not merely at spec construction.
7. Market bars are never forward-filled or interpolated — a timeline
   timestamp with no matching raw bar fails closed
   (`MarketDataBindingError`).
8. `trade_id` is a deterministic pure function of identity fields only
   (never financial-outcome fields) and is recomputed and cross-checked
   on every `TradeRecord` decode.
9. `CostBreakdown.total_cost` is cross-checked against the sum of its own
   itemized components on every decode.
10. `gross_return` is provably scenario-invariant under every cost-
    sensitivity multiplier — only cost components scale.
11. Every financial metric that is undefined for a fold's particular data
    is `skip`ped with an explicit reason — `NaN`/`Infinity` is never
    persisted (enforced by an explicit finiteness check in
    `compute_financial_metrics` that raises rather than silently stores
    a non-finite value).
12. `BacktestStage.FAILED` is reachable and reason-bearing —
    `BacktestRunner._fail` is wired into `_run_locked`'s exception
    handler from the initial implementation (the lesson carried forward
    from the prior milestone's audit, where the equivalent method was
    defined but never called).
13. A crash at any point during or after `run_outer_fold_backtest` is
    always safe to resolve by redoing the entire fold from scratch —
    every mid-fold stage has a legal transition edge back to
    `SOURCES_VERIFIED`.
14. Every backtest artifact carries a schema version, is content-
    addressed, and is independently re-verifiable — never trusted from a
    filename alone.

## 3. Tested behavior (proven by a passing test, not merely asserted)

15. 18 leakage-adversarial proofs (landmine-object instrumentation,
    structural signature checks, and rejection of deliberately-invalid
    configuration) in
    `tests/unit/backtesting/test_backtesting_leakage_adversarial.py` —
    all passing.
16. 12 reference-value/regression tests
    (`test_execution_and_runner_reference_values.py`) covering
    period-return aggregation with simultaneous exits, bucket-analysis
    tercile assignment and insufficient-sample flagging, benchmark math,
    cost-sensitivity monotonicity, and the mixed-directional/flat-signal
    regression (item 22 below).
17. Full end-to-end infrastructure acceptance
    (`test_backtest_runner_end_to_end`): real synthetic historical data →
    a real research dataset → a real prepared experiment → a real
    completed execution → a real completed calibration → a real
    completed backtest, independently re-verified, idempotent on rerun.
18. Resume/crash-window coverage: 5 distinct mid-fold crash points
    (spanning every artifact one fold writes) + 1 distinct post-fold
    stage-transition-burst crash point — all resume to an identical,
    independently-re-verified `COMPLETED` state.
19. `FAILED`-state reachability proven directly (a genuine domain
    exception mid-run leaves an accurate, diagnosable terminal record;
    a further `run()`/`resume()` attempt raises cleanly).
20. Corruption/tampering: bit-flipped artifact detection, `verify_backtest`
    fail-closed on a corrupted fold, a byte-valid-but-semantically-
    tampered `OuterFoldBacktestResult` caught by the financial-metrics
    recomputation proof, a `backtest_id`-mismatch tamper caught by
    key-consistency checking, an identity-field tamper caught at
    `TradeRecord` decode time, a cost-component tamper caught at
    `CostBreakdown` decode time.
21. Concurrency: two simultaneous `run()` attempts for the same backtest
    — exactly one wins, the loser fails fast (never hangs), no double
    publication, verified via `threading.Barrier`-synchronized `os.link`
    interception (deterministic, not `time.sleep`-based).
22. A real defect (accepted-but-`FLAT`-direction signals crashing
    `compute_fill_price`) was found via
    `tests/performance/test_backtesting_throughput.py`'s randomized
    predictions, fixed, and is now permanently regression-covered.
23. A real defect (`annualized_return` overflowing double precision for
    a fold spanning a tiny fraction of a year) was found via
    `test_backtesting_real_model_acceptance.py`'s real logistic-
    regression run (214 real trades placed), fixed with a log-space
    computation and explicit `OverflowError` → skip-with-reason guard.
24. Real-model acceptance (`test_logistic_regression_real_model_end_to_
    end_backtest`): a genuine `logistic_regression` model (not the
    constant test model) against synthetic data with injected AR(1)
    predictive structure, carried through execution → calibration →
    backtest, with `abstention_aware`/`long_short`/`close_and_reverse`
    policies, real trades placed, all financial metrics finite, all
    benchmark/cost-sensitivity artifacts present and internally
    consistent, independently re-verified.
25. All 8 CLI commands smoke-tested end-to-end against real on-disk
    artifacts in this session (`create-backtest-spec`, `run-backtest`,
    `inspect-backtest` [text], `verify-backtest`, `inspect-backtest-fold`,
    `resume-backtest`) — every command returned `rc=0` and produced
    internally consistent output.
26. Determinism: `compute_backtest_identity(spec)` matches the manifest's
    own `backtest_id` in every test that constructs one; a rerun against
    an identical spec is a byte-identical idempotent no-op.

## 4. Assumptions (taken as given, not independently re-verified here)

27. That `calibration.runner.OuterFoldCalibrationResult`'s upstream
    raw-probability generation was itself correct — this milestone
    re-verifies everything **downstream** of the raw probabilities
    (calibration transform, threshold decision), not the base model
    training that produced them.
28. That the historical data pipeline's raw OHLCV bars are themselves
    accurate — this milestone validates relational invariants (high/low
    bounds, chronology, no duplicates) and positional alignment, not the
    ground-truth accuracy of the source data.
29. That `execution.splitters.build_folds_from_split_binding`'s outer-fold
    boundaries (reused unchanged from the execution engine) are
    themselves leakage-safe — already audited in an earlier milestone,
    not re-audited here.
30. That the underlying OS/filesystem's `os.link`-based locking
    (`ml.concurrency.experiment_lock`, reused unchanged) provides the
    process-safety this milestone's crash/concurrency tests rely on.

## 5. Simplified execution models (deliberate, documented approximations)

31. `net_return`'s itemized cost fractions share `entry_observed_price`
    as their denominator rather than each cost's own effective price —
    makes `net_return = gross_return - total_cost` hold exactly rather
    than approximately, at a negligible precision cost at realistic
    (single-to-low-double-digit basis point) cost magnitudes.
32. `return_calculation_policy=LOG` computes `gross_return` in log space
    but `net_return` still subtracts `total_cost` linearly — not a
    rigorous log-space cost composition.
33. Slippage is deterministic-only — no seeded-stochastic slippage model
    exists in this milestone.
34. `exposure` on an `EquityPoint` is a simplified per-trade indicator
    (`exposure_cap` while a trade is open, `0` otherwise), not a true
    concurrent-exposure accounting for `overlap_policy=
    independent_overlapping`.
35. Equity curves are never averaged across outer folds — only scalar
    financial metrics are aggregated fold-wise (mean/std); a genuine
    pooled/stitched walk-forward equity curve is a materially different
    construction this milestone does not implement.
36. Benchmarks (`always_long_zero_cost`/`always_long_net_cost`) price off
    `close`-to-`close` directly, not routed through the full entry/exit
    fill machinery — a simple reference, not a fully simulated strategy.
37. `bars_per_year` uses calendar time (365.25 days/year), not
    trading-session time.
38. Confidence/uncertainty buckets use fixed terciles
    (`[0,1/3)/[1/3,2/3)/[2/3,1]`), never data-dependent percentiles.
39. Cost-sensitivity scenarios re-simulate the full signal set per
    scenario rather than analytically re-deriving net returns from the
    base-cost trade set — simpler and more obviously correct, at the
    cost of `O(scenarios)` re-simulation work (bounded, since the
    scenario count is small and pre-declared).

## 6. Unresolved limitations

40. `price_basis=bid_ask`/`SpreadModelKind.BID_ASK_OBSERVED` are fully
    implemented and structurally validated but cannot be exercised
    end-to-end against this platform's own historical data pipeline,
    which does not currently populate separate bid/ask columns (only
    `tick_volume`/`real_volume`/`spread`) — requires caller-supplied
    market data with real bid/ask depth.
41. Crash-window coverage (5 mid-fold + 1 post-fold-burst injection
    points) is representative, not the fully exhaustive matrix a
    dedicated crash-injection framework might enumerate.
42. No LightGBM/XGBoost variant of the real-model acceptance test was
    added (only `logistic_regression`, per the spec's "Logistic
    Regression preferred" guidance) — both are already registered in
    `ml.model_zoo` and would work with the existing calibration
    pipeline; adding a second acceptance test was deprioritized given
    this milestone's scope.
43. `BacktestRunner._require_compatible_environment` is an explicit
    no-op — this package performs no model fitting, so there is no
    library-version-sensitive numerics comparable to
    `CalibrationRunner`'s scikit-learn dependency to gate resumption on.
    Documented in code as a deliberate parallel, not an oversight.

## 7. Quality gates (this session, in order)

- Ruff (`ruff check src/ tests/`): clean, both before and after the
  final code modification.
- Mypy strict (`mypy src/`, 174 source files): clean, both before and
  after the final code modification.
- Full repository pytest suite, run twice back-to-back after the final
  modification: **2,835 passed, 1 skipped (pre-existing Windows-
  privilege skip, unrelated to this milestone), 0 failed, 0 warnings**
  — both runs, identical results (7m51s, then 7m32s).
- Clean-environment verification: a genuinely fresh temporary venv,
  `pip install -e ".[dev]"` from a clean state, import + ruff + mypy +
  the full backtesting/CLI test subset (88 tests) — all clean.
- Repository-wide safety scan: no `pickle`/`eval`/`exec`/`subprocess`/
  `os.system`/`shell=True`, no unsafe `yaml.load`, no hardcoded
  secrets/credentials, no bare `except:`, no `TODO`/`FIXME`/`HACK`
  markers, in any new file this milestone added.
- One pre-existing test (`tests/unit/test_ml_cli.py::TestBuildParser`)
  needed updating to include the 8 new command names in its exact-set
  assertion — found and fixed as part of this milestone's own
  integration into the shared CLI, not a defect in prior milestones'
  work.

## 8. Explicit non-claims

This framework is **not** claimed to be production-ready. It has **not**
been evaluated against real market data, real broker fills, or any live
execution venue. A backtested result — positive or negative — is **not**
evidence that the underlying strategy is, or would be, profitable in live
trading. Every report this framework produces embeds this disclaimer
verbatim (`backtesting.reporting._STANDARD_LIMITATIONS`). See
`docs/financial_evaluation_backtesting.md`'s "What this framework does
not claim" and "Known limitations and documented defects fixed during
development" sections for the complete, itemized list this report
summarizes.
