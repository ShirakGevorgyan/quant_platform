# Statistical Robustness, Strategy Selection, and Promotion-Gate Framework (Milestone 6)

## Scientific purpose

Milestone 5 answers "what did this one backtest's own numbers say?"
under leakage-safe, independently-verifiable execution simulation. It
never asks *how much of that apparent performance survives resampling
uncertainty, multiple-testing correction, fold-level concentration,
cost/latency stress, or regime narrowness* — nor does it decide whether
a candidate is even worth paper-trading. This milestone answers exactly
those questions, strictly as a **post-hoc statistical analysis** of one
or more already-COMPLETED, already-independently-verified backtests. It
never refits a model, never re-optimizes a parameter to improve a
result, never connects to a broker or MetaTrader 5, and never produces
an `ELIGIBLE_FOR_LIVE_TRADING` decision — that outcome does not exist as
a value this milestone's code can even construct.

## The central rule this entire milestone exists to enforce

**Nothing here is trusted from a manifest's own claim, a persisted
report's own numbers, or a single scalar metric — everything is either
independently re-verified by recomputation from raw sources, or reported
as an explicit resampling-based estimate with its own uncertainty, never
as a guaranteed fact about future performance.**

This is enforced structurally, not by convention:

1. `robustness.source.verify_and_load_source_backtest` re-runs
   `backtesting.verification.verify_backtest`'s full raw-source
   reconstruction before any statistical analysis starts — a backtest
   manifest merely claiming `stage=COMPLETED` is never sufficient.
2. `robustness.verification.verify_robustness` independently
   RECOMPUTES every deterministic analysis (return series, bootstrap,
   downside probabilities, fold stability, sensitivity, stress, regime)
   from the verified source backtest, the declared `RobustnessSpec`, and
   the declared random seed — using the exact same production functions
   the forward pipeline uses, never a second, parallel implementation —
   and compares the fresh result against what was persisted, field by
   field (ignoring only wall-clock timestamps).
3. Every downside/loss probability is computed and reported with
   language such as *"resampling-based estimated probability under the
   observed sample"* — never as a guaranteed future probability.
4. Every bootstrap repetition that cannot define a statistic (zero
   variance, zero losses for profit factor, etc.) is counted and
   reasoned about explicitly (`BootstrapEstimate.skipped_repetitions`/
   `failure_reasons`) — never silently coerced to `0.0`/`NaN`/`inf`.
5. A skipped MANDATORY promotion gate fails closed
   (`PromotionDecisionKind.MANUAL_REVIEW_REQUIRED`, never treated as a
   pass) — see `promotion.py`'s own four-way decision precedence below.
6. `PromotionDecisionKind` has no `ELIGIBLE_FOR_LIVE_TRADING` member —
   not merely "never selected by policy," but structurally
   unconstructible. `PromotionDecision.disclaimer` is a required field on
   every decision, not an optional comment.

## Architecture

New top-level package `quant_platform.robustness`, depending on
`backtesting`, `calibration`, `execution`, `optimization`, `historical`,
`features`, `ml`, and `core` — a strict one-way dependency; none of those
packages import from `robustness`.

### Module layout (17 files)

- `models.py` — every shared enum (`RobustnessStage`, `ReturnSeriesKind`,
  `BootstrapMethodKind`, `MultipleTestingCorrectionKind`,
  `RegimeDimensionKind`, `StressAxisKind`, `PerturbationAxisKind`,
  `GateOutcomeKind`, `PromotionDecisionKind`) and the linear
  `RobustnessStage` transition table.
- `specs.py` — `RobustnessSpec` (the content-addressed identity root)
  and every nested spec it embeds: `BootstrapSpec`, `StabilityThresholds`,
  `StressScenarioSpec` (+ 12 `DEFAULT_STRESS_SCENARIOS`),
  `RegimeDefinitionSpec` (+ 5 `DEFAULT_REGIME_DEFINITIONS`),
  `PerturbationSpec` (+ 3 `DEFAULT_PERTURBATIONS`), `PromotionGateSpec`/
  `PromotionPolicySpec` (+ 14 `DEFAULT_PROMOTION_GATES`).
- `source.py` — Section 3's verified-backtest input contract:
  `verify_and_load_source_backtest`, `SourceVerificationReport`.
- `series.py` — Section 4's return-series contract: `ReturnSeriesBundle`,
  `build_return_series` (7 distinct, never-mixed series kinds).
- `bootstrap.py` — Sections 5/6: 4 dependence-aware bootstrap methods,
  8 named statistic factories, `BootstrapReport`, `DownsideAnalysisReport`.
- `multiple_testing.py` — Section 7: Bonferroni/Holm/Benjamini-Hochberg
  corrections, `StrategyFamily`, probabilistic/deflated Sharpe ratio,
  minimum track record length (Bailey & Lopez de Prado formulas, exact).
- `stability.py` — Section 8: `FoldStabilityReport`, `ConcentrationReport`.
- `resimulation.py` — shared re-simulation helper (a deliberate addition
  beyond the originally-suggested module list): both `sensitivity.py` and
  `stress.py` need the SAME underlying operation (re-run the source
  backtest's own verified predictions/bars through the full production
  pipeline under a `dataclasses.replace`-modified `BacktestSpec`); this
  module reuses `backtesting.runner.recompute_outer_fold_backtest_
  artifacts` — the exact production code path — never a re-implemented
  parallel one.
- `sensitivity.py` — Section 9: `SensitivityReport`, per-axis structural
  applicability, monotonicity-violation counting, cliff detection.
- `stress.py` — Section 10: `StressReport`, named XAUUSD stress profiles,
  break-even bracket search.
- `regimes.py` — Section 11: `RegimeReport`, leakage-safe trailing-window
  and expanding-quantile classification.
- `selection.py` — Section 12: `SelectionReport`, cross-candidate
  eligibility gates and deterministic ranking.
- `promotion.py` — Section 13: `PromotionDecision`, the four-way
  fail-closed decision.
- `manifests.py` — `RobustnessManifest`/`RobustnessManifestStore` and the
  append-only `RobustnessEventStore`, mirroring `backtesting.manifests`.
- `resume.py` — interruption-safe resume via independent artifact
  re-verification, never trusting `manifest.stage` alone.
- `runner.py` — `RobustnessRunner`: orchestrates the full linear pipeline.
- `verification.py` — independent reconstruction of every stage.
- `reporting.py` — `RobustnessReport`, the final consolidated artifact
  index.

## `RobustnessSpec` identity

`robustness_id = fingerprint_json(spec.to_identity_payload())` — a pure
function of every field except `schema_version`, computed via
`compute_robustness_identity`. Two independently constructed specs with
identical settings always produce the same `robustness_id`, verified by
`tests/unit/robustness/test_specs_and_identity.py`.

## Return-series contract (Section 4)

Seven distinct kinds, NEVER silently mixed: `BAR_NET`, `BAR_GROSS`,
`TRADE_NET`, `TRADE_GROSS`, `STITCHED_BAR_NET`, `PER_FOLD_BAR_NET`,
`BENCHMARK_RELATIVE`. Each `ReturnSeriesBundle` carries its own sampling
frequency, observation count, effective sample count, source artifact
content hashes, and time range.

**Effective sample size**: bar-sampled (autocorrelated) series use the
AR(1) approximation `n_eff = n * (1 - rho_1) / (1 + rho_1)`, clamped to
`[1, n]` — a coarse, honestly-labeled approximation, not a rigorous
long-memory correction. Trade-level and fold-level series use
`effective_sample_count == observation_count` unadjusted (these are
already non-overlapping, independent events by construction).

## Dependence-aware bootstrap (Section 5)

Four methods, one honest caveat each:

- **IID**: resamples individual observations independently.
  Explicitly weak for autocorrelated financial time series — implemented
  for comparison/completeness, never this package's recommended default.
- **Moving block** (Kunsch 1989): fixed-length, overlapping blocks with
  replacement. Preserves local dependence up to `block_length`, at the
  cost of "blocky" block-boundary seams.
- **Stationary** (Politis & Romano 1994): each block's length is itself
  random (geometric, mean `block_length`), removing the block-boundary
  seam artifact. **A real bug was found and fixed during this
  milestone's own development** in this method: an earlier version only
  randomized the very first block's start position and then let it drift
  forward contiguously, degenerating into a pure circular rotation of the
  whole series — every "resample" was then numerically identical for
  order-independent statistics like total return, collapsing the
  reported CI to a near-zero width that looked plausible but was
  silently wrong. Found via a direct CI-width comparison against
  `moving_block` on identical data, not by reasoning alone. Fixed by
  re-randomizing the start position and block length for EVERY block,
  not just the first. Regression-covered in `tests/unit/robustness/
  test_bootstrap.py::TestBlockBootstrapBoundaryBehavior`.
- **Fold-level**: resamples whole outer folds with replacement,
  preserving each fold's own internal order — the coarsest, most
  conservative dependence structure.

CIs are computed for at least: total return, mean return, maximum
drawdown, hit rate, profit factor, expectancy (the `STANDARD_STATISTICS`
default set); Sharpe/Sortino/cost-drag/benchmark-relative CIs are
supported via the same `bootstrap_statistic`/`compute_bootstrap_report`
functions with caller-supplied statistic functions and/or a
`BENCHMARK_RELATIVE`-kind series, but are **not** part of the bare
default set (see "Implemented but limited" in the delivery report).

Undefined repetitions (e.g., Sharpe on a zero-variance resample, profit
factor on a resample with zero losing periods) are counted in
`skipped_repetitions`/`failure_reasons` and excluded from the CI — never
coerced to `0.0`/`NaN`/`inf`. If every repetition fails, `BootstrapError`
is raised; no CI is fabricated.

## Downside/loss-probability analysis (Section 6)

`DownsideAnalysisReport` reports resampling-based ESTIMATES (never
guarantees) of: probability total net return ≤ 0, probability mean
return ≤ 0, probability Sharpe ≤ 0, and — only when the caller supplies
the corresponding series — probability of underperforming always-flat/
always-long, probability a cost-stressed variant becomes unprofitable,
and probability maximum drawdown exceeds a configured limit. A
probability the caller did not supply the input series for is `None`,
never `0.0`.

## Multiple-testing and selection-bias control (Section 7)

`StrategyFamily` is the durable, traceable record of every candidate a
selection was drawn from (candidate backtest/experiment/calibration/
optimization identities, search-space identity, selection metric,
eligibility-rules description) — `family_id` is a deterministic function
of this payload, order-independent in the candidate-id tuples.

Bonferroni, Holm (step-down), and Benjamini-Hochberg (step-up FDR)
corrections are implemented and verified against hand-computed textbook
reference values (`tests/unit/robustness/test_multiple_testing.py`).

Probabilistic Sharpe Ratio, Deflated Sharpe Ratio, and Minimum Track
Record Length follow Bailey & Lopez de Prado's published formulas
exactly. Each **fails closed rather than fabricating a value** when its
own required assumption is not available: PSR/MinTRL require ≥10
observations and non-zero variance; DSR requires ≥2 real, caller-supplied
family Sharpe ratios (never an assumed canonical `sigma_SR`) with
non-zero variance across them; MinTRL requires the observed Sharpe to
strictly exceed the benchmark Sharpe (otherwise the required track
record length is infinite/undefined, and `MultipleTestingError` is
raised rather than returning `inf`).

## Fold stability and concentration risk (Section 8)

`FoldStabilityReport` covers profitable-fold fraction, positive-Sharpe
fold fraction, median/worst fold return, fold return/Sharpe/trade-count/
exposure dispersion, maximum fold drawdown, worst-fold cost drag,
direction consistency, and benchmark outperformance fraction.
`ConcentrationReport` reports `max(positive contributions) / sum(positive
contributions)` for fold/trade/day/direction/confidence-tercile
groupings — `None` (not `0.0`) when there is no positive total to divide
by — with explicit warning codes when a declared threshold is exceeded.

## Parameter/decision sensitivity (Section 9)

Declared, bounded perturbations around the ALREADY-SELECTED operating
point — never a second optimization pass; `sensitivity.py` never picks
the best-performing perturbed value and never feeds one back into the
analyzed spec. Each perturbation axis is checked for STRUCTURAL
applicability to the source spec (e.g. `PROBABILITY_THRESHOLD` only
applies when `signal_mapping.kind == probability_bands`) before any
`relative_delta` is attempted; an inapplicable axis is skipped entirely
with a stated reason. `ABSTENTION_THRESHOLD` is unconditionally skipped
at this layer: this platform's `respect_calibration_abstention` is a
boolean, and the actual abstention threshold is a calibration-time
concept this backtest-level re-simulation cannot reach.

Reports: the local performance surface (every evaluated point), a
literal, hand-documented monotonicity-violation count (direction
reversals in the delta-sorted return sequence, flat steps not counted),
cliff detection (does the nearest-neighbor perturbation flip
profitability), a coarse single-axis rank-stability proxy, profitable-
neighborhood fraction, and a parameter-sensitivity score (`max
fractional swing in total net return / |baseline|`). These are this
platform's own defined formulas, not a named statistical procedure from
the literature — documented as such to avoid any appearance of borrowed
authority.

## Cost, latency, and execution stress (Section 10)

Deterministic scenarios: zero cost (informational only), base cost,
1.5x/2x/3x spread, 2x/3x slippage, increased commission/financing,
1-bar/2-bar additional latency, and a combined-adverse scenario (12
`DEFAULT_STRESS_SCENARIOS` total). Five illustrative named XAUUSD stress
profiles (`normal_liquidity`, `rollover_spread_expansion`,
`high_impact_macro_release`, `thin_session`, `broker_degradation`) are
provided as opt-in configuration identities — explicitly documented as
NOT claims about actual broker behavior.

Break-even values are searched on a FIXED, documented, deterministic
grid (multipliers `1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0`; latency
`0..10` bars) — never interpolated. The report states the tightest
bracket on that grid where `total_net_return` crosses from positive to
non-positive, or explicitly that no crossing was found within the
declared bounds.

## Regime robustness (Section 11)

Every regime label uses information available strictly AT-OR-BEFORE the
classified bar. Calendar dimensions (session, day-of-week, hour-of-day)
use only that bar's own timestamp. Trailing dimensions (trend direction,
price regime, volatility/liquidity quantile) use a fixed backward-looking
window; quantile dimensions additionally rank each bar's trailing metric
against the EXPANDING set of trailing metrics observed at or before that
same bar. `SPREAD_QUANTILE` is reported UNAVAILABLE at this layer: this
platform's backtesting-layer OHLCV schema carries no spread column.

Per-regime metrics (observation/trade count, gross/net return, Sharpe,
drawdown, hit rate, exposure, transaction cost, benchmark comparison) are
computed per bucket; a bucket below its declared `minimum_regime_samples`
is still reported, `skipped=True`, never dropped. Per-regime maximum
drawdown is computed on a SYNTHETIC equity curve built by compounding
only that bucket's own (generally non-contiguous) bar returns in
chronological order — an illustrative figure, never a real continuously-
held drawdown experience.

## Champion/challenger selection (Section 12)

`compute_selection_report` evaluates every candidate against a
`SelectionPolicy`'s gates (name/mandatory/min/max, closed vocabulary,
unknown names fail closed at construction) and ranks ELIGIBLE candidates
lexicographically across a declared, directionally-fixed metric order
(bootstrap lower bound of return, worst-fold return, worst stress-
scenario net return, drawdown, turnover, strategy complexity — higher/
lower-is-better fixed per metric, never caller-configurable, to avoid a
configuration error silently inverting a ranking). A candidate with an
unmeasurable value on a ranked metric sorts strictly worse than any
measured value; the final, fully deterministic tie-break is ascending
lexicographic order of the candidate's own `robustness_id`. No candidate
is ever dropped from the report because it failed — `candidate_
eligibility` lists every candidate regardless of outcome.

## Paper-trading promotion gates (Section 13)

`evaluate_promotion` produces exactly one of four decisions, in this
platform's own documented precedence:

1. **`REJECTED`** — at least one MANDATORY gate was measured and failed
   outright.
2. **`MANUAL_REVIEW_REQUIRED`** — no mandatory gate failed outright, but
   at least one could not be measured at all (missing evidence). Per
   Section 13's instruction, this is fail-closed: never treated as a
   pass, but distinguished from a definitive failure since a human
   reviewer may be able to supply the missing evidence.
3. **`RESEARCH_ONLY`** — every mandatory gate passed, but at least one
   non-mandatory (advisory) gate failed.
4. **`ELIGIBLE_FOR_PAPER_TRADING`** — every mandatory gate passed, and
   every measured non-mandatory gate also passed.

`ELIGIBLE_FOR_LIVE_TRADING` is not a member of `PromotionDecisionKind`.
`PromotionDecision.disclaimer` is a fixed, required field on every
decision (never a comment or UI afterthought): *"This is a
resampling-based, historical-evidence promotion decision... NOT
financial advice, NOT a guarantee of future profitability... NOT
approval for live trading with real capital."*

## Persistence, resume, and independent verification (Section 14)

`RobustnessManifest.named_artifacts` is keyed by ARTIFACT KIND (not by
`RobustnessStage`) — `BOOTSTRAP_COMPLETED` alone produces two distinct
artifacts (`bootstrap_report`, `downside_analysis_report`), which a
stage-keyed mapping could not represent 1:1. `resume.verify_completed_
robustness_stages` walks the linear stage order forward, re-reading and
re-decoding every artifact each stage is expected to have produced — it
returns the LAST stage whose own artifacts (and every earlier stage's)
verify successfully, never trusting `manifest.stage`'s own claim past
that point. Proven directly in `tests/unit/robustness/
test_manifests_and_resume.py::TestSemanticTamperingDetection` — a
manifest that CLAIMS `STRESS_COMPLETED` but whose artifacts only support
`SOURCE_VERIFIED` is correctly demoted.

`verify_robustness` independently recomputes every deterministic
analysis and compares it field-by-field (timestamps excluded) against
what was persisted — any mismatch is a CRITICAL finding.

## Annualization and statistical assumptions

- Sharpe/Sortino annualization uses `bars_per_year(bar_interval)`
  (`backtesting.metrics`), which assumes calendar time (365.25 days/
  year), not trading-session time — identical assumption to Milestone 5.
- The AR(1) effective-sample-size approximation assumes a first-order
  autoregressive dependence structure; genuinely longer-memory
  dependence would be underestimated by this correction.
- PSR/DSR/MinTRL assume the Cornish-Fisher-style skew/kurtosis expansion
  underlying Bailey & Lopez de Prado's formulas is a reasonable
  approximation for the analyzed return distribution — not verified
  against the true underlying distribution (which is, of course,
  unknown).
- The stationary/moving-block bootstrap's dependence structure is
  governed by a single scalar `block_length` — a simplification of
  whatever the true (possibly time-varying, possibly multi-scale)
  dependence structure of real market returns actually is.

## Unsupported claims (explicit)

- **Historical robustness does not prove future profitability.** Every
  probability, confidence interval, and stress result in this milestone
  describes the observed historical sample and this package's own
  resampling procedure — never a forecast.
- **Paper-trading eligibility is not live-trading approval.**
  `ELIGIBLE_FOR_PAPER_TRADING` means the declared gates were cleared
  against historical evidence under this platform's own promotion
  policy — nothing about real-money execution, slippage realism,
  operational risk, or regulatory compliance is addressed by this
  milestone.
- **Synthetic acceptance results are not market evidence.** The Section
  17 acceptance workflow's injected-AR(1)-signal dataset exists solely to
  prove the INFRASTRUCTURE works end-to-end; its own promotion outcome
  says nothing about any real market or instrument.

## XAUUSD/MT5 future integration points

This milestone builds every extension point Section 10/11 call for
without implementing external-data-dependent features ahead of the data
actually existing:

- `NAMED_XAUUSD_STRESS_PROFILES` (`stress.py`) — ready to use once an
  operator wants to declare a stress scenario under one of these names;
  the multipliers are illustrative, not calibrated to any real broker.
- `RegimeDimensionKind` already enumerates the eight regimes this
  milestone CAN compute leakage-safely from OHLCV alone; London/NY
  session overlap, rollover-period, macro-event-window, risk-on/risk-off,
  real-yield, and DXY regimes are explicitly NOT implemented — Section
  11's own instruction is "do not implement future external-data regimes
  unless the required historical data already exists and is time-
  aligned," which is not yet true for this platform's historical
  pipeline.
- `SPREAD_QUANTILE` regime classification and true bid/ask-based spread
  bootstrap CIs are both blocked on the same underlying gap: this
  platform's `core.types.OHLCV_COLUMNS` schema has no spread column at
  the backtesting layer (mirrors Milestone 5's own identical, already-
  documented limitation for `SpreadModelKind.BID_ASK_OBSERVED`).
