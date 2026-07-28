# Leakage-Safe Financial Evaluation, Signal Simulation, Transaction-Cost Modeling, and Backtesting Framework (Milestone 5)

## Scientific purpose

Milestone 4E answers whether a model's raw output is a *trustworthy
probability* and what *decision* to make from it. It never asks whether
acting on that decision would have made or lost money. This milestone
answers that question — strictly as an **evaluation** of an
already-frozen, already-verified calibrated-prediction policy against
historical market data, under explicitly modeled execution assumptions.
It never selects features, hyperparameters, models, or calibration/
threshold/confidence/uncertainty policies; it never submits a live order,
allocates capital across strategies, or optimizes leverage. See "What
this framework does not claim" below.

## The central rule this entire milestone exists to enforce

**No position is ever entered using information not actually available
at its own decision time, and no position ever crosses an outer-fold
boundary.**

```
Decision available: bar t closes -> prediction/signal known
Earliest possible fill: bar t+1's open (NEXT_BAR_OPEN, delay_bars=1, the default)
Same-bar-close ("bar t close") entry is REJECTED at BacktestSpec
construction time unless allow_same_bar_close=True is explicitly set.

Outer fold [start, end]:
    every position opens flat, every position closes (or is force-closed /
    discarded / excluded, per an explicit policy) at or before `end` --
    no bar past `end` is ever read.
```

This is enforced structurally, not by convention:

1. `backtesting.specs.EntrySpec.__post_init__` raises
   `BacktestValidationError` for `delay_bars=0` unless
   `allow_same_bar_close=True` is explicitly passed.
2. `backtesting.signals.generate_signals` has no market-bar parameter at
   all — its signature is `(predictions: VerifiedPredictionSet, *, spec,
   position_mode, respect_calibration_abstention)`. There is no Python
   expression through which a future price could reach signal
   construction.
3. `backtesting.execution.simulate_outer_fold_trades` walks bar
   *positions* chronologically, bounded by an explicit
   `fold_end_position: int` — every bar read is `bars.iloc[i]` for `i <=
   fold_end_position`, checked before indexing, never after.
4. `tests/unit/backtesting/test_backtesting_leakage_adversarial.py` proves this with
   active, fail-loud instrumentation: a `_Landmine` object placed at
   every bar strictly beyond a signal's own decision bar (or beyond
   `fold_end_position`) raises `AssertionError` on any numeric use, and
   leakage-critical functions are proven to complete successfully with
   landmines in place — for every combination of exit policy and
   final-trade policy.

## Architecture

New top-level package `quant_platform.backtesting`, depending on
`calibration`, `execution`, `ml`, `features`, and `historical` — a strict
one-way dependency; none of those packages import from `backtesting`.

### Why this package does not reuse `engine`/`costs`/`risk`/`strategy`

This platform's original milestone already shipped a bar-by-bar,
cursor-driven backtest engine (`engine.backtest_engine.BacktestEngine`,
`multiframe.cursor.TimeframeCursor`, `costs.models.CostModel` as a
mutable ABC, `risk.position_sizing`, `strategy.interfaces.Strategy`) —
built for a rule-based strategy that receives bars one at a time and
maintains mutable `Position`/`Portfolio` state. That design is a poor fit
here: this milestone evaluates an **already-computed, already-verified**
outer-fold prediction series (`calibration.runner.
OuterFoldCalibrationResult`) against historical bars, deterministically,
with content-addressed immutable specs, resumable/verifiable artifacts,
and no mutable global state — exactly the shape `calibration`/
`optimization`/`execution` already established, not the shape the
original engine established. The MATH from `costs.models.
FixedSpreadCostModel`'s adverse-price sign convention is reused verbatim
(re-expressed against this package's own `PositionDirection` enum); the
mutable `Portfolio`/ABC machinery is not.

### Module layout (15 files, dependency-ordered)

- `models.py` — every shared enum (position/signal-mapping/overlap/
  entry/exit/cost), the market-bar contract (`validate_market_bar_frame`),
  the verified-prediction contract (`VerifiedPredictionSet`), and the
  `BacktestStage` state machine.
- `specs.py` — `BacktestSpec` (the content-addressed identity root),
  `SignalMappingSpec`/`EntrySpec`/`ExitSpec`/`SpreadSpec`/
  `CommissionSpec`/`SlippageSpec`/`FinancingSpec`/
  `CostSensitivityScenario`.
- `costs.py` — adverse-sign spread/slippage price adjustments,
  commission/financing return fractions, the itemized `CostBreakdown`
  (self-verifying: persisted `total_cost` is cross-checked against the
  sum of its own components on every decode).
- `signals.py` — pure prediction → position-intent mapping (Section 8's
  central rule).
- `fills.py` — the one authoritative fill-price function
  (`compute_fill_price`), with entry-kind-aware vs. exit-basis-aware
  reference-price selection kept structurally separate (a real defect
  found and fixed during development — see "Known limitations and
  documented defects fixed during development" below).
- `positions.py` — the transient `OpenPosition` carrier (never persisted
  standalone).
- `execution.py` — the chronological bar-position-walk simulation →
  `TradeRecord`s. The single most complex module; see "Signal
  simulation" below.
- `trades.py` — the terminal, self-verifying `TradeRecord`
  (`trade_id` is a pure function of identity fields — source
  calibration, outer fold, signal sample position, direction, entry/exit
  timestamps — deliberately **not** of financial-outcome fields like
  `net_return`; recomputed and cross-checked on every decode).
- `returns.py` — gross/net return math, compounded/non-compounded
  equity-curve construction.
- `drawdown.py` — independent peak/trough/recovery analysis, recomputed
  directly from a persisted `EquityCurve`.
- `metrics.py` — ~30 financial metrics, `skip`-with-a-reason instead of
  `NaN`/`Infinity` for anything undefined for a fold's particular data.
- `manifests.py` — `BacktestManifest`/`BacktestManifestStore` and the
  append-only `BacktestEventStore`, mirroring `calibration.manifests`'
  exact pattern.
- `runner.py` — `BacktestRunner` (the top-level orchestrator),
  `run_outer_fold_backtest`, `verify_and_load_predictions` (Section 5's
  independent re-verification contract), and the benchmark/cost-
  sensitivity/bucket-analysis computation functions.
- `resume.py` — verified-artifact-based outer-fold resume planning.
- `verification.py` — `verify_backtest`, an independent cross-store
  re-audit including the financial-metrics recomputation proof (see
  "Verification" below).
- `reporting.py` — JSON/Markdown report builders, with mandatory
  hypothetical/non-live disclaimers embedded in every report.
- `__init__.py` — public re-export surface (110 names).

## `BacktestSpec` identity

Mirrors `calibration.specs.compute_calibration_identity` exactly:
`BacktestSpec.to_identity_payload()` is canonicalized and hashed
(`ml.fingerprints.fingerprint_json`) together with an
`identity_schema_version` marker, producing a stable `backtest_id`. Two
scientifically identical specs always produce the same id, regardless of
process, machine, or dict insertion order.

`source_execution_id` is validated to **equal** `source_experiment_id`
exactly (`BacktestSpec.__post_init__`): this platform's
`ExecutionManifestStore` is keyed purely by `experiment_id` — there is no
separate execution identity anywhere in the platform. The field is kept
structurally distinct from `source_experiment_id` only so a caller-visible
field always names "which verified execution run" a backtest is bound
to. `backtesting.runner.resolve_backtest_inputs` performs a genuinely
**new** integrity check beyond what calibration itself requires: it
additionally verifies the referenced `ExecutionManifest.stage is
ExecutionStage.COMPLETED`, refusing to backtest against an in-progress or
failed execution run.

Every identity-bearing field (`dataset_content_id`, `split_plan_
fingerprint`, `instrument_identity`, `bar_interval`) is independently
re-derived from the loaded `CalibrationManifest`/`ExperimentSpec` and
cross-checked at run time — an inconsistency fails closed
(`BacktestValidationError`), never silently proceeds with whichever value
happened to load.

## The source-prediction contract (Section 5)

`backtesting.runner.verify_and_load_predictions` never trusts a
calibration's persisted `OuterFoldCalibrationResult` by filename or hash
alone. It re-derives the frozen calibrator from its own persisted
parameters (`FrozenDecisionPolicy.selected_calibrator()`), re-applies
`.transform()` to the persisted raw outer-test probabilities, and asserts
the result reproduces the persisted calibrated probabilities within
`1e-9` — the exact recomputation check `calibration.verification.
_verify_calibrated_probabilities_reproduce` performs, repeated here
independently rather than imported, since this package must not simply
assume calibration's own verification was ever run.

## The market-data contract (Section 6)

`backtesting.models.validate_market_bar_frame` is a relational-invariant
check (`high >= max(open, close, low)`, `low <= min(open, close, high)`,
positive/finite prices, strict chronological order, no duplicate
timestamps) deliberately **separate from, and in addition to**,
`historical.models.validate_historical_schema` (which checks
column presence/dtype/timezone but not price relationships).

`backtesting.runner.resolve_market_bars_for_timeline` loads raw OHLC(V)
bars via `historical.loader.DatasetLoader` for the same symbol/timeframe
the source experiment was built from, then **positionally aligns** them
to the timeline's own row-index space (`bars.iloc[i]` corresponds to the
same bar as `timeline.iloc[i]`) — this is what lets `execution.py`'s
bar-position walk use `signal.sample_position` (an index into the
timeline) directly as a `bars` row position. Any timeline bar with no
matching raw market bar fails closed (`MarketDataBindingError`) — **no
forward-fill, no interpolation, ever**.

## Signal mapping (Section 8)

`generate_signals` is a pure function of a `VerifiedPredictionSet` and a
`SignalMappingSpec` alone — seven policies (`directional_long_flat`,
`directional_long_short`, `probability_bands`, `abstention_aware`,
`confidence_floor`, `uncertainty_ceiling`, `combined_confidence_
uncertainty`), nine closed reason codes. `MISSING_MARKET_BAR`/
`OVERLAP_POLICY_REJECTION` are deliberately **not** assignable here — both
require market data or already-open-position state this module
structurally never sees; they are assigned downstream, in `execution.py`.

An **accepted** signal can still carry `direction=FLAT`
(`directional_long_flat`'s "predicted negative → flat" case, and
`probability_bands`' middle dead zone both accept the signal while
calling for no position) — this is a documented, deliberate shape (see
"Known limitations and documented defects fixed during development"
below for the real defect this shape caused and how it was fixed).

## Signal simulation (Sections 9-11, 23-24)

`execution.simulate_outer_fold_trades` walks **bar positions**
chronologically — not signals in isolation. A signal-only loop ("for each
accepted signal, decide what happens") is ambiguous the moment two
signals are dense enough that the second's entry bar arrives before the
first's already-determined exit bar; walking bar positions and asking, at
each bar, "does the open position's exit trigger here? does a new
signal's entry trigger here?" resolves this unambiguously and
deterministically for every overlap/exit policy combination.

- **Overlap policies** (what happens when a new signal arrives while a
  position is open): `IGNORE`, `CLOSE_ONLY`, `CLOSE_AND_REVERSE`,
  `QUEUE` (the most recent queued signal is replayed as a fresh entry
  attempt the instant the open position closes), `INDEPENDENT_
  OVERLAPPING` (every accepted signal opens its own independent
  position — requires a self-contained exit policy; `OPPOSITE_SIGNAL`
  exits have no well-defined meaning for multiple simultaneously open,
  mutually independent trades, and are rejected outright for this
  overlap policy).
- **Exit policies**: `FIXED_HORIZON` (a declared bar count),
  `NEXT_BAR_CLOSE`, `END_OF_FOLD`, `OPPOSITE_SIGNAL` (a "flip": the same
  triggering signal that closed the old position is immediately
  re-evaluated, while flat, as if freshly arrived — this is what makes
  `LONG_SHORT` mode's reversal semantics compose naturally with the
  existing open/close primitives, with no special-casing).
- **Final-trade policies** (a trade still open when the fold ends):
  `DISCARD_INCOMPLETE`, `FORCE_CLOSE_AT_FINAL_PRICE`, `MARK_INCOMPLETE_
  EXCLUDE` (the default — never recorded as a closed `TradeRecord` at
  all).

## Fill prices and the entry/exit price-column distinction

`compute_fill_price` is the **one** authoritative fill function, used
identically for both entry and exit (via `is_entry: bool`) — spread and
slippage can never be applied twice or inconsistently between legs.
`EntrySpec.kind` (`next_bar_open`/`next_bar_mid`/`next_bar_side_aware`/
`delayed_bar`) — not the general `price_basis` field — determines WHICH
price column an entry fill is computed relative to; `price_basis`
(`close`/`mid`/`bid_ask`) governs every **exit**, since Section 11's exit
policies are purely about timing, never price-column choice. See "Known
limitations and documented defects fixed during development" for the
real defect this distinction fixes.

## Cost model (Sections 12/13)

Every cost component is an explicit, separately-named PRICE-SPACE
adjustment (spread, slippage — Section 10 requires persisting "observed
market price" separately from "final effective fill price") or
RETURN-FRACTION charge (commission, financing) — never one unexplained
"cost" field. The adverse-price sign convention (long entry increases,
long exit decreases; short entry decreases, short exit increases — every
adjustment always moves the realized price AGAINST the position) mirrors
`costs.models.FixedSpreadCostModel`'s already-validated convention
exactly.

**Slippage is deterministic-only in this milestone** — no seeded-
stochastic model is implemented. This is a documented simplification, not
an oversight (see "Known limitations" below).

## Returns and equity curves (Sections 14/15)

`gross_return` is computed from OBSERVED (unadjusted) entry/exit prices;
every itemized `CostBreakdown` component is computed as a return-fraction
relative to the SAME `entry_observed_price` denominator, never the
trade's own effective price. This deliberate simplification makes
`net_return := gross_return - total_cost` hold **exactly**, not merely
approximately — testable and exact, at the cost of computing net_return's
percentage relative to the reference price rather than the price actually
paid (a negligible difference at realistic, single-to-low-double-digit
basis-point cost magnitudes). `return_calculation_policy=LOG` computes
`gross_return` as a log return; `net_return` still uses the exact linear
`gross_return - total_cost` subtraction in that mode too — a second,
documented simplification, not a rigorous log-space cost composition.

`build_equity_curve` produces one point per **distinct exit timestamp** —
trades closing at the exact same timestamp (possible under
`overlap_policy=independent_overlapping`) are summed into one point
rather than treated as sequential points, so compounding never
double-applies simultaneous outcomes. A fold with zero closed trades
still produces exactly one (zero-return) point, since `EquityCurve.points`
must never be empty.

## Financial metrics (Sections 17/18)

~30 metrics, reusing `ml.metrics.MetricComputationReport`'s "skip with a
reason, never fabricate `NaN`/`Infinity`" shape for anything undefined
for a fold's particular data (zero-variance Sharpe, zero-trade payoff
ratio, zero gross loss profit factor, and so on).
`bar_return_sharpe`/`trade_return_sharpe` are never mixed or presented as
one unlabeled "Sharpe": the former uses the equity curve's period
returns, the latter the per-trade net-return series directly, sharing the
same annualization factor and risk-free-rate convention (both persisted
explicitly, per Section 18). `bars_per_year` is derived from `core.types.
Timeframe.minutes` using calendar time (365.25 days/year), not
trading-session time — a documented simplification, always well-defined
for any supported `Timeframe`.

`annualized_return` extrapolates a fold's total return to a full-year
compounding factor in log space (`exp(log1p(r) / duration_years) - 1`) —
guarded against `OverflowError`, skipped with a reason rather than
crashing or fabricating `inf` when a fold's duration is too short a
fraction of a year (e.g. a few hundred `M1` bars) for the extrapolation
to remain representable in double precision. See "Known limitations and
documented defects fixed during development" below — this guard exists
because of a real crash found via real-model testing.

**Round trips vs. transaction sides (Milestone 5.2, Section 7 correction).**
An earlier delivery report described a fold's transaction count divided
by its bar count as "round-trips per bar" — that is wrong. Each closed
trade (one round trip) contributes exactly TWO transaction sides (one
entry, one exit), so `transaction_count / bars` is the SIDE rate, not
the round-trip rate. Three separate, correctly-named metrics are now
persisted:

- `trades_per_bar` = `trade_count / bars` — the genuine round-trip rate
  (e.g. 314 trades / 350 bars ≈ 0.90). Always reported, even at zero
  trades (a well-defined fact, not an undefined ratio).
- `transaction_sides_per_bar` = `transaction_count / bars` — always
  exactly `2 × trades_per_bar` (e.g. 628 transactions / 350 bars ≈
  1.79). Skipped, like every other trade-dependent metric, when a fold
  has zero closed trades.
- `reversal_rate` — the fraction of chronologically adjacent closed-trade
  pairs whose direction flips (long→short or short→long). A distinct
  concept from either rate above (it describes the SEQUENCE of trade
  directions, not a per-bar frequency); skipped with fewer than two
  closed trades.

A report or dashboard must never print `transaction_sides_per_bar`'s
value under a "round trips" label, or vice versa.

## Drawdown (Section 19)

`compute_drawdown_report` is a single deterministic pass over an
`EquityCurve`, tracking the running peak and detecting peak → trough →
recovery episodes, independently for `gross` and `net` equity bases
(never one unlabeled "drawdown" — `DrawdownReport.equity_basis` names
which). `max_episodes` bounds how many of the *largest* episodes are
persisted in full; `maximum_drawdown`/`longest_drawdown_duration_bars`
are always computed over *every* episode, never just the persisted
subset. An episode still underwater at the equity curve's last point is
reported `recovered=False`, never assumed to recover later.

## Benchmarks (Section 26)

Computed per outer fold, always three: `always_flat` (the trivial
zero-return reference), `always_long_zero_cost` (buy and hold from the
fold's first test-bar close to its last, ignoring all costs — a simple
reference, deliberately not routed through the full entry/exit fill
machinery), `always_long_net_cost` (the same, paying one round-trip
transaction cost).

## Cost sensitivity (Section 20)

`compute_cost_sensitivity_report` re-simulates the SAME signal set under
every scenario in `BacktestSpec.cost_sensitivity_scenarios` — a bounded,
pre-declared set of spread/slippage/commission multipliers (defaults:
`zero_cost`, `base_cost`, `1.5x_spread`, `2x_spread`,
`increased_slippage`) — never a post-hoc "best scenario" search.
`gross_return` is proven scenario-invariant by construction (only cost
components scale); `net_return` differs, monotonically, with the
multiplier.

## Confidence/uncertainty bucket analysis (Section 21)

`compute_bucket_analysis_report` buckets **closed trades** by their
originating signal's confidence/uncertainty into fixed terciles (`[0,
1/3)`, `[1/3, 2/3)`, `[2/3, 1]`) — never data-dependent percentiles, so
bucket boundaries are never chosen post-hoc from the very outer-test
outcomes being analyzed. A bucket with fewer than `minimum_bucket_samples`
(default `5`) trades is flagged `insufficient_sample` and reports `None`
for average return/hit rate, rather than a statistically meaningless
average.

## Artifact model

Twelve `ArtifactCategory` values are specific to this milestone:
`BACKTEST_SPEC`, `VERIFIED_PREDICTION_SET`, `MARKET_DATA_BINDING`,
`SIGNAL_SET`, `TRADING_POLICY`, `TRADE_SET`, `RETURN_SERIES`,
`DRAWDOWN_REPORT`, `OUTER_FOLD_BACKTEST_RESULT`, `BENCHMARK_REPORT`,
`COST_SENSITIVITY_REPORT`, `BUCKET_ANALYSIS_REPORT`, `BACKTEST_REPORT`.
`OuterFoldBacktestResult` bundles references to every per-fold artifact
(signal set, trade set, equity curve, gross/net drawdown, benchmark/
cost-sensitivity/bucket reports) plus financial metrics **inline**,
consistent with `calibration`'s own precedent.

## Determinism

Identical inputs (spec + source calibration/execution artifacts + market
data) produce an identical `backtest_id`, identical signal ordering,
identical trades (including deterministic `trade_id`s), identical equity
curves, and identical content-addressed artifact hashes — verified
directly (`test_backtest_runner_end_to_end`'s idempotent-rerun assertion,
and `compute_backtest_identity(spec)` matching the manifest's own
`backtest_id`). This package performs no model fitting/refitting of its
own — there is no library-version-sensitive numerics comparable to
`CalibrationRunner`'s scikit-learn dependency to gate resumption on;
`BacktestRunner._require_compatible_environment` is kept as an explicit
no-op for this reason, with the parallel to `CalibrationRunner`'s
identically-named method documented in code.

## Resume and crash safety

`BacktestStage`'s state machine (`CREATED` → `SOURCES_VERIFIED` →
`SIGNALS_READY` → `FILLS_READY` → `TRADES_READY` → `RETURNS_READY` →
`METRICS_READY` → `REPORTS_READY` → `VERIFIED` → `COMPLETED`/`FAILED`)
mirrors `CalibrationStage`'s exact mid-fold-restart-edge design: `run_
outer_fold_backtest` computes signals through bucket analysis as one
atomic, pure function of already-fixed inputs, so every stage strictly
between `SOURCES_VERIFIED` and `REPORTS_READY` has a legal edge straight
back to `SOURCES_VERIFIED` — a crash at any point during or immediately
after that call is always safe to resolve by redoing the entire fold from
scratch.

**The `_fail()` reachability lesson, applied proactively.** A prior
milestone's audit found `CalibrationRunner._fail` fully defined but never
actually called, making `CalibrationStage.FAILED` unreachable in
practice. This milestone wired `BacktestRunner._fail` into `_run_locked`'s
exception handler from the initial implementation, and
`test_genuine_domain_exception_mid_run_leaves_an_accurate_failed_record`
proves it: any `QuantPlatformError` other than `ExperimentLockError`
(lock contention/an aborted process — never a real "this backtest's data
or config is wrong" verdict) transitions the manifest to
`BacktestStage.FAILED` with the exception message as `failure_summary`
and appends a `RUN_FAILED` event before re-raising. A further `run()`/
`resume()` attempt against a `FAILED` backtest raises
`BacktestResumeError` cleanly.

Proven with real interrupted runs across five distinct crash points
spanning every artifact one fold writes (signal set, trade set, equity
curve, gross/net drawdown, benchmark/cost-sensitivity/bucket reports —
`test_backtest_runner_resumes_after_mid_fold_crash`), and a distinct crash
window strictly AFTER `run_outer_fold_backtest` returns but DURING the
post-fold manifest stage-transition burst
(`test_crash_during_post_fold_stage_transition_burst_is_still_resumable`)
— resume always redoes the whole fold and reaches an identical
`COMPLETED` terminal state, independently re-verified afterward.
`backtesting.resume.verify_completed_backtest_outer_folds` re-checks
every outer fold the manifest *claims* is completed by content hash,
category, and decoded self-identity before trusting it.

## Verification

`backtesting.verification.verify_backtest` is an independent, read-only
re-audit, structured like `calibration.verification.verify_calibration`:
it re-checks the spec's identity, every outer-fold result artifact and
its dependent references (fully **decoded**, not merely read as bytes —
decoding re-runs `TradeRecord`/`CostBreakdown`'s own self-verifying
recomputation checks), manifest/event-log self-consistency (including
that `SOURCES_VERIFIED` strictly precedes that fold's own `SIGNALS_
GENERATED`/`TRADES_CONSTRUCTED` events), and — the check that makes this
more than a hash-consistency scan — **recomputes** the equity curve,
gross/net drawdown, and financial metrics directly from the persisted
`TradeSet` alone, asserting every recomputed scalar matches the persisted
`financial_metrics` within `1e-6`.

"Hash validity insufficient — semantically wrong hash-consistent
artifacts must be rejected" is proven directly in
`tests/integration/test_backtesting_engine.py`: a byte-valid,
correctly-hashed, but semantically-tampered `OuterFoldBacktestResult`
(a `total_net_return` edited by `+999.0`) is filed under a manifest
reference exactly as a compromised process might, and `verify_backtest`
still reports a `CRITICAL` `financial_metrics_do_not_reproduce` issue.

**What verification cannot independently confirm**: it does not
re-verify the source calibration's own predictions a second time (that
is `verify_and_load_predictions`'s job, already performed once when the
backtest originally ran) — this module verifies that the backtest's OWN
artifacts are internally self-consistent and correctly derived from each
other, not that the upstream calibration was itself correct.

## CLI

```
python -m quant_platform.ml_cli create-backtest-spec --config bt_config.json
python -m quant_platform.ml_cli run-backtest --config bt_config.json
python -m quant_platform.ml_cli resume-backtest --config bt_config.json --backtest-id ID
python -m quant_platform.ml_cli inspect-backtest --config bt_config.json --backtest-id ID [--format json]
python -m quant_platform.ml_cli report-backtest --config bt_config.json --backtest-id ID
python -m quant_platform.ml_cli inspect-backtest-fold --config bt_config.json --backtest-id ID --outer-fold-index N
python -m quant_platform.ml_cli verify-backtest --config bt_config.json --backtest-id ID
python -m quant_platform.ml_cli compare-backtests --config bt_config.json --backtest-id ID --baseline-backtest-id ID --metric total_net_return
```

`create-backtest-spec` is a dry run (validates + prints the deterministic
`backtest_id`, writes nothing). `resume-backtest` against an
already-`COMPLETED` backtest is a safe idempotent no-op (exit `0`).
`report-backtest` is an alias for `inspect-backtest`. Every command
returns `0` on success, non-zero on failure (`2` specifically when a
run/verify did not reach a successful terminal state), and prints an
actionable message — never a raw traceback for a domain error.

`config.backtesting_schemas.BacktestConfig` is the CLI's pydantic config
schema (`frozen=True, extra="forbid"`, mirroring `config.
calibration_schemas` exactly). It binds directly to an already-completed
`source_calibration_id`; `BacktestConfig.build()` takes an already-loaded
`CalibrationManifest`/`ExperimentSpec` and derives every identity-relevant
field from them — never re-typed by a human into the config file, where
it could silently drift from the actual bound calibration/experiment.

## Example: constructing and running a backtest

```python
from quant_platform.backtesting.specs import BacktestSpec, SignalMappingSpec, EntrySpec, ExitSpec, SpreadSpec, CommissionSpec, SlippageSpec, FinancingSpec
from quant_platform.backtesting.models import (
    SignalMappingPolicyKind, PositionMode, EntryPolicyKind, ExitPolicyKind, FinalTradePolicyKind,
    OverlapPolicyKind, PriceBasisKind, SpreadModelKind, CommissionModelKind, SlippageModelKind, FinancingModelKind,
    ReturnCalculationPolicyKind, CompoundingPolicyKind, DecisionTimestampPolicyKind,
)
from quant_platform.backtesting.runner import BacktestRunner
from quant_platform.calibration.models import DeterminismPolicy

spec = BacktestSpec(
    schema_version=1, source_calibration_id=calibration_id, source_experiment_id=experiment_id, source_execution_id=experiment_id,
    dataset_content_id=dataset_content_id, split_plan_fingerprint=split_fingerprint,
    instrument_identity="XAUUSD", market_timezone="UTC", bar_interval=Timeframe.H1,
    decision_timestamp_policy=DecisionTimestampPolicyKind.AFTER_BAR_CLOSE,
    signal_mapping=SignalMappingSpec(kind=SignalMappingPolicyKind.DIRECTIONAL_LONG_FLAT), position_mode=PositionMode.LONG_FLAT,
    entry_spec=EntrySpec(kind=EntryPolicyKind.NEXT_BAR_OPEN, delay_bars=1),
    exit_spec=ExitSpec(kind=ExitPolicyKind.FIXED_HORIZON, holding_period_bars=5, final_trade_policy=FinalTradePolicyKind.MARK_INCOMPLETE_EXCLUDE),
    overlap_policy=OverlapPolicyKind.IGNORE, price_basis=PriceBasisKind.CLOSE,
    spread_spec=SpreadSpec(kind=SpreadModelKind.FIXED_BASIS_POINTS, basis_points=5.0),
    commission_spec=CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=2.0),
    slippage_spec=SlippageSpec(kind=SlippageModelKind.ZERO), financing_spec=FinancingSpec(kind=FinancingModelKind.NONE),
    return_calculation_policy=ReturnCalculationPolicyKind.SIMPLE, compounding_policy=CompoundingPolicyKind.NON_COMPOUNDED,
    initial_notional=10_000.0, determinism_policy=DeterminismPolicy.STRICT,
)
runner = BacktestRunner(
    ml_artifacts_root=root, calibration_manifest_store=cms, experiment_manifest_store=ems,
    execution_manifest_store=xms, research_manifest_store=rms, research_dataset_store=rds, dataset_loader=loader,
)
outcome = runner.run(spec)
assert outcome.manifest.stage.value == "completed"
```

## What this framework does not claim

- **Every result is hypothetical and non-live.** No live order was ever
  submitted, no real capital was ever at risk, and no broker, exchange,
  or liquidity venue was ever involved — every report this framework
  produces embeds this disclaimer verbatim.
- **Past performance, even simulated, does not indicate future results.**
  A positive backtested return does not demonstrate that a strategy is,
  or will be, profitable in live trading.
- **Modeled costs are declared assumptions, not measurements** of any
  specific broker's real-world spread, commission, slippage, latency, or
  liquidity.
- **No live execution, portfolio allocation, or risk management is
  performed.** This framework evaluates one declared strategy in
  isolation — no capital-allocation-across-strategies, leverage
  optimization, or production risk-limit logic exists anywhere in this
  package.
- Equity curves are reported **per outer fold, never averaged together**
  — see "Known limitations" below for why a genuine pooled/stitched
  walk-forward curve is a materially different construction this
  milestone does not attempt.
- The primary bounded end-to-end acceptance run
  (`tests/integration/test_backtesting_engine.py::test_backtest_runner_
  end_to_end`) uses a test-only constant model
  (`ml.testing.ConstantTestModelFactory`) — it is infrastructure evidence
  (the pipeline composes correctly end-to-end), never evidence of
  predictive or market edge.
  `tests/integration/test_backtesting_real_model_acceptance.py`
  separately runs the full pipeline against a real production model
  (`logistic_regression`) on synthetic data with genuine (not
  market-derived) injected predictive structure — this confirms the
  pipeline behaves correctly, with real trades actually placed, still
  never evidence of edge on real market data.

## Known limitations and documented defects fixed during development

- **`price_basis=bid_ask` requires bid/ask columns this platform's
  historical data pipeline does not currently populate** — raw historical
  bars carry `tick_volume`/`real_volume`/`spread`, not separate `bid`/
  `ask` depth. `SpreadModelKind.BID_ASK_OBSERVED`/`PriceBasisKind.BID_ASK`
  are fully implemented and structurally validated, but exercising them
  end-to-end requires caller-supplied market data with real bid/ask
  columns.
- **`exposure` on an `EquityPoint` is a simplified per-trade indicator**
  (`spec.exposure_cap` while a trade is open, `0.0` otherwise) — not a
  true concurrent-exposure accounting for `overlap_policy=
  independent_overlapping`, where multiple trades can be open at once.
  Documented, not silently wrong: the field's meaning is exactly what is
  described here.
- **Equity curves are never averaged across outer folds.**
  `reporting._aggregate_metrics_section` aggregates each fold's already-
  computed SCALAR financial metrics (fold-wise mean/std); it never
  concatenates or averages the folds' own `EquityCurve` objects, since
  folds cover different, non-overlapping calendar windows and a
  genuine pooled/stitched walk-forward curve (chaining each fold's curve
  after the previous one's ending equity) is a materially more involved
  construction this milestone does not implement.
- **Slippage is deterministic-only** — no seeded-stochastic slippage
  model exists in this milestone.
- **`net_return`'s cost fractions share `entry_observed_price` as their
  denominator** rather than each cost's own effective price (see
  "Returns and equity curves" above) — a deliberate, documented
  simplification trading exact testable additivity for a negligible
  precision cost at realistic cost magnitudes.
- **The entry price-column/price-basis conflation (fixed during
  development)**: a first draft of `fills.py` used one
  `select_reference_price()` function governed only by `price_basis`,
  causing `EntrySpec.kind=NEXT_BAR_OPEN` to silently price off `close`
  instead of `open`. Fixed by splitting into `select_entry_reference_
  price()` (driven by `EntrySpec.kind`) and `select_exit_reference_
  price()` (driven by `price_basis`), caught via hand-verification before
  any test suite existed.
- **The opposite-signal "flip" reopen bug (fixed during development)**: a
  first draft used a `consumed_as_exit` flag that incorrectly prevented
  the same triggering signal from reopening a reversed position
  immediately after an `OPPOSITE_SIGNAL` exit in `LONG_SHORT` mode. Fixed
  by removing the flag and letting execution fall through naturally to
  the "if flat, open" branch.
- **The accepted-but-`FLAT`-direction signal crash (fixed during
  development, caught by `tests/performance/test_backtesting_throughput.
  py` using randomized, threshold-crossing predictions rather than a
  near-constant test model)**: `SignalMappingPolicyKind.
  directional_long_flat`'s "predicted negative → flat" case (and
  `probability_bands`' middle dead zone) produce an ACCEPTED `Signal`
  with `direction=FLAT` — `execution.simulate_outer_fold_trades` was
  treating every accepted signal as entry-triggering, crashing on
  `compute_fill_price(direction=FLAT)`. Fixed by filtering
  `direction is FLAT` out of the entry-triggering signal list in
  `simulate_outer_fold_trades`, exactly mirroring `ExitPolicyKind.
  OPPOSITE_SIGNAL`'s own pre-existing close-trigger check (which already
  excluded `FLAT`). Regression-covered in
  `tests/unit/backtesting/test_execution_and_runner_reference_values.py::
  TestMixedDirectionalAndFlatAcceptedSignals`.
- **The `annualized_return` overflow (fixed during development, caught by
  `tests/integration/test_backtesting_real_model_acceptance.py`)**:
  extrapolating a short fold's return to a full-year compounding factor
  via `(1+r) ** (1/duration_years)` overflowed Python's `float` range for
  a fold spanning a tiny fraction of a year (a few hundred `M1` bars)
  combined with a large realized return. Fixed by computing in log space
  and catching `OverflowError`, skipping the metric with an explicit
  reason (never fabricating `inf`) when the extrapolation is not
  numerically representable — see "Financial metrics" above.
- **Cross-fold-boundary crash-window coverage is representative, not
  exhaustive.** Five distinct mid-fold crash points (spanning every
  artifact one fold writes) plus one distinct post-fold-transition-burst
  crash point are covered by real interrupted-run tests; a fully
  exhaustive enumeration of every conceivable crash injection point was
  not attempted given this milestone's scope.
