# Leakage-Safe Feature Selection and Hyperparameter Optimization Engine (Milestone 4D)

## The central rule this entire milestone exists to enforce

**All feature selection, preprocessing choice, early stopping, threshold
choice, and hyperparameter optimization happen entirely inside the outer
fold's training region. The outer test partition is evaluation-only,
touched exactly once, after candidate selection is fully complete.**

Every design decision below is subordinate to this rule. The single
strongest structural enforcement is `optimization.outer_fold.
finalize_outer_fold`: it is the ONLY function anywhere in this package
that ever reads a `Fold.test_indices` value. No other module (`study`,
`trial_executor`, `feature_selection`, `candidates`) accepts, references,
or has any code path capable of obtaining outer-test row positions —
proven structurally (`inspect`-based tests confirm `optimization.
candidates.rank_trials`'s signature contains nothing resembling an
outer-test metric) and empirically (the full nested pipeline, run
end-to-end against a real dataset, is verified never to have evaluated
outer-test before its own `CANDIDATE_SELECTED` event — see "Verification"
below).

## Why this milestone trains no NEW kind of model

Exactly the same constraint carried forward from Milestones 4A-4C:
SHAP/LIME, probability calibration, ensembles/stacking/blending,
reinforcement learning, neural networks, distributed training, online
learning, live trading, and recursive feature elimination are all
explicitly out of scope. This milestone does not add a new predictive
algorithm — it adds a research layer that SELECTS features and
hyperparameters for the models `ml.model_zoo` already provides (LightGBM,
XGBoost, CatBoost, Logistic Regression, Elastic Net, and the baselines),
using the SAME `ml.interfaces.TrainableModel.fit -> FittedModel.predict`
contract Milestones 4B/4C already exercise. Nothing here ever reads or is
influenced by outer-test performance during search (see "The central
rule" above) — outer-test evaluation exists solely to REPORT how the
already-fully-selected candidate generalizes, never to select it.

## Architecture

New top-level package `quant_platform.optimization`, depending on `ml`,
`execution`, `features`, and `validation` — a strict one-way dependency;
none of those existing packages import from `optimization`. Every
existing guarantee from Milestones 4A-4C (experiment identity, dataset
immutability, label-horizon purge, preprocessing fail-closed behavior,
fold chronology, resume guarantees, content-addressed artifacts,
deterministic seed propagation, cross-store verification) is reused
directly, never reimplemented or weakened.

Module layout (16 files, dependency-ordered):

- `search_space.py` — typed hyperparameter search-space model (no raw
  Optuna lambdas ever cross this package's boundary).
- `models.py` — `OptimizationSpec`/`OptimizationIdentity`, the
  `OptimizationStage` state machine, `PruningConfig`/`EarlyStoppingConfig`/
  `PreprocessingPolicy`, and the full hierarchical seed-derivation chain.
- `inner_splits.py` — nested (inner) walk-forward split construction and
  its own independent leakage validator.
- `feature_selection.py` — the immutable candidate feature universe and
  all 6 required feature-selection strategies.
- `objectives.py` — primary-metric/objective compatibility and the
  "undefined metric is never silently zero" aggregation policy.
- `candidates.py` — `TrialSpec`/`TrialResult` and the ONE deterministic
  ranking policy.
- `study.py` — Optuna integration: sampler construction, typed-space-to-
  suggestion translation, the empirically-verified resume mechanism, and
  the platform-owned median-stopping pruning rule.
- `trial_executor.py` — isolated, idempotent single-trial execution (the
  nested inner loop).
- `outer_fold.py` — outer-fold finalization — see "The central rule".
- `stability.py` — feature/hyperparameter stability analysis.
- `manifests.py` — `OptimizationManifest`/`OptimizationManifestStore` and
  `OptimizationEventStore`.
- `resume.py` — verified-artifact-based trial/outer-fold resume planning.
- `verification.py` — `verify_optimization`, an independent cross-store
  re-audit.
- `reporting.py` — JSON/Markdown report builders.
- `runner.py` — `OptimizationRunner`, the one orchestrator.
- `__init__.py` — public re-export surface (74 names), mirroring `ml/
  __init__.py`'s and `execution/__init__.py`'s exact convention.

## OptimizationSpec identity

Mirrors `ml.experiment_identity` exactly: `OptimizationSpec.
to_identity_payload()` is canonicalized (via `ml.persistence.
canonical_json_bytes`) and hashed (via `ml.fingerprints.fingerprint_json`)
together with an `identity_schema_version` marker, producing a stable
`optimization_id`. Two scientifically identical specs always produce the
same id, regardless of process, machine, or dict insertion order.

**Metric direction is never caller-trusted.** `OptimizationSpec.
metric_direction` IS part of the frozen identity payload (protecting a
future registry reinterpretation from silently changing the meaning of an
OLD, already-computed identity), but `__post_init__` independently
re-derives it via `ml.comparison.is_higher_better(primary_metric)` and
raises if the two disagree. Only `build_optimization_spec` — which always
derives the field correctly — is the sanctioned construction path;
hand-constructing an `OptimizationSpec` with a mismatched direction fails
closed at construction time.

## Preprocessing policy (Option A)

Scale-sensitive models (Logistic Regression, Elastic Net) are excluded
from the optimization surface entirely — consistent with Milestone 4C's
already-shipped fail-closed precedent (`ml.model_validation.
_validate_preprocessing_requirements`): this milestone implements no
fold-local preprocessing-refitting framework, so a scale-sensitive model
cannot be safely searched over until one exists. Enforced at TWO layers:

1. **`build_optimization_spec`** fails fast, at spec-construction time,
   whenever a `ModelRegistry` is supplied and the named model declares
   `ModelCapabilities.requires_scaled_numeric_features=True` — a
   friendlier, earlier fail-fast layer, but only opportunistic (fires
   only when a registry happens to be passed).
2. **`trial_executor.run_trial`** unconditionally re-checks the SAME
   condition itself, via a throwaway "probe model" instantiation, before
   running a single inner fold — the REAL, always-active gate. Whatever
   the constructor-time check concluded is never trusted alone.

`run_trial` also unconditionally rejects any model whose
`ModelCapabilities.is_deterministic` is not `True` — enforced structurally
at the same probe-model check point, not as a dead always-true
`OptimizationSpec` field.

## Seed derivation hierarchy

Every seed used anywhere in a nested search is a pure, deterministic
function of `SeedConfiguration.master_seed` (via `ml.seeds.derive_seed`),
never Python's global `random` module or NumPy's global RNG state
(verified directly in tests):

```
sampler_seed(config)
outer_fold_seed(config, outer_fold_index)
  -> trial_seed(config, outer_fold_index, trial_number)
       -> inner_fold_seed(config, outer_fold_index, trial_number, inner_fold_index)
            -> feature_selector_seed(...)
            -> model_fit_seed(...)
outer_train_refit_seed(config, outer_fold_index)              # independent branch
outer_train_feature_selector_seed(config, outer_fold_index)   # independent branch
```

All branches were verified pairwise-distinct across a large sample of
`(outer_fold_index, trial_number, inner_fold_index)` combinations, and
fully deterministic given the same master seed.

## Nested walk-forward construction

`inner_splits.build_inner_fold_plan` reuses `execution.splitters.
generate_expanding_folds`/`generate_rolling_folds` a SECOND time, against
a sub-timeline sliced from one outer fold's OWN `train_indices` — the
exact same splitting engine (`validation.walk_forward.
PurgedWalkForwardSplitter`) the outer engine already uses, nothing
hand-rolled. Local positions the inner splitter returns are mapped back
to GLOBAL timeline positions via one line of numpy fancy-indexing
(`outer_fold.train_indices[local_position]`).

**`InnerSplitConfig` has no separate `purge_bars` field.** Inner purge
always equals `execution.splitters.required_label_purge_bars_for(
label_horizon_bars)` exactly — the identical off-by-one-proven purge floor
Milestone 4B's own audit established for outer folds, applied one level
deeper. Only `embargo_bars` (a genuine policy choice) is caller-
configurable. `test_size_fraction` (not a fixed row count) scales each
outer fold's inner test size automatically with that fold's own —
possibly expanding — train partition size.

**`validate_nested_plan`** is the independent, defense-in-depth leakage
check: every inner fold's positions are proven, via SET INTERSECTION
against `outer_fold.test_indices` (and, for symmetry,
`outer_fold.validation_indices`), never to touch the outer fold's
reserved rows — checked by direct position membership, never inferred
from timestamp ordering alone, so a bug that let inner positions leak past
the outer-train boundary while preserving chronological order would still
be caught. Also checks chronology, gaps, and cross-inner-fold validation
overlap.

## Feature universe and feature selection

`FeatureUniverse.from_experiment_spec` derives the candidate feature set
directly from the parent `ExperimentSpec.feature_binding.feature_names` —
already guaranteed free of label/timestamp/split/audit columns by
Milestone 3/4A construction; no new column-discovery logic was invented.

All 6 required strategies (`FeatureSelectionStrategy`):

- **`NONE`** — the full universe, unconditionally.
- **`VARIANCE_FILTER`** — population variance threshold, deterministic,
  never touches labels.
- **`CORRELATION_FILTER`** — greedy, order-preserving pairwise-correlation
  pruning; a constant feature's undefined (NaN) correlation is treated as
  "keep" rather than silently dropped; never touches labels (verified via
  `inspect.signature` in tests).
- **`UNIVARIATE`** — `sklearn.feature_selection.mutual_info_classif`/
  `mutual_info_regression`, seeded.
- **`MODEL_NATIVE_IMPORTANCE`** — restricted to models whose fitted object
  exposes a duck-typed `feature_importance()` (LightGBM, XGBoost,
  CatBoost only — `MODEL_NATIVE_IMPORTANCE_SUPPORTED_MODELS`).
- **`STABILITY_SELECTION`** — bootstrap-subsample repeats of a base
  strategy, bounded by `MAX_STABILITY_REPEATS = 200` (never unbounded
  compute).

Every strategy is fit ONLY on the data passed to it — for a trial, that is
one inner fold's inner-train partition; for the winning candidate's final
feature set, that is the complete outer-train partition (see "Outer-fold
finalization" below) — never on validation/test rows of any kind.
`FeatureSelectionResult` records exact training-row provenance
(`training_row_count`/`training_row_first_position`/
`training_row_last_position`/`training_row_fingerprint`) so a fitted
selection is independently auditable against the row range that produced
it.

## Search space and Optuna integration

`search_space.py` defines a closed, typed union of parameter kinds
(`IntegerParameter`, `FloatParameter`, `CategoricalParameter`,
`BooleanParameter`, `FixedParameter`) — no raw Optuna `suggest_*` lambda
is ever handed to this package directly. `SearchSpace` is ordered and
fingerprinted; model-specific defaults exist for LightGBM/XGBoost/CatBoost
plus a fixed baseline space. `FixedParameter` never touches Optuna's
suggestion machinery at all, so it consumes no sampler randomness (proven
via a dedicated test: two studies differing only in whether a
`FixedParameter` is present sample IDENTICAL values for every other real
parameter).

`SamplerKind.TPE` (default) and `SamplerKind.RANDOM` (the mandatory
no-pruning-equivalent control) are supported, both LOCAL/in-memory only —
`optuna.create_study` is never given a `storage=` URL; this platform's own
manifests/artifacts are the durable record, and the `optuna.Study` object
itself is reconstructed fresh every process start.

### Why manual `ask()`/`tell()`, never `study.optimize()`

`study.optimize(objective, n_trials=N)` is never called anywhere in this
package. Manual `ask()`/`tell()` is required for three independent
reasons: (1) a `TrialResult` artifact must be persisted BETWEEN sampling a
trial's parameters and reporting its outcome back to the sampler —
`optimize()`'s single-callback shape has no seam for that; (2) resume must
replay a specific, already-known sequence of historical trials before
asking for anything new; (3) a trial this platform marks
`INVALID`/`FAILED`/`PRUNED` must be told to Optuna via
`state=TrialState.FAIL`/`PRUNED` (never a fabricated numeric value), so it
never poisons future TPE proposals.

### The empirically verified deterministic-resume mechanism

This was flagged in the original specification as needing EMPIRICAL
verification, not assumption, and was tested directly against the
installed `optuna==4.9.0` before being relied upon anywhere in this
package.

**What does NOT work.** The commonly suggested approach — replay
completed trials via `study.add_trial(...)` — was tried first and found
to NOT reproduce identical future suggestions. `add_trial` bypasses
`ask()` entirely, so the sampler's internal RNG is never actually consumed
for the replayed trials, diverging from an uninterrupted run's RNG
trajectory (which DOES consume it once per `ask()`/`suggest_*()` call).

**What DOES work, and is what `study.replay_trial`/`rebuild_study_from_
history` implement.** Call `study.ask()` for EVERY historical trial too,
re-execute the EXACT same ordered sequence of `suggest_*()` calls (order
fixed entirely by `SearchSpace.parameters`' declared order), then
`study.tell()` with that trial's already-known final state/value. This
was verified to reproduce byte-identical subsequent suggestions across
both `TPESampler` and `RandomSampler`, across `int`/`float`
(including log-scale)/`categorical`/`boolean` parameters, and across
`COMPLETE`/`FAIL`/`PRUNED` historical trial states — proven both in
isolated Optuna experiments and end-to-end through the full
`OptimizationRunner` resume path (an interrupted-then-resumed run produces
IDENTICAL winning trials to an uninterrupted run: `tests/integration/
test_optimization_engine.py::TestResumeReproducesUninterruptedRun`), and
as a dedicated unit-level proof sweeping EVERY possible interruption point
across a 10-trial run for both sampler kinds
(`tests/unit/optimization/test_study.py::
TestDeterministicResumeAcrossSamplers`).

`replay_trial` additionally ASSERTS — never silently trusts — that each
replayed suggestion matches the originally-recorded `TrialSpec.
sampled_hyperparameters`; any mismatch (a search-space change, an Optuna
version change, or corrupted stored data) raises `OptimizationResumeError`
rather than silently continuing with a diverged sampler. "Do not
reconstruct a TPE study from only the current best trial" is satisfied
structurally: `rebuild_study_from_history` replays EVERY historical
trial, in strict ascending trial-number order, never merely the best one,
and rejects (via `OptimizationResumeError`) any gap or duplicate in the
supplied trial numbers.

## Pruning policy

The one required pruning rule, `study.evaluate_median_stopping`, is
**platform-owned** — it deliberately never routes through Optuna's own
`trial.report()`/`optuna.pruners` machinery, reading only this platform's
own persisted `InnerFoldTrialMetrics` instead. Once the current trial has
completed at least `PruningConfig.min_completed_inner_folds` inner folds,
it is pruned iff its own running primary-metric aggregate (mean of its
own non-`None` values so far) is worse — per the metric's authoritative
direction — than the MEDIAN of every OTHER trial's running aggregate at
the SAME inner-fold count. Returns `False` (never prunes) whenever
`PruningConfig.kind is PruningKind.NONE` (the mandatory no-pruning
control) or there is not yet enough information to decide. Structurally
cannot see outer-test performance — no parameter anywhere in its
signature resembles one.

## Early stopping policy

GBM early stopping (`EarlyStoppingConfig`) is evaluated against
inner-validation only — enforced structurally by `trial_executor`, the
only place `enabled`/`patience`/`validation_fraction` are ever translated
into a model's own already-shipped `early_stopping_rounds`/
`validation_fraction` hyperparameter keys (`GBM_MODEL_NAMES`; reused
directly from `ml.model_zoo`'s existing support, never reimplemented).

`EarlyStoppingConfig.final_round_policy` governs the number of boosting
rounds used when the winning candidate is refit on the complete
outer-train partition, where there is no inner-validation left to
early-stop against: `"median_best_iteration"` uses the rounded median of
every successful inner fold's own `best_iteration` (falling back to the
sampled/declared round count if no inner fold reports one — never
crashing); `"fixed"` always uses the sampled/declared round count. Both
are deterministic; neither ever reads outer-test performance.
`_extract_best_iteration` is a generic, duck-typed accessor
(`getattr(fitted, "best_iteration", None)`, normalizing LightGBM's own
`0`-sentinel to `None`) — never a per-model `if`/`elif` branch.

## Trial execution

`trial_executor.run_trial` is a (near-)pure function of its arguments: it
reads the reconstructed dataset timeline but never writes to it,
constructs a FRESH model/selector per inner fold (never reuses or mutates
one across folds or trials), and every artifact it writes is
content-addressed (writing identical bytes twice is a safe no-op). Two
calls with the same `TrialSpec` and the same underlying data always
produce byte-identical logical results (proven directly — the
`FeatureSelectionResult` ARTIFACTS themselves differ, since each embeds
its own real `fitted_at` audit timestamp by design, but the SELECTIONS
they record are identical).

**A bad hyperparameter combination never crashes the whole optimization.**
A raised exception from feature selection OR model `fit` for one inner
fold demotes ONLY that inner fold to "did not produce a value" (a broad
`except Exception` around each inner fold's own attempt, mirroring the
identical, already-established pattern `execution.executor.
MetricsFoldExecutor` uses for per-fold isolation). If a hyperparameter
combination is fundamentally broken, it fails identically on every inner
fold, and `objectives.aggregate_primary_metric`'s `min_successful_inner_
folds` gate naturally demotes the whole trial to `TrialStatus.INVALID`
with an accumulated reason.

**What is, and is not, persisted per inner fold.** Only the
`FeatureSelectionResult` is written as its own content-addressed artifact
per inner fold (needed for stability analysis and audit). Fitted models
and raw predictions from inner folds are deliberately NOT persisted — with
`outer_folds x trials x inner_folds` potentially numbering in the
thousands, persisting a full fitted-model blob at every one would be an
unbounded storage cost for artifacts nothing ever reloads. The ONE model
ever persisted in full is the winning candidate's outer-train refit.

`pruning_callback`, if given, is invoked after EVERY completed inner fold
with the metrics accumulated so far; returning `True` stops the trial
early (`TrialStatus.PRUNED`) — verified directly to skip all remaining
inner folds (a call-counting model factory proves no later inner fold's
`fit` is ever attempted after a prune decision).

## Deterministic candidate ranking

One fixed, versioned policy (`candidates.rank_trials`,
`RANKING_POLICY_VERSION`, folded into `OptimizationSpec`'s identity) — a
plain, transparent total order, never a pairwise statistical test between
trials, broken into these ordered tie-break criteria (each consulted only
when every earlier one ties):

1. Valid (`COMPLETED` with a defined primary-metric aggregate) before
   invalid/failed/pruned, unconditionally.
2. Primary metric aggregate, better-first per `ml.comparison.
   is_higher_better`'s authoritative direction — never re-decided.
3. More successful inner folds is better.
4. Lower dispersion (population standard deviation) of the primary metric
   across the same inner folds that produced the aggregate.
5. Fewer selected features on average across inner folds.
6. Lower `estimate_model_complexity` (a deliberately minimal
   `(boosting rounds) x (depth proxy)` estimate — reads
   `num_boost_round`/`iterations` and `max_depth`/`depth`/`num_leaves`;
   returns `None`, genuinely skipping this criterion, if no recognized
   rounds key is present — never fabricates a value).
7. Lower trial number — the final, always-available, fully deterministic
   tie-break.

Outer-test performance is never an input to any of the above — no
parameter anywhere in `rank_trials`'s signature can even represent it
(proven structurally via `inspect` in tests, not merely by convention).

## Outer-fold finalization

`outer_fold.finalize_outer_fold` — see "The central rule" at the top of
this document for why this is the one function permitted to read
outer-test row positions. Requires an already-selected, already-
`COMPLETED` `winning_trial` as an input parameter; there is no code path
here, or anywhere else in this package, that tries multiple candidates
against outer-test and picks the best one. Raises `ValueError` if the
supplied winner is not a valid COMPLETED candidate, or if its recorded
`outer_fold_index` does not match the fold being finalized.

**Final feature set: refit one more time on outer-train, never a vote
across inner folds.** A trial's feature selection was fit independently
inside each inner fold and may legitimately have selected a different set
in each. Rather than inventing a new "combine N inner selections"
heuristic, this module extends the SAME methodology one level further —
it reruns the winning trial's own `FeatureSelectionSpec` ONE more time,
now fit on the complete outer-train partition, deterministic, seeded from
its own dedicated branch (`outer_train_feature_selector_seed`). Proven
directly: the persisted `FeatureSelectionResult`'s own recorded training-
row provenance (`training_row_count`/`training_row_first_position`/
`training_row_last_position`) exactly matches `outer_fold.train_indices`,
never extending into `test_indices`.

**Final boosting-round policy: computed externally, never delegated back
to the model's own internal early stopping.** When early stopping is
enabled, the final refit does NOT pass `early_stopping_rounds`/
`validation_fraction` through to the model wrapper (which would carve its
own internal pseudo-validation tail out of outer-train and self-determine
a best iteration, silently overriding this milestone's own policy).
Instead, `trial_executor.resolve_final_round_count` computes the round
count from the winning trial's own already-completed inner-fold
`best_iteration` values, and that number is set DIRECTLY as the boosting-
round hyperparameter for a plain, non-early-stopping fit — deterministic,
inner-fold-derived, never influenced by outer-test in any way.

## Feature and hyperparameter stability analysis

`stability.summarize_feature_stability` reports, per feature: selection
frequency across every fitted `FeatureSelectionResult`, the frequency it
appeared specifically in a WINNING candidate's final set, and (when the
strategy records one) mean score/rank. `pairwise_jaccard_similarity`
across winning feature sets, with a `LOW stability` warning when the mean
falls below threshold. `summarize_hyperparameter_stability` reports, per
numeric parameter: mean/std and boundary-hit frequency (a parameter
repeatedly landing on its declared low/high bound across winning trials is
flagged — evidence the declared range may be too narrow); per categorical
parameter: choice frequencies; plus a trial-score-dispersion check across
outer folds, flagged `UNSTABLE` when dispersion is high. `flag_near_tied_
top_candidates` compares the top two ranked entries within an epsilon
fraction — never a statistical test, a simple, transparent proximity
check.

## State machine, resume, and idempotency

`OptimizationStage` (14 stages: `INITIALIZING`, `LOADING_EXPERIMENT`,
`BUILDING_OUTER_PLAN`, `RUNNING_OUTER_FOLD`, `BUILDING_INNER_PLAN`,
`RUNNING_TRIAL`, `SELECTING_CANDIDATE`, `REFITTING_WINNER`,
`EVALUATING_OUTER_TEST`, `STORING_RESULTS`, `COMPLETED`,
`RECOVERABLE_FAILURE`, `FAILED`, `CANCELLED`) — every terminal stage maps
to an EMPTY legal-transition set, so `OptimizationManifestStore.
transition()` can structurally never modify a manifest again once it
reaches one.

**Bug found and fixed during development (state-machine resume
reachability).** The original legal-transitions table did not allow
`RECOVERABLE_FAILURE` from `SELECTING_CANDIDATE`/`REFITTING_WINNER`/
`EVALUATING_OUTER_TEST`, and did not allow `RUNNING_OUTER_FOLD` to
self-loop. This was caught by end-to-end smoke testing (a simulated
mid-pipeline crash) BEFORE the formal test suite existed. Fixed by adding
a self-loop to `RUNNING_OUTER_FOLD` (mirroring `RUNNING_TRIAL`'s existing
self-loop) and adding `RECOVERABLE_FAILURE` as a legal target from those
three late stages — justified because everything from candidate selection
through outer-test evaluation is a PURE, deterministic function of
already-fixed inputs (the verified trial set, the outer fold's own row
positions); re-entering and redoing it from scratch after a crash
reproduces the identical result bit-for-bit, never a repeated "peek" at a
changing answer. `runner._execute_pipeline` normalizes any mid-outer-fold
crash stage through `RECOVERABLE_FAILURE -> RUNNING_OUTER_FOLD` before
re-entering the outer-fold loop.

`resume.build_trial_resume_plan` never trusts a manifest's claimed trial
references alone — it re-verifies each one (content hash, artifact
category, decoded `trial_number`/`outer_fold_index`/`optimization_id`
matching the key it was filed under) and scans ascending from trial 0,
stopping at the FIRST verification failure: Optuna's sequential sampler
cannot skip over unknown history, so a gap or corruption at trial N
discards trial N and everything after it, even if a later trial's OWN
bytes are individually intact. `resume.verify_completed_outer_folds` has
no such ordering dependency, since outer folds are independent of each
other.

`OptimizationRunner.run()` transparently auto-resumes a non-terminal
optimization and returns the existing manifest
(`was_idempotent_no_op=True`) for an already-`COMPLETED` one; `.resume()`
is the identical pipeline but raises `OptimizationResumeError` up front if
there is nothing to resume, or if the optimization already reached a
terminal stage.

**Why a failed outer fold (no valid trial to select) fails the whole
optimization**, unlike `execution.runner`'s per-fold-independent failure
tolerance: nested cross-validation's scientific claim is "the selected
feature set/hyperparameters generalize across ALL outer folds this
dataset was evaluated over" — silently completing with one outer fold
missing its outer-test evaluation entirely would misrepresent that claim.
`OptimizationRunner` treats this as a hard failure
(`OptimizationStage.FAILED`), never a partial success.

## Locking and concurrency

`OptimizationRunner` acquires a dedicated, run-DURATION lock
(`.optimization_run.lock`) via `ml.concurrency.experiment_lock`, distinct
from `OptimizationManifestStore`'s own brief, per-transition lock
(`.optimization.lock`) — the identical two-lock precedent Milestone 4B's
own audit established (a single shared lock file would self-deadlock,
since `historical.locking.DatasetLock` is not reentrant).
`OptimizationEventStore` reuses the SAME corrected, already-tested
`DatasetLock`/`experiment_lock` for its own append-only writes.

## Manifests and artifacts

`OptimizationManifest`/`OptimizationManifestStore` and
`OptimizationEventStore` mirror `execution.manifests.ExecutionManifest`/
`ExecutionManifestStore` and `ml.tracking.ExperimentEventStore` exactly —
a mutable current-state manifest plus an append-only JSON-Lines event
history, both rooted at `<ml_artifacts_root>/optimizations/<optimization_
id>/`, a SIBLING tree of `experiments/<experiment_id>/` (never nested
inside it, since one parent experiment can be the subject of MANY
independent optimizations).

`OptimizationEventStore` is a NEW class, not a reuse of `ml.tracking.
ExperimentEventStore` — that class hardcodes `"experiments"` as its
storage-root prefix and validates its key as an experiment-id-shaped
string, with no configurable prefix to reuse for a different root
(`"optimizations"`) or a different `EventType` enum. Rather than
generalizing that class with an unnecessary caller-supplied prefix, this
module mirrors its exact design (JSON Lines, gapless sequence numbers,
repair-on-access for a truncated final line) under its own name and root.

**One manifest, not two.** Unlike `execution.runner`'s two-manifest
pattern (a fine-grained `ExecutionManifest` layered on top of the
pre-existing, coarse `ExperimentManifest`), there is no pre-existing
coarse "optimization status" this milestone must preserve —
`OptimizationManifest`'s own `OptimizationStage` already serves both
roles. A deliberate simplification, documented here rather than silently
diverging from the two-manifest precedent without explanation.

Eight new, purely-additive `ArtifactCategory` members: `OPTIMIZATION_SPEC`,
`TRIAL_RESULT`, `FEATURE_SELECTION_RESULT`, `SEARCH_SUMMARY`,
`FEATURE_STABILITY`, `HYPERPARAMETER_STABILITY`, `OUTER_FOLD_SELECTION`,
`OPTIMIZATION_REPORT` — plus `ENVIRONMENT_SNAPSHOT`, a ninth member added
for parity with `TrialResult.environment_snapshot_reference`.

**Bug found and fixed during development (missing serializer merge).**
`OptimizationRunner.__init__` originally stored its `additional_
serializers` argument directly and passed it, unmerged, to `resolve_
serializer` — unlike `ExecutionRunner.__init__`, which merges the
built-in test-only `_SERIALIZER_REGISTRY` with whatever `additional_
serializers` a caller supplies BEFORE ever resolving anything
(`{**_SERIALIZER_REGISTRY, **dict(additional_serializers or {})}`). Since
every real CLI/production call site supplies a non-empty `additional_
serializers` (`ml.model_zoo.default_serializer_registry()`, which
contains only REAL models, by its own documented design — "the real-model
equivalent a caller merges into their own lookup"), `resolve_serializer`
never fell back to the built-in registry, silently making the test-only
model unusable through the optimization engine specifically (while
working fine through `ExecutionRunner`). Caught by a real, end-to-end CLI
test (`optimize` against `constant_test_model`) failing with `No
serializer registered for serializer_id='constant_test_model_json_v1'`.
Fixed by mirroring `ExecutionRunner`'s exact merge in `OptimizationRunner.
__init__`, now covered by `tests/unit/test_optimization_cli.py`'s full
`optimize` command test.

## Verification

`verification.verify_optimization` — an independent, read-only re-audit
of everything an optimization has ever recorded, mirroring `execution.
verification.verify_execution`'s exact philosophy, extended for this
milestone's own specific leakage-safety concerns. Checks: the
`OPTIMIZATION_SPEC` artifact and its recomputed identity match the
manifest it is filed under; the parent `ExperimentManifest`'s
dataset/split bindings and feature-universe fingerprint are still
consistent; every `TRIAL_RESULT` artifact (hash, category, self-identity,
sampled-hyperparameter validity against the declared search space); every
`FeatureSelectionResult` (via `feature_selection.
validate_feature_selection_result`); ranking reproducibility (recomputes
`rank_trials` over every verified trial and compares the winner against
the manifest's recorded one); every `OuterFoldResult` (hash, category,
self-identity, winner consistency, every dependent artifact reference
readable); summary references; manifest-stage-vs-progress consistency;
and a full event-sequence audit — including the literal enforcement of
the central rule: `CANDIDATE_SELECTED` must precede `OUTER_FOLD_FINALIZED`
for every outer fold with both events (`outer_test_evaluated_before_
candidate_selection`, `ValidationSeverity.CRITICAL`).

`ValidationReport.is_ready` is the single authoritative "did this pass"
gate; nothing here raises for an inconsistent-but-loadable optimization —
warnings are reported separately from fatal (CRITICAL/ERROR) consistency
errors.

## Reporting

`reporting.build_optimization_report_json`/`render_optimization_report_
markdown` — pure functions of already-loaded data, mirroring `ml.
reporting`'s identical "no I/O, no dashboard" design. Always includes a
fixed anti-overclaiming limitations block: the report does not label any
result profitable, and does not call a candidate statistically superior
unless `ml.comparison`'s own paired-significance gate actually says so.

## CLI usage

Nine new subcommands on the existing `python -m quant_platform.ml_cli`
tool (28 total commands on the one shared parser — never a separate
binary):

```bash
python -m quant_platform.ml_cli optimize --config opt_config.json
python -m quant_platform.ml_cli resume-optimization --config opt_config.json --optimization-id ID
python -m quant_platform.ml_cli inspect-optimization --config opt_config.json --optimization-id ID [--format json]
python -m quant_platform.ml_cli list-trials --config opt_config.json --optimization-id ID --outer-fold-index N
python -m quant_platform.ml_cli inspect-trial --config opt_config.json --optimization-id ID --outer-fold-index N --trial-number N
python -m quant_platform.ml_cli verify-optimization --config opt_config.json --optimization-id ID
python -m quant_platform.ml_cli compare-optimization-candidates --config opt_config.json --optimization-id ID --outer-fold-index N
python -m quant_platform.ml_cli feature-stability --config opt_config.json --optimization-id ID
python -m quant_platform.ml_cli hyperparameter-stability --config opt_config.json --optimization-id ID
```

`OptimizationConfig.experiment_config_path` points at the SAME
`MLExperimentConfig` JSON `prepare-experiment` already consumes — a human
never hand-copies a dataset id or split strategy into two separate config
files that could silently drift apart. `optimize` requires the referenced
parent experiment to already be `READY` (via a prior `prepare-experiment`
call). `optimize`/`resume-optimization` return 0 on full completion, 2 if
the optimization ends in a non-`COMPLETED` terminal stage, 1 for a
command-level failure. `verify-optimization` returns 0 if ready
(including when only WARNINGs are present), 2 if any CRITICAL/ERROR issue
is found.

## Testing

**Unit** (`tests/unit/optimization/`, 304 tests across 12 files): identity
determinism/sensitivity and metric-direction never-trusted
(`test_optimization_models.py`); search-space validation, fingerprinting,
and model-specific defaults, including the `CategoricalParameter` bug
regression (`test_search_space.py`); nested-split construction and the
dedicated `TestNoOuterTestLeakage` class of hand-crafted leakage attempts,
all caught
(`test_inner_splits.py`); all 6 feature-selection strategies' correctness
and determinism, including the length-mismatch-validation bug regression
(`test_feature_selection.py`); primary-metric aggregation (`test_
objectives.py`); the full 7-level ranking tie-break chain plus a
structural, `inspect`-based proof ranking never references outer-test
(`test_candidates.py`); sampler construction, deterministic resume across
every possible interruption point for both sampler kinds, replay-mismatch
detection, median-stopping pruning, and global-RNG-untouched (`test_
study.py`); the always-active capability gate, per-inner-fold failure
isolation, pruning early-exit, and determinism (`test_trial_executor.
py`); outer-fold finalization's leakage-safety provenance proof and the
GBM round/early-stopping-key-stripping policy (`test_outer_fold.py`);
Jaccard/stability summaries (`test_stability.py`); manifest/store
CRUD, legal-transition enforcement, and concurrent-duplicate-creation
prevention (`test_optimization_manifests.py`); and verified-artifact-based
resume planning, including truncation-at-first-failure for every
corruption mode (`test_optimization_resume.py`). File basenames that would
otherwise collide with a pre-existing sibling test module elsewhere in
`tests/unit/` (`test_manifests.py`, `test_models.py`, `test_resume.py`)
are disambiguated with an `optimization_` prefix, matching the identical
`test_execution_manifests.py`/`test_ml_manifests.py` precedent already
established for the same reason.

**Integration** (`tests/integration/test_optimization_engine.py`, 13
tests, using the REAL historical -> feature -> research-dataset ->
experiment-preparation -> `OptimizationRunner` pipeline, never mocked):
the full nested pipeline completing with outer-test evaluated exactly
once per fold; idempotent rerun; resume after a simulated crash
reproducing an uninterrupted run's exact winning trials; pruning +
stability selection + early stopping combined; Logistic Regression/Elastic
Net rejected by the REAL engine (not just the unit-level spec check); and
six `verify_optimization` perturbation tests, each corrupting exactly one
artifact/manifest field at a time to prove the specific detector fires.

**CLI** (`tests/unit/test_optimization_cli.py`, 15 tests): all 9 commands
exercised in-process via `main([...])`/`capsys`, against a real
(constant-model) prepared experiment, including error paths (unknown
trial/outer-fold index, resuming a completed or nonexistent optimization).

**Performance** (`tests/performance/test_optimization_throughput.py`, 6
benchmarks): inner-split construction, single-trial execution, a full
small nested-CV `OptimizationRunner.run`, Optuna suggestion throughput,
in-memory ranking, and artifact writes — conservative floors, ~6x-130x
below measured numbers on reference hardware, to catch a severe
accidental regression without CI flakiness.

## Adversarial audit findings

Real, material issues found via test-writing and end-to-end smoke
testing (before and during formal test-suite construction), each now
covered by a permanent regression test:

1. **`CategoricalParameter` unhandled `TypeError`**: the duplicate-choice
   check (`set(self.choices)`) ran BEFORE per-choice type validation, so
   an unhashable choice (e.g. a nested dict smuggled past the
   `JsonPrimitive` type hint at runtime) crashed with a raw, unhandled
   `TypeError: unhashable type: 'dict'` instead of a clean `ValueError`.
   Fixed by reordering validation: type/finiteness first, duplicate check
   second.
2. **Missing length-mismatch validation in 3 feature-selection
   strategies**: `select_univariate`/`select_model_native_importance`/
   `select_stability` had no explicit features/labels row-count check,
   producing confusing raw pandas `IndexError`s several stack frames deep
   on a genuine mismatch (caught via the author's OWN mismatched test
   fixtures, which is exactly what such a check exists to catch cleanly).
   Fixed by adding `_require_matching_length`, called at the top of all
   three.
3. **Illegal self-transition in `OptimizationRunner._run_locked`**: the
   original code created the manifest at `INITIALIZING` then immediately
   called `transition(new_stage=INITIALIZING, ...)` to attach the spec
   artifact reference — an illegal self-loop (`INITIALIZING` has none).
   Fixed by embedding the artifact reference directly into the
   manifest's initial constructor call, never via a later transition.
4. **State-machine resume reachability** (`RECOVERABLE_FAILURE` not legal
   from three late stages, `RUNNING_OUTER_FOLD` missing a self-loop) —
   see "State machine, resume, and idempotency" above for the full
   description and fix.
5. **Missing serializer-registry merge in `OptimizationRunner.__init__`**
   — see "Manifests and artifacts" above for the full description and
   fix; the specific bug that made the test-only model unusable through
   the real CLI's `optimize` command.
6. **Flaky (not incorrect) concurrent-manifest-creation test**: barriering
   two threads at the coarse `OptimizationManifestStore.create()` call
   left a timing gap where one thread could win its lock ENTIRELY
   (acquire, write, release) before the other even attempted acquisition,
   making the loser's observed exception type
   (`OptimizationStateError` vs. `ExperimentLockError`) nondeterministic
   — both are legitimate "lost the race" outcomes of the same safe,
   mutually-exclusive critical section, but a nondeterministic test is
   still a defect in the test. Fixed by adopting `tests/unit/historical/
   test_locking.py`'s own established technique: barrier the two threads
   at the precise `os.link` call `DatasetLock.acquire()` uses to publish
   its lock file, the actual atomic decision point.

Also verified via dedicated tests, none requiring a source fix: `1 ==
True` collide under Python's native equality/hashing in
`CategoricalParameter.choices` (matches JSON Schema's own enum-equality
convention) — a deliberate, accepted, documented limitation, not a bug.

## Remaining limitations (honest, as documented)

- Feature-attribution dashboards (SHAP/LIME), probability calibration,
  ensembles/stacking/blending, reinforcement learning, neural networks,
  distributed training, online learning, live trading/order execution,
  and recursive feature elimination are all explicitly out of scope, per
  the original specification's own DO NOT IMPLEMENT list.
- `MODEL_NATIVE_IMPORTANCE` feature selection is restricted to models
  whose fitted object exposes a duck-typed `feature_importance()`
  (LightGBM/XGBoost/CatBoost) — unsupported for every other registered
  model, including the baselines, by design.
- Scale-sensitive models (Logistic Regression, Elastic Net) cannot be
  optimized until a fold-local preprocessing-refitting framework exists
  (Option A, see "Preprocessing policy" above) — the identical, unchanged
  limitation Milestone 4C's execution engine already carries.
- Deterministic sampler resume is verified against `optuna==4.9.0`
  specifically; an Optuna version upgrade that changes internal RNG
  consumption order would be caught (via `replay_trial`'s own mismatch
  assertion, raising `OptimizationResumeError`) rather than silently
  producing a diverged sampler, but resuming an optimization started
  under a different Optuna version is not supported.
- Seed determinism does not extend to third-party non-deterministic
  native/GPU code (inherited, unchanged limitation from Milestone 4A).
- `DatasetLock`/`experiment_lock` remain local advisory locking, not
  distributed consensus (inherited, unchanged from Milestones 2, 4A, 4B).
- `flag_near_tied_top_candidates` and stability warnings are simple,
  transparent proximity/frequency checks — never a formal statistical
  test, by design (see "Do not use statistical tests between every trial
  unless justified and computationally bounded" in the original
  specification).
