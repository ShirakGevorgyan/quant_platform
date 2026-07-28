# Leakage-Safe Prediction Calibration, Decision Thresholding, Confidence, and Uncertainty Framework (Milestone 4E)

## Scientific purpose

Milestone 4C selects and fits a base predictive model. Milestone 4D
selects features and hyperparameters for it, leakage-safely. Neither
milestone asks whether that model's raw output is a *trustworthy
probability*, what *decision boundary* should be applied to it, how
*confident* any single prediction is, or how *uncertain* the platform is
about that confidence. This milestone answers those four questions,
strictly as **post-processing** of an already-selected, already-refit
base model's raw outputs — it never re-selects features, hyperparameters,
or the model itself, and it never backtests, simulates PnL, models
transaction costs, sizes positions, or claims anything about trading
profitability. See "What this framework does not claim" below.

## The central rule this entire milestone exists to enforce

**Calibrator fitting, calibrator selection, threshold optimization,
abstention-boundary selection, confidence-bucket definition, and
uncertainty-boundary definition all happen entirely on inner
out-of-fold (OOF) predictions from the outer-train partition. The outer
test partition is evaluation-only, touched exactly once, after every
post-processing decision is already frozen.**

```
OUTER TRAIN
    |-- inner time-safe walk-forward splits (reusing
    |   optimization.inner_splits directly, unmodified)
            |-- inner train  -> a FRESH base model, fit from scratch
            |-- inner validation -> inner out-of-fold (OOF) predictions

Calibrator candidates are fit, scored, and selected from inner OOF alone.
The decision threshold is optimized from inner OOF alone.
Confidence/uncertainty component definitions are frozen before outer-test.
Abstention boundaries are frozen before outer-test.

OUTER TEST
    never used for: calibrator fitting, calibrator selection, threshold
    optimization, abstention-threshold selection, uncertainty-boundary
    definition, confidence-bucket definition, fallback decisions, model
    ranking, feature selection, early stopping, reporting configuration.
```

After the post-processing policy is frozen using inner OOF data only:
(1) refit the base model on the **complete** outer-train partition; (2)
predict **once** on the untouched outer-test partition; (3) apply the
**frozen** calibrator, then the **frozen** threshold, then compute
confidence/uncertainty/abstention from the **frozen** policies; (4) read
outer-test **labels** for the first time, strictly to compute final
evaluation metrics that influence nothing computed in steps 1–3.

`calibration.runner.run_outer_fold_calibration` is the **only** function
in this package that ever reads an outer fold's test partition — not by
convention, but structurally:

1. `generate_inner_oof_predictions` is called with the outer `Fold`, but
   that function (and everything it calls, transitively — `optimization.
   inner_splits.build_inner_fold_plan`) has no code path that reads
   `.test_indices`/`.validation_indices` for anything other than a
   defense-in-depth overlap *position* check (never row *data*).
2. `fit_decision_policy` is called with only the `InnerOofPredictionSet`
   step 1 produced. Its signature has no `Fold` parameter at all —
   structurally, there is no Python expression through which outer-test
   bytes could reach it, checked directly in
   `tests/unit/calibration/test_leakage_adversarial.py` via
   `inspect.signature`.
3. The resulting calibrator/threshold/reliability policy is bound to a
   local variable and never reassigned ("frozen").
4. Only *now* does the function read `outer_fold.train_indices` (to
   refit) and, separately, `outer_fold.test_indices` — for **features
   only**, never the label column at this point.
5. The refit model's raw predictions are transformed through the
   already-frozen policy from step 3.
6. `timeline.iloc[outer_fold.test_indices][label_column]` — the **first
   and only** read of outer-test labels in the entire function — happens
   here, after every calibration/threshold/confidence/uncertainty/
   abstention decision already exists.

`tests/unit/calibration/test_leakage_adversarial.py` proves this with
active, fail-loud instrumentation rather than passive code reading: a
`_Landmine` object placed at every outer-test row's feature/label value
raises `AssertionError` on any comparison, arithmetic, or float
conversion, and the leakage-critical functions are proven to complete
successfully with landmines in place. A separate test mutates real
outer-test data (including to physically invalid label values) and
proves the resulting `InnerOofPredictionSet`/`FrozenDecisionPolicy` JSON
is byte-identical to a run against unmutated data.

## Architecture

New top-level package `quant_platform.calibration`, depending on `ml`,
`execution`, `features`, and `optimization` — a strict one-way
dependency; none of those packages import from `calibration`. Every
existing guarantee (experiment identity, dataset immutability,
label-horizon purge, fold chronology, `os.link`-based process-safe
locking, content-addressed artifacts, deterministic seed propagation) is
reused directly, never reimplemented or weakened. In particular, inner
(nested) split construction is `optimization.inner_splits.
build_inner_fold_plan`/`validate_nested_plan`, called directly — this
package does not implement a second splitting engine.

Module layout (16 files, dependency-ordered):

- `models.py` — the raw prediction contract (`RawPredictionSet`), every
  shared enum, and the `CalibrationStage` state machine.
- `specs.py` — `CalibrationSpec` (the content-addressed identity root),
  `ThresholdSpec`/`ConfidenceSpec`/`UncertaintySpec`/`AbstentionSpec`/
  `ProbabilityClippingPolicy`/`ReliabilityBinningSpec`/`CostMatrix`, and
  the seed-derivation branches (`calibration_inner_fold_seed`,
  `calibration_outer_refit_seed`).
- `methods.py` — Identity/Platt/Isotonic/Beta calibrators: unfit
  configuration objects with `.fit(...)`, separate frozen "fitted"
  dataclasses holding only explicit JSON-serializable parameters, never
  a wrapped scikit-learn estimator.
- `metrics.py` — log loss, Brier score, ECE, MCE, calibration
  slope/intercept, sharpness, resolution — reusing `ml.metrics.
  MetricComputationReport`'s "skip with a reason, never NaN" shape.
- `diagnostics.py` — reliability-bin construction (equal-width and
  equal-frequency), Wilson-interval per-bin confidence bounds.
- `thresholds.py` — the *one* authoritative `probability >= threshold`
  boundary function, all 8 threshold policies, per-inner-fold stability
  aggregation.
- `confidence.py` / `uncertainty.py` — transparent, documented component
  proxies, combined by a pure weighted-average function that never
  silently zero-fills a missing component.
- `abstention.py` — selective-prediction decisions and evaluation;
  coverage and accepted-sample accuracy are always reported together.
- `fitting.py` — inner OOF generation (a fresh base model per inner
  fold) and calibrator/threshold selection orchestration. The single
  most leakage-critical module.
- `manifests.py` — `CalibrationManifest`/`CalibrationManifestStore` and
  the append-only `CalibrationEventStore`.
- `reporting.py` — JSON/Markdown report builders, with the standard
  unsupported-claims disclaimers embedded in every report.
- `runner.py` — `CalibrationRunner`, the top-level orchestrator, and
  `run_outer_fold_calibration` — see "The central rule" above.
- `resume.py` — verified-artifact-based outer-fold resume planning.
- `verification.py` — `verify_calibration`, an independent cross-store
  re-audit including the recomputation proof (see below).
- `__init__.py` — public re-export surface (106 names), mirroring
  `optimization/__init__.py`'s exact convention.

## `CalibrationSpec` identity

Mirrors `optimization.models.compute_optimization_identity` exactly:
`CalibrationSpec.to_identity_payload()` is canonicalized and hashed
(`ml.fingerprints.fingerprint_json`) together with an
`identity_schema_version` marker, producing a stable `calibration_id`.
Two scientifically identical specs always produce the same id, regardless
of process, machine, or dict insertion order.

Identity-relevant fields include `source_experiment_id` (required),
`source_optimization_id` (optional — see "Two calibration sources"
below), `dataset_content_id`, `split_plan_fingerprint`,
`base_model_definition_identity`, every calibration/threshold/
confidence/uncertainty/abstention policy field, `bin_support_minimum_samples`
(default `20` — participates in identity because it materially changes
persisted confidence/uncertainty *outputs* for otherwise-identical
inputs, not merely an internal fitting detail), and `seed`. All are
independently re-derived and cross-checked at run time by
`calibration.runner.resolve_calibration_inputs` — an inconsistency
between a declared identity field and what is actually loaded fails
closed (`CalibrationDataError`), never silently proceeds with whichever
value happened to load.

### Two calibration sources

`CalibrationSpec` can post-process either:

1. **A baseline experiment's own model** (`source_optimization_id=None`)
   — hyperparameters and features come directly from the bound
   `ExperimentSpec`.
2. **An optimization's winning model, per outer fold**
   (`source_optimization_id` set) — hyperparameters and features come
   from that outer fold's own `OuterFoldResult.final_hyperparameters`/
   `.final_selected_features`, requiring the source optimization to have
   already reached `OptimizationStage.COMPLETED`.

## Fixed positive-class convention

Platform-wide (established by `ml.metrics`, reused here unchanged): the
positive class is always `1.0`. `CalibrationSpec.positive_class_label`
and `RawPredictionSet.class_labels[positive_class_index]` are both
**validated**, not merely assumed, to equal exactly `1.0`.

## Supported tasks

Binary classification is the fully-supported reference implementation.
Every enum and spec structurally represents multiclass classification,
but `CalibrationSpec.__post_init__` explicitly rejects anything outside
binary classification — no model in `ml.model_zoo` declares multiclass
support end-to-end yet, and this milestone fails closed on an
under-tested code path rather than silently attempting one (Section 2:
"fail closed for unsupported task/model combinations"). Regression
calibration is out of scope entirely and not represented at all.

## The raw prediction contract

`RawPredictionSet` is the one shape every calibration-adjacent module
consumes — never a raw `pandas.DataFrame` with feature/label/prediction
columns mixed together. This is a structural leakage guard, not a style
preference: a `RawPredictionSet` for inner-OOF (training-side) data and
one for outer-test data are constructed through different code paths
(`calibration.fitting` vs. `calibration.runner`'s outer-test evaluation
step), so calibration-fitting code has no `DataFrame` in scope that could
even *contain* outer-test labels.

`__post_init__` rejects: row-count mismatch across parallel arrays,
non-ascending `sample_positions`, duplicate sample identities,
non-monotonic timestamps, non-finite scores, out-of-range probabilities,
a class-label set not containing exactly the positive-class convention,
a `positive_class_index` out of bounds, `true_labels` outside the
declared class domain, and — critically — any overlap between
`fitted_on_rows` (which rows the producing model trained on) and
`sample_positions` (which rows it predicted). That last check is the
dataclass-level structural leakage guard requested by Section 6: "verify
that every calibration row was predicted by a model that did not train
on that row." `fitted_on_rows` is `None` only for the final outer-test
prediction set, where row-disjointness is instead guaranteed structurally
by `runner.py`'s separate-object design.

## Inner out-of-fold generation

`calibration.fitting.generate_inner_oof_predictions` reuses
`optimization.inner_splits.build_inner_fold_plan`/`validate_nested_plan`
directly — **not** reimplemented. For each inner fold: a fresh base
model is instantiated and fit on that inner fold's training rows only
(never a warm-started or partially-fit model); prediction happens only
on that inner fold's validation rows; `InnerOofPredictionSet.
__post_init__` cross-checks every inner fold's `sample_positions` against
`InnerFoldPlan.inner_folds[i].validation_indices` and every
`fitted_on_rows` against `.train_indices`, so a bug that mismatched them
would be caught at construction, not silently trusted.

`InnerOofPredictionSet.concatenated()` merges every inner fold's
predictions into one chronologically-ordered `RawPredictionSet` for
calibrator/threshold fitting — `fitted_on_rows=None` at this merged
level (mixed provenance across inner folds; the per-fold disjointness is
what matters and is already independently guaranteed per constituent
set).

Reuse of an optimization's own OOF is deliberately **not implemented** in
this milestone: Section 6 permits reuse only if identities, semantics,
features, hyperparameters, preprocessing, and splits all match *and*
artifacts independently re-verify — a nontrivial cross-package identity
match this milestone chose to defer (see "Known limitations" below).
Every calibration run regenerates its own inner OOF from scratch.

## Calibration methods

| Method | Parameters | Formula |
|---|---|---|
| Identity | none | calibrated = raw |
| Platt | `coefficient`, `intercept` | calibrated = sigmoid(coefficient · logit(raw) + intercept) |
| Isotonic | `x_thresholds`, `y_thresholds` (paired, monotone) | calibrated = linear-interpolate(raw; thresholds), clipped to [0, 1] |
| Beta | `log_p_coefficient`, `log_one_minus_p_coefficient`, `intercept` | calibrated = sigmoid(a·log(p) + b·log(1−p) + c) |

Beta calibration follows Kull, Silva Filho & Flach (2017), fit via a
2-feature logistic regression on `[log(p), log(1−p)]` — implemented
without any dependency beyond scikit-learn/numpy already in use
elsewhere in this platform.

**No executable objects are ever serialized.** `to_json_dict()` on every
fitted method returns only plain JSON-native values; `from_json_dict`
independently re-validates every persisted parameter (finiteness, shape,
monotonic ordering for isotonic thresholds) before trusting it — never a
pickled/joblib-dumped estimator.

**Input-domain validation is uniform across all four methods.**
Identity, Beta, and Isotonic all reject an out-of-`[0, 1]` input
outright; Platt does too when `input_representation=predict_proba` (its
only mode this milestone exercises — `DECISION_FUNCTION` support exists
structurally for a future unbounded-margin base model but is not
currently reachable from `calibration.fitting`, which always supplies
probabilities). This was **not** the original behavior: during
development, `PlattCalibrator`/`IsotonicCalibrator` were found (via
`tests/unit/calibration/test_methods.py`) to silently clamp an
out-of-range "probability" via `_safe_logit`'s internal clipping /
`np.interp`'s extrapolation, rather than rejecting it — inconsistent with
Identity/Beta and with this platform's "never silently repair" principle.
Both were fixed to reject explicitly.

## Calibrator selection

`calibration.fitting.select_calibrator` fits every candidate in
`CalibrationSpec.calibration_method_candidates` on the pooled inner-OOF
`(probabilities, labels)`, computes calibration metrics for each, and
selects one via a fixed, deterministic tie-break chain: (1) the declared
primary selection metric (direction-aware — log loss/Brier/ECE/MCE are
all "lower is better"); (2) simpler-method preference (`identity=0 <
platt=1 < beta=2 < isotonic=3`, fewer fitted parameters first); (3) a
secondary metric (log loss if the primary was Brier-family, Brier score
otherwise); (4) lexical method identifier. The chain itself is fixed in
code, not caller-configurable — `CalibrationTieBreakPolicy` has exactly
one legal value (`CANONICAL`), so a persisted spec still *names* its
tie-break policy explicitly without inventing a second configuration
surface for a chain that must never actually vary.

Identity is always a candidate (`CalibrationSpec.__post_init__` requires
it) and, since it has no fit-time class/sample requirements, is
virtually never unavailable — the guaranteed fallback. A candidate that
fails to fit (insufficient samples, missing class, fit divergence) or
that produces invalid output probabilities is recorded with an explicit
`FailedCandidateReason`, never silently dropped.

## Probability clipping

`ProbabilityClippingPolicy` (`enabled`, `epsilon`) is a caller-facing,
identity-bearing policy applied **after** the calibration transform,
governing the persisted *output* probability — kept structurally
separate from `_LOGIT_CLIP_EPS` (an internal-only `1e-12` used
defensively inside Platt/Beta fitting to keep `log`/logit finite at
exact `0`/`1` boundaries). The two are never conflated; changing one
never silently changes the other.

## Calibration metrics

Log loss, Brier score, ECE, MCE, calibration slope/intercept (via Cox
logistic regression on the logit of the calibrated probability),
sharpness (variance of predicted probabilities), and resolution
(Murphy-decomposition weighted variance of per-bin empirical rates) —
all hand-verified against closed-form formulas in
`tests/unit/calibration/test_calibration_metrics.py`, not merely "runs
without error." Calibration slope/intercept are legitimately undefined for
zero-variance predictions or single-class labels; `compute_calibration_
metrics` reuses `ml.metrics.MetricComputationReport`'s `skipped: dict[str,
str]` shape for this — an explicit reason, **never** a fabricated `NaN`.

Reliability bins (`diagnostics.compute_reliability_bins`) support both
equal-width and equal-frequency binning; equal-frequency's quantile edges
are deduplicated when repeated predicted probabilities collapse a
requested bin count, and `ReliabilityReport.actual_n_bins`/
`.collapsed_edges_note` make this explicit rather than silently returning
fewer bins or crashing on a zero-width bin. Each bin's empirical
positive-rate confidence interval uses a Wilson score interval (better
behavior near 0/1 than a naive normal approximation).

## Decision thresholds

`apply_threshold` is the **one** authoritative
`positive_prediction = probability >= threshold` boundary function in
this platform — `abstention.decide` explicitly reuses it rather than
re-implementing the comparison, and no other module is permitted to.

Eight threshold policies: `FIXED`, `BALANCED_ACCURACY`, `F1`,
`MATTHEWS_CORRCOEF`, `YOUDEN_J`, `MIN_PRECISION_MAX_RECALL`,
`MIN_RECALL_MAX_PRECISION`, `COST_SENSITIVE` (with an explicit, immutable
`CostMatrix`). All non-`FIXED` policies search a deterministic,
evenly-spaced grid (`candidate_grid_size` points across `[0, 1]`) —
never a stochastic or early-stopping search. A constraint-bearing policy
(`MIN_PRECISION_MAX_RECALL`/`MIN_RECALL_MAX_PRECISION`) that finds no
feasible candidate falls back to `infeasible_fallback_threshold`, with
`ThresholdReport.fallback_used`/`.fallback_reason` recording why —
verified in `test_thresholds.py` against a deliberately-infeasible
dataset (the single highest-probability sample mislabeled negative, so
`min_precision=0.999` is structurally unreachable).

Per-inner-fold threshold stability (mean/median/std/min/max/IQR,
objective dispersion, constraint satisfaction rate) is computed from
inner-fold-only thresholds — never influenced by, and never computed
against, outer-test data.

## Confidence and uncertainty

Confidence is **defined**, not assumed to be "the calibrated probability
itself." `calibration.confidence.compute_confidence` never reaches into a
`RawPredictionSet`/reliability report/model ensemble itself — it combines
an already-computed mapping of *named* components (`distance_from_
threshold`, `probability_extremity`, computed by small pure helpers in
this module; `calibration_bin_support`, `calibrator_disagreement`,
computed by `calibration.runner` with the additional context this module
deliberately does not depend on). With `component_weights` empty, this is
single-component ("heuristic") mode; otherwise a weighted-composite mode.
**A missing component is excluded from the average and renormalized over
what remains — never coerced to `0.0`** (verified directly:
`tests/unit/calibration/test_confidence_uncertainty_abstention.py`
constructs the case where zero-filling and renormalizing would produce
different scores, and asserts the renormalized one).

Uncertainty separates aleatoric/epistemic/calibration/decision framing
via five documented, transparent proxies (never a claim of exact
Bayesian posterior uncertainty):

| Proxy | What it measures |
|---|---|
| `entropy_component` | Spread of the predicted distribution itself |
| `margin_component` | Nearness to the decision threshold |
| `model_disagreement_component` | Disagreement across inner-fold models |
| `calibrator_disagreement_component` | Disagreement across candidate calibration methods |
| `bin_support_uncertainty_component` | How little reliability evidence backs this probability region (relative to `CalibrationSpec.bin_support_minimum_samples`, default `20`) |

**`model_disagreement` is structurally unavailable at outer-test
prediction time** in this milestone's design: inner-fold models are
transient (used only to generate inner OOF, never persisted — persisting
N inner models per outer fold was judged out of proportion to this
milestone's scope). When declared in `UncertaintySpec.components`, it is
passed as `None` for every outer-test prediction with an explicit
`component_unavailable:model_disagreement` reason code — never silently
zero-filled. This is a **documented limitation**, not a silent gap; see
"Known limitations" below. `calibrator_disagreement`, by contrast, *is*
available at outer-test time: every successfully-fit candidate's
`.transform()` (not just the selected one) is applied to the outer-test
raw probability and their spread measured.

Validated monotonicity properties (Section 17), each with a dedicated
test: entropy-based uncertainty is maximal at `p=0.5` and monotonically
decreases moving toward either extreme; margin-based uncertainty is the
exact complement of confidence's distance-from-threshold component, so a
larger threshold margin never produces *greater* margin uncertainty. This
milestone does **not** enforce a universal inverse confidence/uncertainty
relationship beyond what is mathematically guaranteed component-by-
component (Section 17's explicit instruction).

## Abstention / selective prediction

Five policies: `NONE`, `SYMMETRIC_BAND`, `MIN_CONFIDENCE`,
`MAX_UNCERTAINTY`, `CLASS_SPECIFIC_BOUNDARIES`. `abstention.decide`
returns a `(Decision, AbstentionReasonCode)` pair with reason codes
`NOT_ABSTAINED`, `BELOW_CONFIDENCE_FLOOR`, `INSIDE_UNCERTAINTY_BAND`,
`UNCERTAINTY_ABOVE_LIMIT`, `INSUFFICIENT_CALIBRATION_SUPPORT`,
`INVALID_PREDICTION`. **Invalid predictions (non-finite, out-of-range)
fail closed unconditionally** — there is no spec flag anywhere that
relaxes this to a silent `ABSTAIN` conversion; a value that should never
reach this function reaching it anyway is a deeper contract violation
this module chooses to surface, not hide.

`evaluate_selective_prediction` always reports coverage and
accepted-sample accuracy together on the same result object — this
platform never presents an improved accepted-sample number without the
coverage cost that produced it. `SelectivePredictionEvaluation.
__post_init__` structurally forbids a non-`None` `accuracy_on_accepted`/
`selective_risk` when `n_accepted == 0`, so a zero-accepted edge case can
never be silently reported as `0.0` or `1.0` accuracy.

## Artifact model and identities

Every artifact carries a schema version, category, and (where
applicable) `calibration_id`/`outer_fold_index`, verified independently
at deserialization time — never trusted from a filename alone. Seven
`ArtifactCategory` values are specific to this milestone:
`CALIBRATION_SPEC`, `INNER_OOF_PREDICTIONS`, `CALIBRATOR_CANDIDATE_
REPORT`, `THRESHOLD_REPORT`, `DECISION_POLICY`, `OUTER_FOLD_CALIBRATION_
RESULT`, `CALIBRATION_REPORT`.

`DECISION_POLICY` (the persisted `FrozenDecisionPolicy`) nests the
selected calibrator, the pooled threshold report, per-inner-fold
threshold stability, and reliability report(s) together — the frozen,
operational policy an outer-test prediction is actually transformed
through. `CALIBRATOR_CANDIDATE_REPORT`/`THRESHOLD_REPORT` separately
persist the full candidate audit trail (every candidate tried, every
failure reason) — a deliberate, minor duplication between "what was
tried" and "what was frozen," serving different audiences (a full audit
vs. the operational policy).

`OUTER_FOLD_CALIBRATION_RESULT` bundles references to the above plus
row-level outer-test predictions (raw probability, calibrated
probability, decision, abstention reason, confidence, uncertainty) and
final evaluation metrics **inline**, as plain JSON arrays — consistent
with `RawPredictionSet`'s own precedent, appropriate at this milestone's
bounded/research data scale.

## Determinism

Identical inputs (spec + source artifacts + seed) produce an identical
`calibration_id`, identical inner-OOF ordering, identical candidate
ranking, identical selected calibrator and its parameters, identical
threshold, identical confidence/uncertainty/abstention outputs, and
identical content-addressed artifact hashes — verified directly (fitting
twice on identical data produces byte-identical `to_json_dict()` output;
running the whole `CalibrationRunner` pipeline against a real dataset
produces a `calibration_id` matching an independently-recomputed
`compute_calibration_identity(spec)`). Every model-fitting seed is a pure
function of `SeedConfiguration.master_seed` via `ml.seeds.derive_seed`,
branched from the pre-existing `SeedDomain.CALIBRATION` root, never
Python's global `random` or NumPy's global RNG state.

## Resume and crash safety

`CalibrationStage`'s state machine (`CREATED` → `INNER_PREDICTIONS_READY`
→ `CALIBRATORS_EVALUATED` → `CALIBRATOR_SELECTED` → `THRESHOLD_SELECTED`
→ `POLICIES_FROZEN` → `OUTER_PREDICTIONS_READY` → `EVALUATED` →
`VERIFIED` → `COMPLETED`/`FAILED`) has **no `RECOVERABLE_FAILURE` stage**,
unlike `optimization.models.OptimizationStage`. This is a deliberate
divergence, not an oversight: `run_outer_fold_calibration` computes
Section 18's steps 2–14 as one atomic, pure function of already-fixed
inputs (unlike optimization's trial-*search* loop, which can fail one
trial and productively continue with others). A crash at any point during
or immediately after that call is always safe to resolve by redoing the
entire fold from scratch — so every stage strictly between
`INNER_PREDICTIONS_READY` and `EVALUATED` has a legal edge straight back
to `INNER_PREDICTIONS_READY`, mirroring `optimization.runner`'s own
`RECOVERABLE_FAILURE → RUNNING_OUTER_FOLD` restart edge, just without a
dedicated intermediate stage name.

Proven with real interrupted runs, not just unit-level state-machine
checks: `tests/integration/test_calibration_engine.py` monkeypatches
`run_outer_fold_calibration` to raise after 0, 2, or 4 artifact writes
(mid-write-sequence crash simulation), asserts the manifest is left
resumable (never falsely claiming completed work), and asserts `.resume
()` redoes the crashed fold completely and reaches an identical
`COMPLETED` terminal state, independently re-verified afterward.

`calibration.resume.verify_completed_calibration_outer_folds` re-checks
every outer fold the manifest *claims* is completed by content hash,
category, and decoded self-identity before trusting it — a
missing/corrupted/mismatched claim is treated as **not** completed and
redone, never silently accepted ("unverified completed work never
trusted").

**A genuine domain exception mid-run leaves an accurate, diagnosable
terminal record.** `CalibrationRunner._run_locked` wraps pipeline
execution and, on any `QuantPlatformError` other than `ExperimentLockError`
(lock contention/an aborted process — never a real "this calibration's
data or config is wrong" verdict), transitions the manifest to
`CalibrationStage.FAILED` with the exception message as `failure_summary`
and appends a `RUN_FAILED` event *before* re-raising — mirroring
`OptimizationRunner`'s per-decision-point `_fail(...)` calls, collapsed
into one boundary here since calibration's per-fold pipeline is a single
atomic sequence rather than a multi-decision search loop. A further
`run()`/`resume()` attempt against a `FAILED` calibration raises
`CalibrationResumeError` cleanly rather than hanging or silently
resurrecting it.

**Resuming under a different installed `scikit-learn` version fails
closed under `DeterminismPolicy.STRICT`** (the default) —
`CalibrationRunner._require_compatible_environment` compares the
`ENVIRONMENT_SNAPSHOT` artifact recorded at calibration-creation time
against the currently installed `scikit-learn` version (the library
backing every calibration method's `.fit()`) before resuming, exactly
mirroring `OptimizationRunner._require_compatible_optuna_version`'s
fail-closed pattern. `DeterminismPolicy.WARN` downgrades a mismatch to a
logged warning instead.

## Verification

`calibration.verification.verify_calibration` is an independent,
read-only re-audit, structured like `optimization.verification.
verify_optimization`: it re-checks the spec's identity, every outer-fold
result artifact and its dependent references, manifest/event-log
self-consistency, and — the check that makes this more than a
hash-consistency scan — **recomputes** the persisted calibrated
probabilities and threshold decisions from the persisted calibrator
parameters and persisted raw probabilities, and asserts they still match.

Every dependent reference (`inner_oof_reference`, `calibrator_selection_
reference`, `threshold_report_reference`, `decision_policy_reference`) is
fully **decoded** into its dataclass, not merely read as bytes and
discarded — decoding re-runs that type's own `__post_init__` structural
validation (e.g. `RawPredictionSet`'s `fitted_on_rows`/`sample_positions`
disjointness check), which is what actually catches a hash-valid-but-
tampered inner-OOF provenance claim. (`model_reference` remains a
bytes-only read: a serialized model has no generic, deserializer-free
decode this module can perform — see "What verification cannot
independently confirm" below.)
"Hash validity insufficient — semantically wrong hash-consistent
artifacts must be rejected" (Section 25) is proven directly in
`tests/integration/test_calibration_engine.py`: a byte-valid,
correctly-hashed, but semantically-tampered `FrozenDecisionPolicy` (a
Platt coefficient shifted by 5.0, or isotonic thresholds shifted upward)
is filed under a manifest reference exactly as a compromised process
might, and `verify_calibration` still reports a `CRITICAL`
`calibrated_probabilities_do_not_reproduce` issue.

**What verification cannot independently confirm**: it does not re-fit
the inner-fold models or the outer-train refit (that would require the
full training-side dataset and be far more expensive than an audit should
be), so it cannot prove the *raw* probabilities themselves came from a
correctly-trained model — only that everything downstream of those raw
probabilities is self-consistent and faithfully reproducible from what
was persisted.

## CLI

```
python -m quant_platform.ml_cli create-calibration-spec --config cal_config.json
python -m quant_platform.ml_cli run-calibration --config cal_config.json
python -m quant_platform.ml_cli resume-calibration --config cal_config.json --calibration-id ID
python -m quant_platform.ml_cli inspect-calibration --config cal_config.json --calibration-id ID [--format json]
python -m quant_platform.ml_cli report-calibration --config cal_config.json --calibration-id ID
python -m quant_platform.ml_cli inspect-calibration-fold --config cal_config.json --calibration-id ID --outer-fold-index N
python -m quant_platform.ml_cli verify-calibration --config cal_config.json --calibration-id ID
python -m quant_platform.ml_cli compare-calibration --config cal_config.json --calibration-id ID --baseline-calibration-id ID --metric accuracy
```

`create-calibration-spec` is a dry run (validates + prints the
deterministic `calibration_id`, writes nothing), mirroring
`validate-experiment`'s "preflight, no side effects" convention.
`resume-calibration` against an already-`COMPLETED` calibration is a safe
idempotent no-op (exit `0`) rather than an error — unlike
`resume-optimization`, which always raises for an already-terminal
optimization; only a calibration with no manifest at all, or one already
`FAILED`, raises `CalibrationResumeError` (exit `1`) on resume.
`report-calibration` is an alias for `inspect-calibration` — a completed
calibration's aggregate report *is* what `inspect-calibration` prints,
mirroring `optimization`'s identical choice not to duplicate content
under a second command name. Every command returns `0` on success,
non-zero on failure (`2` specifically when a run/verify did not reach a
successful terminal state), and prints an actionable message — never a
raw traceback for a domain error.

`config.calibration_schemas.CalibrationConfig` is the CLI's pydantic
config schema (`frozen=True, extra="forbid"`, mirroring `config.
optimization_schemas` exactly). Unlike `OptimizationConfig`, it does not
reference a fresh `MLExperimentConfig` file — it binds directly to an
already-prepared `source_experiment_id`, so it separately declares
`research_storage_root` explicitly (there is no experiment-config file to
read it from).

## Example: constructing and running a calibration

```python
from quant_platform.calibration.specs import CalibrationSpec
from quant_platform.calibration.models import (
    CalibrationMethodKind, CalibrationTieBreakPolicy, DeterminismPolicy,
    SelectionMetric, ThresholdPolicyKind, BinningStrategy, AbstentionPolicyKind,
)
from quant_platform.calibration.specs import (
    ThresholdSpec, ConfidenceSpec, UncertaintySpec, AbstentionSpec,
    ProbabilityClippingPolicy, ReliabilityBinningSpec,
)
from quant_platform.calibration.runner import CalibrationRunner
from quant_platform.optimization.inner_splits import InnerSplitConfig

spec = CalibrationSpec(
    schema_version=1, task=ObjectiveType.BINARY_CLASSIFICATION, positive_class_label=1.0,
    source_experiment_id=experiment_id, base_model_definition_identity=model_fingerprint,
    dataset_content_id=dataset_content_id, split_plan_fingerprint=split_fingerprint,
    calibration_method_candidates=(CalibrationMethodKind.IDENTITY, CalibrationMethodKind.PLATT, CalibrationMethodKind.ISOTONIC),
    calibration_selection_metric=SelectionMetric.LOG_LOSS,
    calibration_tie_break_policy=CalibrationTieBreakPolicy.CANONICAL,
    minimum_calibration_sample_count=30, minimum_samples_per_class=5,
    inner_oof_policy=InnerSplitConfig(strategy="expanding_walk_forward", n_splits=3, test_size_fraction=0.15, embargo_bars=1),
    threshold_spec=ThresholdSpec(policy=ThresholdPolicyKind.F1, candidate_grid_size=101),
    abstention_spec=AbstentionSpec(policy=AbstentionPolicyKind.NONE),
    confidence_spec=ConfidenceSpec(very_low_max=0.2, low_max=0.4, medium_max=0.6, high_max=0.8),
    uncertainty_spec=UncertaintySpec(components=("entropy", "margin", "bin_support"), aggregation="mean"),
    probability_clipping=ProbabilityClippingPolicy(enabled=True, epsilon=1e-6),
    reliability_binning_specs=(ReliabilityBinningSpec(strategy=BinningStrategy.EQUAL_WIDTH, n_bins=10),),
    seed=42, determinism_policy=DeterminismPolicy.STRICT,
)
runner = CalibrationRunner(
    ml_artifacts_root=root, model_registry=registry, research_manifest_store=rms,
    research_dataset_store=rds, experiment_manifest_store=experiment_manifest_store,
)
outcome = runner.run(spec)
assert outcome.manifest.stage.value == "completed"
```

## What this framework does not claim

- A calibrated probability is an estimate fit on **historical** inner
  out-of-fold data; it is **not** guaranteed to remain accurate under a
  future regime change.
- "Confidence" and "uncertainty" are transparent, **documented proxies**
  (distance from threshold, entropy, reliability-bin support, calibrator
  disagreement) — not a claim of exact Bayesian posterior uncertainty,
  and confidence is not the same thing as certainty.
- High confidence or low uncertainty for a prediction does **not** imply
  that acting on it would be profitable. This milestone performs **no**
  backtesting, PnL simulation, transaction-cost modeling, position
  sizing, or portfolio construction, and makes no claim about trading
  outcomes of any kind.
- Outer-fold evaluation metrics are computed once per fold, on a
  partition never used for any post-processing decision — but a small
  number of outer folds (typical for walk-forward evaluation) limits how
  strongly any single calibration run's outer-fold results generalize.
- The primary bounded end-to-end acceptance run
  (`tests/integration/test_calibration_engine.py::test_calibration_
  runner_end_to_end`) uses a test-only constant model
  (`ml.testing.ConstantTestModelFactory`) that predicts nothing
  meaningful — it is infrastructure evidence (the pipeline composes
  correctly end-to-end), never evidence of predictive or market edge.
  `tests/integration/test_calibration_real_model_acceptance.py`
  separately runs the full pipeline against real production models
  (`logistic_regression`, `lightgbm`) on synthetic data with genuine
  (not market-derived) injected predictive structure — this confirms the
  pipeline behaves correctly with a real, non-trivial probability
  distribution flowing through it, still never evidence of edge on real
  market data.

## Known limitations

- **`model_disagreement` uncertainty is structurally unavailable at
  outer-test time** — inner-fold models are transient and never
  persisted (see "Confidence and uncertainty" above). A future milestone
  could persist them if this proxy is judged worth the storage cost.
- **Inner-OOF reuse from a source optimization is not implemented** —
  every calibration run regenerates its own inner OOF, even when a
  source optimization already computed an equivalent one. Section 6
  permits reuse only under a nontrivial cross-package identity match this
  milestone deferred.
- **Multiclass classification and regression calibration are entirely
  unimplemented**, fail closed at `CalibrationSpec` construction —
  intentional scope, not a bug.
