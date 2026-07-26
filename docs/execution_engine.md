# Time-Safe Validation and Experiment Execution Engine (Milestone 4B)

## Why this milestone trains no real model

Exactly the same constraint as Milestone 4A, carried forward: LightGBM,
XGBoost, CatBoost, Random Forest, Logistic Regression, Elastic Net, neural
networks, feature selection, hyperparameter optimization, SHAP,
calibration, and ensembles are all explicitly out of scope. The ONE model
ever registered anywhere in this codebase remains `ml.testing.
ConstantTestModel` (predicts the training label mean/positive rate
regardless of input) -- explicitly test-only, documented as such.

**This milestone's `ExecutionRunner` DOES call real `fit`/`predict`/
`serialize`, and that is a deliberate, documented decision, not scope
creep.** See `execution.executor`'s module docstring for the full
reasoning: exercising `ml.interfaces`' real `TrainableModel.fit ->
FittedModel.predict` contract end-to-end -- using ONLY the harmless,
deterministic test model -- is exactly how Milestone 4A validated its own
artifact/serialization plumbing, and this milestone continues that
philosophy for the orchestration layer that actually runs a fold. "No
model training yet" (the milestone's own Section 5 heading) means no
REAL predictive algorithm ever trains; it does not mean the walk-forward
engine is reduced to an untested stub. `FoldResult.metrics` is,
correspondingly, an explicit, empty-by-default PLACEHOLDER -- no
performance score (RMSE, accuracy, or otherwise) is computed anywhere in
this milestone, since defining what a "correct" metric even means is
itself a modeling decision, out of scope alongside everything else in
the forbidden list above.

## Architecture

New top-level package `quant_platform.execution`, parallel to
`quant_platform.ml` (Milestone 4A) and depending on it (never the
reverse): `ExperimentSpec`, `ModelRegistry`, `MLArtifactStore`,
`ExperimentManifestStore`, `ExperimentEventStore`, and `ml.concurrency.
experiment_lock` are all reused directly, never duplicated. Also depends
on `features.manifests` (to read an already-built research dataset's
content) and `validation.walk_forward`/`features.splitting` (the actual
purge/embargo/expanding/rolling math, reused, never reimplemented).

Module layout (consolidated from the milestone's own suggested 17-file
list -- see each consolidation's reasoning inline below -- into 12 focused
modules, mirroring how Milestone 4A's own delivery deviated from its
suggested layout where a suggestion didn't fit the actual work):

- `state_machine.py` -- `ExecutionStage` enum + legal-transition table.
- `splitters.py` -- consolidates the suggested `folds.py`/`splitters.py`/
  `walk_forward.py`/`rolling.py`/`expanding.py`/`purging.py`/`embargo.py`
  into ONE module, since `features.splitting` already demonstrates that
  expanding/rolling/purge/embargo are facets of a single underlying
  engine (`validation.walk_forward.PurgedWalkForwardSplitter`), not seven
  independent algorithms.
- `execution_validation.py` -- fold-plan time-safety validation.
- `context.py` -- immutable `FoldExecutionContext`.
- `results.py` -- `FoldResult`, `AggregatedExecutionResult`.
- `manifests.py` -- `ExecutionManifest`/`ExecutionManifestStore`.
- `executor.py` -- `FoldExecutor` protocol + `DeterministicFoldExecutor`
  (consolidates the suggested `execution.py`/`executor.py`).
- `timeline.py` -- per-fold time-bound `Timeline`.
- `resume.py` -- resume detection/planning.
- `runner.py` -- `ExecutionRunner`, the one orchestrator (consolidates the
  suggested `runner.py`/`scheduler.py` -- there is no actual multi-
  process/distributed scheduling concern in this milestone's single-
  process, sequential-fold design, so a separate scheduler module would
  hold no real logic). Also owns `extract_label_horizon_bars`/
  `assert_preprocessing_is_safe_for_execution` (post-approval audit fix)
  -- the two functions that bridge the bound `ResearchDatasetManifest`
  into this engine's own fold-plan construction.
- `verification.py` -- (post-approval audit fix) `verify_execution`,
  an independent cross-store consistency re-audit; see "Verify-execution"
  below.
- `reporting.py` -- execution/fold summary reports.
- `__init__.py` -- public re-export surface, mirroring `ml/__init__.py`'s
  exact convention, deliberately excluding `ml.testing.ConstantTestModel`
  for the identical reason 4A's own `__init__.py` does.

Dependency order (acyclic): `state_machine` -> `splitters` ->
`execution_validation` -> `context`/`results` -> `manifests` ->
`executor` -> `timeline` -> `resume` -> `runner` -> `reporting`.

## Execution lifecycle (Section 2)

Two INDEPENDENT manifests track one experiment's execution, at two
different levels of detail -- exactly how Milestone 4A already splits
"current state" (`ExperimentManifest`) from "append-only history"
(`ExperimentEventStore`) across two focused stores sharing one root:

- **`ml.manifests.ExperimentManifest.status`** (`ExperimentStatus`,
  Milestone 4A, UNMODIFIED): sees exactly two transitions from this
  milestone -- `READY -> RUNNING` once, at the very start of `.run()`,
  and `RUNNING -> COMPLETED`/`RUNNING -> FAILED` once, at the very end.
- **`execution.manifests.ExecutionManifest.stage`** (`ExecutionStage`,
  new): the fine-grained progress this milestone actually adds --
  `INITIALIZING -> LOADING_DATASET -> BUILDING_SPLITS -> RUNNING_FOLD ->
  STORING_RESULTS -> COMPLETED`, with `FAILED`/`CANCELLED`/
  `RECOVERABLE_FAILURE` reachable from most non-terminal stages. Every
  terminal stage maps to an EMPTY legal-transition set (`state_machine.
  _LEGAL_TRANSITIONS`), so `ExecutionManifestStore.transition()` can
  structurally never modify a manifest again once it reaches one --
  exactly the same structural guarantee `ml.models.is_legal_transition`
  already gives `ExperimentStatus`.

`RUNNING_FOLD` and `STORING_RESULTS` alternate once per fold (`RUNNING_FOLD
-> STORING_RESULTS -> RUNNING_FOLD -> ... -> STORING_RESULTS ->
COMPLETED`) rather than appearing exactly once each -- the milestone's own
lifecycle diagram is illustrative of the overall shape, not a claim every
stage is visited only once. **No stage may transition to itself** (there
are no `X -> X` entries anywhere in the table); recording a resume
attempt without moving the stage forward is `ExecutionManifestStore.
bump_resume_count`, which deliberately bypasses the transition-legality
check rather than requiring meaningless self-loops for every stage (see
"Bug found" below).

`RECOVERABLE_FAILURE` vs. `FAILED`: a fold-level exception (bad data/model
input for THAT specific fold) is recorded as `FoldResult(status=FAILED)`
-- the loop continues to the remaining folds, and the OVERALL execution
ends `FAILED` once every fold has been attempted, never silently reported
as a success. An `ExperimentLockError` raised while WRITING an artifact
(the one clearly-recognizable "transient, worth retrying" condition
available via `ml.concurrency.experiment_lock`) stops the fold loop
immediately and transitions to `RECOVERABLE_FAILURE`, re-raising the
original error -- a later `resume()` call picks up exactly where this
left off. A true process kill (no code runs at all) is handled
differently still: the manifest simply stays at whatever stage it was
last durably written to, and `resume()`'s fold-by-fold re-verification
(see "Resume support" below) figures out what actually happened.

## Splitters (Section 3) and purge/embargo (Section 4)

All of Expanding Window, Rolling Window, Walk Forward, Grouped Walk
Forward, Blocked Time Split, Purged Split, and Embargo are provided --
every one of them a thin, explicitly-purposed wrapper around
`validation.walk_forward.PurgedWalkForwardSplitter`, never a
reimplementation of its gap arithmetic:

- `generate_expanding_folds`/`generate_rolling_folds` -- direct wrappers
  (`max_train_size=None`/set, respectively).
- `generate_blocked_time_folds` -- `test_size = len(timeline) // n_blocks`,
  `n_splits = n_blocks - 1`, delegated straight to the expanding
  generator; a convenience constructor, not a new algorithm.
- `generate_grouped_walk_forward_folds` -- an independent expanding
  walk-forward split generated WITHIN each distinct value of a group
  column (e.g. a per-instrument key in a hypothetical multi-symbol
  dataset -- no such dataset exists yet in this single-symbol-focused
  codebase, so this is exercised only via synthetic multi-group test
  data, honestly documented as forward-looking), fold indices mapped back
  to row POSITIONS in the full timeline (proven, via a dedicated
  interleaved-groups test, never to cross-contaminate positions between
  groups) and renumbered sequentially across all groups.
- `PurgeSpec`/`EmbargoSpec` -- support `bars` (int), `timedelta`
  (`pandas.Timedelta`, converted via `Timeframe.duration`, always
  rounding UP so a partial bar is never under-purged), or `calendar_days`
  (a PLAIN calendar-day count, `pandas.Timedelta(days=N)` -- **not**
  trading-session-aware; weekends/holidays are not excluded, a documented
  limitation rather than an invented trading-calendar subsystem).

**Every fold optionally carves a VALIDATION slice off the chronological
tail of its train segment** (`validation_fraction`), separated from the
now-shorter train segment by the SAME `purge_bars` gap already used at
the train/test boundary (documented as a deliberate simplification: the
validation/test gap gets the FULL `purge_bars + embargo_bars`, since
validation's tail sits exactly where the original train/test gap began;
the train/validation gap uses `purge_bars` alone, not embargo, since
correctness there only requires no direct label-window overlap -- a
best-effort convenience, not a rigorous CV boundary like train/test is).

`Fold`/`FoldPlan` (this milestone's own execution-transient types, never
persisted directly) inherit the identical "raw `numpy.ndarray` field
makes `==` comparison unsafe" landmine `features.splitting.DatasetSplit`
already has -- documented, not worked around, consistent with that
precedent; tests compare via explicit field access, never bare `==`.

## Label-information purge (post-approval audit fix)

**The bug.** Before this fix, the split engine derived its required
train/test gap ONLY from user-declared `SplitBinding.params`
(`purge_bars`/`embargo_bars`); the fold validator likewise checked only
`purge_bars + embargo_bars`. Neither ever consulted the bound dataset's
REAL label horizon (`ResearchDatasetManifest.label_definition.
horizon_bars`). This permitted a scientifically invalid configuration --
e.g. `horizon_bars=12, purge_bars=0, embargo_bars=0` -- where a training
sample near the train/test boundary has a target computed from prices
inside the validation or test period: genuine information leakage,
silently accepted.

**Exact label target information interval, and the off-by-one proof.**
Every label kind `features.labels` implements resolves using price data
through row `i + horizon_bars` INCLUSIVE, never beyond:
`compute_future_return`/`compute_future_log_return`/
`compute_binary_direction`/`compute_vol_adjusted_return` all call
`close.shift(-horizon_bars)`; `build_triple_barrier_labels` loops
`offset in range(1, horizon_bars + 1)`. So row `i`'s label target
information interval is exactly `[i+1, i+horizon_bars]` -- never
`i+horizon_bars+1` or beyond, for any label kind this codebase
implements.

Let `T` be the last kept training row position and `S` the first
validation/test row position (`S > T`). Row `T`'s label depends on data
through row `T + horizon_bars`. No leak requires that row stay strictly
BEFORE `S`:

    T + horizon_bars < S
    S - T > horizon_bars
    (S - T - 1) >= horizon_bars                    [integers]

This codebase's own "gap" convention is exactly `S - T - 1` (the count of
SKIPPED rows between the last train row and the first validation/test
row -- see `execution_validation._validate_purge_embargo` and
`validation.walk_forward.PurgedWalkForwardSplitter`'s own
`train_end = test_start - gap` construction). Substituting:
`gap >= horizon_bars`. **The required minimum is therefore exactly
`horizon_bars`, not `horizon_bars - 1`** -- a gap of `horizon_bars - 1`
leaves row `T`'s label depending on row `T + horizon_bars = S`, the
first validation/test row itself: a genuine leak. `execution.splitters.
required_label_purge_bars_for` implements and documents this proof
in full, and is an IDENTITY function today (kept as its own named
function, not inlined, so a hypothetical future label kind with
different forward-window semantics has one obvious place to change the
formula). `tests/unit/execution/test_splitters.py::
TestRequiredLabelPurgeBarsFor` and `tests/unit/execution/
test_execution_validation.py::TestLabelHorizonPurgeCheck` prove both
edges (one bar under rejected, exactly at accepted) directly.

**Declared vs. required vs. effective purge, and the chosen policy.**
Three distinct quantities, all recorded:

- `declared_purge_bars` -- `FoldPlan.purge_bars`, from `SplitBinding.
  params` (identity-relevant, unchanged by this fix).
- `required_label_purge_bars` -- the dataset-manifest-derived FACT
  (`required_label_purge_bars_for(label_horizon_bars)`).
- `effective_purge_bars` -- the purge actually in force.

Two policies were considered: **(A)** silently widen,
`effective_purge_bars = max(declared, required)`; **(B)** REJECT unless
`declared >= required`. **(B) was chosen.** Silently widening would let
two runs sharing the identical `ExperimentSpec` identity payload (since
`split_binding.params` is unchanged) execute materially DIFFERENT split
semantics whenever `required_label_purge_bars` exceeded the declared
value -- the identity payload would no longer describe the execution
actually run. Rejection keeps "same identity, same semantics" exact.
Consequently, **`effective_purge_bars` always equals `declared_purge_bars`
in this codebase**, for every execution that reaches `RUNNING_FOLD` --
`effective_purge_bars` is still recorded as its own explicitly-named
field (not left for a reader to re-derive) so the manifest is
self-documenting without requiring a reader to already know the policy.

**Embargo is not a substitute.** `execution_validation.
_validate_label_horizon_purge` compares `declared_purge_bars` ALONE
against `required_label_purge_bars` -- `embargo_bars` never enters this
comparison. An arbitrarily large `embargo_bars` cannot mask an
insufficient `purge_bars`; proven end-to-end in
`test_runner.py::TestLabelHorizonPurgeEnforcement::
test_large_embargo_cannot_mask_insufficient_purge`.

**Train/validation/test boundaries.** `_carve_validation` reuses
`purge_bars` for the train->validation gap and `purge_bars + embargo_bars`
for the (validation or train)->test gap -- both PRE-EXISTING arithmetic,
unchanged by this fix. Once `_validate_label_horizon_purge` guarantees
`purge_bars >= required_label_purge_bars`, and `_validate_purge_embargo`
guarantees the ACTUAL constructed gap is `>= purge_bars`, all THREE
boundaries are transitively protected without further arithmetic
changes:

- **train -> validation**: actual gap `>= purge_bars >=
  required_label_purge_bars`. Proven end-to-end in
  `test_execution_validation.py::TestLabelHorizonPurgeCheck::
  test_train_validation_boundary_is_label_horizon_safe_end_to_end`.
- **train -> test (no validation)**: identical reasoning; the standard
  case every purge test in this suite exercises.
- **validation -> test**: gets the FULL `purge_bars + embargo_bars`,
  which is `>= purge_bars >= required_label_purge_bars` -- at least as
  protected as the train/validation boundary. This milestone's
  `FoldResult`/`AggregatedExecutionResult` do not YET use validation
  labels for any evaluation, selection, or threshold decision (`metrics`
  is an explicit, empty-by-default placeholder -- see "Why this
  milestone trains no real model"); if a future milestone starts using
  validation predictions for model selection, this boundary's adequacy
  should be re-reviewed against that NEW use, not assumed unchanged.

This check operates on POSITIONS, not merely "no overlap": chronology is
independently enforced by `_validate_chronology` (strictly ascending,
train-before-test), so a sufficient gap in row-count terms is equivalent
to a sufficient gap in TIME for any chronologically-sorted timeline --
there is no separate "time gap" to check independently of the row gap.

**Dataset manifest integration; no new identity field.**
`execution.runner.extract_label_horizon_bars` derives `label_horizon_bars`
from the ALREADY-LOADED, immutable `ResearchDatasetManifest.
label_definition`, via the typed `features.labels.LabelDefinition.
from_json_dict` -- never a private JSON key read, never a duplicated
schema. `build_folds_from_split_binding` takes `label_horizon_bars` as a
new REQUIRED keyword argument, deliberately NOT read from
`SplitBinding.params` -- a `params["label_horizon_bars"]` entry, if
present, is simply inert (proven in `test_splitters.py::
TestBuildFoldsFromSplitBinding::
test_label_horizon_bars_param_in_split_binding_is_ignored` and
`test_runner.py::TestLabelHorizonPurgeEnforcement::
test_dataset_manifest_is_the_source_not_split_binding_params`). No new
`SplitBinding`/`ExperimentSpec` field was introduced for this: research
dataset manifests are immutable once saved, and `label_definition` is
already one of `features.manifests.compute_dataset_id`'s own hashed
inputs -- so `label_horizon_bars` is a pure function of `dataset_binding.
dataset_id`/`manifest_version`, BOTH already identity-relevant fields in
`ExperimentSpec.to_identity_payload()`. Two experiments bound to datasets
with different label horizons already get different `dataset_id`s, hence
different experiment identities, with no further change required --
proven directly in `test_label_horizon_purge.py::
TestLabelDefinitionIdentityImplications`.

**Persisted for audit.** `ExecutionManifest` gains six new fields, all
recorded together at the `BUILDING_SPLITS -> RUNNING_FOLD` transition:
`declared_purge_bars`, `required_label_purge_bars`,
`effective_purge_bars`, `embargo_bars`, `label_horizon_source` (always
`"research_dataset_manifest"`), `split_policy` (always
`"reject_if_declared_purge_below_required_label_horizon"`).

**Fail-closed extraction.** `extract_label_horizon_bars` converts a
malformed/missing `label_definition` (missing `name`/`kind`/
`horizon_bars`, an unknown `kind`, a non-positive `horizon_bars`, a
malformed `params` mapping) into ONE clear `FoldValidationError`, never a
raw `KeyError`/`ValueError`/`TypeError`/`FeatureError` -- proven in
`test_label_horizon_purge.py::TestExtractLabelHorizonBars` (7 fail-closed
cases against the REAL `LabelDefinition.from_json_dict`).

## Preprocessing safety (post-approval audit fix)

**The risk.** `execution.splitters.reconstruct_dataset_timeline`
reassembles a dataset's FULL timeline and this engine re-splits it using
ITS OWN, independent fold configuration -- which does not, and cannot,
align with whatever fold-group boundaries `features.dataset_builder.
ResearchDatasetBuilder` used when it originally fit any
`TransformPipeline` (`_fold_groups`, then one `TransformPipeline.fit()`
per group, fit ONLY on that group's own train indices). That per-group
fitting is CORRECT for Milestone 3's OWN usage, but says nothing about
whether the SAME baked-in feature values stay safe under a DIFFERENT
(this execution's) train/test boundary -- the fitted statistics may span
rows this execution's own folds consider "future".

**Code-evidence audit result.** Direct inspection of
`features/dataset_builder.py` confirms: `ResearchDatasetBuilder.build()`
computes RAW features via `FeatureEngine.compute(...)` BEFORE any
split/fold logic exists at all (split-independent); `fitted_preprocessing_
fingerprint = _combined_fold_fingerprint(...) if request.preprocessing
else None` -- the fingerprint is non-None if, and only if,
`request.preprocessing` (a `dict[str, TransformKind]`) was itself
non-empty, so the fingerprint and `preprocessing_definition` signals
always agree today. A grep across every currently-registered feature
module (`technical/price.py`, `temporal/calendar_features.py`,
`multi_timeframe.py`, `cross_asset/cross_asset.py`,
`macro/macro_features.py`) confirms every one of them uses the DEFAULT
`MissingPolicySpec()` (`MissingPolicyKind.PRESERVE_NULL`) -- none uses
`TRAINING_STATISTIC_FILL` (a separate, per-feature, fold-group-fitted
null-fill policy, independent of `TransformPipeline`). **Result: as of
this milestone, every dataset any registered feature module can produce,
when built with NO `preprocessing` transforms requested, contains only
causal/raw feature values -- proven by code reference (not merely
assumed) and by `tests/integration/test_execution_engine.py::
test_causal_only_dataset_explicitly_satisfies_preprocessing_safety_check`,
which asserts this against a dataset built through the REAL
`ResearchDatasetBuilder` pipeline.**

**Fail-closed policy (not a full preprocessing framework).** This
milestone does not attempt to refit/re-validate preprocessing per
execution fold (out of scope, same as feature selection/hyperparameter
optimization/calibration). Instead, `execution.runner.
assert_preprocessing_is_safe_for_execution` rejects an execution outright
whenever EITHER signal indicates fitted preprocessing:
`fitted_preprocessing_fingerprint is not None` OR
`preprocessing_definition` is non-empty -- checked independently, NOT
merely trusting the fingerprint alone (an explicit audit requirement),
so this check does not silently stop detecting the unsafe state if a
future Milestone 3 change ever let the two signals diverge. Proven end-
to-end against a dataset built through the REAL pipeline WITH a
`standard_scale` transform in `tests/integration/
test_execution_engine.py::
test_unsafe_globally_fitted_preprocessing_is_rejected_before_any_fold_runs`.

**Documented residual gap.** `MissingPolicyKind.TRAINING_STATISTIC_FILL`
is currently unused by every registered feature module, so it is
dormant, not actively exploited -- but the `ResearchDatasetManifest`
carries no typed signal recording whether any feature used it (adding
one would itself be "a full preprocessing framework", explicitly out of
scope). If a future feature module adopts `TRAINING_STATISTIC_FILL`,
THIS check will not detect it, since it inspects only
`preprocessing_definition`/`fitted_preprocessing_fingerprint`. Documented
here rather than silently ignored; closing it requires either a new
manifest signal (Milestone 3's decision) or extending this check once
that signal exists.

### Reassembling a dataset's full timeline

`execution.splitters.reconstruct_dataset_timeline` concatenates EVERY
split a research dataset's manifest recorded (train/validation/test, or
`fold_k_train`/`fold_k_test`/..., whatever it was built with) and sorts by
timestamp -- safe specifically because every split `features.
dataset_builder.ResearchDatasetBuilder` produces is non-overlapping and
internally sorted; the duplicate-timestamp and monotonicity checks this
function performs are a defense-in-depth VERIFICATION of that invariant,
not a workaround for a known violation of it. This is what lets ONE
stored dataset be walk-forward-split MANY different ways at execution
time (different `n_splits`/`test_size`/purge/embargo), independently of
however Milestone 3 originally split it at build time.

`ExperimentSpec.split_binding` (Milestone 4A's own, already
identity-relevant field -- `strategy` + JSON-primitive `params`) is
reused AS-IS for this configuration; this milestone introduces zero new
identity-relevant fields. `build_folds_from_split_binding` dispatches on
`strategy` (`"expanding_walk_forward"`, `"rolling_walk_forward"`,
`"blocked_time_split"`) and extracts `n_splits`/`test_size`/`purge_bars`/
`embargo_bars`/`max_train_size`/`n_blocks`/`validation_fraction` from
`params`.

## Walk-forward engine (Section 5)

`ExecutionRunner`'s pipeline, and how it maps onto `ExecutionStage`:

    load dataset        -> INITIALIZING -> LOADING_DATASET
    generate fold        -> LOADING_DATASET -> BUILDING_SPLITS
    validate fold        -> (still BUILDING_SPLITS; FAILED, non-resumable, if invalid)
    run fold             -> BUILDING_SPLITS -> RUNNING_FOLD (repeats)
    collect outputs       -> (within RUNNING_FOLD, per fold)
    store fold artifacts  -> RUNNING_FOLD -> STORING_RESULTS (repeats)
    aggregate            -> STORING_RESULTS -> COMPLETED (final)

A fold-plan validation failure (Section 13 checks; see below) is treated
as FATAL, never resumable -- it is deterministic given the same
`split_binding`/dataset, so retrying reproduces the identical failure.

**Ordering invariant** (mirrors the identical, previously-audited
Milestone 4A invariant for `ExperimentManifest`/events): every manifest
write/transition happens-before its corresponding event append.

## Execution context (Section 6)

`execution.context.FoldExecutionContext` -- one fresh, immutable object
per fold: `experiment_id`, `fold_index`, `split_id`, `dataset_content_id`
(the dataset fingerprint), a reference to the already-loaded
`ExperimentManifest`, a per-fold derived `seed`, an `EnvironmentSnapshot`,
the artifact/event stores, the artifacts root, and a `started_at`
timestamp. "Immutable" describes this object's OWN fields (a fresh
instance per fold, never reassigned); like `ml.registry.ModelDefinition`
holding a `ModelFactory`, it may still reference already-constructed,
longer-lived collaborator objects (the stores) that are themselves
stateful services, not pure data.

Per-fold seeds are derived from `ExperimentSpec.seed_configuration`
(Milestone 4A, unmodified) via `SeedDomain.CROSS_VALIDATION` combined with
the fold index in the domain string (`derive(f"cross_validation:{fold_index}")`)
-- reusing 4A's existing SHA-256-based derivation machinery, never a new
seeding scheme.

## Fold results and aggregation (Sections 7-8)

`execution.results.FoldResult` -- immutable, one per fold: index, time
bounds (train/validation/test), sizes, `duration_seconds`, `status`
(`COMPLETED`/`FAILED`), `artifact_references` (model + predictions, for a
completed fold), and `metrics` (an explicit, validated, empty-by-default
placeholder -- see "Why this milestone trains no real model" above).

`execution.results.AggregatedExecutionResult` -- the final, immutable
summary of one execution run: completed/failed fold indices, overall
status, timing, `resume_count`, and artifact references (the timeline).
**Deliberately excludes a self-reference to its own `EXECUTION_SUMMARY`
artifact** -- content addressing means an object's hash cannot be known
until its own bytes are fixed, so it cannot reference the artifact being
written FROM it (see "Bug found" below); the mutable `ExecutionManifest`
(not content-addressed) is where BOTH the timeline and the summary
artifact references are recorded together.

## Resume support (Section 9)

`execution.resume.verify_completed_folds` NEVER trusts
`ExecutionManifest.completed_fold_indices` alone -- it re-checks each
claim in THREE independent ways, any one of which demotes a fold to
`needs_rerun`: (1) `MLArtifactStore.read_artifact` re-verifies the
SHA-256 content hash on every read, never skipped; (2) the reference's
own `category` must actually be `FOLD_RESULT`; (3) **(post-approval audit
fix)** the artifact's bytes must decode as a `FoldResult` whose OWN
`fold_index` field matches the `fold_result_references` dict KEY it was
filed under -- a valid hash proves the bytes are intact, not that they
were filed under the correct key. A `fold_result_references` entry that
(a hand edit, a future caller's bug, or any other means) points a
genuine fold-3 result at key 5 is rejected here even though its hash
checks out cleanly. All three failure modes (missing/corrupted artifact,
wrong category, undecodable JSON, fold_index mismatch) are caught
explicitly and converted to "needs rerun" -- none ever escapes this
module as a raw `KeyError`/`ValueError`/`TypeError`/JSON decode
exception. Proven in `tests/unit/execution/test_resume.py::
TestVerifyCompletedFolds` and end-to-end (real dataset, real corruption,
real resume) in `tests/integration/test_execution_engine.py::
test_resume_replaces_a_corrupted_completed_fold_and_rebuilds_the_aggregate`
-- which also proves the OLD reference is genuinely replaced (never
carried forward: `ExecutionRunner._execute_pipeline` seeds its working
`fold_result_refs` dict ONLY from `resume_plan.verified_complete`, so a
demoted fold is simply absent from that seed, not present-with-a-stale-
value), completed/failed sets stay disjoint, and the final aggregate is
rebuilt from the CURRENTLY verified fold results. `build_resume_plan`
combines this with an explicit, narrowly-scoped notion of "force":
**this milestone does not support restarting an already-terminal
execution from scratch in place** (that would require deciding how to
archive/version the prior terminal record -- a real design question left
to a future milestone); `force_rerun_folds` instead lets a caller
resuming a non-terminal execution name SPECIFIC fold indices to rerun
even if verified-complete, satisfying "never rerun completed folds
unless explicitly forced" at a genuinely useful, honestly-bounded scope.

`ExecutionRunner.run()` transparently auto-resumes a non-terminal
execution and returns the existing aggregate (`was_idempotent_no_op=True`)
for an already-`COMPLETED` one; `.resume()` is the identical pipeline but
raises `ExecutionResumeError` up front if there is no prior execution
manifest at all, for callers that want to assert this is genuinely a
resume rather than a first attempt.

## Failure recovery (Section 10)

- **Interrupted execution** (process killed mid-run): the last durably
  written `ExecutionManifest` stage/fold-result-references are the
  source of truth; `resume()` re-verifies and continues.
- **Corrupted fold artifact**: caught by `verify_completed_folds`'s
  content-hash re-verification; treated as needing rerun.
- **Partial result write**: every manifest/artifact write in this
  milestone is atomic (temp-file-then-rename, reusing `ml.persistence.
  write_json_atomic`/`MLArtifactStore`'s existing race-tolerant pattern)
  -- a reader never observes a half-written file.
- **Invalid fold / missing artifact**: `execution_validation.
  validate_fold_plan` (fatal, pre-execution) and `MLArtifactStore.
  read_artifact` (`ArtifactNotFoundError`/`ArtifactCorruptionError`,
  post-hoc) respectively.
- **Lock timeout**: `ml.concurrency.experiment_lock` translates a
  contested lock into `ExperimentLockError` at the boundary, exactly as
  Milestone 4A's own audit established for `ml.manifests`/`ml.tracking`.

Recoverable (resumable): a per-fold data/model error (fold marked
FAILED, execution continues and ends FAILED, but IS resumable if e.g. the
underlying data issue gets fixed and folds are force-rerun); an
`ExperimentLockError` during an artifact write (`RECOVERABLE_FAILURE`,
directly resumable). Fatal (never resumable): fold-plan validation
failure (deterministic, would reproduce identically); an already-terminal
execution (`COMPLETED`/`FAILED`/`CANCELLED`).

## Execution locking (Section 11)

`ExecutionRunner` acquires a DEDICATED, run-DURATION lock
(`.execution_run.lock`) via `ml.concurrency.experiment_lock`, distinct
from `ExecutionManifestStore`'s own brief, per-transition lock
(`.execution.lock`) -- sharing one file between the two would
self-deadlock, since `historical.locking.DatasetLock` is not reentrant
(see "Bug found" below). A second concurrent `.run()`/`.resume()` call
for the SAME experiment_id fails fast with `ExperimentLockError` (never
silently corrupts, never blocks indefinitely) while the first is running.

## Artifacts (Section 12)

Reuses `ml.artifacts.MLArtifactStore` directly, with three new,
purely-additive `ArtifactCategory` members: `FOLD_RESULT` (one per
completed OR failed fold), `EXECUTION_SUMMARY` (the final aggregate),
`TIMELINE` (per-fold time bounds). "Fold manifest"/"execution manifest"
are the SEPARATE, mutable `ExecutionManifest` (not content-addressed,
since it changes throughout a run) -- see "Execution lifecycle" above.
"Execution log" is satisfied by the REUSED `ml.tracking.
ExperimentEventStore`, extended with four new, purely-additive
`EventType` members (`FOLD_STARTED`/`FOLD_COMPLETED`/`FOLD_FAILED`/
`EXECUTION_RESUMED`) rather than a second, redundant logging system.

## Validation (Section 13)

`execution.execution_validation.validate_fold_plan` reuses `ml.models.
ValidationIssue`/`ValidationReport`/`ValidationSeverity` directly (never
a parallel reporting type), checking: chronology (train strictly before
test, indices strictly ascending), purge/embargo gap sufficiency (exact
required-gap arithmetic, including the train/validation and
validation/test sub-gaps when a validation slice exists), no within-fold
or cross-fold-test overlap (expanding/grouped train sets legitimately
overlapping ACROSS folds is explicitly NOT flagged), no empty folds,
duplicate/out-of-order timestamps, and dataset-row-count/position
compatibility. Every check contributes a `ValidationIssue`, never a
raised exception, mirroring `ml.validation`'s identical guarantee.

## Verify-execution (post-approval audit fix)

`execution.verification.verify_execution` is an independent, read-only
re-audit of everything an execution has recorded across its FOUR
separate durable stores (`ExecutionManifest`, `ExperimentManifest`, the
artifact store, the event log), proving they still agree with each
other -- distinct from `execution.resume.verify_completed_folds`'s
narrower, forward-looking "which folds can I trust well enough to skip
re-running" question. Reuses `ml.models.ValidationIssue`/
`ValidationReport`/`ValidationSeverity` directly, exactly like
`execution_validation.validate_fold_plan` -- never raises for an
inconsistent-but-loadable execution (only a missing `ExecutionManifest`/
`ExperimentManifest` entirely propagates the underlying store's own
`ArtifactNotFoundError`, matching every other `ml_cli.py` load-based
command).

Checks performed: every recorded `FOLD_RESULT` reference exists, passes
content verification, decodes to a `FoldResult`, has the correct artifact
category, and its decoded `fold_index` matches the manifest key (plus a
bonus check: decoded `status` is consistent with which of
`completed_fold_indices`/`failed_fold_indices` it appears in); the
`EXECUTION_SUMMARY` aggregate exists, verifies, and its
completed/failed indices and `total_folds` match the execution manifest;
the `TIMELINE` artifact exists and verifies; the terminal
`ExecutionStage` is compatible with the aggregate's `overall_status`;
`ExperimentManifest.status` is compatible with `ExecutionManifest.stage`
(every non-terminal stage implies `RUNNING`; each terminal stage implies
the matching terminal status); the event sequence contains no impossible
transitions (at most one `RUN_STARTED`; `RUN_COMPLETED`/`RUN_FAILED`, if
present, is always the LAST event; every `FOLD_COMPLETED`/`FOLD_FAILED`
has an earlier `FOLD_STARTED` for the same fold index;
`EXECUTION_RESUMED` never precedes the first `RUN_STARTED`).

**Do not require false atomicity across files.** `ExecutionManifest`,
`ExperimentManifest`, artifacts, and the event log are four separate
files/directories, never written inside one cross-file transaction. This
codebase's own write ordering creates two DIFFERENT, known crash
windows, and this module handles each honestly rather than treating
either as corruption:

- **Per-fold**, the event (`FOLD_STARTED`/`FOLD_COMPLETED`/
  `FOLD_FAILED`) is appended BEFORE the manifest transition that folds
  it into `completed_fold_indices`/`failed_fold_indices` -- a crash here
  can leave the event log momentarily AHEAD of the manifest. Benign and
  self-healing (a future `resume()` re-verifies independently); not
  flagged as an error for a still-running execution.
- **At the very end**, the terminal `ExecutionManifest` transition (then
  the terminal `ExperimentManifest` transition) is written BEFORE the
  describing `RUN_COMPLETED`/`RUN_FAILED` event is appended -- a crash
  here can leave a terminal, authoritative manifest whose event log
  never got its closing entry. **This is the crash window the audit
  specifically asked to be surfaced.** Detected and reported as
  `terminal_manifest_missing_terminal_event`
  (`ValidationSeverity.WARNING` -- `ValidationReport.is_ready` stays
  `True`; the manifest remains authoritative, only the event log's
  history of the run is incomplete). Never silently dropped.

`ml_cli.py verify-execution` prints every issue as `[severity] code:
message` plus a final `is_ready: <bool>` line, returning 0 if ready
(including when only WARNINGs are present), 2 if any CRITICAL/ERROR
issue is found -- the same "0 unless not ready" convention
`validate-experiment` already uses.

## Execution reporting (Section 14)

`execution.reporting.build_execution_report_json`/
`render_execution_report_markdown` -- pure functions of already-loaded
data (an `ExecutionManifest` plus optionally an already-fetched
`AggregatedExecutionResult`/`Timeline`), mirroring `ml.reporting`'s
identical "no I/O, no dashboard" design exactly.

## CLI usage (Section 15)

Extends the existing `python -m quant_platform.ml_cli` tool (never a
separate binary) with six new subcommands, all using
`DeterministicFoldExecutor` only:

```bash
python -m quant_platform.ml_cli execute --config config.json --experiment-id ID
python -m quant_platform.ml_cli resume --config config.json --experiment-id ID [--force-rerun-fold N ...]
python -m quant_platform.ml_cli inspect-execution --config config.json --experiment-id ID [--format json]
python -m quant_platform.ml_cli inspect-fold --config config.json --experiment-id ID --fold-index N
python -m quant_platform.ml_cli list-folds --config config.json --experiment-id ID
python -m quant_platform.ml_cli verify-execution --config config.json --experiment-id ID
```

`execute`/`resume` return 0 on full completion, 2 if the execution ends
with one or more failed folds (never a traceback), 1 for a command-level
failure (bad config, contested lock, not-yet-`ready` experiment).

## Testing (Section 16)

Unit (`tests/unit/execution/`, 273 tests): splitters (incl. the
label-horizon-purge off-by-one proof and identity-agnostic `SplitBinding.
params` isolation), purge/embargo, execution validation (incl. the
dedicated label-horizon-purge policy gate), execution context, fold
results/aggregation, execution manifests (incl. concurrency, deterministic
lock-translation tests, and the six new purge/policy audit fields),
verify-execution (`test_verification.py`, new), label-horizon-purge/
preprocessing-safety runner-level building blocks and identity
implications (`test_label_horizon_purge.py`, new), the deterministic
executor, timeline, resume planning (incl. the fold_index/category
cross-check), the full `ExecutionRunner` lifecycle (happy path,
idempotency, fold-level failure isolation, recoverable failure + resume,
force-rerun, duplicate-execution locking, fold-plan validation failure,
not-yet-ready experiments, label-horizon-purge enforcement), reporting,
and the package's own public-surface regression test. Integration
(`tests/integration/test_execution_engine.py`, 8 tests, using the REAL
historical -> feature -> research-dataset -> experiment-preparation
pipeline, never mocked): full walk-forward execution, resume after a
simulated interruption, duplicate/parallel execution, a corrupted fold
artifact, label-horizon-purge rejection against the real dataset builder
(the audit's own horizon=12/purge=0/embargo=0 example), unsafe globally-
fitted preprocessing rejection, causal-only-dataset safety confirmation,
and resume replacing a corrupted completed fold end-to-end (old reference
replaced, aggregate rebuilt from currently-verified results, `verify-
execution` confirms the rebuilt state). Property-based
(`tests/unit/execution/test_property_based.py`, Hypothesis): chronology
and exact purge/embargo gap for arbitrary expanding/rolling
configurations (with `label_horizon_bars` held fixed at 0, since that
property is independent of the label-horizon-purge check -- see that
file's own comment), cross-fold non-overlap, purge-spec timedelta-to-bars
rounding never under-purges, `ExecutionStage` transition-legality
purity/determinism and terminal-stage-has-zero-targets for every enum
member, and resume-plan purity (identical inputs always produce an
identical plan).

`tests/unit/test_ml_cli.py` (24 tests) exercises the CLI end-to-end,
including `verify-execution`'s new, richer report output; its
`_build_research_dataset` fixture now builds a preprocessing-free
(causal-only) dataset by default -- a `preprocessing` argument lets a
test opt into building an UNSAFE dataset specifically to prove
rejection, rather than every CLI happy-path test incidentally exercising
an unsafe configuration.

## Performance (Section 17)

`tests/performance/test_execution_throughput.py`: split generation,
full walk-forward execution (distinct experiments, never idempotent),
idempotent resume, fold-result artifact writes, and in-memory aggregate
construction -- see that file's own docstring for exact measured numbers
and floors (conservative, ~10x-100x below measurement, to catch a severe
regression without CI flakiness).

## Adversarial audit findings (Section 19)

Three real, material bugs were found via direct end-to-end smoke testing
against a real dataset (BEFORE the formal test suite existed) and fixed,
each now covered by a permanent regression test:

1. **Lock self-deadlock**: the runner's own whole-execution lock and
   `ExecutionManifestStore`'s internal per-transition lock originally
   shared the same file name (`.execution.lock`) -- since
   `historical.locking.DatasetLock` is not reentrant, the SAME process
   acquiring it twice (outer run-lock, then an inner manifest-transition
   lock) deadlocked immediately. Fixed by renaming the outer lock to
   `.execution_run.lock`, distinct from the manifest store's own.
2. **Self-referencing artifact**: `AggregatedExecutionResult` was
   originally built with `artifact_references` including a reference to
   the `EXECUTION_SUMMARY` artifact being written FROM that very object
   -- impossible under content addressing (a hash cannot be known before
   its own bytes are fixed), causing the object returned from `.run()`
   to differ from the object later reloaded from disk. Fixed by never
   including a self-reference in the aggregate; both refs (timeline +
   summary) are recorded on the separate, non-content-addressed
   `ExecutionManifest` instead.
3. **Illegal self-transition on resume**: resuming used to call
   `transition(new_stage=current.stage, ...)` merely to bump
   `resume_count`, which is illegal for every stage (no stage transitions
   to itself). Fixed by adding `ExecutionManifestStore.bump_resume_count`,
   which updates the counter without going through transition-legality
   checking at all.

Also verified via dedicated tests, none requiring a fix: fold-plan
validation genuinely rejects duplicate/out-of-order timestamps and
overlapping folds (both within one fold and across folds' test sets,
while correctly NOT flagging expanding/grouped train overlap); an
interrupted manifest write (fault-injected via a patched `Path.replace`)
never exposes a truncated file; concurrent duplicate execution attempts
never corrupt the manifest (exactly one winner); a stale/contested
execution lock is translated to `ExperimentLockError` deterministically.

**Fourth finding, found via an independent post-approval source-inspection
audit (not this milestone's own testing), fixed before commit:**

4. **Label-information horizon leakage**: the split engine derived its
   required train/test gap ONLY from user-declared `purge_bars`/
   `embargo_bars`, never checking the bound dataset's real label horizon
   (`ResearchDatasetManifest.label_definition.horizon_bars`) -- permitting
   a configuration such as `horizon_bars=12, purge_bars=0, embargo_bars=0`
   where a training sample's label depends on prices inside the
   validation/test period. Fixed via a new, independent validation gate
   (`execution_validation._validate_label_horizon_purge`) comparing the
   declared purge against a manifest-derived required minimum, with
   REJECTION (not silent widening) as the chosen policy. See
   "Label-information purge" above for the full proof, policy rationale,
   and boundary-by-boundary reasoning. The SAME audit additionally
   identified two related, not-yet-exploited-but-real gaps, both closed
   in the same pass: (a) no fail-closed check existed for a dataset built
   with globally-fitted preprocessing, which this engine's independent
   re-splitting would make unsafe (see "Preprocessing safety" above);
   (b) `execution.resume.verify_completed_folds` verified an artifact's
   content HASH but never cross-checked its decoded `fold_index` against
   the manifest key it was filed under, and `ml_cli.py verify-execution`
   only checked artifact hashes, never cross-store consistency (see
   "Verify-execution" above).

## Remaining limitations (honest, as documented)

- `calendar_days` in `PurgeSpec`/`EmbargoSpec` is plain calendar time, not
  a trading-session-aware business-day calendar.
- `generate_grouped_walk_forward_folds` has no real multi-symbol dataset
  to exercise it against yet in this codebase (single-symbol-focused
  throughout Milestones 1-4B) -- proven correct against synthetic
  multi-group data only.
- Restarting an already-terminal execution from scratch in place is not
  supported (would require archival/versioning semantics, a future
  milestone's design question) -- `force_rerun_folds` covers the
  narrower, still-genuinely-useful "resume a non-terminal execution but
  re-verify specific folds anyway" case.
- The manifest/event-log ordering gap Milestone 4A's own correctness
  audit documented (a crash strictly between a manifest write and its
  describing event append can leave that one event missing) applies
  identically here -- no new gap introduced; `execution.verification.
  verify_execution` now DETECTS and honestly reports this specific
  window (`terminal_manifest_missing_terminal_event`, WARNING) rather
  than leaving it silently undetectable, but the underlying non-atomic
  write ordering itself is unchanged (closing it fully would require
  true cross-file transactions, a larger design change this milestone
  does not attempt).
- `MissingPolicyKind.TRAINING_STATISTIC_FILL` (a per-feature,
  fold-group-fitted null-fill policy, independent of `TransformPipeline`)
  is unused by every feature module registered as of this milestone, so
  `assert_preprocessing_is_safe_for_execution`'s fail-closed check
  (which inspects only `preprocessing_definition`/
  `fitted_preprocessing_fingerprint`) cannot detect it if a future
  feature module adopts it -- documented, not silently ignored; see
  "Preprocessing safety" above.
- `execution.verification`'s "no impossible event transitions" check is a
  concrete, USEFUL subset of everything that could theoretically be
  checked (at most one `RUN_STARTED`; a terminal run event is always
  last; every fold completion has a matching start; no resume precedes
  its run's start) -- not an exhaustive formal state-machine replay of
  the event log against `execution.state_machine`'s legal-transition
  table.
- `FoldResult.metrics` computes no real performance score anywhere in
  this milestone (see "Why this milestone trains no real model" above)
  -- reserved for a future milestone.
- Seed determinism does not extend to third-party non-deterministic
  native/GPU code (inherited, unchanged limitation from Milestone 4A).
- `DatasetLock`/`experiment_lock` remain local advisory locking, not
  distributed consensus (inherited, unchanged from Milestones 2 and 4A).
