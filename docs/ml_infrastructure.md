# ML Core Infrastructure and Artifact Foundation (Milestone 4A)

## Why this milestone trains no model

This milestone builds the scaffolding a real model will eventually run
inside: deterministic experiment identity, a content-addressed artifact
store, an append-only event log, pre-run validation, and a preparation
service that turns an `ExperimentSpec` into a validated, `ready`-or-
`failed` manifest. It deliberately implements **no** real predictive
algorithm -- no LightGBM/XGBoost/CatBoost/Random Forest/Logistic
Regression/Elastic Net/neural network, no hyperparameter optimization,
no feature selection, no calibration, no ensembling, no SHAP, no
production serving, no live trading, and no automatic profitability
claim. The only model registered anywhere in this milestone is
`ml.testing.ConstantTestModel` -- a model that always predicts the
training label's mean (regression) or positive rate (binary
classification) regardless of input, whose entire purpose is to exercise
the fit -> predict -> serialize -> deserialize path end to end without
a genuine ML library dependency. Every module and CLI command in this
document should be read with that boundary in mind: this is
infrastructure for preparing experiments, not for running them.

## Architecture

```
src/quant_platform/ml/
    persistence.py        deterministic JSON, atomic writes, path safety
    fingerprints.py        sha256 hex helpers, fingerprint_json
    seeds.py                SeedDomain, SeedConfiguration, derive_seed
    models.py               domain models: bindings, statuses, artifact
                             metadata, validation types
    environment.py          EnvironmentSnapshot + CodeRevisionBinding capture
    interfaces.py           ModelFactory/TrainableModel/FittedModel/
                             Predictor/Serializer Protocols
    testing.py              ConstantTestModel -- TEST-ONLY, not real ML
    registry.py             ModelRegistry (definitions + factories)
    experiment_spec.py      ExperimentSpec, canonical serialization
    experiment_identity.py  compute_experiment_identity, golden hashes
    artifacts.py            MLArtifactStore (content-addressed)
    manifests.py            ExperimentManifest + ExperimentManifestStore
    tracking.py             ExperimentEventStore (append-only JSONL)
    validation.py           validate_experiment_spec (preflight checks)
    experiment_manager.py   ExperimentPreparer (the one orchestrator)
    reporting.py            build_report_json / render_report_markdown
```

Dependency order (no cycles): `persistence` -> `fingerprints` -> `seeds`
-> `models` -> `environment` / `interfaces` -> `testing` -> `registry` ->
`experiment_spec` -> `experiment_identity` -> `manifests` / `validation`
-> `experiment_manager` -> `reporting`. `validation.py` and
`experiment_manager.py` depend on `features.manifests` (the Milestone 3
research dataset subsystem) through its public `ResearchDatasetManifest`/
`ResearchManifestStore` interface only -- nothing here re-implements or
duplicates research dataset storage.

`config/ml_schemas.py` (pydantic config) and `ml_cli.py` (the CLI
entrypoint) sit outside the `ml/` package proper, mirroring how
`config/feature_schemas.py` and `feature_cli.py` relate to
`features/`.

## Model contracts (Section 3)

`ml/interfaces.py` defines the shape any future real model must satisfy.
The central design choice: **`TrainableModel.fit(...)` returns a NEW
`FittedModel` object rather than mutating `self`.** This makes "predict
before fit must fail" a structural property, not a runtime flag check --
an unfit model has no `predict` attribute at all (`hasattr(unfit,
"predict")` is `False`); calling it raises `AttributeError` at the
object's actual shape. A future model MAY additionally track fitted
state internally and raise `ModelNotFittedError` defensively (both
patterns satisfy the same Protocols); `ml.testing.ConstantTestModel` uses
the structural pattern exclusively.

Other invalid operations this module makes explicit:
- `predict_proba` on a model whose `ModelCapabilities.
  supports_predict_proba` is `False` raises `UnsupportedObjectiveError`
  via the shared `ModelCapabilities.require_predict_proba` guard.
- **`isinstance(fitted, ProbabilisticPredictor)` is not itself a
  capability check.** `ProbabilisticPredictor` is a `@runtime_checkable`
  `Protocol`, which only verifies that `predict_proba`/`class_labels`
  exist as callable attributes -- not that they're semantically valid for
  this model's declared capabilities. `ml.testing.FittedConstantTestModel`
  always defines both methods, so the isinstance check is `True` even for
  a regression-only instance; the actual gate is `require_predict_proba`
  raising at call time. Any code that needs "does this model really
  support probability output" must ask `capabilities.supports_predict_proba`,
  never rely on `isinstance` alone.
- `ProbabilisticPredictor.class_labels` is the authoritative, ordered
  tuple describing what each `predict_proba` output column means --
  never an implicit alphabetical/insertion-order assumption.
- `FeatureSchema.validate_frame` enforces column completeness/order
  under an explicit `FeatureColumnPolicy` (`STRICT` rejects both missing
  AND extra columns; `IGNORE_EXTRA` rejects only missing columns) --
  never a silent reorder or silent subset selection with no named
  policy.
- `ModelDeserializer.deserialize`'s optional `expected_metadata` asserts
  the reconstructed model's metadata matches what the caller expects,
  raising `FeatureSchemaMismatchError` on mismatch.

## Model registry (Section 4)

`ModelRegistry` stores `ModelDefinition`s (name + version + description +
capabilities + factory + serializer id) -- never fitted instances.
Append-only registration; duplicate `(name, version)` rejected with
`DuplicateModelDefinitionError`; `list_definitions()` always returns a
deterministic `(name, version)`-sorted order. `ModelRegistry.fingerprint()`
is a pure function of every registered definition's own JSON-serializable
metadata (`ModelDefinition.to_json_dict()`, which explicitly excludes the
`factory` field) -- it is identical across two independently-constructed
registries with the same content, regardless of registration order,
Python object identity, process ID, memory addresses, or wall-clock
time, because none of those ever enter `to_json_dict()`.

No cycle detection: unlike `features.registry.FeatureRegistry` (where one
feature's `compute` can legitimately depend on another feature's output),
nothing in this milestone lets one `ModelDefinition` depend on another --
building dependency-graph machinery for a graph that structurally cannot
exist would be unjustified complexity.

## Experiment identity (Sections 5-6) -- the architectural centerpiece

`ExperimentSpec` (`experiment_spec.py`) is the complete, immutable binding
of an experiment to its exact scientific inputs: dataset binding, feature
binding (ordered names + versions + registry fingerprint), label binding,
split binding, preprocessing binding (including the fitted preprocessing
fingerprint), model name + version, hyperparameters, objective, seed
configuration, code revision binding, and `environment_requirements`.

**Identity-relevant vs. descriptive fields** is the central design
principle: `ExperimentSpec.to_identity_payload()` returns ONLY the
scientifically-relevant fields above; `to_json_dict()` returns that
payload PLUS `primary_metric`, `tags`, and `notes`. Changing notes, tags,
or the declared primary metric NEVER changes `to_identity_payload()`'s
output and therefore never changes the experiment ID -- a human editing
a description of an experiment after the fact does not create a new
experiment.

**`primary_metric` identity audit** (post-4A correctness review):
re-confirmed that `primary_metric` is read in exactly three places in
this package -- the spec's own non-empty check, `to_json_dict()`'s
round-trip, and `ml_cli.py`'s config passthrough -- and drives no
computation, comparison, selection, threshold, fold choice, or model/
hyperparameter decision anywhere. The decision to keep it
identity-exempt is therefore RETAINED, not changed. This is conditional,
not permanent: the moment a future milestone uses `primary_metric` to
select between models/hyperparameters/thresholds/folds, it must move into
`to_identity_payload()` at that point, with `IDENTITY_SCHEMA_VERSION`
bumped alongside so old and new IDs never silently collide -- see
`experiment_spec.py`'s module docstring for the full reasoning.

`experiment_identity.compute_experiment_identity(spec)` hashes
`to_identity_payload()` (plus an explicit `IDENTITY_SCHEMA_VERSION`, so
a future change to what counts as identity-relevant always changes every
ID rather than silently colliding old and new schemes) through
`persistence.canonical_json_bytes` (sorted keys, no non-finite floats,
never a raw `repr()`) and SHA-256. Consequences, all covered by golden
and property-based tests in `tests/unit/ml/test_experiment_identity.py`:

- The same scientifically-relevant spec always produces the same
  `experiment_id`, in any process, on any machine, forever.
- Dict/JSON key insertion order (feature_versions, hyperparameters,
  environment_requirements) never affects the ID.
- Feature *order* DOES affect the ID when it changes (a fixed-width
  feature vector's column order is semantically meaningful) -- this is
  intentionally different from dict-key order, which never is.
- Changing dataset content, feature versions/order, label definition,
  split definition, preprocessing fingerprint, model version,
  hyperparameters, objective, master seed, code revision, or an
  explicitly-declared `environment_requirements` entry all change the ID.
- Wall-clock timestamps, process IDs, memory addresses, local/temp file
  paths, and Python's randomized `hash()` never appear anywhere in the
  identity payload -- there is nothing to accidentally leak.

**Why `environment_requirements` is identity-relevant but
`EnvironmentSnapshot` is not** (Section 8): `EnvironmentSnapshot`
(captured Python/OS/package versions, CPU count) is always informational
-- two scientifically identical experiments prepared on different
machines get the SAME ID regardless of OS/CPU/package patch version.
`environment_requirements` is the deliberate escape hatch: a human
DECLARES a requirement (e.g. `{"numpy": ">=2.0"}`); it defaults to empty,
so it's a no-op unless explicitly used, and even then it's the same
declared string across machines unless a human deliberately writes
different declarations on different machines -- exactly the "unless
explicitly configured" case the milestone spec calls for.

## Random seed management (Section 7)

`SeedConfiguration` (`seeds.py`) holds exactly one field of substance:
`master_seed` (plus a schema version) -- deliberately no notes/
descriptive field, which is what makes "changing an unused descriptive
seed field must not be possible" true by construction: no such field
exists. `derive_seed(master_seed, domain)` derives a per-domain child
seed via SHA-256 of `f"{master_seed}:{domain_name}"`, truncated to 4
bytes -- never Python's randomized `hash()`, so derivation is identical
across processes regardless of `PYTHONHASHSEED` (verified in
`test_seeds.py` via a subprocess with `PYTHONHASHSEED=random`).
`SeedDomain` enumerates `GLOBAL, MODEL_INIT, DATA_OPERATIONS,
CROSS_VALIDATION, HYPERPARAMETER_SEARCH, FEATURE_SELECTION, CALIBRATION,
ENSEMBLE` -- most are unimplemented in this milestone but the domain
enum exists so future milestones need no seed-derivation changes.
`.random_for(domain)`/`.numpy_generator_for(domain)` return seeded
`random.Random`/`numpy.random.Generator` instances. Importing `ml.seeds`
does not itself mutate global `random`/`numpy.random` state (verified by
a subprocess test comparing pre/post-import draw sequences).

**Documented limitation**: seeding `random` and NumPy's `Generator`
deterministically does not guarantee bit-for-bit determinism from a
future third-party library with its own non-deterministic reduction
order (e.g. GPU-accelerated summation). This module seeds what it
controls and records, rather than hides, that boundary.

## Environment and code revision capture (Section 8)

`environment.py` has two independent halves:

1. `capture_environment_snapshot()` -> `EnvironmentSnapshot` --
   INFORMATIONAL ONLY. Python version, OS/architecture, tracked package
   versions (numpy, pandas, pydantic, pyarrow, quant-platform via
   `importlib.metadata`, `None` if not resolvable), CPU core count (no
   new third-party dependency was added purely to capture memory info;
   it is simply omitted).
2. `capture_code_revision_binding()` -> `CodeRevisionBinding` --
   IDENTITY-RELEVANT. Tries `git rev-parse HEAD` + `git status
   --porcelain` (broadened from `historical.code_revision`'s
   single-subpackage scope to hash the entire `quant_platform` source
   tree for the content-hash fallback, since an ML experiment's behavior
   can depend on any part of the codebase); falls back to a
   `content:<sha256>` marker when no git repository is found. `git` is
   invoked via `subprocess.run` with an explicit argument list, a 5s
   timeout, and `check=False` -- never `shell=True`, never an unsafe
   shell string. A failed/missing git command or absent `.git` directory
   is treated as "no git available," never raised.

## Content-addressed artifact storage (Section 9)

`MLArtifactStore` (`artifacts.py`) layout:

```
<ml_artifacts_root>/
    content/<hash[:2]>/<sha256>     raw artifact bytes, write-once
    metadata/<sha256>.json          {schema_version, content_hash,
                                      category, size_bytes, created_at}
```

`experiments/<experiment_id>/manifest.json` (handled by
`ExperimentManifestStore`, a sibling store over the same root) completes
the suggested layout. **Deliberate deviation: no `CURRENT` pointer.**
`CanonicalStore`/`ResearchDatasetStore` each manage ONE evolving named
entity with a meaningful "latest version"; this store holds many
unrelated, independently content-addressed blobs with no such entity --
a `CURRENT` pointer would have no referent. Every caller that needs an
artifact again already holds its `ArtifactReference`, recorded explicitly
in an `ExperimentManifest`.

Guarantees: SHA-256 content addressing; atomic temp-file-then-rename
writes; content re-verified on every read (never trusted from the
filename alone); duplicate writes of identical bytes are idempotent
no-ops; writing the SAME hash under a DIFFERENT declared `category` is
treated as corruption (`ArtifactCorruptionError`) -- "no silent overwrite"
applies to metadata, not just content; path traversal and Windows
symlink escapes are blocked via `persistence.assert_within_root`
(resolves and checks containment, following symlinks); **the generic
layer never deserializes anything beyond raw `bytes` -- no pickle, no
`eval`, no dynamic import.** A caller reconstructing a fitted model must
go through an explicitly-trusted `ModelDeserializer` (`ml.interfaces`),
never this layer.

**Concurrency**: two threads/processes writing byte-identical content
race harmlessly to the same destination. On Windows, `Path.replace`
racing onto the same destination can raise `PermissionError` rather than
POSIX's silent atomic "one wins" -- both the content write and the
metadata sidecar write catch this and check whether the destination now
exists before deciding whether to re-raise (mirrors
`historical.canonical_store.write_partition`'s identical race-tolerant
rename; found and fixed via this milestone's own concurrency tests, see
"Adversarial audit" below).

## Experiment manifest and lifecycle (Section 10)

`ExperimentManifest` (`manifests.py`) is the durable record of one
experiment's preparation: schema version, `ExperimentIdentity`, the full
embedded `ExperimentSpec` (never duplicated as separate top-level
binding fields -- one source of truth), `model_definition_fingerprint`
(the registry definition's content at preparation time, distinct from
the spec's own `model_name`/`model_version`), status, `EnvironmentSnapshot`,
artifact references, an optional validation report reference, creation/
completion timestamps, an optional failure summary, an optional parent
experiment id, and descriptive limitations.

**Status lifecycle** (`ExperimentStatus`): `CREATED -> VALIDATING ->
READY -> RUNNING -> COMPLETED`, with `FAILED` reachable from
`VALIDATING`/`RUNNING` and `CANCELLED` reachable from several
non-terminal states. Legality is an explicit adjacency table
(`models._LEGAL_TRANSITIONS`) checked by `is_legal_transition`. Every
terminal status (`COMPLETED`, `FAILED`, `CANCELLED`) maps to an EMPTY
legal-transitions set -- `ExperimentManifestStore.transition()` can
therefore never modify a manifest again once it reaches one; this is a
structural consequence of the table, not a separate check bolted on top.

**Why a mutable "current manifest" file, not versioned revisions**: this
milestone's spec allows either design. `manifests.py` chooses "one
current-state file, overwritten atomically in place on each legal
transition" and relies on `tracking.py`'s append-only event log for
history -- duplicating that history a second time as numbered manifest
revisions would be redundant.

`ExperimentManifest.__post_init__` recomputes
`compute_experiment_identity(self.spec)` and compares it to
`self.identity`, raising `ExperimentIdentityError` on any mismatch --
**a manifest can never exist, in memory or reconstructed from disk, with
an identity inconsistent with its own embedded spec.** This is how "same
experiment ID pointing to inconsistent content = corruption, fail
loudly" is enforced: not by a special check in the store, but by every
manifest object's own construction.

`ExperimentManifestStore.create()` never overwrites an existing manifest
file (raises `ExperimentStateError`); `.transition()` rejects illegal
transitions the same way. Both wrap their file operations in
`ml.concurrency.experiment_lock` (a thin adapter over
`historical.locking.DatasetLock`, reused as-is, keyed per-experiment) --
see "Concurrency and crash consistency" below.

**Lifecycle invariants proven by tests** (post-4A correctness audit,
`test_ml_manifests.py::TestExperimentManifestStoreTransitions`): every
legal transition leaves `spec` (and therefore every embedded dataset/
feature/label/preprocessing/model/seed/code binding), `identity`,
`model_definition_fingerprint`, `environment_snapshot`, and `created_at`
byte-for-byte unchanged -- `transition()`'s signature structurally has no
parameter that could rewrite any of them, verified end-to-end rather
than by code inspection alone. `artifact_references`/
`validation_report_reference` are the only fields a transition can
change, and only when explicitly passed -- omitting them on a later
transition carries the existing values forward unchanged, never resets
them. Every terminal status (`COMPLETED`, `FAILED`, `CANCELLED`) rejects
transition attempts to EVERY possible target status, not just the one
example previously covered.

## Experiment event tracking (Section 11)

`ExperimentEventStore` (`tracking.py`) is a lightweight, local,
append-only JSON-Lines log at
`<root>/experiments/<experiment_id>/events.jsonl` -- one canonical-JSON
`EventRecord` per line (`experiment_created`, `validation_started`,
`validation_passed`/`validation_failed`, `run_started`, `artifact_written`,
`run_completed`/`run_failed`, `experiment_cancelled`). Every event carries
an explicit, gapless, strictly-increasing `sequence` number starting at
1 -- never relative wall-clock ordering alone. `details` is restricted to
JSON-primitive values (`models.validate_json_primitive_mapping`), so an
event can carry small structured context but never an arbitrary object,
environment-variable dump, or credential. Timestamps and events here are
operational history -- never part of scientific identity.

**Crash safety is "repair-on-access," not true atomic append**: unlike
this package's content-addressed writes, there is no portable way to
atomically append an arbitrary-length line to an existing file on both
POSIX and Windows. Every access (`append` and `read_events`) tolerates
and silently repairs (truncates away) an unparseable FINAL line -- the
signature of a crash mid-write -- while treating an unparseable line
ANYWHERE ELSE as unrecoverable corruption (`ArtifactCorruptionError`).
This is a documented, deliberately weaker guarantee than the
content-addressed stores, appropriate for an operational log.

## Pre-run validation (Section 13)

`validation.validate_experiment_spec` checks an `ExperimentSpec` against
everything it claims to bind to: the model registry (definition exists,
supports the declared objective) and the resolved
`features.manifests.ResearchDatasetManifest` (dataset identity/content
match, exact features present in matching order, feature registry
fingerprint match, preprocessing definition/fingerprint consistency,
split strategy consistency). It also re-confirms label/objective
compatibility, seed validity, non-empty code revision, and identity
consistency, at `INFO` severity -- these are already guaranteed
structurally by the domain models at construction time, but are reported
explicitly rather than silently assumed, for a complete audit trail.
Every check contributes exactly one `ValidationIssue`
(`INFO`/`WARNING`/`ERROR`/`CRITICAL`); `ValidationReport.is_ready` is
`True` iff no `ERROR`/`CRITICAL` issue exists. **Validation never raises
for a bad experiment** -- only genuinely unusable input (e.g. an empty
artifact root) surfaces as a `CRITICAL` issue.

**Label/dataset cross-validation** (added in the post-4A correctness
audit): `_validate_label_binding` cross-checks `LabelBinding` against the
research dataset manifest's own recorded `label_definition` -- closing a
gap where `_validate_dataset_binding`'s exact `dataset_id`/
`manifest_version`/`content_id` match says nothing about whether a
human-declared `LabelBinding` agrees with what that dataset actually is.
`name`, `kind`, and `horizon_bars` are compared as exact (case-sensitive,
un-normalized) values; `params` is compared as a dict (a missing
manifest `"params"` key defaults to `{}`, matching both sides' own
default, so this is not treated as missing metadata). A manifest missing
`name`/`kind`/`horizon_bars` outright is reported as
`research_dataset_label_binding_incomplete` (CRITICAL) rather than
compared. Mismatches are reported as `label_name_mismatch`,
`label_kind_mismatch`, `label_horizon_mismatch`, or `label_params_mismatch`
(each CRITICAL), plus an aggregate `research_dataset_label_binding_mismatch`
whenever any of them fire. Three checklist items from the original
request have NO representable analogue in Milestone 3's manifest schema
and are deliberately not cross-validated, documented rather than invented:
`LabelBinding.label_type` (an ML-layer-only classification the manifest
never records -- still independently governed by the objective-
compatibility check below); a "target column" (labels are written under
a fixed `"label"` column name by `dataset_builder.py`, never a manifest
field); and a separate "label version"/fingerprint distinct from the
fields above (`label_definition` is already baked into `dataset_id`'s own
hash, so a change there already changes `dataset_id`, independently
caught by `_validate_dataset_binding`). See `_validate_label_binding`'s
own docstring in `validation.py` for the full, exact semantics.

## Experiment creation service (Section 14)

`ExperimentPreparer.prepare(spec)` (`experiment_manager.py`) is the ONE
orchestrator this milestone ships, and it does creation and preparation
ONLY -- there is no method here that resolves a model factory and calls
`.create()`/`.fit()` on it.

Pipeline: compute identity -> if a manifest already exists for that
identity, return it AS-IS (idempotent, see below) -> resolve the model
definition (propagates `UnknownModelDefinitionError` directly, no
manifest created yet) -> resolve the research dataset manifest
(propagates `ResearchDatasetError`/`ManifestError` directly) -> capture
environment -> create the immutable initial manifest at `CREATED`,
append `experiment_created` -> transition to `VALIDATING`, append
`validation_started` -> run `validate_experiment_spec`, write the report
as a content-addressed `REPORT` artifact -> transition to `READY` or
`FAILED` (with a `failure_summary` joining every blocking issue),
recording the validation report's `ArtifactReference` on the manifest
either way -> THEN append `validation_passed`/`validation_failed`.

**Idempotency and descriptive metadata**: two calls with
scientifically-identical specs (same `to_identity_payload()`) but
different `notes`/`tags`/`primary_metric` produce the SAME
`experiment_id`. The FIRST call's descriptive metadata is what gets
durably recorded; a later call returns the existing manifest UNCHANGED --
there is no mechanism in this milestone to update an experiment's
descriptive metadata after first preparation. This is a deliberate
consequence of "never overwrite." This also means a FAILED experiment
(e.g. from a mismatched `LabelBinding`) can never be silently
re-prepared into `ready` merely by calling `prepare()` again with the
identical spec -- the identical spec has the identical identity, finds
the existing FAILED manifest, and is returned as-is, unchanged.

**Manifest/event ordering invariant** (post-4A correctness audit
finding): every step above writes/transitions the manifest FIRST and
appends the event describing it SECOND -- verified by a dedicated
call-order regression test in `test_experiment_manager.py`. Originally,
the final `READY`/`FAILED` step did this backwards (event appended
before the transition); this was corrected because it was the only step
out of step with the others and, in the old order, a crash between the
two calls could leave the event log claiming a transition the manifest
had not yet durably recorded -- strictly worse than the corrected
order's residual gap (a transition recorded with its describing event
still missing, a recoverable incompleteness, since the manifest is the
authoritative record). A separate, still-open limitation: a crash
between `create()`/`transition(VALIDATING)` and the final `READY`/
`FAILED` transition leaves that experiment_id permanently parked at its
last-recorded non-terminal status, since `prepare()`'s idempotency check
treats ANY existing manifest -- terminal or not -- as "already prepared"
and never resumes it. Fixing that would mean redesigning this class's
idempotency/retry semantics, out of scope for this audit.

## Reporting (Section 15)

`reporting.build_report_json(manifest, validation_report=None)` and
`reporting.render_report_markdown(...)` are pure functions of
already-loaded data (no I/O) -- summarizing experiment id, dataset,
model, objective, feature count, label/split/preprocessing, seed
fingerprint, code revision, environment summary, validation result
(expanded if the caller loaded the full `ValidationReport`, otherwise
just the artifact reference), status, artifact references, and
limitations (a standing "no model was fitted, no profitability claim"
notice plus any manifest-specific ones). No charts, no dashboard.

## Configuration (Section 16)

`config/ml_schemas.py` mirrors `config/feature_schemas.py`'s conventions
(frozen pydantic models, `extra="forbid"`, `.build()` factories).
`MLDatasetConfig` deliberately asks only for `dataset_id` +
`manifest_version` + `research_storage_root` -- the actual `content_id`,
`feature_versions`, `feature_registry_fingerprint`,
`preprocessing_definition`, and `fitted_preprocessing_fingerprint` are
all derived live from the loaded `ResearchDatasetManifest` by `ml_cli.py`,
never hand-typed into a config file (a human mistyping a 64-character
hex hash is exactly the class of error this avoids).
`examples/xauusd_ml_experiment.example.json` is a complete worked
example -- see below.

## CLI usage (Section 17)

```
python -m quant_platform.ml_cli list-model-definitions
python -m quant_platform.ml_cli describe-model-definition --name constant_test_model --version 1
python -m quant_platform.ml_cli prepare-experiment --config config.json
python -m quant_platform.ml_cli validate-experiment --config config.json
python -m quant_platform.ml_cli inspect-experiment --config config.json --experiment-id ID [--format json]
python -m quant_platform.ml_cli inspect-experiment-manifest --config config.json --experiment-id ID
python -m quant_platform.ml_cli verify-artifact --config config.json --content-hash HASH
python -m quant_platform.ml_cli list-experiment-events --config config.json --experiment-id ID
```

`validate-experiment` is a pure dry run -- it builds the spec and runs
preflight validation WITHOUT calling `prepare`, writing no manifest,
artifact, or event. `prepare-experiment` and `validate-experiment` exit
0 when the experiment/spec is ready, 2 when it is not (distinct from 1,
which means the command itself failed -- bad config, missing dataset).
There is no `ml train`/`ml predict` command; the only model registered
by `ml_cli.build_model_registry()` is the explicitly test-labeled
`constant_test_model`.

## Worked XAUUSD example

`examples/xauusd_ml_experiment.example.json` prepares an experiment
against the XAUUSD research dataset built by
`examples/xauusd_research_dataset.example.json` (Milestone 3), using
ONLY the deterministic test-only model -- no real model is trained, and
no profitability claim is made or implied:

```bash
python -m quant_platform.feature_cli build-research-dataset \
    --config examples/xauusd_research_dataset.example.json
# note the printed dataset_id and version, then edit
# examples/xauusd_ml_experiment.example.json's "dataset" block to match

python -m quant_platform.ml_cli validate-experiment \
    --config examples/xauusd_ml_experiment.example.json
python -m quant_platform.ml_cli prepare-experiment \
    --config examples/xauusd_ml_experiment.example.json
python -m quant_platform.ml_cli inspect-experiment \
    --config examples/xauusd_ml_experiment.example.json --experiment-id <printed id>
```

(Requires an already-built XAUUSD research dataset first -- this
milestone prepares experiments FROM research datasets, it does not
build them.)

## Concurrency and crash consistency (Section 19)

Every store in `ml/` reuses `historical.locking.DatasetLock` for
per-experiment mutual exclusion rather than a second hand-rolled lock
implementation, through `ml.concurrency.experiment_lock` (added in the
post-4A correctness audit) -- a thin adapter that translates lock-
ACQUISITION failures into `ExperimentLockError` at the ML public
boundary, preserving the original exception as `__cause__`. `DatasetLock`
itself is a **local filesystem advisory lock**, fail-fast (raises
`DatasetLockError` immediately rather than blocking), appropriate for
this platform's documented single-machine, single-writer-at-a-time
design target -- **not distributed consensus**, and it does not protect
against a process that ignores it. Before this adapter existed, a
contested or racing lock acquisition surfaced as a raw `historical`-layer
`DatasetLockError` (or, rarely, a raw `PermissionError` -- see below) --
an ML caller should never need to import a Milestone 2 exception type or
catch a bare `PermissionError` to handle "another process is preparing
this experiment right now." `test_ml_manifests.py::
TestExperimentLockErrorTranslation` and `test_tracking.py::
test_append_translates_contested_lock_deterministically` prove the
translation directly (one thread holds the lock deterministically,
no timing race needed); failures from the PROTECTED BODY (e.g. a genuine
disk error while writing a manifest) are never touched by this adapter
and keep their original type.

Concurrent `create()`/`append()` calls on the SAME experiment_id are
verified safe (exactly one winner, others fail loudly, never silent
corruption) in `test_ml_manifests.py`/`test_ml_tracking.py`. Under
many-way (6+) simultaneous FIRST lock acquisition specifically, a rare,
pre-existing Windows-specific race in `DatasetLock`'s own stale-lock
reclaim path can surface a raw `PermissionError` -- `experiment_lock` now
catches this too and translates it into the same `ExperimentLockError`,
so it no longer needs to be tolerated as a distinct raw exception type at
the ML boundary (it remains a known, documented limitation of reused
Milestone 2 code, out of scope to fix there); this milestone's
concurrency tests use a moderate thread count and tolerate either
`ExperimentLockError` or `ExperimentStateError`, since both represent
"lost the race safely," never corruption.

Interrupted writes: content-addressed stores use atomic temp-file-then-
rename (a reader never observes a partial file at the final path); the
event log tolerates and repairs a torn FINAL line specifically (see
above). `test_ml_manifests.py` additionally proves this deterministically
for `ExperimentManifestStore` via fault injection (`Path.replace` patched
to raise mid-write): an interrupted `create()` leaves no manifest file at
all, and an interrupted `transition()` leaves the PREVIOUS, still-valid
manifest fully intact and loadable -- never a truncated file, never a
"partly applied" transition.

## Security and trust boundaries (Section 18)

- The generic artifact layer never executes code on read -- no pickle,
  no `eval`, no dynamic import; only raw `bytes` in, raw `bytes` out.
- JSON parsing (`persistence.parse_json_strict`) rejects `NaN`/
  `Infinity`/`-Infinity` tokens Python's `json` module otherwise accepts
  as a non-standard extension.
- All paths are checked via `persistence.assert_within_root`, which
  resolves (following symlinks) and verifies containment under the
  configured root -- blocking both literal `../` traversal and
  symlink-based escapes.
- No secrets, credentials, or environment-variable dumps are captured
  anywhere; `EnvironmentSnapshot` captures only package/OS/architecture
  metadata.
- `git` is invoked with an explicit argument list and a timeout, never
  `shell=True`.

## Reproducibility limitations (honest, as documented)

- Seed determinism does not extend to third-party non-deterministic
  native/GPU code (see "Random seed management" above).
- `DatasetLock` is local advisory locking, not distributed consensus.
- Under extreme concurrent contention, `DatasetLock`'s reclaim path can
  rarely surface a raw OS error, which `ml.concurrency.experiment_lock`
  now catches and translates into `ExperimentLockError` at the ML
  boundary (see "Concurrency" above) -- the underlying race itself is
  pre-existing Milestone 2 behavior, not fixed (out of scope).
- `LabelBinding`'s `label_type` (an ML-layer-only classification) and a
  "target column"/"label version" concept have no analogue anywhere in
  the research dataset manifest's schema, so they are not (and cannot be)
  cross-validated against it -- documented, not invented (see "Pre-run
  validation" above for what IS cross-validated: name/kind/horizon_bars/
  params).
- A process crash strictly between a manifest write/transition and the
  event-store append describing it can leave that one event missing from
  the log (the manifest itself, the authoritative record, is never
  wrong) -- there is no cross-file atomic transaction spanning both
  stores (see "Experiment creation service" above).
- A process crash between `create()`/`transition(VALIDATING)` and the
  final `READY`/`FAILED` transition leaves that experiment_id permanently
  parked at its last-recorded non-terminal status; `prepare()`'s
  idempotency check does not distinguish a genuinely-finished manifest
  from an interrupted one and never resumes/retries the latter (see
  "Experiment creation service" above) -- resuming interrupted
  preparation would require redesigning this class's idempotency
  semantics, out of scope for this audit.
- Environment capture is best-effort: a package not resolvable via
  `importlib.metadata` is recorded as `None`, never guessed at.
