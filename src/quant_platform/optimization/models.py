"""`OptimizationSpec`: the complete, immutable binding of one leakage-safe
feature-selection-and-hyperparameter-search run to its exact scientific
inputs, plus `OptimizationStage` -- the state machine tracking one
optimization's coarse lifecycle. Mirrors `ml.experiment_spec.ExperimentSpec`/
`ml.experiment_identity`/`execution.state_machine` exactly: this milestone
reuses that architecture rather than inventing a parallel one.

PREPROCESSING POLICY: OPTION A, CHOSEN AND DOCUMENTED HERE
--------------------------------------------------------------------------
The spec offers two choices: (A) keep scale-sensitive models excluded from
optimization, fail closed; (B) implement fold-local preprocessing
(fit-inner-train-only scaling, refit on outer-train, safe non-pickle
serialization, full identity integration). This milestone implements
**Option A** -- `PreprocessingPolicy.EXCLUDE_SCALE_SENSITIVE` is the ONLY
value `OptimizationSpec.preprocessing_policy` may hold, enforced in
`__post_init__`.

Why: `ml.model_validation._validate_preprocessing_requirements` (Milestone
4C) ALREADY fails closed for any model declaring `ModelCapabilities.
requires_scaled_numeric_features=True` (Logistic Regression, Elastic Net)
through the real, orchestrated execution path -- "this milestone
implements no fold-local preprocessing-refitting framework, so this model
cannot be trained through the real execution engine until one exists" is
that module's own, already-shipped, already-tested statement. Building a
genuinely rigorous fold-local preprocessing abstraction (fit-inner-train-
only, refit-on-outer-train, safe serialization, full identity/fingerprint
integration, resume-safety, adversarial testing) is a substantial project
in its own right -- attempting it "casually" inside an already-enormous
milestone is exactly what the spec's own instruction warns against ("do
not implement B casually. If it cannot be completed rigorously, choose A
and document it"). Choosing A keeps this milestone consistent with the
existing, tested, just-reaffirmed fail-closed precedent rather than
introducing a second, hastily-built preprocessing mechanism alongside it.
Concretely: `build_optimization_spec` rejects (fail closed, at spec-
construction time, before a single trial ever runs) any `model_name`
whose registered `ModelCapabilities.requires_scaled_numeric_features` is
`True` when a `ModelRegistry` is supplied; `optimization.trial_executor`
enforces the identical check unconditionally as the real, always-active
gate (the constructor-time check is a friendlier, earlier fail-fast layer
on top of it, not a replacement for it).

WHY "DETERMINISTIC REQUIREMENTS" IS NOT A SEPARATE SPEC FIELD
--------------------------------------------------------------------------
The spec's identity-field list includes "deterministic requirements".
This is enforced STRUCTURALLY rather than as a new, always-`True` toggle
field: every model this package can resolve already declares
`ModelCapabilities.is_deterministic`, and `optimization.trial_executor`
refuses (fail closed) to run any trial against a model declaring
`is_deterministic=False`. A field that could only ever legally be `True`
in this milestone's implementation would be dead configuration space;
this choice is called out explicitly here (and in the delivery report) as
a documented architectural deviation from the spec's suggested field list.

METRIC DIRECTION: DERIVED AND VERIFIED, NEVER TRUSTED FROM CALLER INPUT
--------------------------------------------------------------------------
`OptimizationSpec.metric_direction` IS part of the frozen identity payload
(freezing today's registry-derived answer against a hypothetical future
change to `ml.comparison`'s direction table silently reinterpreting an
old spec) -- but `__post_init__` re-derives it from `ml.comparison.
is_higher_better(primary_metric)` and raises if the stored value disagrees.
A caller can never hand-construct an `OptimizationSpec` with a direction
that contradicts the authoritative registry; only `build_optimization_spec`
(which always derives it correctly) is the intended construction path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from quant_platform.ml.comparison import is_higher_better
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.fingerprints import fingerprint_json, is_valid_sha256_hex
from quant_platform.ml.models import (
    DatasetBinding,
    JsonPrimitive,
    ObjectiveType,
    SplitBinding,
    validate_json_primitive_mapping,
)
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version
from quant_platform.ml.registry import ModelRegistry
from quant_platform.ml.seeds import SeedConfiguration, derive_seed
from quant_platform.optimization.feature_selection import FeatureSelectionSpec, FeatureUniverse
from quant_platform.optimization.inner_splits import InnerSplitConfig
from quant_platform.optimization.search_space import SearchSpace

OPTIMIZATION_SCHEMA_VERSION = 1
OPTIMIZATION_IDENTITY_SCHEMA_VERSION = 1
RANKING_POLICY_VERSION = 1
"""Versions `optimization.candidates`' ONE fixed, deterministic ranking
algorithm -- folded into `OptimizationSpec`'s identity payload exactly
like `IDENTITY_SCHEMA_VERSION` is, so a future change to the ranking rule
itself always produces a different `optimization_id` rather than silently
reinterpreting an old spec's already-recorded trials under a new policy."""


class SamplerKind(Enum):
    TPE = "tpe"
    RANDOM = "random"


class PruningKind(Enum):
    NONE = "none"
    """The mandatory no-pruning control."""
    MEDIAN_STOPPING = "median_stopping"
    """Prune a trial iff, after `min_completed_inner_folds` inner folds,
    its running aggregate is worse (per the primary metric's authoritative
    direction) than the MEDIAN running aggregate of every OTHER trial's
    completed, non-pruned folds at the same inner-fold count -- a
    deterministic aggregate over completed inner-fold metrics ONLY, never
    outer-test."""


class PreprocessingPolicy(Enum):
    EXCLUDE_SCALE_SENSITIVE = "exclude_scale_sensitive"


class OptimizationStage(Enum):
    INITIALIZING = "initializing"
    LOADING_EXPERIMENT = "loading_experiment"
    BUILDING_OUTER_PLAN = "building_outer_plan"
    RUNNING_OUTER_FOLD = "running_outer_fold"
    BUILDING_INNER_PLAN = "building_inner_plan"
    RUNNING_TRIAL = "running_trial"
    SELECTING_CANDIDATE = "selecting_candidate"
    REFITTING_WINNER = "refitting_winner"
    EVALUATING_OUTER_TEST = "evaluating_outer_test"
    STORING_RESULTS = "storing_results"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERABLE_FAILURE = "recoverable_failure"


TERMINAL_OPTIMIZATION_STAGES: frozenset[OptimizationStage] = frozenset(
    {OptimizationStage.COMPLETED, OptimizationStage.FAILED, OptimizationStage.CANCELLED}
)

_LEGAL_OPTIMIZATION_TRANSITIONS: dict[OptimizationStage, frozenset[OptimizationStage]] = {
    OptimizationStage.INITIALIZING: frozenset({OptimizationStage.LOADING_EXPERIMENT, OptimizationStage.FAILED, OptimizationStage.CANCELLED}),
    OptimizationStage.LOADING_EXPERIMENT: frozenset({OptimizationStage.BUILDING_OUTER_PLAN, OptimizationStage.FAILED, OptimizationStage.CANCELLED}),
    OptimizationStage.BUILDING_OUTER_PLAN: frozenset({OptimizationStage.RUNNING_OUTER_FOLD, OptimizationStage.FAILED, OptimizationStage.CANCELLED}),
    OptimizationStage.RUNNING_OUTER_FOLD: frozenset({
        # Self-loop: mirrors RUNNING_TRIAL's own self-loop below -- the
        # per-outer-fold `current_outer_fold_index` counter advancing to
        # the NEXT fold while still in the overall "running outer folds"
        # phase, and the resume-normalization path (see runner.py) that
        # re-enters this stage from RECOVERABLE_FAILURE before resuming
        # the SAME outer fold's own inner trial loop.
        OptimizationStage.RUNNING_OUTER_FOLD,
        OptimizationStage.BUILDING_INNER_PLAN, OptimizationStage.RECOVERABLE_FAILURE, OptimizationStage.FAILED, OptimizationStage.CANCELLED,
    }),
    OptimizationStage.BUILDING_INNER_PLAN: frozenset({
        OptimizationStage.RUNNING_TRIAL, OptimizationStage.RECOVERABLE_FAILURE, OptimizationStage.FAILED, OptimizationStage.CANCELLED,
    }),
    OptimizationStage.RUNNING_TRIAL: frozenset({
        # Self-loop: trial N complete, trial N+1 begins. Deliberately NOT
        # split into a separate "storing trial" stage the way the outer
        # execution engine splits RUNNING_FOLD/STORING_RESULTS -- each
        # trial's own artifact write and manifest trial-count update
        # happen WITHIN this stage; RUNNING_TRIAL -> RUNNING_TRIAL already
        # unambiguously means "one more trial completed, still trialing",
        # and a dedicated intermediate stage would carry no additional
        # legality information the manifest's own trial-progress fields
        # do not already provide. See module docstring.
        OptimizationStage.RUNNING_TRIAL, OptimizationStage.SELECTING_CANDIDATE,
        OptimizationStage.RECOVERABLE_FAILURE, OptimizationStage.FAILED, OptimizationStage.CANCELLED,
    }),
    OptimizationStage.SELECTING_CANDIDATE: frozenset({
        # RECOVERABLE_FAILURE is legal here (and from REFITTING_WINNER/
        # EVALUATING_OUTER_TEST below) because everything from candidate
        # selection through outer-test evaluation is a PURE, deterministic
        # function of already-fixed inputs (the verified trial set, the
        # outer fold's own row positions) -- re-entering and redoing it
        # from scratch after a crash reproduces the identical result
        # bit-for-bit; it is never a repeated "peek" at a changing answer.
        OptimizationStage.REFITTING_WINNER, OptimizationStage.RECOVERABLE_FAILURE, OptimizationStage.FAILED, OptimizationStage.CANCELLED,
    }),
    OptimizationStage.REFITTING_WINNER: frozenset({
        OptimizationStage.EVALUATING_OUTER_TEST, OptimizationStage.RECOVERABLE_FAILURE, OptimizationStage.FAILED, OptimizationStage.CANCELLED,
    }),
    OptimizationStage.EVALUATING_OUTER_TEST: frozenset({
        OptimizationStage.STORING_RESULTS, OptimizationStage.RECOVERABLE_FAILURE, OptimizationStage.FAILED, OptimizationStage.CANCELLED,
    }),
    OptimizationStage.STORING_RESULTS: frozenset({
        # Loop back for the NEXT outer fold, or finish entirely.
        OptimizationStage.RUNNING_OUTER_FOLD, OptimizationStage.COMPLETED, OptimizationStage.FAILED, OptimizationStage.CANCELLED,
    }),
    OptimizationStage.RECOVERABLE_FAILURE: frozenset({
        OptimizationStage.RUNNING_OUTER_FOLD, OptimizationStage.RUNNING_TRIAL, OptimizationStage.FAILED, OptimizationStage.CANCELLED,
    }),
    OptimizationStage.COMPLETED: frozenset(),
    OptimizationStage.FAILED: frozenset(),
    OptimizationStage.CANCELLED: frozenset(),
}


def is_legal_optimization_transition(current: OptimizationStage, target: OptimizationStage) -> bool:
    return target in _LEGAL_OPTIMIZATION_TRANSITIONS[current]


def is_terminal_optimization_stage(stage: OptimizationStage) -> bool:
    return stage in TERMINAL_OPTIMIZATION_STAGES


@dataclass(frozen=True, slots=True)
class PruningConfig:
    kind: PruningKind
    min_completed_inner_folds: int = 1
    params: Mapping[str, JsonPrimitive] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_completed_inner_folds < 1:
            raise ValueError(f"PruningConfig.min_completed_inner_folds must be >= 1, got {self.min_completed_inner_folds}")
        validate_json_primitive_mapping(self.params, field_name="PruningConfig.params")

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "min_completed_inner_folds": self.min_completed_inner_folds, "params": dict(sorted(self.params.items()))}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PruningConfig:
        return cls(
            kind=PruningKind(raw["kind"]), min_completed_inner_folds=int(str(raw.get("min_completed_inner_folds", 1))),
            params=as_json_dict(raw.get("params") or {}, field_name="PruningConfig.params"),
        )


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    """GBM early stopping, ALWAYS evaluated against inner-validation only
    (never outer-test, never the complete outer-train, never future
    rows) -- enforced structurally by `optimization.trial_executor`,
    which is the only place `enabled`/`patience`/`validation_fraction`
    are ever translated into a model's own `early_stopping_rounds`/
    `validation_fraction` hyperparameter keys (see `ml.model_zoo.
    lightgbm_model`/`xgboost_model`/`catboost_model`'s own, already-
    shipped support for exactly those keys -- reused directly, never
    reimplemented here).

    `final_round_policy` governs the number of boosting rounds used when
    the winning candidate is refit on the COMPLETE outer-train partition
    (where there is no inner-validation left to early-stop against):
    `"median_best_iteration"` uses the rounded median of every successful
    inner fold's own best iteration (falling back to the sampled/declared
    round count if no inner fold reports one); `"fixed"` always uses the
    sampled/declared round count, ignoring any inner fold's best
    iteration entirely. Both are deterministic; neither ever reads
    outer-test performance."""

    enabled: bool
    patience: int | None = None
    validation_fraction: float = 0.1
    final_round_policy: str = "median_best_iteration"

    def __post_init__(self) -> None:
        if self.enabled:
            if self.patience is None or self.patience < 1:
                raise ValueError(f"EarlyStoppingConfig.patience must be >= 1 when enabled, got {self.patience}")
            if not (0.0 < self.validation_fraction < 1.0):
                raise ValueError(f"EarlyStoppingConfig.validation_fraction must be in (0, 1), got {self.validation_fraction}")
        if self.final_round_policy not in ("median_best_iteration", "fixed"):
            raise ValueError(f"EarlyStoppingConfig.final_round_policy must be 'median_best_iteration' or 'fixed', got {self.final_round_policy!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled, "patience": self.patience, "validation_fraction": self.validation_fraction,
            "final_round_policy": self.final_round_policy,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> EarlyStoppingConfig:
        patience_raw = raw.get("patience")
        return cls(
            enabled=bool(raw["enabled"]), patience=(None if patience_raw is None else int(str(patience_raw))),
            validation_fraction=float(str(raw.get("validation_fraction", 0.1))),
            final_round_policy=str(raw.get("final_round_policy", "median_best_iteration")),
        )


@dataclass(frozen=True, slots=True)
class OptimizationSpec:
    schema_version: int
    parent_experiment_id: str
    dataset_binding: DatasetBinding
    model_name: str
    model_version: str
    objective: ObjectiveType
    primary_metric: str
    metric_direction: str
    outer_split_binding: SplitBinding
    inner_split_config: InnerSplitConfig
    feature_selection_spec: FeatureSelectionSpec
    feature_universe_fingerprint: str
    search_space: SearchSpace
    sampler_kind: SamplerKind
    pruning_config: PruningConfig
    early_stopping_config: EarlyStoppingConfig
    preprocessing_policy: PreprocessingPolicy
    max_trials: int
    min_successful_inner_folds: int
    ranking_policy_version: int
    seed_configuration: SeedConfiguration
    timeout_seconds: int | None = None
    max_failed_trials: int | None = None
    tags: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not is_valid_sha256_hex(self.parent_experiment_id):
            raise ValueError(f"OptimizationSpec.parent_experiment_id must be a valid sha256 hex id, got {self.parent_experiment_id!r}")
        if not self.model_name:
            raise ValueError("OptimizationSpec.model_name must not be empty")
        if not self.model_version:
            raise ValueError("OptimizationSpec.model_version must not be empty")
        if not self.primary_metric:
            raise ValueError("OptimizationSpec.primary_metric must not be empty")
        expected_direction = "maximize" if is_higher_better(self.primary_metric) else "minimize"
        if self.metric_direction != expected_direction:
            raise ValueError(
                f"OptimizationSpec.metric_direction={self.metric_direction!r} does not match the authoritative "
                f"ml.comparison registry's derived direction {expected_direction!r} for primary_metric "
                f"{self.primary_metric!r} -- metric direction must never be caller-trusted"
            )
        if self.preprocessing_policy is not PreprocessingPolicy.EXCLUDE_SCALE_SENSITIVE:
            raise ValueError(
                f"OptimizationSpec.preprocessing_policy={self.preprocessing_policy!r} is not implemented -- only "
                "PreprocessingPolicy.EXCLUDE_SCALE_SENSITIVE (Option A) exists in this milestone, see module docstring"
            )
        if self.max_trials < 1:
            raise ValueError(f"OptimizationSpec.max_trials must be >= 1 (bounded compute budget), got {self.max_trials}")
        if self.min_successful_inner_folds < 1:
            raise ValueError(f"OptimizationSpec.min_successful_inner_folds must be >= 1, got {self.min_successful_inner_folds}")
        if self.min_successful_inner_folds > self.inner_split_config.n_splits:
            raise ValueError(
                f"OptimizationSpec.min_successful_inner_folds ({self.min_successful_inner_folds}) cannot exceed "
                f"inner_split_config.n_splits ({self.inner_split_config.n_splits})"
            )
        if self.ranking_policy_version != RANKING_POLICY_VERSION:
            raise ValueError(f"OptimizationSpec.ranking_policy_version must be {RANKING_POLICY_VERSION}, got {self.ranking_policy_version}")
        if self.timeout_seconds is not None and self.timeout_seconds < 1:
            raise ValueError(f"OptimizationSpec.timeout_seconds must be >= 1 if set, got {self.timeout_seconds}")
        if self.max_failed_trials is not None and self.max_failed_trials < 0:
            raise ValueError(f"OptimizationSpec.max_failed_trials must be >= 0 if set, got {self.max_failed_trials}")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("OptimizationSpec.tags must not contain duplicates")

    def to_identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "parent_experiment_id": self.parent_experiment_id,
            "dataset_binding": self.dataset_binding.to_json_dict(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "objective": self.objective.value,
            "primary_metric": self.primary_metric,
            "metric_direction": self.metric_direction,
            "outer_split_binding": self.outer_split_binding.to_json_dict(),
            "inner_split_config": self.inner_split_config.to_json_dict(),
            "feature_selection_spec": self.feature_selection_spec.to_json_dict(),
            "feature_universe_fingerprint": self.feature_universe_fingerprint,
            "search_space": self.search_space.to_json_dict(),
            "sampler_kind": self.sampler_kind.value,
            "pruning_config": self.pruning_config.to_json_dict(),
            "early_stopping_config": self.early_stopping_config.to_json_dict(),
            "preprocessing_policy": self.preprocessing_policy.value,
            "max_trials": self.max_trials,
            "min_successful_inner_folds": self.min_successful_inner_folds,
            "ranking_policy_version": self.ranking_policy_version,
            "seed_configuration": self.seed_configuration.to_json_dict(),
            "timeout_seconds": self.timeout_seconds,
            "max_failed_trials": self.max_failed_trials,
        }

    def to_json_dict(self) -> dict[str, object]:
        payload = self.to_identity_payload()
        payload["tags"] = list(self.tags)
        payload["notes"] = self.notes
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> OptimizationSpec:
        require_schema_version(raw, supported=OPTIMIZATION_SCHEMA_VERSION, context="OptimizationSpec")
        return cls(
            schema_version=OPTIMIZATION_SCHEMA_VERSION,
            parent_experiment_id=str(raw["parent_experiment_id"]),
            dataset_binding=DatasetBinding.from_json_dict(as_json_dict(raw["dataset_binding"], field_name="dataset_binding")),
            model_name=str(raw["model_name"]), model_version=str(raw["model_version"]),
            objective=ObjectiveType(raw["objective"]), primary_metric=str(raw["primary_metric"]),
            metric_direction=str(raw["metric_direction"]),
            outer_split_binding=SplitBinding.from_json_dict(as_json_dict(raw["outer_split_binding"], field_name="outer_split_binding")),
            inner_split_config=InnerSplitConfig.from_json_dict(as_json_dict(raw["inner_split_config"], field_name="inner_split_config")),
            feature_selection_spec=FeatureSelectionSpec.from_json_dict(as_json_dict(raw["feature_selection_spec"], field_name="feature_selection_spec")),
            feature_universe_fingerprint=str(raw["feature_universe_fingerprint"]),
            search_space=SearchSpace.from_json_dict(as_json_dict(raw["search_space"], field_name="search_space")),
            sampler_kind=SamplerKind(raw["sampler_kind"]),
            pruning_config=PruningConfig.from_json_dict(as_json_dict(raw["pruning_config"], field_name="pruning_config")),
            early_stopping_config=EarlyStoppingConfig.from_json_dict(as_json_dict(raw["early_stopping_config"], field_name="early_stopping_config")),
            preprocessing_policy=PreprocessingPolicy(raw["preprocessing_policy"]),
            max_trials=int(str(raw["max_trials"])), min_successful_inner_folds=int(str(raw["min_successful_inner_folds"])),
            ranking_policy_version=int(str(raw["ranking_policy_version"])),
            seed_configuration=SeedConfiguration.from_json_dict(as_json_dict(raw["seed_configuration"], field_name="seed_configuration")),
            timeout_seconds=(None if raw.get("timeout_seconds") is None else int(str(raw["timeout_seconds"]))),
            max_failed_trials=(None if raw.get("max_failed_trials") is None else int(str(raw["max_failed_trials"]))),
            tags=tuple(str(t) for t in as_json_list(raw.get("tags") or [], field_name="tags")),
            notes=str(raw.get("notes", "")),
        )


def build_optimization_spec(
    *, experiment: ExperimentSpec, parent_experiment_id: str, model_name: str, model_version: str, primary_metric: str,
    inner_split_config: InnerSplitConfig, feature_selection_spec: FeatureSelectionSpec, search_space: SearchSpace,
    sampler_kind: SamplerKind, pruning_config: PruningConfig, early_stopping_config: EarlyStoppingConfig,
    max_trials: int, min_successful_inner_folds: int, seed_configuration: SeedConfiguration,
    timeout_seconds: int | None = None, max_failed_trials: int | None = None, tags: tuple[str, ...] = (), notes: str = "",
    model_registry: ModelRegistry | None = None,
) -> OptimizationSpec:
    """The intended `OptimizationSpec` construction path -- derives
    `dataset_binding`/`outer_split_binding`/`feature_universe_fingerprint`/
    `metric_direction` from the parent `ExperimentSpec` and the
    authoritative metric registry rather than trusting a caller to supply
    consistent values by hand. When `model_registry` is given, fails
    closed immediately (before any trial ever runs) if `model_name`
    requires scaled numeric features -- see module docstring."""
    if model_registry is not None:
        capabilities = model_registry.get(model_name, model_version).capabilities
        if capabilities.requires_scaled_numeric_features:
            raise ValueError(
                f"Model {model_name!r}@{model_version!r} requires scaled numeric features, but this milestone's "
                "preprocessing policy (Option A) excludes scale-sensitive models from optimization entirely -- "
                "see optimization.models' module docstring"
            )
        if not capabilities.supports(experiment.objective):
            raise ValueError(f"Model {model_name!r}@{model_version!r} does not support objective {experiment.objective.value!r}")

    universe = FeatureUniverse.from_experiment_spec(experiment)
    metric_direction = "maximize" if is_higher_better(primary_metric) else "minimize"
    return OptimizationSpec(
        schema_version=OPTIMIZATION_SCHEMA_VERSION, parent_experiment_id=parent_experiment_id,
        dataset_binding=experiment.dataset_binding, model_name=model_name, model_version=model_version,
        objective=experiment.objective, primary_metric=primary_metric, metric_direction=metric_direction,
        outer_split_binding=experiment.split_binding, inner_split_config=inner_split_config,
        feature_selection_spec=feature_selection_spec, feature_universe_fingerprint=universe.fingerprint,
        search_space=search_space, sampler_kind=sampler_kind, pruning_config=pruning_config,
        early_stopping_config=early_stopping_config, preprocessing_policy=PreprocessingPolicy.EXCLUDE_SCALE_SENSITIVE,
        max_trials=max_trials, min_successful_inner_folds=min_successful_inner_folds,
        ranking_policy_version=RANKING_POLICY_VERSION, seed_configuration=seed_configuration,
        timeout_seconds=timeout_seconds, max_failed_trials=max_failed_trials, tags=tags, notes=notes,
    )


@dataclass(frozen=True, slots=True)
class OptimizationIdentity:
    schema_version: int
    optimization_id: str

    def __post_init__(self) -> None:
        if not is_valid_sha256_hex(self.optimization_id):
            raise ValueError(f"OptimizationIdentity.optimization_id must be a valid sha256 hex digest, got {self.optimization_id!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "optimization_id": self.optimization_id}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> OptimizationIdentity:
        require_schema_version(raw, supported=OPTIMIZATION_IDENTITY_SCHEMA_VERSION, context="OptimizationIdentity")
        return cls(schema_version=OPTIMIZATION_IDENTITY_SCHEMA_VERSION, optimization_id=str(raw["optimization_id"]))


def compute_optimization_identity(spec: OptimizationSpec) -> OptimizationIdentity:
    """Pure function, mirroring `experiment_identity.compute_experiment_
    identity` exactly: two scientifically identical `OptimizationSpec`s
    (identical `to_identity_payload()`) always produce the same
    `optimization_id`, regardless of process, machine, dict insertion
    order, or wall-clock time; a materially different spec always
    produces a different one."""
    payload = dict(spec.to_identity_payload())
    payload["identity_schema_version"] = OPTIMIZATION_IDENTITY_SCHEMA_VERSION
    optimization_id = fingerprint_json(payload)
    return OptimizationIdentity(schema_version=OPTIMIZATION_IDENTITY_SCHEMA_VERSION, optimization_id=optimization_id)


def verify_optimization_identity(spec: OptimizationSpec, identity: OptimizationIdentity) -> bool:
    return compute_optimization_identity(spec) == identity


# --------------------------------------------------------------------------
# Seed derivation hierarchy -- see module-level docstring in
# optimization/__init__.py for the full chain diagram. Every function here
# is a pure composition of `ml.seeds.SeedConfiguration.derive`/`derive_seed`
# -- none ever touches Python's global `random` module or NumPy's global
# RNG state.
# --------------------------------------------------------------------------
def sampler_seed(seed_configuration: SeedConfiguration) -> int:
    return seed_configuration.derive("optimization_sampler")


def outer_fold_seed(seed_configuration: SeedConfiguration, outer_fold_index: int) -> int:
    return seed_configuration.derive(f"optimization_outer_fold:{outer_fold_index}")


def trial_seed(seed_configuration: SeedConfiguration, outer_fold_index: int, trial_number: int) -> int:
    return derive_seed(outer_fold_seed(seed_configuration, outer_fold_index), f"optimization_trial:{trial_number}")


def inner_fold_seed(seed_configuration: SeedConfiguration, outer_fold_index: int, trial_number: int, inner_fold_index: int) -> int:
    return derive_seed(trial_seed(seed_configuration, outer_fold_index, trial_number), f"optimization_inner_fold:{inner_fold_index}")


def feature_selector_seed(seed_configuration: SeedConfiguration, outer_fold_index: int, trial_number: int, inner_fold_index: int) -> int:
    return derive_seed(inner_fold_seed(seed_configuration, outer_fold_index, trial_number, inner_fold_index), "optimization_feature_selection")


def model_fit_seed(seed_configuration: SeedConfiguration, outer_fold_index: int, trial_number: int, inner_fold_index: int) -> int:
    return derive_seed(inner_fold_seed(seed_configuration, outer_fold_index, trial_number, inner_fold_index), "optimization_model_fit")


def outer_train_refit_seed(seed_configuration: SeedConfiguration, outer_fold_index: int) -> int:
    """The seed used when refitting the winning candidate on the COMPLETE
    outer-train partition -- deliberately its own, distinct branch (from
    `outer_fold_seed`, never from any one trial's `trial_seed`), so the
    final refit is not accidentally coupled to whichever trial number
    happened to win."""
    return derive_seed(outer_fold_seed(seed_configuration, outer_fold_index), "optimization_outer_train_refit")


def outer_train_feature_selector_seed(seed_configuration: SeedConfiguration, outer_fold_index: int) -> int:
    """The seed used when refitting feature selection ONE final time on
    the COMPLETE outer-train partition, before the outer-train refit
    itself -- see `optimization.outer_fold`'s module docstring for why
    this is the chosen "final selected feature set" policy."""
    return derive_seed(outer_fold_seed(seed_configuration, outer_fold_index), "optimization_outer_train_feature_selection")


__all__ = [
    "OPTIMIZATION_IDENTITY_SCHEMA_VERSION",
    "OPTIMIZATION_SCHEMA_VERSION",
    "RANKING_POLICY_VERSION",
    "TERMINAL_OPTIMIZATION_STAGES",
    "EarlyStoppingConfig",
    "OptimizationIdentity",
    "OptimizationSpec",
    "OptimizationStage",
    "PreprocessingPolicy",
    "PruningConfig",
    "PruningKind",
    "SamplerKind",
    "build_optimization_spec",
    "compute_optimization_identity",
    "feature_selector_seed",
    "inner_fold_seed",
    "is_legal_optimization_transition",
    "is_terminal_optimization_stage",
    "model_fit_seed",
    "outer_fold_seed",
    "outer_train_feature_selector_seed",
    "outer_train_refit_seed",
    "sampler_seed",
    "trial_seed",
    "verify_optimization_identity",
]
