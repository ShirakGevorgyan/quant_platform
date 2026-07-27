"""Leakage-safe feature selection (Milestone 4D) -- the candidate feature
universe, six selection strategies, and the immutable `FeatureSelectionResult`
every one of them produces.

THE CANDIDATE FEATURE UNIVERSE IS NEVER DISCOVERED AT RUNTIME
--------------------------------------------------------------------------
`FeatureUniverse.from_experiment_spec` builds the universe from the parent
`ExperimentSpec.feature_binding.feature_names` alone -- the exact, ordered,
already-declared feature list Milestone 3/4A's `features.registry`/
`ml.experiment_spec` already established for this experiment. That binding
is, by construction, already free of the label column, timestamps, split
metadata, group identifiers, and leakage/audit columns: `execution.runner.
_run_one_fold` selects a fold's feature frame via this exact same list
(`feature_names = list(spec.feature_binding.feature_names)`), never by
introspecting a DataFrame's columns at runtime. This module therefore adds
NO new column-exclusion logic of its own -- it reuses Milestone 4A's
declaration, satisfying "do not discover arbitrary extra columns from a
DataFrame at runtime" by construction rather than by a new filter.

EVERY STRATEGY IS FIT ON INNER-TRAIN ONLY, EVERY TIME IT IS CALLED
--------------------------------------------------------------------------
None of the six functions below (`select_none` through
`select_stability`) accept or reference anything resembling "the whole
dataset", "outer-test", or a previously-computed selection to reuse
across folds -- each is a pure function of the ONE inner-train partition
(features/labels/row-positions) it is given. `optimization.trial_executor`
is responsible for calling the right one fresh, independently, inside
every inner fold; nothing here has the information to do otherwise (there
is no dataset handle, no store, no cache -- only the rows passed in).

MODEL-NATIVE IMPORTANCE: EXPLICITLY-DECLARED SUPPORTED MODELS ONLY
--------------------------------------------------------------------------
`MODEL_NATIVE_IMPORTANCE_SUPPORTED_MODELS` is deliberately narrow (LightGBM/
XGBoost/CatBoost) -- exactly the model families this platform's own
`ml.model_zoo` gives a concrete, already-tested `feature_importance()`
method to (detected via `getattr`, never a per-model `if`/`elif` chain, so
adding a future supported model needs no branching change here, only an
addition to this tuple once that model's wrapper grows the same method).
Logistic Regression/Elastic Net are excluded from this milestone's
optimization surface entirely (see `optimization.models`' module
docstring on the fail-closed preprocessing policy), so they are moot here
regardless. Never SHAP, never permutation importance -- both explicitly
out of scope.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.feature_selection import (  # type: ignore[import-untyped]
    mutual_info_classif,
    mutual_info_regression,
)

from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.fingerprints import fingerprint_json
from quant_platform.ml.interfaces import FeatureSchema, ModelFactory
from quant_platform.ml.models import (
    JsonPrimitive,
    ModelHyperparameters,
    ObjectiveType,
    validate_json_primitive_mapping,
)
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)
from quant_platform.ml.seeds import SeedConfiguration, derive_seed
from quant_platform.optimization.search_space import (
    CATBOOST_MODEL_NAME,
    LIGHTGBM_MODEL_NAME,
    XGBOOST_MODEL_NAME,
)

FEATURE_SELECTION_SCHEMA_VERSION = 1
MAX_STABILITY_REPEATS = 200
"""Bounded compute budget for STABILITY_SELECTION -- "bounded compute
budget" is an explicit spec requirement; this is a conservative, documented
ceiling, not a performance-tuned value."""

MODEL_NATIVE_IMPORTANCE_SUPPORTED_MODELS: tuple[str, ...] = (LIGHTGBM_MODEL_NAME, XGBOOST_MODEL_NAME, CATBOOST_MODEL_NAME)
_STABILITY_BASE_STRATEGIES = ("variance_filter", "correlation_filter", "univariate")


class FeatureSelectionStrategy(Enum):
    NONE = "none"
    VARIANCE_FILTER = "variance_filter"
    CORRELATION_FILTER = "correlation_filter"
    UNIVARIATE = "univariate"
    MODEL_NATIVE_IMPORTANCE = "model_native_importance"
    STABILITY_SELECTION = "stability_selection"


@dataclass(frozen=True, slots=True)
class FeatureUniverse:
    """The immutable candidate feature universe -- ordered exactly as
    `ExperimentSpec.feature_binding` declares it. `fingerprint` binds both
    the ordered name tuple AND the upstream feature-registry fingerprint,
    so a universe fingerprint changes if either the SET/ORDER of features
    changes or the underlying feature DEFINITIONS do (a silent feature
    redefinition must not produce an identical universe fingerprint)."""

    feature_names: tuple[str, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError("FeatureUniverse.feature_names must not be empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("FeatureUniverse.feature_names must not contain duplicates")

    @classmethod
    def from_experiment_spec(cls, spec: ExperimentSpec) -> FeatureUniverse:
        names = spec.feature_binding.feature_names
        fp = fingerprint_json({
            "feature_names": list(names), "feature_registry_fingerprint": spec.feature_binding.feature_registry_fingerprint,
        })
        return cls(feature_names=names, fingerprint=fp)

    def to_json_dict(self) -> dict[str, object]:
        return {"feature_names": list(self.feature_names), "fingerprint": self.fingerprint}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureUniverse:
        return cls(
            feature_names=tuple(str(n) for n in as_json_list(raw["feature_names"], field_name="feature_names")),
            fingerprint=str(raw["fingerprint"]),
        )


def _require_positive_int(params: Mapping[str, JsonPrimitive], key: str) -> int:
    value = params.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"FeatureSelectionSpec.params[{key!r}] must be a positive integer, got {value!r}")
    return value


def _require_unit_float(params: Mapping[str, JsonPrimitive], key: str, *, low: float = 0.0, high: float = 1.0) -> float:
    value = params.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"FeatureSelectionSpec.params[{key!r}] must be numeric, got {value!r}")
    numeric = float(value)
    if not (low <= numeric <= high):
        raise ValueError(f"FeatureSelectionSpec.params[{key!r}]={numeric} must be in [{low}, {high}]")
    return numeric


@dataclass(frozen=True, slots=True)
class FeatureSelectionSpec:
    """The durable, identity-relevant DECLARATION of which strategy to run
    and its parameters -- never a fitted result. Part of `OptimizationSpec`'s
    own canonical identity payload."""

    strategy: FeatureSelectionStrategy
    params: Mapping[str, JsonPrimitive] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_json_primitive_mapping(self.params, field_name="FeatureSelectionSpec.params")
        if self.strategy is FeatureSelectionStrategy.NONE:
            if self.params:
                raise ValueError("FeatureSelectionSpec: strategy=NONE must not declare any params")
        elif self.strategy is FeatureSelectionStrategy.VARIANCE_FILTER:
            min_variance = self.params.get("min_variance")
            if isinstance(min_variance, bool) or not isinstance(min_variance, (int, float)) or float(min_variance) < 0:
                raise ValueError(f"VARIANCE_FILTER requires params['min_variance'] >= 0, got {min_variance!r}")
        elif self.strategy is FeatureSelectionStrategy.CORRELATION_FILTER:
            _require_unit_float(self.params, "max_abs_correlation", low=1e-9, high=1.0)
        elif self.strategy is FeatureSelectionStrategy.UNIVARIATE:
            self._validate_topk_or_percentile()
        elif self.strategy is FeatureSelectionStrategy.MODEL_NATIVE_IMPORTANCE:
            mode = self.params.get("mode")
            if mode == "top_k":
                _require_positive_int(self.params, "k")
            elif mode == "threshold":
                min_importance = self.params.get("min_importance")
                if isinstance(min_importance, bool) or not isinstance(min_importance, (int, float)) or float(min_importance) < 0:
                    raise ValueError(f"MODEL_NATIVE_IMPORTANCE mode='threshold' requires params['min_importance'] >= 0, got {min_importance!r}")
            else:
                raise ValueError(f"MODEL_NATIVE_IMPORTANCE requires params['mode'] in {{'top_k', 'threshold'}}, got {mode!r}")
        elif self.strategy is FeatureSelectionStrategy.STABILITY_SELECTION:
            base_strategy = self.params.get("base_strategy")
            if base_strategy not in _STABILITY_BASE_STRATEGIES:
                raise ValueError(f"STABILITY_SELECTION requires params['base_strategy'] in {_STABILITY_BASE_STRATEGIES}, got {base_strategy!r}")
            n_repeats = _require_positive_int(self.params, "n_repeats")
            if n_repeats > MAX_STABILITY_REPEATS:
                raise ValueError(f"STABILITY_SELECTION params['n_repeats']={n_repeats} exceeds the bounded maximum of {MAX_STABILITY_REPEATS}")
            _require_unit_float(self.params, "subsample_fraction", low=1e-9, high=1.0) if "subsample_fraction" in self.params else None
            _require_unit_float(self.params, "min_frequency", low=0.0, high=1.0)
        else:  # pragma: no cover - exhaustive over FeatureSelectionStrategy
            raise ValueError(f"Unknown FeatureSelectionStrategy {self.strategy!r}")

    def _validate_topk_or_percentile(self) -> None:
        mode = self.params.get("mode", "top_k")
        if mode == "top_k":
            _require_positive_int(self.params, "k")
        elif mode == "percentile":
            _require_unit_float(self.params, "percentile", low=1e-9, high=100.0)
        else:
            raise ValueError(f"UNIVARIATE requires params['mode'] in {{'top_k', 'percentile'}}, got {mode!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {"strategy": self.strategy.value, "params": dict(sorted(self.params.items()))}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureSelectionSpec:
        return cls(
            strategy=FeatureSelectionStrategy(raw["strategy"]),
            params=as_json_dict(raw.get("params") or {}, field_name="FeatureSelectionSpec.params"),
        )

    def fingerprint(self) -> str:
        return fingerprint_json(self.to_json_dict())


@dataclass(frozen=True, slots=True)
class FeatureSelectionResult:
    schema_version: int
    strategy: FeatureSelectionStrategy
    selected_features: tuple[str, ...]
    rejected_features: tuple[str, ...]
    feature_universe_fingerprint: str
    selector_params: Mapping[str, JsonPrimitive]
    selector_seed: int
    training_row_count: int
    training_row_first_position: int
    training_row_last_position: int
    training_row_fingerprint: str
    selection_reason: str
    fitted_at: str
    per_feature_score: Mapping[str, float] | None = None
    per_feature_rank: Mapping[str, int] | None = None
    stability_frequency: Mapping[str, float] | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.selected_features:
            raise ValueError("FeatureSelectionResult.selected_features must not be empty")
        if len(set(self.selected_features)) != len(self.selected_features):
            raise ValueError("FeatureSelectionResult.selected_features must not contain duplicates")
        if len(set(self.rejected_features)) != len(self.rejected_features):
            raise ValueError("FeatureSelectionResult.rejected_features must not contain duplicates")
        overlap = set(self.selected_features) & set(self.rejected_features)
        if overlap:
            raise ValueError(f"FeatureSelectionResult.selected_features/rejected_features must be disjoint, overlap={sorted(overlap)}")
        if self.training_row_count < 1:
            raise ValueError(f"FeatureSelectionResult.training_row_count must be >= 1, got {self.training_row_count}")
        if self.selector_seed < 0:
            raise ValueError(f"FeatureSelectionResult.selector_seed must be >= 0, got {self.selector_seed}")
        validate_json_primitive_mapping(self.selector_params, field_name="FeatureSelectionResult.selector_params")
        if self.per_feature_score is not None:
            for name, score in self.per_feature_score.items():
                if not math.isfinite(score):
                    raise ValueError(f"FeatureSelectionResult.per_feature_score[{name!r}]={score!r} must be finite")
        if self.stability_frequency is not None:
            for name, freq in self.stability_frequency.items():
                if not (0.0 <= freq <= 1.0):
                    raise ValueError(f"FeatureSelectionResult.stability_frequency[{name!r}]={freq!r} must be in [0, 1]")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "strategy": self.strategy.value,
            "selected_features": list(self.selected_features), "rejected_features": list(self.rejected_features),
            "feature_universe_fingerprint": self.feature_universe_fingerprint,
            "selector_params": dict(sorted(self.selector_params.items())), "selector_seed": self.selector_seed,
            "training_row_count": self.training_row_count, "training_row_first_position": self.training_row_first_position,
            "training_row_last_position": self.training_row_last_position, "training_row_fingerprint": self.training_row_fingerprint,
            "selection_reason": self.selection_reason, "fitted_at": self.fitted_at,
            "per_feature_score": (None if self.per_feature_score is None else dict(sorted(self.per_feature_score.items()))),
            "per_feature_rank": (None if self.per_feature_rank is None else dict(sorted(self.per_feature_rank.items()))),
            "stability_frequency": (None if self.stability_frequency is None else dict(sorted(self.stability_frequency.items()))),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureSelectionResult:
        require_schema_version(raw, supported=FEATURE_SELECTION_SCHEMA_VERSION, context="FeatureSelectionResult")
        per_feature_score_raw = raw.get("per_feature_score")
        per_feature_rank_raw = raw.get("per_feature_rank")
        stability_frequency_raw = raw.get("stability_frequency")
        return cls(
            schema_version=FEATURE_SELECTION_SCHEMA_VERSION, strategy=FeatureSelectionStrategy(raw["strategy"]),
            selected_features=tuple(str(n) for n in as_json_list(raw["selected_features"], field_name="selected_features")),
            rejected_features=tuple(str(n) for n in as_json_list(raw["rejected_features"], field_name="rejected_features")),
            feature_universe_fingerprint=str(raw["feature_universe_fingerprint"]),
            selector_params=as_json_dict(raw.get("selector_params") or {}, field_name="selector_params"),
            selector_seed=int(str(raw["selector_seed"])), training_row_count=int(str(raw["training_row_count"])),
            training_row_first_position=int(str(raw["training_row_first_position"])),
            training_row_last_position=int(str(raw["training_row_last_position"])),
            training_row_fingerprint=str(raw["training_row_fingerprint"]), selection_reason=str(raw["selection_reason"]),
            fitted_at=str(raw["fitted_at"]),
            per_feature_score=(
                None if per_feature_score_raw is None
                else {str(k): float(v) for k, v in as_json_dict(per_feature_score_raw, field_name="per_feature_score").items()}
            ),
            per_feature_rank=(
                None if per_feature_rank_raw is None
                else {str(k): int(v) for k, v in as_json_dict(per_feature_rank_raw, field_name="per_feature_rank").items()}
            ),
            stability_frequency=(
                None if stability_frequency_raw is None
                else {str(k): float(v) for k, v in as_json_dict(stability_frequency_raw, field_name="stability_frequency").items()}
            ),
            warnings=tuple(str(w) for w in as_json_list(raw.get("warnings") or [], field_name="warnings")),
        )


def validate_feature_selection_result(result: FeatureSelectionResult, universe: FeatureUniverse) -> None:
    """Cross-checks a `FeatureSelectionResult` against the `FeatureUniverse`
    it claims to have selected from -- raises `ValueError` (never silently
    accepts) if the fingerprint does not match, if any selected/rejected
    name is not in the universe, or if selected+rejected does not exactly
    partition it."""
    if result.feature_universe_fingerprint != universe.fingerprint:
        raise ValueError(
            f"FeatureSelectionResult.feature_universe_fingerprint {result.feature_universe_fingerprint!r} does "
            f"not match the given FeatureUniverse's fingerprint {universe.fingerprint!r}"
        )
    universe_set = set(universe.feature_names)
    undeclared = (set(result.selected_features) | set(result.rejected_features)) - universe_set
    if undeclared:
        raise ValueError(f"FeatureSelectionResult references feature(s) not in the candidate universe: {sorted(undeclared)}")
    union = set(result.selected_features) | set(result.rejected_features)
    if union != universe_set:
        raise ValueError(
            f"FeatureSelectionResult.selected_features + rejected_features does not exactly partition the "
            f"candidate universe -- missing: {sorted(universe_set - union)}"
        )


def _row_bounds_and_fingerprint(row_positions: np.ndarray) -> tuple[int, int, int, str]:
    positions = [int(p) for p in row_positions.tolist()]
    if not positions:
        raise ValueError("row_positions must not be empty")
    return len(positions), positions[0], positions[-1], fingerprint_json({"positions": positions})


def _stable_order_by_score_desc(names: Sequence[str], scores: Mapping[str, float]) -> list[str]:
    return sorted(names, key=lambda n: (-scores[n], n))


def _ranks_from_scores(scores: Mapping[str, float], names: Sequence[str]) -> dict[str, int]:
    ordered = _stable_order_by_score_desc(names, scores)
    return {name: index + 1 for index, name in enumerate(ordered)}


def _build_result(
    *, strategy: FeatureSelectionStrategy, universe: FeatureUniverse, selected: tuple[str, ...],
    params: Mapping[str, JsonPrimitive], seed: int, row_positions: np.ndarray, reason: str,
    scores: Mapping[str, float] | None = None, ranks: Mapping[str, int] | None = None,
    stability_frequency: Mapping[str, float] | None = None, warnings: tuple[str, ...] = (),
) -> FeatureSelectionResult:
    if not selected:
        raise ValueError(f"{strategy.value}: selection produced an empty feature set for params={dict(params)!r}")
    rejected = tuple(name for name in universe.feature_names if name not in set(selected))
    row_count, first_pos, last_pos, row_fp = _row_bounds_and_fingerprint(row_positions)
    return FeatureSelectionResult(
        schema_version=FEATURE_SELECTION_SCHEMA_VERSION, strategy=strategy, selected_features=selected,
        rejected_features=rejected, feature_universe_fingerprint=universe.fingerprint, selector_params=dict(params),
        selector_seed=seed, training_row_count=row_count, training_row_first_position=first_pos,
        training_row_last_position=last_pos, training_row_fingerprint=row_fp, selection_reason=reason,
        fitted_at=format_utc_timestamp(utc_now()), per_feature_score=scores, per_feature_rank=ranks,
        stability_frequency=stability_frequency, warnings=warnings,
    )


# --------------------------------------------------------------------------
# Strategy 1: NONE
# --------------------------------------------------------------------------
def select_none(*, universe: FeatureUniverse, row_positions: np.ndarray) -> FeatureSelectionResult:
    return _build_result(
        strategy=FeatureSelectionStrategy.NONE, universe=universe, selected=universe.feature_names, params={},
        seed=0, row_positions=row_positions,
        reason="NONE: complete candidate universe retained unconditionally (mandatory scientific control)",
    )


# --------------------------------------------------------------------------
# Strategy 2: VARIANCE_FILTER
# --------------------------------------------------------------------------
def select_variance_filter(
    *, universe: FeatureUniverse, features: pd.DataFrame, row_positions: np.ndarray, params: Mapping[str, JsonPrimitive], seed: int,
) -> FeatureSelectionResult:
    min_variance = float(str(params["min_variance"]))
    ordered = features[list(universe.feature_names)]
    variances = {str(name): float(value) for name, value in ordered.var(ddof=0).items()}
    selected = tuple(name for name in universe.feature_names if variances[name] > min_variance)
    ranks = _ranks_from_scores(variances, universe.feature_names)
    return _build_result(
        strategy=FeatureSelectionStrategy.VARIANCE_FILTER, universe=universe, selected=selected, params=params,
        seed=seed, row_positions=row_positions, scores=variances, ranks=ranks,
        reason=f"VARIANCE_FILTER: retained {len(selected)}/{len(universe.feature_names)} feature(s) with "
        f"population variance > {min_variance} on inner-train",
    )


# --------------------------------------------------------------------------
# Strategy 3: CORRELATION_FILTER
# --------------------------------------------------------------------------
def select_correlation_filter(
    *, universe: FeatureUniverse, features: pd.DataFrame, row_positions: np.ndarray, params: Mapping[str, JsonPrimitive],
    seed: int,  # noqa: ARG001 -- accepted for a uniform strategy-function signature; unused, see docstring (no randomness here)
) -> FeatureSelectionResult:
    """Greedy, order-preserving, label-blind filter: features are visited
    in UNIVERSE order; a feature is rejected iff its absolute Pearson
    correlation with some ALREADY-KEPT feature exceeds the threshold. This
    always keeps the first-encountered member of any highly-correlated
    cluster and is fully deterministic given a fixed universe order -- no
    seed is actually consumed (recorded as 0 for uniformity with every
    other strategy's result shape). A feature with zero variance in this
    inner-train partition has an undefined (NaN) correlation with every
    other feature; NaN is treated as "not correlated" (kept) here -- pair
    with VARIANCE_FILTER first if constant features should also be
    removed."""
    max_abs_correlation = float(str(params["max_abs_correlation"]))
    ordered = features[list(universe.feature_names)]
    corr = ordered.corr().abs()
    kept: list[str] = []
    for name in universe.feature_names:
        if not kept:
            kept.append(name)
            continue
        max_corr_with_kept = corr.loc[[name], kept].to_numpy().max()
        if pd.isna(max_corr_with_kept) or float(max_corr_with_kept) <= max_abs_correlation:
            kept.append(name)
    selected = tuple(name for name in universe.feature_names if name in set(kept))
    return _build_result(
        strategy=FeatureSelectionStrategy.CORRELATION_FILTER, universe=universe, selected=selected, params=params,
        seed=0, row_positions=row_positions,
        reason=f"CORRELATION_FILTER: retained {len(selected)}/{len(universe.feature_names)} feature(s) after "
        f"greedy pairwise-correlation pruning at |r| <= {max_abs_correlation} on inner-train",
    )


# --------------------------------------------------------------------------
# Strategy 4: UNIVARIATE
# --------------------------------------------------------------------------
def _require_matching_length(features: pd.DataFrame, labels: pd.Series, *, strategy_name: str) -> None:
    """Fails closed with a clear, actionable message -- never a raw
    pandas/numpy `IndexError` several stack frames deep -- for the three
    strategies that actually read `labels` (`select_univariate`,
    `select_model_native_importance`, `select_stability`)."""
    if len(features) != len(labels):
        raise ValueError(f"{strategy_name}: features ({len(features)} rows) and labels ({len(labels)} rows) must have matching length")


def select_univariate(
    *, universe: FeatureUniverse, features: pd.DataFrame, labels: pd.Series, row_positions: np.ndarray,
    params: Mapping[str, JsonPrimitive], seed: int, objective: ObjectiveType,
) -> FeatureSelectionResult:
    _require_matching_length(features, labels, strategy_name="UNIVARIATE")
    ordered = features[list(universe.feature_names)]
    matrix = ordered.to_numpy(dtype="float64")
    y = labels.to_numpy()
    if objective is ObjectiveType.REGRESSION:
        raw_scores = mutual_info_regression(matrix, y, random_state=seed)
    else:
        raw_scores = mutual_info_classif(matrix, y, random_state=seed)
    scores = dict(zip(universe.feature_names, (float(s) for s in raw_scores), strict=True))
    ranked_names = _stable_order_by_score_desc(universe.feature_names, scores)

    mode = str(params.get("mode", "top_k"))
    if mode == "top_k":
        k = min(int(str(params["k"])), len(universe.feature_names))
        selected_set = set(ranked_names[:k])
    else:
        percentile = float(str(params["percentile"]))
        count = max(1, math.ceil(len(universe.feature_names) * percentile / 100.0))
        selected_set = set(ranked_names[:count])
    selected = tuple(name for name in universe.feature_names if name in selected_set)
    ranks = _ranks_from_scores(scores, universe.feature_names)
    return _build_result(
        strategy=FeatureSelectionStrategy.UNIVARIATE, universe=universe, selected=selected, params=params, seed=seed,
        row_positions=row_positions, scores=scores, ranks=ranks,
        reason=f"UNIVARIATE (mutual information, mode={mode!r}): retained {len(selected)}/{len(universe.feature_names)} "
        "feature(s) fitted on inner-train",
    )


# --------------------------------------------------------------------------
# Strategy 5: MODEL_NATIVE_IMPORTANCE
# --------------------------------------------------------------------------
def select_model_native_importance(
    *, universe: FeatureUniverse, features: pd.DataFrame, labels: pd.Series, row_positions: np.ndarray,
    params: Mapping[str, JsonPrimitive], seed: int, model_name: str, model_factory: ModelFactory,
    hyperparameters: ModelHyperparameters, objective: ObjectiveType,
) -> FeatureSelectionResult:
    if model_name not in MODEL_NATIVE_IMPORTANCE_SUPPORTED_MODELS:
        raise ValueError(
            f"MODEL_NATIVE_IMPORTANCE selection is not supported for model {model_name!r} -- supported model "
            f"families: {MODEL_NATIVE_IMPORTANCE_SUPPORTED_MODELS}"
        )
    _require_matching_length(features, labels, strategy_name="MODEL_NATIVE_IMPORTANCE")
    ordered = features[list(universe.feature_names)]
    feature_schema = FeatureSchema(feature_names=universe.feature_names)
    selector_model = model_factory.create(hyperparameters=hyperparameters, feature_schema=feature_schema, objective=objective)
    fitted = selector_model.fit(ordered, labels, seeds=SeedConfiguration(master_seed=seed))
    importance_fn = getattr(fitted, "feature_importance", None)
    if importance_fn is None:
        raise ValueError(f"Fitted model {model_name!r} does not expose a feature_importance() method")
    raw_scores = importance_fn()
    scores = {str(name): float(value) for name, value in raw_scores.items()}

    mode = str(params.get("mode"))
    ranked_names = _stable_order_by_score_desc(universe.feature_names, scores)
    if mode == "top_k":
        k = min(int(str(params["k"])), len(universe.feature_names))
        selected_set = set(ranked_names[:k])
    else:
        min_importance = float(str(params["min_importance"]))
        selected_set = {name for name in universe.feature_names if scores[name] > min_importance}
    selected = tuple(name for name in universe.feature_names if name in selected_set)
    ranks = _ranks_from_scores(scores, universe.feature_names)
    return _build_result(
        strategy=FeatureSelectionStrategy.MODEL_NATIVE_IMPORTANCE, universe=universe, selected=selected, params=params,
        seed=seed, row_positions=row_positions, scores=scores, ranks=ranks,
        reason=f"MODEL_NATIVE_IMPORTANCE ({model_name}, mode={mode!r}): retained {len(selected)}/"
        f"{len(universe.feature_names)} feature(s), selector model trained on inner-train only",
    )


# --------------------------------------------------------------------------
# Strategy 6: STABILITY_SELECTION
# --------------------------------------------------------------------------
def select_stability(
    *, universe: FeatureUniverse, features: pd.DataFrame, labels: pd.Series, row_positions: np.ndarray,
    params: Mapping[str, JsonPrimitive], seed: int, objective: ObjectiveType,
) -> FeatureSelectionResult:
    _require_matching_length(features, labels, strategy_name="STABILITY_SELECTION")
    base_strategy = str(params["base_strategy"])
    n_repeats = int(str(params["n_repeats"]))
    subsample_fraction = float(str(params.get("subsample_fraction", 0.8)))
    min_frequency = float(str(params["min_frequency"]))
    base_params = {
        k: v for k, v in params.items() if k not in {"base_strategy", "n_repeats", "subsample_fraction", "min_frequency"}
    }

    n = len(features)
    subsample_size = max(1, round(n * subsample_fraction))
    counts = dict.fromkeys(universe.feature_names, 0)
    for repeat in range(n_repeats):
        repeat_seed = derive_seed(seed, f"stability_repeat:{repeat}")
        rng = np.random.default_rng(repeat_seed)
        subsample_positions = np.sort(rng.choice(n, size=subsample_size, replace=False))
        sub_features = features.iloc[subsample_positions]
        sub_labels = labels.iloc[subsample_positions]
        sub_row_positions = row_positions[subsample_positions]

        if base_strategy == "variance_filter":
            sub_result = select_variance_filter(
                universe=universe, features=sub_features, row_positions=sub_row_positions, params=base_params, seed=repeat_seed,
            )
        elif base_strategy == "correlation_filter":
            sub_result = select_correlation_filter(
                universe=universe, features=sub_features, row_positions=sub_row_positions, params=base_params, seed=repeat_seed,
            )
        else:
            sub_result = select_univariate(
                universe=universe, features=sub_features, labels=sub_labels, row_positions=sub_row_positions,
                params=base_params, seed=repeat_seed, objective=objective,
            )
        for name in sub_result.selected_features:
            counts[name] += 1

    frequencies = {name: counts[name] / n_repeats for name in universe.feature_names}
    selected = tuple(name for name in universe.feature_names if frequencies[name] >= min_frequency)
    return _build_result(
        strategy=FeatureSelectionStrategy.STABILITY_SELECTION, universe=universe, selected=selected, params=params,
        seed=seed, row_positions=row_positions, stability_frequency=frequencies,
        reason=f"STABILITY_SELECTION (base={base_strategy!r}, n_repeats={n_repeats}, "
        f"subsample_fraction={subsample_fraction}): retained {len(selected)}/{len(universe.feature_names)} "
        f"feature(s) with selection frequency >= {min_frequency} across bootstrap repeats confined to inner-train",
    )


def run_feature_selection(
    spec: FeatureSelectionSpec, *, universe: FeatureUniverse, features: pd.DataFrame, labels: pd.Series,
    row_positions: np.ndarray, seed: int, objective: ObjectiveType, model_name: str | None = None,
    model_factory: ModelFactory | None = None, hyperparameters: ModelHyperparameters | None = None,
) -> FeatureSelectionResult:
    """The one dispatch entry point `optimization.trial_executor` calls,
    fitted fresh inside every inner fold -- see module docstring."""
    if spec.strategy is FeatureSelectionStrategy.NONE:
        return select_none(universe=universe, row_positions=row_positions)
    if spec.strategy is FeatureSelectionStrategy.VARIANCE_FILTER:
        return select_variance_filter(universe=universe, features=features, row_positions=row_positions, params=spec.params, seed=seed)
    if spec.strategy is FeatureSelectionStrategy.CORRELATION_FILTER:
        return select_correlation_filter(universe=universe, features=features, row_positions=row_positions, params=spec.params, seed=seed)
    if spec.strategy is FeatureSelectionStrategy.UNIVARIATE:
        return select_univariate(universe=universe, features=features, labels=labels, row_positions=row_positions, params=spec.params, seed=seed, objective=objective)
    if spec.strategy is FeatureSelectionStrategy.MODEL_NATIVE_IMPORTANCE:
        if model_name is None or model_factory is None or hyperparameters is None:
            raise ValueError("MODEL_NATIVE_IMPORTANCE requires model_name, model_factory, and hyperparameters")
        return select_model_native_importance(
            universe=universe, features=features, labels=labels, row_positions=row_positions, params=spec.params,
            seed=seed, model_name=model_name, model_factory=model_factory, hyperparameters=hyperparameters, objective=objective,
        )
    if spec.strategy is FeatureSelectionStrategy.STABILITY_SELECTION:
        return select_stability(universe=universe, features=features, labels=labels, row_positions=row_positions, params=spec.params, seed=seed, objective=objective)
    raise ValueError(f"Unknown FeatureSelectionStrategy {spec.strategy!r}")  # pragma: no cover - exhaustive over the enum


__all__ = [
    "FEATURE_SELECTION_SCHEMA_VERSION",
    "MAX_STABILITY_REPEATS",
    "MODEL_NATIVE_IMPORTANCE_SUPPORTED_MODELS",
    "FeatureSelectionResult",
    "FeatureSelectionSpec",
    "FeatureSelectionStrategy",
    "FeatureUniverse",
    "run_feature_selection",
    "select_correlation_filter",
    "select_model_native_importance",
    "select_none",
    "select_stability",
    "select_univariate",
    "select_variance_filter",
    "validate_feature_selection_result",
]
