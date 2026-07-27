"""Typed, serializable hyperparameter search-space definitions (Milestone
4D). "Do not pass raw Optuna lambdas or arbitrary Python callables as the
durable search-space definition" -- every parameter a trial can sample is
one of exactly five closed, JSON-round-trippable shapes (`IntegerParameter`,
`FloatParameter`, `CategoricalParameter`, `BooleanParameter`,
`FixedParameter`), never an opaque closure. This module is deliberately
Optuna-free: it has no dependency on the `optuna` package at all, so a
`SearchSpace` can be constructed, validated, canonicalized, and
fingerprinted without that (or any) search backend installed.
`optimization.study` is the one place that translates a `SearchSpace` into
concrete Optuna suggestions -- kept as a separate, thin adapter so this
module's own tests never need Optuna, and so a future alternative search
backend would only need a new adapter, not a change here.

WHY NO CONDITIONAL-ACTIVATION MODEL IN THIS MILESTONE
--------------------------------------------------------------------------
The spec allows a "typed restricted condition model if necessary". None of
the model-specific search spaces this milestone defines (LightGBM/XGBoost/
CatBoost/baselines) need one -- every declared parameter is always sampled
independently of every other. Adding conditional-activation machinery
nothing yet exercises would be exactly the kind of speculative complexity
this platform's own engineering conventions reject; if a future model
genuinely needs "parameter B only matters when A == x", add a minimal
typed condition then, backed by a real test that needs it.

CANONICALIZATION AND FINGERPRINTING
--------------------------------------------------------------------------
`SearchSpace.fingerprint()` hashes the canonical JSON encoding of the
ordered parameter list (`ml.fingerprints.fingerprint_json`, the same
sha256-of-canonical-JSON every other fingerprint in this platform uses).
Parameter ORDER is part of the canonical payload (never sorted out from
under the caller) so two search spaces that declare the same parameters
in a different order are -- correctly -- different fingerprints unless a
caller builds them identically; `OptimizationSpec`'s own identity payload
uses this fingerprint, never a re-serialization of the whole space.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from quant_platform.ml.fingerprints import fingerprint_json
from quant_platform.ml.models import JsonPrimitive, validate_json_primitive_mapping
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

SEARCH_SPACE_SCHEMA_VERSION = 1


class ParameterKind(Enum):
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    FIXED = "fixed"


def _validate_name(name: str, *, field_name: str = "name") -> None:
    if not name:
        raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True, slots=True)
class IntegerParameter:
    name: str
    low: int
    high: int
    step: int = 1
    log: bool = False

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if self.low > self.high:
            raise ValueError(f"IntegerParameter {self.name!r}: low ({self.low}) must be <= high ({self.high})")
        if self.step <= 0:
            raise ValueError(f"IntegerParameter {self.name!r}: step must be positive, got {self.step}")
        if self.log and self.low <= 0:
            raise ValueError(f"IntegerParameter {self.name!r}: log=True requires a strictly positive lower bound, got low={self.low}")
        if self.log and self.step != 1:
            raise ValueError(f"IntegerParameter {self.name!r}: log=True is only supported with step=1, got step={self.step}")

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": ParameterKind.INTEGER.value, "name": self.name, "low": self.low, "high": self.high, "step": self.step, "log": self.log}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> IntegerParameter:
        return cls(name=str(raw["name"]), low=int(str(raw["low"])), high=int(str(raw["high"])), step=int(str(raw.get("step", 1))), log=bool(raw.get("log", False)))


@dataclass(frozen=True, slots=True)
class FloatParameter:
    name: str
    low: float
    high: float
    step: float | None = None
    log: bool = False

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError(f"FloatParameter {self.name!r}: low/high must be finite")
        if self.low > self.high:
            raise ValueError(f"FloatParameter {self.name!r}: low ({self.low}) must be <= high ({self.high})")
        if self.step is not None:
            if not math.isfinite(self.step) or self.step <= 0:
                raise ValueError(f"FloatParameter {self.name!r}: step must be a finite positive number, got {self.step}")
            if self.log:
                raise ValueError(f"FloatParameter {self.name!r}: log=True cannot be combined with a discretization step")
        if self.log and self.low <= 0:
            raise ValueError(f"FloatParameter {self.name!r}: log=True requires a strictly positive lower bound, got low={self.low}")

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": ParameterKind.FLOAT.value, "name": self.name, "low": self.low, "high": self.high, "step": self.step, "log": self.log}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FloatParameter:
        step_raw = raw.get("step")
        return cls(name=str(raw["name"]), low=float(str(raw["low"])), high=float(str(raw["high"])), step=(None if step_raw is None else float(str(step_raw))), log=bool(raw.get("log", False)))


@dataclass(frozen=True, slots=True)
class CategoricalParameter:
    name: str
    choices: tuple[JsonPrimitive, ...]

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if not self.choices:
            raise ValueError(f"CategoricalParameter {self.name!r}: choices must not be empty")
        # Type/finiteness validated BEFORE the duplicate check: `set(...)`
        # raises an unhandled `TypeError` for an unhashable choice (e.g. a
        # nested dict/list smuggled past the `JsonPrimitive` type hint at
        # runtime) -- checking shape first guarantees every choice is
        # already hashable by the time `set()` is ever called, so a
        # malformed choice always fails with this module's own clear
        # `ValueError`, never a raw `TypeError`.
        for choice in self.choices:
            if isinstance(choice, bool) or choice is None or isinstance(choice, (str, int)):
                continue
            if isinstance(choice, float):
                if not math.isfinite(choice):
                    raise ValueError(f"CategoricalParameter {self.name!r}: choice {choice!r} must be finite")
                continue
            raise ValueError(f"CategoricalParameter {self.name!r}: choice {choice!r} is not a JSON primitive")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError(f"CategoricalParameter {self.name!r}: choices must not contain duplicates, got {self.choices}")

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": ParameterKind.CATEGORICAL.value, "name": self.name, "choices": list(self.choices)}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CategoricalParameter:
        return cls(name=str(raw["name"]), choices=tuple(as_json_list(raw["choices"], field_name="choices")))


@dataclass(frozen=True, slots=True)
class BooleanParameter:
    name: str

    def __post_init__(self) -> None:
        _validate_name(self.name)

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": ParameterKind.BOOLEAN.value, "name": self.name}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BooleanParameter:
        return cls(name=str(raw["name"]))


@dataclass(frozen=True, slots=True)
class FixedParameter:
    """A parameter that is part of the trial's hyperparameter dict but is
    never actually searched over -- always resolves to the same declared
    value. Used both for baseline models (no tunable space at all, only
    "explicitly meaningful fixed parameters") and for pinning a real
    model's platform-reserved keys (e.g. a fixed `bagging_freq` so
    LightGBM's sampled `subsample` alias actually takes effect)."""

    name: str
    value: JsonPrimitive

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError(f"FixedParameter {self.name!r}: value must be finite, got {self.value!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": ParameterKind.FIXED.value, "name": self.name, "value": self.value}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FixedParameter:
        return cls(name=str(raw["name"]), value=raw.get("value"))  # type: ignore[arg-type]


ParameterDefinition = IntegerParameter | FloatParameter | CategoricalParameter | BooleanParameter | FixedParameter

_KIND_DECODERS: dict[str, type[ParameterDefinition]] = {
    ParameterKind.INTEGER.value: IntegerParameter,
    ParameterKind.FLOAT.value: FloatParameter,
    ParameterKind.CATEGORICAL.value: CategoricalParameter,
    ParameterKind.BOOLEAN.value: BooleanParameter,
    ParameterKind.FIXED.value: FixedParameter,
}


def _decode_parameter(raw: object) -> ParameterDefinition:
    payload = as_json_dict(raw, field_name="SearchSpace.parameters[]")
    kind = str(payload.get("kind"))
    decoder = _KIND_DECODERS.get(kind)
    if decoder is None:
        raise ValueError(f"Unknown search-space parameter kind {kind!r}")
    return decoder.from_json_dict(payload)


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """An ordered, immutable collection of parameter definitions -- the
    complete, durable description of what a sampler is allowed to
    propose. Order is part of the canonical identity (see module
    docstring); two spaces with the same parameters in a different order
    fingerprint differently."""

    schema_version: int
    parameters: tuple[ParameterDefinition, ...]

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("SearchSpace.parameters must not be empty")
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"SearchSpace.parameters must have unique names, got {names}")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "parameters": [p.to_json_dict() for p in self.parameters]}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SearchSpace:
        require_schema_version(raw, supported=SEARCH_SPACE_SCHEMA_VERSION, context="SearchSpace")
        return cls(
            schema_version=SEARCH_SPACE_SCHEMA_VERSION,
            parameters=tuple(_decode_parameter(p) for p in as_json_list(raw["parameters"], field_name="parameters")),
        )

    def fingerprint(self) -> str:
        return fingerprint_json(self.to_json_dict())


def build_search_space(parameters: list[ParameterDefinition]) -> SearchSpace:
    return SearchSpace(schema_version=SEARCH_SPACE_SCHEMA_VERSION, parameters=tuple(parameters))


def validate_sampled_values(space: SearchSpace, values: dict[str, JsonPrimitive]) -> None:
    """Defense-in-depth: confirms a sampled value dict actually matches
    `space` -- every declared name present, no undeclared name, and every
    value within its parameter's own declared bounds/choices. Called by
    `optimization.study` immediately after asking a sampler for a
    suggestion, and by `optimization.candidates` before a `TrialSpec` is
    considered valid -- a sampler bug or a corrupted resumed trial must
    never silently proceed with an out-of-space value."""
    validate_json_primitive_mapping(values, field_name="sampled values")
    declared = set(space.parameter_names)
    given = set(values)
    missing = declared - given
    if missing:
        raise ValueError(f"Sampled values are missing declared parameter(s): {sorted(missing)}")
    extra = given - declared
    if extra:
        raise ValueError(f"Sampled values contain undeclared parameter(s): {sorted(extra)}")
    for parameter in space.parameters:
        value = values[parameter.name]
        _validate_one_value(parameter, value)


def _validate_one_value(parameter: ParameterDefinition, value: JsonPrimitive) -> None:
    if isinstance(parameter, IntegerParameter):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Parameter {parameter.name!r} expects an int, got {value!r}")
        if not (parameter.low <= value <= parameter.high):
            raise ValueError(f"Parameter {parameter.name!r} value {value} is outside [{parameter.low}, {parameter.high}]")
        if (value - parameter.low) % parameter.step != 0:
            raise ValueError(f"Parameter {parameter.name!r} value {value} is not reachable from low={parameter.low} with step={parameter.step}")
    elif isinstance(parameter, FloatParameter):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Parameter {parameter.name!r} expects a float, got {value!r}")
        numeric = float(value)
        if not (parameter.low <= numeric <= parameter.high):
            raise ValueError(f"Parameter {parameter.name!r} value {numeric} is outside [{parameter.low}, {parameter.high}]")
    elif isinstance(parameter, CategoricalParameter):
        if value not in parameter.choices:
            raise ValueError(f"Parameter {parameter.name!r} value {value!r} is not one of {parameter.choices}")
    elif isinstance(parameter, BooleanParameter):
        if not isinstance(value, bool):
            raise ValueError(f"Parameter {parameter.name!r} expects a bool, got {value!r}")
    elif isinstance(parameter, FixedParameter):
        if value != parameter.value:
            raise ValueError(f"Parameter {parameter.name!r} is fixed at {parameter.value!r}, got {value!r}")
    else:  # pragma: no cover - exhaustive over ParameterDefinition's closed union
        raise TypeError(f"Unknown parameter definition type {type(parameter).__name__}")


# --------------------------------------------------------------------------
# Model-specific default search spaces
# --------------------------------------------------------------------------
# Boosting-round-count key names are each model wrapper's OWN recognized
# hyperparameter key (see `ml.model_zoo.lightgbm_model`/`xgboost_model`/
# `catboost_model`'s `_split_hyperparameters` -- these are popped out of
# the hyperparameter dict before the rest is forwarded to the underlying
# library, never passed through as a native library parameter). Every
# other name below is the literal library parameter name (resolved
# through the library's own alias table where the platform's wrapper
# forwards it unchanged, e.g. LightGBM's `subsample`/`colsample_bytree`/
# `reg_alpha`/`reg_lambda` aliases for `bagging_fraction`/`feature_fraction`/
# `lambda_l1`/`lambda_l2`).
LIGHTGBM_MODEL_NAME = "lightgbm"
XGBOOST_MODEL_NAME = "xgboost"
CATBOOST_MODEL_NAME = "catboost"


def lightgbm_default_search_space() -> SearchSpace:
    return build_search_space([
        FloatParameter(name="learning_rate", low=0.01, high=0.3, log=True),
        IntegerParameter(name="num_leaves", low=8, high=256),
        IntegerParameter(name="max_depth", low=-1, high=16),
        IntegerParameter(name="min_child_samples", low=5, high=100),
        FloatParameter(name="subsample", low=0.5, high=1.0),
        FloatParameter(name="colsample_bytree", low=0.5, high=1.0),
        FloatParameter(name="reg_alpha", low=0.0, high=5.0),
        FloatParameter(name="reg_lambda", low=0.0, high=5.0),
        IntegerParameter(name="num_boost_round", low=50, high=500),
        # `bagging_freq` must be > 0 for LightGBM's `bagging_fraction`
        # (aliased here as `subsample`) to have any effect at all --
        # fixed rather than searched, since it is a mechanism switch, not
        # a meaningful tunable quantity on its own.
        FixedParameter(name="bagging_freq", value=1),
    ])


def xgboost_default_search_space() -> SearchSpace:
    return build_search_space([
        FloatParameter(name="learning_rate", low=0.01, high=0.3, log=True),
        IntegerParameter(name="max_depth", low=2, high=12),
        FloatParameter(name="min_child_weight", low=1.0, high=10.0),
        FloatParameter(name="subsample", low=0.5, high=1.0),
        FloatParameter(name="colsample_bytree", low=0.5, high=1.0),
        FloatParameter(name="gamma", low=0.0, high=5.0),
        FloatParameter(name="reg_alpha", low=0.0, high=5.0),
        FloatParameter(name="reg_lambda", low=0.1, high=5.0, log=True),
        IntegerParameter(name="num_boost_round", low=50, high=500),
    ])


def catboost_default_search_space() -> SearchSpace:
    return build_search_space([
        FloatParameter(name="learning_rate", low=0.01, high=0.3, log=True),
        IntegerParameter(name="depth", low=4, high=10),
        FloatParameter(name="l2_leaf_reg", low=1.0, high=10.0),
        FloatParameter(name="random_strength", low=0.0, high=10.0),
        FloatParameter(name="bagging_temperature", low=0.0, high=5.0),
        IntegerParameter(name="iterations", low=50, high=500),
    ])


def baseline_fixed_search_space(fixed_values: dict[str, JsonPrimitive]) -> SearchSpace:
    """"Baselines (no optimization except explicitly meaningful fixed
    parameters)" -- every parameter is a `FixedParameter`, so a baseline
    still flows through the exact same trial-execution/ranking machinery
    (as a single-trial "search"), never a special-cased code path."""
    if not fixed_values:
        raise ValueError("baseline_fixed_search_space requires at least one fixed parameter")
    return build_search_space([FixedParameter(name=name, value=value) for name, value in fixed_values.items()])


_DEFAULT_SEARCH_SPACES: dict[str, Callable[[], SearchSpace]] = {
    LIGHTGBM_MODEL_NAME: lightgbm_default_search_space,
    XGBOOST_MODEL_NAME: xgboost_default_search_space,
    CATBOOST_MODEL_NAME: catboost_default_search_space,
}


def default_search_space_for_model(model_name: str) -> SearchSpace:
    """The one place that maps a registered model name to its declared
    default search space. Raises for any model this milestone does not
    define a default for (logistic_regression/elastic_net -- excluded by
    the fail-closed preprocessing policy, see `optimization.models`'
    module docstring; every baseline -- use `baseline_fixed_search_space`
    explicitly instead, since "fixed parameters" are inherently caller-
    declared, not a single universal default)."""
    factory = _DEFAULT_SEARCH_SPACES.get(model_name)
    if factory is None:
        raise ValueError(
            f"No default search space is defined for model {model_name!r} -- supported: "
            f"{sorted(_DEFAULT_SEARCH_SPACES)} (baselines must use baseline_fixed_search_space explicitly)"
        )
    return factory()


__all__ = [
    "CATBOOST_MODEL_NAME",
    "LIGHTGBM_MODEL_NAME",
    "SEARCH_SPACE_SCHEMA_VERSION",
    "XGBOOST_MODEL_NAME",
    "BooleanParameter",
    "CategoricalParameter",
    "FixedParameter",
    "FloatParameter",
    "IntegerParameter",
    "ParameterDefinition",
    "ParameterKind",
    "SearchSpace",
    "baseline_fixed_search_space",
    "build_search_space",
    "catboost_default_search_space",
    "default_search_space_for_model",
    "lightgbm_default_search_space",
    "validate_sampled_values",
    "xgboost_default_search_space",
]
