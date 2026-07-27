"""Shared fixtures/builders for the Milestone 4D optimization engine test
suite. Mirrors `tests.unit.ml.conftest`'s exact "hand-built, lightweight
fixture objects for unit tests" philosophy -- real datasets are only
built where a test genuinely needs one (see `tests/integration/
test_optimization_engine.py`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.unit.ml.conftest import make_experiment_spec_kwargs

from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.fingerprints import fingerprint_json
from quant_platform.ml.models import FeatureBinding
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.optimization.feature_selection import (
    FeatureSelectionSpec,
    FeatureSelectionStrategy,
    FeatureUniverse,
)
from quant_platform.optimization.inner_splits import InnerSplitConfig
from quant_platform.optimization.models import (
    EarlyStoppingConfig,
    OptimizationSpec,
    PruningConfig,
    PruningKind,
    SamplerKind,
    build_optimization_spec,
)
from quant_platform.optimization.search_space import lightgbm_default_search_space

PARENT_EXPERIMENT_ID = "d" * 64
DEFAULT_FEATURE_NAMES = ("f1", "f2", "f3", "f4", "f5", "f6")


def make_experiment_spec(*, feature_names: tuple[str, ...] | None = None, **overrides: object) -> ExperimentSpec:
    kwargs = make_experiment_spec_kwargs(**overrides)
    if feature_names is not None and "feature_binding" not in overrides:
        kwargs["feature_binding"] = FeatureBinding(
            feature_names=feature_names, feature_versions=dict.fromkeys(feature_names, "1"),
            feature_registry_fingerprint="b" * 64,
        )
    return ExperimentSpec(**kwargs)  # type: ignore[arg-type]


def make_optimization_spec(*, experiment: ExperimentSpec | None = None, **overrides: object) -> OptimizationSpec:
    experiment = experiment if experiment is not None else make_experiment_spec(feature_names=DEFAULT_FEATURE_NAMES)
    primary_metric = "rmse" if experiment.objective.value == "regression" else "accuracy"
    base: dict[str, object] = {
        "experiment": experiment, "parent_experiment_id": PARENT_EXPERIMENT_ID, "model_name": "lightgbm", "model_version": "1",
        "primary_metric": primary_metric,
        "inner_split_config": InnerSplitConfig(strategy="expanding_walk_forward", n_splits=2, test_size_fraction=0.2),
        "feature_selection_spec": FeatureSelectionSpec(strategy=FeatureSelectionStrategy.NONE),
        "search_space": lightgbm_default_search_space(), "sampler_kind": SamplerKind.TPE,
        "pruning_config": PruningConfig(kind=PruningKind.NONE), "early_stopping_config": EarlyStoppingConfig(enabled=False),
        "max_trials": 5, "min_successful_inner_folds": 1, "seed_configuration": SeedConfiguration(master_seed=7),
    }
    base.update(overrides)
    return build_optimization_spec(**base)  # type: ignore[arg-type]


def make_feature_universe(names: tuple[str, ...] = DEFAULT_FEATURE_NAMES) -> FeatureUniverse:
    fp = fingerprint_json({"feature_names": list(names), "feature_registry_fingerprint": "b" * 64})
    return FeatureUniverse(feature_names=names, fingerprint=fp)


def make_feature_frame(n_rows: int = 300, n_features: int = 6, *, seed: int = 0, informative: bool = False, label: pd.Series | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns: dict[str, np.ndarray] = {}
    for i in range(n_features):
        base = rng.normal(size=n_rows)
        if informative and label is not None and i == 0:
            base = base + label.to_numpy() * 3.0
        columns[f"f{i + 1}"] = base
    return pd.DataFrame(columns)


def make_binary_labels(n_rows: int = 300, *, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(0, 2, size=n_rows).astype(float), name="label")


def make_row_positions(n_rows: int = 300, *, start: int = 0) -> np.ndarray:
    return np.arange(start, start + n_rows, dtype=np.int64)


@pytest.fixture
def experiment_spec() -> ExperimentSpec:
    return make_experiment_spec(feature_names=DEFAULT_FEATURE_NAMES)


@pytest.fixture
def optimization_spec(experiment_spec: ExperimentSpec) -> OptimizationSpec:
    return make_optimization_spec(experiment=experiment_spec)


@pytest.fixture
def feature_universe() -> FeatureUniverse:
    return make_feature_universe()
