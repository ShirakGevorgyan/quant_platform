"""Performance benchmarks for Milestone 4C's `ml.model_zoo` predictive
models, `ml.metrics`, and `ml.comparison`. Same philosophy as
`test_execution_throughput.py`/`test_ml_infrastructure_throughput.py`:
conservative floors (roughly 10x-100x below measured numbers on
reference hardware) to catch a severe accidental regression without
being flaky on a slower CI runner -- these are NOT production throughput
guarantees, and no correctness work (seed propagation, native
serialization, pre-fit validation) is ever skipped to make a number
look better.

Measured on reference hardware (informational; one real run of this
file's own benchmarks, Windows 11 / NTFS; expect run-to-run variance of
at least +/-30%, and considerably more for the gradient-boosting
libraries, whose fit cost is dominated by native (non-Python) internals
this suite does not control):
  - `LightGBMModel.fit` (2000 rows, 10 features, 50 rounds), 20 iter:
    20.6ms/iter median, p95 26.4ms.
  - `XGBoostModel.fit` (same shape), 20 iter: 25.4ms/iter median, p95
    94.7ms (XGBoost showed the widest run-to-run spread of the three).
  - `CatBoostModel.fit` (same shape, 50 iterations), 20 iter: 172.9ms/
    iter median, p95 292.5ms -- clearly the slowest fit of the three
    gradient-boosting libraries at this scale.
  - `LogisticRegressionModel.fit`, 50 iter: 2.5ms/iter median.
    `ElasticNetModel.fit`, 50 iter: 0.7ms/iter median.
  - Fitted `predict()` (500-row batch), 200 iter: LightGBM 1.08ms/iter,
    XGBoost 1.52ms/iter, CatBoost 2.58ms/iter median.
  - Native serialize+deserialize round trip, 50 iter: LightGBM 0.20ms,
    XGBoost 0.19ms, CatBoost 0.16ms/iter median -- CatBoost's temp-file-
    backed `.cbm` round trip is, perhaps surprisingly, not the slowest
    of the three at this model size (OS page-cache-warm temp-file I/O
    is cheap relative to the text/binary encoding the other two do).
  - `compute_classification_metrics` (1000 rows, with proba), 1000 iter:
    5.79ms/iter median (ROC AUC/PR AUC's O(n log n) sort dominates).
    `compute_regression_metrics` (1000 rows), 1000 iter: 0.58ms/iter
    median (no such sort).
  - `compare_to_baselines` (1 candidate vs. 4 baselines, 8 folds, 2
    metrics each), 200 iter: 3.78ms/iter median (dominated by 8 calls
    to `scipy.stats.wilcoxon`, not this module's own code).
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pandas as pd
import pytest

from quant_platform.ml.comparison import ModelFoldMetrics, compare_to_baselines
from quant_platform.ml.interfaces import FeatureSchema
from quant_platform.ml.metrics import compute_classification_metrics, compute_regression_metrics
from quant_platform.ml.model_zoo import catboost_model as cb_wrapper
from quant_platform.ml.model_zoo import lightgbm_model as lgbm_wrapper
from quant_platform.ml.model_zoo import linear as linear_wrapper
from quant_platform.ml.model_zoo import xgboost_model as xgb_wrapper
from quant_platform.ml.models import ModelHyperparameters, ObjectiveType
from quant_platform.ml.seeds import SeedConfiguration

pytestmark = pytest.mark.performance

_N_ROWS = 2000
_FEATURE_NAMES = tuple(f"f{i}" for i in range(10))
_SCHEMA = FeatureSchema(feature_names=_FEATURE_NAMES)


def _timed_iterations(fn, count: int) -> list[float]:
    timings = []
    for _ in range(count):
        started = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - started)
    return timings


def _report(label: str, timings: list[float]) -> float:
    median = statistics.median(timings)
    rate = 1.0 / median if median > 0 else float("inf")
    print(f"\n{label}: n={len(timings)} median={median * 1000:.3f}ms p95={sorted(timings)[int(len(timings) * 0.95)] * 1000:.3f}ms rate={rate:,.0f}/sec")
    return median


def _features(n: int = _N_ROWS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({name: rng.normal(size=n) for name in _FEATURE_NAMES})


def _classification_labels(features: pd.DataFrame) -> pd.Series:
    signal = sum(features[name] for name in _FEATURE_NAMES[:3])
    return pd.Series((signal > 0).astype(int))


def _regression_labels(features: pd.DataFrame) -> pd.Series:
    signal = sum(features[name] for name in _FEATURE_NAMES[:3])
    return pd.Series(signal + 0.1)


class TestModelFitThroughput:
    def test_lightgbm_fit(self) -> None:
        features, labels = _features(), None
        labels = _classification_labels(features)
        factory = lgbm_wrapper.LightGBMModelFactory()

        def fit() -> None:
            model = factory.create(
                hyperparameters=ModelHyperparameters(values={"num_boost_round": 50, "num_leaves": 15}),
                feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
            )
            model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))

        median = _report(f"LightGBMModel.fit ({_N_ROWS} rows, {len(_FEATURE_NAMES)} features, 50 rounds)", _timed_iterations(fit, 20))
        assert median < 1.0, "fitting 50-round LightGBM on 2000x10 should not take >1s (conservative floor)"

    def test_xgboost_fit(self) -> None:
        features = _features()
        labels = _classification_labels(features)
        factory = xgb_wrapper.XGBoostModelFactory()

        def fit() -> None:
            model = factory.create(
                hyperparameters=ModelHyperparameters(values={"num_boost_round": 50, "max_depth": 4}),
                feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
            )
            model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))

        median = _report(f"XGBoostModel.fit ({_N_ROWS} rows, {len(_FEATURE_NAMES)} features, 50 rounds)", _timed_iterations(fit, 20))
        assert median < 1.0, "fitting 50-round XGBoost on 2000x10 should not take >1s (conservative floor)"

    def test_catboost_fit(self) -> None:
        features = _features()
        labels = _classification_labels(features)
        factory = cb_wrapper.CatBoostModelFactory()

        def fit() -> None:
            model = factory.create(
                hyperparameters=ModelHyperparameters(values={"iterations": 50}),
                feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
            )
            model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))

        median = _report(f"CatBoostModel.fit ({_N_ROWS} rows, {len(_FEATURE_NAMES)} features, 50 iterations)", _timed_iterations(fit, 20))
        assert median < 3.0, "fitting 50-iteration CatBoost on 2000x10 should not take >3s (conservative floor)"

    def test_logistic_regression_fit(self) -> None:
        features = _features()
        labels = _classification_labels(features)
        factory = linear_wrapper.LogisticRegressionModelFactory()

        def fit() -> None:
            model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION)
            model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))

        median = _report(f"LogisticRegressionModel.fit ({_N_ROWS} rows, {len(_FEATURE_NAMES)} features)", _timed_iterations(fit, 50))
        assert median < 0.5, "fitting sklearn LogisticRegression on 2000x10 should not take >500ms"

    def test_elastic_net_fit(self) -> None:
        features = _features()
        labels = _regression_labels(features)
        factory = linear_wrapper.ElasticNetModelFactory()

        def fit() -> None:
            model = factory.create(hyperparameters=ModelHyperparameters(), feature_schema=_SCHEMA, objective=ObjectiveType.REGRESSION)
            model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))

        median = _report(f"ElasticNetModel.fit ({_N_ROWS} rows, {len(_FEATURE_NAMES)} features)", _timed_iterations(fit, 50))
        assert median < 0.5, "fitting sklearn ElasticNet on 2000x10 should not take >500ms"


class TestModelPredictThroughput:
    def test_lightgbm_predict(self) -> None:
        features = _features()
        labels = _classification_labels(features)
        factory = lgbm_wrapper.LightGBMModelFactory()
        model = factory.create(
            hyperparameters=ModelHyperparameters(values={"num_boost_round": 50, "num_leaves": 15}),
            feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
        )
        fitted = model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))
        batch = features.iloc[:500]

        median = _report("FittedLightGBMModel.predict (500-row batch)", _timed_iterations(lambda: fitted.predict(batch), 200))
        assert median < 0.1, "predicting 500 rows with a 50-round LightGBM model should not take >100ms"

    def test_xgboost_predict(self) -> None:
        features = _features()
        labels = _classification_labels(features)
        factory = xgb_wrapper.XGBoostModelFactory()
        model = factory.create(
            hyperparameters=ModelHyperparameters(values={"num_boost_round": 50, "max_depth": 4}),
            feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
        )
        fitted = model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))
        batch = features.iloc[:500]

        median = _report("FittedXGBoostModel.predict (500-row batch)", _timed_iterations(lambda: fitted.predict(batch), 200))
        assert median < 0.1, "predicting 500 rows with a 50-round XGBoost model should not take >100ms"

    def test_catboost_predict(self) -> None:
        features = _features()
        labels = _classification_labels(features)
        factory = cb_wrapper.CatBoostModelFactory()
        model = factory.create(
            hyperparameters=ModelHyperparameters(values={"iterations": 50}), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
        )
        fitted = model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))
        batch = features.iloc[:500]

        median = _report("FittedCatBoostModel.predict (500-row batch)", _timed_iterations(lambda: fitted.predict(batch), 200))
        assert median < 0.2, "predicting 500 rows with a 50-iteration CatBoost model should not take >200ms"


class TestSerializationThroughput:
    def test_lightgbm_serialize_deserialize_round_trip(self) -> None:
        features = _features()
        labels = _classification_labels(features)
        factory = lgbm_wrapper.LightGBMModelFactory()
        model = factory.create(
            hyperparameters=ModelHyperparameters(values={"num_boost_round": 50, "num_leaves": 15}),
            feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
        )
        fitted = model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))
        serializer, deserializer = lgbm_wrapper.LightGBMModelSerializer(), lgbm_wrapper.LightGBMModelDeserializer()

        def round_trip() -> None:
            data = serializer.serialize(fitted)
            deserializer.deserialize(data, expected_metadata=fitted.metadata)

        median = _report("LightGBM serialize+deserialize round trip (50-round model)", _timed_iterations(round_trip, 50))
        assert median < 0.5, "a LightGBM serialize+deserialize round trip should not take >500ms"

    def test_xgboost_serialize_deserialize_round_trip(self) -> None:
        features = _features()
        labels = _classification_labels(features)
        factory = xgb_wrapper.XGBoostModelFactory()
        model = factory.create(
            hyperparameters=ModelHyperparameters(values={"num_boost_round": 50, "max_depth": 4}),
            feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
        )
        fitted = model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))
        serializer, deserializer = xgb_wrapper.XGBoostModelSerializer(), xgb_wrapper.XGBoostModelDeserializer()

        def round_trip() -> None:
            data = serializer.serialize(fitted)
            deserializer.deserialize(data, expected_metadata=fitted.metadata)

        median = _report("XGBoost serialize+deserialize round trip (50-round model)", _timed_iterations(round_trip, 50))
        assert median < 0.5, "an XGBoost serialize+deserialize round trip should not take >500ms"

    def test_catboost_serialize_deserialize_round_trip(self) -> None:
        features = _features()
        labels = _classification_labels(features)
        factory = cb_wrapper.CatBoostModelFactory()
        model = factory.create(
            hyperparameters=ModelHyperparameters(values={"iterations": 50}), feature_schema=_SCHEMA, objective=ObjectiveType.BINARY_CLASSIFICATION,
        )
        fitted = model.fit(features, labels, seeds=SeedConfiguration(master_seed=1))
        serializer, deserializer = cb_wrapper.CatBoostModelSerializer(), cb_wrapper.CatBoostModelDeserializer()

        def round_trip() -> None:
            data = serializer.serialize(fitted)
            deserializer.deserialize(data, expected_metadata=fitted.metadata)

        median = _report(
            "CatBoost serialize+deserialize round trip (50-iteration model, temp-file-backed .cbm)",
            _timed_iterations(round_trip, 50),
        )
        assert median < 0.2, "a CatBoost serialize+deserialize round trip should not take >200ms (temp-file I/O included)"


class TestMetricsComputationThroughput:
    def test_compute_classification_metrics(self) -> None:
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=1000).astype("float64")
        y_pred = rng.integers(0, 2, size=1000).astype("float64")
        y_proba = rng.uniform(size=1000)

        median = _report(
            "compute_classification_metrics (1000 rows, with proba)",
            _timed_iterations(lambda: compute_classification_metrics(y_true, y_pred, y_proba), 1000),
        )
        assert median < 0.05, "computing all classification metrics for 1000 rows should not take >50ms"

    def test_compute_regression_metrics(self) -> None:
        rng = np.random.default_rng(0)
        y_true = rng.normal(size=1000)
        y_pred = rng.normal(size=1000)

        median = _report("compute_regression_metrics (1000 rows)", _timed_iterations(lambda: compute_regression_metrics(y_true, y_pred), 1000))
        assert median < 0.02, "computing all regression metrics for 1000 rows should not take >20ms"


class TestComparisonThroughput:
    def test_compare_to_baselines(self) -> None:
        rng = np.random.default_rng(0)

        def _fold_metrics(name: str) -> ModelFoldMetrics:
            return ModelFoldMetrics(
                model_name=name,
                per_fold_metrics=tuple({"accuracy": float(v), "f1": float(v * 0.9)} for v in rng.uniform(0.5, 0.9, size=8)),
            )

        candidate = _fold_metrics("candidate")
        baselines = [_fold_metrics(f"baseline_{i}") for i in range(4)]

        median = _report(
            "compare_to_baselines (1 candidate vs. 4 baselines, 8 folds, 2 metrics each)",
            _timed_iterations(lambda: compare_to_baselines(candidate, baselines), 200),
        )
        assert median < 0.05, "comparing against 4 baselines over 8 folds should not take >50ms"
