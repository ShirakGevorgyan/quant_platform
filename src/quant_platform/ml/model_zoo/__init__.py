"""`quant_platform.ml.model_zoo` -- Milestone 4C's production predictive
model layer: real, registrable `TrainableModel`/`FittedModel`/
`ModelFactory`/`ModelSerializer`/`ModelDeserializer` implementations for
LightGBM, XGBoost, CatBoost, scikit-learn Logistic Regression/Elastic
Net, and four mandatory comparison baselines (Constant/Random/Majority/
Dummy-Mean-Regressor).

NOT A NEW ABSTRACTION LAYER
--------------------------------------------------------------------------
Every model here satisfies `ml.interfaces`'s EXISTING Protocols exactly
as `ml.testing.ConstantTestModel` already did -- there is no model-
specific branch anywhere in `execution.executor`/`execution.runner`, and
none is added by this package. `ModelRegistry`/`ModelDefinition` (`ml.
registry`) are reused unmodified; `register_default_models` below is the
one new convenience this package adds, building a `ModelRegistry`
pre-populated with every model this milestone ships -- a caller may
still register a subset, or additional definitions, directly against
`ml.registry.ModelRegistry` without going through it at all.

`predict()` vs. `predict_proba()` CONVENTION (binding on every model in
this package)
--------------------------------------------------------------------------
For `BINARY_CLASSIFICATION`, `predict()` returns HARD class labels
(`{0.0, 1.0}`), never a probability -- `predict_proba()` (only on models
declaring `supports_predict_proba=True`) is the SEPARATE, additional
capability that returns continuous probabilities, with class-label
column order given by `class_labels`. This mirrors scikit-learn's own
`predict`/`predict_proba` split and is what makes `ml.metrics.
compute_classification_metrics`'s dual (`y_pred` for accuracy/precision/
recall/F1/MCC, `y_proba` for ROC AUC/PR AUC) signature meaningful.

NO MODEL IN THIS PACKAGE PICKLES ANYTHING
--------------------------------------------------------------------------
LightGBM/XGBoost/CatBoost each have a native, non-pickle serialization
format (text for LightGBM, a versioned binary format for XGBoost,
CatBoost's own `.cbm` format); scikit-learn's linear models and every
baseline are serialized by extracting their fitted NumPy-array
parameters into an explicit, versioned JSON envelope and reconstructing
a fresh estimator on deserialize -- never `pickle`/`joblib.dump` on an
arbitrary estimator object. See `model_zoo.common`'s module docstring.
"""

from __future__ import annotations

from quant_platform.ml.interfaces import ModelDeserializer, ModelSerializer
from quant_platform.ml.model_zoo.baselines import (
    CONSTANT_PREDICTOR_NAME,
    DUMMY_MEAN_REGRESSOR_NAME,
    MAJORITY_PREDICTOR_NAME,
    RANDOM_PREDICTOR_NAME,
    ConstantPredictorDeserializer,
    ConstantPredictorFactory,
    ConstantPredictorSerializer,
    DummyMeanRegressorDeserializer,
    DummyMeanRegressorFactory,
    DummyMeanRegressorSerializer,
    MajorityPredictorDeserializer,
    MajorityPredictorFactory,
    MajorityPredictorSerializer,
    RandomPredictorDeserializer,
    RandomPredictorFactory,
    RandomPredictorSerializer,
)
from quant_platform.ml.model_zoo.catboost_model import (
    CATBOOST_MODEL_NAME,
    CatBoostModelDeserializer,
    CatBoostModelFactory,
    CatBoostModelSerializer,
)
from quant_platform.ml.model_zoo.lightgbm_model import (
    LIGHTGBM_MODEL_NAME,
    LightGBMModelDeserializer,
    LightGBMModelFactory,
    LightGBMModelSerializer,
)
from quant_platform.ml.model_zoo.linear import (
    ELASTIC_NET_MODEL_NAME,
    LOGISTIC_REGRESSION_MODEL_NAME,
    ElasticNetModelDeserializer,
    ElasticNetModelFactory,
    ElasticNetModelSerializer,
    LogisticRegressionModelDeserializer,
    LogisticRegressionModelFactory,
    LogisticRegressionModelSerializer,
)
from quant_platform.ml.model_zoo.xgboost_model import (
    XGBOOST_MODEL_NAME,
    XGBoostModelDeserializer,
    XGBoostModelFactory,
    XGBoostModelSerializer,
)
from quant_platform.ml.registry import ModelDefinition, ModelRegistry

MODEL_ZOO_VERSION = "1.0.0"

BASELINE_MODEL_NAMES: tuple[str, ...] = (
    CONSTANT_PREDICTOR_NAME, RANDOM_PREDICTOR_NAME, MAJORITY_PREDICTOR_NAME, DUMMY_MEAN_REGRESSOR_NAME,
)
"""Every real, comparison-worthy model must be proven to statistically
outperform every one of these -- see `ml.comparison`."""

_SERIALIZER_ID_SUFFIX = "_v1"


def _serializer_id_for(name: str) -> str:
    return f"{name}{_SERIALIZER_ID_SUFFIX}"


def register_default_models(registry: ModelRegistry | None = None) -> ModelRegistry:
    """Registers every model this milestone ships, version `"1"`, into
    `registry` (a fresh `ModelRegistry` if none is given) and returns it.
    A caller wanting only a subset should call `ModelRegistry.register`
    directly with the specific `ModelDefinition`s they want instead of
    calling this at all -- nothing here is required to use any single
    model in this package."""
    registry = registry if registry is not None else ModelRegistry()

    registry.register(
        ModelDefinition(
            name=CONSTANT_PREDICTOR_NAME, version="1",
            description="Baseline: predicts a fixed, hyperparameter-declared constant regardless of input. "
            "Mandatory scientific-comparison floor -- see 'MODEL COMPARISON'.",
            capabilities=ConstantPredictorFactory.capabilities(), factory=ConstantPredictorFactory(),
            serializer_id=_serializer_id_for(CONSTANT_PREDICTOR_NAME),
        )
    )
    registry.register(
        ModelDefinition(
            name=RANDOM_PREDICTOR_NAME, version="1",
            description="Baseline: predicts by resampling (with replacement, seeded) from the TRAINING "
            "label distribution. Mandatory scientific-comparison floor.",
            capabilities=RandomPredictorFactory.capabilities(), factory=RandomPredictorFactory(),
            serializer_id=_serializer_id_for(RANDOM_PREDICTOR_NAME),
        )
    )
    registry.register(
        ModelDefinition(
            name=MAJORITY_PREDICTOR_NAME, version="1",
            description="Baseline (classification only): always predicts the majority training class. "
            "Mandatory scientific-comparison floor.",
            capabilities=MajorityPredictorFactory.capabilities(), factory=MajorityPredictorFactory(),
            serializer_id=_serializer_id_for(MAJORITY_PREDICTOR_NAME),
        )
    )
    registry.register(
        ModelDefinition(
            name=DUMMY_MEAN_REGRESSOR_NAME, version="1",
            description="Baseline (regression only): always predicts the training label mean. Mandatory "
            "scientific-comparison floor.",
            capabilities=DummyMeanRegressorFactory.capabilities(), factory=DummyMeanRegressorFactory(),
            serializer_id=_serializer_id_for(DUMMY_MEAN_REGRESSOR_NAME),
        )
    )
    registry.register(
        ModelDefinition(
            name=LOGISTIC_REGRESSION_MODEL_NAME, version="1",
            description="scikit-learn LogisticRegression (binary classification), deterministic (lbfgs solver).",
            capabilities=LogisticRegressionModelFactory.capabilities(), factory=LogisticRegressionModelFactory(),
            serializer_id=_serializer_id_for(LOGISTIC_REGRESSION_MODEL_NAME),
        )
    )
    registry.register(
        ModelDefinition(
            name=ELASTIC_NET_MODEL_NAME, version="1",
            description="scikit-learn ElasticNet (regression), deterministic (cyclic coordinate descent).",
            capabilities=ElasticNetModelFactory.capabilities(), factory=ElasticNetModelFactory(),
            serializer_id=_serializer_id_for(ELASTIC_NET_MODEL_NAME),
        )
    )
    registry.register(
        ModelDefinition(
            name=LIGHTGBM_MODEL_NAME, version="1",
            description="LightGBM gradient boosting (binary classification or regression), deterministic mode.",
            capabilities=LightGBMModelFactory.capabilities(), factory=LightGBMModelFactory(),
            serializer_id=_serializer_id_for(LIGHTGBM_MODEL_NAME),
        )
    )
    registry.register(
        ModelDefinition(
            name=XGBOOST_MODEL_NAME, version="1",
            description="XGBoost gradient boosting (binary classification or regression), deterministic histogram method.",
            capabilities=XGBoostModelFactory.capabilities(), factory=XGBoostModelFactory(),
            serializer_id=_serializer_id_for(XGBOOST_MODEL_NAME),
        )
    )
    registry.register(
        ModelDefinition(
            name=CATBOOST_MODEL_NAME, version="1",
            description="CatBoost gradient boosting (binary classification or regression), native categorical support, deterministic mode.",
            capabilities=CatBoostModelFactory.capabilities(), factory=CatBoostModelFactory(),
            serializer_id=_serializer_id_for(CATBOOST_MODEL_NAME),
        )
    )
    return registry


def default_serializer_registry() -> dict[str, tuple[ModelSerializer, ModelDeserializer]]:
    """`serializer_id -> (ModelSerializer, ModelDeserializer)` for every
    model `register_default_models` registers -- the SAME "resolved by a
    caller-provided lookup, never instantiated by the registry itself"
    pattern `execution.runner._SERIALIZER_REGISTRY` already established
    for `ml.testing`'s test-only model; this is the real-model
    equivalent a caller merges into their own lookup. Each entry needs its
    own `# type: ignore[dict-item]` for the same reason `execution.
    runner._SERIALIZER_REGISTRY`'s one built-in entry already does: every
    concrete `serialize`/`deserialize` pair's parameter is typed to its
    OWN specific fitted-model dataclass, narrower than the `ModelSerializer`/
    `ModelDeserializer` protocols' `FittedModel` parameter -- sound in
    practice (this registry is only ever consulted for a model whose OWN
    factory produced that exact fitted type), but not something a
    structural type checker can prove."""
    registry: dict[str, tuple[ModelSerializer, ModelDeserializer]] = {
        _serializer_id_for(CONSTANT_PREDICTOR_NAME): (ConstantPredictorSerializer(), ConstantPredictorDeserializer()),  # type: ignore[dict-item]
        _serializer_id_for(RANDOM_PREDICTOR_NAME): (RandomPredictorSerializer(), RandomPredictorDeserializer()),  # type: ignore[dict-item]
        _serializer_id_for(MAJORITY_PREDICTOR_NAME): (MajorityPredictorSerializer(), MajorityPredictorDeserializer()),  # type: ignore[dict-item]
        _serializer_id_for(DUMMY_MEAN_REGRESSOR_NAME): (DummyMeanRegressorSerializer(), DummyMeanRegressorDeserializer()),  # type: ignore[dict-item]
        _serializer_id_for(LOGISTIC_REGRESSION_MODEL_NAME): (LogisticRegressionModelSerializer(), LogisticRegressionModelDeserializer()),  # type: ignore[dict-item]
        _serializer_id_for(ELASTIC_NET_MODEL_NAME): (ElasticNetModelSerializer(), ElasticNetModelDeserializer()),  # type: ignore[dict-item]
        _serializer_id_for(LIGHTGBM_MODEL_NAME): (LightGBMModelSerializer(), LightGBMModelDeserializer()),  # type: ignore[dict-item]
        _serializer_id_for(XGBOOST_MODEL_NAME): (XGBoostModelSerializer(), XGBoostModelDeserializer()),  # type: ignore[dict-item]
        _serializer_id_for(CATBOOST_MODEL_NAME): (CatBoostModelSerializer(), CatBoostModelDeserializer()),  # type: ignore[dict-item]
    }
    return registry


__all__ = [
    "BASELINE_MODEL_NAMES",
    "CATBOOST_MODEL_NAME",
    "CONSTANT_PREDICTOR_NAME",
    "DUMMY_MEAN_REGRESSOR_NAME",
    "ELASTIC_NET_MODEL_NAME",
    "LIGHTGBM_MODEL_NAME",
    "LOGISTIC_REGRESSION_MODEL_NAME",
    "MAJORITY_PREDICTOR_NAME",
    "MODEL_ZOO_VERSION",
    "RANDOM_PREDICTOR_NAME",
    "XGBOOST_MODEL_NAME",
    "default_serializer_registry",
    "register_default_models",
]
