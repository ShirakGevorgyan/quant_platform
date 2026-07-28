"""Calibration methods (Milestone 4E, Sections 7/20): Identity, Platt
(logistic sigmoid), Isotonic regression, and Beta calibration.

THE FIT/TRANSFORM SHAPE
--------------------------------------------------------------------------
Every method here follows `ml.interfaces.TrainableModel`'s established
"fit returns a NEW object" convention: an unfit `<X>Calibrator` (holds
only hyperparameters, e.g. none for these four) exposes `.fit(raw_values,
labels) -> Fitted<X>Calibrator`; the fitted object is a SEPARATE, frozen
dataclass holding only explicit, JSON-serializable parameters -- never a
wrapped `sklearn` estimator object. `FittedCalibrationMethod` (a Protocol)
is the common shape `calibration.fitting`'s candidate-evaluation loop
programs against, so adding a fifth method later requires no change to
that loop.

NO EXECUTABLE OBJECTS ARE EVER SERIALIZED
--------------------------------------------------------------------------
`to_json_dict()` on every fitted method below returns ONLY plain
JSON-native values (floats, a list of floats) -- coefficients, an
intercept, or (isotonic) explicit threshold/value arrays reconstructed
via `numpy.interp` on deserialize, never a pickled/joblib-dumped
`sklearn` estimator. `from_json_dict` independently validates every
persisted parameter (finiteness, shape, monotonic ordering where
applicable, class count) before trusting it -- Section 20's "Validate
deserialization independently."
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from quant_platform.calibration.models import CalibrationMethodKind, ProbabilityRepresentation
from quant_platform.core.exceptions import CalibrationFitError, CalibrationValidationError
from quant_platform.ml.persistence import as_json_list, require_schema_version

METHOD_SCHEMA_VERSION = 1
_LOGIT_CLIP_EPS = 1e-12
"""Internal-only clipping applied strictly WITHIN Platt/Beta fitting to
keep `log`/`logit` finite for input probabilities at the exact [0, 1]
boundary -- distinct from, and never a substitute for, the caller-facing
`ProbabilityClippingPolicy` (Section 9), which governs OUTPUT probability
clipping, not this internal fitting safeguard."""

_METHOD_COMPLEXITY_RANK: dict[CalibrationMethodKind, int] = {
    CalibrationMethodKind.IDENTITY: 0,
    CalibrationMethodKind.PLATT: 1,
    CalibrationMethodKind.BETA: 2,
    CalibrationMethodKind.ISOTONIC: 3,
}
"""Section 8's tie-break chain, step 2: "simpler method preference" --
fewer fitted parameters first. Identity (0 parameters) < Platt (2) < Beta
(3) < Isotonic (up to `n_samples` piecewise knots)."""


def method_complexity_rank(kind: CalibrationMethodKind) -> int:
    return _METHOD_COMPLEXITY_RANK[kind]


@runtime_checkable
class FittedCalibrationMethod(Protocol):
    @property
    def kind(self) -> CalibrationMethodKind: ...

    @property
    def input_representation(self) -> ProbabilityRepresentation: ...

    def transform(self, raw_values: np.ndarray) -> np.ndarray: ...

    def to_json_dict(self) -> dict[str, object]: ...

    def summary(self) -> str: ...


def _validate_fit_inputs(raw_values: np.ndarray, labels: np.ndarray, *, method_name: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(raw_values, dtype="float64")
    y = np.asarray(labels, dtype="float64")
    if x.ndim != 1 or y.ndim != 1:
        raise CalibrationFitError(f"{method_name}.fit requires 1-dimensional arrays, got shapes {x.shape}/{y.shape}")
    if x.shape != y.shape:
        raise CalibrationFitError(f"{method_name}.fit: raw_values shape {x.shape} does not match labels shape {y.shape}")
    if len(x) == 0:
        raise CalibrationFitError(f"{method_name}.fit requires at least one sample")
    if not np.all(np.isfinite(x)):
        raise CalibrationFitError(f"{method_name}.fit: raw_values contains non-finite value(s)")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise CalibrationFitError(f"{method_name}.fit: labels must be binary (0.0/1.0) valued")
    n_classes = len(set(y.tolist()))
    if n_classes < 2:
        raise CalibrationFitError(f"{method_name}.fit requires both classes present, got {n_classes} distinct class(es)")
    return x, y


def _validate_transform_input(raw_values: np.ndarray, *, method_name: str) -> np.ndarray:
    x = np.asarray(raw_values, dtype="float64")
    if x.ndim != 1:
        raise CalibrationValidationError(f"{method_name}.transform requires a 1-dimensional array, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise CalibrationValidationError(f"{method_name}.transform: raw_values contains non-finite value(s)")
    return x


def _require_finite(value: float, *, field_name: str) -> None:
    if not math.isfinite(value):
        raise CalibrationValidationError(f"{field_name} must be finite, got {value!r}")


# --------------------------------------------------------------------------
# A. Identity / no calibration
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class IdentityCalibrator:
    """Always available (Section 8: "The identity method must always be
    available as a baseline") -- requires only that `raw_values` are
    already valid probabilities, never touches `labels` at fit time."""

    def fit(self, raw_values: np.ndarray, labels: np.ndarray) -> FittedIdentityCalibrator:
        x = np.asarray(raw_values, dtype="float64")
        y = np.asarray(labels, dtype="float64")
        if x.shape != y.shape:
            raise CalibrationFitError(f"IdentityCalibrator.fit: raw_values shape {x.shape} does not match labels shape {y.shape}")
        if len(x) == 0:
            raise CalibrationFitError("IdentityCalibrator.fit requires at least one sample")
        if not np.all(np.isfinite(x)):
            raise CalibrationFitError("IdentityCalibrator.fit: raw_values contains non-finite value(s)")
        if np.any((x < 0.0) | (x > 1.0)):
            raise CalibrationFitError("IdentityCalibrator.fit: raw_values must already be valid probabilities in [0, 1]")
        return FittedIdentityCalibrator()


@dataclass(frozen=True, slots=True)
class FittedIdentityCalibrator:
    kind: CalibrationMethodKind = CalibrationMethodKind.IDENTITY
    input_representation: ProbabilityRepresentation = ProbabilityRepresentation.PREDICT_PROBA

    def transform(self, raw_values: np.ndarray) -> np.ndarray:
        x = _validate_transform_input(raw_values, method_name="FittedIdentityCalibrator")
        if np.any((x < 0.0) | (x > 1.0)):
            raise CalibrationValidationError("FittedIdentityCalibrator.transform: raw_values must be valid probabilities in [0, 1]")
        return x.copy()

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": METHOD_SCHEMA_VERSION, "kind": self.kind.value, "input_representation": self.input_representation.value}

    def summary(self) -> str:
        return "Identity: calibrated probability equals the raw predict_proba output unchanged."

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FittedIdentityCalibrator:
        require_schema_version(raw, supported=METHOD_SCHEMA_VERSION, context="FittedIdentityCalibrator")
        if CalibrationMethodKind(raw["kind"]) is not CalibrationMethodKind.IDENTITY:
            raise CalibrationValidationError(f"FittedIdentityCalibrator.from_json_dict: kind mismatch, got {raw['kind']!r}")
        return cls(input_representation=ProbabilityRepresentation(raw["input_representation"]))


# --------------------------------------------------------------------------
# B. Platt scaling / logistic sigmoid calibration
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    input_representation: ProbabilityRepresentation = ProbabilityRepresentation.PREDICT_PROBA

    def fit(self, raw_values: np.ndarray, labels: np.ndarray) -> FittedPlattCalibrator:
        x, y = _validate_fit_inputs(raw_values, labels, method_name="PlattCalibrator")
        feature = self._feature(x)
        model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
        try:
            model.fit(feature.reshape(-1, 1), y)
        except ValueError as exc:
            raise CalibrationFitError(f"PlattCalibrator.fit failed: {exc}") from exc
        return FittedPlattCalibrator(
            coefficient=float(model.coef_[0][0]), intercept=float(model.intercept_[0]),
            input_representation=self.input_representation,
        )

    def _feature(self, x: np.ndarray) -> np.ndarray:
        if self.input_representation is ProbabilityRepresentation.DECISION_FUNCTION:
            return x
        clipped = np.clip(x, _LOGIT_CLIP_EPS, 1.0 - _LOGIT_CLIP_EPS)
        return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class FittedPlattCalibrator:
    coefficient: float
    intercept: float
    kind: CalibrationMethodKind = CalibrationMethodKind.PLATT
    input_representation: ProbabilityRepresentation = ProbabilityRepresentation.PREDICT_PROBA

    def __post_init__(self) -> None:
        _require_finite(self.coefficient, field_name="FittedPlattCalibrator.coefficient")
        _require_finite(self.intercept, field_name="FittedPlattCalibrator.intercept")

    def transform(self, raw_values: np.ndarray) -> np.ndarray:
        x = _validate_transform_input(raw_values, method_name="FittedPlattCalibrator")
        if self.input_representation is ProbabilityRepresentation.DECISION_FUNCTION:
            feature = x
        else:
            # PREDICT_PROBA mode asserts `x` IS already a probability --
            # silently clamping an out-of-[0, 1] value via `_safe_logit`
            # would treat invalid input as if it were valid (this
            # platform's "never silently repair" rule), so it is
            # rejected here exactly like `FittedIdentityCalibrator`/
            # `FittedBetaCalibrator` already do.
            if np.any((x < 0.0) | (x > 1.0)):
                raise CalibrationValidationError("FittedPlattCalibrator.transform: raw_values must be valid probabilities in [0, 1] when input_representation=predict_proba")
            feature = _safe_logit(x)
        # A tampered (Section 7 adversarial audit) or otherwise extreme
        # `coefficient`/`feature` product can overflow to +/-inf --
        # mathematically CORRECT (the sigmoid saturates to exactly 1.0/0.0
        # in that limit), not a computation error, so the overflow
        # RuntimeWarning is suppressed rather than the result being wrong;
        # `over="ignore"` is scoped to only these two operations.
        with np.errstate(over="ignore"):
            z = self.coefficient * feature + self.intercept
            return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": METHOD_SCHEMA_VERSION, "kind": self.kind.value, "coefficient": self.coefficient,
            "intercept": self.intercept, "input_representation": self.input_representation.value,
        }

    def summary(self) -> str:
        return f"Platt scaling: calibrated = sigmoid({self.coefficient:.6g} * raw + {self.intercept:.6g})"

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FittedPlattCalibrator:
        require_schema_version(raw, supported=METHOD_SCHEMA_VERSION, context="FittedPlattCalibrator")
        if CalibrationMethodKind(raw["kind"]) is not CalibrationMethodKind.PLATT:
            raise CalibrationValidationError(f"FittedPlattCalibrator.from_json_dict: kind mismatch, got {raw['kind']!r}")
        return cls(
            coefficient=float(str(raw["coefficient"])), intercept=float(str(raw["intercept"])),
            input_representation=ProbabilityRepresentation(raw["input_representation"]),
        )


def _safe_logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, _LOGIT_CLIP_EPS, 1.0 - _LOGIT_CLIP_EPS)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


# --------------------------------------------------------------------------
# C. Isotonic regression
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    def fit(self, raw_values: np.ndarray, labels: np.ndarray) -> FittedIsotonicCalibrator:
        x, y = _validate_fit_inputs(raw_values, labels, method_name="IsotonicCalibrator")
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        try:
            model.fit(x, y)
        except ValueError as exc:
            raise CalibrationFitError(f"IsotonicCalibrator.fit failed: {exc}") from exc
        x_thresholds = tuple(float(v) for v in model.X_thresholds_)
        y_thresholds = tuple(float(v) for v in model.y_thresholds_)
        return FittedIsotonicCalibrator(x_thresholds=x_thresholds, y_thresholds=y_thresholds)


@dataclass(frozen=True, slots=True)
class FittedIsotonicCalibrator:
    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]
    kind: CalibrationMethodKind = CalibrationMethodKind.ISOTONIC
    input_representation: ProbabilityRepresentation = ProbabilityRepresentation.PREDICT_PROBA

    def __post_init__(self) -> None:
        if len(self.x_thresholds) != len(self.y_thresholds):
            raise CalibrationValidationError(
                f"FittedIsotonicCalibrator: x_thresholds (len={len(self.x_thresholds)}) and y_thresholds "
                f"(len={len(self.y_thresholds)}) must have equal length"
            )
        if len(self.x_thresholds) < 1:
            raise CalibrationValidationError("FittedIsotonicCalibrator requires at least one threshold pair")
        for i, v in enumerate(self.x_thresholds):
            _require_finite(v, field_name=f"FittedIsotonicCalibrator.x_thresholds[{i}]")
        for i, v in enumerate(self.y_thresholds):
            _require_finite(v, field_name=f"FittedIsotonicCalibrator.y_thresholds[{i}]")
            if not (0.0 <= v <= 1.0):
                raise CalibrationValidationError(f"FittedIsotonicCalibrator.y_thresholds[{i}] must be in [0, 1], got {v!r}")
        if list(self.x_thresholds) != sorted(self.x_thresholds):
            raise CalibrationValidationError("FittedIsotonicCalibrator.x_thresholds must be non-decreasing")
        if list(self.y_thresholds) != sorted(self.y_thresholds):
            raise CalibrationValidationError(
                "FittedIsotonicCalibrator.y_thresholds must be non-decreasing (monotonicity is the entire point "
                "of isotonic regression)"
            )

    def transform(self, raw_values: np.ndarray) -> np.ndarray:
        x = _validate_transform_input(raw_values, method_name="FittedIsotonicCalibrator")
        # Isotonic has no DECISION_FUNCTION mode at all (unlike Platt) --
        # `x` is unconditionally asserted to be a probability, so an
        # out-of-[0, 1] value is rejected rather than silently clamped
        # by `np.interp`'s left=/right= extrapolation (which exists to
        # handle in-domain values outside the FITTED threshold range,
        # not to launder invalid input).
        if np.any((x < 0.0) | (x > 1.0)):
            raise CalibrationValidationError("FittedIsotonicCalibrator.transform: raw_values must be valid probabilities in [0, 1]")
        result = np.interp(x, self.x_thresholds, self.y_thresholds, left=self.y_thresholds[0], right=self.y_thresholds[-1])
        return np.asarray(np.clip(result, 0.0, 1.0), dtype=np.float64)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": METHOD_SCHEMA_VERSION, "kind": self.kind.value,
            "x_thresholds": list(self.x_thresholds), "y_thresholds": list(self.y_thresholds),
            "interpolation": "linear", "out_of_bounds": "clip", "input_representation": self.input_representation.value,
        }

    def summary(self) -> str:
        return f"Isotonic regression: {len(self.x_thresholds)} monotone threshold pair(s), linear interpolation, clipped out-of-bounds."

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FittedIsotonicCalibrator:
        require_schema_version(raw, supported=METHOD_SCHEMA_VERSION, context="FittedIsotonicCalibrator")
        if CalibrationMethodKind(raw["kind"]) is not CalibrationMethodKind.ISOTONIC:
            raise CalibrationValidationError(f"FittedIsotonicCalibrator.from_json_dict: kind mismatch, got {raw['kind']!r}")
        if str(raw.get("interpolation", "linear")) != "linear":
            raise CalibrationValidationError(f"FittedIsotonicCalibrator: unsupported interpolation {raw.get('interpolation')!r}")
        if str(raw.get("out_of_bounds", "clip")) != "clip":
            raise CalibrationValidationError(f"FittedIsotonicCalibrator: unsupported out_of_bounds {raw.get('out_of_bounds')!r}")
        return cls(
            x_thresholds=tuple(float(v) for v in as_json_list(raw["x_thresholds"], field_name="x_thresholds")),
            y_thresholds=tuple(float(v) for v in as_json_list(raw["y_thresholds"], field_name="y_thresholds")),
            input_representation=ProbabilityRepresentation(raw["input_representation"]),
        )


# --------------------------------------------------------------------------
# D. Beta calibration (Kull, Silva Filho & Flach, 2017)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BetaCalibrator:
    """`logit(calibrated) = a*log(p) + b*log(1-p) + c`, fit via a
    2-feature logistic regression -- implemented entirely with `sklearn`/
    `numpy` (Section 7D: "if implemented without unsafe or obscure
    dependencies"), no additional third-party calibration package."""

    def fit(self, raw_values: np.ndarray, labels: np.ndarray) -> FittedBetaCalibrator:
        x, y = _validate_fit_inputs(raw_values, labels, method_name="BetaCalibrator")
        if np.any((x < 0.0) | (x > 1.0)):
            raise CalibrationFitError("BetaCalibrator.fit requires raw_values to be valid probabilities in [0, 1]")
        clipped = np.clip(x, _LOGIT_CLIP_EPS, 1.0 - _LOGIT_CLIP_EPS)
        features = np.column_stack([np.log(clipped), np.log(1.0 - clipped)])
        model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
        try:
            model.fit(features, y)
        except ValueError as exc:
            raise CalibrationFitError(f"BetaCalibrator.fit failed: {exc}") from exc
        return FittedBetaCalibrator(
            log_p_coefficient=float(model.coef_[0][0]), log_one_minus_p_coefficient=float(model.coef_[0][1]),
            intercept=float(model.intercept_[0]),
        )


@dataclass(frozen=True, slots=True)
class FittedBetaCalibrator:
    log_p_coefficient: float
    log_one_minus_p_coefficient: float
    intercept: float
    kind: CalibrationMethodKind = CalibrationMethodKind.BETA
    input_representation: ProbabilityRepresentation = ProbabilityRepresentation.PREDICT_PROBA

    def __post_init__(self) -> None:
        _require_finite(self.log_p_coefficient, field_name="FittedBetaCalibrator.log_p_coefficient")
        _require_finite(self.log_one_minus_p_coefficient, field_name="FittedBetaCalibrator.log_one_minus_p_coefficient")
        _require_finite(self.intercept, field_name="FittedBetaCalibrator.intercept")

    def transform(self, raw_values: np.ndarray) -> np.ndarray:
        x = _validate_transform_input(raw_values, method_name="FittedBetaCalibrator")
        if np.any((x < 0.0) | (x > 1.0)):
            raise CalibrationValidationError("FittedBetaCalibrator.transform requires raw_values to be valid probabilities in [0, 1]")
        clipped = np.clip(x, _LOGIT_CLIP_EPS, 1.0 - _LOGIT_CLIP_EPS)
        # See FittedPlattCalibrator.transform's identical comment: an
        # extreme (e.g. tampered) coefficient can overflow this product to
        # +/-inf, which is the mathematically correct saturated sigmoid
        # limit, not an error.
        with np.errstate(over="ignore"):
            z = self.log_p_coefficient * np.log(clipped) + self.log_one_minus_p_coefficient * np.log(1.0 - clipped) + self.intercept
            return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": METHOD_SCHEMA_VERSION, "kind": self.kind.value, "log_p_coefficient": self.log_p_coefficient,
            "log_one_minus_p_coefficient": self.log_one_minus_p_coefficient, "intercept": self.intercept,
            "input_representation": self.input_representation.value,
        }

    def summary(self) -> str:
        return (
            f"Beta calibration: logit(calibrated) = {self.log_p_coefficient:.6g}*log(p) + "
            f"{self.log_one_minus_p_coefficient:.6g}*log(1-p) + {self.intercept:.6g}"
        )

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FittedBetaCalibrator:
        require_schema_version(raw, supported=METHOD_SCHEMA_VERSION, context="FittedBetaCalibrator")
        if CalibrationMethodKind(raw["kind"]) is not CalibrationMethodKind.BETA:
            raise CalibrationValidationError(f"FittedBetaCalibrator.from_json_dict: kind mismatch, got {raw['kind']!r}")
        return cls(
            log_p_coefficient=float(str(raw["log_p_coefficient"])),
            log_one_minus_p_coefficient=float(str(raw["log_one_minus_p_coefficient"])),
            intercept=float(str(raw["intercept"])), input_representation=ProbabilityRepresentation(raw["input_representation"]),
        )


FittedMethodUnion = FittedIdentityCalibrator | FittedPlattCalibrator | FittedIsotonicCalibrator | FittedBetaCalibrator


def fitted_method_from_json_dict(raw: dict[str, object]) -> FittedMethodUnion:
    """Dispatches on the persisted `kind` field to the correct
    `Fitted<X>Calibrator.from_json_dict` -- the one place this package
    reconstructs a fitted calibrator from durable JSON without the
    caller needing to already know which method was selected."""
    kind = CalibrationMethodKind(raw.get("kind"))
    if kind is CalibrationMethodKind.IDENTITY:
        return FittedIdentityCalibrator.from_json_dict(raw)
    if kind is CalibrationMethodKind.PLATT:
        return FittedPlattCalibrator.from_json_dict(raw)
    if kind is CalibrationMethodKind.ISOTONIC:
        return FittedIsotonicCalibrator.from_json_dict(raw)
    if kind is CalibrationMethodKind.BETA:
        return FittedBetaCalibrator.from_json_dict(raw)
    raise CalibrationValidationError(f"Unknown calibration method kind {kind!r}")  # pragma: no cover - exhaustive enum


def build_unfit_method(kind: CalibrationMethodKind) -> IdentityCalibrator | PlattCalibrator | IsotonicCalibrator | BetaCalibrator:
    if kind is CalibrationMethodKind.IDENTITY:
        return IdentityCalibrator()
    if kind is CalibrationMethodKind.PLATT:
        return PlattCalibrator()
    if kind is CalibrationMethodKind.ISOTONIC:
        return IsotonicCalibrator()
    if kind is CalibrationMethodKind.BETA:
        return BetaCalibrator()
    raise CalibrationValidationError(f"Unknown calibration method kind {kind!r}")  # pragma: no cover - exhaustive enum


__all__ = [
    "METHOD_SCHEMA_VERSION",
    "BetaCalibrator",
    "FittedBetaCalibrator",
    "FittedCalibrationMethod",
    "FittedIdentityCalibrator",
    "FittedIsotonicCalibrator",
    "FittedMethodUnion",
    "FittedPlattCalibrator",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "build_unfit_method",
    "fitted_method_from_json_dict",
    "method_complexity_rank",
]
