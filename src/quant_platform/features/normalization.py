"""Train-only fitted preprocessing transforms: standard scaling, robust
scaling, winsorization/clipping, and a signed-log transform. No dependency
on scikit-learn -- these are simple enough (a handful of scalar parameters
each) that a lightweight internal implementation is both sufficient and
keeps this platform's dependency surface unchanged (`pyproject.toml`'s
dependency list is not touched by Milestone 3).

THE LEAKAGE GUARANTEE THIS MODULE STRUCTURALLY ENFORCES
--------------------------------------------------------------------------
`fit_transform` computes parameters from whatever series it is given --
callers (specifically `dataset_builder.ResearchDatasetBuilder`) must only
ever call it against the TRAINING partition. `apply_transform` never
recomputes anything; it is a pure function of already-frozen parameters.
`TransformPipeline` adds one more structural guard on top: once `fit()` has
been called, a second `fit()` call raises `PreprocessingLeakageError`
unless the caller explicitly passes `allow_refit=True` -- making "the
validation/test split accidentally got fit on" a loud, immediate failure
instead of a silent, hard-to-notice numerical bug.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import FeatureError, PreprocessingLeakageError


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FeatureError(f"Expected a JSON object, got {type(value).__name__}")
    return value


class TransformKind(Enum):
    STANDARD_SCALE = "standard_scale"
    ROBUST_SCALE = "robust_scale"
    WINSORIZE = "winsorize"
    SIGNED_LOG1P = "signed_log1p"


@dataclass(frozen=True, slots=True)
class FittedTransform:
    kind: TransformKind
    feature_name: str
    params: dict[str, float]
    fit_row_count: int
    fit_data_fingerprint: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "feature_name": self.feature_name,
            "params": self.params,
            "fit_row_count": self.fit_row_count,
            "fit_data_fingerprint": self.fit_data_fingerprint,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FittedTransform:
        return cls(
            kind=TransformKind(raw["kind"]),
            feature_name=str(raw["feature_name"]),
            params={str(k): float(str(v)) for k, v in _as_dict(raw["params"]).items()},
            fit_row_count=int(str(raw["fit_row_count"])),
            fit_data_fingerprint=str(raw["fit_data_fingerprint"]),
        )


def fit_transform(
    series: pd.Series, *, kind: TransformKind, feature_name: str, clip_quantiles: tuple[float, float] = (0.01, 0.99)
) -> FittedTransform:
    """Compute frozen parameters for `kind` from `series`. Callers are
    responsible for only ever passing a training-partition slice -- see
    module docstring."""
    clean = series.dropna()

    if kind is TransformKind.STANDARD_SCALE:
        mean = float(clean.mean()) if len(clean) else 0.0
        std = float(clean.std()) if len(clean) else 1.0
        params = {"mean": mean, "std": std if std > 0 else 1.0}
    elif kind is TransformKind.ROBUST_SCALE:
        median = float(clean.median()) if len(clean) else 0.0
        q1 = float(clean.quantile(0.25)) if len(clean) else 0.0
        q3 = float(clean.quantile(0.75)) if len(clean) else 0.0
        iqr = q3 - q1
        params = {"median": median, "iqr": iqr if iqr > 0 else 1.0}
    elif kind is TransformKind.WINSORIZE:
        lower = float(clean.quantile(clip_quantiles[0])) if len(clean) else 0.0
        upper = float(clean.quantile(clip_quantiles[1])) if len(clean) else 0.0
        params = {"lower": lower, "upper": upper}
    elif kind is TransformKind.SIGNED_LOG1P:
        params = {}
    else:  # pragma: no cover - exhaustive over TransformKind
        raise AssertionError(f"Unhandled TransformKind: {kind}")

    clean_sum = float(clean.sum()) if len(clean) else 0.0
    fingerprint_source = f"{feature_name}|{kind.value}|{len(clean)}|{clean_sum!r}"
    fit_data_fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:16]
    return FittedTransform(
        kind=kind, feature_name=feature_name, params=params, fit_row_count=len(clean),
        fit_data_fingerprint=fit_data_fingerprint,
    )


def apply_transform(series: pd.Series, fitted: FittedTransform) -> pd.Series:
    """Pure application of already-frozen `fitted` parameters -- never
    computes anything from `series` itself (no mean/median/quantile call
    appears in this function), which is what makes it safe to call on
    validation/test data without any risk of leakage back into the fitted
    parameters."""
    if fitted.kind is TransformKind.STANDARD_SCALE:
        return (series - fitted.params["mean"]) / fitted.params["std"]
    if fitted.kind is TransformKind.ROBUST_SCALE:
        return (series - fitted.params["median"]) / fitted.params["iqr"]
    if fitted.kind is TransformKind.WINSORIZE:
        return series.clip(lower=fitted.params["lower"], upper=fitted.params["upper"])
    if fitted.kind is TransformKind.SIGNED_LOG1P:
        return pd.Series(np.sign(series) * np.log1p(series.abs()), index=series.index)
    raise AssertionError(f"Unhandled TransformKind: {fitted.kind}")  # pragma: no cover


@dataclass(slots=True)
class TransformPipeline:
    """A named collection of per-feature fitted transforms, fit exactly
    once (by default) against a training partition and then applied
    unchanged to every other split."""

    _fitted: dict[str, FittedTransform] = field(default_factory=dict)
    _is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, train_df: pd.DataFrame, *, specs: dict[str, TransformKind], allow_refit: bool = False) -> None:
        if self._is_fitted and not allow_refit:
            raise PreprocessingLeakageError(
                "TransformPipeline.fit() called on an already-fitted pipeline. Refitting would "
                "silently replace parameters frozen from the training split with parameters derived "
                "from whatever data this call was just given -- if that data includes validation/test "
                "rows, this is exactly how preprocessing leakage happens. Pass allow_refit=True only "
                "if you are deliberately rebuilding the pipeline from scratch on a fresh training set."
            )
        fitted: dict[str, FittedTransform] = {}
        for name, kind in specs.items():
            if name not in train_df.columns:
                raise KeyError(f"TransformPipeline.fit: column {name!r} not present in train_df")
            fitted[name] = fit_transform(train_df[name], kind=kind, feature_name=name)
        self._fitted = fitted
        self._is_fitted = True

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise PreprocessingLeakageError("TransformPipeline.apply() called before fit()")
        result = df.copy()
        for name, fitted in self._fitted.items():
            if name in result.columns:
                result[name] = apply_transform(result[name], fitted)
        return result

    def to_json_dict(self) -> dict[str, object]:
        return {"fitted": {name: ft.to_json_dict() for name, ft in self._fitted.items()}}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> TransformPipeline:
        fitted = {
            name: FittedTransform.from_json_dict(_as_dict(payload))
            for name, payload in _as_dict(raw["fitted"]).items()
        }
        pipeline = cls(_fitted=fitted, _is_fitted=len(fitted) > 0)
        return pipeline

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_json_dict(), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


__all__ = ["FittedTransform", "TransformKind", "TransformPipeline", "apply_transform", "fit_transform"]
