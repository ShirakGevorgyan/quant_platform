"""`LabelDrift` (Milestone 11, Phase 3, Part C): distribution drift
between two label bundles -- PSI, KL divergence, JS divergence, class
drift, and rolling drift (PSI computed per rolling window of the
candidate against a fixed baseline distribution). Compares two already-
generated `labels.builder.LabelBundle`s (e.g. two different time
windows, or two different generations of the SAME specification) --
never estimates predictive power, never compares a label to any
prediction target."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import LabelValidationRequestError
from quant_platform.historical.quality import Severity
from quant_platform.label_validation.distribution import bucket_values
from quant_platform.label_validation.evidence import (
    LabelEvidence,
    LabelValidationDimensionKind,
    make_evidence,
)
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.models import LabelFamily
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = ["LABEL_DRIFT_SCHEMA_VERSION", "SIGNIFICANT_PSI_THRESHOLD", "LabelDrift", "compute_label_drift"]

LABEL_DRIFT_SCHEMA_VERSION = 1
SIGNIFICANT_PSI_THRESHOLD = 0.2
"""The commonly-used PSI threshold beyond which a distribution shift is
considered significant (0.1 = moderate, 0.2 = significant) -- a
documented, disclosed convention, not a claim of statistical
significance in the formal sense."""
_EPSILON = 1e-6


def _class_fractions(values: pd.Series, *, label_family: LabelFamily, bucket_count: int) -> dict[str, float]:
    buckets = bucket_values(values, label_family=label_family, bucket_count=bucket_count).dropna()
    if len(buckets) == 0:
        return {}
    counts = buckets.value_counts()
    return {str(k): float(v) / len(buckets) for k, v in counts.items()}


def _aligned_probabilities(baseline_fractions: dict[str, float], candidate_fractions: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    classes = sorted(set(baseline_fractions) | set(candidate_fractions))
    p = np.array([baseline_fractions.get(c, 0.0) + _EPSILON for c in classes])
    q = np.array([candidate_fractions.get(c, 0.0) + _EPSILON for c in classes])
    return p / p.sum(), q / q.sum()


def _psi(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum((q - p) * np.log(q / p)))


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return float(0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m))


@dataclass(frozen=True, slots=True)
class LabelDrift:
    schema_version: int
    baseline_label_specification_id: str
    candidate_label_specification_id: str
    psi: float | None
    kl_divergence: float | None
    js_divergence: float | None
    class_drift: dict[str, float]
    """`candidate_fraction - baseline_fraction` per class."""
    rolling_drift: tuple[float, ...]
    """PSI of each successive rolling window of the candidate's values against the fixed baseline distribution."""
    evidence: tuple[LabelEvidence, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "baseline_label_specification_id": self.baseline_label_specification_id,
            "candidate_label_specification_id": self.candidate_label_specification_id, "psi": self.psi,
            "kl_divergence": self.kl_divergence, "js_divergence": self.js_divergence, "class_drift": self.class_drift,
            "rolling_drift": list(self.rolling_drift), "evidence": [e.to_json_dict() for e in self.evidence],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelDrift:
        require_schema_version(raw, supported=LABEL_DRIFT_SCHEMA_VERSION, context="LabelDrift")

        def _optional_float(key: str) -> float | None:
            value = raw.get(key)
            return None if value is None else float(str(value))

        return cls(
            schema_version=LABEL_DRIFT_SCHEMA_VERSION, baseline_label_specification_id=str(raw["baseline_label_specification_id"]),
            candidate_label_specification_id=str(raw["candidate_label_specification_id"]), psi=_optional_float("psi"),
            kl_divergence=_optional_float("kl_divergence"), js_divergence=_optional_float("js_divergence"),
            class_drift={str(k): float(str(v)) for k, v in as_json_dict(raw.get("class_drift") or {}, field_name="class_drift").items()},
            rolling_drift=tuple(float(str(v)) for v in as_json_list(raw.get("rolling_drift") or [], field_name="rolling_drift")),
            evidence=tuple(
                LabelEvidence.from_json_dict(as_json_dict(e, field_name="evidence[]"))
                for e in as_json_list(raw.get("evidence") or [], field_name="evidence")
            ),
        )


def compute_label_drift(baseline: LabelBundle, candidate: LabelBundle, *, bucket_count: int = 10, rolling_window_bars: int = 20) -> LabelDrift:
    if baseline.specification.label_family != candidate.specification.label_family:
        raise LabelValidationRequestError(
            f"Cannot compare drift between different label families: baseline={baseline.specification.label_family.value!r} "
            f"candidate={candidate.specification.label_family.value!r}",
            context={"baseline_family": baseline.specification.label_family.value, "candidate_family": candidate.specification.label_family.value},
        )

    label_family = baseline.specification.label_family
    baseline_fractions = _class_fractions(baseline.values, label_family=label_family, bucket_count=bucket_count)
    candidate_fractions = _class_fractions(candidate.values, label_family=label_family, bucket_count=bucket_count)
    baseline_id = baseline.specification.label_specification_id
    candidate_id = candidate.specification.label_specification_id

    if not baseline_fractions or not candidate_fractions:
        return LabelDrift(
            schema_version=LABEL_DRIFT_SCHEMA_VERSION, baseline_label_specification_id=baseline_id, candidate_label_specification_id=candidate_id,
            psi=None, kl_divergence=None, js_divergence=None, class_drift={}, rolling_drift=(), evidence=(),
        )

    p, q = _aligned_probabilities(baseline_fractions, candidate_fractions)
    psi = _psi(p, q)
    kl = _kl_divergence(p, q)
    js = _js_divergence(p, q)
    classes = sorted(set(baseline_fractions) | set(candidate_fractions))
    class_drift = {c: candidate_fractions.get(c, 0.0) - baseline_fractions.get(c, 0.0) for c in classes}

    rolling_drift: list[float] = []
    candidate_valid = candidate.values.dropna()
    for start in range(0, len(candidate_valid), rolling_window_bars):
        window = candidate_valid.iloc[start : start + rolling_window_bars]
        if len(window) < max(2, rolling_window_bars // 2):
            continue
        window_fractions = _class_fractions(window, label_family=label_family, bucket_count=bucket_count)
        if not window_fractions:
            continue
        wp, wq = _aligned_probabilities(baseline_fractions, window_fractions)
        rolling_drift.append(_psi(wp, wq))

    evidence: list[LabelEvidence] = []
    if psi > SIGNIFICANT_PSI_THRESHOLD:
        evidence.append(make_evidence(
            finding=f"PSI is {psi:.4f}, above the significant-drift threshold ({SIGNIFICANT_PSI_THRESHOLD})",
            evidence=(f"class_drift={class_drift}",), dimension=LabelValidationDimensionKind.DRIFT, severity=Severity.WARNING,
            affected_labels=(baseline_id, candidate_id), statistics={"psi": psi, "kl_divergence": kl, "js_divergence": js},
        ))

    return LabelDrift(
        schema_version=LABEL_DRIFT_SCHEMA_VERSION, baseline_label_specification_id=baseline_id, candidate_label_specification_id=candidate_id,
        psi=psi, kl_divergence=kl, js_divergence=js, class_drift=class_drift, rolling_drift=tuple(rolling_drift), evidence=tuple(evidence),
    )
