"""Overlap detection (Milestone 11, Phase 3, Part C): horizon overlap,
barrier overlap, duplicate targets, redundant targets -- across a SET
of `labels.builder.LabelBundle`s. `duplicate_target` is an EXACT check
(`identity.content_id` equality -- the same content-addressed identity
Part A already computes, never recomputed a second way here);
`redundant_target` is a disclosed statistical-similarity heuristic
(Pearson correlation above a documented threshold) -- never a claim of
predictive redundancy, since this package has no notion of a prediction
target at all."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.models import LabelFamily
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = ["OVERLAP_SCHEMA_VERSION", "REDUNDANCY_CORRELATION_THRESHOLD", "OverlapFinding", "OverlapReport", "detect_overlap"]

OVERLAP_SCHEMA_VERSION = 1
REDUNDANCY_CORRELATION_THRESHOLD = 0.999
"""Pearson correlation above which two DIFFERENTLY-identified label
bundles are reported as "redundant" -- a documented, disclosed
similarity threshold, never a predictive-power claim."""

_TRIPLE_BARRIER_PARAMETER_KEYS = ("profit_multiplier", "loss_multiplier", "max_holding_bars", "volatility_estimator_reference")


def _safe_pearson_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return None
    a_valid, b_valid = a[mask], b[mask]
    if np.std(a_valid) == 0 or np.std(b_valid) == 0:
        return None
    return float(np.corrcoef(a_valid, b_valid)[0, 1])


@dataclass(frozen=True, slots=True)
class OverlapFinding:
    kind: str
    label_specification_id_a: str
    label_specification_id_b: str
    message: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind, "label_specification_id_a": self.label_specification_id_a,
            "label_specification_id_b": self.label_specification_id_b, "message": self.message,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> OverlapFinding:
        return cls(
            kind=str(raw["kind"]), label_specification_id_a=str(raw["label_specification_id_a"]),
            label_specification_id_b=str(raw["label_specification_id_b"]), message=str(raw["message"]),
        )


@dataclass(frozen=True, slots=True)
class OverlapReport:
    schema_version: int
    findings: tuple[OverlapFinding, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "findings": [f.to_json_dict() for f in self.findings]}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> OverlapReport:
        require_schema_version(raw, supported=OVERLAP_SCHEMA_VERSION, context="OverlapReport")
        return cls(
            schema_version=OVERLAP_SCHEMA_VERSION,
            findings=tuple(
                OverlapFinding.from_json_dict(as_json_dict(f, field_name="findings[]"))
                for f in as_json_list(raw.get("findings") or [], field_name="findings")
            ),
        )


def detect_overlap(bundles: tuple[LabelBundle, ...], *, redundancy_correlation_threshold: float = REDUNDANCY_CORRELATION_THRESHOLD) -> OverlapReport:
    findings: list[OverlapFinding] = []
    for i in range(len(bundles)):
        for j in range(i + 1, len(bundles)):
            a, b = bundles[i], bundles[j]
            spec_a, spec_b = a.specification, b.specification
            id_a, id_b = spec_a.label_specification_id, spec_b.label_specification_id

            if (
                spec_a.label_family == spec_b.label_family and spec_a.parameters.get("horizon_bars") is not None
                and spec_a.parameters.get("horizon_bars") == spec_b.parameters.get("horizon_bars")
            ):
                findings.append(OverlapFinding(
                    kind="horizon_overlap", label_specification_id_a=id_a, label_specification_id_b=id_b,
                    message=f"same label_family={spec_a.label_family.value!r} and horizon_bars={spec_a.parameters['horizon_bars']!r}",
                ))

            if (
                spec_a.label_family is LabelFamily.TRIPLE_BARRIER and spec_b.label_family is LabelFamily.TRIPLE_BARRIER
                and all(spec_a.parameters.get(k) == spec_b.parameters.get(k) for k in _TRIPLE_BARRIER_PARAMETER_KEYS)
            ):
                findings.append(OverlapFinding(
                    kind="barrier_overlap", label_specification_id_a=id_a, label_specification_id_b=id_b,
                    message="identical profit_multiplier/loss_multiplier/max_holding_bars/volatility_estimator_reference",
                ))

            if a.identity.content_id == b.identity.content_id:
                findings.append(OverlapFinding(
                    kind="duplicate_target", label_specification_id_a=id_a, label_specification_id_b=id_b,
                    message=f"identical content_id={a.identity.content_id!r}",
                ))
                continue  # a duplicate is trivially also "redundant" -- do not double-report

            correlation = _safe_pearson_correlation(a.values.to_numpy(), b.values.to_numpy())
            if correlation is not None and correlation >= redundancy_correlation_threshold:
                findings.append(OverlapFinding(
                    kind="redundant_target", label_specification_id_a=id_a, label_specification_id_b=id_b,
                    message=f"correlation={correlation:.6f} (>= {redundancy_correlation_threshold})",
                ))

    return OverlapReport(schema_version=OVERLAP_SCHEMA_VERSION, findings=tuple(findings))
