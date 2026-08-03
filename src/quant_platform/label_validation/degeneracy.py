"""`LabelDegeneracy` (Milestone 11, Phase 3, Part C): constant labels,
near-constant labels, single class, empty labels, all-neutral, zero
variance, impossible labels -- structural defects that make a label
UNUSABLE for ML research, regardless of any prediction target. Unlike
`balance.py`, degeneracy findings CAN be blocking: a constant or empty
label carries zero information no model could ever learn from.

`detect_duplicate_labels` (records-based, not bundle-based -- see
its own docstring) lives here too, since a duplicate is a degeneracy
of a DIFFERENT kind (an identity collision) rather than a distribution
shape."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.historical.quality import Severity
from quant_platform.label_validation.distribution import bucket_values
from quant_platform.label_validation.evidence import (
    BlockingFindingCode,
    LabelEvidence,
    LabelValidationDimensionKind,
    make_evidence,
)
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.models import LabelFamily
from quant_platform.labels.records import LabelRecord
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = [
    "LABEL_DEGENERACY_SCHEMA_VERSION",
    "NEAR_CONSTANT_FRACTION_THRESHOLD",
    "LabelDegeneracy",
    "compute_label_degeneracy",
    "detect_duplicate_labels",
]

LABEL_DEGENERACY_SCHEMA_VERSION = 1
NEAR_CONSTANT_FRACTION_THRESHOLD = 0.99
"""The majority-value fraction beyond which a non-constant label is
still reported as "near constant" -- a documented, disclosed threshold."""

_DISCRETE_DOMAINS: dict[LabelFamily, frozenset[float]] = {
    LabelFamily.DIRECTION: frozenset({-1.0, 0.0, 1.0}),
    LabelFamily.TRIPLE_BARRIER: frozenset({-1.0, 0.0, 1.0}),
}


def _domain_violation_count(values: pd.Series, label_family: LabelFamily) -> int:
    valid = values.dropna()
    if valid.empty:
        return 0
    if label_family in _DISCRETE_DOMAINS:
        domain = _DISCRETE_DOMAINS[label_family]
        return int((~valid.isin(domain)).sum())
    if label_family is LabelFamily.FORWARD_VOLATILITY:
        return int((valid < 0).sum())
    if label_family in (LabelFamily.NEXT_RETURN, LabelFamily.MULTI_HORIZON_RETURN):
        return int((valid <= -1.0).sum())
    return 0  # FUTURE_EXTENSION_PLACEHOLDER (or any unknown family): no domain to validate against


@dataclass(frozen=True, slots=True)
class LabelDegeneracy:
    schema_version: int
    label_specification_id: str
    is_empty: bool
    is_constant: bool
    is_near_constant: bool
    is_single_class: bool
    """Whether the BUCKETED representation (see `distribution.
    bucket_values`) collapses to a single class -- distinct from
    `is_constant`, which checks the raw values."""
    is_all_neutral: bool
    """`LabelFamily.DIRECTION` only; `False` for every other family."""
    is_zero_variance: bool
    has_impossible_labels: bool
    impossible_value_count: int
    evidence: tuple[LabelEvidence, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id, "is_empty": self.is_empty,
            "is_constant": self.is_constant, "is_near_constant": self.is_near_constant, "is_single_class": self.is_single_class,
            "is_all_neutral": self.is_all_neutral, "is_zero_variance": self.is_zero_variance,
            "has_impossible_labels": self.has_impossible_labels, "impossible_value_count": self.impossible_value_count,
            "evidence": [e.to_json_dict() for e in self.evidence],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelDegeneracy:
        require_schema_version(raw, supported=LABEL_DEGENERACY_SCHEMA_VERSION, context="LabelDegeneracy")
        return cls(
            schema_version=LABEL_DEGENERACY_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            is_empty=bool(raw["is_empty"]), is_constant=bool(raw["is_constant"]), is_near_constant=bool(raw["is_near_constant"]),
            is_single_class=bool(raw["is_single_class"]), is_all_neutral=bool(raw["is_all_neutral"]),
            is_zero_variance=bool(raw["is_zero_variance"]), has_impossible_labels=bool(raw["has_impossible_labels"]),
            impossible_value_count=int(str(raw["impossible_value_count"])),
            evidence=tuple(
                LabelEvidence.from_json_dict(as_json_dict(e, field_name="evidence[]"))
                for e in as_json_list(raw.get("evidence") or [], field_name="evidence")
            ),
        )

    @property
    def is_blocking(self) -> bool:
        return any(e.blocking for e in self.evidence)


def compute_label_degeneracy(bundle: LabelBundle) -> LabelDegeneracy:
    spec_id = bundle.specification.label_specification_id
    label_family = bundle.specification.label_family
    values = bundle.values
    valid = values.dropna()
    evidence: list[LabelEvidence] = []

    is_empty = len(valid) == 0
    if is_empty:
        evidence.append(make_evidence(
            finding="label has zero valid (non-NaN) values", evidence=(f"row_count={bundle.row_count}",),
            dimension=LabelValidationDimensionKind.DEGENERACY, severity=Severity.CRITICAL, affected_labels=(spec_id,),
            blocking=True, blocking_code=BlockingFindingCode.EMPTY_LABELS,
        ))
        return LabelDegeneracy(
            schema_version=LABEL_DEGENERACY_SCHEMA_VERSION, label_specification_id=spec_id, is_empty=True, is_constant=False,
            is_near_constant=False, is_single_class=False, is_all_neutral=False, is_zero_variance=False, has_impossible_labels=False,
            impossible_value_count=0, evidence=tuple(evidence),
        )

    value_counts = valid.value_counts()
    majority_fraction = float(value_counts.iloc[0]) / len(valid)
    is_constant = valid.nunique() == 1
    is_near_constant = (not is_constant) and majority_fraction >= NEAR_CONSTANT_FRACTION_THRESHOLD
    is_zero_variance = bool(np.isclose(float(valid.std(ddof=0)), 0.0)) if len(valid) > 1 else True

    buckets = bucket_values(values, label_family=label_family).dropna()
    is_single_class = bool(buckets.nunique() <= 1) if len(buckets) > 0 else False

    is_all_neutral = False
    if label_family is LabelFamily.DIRECTION:
        is_all_neutral = bool((valid == 0.0).all())

    impossible_value_count = _domain_violation_count(values, label_family)
    has_impossible_labels = impossible_value_count > 0

    if is_constant:
        evidence.append(make_evidence(
            finding=f"label is constant (single value {valid.iloc[0]!r} for every valid row)", evidence=(f"valid_count={len(valid)}",),
            dimension=LabelValidationDimensionKind.DEGENERACY, severity=Severity.CRITICAL, affected_labels=(spec_id,),
            statistics={"majority_fraction": majority_fraction}, blocking=True, blocking_code=BlockingFindingCode.CONSTANT_LABELS,
        ))
    elif is_near_constant:
        evidence.append(make_evidence(
            finding=f"label is near-constant (majority fraction {majority_fraction:.4f})",
            evidence=(f"majority_fraction={majority_fraction:.4f}",), dimension=LabelValidationDimensionKind.DEGENERACY,
            severity=Severity.WARNING, affected_labels=(spec_id,), statistics={"majority_fraction": majority_fraction},
        ))

    if is_all_neutral:
        evidence.append(make_evidence(
            finding="every valid row is NEUTRAL", evidence=(f"valid_count={len(valid)}",), dimension=LabelValidationDimensionKind.DEGENERACY,
            severity=Severity.CRITICAL, affected_labels=(spec_id,), blocking=True, blocking_code=BlockingFindingCode.CONSTANT_LABELS,
        ))

    if has_impossible_labels:
        evidence.append(make_evidence(
            finding=f"{impossible_value_count} value(s) fall outside {label_family.value}'s valid domain",
            evidence=(f"impossible_value_count={impossible_value_count}",), dimension=LabelValidationDimensionKind.DEGENERACY,
            severity=Severity.CRITICAL, affected_labels=(spec_id,), statistics={"impossible_value_count": float(impossible_value_count)},
            blocking=True, blocking_code=BlockingFindingCode.BARRIER_VIOLATION,
        ))

    return LabelDegeneracy(
        schema_version=LABEL_DEGENERACY_SCHEMA_VERSION, label_specification_id=spec_id, is_empty=False, is_constant=is_constant,
        is_near_constant=is_near_constant, is_single_class=is_single_class, is_all_neutral=is_all_neutral, is_zero_variance=is_zero_variance,
        has_impossible_labels=has_impossible_labels, impossible_value_count=impossible_value_count, evidence=tuple(evidence),
    )


def detect_duplicate_labels(records: tuple[LabelRecord, ...]) -> tuple[str, ...]:
    """A `label_id` collision -- two DIFFERENT records (necessarily
    different `row_identity`/`label_specification_id` combinations,
    since `label_id = sha256(label_specification_id | row_identity)`)
    sharing the same `label_id` would mean a genuine hash collision or
    identity-computation bug. In a healthy system this is always empty
    -- included as a real, structural check rather than a symbolic one,
    per the governing specification's own "duplicate labels" item."""
    counts = Counter(r.label_id for r in records)
    return tuple(sorted(label_id for label_id, count in counts.items() if count > 1))
