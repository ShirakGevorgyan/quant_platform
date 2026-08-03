"""`LabelCoverage` (Milestone 11, Phase 3, Part C): how much of a label
bundle is actually usable -- row/valid/missing counts, coverage
fraction, and a breakdown of WHERE the missing rows are (leading
warmup, trailing embargo tail, or an interior "hole" -- the last of
which is a genuine anomaly for any of the 5 concrete label families,
whose only legitimate `NaN` source is an unresolved trailing tail, per
`labels.diagnostics`'s own established trailing-NaN-tail convention)."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.historical.quality import Severity
from quant_platform.label_validation.evidence import (
    LabelEvidence,
    LabelValidationDimensionKind,
    make_evidence,
)
from quant_platform.labels.builder import LabelBundle
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = ["LABEL_COVERAGE_SCHEMA_VERSION", "LOW_COVERAGE_FRACTION_THRESHOLD", "LabelCoverage", "compute_label_coverage"]

LABEL_COVERAGE_SCHEMA_VERSION = 1
LOW_COVERAGE_FRACTION_THRESHOLD = 0.5


def _leading_run_length(is_na: list[bool]) -> int:
    count = 0
    for flag in is_na:
        if not flag:
            break
        count += 1
    return count


def _trailing_run_length(is_na: list[bool]) -> int:
    return _leading_run_length(list(reversed(is_na)))


@dataclass(frozen=True, slots=True)
class LabelCoverage:
    schema_version: int
    label_specification_id: str
    row_count: int
    valid_count: int
    missing_count: int
    coverage_fraction: float
    warmup_row_count: int
    """Leading run of `NaN` -- rows before the label could first resolve."""
    trailing_unresolved_count: int
    """Trailing run of `NaN` -- rows too close to the end of the data to resolve yet."""
    interior_missing_count: int
    """`NaN` rows that are NEITHER part of the leading warmup NOR the
    trailing tail -- an anomaly; no concrete label family's generator
    legitimately produces one."""
    evidence: tuple[LabelEvidence, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id, "row_count": self.row_count,
            "valid_count": self.valid_count, "missing_count": self.missing_count, "coverage_fraction": self.coverage_fraction,
            "warmup_row_count": self.warmup_row_count, "trailing_unresolved_count": self.trailing_unresolved_count,
            "interior_missing_count": self.interior_missing_count, "evidence": [e.to_json_dict() for e in self.evidence],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelCoverage:
        require_schema_version(raw, supported=LABEL_COVERAGE_SCHEMA_VERSION, context="LabelCoverage")
        return cls(
            schema_version=LABEL_COVERAGE_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            row_count=int(str(raw["row_count"])), valid_count=int(str(raw["valid_count"])), missing_count=int(str(raw["missing_count"])),
            coverage_fraction=float(str(raw["coverage_fraction"])), warmup_row_count=int(str(raw["warmup_row_count"])),
            trailing_unresolved_count=int(str(raw["trailing_unresolved_count"])), interior_missing_count=int(str(raw["interior_missing_count"])),
            evidence=tuple(
                LabelEvidence.from_json_dict(as_json_dict(e, field_name="evidence[]"))
                for e in as_json_list(raw.get("evidence") or [], field_name="evidence")
            ),
        )


def compute_label_coverage(bundle: LabelBundle) -> LabelCoverage:
    spec_id = bundle.specification.label_specification_id
    is_na = bundle.values.isna().tolist()
    row_count = len(is_na)
    missing_count = sum(is_na)
    valid_count = row_count - missing_count
    coverage_fraction = (valid_count / row_count) if row_count else 0.0

    warmup = _leading_run_length(is_na)
    trailing = _trailing_run_length(is_na) if valid_count > 0 else 0
    interior = max(0, missing_count - warmup - trailing)

    evidence: list[LabelEvidence] = []
    if interior > 0:
        evidence.append(make_evidence(
            finding=f"{interior} missing row(s) fall outside the leading warmup / trailing tail (an interior hole)",
            evidence=(f"warmup={warmup}", f"trailing={trailing}", f"interior={interior}"), dimension=LabelValidationDimensionKind.COVERAGE,
            severity=Severity.WARNING, affected_labels=(spec_id,), statistics={"interior_missing_count": float(interior)},
        ))
    if coverage_fraction < LOW_COVERAGE_FRACTION_THRESHOLD:
        evidence.append(make_evidence(
            finding=f"coverage fraction is low ({coverage_fraction:.4f})", evidence=(f"valid_count={valid_count}", f"row_count={row_count}"),
            dimension=LabelValidationDimensionKind.COVERAGE, severity=Severity.WARNING, affected_labels=(spec_id,),
            statistics={"coverage_fraction": coverage_fraction},
        ))

    return LabelCoverage(
        schema_version=LABEL_COVERAGE_SCHEMA_VERSION, label_specification_id=spec_id, row_count=row_count, valid_count=valid_count,
        missing_count=missing_count, coverage_fraction=coverage_fraction, warmup_row_count=warmup, trailing_unresolved_count=trailing,
        interior_missing_count=interior, evidence=tuple(evidence),
    )
