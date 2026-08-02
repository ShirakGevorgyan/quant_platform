"""The evidence model (Milestone 11, Phase 1, Part 2): the atomic unit
every deep-diagnostic check in `diagnostics.py` emits. Deliberately a
NEW, additive type -- it does not replace `DimensionResult`'s plain
`findings`/`warnings`/`recommendations` strings from Part 1 (that shape
stays exactly as `engine.py`/`dimensions.py` built it), it gives the
deep-diagnostics layer introduced in Part 2 a richer, structured record
to explain WHY a dimension scored the way it did.

Every `Evidence.affected_artifacts` entry references an IMMUTABLE
identity -- `dataset_id`, `content_id`, `version`, and/or a split name
-- never a filesystem path (paths can move; content-addressed ids and
version strings cannot silently change without the resulting object
being, by definition, a different object). Row-level evidence cites the
row's position WITHIN a specific, checksum-verified `(dataset_id,
content_id, split)` triple -- the closest thing to a stable row identity
this system has, since individual rows carry no UUID of their own."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.historical.quality import Severity
from quant_platform.ml.persistence import as_json_list, require_schema_version
from quant_platform.qualification.models import QualificationDimensionKind

__all__ = ["EVIDENCE_SCHEMA_VERSION", "Evidence", "affected_split", "make_evidence"]

EVIDENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Evidence:
    finding: str
    evidence: tuple[str, ...]
    severity: Severity
    dimension: QualificationDimensionKind
    recommendation: str | None
    affected_artifacts: tuple[str, ...]
    blocking: bool = False

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION, "finding": self.finding, "evidence": list(self.evidence),
            "severity": self.severity.value, "dimension": self.dimension.value, "recommendation": self.recommendation,
            "affected_artifacts": list(self.affected_artifacts), "blocking": self.blocking,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> Evidence:
        require_schema_version(raw, supported=EVIDENCE_SCHEMA_VERSION, context="Evidence")
        return cls(
            finding=str(raw["finding"]),
            evidence=tuple(str(s) for s in as_json_list(raw.get("evidence") or [], field_name="evidence")),
            severity=Severity(raw["severity"]), dimension=QualificationDimensionKind(raw["dimension"]),
            recommendation=(None if raw.get("recommendation") is None else str(raw["recommendation"])),
            affected_artifacts=tuple(str(s) for s in as_json_list(raw.get("affected_artifacts") or [], field_name="affected_artifacts")),
            blocking=bool(raw.get("blocking", False)),
        )


def affected_split(dataset_id: str, content_id: str, split_name: str) -> str:
    """The one, canonical way every check in this package spells a
    split's immutable identity -- so two `Evidence` records about the
    SAME split are always textually identical, never accidentally
    diverging (e.g. `split=train` vs `split:train`)."""
    return f"dataset_id={dataset_id} content_id={content_id} split={split_name}"


def make_evidence(
    *, finding: str, evidence: tuple[str, ...], severity: Severity, dimension: QualificationDimensionKind,
    affected_artifacts: tuple[str, ...], recommendation: str | None = None, blocking: bool = False,
) -> Evidence:
    return Evidence(
        finding=finding, evidence=evidence, severity=severity, dimension=dimension, recommendation=recommendation,
        affected_artifacts=affected_artifacts, blocking=blocking,
    )
