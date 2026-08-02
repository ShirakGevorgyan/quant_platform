"""The evidence model (Milestone 11, Phase 3, Part A): the atomic unit
every dimension evaluator in `diagnostics.py` emits. Mirrors
`feature_discovery.evidence.FeatureDiscoveryEvidence` and
`qualification`'s evidence model exactly -- `affected_specification`
plays the role `affected_feature`/`affected_split` play in those two
packages. `LabelDimensionKind` (the 7 structural dimensions this package
evaluates) and `LabelEvidenceCode` (the named blocking conditions) live
here too, since `LabelEvidence` is their first, lowest-level consumer.

Every dimension concerns STRUCTURE -- identity, versioning, availability
semantics, manifest integrity, determinism, reproducibility, lineage --
never the scientific quality of a label VALUE. This package has no
concept of a "good" or "bad" label; only a structurally sound or
unsound one."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from quant_platform.historical.quality import Severity
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = [
    "LABEL_DIMENSION_ORDER",
    "LABEL_EVIDENCE_SCHEMA_VERSION",
    "LabelDimensionKind",
    "LabelEvidence",
    "LabelEvidenceCode",
    "make_evidence",
]

LABEL_EVIDENCE_SCHEMA_VERSION = 1


class LabelDimensionKind(Enum):
    IDENTITY = "identity"
    VERSIONING = "versioning"
    AVAILABILITY = "availability"
    MANIFEST_INTEGRITY = "manifest_integrity"
    DETERMINISM = "determinism"
    REPRODUCIBILITY = "reproducibility"
    LINEAGE = "lineage"


LABEL_DIMENSION_ORDER: tuple[LabelDimensionKind, ...] = (
    LabelDimensionKind.IDENTITY,
    LabelDimensionKind.VERSIONING,
    LabelDimensionKind.AVAILABILITY,
    LabelDimensionKind.MANIFEST_INTEGRITY,
    LabelDimensionKind.DETERMINISM,
    LabelDimensionKind.REPRODUCIBILITY,
    LabelDimensionKind.LINEAGE,
)
"""Fixed, canonical iteration order -- every report/diagnostics function
in this package iterates dimensions in EXACTLY this order, never dict/set
iteration order, so two diagnostic runs over the same bundle always
produce byte-identical `LabelDiagnostics.dimension_results` ordering."""


class LabelEvidenceCode(Enum):
    IDENTITY_MISMATCH = "identity_mismatch"
    SPECIFICATION_TAMPERED = "specification_tampered"
    MANIFEST_MISMATCH = "manifest_mismatch"
    NON_DETERMINISTIC = "non_deterministic"
    MUTABLE_ALIAS = "mutable_alias"
    UNKNOWN_IDENTITY_ALGORITHM = "unknown_identity_algorithm"
    LINEAGE_INCOMPLETE = "lineage_incomplete"


@dataclass(frozen=True, slots=True)
class LabelEvidence:
    """Never a bare verdict. `evidence` is a tuple of human-readable
    facts, always referencing an immutable identity
    (`label_specification_id`/`content_id`, never a filesystem path or
    a wall-clock timestamp)."""

    finding: str
    evidence: tuple[str, ...]
    dimension: LabelDimensionKind
    severity: Severity
    recommendation: str | None
    affected_specification: str
    supporting_statistics: dict[str, float] = field(default_factory=dict)
    blocking: bool = False
    blocking_code: LabelEvidenceCode | None = None

    def __post_init__(self) -> None:
        if self.blocking and self.blocking_code is None:
            raise ValueError(f"LabelEvidence: blocking=True requires a blocking_code (specification={self.affected_specification!r})")
        if not self.blocking and self.blocking_code is not None:
            raise ValueError(f"LabelEvidence: blocking_code set without blocking=True (specification={self.affected_specification!r})")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": LABEL_EVIDENCE_SCHEMA_VERSION, "finding": self.finding, "evidence": list(self.evidence),
            "dimension": self.dimension.value, "severity": self.severity.value, "recommendation": self.recommendation,
            "affected_specification": self.affected_specification, "supporting_statistics": self.supporting_statistics,
            "blocking": self.blocking, "blocking_code": (self.blocking_code.value if self.blocking_code else None),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelEvidence:
        require_schema_version(raw, supported=LABEL_EVIDENCE_SCHEMA_VERSION, context="LabelEvidence")
        blocking_code_raw = raw.get("blocking_code")
        return cls(
            finding=str(raw["finding"]), evidence=tuple(str(s) for s in as_json_list(raw.get("evidence") or [], field_name="evidence")),
            dimension=LabelDimensionKind(raw["dimension"]), severity=Severity(raw["severity"]),
            recommendation=(None if raw.get("recommendation") is None else str(raw["recommendation"])),
            affected_specification=str(raw["affected_specification"]),
            supporting_statistics={str(k): float(str(v)) for k, v in as_json_dict(raw.get("supporting_statistics") or {}, field_name="supporting_statistics").items()},
            blocking=bool(raw.get("blocking", False)), blocking_code=(None if blocking_code_raw is None else LabelEvidenceCode(blocking_code_raw)),
        )


def make_evidence(
    *, finding: str, evidence: tuple[str, ...], dimension: LabelDimensionKind, severity: Severity,
    affected_specification: str, recommendation: str | None = None, supporting_statistics: dict[str, float] | None = None,
    blocking: bool = False, blocking_code: LabelEvidenceCode | None = None,
) -> LabelEvidence:
    return LabelEvidence(
        finding=finding, evidence=evidence, dimension=dimension, severity=severity, recommendation=recommendation,
        affected_specification=affected_specification, supporting_statistics=(supporting_statistics or {}), blocking=blocking, blocking_code=blocking_code,
    )
