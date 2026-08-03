"""The evidence model (Milestone 11, Phase 3, Part C): the atomic unit
every dimension evaluator in `diagnostics.py` emits. Mirrors
`labels.evidence.LabelEvidence`/`feature_discovery.evidence.
FeatureDiscoveryEvidence`/`qualification`'s own evidence model exactly
-- a DIFFERENT type in a DIFFERENT package (this package is never
merged into `labels/`), sharing only the established shape. Every
field the governing specification names: Finding, Evidence, Severity,
Affected labels, Statistics, Recommendation, Blocking flag.

`LabelValidationDimensionKind` (the 8 dimensions this package
evaluates) and `BlockingFindingCode` (the named blocking conditions)
live here too, since this evidence record is their first, lowest-level
consumer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from quant_platform.historical.quality import Severity
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = [
    "LABEL_VALIDATION_DIMENSION_ORDER",
    "LABEL_VALIDATION_EVIDENCE_SCHEMA_VERSION",
    "BlockingFindingCode",
    "LabelEvidence",
    "LabelValidationDimensionKind",
    "make_evidence",
]

LABEL_VALIDATION_EVIDENCE_SCHEMA_VERSION = 1


class LabelValidationDimensionKind(Enum):
    DISTRIBUTION = "distribution"
    BALANCE = "balance"
    DEGENERACY = "degeneracy"
    COVERAGE = "coverage"
    TEMPORAL_STABILITY = "temporal_stability"
    REGIME_STABILITY = "regime_stability"
    DRIFT = "drift"
    LEAKAGE = "leakage"


LABEL_VALIDATION_DIMENSION_ORDER: tuple[LabelValidationDimensionKind, ...] = (
    LabelValidationDimensionKind.DISTRIBUTION,
    LabelValidationDimensionKind.BALANCE,
    LabelValidationDimensionKind.DEGENERACY,
    LabelValidationDimensionKind.COVERAGE,
    LabelValidationDimensionKind.TEMPORAL_STABILITY,
    LabelValidationDimensionKind.REGIME_STABILITY,
    LabelValidationDimensionKind.DRIFT,
    LabelValidationDimensionKind.LEAKAGE,
)
"""Fixed, canonical iteration order -- every report/engine/reconciliation
in this package iterates dimensions in EXACTLY this order, never dict/set
iteration order, so two qualification runs over the same bundle always
produce byte-identical `LabelDiagnostics.dimension_results` ordering."""


class BlockingFindingCode(Enum):
    EMPTY_LABELS = "empty_labels"
    CONSTANT_LABELS = "constant_labels"
    IDENTITY_MISMATCH = "identity_mismatch"
    MANIFEST_MISMATCH = "manifest_mismatch"
    BARRIER_VIOLATION = "barrier_violation"
    AVAILABILITY_VIOLATION = "availability_violation"
    REPLAY_DIVERGENCE = "replay_divergence"


@dataclass(frozen=True, slots=True)
class LabelEvidence:
    """Never a bare verdict. `evidence` is a tuple of human-readable
    facts, always referencing an immutable identity
    (`label_specification_id`/`content_id`, never a filesystem path or
    a wall-clock timestamp). `affected_labels` is a tuple rather than a
    single id because some findings (drift, overlap, reconciliation)
    inherently concern a PAIR or SET of labels at once."""

    finding: str
    evidence: tuple[str, ...]
    dimension: LabelValidationDimensionKind
    severity: Severity
    affected_labels: tuple[str, ...]
    statistics: dict[str, float] = field(default_factory=dict)
    recommendation: str | None = None
    blocking: bool = False
    blocking_code: BlockingFindingCode | None = None

    def __post_init__(self) -> None:
        if self.blocking and self.blocking_code is None:
            raise ValueError(f"LabelEvidence: blocking=True requires a blocking_code (affected_labels={self.affected_labels!r})")
        if not self.blocking and self.blocking_code is not None:
            raise ValueError(f"LabelEvidence: blocking_code set without blocking=True (affected_labels={self.affected_labels!r})")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": LABEL_VALIDATION_EVIDENCE_SCHEMA_VERSION, "finding": self.finding, "evidence": list(self.evidence),
            "dimension": self.dimension.value, "severity": self.severity.value, "affected_labels": list(self.affected_labels),
            "statistics": self.statistics, "recommendation": self.recommendation, "blocking": self.blocking,
            "blocking_code": (self.blocking_code.value if self.blocking_code else None),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelEvidence:
        require_schema_version(raw, supported=LABEL_VALIDATION_EVIDENCE_SCHEMA_VERSION, context="LabelEvidence")
        blocking_code_raw = raw.get("blocking_code")
        return cls(
            finding=str(raw["finding"]), evidence=tuple(str(s) for s in as_json_list(raw.get("evidence") or [], field_name="evidence")),
            dimension=LabelValidationDimensionKind(raw["dimension"]), severity=Severity(raw["severity"]),
            affected_labels=tuple(str(s) for s in as_json_list(raw.get("affected_labels") or [], field_name="affected_labels")),
            statistics={str(k): float(str(v)) for k, v in as_json_dict(raw.get("statistics") or {}, field_name="statistics").items()},
            recommendation=(None if raw.get("recommendation") is None else str(raw["recommendation"])),
            blocking=bool(raw.get("blocking", False)), blocking_code=(None if blocking_code_raw is None else BlockingFindingCode(blocking_code_raw)),
        )


def make_evidence(
    *, finding: str, evidence: tuple[str, ...], dimension: LabelValidationDimensionKind, severity: Severity,
    affected_labels: tuple[str, ...], statistics: dict[str, float] | None = None, recommendation: str | None = None,
    blocking: bool = False, blocking_code: BlockingFindingCode | None = None,
) -> LabelEvidence:
    return LabelEvidence(
        finding=finding, evidence=evidence, dimension=dimension, severity=severity, affected_labels=affected_labels,
        statistics=(statistics or {}), recommendation=recommendation, blocking=blocking, blocking_code=blocking_code,
    )
