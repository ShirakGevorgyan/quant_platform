"""The evidence model (Milestone 11, Phase 2, Part 1): the atomic unit
every dimension evaluator in `diagnostics.py` emits. Every conclusion
this package draws is a `FeatureDiscoveryEvidence` record -- never a
bare `"bad feature"` string. `FeatureDiscoveryDimensionKind` (the 10
required dimensions) and `BlockingFindingCode` (the 6 named blocking
conditions) live here too, since `FeatureDiscoveryEvidence` is their
first, lowest-level consumer -- `models.py`'s composite types
(`FeatureDimensionResult`, `FeatureSignalDiagnostics`,
`FeatureDiscoveryReport`) import both from this module, never the
reverse."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from quant_platform.historical.quality import Severity
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = [
    "FEATURE_DISCOVERY_DIMENSION_ORDER",
    "FEATURE_DISCOVERY_EVIDENCE_SCHEMA_VERSION",
    "BlockingFindingCode",
    "FeatureDiscoveryDimensionKind",
    "FeatureDiscoveryEvidence",
    "make_evidence",
]

FEATURE_DISCOVERY_EVIDENCE_SCHEMA_VERSION = 1


class FeatureDiscoveryDimensionKind(Enum):
    INFORMATION_CONTENT = "information_content"
    TEMPORAL_STABILITY = "temporal_stability"
    REGIME_STABILITY = "regime_stability"
    DRIFT_BEHAVIOUR = "drift_behaviour"
    REDUNDANCY = "redundancy"
    COVERAGE = "coverage"
    AVAILABILITY = "availability"
    LEAKAGE_SAFETY = "leakage_safety"
    DETERMINISM = "determinism"
    REPRODUCIBILITY = "reproducibility"


FEATURE_DISCOVERY_DIMENSION_ORDER: tuple[FeatureDiscoveryDimensionKind, ...] = (
    FeatureDiscoveryDimensionKind.INFORMATION_CONTENT,
    FeatureDiscoveryDimensionKind.TEMPORAL_STABILITY,
    FeatureDiscoveryDimensionKind.REGIME_STABILITY,
    FeatureDiscoveryDimensionKind.DRIFT_BEHAVIOUR,
    FeatureDiscoveryDimensionKind.REDUNDANCY,
    FeatureDiscoveryDimensionKind.COVERAGE,
    FeatureDiscoveryDimensionKind.AVAILABILITY,
    FeatureDiscoveryDimensionKind.LEAKAGE_SAFETY,
    FeatureDiscoveryDimensionKind.DETERMINISM,
    FeatureDiscoveryDimensionKind.REPRODUCIBILITY,
)
"""Fixed, canonical iteration order -- every report/engine/reporting
function in this package iterates dimensions in EXACTLY this order,
never dict/set iteration order, so two discovery runs over the same
dataset always produce byte-identical `FeatureSignalDiagnostics.
dimension_results` ordering."""


class BlockingFindingCode(Enum):
    LEAKAGE = "leakage"
    FUTURE_VISIBILITY = "future_visibility"
    NON_DETERMINISTIC_FEATURE = "non_deterministic_feature"
    IDENTITY_MISMATCH = "identity_mismatch"
    MANIFEST_MISMATCH = "manifest_mismatch"
    AVAILABILITY_VIOLATION = "availability_violation"


@dataclass(frozen=True, slots=True)
class FeatureDiscoveryEvidence:
    """Never a bare verdict. `evidence` is a tuple of human-readable
    facts (always referencing an immutable identity -- `dataset_id`/
    `content_id`/`feature_name`/split name -- never a filesystem path);
    `supporting_statistics` is the same evidence in STRUCTURED,
    machine-checkable form (the exact numeric values a `finding`
    describes, e.g. `{"variance": 0.0}` for a constant-feature
    finding)."""

    finding: str
    evidence: tuple[str, ...]
    dimension: FeatureDiscoveryDimensionKind
    severity: Severity
    recommendation: str | None
    affected_feature: str
    supporting_statistics: dict[str, float] = field(default_factory=dict)
    blocking: bool = False
    blocking_code: BlockingFindingCode | None = None

    def __post_init__(self) -> None:
        if self.blocking and self.blocking_code is None:
            raise ValueError(f"FeatureDiscoveryEvidence: blocking=True requires a blocking_code (feature={self.affected_feature!r})")
        if not self.blocking and self.blocking_code is not None:
            raise ValueError(f"FeatureDiscoveryEvidence: blocking_code set without blocking=True (feature={self.affected_feature!r})")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": FEATURE_DISCOVERY_EVIDENCE_SCHEMA_VERSION, "finding": self.finding, "evidence": list(self.evidence),
            "dimension": self.dimension.value, "severity": self.severity.value, "recommendation": self.recommendation,
            "affected_feature": self.affected_feature, "supporting_statistics": self.supporting_statistics,
            "blocking": self.blocking, "blocking_code": (self.blocking_code.value if self.blocking_code else None),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureDiscoveryEvidence:
        require_schema_version(raw, supported=FEATURE_DISCOVERY_EVIDENCE_SCHEMA_VERSION, context="FeatureDiscoveryEvidence")
        blocking_code_raw = raw.get("blocking_code")
        return cls(
            finding=str(raw["finding"]), evidence=tuple(str(s) for s in as_json_list(raw.get("evidence") or [], field_name="evidence")),
            dimension=FeatureDiscoveryDimensionKind(raw["dimension"]), severity=Severity(raw["severity"]),
            recommendation=(None if raw.get("recommendation") is None else str(raw["recommendation"])),
            affected_feature=str(raw["affected_feature"]),
            supporting_statistics={str(k): float(str(v)) for k, v in as_json_dict(raw.get("supporting_statistics") or {}, field_name="supporting_statistics").items()},
            blocking=bool(raw.get("blocking", False)), blocking_code=(None if blocking_code_raw is None else BlockingFindingCode(blocking_code_raw)),
        )


def make_evidence(
    *, finding: str, evidence: tuple[str, ...], dimension: FeatureDiscoveryDimensionKind, severity: Severity,
    affected_feature: str, recommendation: str | None = None, supporting_statistics: dict[str, float] | None = None,
    blocking: bool = False, blocking_code: BlockingFindingCode | None = None,
) -> FeatureDiscoveryEvidence:
    return FeatureDiscoveryEvidence(
        finding=finding, evidence=evidence, dimension=dimension, severity=severity, recommendation=recommendation,
        affected_feature=affected_feature, supporting_statistics=(supporting_statistics or {}), blocking=blocking, blocking_code=blocking_code,
    )
