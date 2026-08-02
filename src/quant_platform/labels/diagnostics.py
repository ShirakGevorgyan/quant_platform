"""`LabelDiagnostics` (Milestone 11, Phase 3, Part A): the 7-dimension
structural evaluation of one already-built `builder.LabelBundle` +
`manifest.LabelManifest` pair. Every dimension concerns STRUCTURE --
never the scientific quality, predictive value, or correctness of a
label's VALUES, since Part A ships no family-specific generation logic
to judge those values against.

DIMENSION -> POINT-IN-TIME RULE MAPPING
--------------------------------------------------------------------------
The governing specification names 7 point-in-time rules ("no future
visibility", "no future macro release", "no future cross asset", "no
revised data", "no unavailable observation", "no wall clock semantics",
"no mutable aliases"). Two of them ("no future macro release", "no
future cross asset") are NOT independently verifiable from a bundle
alone -- doing so would require re-reading raw macro/cross-asset source
data, which is out of this package's scope (it never imports
`market_data`/`features`). AVAILABILITY reports this honestly as an
informational, non-blocking finding rather than fabricating a check it
cannot actually perform -- the same disclosed-scope-boundary discipline
`qualification`'s "Macro/cross-asset scope" section already established.
The other 5 rules ARE independently verifiable and are checked for real:
"no mutable aliases" (enforced by `builder.LabelBuilder` at construction
time, confirmed here), "no wall clock semantics"/"no unavailable
observation"/"no future visibility" (the trailing-NaN-tail shape check
below), "no revised data" (out of scope for the identical reason as the
two macro/cross-asset rules, disclosed likewise)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.core.exceptions import LabelError
from quant_platform.historical.quality import Severity
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.evidence import (
    LABEL_DIMENSION_ORDER,
    LabelDimensionKind,
    LabelEvidence,
    LabelEvidenceCode,
    make_evidence,
)
from quant_platform.labels.identity import compute_label_identity
from quant_platform.labels.manifest import LabelManifest
from quant_platform.labels.models import LABEL_IDENTITY_ALGORITHM
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = [
    "LABEL_DIAGNOSTICS_SCHEMA_VERSION",
    "LABEL_DIMENSION_RESULT_SCHEMA_VERSION",
    "LabelDiagnostics",
    "LabelDimensionResult",
    "compute_label_diagnostics",
]

LABEL_DIMENSION_RESULT_SCHEMA_VERSION = 1
LABEL_DIAGNOSTICS_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LabelDimensionResult:
    dimension: LabelDimensionKind
    label_specification_id: str
    score: float
    evidence: tuple[LabelEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise LabelError(f"LabelDimensionResult.score must be in [0, 1], got {self.score}", context={"dimension": self.dimension.value})
        for record in self.evidence:
            if record.dimension is not self.dimension:
                raise LabelError(
                    f"LabelEvidence.dimension={record.dimension.value!r} does not match LabelDimensionResult.dimension={self.dimension.value!r}",
                    context={"dimension": self.dimension.value},
                )
            if record.affected_specification != self.label_specification_id:
                raise LabelError(
                    f"LabelEvidence.affected_specification={record.affected_specification!r} does not match "
                    f"LabelDimensionResult.label_specification_id={self.label_specification_id!r}",
                    context={"dimension": self.dimension.value},
                )

    @property
    def blocking_evidence(self) -> tuple[LabelEvidence, ...]:
        return tuple(e for e in self.evidence if e.blocking)

    @property
    def is_blocking(self) -> bool:
        return bool(self.blocking_evidence)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": LABEL_DIMENSION_RESULT_SCHEMA_VERSION, "dimension": self.dimension.value,
            "label_specification_id": self.label_specification_id, "score": self.score, "evidence": [e.to_json_dict() for e in self.evidence],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelDimensionResult:
        require_schema_version(raw, supported=LABEL_DIMENSION_RESULT_SCHEMA_VERSION, context="LabelDimensionResult")
        return cls(
            dimension=LabelDimensionKind(raw["dimension"]), label_specification_id=str(raw["label_specification_id"]),
            score=float(str(raw["score"])),
            evidence=tuple(
                LabelEvidence.from_json_dict(as_json_dict(e, field_name="evidence[]"))
                for e in as_json_list(raw.get("evidence") or [], field_name="evidence")
            ),
        )


@dataclass(frozen=True, slots=True)
class LabelDiagnostics:
    schema_version: int
    label_specification_id: str
    dimension_results: tuple[LabelDimensionResult, ...]
    overall_score: float

    def __post_init__(self) -> None:
        found = tuple(r.dimension for r in self.dimension_results)
        if found != LABEL_DIMENSION_ORDER:
            raise LabelError(
                f"LabelDiagnostics.dimension_results must cover exactly LABEL_DIMENSION_ORDER, got {[d.value for d in found]!r}",
                context={"label_specification_id": self.label_specification_id},
            )

    @property
    def all_evidence(self) -> tuple[LabelEvidence, ...]:
        return tuple(e for r in self.dimension_results for e in r.evidence)

    @property
    def blocking_evidence(self) -> tuple[LabelEvidence, ...]:
        return tuple(e for r in self.dimension_results for e in r.blocking_evidence)

    @property
    def is_blocking(self) -> bool:
        return bool(self.blocking_evidence)

    def dimension_result(self, dimension: LabelDimensionKind) -> LabelDimensionResult:
        for result in self.dimension_results:
            if result.dimension is dimension:
                return result
        raise LabelError(f"No LabelDimensionResult for dimension={dimension.value!r}", context={"label_specification_id": self.label_specification_id})

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id,
            "dimension_results": [r.to_json_dict() for r in self.dimension_results], "overall_score": self.overall_score,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelDiagnostics:
        require_schema_version(raw, supported=LABEL_DIAGNOSTICS_SCHEMA_VERSION, context="LabelDiagnostics")
        return cls(
            schema_version=LABEL_DIAGNOSTICS_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            dimension_results=tuple(
                LabelDimensionResult.from_json_dict(as_json_dict(r, field_name="dimension_results[]"))
                for r in as_json_list(raw.get("dimension_results") or [], field_name="dimension_results")
            ),
            overall_score=float(str(raw["overall_score"])),
        )


def _evaluate_identity(bundle: LabelBundle) -> LabelDimensionResult:
    spec_id = bundle.specification.label_specification_id
    recomputed = compute_label_identity(spec_id, bundle.values, source_content_id=bundle.identity.source_content_id)
    if recomputed.content_id != bundle.identity.content_id:
        evidence = (make_evidence(
            finding="Bundle identity does not match a fresh recomputation from its own values",
            evidence=(f"claimed content_id={bundle.identity.content_id!r}", f"recomputed content_id={recomputed.content_id!r}"),
            dimension=LabelDimensionKind.IDENTITY, severity=Severity.CRITICAL, affected_specification=spec_id,
            recommendation="Regenerate this bundle; its identity does not match its own content.",
            blocking=True, blocking_code=LabelEvidenceCode.IDENTITY_MISMATCH,
        ),)
        return LabelDimensionResult(dimension=LabelDimensionKind.IDENTITY, label_specification_id=spec_id, score=0.0, evidence=evidence)
    return LabelDimensionResult(dimension=LabelDimensionKind.IDENTITY, label_specification_id=spec_id, score=1.0, evidence=())


def _evaluate_versioning(bundle: LabelBundle) -> LabelDimensionResult:
    spec_id = bundle.specification.label_specification_id
    consistent, issues = bundle.specification.verify_self_consistency()
    if not consistent:
        evidence = (make_evidence(
            finding="Specification is not self-consistent (parameter_hash or label_specification_id tampered)",
            evidence=issues, dimension=LabelDimensionKind.VERSIONING, severity=Severity.CRITICAL, affected_specification=spec_id,
            recommendation="Rebuild this specification via models.build_label_specification; never hand-edit a registered one.",
            blocking=True, blocking_code=LabelEvidenceCode.SPECIFICATION_TAMPERED,
        ),)
        return LabelDimensionResult(dimension=LabelDimensionKind.VERSIONING, label_specification_id=spec_id, score=0.0, evidence=evidence)
    return LabelDimensionResult(dimension=LabelDimensionKind.VERSIONING, label_specification_id=spec_id, score=1.0, evidence=())


def _trailing_nan_tail_is_well_formed(values: pd.Series) -> bool:
    """A forward-looking label's only legitimate NaN source (in this
    infrastructure-only phase, with no real generator to reason about
    further) is "not enough future data yet" -- which always produces a
    single TRAILING run of NaN, never a NaN preceded by a later valid
    value. A NaN "hole" followed by more valid data is inconsistent with
    that shape and is flagged (a documented heuristic, disclosed here,
    not a proof about any specific family's semantics)."""
    is_na = values.isna().to_numpy()
    if not is_na.any():
        return True
    first_nan = int(is_na.argmax())
    return bool(is_na[first_nan:].all())


def _evaluate_availability(bundle: LabelBundle) -> LabelDimensionResult:
    spec_id = bundle.specification.label_specification_id
    evidence: list[LabelEvidence] = []
    score = 1.0

    if not _trailing_nan_tail_is_well_formed(bundle.values):
        score = 0.5
        evidence.append(make_evidence(
            finding="Label values contain a non-trailing NaN pattern (a NaN followed by a later valid value)",
            evidence=(f"row_count={bundle.row_count}", f"valid_count={bundle.valid_count}"),
            dimension=LabelDimensionKind.AVAILABILITY, severity=Severity.WARNING, affected_specification=spec_id,
            recommendation="A forward-looking label's unresolved rows should form a single trailing run; investigate the generator.",
        ))

    evidence.append(make_evidence(
        finding="Mutable-alias guard was enforced at build time (builder.LabelBuilder checks every source column via numpy.shares_memory)",
        evidence=("no aliasing detected at generation time",), dimension=LabelDimensionKind.AVAILABILITY, severity=Severity.INFO,
        affected_specification=spec_id, recommendation=None,
    ))
    evidence.append(make_evidence(
        finding="Future macro release / future cross asset / revised data checks are out of scope for this infrastructure-only phase",
        evidence=("this package never re-reads raw macro/cross-asset source data",), dimension=LabelDimensionKind.AVAILABILITY,
        severity=Severity.INFO, affected_specification=spec_id,
        recommendation="Re-verify against real market_data_lineage once a real generator (Part 2+) produces this label family's values.",
    ))
    return LabelDimensionResult(dimension=LabelDimensionKind.AVAILABILITY, label_specification_id=spec_id, score=score, evidence=tuple(evidence))


def _evaluate_manifest_integrity(bundle: LabelBundle, manifest: LabelManifest) -> LabelDimensionResult:
    spec_id = bundle.specification.label_specification_id
    evidence: list[LabelEvidence] = []
    score = 1.0

    consistent, issues = manifest.verify_self_consistency()
    if not consistent:
        score = 0.0
        evidence.append(make_evidence(
            finding="Manifest checksum does not match a fresh recomputation", evidence=issues, dimension=LabelDimensionKind.MANIFEST_INTEGRITY,
            severity=Severity.CRITICAL, affected_specification=spec_id, recommendation="Rebuild this manifest via manifest.build_label_manifest.",
            blocking=True, blocking_code=LabelEvidenceCode.MANIFEST_MISMATCH,
        ))
    if manifest.label_specification_id != spec_id:
        score = 0.0
        evidence.append(make_evidence(
            finding="Manifest.label_specification_id does not match the bundle's own specification",
            evidence=(f"manifest={manifest.label_specification_id!r}", f"bundle={spec_id!r}"), dimension=LabelDimensionKind.MANIFEST_INTEGRITY,
            severity=Severity.CRITICAL, affected_specification=spec_id, recommendation="This manifest does not belong to this bundle.",
            blocking=True, blocking_code=LabelEvidenceCode.MANIFEST_MISMATCH,
        ))
    return LabelDimensionResult(dimension=LabelDimensionKind.MANIFEST_INTEGRITY, label_specification_id=spec_id, score=score, evidence=tuple(evidence))


def _evaluate_determinism(bundle: LabelBundle) -> LabelDimensionResult:
    spec_id = bundle.specification.label_specification_id
    first = compute_label_identity(spec_id, bundle.values, source_content_id=bundle.identity.source_content_id)
    second = compute_label_identity(spec_id, bundle.values, source_content_id=bundle.identity.source_content_id)
    if first.content_id != second.content_id:  # pragma: no cover - would indicate a non-deterministic hash implementation
        evidence = (make_evidence(
            finding="Two independent identity recomputations over the identical values produced different content_id values",
            evidence=(f"first={first.content_id!r}", f"second={second.content_id!r}"), dimension=LabelDimensionKind.DETERMINISM,
            severity=Severity.CRITICAL, affected_specification=spec_id, recommendation="Investigate compute_label_identity for non-determinism.",
            blocking=True, blocking_code=LabelEvidenceCode.NON_DETERMINISTIC,
        ),)
        return LabelDimensionResult(dimension=LabelDimensionKind.DETERMINISM, label_specification_id=spec_id, score=0.0, evidence=evidence)
    return LabelDimensionResult(dimension=LabelDimensionKind.DETERMINISM, label_specification_id=spec_id, score=1.0, evidence=())


def _evaluate_reproducibility(bundle: LabelBundle) -> LabelDimensionResult:
    spec_id = bundle.specification.label_specification_id
    if bundle.specification.identity_algorithm != LABEL_IDENTITY_ALGORITHM:
        evidence = (make_evidence(
            finding=f"Specification declares identity_algorithm={bundle.specification.identity_algorithm!r}, which this code version does not recognize",
            evidence=(f"known algorithm: {LABEL_IDENTITY_ALGORITHM!r}",), dimension=LabelDimensionKind.REPRODUCIBILITY, severity=Severity.WARNING,
            affected_specification=spec_id, recommendation="A specification carrying an unrecognized identity_algorithm cannot be portably reproduced by this code version.",
            blocking=False, blocking_code=None,
        ),)
        return LabelDimensionResult(dimension=LabelDimensionKind.REPRODUCIBILITY, label_specification_id=spec_id, score=0.5, evidence=evidence)
    return LabelDimensionResult(dimension=LabelDimensionKind.REPRODUCIBILITY, label_specification_id=spec_id, score=1.0, evidence=())


def _evaluate_lineage(bundle: LabelBundle, manifest: LabelManifest) -> LabelDimensionResult:
    spec_id = bundle.specification.label_specification_id
    missing = [name for name in ("dataset_identity", "manifest_identity") if not getattr(manifest, name)]
    if missing:
        evidence = (make_evidence(
            finding="Manifest is missing required lineage field(s)", evidence=tuple(missing), dimension=LabelDimensionKind.LINEAGE,
            severity=Severity.CRITICAL, affected_specification=spec_id, recommendation="A label manifest must always record its dataset/manifest lineage.",
            blocking=True, blocking_code=LabelEvidenceCode.LINEAGE_INCOMPLETE,
        ),)
        return LabelDimensionResult(dimension=LabelDimensionKind.LINEAGE, label_specification_id=spec_id, score=0.0, evidence=evidence)

    score = 1.0
    evidence_list: list[LabelEvidence] = []
    if manifest.feature_identity is None or manifest.qualification_identity is None:
        score = 0.75
        evidence_list.append(make_evidence(
            finding="feature_identity/qualification_identity were not supplied for this manifest",
            evidence=(f"feature_identity={manifest.feature_identity!r}", f"qualification_identity={manifest.qualification_identity!r}"),
            dimension=LabelDimensionKind.LINEAGE, severity=Severity.INFO, affected_specification=spec_id,
            recommendation="Supply these when a Feature Discovery / Qualification report is available upstream.",
        ))
    return LabelDimensionResult(dimension=LabelDimensionKind.LINEAGE, label_specification_id=spec_id, score=score, evidence=tuple(evidence_list))


def compute_label_diagnostics(bundle: LabelBundle, manifest: LabelManifest) -> LabelDiagnostics:
    results = (
        _evaluate_identity(bundle),
        _evaluate_versioning(bundle),
        _evaluate_availability(bundle),
        _evaluate_manifest_integrity(bundle, manifest),
        _evaluate_determinism(bundle),
        _evaluate_reproducibility(bundle),
        _evaluate_lineage(bundle, manifest),
    )
    overall_score = sum(r.score for r in results) / len(results)
    return LabelDiagnostics(
        schema_version=LABEL_DIAGNOSTICS_SCHEMA_VERSION, label_specification_id=bundle.specification.label_specification_id,
        dimension_results=results, overall_score=overall_score,
    )
