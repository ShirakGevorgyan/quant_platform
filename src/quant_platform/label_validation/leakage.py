"""Leakage validation (Milestone 11, Phase 3, Part C): future
timestamps, future macro releases, future cross-asset values, future
revisions, availability violations, barrier violations. An INDEPENDENT
re-verification -- this module deliberately does NOT call `labels.
diagnostics`'s own point-in-time checks; it re-derives its own
conclusions from the bundle/manifest/records alone, mirroring this
platform's established "verify, never trust a sibling package's cached
conclusion" discipline (see `feature_discovery.infra_verification`'s
identical `verify_bundle_self_consistency` note).

Two of the 6 named checks ("future macro release", "future cross
asset") are NOT independently verifiable from a bundle/manifest/records
alone -- doing so would require re-reading raw macro/cross-asset source
data, which is out of THIS package's scope too (`label_validation`
never imports `market_data`/`features`, exactly like `labels/` itself).
Reported honestly as INFO-severity, non-blocking evidence rather than a
fabricated check -- the same disclosed-scope-boundary discipline
`qualification`/`labels.diagnostics` already established. "Future
revisions" is out of scope for the identical reason."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.historical.quality import Severity
from quant_platform.label_validation.evidence import (
    BlockingFindingCode,
    LabelEvidence,
    LabelValidationDimensionKind,
    make_evidence,
)
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.identity import compute_label_identity
from quant_platform.labels.manifest import LabelManifest
from quant_platform.labels.models import LabelFamily
from quant_platform.labels.records import LabelRecord
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = ["LEAKAGE_VALIDATION_SCHEMA_VERSION", "LeakageValidationResult", "validate_leakage"]

LEAKAGE_VALIDATION_SCHEMA_VERSION = 1

_BARRIER_DOMAIN = frozenset({-1.0, 0.0, 1.0})


def _trailing_nan_tail_is_well_formed(values: pd.Series) -> bool:
    """Independently reimplemented (not imported from `labels.
    diagnostics._trailing_nan_tail_is_well_formed`) -- the SAME shape
    check, re-derived rather than trusted."""
    is_na = values.isna().to_numpy()
    if not is_na.any():
        return True
    first_nan = int(is_na.argmax())
    return bool(is_na[first_nan:].all())


@dataclass(frozen=True, slots=True)
class LeakageValidationResult:
    schema_version: int
    label_specification_id: str
    trailing_nan_tail_well_formed: bool
    availability_time_consistent: bool | None
    """`None` when no `LabelRecord`s were supplied -- there is nothing to check."""
    barrier_domain_valid: bool
    identity_consistent: bool
    """Fresh `labels.identity.compute_label_identity` recomputation over
    `bundle.values` matches `bundle.identity.content_id` -- an
    INDEPENDENT tamper check, never trusting the bundle's own claim."""
    manifest_self_consistent: bool
    records_self_consistent: bool | None
    """`None` when no `LabelRecord`s were supplied."""
    evidence: tuple[LabelEvidence, ...]

    @property
    def is_blocking(self) -> bool:
        return any(e.blocking for e in self.evidence)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id,
            "trailing_nan_tail_well_formed": self.trailing_nan_tail_well_formed,
            "availability_time_consistent": self.availability_time_consistent, "barrier_domain_valid": self.barrier_domain_valid,
            "identity_consistent": self.identity_consistent, "manifest_self_consistent": self.manifest_self_consistent,
            "records_self_consistent": self.records_self_consistent, "evidence": [e.to_json_dict() for e in self.evidence],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LeakageValidationResult:
        require_schema_version(raw, supported=LEAKAGE_VALIDATION_SCHEMA_VERSION, context="LeakageValidationResult")
        availability_raw = raw.get("availability_time_consistent")
        records_raw = raw.get("records_self_consistent")
        return cls(
            schema_version=LEAKAGE_VALIDATION_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            trailing_nan_tail_well_formed=bool(raw["trailing_nan_tail_well_formed"]),
            availability_time_consistent=(None if availability_raw is None else bool(availability_raw)),
            barrier_domain_valid=bool(raw["barrier_domain_valid"]), identity_consistent=bool(raw["identity_consistent"]),
            manifest_self_consistent=bool(raw["manifest_self_consistent"]),
            records_self_consistent=(None if records_raw is None else bool(records_raw)),
            evidence=tuple(
                LabelEvidence.from_json_dict(as_json_dict(e, field_name="evidence[]"))
                for e in as_json_list(raw.get("evidence") or [], field_name="evidence")
            ),
        )


def validate_leakage(bundle: LabelBundle, manifest: LabelManifest, *, records: tuple[LabelRecord, ...] | None = None) -> LeakageValidationResult:
    spec_id = bundle.specification.label_specification_id
    evidence: list[LabelEvidence] = []

    if manifest.label_specification_id != spec_id:
        evidence.append(make_evidence(
            finding="manifest.label_specification_id does not match the bundle being validated -- availability semantics cannot be trusted",
            evidence=(f"manifest.label_specification_id={manifest.label_specification_id!r}", f"bundle.specification.label_specification_id={spec_id!r}"),
            dimension=LabelValidationDimensionKind.LEAKAGE, severity=Severity.CRITICAL, affected_labels=(spec_id,),
            blocking=True, blocking_code=BlockingFindingCode.MANIFEST_MISMATCH,
        ))

    manifest_self_consistent, manifest_issues = manifest.verify_self_consistency()
    if not manifest_self_consistent:
        evidence.append(make_evidence(
            finding="manifest is not self-consistent (its own checksum does not match a fresh recomputation)", evidence=manifest_issues,
            dimension=LabelValidationDimensionKind.LEAKAGE, severity=Severity.CRITICAL, affected_labels=(spec_id,),
            blocking=True, blocking_code=BlockingFindingCode.MANIFEST_MISMATCH,
        ))

    recomputed_identity = compute_label_identity(spec_id, bundle.values, source_content_id=bundle.identity.source_content_id)
    identity_consistent = recomputed_identity.content_id == bundle.identity.content_id
    if not identity_consistent:
        evidence.append(make_evidence(
            finding="bundle identity does not match a fresh recomputation from its own values", evidence=(
                f"claimed content_id={bundle.identity.content_id!r}", f"recomputed content_id={recomputed_identity.content_id!r}",
            ),
            dimension=LabelValidationDimensionKind.LEAKAGE, severity=Severity.CRITICAL, affected_labels=(spec_id,),
            blocking=True, blocking_code=BlockingFindingCode.IDENTITY_MISMATCH,
        ))

    records_self_consistent: bool | None = None
    if records is not None:
        inconsistent_records = [r for r in records if not r.verify_self_consistency()[0]]
        records_self_consistent = not inconsistent_records
        if inconsistent_records:
            evidence.append(make_evidence(
                finding=f"{len(inconsistent_records)} record(s) are not self-consistent (tampered label_id/content_hash/row_identity)",
                evidence=(f"first_inconsistent_row_identity={inconsistent_records[0].row_identity}",), dimension=LabelValidationDimensionKind.LEAKAGE,
                severity=Severity.CRITICAL, affected_labels=(spec_id,), statistics={"inconsistent_record_count": float(len(inconsistent_records))},
                blocking=True, blocking_code=BlockingFindingCode.IDENTITY_MISMATCH,
            ))

    trailing_ok = _trailing_nan_tail_is_well_formed(bundle.values)
    if not trailing_ok:
        evidence.append(make_evidence(
            finding="label values contain a non-trailing NaN pattern (future-visibility shape anomaly)", evidence=(f"label_specification_id={spec_id}",),
            dimension=LabelValidationDimensionKind.LEAKAGE, severity=Severity.WARNING, affected_labels=(spec_id,),
        ))

    availability_consistent: bool | None = None
    if records is not None:
        violations = [r for r in records if pd.Timestamp(r.availability_time) < pd.Timestamp(r.event_time)]
        availability_consistent = not violations
        if violations:
            evidence.append(make_evidence(
                finding=f"{len(violations)} record(s) claim availability_time before their own event_time",
                evidence=(f"first_violation_row_identity={violations[0].row_identity}",), dimension=LabelValidationDimensionKind.LEAKAGE,
                severity=Severity.CRITICAL, affected_labels=(spec_id,), statistics={"violation_count": float(len(violations))},
                blocking=True, blocking_code=BlockingFindingCode.AVAILABILITY_VIOLATION,
            ))

    barrier_domain_valid = True
    if bundle.specification.label_family is LabelFamily.TRIPLE_BARRIER:
        valid = bundle.values.dropna()
        violation_count = int((~valid.isin(_BARRIER_DOMAIN)).sum())
        barrier_domain_valid = violation_count == 0
        if not barrier_domain_valid:
            evidence.append(make_evidence(
                finding=f"{violation_count} triple-barrier value(s) fall outside {{-1, 0, 1}}", evidence=(f"violation_count={violation_count}",),
                dimension=LabelValidationDimensionKind.LEAKAGE, severity=Severity.CRITICAL, affected_labels=(spec_id,),
                statistics={"violation_count": float(violation_count)}, blocking=True, blocking_code=BlockingFindingCode.BARRIER_VIOLATION,
            ))

    evidence.append(make_evidence(
        finding="future macro release / future cross-asset checks are out of scope for this package",
        evidence=("label_validation never reads raw macro/cross-asset source data",), dimension=LabelValidationDimensionKind.LEAKAGE,
        severity=Severity.INFO, affected_labels=(spec_id,),
    ))

    return LeakageValidationResult(
        schema_version=LEAKAGE_VALIDATION_SCHEMA_VERSION, label_specification_id=spec_id, trailing_nan_tail_well_formed=trailing_ok,
        availability_time_consistent=availability_consistent, barrier_domain_valid=barrier_domain_valid, identity_consistent=identity_consistent,
        manifest_self_consistent=manifest_self_consistent, records_self_consistent=records_self_consistent, evidence=tuple(evidence),
    )
