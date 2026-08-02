"""`LabelReconciliation` (Milestone 11, Phase 3, Part A): compares two
`builder.LabelBundle` + `manifest.LabelManifest` pairs for the SAME
`label_specification_id`, detecting 4 drift kinds: specification drift,
identity drift, manifest drift, lineage drift. Mirrors `qualification.
reconciliation`/`feature_discovery.reconciliation`'s established shape
exactly: reconciling two bundles for DIFFERENT specifications is a
structural precondition violation (there is nothing to reconcile) and
raises `LabelReconciliationError`; every other disagreement is a normal,
expected, non-raising `LabelReconciliationIssue`."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.core.exceptions import LabelReconciliationError
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.manifest import LabelManifest
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)

__all__ = ["LABEL_RECONCILIATION_SCHEMA_VERSION", "LabelReconciliation", "LabelReconciliationIssue", "LabelReconciliationResult"]

LABEL_RECONCILIATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LabelReconciliationIssue:
    kind: str
    message: str

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "message": self.message}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelReconciliationIssue:
        return cls(kind=str(raw["kind"]), message=str(raw["message"]))


@dataclass(frozen=True, slots=True)
class LabelReconciliationResult:
    schema_version: int
    label_specification_id: str
    baseline_content_id: str
    candidate_content_id: str
    reconciled: bool
    issues: tuple[LabelReconciliationIssue, ...]
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id,
            "baseline_content_id": self.baseline_content_id, "candidate_content_id": self.candidate_content_id,
            "reconciled": self.reconciled, "issues": [i.to_json_dict() for i in self.issues], "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelReconciliationResult:
        require_schema_version(raw, supported=LABEL_RECONCILIATION_SCHEMA_VERSION, context="LabelReconciliationResult")
        return cls(
            schema_version=LABEL_RECONCILIATION_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            baseline_content_id=str(raw["baseline_content_id"]), candidate_content_id=str(raw["candidate_content_id"]),
            reconciled=bool(raw["reconciled"]),
            issues=tuple(
                LabelReconciliationIssue.from_json_dict(as_json_dict(i, field_name="issues[]"))
                for i in as_json_list(raw.get("issues") or [], field_name="issues")
            ),
            generated_at=str(raw["generated_at"]),
        )


class LabelReconciliation:
    def reconcile(
        self, baseline: LabelBundle, candidate: LabelBundle, *, baseline_manifest: LabelManifest, candidate_manifest: LabelManifest,
    ) -> LabelReconciliationResult:
        baseline_id = baseline.specification.label_specification_id
        candidate_id = candidate.specification.label_specification_id
        if baseline_id != candidate_id:
            raise LabelReconciliationError(
                f"Cannot reconcile bundles for different label_specification_id values: baseline={baseline_id!r} candidate={candidate_id!r}",
                context={"baseline_label_specification_id": baseline_id, "candidate_label_specification_id": candidate_id},
            )

        issues: list[LabelReconciliationIssue] = []
        if baseline.specification != candidate.specification:
            issues.append(LabelReconciliationIssue(
                kind="specification_drift", message=f"specification changed: baseline={baseline.specification} candidate={candidate.specification}",
            ))
        if baseline.identity.content_id != candidate.identity.content_id:
            issues.append(LabelReconciliationIssue(
                kind="identity_drift", message=f"content_id changed: baseline={baseline.identity.content_id} candidate={candidate.identity.content_id}",
            ))
        if baseline_manifest.manifest_checksum != candidate_manifest.manifest_checksum:
            issues.append(LabelReconciliationIssue(
                kind="manifest_drift",
                message=f"manifest_checksum changed: baseline={baseline_manifest.manifest_checksum} candidate={candidate_manifest.manifest_checksum}",
            ))
        baseline_lineage = (baseline_manifest.dataset_identity, baseline_manifest.manifest_identity, baseline_manifest.feature_identity, baseline_manifest.qualification_identity)
        candidate_lineage = (candidate_manifest.dataset_identity, candidate_manifest.manifest_identity, candidate_manifest.feature_identity, candidate_manifest.qualification_identity)
        if baseline_lineage != candidate_lineage:
            issues.append(LabelReconciliationIssue(kind="lineage_drift", message=f"lineage changed: baseline={baseline_lineage} candidate={candidate_lineage}"))

        return LabelReconciliationResult(
            schema_version=LABEL_RECONCILIATION_SCHEMA_VERSION, label_specification_id=baseline_id, baseline_content_id=baseline.identity.content_id,
            candidate_content_id=candidate.identity.content_id, reconciled=not issues, issues=tuple(issues), generated_at=format_utc_timestamp(utc_now()),
        )
