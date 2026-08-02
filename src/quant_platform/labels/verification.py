"""`LabelVerifier` (Milestone 11, Phase 3, Part A): independent
verification of a `builder.LabelBundle` + `manifest.LabelManifest` pair.
Never trusts a cached bundle. Two checks, mirroring `qualification.
verification`/`feature_discovery.infra_verification`'s established
pattern exactly:

1. `verify_bundle_self_consistency` -- pure, no I/O -- independently
   recomputes the bundle's identity from its own `values`, the
   specification's own self-consistency, and the manifest's own
   checksum, comparing each against what the objects themselves claim.
2. Full re-derivation -- a FRESH `builder.LabelBuilder.build()` call
   against the same `LabelDefinition`/source data/`source_content_id`,
   diffed against the supplied bundle via `reconciliation.
   LabelReconciliation`.

A mismatch in either is a normal, non-raising outcome
(`LabelVerificationResult.verified=False`), never an exception.
`LabelVerificationError` is reserved for genuinely being unable to
ATTEMPT the re-derivation at all."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.core.exceptions import LabelError, LabelVerificationError
from quant_platform.labels.builder import LabelBuilder, LabelBundle, LabelDefinition
from quant_platform.labels.identity import compute_label_identity
from quant_platform.labels.manifest import LabelManifest, build_label_manifest
from quant_platform.labels.reconciliation import LabelReconciliation, LabelReconciliationResult
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)

__all__ = ["LABEL_VERIFICATION_SCHEMA_VERSION", "LabelVerificationResult", "LabelVerifier", "verify_bundle_self_consistency"]

LABEL_VERIFICATION_SCHEMA_VERSION = 1


def verify_bundle_self_consistency(bundle: LabelBundle, manifest: LabelManifest) -> tuple[bool, tuple[str, ...]]:
    issues: list[str] = []

    recomputed_identity = compute_label_identity(
        bundle.specification.label_specification_id, bundle.values, source_content_id=bundle.identity.source_content_id,
    )
    if recomputed_identity.content_id != bundle.identity.content_id:
        issues.append(f"identity.content_id: bundle claims {bundle.identity.content_id!r}, recomputed from values is {recomputed_identity.content_id!r}")

    spec_consistent, spec_issues = bundle.specification.verify_self_consistency()
    if not spec_consistent:
        issues.extend(f"specification: {issue}" for issue in spec_issues)

    manifest_consistent, manifest_issues = manifest.verify_self_consistency()
    if not manifest_consistent:
        issues.extend(f"manifest: {issue}" for issue in manifest_issues)

    if manifest.label_specification_id != bundle.specification.label_specification_id:
        issues.append(
            f"manifest.label_specification_id={manifest.label_specification_id!r} does not match "
            f"bundle.specification.label_specification_id={bundle.specification.label_specification_id!r}"
        )

    return (not issues, tuple(issues))


@dataclass(frozen=True, slots=True)
class LabelVerificationResult:
    schema_version: int
    label_specification_id: str
    verified: bool
    self_consistent: bool
    self_consistency_issues: tuple[str, ...]
    reconciliation: LabelReconciliationResult
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id, "verified": self.verified,
            "self_consistent": self.self_consistent, "self_consistency_issues": list(self.self_consistency_issues),
            "reconciliation": self.reconciliation.to_json_dict(), "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelVerificationResult:
        require_schema_version(raw, supported=LABEL_VERIFICATION_SCHEMA_VERSION, context="LabelVerificationResult")
        return cls(
            schema_version=LABEL_VERIFICATION_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            verified=bool(raw["verified"]), self_consistent=bool(raw["self_consistent"]),
            self_consistency_issues=tuple(str(s) for s in as_json_list(raw.get("self_consistency_issues") or [], field_name="self_consistency_issues")),
            reconciliation=LabelReconciliationResult.from_json_dict(as_json_dict(raw["reconciliation"], field_name="reconciliation")),
            generated_at=str(raw["generated_at"]),
        )


class LabelVerifier:
    def verify(
        self, bundle: LabelBundle, manifest: LabelManifest, definition: LabelDefinition, source_data: pd.DataFrame, *, source_content_id: str,
    ) -> LabelVerificationResult:
        self_consistent, self_consistency_issues = verify_bundle_self_consistency(bundle, manifest)

        try:
            fresh_bundle = LabelBuilder().build(definition, source_data, source_content_id=source_content_id)
            fresh_manifest = build_label_manifest(
                definition.specification, generation_timestamp=format_utc_timestamp(utc_now()),
                feature_identity=manifest.feature_identity, qualification_identity=manifest.qualification_identity,
            )
            reconciliation = LabelReconciliation().reconcile(bundle, fresh_bundle, baseline_manifest=manifest, candidate_manifest=fresh_manifest)
        except LabelError as exc:
            raise LabelVerificationError(
                f"LabelVerifier.verify could not complete re-derivation for label_specification_id={bundle.specification.label_specification_id!r}: {exc}",
                context={"label_specification_id": bundle.specification.label_specification_id},
            ) from exc

        return LabelVerificationResult(
            schema_version=LABEL_VERIFICATION_SCHEMA_VERSION, label_specification_id=bundle.specification.label_specification_id,
            verified=(self_consistent and reconciliation.reconciled), self_consistent=self_consistent,
            self_consistency_issues=self_consistency_issues, reconciliation=reconciliation, generated_at=format_utc_timestamp(utc_now()),
        )
