"""`FeatureInfrastructureVerifier` (Milestone 11, Phase 2, Part 2):
independent verification of the dependency graph, lineage, manifest,
identity, metadata, and catalog. Never trusts a cached
`FeatureInfrastructureBundle`. Two independent checks:

1. `verify_bundle_self_consistency` -- pure, no I/O -- independently
   recomputes `FeatureManifest.manifest_id` from `snapshot.metadata`'s
   own `feature_id`s (a small, deliberately separate reimplementation
   of `catalog.build_feature_manifest`'s own hash formula -- a bug
   shared between the two would otherwise go undetected) and compares
   against what the bundle's `manifest` field itself claims. Also
   confirms every catalog entry's `deterministic_identity` still
   matches a fresh `FeatureSpec.fingerprint()` recomputation wherever a
   live `registry` is supplied (identity verification).
2. Full re-capture -- a fresh `catalog.build_feature_infrastructure_
   bundle()` run against the live `registry`/`manifest`, diffed against
   the supplied bundle via `FeatureInfrastructureReconciliation`.

A mismatch in either is a normal, non-raising outcome
(`FeatureInfrastructureVerificationResult.verified=False`), never an
exception."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from quant_platform.core.exceptions import UnknownFeatureError
from quant_platform.feature_discovery.catalog import (
    FeatureInfrastructureBundle,
    build_feature_infrastructure_bundle,
)
from quant_platform.feature_discovery.infra_reconciliation import (
    FEATURE_INFRASTRUCTURE_RECONCILIATION_SCHEMA_VERSION,
    FeatureInfrastructureReconciliation,
    FeatureInfrastructureReconciliationIssue,
    FeatureInfrastructureReconciliationResult,
)
from quant_platform.features.manifests import ResearchDatasetManifest
from quant_platform.features.registry import FeatureRegistry
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)

__all__ = [
    "FEATURE_INFRASTRUCTURE_VERIFICATION_SCHEMA_VERSION",
    "FeatureInfrastructureVerificationResult",
    "FeatureInfrastructureVerifier",
    "verify_bundle_self_consistency",
]

FEATURE_INFRASTRUCTURE_VERIFICATION_SCHEMA_VERSION = 1


def verify_bundle_self_consistency(bundle: FeatureInfrastructureBundle, *, registry: FeatureRegistry | None = None) -> tuple[bool, tuple[str, ...]]:
    issues: list[str] = []

    feature_ids = tuple(sorted(m.feature_id for m in bundle.snapshot.metadata))
    recomputed_manifest_id = hashlib.sha256(f"{bundle.snapshot.dataset_id}|{','.join(feature_ids)}".encode()).hexdigest()[:16]
    if recomputed_manifest_id != bundle.manifest.manifest_id:
        issues.append(f"manifest_id: bundle claims {bundle.manifest.manifest_id!r}, recomputed from snapshot.metadata is {recomputed_manifest_id!r}")

    if bundle.manifest.feature_ids != feature_ids:
        issues.append(f"manifest.feature_ids: bundle claims {bundle.manifest.feature_ids!r}, recomputed from snapshot.metadata is {feature_ids!r}")

    if registry is not None:
        for entry in bundle.snapshot.metadata:
            try:
                spec = registry.get(entry.feature_name, entry.feature_id.rsplit("@", 1)[-1]).spec
            except UnknownFeatureError:
                issues.append(f"identity: feature {entry.feature_name!r} (feature_id={entry.feature_id!r}) not resolvable in the supplied registry")
                continue
            fresh_fingerprint = spec.fingerprint()
            if fresh_fingerprint != entry.deterministic_identity:
                issues.append(
                    f"identity: feature {entry.feature_name!r} deterministic_identity={entry.deterministic_identity!r} does not match "
                    f"a fresh FeatureSpec.fingerprint() recomputation {fresh_fingerprint!r}"
                )

    return (not issues, tuple(issues))


@dataclass(frozen=True, slots=True)
class FeatureInfrastructureVerificationResult:
    schema_version: int
    dataset_id: str
    manifest_id: str
    verified: bool
    self_consistent: bool
    self_consistency_issues: tuple[str, ...]
    reconciliation: FeatureInfrastructureReconciliationResult
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "manifest_id": self.manifest_id, "verified": self.verified,
            "self_consistent": self.self_consistent, "self_consistency_issues": list(self.self_consistency_issues),
            "reconciliation": self.reconciliation.to_json_dict(), "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureInfrastructureVerificationResult:
        require_schema_version(raw, supported=FEATURE_INFRASTRUCTURE_VERIFICATION_SCHEMA_VERSION, context="FeatureInfrastructureVerificationResult")
        return cls(
            schema_version=FEATURE_INFRASTRUCTURE_VERIFICATION_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]), manifest_id=str(raw["manifest_id"]),
            verified=bool(raw["verified"]), self_consistent=bool(raw["self_consistent"]),
            self_consistency_issues=tuple(str(s) for s in as_json_list(raw.get("self_consistency_issues") or [], field_name="self_consistency_issues")),
            reconciliation=FeatureInfrastructureReconciliationResult.from_json_dict(as_json_dict(raw["reconciliation"], field_name="reconciliation")),
            generated_at=str(raw["generated_at"]),
        )


class FeatureInfrastructureVerifier:
    def verify(self, bundle: FeatureInfrastructureBundle, registry: FeatureRegistry, manifest: ResearchDatasetManifest) -> FeatureInfrastructureVerificationResult:
        self_consistent, self_consistency_issues = verify_bundle_self_consistency(bundle, registry=registry)

        if bundle.snapshot.dataset_id != manifest.dataset_id:
            reconciliation = FeatureInfrastructureReconciliationResult(
                schema_version=FEATURE_INFRASTRUCTURE_RECONCILIATION_SCHEMA_VERSION, dataset_id=bundle.snapshot.dataset_id,
                baseline_manifest_id=bundle.manifest.manifest_id, candidate_manifest_id=manifest.dataset_id, reconciled=False,
                issues=(FeatureInfrastructureReconciliationIssue(
                    kind="dataset_id_mismatch", feature_name=None,
                    message=f"bundle.snapshot.dataset_id={bundle.snapshot.dataset_id!r} does not match manifest.dataset_id={manifest.dataset_id!r}",
                ),),
                generated_at=format_utc_timestamp(utc_now()),
            )
            return FeatureInfrastructureVerificationResult(
                schema_version=FEATURE_INFRASTRUCTURE_VERIFICATION_SCHEMA_VERSION, dataset_id=bundle.snapshot.dataset_id,
                manifest_id=bundle.manifest.manifest_id, verified=False, self_consistent=self_consistent,
                self_consistency_issues=self_consistency_issues, reconciliation=reconciliation, generated_at=format_utc_timestamp(utc_now()),
            )

        recomputed_bundle = build_feature_infrastructure_bundle(registry, manifest)
        reconciliation = FeatureInfrastructureReconciliation().reconcile(bundle, recomputed_bundle)

        return FeatureInfrastructureVerificationResult(
            schema_version=FEATURE_INFRASTRUCTURE_VERIFICATION_SCHEMA_VERSION, dataset_id=bundle.snapshot.dataset_id,
            manifest_id=bundle.manifest.manifest_id, verified=(self_consistent and reconciliation.reconciled), self_consistent=self_consistent,
            self_consistency_issues=self_consistency_issues, reconciliation=reconciliation, generated_at=format_utc_timestamp(utc_now()),
        )
