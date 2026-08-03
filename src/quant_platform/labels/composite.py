"""`CompositeLabelBundle` (Milestone 11, Phase 3, Part B): "Label
Bundles" -- a deterministic grouping of several independently-generated,
independently-identified `builder.LabelBundle`s (e.g. Return +
Direction, Return + Volatility, Direction + Triple Barrier, Return +
Direction + Volatility) under one content-addressed `composite_id`.

This is NOT "one family depending on another's output" -- every member
bundle is generated independently (via its own `LabelDefinition` and
Part A's `builder.LabelBuilder`, never by one family reading another's
values) and simply grouped together afterward. `composite_id` is a
sha256 over `(dataset_id, sorted member content_ids)` -- adding,
removing, or regenerating any member with different values always
changes it.

Verification/replay/reconciliation at the composite level are thin
aggregations over Part A's own single-bundle `LabelVerifier`/
`LabelReplay`/`LabelReconciliation` -- one call per member, results
collected -- never a parallel reimplementation of any of the three."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from quant_platform.core.exceptions import LabelReconciliationError, LabelRequestError, LabelVerificationError
from quant_platform.labels.builder import LabelBuilder, LabelBundle, LabelDefinition
from quant_platform.labels.manifest import LabelManifest
from quant_platform.labels.reconciliation import LabelReconciliation, LabelReconciliationResult
from quant_platform.labels.replay import LabelReplay, LabelReplayResult
from quant_platform.labels.verification import LabelVerificationResult, LabelVerifier
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)

__all__ = [
    "COMPOSITE_LABEL_BUNDLE_SCHEMA_VERSION",
    "CompositeLabelBundle",
    "CompositeReconciliationResult",
    "CompositeReplayResult",
    "CompositeVerificationResult",
    "build_composite_from_definitions",
    "build_composite_label_bundle",
    "compute_composite_id",
    "reconcile_composite",
    "replay_composite",
    "verify_composite",
]

COMPOSITE_LABEL_BUNDLE_SCHEMA_VERSION = 1


def compute_composite_id(dataset_id: str, member_content_ids: tuple[str, ...]) -> str:
    payload = json.dumps({"dataset_id": dataset_id, "member_content_ids": sorted(member_content_ids)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CompositeLabelBundle:
    schema_version: int
    composite_id: str
    dataset_id: str
    members: tuple[LabelBundle, ...]
    """Sorted by `label_specification_id` -- deterministic regardless of
    the order members were generated/supplied in."""
    generated_at: str

    def member(self, label_specification_id: str) -> LabelBundle:
        for m in self.members:
            if m.specification.label_specification_id == label_specification_id:
                return m
        raise KeyError(label_specification_id)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "composite_id": self.composite_id, "dataset_id": self.dataset_id,
            "members": [m.to_json_dict() for m in self.members], "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CompositeLabelBundle:
        require_schema_version(raw, supported=COMPOSITE_LABEL_BUNDLE_SCHEMA_VERSION, context="CompositeLabelBundle")
        return cls(
            schema_version=COMPOSITE_LABEL_BUNDLE_SCHEMA_VERSION, composite_id=str(raw["composite_id"]), dataset_id=str(raw["dataset_id"]),
            members=tuple(
                LabelBundle.from_json_dict(as_json_dict(m, field_name="members[]"))
                for m in as_json_list(raw.get("members") or [], field_name="members")
            ),
            generated_at=str(raw["generated_at"]),
        )

    def verify_self_consistency(self) -> tuple[bool, tuple[str, ...]]:
        member_content_ids = tuple(m.identity.content_id for m in self.members)
        recomputed = compute_composite_id(self.dataset_id, member_content_ids)
        if recomputed != self.composite_id:
            return False, (f"composite_id: claims {self.composite_id!r}, recomputed is {recomputed!r}",)
        return True, ()


def build_composite_label_bundle(dataset_id: str, members: tuple[LabelBundle, ...]) -> CompositeLabelBundle:
    if not members:
        raise LabelRequestError("A composite label bundle requires at least one member", context={"dataset_id": dataset_id})
    spec_ids = [m.specification.label_specification_id for m in members]
    if len(set(spec_ids)) != len(spec_ids):
        raise LabelRequestError("Composite members must have distinct label_specification_id values", context={"dataset_id": dataset_id})

    sorted_members = tuple(sorted(members, key=lambda m: m.specification.label_specification_id))
    member_content_ids = tuple(m.identity.content_id for m in sorted_members)
    composite_id = compute_composite_id(dataset_id, member_content_ids)
    return CompositeLabelBundle(
        schema_version=COMPOSITE_LABEL_BUNDLE_SCHEMA_VERSION, composite_id=composite_id, dataset_id=dataset_id, members=sorted_members,
        generated_at=format_utc_timestamp(utc_now()),
    )


def build_composite_from_definitions(
    definitions: tuple[LabelDefinition, ...], source_data: pd.DataFrame, *, dataset_id: str, source_content_id: str,
) -> CompositeLabelBundle:
    builder = LabelBuilder()
    members = tuple(builder.build(definition, source_data, source_content_id=source_content_id) for definition in definitions)
    return build_composite_label_bundle(dataset_id, members)


@dataclass(frozen=True, slots=True)
class CompositeVerificationResult:
    schema_version: int
    composite_id: str
    verified: bool
    self_consistent: bool
    member_results: tuple[LabelVerificationResult, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "composite_id": self.composite_id, "verified": self.verified,
            "self_consistent": self.self_consistent, "member_results": [r.to_json_dict() for r in self.member_results],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CompositeVerificationResult:
        require_schema_version(raw, supported=COMPOSITE_LABEL_BUNDLE_SCHEMA_VERSION, context="CompositeVerificationResult")
        return cls(
            schema_version=COMPOSITE_LABEL_BUNDLE_SCHEMA_VERSION, composite_id=str(raw["composite_id"]), verified=bool(raw["verified"]),
            self_consistent=bool(raw["self_consistent"]),
            member_results=tuple(
                LabelVerificationResult.from_json_dict(as_json_dict(r, field_name="member_results[]"))
                for r in as_json_list(raw.get("member_results") or [], field_name="member_results")
            ),
        )


def verify_composite(
    composite: CompositeLabelBundle, manifests: tuple[LabelManifest, ...], definitions: tuple[LabelDefinition, ...], source_data: pd.DataFrame,
    *, source_content_id: str,
) -> CompositeVerificationResult:
    self_consistent, _issues = composite.verify_self_consistency()
    manifest_by_spec = {m.label_specification_id: m for m in manifests}
    definition_by_spec = {d.label_specification_id: d for d in definitions}

    results = []
    for member in composite.members:
        spec_id = member.specification.label_specification_id
        manifest = manifest_by_spec.get(spec_id)
        definition = definition_by_spec.get(spec_id)
        if manifest is None or definition is None:
            raise LabelVerificationError(
                f"No manifest/definition supplied for composite member label_specification_id={spec_id!r}", context={"label_specification_id": spec_id},
            )
        results.append(LabelVerifier().verify(member, manifest, definition, source_data, source_content_id=source_content_id))

    verified = self_consistent and all(r.verified for r in results)
    return CompositeVerificationResult(
        schema_version=COMPOSITE_LABEL_BUNDLE_SCHEMA_VERSION, composite_id=composite.composite_id, verified=verified,
        self_consistent=self_consistent, member_results=tuple(results),
    )


@dataclass(frozen=True, slots=True)
class CompositeReplayResult:
    schema_version: int
    composite_id: str
    replayed: bool
    member_results: tuple[LabelReplayResult, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "composite_id": self.composite_id, "replayed": self.replayed,
            "member_results": [r.to_json_dict() for r in self.member_results],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CompositeReplayResult:
        require_schema_version(raw, supported=COMPOSITE_LABEL_BUNDLE_SCHEMA_VERSION, context="CompositeReplayResult")
        return cls(
            schema_version=COMPOSITE_LABEL_BUNDLE_SCHEMA_VERSION, composite_id=str(raw["composite_id"]), replayed=bool(raw["replayed"]),
            member_results=tuple(
                LabelReplayResult.from_json_dict(as_json_dict(r, field_name="member_results[]"))
                for r in as_json_list(raw.get("member_results") or [], field_name="member_results")
            ),
        )


def replay_composite(composite: CompositeLabelBundle, definitions: tuple[LabelDefinition, ...], source_data: pd.DataFrame, *, source_content_id: str) -> CompositeReplayResult:
    definition_by_spec = {d.label_specification_id: d for d in definitions}
    results = []
    for member in composite.members:
        spec_id = member.specification.label_specification_id
        definition = definition_by_spec.get(spec_id)
        if definition is None:
            raise LabelRequestError(f"No definition supplied for composite member label_specification_id={spec_id!r}", context={"label_specification_id": spec_id})
        results.append(LabelReplay().replay(definition, source_data, source_content_id=source_content_id, original=member))
    return CompositeReplayResult(
        schema_version=COMPOSITE_LABEL_BUNDLE_SCHEMA_VERSION, composite_id=composite.composite_id, replayed=all(r.replayed for r in results),
        member_results=tuple(results),
    )


@dataclass(frozen=True, slots=True)
class CompositeReconciliationIssue:
    kind: str
    message: str

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "message": self.message}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CompositeReconciliationIssue:
        return cls(kind=str(raw["kind"]), message=str(raw["message"]))


@dataclass(frozen=True, slots=True)
class CompositeReconciliationResult:
    schema_version: int
    baseline_composite_id: str
    candidate_composite_id: str
    reconciled: bool
    issues: tuple[CompositeReconciliationIssue, ...]
    member_results: tuple[LabelReconciliationResult, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "baseline_composite_id": self.baseline_composite_id,
            "candidate_composite_id": self.candidate_composite_id, "reconciled": self.reconciled,
            "issues": [i.to_json_dict() for i in self.issues], "member_results": [r.to_json_dict() for r in self.member_results],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CompositeReconciliationResult:
        require_schema_version(raw, supported=COMPOSITE_LABEL_BUNDLE_SCHEMA_VERSION, context="CompositeReconciliationResult")
        return cls(
            schema_version=COMPOSITE_LABEL_BUNDLE_SCHEMA_VERSION, baseline_composite_id=str(raw["baseline_composite_id"]),
            candidate_composite_id=str(raw["candidate_composite_id"]), reconciled=bool(raw["reconciled"]),
            issues=tuple(
                CompositeReconciliationIssue.from_json_dict(as_json_dict(i, field_name="issues[]"))
                for i in as_json_list(raw.get("issues") or [], field_name="issues")
            ),
            member_results=tuple(
                LabelReconciliationResult.from_json_dict(as_json_dict(r, field_name="member_results[]"))
                for r in as_json_list(raw.get("member_results") or [], field_name="member_results")
            ),
        )


def reconcile_composite(
    baseline: CompositeLabelBundle, candidate: CompositeLabelBundle, *, baseline_manifests: tuple[LabelManifest, ...],
    candidate_manifests: tuple[LabelManifest, ...],
) -> CompositeReconciliationResult:
    if baseline.dataset_id != candidate.dataset_id:
        raise LabelReconciliationError(
            f"Cannot reconcile composites for different dataset_id values: baseline={baseline.dataset_id!r} candidate={candidate.dataset_id!r}",
            context={"baseline_dataset_id": baseline.dataset_id, "candidate_dataset_id": candidate.dataset_id},
        )

    issues: list[CompositeReconciliationIssue] = []
    baseline_ids = {m.specification.label_specification_id for m in baseline.members}
    candidate_ids = {m.specification.label_specification_id for m in candidate.members}
    if baseline_ids != candidate_ids:
        issues.append(CompositeReconciliationIssue(
            kind="member_set_drift",
            message=f"member sets differ: only_in_baseline={sorted(baseline_ids - candidate_ids)} only_in_candidate={sorted(candidate_ids - baseline_ids)}",
        ))
    if baseline.composite_id != candidate.composite_id:
        issues.append(CompositeReconciliationIssue(kind="bundle_drift", message=f"composite_id changed: baseline={baseline.composite_id} candidate={candidate.composite_id}"))

    baseline_manifest_by_spec = {m.label_specification_id: m for m in baseline_manifests}
    candidate_manifest_by_spec = {m.label_specification_id: m for m in candidate_manifests}
    member_results = []
    for spec_id in sorted(baseline_ids & candidate_ids):
        baseline_member = baseline.member(spec_id)
        candidate_member = candidate.member(spec_id)
        baseline_manifest = baseline_manifest_by_spec.get(spec_id)
        candidate_manifest = candidate_manifest_by_spec.get(spec_id)
        if baseline_manifest is None or candidate_manifest is None:
            issues.append(CompositeReconciliationIssue(kind="manifest_drift", message=f"missing manifest for member {spec_id!r}"))
            continue
        result = LabelReconciliation().reconcile(baseline_member, candidate_member, baseline_manifest=baseline_manifest, candidate_manifest=candidate_manifest)
        member_results.append(result)
        if not result.reconciled:
            issues.append(CompositeReconciliationIssue(kind="member_drift", message=f"member {spec_id!r} diverged: {len(result.issues)} issue(s)"))

    return CompositeReconciliationResult(
        schema_version=COMPOSITE_LABEL_BUNDLE_SCHEMA_VERSION, baseline_composite_id=baseline.composite_id, candidate_composite_id=candidate.composite_id,
        reconciled=not issues, issues=tuple(issues), member_results=tuple(member_results),
    )
