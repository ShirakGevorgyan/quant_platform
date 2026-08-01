"""Curated dataset layout (Milestone 10, Phase 4B): one immutable
COMPONENT dataset manifest per macro series, plus one COMBINED manifest
binding the exact component versions that make up one curated universe
backfill. Deliberately separate datasets per series -- unlike
frequencies (`DGS10` daily, `CPIAUCSL` monthly) are never forced into
one physically regular time series; no implicit resampling, forward-
fill, or alignment happens anywhere in this module.

VERSIONING IS A STORE-LEVEL CONCEPT, NOT BAKED INTO THE MANIFEST'S OWN
CONTENT HASH: `ComponentDatasetManifest`/`CombinedUniverseManifest` are
each content-addressed purely from their own observed facts (which
observations are included, coverage, missing/revision counts) --
"version N" is simply "the Nth distinct manifest ever durably recorded
for this series/namespace" (`len(history)`), read from the append-only
store, mirroring how `market_data.manifests.DatasetManifestStore`
already separates a manifest's own identity from its position in
history."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quant_platform.core.exceptions import CombinedManifestError, MarketDataPersistenceError
from quant_platform.core.json import canonical_json_bytes, parse_json_strict
from quant_platform.market_data.collectors.curated.macro_observation import CuratedMacroObservation
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)

__all__ = [
    "COMBINED_UNIVERSE_MANIFEST_KIND",
    "COMPONENT_DATASET_MANIFEST_KIND",
    "CombinedUniverseManifest",
    "CombinedUniverseManifestStore",
    "CompletenessStatus",
    "ComponentDatasetManifest",
    "ComponentDatasetManifestStore",
    "create_combined_universe_manifest",
    "create_component_dataset_manifest",
]

COMPONENT_DATASET_MANIFEST_KIND = "curated_component_dataset_manifest"
COMBINED_UNIVERSE_MANIFEST_KIND = "curated_combined_universe_manifest"


class CompletenessStatus:
    COMPLETE = "complete"
    PARTIAL = "partial"
    _VALID = frozenset({COMPLETE, PARTIAL})


@dataclass(frozen=True, slots=True)
class ComponentDatasetManifest:
    component_manifest_id: str
    series_id: str
    canonical_series_name: str
    coverage_start: str | None
    coverage_end: str | None
    native_frequency: str
    observation_count: int
    missing_count: int
    revision_count: int
    """Number of DISTINCT `observation_date`s with more than one
    recorded vintage in this component -- never collapsed, always
    counted."""
    semantic_digest: str
    creation_time: datetime

    def __post_init__(self) -> None:
        require_non_empty(self.series_id, field_name="ComponentDatasetManifest.series_id")
        require_non_empty(self.canonical_series_name, field_name="ComponentDatasetManifest.canonical_series_name")
        require_tz_aware(self.creation_time, field_name="ComponentDatasetManifest.creation_time")
        if self.observation_count < 0 or self.missing_count < 0 or self.revision_count < 0:
            raise CombinedManifestError("ComponentDatasetManifest counts must all be >= 0")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": COMPONENT_DATASET_MANIFEST_KIND, "component_manifest_id": self.component_manifest_id, "series_id": self.series_id,
            "canonical_series_name": self.canonical_series_name, "coverage_start": self.coverage_start, "coverage_end": self.coverage_end,
            "native_frequency": self.native_frequency, "observation_count": self.observation_count, "missing_count": self.missing_count,
            "revision_count": self.revision_count, "semantic_digest": self.semantic_digest,
            "creation_time": serialize_timestamp(self.creation_time, field_name="creation_time"),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["component_manifest_id"]
        del payload["creation_time"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ComponentDatasetManifest:
        return cls(
            component_manifest_id=str(raw["component_manifest_id"]), series_id=str(raw["series_id"]), canonical_series_name=str(raw["canonical_series_name"]),
            coverage_start=(None if raw.get("coverage_start") is None else str(raw["coverage_start"])),
            coverage_end=(None if raw.get("coverage_end") is None else str(raw["coverage_end"])), native_frequency=str(raw["native_frequency"]),
            observation_count=int(str(raw["observation_count"])), missing_count=int(str(raw["missing_count"])), revision_count=int(str(raw["revision_count"])),
            semantic_digest=str(raw["semantic_digest"]), creation_time=deserialize_timestamp(raw["creation_time"], field_name="creation_time"),
        )


def create_component_dataset_manifest(
    *, series_id: str, canonical_series_name: str, observations: tuple[CuratedMacroObservation, ...], missing_count: int, creation_time: datetime,
) -> ComponentDatasetManifest:
    dates = sorted(o.observation_date for o in observations)
    coverage_start = dates[0] if dates else None
    coverage_end = dates[-1] if dates else None
    native_frequency = observations[0].native_frequency if observations else ""
    date_counts: dict[str, int] = {}
    for o in observations:
        date_counts[o.observation_date] = date_counts.get(o.observation_date, 0) + 1
    revision_count = sum(1 for count in date_counts.values() if count > 1)
    semantic_digest = compute_content_id("curated_component_semantic_digest", {"observation_ids": sorted(o.observation_id for o in observations)})

    provisional = ComponentDatasetManifest(
        component_manifest_id="0" * 64, series_id=series_id, canonical_series_name=canonical_series_name, coverage_start=coverage_start,
        coverage_end=coverage_end, native_frequency=native_frequency, observation_count=len(observations), missing_count=missing_count,
        revision_count=revision_count, semantic_digest=semantic_digest, creation_time=creation_time,
    )
    component_manifest_id = compute_content_id(COMPONENT_DATASET_MANIFEST_KIND, provisional.to_identity_payload())
    return ComponentDatasetManifest(
        component_manifest_id=component_manifest_id, series_id=series_id, canonical_series_name=canonical_series_name, coverage_start=coverage_start,
        coverage_end=coverage_end, native_frequency=native_frequency, observation_count=len(observations), missing_count=missing_count,
        revision_count=revision_count, semantic_digest=semantic_digest, creation_time=creation_time,
    )


class ComponentDatasetManifestStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _dir(self, provider: str, series_id: str) -> Path:
        return self._root / "collectors" / "curated" / "component_manifests" / provider / series_id

    def _path(self, provider: str, series_id: str) -> Path:
        return self._dir(provider, series_id) / "manifests.jsonl"

    def read_history(self, provider: str, series_id: str) -> list[ComponentDatasetManifest]:
        path = self._path(provider, series_id)
        if not path.is_file():
            return []
        manifests: list[ComponentDatasetManifest] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted component manifest history line for {provider}/{series_id}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted component manifest history line for {provider}/{series_id}: expected a JSON object")
            manifests.append(ComponentDatasetManifest.from_json_dict(raw))
        return manifests

    def read_current(self, provider: str, series_id: str) -> ComponentDatasetManifest | None:
        history = self.read_history(provider, series_id)
        return history[-1] if history else None

    def current_version(self, provider: str, series_id: str) -> int:
        return len(self.read_history(provider, series_id))

    def append(self, provider: str, manifest: ComponentDatasetManifest) -> tuple[ComponentDatasetManifest, int]:
        """Idempotent no-op (returns the existing version number) if the
        CURRENT manifest already has this exact `component_manifest_id`
        -- an exact no-op update (see `update_plan.py`) must never mint
        a new version."""
        current = self.read_current(provider, manifest.series_id)
        if current is not None and current.component_manifest_id == manifest.component_manifest_id:
            return current, self.current_version(provider, manifest.series_id)
        self._dir(provider, manifest.series_id).mkdir(parents=True, exist_ok=True)
        path = self._path(provider, manifest.series_id)
        with path.open("ab") as handle:
            handle.write(canonical_json_bytes(manifest.to_json_dict()))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return manifest, self.current_version(provider, manifest.series_id)


@dataclass(frozen=True, slots=True)
class CombinedUniverseManifest:
    combined_manifest_id: str
    curated_registry_id: str
    backfill_plan_id: str
    target_dataset_namespace: str
    component_manifest_ids: dict[str, str]
    coverage_by_series: dict[str, tuple[str | None, str | None]]
    frequencies_by_series: dict[str, str]
    availability_policy_ids_by_series: dict[str, str]
    revision_policy_id: str
    missing_counts_by_series: dict[str, int]
    semantic_digest: str
    completeness_status: str
    creation_time: datetime

    def __post_init__(self) -> None:
        require_non_empty(self.curated_registry_id, field_name="CombinedUniverseManifest.curated_registry_id")
        require_non_empty(self.backfill_plan_id, field_name="CombinedUniverseManifest.backfill_plan_id")
        require_non_empty(self.target_dataset_namespace, field_name="CombinedUniverseManifest.target_dataset_namespace")
        require_tz_aware(self.creation_time, field_name="CombinedUniverseManifest.creation_time")
        if self.completeness_status not in CompletenessStatus._VALID:
            raise CombinedManifestError(f"CombinedUniverseManifest.completeness_status must be one of {sorted(CompletenessStatus._VALID)!r}, got {self.completeness_status!r}")
        if not self.component_manifest_ids:
            raise CombinedManifestError("CombinedUniverseManifest.component_manifest_ids must not be empty")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": COMBINED_UNIVERSE_MANIFEST_KIND, "combined_manifest_id": self.combined_manifest_id, "curated_registry_id": self.curated_registry_id,
            "backfill_plan_id": self.backfill_plan_id, "target_dataset_namespace": self.target_dataset_namespace,
            "component_manifest_ids": dict(sorted(self.component_manifest_ids.items())),
            "coverage_by_series": {k: list(v) for k, v in sorted(self.coverage_by_series.items())},
            "frequencies_by_series": dict(sorted(self.frequencies_by_series.items())),
            "availability_policy_ids_by_series": dict(sorted(self.availability_policy_ids_by_series.items())),
            "revision_policy_id": self.revision_policy_id, "missing_counts_by_series": dict(sorted(self.missing_counts_by_series.items())),
            "semantic_digest": self.semantic_digest, "completeness_status": self.completeness_status,
            "creation_time": serialize_timestamp(self.creation_time, field_name="creation_time"),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["combined_manifest_id"]
        del payload["creation_time"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CombinedUniverseManifest:
        from quant_platform.ml.persistence import as_json_dict

        coverage_raw = as_json_dict(raw["coverage_by_series"], field_name="coverage_by_series")
        coverage: dict[str, tuple[str | None, str | None]] = {}
        for series_id, pair in coverage_raw.items():
            assert isinstance(pair, list) and len(pair) == 2
            coverage[series_id] = (pair[0], pair[1])
        return cls(
            combined_manifest_id=str(raw["combined_manifest_id"]), curated_registry_id=str(raw["curated_registry_id"]),
            backfill_plan_id=str(raw["backfill_plan_id"]), target_dataset_namespace=str(raw["target_dataset_namespace"]),
            component_manifest_ids={str(k): str(v) for k, v in as_json_dict(raw["component_manifest_ids"], field_name="component_manifest_ids").items()},
            coverage_by_series=coverage,
            frequencies_by_series={str(k): str(v) for k, v in as_json_dict(raw["frequencies_by_series"], field_name="frequencies_by_series").items()},
            availability_policy_ids_by_series={str(k): str(v) for k, v in as_json_dict(raw["availability_policy_ids_by_series"], field_name="availability_policy_ids_by_series").items()},
            revision_policy_id=str(raw["revision_policy_id"]),
            missing_counts_by_series={str(k): int(str(v)) for k, v in as_json_dict(raw["missing_counts_by_series"], field_name="missing_counts_by_series").items()},
            semantic_digest=str(raw["semantic_digest"]), completeness_status=str(raw["completeness_status"]),
            creation_time=deserialize_timestamp(raw["creation_time"], field_name="creation_time"),
        )


def create_combined_universe_manifest(
    *, curated_registry_id: str, backfill_plan_id: str, target_dataset_namespace: str, component_manifests: dict[str, ComponentDatasetManifest],
    availability_policy_ids_by_series: dict[str, str], revision_policy_id: str, completeness_status: str, creation_time: datetime,
) -> CombinedUniverseManifest:
    component_manifest_ids = {series_id: m.component_manifest_id for series_id, m in component_manifests.items()}
    coverage_by_series = {series_id: (m.coverage_start, m.coverage_end) for series_id, m in component_manifests.items()}
    frequencies_by_series = {series_id: m.native_frequency for series_id, m in component_manifests.items()}
    missing_counts_by_series = {series_id: m.missing_count for series_id, m in component_manifests.items()}
    semantic_digest = compute_content_id("curated_combined_semantic_digest", {"component_manifest_ids": sorted(component_manifest_ids.values())})

    provisional = CombinedUniverseManifest(
        combined_manifest_id="0" * 64, curated_registry_id=curated_registry_id, backfill_plan_id=backfill_plan_id,
        target_dataset_namespace=target_dataset_namespace, component_manifest_ids=component_manifest_ids, coverage_by_series=coverage_by_series,
        frequencies_by_series=frequencies_by_series, availability_policy_ids_by_series=availability_policy_ids_by_series,
        revision_policy_id=revision_policy_id, missing_counts_by_series=missing_counts_by_series, semantic_digest=semantic_digest,
        completeness_status=completeness_status, creation_time=creation_time,
    )
    combined_manifest_id = compute_content_id(COMBINED_UNIVERSE_MANIFEST_KIND, provisional.to_identity_payload())
    return CombinedUniverseManifest(
        combined_manifest_id=combined_manifest_id, curated_registry_id=curated_registry_id, backfill_plan_id=backfill_plan_id,
        target_dataset_namespace=target_dataset_namespace, component_manifest_ids=component_manifest_ids, coverage_by_series=coverage_by_series,
        frequencies_by_series=frequencies_by_series, availability_policy_ids_by_series=availability_policy_ids_by_series,
        revision_policy_id=revision_policy_id, missing_counts_by_series=missing_counts_by_series, semantic_digest=semantic_digest,
        completeness_status=completeness_status, creation_time=creation_time,
    )


class CombinedUniverseManifestStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _dir(self, target_dataset_namespace: str) -> Path:
        return self._root / "collectors" / "curated" / "combined_manifests" / target_dataset_namespace

    def _path(self, target_dataset_namespace: str) -> Path:
        return self._dir(target_dataset_namespace) / "manifests.jsonl"

    def read_history(self, target_dataset_namespace: str) -> list[CombinedUniverseManifest]:
        path = self._path(target_dataset_namespace)
        if not path.is_file():
            return []
        manifests: list[CombinedUniverseManifest] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted combined manifest history line for namespace {target_dataset_namespace!r}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted combined manifest history line for namespace {target_dataset_namespace!r}: expected a JSON object")
            manifests.append(CombinedUniverseManifest.from_json_dict(raw))
        return manifests

    def read_current(self, target_dataset_namespace: str) -> CombinedUniverseManifest | None:
        history = self.read_history(target_dataset_namespace)
        return history[-1] if history else None

    def current_version(self, target_dataset_namespace: str) -> int:
        return len(self.read_history(target_dataset_namespace))

    def append(self, manifest: CombinedUniverseManifest) -> tuple[CombinedUniverseManifest, int]:
        current = self.read_current(manifest.target_dataset_namespace)
        if current is not None and current.combined_manifest_id == manifest.combined_manifest_id:
            return current, self.current_version(manifest.target_dataset_namespace)
        self._dir(manifest.target_dataset_namespace).mkdir(parents=True, exist_ok=True)
        path = self._path(manifest.target_dataset_namespace)
        with path.open("ab") as handle:
            handle.write(canonical_json_bytes(manifest.to_json_dict()))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return manifest, self.current_version(manifest.target_dataset_namespace)
