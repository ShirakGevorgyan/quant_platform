"""Curated cross-asset dataset layout (Milestone 10, Phase 4C, spec
Section 19): one immutable COMPONENT dataset manifest per
`ProviderSymbolMapping` (the natural component key -- a mapping already
binds provider + provider_symbol + canonical_driver_id + instrument_form
+ currency + adjustment_policy_kind + continuation_policy_id + mapping_
version, exactly the granularity the spec requires components to never
be merged across), plus one COMBINED manifest binding the exact
component versions that make up one curated cross-asset universe
backfill. Deliberately separate datasets per mapping -- unlike calendars
and frequencies are never forced into physical alignment; no implicit
resampling, forward-fill, or cross-mapping merge happens anywhere in
this module.

VERSIONING IS A STORE-LEVEL CONCEPT, NOT BAKED INTO THE MANIFEST'S OWN
CONTENT HASH -- mirrors `curated.datasets` exactly: "version N" is
"the Nth distinct manifest ever durably recorded for this mapping/
namespace" (`len(history)`), read from the append-only store."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quant_platform.core.exceptions import MarketCombinedManifestError, MarketDataPersistenceError
from quant_platform.core.json import canonical_json_bytes, parse_json_strict
from quant_platform.market_data.collectors.cross_asset.market_record import MarketDriverBar
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)

__all__ = [
    "COMBINED_CROSS_ASSET_MANIFEST_KIND",
    "COMPONENT_MARKET_DATASET_MANIFEST_KIND",
    "CombinedCrossAssetManifest",
    "CombinedCrossAssetManifestStore",
    "CompletenessStatus",
    "ComponentMarketDatasetManifest",
    "ComponentMarketDatasetManifestStore",
    "create_combined_cross_asset_manifest",
    "create_component_market_dataset_manifest",
]

COMPONENT_MARKET_DATASET_MANIFEST_KIND = "cross_asset_component_dataset_manifest"
COMBINED_CROSS_ASSET_MANIFEST_KIND = "cross_asset_combined_universe_manifest"


class CompletenessStatus:
    COMPLETE = "complete"
    PARTIAL = "partial"
    _VALID = frozenset({COMPLETE, PARTIAL})


@dataclass(frozen=True, slots=True)
class ComponentMarketDatasetManifest:
    component_manifest_id: str
    mapping_id: str
    canonical_driver_id: str
    provider: str
    provider_symbol: str
    instrument_form: str
    timeframe: str
    adjustment_policy_id: str
    continuation_policy_id: str | None
    session_policy_id: str
    availability_policy_id: str
    coverage_start: str | None
    coverage_end: str | None
    bar_count: int
    missing_business_day_count: int
    conflicting_coordinate_count: int
    semantic_digest: str
    creation_time: datetime

    def __post_init__(self) -> None:
        require_non_empty(self.mapping_id, field_name="ComponentMarketDatasetManifest.mapping_id")
        require_non_empty(self.canonical_driver_id, field_name="ComponentMarketDatasetManifest.canonical_driver_id")
        require_non_empty(self.provider, field_name="ComponentMarketDatasetManifest.provider")
        require_non_empty(self.provider_symbol, field_name="ComponentMarketDatasetManifest.provider_symbol")
        require_tz_aware(self.creation_time, field_name="ComponentMarketDatasetManifest.creation_time")
        if self.bar_count < 0 or self.missing_business_day_count < 0 or self.conflicting_coordinate_count < 0:
            raise MarketCombinedManifestError("ComponentMarketDatasetManifest counts must all be >= 0")
        if self.conflicting_coordinate_count > 0:
            raise MarketCombinedManifestError(
                f"ComponentMarketDatasetManifest for mapping_id {self.mapping_id!r} has {self.conflicting_coordinate_count} conflicting "
                "coordinate(s) -- a component with unresolved conflicting duplicate bars must never be committed"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": COMPONENT_MARKET_DATASET_MANIFEST_KIND, "component_manifest_id": self.component_manifest_id, "mapping_id": self.mapping_id,
            "canonical_driver_id": self.canonical_driver_id, "provider": self.provider, "provider_symbol": self.provider_symbol,
            "instrument_form": self.instrument_form, "timeframe": self.timeframe, "adjustment_policy_id": self.adjustment_policy_id,
            "continuation_policy_id": self.continuation_policy_id, "session_policy_id": self.session_policy_id,
            "availability_policy_id": self.availability_policy_id, "coverage_start": self.coverage_start, "coverage_end": self.coverage_end,
            "bar_count": self.bar_count, "missing_business_day_count": self.missing_business_day_count,
            "conflicting_coordinate_count": self.conflicting_coordinate_count, "semantic_digest": self.semantic_digest,
            "creation_time": serialize_timestamp(self.creation_time, field_name="creation_time"),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["component_manifest_id"]
        del payload["creation_time"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ComponentMarketDatasetManifest:
        return cls(
            component_manifest_id=str(raw["component_manifest_id"]), mapping_id=str(raw["mapping_id"]),
            canonical_driver_id=str(raw["canonical_driver_id"]), provider=str(raw["provider"]), provider_symbol=str(raw["provider_symbol"]),
            instrument_form=str(raw["instrument_form"]), timeframe=str(raw["timeframe"]), adjustment_policy_id=str(raw["adjustment_policy_id"]),
            continuation_policy_id=(None if raw.get("continuation_policy_id") is None else str(raw["continuation_policy_id"])),
            session_policy_id=str(raw["session_policy_id"]), availability_policy_id=str(raw["availability_policy_id"]),
            coverage_start=(None if raw.get("coverage_start") is None else str(raw["coverage_start"])),
            coverage_end=(None if raw.get("coverage_end") is None else str(raw["coverage_end"])), bar_count=int(str(raw["bar_count"])),
            missing_business_day_count=int(str(raw["missing_business_day_count"])),
            conflicting_coordinate_count=int(str(raw["conflicting_coordinate_count"])), semantic_digest=str(raw["semantic_digest"]),
            creation_time=deserialize_timestamp(raw["creation_time"], field_name="creation_time"),
        )


def create_component_market_dataset_manifest(
    *, mapping_id: str, canonical_driver_id: str, provider: str, provider_symbol: str, instrument_form: str, timeframe: str,
    adjustment_policy_id: str, session_policy_id: str, availability_policy_id: str, bars: tuple[MarketDriverBar, ...],
    missing_business_day_count: int, conflicting_coordinate_count: int, creation_time: datetime, continuation_policy_id: str | None = None,
) -> ComponentMarketDatasetManifest:
    open_times = sorted(b.open_time for b in bars)
    coverage_start = open_times[0].isoformat() if open_times else None
    coverage_end = open_times[-1].isoformat() if open_times else None
    semantic_digest = compute_content_id("cross_asset_component_semantic_digest", {"bar_ids": sorted(b.bar_id for b in bars)})

    provisional = ComponentMarketDatasetManifest(
        component_manifest_id="0" * 64, mapping_id=mapping_id, canonical_driver_id=canonical_driver_id, provider=provider,
        provider_symbol=provider_symbol, instrument_form=instrument_form, timeframe=timeframe, adjustment_policy_id=adjustment_policy_id,
        continuation_policy_id=continuation_policy_id, session_policy_id=session_policy_id, availability_policy_id=availability_policy_id,
        coverage_start=coverage_start, coverage_end=coverage_end, bar_count=len(bars), missing_business_day_count=missing_business_day_count,
        conflicting_coordinate_count=conflicting_coordinate_count, semantic_digest=semantic_digest, creation_time=creation_time,
    )
    component_manifest_id = compute_content_id(COMPONENT_MARKET_DATASET_MANIFEST_KIND, provisional.to_identity_payload())
    return ComponentMarketDatasetManifest(
        component_manifest_id=component_manifest_id, mapping_id=mapping_id, canonical_driver_id=canonical_driver_id, provider=provider,
        provider_symbol=provider_symbol, instrument_form=instrument_form, timeframe=timeframe, adjustment_policy_id=adjustment_policy_id,
        continuation_policy_id=continuation_policy_id, session_policy_id=session_policy_id, availability_policy_id=availability_policy_id,
        coverage_start=coverage_start, coverage_end=coverage_end, bar_count=len(bars), missing_business_day_count=missing_business_day_count,
        conflicting_coordinate_count=conflicting_coordinate_count, semantic_digest=semantic_digest, creation_time=creation_time,
    )


class ComponentMarketDatasetManifestStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _dir(self, mapping_id: str) -> Path:
        return self._root / "collectors" / "cross_asset" / "component_manifests" / mapping_id

    def _path(self, mapping_id: str) -> Path:
        return self._dir(mapping_id) / "manifests.jsonl"

    def read_history(self, mapping_id: str) -> list[ComponentMarketDatasetManifest]:
        path = self._path(mapping_id)
        if not path.is_file():
            return []
        manifests: list[ComponentMarketDatasetManifest] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted component manifest history line for mapping {mapping_id}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted component manifest history line for mapping {mapping_id}: expected a JSON object")
            manifests.append(ComponentMarketDatasetManifest.from_json_dict(raw))
        return manifests

    def read_current(self, mapping_id: str) -> ComponentMarketDatasetManifest | None:
        history = self.read_history(mapping_id)
        return history[-1] if history else None

    def current_version(self, mapping_id: str) -> int:
        return len(self.read_history(mapping_id))

    def append(self, manifest: ComponentMarketDatasetManifest) -> tuple[ComponentMarketDatasetManifest, int]:
        """Idempotent no-op (returns the existing version number) if the
        CURRENT manifest already has this exact `component_manifest_id`
        -- an exact no-op update (see `update_plan.py`) must never mint
        a new version."""
        current = self.read_current(manifest.mapping_id)
        if current is not None and current.component_manifest_id == manifest.component_manifest_id:
            return current, self.current_version(manifest.mapping_id)
        self._dir(manifest.mapping_id).mkdir(parents=True, exist_ok=True)
        path = self._path(manifest.mapping_id)
        with path.open("ab") as handle:
            handle.write(canonical_json_bytes(manifest.to_json_dict()))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return manifest, self.current_version(manifest.mapping_id)


@dataclass(frozen=True, slots=True)
class CombinedCrossAssetManifest:
    combined_manifest_id: str
    curated_registry_id: str
    backfill_plan_id: str
    target_dataset_namespace: str
    component_manifest_ids: dict[str, str]
    """Keyed by `mapping_id`."""
    driver_id_by_mapping: dict[str, str]
    coverage_by_mapping: dict[str, tuple[str | None, str | None]]
    missing_counts_by_mapping: dict[str, int]
    required_driver_ids: tuple[str, ...]
    missing_required_driver_ids: tuple[str, ...]
    """Required driver ids with NO successful component in this
    manifest -- non-empty forces `completeness_status=PARTIAL`
    regardless of any other signal (spec Section 18's "no completed
    universe with silently missing required driver")."""
    semantic_digest: str
    completeness_status: str
    creation_time: datetime

    def __post_init__(self) -> None:
        require_non_empty(self.curated_registry_id, field_name="CombinedCrossAssetManifest.curated_registry_id")
        require_non_empty(self.backfill_plan_id, field_name="CombinedCrossAssetManifest.backfill_plan_id")
        require_non_empty(self.target_dataset_namespace, field_name="CombinedCrossAssetManifest.target_dataset_namespace")
        require_tz_aware(self.creation_time, field_name="CombinedCrossAssetManifest.creation_time")
        if self.completeness_status not in CompletenessStatus._VALID:
            raise MarketCombinedManifestError(f"CombinedCrossAssetManifest.completeness_status must be one of {sorted(CompletenessStatus._VALID)!r}, got {self.completeness_status!r}")
        if not self.component_manifest_ids:
            raise MarketCombinedManifestError("CombinedCrossAssetManifest.component_manifest_ids must not be empty")
        if self.missing_required_driver_ids and self.completeness_status != CompletenessStatus.PARTIAL:
            raise MarketCombinedManifestError(
                f"CombinedCrossAssetManifest has missing_required_driver_ids={self.missing_required_driver_ids!r} but "
                f"completeness_status={self.completeness_status!r} -- a universe missing a required driver can never be COMPLETE"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": COMBINED_CROSS_ASSET_MANIFEST_KIND, "combined_manifest_id": self.combined_manifest_id,
            "curated_registry_id": self.curated_registry_id, "backfill_plan_id": self.backfill_plan_id,
            "target_dataset_namespace": self.target_dataset_namespace,
            "component_manifest_ids": dict(sorted(self.component_manifest_ids.items())),
            "driver_id_by_mapping": dict(sorted(self.driver_id_by_mapping.items())),
            "coverage_by_mapping": {k: list(v) for k, v in sorted(self.coverage_by_mapping.items())},
            "missing_counts_by_mapping": dict(sorted(self.missing_counts_by_mapping.items())),
            "required_driver_ids": sorted(self.required_driver_ids), "missing_required_driver_ids": sorted(self.missing_required_driver_ids),
            "semantic_digest": self.semantic_digest, "completeness_status": self.completeness_status,
            "creation_time": serialize_timestamp(self.creation_time, field_name="creation_time"),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["combined_manifest_id"]
        del payload["creation_time"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CombinedCrossAssetManifest:
        from quant_platform.ml.persistence import as_json_dict, as_json_list

        coverage_raw = as_json_dict(raw["coverage_by_mapping"], field_name="coverage_by_mapping")
        coverage: dict[str, tuple[str | None, str | None]] = {}
        for mapping_id, pair in coverage_raw.items():
            assert isinstance(pair, list) and len(pair) == 2
            coverage[mapping_id] = (pair[0], pair[1])
        return cls(
            combined_manifest_id=str(raw["combined_manifest_id"]), curated_registry_id=str(raw["curated_registry_id"]),
            backfill_plan_id=str(raw["backfill_plan_id"]), target_dataset_namespace=str(raw["target_dataset_namespace"]),
            component_manifest_ids={str(k): str(v) for k, v in as_json_dict(raw["component_manifest_ids"], field_name="component_manifest_ids").items()},
            driver_id_by_mapping={str(k): str(v) for k, v in as_json_dict(raw["driver_id_by_mapping"], field_name="driver_id_by_mapping").items()},
            coverage_by_mapping=coverage,
            missing_counts_by_mapping={str(k): int(str(v)) for k, v in as_json_dict(raw["missing_counts_by_mapping"], field_name="missing_counts_by_mapping").items()},
            required_driver_ids=tuple(str(s) for s in as_json_list(raw["required_driver_ids"], field_name="required_driver_ids")),
            missing_required_driver_ids=tuple(str(s) for s in as_json_list(raw["missing_required_driver_ids"], field_name="missing_required_driver_ids")),
            semantic_digest=str(raw["semantic_digest"]), completeness_status=str(raw["completeness_status"]),
            creation_time=deserialize_timestamp(raw["creation_time"], field_name="creation_time"),
        )


def create_combined_cross_asset_manifest(
    *, curated_registry_id: str, backfill_plan_id: str, target_dataset_namespace: str,
    component_manifests: dict[str, ComponentMarketDatasetManifest], required_driver_ids: tuple[str, ...], creation_time: datetime,
) -> CombinedCrossAssetManifest:
    component_manifest_ids = {mapping_id: m.component_manifest_id for mapping_id, m in component_manifests.items()}
    driver_id_by_mapping = {mapping_id: m.canonical_driver_id for mapping_id, m in component_manifests.items()}
    coverage_by_mapping = {mapping_id: (m.coverage_start, m.coverage_end) for mapping_id, m in component_manifests.items()}
    missing_counts_by_mapping = {mapping_id: m.missing_business_day_count for mapping_id, m in component_manifests.items()}
    satisfied_driver_ids = set(driver_id_by_mapping.values())
    missing_required = tuple(sorted(set(required_driver_ids) - satisfied_driver_ids))
    completeness_status = CompletenessStatus.PARTIAL if missing_required else CompletenessStatus.COMPLETE
    semantic_digest = compute_content_id("cross_asset_combined_semantic_digest", {"component_manifest_ids": sorted(component_manifest_ids.values())})

    provisional = CombinedCrossAssetManifest(
        combined_manifest_id="0" * 64, curated_registry_id=curated_registry_id, backfill_plan_id=backfill_plan_id,
        target_dataset_namespace=target_dataset_namespace, component_manifest_ids=component_manifest_ids, driver_id_by_mapping=driver_id_by_mapping,
        coverage_by_mapping=coverage_by_mapping, missing_counts_by_mapping=missing_counts_by_mapping, required_driver_ids=tuple(sorted(set(required_driver_ids))),
        missing_required_driver_ids=missing_required, semantic_digest=semantic_digest, completeness_status=completeness_status, creation_time=creation_time,
    )
    combined_manifest_id = compute_content_id(COMBINED_CROSS_ASSET_MANIFEST_KIND, provisional.to_identity_payload())
    return CombinedCrossAssetManifest(
        combined_manifest_id=combined_manifest_id, curated_registry_id=curated_registry_id, backfill_plan_id=backfill_plan_id,
        target_dataset_namespace=target_dataset_namespace, component_manifest_ids=component_manifest_ids, driver_id_by_mapping=driver_id_by_mapping,
        coverage_by_mapping=coverage_by_mapping, missing_counts_by_mapping=missing_counts_by_mapping, required_driver_ids=tuple(sorted(set(required_driver_ids))),
        missing_required_driver_ids=missing_required, semantic_digest=semantic_digest, completeness_status=completeness_status, creation_time=creation_time,
    )


class CombinedCrossAssetManifestStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _dir(self, target_dataset_namespace: str) -> Path:
        return self._root / "collectors" / "cross_asset" / "combined_manifests" / target_dataset_namespace

    def _path(self, target_dataset_namespace: str) -> Path:
        return self._dir(target_dataset_namespace) / "manifests.jsonl"

    def read_history(self, target_dataset_namespace: str) -> list[CombinedCrossAssetManifest]:
        path = self._path(target_dataset_namespace)
        if not path.is_file():
            return []
        manifests: list[CombinedCrossAssetManifest] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted combined manifest history line for namespace {target_dataset_namespace!r}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted combined manifest history line for namespace {target_dataset_namespace!r}: expected a JSON object")
            manifests.append(CombinedCrossAssetManifest.from_json_dict(raw))
        return manifests

    def read_current(self, target_dataset_namespace: str) -> CombinedCrossAssetManifest | None:
        history = self.read_history(target_dataset_namespace)
        return history[-1] if history else None

    def current_version(self, target_dataset_namespace: str) -> int:
        return len(self.read_history(target_dataset_namespace))

    def append(self, manifest: CombinedCrossAssetManifest) -> tuple[CombinedCrossAssetManifest, int]:
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
