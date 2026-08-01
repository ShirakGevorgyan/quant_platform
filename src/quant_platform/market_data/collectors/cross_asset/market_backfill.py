"""Immutable, content-addressed multi-driver curated cross-asset backfill
spec (Milestone 10, Phase 4C, spec Section 17). Same semantic plan
always produces the same `backfill_plan_id` -- constructed ONLY via
`create_market_backfill_spec`, which validates every selected mapping
against a `CuratedMarketDriverRegistry` AND a `SymbolMappingSet` (unknown
driver, disabled driver, unknown mapping, or a mapping belonging to a
driver not in `selected_driver_ids` are all rejected) before a spec can
even exist.

`selected_mapping_ids` is deliberately separate from `selected_driver_ids`
(mirrors the spec's own "selected driver ids; provider selection/mapping
ids" as two distinct fields): a driver id names the ECONOMIC CONCEPT
being backfilled; a mapping id names the EXACT provider surface used to
supply it. More than one mapping per driver is permitted (the
cross-provider conflict model, Section 20, depends on this: fetching the
same concept from two providers simultaneously is a deliberate choice a
caller makes explicitly, never an automatic behind-the-scenes
arbitration)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from quant_platform.core.exceptions import MarketBackfillSpecError
from quant_platform.market_data.collectors.cross_asset.registry import CuratedMarketDriverRegistry
from quant_platform.market_data.collectors.cross_asset.symbol_mapping import SymbolMappingSet
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)

__all__ = [
    "MARKET_BACKFILL_SPEC_KIND",
    "MarketBackfillSpec",
    "MarketCachePolicy",
    "create_market_backfill_spec",
]

MARKET_BACKFILL_SPEC_KIND = "cross_asset_market_backfill_spec"


class MarketCachePolicy(Enum):
    PREFER_CACHE = "prefer_cache"
    """Reuse an already-cached response for the same semantic request
    when one exists, falling back to fresh only when nothing is cached
    yet -- see `market_orchestration.py`."""

    FORCE_FRESH = "force_fresh"
    """Always fetch fresh, even if a cached response already exists -- a
    fresh-network fetch MAY produce a new response version (spec Section
    18's own "fresh-network policy can create new response version")."""


@dataclass(frozen=True, slots=True)
class MarketBackfillSpec:
    backfill_plan_id: str
    curated_registry_id: str
    selected_driver_ids: tuple[str, ...]
    """Always sorted -- identity-independent-of-declaration-order AND
    stable driver processing order (spec Section 18) are the SAME sort."""
    selected_mapping_ids: tuple[str, ...]
    """Always sorted. Every id must belong to `SymbolMappingSet` AND its
    `canonical_driver_id` must be a member of `selected_driver_ids` --
    validated in `create_market_backfill_spec`."""
    start_time: datetime
    end_time: datetime
    requested_granularity: str
    cache_policy: MarketCachePolicy
    target_dataset_namespace: str
    fail_fast: bool
    max_driver_count: int
    max_records_per_mapping: int
    max_total_raw_bytes: int

    def __post_init__(self) -> None:
        require_non_empty(self.curated_registry_id, field_name="MarketBackfillSpec.curated_registry_id")
        require_non_empty(self.requested_granularity, field_name="MarketBackfillSpec.requested_granularity")
        require_non_empty(self.target_dataset_namespace, field_name="MarketBackfillSpec.target_dataset_namespace")
        if not self.selected_driver_ids:
            raise MarketBackfillSpecError("MarketBackfillSpec.selected_driver_ids must not be empty")
        if len(set(self.selected_driver_ids)) != len(self.selected_driver_ids):
            raise MarketBackfillSpecError(f"MarketBackfillSpec.selected_driver_ids contains duplicates: {self.selected_driver_ids!r}")
        if tuple(sorted(self.selected_driver_ids)) != self.selected_driver_ids:
            raise MarketBackfillSpecError("MarketBackfillSpec.selected_driver_ids must be constructed via create_market_backfill_spec (which sorts them)")
        if not self.selected_mapping_ids:
            raise MarketBackfillSpecError("MarketBackfillSpec.selected_mapping_ids must not be empty")
        if len(set(self.selected_mapping_ids)) != len(self.selected_mapping_ids):
            raise MarketBackfillSpecError(f"MarketBackfillSpec.selected_mapping_ids contains duplicates: {self.selected_mapping_ids!r}")
        if tuple(sorted(self.selected_mapping_ids)) != self.selected_mapping_ids:
            raise MarketBackfillSpecError("MarketBackfillSpec.selected_mapping_ids must be constructed via create_market_backfill_spec (which sorts them)")
        require_tz_aware(self.start_time, field_name="MarketBackfillSpec.start_time")
        require_tz_aware(self.end_time, field_name="MarketBackfillSpec.end_time")
        if self.end_time < self.start_time:
            raise MarketBackfillSpecError(f"MarketBackfillSpec.end_time ({self.end_time}) must be >= start_time ({self.start_time})")
        if self.max_driver_count < 1:
            raise MarketBackfillSpecError(f"MarketBackfillSpec.max_driver_count must be >= 1, got {self.max_driver_count}")
        if len(self.selected_driver_ids) > self.max_driver_count:
            raise MarketBackfillSpecError(f"MarketBackfillSpec selects {len(self.selected_driver_ids)} drivers, exceeding max_driver_count={self.max_driver_count}")
        if self.max_records_per_mapping < 1:
            raise MarketBackfillSpecError(f"MarketBackfillSpec.max_records_per_mapping must be >= 1, got {self.max_records_per_mapping}")
        if self.max_total_raw_bytes < 1:
            raise MarketBackfillSpecError(f"MarketBackfillSpec.max_total_raw_bytes must be >= 1, got {self.max_total_raw_bytes}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": MARKET_BACKFILL_SPEC_KIND, "backfill_plan_id": self.backfill_plan_id, "curated_registry_id": self.curated_registry_id,
            "selected_driver_ids": list(self.selected_driver_ids), "selected_mapping_ids": list(self.selected_mapping_ids),
            "start_time": serialize_timestamp(self.start_time, field_name="start_time"),
            "end_time": serialize_timestamp(self.end_time, field_name="end_time"), "requested_granularity": self.requested_granularity,
            "cache_policy": self.cache_policy.value, "target_dataset_namespace": self.target_dataset_namespace, "fail_fast": self.fail_fast,
            "max_driver_count": self.max_driver_count, "max_records_per_mapping": self.max_records_per_mapping,
            "max_total_raw_bytes": self.max_total_raw_bytes,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["backfill_plan_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> MarketBackfillSpec:
        from quant_platform.ml.persistence import as_json_list

        return cls(
            backfill_plan_id=str(raw["backfill_plan_id"]), curated_registry_id=str(raw["curated_registry_id"]),
            selected_driver_ids=tuple(str(s) for s in as_json_list(raw["selected_driver_ids"], field_name="selected_driver_ids")),
            selected_mapping_ids=tuple(str(s) for s in as_json_list(raw["selected_mapping_ids"], field_name="selected_mapping_ids")),
            start_time=deserialize_timestamp(raw["start_time"], field_name="start_time"),
            end_time=deserialize_timestamp(raw["end_time"], field_name="end_time"), requested_granularity=str(raw["requested_granularity"]),
            cache_policy=MarketCachePolicy(raw["cache_policy"]), target_dataset_namespace=str(raw["target_dataset_namespace"]),
            fail_fast=bool(raw["fail_fast"]), max_driver_count=int(str(raw["max_driver_count"])),
            max_records_per_mapping=int(str(raw["max_records_per_mapping"])), max_total_raw_bytes=int(str(raw["max_total_raw_bytes"])),
        )


def create_market_backfill_spec(
    *, registry: CuratedMarketDriverRegistry, mapping_set: SymbolMappingSet, selected_driver_ids: tuple[str, ...],
    selected_mapping_ids: tuple[str, ...], start_time: datetime, end_time: datetime, requested_granularity: str,
    target_dataset_namespace: str, cache_policy: MarketCachePolicy = MarketCachePolicy.PREFER_CACHE, fail_fast: bool = True,
    max_driver_count: int = 32, max_records_per_mapping: int = 200_000, max_total_raw_bytes: int = 500_000_000,
) -> MarketBackfillSpec:
    if not selected_driver_ids:
        raise MarketBackfillSpecError("selected_driver_ids must not be empty")
    if not selected_mapping_ids:
        raise MarketBackfillSpecError("selected_mapping_ids must not be empty")

    ordered_driver_ids = tuple(sorted(set(selected_driver_ids)))
    for driver_id in ordered_driver_ids:
        spec = registry.get(driver_id)
        if spec is None:
            raise MarketBackfillSpecError(f"unknown canonical_driver_id {driver_id!r} is not present in registry {registry.registry_id!r}")
        if not spec.enabled:
            raise MarketBackfillSpecError(f"canonical_driver_id {driver_id!r} is disabled in the registry -- enable it before selecting it for a backfill")

    ordered_mapping_ids = tuple(sorted(set(selected_mapping_ids)))
    for mapping_id in ordered_mapping_ids:
        mapping = mapping_set.get(mapping_id)
        if mapping is None:
            raise MarketBackfillSpecError(f"unknown mapping_id {mapping_id!r} is not present in the supplied SymbolMappingSet")
        if not mapping.enabled:
            raise MarketBackfillSpecError(f"mapping_id {mapping_id!r} is disabled -- enable it before selecting it for a backfill")
        if mapping.canonical_driver_id not in ordered_driver_ids:
            raise MarketBackfillSpecError(
                f"mapping_id {mapping_id!r} belongs to canonical_driver_id {mapping.canonical_driver_id!r}, which is not in selected_driver_ids"
            )

    provisional = MarketBackfillSpec(
        backfill_plan_id="0" * 64, curated_registry_id=registry.registry_id, selected_driver_ids=ordered_driver_ids,
        selected_mapping_ids=ordered_mapping_ids, start_time=start_time, end_time=end_time, requested_granularity=requested_granularity,
        cache_policy=cache_policy, target_dataset_namespace=target_dataset_namespace, fail_fast=fail_fast, max_driver_count=max_driver_count,
        max_records_per_mapping=max_records_per_mapping, max_total_raw_bytes=max_total_raw_bytes,
    )
    backfill_plan_id = compute_content_id(MARKET_BACKFILL_SPEC_KIND, provisional.to_identity_payload())
    return MarketBackfillSpec(
        backfill_plan_id=backfill_plan_id, curated_registry_id=registry.registry_id, selected_driver_ids=ordered_driver_ids,
        selected_mapping_ids=ordered_mapping_ids, start_time=start_time, end_time=end_time, requested_granularity=requested_granularity,
        cache_policy=cache_policy, target_dataset_namespace=target_dataset_namespace, fail_fast=fail_fast, max_driver_count=max_driver_count,
        max_records_per_mapping=max_records_per_mapping, max_total_raw_bytes=max_total_raw_bytes,
    )
