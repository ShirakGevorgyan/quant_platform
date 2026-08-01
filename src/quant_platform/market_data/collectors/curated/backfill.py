"""Immutable, content-addressed multi-series curated backfill spec
(Milestone 10, Phase 4B). Same semantic plan always produces the same
`backfill_plan_id` -- constructed ONLY via `create_curated_backfill_spec`,
which validates every selection against a `CuratedFredRegistry` (unknown
or disabled series rejected) before a spec can even exist."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from quant_platform.core.exceptions import CuratedBackfillSpecError
from quant_platform.market_data.collectors.curated.registry import CuratedFredRegistry
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)

__all__ = [
    "CURATED_BACKFILL_SPEC_KIND",
    "CachePolicy",
    "CuratedBackfillSpec",
    "create_curated_backfill_spec",
]

CURATED_BACKFILL_SPEC_KIND = "curated_backfill_spec"


class CachePolicy(Enum):
    PREFER_CACHE = "prefer_cache"
    """Reuse an already-cached response for the same semantic request
    when one exists (`FetchMode.CACHED_REPLAY` per series, falling back
    to `FRESH` only when nothing is cached yet) -- see `orchestration.py`."""

    FORCE_FRESH = "force_fresh"
    """Always fetch fresh (`FetchMode.FRESH` for every series), even if
    a cached response already exists -- the spec's own "a fresh-network
    policy may produce a new response version" case, made explicit."""


@dataclass(frozen=True, slots=True)
class CuratedBackfillSpec:
    backfill_plan_id: str
    curated_registry_id: str
    selected_series_ids: tuple[str, ...]
    """Always sorted -- both the identity-independent-of-declaration-
    order requirement AND the "stable series processing order"
    orchestration requirement are satisfied by the SAME sort."""
    observation_start: datetime
    observation_end: datetime
    realtime_start: datetime | None
    realtime_end: datetime | None
    revision_policy_id: str
    output_type: int | None
    page_size: int
    cache_policy: CachePolicy
    availability_policy_registry_id: str | None
    """`None` means "use each selected series' own declared
    `release_availability_policy_id` from the registry" -- set only to
    UNIFORMLY override every selected series onto a single policy."""
    normalization_registry_id: str | None
    """`None` means "use each selected series' own declared
    `normalization_kind`/`unit_conversion` from the registry" -- set
    only to override onto a single shared `macro_normalization.
    UnitMappingSpec.unit_mapping_id`."""
    target_dataset_namespace: str
    fail_fast: bool
    max_series_count: int
    max_observations_per_series: int
    max_total_raw_bytes: int

    def __post_init__(self) -> None:
        require_non_empty(self.curated_registry_id, field_name="CuratedBackfillSpec.curated_registry_id")
        require_non_empty(self.revision_policy_id, field_name="CuratedBackfillSpec.revision_policy_id")
        require_non_empty(self.target_dataset_namespace, field_name="CuratedBackfillSpec.target_dataset_namespace")
        if not self.selected_series_ids:
            raise CuratedBackfillSpecError("CuratedBackfillSpec.selected_series_ids must not be empty")
        if len(set(self.selected_series_ids)) != len(self.selected_series_ids):
            raise CuratedBackfillSpecError(f"CuratedBackfillSpec.selected_series_ids contains duplicates: {self.selected_series_ids!r}")
        if tuple(sorted(self.selected_series_ids)) != self.selected_series_ids:
            raise CuratedBackfillSpecError("CuratedBackfillSpec.selected_series_ids must be constructed via create_curated_backfill_spec (which sorts them)")
        require_tz_aware(self.observation_start, field_name="CuratedBackfillSpec.observation_start")
        require_tz_aware(self.observation_end, field_name="CuratedBackfillSpec.observation_end")
        if self.observation_end < self.observation_start:
            raise CuratedBackfillSpecError(f"CuratedBackfillSpec.observation_end ({self.observation_end}) must be >= observation_start ({self.observation_start})")
        if self.realtime_start is not None:
            require_tz_aware(self.realtime_start, field_name="CuratedBackfillSpec.realtime_start")
        if self.realtime_end is not None:
            require_tz_aware(self.realtime_end, field_name="CuratedBackfillSpec.realtime_end")
        if not (1 <= self.page_size <= 100_000):
            raise CuratedBackfillSpecError(f"CuratedBackfillSpec.page_size must be in [1, 100000], got {self.page_size}")
        if self.max_series_count < 1:
            raise CuratedBackfillSpecError(f"CuratedBackfillSpec.max_series_count must be >= 1, got {self.max_series_count}")
        if len(self.selected_series_ids) > self.max_series_count:
            raise CuratedBackfillSpecError(f"CuratedBackfillSpec selects {len(self.selected_series_ids)} series, exceeding max_series_count={self.max_series_count}")
        if self.max_observations_per_series < 1:
            raise CuratedBackfillSpecError(f"CuratedBackfillSpec.max_observations_per_series must be >= 1, got {self.max_observations_per_series}")
        if self.max_total_raw_bytes < 1:
            raise CuratedBackfillSpecError(f"CuratedBackfillSpec.max_total_raw_bytes must be >= 1, got {self.max_total_raw_bytes}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": CURATED_BACKFILL_SPEC_KIND, "backfill_plan_id": self.backfill_plan_id, "curated_registry_id": self.curated_registry_id,
            "selected_series_ids": list(self.selected_series_ids),
            "observation_start": serialize_timestamp(self.observation_start, field_name="observation_start"),
            "observation_end": serialize_timestamp(self.observation_end, field_name="observation_end"),
            "realtime_start": (None if self.realtime_start is None else serialize_timestamp(self.realtime_start, field_name="realtime_start")),
            "realtime_end": (None if self.realtime_end is None else serialize_timestamp(self.realtime_end, field_name="realtime_end")),
            "revision_policy_id": self.revision_policy_id, "output_type": self.output_type, "page_size": self.page_size,
            "cache_policy": self.cache_policy.value, "availability_policy_registry_id": self.availability_policy_registry_id,
            "normalization_registry_id": self.normalization_registry_id, "target_dataset_namespace": self.target_dataset_namespace,
            "fail_fast": self.fail_fast, "max_series_count": self.max_series_count, "max_observations_per_series": self.max_observations_per_series,
            "max_total_raw_bytes": self.max_total_raw_bytes,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["backfill_plan_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CuratedBackfillSpec:
        from quant_platform.ml.persistence import as_json_list

        raw_rs = raw.get("realtime_start")
        raw_re = raw.get("realtime_end")
        raw_ot = raw.get("output_type")
        return cls(
            backfill_plan_id=str(raw["backfill_plan_id"]), curated_registry_id=str(raw["curated_registry_id"]),
            selected_series_ids=tuple(str(s) for s in as_json_list(raw["selected_series_ids"], field_name="selected_series_ids")),
            observation_start=deserialize_timestamp(raw["observation_start"], field_name="observation_start"),
            observation_end=deserialize_timestamp(raw["observation_end"], field_name="observation_end"),
            realtime_start=(None if raw_rs is None else deserialize_timestamp(raw_rs, field_name="realtime_start")),
            realtime_end=(None if raw_re is None else deserialize_timestamp(raw_re, field_name="realtime_end")),
            revision_policy_id=str(raw["revision_policy_id"]), output_type=(None if raw_ot is None else int(str(raw_ot))),
            page_size=int(str(raw["page_size"])), cache_policy=CachePolicy(raw["cache_policy"]),
            availability_policy_registry_id=(None if raw.get("availability_policy_registry_id") is None else str(raw["availability_policy_registry_id"])),
            normalization_registry_id=(None if raw.get("normalization_registry_id") is None else str(raw["normalization_registry_id"])),
            target_dataset_namespace=str(raw["target_dataset_namespace"]), fail_fast=bool(raw["fail_fast"]),
            max_series_count=int(str(raw["max_series_count"])), max_observations_per_series=int(str(raw["max_observations_per_series"])),
            max_total_raw_bytes=int(str(raw["max_total_raw_bytes"])),
        )


def create_curated_backfill_spec(
    *, registry: CuratedFredRegistry, selected_series_ids: tuple[str, ...], observation_start: datetime, observation_end: datetime,
    revision_policy_id: str, target_dataset_namespace: str, realtime_start: datetime | None = None, realtime_end: datetime | None = None,
    output_type: int | None = None, page_size: int = 100_000, cache_policy: CachePolicy = CachePolicy.PREFER_CACHE,
    availability_policy_registry_id: str | None = None, normalization_registry_id: str | None = None, fail_fast: bool = True,
    max_series_count: int = 64, max_observations_per_series: int = 200_000, max_total_raw_bytes: int = 500_000_000,
) -> CuratedBackfillSpec:
    if not selected_series_ids:
        raise CuratedBackfillSpecError("selected_series_ids must not be empty")
    for series_id in selected_series_ids:
        spec = registry.get(series_id)
        if spec is None:
            raise CuratedBackfillSpecError(f"unknown series_id {series_id!r} is not present in registry {registry.registry_id!r}")
        if not spec.enabled:
            raise CuratedBackfillSpecError(f"series_id {series_id!r} is disabled in the registry -- enable it before selecting it for a backfill")

    ordered_series_ids = tuple(sorted(set(selected_series_ids)))
    provisional = CuratedBackfillSpec(
        backfill_plan_id="0" * 64, curated_registry_id=registry.registry_id, selected_series_ids=ordered_series_ids,
        observation_start=observation_start, observation_end=observation_end, realtime_start=realtime_start, realtime_end=realtime_end,
        revision_policy_id=revision_policy_id, output_type=output_type, page_size=page_size, cache_policy=cache_policy,
        availability_policy_registry_id=availability_policy_registry_id, normalization_registry_id=normalization_registry_id,
        target_dataset_namespace=target_dataset_namespace, fail_fast=fail_fast, max_series_count=max_series_count,
        max_observations_per_series=max_observations_per_series, max_total_raw_bytes=max_total_raw_bytes,
    )
    backfill_plan_id = compute_content_id(CURATED_BACKFILL_SPEC_KIND, provisional.to_identity_payload())
    return CuratedBackfillSpec(
        backfill_plan_id=backfill_plan_id, curated_registry_id=registry.registry_id, selected_series_ids=ordered_series_ids,
        observation_start=observation_start, observation_end=observation_end, realtime_start=realtime_start, realtime_end=realtime_end,
        revision_policy_id=revision_policy_id, output_type=output_type, page_size=page_size, cache_policy=cache_policy,
        availability_policy_registry_id=availability_policy_registry_id, normalization_registry_id=normalization_registry_id,
        target_dataset_namespace=target_dataset_namespace, fail_fast=fail_fast, max_series_count=max_series_count,
        max_observations_per_series=max_observations_per_series, max_total_raw_bytes=max_total_raw_bytes,
    )
