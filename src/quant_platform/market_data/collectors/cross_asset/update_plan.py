"""Pure, deterministic incremental cross-asset update planning
(Milestone 10, Phase 4C, spec Section 22) -- NEVER reads the wall clock
("today"), NEVER touches the network. Compares each selected mapping's
CURRENT `ComponentMarketDatasetManifest` (already-durable coverage)
against a caller-supplied `desired_end_time`, deciding per mapping
whether nothing needs to happen, new bars need appending, or a POLICY
CHANGE (adjustment or continuation policy) requires a full refresh --
since a different policy can legitimately change what "the value" even
means for already-covered dates.

EXACT NO-OP GUARANTEE: when every mapping resolves to `NO_UPDATE_NEEDED`,
this plan recommends running nothing at all -- and even if a caller runs
`run_cross_asset_backfill_operation` anyway with an unchanged interval,
`ComponentMarketDatasetManifestStore.append`'s own idempotent no-op-on-
identical-id behavior (see `datasets.py`) independently guarantees no
new dataset version is minted. This module's OWN job is simply to
report `NO_UPDATE_NEEDED` correctly, not to re-implement that
guarantee."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from quant_platform.core.exceptions import MarketUpdatePlanError
from quant_platform.market_data.collectors.cross_asset.datasets import (
    CombinedCrossAssetManifest,
    ComponentMarketDatasetManifestStore,
)
from quant_platform.market_data.collectors.cross_asset.instrument_form import InstrumentForm
from quant_platform.market_data.collectors.cross_asset.registry import CuratedMarketDriverRegistry
from quant_platform.market_data.collectors.cross_asset.symbol_mapping import SymbolMappingSet
from quant_platform.market_data.identity import compute_content_id, require_tz_aware

__all__ = [
    "CROSS_ASSET_UPDATE_PLAN_KIND",
    "CrossAssetUpdatePlan",
    "MappingUpdateAction",
    "MappingUpdatePlanEntry",
    "create_cross_asset_update_plan",
]

CROSS_ASSET_UPDATE_PLAN_KIND = "cross_asset_update_plan"

_FUTURES_FORMS = frozenset({InstrumentForm.EXCHANGE_FUTURES_CONTRACT, InstrumentForm.PROVIDER_CONTINUOUS_FUTURES})


class MappingUpdateAction(Enum):
    NO_UPDATE_NEEDED = "no_update_needed"
    APPEND_BARS = "append_bars"
    POLICY_REFRESH = "policy_refresh"
    """The mapping's own `adjustment_policy_kind`/`continuation_policy_id`
    no longer matches what the CURRENT component manifest was built
    under -- a full refresh from scratch is required, since already-
    covered bars may now resolve to a DIFFERENT value under the new
    policy (spec Section 10/12's own "policy CHANGE changes dataset
    identity")."""


@dataclass(frozen=True, slots=True)
class MappingUpdatePlanEntry:
    mapping_id: str
    canonical_driver_id: str
    action: MappingUpdateAction
    requested_interval_start: datetime | None
    requested_interval_end: datetime | None
    current_component_manifest_id: str | None
    overlap_interval: tuple[str | None, str | None]
    """`(existing_coverage_start, existing_coverage_end)` -- reported
    purely for visibility, never used to skip a value the requested
    interval legitimately re-touches (e.g. a `POLICY_REFRESH`)."""
    needs_futures_roll_refresh: bool
    """`True` whenever `instrument_form` is a futures form -- roll state
    may have advanced since the mapping was last fetched, and this
    module does not itself track roll timing (spec Section 22's own
    "futures-roll refresh needs"); an honest flag, never a computed
    roll decision."""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "mapping_id": self.mapping_id, "canonical_driver_id": self.canonical_driver_id, "action": self.action.value,
            "requested_interval_start": (None if self.requested_interval_start is None else self.requested_interval_start.isoformat()),
            "requested_interval_end": (None if self.requested_interval_end is None else self.requested_interval_end.isoformat()),
            "current_component_manifest_id": self.current_component_manifest_id, "overlap_interval": list(self.overlap_interval),
            "needs_futures_roll_refresh": self.needs_futures_roll_refresh,
        }


@dataclass(frozen=True, slots=True)
class CrossAssetUpdatePlan:
    update_plan_id: str
    curated_registry_id: str
    target_dataset_namespace: str
    desired_end_time: datetime
    entries: tuple[MappingUpdatePlanEntry, ...]
    plan_issues: tuple[str, ...]
    planning_time: datetime

    def __post_init__(self) -> None:
        require_tz_aware(self.desired_end_time, field_name="CrossAssetUpdatePlan.desired_end_time")
        require_tz_aware(self.planning_time, field_name="CrossAssetUpdatePlan.planning_time")

    def mappings_requiring_update(self) -> tuple[str, ...]:
        return tuple(e.mapping_id for e in self.entries if e.action is not MappingUpdateAction.NO_UPDATE_NEEDED)

    def is_exact_no_op(self) -> bool:
        return all(e.action is MappingUpdateAction.NO_UPDATE_NEEDED for e in self.entries) and not self.plan_issues

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": CROSS_ASSET_UPDATE_PLAN_KIND, "update_plan_id": self.update_plan_id, "curated_registry_id": self.curated_registry_id,
            "target_dataset_namespace": self.target_dataset_namespace, "desired_end_time": self.desired_end_time.isoformat(),
            "entries": [e.to_json_dict() for e in self.entries], "plan_issues": list(self.plan_issues), "planning_time": self.planning_time.isoformat(),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["update_plan_id"]
        del payload["planning_time"]
        return payload


def create_cross_asset_update_plan(
    *, existing_combined_manifest: CombinedCrossAssetManifest | None, component_store: ComponentMarketDatasetManifestStore,
    registry: CuratedMarketDriverRegistry, mapping_set: SymbolMappingSet, selected_mapping_ids: tuple[str, ...], target_dataset_namespace: str,
    desired_end_time: datetime, planning_time: datetime,
) -> CrossAssetUpdatePlan:
    require_tz_aware(desired_end_time, field_name="desired_end_time")
    require_tz_aware(planning_time, field_name="planning_time")
    if not selected_mapping_ids:
        raise MarketUpdatePlanError("create_cross_asset_update_plan requires a non-empty selected_mapping_ids")
    if existing_combined_manifest is not None and existing_combined_manifest.curated_registry_id != registry.registry_id:
        raise MarketUpdatePlanError(
            f"existing_combined_manifest.curated_registry_id={existing_combined_manifest.curated_registry_id!r} does not match registry.registry_id={registry.registry_id!r}"
        )

    entries: list[MappingUpdatePlanEntry] = []
    issues: list[str] = []
    for mapping_id in sorted(set(selected_mapping_ids)):
        mapping = mapping_set.get(mapping_id)
        if mapping is None:
            issues.append(f"unknown_mapping:{mapping_id}")
            continue
        if not mapping.enabled:
            issues.append(f"disabled_mapping:{mapping_id}")
            continue
        driver_spec = registry.get(mapping.canonical_driver_id)
        if driver_spec is None:
            issues.append(f"unknown_driver_for_mapping:{mapping_id}")
            continue
        if not driver_spec.enabled:
            issues.append(f"disabled_driver_for_mapping:{mapping_id}")
            continue

        needs_roll_refresh = mapping.instrument_form in _FUTURES_FORMS
        component = component_store.read_current(mapping_id)
        if component is None:
            entries.append(MappingUpdatePlanEntry(
                mapping_id=mapping_id, canonical_driver_id=mapping.canonical_driver_id, action=MappingUpdateAction.APPEND_BARS,
                requested_interval_start=None, requested_interval_end=desired_end_time, current_component_manifest_id=None,
                overlap_interval=(None, None), needs_futures_roll_refresh=needs_roll_refresh,
            ))
            continue

        policy_changed = component.adjustment_policy_id != driver_spec.adjustment_policy_id or component.continuation_policy_id != mapping.continuation_policy_id
        if policy_changed:
            entries.append(MappingUpdatePlanEntry(
                mapping_id=mapping_id, canonical_driver_id=mapping.canonical_driver_id, action=MappingUpdateAction.POLICY_REFRESH,
                requested_interval_start=None, requested_interval_end=desired_end_time, current_component_manifest_id=component.component_manifest_id,
                overlap_interval=(component.coverage_start, component.coverage_end), needs_futures_roll_refresh=needs_roll_refresh,
            ))
            continue

        desired_end_text = desired_end_time.isoformat()
        if component.coverage_end is None or desired_end_text > component.coverage_end:
            append_start = None if component.coverage_end is None else datetime.fromisoformat(component.coverage_end) + timedelta(seconds=1)
            entries.append(MappingUpdatePlanEntry(
                mapping_id=mapping_id, canonical_driver_id=mapping.canonical_driver_id, action=MappingUpdateAction.APPEND_BARS,
                requested_interval_start=append_start, requested_interval_end=desired_end_time, current_component_manifest_id=component.component_manifest_id,
                overlap_interval=(component.coverage_start, component.coverage_end), needs_futures_roll_refresh=needs_roll_refresh,
            ))
            continue

        entries.append(MappingUpdatePlanEntry(
            mapping_id=mapping_id, canonical_driver_id=mapping.canonical_driver_id, action=MappingUpdateAction.NO_UPDATE_NEEDED,
            requested_interval_start=None, requested_interval_end=None, current_component_manifest_id=component.component_manifest_id,
            overlap_interval=(component.coverage_start, component.coverage_end), needs_futures_roll_refresh=needs_roll_refresh,
        ))

    provisional = CrossAssetUpdatePlan(
        update_plan_id="0" * 64, curated_registry_id=registry.registry_id, target_dataset_namespace=target_dataset_namespace,
        desired_end_time=desired_end_time, entries=tuple(entries), plan_issues=tuple(issues), planning_time=planning_time,
    )
    update_plan_id = compute_content_id(CROSS_ASSET_UPDATE_PLAN_KIND, provisional.to_identity_payload())
    return CrossAssetUpdatePlan(
        update_plan_id=update_plan_id, curated_registry_id=registry.registry_id, target_dataset_namespace=target_dataset_namespace,
        desired_end_time=desired_end_time, entries=tuple(entries), plan_issues=tuple(issues), planning_time=planning_time,
    )
