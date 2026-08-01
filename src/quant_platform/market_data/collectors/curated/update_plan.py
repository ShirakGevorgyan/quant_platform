"""Pure, deterministic incremental update planning (Milestone 10, Phase
4B) -- NEVER reads the wall clock ("today"), NEVER touches the network.
Compares each selected series' CURRENT `ComponentDatasetManifest`
(already-durable coverage) against a caller-supplied
`desired_observation_end` and `RevisionPolicy`, deciding per series
whether nothing needs to happen, new observations need appending, or a
revision-policy CHANGE requires a full refresh (since a different
policy can legitimately change what "the value" even means for
already-covered dates).

EXACT NO-OP GUARANTEE: when every series resolves to `NO_UPDATE_NEEDED`,
this plan recommends running nothing at all -- and even if a caller
runs `run_curated_backfill_operation` anyway with an unchanged interval,
`ComponentDatasetManifestStore.append`'s own idempotent no-op-on-
identical-id behavior (see `datasets.py`) independently guarantees no
new dataset version is minted. This module's OWN job is simply to
report `NO_UPDATE_NEEDED` correctly, not to re-implement that
guarantee."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from quant_platform.core.exceptions import UpdatePlanError
from quant_platform.market_data.collectors.curated.datasets import (
    CombinedUniverseManifest,
    ComponentDatasetManifestStore,
)
from quant_platform.market_data.collectors.curated.registry import CuratedFredRegistry
from quant_platform.market_data.collectors.curated.revision_policy import RevisionPolicy
from quant_platform.market_data.identity import compute_content_id, require_tz_aware

__all__ = [
    "CURATED_UPDATE_PLAN_KIND",
    "CuratedUpdatePlan",
    "SeriesUpdateAction",
    "SeriesUpdatePlanEntry",
    "create_curated_update_plan",
]

CURATED_UPDATE_PLAN_KIND = "curated_update_plan"


class SeriesUpdateAction(Enum):
    NO_UPDATE_NEEDED = "no_update_needed"
    APPEND_OBSERVATIONS = "append_observations"
    REVISION_REFRESH = "revision_refresh"
    """The `RevisionPolicy` in effect changed since the series was last
    fetched -- a full refresh from the series' own
    `default_observation_start` is required, since already-covered
    dates may now resolve to DIFFERENT vintages under the new policy."""


@dataclass(frozen=True, slots=True)
class SeriesUpdatePlanEntry:
    series_id: str
    action: SeriesUpdateAction
    requested_interval_start: datetime | None
    requested_interval_end: datetime | None
    current_component_manifest_id: str | None
    overlap_interval: tuple[str | None, str | None]
    """`(existing_coverage_start, existing_coverage_end)` -- reported
    purely for visibility into what is ALREADY covered and would
    overlap the newly requested interval; never used to skip a value
    the requested interval legitimately re-touches (e.g. a
    `REVISION_REFRESH`)."""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "series_id": self.series_id, "action": self.action.value,
            "requested_interval_start": (None if self.requested_interval_start is None else self.requested_interval_start.isoformat()),
            "requested_interval_end": (None if self.requested_interval_end is None else self.requested_interval_end.isoformat()),
            "current_component_manifest_id": self.current_component_manifest_id, "overlap_interval": list(self.overlap_interval),
        }


@dataclass(frozen=True, slots=True)
class CuratedUpdatePlan:
    update_plan_id: str
    curated_registry_id: str
    target_dataset_namespace: str
    desired_observation_end: datetime
    revision_policy_id: str
    entries: tuple[SeriesUpdatePlanEntry, ...]
    plan_issues: tuple[str, ...]
    planning_time: datetime

    def __post_init__(self) -> None:
        require_tz_aware(self.desired_observation_end, field_name="CuratedUpdatePlan.desired_observation_end")
        require_tz_aware(self.planning_time, field_name="CuratedUpdatePlan.planning_time")

    def series_requiring_update(self) -> tuple[str, ...]:
        return tuple(e.series_id for e in self.entries if e.action is not SeriesUpdateAction.NO_UPDATE_NEEDED)

    def is_exact_no_op(self) -> bool:
        return all(e.action is SeriesUpdateAction.NO_UPDATE_NEEDED for e in self.entries) and not self.plan_issues

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": CURATED_UPDATE_PLAN_KIND, "update_plan_id": self.update_plan_id, "curated_registry_id": self.curated_registry_id,
            "target_dataset_namespace": self.target_dataset_namespace, "desired_observation_end": self.desired_observation_end.isoformat(),
            "revision_policy_id": self.revision_policy_id, "entries": [e.to_json_dict() for e in self.entries], "plan_issues": list(self.plan_issues),
            "planning_time": self.planning_time.isoformat(),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["update_plan_id"]
        del payload["planning_time"]
        return payload


def create_curated_update_plan(
    *, existing_combined_manifest: CombinedUniverseManifest | None, component_store: ComponentDatasetManifestStore, registry: CuratedFredRegistry,
    selected_series_ids: tuple[str, ...], target_dataset_namespace: str, desired_observation_end: datetime, revision_policy: RevisionPolicy,
    planning_time: datetime, provider: str = "fred",
) -> CuratedUpdatePlan:
    require_tz_aware(desired_observation_end, field_name="desired_observation_end")
    require_tz_aware(planning_time, field_name="planning_time")
    if not selected_series_ids:
        raise UpdatePlanError("create_curated_update_plan requires a non-empty selected_series_ids")

    policy_changed = existing_combined_manifest is not None and existing_combined_manifest.revision_policy_id != revision_policy.revision_policy_id
    desired_end_text = desired_observation_end.strftime("%Y-%m-%d")

    entries: list[SeriesUpdatePlanEntry] = []
    issues: list[str] = []
    for series_id in sorted(set(selected_series_ids)):
        spec = registry.get(series_id)
        if spec is None:
            issues.append(f"unknown_series:{series_id}")
            continue
        if not spec.enabled:
            issues.append(f"disabled_series:{series_id}")
            continue

        component = component_store.read_current(provider, series_id)
        if component is None:
            entries.append(SeriesUpdatePlanEntry(
                series_id=series_id, action=SeriesUpdateAction.APPEND_OBSERVATIONS, requested_interval_start=spec.default_observation_start,
                requested_interval_end=desired_observation_end, current_component_manifest_id=None, overlap_interval=(None, None),
            ))
            continue

        if policy_changed:
            entries.append(SeriesUpdatePlanEntry(
                series_id=series_id, action=SeriesUpdateAction.REVISION_REFRESH, requested_interval_start=spec.default_observation_start,
                requested_interval_end=desired_observation_end, current_component_manifest_id=component.component_manifest_id,
                overlap_interval=(component.coverage_start, component.coverage_end),
            ))
            continue

        if component.coverage_end is None or desired_end_text > component.coverage_end:
            from datetime import timedelta

            append_start = spec.default_observation_start if component.coverage_end is None else datetime.fromisoformat(component.coverage_end).replace(tzinfo=desired_observation_end.tzinfo) + timedelta(days=1)
            entries.append(SeriesUpdatePlanEntry(
                series_id=series_id, action=SeriesUpdateAction.APPEND_OBSERVATIONS, requested_interval_start=append_start,
                requested_interval_end=desired_observation_end, current_component_manifest_id=component.component_manifest_id,
                overlap_interval=(component.coverage_start, component.coverage_end),
            ))
            continue

        entries.append(SeriesUpdatePlanEntry(
            series_id=series_id, action=SeriesUpdateAction.NO_UPDATE_NEEDED, requested_interval_start=None, requested_interval_end=None,
            current_component_manifest_id=component.component_manifest_id, overlap_interval=(component.coverage_start, component.coverage_end),
        ))

    curated_registry_id = registry.registry_id
    provisional = CuratedUpdatePlan(
        update_plan_id="0" * 64, curated_registry_id=curated_registry_id, target_dataset_namespace=target_dataset_namespace,
        desired_observation_end=desired_observation_end, revision_policy_id=revision_policy.revision_policy_id, entries=tuple(entries),
        plan_issues=tuple(issues), planning_time=planning_time,
    )
    update_plan_id = compute_content_id(CURATED_UPDATE_PLAN_KIND, provisional.to_identity_payload())
    return CuratedUpdatePlan(
        update_plan_id=update_plan_id, curated_registry_id=curated_registry_id, target_dataset_namespace=target_dataset_namespace,
        desired_observation_end=desired_observation_end, revision_policy_id=revision_policy.revision_policy_id, entries=tuple(entries),
        plan_issues=tuple(issues), planning_time=planning_time,
    )
