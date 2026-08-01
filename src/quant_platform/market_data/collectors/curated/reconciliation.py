"""Cross-store consistency reconciliation for the curated universe
(Milestone 10, Phase 4B). Mirrors `collectors.reconciliation.
reconcile_fred_macro_dataset`'s own scan-and-report shape (itself
mirroring `market_data.provenance.find_provenance_conflicts`), extended
to span an entire multi-series curated universe: the registry, every
component dataset, and the combined manifest that binds them."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import pandas as pd

from quant_platform.market_data.collectors.curated.datasets import (
    CombinedUniverseManifestStore,
    ComponentDatasetManifestStore,
)
from quant_platform.market_data.collectors.curated.macro_observation import CuratedObservationStore
from quant_platform.market_data.collectors.curated.registry import CuratedFredRegistry
from quant_platform.market_data.identity import require_tz_aware
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.provenance import ProvenanceStore
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.ml.models import ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import format_utc_timestamp

__all__ = ["CURATED_RECONCILIATION_REPORT_SCHEMA_VERSION", "reconcile_curated_universe"]

CURATED_RECONCILIATION_REPORT_SCHEMA_VERSION = 1


def _issue(severity: ValidationSeverity, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message)


def _report(issues: list[ValidationIssue], *, as_of: datetime) -> ValidationReport:
    return ValidationReport(schema_version=CURATED_RECONCILIATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))


def reconcile_curated_universe(
    *, repository: MarketDataRepository, registry: CuratedFredRegistry, target_dataset_namespace: str, as_of: datetime, provider: str = "fred",
) -> ValidationReport:
    require_tz_aware(as_of, field_name="as_of")
    issues: list[ValidationIssue] = []

    combined_store = CombinedUniverseManifestStore(repository.root)
    component_store = ComponentDatasetManifestStore(repository.root)
    observation_store = CuratedObservationStore(repository.root)
    provenance_store = ProvenanceStore(repository.root)

    combined = combined_store.read_current(target_dataset_namespace)
    if combined is None:
        issues.append(_issue(ValidationSeverity.CRITICAL, "combined_manifest_missing", f"No combined manifest exists yet for namespace {target_dataset_namespace!r}."))
        return _report(issues, as_of=as_of)

    if combined.curated_registry_id != registry.registry_id:
        issues.append(_issue(ValidationSeverity.WARNING, "registry_version_mismatch", f"Combined manifest was built against registry_id={combined.curated_registry_id!r}, caller supplied a different registry_id={registry.registry_id!r}."))

    for series_id, component_manifest_id in combined.component_manifest_ids.items():
        spec = registry.get(series_id)
        if spec is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "curated_registry_series_missing", f"Combined manifest references series_id={series_id!r}, which is absent from the supplied registry."))
            continue
        if not spec.enabled:
            issues.append(_issue(ValidationSeverity.WARNING, "combined_manifest_references_disabled_series", f"Combined manifest references series_id={series_id!r}, which is disabled in the current registry."))

        # ---- component dataset manifest linkage ----
        current_component = component_store.read_current(provider, series_id)
        if current_component is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "component_manifest_missing", f"series_id={series_id!r}: combined manifest references component_manifest_id={component_manifest_id!r}, but no component manifest history exists."))
            continue
        if current_component.component_manifest_id != component_manifest_id:
            issues.append(_issue(ValidationSeverity.CRITICAL, "component_manifest_version_mismatch", f"series_id={series_id!r}: combined manifest references component_manifest_id={component_manifest_id!r}, but the CURRENT component manifest is {current_component.component_manifest_id!r}."))

        # ---- coverage intervals recomputed from the actual observation store ----
        observations = observation_store.read_observations(provider, series_id)
        dates = sorted(o.observation_date for o in observations)
        recomputed_start = dates[0] if dates else None
        recomputed_end = dates[-1] if dates else None
        if current_component.coverage_start != recomputed_start or current_component.coverage_end != recomputed_end:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "coverage_interval_mismatch",
                f"series_id={series_id!r}: component manifest reports coverage=[{current_component.coverage_start!r}, {current_component.coverage_end!r}], recomputed from the observation store=[{recomputed_start!r}, {recomputed_end!r}].",
            ))
        if current_component.observation_count != len(observations):
            issues.append(_issue(ValidationSeverity.CRITICAL, "observation_count_mismatch", f"series_id={series_id!r}: component manifest reports observation_count={current_component.observation_count}, observation store has {len(observations)}."))

        # ---- vintage/revision uniqueness: same (date, realtime_start) must never carry two different values ----
        by_vintage: dict[tuple[str, str], set[str]] = defaultdict(set)
        for o in observations:
            by_vintage[(o.observation_date, o.realtime_start or "")].add("MISSING" if o.is_missing else str(o.value))
        for (date, realtime_start), values in by_vintage.items():
            if len(values) > 1:
                issues.append(_issue(ValidationSeverity.CRITICAL, "conflicting_vintage", f"series_id={series_id!r}: observation_date={date!r} at realtime_start={realtime_start!r} has {len(values)} different recorded values: {sorted(values)!r}."))

        # ---- provenance completeness: every observation must have a provenance record, and vice versa ----
        dataset_key = DatasetKey(dataset_kind=DatasetKind.MACRO_OBSERVATIONS, provider=provider, instrument_id=series_id)
        provenance_records = provenance_store.read_all(dataset_key)
        provenance_event_ids = {p.event_id for p in provenance_records}
        for o in observations:
            if o.observation_id not in provenance_event_ids:
                issues.append(_issue(ValidationSeverity.CRITICAL, "observation_without_provenance", f"series_id={series_id!r}: observation_id={o.observation_id!r} has no matching provenance record."))
        observation_ids = {o.observation_id for o in observations}
        for p in provenance_records:
            if p.event_id not in observation_ids:
                issues.append(_issue(ValidationSeverity.CRITICAL, "provenance_without_observation", f"series_id={series_id!r}: provenance_id={p.provenance_id!r} references observation_id={p.event_id!r}, absent from the observation store."))
            coordinate_matches = [q for q in provenance_records if q.source_manifest_id == p.source_manifest_id and q.source_row_index == p.source_row_index]
            if len(coordinate_matches) > 1:
                issues.append(_issue(ValidationSeverity.CRITICAL, "duplicate_provenance_coordinate", f"series_id={series_id!r}: source coordinate (source_manifest_id={p.source_manifest_id!r}, row_index={p.source_row_index}) has {len(coordinate_matches)} provenance records."))

    for series_id in registry.enabled_series_ids():
        if series_id not in combined.component_manifest_ids and series_id in combined.frequencies_by_series:
            issues.append(_issue(ValidationSeverity.WARNING, "enabled_series_not_in_combined_manifest", f"series_id={series_id!r} is enabled in the registry but not referenced by the current combined manifest."))

    return _report(issues, as_of=as_of)
