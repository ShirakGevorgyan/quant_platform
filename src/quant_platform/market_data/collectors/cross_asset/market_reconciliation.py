"""Cross-store consistency reconciliation for the curated cross-asset
universe (Milestone 10, Phase 4C, spec Section 26). Mirrors
`curated.reconciliation.reconcile_curated_universe`'s own scan-and-report
shape, extended to span the registry, every component dataset, the
combined manifest that binds them, AND -- the cross-asset-specific
addition spec Section 20 requires -- the CROSS-PROVIDER CONFLICT MODEL:
when more than one mapping serves the SAME `canonical_driver_id`,
overlapping bars are compared deterministically and classified as exact
equality, tolerance-level difference, or material conflict. Provider
prices are NEVER averaged or silently reconciled into one truth -- every
component dataset stays independently readable; this function only
REPORTS the comparison."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from decimal import Decimal

import pandas as pd

from quant_platform.market_data.collectors.cross_asset.datasets import (
    CombinedCrossAssetManifestStore,
    ComponentMarketDatasetManifestStore,
)
from quant_platform.market_data.collectors.cross_asset.gap_policy import analyze_bar_gaps
from quant_platform.market_data.collectors.cross_asset.market_record import MarketDriverBarStore
from quant_platform.market_data.collectors.cross_asset.registry import CuratedMarketDriverRegistry
from quant_platform.market_data.collectors.cross_asset.sessions import TimezoneSessionPolicy
from quant_platform.market_data.collectors.cross_asset.symbol_mapping import SymbolMappingSet
from quant_platform.market_data.identity import require_tz_aware
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.provenance import ProvenanceStore
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.ml.models import ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import format_utc_timestamp

__all__ = [
    "CROSS_ASSET_RECONCILIATION_REPORT_SCHEMA_VERSION",
    "PRICE_TOLERANCE_RATIO",
    "reconcile_cross_asset_universe",
]

CROSS_ASSET_RECONCILIATION_REPORT_SCHEMA_VERSION = 1
PRICE_TOLERANCE_RATIO = Decimal("0.005")
"""0.5% -- close prices within this relative tolerance across providers
for the SAME driver/coordinate are classified `tolerance_level_difference`
rather than `material_conflict`. A deliberately conservative, disclosed
constant -- never silently widened to hide real disagreement."""


def _issue(severity: ValidationSeverity, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message)


def _report(issues: list[ValidationIssue], *, as_of: datetime) -> ValidationReport:
    return ValidationReport(schema_version=CROSS_ASSET_RECONCILIATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))


def reconcile_cross_asset_universe(
    *, repository: MarketDataRepository, registry: CuratedMarketDriverRegistry, mapping_set: SymbolMappingSet, target_dataset_namespace: str,
    session_policies: dict[str, TimezoneSessionPolicy], as_of: datetime,
) -> ValidationReport:
    require_tz_aware(as_of, field_name="as_of")
    issues: list[ValidationIssue] = []

    combined_store = CombinedCrossAssetManifestStore(repository.root)
    component_store = ComponentMarketDatasetManifestStore(repository.root)
    bar_store = MarketDriverBarStore(repository.root)
    provenance_store = ProvenanceStore(repository.root)

    combined = combined_store.read_current(target_dataset_namespace)
    if combined is None:
        issues.append(_issue(ValidationSeverity.CRITICAL, "combined_manifest_missing", f"No combined manifest exists yet for namespace {target_dataset_namespace!r}."))
        return _report(issues, as_of=as_of)

    if combined.curated_registry_id != registry.registry_id:
        issues.append(_issue(ValidationSeverity.WARNING, "registry_version_mismatch", f"Combined manifest was built against registry_id={combined.curated_registry_id!r}, caller supplied a different registry_id={registry.registry_id!r}."))

    bars_by_mapping = {}
    for mapping_id, component_manifest_id in combined.component_manifest_ids.items():
        mapping = mapping_set.get(mapping_id)
        if mapping is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "unknown_mapping_in_combined_manifest", f"Combined manifest references mapping_id={mapping_id!r}, absent from the supplied SymbolMappingSet."))
            continue
        driver_spec = registry.get(mapping.canonical_driver_id)
        if driver_spec is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "curated_registry_driver_missing", f"mapping_id={mapping_id!r}: canonical_driver_id={mapping.canonical_driver_id!r} is absent from the supplied registry."))
            continue
        if not driver_spec.enabled:
            issues.append(_issue(ValidationSeverity.WARNING, "combined_manifest_references_disabled_driver", f"mapping_id={mapping_id!r}: canonical_driver_id={mapping.canonical_driver_id!r} is disabled in the current registry."))

        current_component = component_store.read_current(mapping_id)
        if current_component is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "component_manifest_missing", f"mapping_id={mapping_id!r}: combined manifest references component_manifest_id={component_manifest_id!r}, but no component manifest history exists."))
            continue
        if current_component.component_manifest_id != component_manifest_id:
            issues.append(_issue(ValidationSeverity.CRITICAL, "component_manifest_version_mismatch", f"mapping_id={mapping_id!r}: combined manifest references component_manifest_id={component_manifest_id!r}, but the CURRENT component manifest is {current_component.component_manifest_id!r}."))

        bars = bar_store.read_bars(mapping.provider, mapping.canonical_driver_id, mapping.instrument_form)
        bars_by_mapping[mapping_id] = bars
        recomputed_count = len(bars)
        if current_component.bar_count != recomputed_count:
            issues.append(_issue(ValidationSeverity.CRITICAL, "bar_count_mismatch", f"mapping_id={mapping_id!r}: component manifest reports bar_count={current_component.bar_count}, bar store has {recomputed_count}."))

        session_policy = session_policies.get(driver_spec.session_policy_id)
        if session_policy is not None and bars:
            gap_report = analyze_bar_gaps(tuple(bars), session_policy=session_policy)
            if gap_report.has_conflicting_coordinates:
                issues.append(_issue(ValidationSeverity.CRITICAL, "conflicting_bar_coordinates", f"mapping_id={mapping_id!r}: {gap_report.conflicting_coordinate_count} conflicting duplicate bar coordinate(s) found in the committed bar store."))

        # ---- provenance completeness ----
        dataset_key = DatasetKey(dataset_kind=DatasetKind.CROSS_ASSET_MARKET_BARS, provider=mapping.provider, instrument_id=f"{mapping.canonical_driver_id}__{mapping.instrument_form.value}")
        provenance_records = provenance_store.read_all(dataset_key)
        provenance_event_ids = {p.event_id for p in provenance_records}
        for bar in bars:
            if bar.bar_id not in provenance_event_ids:
                issues.append(_issue(ValidationSeverity.CRITICAL, "bar_without_provenance", f"mapping_id={mapping_id!r}: bar_id={bar.bar_id!r} has no matching provenance record."))
        bar_ids = {bar.bar_id for bar in bars}
        for p in provenance_records:
            if p.event_id not in bar_ids:
                issues.append(_issue(ValidationSeverity.CRITICAL, "provenance_without_bar", f"mapping_id={mapping_id!r}: provenance_id={p.provenance_id!r} references bar_id={p.event_id!r}, absent from the bar store."))

    for driver_id in registry.enabled_driver_ids():
        driver_mapping_ids = {m.mapping_id for m in mapping_set.for_driver(driver_id)}
        if driver_id in combined.driver_id_by_mapping.values():
            continue
        if driver_mapping_ids & set(combined.component_manifest_ids.keys()):
            continue
        if driver_id in registry.required_driver_ids():
            issues.append(_issue(ValidationSeverity.WARNING, "enabled_required_driver_not_in_combined_manifest", f"canonical_driver_id={driver_id!r} is enabled and required but not referenced by the current combined manifest."))

    # ---- Cross-provider conflict model (spec Section 20) ----
    mappings_by_driver: dict[str, list[str]] = defaultdict(list)
    for mapping_id in bars_by_mapping:
        mapping = mapping_set.get(mapping_id)
        assert mapping is not None
        mappings_by_driver[mapping.canonical_driver_id].append(mapping_id)

    for driver_id, sibling_mapping_ids in mappings_by_driver.items():
        if len(sibling_mapping_ids) < 2:
            continue
        for i in range(len(sibling_mapping_ids)):
            for j in range(i + 1, len(sibling_mapping_ids)):
                left_id, right_id = sorted((sibling_mapping_ids[i], sibling_mapping_ids[j]))
                left_by_time = {b.open_time: b for b in bars_by_mapping[left_id]}
                right_by_time = {b.open_time: b for b in bars_by_mapping[right_id]}
                overlap = sorted(set(left_by_time) & set(right_by_time))
                exact, tolerance, material = 0, 0, 0
                for open_time in overlap:
                    lb, rb = left_by_time[open_time], right_by_time[open_time]
                    if lb.close == rb.close:
                        exact += 1
                    else:
                        ratio = abs(lb.close - rb.close) / max(abs(lb.close), abs(rb.close), Decimal(1))
                        if ratio <= PRICE_TOLERANCE_RATIO:
                            tolerance += 1
                        else:
                            material += 1
                if material > 0:
                    issues.append(_issue(
                        ValidationSeverity.WARNING, "cross_provider_material_conflict",
                        f"canonical_driver_id={driver_id!r}: mapping {left_id!r} vs {right_id!r} disagree materially on {material} of {len(overlap)} overlapping bar(s) (exact={exact}, tolerance={tolerance}).",
                    ))
                elif tolerance > 0:
                    issues.append(_issue(
                        ValidationSeverity.WARNING, "cross_provider_tolerance_difference",
                        f"canonical_driver_id={driver_id!r}: mapping {left_id!r} vs {right_id!r} differ within tolerance on {tolerance} of {len(overlap)} overlapping bar(s).",
                    ))

    return _report(issues, as_of=as_of)
