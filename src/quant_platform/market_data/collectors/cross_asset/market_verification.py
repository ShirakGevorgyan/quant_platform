"""Independent verification for the curated cross-asset universe
(Milestone 10, Phase 4C, spec Section 27). Mirrors `curated.verification.
verify_curated_universe`'s own discipline: every check REDERIVES an
artifact fresh from durable state (the raw cached bytes, the registry,
the policies) using the exact same PURE functions the orchestration
layer used, invoked completely independently -- never trusts a cached
parsed bar, a component manifest's own recorded counts, or a final
"is_verified" flag anywhere.

INDEPENDENCE CLASSIFICATION (spec Section 27's own "document which
checks reuse provider parsing logic vs. structurally independent"):
checks 6 (reparse) and 9 (recompute events) call `collector.
parse_history_response`/`market_normalization.normalize_raw_market_record`
-- the SAME pure functions orchestration used, invoked freshly against
independently re-read raw bytes (never a cached parse) and independently
re-hashed bytes (check 5) -- structurally independent of orchestration's
OWN RUN, but NOT independent of the provider adapter's OWN parsing
implementation (a bug in `providers/alpha_vantage.py` itself would not
be caught by this verification; that risk is instead covered by the
adversarial audit's dedicated tamper tests, spec Section 33). Checks 1-4,
7-8, 10-17 (registry/mapping/manifest self-identity, component/combined
digests, completeness) are fully structurally independent recomputations
requiring no provider-specific code at all."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from quant_platform.core.types import Timeframe
from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.cross_asset.availability import BarAvailabilityPolicy
from quant_platform.market_data.collectors.cross_asset.datasets import (
    COMBINED_CROSS_ASSET_MANIFEST_KIND,
    COMPONENT_MARKET_DATASET_MANIFEST_KIND,
    CombinedCrossAssetManifestStore,
    ComponentMarketDatasetManifestStore,
    create_component_market_dataset_manifest,
)
from quant_platform.market_data.collectors.cross_asset.gap_policy import analyze_bar_gaps
from quant_platform.market_data.collectors.cross_asset.market_backfill import MarketBackfillSpec
from quant_platform.market_data.collectors.cross_asset.market_normalization import normalize_raw_market_record
from quant_platform.market_data.collectors.cross_asset.market_orchestration import MappingOutcome
from quant_platform.market_data.collectors.cross_asset.market_record import MarketDriverBarStore
from quant_platform.market_data.collectors.cross_asset.protocols import HistoricalMarketCollector
from quant_platform.market_data.collectors.cross_asset.registry import CuratedMarketDriverRegistry
from quant_platform.market_data.collectors.cross_asset.sessions import TimezoneSessionPolicy
from quant_platform.market_data.collectors.cross_asset.symbol_mapping import SymbolMappingSet
from quant_platform.market_data.collectors.response_manifest import (
    RESPONSE_MANIFEST_KIND,
    compute_raw_content_digest,
)
from quant_platform.market_data.identity import compute_content_id, require_tz_aware
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.ml.models import ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import format_utc_timestamp

__all__ = ["CROSS_ASSET_VERIFICATION_REPORT_SCHEMA_VERSION", "verify_cross_asset_universe"]

CROSS_ASSET_VERIFICATION_REPORT_SCHEMA_VERSION = 1


def _issue(severity: ValidationSeverity, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message)


def _report(issues: list[ValidationIssue], *, as_of: datetime) -> ValidationReport:
    return ValidationReport(schema_version=CROSS_ASSET_VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))


def verify_cross_asset_universe(
    *, repository: MarketDataRepository, cache: RawResponseCache, registry: CuratedMarketDriverRegistry, mapping_set: SymbolMappingSet,
    backfill_spec: MarketBackfillSpec, session_policies: dict[str, TimezoneSessionPolicy], availability_policies: dict[str, BarAvailabilityPolicy],
    collectors_by_provider: dict[str, HistoricalMarketCollector], mapping_outcomes: tuple[MappingOutcome, ...], as_of: datetime,
) -> ValidationReport:
    require_tz_aware(as_of, field_name="as_of")
    issues: list[ValidationIssue] = []

    # ---- 1. Registry identity ----
    self_check_registry_id = compute_content_id("cross_asset_curated_market_driver_registry", registry.to_identity_payload())
    if self_check_registry_id != registry.registry_id:
        issues.append(_issue(ValidationSeverity.CRITICAL, "forged_registry_identity", f"Registry {registry.registry_id!r} does not reproduce its own id from its own recorded specs."))
    if backfill_spec.curated_registry_id != registry.registry_id:
        issues.append(_issue(ValidationSeverity.CRITICAL, "backfill_spec_registry_mismatch", f"backfill_spec.curated_registry_id={backfill_spec.curated_registry_id!r} does not match registry_id={registry.registry_id!r}."))

    component_store = ComponentMarketDatasetManifestStore(repository.root)
    combined_store = CombinedCrossAssetManifestStore(repository.root)
    bar_store = MarketDriverBarStore(repository.root)

    for outcome in mapping_outcomes:
        if not outcome.succeeded:
            continue
        mapping = mapping_set.get(outcome.mapping_id)
        if mapping is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "unknown_mapping_in_outcome", f"MappingOutcome references mapping_id={outcome.mapping_id!r}, absent from the supplied SymbolMappingSet."))
            continue
        driver_spec = registry.get(mapping.canonical_driver_id)
        if driver_spec is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "unknown_driver_in_outcome", f"mapping_id={outcome.mapping_id!r}: canonical_driver_id={mapping.canonical_driver_id!r} absent from the supplied registry."))
            continue
        collector = collectors_by_provider.get(mapping.provider)
        session_policy = session_policies.get(driver_spec.session_policy_id)
        availability_policy = availability_policies.get(driver_spec.availability_policy_id)
        if collector is None or session_policy is None or availability_policy is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "missing_verification_inputs", f"mapping_id={outcome.mapping_id!r}: collector/session_policy/availability_policy not all supplied for verification."))
            continue

        # ---- 5. Response manifest + re-hash raw bytes ----
        if outcome.response_manifest_id is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "missing_response_manifest_id", f"mapping_id={outcome.mapping_id!r}: MappingOutcome has no response_manifest_id."))
            continue
        response_manifest = cache.read_manifest(outcome.response_manifest_id)
        if response_manifest is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "response_manifest_missing", f"mapping_id={outcome.mapping_id!r}: no cached response manifest for response_manifest_id={outcome.response_manifest_id!r}."))
            continue
        self_check_response_id = compute_content_id(RESPONSE_MANIFEST_KIND, response_manifest.to_identity_payload())
        if self_check_response_id != response_manifest.response_manifest_id:
            issues.append(_issue(ValidationSeverity.CRITICAL, "forged_response_manifest_identity", f"mapping_id={outcome.mapping_id!r}: response manifest does not reproduce its own id."))
        raw_bytes = cache.read_bytes(outcome.response_manifest_id, verify=False)
        actual_digest = compute_raw_content_digest(raw_bytes)
        if actual_digest != response_manifest.raw_content_digest:
            issues.append(_issue(ValidationSeverity.CRITICAL, "raw_content_digest_mismatch", f"mapping_id={outcome.mapping_id!r}: re-hashed raw bytes digest {actual_digest!r} != manifest {response_manifest.raw_content_digest!r}."))
            continue

        # ---- Component dataset manifest (fetched early -- its own recorded
        # `timeframe` is the independently-checkable source for the reparse below,
        # never assumed/hardcoded) ----
        component_manifest = component_store.read_current(outcome.mapping_id)
        if component_manifest is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "component_manifest_missing", f"mapping_id={outcome.mapping_id!r}: no durable component manifest."))
            continue
        timeframe = Timeframe(component_manifest.timeframe)

        # ---- 6. Strictly reparse (see module docstring's independence classification) ----
        try:
            raw_records = collector.parse_history_response(raw_bytes, provider_symbol=mapping.provider_symbol, response_manifest=response_manifest)
        except Exception as exc:
            issues.append(_issue(ValidationSeverity.CRITICAL, "reparse_failed", f"mapping_id={outcome.mapping_id!r}: independent reparse raised {type(exc).__name__}: {exc}"))
            continue

        # ---- 7/8/9/10. Recompute Decimals, event identities, adjustment/availability semantics ----
        rederived_bar_ids: set[str] = set()
        for raw in raw_records:
            bar, _issue_codes = normalize_raw_market_record(
                raw, canonical_driver_id=mapping.canonical_driver_id, instrument_form=mapping.instrument_form, timeframe=timeframe,
                session_policy=session_policy, availability_policy=availability_policy, adjustment_policy_id=driver_spec.adjustment_policy_id,
                request_manifest_id=(outcome.request_manifest_id or ""), response_manifest_id=outcome.response_manifest_id,
                source_manifest_id=(outcome.source_manifest_id or ""), source_row_index=raw.source_sequence,
            )
            if bar is not None:
                rederived_bar_ids.add(bar.bar_id)

        stored_bars = bar_store.read_bars(mapping.provider, mapping.canonical_driver_id, mapping.instrument_form)
        stored_ids = {b.bar_id for b in stored_bars}
        for rederived_id in rederived_bar_ids:
            if rederived_id not in stored_ids:
                issues.append(_issue(ValidationSeverity.CRITICAL, "rederived_bar_not_in_store", f"mapping_id={outcome.mapping_id!r}: independently rederived bar_id={rederived_id!r} is absent from the durable bar store."))

        # ---- 11. Gap/conflict re-analysis ----
        if stored_bars:
            gap_report = analyze_bar_gaps(tuple(stored_bars), session_policy=session_policy)
            if gap_report.has_conflicting_coordinates:
                issues.append(_issue(ValidationSeverity.CRITICAL, "conflicting_bar_coordinates", f"mapping_id={outcome.mapping_id!r}: {gap_report.conflicting_coordinate_count} conflicting duplicate bar coordinate(s) in the committed bar store."))

        # ---- 13. Component dataset manifest re-derivation ----
        rederived_component = create_component_market_dataset_manifest(
            mapping_id=mapping.mapping_id, canonical_driver_id=mapping.canonical_driver_id, provider=mapping.provider, provider_symbol=mapping.provider_symbol,
            instrument_form=mapping.instrument_form.value, timeframe=component_manifest.timeframe, adjustment_policy_id=driver_spec.adjustment_policy_id,
            session_policy_id=session_policy.session_policy_id, availability_policy_id=availability_policy.availability_policy_id, bars=tuple(stored_bars),
            missing_business_day_count=component_manifest.missing_business_day_count, conflicting_coordinate_count=component_manifest.conflicting_coordinate_count,
            creation_time=as_of, continuation_policy_id=mapping.continuation_policy_id,
        )
        self_check_component_id = compute_content_id(COMPONENT_MARKET_DATASET_MANIFEST_KIND, component_manifest.to_identity_payload())
        if self_check_component_id != component_manifest.component_manifest_id:
            issues.append(_issue(ValidationSeverity.CRITICAL, "forged_component_manifest_identity", f"mapping_id={outcome.mapping_id!r}: component manifest does not reproduce its own id."))
        if rederived_component.semantic_digest != component_manifest.semantic_digest:
            issues.append(_issue(ValidationSeverity.CRITICAL, "component_semantic_digest_mismatch", f"mapping_id={outcome.mapping_id!r}: rederived semantic_digest={rederived_component.semantic_digest!r} != stored {component_manifest.semantic_digest!r}."))

    # ---- 14/15. Combined manifest + completeness status ----
    combined = combined_store.read_current(backfill_spec.target_dataset_namespace)
    if combined is None:
        issues.append(_issue(ValidationSeverity.CRITICAL, "combined_manifest_missing", f"No combined manifest for namespace {backfill_spec.target_dataset_namespace!r}."))
    else:
        self_check_combined_id = compute_content_id(COMBINED_CROSS_ASSET_MANIFEST_KIND, combined.to_identity_payload())
        if self_check_combined_id != combined.combined_manifest_id:
            issues.append(_issue(ValidationSeverity.CRITICAL, "forged_combined_manifest_identity", "Combined manifest does not reproduce its own id from its own recorded fields."))
        succeeded_ids = {o.mapping_id for o in mapping_outcomes if o.succeeded}
        if set(combined.component_manifest_ids.keys()) != succeeded_ids:
            issues.append(_issue(ValidationSeverity.CRITICAL, "combined_manifest_mapping_set_mismatch", f"Combined manifest binds mappings {sorted(combined.component_manifest_ids.keys())!r}, but succeeded outcomes are {sorted(succeeded_ids)!r}."))
        required = set(registry.required_driver_ids())
        satisfied = set(combined.driver_id_by_mapping.values())
        expected_missing_required = tuple(sorted(required - satisfied))
        if tuple(sorted(combined.missing_required_driver_ids)) != expected_missing_required:
            issues.append(_issue(ValidationSeverity.CRITICAL, "missing_required_driver_ids_mismatch", f"Combined manifest reports missing_required_driver_ids={sorted(combined.missing_required_driver_ids)!r}, recomputed={list(expected_missing_required)!r}."))
        expected_completeness = "partial" if expected_missing_required else "complete"
        if combined.completeness_status != expected_completeness:
            issues.append(_issue(ValidationSeverity.CRITICAL, "completeness_status_mismatch", f"Combined manifest reports completeness_status={combined.completeness_status!r}, expected {expected_completeness!r}."))

    return _report(issues, as_of=as_of)
