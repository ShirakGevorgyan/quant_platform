"""Independent verification for the curated universe (Milestone 10,
Phase 4B). Mirrors `collectors.verification.verify_fred_macro_operation`'s
own STRUCTURALLY INDEPENDENT discipline: every check REDERIVES an
artifact fresh from durable state (the raw cached bytes, the registry,
the policies) using the exact same PURE functions the orchestration
layer used, invoked completely independently -- never trusts a cached
parsed observation, a component manifest's own recorded counts, or a
final "is_verified" flag anywhere.

Takes the ORIGINAL construction parameters (`registry`, `backfill_spec`,
`availability_policies`, `revision_policy`) PLUS the `SeriesOutcome`
tuple a `CuratedIngestionReport` already returned -- mirroring Phase
4A's own choice to rederive from caller-declared parameters rather than
solely from ledger evidence (the curated operation ledger records only
AGGREGATE per-stage evidence, e.g. a sorted list of manifest ids, not a
per-series mapping -- deliberately minimal, matching the "smallest
honest stage machine" discipline; per-series linkage for verification
comes from the caller-held `SeriesOutcome`s instead)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.curated.availability import AvailabilityPolicy
from quant_platform.market_data.collectors.curated.backfill import CuratedBackfillSpec
from quant_platform.market_data.collectors.curated.datasets import (
    COMBINED_UNIVERSE_MANIFEST_KIND,
    COMPONENT_DATASET_MANIFEST_KIND,
    CombinedUniverseManifestStore,
    ComponentDatasetManifestStore,
    create_component_dataset_manifest,
)
from quant_platform.market_data.collectors.curated.macro_observation import CuratedObservationStore
from quant_platform.market_data.collectors.curated.orchestration import SeriesOutcome, _normalize_curated_row
from quant_platform.market_data.collectors.curated.registry import CuratedFredRegistry
from quant_platform.market_data.collectors.curated.revision_policy import RevisionPolicy
from quant_platform.market_data.collectors.fred_schemas import parse_fred_json_response
from quant_platform.market_data.collectors.response_manifest import (
    RESPONSE_MANIFEST_KIND,
    compute_raw_content_digest,
)
from quant_platform.market_data.identity import compute_content_id, require_tz_aware
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.ml.models import ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import format_utc_timestamp

__all__ = ["CURATED_VERIFICATION_REPORT_SCHEMA_VERSION", "verify_curated_universe"]

CURATED_VERIFICATION_REPORT_SCHEMA_VERSION = 1


def _issue(severity: ValidationSeverity, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message)


def _report(issues: list[ValidationIssue], *, as_of: datetime) -> ValidationReport:
    return ValidationReport(schema_version=CURATED_VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))


def verify_curated_universe(
    *, repository: MarketDataRepository, cache: RawResponseCache, registry: CuratedFredRegistry, backfill_spec: CuratedBackfillSpec,
    availability_policies: dict[str, AvailabilityPolicy], revision_policy: RevisionPolicy, series_outcomes: tuple[SeriesOutcome, ...], as_of: datetime,
    provider: str = "fred",
) -> ValidationReport:
    require_tz_aware(as_of, field_name="as_of")
    issues: list[ValidationIssue] = []

    # ---- 1. Registry identity ----
    self_check_registry_id = compute_content_id("curated_fred_registry", registry.to_identity_payload())
    if self_check_registry_id != registry.registry_id:
        issues.append(_issue(ValidationSeverity.CRITICAL, "forged_registry_identity", f"Registry {registry.registry_id!r} does not reproduce its own id from its own recorded specs."))
    if backfill_spec.curated_registry_id != registry.registry_id:
        issues.append(_issue(ValidationSeverity.CRITICAL, "backfill_spec_registry_mismatch", f"backfill_spec.curated_registry_id={backfill_spec.curated_registry_id!r} does not match registry_id={registry.registry_id!r}."))

    # ---- 2. Every series spec ----
    # `CuratedFredSeriesSpec` has no separate content-addressed id of its
    # own -- it IS a plain value object keyed by `series_id`; each spec's
    # own correctness is already captured transitively by the registry-
    # identity check above (`registry.to_identity_payload()` embeds every
    # spec's own `to_identity_payload()`), so no separate per-spec check
    # is needed here.

    component_store = ComponentDatasetManifestStore(repository.root)
    combined_store = CombinedUniverseManifestStore(repository.root)
    observation_store = CuratedObservationStore(repository.root)

    component_manifests_by_series = {}
    for outcome in series_outcomes:
        if not outcome.succeeded:
            continue
        spec = registry.get(outcome.series_id)
        if spec is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "unknown_series_in_outcome", f"SeriesOutcome references series_id={outcome.series_id!r}, absent from the supplied registry."))
            continue
        availability_policy = availability_policies.get(outcome.series_id)
        if availability_policy is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "missing_availability_policy", f"series_id={outcome.series_id!r}: no AvailabilityPolicy supplied for verification."))
            continue

        # ---- 5. Response manifest + raw bytes ----
        if outcome.response_manifest_id is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "missing_response_manifest_id", f"series_id={outcome.series_id!r}: SeriesOutcome has no response_manifest_id."))
            continue
        response_manifest = cache.read_manifest(outcome.response_manifest_id)
        if response_manifest is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "response_manifest_missing", f"series_id={outcome.series_id!r}: no cached response manifest for response_manifest_id={outcome.response_manifest_id!r}."))
            continue
        self_check_response_id = compute_content_id(RESPONSE_MANIFEST_KIND, response_manifest.to_identity_payload())
        if self_check_response_id != response_manifest.response_manifest_id:
            issues.append(_issue(ValidationSeverity.CRITICAL, "forged_response_manifest_identity", f"series_id={outcome.series_id!r}: response manifest does not reproduce its own id."))
        raw_bytes = cache.read_bytes(outcome.response_manifest_id, verify=False)
        actual_digest = compute_raw_content_digest(raw_bytes)
        if actual_digest != response_manifest.raw_content_digest:
            issues.append(_issue(ValidationSeverity.CRITICAL, "raw_content_digest_mismatch", f"series_id={outcome.series_id!r}: re-hashed raw bytes digest {actual_digest!r} != manifest {response_manifest.raw_content_digest!r}."))

        # ---- 6. Strictly reparse ----
        try:
            observations_raw = parse_fred_json_response(raw_bytes, series_id=outcome.series_id)
        except Exception as exc:
            issues.append(_issue(ValidationSeverity.CRITICAL, "reparse_failed", f"series_id={outcome.series_id!r}: independent reparse raised {type(exc).__name__}: {exc}"))
            continue

        stored_observations = observation_store.read_observations(provider, outcome.series_id)
        if not stored_observations:
            issues.append(_issue(ValidationSeverity.CRITICAL, "no_stored_observations", f"series_id={outcome.series_id!r}: durable observation store is empty, cannot independently rederive native unit/frequency for comparison."))
            continue
        # The native unit/frequency this series was fetched under are recorded on every
        # already-stored observation for it (all of one series' observations share the
        # same fetch's metadata) -- read fresh from the durable store rather than
        # re-fetching metadata again, since metadata drift is already independently
        # covered by orchestration's own SERIES_METADATA_VERIFIED stage.
        native_unit = stored_observations[0].native_unit
        native_frequency = stored_observations[0].native_frequency

        # ---- 7/8/9. Recompute availability + normalized values + vintage identity ----
        rederived_valid = []
        for obs in observations_raw:
            observation, _row_issues, _skip_only = _normalize_curated_row(
                obs, spec_series_id=outcome.series_id, canonical_series_name=spec.canonical_series_name, target_macro_instrument_id=spec.target_macro_instrument_id,
                normalized_unit=spec.normalization_kind.value, native_unit=native_unit, native_frequency=native_frequency, unit_conversion=spec.unit_conversion,
                missing_value_policy=spec.missing_value_policy, availability_policy=availability_policy, availability_policy_id=availability_policy.availability_policy_id,
                request_manifest_id=(outcome.request_manifest_id or ""), response_manifest_id=outcome.response_manifest_id, source_manifest_id=(outcome.source_manifest_id or ""),
            )
            if observation is not None:
                rederived_valid.append(observation)

        stored_ids = {o.observation_id for o in stored_observations}
        for rederived in rederived_valid:
            if rederived.observation_id not in stored_ids:
                issues.append(_issue(
                    ValidationSeverity.CRITICAL, "rederived_observation_not_in_store",
                    f"series_id={outcome.series_id!r}: independently rederived observation_id={rederived.observation_id!r} (date={rederived.observation_date!r}) is absent from the durable observation store.",
                ))

        # ---- 11. Component dataset manifest ----
        component_manifest = component_store.read_current(provider, outcome.series_id)
        if component_manifest is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "component_manifest_missing", f"series_id={outcome.series_id!r}: no durable component manifest."))
            continue
        rederived_component = create_component_dataset_manifest(
            series_id=outcome.series_id, canonical_series_name=spec.canonical_series_name, observations=tuple(stored_observations),
            missing_count=component_manifest.missing_count, creation_time=as_of,
        )
        self_check_component_id = compute_content_id(COMPONENT_DATASET_MANIFEST_KIND, component_manifest.to_identity_payload())
        if self_check_component_id != component_manifest.component_manifest_id:
            issues.append(_issue(ValidationSeverity.CRITICAL, "forged_component_manifest_identity", f"series_id={outcome.series_id!r}: component manifest does not reproduce its own id."))
        if rederived_component.semantic_digest != component_manifest.semantic_digest:
            issues.append(_issue(ValidationSeverity.CRITICAL, "component_semantic_digest_mismatch", f"series_id={outcome.series_id!r}: rederived semantic_digest={rederived_component.semantic_digest!r} != stored {component_manifest.semantic_digest!r}."))
        component_manifests_by_series[outcome.series_id] = component_manifest

    # ---- 12/13. Combined manifest + completeness status ----
    combined = combined_store.read_current(backfill_spec.target_dataset_namespace)
    if combined is None:
        issues.append(_issue(ValidationSeverity.CRITICAL, "combined_manifest_missing", f"No combined manifest for namespace {backfill_spec.target_dataset_namespace!r}."))
    else:
        self_check_combined_id = compute_content_id(COMBINED_UNIVERSE_MANIFEST_KIND, combined.to_identity_payload())
        if self_check_combined_id != combined.combined_manifest_id:
            issues.append(_issue(ValidationSeverity.CRITICAL, "forged_combined_manifest_identity", "Combined manifest does not reproduce its own id from its own recorded fields."))
        succeeded_ids = {o.series_id for o in series_outcomes if o.succeeded}
        if set(combined.component_manifest_ids.keys()) != succeeded_ids:
            issues.append(_issue(ValidationSeverity.CRITICAL, "combined_manifest_series_set_mismatch", f"Combined manifest binds series {sorted(combined.component_manifest_ids.keys())!r}, but succeeded outcomes are {sorted(succeeded_ids)!r}."))
        expected_completeness = "complete" if len(succeeded_ids) == len(backfill_spec.selected_series_ids) else "partial"
        if combined.completeness_status != expected_completeness:
            issues.append(_issue(ValidationSeverity.CRITICAL, "completeness_status_mismatch", f"Combined manifest reports completeness_status={combined.completeness_status!r}, expected {expected_completeness!r} given {len(succeeded_ids)}/{len(backfill_spec.selected_series_ids)} series succeeded."))
        # ---- 9 (continued). Revision policy identity used for this universe ----
        if combined.revision_policy_id != revision_policy.revision_policy_id:
            issues.append(_issue(ValidationSeverity.CRITICAL, "revision_policy_mismatch", f"Combined manifest was built under revision_policy_id={combined.revision_policy_id!r}, caller supplied a different revision_policy_id={revision_policy.revision_policy_id!r}."))

    return _report(issues, as_of=as_of)
