"""Cross-store consistency reconciliation for the Phase 4A collector
layer (Milestone 10). Unlike `verification.verify_fred_macro_operation`
(which REDERIVES every artifact fresh from caller-declared collector
construction parameters and compares), this module reconciles
ALREADY-STORED durable evidence purely against ITSELF -- it needs no
original request parameters, only a `provider`/`series_id`, exactly the
shape of check an operations/recovery tool runs without necessarily
knowing how a given operation was originally constructed. Mirrors
`market_data.provenance.find_provenance_conflicts`'s own scan-and-report
shape, and reuses `ml.models.ValidationReport` like every other
verification/reconciliation function in this repository.

SCOPE NOTE on "stale checkpoint": Phase 4A deliberately did not add a
separate `CollectorCheckpoint`/`CollectorCheckpointStore` -- Phase 3's
own `checkpoints.py` is shaped for partitioned, multi-batch raw-
ingestion backfills, a materially different resumability concern than a
single per-`operation_id` FRED fetch, which `orchestration.
CollectorOperationStore` already makes fully resumable on its own (see
`docs/milestone10_phase4a_delivery_report.md`, Known Non-Blocking
Limitations). The closest honest analogue to "stale checkpoint" here is
an operation ledger entry that never reached `VERIFICATION_COMPLETED`;
`reconcile_fred_macro_dataset` reports these as `stalled_operation`."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import pandas as pd

from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.orchestration import (
    CollectorOperationStage,
    CollectorOperationStore,
)
from quant_platform.market_data.collectors.response_manifest import compute_raw_content_digest
from quant_platform.market_data.identity import require_tz_aware
from quant_platform.market_data.macro import MacroEventStore
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.provenance import ProvenanceStore
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.ml.models import ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import format_utc_timestamp

__all__ = ["RECONCILIATION_REPORT_SCHEMA_VERSION", "reconcile_fred_macro_dataset"]

RECONCILIATION_REPORT_SCHEMA_VERSION = 1

_ACCEPTABLE_CONTENT_TYPE_SUBSTRINGS = ("json", "csv", "text", "plain")


def _issue(severity: ValidationSeverity, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message)


def _report(issues: list[ValidationIssue], *, as_of: datetime) -> ValidationReport:
    return ValidationReport(schema_version=RECONCILIATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))


def reconcile_fred_macro_dataset(*, repository: MarketDataRepository, cache: RawResponseCache, provider: str, series_id: str, as_of: datetime) -> ValidationReport:
    require_tz_aware(as_of, field_name="as_of")
    issues: list[ValidationIssue] = []
    dataset_key = DatasetKey(dataset_kind=DatasetKind.MACRO_OBSERVATIONS, provider=provider, instrument_id=series_id)

    operation_store = CollectorOperationStore(repository.root)
    by_operation: dict[str, list] = defaultdict(list)  # type: ignore[type-arg]
    for record in operation_store.read_all(dataset_key):
        by_operation[record.operation_id].append(record)

    provenance_store = ProvenanceStore(repository.root)
    macro_store = MacroEventStore(repository.root)
    provenance_records = provenance_store.read_all(dataset_key)
    macro_events = macro_store.read_events(provider, series_id)
    macro_events_by_id = {e.event_id: e for e in macro_events}

    # ---- 1. Per-operation structural checks ----
    for operation_id, records in by_operation.items():
        stage_map = {r.stage: r for r in records}
        latest_stage = records[-1].stage
        if latest_stage is not CollectorOperationStage.VERIFICATION_COMPLETED:
            issues.append(_issue(ValidationSeverity.WARNING, "stalled_operation", f"operation_id={operation_id!r} last reached stage {latest_stage.value!r}, never VERIFICATION_COMPLETED."))
            continue

        response_stage = stage_map.get(CollectorOperationStage.RESPONSE_DOWNLOADED)
        if response_stage is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "missing_raw_response", f"operation_id={operation_id!r} reached VERIFICATION_COMPLETED but has no RESPONSE_DOWNLOADED stage evidence."))
            continue
        response_manifest_id = str(response_stage.stage_evidence["response_manifest_id"])
        response_manifest = cache.read_manifest(response_manifest_id)
        if response_manifest is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "missing_raw_response", f"operation_id={operation_id!r} references response_manifest_id={response_manifest_id!r}, but no cached manifest exists."))
            continue
        try:
            raw_bytes = cache.read_bytes(response_manifest_id, verify=False)
        except Exception as exc:
            issues.append(_issue(ValidationSeverity.CRITICAL, "missing_raw_response", f"operation_id={operation_id!r}: could not read cached raw bytes for response_manifest_id={response_manifest_id!r}: {exc}"))
            continue

        actual_digest = compute_raw_content_digest(raw_bytes)
        if actual_digest != response_manifest.raw_content_digest:
            issues.append(_issue(ValidationSeverity.CRITICAL, "digest_mismatch", f"operation_id={operation_id!r}: cached raw bytes for response_manifest_id={response_manifest_id!r} re-hash to {actual_digest!r}, manifest records {response_manifest.raw_content_digest!r}."))
        if len(raw_bytes) != response_manifest.byte_length:
            issues.append(_issue(ValidationSeverity.CRITICAL, "truncated_payload", f"operation_id={operation_id!r}: cached raw bytes for response_manifest_id={response_manifest_id!r} are {len(raw_bytes)} bytes, manifest records byte_length={response_manifest.byte_length}."))
        if response_manifest.content_type is not None and not any(token in response_manifest.content_type.lower() for token in _ACCEPTABLE_CONTENT_TYPE_SUBSTRINGS):
            issues.append(_issue(ValidationSeverity.WARNING, "unexpected_content_type", f"operation_id={operation_id!r}: response_manifest_id={response_manifest_id!r} has content_type={response_manifest.content_type!r}."))

        request_stage = stage_map.get(CollectorOperationStage.REQUEST_MANIFEST_COMMITTED)
        if request_stage is not None:
            recorded_request_id = str(request_stage.stage_evidence["request_manifest_id"])
            if response_manifest.request_manifest_id != recorded_request_id:
                issues.append(_issue(ValidationSeverity.CRITICAL, "wrong_request_response_linkage", f"operation_id={operation_id!r}: response_manifest_id={response_manifest_id!r} is linked to request_manifest_id={response_manifest.request_manifest_id!r}, ledger recorded {recorded_request_id!r}."))

        source_stage = stage_map.get(CollectorOperationStage.SOURCE_MANIFEST_CREATED)
        if source_stage is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "source_manifest_mismatch", f"operation_id={operation_id!r} reached VERIFICATION_COMPLETED but has no SOURCE_MANIFEST_CREATED stage evidence."))
            continue
        source_manifest_id = str(source_stage.stage_evidence["source_manifest_id"])

        normalize_stage = stage_map.get(CollectorOperationStage.NORMALIZED_RECORDS_PRODUCED)
        if normalize_stage is None:
            continue
        expected_valid_row_count = int(str(normalize_stage.stage_evidence.get("valid_row_count", 0)))

        matching_provenance = [p for p in provenance_records if p.source_manifest_id == source_manifest_id]
        matching_row_indices = [p.source_row_index for p in matching_provenance]
        if len(matching_row_indices) != len(set(matching_row_indices)):
            issues.append(_issue(ValidationSeverity.CRITICAL, "duplicate_observation_coordinate", f"operation_id={operation_id!r}: source_manifest_id={source_manifest_id!r} has {len(matching_provenance)} provenance records over only {len(set(matching_row_indices))} distinct row indices."))
        if len(set(matching_row_indices)) != expected_valid_row_count:
            issues.append(_issue(ValidationSeverity.ERROR, "missing_observation", f"operation_id={operation_id!r}: source_manifest_id={source_manifest_id!r} expected {expected_valid_row_count} valid provenance-backed observations, found {len(set(matching_row_indices))}."))
        for p in matching_provenance:
            if p.event_id not in macro_events_by_id:
                issues.append(_issue(ValidationSeverity.CRITICAL, "repository_record_missing_for_provenance", f"operation_id={operation_id!r}: provenance_id={p.provenance_id!r} references event_id={p.event_id!r}, absent from the macro event store."))

    # ---- 2. repository record without source evidence: every macro event must have a provenance record ----
    provenance_event_ids = {p.event_id for p in provenance_records}
    for event in macro_events:
        if event.event_id not in provenance_event_ids:
            issues.append(_issue(ValidationSeverity.CRITICAL, "repository_record_without_source_evidence", f"event_id={event.event_id!r} (sequence={event.sequence}) has no matching provenance record."))

    # ---- 3. wrong unit mapping: the SAME series must never be ingested under two different unit mappings ----
    distinct_unit_mappings = {p.instrument_mapping_id for p in provenance_records}
    if len(distinct_unit_mappings) > 1:
        issues.append(_issue(ValidationSeverity.CRITICAL, "wrong_unit_mapping", f"series {series_id!r} has provenance evidence of {len(distinct_unit_mappings)} distinct unit mappings: {sorted(distinct_unit_mappings)!r} -- values are not comparable across a mapping change."))

    # ---- 4. conflicting vintage: same (source_event_id, vintage=event_time) recorded with different values ----
    by_source_event_and_vintage: dict[tuple[str, str], set[str]] = defaultdict(set)
    for event in macro_events:
        key = (event.source_event_id or "", event.event_time.isoformat())
        by_source_event_and_vintage[key].add(str(event.value))
    for (source_event_id, vintage), values in by_source_event_and_vintage.items():
        if len(values) > 1:
            issues.append(_issue(ValidationSeverity.CRITICAL, "conflicting_vintage", f"source_event_id={source_event_id!r} at vintage(event_time)={vintage!r} has {len(values)} different recorded values: {sorted(values)!r}."))

    return _report(issues, as_of=as_of)
