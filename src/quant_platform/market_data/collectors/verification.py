"""Independent verification for the Phase 4A collector layer (Milestone
10) -- mirrors `market_data.verification`'s own STRUCTURALLY INDEPENDENT
discipline exactly, reusing its `ValidationIssue`/`ValidationReport`/
`ValidationSeverity` vocabulary directly (the same reuse that module
itself makes of `ml.models`'). Every check below recomputes an
artifact's OWN content id from its own recorded fields (never trusts a
stored digest at face value), re-hashes raw bytes fresh on every read,
STRICTLY REPARSES the FRED response from scratch via `fred_schemas.
parse_fred_json_response`/`parse_fred_csv_response` (never reads back a
previously-parsed in-memory result or a cached `FredSourceAdapter`), and
cross-checks evidence across the request manifest, response manifest,
raw cache, source manifest, provenance store, and macro event store.

"Do not trust cached parsed results" is the organizing principle:
`verify_fred_macro_operation` never calls `fred.load_fred_adapter_from_
cache` (which exists to serve the ORCHESTRATION path, not verification)
-- it independently rebuilds the request manifest via `fred.
build_fred_request_manifest`, re-reads raw bytes via `cache.
RawResponseCache.read_bytes`, reparses them via `fred_schemas.
parse_fred_*_response` directly, and re-normalizes every row via
`macro_normalization.normalize_macro_row` -- the exact same PURE
functions the orchestration layer used, but invoked completely
independently against durable artifacts only, never against an
orchestration-produced report or any other in-memory intermediate."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from quant_platform.market_data.adapters import SourceRowCoordinate
from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.fred import (
    FRED_COLLECTOR_VERSION,
    _observation_to_raw_fields,
    build_fred_request_manifest,
)
from quant_platform.market_data.collectors.fred_schemas import (
    parse_fred_csv_response,
    parse_fred_json_response,
)
from quant_platform.market_data.collectors.macro_normalization import (
    NormalizedMacroObservation,
    UnitMappingSpec,
    fred_timezone_policy_id,
    normalize_macro_row,
)
from quant_platform.market_data.collectors.orchestration import (
    CollectorOperationStage,
    CollectorOperationStore,
)
from quant_platform.market_data.collectors.request_manifest import REQUEST_MANIFEST_KIND, CredentialMode
from quant_platform.market_data.collectors.response_manifest import (
    RESPONSE_MANIFEST_KIND,
    CompletionStatus,
    compute_raw_content_digest,
)
from quant_platform.market_data.identity import compute_content_id, require_tz_aware
from quant_platform.market_data.macro import MacroEventStore, create_macro_event
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.provenance import ProvenanceStore
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.market_data.source_manifests import (
    SOURCE_MANIFEST_KIND,
    RecordKind,
    SourceKind,
    create_source_manifest,
)
from quant_platform.ml.models import ValidationIssue, ValidationReport, ValidationSeverity
from quant_platform.ml.persistence import format_utc_timestamp

__all__ = [
    "VERIFICATION_REPORT_SCHEMA_VERSION",
    "verify_fred_macro_operation",
    "verify_secret_absence",
]

VERIFICATION_REPORT_SCHEMA_VERSION = 1


def _issue(severity: ValidationSeverity, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message)


def _report(issues: list[ValidationIssue], *, as_of: datetime) -> ValidationReport:
    return ValidationReport(schema_version=VERIFICATION_REPORT_SCHEMA_VERSION, issues=tuple(issues), generated_at=format_utc_timestamp(pd.Timestamp(as_of)))


def verify_secret_absence(secret: str, *artifacts: object, as_of: datetime) -> ValidationReport:
    """`artifacts` may be manifests/reports/records/plain strings --
    anything with a JSON-serializable `to_json_dict()` is rendered via
    `repr()` of that dict; everything else via plain `str()`. Every
    rendered form is scanned for `secret` as a literal substring."""
    require_tz_aware(as_of, field_name="as_of")
    issues: list[ValidationIssue] = []
    if not secret:
        return _report(issues, as_of=as_of)
    for index, artifact in enumerate(artifacts):
        text = repr(artifact.to_json_dict()) if hasattr(artifact, "to_json_dict") else str(artifact)
        if secret in text:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "secret_exposed_in_durable_artifact",
                f"artifact at position {index} ({type(artifact).__name__}) contains the secret value as a literal substring.",
            ))
    return _report(issues, as_of=as_of)


def verify_fred_macro_operation(
    *,
    repository: MarketDataRepository,
    cache: RawResponseCache,
    operation_id: str,
    series_id: str,
    provider: str,
    unit_mapping: UnitMappingSpec,
    as_of: datetime,
    observation_start: datetime | None = None,
    observation_end: datetime | None = None,
    response_format: str = "json",
    timeout_policy_id: str = "0" * 64,
    retry_policy_id: str = "0" * 64,
    rate_limit_policy_id: str = "0" * 64,
    credential_mode: CredentialMode = CredentialMode.ANONYMOUS,
    request_time: datetime | None = None,
) -> ValidationReport:
    """Independently reproduces every artifact `run_fred_macro_
    ingestion_operation` claims to have produced for `operation_id`,
    purely from durable state (`repository`, `cache`, the operation
    ledger) plus the caller-declared collector parameters -- and reports
    every mismatch as a `ValidationIssue`. Never raises on a detected
    mismatch (mirrors every other `verify_*` function in this
    repository): a caller wanting fail-closed behavior inspects
    `report.criticals`."""
    require_tz_aware(as_of, field_name="as_of")
    issues: list[ValidationIssue] = []
    dataset_key = DatasetKey(dataset_kind=DatasetKind.MACRO_OBSERVATIONS, provider=provider, instrument_id=series_id)

    history = [r for r in CollectorOperationStore(repository.root).read_all(dataset_key) if r.operation_id == operation_id]
    if not history:
        issues.append(_issue(ValidationSeverity.CRITICAL, "operation_not_found", f"No operation ledger history for operation_id {operation_id!r} under dataset_key {dataset_key.storage_path_parts()!r}."))
        return _report(issues, as_of=as_of)
    by_stage = {r.stage: r for r in history}

    # ---- 1. Rederive request manifest identity ----
    resolved_request_time = request_time if request_time is not None else as_of
    request_manifest = build_fred_request_manifest(
        series_id=series_id, observation_start=observation_start, observation_end=observation_end, response_format=response_format,
        timeout_policy_id=timeout_policy_id, retry_policy_id=retry_policy_id, rate_limit_policy_id=rate_limit_policy_id,
        credential_mode=credential_mode, request_time=resolved_request_time, collector_version=FRED_COLLECTOR_VERSION,
    )
    self_check_request_id = compute_content_id(REQUEST_MANIFEST_KIND, request_manifest.to_identity_payload())
    if self_check_request_id != request_manifest.request_manifest_id:
        issues.append(_issue(ValidationSeverity.CRITICAL, "forged_request_manifest_identity", f"Rederived CollectorRequestManifest {request_manifest.request_manifest_id!r} does not reproduce its own id from its own recorded fields."))

    request_stage = by_stage.get(CollectorOperationStage.REQUEST_MANIFEST_COMMITTED)
    if request_stage is None:
        issues.append(_issue(ValidationSeverity.CRITICAL, "request_manifest_stage_missing", f"Operation {operation_id!r} has no REQUEST_MANIFEST_COMMITTED stage evidence."))
    else:
        recorded_request_manifest_id = str(request_stage.stage_evidence.get("request_manifest_id"))
        if recorded_request_manifest_id != request_manifest.request_manifest_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "request_manifest_identity_mismatch",
                f"Rederived request_manifest_id {request_manifest.request_manifest_id!r} does not match the ledger's recorded {recorded_request_manifest_id!r} -- "
                "the caller-declared collector parameters do not reproduce the operation this ledger actually recorded.",
            ))

    # ---- 2. Response manifest identity + request linkage ----
    response_stage = by_stage.get(CollectorOperationStage.RESPONSE_DOWNLOADED)
    if response_stage is None:
        issues.append(_issue(ValidationSeverity.CRITICAL, "response_manifest_stage_missing", f"Operation {operation_id!r} has no RESPONSE_DOWNLOADED stage evidence."))
        return _report(issues, as_of=as_of)
    response_manifest_id = str(response_stage.stage_evidence.get("response_manifest_id"))
    response_manifest = cache.read_manifest(response_manifest_id)
    if response_manifest is None:
        issues.append(_issue(ValidationSeverity.CRITICAL, "response_manifest_missing", f"No cached response manifest for response_manifest_id {response_manifest_id!r}."))
        return _report(issues, as_of=as_of)

    self_check_response_id = compute_content_id(RESPONSE_MANIFEST_KIND, response_manifest.to_identity_payload())
    if self_check_response_id != response_manifest.response_manifest_id:
        issues.append(_issue(ValidationSeverity.CRITICAL, "forged_response_manifest_identity", f"CollectorResponseManifest {response_manifest.response_manifest_id!r} does not reproduce its own id from its own recorded fields -- forged or tampered."))
    if response_manifest.request_manifest_id != request_manifest.request_manifest_id:
        issues.append(_issue(ValidationSeverity.CRITICAL, "response_request_linkage_mismatch", f"CollectorResponseManifest {response_manifest.response_manifest_id!r} is linked to request_manifest_id {response_manifest.request_manifest_id!r}, not the rederived {request_manifest.request_manifest_id!r}."))

    # ---- 3. Re-hash raw bytes independently ----
    raw_bytes = cache.read_bytes(response_manifest_id, verify=False)
    actual_digest = compute_raw_content_digest(raw_bytes)
    if actual_digest != response_manifest.raw_content_digest:
        issues.append(_issue(ValidationSeverity.CRITICAL, "raw_content_digest_mismatch", f"Re-hashed raw bytes for response_manifest_id {response_manifest_id!r} produce digest {actual_digest!r}, but the manifest records {response_manifest.raw_content_digest!r}."))
    if len(raw_bytes) != response_manifest.byte_length:
        issues.append(_issue(ValidationSeverity.CRITICAL, "byte_length_mismatch", f"Cached raw bytes for response_manifest_id {response_manifest_id!r} are {len(raw_bytes)} bytes, manifest records byte_length={response_manifest.byte_length}."))

    # ---- 4. Validate response metadata ----
    if response_manifest.completion_status is not CompletionStatus.COMPLETE:
        issues.append(_issue(ValidationSeverity.CRITICAL, "incomplete_response", f"response_manifest_id {response_manifest_id!r} has completion_status={response_manifest.completion_status.value!r}, expected COMPLETE."))
    if not (200 <= response_manifest.http_status < 300):
        issues.append(_issue(ValidationSeverity.ERROR, "non_success_http_status", f"response_manifest_id {response_manifest_id!r} recorded http_status={response_manifest.http_status}."))

    # ---- 5. Strictly reparse (never trust a cached parsed result) ----
    try:
        if response_format == "json":
            observations = parse_fred_json_response(raw_bytes, series_id=series_id)
        elif response_format == "csv":
            observations = parse_fred_csv_response(raw_bytes, series_id=series_id)
        else:
            issues.append(_issue(ValidationSeverity.CRITICAL, "unsupported_response_format", f"response_format {response_format!r} is not 'json' or 'csv'."))
            return _report(issues, as_of=as_of)
    except Exception as exc:
        issues.append(_issue(ValidationSeverity.CRITICAL, "reparse_failed", f"Independent reparse of response_manifest_id {response_manifest_id!r} raised {type(exc).__name__}: {exc}"))
        return _report(issues, as_of=as_of)

    # ---- 6. Rederive source manifest ----
    timezone_policy_id = fred_timezone_policy_id()
    source_manifest = create_source_manifest(
        source_name=f"fred:{series_id}", source_kind=SourceKind.FRED_API, source_schema_version=1, record_kind=RecordKind.MACRO_OBSERVATION,
        source_label=f"fred:{series_id}:{response_manifest_id[:16]}", content_digest=response_manifest.raw_content_digest,
        byte_size=response_manifest.byte_length, encoding=response_manifest.encoding or "utf-8", instrument_mapping_id=unit_mapping.unit_mapping_id,
        timezone_policy_id=timezone_policy_id, unit_normalization_version=unit_mapping.unit_mapping_version, creation_time=as_of,
        expected_start=observation_start, expected_end=observation_end,
    )
    self_check_source_id = compute_content_id(SOURCE_MANIFEST_KIND, source_manifest.to_identity_payload())
    if self_check_source_id != source_manifest.source_manifest_id:
        issues.append(_issue(ValidationSeverity.CRITICAL, "forged_source_manifest_identity", f"Rederived SourceManifest {source_manifest.source_manifest_id!r} does not reproduce its own id from its own recorded fields."))

    source_stage = by_stage.get(CollectorOperationStage.SOURCE_MANIFEST_CREATED)
    if source_stage is None:
        issues.append(_issue(ValidationSeverity.CRITICAL, "source_manifest_stage_missing", f"Operation {operation_id!r} has no SOURCE_MANIFEST_CREATED stage evidence."))
    else:
        recorded_source_manifest_id = str(source_stage.stage_evidence.get("source_manifest_id"))
        if recorded_source_manifest_id != source_manifest.source_manifest_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "source_manifest_identity_mismatch",
                f"Rederived source_manifest_id {source_manifest.source_manifest_id!r} does not match the ledger's recorded {recorded_source_manifest_id!r}.",
            ))

    # ---- 7. Rederive normalized observations ----
    valid_rows: list[tuple[int, NormalizedMacroObservation]] = []
    for obs in observations:
        raw_fields = _observation_to_raw_fields(obs)
        normalized, row_issue_codes = normalize_macro_row(raw_fields, series_id=series_id, unit_mapping=unit_mapping)
        if not row_issue_codes:
            assert normalized is not None
            valid_rows.append((obs.row_index, normalized))

    # ---- 8/9. Verify provenance and repository dataset for every rederived valid row ----
    normalize_stage = by_stage.get(CollectorOperationStage.NORMALIZED_RECORDS_PRODUCED)
    if normalize_stage is None:
        issues.append(_issue(ValidationSeverity.CRITICAL, "normalized_records_stage_missing", f"Operation {operation_id!r} has no NORMALIZED_RECORDS_PRODUCED stage evidence."))
        return _report(issues, as_of=as_of)
    sequence_start_raw = normalize_stage.stage_evidence.get("macro_sequence_start")
    if sequence_start_raw is None:
        issues.append(_issue(ValidationSeverity.CRITICAL, "sequence_start_missing", f"Operation {operation_id!r}'s NORMALIZED_RECORDS_PRODUCED evidence has no macro_sequence_start."))
        return _report(issues, as_of=as_of)
    sequence_start = int(str(sequence_start_raw))

    provenance_store = ProvenanceStore(repository.root)
    macro_store = MacroEventStore(repository.root)
    existing_event_ids = {e.event_id for e in macro_store.read_events(provider, series_id)}

    for index, (row_index, normalized) in enumerate(valid_rows):
        expected_event = create_macro_event(
            series_id=series_id, provider=provider, event_time=normalized.event_time, sequence=sequence_start + index,
            value=normalized.value, unit=normalized.unit.value, source_event_id=normalized.source_event_id,
        )
        provenance_record = provenance_store.read_by_source_coordinate(dataset_key, SourceRowCoordinate(source_manifest_id=source_manifest.source_manifest_id, row_index=row_index))
        if provenance_record is None:
            issues.append(_issue(ValidationSeverity.CRITICAL, "missing_provenance", f"No provenance record for (source_manifest_id={source_manifest.source_manifest_id!r}, row_index={row_index})."))
            continue
        if provenance_record.event_id != expected_event.event_id:
            issues.append(_issue(
                ValidationSeverity.CRITICAL, "provenance_event_id_mismatch",
                f"Provenance for row_index={row_index} is bound to event_id {provenance_record.event_id!r}, but independent rederivation expects {expected_event.event_id!r}.",
            ))
            continue
        if expected_event.event_id not in existing_event_ids:
            issues.append(_issue(ValidationSeverity.CRITICAL, "repository_record_missing", f"Rederived event_id {expected_event.event_id!r} for row_index={row_index} has provenance but is absent from the macro event store."))

    return _report(issues, as_of=as_of)
