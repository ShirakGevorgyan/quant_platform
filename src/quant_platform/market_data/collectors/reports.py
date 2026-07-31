"""Deterministic reporting for the Phase 4A collector layer (Milestone
10) -- mirrors `market_data.reports`'s own convention exactly: every
function here wraps an object this package already produces (a
`CollectorRequestManifest`, `CollectorResponseManifest`, tuple of
`RetryAttemptRecord`, `TokenBucketState`, or `CollectorIngestionReport`)
into a stable, deterministic dict, never re-deriving new facts of its
own. `generate_quarantine_summary_report`/`generate_provenance_summary_
report`/`generate_reconciliation_report`/`generate_verification_report`
are re-exported UNCHANGED from `market_data.reports`: those four are
already `DatasetKey`-generic (no candle/tick/FRED-specific logic), so
Phase 3's own implementations serve macro datasets exactly as they are.

SECRET-FREE BY CONSTRUCTION: every dict built here is assembled purely
from `to_json_dict()`-style calls on objects that are themselves
structurally secret-free (`CollectorRequestManifest.__post_init__`
rejects secret-shaped query params/headers; `CollectorResponseManifest`
only ever carries an ALLOWLISTED header subset; `RetryAttemptRecord` has
no field for a raw credential). No function in this module ever accepts
an `api_key`/raw credential as an argument -- there is nothing here that
COULD leak one."""

from __future__ import annotations

from quant_platform.market_data.collectors.orchestration import CollectorIngestionReport
from quant_platform.market_data.collectors.rate_limit import (
    RateLimitPolicy,
    TokenBucketState,
    seconds_until_available,
)
from quant_platform.market_data.collectors.request_manifest import CollectorRequestManifest
from quant_platform.market_data.collectors.response_manifest import CollectorResponseManifest
from quant_platform.market_data.collectors.retry import RetryAttemptRecord
from quant_platform.market_data.reports import (
    generate_provenance_summary_report,
    generate_quarantine_summary_report,
    generate_reconciliation_report,
    generate_verification_report,
)

__all__ = [
    "generate_fred_ingestion_operation_report",
    "generate_provenance_summary_report",
    "generate_quarantine_summary_report",
    "generate_rate_limit_status_report",
    "generate_reconciliation_report",
    "generate_replay_comparison_report",
    "generate_request_manifest_report",
    "generate_response_manifest_report",
    "generate_retry_attempts_report",
    "generate_verification_report",
]


def generate_request_manifest_report(manifest: CollectorRequestManifest) -> dict[str, object]:
    return {
        "request_manifest_id": manifest.request_manifest_id, "collector_name": manifest.collector_name,
        "collector_version": manifest.collector_version, "endpoint_host": manifest.endpoint_host, "endpoint_path": manifest.endpoint_path,
        "requested_series_or_dataset": manifest.requested_series_or_dataset,
        "requested_interval_start": (None if manifest.requested_interval_start is None else manifest.requested_interval_start.isoformat()),
        "requested_interval_end": (None if manifest.requested_interval_end is None else manifest.requested_interval_end.isoformat()),
        "response_format": manifest.response_format, "credential_mode": manifest.credential_mode.value,
        "canonical_query_param_keys": sorted(manifest.canonical_query_params.keys()),
        "canonical_header_keys": sorted(manifest.canonical_headers.keys()),
    }


def generate_response_manifest_report(manifest: CollectorResponseManifest) -> dict[str, object]:
    return {
        "response_manifest_id": manifest.response_manifest_id, "request_manifest_id": manifest.request_manifest_id,
        "http_status": manifest.http_status, "byte_length": manifest.byte_length, "raw_content_digest": manifest.raw_content_digest,
        "content_type": manifest.content_type, "encoding": manifest.encoding, "completion_status": manifest.completion_status.value,
        "transport_attempt_count": manifest.transport_attempt_count, "canonical_selected_header_keys": sorted(manifest.canonical_selected_headers.keys()),
    }


def generate_retry_attempts_report(attempts: tuple[RetryAttemptRecord, ...]) -> dict[str, object]:
    outcome_counts: dict[str, int] = {}
    for attempt in attempts:
        outcome_counts[attempt.outcome] = outcome_counts.get(attempt.outcome, 0) + 1
    return {"attempt_count": len(attempts), "outcome_counts": outcome_counts, "attempts": [a.to_json_dict() for a in attempts]}


def generate_rate_limit_status_report(*, state: TokenBucketState, policy: RateLimitPolicy) -> dict[str, object]:
    return {
        "rate_limit_policy_id": policy.rate_limit_policy_id, "max_tokens": str(policy.max_tokens),
        "refill_rate_per_second": str(policy.refill_rate_per_second), "current_tokens": str(state.tokens),
        "last_refill_time": state.last_refill_time.isoformat(), "seconds_until_one_token_available": str(seconds_until_available(state, policy)),
    }


def generate_fred_ingestion_operation_report(report: CollectorIngestionReport) -> dict[str, object]:
    return report.to_json_dict()


def generate_replay_comparison_report(*, original: CollectorIngestionReport, replayed: CollectorIngestionReport) -> dict[str, object]:
    """`identical` is true iff the two reports agree on every field
    offline replay promises to reproduce -- `operation_id`/`dataset_key`/
    `is_dry_run`/`fetch_mode` are deliberately EXCLUDED (a replay
    legitimately targets a different `operation_id` and reads via
    `FetchMode.CACHED_REPLAY` rather than the original's `FetchMode.
    FRESH`; see `orchestration.py`'s own module docstring)."""
    compared_fields = (
        "request_manifest_id", "response_manifest_id", "source_manifest_id", "parsed_row_count", "valid_row_count",
        "quarantined_row_count", "quarantine_issue_counts", "committed_event_count", "normalized_events_digest",
    )
    differences = {field: (getattr(original, field), getattr(replayed, field)) for field in compared_fields if getattr(original, field) != getattr(replayed, field)}
    return {
        "identical": not differences,
        "differences": {k: {"original": v[0], "replayed": v[1]} for k, v in differences.items()},
        "original": original.to_json_dict(), "replayed": replayed.to_json_dict(),
    }
