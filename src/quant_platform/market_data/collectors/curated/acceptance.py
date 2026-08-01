"""Opt-in REAL-FRED acceptance workflow (Milestone 10, Phase 4B) -- the
ONLY place anywhere in `collectors/` (including `collectors/curated/`)
that reads an environment variable. Every other module in this package
requires a credential to be passed explicitly by the caller; this
module exists purely so a human running the acceptance workflow does
not need to write a real key into any file, CLI argument history, or
example config -- consistent with "Do not automatically read arbitrary
environment variables in pure domain code": `resolve_fred_api_key_from_
environment` is not domain code, it is this ONE, explicitly-named,
narrowly-scoped opt-in entry point.

Disabled by default: `run_real_fred_acceptance_workflow` requires an
explicit `api_key: str` argument -- nothing in this module implicitly
falls back to "just try it anyway." A caller (a pytest test, a manual
script) that finds no key via `resolve_fred_api_key_from_environment`
is expected to skip cleanly with a precise reason and make ZERO network
calls, never treating a missing credential as an application failure."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.curated.availability import AvailabilityPolicy
from quant_platform.market_data.collectors.curated.backfill import CachePolicy, create_curated_backfill_spec
from quant_platform.market_data.collectors.curated.orchestration import (
    CuratedIngestionReport,
    run_curated_backfill_operation,
)
from quant_platform.market_data.collectors.curated.reconciliation import reconcile_curated_universe
from quant_platform.market_data.collectors.curated.registry import (
    create_curated_registry,
    default_core_series_specs,
)
from quant_platform.market_data.collectors.curated.revision_policy import RevisionPolicy
from quant_platform.market_data.collectors.curated.verification import verify_curated_universe
from quant_platform.market_data.collectors.protocols import HistoricalHttpTransport
from quant_platform.market_data.collectors.rate_limit import RateLimitPolicy, TokenBucketState
from quant_platform.market_data.collectors.request_manifest import CredentialMode
from quant_platform.market_data.collectors.retry import RetryPolicy
from quant_platform.market_data.identity import require_tz_aware
from quant_platform.market_data.repository import MarketDataRepository

__all__ = [
    "FRED_API_KEY_ENV_VAR",
    "RedactedAcceptanceReport",
    "resolve_fred_api_key_from_environment",
    "run_real_fred_acceptance_workflow",
]

FRED_API_KEY_ENV_VAR = "FRED_API_KEY"


def resolve_fred_api_key_from_environment(env: Mapping[str, str] | None = None) -> str | None:
    """Returns `None` (never raises, never falls back to a placeholder)
    when `FRED_API_KEY` is absent or blank -- the caller decides what
    "no key" means (skip cleanly, in every sanctioned use of this
    function)."""
    source = env if env is not None else os.environ
    value = source.get(FRED_API_KEY_ENV_VAR)
    if value is None or not value.strip():
        return None
    return value.strip()


@dataclass(frozen=True, slots=True)
class RedactedAcceptanceReport:
    ran: bool
    series_checked: tuple[str, ...]
    backfill_stage: str
    backfill_completeness_status: str
    committed_observation_counts: dict[str, int]
    reconciliation_critical_count: int
    verification_critical_count: int
    replay_identical: bool
    notes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        """Deliberately narrow: no request URL, no manifest object, no
        header, no raw response -- only aggregate counts and booleans.
        The API key CANNOT appear here even by accident, since this
        function's own return type has no field capable of holding one."""
        return {
            "ran": self.ran, "series_checked": list(self.series_checked), "backfill_stage": self.backfill_stage,
            "backfill_completeness_status": self.backfill_completeness_status, "committed_observation_counts": dict(self.committed_observation_counts),
            "reconciliation_critical_count": self.reconciliation_critical_count, "verification_critical_count": self.verification_critical_count,
            "replay_identical": self.replay_identical, "notes": list(self.notes),
        }


def run_real_fred_acceptance_workflow(
    *, api_key: str, repository: MarketDataRepository, cache: RawResponseCache, transport: HistoricalHttpTransport, retry_policy: RetryPolicy,
    rate_limit_policy: RateLimitPolicy, rate_limit_state: TokenBucketState, revision_policy: RevisionPolicy, availability_policies: dict[str, AvailabilityPolicy],
    observation_start: datetime, observation_end: datetime, operation_time: datetime, operation_id: str = "fred-acceptance",
    target_dataset_namespace: str = "fred_acceptance_workflow", page_size: int = 100,
) -> RedactedAcceptanceReport:
    """Requests at least the four CORE series over a caller-BOUNDED
    interval (never "since inception"), with a small `page_size`. Runs
    the full pipeline (metadata verify -> backfill -> component/combined
    datasets -> reconciliation -> verification), then a SECOND,
    `FetchMode`-equivalent CACHED_REPLAY pass with a `ForbiddenTransport`
    to prove offline replay makes zero network calls, comparing the
    two runs' semantic results."""
    if not api_key or not api_key.strip():
        raise ValueError("run_real_fred_acceptance_workflow requires a non-empty api_key -- callers must resolve one (e.g. via resolve_fred_api_key_from_environment) before calling this")
    require_tz_aware(observation_start, field_name="observation_start")
    require_tz_aware(observation_end, field_name="observation_end")
    require_tz_aware(operation_time, field_name="operation_time")

    core_specs = default_core_series_specs(
        registry_version=1, revision_policy_id=revision_policy.revision_policy_id,
        release_availability_policy_id_daily=next(iter(availability_policies.values())).availability_policy_id,
        release_availability_policy_id_monthly=next(iter(availability_policies.values())).availability_policy_id, default_observation_start=observation_start,
    )
    registry = create_curated_registry(registry_version=1, specs=core_specs)
    backfill_spec = create_curated_backfill_spec(
        registry=registry, selected_series_ids=tuple(sorted(s.series_id for s in core_specs)), observation_start=observation_start,
        observation_end=observation_end, revision_policy_id=revision_policy.revision_policy_id, target_dataset_namespace=target_dataset_namespace,
        page_size=page_size, cache_policy=CachePolicy.PREFER_CACHE, fail_fast=False,
    )

    original_report = run_curated_backfill_operation(
        repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies,
        revision_policy=revision_policy, operation_id=operation_id, operation_time=operation_time, transport=transport, api_key=api_key,
        retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, rate_limit_state=rate_limit_state, credential_mode=CredentialMode.API_KEY,
    )

    reconciliation_result = reconcile_curated_universe(repository=repository, registry=registry, target_dataset_namespace=target_dataset_namespace, as_of=operation_time)
    verification_result = verify_curated_universe(
        repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies,
        revision_policy=revision_policy, series_outcomes=original_report.series_outcomes, as_of=operation_time,
    )

    class _ForbiddenTransport:
        def get(self, _request: object) -> object:
            raise AssertionError("offline replay must perform zero network calls")

    replay_report = run_curated_backfill_operation(
        repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies,
        revision_policy=revision_policy, operation_id=operation_id, operation_time=operation_time, transport=_ForbiddenTransport(),  # type: ignore[arg-type]
        credential_mode=CredentialMode.API_KEY,
    )
    replay_identical = _reports_semantically_identical(original_report, replay_report)

    committed_counts = {o.series_id: o.committed_observation_count for o in original_report.series_outcomes}
    notes = tuple(f"series {o.series_id} failed: {o.failure_reason}" for o in original_report.series_outcomes if not o.succeeded)
    return RedactedAcceptanceReport(
        ran=True, series_checked=backfill_spec.selected_series_ids, backfill_stage=original_report.stage.value,
        backfill_completeness_status=original_report.completeness_status, committed_observation_counts=committed_counts,
        reconciliation_critical_count=len(reconciliation_result.criticals), verification_critical_count=len(verification_result.criticals),
        replay_identical=replay_identical, notes=notes,
    )


def _reports_semantically_identical(original: CuratedIngestionReport, replayed: CuratedIngestionReport) -> bool:
    if original.completeness_status != replayed.completeness_status:
        return False
    if original.combined_manifest_id != replayed.combined_manifest_id:
        return False
    original_by_series = {o.series_id: o for o in original.series_outcomes}
    replayed_by_series = {o.series_id: o for o in replayed.series_outcomes}
    if set(original_by_series) != set(replayed_by_series):
        return False
    return all(
        original_by_series[sid].committed_observation_count == replayed_by_series[sid].committed_observation_count
        and original_by_series[sid].component_manifest_id == replayed_by_series[sid].component_manifest_id
        for sid in original_by_series
    )
