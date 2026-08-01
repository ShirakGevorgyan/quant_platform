"""Opt-in REAL-Alpha-Vantage acceptance workflow (Milestone 10, Phase
4C) -- the ONLY place anywhere in `collectors/cross_asset/` that reads
an environment variable. Every other module in this package requires a
credential to be passed explicitly by the caller; this module exists
purely so a human running the acceptance workflow does not need to
write a real key into any file, CLI argument history, or example
config -- consistent with "Do not automatically read arbitrary
environment variables in pure domain code": `resolve_alpha_vantage_api_key_from_environment`
is not domain code, it is this ONE, explicitly-named, narrowly-scoped
opt-in entry point.

Disabled by default: `run_real_alpha_vantage_acceptance_workflow`
requires an explicit `api_key: str` argument -- nothing in this module
implicitly falls back to "just try it anyway." A caller (a pytest test,
a manual script) that finds no key via `resolve_alpha_vantage_api_key_from_environment`
is expected to skip cleanly with a precise reason and make ZERO network
calls, never treating a missing credential as an application failure.

BOUNDED SUBSET (spec Section 24): Alpha Vantage's shipped endpoint
supplies ETF-form proxies only -- this workflow exercises the ONE
highest-value core driver this platform can genuinely verify against a
live provider, `gold_reference` (via `GLD`), never claiming to validate
the full 10-concept universe against a real provider. Fixture-based
acceptance (`tests/unit/market_data/test_collectors_cross_asset_fixture_acceptance.py`)
is what covers the full conceptual universe; this module supplements it,
never replaces it."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, time

from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.cross_asset.adjustment import (
    AdjustmentPolicyKind,
    create_adjustment_policy,
)
from quant_platform.market_data.collectors.cross_asset.availability import (
    BarAvailabilityPolicyKind,
    create_bar_availability_policy,
)
from quant_platform.market_data.collectors.cross_asset.instrument_form import (
    InstrumentForm,
    ProxyQuality,
    create_proxy_policy,
)
from quant_platform.market_data.collectors.cross_asset.market_backfill import (
    MarketCachePolicy,
    create_market_backfill_spec,
)
from quant_platform.market_data.collectors.cross_asset.market_orchestration import (
    CrossAssetIngestionReport,
    run_cross_asset_backfill_operation,
)
from quant_platform.market_data.collectors.cross_asset.market_reconciliation import (
    reconcile_cross_asset_universe,
)
from quant_platform.market_data.collectors.cross_asset.market_verification import verify_cross_asset_universe
from quant_platform.market_data.collectors.cross_asset.protocols import HistoricalMarketCollector
from quant_platform.market_data.collectors.cross_asset.providers.alpha_vantage import (
    ALPHA_VANTAGE_ALLOWED_HOSTS,
    ALPHA_VANTAGE_API_KEY_ENV_VAR,
    ALPHA_VANTAGE_COLLECTOR_NAME,
    AlphaVantageCollector,
)
from quant_platform.market_data.collectors.cross_asset.registry import (
    DriverTier,
    create_curated_market_driver_registry,
    create_curated_market_driver_spec,
)
from quant_platform.market_data.collectors.cross_asset.sessions import (
    CandleTimestampConvention,
    create_timezone_session_policy,
)
from quant_platform.market_data.collectors.cross_asset.symbol_mapping import (
    create_provider_symbol_mapping,
    create_symbol_mapping_set,
)
from quant_platform.market_data.collectors.protocols import HistoricalHttpTransport
from quant_platform.market_data.collectors.rate_limit import RateLimitPolicy, TokenBucketState
from quant_platform.market_data.collectors.request_manifest import CredentialMode
from quant_platform.market_data.collectors.retry import RetryPolicy
from quant_platform.market_data.identity import require_tz_aware
from quant_platform.market_data.repository import MarketDataRepository

__all__ = [
    "ALPHA_VANTAGE_ACCEPTANCE_DRIVER_ID",
    "ALPHA_VANTAGE_ACCEPTANCE_SYMBOL",
    "RedactedCrossAssetAcceptanceReport",
    "resolve_alpha_vantage_api_key_from_environment",
    "run_real_alpha_vantage_acceptance_workflow",
]

ALPHA_VANTAGE_ACCEPTANCE_DRIVER_ID = "gold_reference"
ALPHA_VANTAGE_ACCEPTANCE_SYMBOL = "GLD"


def resolve_alpha_vantage_api_key_from_environment(env: Mapping[str, str] | None = None) -> str | None:
    """Returns `None` (never raises, never falls back to a placeholder)
    when `ALPHA_VANTAGE_API_KEY` is absent or blank -- the caller
    decides what "no key" means (skip cleanly, in every sanctioned use
    of this function)."""
    source = env if env is not None else os.environ
    value = source.get(ALPHA_VANTAGE_API_KEY_ENV_VAR)
    if value is None or not value.strip():
        return None
    return value.strip()


@dataclass(frozen=True, slots=True)
class RedactedCrossAssetAcceptanceReport:
    ran: bool
    mappings_checked: tuple[str, ...]
    backfill_stage: str
    backfill_completeness_status: str
    committed_bar_counts: dict[str, int]
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
            "ran": self.ran, "mappings_checked": list(self.mappings_checked), "backfill_stage": self.backfill_stage,
            "backfill_completeness_status": self.backfill_completeness_status, "committed_bar_counts": dict(self.committed_bar_counts),
            "reconciliation_critical_count": self.reconciliation_critical_count, "verification_critical_count": self.verification_critical_count,
            "replay_identical": self.replay_identical, "notes": list(self.notes),
        }


def run_real_alpha_vantage_acceptance_workflow(
    *, api_key: str, repository: MarketDataRepository, cache: RawResponseCache, transport: HistoricalHttpTransport, retry_policy: RetryPolicy,
    rate_limit_policy: RateLimitPolicy, rate_limit_state: TokenBucketState, operation_time: datetime, operation_id: str = "alpha-vantage-acceptance",
    target_dataset_namespace: str = "alpha_vantage_acceptance_workflow", max_records_per_mapping: int = 20_000,
) -> RedactedCrossAssetAcceptanceReport:
    """Requests the ONE bounded, real-verifiable mapping (`GLD` for
    `gold_reference`) over the provider's own full-history response
    shape (see module docstring), runs the full pipeline (metadata
    verify -> backfill -> component/combined datasets -> reconciliation
    -> verification), then a SECOND, cached-replay pass with a
    `_ForbiddenTransport` to prove offline replay makes zero network
    calls, comparing the two runs' semantic results."""
    if not api_key or not api_key.strip():
        raise ValueError(
            "run_real_alpha_vantage_acceptance_workflow requires a non-empty api_key -- callers must resolve one "
            "(e.g. via resolve_alpha_vantage_api_key_from_environment) before calling this"
        )
    require_tz_aware(operation_time, field_name="operation_time")

    adjustment_policy = create_adjustment_policy(kind=AdjustmentPolicyKind.RAW_UNADJUSTED)
    session_policy = create_timezone_session_policy(
        timezone_key="America/New_York", is_24_hour_session=False, timestamp_convention=CandleTimestampConvention.OPEN_LABELED,
        provider_session_note="Alpha Vantage TIME_SERIES_DAILY daily bar date treated as the NYSE session date for this US-listed ETF.",
        session_open_time=time(9, 30), session_close_time=time(16, 0),
    )
    availability_policy = create_bar_availability_policy(
        kind=BarAvailabilityPolicyKind.CLOSE_PLUS_CONSERVATIVE_DELAY, timezone_key="America/New_York", delay_minutes=1440,
        notes="Alpha Vantage documents no exact intraday publication SLA for daily bars; conservatively assume next-day availability.",
    )
    proxy_policy = create_proxy_policy(
        is_proxy=True, proxy_for=ALPHA_VANTAGE_ACCEPTANCE_DRIVER_ID, proxy_quality=ProxyQuality.HIGH,
        known_basis_risk="Physically-backed gold-bullion ETF; small tracking error vs. spot from fund expenses/creation-redemption mechanics.",
    )
    mapping = create_provider_symbol_mapping(
        provider=ALPHA_VANTAGE_COLLECTOR_NAME, provider_symbol=ALPHA_VANTAGE_ACCEPTANCE_SYMBOL, canonical_driver_id=ALPHA_VANTAGE_ACCEPTANCE_DRIVER_ID,
        instrument_form=InstrumentForm.ETF, currency="USD", adjustment_policy_kind=AdjustmentPolicyKind.RAW_UNADJUSTED, proxy_policy=proxy_policy,
        exchange_or_venue="NYSEARCA",
    )
    mapping_set = create_symbol_mapping_set((mapping,))
    driver_spec = create_curated_market_driver_spec(
        canonical_driver_id=ALPHA_VANTAGE_ACCEPTANCE_DRIVER_ID, canonical_name="Gold (XAUUSD Reference Market)", registry_version=1,
        tier=DriverTier.CORE_XAUUSD_MARKET_DRIVER, economic_role="The XAUUSD market itself, verified against a real provider this run.",
        is_required=True, asset_class="precious_metal", preferred_instrument_form=InstrumentForm.SPOT,
        allowed_instrument_forms=(InstrumentForm.SPOT, InstrumentForm.ETF), canonical_currency="USD", canonical_quote_unit="usd_per_troy_ounce",
        expected_frequency="daily", session_policy_id=session_policy.session_policy_id, adjustment_policy=adjustment_policy,
        availability_policy_id=availability_policy.availability_policy_id, provider_mapping_ids=(mapping.mapping_id,), enabled=True,
    )
    registry = create_curated_market_driver_registry(registry_version=1, specs=(driver_spec,))

    backfill_spec = create_market_backfill_spec(
        registry=registry, mapping_set=mapping_set, selected_driver_ids=(ALPHA_VANTAGE_ACCEPTANCE_DRIVER_ID,), selected_mapping_ids=(mapping.mapping_id,),
        start_time=operation_time, end_time=operation_time, requested_granularity="1d", target_dataset_namespace=target_dataset_namespace,
        cache_policy=MarketCachePolicy.PREFER_CACHE, fail_fast=False, max_records_per_mapping=max_records_per_mapping,
    )
    collectors_by_provider: dict[str, HistoricalMarketCollector] = {ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector()}
    allowed_hosts_by_provider = {ALPHA_VANTAGE_COLLECTOR_NAME: ALPHA_VANTAGE_ALLOWED_HOSTS}
    session_policies = {session_policy.session_policy_id: session_policy}
    availability_policies = {availability_policy.availability_policy_id: availability_policy}

    original_report = run_cross_asset_backfill_operation(
        repository=repository, cache=cache, registry=registry, mapping_set=mapping_set, backfill_spec=backfill_spec, session_policies=session_policies,
        availability_policies=availability_policies, collectors_by_provider=collectors_by_provider, allowed_hosts_by_provider=allowed_hosts_by_provider,
        operation_id=operation_id, operation_time=operation_time, transport=transport, api_key=api_key, retry_policy=retry_policy,
        rate_limit_policy=rate_limit_policy, rate_limit_state=rate_limit_state, credential_mode=CredentialMode.API_KEY,
    )

    reconciliation_result = reconcile_cross_asset_universe(
        repository=repository, registry=registry, mapping_set=mapping_set, target_dataset_namespace=target_dataset_namespace,
        session_policies=session_policies, as_of=operation_time,
    )
    verification_result = verify_cross_asset_universe(
        repository=repository, cache=cache, registry=registry, mapping_set=mapping_set, backfill_spec=backfill_spec, session_policies=session_policies,
        availability_policies=availability_policies, collectors_by_provider=collectors_by_provider, mapping_outcomes=original_report.mapping_outcomes,
        as_of=operation_time,
    )

    class _ForbiddenTransport:
        def get(self, _request: object) -> object:
            raise AssertionError("offline replay must perform zero network calls")

    replay_report = run_cross_asset_backfill_operation(
        repository=repository, cache=cache, registry=registry, mapping_set=mapping_set, backfill_spec=backfill_spec, session_policies=session_policies,
        availability_policies=availability_policies, collectors_by_provider=collectors_by_provider, allowed_hosts_by_provider=allowed_hosts_by_provider,
        operation_id=operation_id, operation_time=operation_time, transport=_ForbiddenTransport(), credential_mode=CredentialMode.API_KEY,  # type: ignore[arg-type]
    )
    replay_identical = _reports_semantically_identical(original_report, replay_report)

    committed_counts = {o.mapping_id: o.committed_bar_count for o in original_report.mapping_outcomes}
    notes = tuple(f"mapping {o.mapping_id} failed: {o.failure_reason}" for o in original_report.mapping_outcomes if not o.succeeded)
    return RedactedCrossAssetAcceptanceReport(
        ran=True, mappings_checked=backfill_spec.selected_mapping_ids, backfill_stage=original_report.stage.value,
        backfill_completeness_status=original_report.completeness_status, committed_bar_counts=committed_counts,
        reconciliation_critical_count=len(reconciliation_result.criticals), verification_critical_count=len(verification_result.criticals),
        replay_identical=replay_identical, notes=notes,
    )


def _reports_semantically_identical(original: CrossAssetIngestionReport, replayed: CrossAssetIngestionReport) -> bool:
    if original.completeness_status != replayed.completeness_status:
        return False
    if original.combined_manifest_id != replayed.combined_manifest_id:
        return False
    original_by_mapping = {o.mapping_id: o for o in original.mapping_outcomes}
    replayed_by_mapping = {o.mapping_id: o for o in replayed.mapping_outcomes}
    if set(original_by_mapping) != set(replayed_by_mapping):
        return False
    return all(
        original_by_mapping[mid].committed_bar_count == replayed_by_mapping[mid].committed_bar_count
        and original_by_mapping[mid].component_manifest_id == replayed_by_mapping[mid].component_manifest_id
        for mid in original_by_mapping
    )
