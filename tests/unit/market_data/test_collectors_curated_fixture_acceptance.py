"""Mandatory fixture-based, no-network acceptance workflow for the
curated FRED universe (Milestone 10, Phase 4B) -- the ordinary,
always-run counterpart to the opt-in real-FRED acceptance workflow in
`test_collectors_curated_acceptance.py`. Exercises the COMPLETE
pipeline (registry -> backfill spec -> orchestration -> component/
combined datasets -> reconciliation -> verification -> offline replay)
against realistic, hand-built FRED fixtures covering all 4 core series,
multiple native frequencies, one missing observation, and one revised
observation -- all fully deterministic, zero network calls anywhere in
this file."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from _curated_test_helpers import (
    CORE_METADATA_BODIES,
    OBS_END,
    OBS_START,
    T0,
    default_availability_policies,
    default_core_registry,
    default_rate_limit_policy,
    default_retry_policy,
    default_revision_policy,
    fresh_repository_and_cache,
    observations_body,
)

from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.curated.backfill import create_curated_backfill_spec
from quant_platform.market_data.collectors.curated.datasets import (
    CombinedUniverseManifestStore,
    ComponentDatasetManifestStore,
)
from quant_platform.market_data.collectors.curated.macro_observation import CuratedObservationStore
from quant_platform.market_data.collectors.curated.orchestration import (
    CuratedOperationStage,
    run_curated_backfill_operation,
)
from quant_platform.market_data.collectors.curated.reconciliation import reconcile_curated_universe
from quant_platform.market_data.collectors.curated.reports import (
    generate_combined_universe_report,
    generate_curated_ingestion_report,
    generate_curated_reconciliation_report,
    generate_curated_verification_report,
)
from quant_platform.market_data.collectors.curated.verification import verify_curated_universe
from quant_platform.market_data.collectors.protocols import TransportRequest, TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.request_manifest import CredentialMode
from quant_platform.market_data.repository import MarketDataRepository

CORE_ORDER = ("CPIAUCSL", "DFF", "DFII10", "DGS10")
NAMESPACE = "xauusd_macro_fixture_acceptance"

# Realistic fixture rows: CPI monthly (2 obs), DFF daily with ONE missing value
# (weekend "."), DFII10 daily (1 obs), DGS10 daily with ONE REVISION (same
# observation_date, two vintages with different values and realtime_start).
_ROWS = {
    "CPIAUCSL": [
        {"date": "2024-01-01", "value": "308.417", "realtime_start": "2024-02-13", "realtime_end": "9999-12-31"},
        {"date": "2024-02-01", "value": "310.326", "realtime_start": "2024-03-12", "realtime_end": "9999-12-31"},
    ],
    "DFF": [
        {"date": "2024-01-02", "value": "5.33", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"},
        {"date": "2024-01-06", "value": ".", "realtime_start": "2024-01-06", "realtime_end": "9999-12-31"},
    ],
    "DFII10": [{"date": "2024-01-02", "value": "1.85", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"}],
    "DGS10": [
        {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02", "realtime_end": "2024-05-31"},
        {"date": "2024-01-02", "value": "4.05", "realtime_start": "2024-06-01", "realtime_end": "9999-12-31"},
    ],
}


@dataclass
class RoutingFakeTransport:
    """Routes by URL shape (never assumes call order) -- see
    `test_collectors_curated_pit_concurrency_adversarial.py` for why a
    positional response list is unsafe under any partial-cache scenario."""

    bodies_by_series: dict[str, tuple[bytes, bytes]]
    calls: list[str] = field(default_factory=list)

    def get(self, request: TransportRequest) -> TransportResponse:
        self.calls.append(request.url)
        for series_id, (metadata_body, obs_body) in self.bodies_by_series.items():
            if f"series_id={series_id}" in request.url:
                body = obs_body if "/series/observations" in request.url else metadata_body
                return TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=body, final_url=request.url)
        raise AssertionError(f"RoutingFakeTransport: no fixture registered for {request.url!r}")


class ForbiddenTransport:
    def get(self, request: TransportRequest) -> TransportResponse:
        raise AssertionError(f"offline replay must perform zero network calls (got {request.url!r})")


def _bodies_by_series() -> dict[str, tuple[bytes, bytes]]:
    return {series_id: (CORE_METADATA_BODIES[series_id], observations_body(_ROWS[series_id])) for series_id in CORE_ORDER}


def _run_full_workflow(root: Path, repository: MarketDataRepository, cache: RawResponseCache, *, operation_id: str, transport: object):
    registry = default_core_registry()
    availability_policies = default_availability_policies()
    revision_policy = default_revision_policy()
    retry_policy = default_retry_policy()
    rate_limit_policy = default_rate_limit_policy()
    backfill_spec = create_curated_backfill_spec(
        registry=registry, selected_series_ids=CORE_ORDER, observation_start=OBS_START, observation_end=OBS_END,
        revision_policy_id=revision_policy.revision_policy_id, target_dataset_namespace=NAMESPACE,
    )
    report = run_curated_backfill_operation(
        repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies,
        revision_policy=revision_policy, operation_id=operation_id, operation_time=T0, transport=transport,
        retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0),
        credential_mode=CredentialMode.ANONYMOUS,
    )
    return registry, availability_policies, revision_policy, backfill_spec, report


class TestCompleteFixtureWorkflow:
    def test_full_pipeline_reaches_completed_and_complete(self) -> None:
        root, repository, cache = fresh_repository_and_cache()
        transport = RoutingFakeTransport(bodies_by_series=_bodies_by_series())
        _registry, _availability_policies, _revision_policy, _backfill_spec, report = _run_full_workflow(root, repository, cache, operation_id="fixture-op", transport=transport)
        assert report.stage is CuratedOperationStage.COMPLETED
        assert report.completeness_status == "complete"
        assert all(o.succeeded for o in report.series_outcomes)
        assert len(transport.calls) == 8  # 4 series x (metadata + observations)


class TestMissingValueHandling:
    def test_dff_missing_value_quarantined_and_reported(self) -> None:
        root, repository, cache = fresh_repository_and_cache()
        transport = RoutingFakeTransport(bodies_by_series=_bodies_by_series())
        _registry, _availability_policies, _revision_policy, _backfill_spec, report = _run_full_workflow(root, repository, cache, operation_id="fixture-op", transport=transport)
        dff_outcome = next(o for o in report.series_outcomes if o.series_id == "DFF")
        assert dff_outcome.quarantined_row_count == 1
        assert dff_outcome.missing_count == 1
        assert dff_outcome.valid_row_count == 1

        obs_store = CuratedObservationStore(root)
        dff_observations = obs_store.read_observations("fred", "DFF")
        assert len(dff_observations) == 1  # the quarantined "." row never becomes a durable observation
        assert dff_observations[0].observation_date == "2024-01-02"


class TestRevisionHandling:
    def test_dgs10_revision_preserved_as_two_distinct_observations(self) -> None:
        root, repository, cache = fresh_repository_and_cache()
        transport = RoutingFakeTransport(bodies_by_series=_bodies_by_series())
        _registry, _availability_policies, _revision_policy, _backfill_spec, _report = _run_full_workflow(root, repository, cache, operation_id="fixture-op", transport=transport)

        obs_store = CuratedObservationStore(root)
        dgs10_observations = obs_store.read_observations("fred", "DGS10")
        assert len(dgs10_observations) == 2
        assert {o.value for o in dgs10_observations} == {Decimal("4.02"), Decimal("4.05")}
        assert len({o.observation_id for o in dgs10_observations}) == 2  # never collapsed

        component_store = ComponentDatasetManifestStore(root)
        dgs10_component = component_store.read_current("fred", "DGS10")
        assert dgs10_component.revision_count == 1  # one observation_date with >1 vintage
        assert dgs10_component.observation_count == 2


class TestMultiFrequencyPreservation:
    def test_monthly_and_daily_component_manifests_never_merged(self) -> None:
        root, repository, cache = fresh_repository_and_cache()
        transport = RoutingFakeTransport(bodies_by_series=_bodies_by_series())
        _run_full_workflow(root, repository, cache, operation_id="fixture-op", transport=transport)

        component_store = ComponentDatasetManifestStore(root)
        assert component_store.read_current("fred", "CPIAUCSL").native_frequency == "M"
        for series_id in ("DFF", "DFII10", "DGS10"):
            assert component_store.read_current("fred", series_id).native_frequency == "D"

        combined_store = CombinedUniverseManifestStore(root)
        combined = combined_store.read_current(NAMESPACE)
        assert combined is not None
        assert combined.frequencies_by_series == {"CPIAUCSL": "M", "DFF": "D", "DFII10": "D", "DGS10": "D"}


class TestReconciliationAndVerification:
    def test_reconciliation_and_verification_both_report_zero_criticals(self) -> None:
        root, repository, cache = fresh_repository_and_cache()
        transport = RoutingFakeTransport(bodies_by_series=_bodies_by_series())
        registry, availability_policies, revision_policy, backfill_spec, report = _run_full_workflow(root, repository, cache, operation_id="fixture-op", transport=transport)

        reconciliation_result = reconcile_curated_universe(repository=repository, registry=registry, target_dataset_namespace=NAMESPACE, as_of=T0)
        assert reconciliation_result.criticals == ()

        verification_result = verify_curated_universe(
            repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies,
            revision_policy=revision_policy, series_outcomes=report.series_outcomes, as_of=T0,
        )
        assert verification_result.criticals == ()

        # Reports round-trip cleanly and stay JSON-serializable / secret-free.
        import json

        for payload in (
            generate_curated_ingestion_report(report),
            generate_combined_universe_report(CombinedUniverseManifestStore(root).read_current(NAMESPACE)),
            generate_curated_reconciliation_report(report=reconciliation_result, target_dataset_namespace=NAMESPACE),
            generate_curated_verification_report(report=verification_result, target_dataset_namespace=NAMESPACE),
        ):
            json.dumps(payload)  # must not raise


class TestOfflineReplayEquality:
    def test_replay_with_forbidden_transport_is_semantically_identical(self) -> None:
        root, repository, cache = fresh_repository_and_cache()
        original_transport = RoutingFakeTransport(bodies_by_series=_bodies_by_series())
        _registry, _availability_policies, _revision_policy, _backfill_spec, original_report = _run_full_workflow(root, repository, cache, operation_id="fixture-op", transport=original_transport)

        _registry2, _ap2, _rp2, _bs2, replay_report = _run_full_workflow(root, repository, cache, operation_id="fixture-op", transport=ForbiddenTransport())

        assert replay_report.completeness_status == original_report.completeness_status
        assert replay_report.combined_manifest_id == original_report.combined_manifest_id
        original_by_series = {o.series_id: o for o in original_report.series_outcomes}
        replay_by_series = {o.series_id: o for o in replay_report.series_outcomes}
        assert set(original_by_series) == set(replay_by_series)
        for series_id in original_by_series:
            assert original_by_series[series_id].component_manifest_id == replay_by_series[series_id].component_manifest_id
            assert original_by_series[series_id].committed_observation_count == replay_by_series[series_id].committed_observation_count


class TestDifferentTempRootsProduceIdenticalIdentity:
    def test_two_independent_repositories_converge_on_identical_ids(self) -> None:
        """Running the SAME semantic fixture workflow from two completely
        SEPARATE, unrelated temp directories must produce byte-identical
        `combined_manifest_id`/component ids -- identity is purely
        content-addressed, never influenced by filesystem path."""
        root_a, repository_a, cache_a = fresh_repository_and_cache()
        root_b, repository_b, cache_b = fresh_repository_and_cache()
        assert root_a != root_b

        _r1, _a1, _rv1, _b1, report_a = _run_full_workflow(root_a, repository_a, cache_a, operation_id="fixture-op", transport=RoutingFakeTransport(bodies_by_series=_bodies_by_series()))
        _r2, _a2, _rv2, _b2, report_b = _run_full_workflow(root_b, repository_b, cache_b, operation_id="fixture-op", transport=RoutingFakeTransport(bodies_by_series=_bodies_by_series()))

        assert report_a.combined_manifest_id == report_b.combined_manifest_id
        by_series_a = {o.series_id: o.component_manifest_id for o in report_a.series_outcomes}
        by_series_b = {o.series_id: o.component_manifest_id for o in report_b.series_outcomes}
        assert by_series_a == by_series_b

        obs_store_a = CuratedObservationStore(root_a)
        obs_store_b = CuratedObservationStore(root_b)
        for series_id in CORE_ORDER:
            ids_a = sorted(o.observation_id for o in obs_store_a.read_observations("fred", series_id))
            ids_b = sorted(o.observation_id for o in obs_store_b.read_observations("fred", series_id))
            assert ids_a == ids_b


class TestDifferentPythonHashSeedProducesIdenticalIdentity:
    def test_identity_is_stable_across_pythonhashseed(self) -> None:
        """Content-addressed identity (`compute_content_id`, sha256 over
        CANONICAL json with sorted keys) must be completely independent
        of `PYTHONHASHSEED` -- unlike raw `dict`/`set` iteration order or
        `hash()`, which Python deliberately randomizes per-process by
        default. Spawns two child interpreters with DIFFERENT explicit
        seeds and confirms they compute the identical `combined_manifest_id`."""
        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
            from _curated_test_helpers import (
                CORE_METADATA_BODIES, OBS_END, OBS_START, T0, default_availability_policies, default_core_registry,
                default_rate_limit_policy, default_retry_policy, default_revision_policy, fresh_repository_and_cache, observations_body,
            )
            from quant_platform.market_data.collectors.curated.backfill import create_curated_backfill_spec
            from quant_platform.market_data.collectors.curated.orchestration import run_curated_backfill_operation
            from quant_platform.market_data.collectors.protocols import TransportResponse
            from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
            from quant_platform.market_data.collectors.request_manifest import CredentialMode

            ROWS = {_ROWS!r}
            CORE_ORDER = {CORE_ORDER!r}

            class RoutingFakeTransport:
                def __init__(self, bodies):
                    self.bodies = bodies
                def get(self, request):
                    for series_id, (meta, obs) in self.bodies.items():
                        if f"series_id={{series_id}}" in request.url:
                            body = obs if "/series/observations" in request.url else meta
                            return TransportResponse(status_code=200, headers={{"Content-Type": "application/json"}}, body=body, final_url=request.url)
                    raise AssertionError("no fixture")

            bodies = {{sid: (CORE_METADATA_BODIES[sid], observations_body(ROWS[sid])) for sid in CORE_ORDER}}
            root, repository, cache = fresh_repository_and_cache()
            registry = default_core_registry()
            availability_policies = default_availability_policies()
            revision_policy = default_revision_policy()
            retry_policy = default_retry_policy()
            rate_limit_policy = default_rate_limit_policy()
            backfill_spec = create_curated_backfill_spec(
                registry=registry, selected_series_ids=CORE_ORDER, observation_start=OBS_START, observation_end=OBS_END,
                revision_policy_id=revision_policy.revision_policy_id, target_dataset_namespace="hashseed_check",
            )
            report = run_curated_backfill_operation(
                repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies,
                revision_policy=revision_policy, operation_id="op", operation_time=T0, transport=RoutingFakeTransport(bodies),
                retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0),
                credential_mode=CredentialMode.ANONYMOUS,
            )
            print(report.combined_manifest_id)
        """)
        env_a = {"PYTHONHASHSEED": "0"}
        env_b = {"PYTHONHASHSEED": "999983"}

        result_a = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env={**os.environ, **env_a}, timeout=60)
        result_b = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env={**os.environ, **env_b}, timeout=60)
        assert result_a.returncode == 0, result_a.stderr
        assert result_b.returncode == 0, result_b.stderr
        id_a = result_a.stdout.strip().splitlines()[-1]
        id_b = result_b.stdout.strip().splitlines()[-1]
        assert id_a == id_b
        assert len(id_a) == 64
