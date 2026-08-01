"""Cross-store curated-universe reconciliation tests (Milestone 10,
Phase 4B) -- `reconcile_curated_universe` scans registry vs. selected
series, official metadata vs. curated spec linkage, component/combined
manifest linkage, and vintage uniqueness."""

from __future__ import annotations

from dataclasses import dataclass, field

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

from quant_platform.market_data.collectors.curated.backfill import create_curated_backfill_spec
from quant_platform.market_data.collectors.curated.orchestration import run_curated_backfill_operation
from quant_platform.market_data.collectors.curated.reconciliation import reconcile_curated_universe
from quant_platform.market_data.collectors.curated.registry import (
    create_curated_registry,
    default_core_series_specs,
)
from quant_platform.market_data.collectors.protocols import TransportRequest, TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.request_manifest import CredentialMode
from quant_platform.ml.models import ValidationSeverity

CORE_ORDER = ("CPIAUCSL", "DFF", "DFII10", "DGS10")

_ROWS = {
    "CPIAUCSL": [{"date": "2024-01-01", "value": "308.417", "realtime_start": "2024-02-13", "realtime_end": "9999-12-31"}],
    "DFF": [{"date": "2024-01-02", "value": "5.33", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"}],
    "DFII10": [{"date": "2024-01-02", "value": "1.85", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"}],
    "DGS10": [{"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"}],
}


@dataclass
class FakeTransport:
    responses: list[object] = field(default_factory=list)
    calls: list[TransportRequest] = field(default_factory=list)

    def get(self, request: TransportRequest) -> TransportResponse:
        self.calls.append(request)
        item = self.responses.pop(0)
        assert isinstance(item, TransportResponse)
        return item


def _resp(body: bytes) -> TransportResponse:
    return TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=body, final_url="x")


def _responses_for(series_ids: tuple[str, ...]) -> list[object]:
    out: list[object] = []
    for series_id in series_ids:
        out.append(_resp(CORE_METADATA_BODIES[series_id]))
        out.append(_resp(observations_body(_ROWS[series_id])))
    return out


def _seed_universe(namespace: str = "xauusd_macro_reconcile"):
    root, repository, cache = fresh_repository_and_cache()
    registry = default_core_registry()
    availability_policies = default_availability_policies()
    revision_policy = default_revision_policy()
    retry_policy = default_retry_policy()
    rate_limit_policy = default_rate_limit_policy()
    backfill_spec = create_curated_backfill_spec(
        registry=registry, selected_series_ids=CORE_ORDER, observation_start=OBS_START, observation_end=OBS_END,
        revision_policy_id=revision_policy.revision_policy_id, target_dataset_namespace=namespace,
    )
    run_curated_backfill_operation(
        repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies,
        revision_policy=revision_policy, operation_id="seed", operation_time=T0, transport=FakeTransport(responses=_responses_for(CORE_ORDER)),
        retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0),
        credential_mode=CredentialMode.ANONYMOUS,
    )
    return root, repository, registry, namespace


class TestCleanUniverse:
    def test_freshly_seeded_universe_has_no_critical_issues(self) -> None:
        _root, repository, registry, namespace = _seed_universe()
        report = reconcile_curated_universe(repository=repository, registry=registry, target_dataset_namespace=namespace, as_of=T0)
        assert report.criticals == ()


class TestMissingCombinedManifest:
    def test_unknown_namespace_reports_critical(self) -> None:
        _root, repository, registry, _namespace = _seed_universe()
        report = reconcile_curated_universe(repository=repository, registry=registry, target_dataset_namespace="never_seeded_namespace", as_of=T0)
        assert any(i.code == "combined_manifest_missing" for i in report.criticals)


class TestRegistryVersionMismatch:
    def test_different_registry_id_reported_as_warning(self) -> None:
        _root, repository, _registry, namespace = _seed_universe()
        other_specs = default_core_series_specs(
            registry_version=2, revision_policy_id="a" * 64, release_availability_policy_id_daily="b" * 64,
            release_availability_policy_id_monthly="c" * 64, default_observation_start=OBS_START,
        )
        other_registry = create_curated_registry(registry_version=2, specs=other_specs)
        report = reconcile_curated_universe(repository=repository, registry=other_registry, target_dataset_namespace=namespace, as_of=T0)
        assert any(i.code == "registry_version_mismatch" and i.severity is ValidationSeverity.WARNING for i in report.issues)


class TestSeriesAbsentFromRegistry:
    def test_series_referenced_by_combined_manifest_but_absent_from_registry_is_critical(self) -> None:
        _root, repository, registry, namespace = _seed_universe()
        # A registry missing DGS10 entirely, even though the combined manifest references it.
        remaining_specs = tuple(s for s in registry.specs if s.series_id != "DGS10")
        pruned_registry = create_curated_registry(registry_version=registry.registry_version, specs=remaining_specs)
        report = reconcile_curated_universe(repository=repository, registry=pruned_registry, target_dataset_namespace=namespace, as_of=T0)
        assert any(i.code == "curated_registry_series_missing" for i in report.criticals)


class TestConflictingVintage:
    def test_two_different_values_for_the_same_date_and_realtime_start_flagged(self) -> None:
        from decimal import Decimal

        from quant_platform.market_data.collectors.curated.macro_observation import (
            CuratedObservationStore,
            create_curated_macro_observation,
        )

        root, repository, registry, namespace = _seed_universe()
        obs_store = CuratedObservationStore(root)
        # Directly inject a SECOND, conflicting observation sharing (observation_date, realtime_start)
        # with an already-committed DGS10 row but a DIFFERENT value -- an integrity violation that
        # must never happen through the orchestrator (content-addressed identity would differ,
        # producing two DISTINCT observation_ids), simulated here to prove reconciliation catches it.
        conflicting = create_curated_macro_observation(
            series_id="DGS10", canonical_series_name="us_10y_nominal_yield", target_macro_instrument_id="us_10y_nominal_yield",
            observation_date="2024-01-02", value=Decimal("9.99"), is_missing=False, normalized_unit="percent", native_unit="%", native_frequency="D",
            realtime_start="2024-01-02", realtime_end="9999-12-31", availability_time=T0, availability_policy_id="a" * 64,
            request_manifest_id="b" * 64, response_manifest_id="c" * 64, source_manifest_id="d" * 64, source_row_index=999,
        )
        obs_store.append("fred", conflicting)
        report = reconcile_curated_universe(repository=repository, registry=registry, target_dataset_namespace=namespace, as_of=T0)
        assert any(i.code == "conflicting_vintage" for i in report.criticals)
        # This also desynchronizes the component manifest's own recorded observation_count
        # from what the observation store now holds -- both are legitimately reported.
        assert any(i.code == "observation_count_mismatch" for i in report.criticals)
