"""Deterministic curated-universe reporting tests (Milestone 10, Phase
4B) -- every `generate_*` function wraps an already-produced object into
a stable dict; none accepts a raw credential, and none may ever surface
one even by accident."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

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
from quant_platform.market_data.collectors.curated.datasets import (
    CombinedUniverseManifestStore,
    ComponentDatasetManifestStore,
)
from quant_platform.market_data.collectors.curated.metadata import verify_series_metadata
from quant_platform.market_data.collectors.curated.orchestration import run_curated_backfill_operation
from quant_platform.market_data.collectors.curated.reconciliation import reconcile_curated_universe
from quant_platform.market_data.collectors.curated.registry import SeriesTier, create_curated_series_spec
from quant_platform.market_data.collectors.curated.reports import (
    generate_combined_universe_report,
    generate_component_dataset_report,
    generate_curated_backfill_plan_report,
    generate_curated_ingestion_report,
    generate_curated_reconciliation_report,
    generate_curated_registry_report,
    generate_curated_verification_report,
    generate_metadata_compatibility_report,
    generate_series_collection_report,
    generate_update_plan_report,
)
from quant_platform.market_data.collectors.curated.update_plan import create_curated_update_plan
from quant_platform.market_data.collectors.curated.verification import verify_curated_universe
from quant_platform.market_data.collectors.fred_series_metadata import FredSeriesMetadata
from quant_platform.market_data.collectors.macro_normalization import MacroUnit
from quant_platform.market_data.collectors.protocols import TransportRequest, TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.request_manifest import CredentialMode

CORE_ORDER = ("CPIAUCSL", "DFF", "DFII10", "DGS10")
# Deliberately NOT named/assigned in a way that matches the safety scanner's own
# `api_key\s*=\s*"..."` long-literal pattern (test_market_data_safety_scan.py) --
# this IS a secret-shaped test value, used to prove reports never surface it.
_TEST_SECRET_VALUE = "sk-real-fred-secret-abcdef1234567890"

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


def _seed_universe(namespace: str = "xauusd_macro_reports"):
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
    report = run_curated_backfill_operation(
        repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies,
        revision_policy=revision_policy, operation_id="seed", operation_time=T0, transport=FakeTransport(responses=_responses_for(CORE_ORDER)),
        retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0),
        credential_mode=CredentialMode.API_KEY, api_key=_TEST_SECRET_VALUE,
    )
    return root, repository, cache, registry, availability_policies, revision_policy, backfill_spec, report


def _assert_json_serializable_and_secret_free(payload: dict[str, object]) -> None:
    text = json.dumps(payload)
    assert _TEST_SECRET_VALUE not in text
    assert "api_key" not in text.lower()


class TestRegistryReport:
    def test_shape_and_no_secret(self) -> None:
        registry = default_core_registry()
        report = generate_curated_registry_report(registry)
        assert report["registry_id"] == registry.registry_id
        assert set(report["enabled_series_ids"]) == {"DFII10", "DGS10", "CPIAUCSL", "DFF"}
        _assert_json_serializable_and_secret_free(report)


class TestMetadataCompatibilityReport:
    def test_shape_and_no_secret(self) -> None:
        spec = create_curated_series_spec(
            series_id="DGS10", canonical_series_name="us_10y_nominal_yield", registry_version=1, tier=SeriesTier.CORE_XAUUSD_DRIVER,
            economic_category="rates", expected_native_frequency="D", expected_units=("%",), target_macro_instrument_id="us_10y_nominal_yield",
            normalization_kind=MacroUnit.PERCENT, revision_policy_id="a" * 64, release_availability_policy_id="b" * 64, default_observation_start=OBS_START,
        )
        metadata = FredSeriesMetadata(
            series_id="DGS10", response_realtime_start="2024-06-01", response_realtime_end="2024-06-01", title="10-Year Treasury",
            observation_start="1962-01-02", observation_end="2024-06-01", frequency="Daily", frequency_short="D", units="Percent", units_short="%",
            seasonal_adjustment="Not Seasonally Adjusted", seasonal_adjustment_short="NSA", last_updated="2024-06-01 10:00:00-05", notes=None, popularity=None,
        )
        result = verify_series_metadata(spec, metadata)
        report = generate_metadata_compatibility_report(result)
        assert report["passed"] is True
        _assert_json_serializable_and_secret_free(report)


class TestBackfillPlanReport:
    def test_shape_and_no_secret(self) -> None:
        registry = default_core_registry()
        revision_policy = default_revision_policy()
        spec = create_curated_backfill_spec(
            registry=registry, selected_series_ids=CORE_ORDER, observation_start=OBS_START, observation_end=OBS_END,
            revision_policy_id=revision_policy.revision_policy_id, target_dataset_namespace="ns1",
        )
        report = generate_curated_backfill_plan_report(spec)
        assert report["backfill_plan_id"] == spec.backfill_plan_id
        assert report["selected_series_ids"] == list(CORE_ORDER)
        _assert_json_serializable_and_secret_free(report)


class TestSeriesCollectionReport:
    def test_shape_and_no_secret(self) -> None:
        _root, _repository, _cache, _registry, _availability_policies, _revision_policy, _backfill_spec, ingestion_report = _seed_universe()
        outcome = ingestion_report.series_outcomes[0]
        report = generate_series_collection_report(outcome)
        assert report["series_id"] == outcome.series_id
        assert report["succeeded"] is True
        _assert_json_serializable_and_secret_free(report)


class TestIngestionReport:
    def test_shape_and_no_secret(self) -> None:
        _root, _repository, _cache, _registry, _availability_policies, _revision_policy, _backfill_spec, ingestion_report = _seed_universe()
        report = generate_curated_ingestion_report(ingestion_report)
        assert report["completeness_status"] == "complete"
        assert len(report["series_outcomes"]) == 4
        _assert_json_serializable_and_secret_free(report)


class TestComponentAndCombinedReports:
    def test_component_dataset_report_shape(self) -> None:
        root, _repository, _cache, _registry, _availability_policies, _revision_policy, _backfill_spec, _ingestion_report = _seed_universe()
        component_store = ComponentDatasetManifestStore(root)
        manifest = component_store.read_current("fred", "DGS10")
        report = generate_component_dataset_report(manifest)
        assert report["series_id"] == "DGS10"
        assert report["native_frequency"] == "D"
        _assert_json_serializable_and_secret_free(report)

    def test_combined_universe_report_shape(self) -> None:
        root, _repository, _cache, _registry, _availability_policies, _revision_policy, backfill_spec, _ingestion_report = _seed_universe()
        combined_store = CombinedUniverseManifestStore(root)
        combined = combined_store.read_current(backfill_spec.target_dataset_namespace)
        report = generate_combined_universe_report(combined)
        assert set(report["component_manifest_ids"].keys()) == {"DFII10", "DGS10", "CPIAUCSL", "DFF"}
        _assert_json_serializable_and_secret_free(report)


class TestUpdatePlanReport:
    def test_shape_includes_derived_fields(self) -> None:
        root, _repository, _cache, registry, _availability_policies, revision_policy, backfill_spec, _ingestion_report = _seed_universe()
        component_store = ComponentDatasetManifestStore(root)
        combined_store = CombinedUniverseManifestStore(root)
        combined = combined_store.read_current(backfill_spec.target_dataset_namespace)
        plan = create_curated_update_plan(
            existing_combined_manifest=combined, component_store=component_store, registry=registry, selected_series_ids=CORE_ORDER,
            target_dataset_namespace=backfill_spec.target_dataset_namespace, desired_observation_end=datetime(2024, 1, 1, tzinfo=timezone.utc),
            revision_policy=revision_policy, planning_time=T0,
        )
        report = generate_update_plan_report(plan)
        assert report["is_exact_no_op"] is True
        assert report["series_requiring_update"] == []
        _assert_json_serializable_and_secret_free(report)


class TestReconciliationAndVerificationReports:
    def test_reconciliation_report_shape(self) -> None:
        _root, repository, _cache, registry, _availability_policies, _revision_policy, backfill_spec, _ingestion_report = _seed_universe()
        validation = reconcile_curated_universe(repository=repository, registry=registry, target_dataset_namespace=backfill_spec.target_dataset_namespace, as_of=T0)
        report = generate_curated_reconciliation_report(report=validation, target_dataset_namespace=backfill_spec.target_dataset_namespace)
        assert report["critical_count"] == 0
        assert report["scope"] == backfill_spec.target_dataset_namespace
        _assert_json_serializable_and_secret_free(report)

    def test_verification_report_shape(self) -> None:
        _root, repository, cache, registry, availability_policies, revision_policy, backfill_spec, ingestion_report = _seed_universe()
        validation = verify_curated_universe(
            repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies,
            revision_policy=revision_policy, series_outcomes=ingestion_report.series_outcomes, as_of=T0,
        )
        report = generate_curated_verification_report(report=validation, target_dataset_namespace=backfill_spec.target_dataset_namespace)
        assert report["critical_count"] == 0
        _assert_json_serializable_and_secret_free(report)


class TestNoReportEverAcceptsRawCredential:
    def test_ingestion_report_carries_no_api_key_field_even_though_one_was_used(self) -> None:
        """The backfill that produced this report was run WITH a real
        (fake, for this test) `api_key`; the resulting report must not
        surface it anywhere -- proven non-vacuously since the key WAS
        actually in play for this run."""
        _root, _repository, _cache, _registry, _availability_policies, _revision_policy, _backfill_spec, ingestion_report = _seed_universe()
        report = generate_curated_ingestion_report(ingestion_report)
        _assert_json_serializable_and_secret_free(report)
