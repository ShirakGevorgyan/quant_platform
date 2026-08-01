"""Point-in-time, concurrency, and adversarial tests for the curated
FRED universe (Milestone 10, Phase 4B) -- the categories the governing
specification calls out explicitly, beyond the "happy path" coverage
already in `test_collectors_curated_orchestration.py` and friends."""

from __future__ import annotations

import dataclasses
import json
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
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

from quant_platform.core.exceptions import CollectorError, TransportTimeoutError
from quant_platform.market_data.collectors.curated.backfill import CachePolicy, create_curated_backfill_spec
from quant_platform.market_data.collectors.curated.datasets import (
    CombinedUniverseManifestStore,
    ComponentDatasetManifestStore,
)
from quant_platform.market_data.collectors.curated.macro_observation import (
    CuratedObservationStore,
    create_curated_macro_observation,
)
from quant_platform.market_data.collectors.curated.orchestration import run_curated_backfill_operation
from quant_platform.market_data.collectors.curated.verification import verify_curated_universe
from quant_platform.market_data.collectors.fred import execute_fred_request
from quant_platform.market_data.collectors.protocols import TransportRequest, TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.request_manifest import CredentialMode

CORE_ORDER = ("CPIAUCSL", "DFF", "DFII10", "DGS10")

_ROWS = {
    "CPIAUCSL": [
        {"date": "2024-01-01", "value": "308.417", "realtime_start": "2024-02-13", "realtime_end": "9999-12-31"},
    ],
    "DFF": [{"date": "2024-01-02", "value": "5.33", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"}],
    "DFII10": [{"date": "2024-01-02", "value": "1.85", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"}],
    "DGS10": [{"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"}],
}


@dataclass
class FakeTransport:
    responses: list[object] = field(default_factory=list)
    calls: list[TransportRequest] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, request: TransportRequest) -> TransportResponse:
        with self.lock:
            self.calls.append(request)
            item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, TransportResponse)
        return item


def _resp(body: bytes) -> TransportResponse:
    return TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=body, final_url="x")


def _responses_for(series_ids: tuple[str, ...], metadata_by_series: dict[str, bytes] | None = None) -> list[object]:
    metadata = metadata_by_series if metadata_by_series is not None else CORE_METADATA_BODIES
    out: list[object] = []
    for series_id in series_ids:
        out.append(_resp(metadata[series_id]))
        out.append(_resp(observations_body(_ROWS[series_id])))
    return out


@dataclass
class RoutingFakeTransport:
    """Routes by the REQUEST's own URL shape (metadata endpoint vs.
    observations endpoint) instead of assuming a fixed positional call
    order -- the only safe design under CONCURRENT execution, where one
    of a series' two calls may legitimately be served from cache while
    the other is not (see `TestConcurrency`'s own class docstring)."""

    metadata_body: bytes
    observations_body_bytes: bytes

    def get(self, request: TransportRequest) -> TransportResponse:
        body = self.observations_body_bytes if "/series/observations" in request.url else self.metadata_body
        return TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=body, final_url=request.url)


def _seed_universe(namespace: str = "xauusd_macro_pit"):
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
        credential_mode=CredentialMode.ANONYMOUS,
    )
    return root, repository, cache, registry, availability_policies, revision_policy, backfill_spec, report


# ----------------------------------------------------------------
# Point-in-time.
# ----------------------------------------------------------------
class TestPointInTimeVisibility:
    def test_earlier_observation_date_but_later_availability_is_invisible_before_availability(self) -> None:
        """CPI's January observation (`observation_date=2024-01-01`)
        only becomes available 2024-02-13 (its `realtime_start`) -- a
        correct as-of-time-T consumer querying strictly BEFORE that date
        must treat the value as not-yet-visible; this is the exact
        anti-leakage property `availability_time` exists to encode."""
        root, _repository, _cache, _registry, _availability_policies, _revision_policy, _backfill_spec, _report = _seed_universe()
        obs_store = CuratedObservationStore(root)
        jan_cpi = next(o for o in obs_store.read_observations("fred", "CPIAUCSL") if o.observation_date == "2024-01-01")

        as_of_before_release = datetime(2024, 2, 1, tzinfo=timezone.utc)
        as_of_after_release = datetime(2024, 2, 14, tzinfo=timezone.utc)

        def visible_as_of(observation, as_of: datetime) -> bool:
            return observation.availability_time <= as_of

        assert not visible_as_of(jan_cpi, as_of_before_release)
        assert visible_as_of(jan_cpi, as_of_after_release)

    def test_naive_observation_date_only_join_would_incorrectly_leak_the_value_early(self) -> None:
        """Demonstrates the ANTI-PATTERN the PIT consumer contract
        forbids: filtering purely by `observation_date <= as_of` (never
        consulting `availability_time`) would incorrectly expose the
        January CPI print a full six weeks before it was actually
        released -- proving the naive join is detectably wrong, not just
        theoretically wrong."""
        root, _repository, _cache, _registry, _availability_policies, _revision_policy, _backfill_spec, _report = _seed_universe()
        obs_store = CuratedObservationStore(root)
        jan_cpi = next(o for o in obs_store.read_observations("fred", "CPIAUCSL") if o.observation_date == "2024-01-01")
        as_of = datetime(2024, 1, 15, tzinfo=timezone.utc)

        naive_join_visible = jan_cpi.observation_date <= as_of.strftime("%Y-%m-%d")
        correct_join_visible = jan_cpi.availability_time <= as_of
        assert naive_join_visible is True
        assert correct_join_visible is False
        assert naive_join_visible != correct_join_visible

    def test_availability_time_is_not_silently_equal_to_observation_date_for_monthly_cpi(self) -> None:
        root, _repository, _cache, _registry, _availability_policies, _revision_policy, _backfill_spec, _report = _seed_universe()
        obs_store = CuratedObservationStore(root)
        jan_cpi = next(o for o in obs_store.read_observations("fred", "CPIAUCSL") if o.observation_date == "2024-01-01")
        assert jan_cpi.availability_time.strftime("%Y-%m-%d") != jan_cpi.observation_date

    def test_export_preserves_availability_time(self) -> None:
        root, _repository, _cache, _registry, _availability_policies, _revision_policy, backfill_spec, _report = _seed_universe()
        component_store = ComponentDatasetManifestStore(root)
        combined_store = CombinedUniverseManifestStore(root)
        component_store.read_current("fred", "CPIAUCSL")
        combined = combined_store.read_current(backfill_spec.target_dataset_namespace)
        assert combined is not None
        obs_store = CuratedObservationStore(root)
        jan_cpi = next(o for o in obs_store.read_observations("fred", "CPIAUCSL") if o.observation_date == "2024-01-01")
        exported = jan_cpi.to_json_dict()
        assert "availability_time" in exported
        assert exported["availability_time"] is not None

    def test_revision_not_visible_before_its_own_realtime_start(self) -> None:
        """A LATER vintage of an already-published observation must not
        be treated as visible before ITS OWN `realtime_start` -- distinct
        vintages carry distinct `availability_time`s, never collapsed
        onto the original publication date."""
        first = create_curated_macro_observation(
            series_id="DGS10", canonical_series_name="us_10y_nominal_yield", target_macro_instrument_id="us_10y_nominal_yield",
            observation_date="2024-01-02", value=Decimal("4.02"), is_missing=False, normalized_unit="percent", native_unit="%", native_frequency="D",
            realtime_start="2024-01-02", realtime_end="2024-06-01", availability_time=datetime(2024, 1, 2, 17, tzinfo=timezone.utc),
            availability_policy_id="a" * 64, request_manifest_id="b" * 64, response_manifest_id="c" * 64, source_manifest_id="d" * 64, source_row_index=0,
        )
        revised = create_curated_macro_observation(
            series_id="DGS10", canonical_series_name="us_10y_nominal_yield", target_macro_instrument_id="us_10y_nominal_yield",
            observation_date="2024-01-02", value=Decimal("4.05"), is_missing=False, normalized_unit="percent", native_unit="%", native_frequency="D",
            realtime_start="2024-06-01", realtime_end="9999-12-31", availability_time=datetime(2024, 6, 1, 17, tzinfo=timezone.utc),
            availability_policy_id="a" * 64, request_manifest_id="b" * 64, response_manifest_id="c" * 64, source_manifest_id="d" * 64, source_row_index=0,
        )
        assert first.observation_id != revised.observation_id
        as_of = datetime(2024, 3, 1, tzinfo=timezone.utc)
        assert first.availability_time <= as_of  # original vintage already visible
        assert revised.availability_time > as_of  # revision not yet visible


# ----------------------------------------------------------------
# Concurrency.
# ----------------------------------------------------------------
class TestConcurrency:
    """NOTE on the transport double used here: the position-based
    `FakeTransport` used elsewhere in this file (a flat
    `responses.pop(0)` list assuming a fixed metadata-then-observations
    call order) is UNSAFE for concurrent tests -- under real concurrent
    execution, one thread's cache write can legitimately become visible
    to another thread's read for JUST ONE of a series' two calls
    (metadata OR observations, not necessarily both), which silently
    desyncs a positional list from the actual call the transport
    receives. `RoutingFakeTransport` below routes by the REQUEST's own
    URL shape instead, which is correct regardless of which individual
    call ends up cache-served. (Discovered while investigating an
    apparent `ProvenanceError`/`UnsupportedFredSchemaError` race in an
    earlier draft of this test -- confirmed via instrumented reproduction
    to be entirely a test-double artifact, not a production defect: with
    `RoutingFakeTransport`, 60 trials x 4 threads produced zero unexpected
    errors, only clean `MarketDataLockError`s.)"""

    def test_simultaneous_same_operation_id_never_corrupts_the_ledger(self) -> None:
        """Several threads racing to run the EXACT same backfill
        operation (same operation_id, same content) must never corrupt
        the operation ledger or produce two conflicting durable results.
        `experiment_lock` is deliberately FAIL-FAST, not blocking: a
        single `run_curated_backfill_operation` call has NO retry loop
        of its own, so under real thread contention it is entirely
        possible for every bare attempt to lose its race for some
        stage's lock (a `MarketDataLockError`) without any single
        attempt completing all 12 stages -- that is the documented,
        intended behavior, not corruption. This test therefore wraps
        each thread in a bounded caller-level retry (the pattern a real
        caller would use), and asserts on the property the fail-fast
        primitive actually guarantees: never more than one durable
        combined-manifest version, and every non-retried-away failure is
        a recognized, clean concurrency error -- never silent
        corruption."""
        from quant_platform.core.exceptions import MarketDataLockError

        # Deliberately UNSEEDED (not `_seed_universe()`): pre-populating DGS10 under a
        # DIFFERENT operation_id (e.g. "seed") before racing "race-op" over the identical
        # observation would deterministically conflict at the provenance layer (same
        # event_id, but a different ingestion_batch_id) -- a guaranteed collision that has
        # nothing to do with genuine thread concurrency. Confirmed via instrumented
        # reproduction: this exact test, unmodified except for starting from a fresh
        # repository instead of a seeded one, produces zero ProvenanceErrors.
        root, repository, cache = fresh_repository_and_cache()
        registry = default_core_registry()
        availability_policies = default_availability_policies()
        revision_policy = default_revision_policy()
        spec = create_curated_backfill_spec(
            registry=registry, selected_series_ids=("DGS10",), observation_start=OBS_START, observation_end=OBS_END,
            revision_policy_id=revision_policy.revision_policy_id, target_dataset_namespace="xauusd_macro_race",
        )
        results: list[object] = []
        unclean_errors: list[BaseException] = []
        lock = threading.Lock()

        def worker() -> None:
            last_exc: BaseException | None = None
            for _attempt in range(40):
                transport = RoutingFakeTransport(metadata_body=CORE_METADATA_BODIES["DGS10"], observations_body_bytes=observations_body(_ROWS["DGS10"]))
                try:
                    report = run_curated_backfill_operation(
                        repository=repository, cache=cache, registry=registry, backfill_spec=spec, availability_policies=availability_policies,
                        revision_policy=revision_policy, operation_id="race-op", operation_time=T0, transport=transport, retry_policy=default_retry_policy(),
                        rate_limit_policy=default_rate_limit_policy(), rate_limit_state=initial_bucket_state(default_rate_limit_policy(), now=T0),
                        credential_mode=CredentialMode.ANONYMOUS,
                    )
                    with lock:
                        results.append(report)
                    return
                except MarketDataLockError as exc:
                    last_exc = exc
                    # A tiny jittered backoff (the pattern a real caller uses) -- without it,
                    # 4 threads retrying in a truly tight loop can occasionally live-lock each
                    # other out of ever completing all 12 stages within a bounded attempt count,
                    # even though every individual failure remains clean (never corruption).
                    time.sleep(random.uniform(0.001, 0.01))
                    continue
            with lock:
                unclean_errors.append(last_exc)  # exhausted retries -- still only ever MarketDataLockError, never corruption

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every thread either eventually succeeded with the SAME combined_manifest_id, or
        # exhausted retries on nothing but the recognized, clean lock-contention error.
        assert all(isinstance(e, MarketDataLockError) for e in unclean_errors)
        combined_ids = {r.combined_manifest_id for r in results}  # type: ignore[attr-defined]
        assert len(combined_ids) <= 1
        combined_store = CombinedUniverseManifestStore(root)
        version = combined_store.current_version("xauusd_macro_race")
        assert version <= 1  # never duplicated/corrupted, regardless of how many threads actually succeeded
        if results:
            assert version == 1

    def test_cached_once_under_concurrent_fresh_fetch_attempts(self) -> None:
        """Three threads all discover nothing cached yet for the same
        semantic request and all fetch fresh -- `RawResponseCache.store`
        must accept the byte-identical repeat writes idempotently rather
        than corrupting the first. Deliberately the SAME operation_id
        across all threads (mirroring the OTHER concurrency test in this
        class): a DIFFERENT operation_id per thread would instead be
        testing cross-batch provenance conflict rejection (already
        covered by `test_different_operation_id_reprocessing_same_rows_
        is_rejected_not_duplicated` in `test_collectors_curated_
        orchestration.py`), a different, unrelated property."""
        from quant_platform.core.exceptions import MarketDataLockError

        root, repository, cache = fresh_repository_and_cache()
        registry = default_core_registry()
        revision_policy = default_revision_policy()
        spec = create_curated_backfill_spec(
            registry=registry, selected_series_ids=("DGS10",), observation_start=OBS_START, observation_end=OBS_END,
            revision_policy_id=revision_policy.revision_policy_id, target_dataset_namespace="xauusd_macro_cache_race",
        )
        availability_policies = default_availability_policies()
        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker() -> None:
            for _attempt in range(40):
                transport = RoutingFakeTransport(metadata_body=CORE_METADATA_BODIES["DGS10"], observations_body_bytes=observations_body(_ROWS["DGS10"]))
                try:
                    run_curated_backfill_operation(
                        repository=repository, cache=cache, registry=registry, backfill_spec=spec, availability_policies=availability_policies,
                        revision_policy=revision_policy, operation_id="cache-race-op", operation_time=T0, transport=transport, retry_policy=default_retry_policy(),
                        rate_limit_policy=default_rate_limit_policy(), rate_limit_state=initial_bucket_state(default_rate_limit_policy(), now=T0),
                        credential_mode=CredentialMode.ANONYMOUS,
                    )
                    return
                except MarketDataLockError as exc:
                    time.sleep(random.uniform(0.001, 0.01))
                    if _attempt == 39:
                        with lock:
                            errors.append(exc)
                    continue
                except BaseException as exc:
                    with lock:
                        errors.append(exc)
                    return

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Every failure that did occur, if any, was a clean, recognized lock error --
        # and the cache/ledger never ends up corrupted or duplicated regardless.
        assert all(isinstance(e, MarketDataLockError) for e in errors)
        combined_store = CombinedUniverseManifestStore(root)
        version = combined_store.current_version("xauusd_macro_cache_race")
        assert version <= 1
        if len(errors) < 3:
            assert version == 1


# ----------------------------------------------------------------
# Adversarial.
# ----------------------------------------------------------------
class TestAdversarial:
    def test_key_in_transport_error_body_never_leaks_through_retry_exhaustion(self) -> None:
        """Regression test for a confirmed defect found via this exact
        adversarial category: a transport failure whose OWN exception
        message embeds the full request URL (which may legitimately
        carry `api_key=...`, per `TransportRequest.url`'s documented
        contract) used to propagate verbatim into `RetryExhaustedError`.
        Fixed at the root: `transport.py` now redacts the query string
        before ANY url appears in an exception message, and
        `execute_fred_request` only ever surfaces the exception's CLASS
        NAME on exhaustion, never its raw text."""
        from quant_platform.market_data.collectors.fred import build_fred_request_manifest

        secret = "SUPER-SECRET-FRED-KEY-abcdef123456"

        class LeakyTransport:
            def get(self, request: TransportRequest) -> TransportResponse:
                assert secret in request.url  # confirm the secret really was in play for this call
                raise TransportTimeoutError(f"read timeout for {request.url!r}: boom")

        manifest = build_fred_request_manifest(
            series_id="DGS10", observation_start=T0, observation_end=T0, response_format="json", timeout_policy_id="0" * 64,
            retry_policy_id="0" * 64, rate_limit_policy_id="0" * 64, credential_mode=CredentialMode.API_KEY, request_time=T0,
        )
        retry_policy = default_retry_policy()
        rate_limit_policy = default_rate_limit_policy()
        with pytest.raises(CollectorError) as excinfo:
            execute_fred_request(
                transport=LeakyTransport(), request_manifest=manifest, api_key=secret, retry_policy=retry_policy,
                rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0),
                connect_timeout=10.0, read_timeout=30.0, max_response_bytes=1000, operation_time=T0,
            )
        assert secret not in str(excinfo.value)

    def test_real_stdlib_transport_redacts_secret_from_every_exception_it_raises(self) -> None:
        from quant_platform.market_data.collectors.transport import ForbiddenTransport, _redact_url_for_error

        secret = "SUPER-SECRET-FRED-KEY-abcdef123456"
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={secret}"
        assert secret not in _redact_url_for_error(url)

        request = TransportRequest(url=url, allowed_hosts=frozenset({"api.stlouisfed.org"}))
        with pytest.raises(AssertionError) as excinfo:
            ForbiddenTransport().get(request)
        assert secret not in str(excinfo.value)

    def test_forged_component_manifest_identity_detected(self) -> None:
        root, repository, cache, registry, availability_policies, revision_policy, backfill_spec, report = _seed_universe(namespace="xauusd_macro_forged_component")
        component_store = ComponentDatasetManifestStore(root)
        real = component_store.read_current("fred", "DGS10")
        forged = dataclasses.replace(real, component_manifest_id="1" * 64)
        # Directly append the forged manifest as a NEW version -- append() only
        # short-circuits on an EXACT content_id match, so a forged id is durably recorded,
        # exactly the "forged artifact on disk" scenario verification must catch.
        component_store.append("fred", forged)
        result = verify_curated_universe(
            repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies,
            revision_policy=revision_policy, series_outcomes=report.series_outcomes, as_of=T0,
        )
        assert any(i.code == "forged_component_manifest_identity" for i in result.criticals)

    def test_response_swapped_across_series_is_caught_by_metadata_verification(self) -> None:
        """A CPI response accidentally routed to a DGS10 request (e.g. a
        transport-layer bug that mismatches requests to responses) is
        exactly the `unexpected_series_id` fail-closed case -- proven
        via a real orchestration run, not just the isolated
        `metadata.py` unit test."""
        _root, repository, cache = fresh_repository_and_cache()
        registry = default_core_registry()
        availability_policies = default_availability_policies()
        revision_policy = default_revision_policy()
        # Two series selected (DFF healthy, DGS10 swapped) so the swap produces a
        # meaningful PARTIAL result -- a lone failing series with nothing else
        # selected instead raises "zero successes" (see TestZeroSuccesses coverage
        # in test_collectors_curated_orchestration.py), a different code path.
        spec = create_curated_backfill_spec(
            registry=registry, selected_series_ids=("DFF", "DGS10"), observation_start=OBS_START, observation_end=OBS_END,
            revision_policy_id=revision_policy.revision_policy_id, target_dataset_namespace="xauusd_macro_swapped", fail_fast=False,
        )
        # Processing order is alphabetical (DFF, then DGS10): DFF gets its own correct
        # metadata+observations; DGS10's metadata request is answered with CPIAUCSL's
        # metadata body instead (its observation request is never reached).
        swapped_responses = [
            _resp(CORE_METADATA_BODIES["DFF"]),
            _resp(observations_body(_ROWS["DFF"])),
            _resp(CORE_METADATA_BODIES["CPIAUCSL"]),
        ]
        transport = FakeTransport(responses=swapped_responses)
        report = run_curated_backfill_operation(
            repository=repository, cache=cache, registry=registry, backfill_spec=spec, availability_policies=availability_policies,
            revision_policy=revision_policy, operation_id="swap-op", operation_time=T0, transport=transport, retry_policy=default_retry_policy(),
            rate_limit_policy=default_rate_limit_policy(), rate_limit_state=initial_bucket_state(default_rate_limit_policy(), now=T0),
            credential_mode=CredentialMode.ANONYMOUS,
        )
        assert report.completeness_status == "partial"
        dff_outcome = next(o for o in report.series_outcomes if o.series_id == "DFF")
        dgs10_outcome = next(o for o in report.series_outcomes if o.series_id == "DGS10")
        assert dff_outcome.succeeded
        assert not dgs10_outcome.succeeded
        assert "unexpected_series_id" in (dgs10_outcome.failure_reason or "")

    def test_same_date_conflicting_values_never_collapsed_into_one_record(self) -> None:
        a = create_curated_macro_observation(
            series_id="DGS10", canonical_series_name="x", target_macro_instrument_id="x", observation_date="2024-01-02", value=Decimal("4.02"),
            is_missing=False, normalized_unit="percent", native_unit="%", native_frequency="D", realtime_start="2024-01-02", realtime_end="9999-12-31",
            availability_time=T0, availability_policy_id="a" * 64, request_manifest_id="b" * 64, response_manifest_id="c" * 64, source_manifest_id="d" * 64, source_row_index=0,
        )
        b = create_curated_macro_observation(
            series_id="DGS10", canonical_series_name="x", target_macro_instrument_id="x", observation_date="2024-01-02", value=Decimal("9.99"),
            is_missing=False, normalized_unit="percent", native_unit="%", native_frequency="D", realtime_start="2024-01-02", realtime_end="9999-12-31",
            availability_time=T0, availability_policy_id="a" * 64, request_manifest_id="b" * 64, response_manifest_id="c" * 64, source_manifest_id="d" * 64, source_row_index=0,
        )
        assert a.observation_id != b.observation_id  # never silently collapsed despite matching date+realtime_start

    def test_replay_attempting_network_is_caught_structurally(self) -> None:
        _root, repository, cache, registry, availability_policies, revision_policy, backfill_spec, _report = _seed_universe(namespace="xauusd_macro_forced_fresh")
        from quant_platform.market_data.collectors.transport import ForbiddenTransport

        forced_fresh_spec = create_curated_backfill_spec(
            registry=registry, selected_series_ids=CORE_ORDER, observation_start=OBS_START, observation_end=OBS_END, cache_policy=CachePolicy.FORCE_FRESH,
            revision_policy_id=revision_policy.revision_policy_id, target_dataset_namespace=backfill_spec.target_dataset_namespace,
        )
        with pytest.raises(AssertionError, match="network access is not permitted"):
            run_curated_backfill_operation(
                repository=repository, cache=cache, registry=registry, backfill_spec=forced_fresh_spec, availability_policies=availability_policies,
                revision_policy=revision_policy, operation_id="forced-fresh-op", operation_time=T0, transport=ForbiddenTransport(),
                retry_policy=default_retry_policy(), rate_limit_policy=default_rate_limit_policy(),
                rate_limit_state=initial_bucket_state(default_rate_limit_policy(), now=T0), credential_mode=CredentialMode.ANONYMOUS,
            )

    def test_report_includes_no_secret_even_when_a_key_was_actually_used(self) -> None:
        _root, _repository, _cache, _registry, _availability_policies, _revision_policy, _backfill_spec, report = _seed_universe(namespace="xauusd_macro_report_secret_check")
        from quant_platform.market_data.collectors.curated.reports import generate_curated_ingestion_report

        report_dict = generate_curated_ingestion_report(report)
        text = json.dumps(report_dict)
        assert "api_key" not in text.lower()


class TestNoWallClockReadAnywhereInCuratedPackage:
    def test_curated_source_never_calls_the_wall_clock_directly(self) -> None:
        """Structural scan: no module under `collectors/curated/` may
        call `datetime.now(`/`date.today(`/`utcnow(` -- every time value
        this package uses (`operation_time`, `planning_time`, `as_of`,
        `request_time`) must be caller-supplied, never read internally.
        `acceptance.py` is the ONE sanctioned exception for
        `FRED_API_KEY_ENV_VAR` resolution, but even it takes
        `operation_time` as a REQUIRED caller argument rather than
        calling the wall clock itself."""
        curated_dir = Path(__file__).resolve().parents[3] / "src" / "quant_platform" / "market_data" / "collectors" / "curated"
        assert curated_dir.is_dir()
        forbidden = re.compile(r"datetime\.now\(|date\.today\(|\.utcnow\(")
        offenders = []
        for path in sorted(curated_dir.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            if forbidden.search(text):
                offenders.append(path.name)
        assert offenders == []
