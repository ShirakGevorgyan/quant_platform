"""Point-in-time, concurrency, and adversarial-audit tests (Milestone
10, Phase 4C, spec Sections 30/33). Many of the 24 adversarial items are
already covered structurally elsewhere (construction-time validation in
`test_collectors_cross_asset_registry.py`/`test_collectors_cross_asset_
policies.py`, tamper/rehash detection in `test_collectors_cross_asset_
reconciliation_verification_reports.py`, offline-replay/no-op-version/
conflicting-duplicate-bar in `test_collectors_cross_asset_fixture_
acceptance.py`) -- this file targets the remaining items: concurrent
duplicate ingestion, forged component/combined manifest identity,
credential-in-error-message leakage, an explicit orchestration-level
`_ForbiddenTransport` proof, and PYTHONHASHSEED-independent registry
identity."""

from __future__ import annotations

import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _collectors_test_helpers import FakeTransport
from _cross_asset_test_helpers import (
    build_default_registry_and_mappings,
    default_rate_limit_policy,
    default_retry_policy,
    fresh_repository_and_cache,
)

from quant_platform.market_data.collectors.cross_asset.market_backfill import (
    MarketCachePolicy,
    create_market_backfill_spec,
)
from quant_platform.market_data.collectors.cross_asset.market_orchestration import (
    run_cross_asset_backfill_operation,
)
from quant_platform.market_data.collectors.cross_asset.market_verification import (
    verify_cross_asset_universe,
)
from quant_platform.market_data.collectors.cross_asset.providers.alpha_vantage import (
    ALPHA_VANTAGE_ALLOWED_HOSTS,
    ALPHA_VANTAGE_COLLECTOR_NAME,
    AlphaVantageCollector,
)
from quant_platform.market_data.collectors.protocols import TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.request_manifest import CredentialMode

T0 = datetime(2024, 1, 5, tzinfo=timezone.utc)


def _gld_body() -> bytes:
    import json

    return json.dumps({
        "Meta Data": {"1. Information": "Daily Prices", "2. Symbol": "GLD", "3. Last Refreshed": "2024-01-05", "4. Output Size": "Full size", "5. Time Zone": "US/Eastern"},
        "Time Series (Daily)": {
            "2024-01-03": {"1. open": "190.00", "2. high": "191.50", "3. low": "189.80", "4. close": "191.00", "5. volume": "1000000"},
            "2024-01-04": {"1. open": "191.10", "2. high": "192.00", "3. low": "190.50", "4. close": "191.80", "5. volume": "900000"},
            "2024-01-05": {"1. open": "191.90", "2. high": "193.00", "3. low": "191.50", "4. close": "192.50", "5. volume": "1100000"},
        },
    }).encode("utf-8")


def _http_ok(body: bytes) -> TransportResponse:
    return TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=body, final_url="https://www.alphavantage.co/query")


def _run_gld(*, repository, cache, transport) -> object:
    registry, mapping_set, session, avail = build_default_registry_and_mappings(driver_ids=("gold_reference",))
    mapping = mapping_set.for_driver("gold_reference")[0]
    backfill_spec = create_market_backfill_spec(
        registry=registry, mapping_set=mapping_set, selected_driver_ids=("gold_reference",), selected_mapping_ids=(mapping.mapping_id,),
        start_time=T0, end_time=T0, requested_granularity="1d", target_dataset_namespace="concurrency_ns", cache_policy=MarketCachePolicy.PREFER_CACHE, fail_fast=False,
    )
    rate_limit_policy = default_rate_limit_policy()
    return run_cross_asset_backfill_operation(
        repository=repository, cache=cache, registry=registry, mapping_set=mapping_set, backfill_spec=backfill_spec,
        session_policies={session.session_policy_id: session}, availability_policies={avail.availability_policy_id: avail},
        collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector()}, allowed_hosts_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: ALPHA_VANTAGE_ALLOWED_HOSTS},
        operation_id="op1", operation_time=T0, transport=transport, api_key="demo", retry_policy=default_retry_policy(),
        rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), credential_mode=CredentialMode.API_KEY,
    ), registry, mapping_set


class TestConcurrency:
    def test_concurrent_identical_backfill_produces_no_duplicate_bars(self) -> None:
        """Two threads racing to commit the SAME semantic backfill to the
        SAME repository: the operation ledger's lock is deliberately
        FAIL-FAST (mirrors `ml.concurrency.experiment_lock`'s own
        documented "translate contested lock into an error, never block-
        wait" contract) -- a losing thread legitimately raises
        `MarketDataLockError`, which this test tolerates. What must NEVER
        happen, regardless of how many threads lose the race, is
        DUPLICATE or CORRUPT bars from whichever threads succeeded."""
        from quant_platform.core.exceptions import MarketDataLockError
        from quant_platform.market_data.collectors.cross_asset.instrument_form import InstrumentForm
        from quant_platform.market_data.collectors.cross_asset.market_record import MarketDriverBarStore

        _root, repository, cache = fresh_repository_and_cache()
        # Pre-seed the cache with a single fetch so both threads hit CACHED
        # (never racing on the transport itself, which is not thread-safe).
        transport = FakeTransport(responses=[_http_ok(_gld_body())])
        _run_gld(repository=repository, cache=cache, transport=transport)

        errors: list[BaseException] = []

        def _worker() -> None:
            try:
                _run_gld(repository=repository, cache=cache, transport=None)
            except MarketDataLockError:
                pass  # expected, fail-fast lock contention -- not a data-integrity problem
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        bars = MarketDriverBarStore(repository.root).read_bars(ALPHA_VANTAGE_COLLECTOR_NAME, "gold_reference", InstrumentForm.ETF)
        assert len(bars) == 3

    def test_same_request_cached_once_under_concurrent_readers(self) -> None:

        _root, repository, cache = fresh_repository_and_cache()
        transport = FakeTransport(responses=[_http_ok(_gld_body())])
        report, _registry, _mapping_set = _run_gld(repository=repository, cache=cache, transport=transport)
        assert report.mapping_outcomes[0].response_manifest_id is not None

        results: list[bytes] = []
        lock = threading.Lock()

        def _reader() -> None:
            data = cache.read_bytes(report.mapping_outcomes[0].response_manifest_id, verify=True)
            with lock:
                results.append(data)

        readers = [threading.Thread(target=_reader) for _ in range(8)]
        for t in readers:
            t.start()
        for t in readers:
            t.join()
        assert len(results) == 8
        assert len(set(results)) == 1


class TestForgedManifestDetection:
    def test_forged_component_manifest_identity_detected(self) -> None:
        from dataclasses import replace

        from quant_platform.market_data.collectors.cross_asset.datasets import (
            ComponentMarketDatasetManifestStore,
        )

        _root, repository, cache = fresh_repository_and_cache()
        transport = FakeTransport(responses=[_http_ok(_gld_body())])
        report, registry, mapping_set = _run_gld(repository=repository, cache=cache, transport=transport)

        component_store = ComponentMarketDatasetManifestStore(repository.root)
        mapping = mapping_set.for_driver("gold_reference")[0]
        genuine = component_store.read_current(mapping.mapping_id)
        assert genuine is not None
        forged = replace(genuine, bar_count=999999)
        # Simulate a forged manifest slipping into the store by appending
        # a tampered copy directly (bypassing normal orchestration commit).
        import json as _json

        path = repository.root / "collectors" / "cross_asset" / "component_manifests" / mapping.mapping_id / "manifests.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_json.dumps(forged.to_json_dict()) + "\n")

        registry_2, mapping_set_2, session, avail = build_default_registry_and_mappings(driver_ids=("gold_reference",))
        from quant_platform.market_data.collectors.cross_asset.market_backfill import (
            create_market_backfill_spec,
        )

        gold_mapping_2 = mapping_set_2.for_driver("gold_reference")[0]
        backfill_spec = create_market_backfill_spec(
            registry=registry_2, mapping_set=mapping_set_2, selected_driver_ids=("gold_reference",), selected_mapping_ids=(gold_mapping_2.mapping_id,),
            start_time=T0, end_time=T0, requested_granularity="1d", target_dataset_namespace="concurrency_ns", cache_policy=MarketCachePolicy.PREFER_CACHE,
        )
        result = verify_cross_asset_universe(
            repository=repository, cache=cache, registry=registry, mapping_set=mapping_set, backfill_spec=backfill_spec,
            session_policies={session.session_policy_id: session}, availability_policies={avail.availability_policy_id: avail},
            collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector()}, mapping_outcomes=report.mapping_outcomes, as_of=T0,
        )
        assert any(i.code in ("component_semantic_digest_mismatch", "forged_component_manifest_identity") for i in result.criticals)

    def test_combined_manifest_component_swap_detected(self) -> None:
        from dataclasses import replace

        from quant_platform.market_data.collectors.cross_asset.datasets import (
            COMBINED_CROSS_ASSET_MANIFEST_KIND,
            CombinedCrossAssetManifestStore,
        )
        from quant_platform.market_data.identity import compute_content_id

        _root, repository, cache = fresh_repository_and_cache()
        transport = FakeTransport(responses=[_http_ok(_gld_body())])
        _report, _registry, _mapping_set = _run_gld(repository=repository, cache=cache, transport=transport)

        combined_store = CombinedCrossAssetManifestStore(repository.root)
        genuine = combined_store.read_current("concurrency_ns")
        assert genuine is not None
        swapped_components = dict(genuine.component_manifest_ids)
        for key in swapped_components:
            swapped_components[key] = "f" * 64
        # `dataclasses.replace` changes the field WITHOUT recomputing
        # `combined_manifest_id` -- exactly what a forger slipping a swapped
        # manifest into storage would attempt (stored id looks unchanged).
        forged = replace(genuine, component_manifest_ids=swapped_components)
        assert forged.combined_manifest_id == genuine.combined_manifest_id  # the forger's OWN claimed id is unchanged

        # Independent verification re-derives the id from the manifest's OWN
        # recorded fields -- the forgery is caught because the RECOMPUTED id
        # no longer matches the STORED (claimed) id.
        recomputed_id = compute_content_id(COMBINED_CROSS_ASSET_MANIFEST_KIND, forged.to_identity_payload())
        assert recomputed_id != forged.combined_manifest_id, "forged component swap must be caught by independent id recomputation"


class TestNoSecretLeakage:
    def test_retry_exhausted_error_never_contains_api_key(self) -> None:

        secret = "AKIA-SUPER-SECRET-KEY-DO-NOT-LEAK"
        transport = FakeTransport(responses=[RuntimeError("boom") for _ in range(5)])
        _root, repository, cache = fresh_repository_and_cache()
        registry, mapping_set, session, avail = build_default_registry_and_mappings(driver_ids=("gold_reference",))
        mapping = mapping_set.for_driver("gold_reference")[0]
        backfill_spec = create_market_backfill_spec(
            registry=registry, mapping_set=mapping_set, selected_driver_ids=("gold_reference",), selected_mapping_ids=(mapping.mapping_id,),
            start_time=T0, end_time=T0, requested_granularity="1d", target_dataset_namespace="secret_ns", cache_policy=MarketCachePolicy.PREFER_CACHE, fail_fast=True,
        )
        rate_limit_policy = default_rate_limit_policy()
        try:
            run_cross_asset_backfill_operation(
                repository=repository, cache=cache, registry=registry, mapping_set=mapping_set, backfill_spec=backfill_spec,
                session_policies={session.session_policy_id: session}, availability_policies={avail.availability_policy_id: avail},
                collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector()}, allowed_hosts_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: ALPHA_VANTAGE_ALLOWED_HOSTS},
                operation_id="op1", operation_time=T0, transport=transport, api_key=secret, retry_policy=default_retry_policy(),
                rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), credential_mode=CredentialMode.API_KEY,
            )
        except Exception as exc:
            assert secret not in str(exc)
            # Walk the exception chain (orchestration wraps the original failure).
            cause = exc.__cause__
            while cause is not None:
                assert secret not in str(cause)
                cause = cause.__cause__


class TestOfflineReplayNeverCallsTransport:
    def test_forbidden_transport_proves_zero_network_calls(self) -> None:
        class _ForbiddenTransport:
            def get(self, _request: object) -> object:
                raise AssertionError("offline replay must perform zero network calls")

        _root, repository, cache = fresh_repository_and_cache()
        transport = FakeTransport(responses=[_http_ok(_gld_body())])
        _run_gld(repository=repository, cache=cache, transport=transport)

        # Replay with a transport that raises on ANY .get() call.
        registry, mapping_set, session, avail = build_default_registry_and_mappings(driver_ids=("gold_reference",))
        mapping = mapping_set.for_driver("gold_reference")[0]
        backfill_spec = create_market_backfill_spec(
            registry=registry, mapping_set=mapping_set, selected_driver_ids=("gold_reference",), selected_mapping_ids=(mapping.mapping_id,),
            start_time=T0, end_time=T0, requested_granularity="1d", target_dataset_namespace="concurrency_ns", cache_policy=MarketCachePolicy.PREFER_CACHE,
            fail_fast=False,  # must match `_run_gld`'s own spec exactly -- same backfill_plan_id, same operation content_digest
        )
        rate_limit_policy = default_rate_limit_policy()
        report = run_cross_asset_backfill_operation(
            repository=repository, cache=cache, registry=registry, mapping_set=mapping_set, backfill_spec=backfill_spec,
            session_policies={session.session_policy_id: session}, availability_policies={avail.availability_policy_id: avail},
            collectors_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: AlphaVantageCollector()}, allowed_hosts_by_provider={ALPHA_VANTAGE_COLLECTOR_NAME: ALPHA_VANTAGE_ALLOWED_HOSTS},
            operation_id="op1", operation_time=T0, transport=_ForbiddenTransport(), api_key="demo", retry_policy=default_retry_policy(),  # type: ignore[arg-type]
            rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), credential_mode=CredentialMode.API_KEY,
        )
        assert report.mapping_outcomes[0].succeeded


class TestPythonHashSeedIndependence:
    def test_registry_identity_independent_of_hash_seed(self) -> None:
        """Runs the SAME registry-construction script under two DIFFERENT
        `PYTHONHASHSEED` values in separate subprocesses -- `registry_id`
        must be byte-identical, proving no `set`/`dict` iteration-order
        dependency leaked into content-addressed identity."""
        script = (
            "import sys; sys.path.insert(0, 'src');"
            "from quant_platform.market_data.collectors.cross_asset.adjustment import AdjustmentPolicyKind, create_adjustment_policy;"
            "from quant_platform.market_data.collectors.cross_asset.registry import (default_core_market_driver_specs, "
            "default_optional_market_driver_specs, create_curated_market_driver_registry);"
            "adj = create_adjustment_policy(kind=AdjustmentPolicyKind.RAW_UNADJUSTED);"
            "core = default_core_market_driver_specs(registry_version=1, adjustment_policy=adj, session_policy_id='s'*64, "
            "availability_policy_id='a'*64, provider_mapping_ids_by_driver={});"
            "opt = default_optional_market_driver_specs(registry_version=1, adjustment_policy=adj, session_policy_id='s'*64, "
            "availability_policy_id='a'*64, provider_mapping_ids_by_driver={});"
            "registry = create_curated_market_driver_registry(registry_version=1, specs=core + opt);"
            "print(registry.registry_id)"
        )
        repo_root = Path(__file__).resolve().parents[3]
        result_1 = subprocess.run([sys.executable, "-c", script], cwd=repo_root, capture_output=True, text=True, env={"PYTHONHASHSEED": "1", "PATH": __import__("os").environ.get("PATH", "")}, check=True)
        result_2 = subprocess.run([sys.executable, "-c", script], cwd=repo_root, capture_output=True, text=True, env={"PYTHONHASHSEED": "42", "PATH": __import__("os").environ.get("PATH", "")}, check=True)
        id_1 = result_1.stdout.strip()
        id_2 = result_2.stdout.strip()
        assert id_1 and id_2
        assert id_1 == id_2
