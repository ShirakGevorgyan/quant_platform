"""Independent verification tests (Milestone 10, Phase 4A) -- confirms
`verify_fred_macro_operation` genuinely REDERIVES every artifact from
durable state rather than trusting a cached parsed result, and that it
correctly detects tampering, wrong parameters, and unknown operations."""

from __future__ import annotations

import tempfile
from pathlib import Path

from _collectors_test_helpers import (
    T0,
    FakeTransport,
    default_rate_limit_policy,
    default_retry_policy,
    fred_json_body,
)

from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.fred import build_fred_request_manifest
from quant_platform.market_data.collectors.macro_normalization import (
    MacroUnit,
    UnitMappingEntry,
    create_unit_mapping_spec,
)
from quant_platform.market_data.collectors.orchestration import FetchMode, run_fred_macro_ingestion_operation
from quant_platform.market_data.collectors.protocols import TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.request_manifest import CredentialMode
from quant_platform.market_data.collectors.verification import (
    verify_fred_macro_operation,
    verify_secret_absence,
)

UNIT_MAPPING = create_unit_mapping_spec(unit_mapping_version=1, entries=(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT),))


def _env():
    root = Path(tempfile.mkdtemp())
    from quant_platform.market_data.repository import MarketDataRepository

    return root, MarketDataRepository.open(root), RawResponseCache(root)


def _run(repository, cache, operation_id: str = "op-verify-1"):
    retry_policy = default_retry_policy()
    rate_limit_policy = default_rate_limit_policy()
    transport = FakeTransport(responses=[
        TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
            {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"},
        ]), final_url="https://api.stlouisfed.org/fred/series/observations"),
    ])
    report = run_fred_macro_ingestion_operation(
        repository=repository, cache=cache, operation_id=operation_id, operation_time=T0, series_id="DGS10", provider="fred",
        unit_mapping=UNIT_MAPPING, fetch_mode=FetchMode.FRESH, observation_start=T0, observation_end=T0,
        credential_mode=CredentialMode.ANONYMOUS, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
        rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), transport=transport,
    )
    return report, retry_policy, rate_limit_policy


class TestCleanOperationVerifiesWithNoIssues:
    def test_no_criticals_or_errors(self) -> None:
        _root, repository, cache = _env()
        _report, retry_policy, rate_limit_policy = _run(repository, cache)
        result = verify_fred_macro_operation(
            repository=repository, cache=cache, operation_id="op-verify-1", series_id="DGS10", provider="fred", unit_mapping=UNIT_MAPPING,
            as_of=T0, observation_start=T0, observation_end=T0, credential_mode=CredentialMode.ANONYMOUS, request_time=T0,
            retry_policy_id=retry_policy.retry_policy_id, rate_limit_policy_id=rate_limit_policy.rate_limit_policy_id,
        )
        assert result.criticals == ()
        assert result.errors == ()


class TestUnknownOperation:
    def test_unknown_operation_id_is_flagged(self) -> None:
        _root, repository, cache = _env()
        _run(repository, cache)
        result = verify_fred_macro_operation(
            repository=repository, cache=cache, operation_id="does-not-exist", series_id="DGS10", provider="fred", unit_mapping=UNIT_MAPPING,
            as_of=T0, observation_start=T0, observation_end=T0, credential_mode=CredentialMode.ANONYMOUS, request_time=T0,
        )
        assert any(i.code == "operation_not_found" for i in result.criticals)


class TestWrongParametersFailToRederive:
    def test_different_observation_window_produces_a_mismatch(self) -> None:
        from datetime import datetime, timezone

        _root, repository, cache = _env()
        _run(repository, cache)
        result = verify_fred_macro_operation(
            repository=repository, cache=cache, operation_id="op-verify-1", series_id="DGS10", provider="fred", unit_mapping=UNIT_MAPPING,
            as_of=T0, observation_start=datetime(2020, 1, 1, tzinfo=timezone.utc), observation_end=datetime(2020, 1, 1, tzinfo=timezone.utc),
            credential_mode=CredentialMode.ANONYMOUS, request_time=T0,
        )
        assert any(i.code == "request_manifest_identity_mismatch" for i in result.criticals)


class TestTamperingIsDetected:
    def test_tampered_raw_bytes_are_caught_via_independent_rehash(self) -> None:
        _root, repository, cache = _env()
        report, retry_policy, rate_limit_policy = _run(repository, cache)

        candidates = [p for p in _root.rglob("body.bin") if report.response_manifest_id in str(p.parent)]
        assert len(candidates) == 1
        raw_bytes_path = candidates[0]
        original = raw_bytes_path.read_bytes()
        raw_bytes_path.write_bytes(original + b"TAMPERED")
        try:
            result = verify_fred_macro_operation(
                repository=repository, cache=cache, operation_id="op-verify-1", series_id="DGS10", provider="fred", unit_mapping=UNIT_MAPPING,
                as_of=T0, observation_start=T0, observation_end=T0, credential_mode=CredentialMode.ANONYMOUS, request_time=T0,
                retry_policy_id=retry_policy.retry_policy_id, rate_limit_policy_id=rate_limit_policy.rate_limit_policy_id,
            )
            assert any(i.code in ("raw_content_digest_mismatch", "byte_length_mismatch") for i in result.criticals)
        finally:
            raw_bytes_path.write_bytes(original)

    def test_forged_response_manifest_field_is_caught_by_self_consistency_check(self) -> None:
        """Directly tampers the ON-DISK response manifest JSON (not the
        bytes) to claim a different `http_status` than what its own
        content id was computed from -- the self-consistency recompute
        must catch this."""
        _root, repository, cache = _env()
        report, retry_policy, rate_limit_policy = _run(repository, cache)

        manifest_path = next(p for p in _root.rglob("manifest.json") if report.response_manifest_id in str(p.parent))
        import json

        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw["http_status"] = 500
        manifest_path.write_text(json.dumps(raw), encoding="utf-8")

        result = verify_fred_macro_operation(
            repository=repository, cache=cache, operation_id="op-verify-1", series_id="DGS10", provider="fred", unit_mapping=UNIT_MAPPING,
            as_of=T0, observation_start=T0, observation_end=T0, credential_mode=CredentialMode.ANONYMOUS, request_time=T0,
            retry_policy_id=retry_policy.retry_policy_id, rate_limit_policy_id=rate_limit_policy.rate_limit_policy_id,
        )
        assert any(i.code == "forged_response_manifest_identity" for i in result.criticals)


class TestVerifySecretAbsence:
    class _FakeArtifact:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def to_json_dict(self) -> dict:
            return self._payload

    def test_leaked_secret_is_detected(self) -> None:
        leaky = self._FakeArtifact({"field": "leaked=SECRET_VALUE"})
        result = verify_secret_absence("SECRET_VALUE", leaky, as_of=T0)
        assert len(result.criticals) == 1

    def test_clean_artifact_is_not_flagged(self) -> None:
        clean = self._FakeArtifact({"field": "nothing_here"})
        result = verify_secret_absence("SECRET_VALUE", clean, as_of=T0)
        assert result.criticals == ()

    def test_real_manifests_never_leak_the_secret(self) -> None:
        manifest = build_fred_request_manifest(
            series_id="DGS10", observation_start=T0, observation_end=T0, response_format="json", timeout_policy_id="a" * 64,
            retry_policy_id="b" * 64, rate_limit_policy_id="c" * 64, credential_mode=CredentialMode.API_KEY, request_time=T0,
        )
        result = verify_secret_absence("SECRET_VALUE", manifest, as_of=T0)
        assert result.criticals == ()

    def test_empty_secret_is_a_no_op(self) -> None:
        result = verify_secret_absence("", self._FakeArtifact({"field": "x"}), as_of=T0)
        assert result.criticals == ()
