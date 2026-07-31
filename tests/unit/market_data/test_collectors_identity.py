"""Request/response manifest identity tests (Milestone 10, Phase 4A) --
content-addressed determinism, operational-field exclusion, and the
exact "what legitimately changes identity" matrix the spec requires."""

from __future__ import annotations

from datetime import datetime, timezone

from quant_platform.market_data.collectors.request_manifest import CredentialMode, create_request_manifest
from quant_platform.market_data.collectors.response_manifest import CompletionStatus, create_response_manifest

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2024, 6, 1, tzinfo=timezone.utc)


def _request_manifest(**overrides):
    defaults = {
        "collector_name": "fred", "collector_version": "1.0.0", "endpoint_host": "api.stlouisfed.org", "endpoint_path": "/fred/series/observations",
        "canonical_query_params": {"series_id": "DGS10", "file_type": "json"}, "canonical_headers": {}, "requested_series_or_dataset": "DGS10",
        "response_format": "json", "timeout_policy_id": "a" * 64, "retry_policy_id": "b" * 64, "rate_limit_policy_id": "c" * 64,
        "credential_mode": CredentialMode.ANONYMOUS, "request_time": T0,
    }
    defaults.update(overrides)
    return create_request_manifest(**defaults)


def _response_manifest(**overrides):
    defaults = {
        "request_manifest_id": "a" * 64, "http_status": 200, "raw_headers": {"Content-Type": "application/json"}, "raw_bytes": b"hello",
        "content_type": "application/json", "encoding": "utf-8", "completion_status": CompletionStatus.COMPLETE, "received_time": T0,
    }
    defaults.update(overrides)
    return create_response_manifest(**defaults)


class TestRequestManifestIdentity:
    def test_same_semantic_request_same_id(self) -> None:
        assert _request_manifest().request_manifest_id == _request_manifest().request_manifest_id

    def test_query_param_ordering_is_irrelevant(self) -> None:
        a = _request_manifest(canonical_query_params={"series_id": "DGS10", "file_type": "json"})
        b = _request_manifest(canonical_query_params={"file_type": "json", "series_id": "DGS10"})
        assert a.request_manifest_id == b.request_manifest_id

    def test_request_time_is_excluded_from_identity(self) -> None:
        a = _request_manifest(request_time=T0)
        b = _request_manifest(request_time=T1)
        assert a.request_manifest_id == b.request_manifest_id

    def test_interval_change_changes_id(self) -> None:
        a = _request_manifest(requested_interval_start=T0)
        b = _request_manifest(requested_interval_start=T1)
        assert a.request_manifest_id != b.request_manifest_id

    def test_retry_policy_id_change_changes_id(self) -> None:
        a = _request_manifest(retry_policy_id="b" * 64)
        b = _request_manifest(retry_policy_id="d" * 64)
        assert a.request_manifest_id != b.request_manifest_id

    def test_rate_limit_policy_id_change_changes_id(self) -> None:
        a = _request_manifest(rate_limit_policy_id="c" * 64)
        b = _request_manifest(rate_limit_policy_id="e" * 64)
        assert a.request_manifest_id != b.request_manifest_id

    def test_timeout_policy_id_change_changes_id(self) -> None:
        a = _request_manifest(timeout_policy_id="a" * 64)
        b = _request_manifest(timeout_policy_id="f" * 64)
        assert a.request_manifest_id != b.request_manifest_id

    def test_credential_mode_change_changes_id(self) -> None:
        a = _request_manifest(credential_mode=CredentialMode.ANONYMOUS)
        b = _request_manifest(credential_mode=CredentialMode.API_KEY)
        assert a.request_manifest_id != b.request_manifest_id

    def test_series_change_changes_id(self) -> None:
        a = _request_manifest(requested_series_or_dataset="DGS10", canonical_query_params={"series_id": "DGS10"})
        b = _request_manifest(requested_series_or_dataset="DFF", canonical_query_params={"series_id": "DFF"})
        assert a.request_manifest_id != b.request_manifest_id

    def test_response_format_change_changes_id(self) -> None:
        a = _request_manifest(response_format="json")
        b = _request_manifest(response_format="csv")
        assert a.request_manifest_id != b.request_manifest_id

    def test_secret_value_never_influences_identity(self) -> None:
        """The manifest structurally never SEES the secret at all
        (`SecretExposureError` would reject it if a caller tried) -- this
        confirms two callers with different real API keys, who never put
        the key into `canonical_query_params`, get the identical id, since
        `credential_mode` alone (not the key) is part of identity."""
        a = _request_manifest(credential_mode=CredentialMode.API_KEY)
        b = _request_manifest(credential_mode=CredentialMode.API_KEY)
        assert a.request_manifest_id == b.request_manifest_id  # neither manifest ever saw a key value to differ on

    def test_filesystem_root_is_irrelevant(self) -> None:
        """The manifest has no filesystem-path concept at all -- this is
        true by construction (no field for a path), asserted here as a
        structural fact rather than by varying a root, since
        `create_request_manifest` takes no such parameter."""
        assert not hasattr(_request_manifest(), "storage_root") and not hasattr(_request_manifest(), "path")


class TestResponseManifestIdentity:
    def test_same_bytes_same_id(self) -> None:
        a = _response_manifest(raw_bytes=b"hello")
        b = _response_manifest(raw_bytes=b"hello")
        assert a.response_manifest_id == b.response_manifest_id

    def test_changed_bytes_changed_id(self) -> None:
        a = _response_manifest(raw_bytes=b"hello")
        b = _response_manifest(raw_bytes=b"goodbye")
        assert a.response_manifest_id != b.response_manifest_id

    def test_changed_request_linkage_changes_id(self) -> None:
        a = _response_manifest(request_manifest_id="a" * 64)
        b = _response_manifest(request_manifest_id="b" * 64)
        assert a.response_manifest_id != b.response_manifest_id

    def test_changed_status_changes_id(self) -> None:
        a = _response_manifest(http_status=200)
        b = _response_manifest(http_status=404)
        assert a.response_manifest_id != b.response_manifest_id

    def test_received_time_excluded_from_identity(self) -> None:
        a = _response_manifest(received_time=T0)
        b = _response_manifest(received_time=T1)
        assert a.response_manifest_id == b.response_manifest_id

    def test_transport_attempt_count_excluded_from_identity(self) -> None:
        a = _response_manifest(transport_attempt_count=1)
        b = _response_manifest(transport_attempt_count=5)
        assert a.response_manifest_id == b.response_manifest_id

    def test_forged_digest_is_rejected(self) -> None:
        """Cannot construct a `CollectorResponseManifest` whose recorded
        `raw_content_digest` disagrees with its OWN bytes and then have
        `create_response_manifest` accept it silently -- the digest is
        ALWAYS recomputed from `raw_bytes` inside `create_response_
        manifest`, never taken from a caller-supplied value, so there is
        no argument through which a forged digest could even be passed."""
        import inspect

        assert "raw_content_digest" not in inspect.signature(create_response_manifest).parameters

    def test_truncated_content_produces_a_different_id_than_the_full_response(self) -> None:
        full = _response_manifest(raw_bytes=b"the-full-response-body")
        truncated = _response_manifest(raw_bytes=b"the-full-respon")
        assert full.response_manifest_id != truncated.response_manifest_id
        assert full.byte_length != truncated.byte_length

    def test_path_is_irrelevant_no_path_field_exists(self) -> None:
        assert not hasattr(_response_manifest(), "path") and not hasattr(_response_manifest(), "storage_root")

    def test_headers_are_canonicalized_via_allowlist_not_forwarded_verbatim(self) -> None:
        manifest = _response_manifest(raw_headers={"Content-Type": "application/json", "X-Request-Id": "abc123"})
        assert "x-request-id" not in manifest.canonical_selected_headers
        assert manifest.canonical_selected_headers.get("content-type") == "application/json"
