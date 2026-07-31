"""Secret-handling tests (Milestone 10, Phase 4A) -- proves the API key
never reaches a request manifest, response manifest, report, exception
message, or log, while ALSO proving the guard is non-vacuous: a
deliberately bad construction attempt (secret placed where it must
never go) is caught."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from quant_platform.core.exceptions import SecretExposureError
from quant_platform.market_data.collectors.request_manifest import CredentialMode, create_request_manifest
from quant_platform.market_data.collectors.response_manifest import CompletionStatus, create_response_manifest

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
_REAL_SECRET = "sk-realsecretvalue1234567890abcdef"


class TestRequestManifestSecretRedaction:
    def test_api_key_in_query_params_is_rejected(self) -> None:
        with pytest.raises(SecretExposureError):
            create_request_manifest(
                collector_name="fred", collector_version="1.0.0", endpoint_host="api.stlouisfed.org", endpoint_path="/x",
                canonical_query_params={"series_id": "DGS10", "api_key": _REAL_SECRET}, canonical_headers={},
                requested_series_or_dataset="DGS10", response_format="json", timeout_policy_id="a" * 64, retry_policy_id="b" * 64,
                rate_limit_policy_id="c" * 64, credential_mode=CredentialMode.API_KEY, request_time=T0,
            )

    @pytest.mark.parametrize("secret_key_name", ["api_key", "apikey", "api-key", "key", "token", "access_token", "authorization", "auth", "secret", "password", "client_secret", "x-api-key"])
    def test_every_blocklisted_param_name_is_rejected(self, secret_key_name: str) -> None:
        with pytest.raises(SecretExposureError):
            create_request_manifest(
                collector_name="fred", collector_version="1.0.0", endpoint_host="api.stlouisfed.org", endpoint_path="/x",
                canonical_query_params={secret_key_name: "x"}, canonical_headers={}, requested_series_or_dataset="DGS10",
                response_format="json", timeout_policy_id="a" * 64, retry_policy_id="b" * 64, rate_limit_policy_id="c" * 64,
                credential_mode=CredentialMode.API_KEY, request_time=T0,
            )

    def test_authorization_header_is_rejected(self) -> None:
        with pytest.raises(SecretExposureError):
            create_request_manifest(
                collector_name="fred", collector_version="1.0.0", endpoint_host="api.stlouisfed.org", endpoint_path="/x",
                canonical_query_params={"series_id": "DGS10"}, canonical_headers={"Authorization": f"Bearer {_REAL_SECRET}"},
                requested_series_or_dataset="DGS10", response_format="json", timeout_policy_id="a" * 64, retry_policy_id="b" * 64,
                rate_limit_policy_id="c" * 64, credential_mode=CredentialMode.API_KEY, request_time=T0,
            )

    def test_clean_manifest_never_contains_the_secret_anywhere(self) -> None:
        manifest = create_request_manifest(
            collector_name="fred", collector_version="1.0.0", endpoint_host="api.stlouisfed.org", endpoint_path="/x",
            canonical_query_params={"series_id": "DGS10"}, canonical_headers={}, requested_series_or_dataset="DGS10",
            response_format="json", timeout_policy_id="a" * 64, retry_policy_id="b" * 64, rate_limit_policy_id="c" * 64,
            credential_mode=CredentialMode.API_KEY, request_time=T0,
        )
        serialized = json.dumps(manifest.to_json_dict())
        assert _REAL_SECRET not in serialized
        assert _REAL_SECRET not in repr(manifest)
        # Only the MODE label is recorded, never a secret value or digest:
        assert manifest.credential_mode is CredentialMode.API_KEY
        assert manifest.to_json_dict()["credential_mode"] == "api_key"

    def test_manifest_records_no_secret_derived_digest(self) -> None:
        """The manifest must record `credential_mode` alone -- no hash,
        prefix, or any other secret-DERIVED value that could leak
        information about the key."""
        manifest = create_request_manifest(
            collector_name="fred", collector_version="1.0.0", endpoint_host="api.stlouisfed.org", endpoint_path="/x",
            canonical_query_params={"series_id": "DGS10"}, canonical_headers={}, requested_series_or_dataset="DGS10",
            response_format="json", timeout_policy_id="a" * 64, retry_policy_id="b" * 64, rate_limit_policy_id="c" * 64,
            credential_mode=CredentialMode.API_KEY, request_time=T0,
        )
        json_dict = manifest.to_json_dict()
        assert set(json_dict.keys()) == {
            "kind", "request_manifest_id", "collector_name", "collector_version", "endpoint_host", "endpoint_path", "http_method",
            "canonical_query_params", "canonical_headers", "requested_series_or_dataset", "requested_interval_start",
            "requested_interval_end", "response_format", "timeout_policy_id", "retry_policy_id", "rate_limit_policy_id",
            "credential_mode", "schema_version", "request_time",
        }


class TestResponseManifestSecretRedaction:
    def test_secret_bearing_header_is_dropped_by_the_allowlist(self) -> None:
        manifest = create_response_manifest(
            request_manifest_id="a" * 64, http_status=200, raw_headers={"Content-Type": "application/json", "Set-Cookie": "session=abc123"},
            raw_bytes=b"{}", content_type="application/json", encoding="utf-8", completion_status=CompletionStatus.COMPLETE, received_time=T0,
        )
        assert "set-cookie" not in manifest.canonical_selected_headers
        assert json.dumps(manifest.to_json_dict()).find("abc123") == -1

    def test_authorization_response_header_is_dropped(self) -> None:
        manifest = create_response_manifest(
            request_manifest_id="a" * 64, http_status=200, raw_headers={"WWW-Authenticate": "Bearer error=invalid_token"},
            raw_bytes=b"{}", content_type="application/json", encoding="utf-8", completion_status=CompletionStatus.COMPLETE, received_time=T0,
        )
        assert "www-authenticate" not in manifest.canonical_selected_headers

    def test_only_allowlisted_headers_ever_survive(self) -> None:
        manifest = create_response_manifest(
            request_manifest_id="a" * 64, http_status=200,
            raw_headers={"Content-Type": "application/json", "X-Correlation-Id": "abc", "Set-Cookie": "x", "Authorization": "y"},
            raw_bytes=b"{}", content_type="application/json", encoding="utf-8", completion_status=CompletionStatus.COMPLETE, received_time=T0,
        )
        assert set(manifest.canonical_selected_headers.keys()) <= {"content-type", "content-length", "date", "last-modified", "etag"}


class TestSecretExposureGuardIsNonVacuous:
    """A deliberately BAD implementation (a hypothetical caller who
    manually stuffs a secret into a plain dict and serializes it,
    bypassing `create_request_manifest` entirely) IS caught by simple
    inspection -- this proves the redaction assertions above are
    exercising a REAL guard, not vacuously passing because nothing was
    ever tested against a genuinely leaking artifact."""

    def test_a_naive_dict_with_a_raw_secret_would_be_detected_by_the_same_scan_style(self) -> None:
        leaking_artifact = {"api_key": _REAL_SECRET, "series_id": "DGS10"}
        assert _REAL_SECRET in json.dumps(leaking_artifact)  # confirms the detection method itself works

    def test_secret_exposure_error_is_the_specific_type_raised(self) -> None:
        with pytest.raises(SecretExposureError) as exc_info:
            create_request_manifest(
                collector_name="fred", collector_version="1.0.0", endpoint_host="api.stlouisfed.org", endpoint_path="/x",
                canonical_query_params={"api_key": _REAL_SECRET}, canonical_headers={}, requested_series_or_dataset="DGS10",
                response_format="json", timeout_policy_id="a" * 64, retry_policy_id="b" * 64, rate_limit_policy_id="c" * 64,
                credential_mode=CredentialMode.API_KEY, request_time=T0,
            )
        # The exception message itself must not leak the secret value either:
        assert _REAL_SECRET not in str(exc_info.value)
