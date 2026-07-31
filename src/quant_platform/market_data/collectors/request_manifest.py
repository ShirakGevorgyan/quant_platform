"""Immutable, self-verifying collector request manifest (Milestone 10,
Phase 4A). Binds everything that defines "what was requested" EXCEPT the
credential itself: `canonical_query_params`/`canonical_headers` are the
SECRET-FREE view of the actual request (the FRED `api_key` query
parameter -- or any header-based credential a future collector might
use -- is never recorded here, only `credential_mode`, a bare label).
`__post_init__` enforces this structurally, not merely by convention: it
refuses to construct a manifest whose `canonical_query_params`/
`canonical_headers` contain any KEY on a small, explicit
secret-parameter-name blocklist (`api_key`, `token`, `authorization`,
...), raising `SecretExposureError` -- so a caller who accidentally
passes the real key into the CANONICAL (manifest-bound) params, instead
of only into the actual `TransportRequest` sent over the wire, fails
loudly rather than silently persisting a secret.

`request_time` is caller-supplied OPERATIONAL metadata (when this
request was planned/submitted) and, like `creation_time` everywhere else
in this package, is EXCLUDED from identity: the same semantic request
resubmitted at a later `request_time` is still "the same request" -- see
module-level identity discussion in `docs/market_data_architecture.md`'s
Phase 4A section. What legitimately changes between two calls with an
identical `CollectorRequestManifest` is the RESPONSE (a different
`CollectorResponseManifest`), never the request's own identity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from quant_platform.core.exceptions import CollectorRequestManifestError, SecretExposureError
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)

__all__ = [
    "REQUEST_MANIFEST_KIND",
    "CollectorRequestManifest",
    "CredentialMode",
    "create_request_manifest",
]

REQUEST_MANIFEST_KIND = "collector_request_manifest"

_SECRET_PARAM_NAME_BLOCKLIST = frozenset({
    "api_key", "apikey", "api-key", "key", "token", "access_token", "authorization", "auth", "secret",
    "password", "client_secret", "x-api-key",
})


class CredentialMode(Enum):
    ANONYMOUS = "anonymous"
    API_KEY = "api_key"


def _reject_secret_shaped_keys(mapping: dict[str, str], *, field_name: str) -> None:
    offending = sorted(k for k in mapping if k.lower() in _SECRET_PARAM_NAME_BLOCKLIST)
    if offending:
        raise SecretExposureError(f"{field_name} must never carry a secret-shaped key; found: {offending}")


@dataclass(frozen=True, slots=True)
class CollectorRequestManifest:
    request_manifest_id: str
    collector_name: str
    collector_version: str
    endpoint_host: str
    endpoint_path: str
    http_method: str
    canonical_query_params: dict[str, str]
    canonical_headers: dict[str, str]
    requested_series_or_dataset: str
    requested_interval_start: datetime | None
    requested_interval_end: datetime | None
    response_format: str
    timeout_policy_id: str
    retry_policy_id: str
    rate_limit_policy_id: str
    credential_mode: CredentialMode
    schema_version: int
    request_time: datetime

    def __post_init__(self) -> None:
        require_non_empty(self.collector_name, field_name="CollectorRequestManifest.collector_name")
        require_non_empty(self.collector_version, field_name="CollectorRequestManifest.collector_version")
        require_non_empty(self.endpoint_host, field_name="CollectorRequestManifest.endpoint_host")
        require_non_empty(self.endpoint_path, field_name="CollectorRequestManifest.endpoint_path")
        if self.http_method != "GET":
            raise CollectorRequestManifestError(f"CollectorRequestManifest.http_method must be 'GET' (this phase supports no other method), got {self.http_method!r}")
        require_non_empty(self.requested_series_or_dataset, field_name="CollectorRequestManifest.requested_series_or_dataset")
        require_non_empty(self.response_format, field_name="CollectorRequestManifest.response_format")
        require_non_empty(self.timeout_policy_id, field_name="CollectorRequestManifest.timeout_policy_id")
        require_non_empty(self.retry_policy_id, field_name="CollectorRequestManifest.retry_policy_id")
        require_non_empty(self.rate_limit_policy_id, field_name="CollectorRequestManifest.rate_limit_policy_id")
        if self.schema_version < 1:
            raise CollectorRequestManifestError(f"CollectorRequestManifest.schema_version must be >= 1, got {self.schema_version}")
        require_tz_aware(self.request_time, field_name="CollectorRequestManifest.request_time")
        if self.requested_interval_start is not None:
            require_tz_aware(self.requested_interval_start, field_name="CollectorRequestManifest.requested_interval_start")
        if self.requested_interval_end is not None:
            require_tz_aware(self.requested_interval_end, field_name="CollectorRequestManifest.requested_interval_end")
        if self.requested_interval_start is not None and self.requested_interval_end is not None and self.requested_interval_end < self.requested_interval_start:
            raise CollectorRequestManifestError(
                f"CollectorRequestManifest.requested_interval_end ({self.requested_interval_end}) must be >= "
                f"requested_interval_start ({self.requested_interval_start})"
            )
        _reject_secret_shaped_keys(self.canonical_query_params, field_name="CollectorRequestManifest.canonical_query_params")
        _reject_secret_shaped_keys(self.canonical_headers, field_name="CollectorRequestManifest.canonical_headers")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": REQUEST_MANIFEST_KIND, "request_manifest_id": self.request_manifest_id, "collector_name": self.collector_name,
            "collector_version": self.collector_version, "endpoint_host": self.endpoint_host, "endpoint_path": self.endpoint_path,
            "http_method": self.http_method, "canonical_query_params": dict(self.canonical_query_params),
            "canonical_headers": dict(self.canonical_headers), "requested_series_or_dataset": self.requested_series_or_dataset,
            "requested_interval_start": (None if self.requested_interval_start is None else serialize_timestamp(self.requested_interval_start, field_name="requested_interval_start")),
            "requested_interval_end": (None if self.requested_interval_end is None else serialize_timestamp(self.requested_interval_end, field_name="requested_interval_end")),
            "response_format": self.response_format, "timeout_policy_id": self.timeout_policy_id, "retry_policy_id": self.retry_policy_id,
            "rate_limit_policy_id": self.rate_limit_policy_id, "credential_mode": self.credential_mode.value,
            "schema_version": self.schema_version, "request_time": serialize_timestamp(self.request_time, field_name="request_time"),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["request_manifest_id"]
        del payload["request_time"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CollectorRequestManifest:
        from quant_platform.ml.persistence import as_json_dict

        raw_start = raw.get("requested_interval_start")
        raw_end = raw.get("requested_interval_end")
        return cls(
            request_manifest_id=str(raw["request_manifest_id"]), collector_name=str(raw["collector_name"]),
            collector_version=str(raw["collector_version"]), endpoint_host=str(raw["endpoint_host"]), endpoint_path=str(raw["endpoint_path"]),
            http_method=str(raw["http_method"]),
            canonical_query_params={str(k): str(v) for k, v in as_json_dict(raw["canonical_query_params"], field_name="canonical_query_params").items()},
            canonical_headers={str(k): str(v) for k, v in as_json_dict(raw["canonical_headers"], field_name="canonical_headers").items()},
            requested_series_or_dataset=str(raw["requested_series_or_dataset"]),
            requested_interval_start=(None if raw_start is None else deserialize_timestamp(raw_start, field_name="requested_interval_start")),
            requested_interval_end=(None if raw_end is None else deserialize_timestamp(raw_end, field_name="requested_interval_end")),
            response_format=str(raw["response_format"]), timeout_policy_id=str(raw["timeout_policy_id"]),
            retry_policy_id=str(raw["retry_policy_id"]), rate_limit_policy_id=str(raw["rate_limit_policy_id"]),
            credential_mode=CredentialMode(raw["credential_mode"]), schema_version=int(str(raw["schema_version"])),
            request_time=deserialize_timestamp(raw["request_time"], field_name="request_time"),
        )


def create_request_manifest(
    *, collector_name: str, collector_version: str, endpoint_host: str, endpoint_path: str, canonical_query_params: dict[str, str],
    canonical_headers: dict[str, str], requested_series_or_dataset: str, response_format: str, timeout_policy_id: str, retry_policy_id: str,
    rate_limit_policy_id: str, credential_mode: CredentialMode, request_time: datetime, requested_interval_start: datetime | None = None,
    requested_interval_end: datetime | None = None, http_method: str = "GET", schema_version: int = 1,
) -> CollectorRequestManifest:
    provisional = CollectorRequestManifest(
        request_manifest_id="0" * 64, collector_name=collector_name, collector_version=collector_version, endpoint_host=endpoint_host,
        endpoint_path=endpoint_path, http_method=http_method, canonical_query_params=canonical_query_params, canonical_headers=canonical_headers,
        requested_series_or_dataset=requested_series_or_dataset, requested_interval_start=requested_interval_start,
        requested_interval_end=requested_interval_end, response_format=response_format, timeout_policy_id=timeout_policy_id,
        retry_policy_id=retry_policy_id, rate_limit_policy_id=rate_limit_policy_id, credential_mode=credential_mode,
        schema_version=schema_version, request_time=request_time,
    )
    request_manifest_id = compute_content_id(REQUEST_MANIFEST_KIND, provisional.to_identity_payload())
    return CollectorRequestManifest(
        request_manifest_id=request_manifest_id, collector_name=collector_name, collector_version=collector_version, endpoint_host=endpoint_host,
        endpoint_path=endpoint_path, http_method=http_method, canonical_query_params=canonical_query_params, canonical_headers=canonical_headers,
        requested_series_or_dataset=requested_series_or_dataset, requested_interval_start=requested_interval_start,
        requested_interval_end=requested_interval_end, response_format=response_format, timeout_policy_id=timeout_policy_id,
        retry_policy_id=retry_policy_id, rate_limit_policy_id=rate_limit_policy_id, credential_mode=credential_mode,
        schema_version=schema_version, request_time=request_time,
    )
