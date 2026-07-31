"""Immutable, self-verifying collector response manifest (Milestone 10,
Phase 4A). Binds a content digest of the RAW response bytes -- never a
filesystem path -- plus linkage back to the request that produced it.

`canonicalize_response_headers` is an ALLOWLIST, not a blocklist: only a
small, explicit set of known-safe headers (`content-type`,
`content-length`, `date`, `last-modified`, `etag`) is ever kept: any
header NOT on this list is dropped by construction, including any future
header this module has never seen -- a blocklist would need to
anticipate every possible secret-carrying header name in advance
(`Authorization`, `Set-Cookie`, a vendor-specific session header, ...);
an allowlist cannot leak an unanticipated one. `__post_init__` still
defensively re-checks `canonical_selected_headers` against the same
secret-shaped-key blocklist `request_manifest.py` uses, as a second,
independent layer.

`received_time` and `transport_attempt_count` are OPERATIONAL (like
`creation_time` elsewhere in this package) and excluded from identity:
identical response BYTES, for the same request, received on a different
attempt count or at a different wall-clock moment, are still the SAME
response. `completion_status` distinguishes a fully-received body from a
partial one; `cache.py`'s `RawResponseCache.store` -- not this module --
is what refuses to durably commit a `PARTIAL` manifest as if it were
complete (see that module's own docstring)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from quant_platform.core.exceptions import CollectorResponseManifestError, SecretExposureError
from quant_platform.core.json import sha256_hex_bytes
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)

__all__ = [
    "RESPONSE_MANIFEST_KIND",
    "CollectorResponseManifest",
    "CompletionStatus",
    "canonicalize_response_headers",
    "compute_raw_content_digest",
    "create_response_manifest",
]

RESPONSE_MANIFEST_KIND = "collector_response_manifest"

_ALLOWED_RESPONSE_HEADER_NAMES = frozenset({"content-type", "content-length", "date", "last-modified", "etag"})
_SECRET_HEADER_NAME_BLOCKLIST = frozenset({"authorization", "set-cookie", "x-api-key", "www-authenticate", "proxy-authenticate"})


class CompletionStatus(Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"


def canonicalize_response_headers(raw_headers: dict[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in raw_headers.items() if k.lower() in _ALLOWED_RESPONSE_HEADER_NAMES}


def compute_raw_content_digest(data: bytes) -> str:
    return sha256_hex_bytes(data)


@dataclass(frozen=True, slots=True)
class CollectorResponseManifest:
    response_manifest_id: str
    request_manifest_id: str
    http_status: int
    canonical_selected_headers: dict[str, str]
    byte_length: int
    raw_content_digest: str
    content_type: str | None
    encoding: str | None
    completion_status: CompletionStatus
    pagination_page_index: int | None
    pagination_next_token: str | None
    received_time: datetime
    transport_attempt_count: int

    def __post_init__(self) -> None:
        require_non_empty(self.request_manifest_id, field_name="CollectorResponseManifest.request_manifest_id")
        if not (100 <= self.http_status <= 599):
            raise CollectorResponseManifestError(f"CollectorResponseManifest.http_status must be in [100, 599], got {self.http_status}")
        if self.byte_length < 0:
            raise CollectorResponseManifestError(f"CollectorResponseManifest.byte_length must be >= 0, got {self.byte_length}")
        if len(self.raw_content_digest) != 64:
            raise CollectorResponseManifestError(f"CollectorResponseManifest.raw_content_digest must be a 64-char sha256 hex digest, got {self.raw_content_digest!r}")
        if self.transport_attempt_count < 1:
            raise CollectorResponseManifestError(f"CollectorResponseManifest.transport_attempt_count must be >= 1, got {self.transport_attempt_count}")
        if self.pagination_page_index is not None and self.pagination_page_index < 0:
            raise CollectorResponseManifestError(f"CollectorResponseManifest.pagination_page_index must be >= 0 or None, got {self.pagination_page_index}")
        require_tz_aware(self.received_time, field_name="CollectorResponseManifest.received_time")
        offending = sorted(k for k in self.canonical_selected_headers if k.lower() in _SECRET_HEADER_NAME_BLOCKLIST)
        if offending:
            raise SecretExposureError(f"CollectorResponseManifest.canonical_selected_headers must never carry a secret-shaped key; found: {offending}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": RESPONSE_MANIFEST_KIND, "response_manifest_id": self.response_manifest_id, "request_manifest_id": self.request_manifest_id,
            "http_status": self.http_status, "canonical_selected_headers": dict(self.canonical_selected_headers), "byte_length": self.byte_length,
            "raw_content_digest": self.raw_content_digest, "content_type": self.content_type, "encoding": self.encoding,
            "completion_status": self.completion_status.value, "pagination_page_index": self.pagination_page_index,
            "pagination_next_token": self.pagination_next_token, "received_time": serialize_timestamp(self.received_time, field_name="received_time"),
            "transport_attempt_count": self.transport_attempt_count,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["response_manifest_id"]
        del payload["received_time"]
        del payload["transport_attempt_count"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CollectorResponseManifest:
        from quant_platform.ml.persistence import as_json_dict

        raw_page_index = raw.get("pagination_page_index")
        return cls(
            response_manifest_id=str(raw["response_manifest_id"]), request_manifest_id=str(raw["request_manifest_id"]),
            http_status=int(str(raw["http_status"])),
            canonical_selected_headers={str(k): str(v) for k, v in as_json_dict(raw["canonical_selected_headers"], field_name="canonical_selected_headers").items()},
            byte_length=int(str(raw["byte_length"])), raw_content_digest=str(raw["raw_content_digest"]),
            content_type=(None if raw.get("content_type") is None else str(raw["content_type"])),
            encoding=(None if raw.get("encoding") is None else str(raw["encoding"])),
            completion_status=CompletionStatus(raw["completion_status"]),
            pagination_page_index=(None if raw_page_index is None else int(str(raw_page_index))),
            pagination_next_token=(None if raw.get("pagination_next_token") is None else str(raw["pagination_next_token"])),
            received_time=deserialize_timestamp(raw["received_time"], field_name="received_time"),
            transport_attempt_count=int(str(raw["transport_attempt_count"])),
        )


def create_response_manifest(
    *, request_manifest_id: str, http_status: int, raw_headers: dict[str, str], raw_bytes: bytes, content_type: str | None, encoding: str | None,
    completion_status: CompletionStatus, received_time: datetime, transport_attempt_count: int = 1, pagination_page_index: int | None = None,
    pagination_next_token: str | None = None,
) -> CollectorResponseManifest:
    canonical_headers = canonicalize_response_headers(raw_headers)
    digest = compute_raw_content_digest(raw_bytes)
    provisional = CollectorResponseManifest(
        response_manifest_id="0" * 64, request_manifest_id=request_manifest_id, http_status=http_status, canonical_selected_headers=canonical_headers,
        byte_length=len(raw_bytes), raw_content_digest=digest, content_type=content_type, encoding=encoding, completion_status=completion_status,
        pagination_page_index=pagination_page_index, pagination_next_token=pagination_next_token, received_time=received_time,
        transport_attempt_count=transport_attempt_count,
    )
    response_manifest_id = compute_content_id(RESPONSE_MANIFEST_KIND, provisional.to_identity_payload())
    return CollectorResponseManifest(
        response_manifest_id=response_manifest_id, request_manifest_id=request_manifest_id, http_status=http_status,
        canonical_selected_headers=canonical_headers, byte_length=len(raw_bytes), raw_content_digest=digest, content_type=content_type,
        encoding=encoding, completion_status=completion_status, pagination_page_index=pagination_page_index,
        pagination_next_token=pagination_next_token, received_time=received_time, transport_attempt_count=transport_attempt_count,
    )
