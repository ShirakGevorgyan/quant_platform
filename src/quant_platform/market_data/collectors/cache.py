"""Repository-consistent, append-only raw response cache (Milestone 10,
Phase 4A). Storage layout:
`{storage_root}/collectors/raw_responses/{response_manifest_id}/manifest.json`
+ `.../body.bin`, plus an append-only per-request index at
`{storage_root}/collectors/request_index/{request_manifest_id}/responses.jsonl`
recording, in insertion order, every DISTINCT `response_manifest_id`
this SEMANTIC request has ever received -- the explicit distinction the
specification asks for: `response_manifest_id` is the ACTUAL RESPONSE's
own content identity (changes if the bytes differ), `request_manifest_id`
is the SEMANTIC REQUEST's identity (stable across repeated calls, even
if the server legitimately returns different content at a later
`request_time` -- e.g. FRED publishing a new observation since the last
call). `read_latest_response_for_request` is the CACHE LOOKUP POLICY: by
default "most recently stored," which a caller wanting offline replay of
one specific historical fetch instead pins by response_manifest_id
directly.

Every `response_manifest_id`/`request_manifest_id` used as a path
component is validated as EXACTLY 64 lowercase hex characters (the shape
`compute_content_id`'s sha256 output always has) before touching the
filesystem -- path traversal is structurally unreachable, never merely
discouraged."""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from quant_platform.core.exceptions import (
    CacheCorruptionError,
    CollectorResponseManifestError,
    ExperimentLockError,
    MarketDataLockError,
    MarketDataPathSecurityError,
    MarketDataPersistenceError,
    ResponseIntegrityError,
)
from quant_platform.core.json import canonical_json_bytes, write_json_atomic
from quant_platform.market_data.collectors.response_manifest import (
    CollectorResponseManifest,
    CompletionStatus,
    compute_raw_content_digest,
)
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.persistence import parse_json_strict

__all__ = ["RawResponseCache"]

_SAFE_HEX_ID = re.compile(r"^[0-9a-f]{64}$")


def _validate_hex_id(value: str, *, field_name: str) -> str:
    if not _SAFE_HEX_ID.match(value):
        raise MarketDataPathSecurityError(f"{field_name} must be a 64-char lowercase sha256 hex digest to be used as a path component, got {value!r}")
    return value


@contextmanager
def _cache_lock(lock_path: Path) -> Iterator[None]:
    try:
        with experiment_lock(lock_path):
            yield
    except ExperimentLockError as exc:
        raise MarketDataLockError(f"Could not acquire raw-response cache lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
    except OSError as exc:
        raise MarketDataLockError(f"Raw-response cache lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp_path.write_bytes(data)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


class RawResponseCache:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _response_dir(self, response_manifest_id: str) -> Path:
        return self._root / "collectors" / "raw_responses" / _validate_hex_id(response_manifest_id, field_name="response_manifest_id")

    def _manifest_path(self, response_manifest_id: str) -> Path:
        return self._response_dir(response_manifest_id) / "manifest.json"

    def _body_path(self, response_manifest_id: str) -> Path:
        return self._response_dir(response_manifest_id) / "body.bin"

    def _lock_path(self, response_manifest_id: str) -> Path:
        return self._response_dir(response_manifest_id) / ".cache.lock"

    def _request_index_dir(self, request_manifest_id: str) -> Path:
        return self._root / "collectors" / "request_index" / _validate_hex_id(request_manifest_id, field_name="request_manifest_id")

    def _request_index_path(self, request_manifest_id: str) -> Path:
        return self._request_index_dir(request_manifest_id) / "responses.jsonl"

    def _request_index_lock_path(self, request_manifest_id: str) -> Path:
        return self._request_index_dir(request_manifest_id) / ".index.lock"

    # ----------------------------------------------------------------
    # Store.
    # ----------------------------------------------------------------
    def store(self, manifest: CollectorResponseManifest, raw_bytes: bytes) -> CollectorResponseManifest:
        """Idempotent for an EXACT retry (same `response_manifest_id`,
        byte-identical body already on disk); raises `CacheCorruptionError`
        for a CONFLICTING write (same `response_manifest_id`, DIFFERENT
        bytes -- reachable only via a caller bug or a hash collision,
        never silently overwritten). Raises `ResponseIntegrityError` if
        `manifest.raw_content_digest` does not match `raw_bytes` BEFORE
        anything is written. Refuses to persist a `PARTIAL` manifest --
        "partial responses must never be marked complete" is enforced
        HERE, at the only point a response becomes durable."""
        if manifest.completion_status is not CompletionStatus.COMPLETE:
            raise CollectorResponseManifestError(
                f"refusing to cache a {manifest.completion_status.value!r} response as durable evidence -- only COMPLETE responses may be stored"
            )
        actual_digest = compute_raw_content_digest(raw_bytes)
        if actual_digest != manifest.raw_content_digest:
            raise ResponseIntegrityError(
                f"raw_bytes digest {actual_digest!r} does not match manifest.raw_content_digest {manifest.raw_content_digest!r} for "
                f"response_manifest_id {manifest.response_manifest_id!r}"
            )
        if len(raw_bytes) != manifest.byte_length:
            raise ResponseIntegrityError(
                f"raw_bytes length {len(raw_bytes)} does not match manifest.byte_length {manifest.byte_length} for "
                f"response_manifest_id {manifest.response_manifest_id!r}"
            )

        lock_path = self._lock_path(manifest.response_manifest_id)
        self._response_dir(manifest.response_manifest_id).mkdir(parents=True, exist_ok=True)
        with _cache_lock(lock_path):
            body_path = self._body_path(manifest.response_manifest_id)
            if body_path.is_file():
                existing_bytes = body_path.read_bytes()
                existing_digest = compute_raw_content_digest(existing_bytes)
                if existing_digest != manifest.raw_content_digest:
                    raise CacheCorruptionError(
                        f"response_manifest_id {manifest.response_manifest_id!r} already has DIFFERENT bytes on disk "
                        f"(existing digest {existing_digest!r} != new digest {manifest.raw_content_digest!r}) -- refusing to overwrite"
                    )
                return manifest  # idempotent no-op: byte-identical exact retry
            _atomic_write_bytes(body_path, raw_bytes)
            write_json_atomic(self._manifest_path(manifest.response_manifest_id), manifest.to_json_dict())

        self._append_to_request_index(manifest.request_manifest_id, manifest.response_manifest_id)
        return manifest

    def _append_to_request_index(self, request_manifest_id: str, response_manifest_id: str) -> None:
        lock_path = self._request_index_lock_path(request_manifest_id)
        self._request_index_dir(request_manifest_id).mkdir(parents=True, exist_ok=True)
        with _cache_lock(lock_path):
            existing = self._read_request_index(request_manifest_id)
            if response_manifest_id in existing:
                return
            path = self._request_index_path(request_manifest_id)
            with path.open("ab") as handle:
                handle.write(canonical_json_bytes({"response_manifest_id": response_manifest_id}))
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _read_request_index(self, request_manifest_id: str) -> list[str]:
        path = self._request_index_path(request_manifest_id)
        if not path.is_file():
            return []
        ids: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted request index line for request_manifest_id {request_manifest_id!r}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted request index line for request_manifest_id {request_manifest_id!r}: expected a JSON object")
            ids.append(str(raw["response_manifest_id"]))
        return ids

    # ----------------------------------------------------------------
    # Read.
    # ----------------------------------------------------------------
    def read_manifest(self, response_manifest_id: str) -> CollectorResponseManifest | None:
        path = self._manifest_path(response_manifest_id)
        if not path.is_file():
            return None
        try:
            raw = parse_json_strict(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise MarketDataPersistenceError(f"Corrupted response manifest for response_manifest_id {response_manifest_id!r}: {exc}") from exc
        if not isinstance(raw, dict):
            raise MarketDataPersistenceError(f"Corrupted response manifest for response_manifest_id {response_manifest_id!r}: expected a JSON object")
        return CollectorResponseManifest.from_json_dict(raw)

    def read_bytes(self, response_manifest_id: str, *, verify: bool = True) -> bytes:
        """Re-hashes on every read by default (`verify=True`) --
        independent integrity verification is the NORMAL path, not an
        opt-in extra. Raises `CacheCorruptionError` if bytes are missing
        for a manifest that claims to exist, or if re-hashing does not
        reproduce the recorded digest."""
        manifest = self.read_manifest(response_manifest_id)
        if manifest is None:
            raise CacheCorruptionError(f"no cached manifest for response_manifest_id {response_manifest_id!r}")
        body_path = self._body_path(response_manifest_id)
        if not body_path.is_file():
            raise CacheCorruptionError(f"manifest exists but body.bin is missing for response_manifest_id {response_manifest_id!r}")
        data = body_path.read_bytes()
        if verify:
            actual_digest = compute_raw_content_digest(data)
            if actual_digest != manifest.raw_content_digest:
                raise CacheCorruptionError(
                    f"re-hash mismatch for response_manifest_id {response_manifest_id!r}: on-disk digest {actual_digest!r} != "
                    f"manifest.raw_content_digest {manifest.raw_content_digest!r}"
                )
            if len(data) != manifest.byte_length:
                raise CacheCorruptionError(
                    f"re-read length mismatch for response_manifest_id {response_manifest_id!r}: on-disk length {len(data)} != "
                    f"manifest.byte_length {manifest.byte_length}"
                )
        return data

    def read_responses_for_request(self, request_manifest_id: str) -> list[CollectorResponseManifest]:
        """Every DISTINCT response this semantic request has ever
        received, in the order first stored."""
        manifests: list[CollectorResponseManifest] = []
        for response_manifest_id in self._read_request_index(request_manifest_id):
            manifest = self.read_manifest(response_manifest_id)
            if manifest is not None:
                manifests.append(manifest)
        return manifests

    def read_latest_response_for_request(self, request_manifest_id: str) -> CollectorResponseManifest | None:
        responses = self.read_responses_for_request(request_manifest_id)
        return responses[-1] if responses else None
