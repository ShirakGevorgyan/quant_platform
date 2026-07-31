"""Raw response cache tests (Milestone 10, Phase 4A) -- content-addressed
storage, atomic write, exact-retry idempotency, conflict detection,
path-traversal rejection, and the request-index's own explicit modeling
of "same semantic request, different response over time"."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_platform.core.exceptions import (
    CacheCorruptionError,
    CollectorResponseManifestError,
    MarketDataPathSecurityError,
    ResponseIntegrityError,
)
from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.response_manifest import CompletionStatus, create_response_manifest

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
_REQUEST_ID = "a" * 64


def _cache() -> RawResponseCache:
    return RawResponseCache(Path(tempfile.mkdtemp()))


def _manifest_and_bytes(body: bytes = b"hello", *, request_manifest_id: str = _REQUEST_ID, received_time: datetime = T0):
    manifest = create_response_manifest(
        request_manifest_id=request_manifest_id, http_status=200, raw_headers={"Content-Type": "application/json"}, raw_bytes=body,
        content_type="application/json", encoding="utf-8", completion_status=CompletionStatus.COMPLETE, received_time=received_time,
    )
    return manifest, body


class TestStoreAndRead:
    def test_first_store_then_read_round_trips(self) -> None:
        cache = _cache()
        manifest, body = _manifest_and_bytes()
        cache.store(manifest, body)
        assert cache.read_bytes(manifest.response_manifest_id) == body
        read_back = cache.read_manifest(manifest.response_manifest_id)
        assert read_back is not None and read_back.response_manifest_id == manifest.response_manifest_id

    def test_exact_retry_is_idempotent(self) -> None:
        cache = _cache()
        manifest, body = _manifest_and_bytes()
        cache.store(manifest, body)
        result = cache.store(manifest, body)  # exact same manifest+bytes again
        assert result.response_manifest_id == manifest.response_manifest_id

    def test_conflicting_write_under_same_identity_is_rejected(self) -> None:
        """Cannot literally happen through normal API use (the digest IS
        the id), so this simulates it directly: corrupt the on-disk body
        after a legitimate store, then attempt to store again."""
        cache = _cache()
        manifest, body = _manifest_and_bytes()
        cache.store(manifest, body)
        body_path = cache._body_path(manifest.response_manifest_id)
        body_path.write_bytes(b"corrupted-different-bytes")
        with pytest.raises(CacheCorruptionError):
            cache.store(manifest, body)

    def test_digest_mismatch_between_manifest_and_bytes_is_rejected_before_write(self) -> None:
        cache = _cache()
        manifest, _ = _manifest_and_bytes(body=b"hello")
        with pytest.raises(ResponseIntegrityError):
            cache.store(manifest, b"different bytes entirely")
        assert cache.read_manifest(manifest.response_manifest_id) is None  # nothing was written

    def test_partial_response_is_refused(self) -> None:
        cache = _cache()
        manifest = create_response_manifest(
            request_manifest_id=_REQUEST_ID, http_status=200, raw_headers={}, raw_bytes=b"partial",
            content_type="application/json", encoding="utf-8", completion_status=CompletionStatus.PARTIAL, received_time=T0,
        )
        with pytest.raises(CollectorResponseManifestError):
            cache.store(manifest, b"partial")

    def test_missing_bytes_on_disk_is_detected(self) -> None:
        cache = _cache()
        manifest, body = _manifest_and_bytes()
        cache.store(manifest, body)
        cache._body_path(manifest.response_manifest_id).unlink()
        with pytest.raises(CacheCorruptionError):
            cache.read_bytes(manifest.response_manifest_id)

    def test_corrupted_bytes_detected_on_read_via_rehash(self) -> None:
        cache = _cache()
        manifest, body = _manifest_and_bytes()
        cache.store(manifest, body)
        cache._body_path(manifest.response_manifest_id).write_bytes(b"tampered-after-store")
        with pytest.raises(CacheCorruptionError):
            cache.read_bytes(manifest.response_manifest_id, verify=True)

    def test_read_without_verify_skips_rehash(self) -> None:
        cache = _cache()
        manifest, body = _manifest_and_bytes()
        cache.store(manifest, body)
        cache._body_path(manifest.response_manifest_id).write_bytes(b"tampered-after-store")
        assert cache.read_bytes(manifest.response_manifest_id, verify=False) == b"tampered-after-store"

    def test_read_unknown_manifest_returns_none(self) -> None:
        cache = _cache()
        assert cache.read_manifest("f" * 64) is None

    def test_read_bytes_for_unknown_manifest_raises(self) -> None:
        cache = _cache()
        with pytest.raises(CacheCorruptionError):
            cache.read_bytes("f" * 64)


class TestPathSafety:
    def test_path_traversal_shaped_response_manifest_id_is_rejected(self) -> None:
        cache = _cache()
        for bad_id in ("../../../etc/passwd", "..\\..\\windows\\system32", "not-hex", "a" * 63, "A" * 64):
            with pytest.raises(MarketDataPathSecurityError):
                cache.read_manifest(bad_id)

    def test_path_traversal_shaped_request_manifest_id_is_rejected(self) -> None:
        cache = _cache()
        with pytest.raises(MarketDataPathSecurityError):
            cache.read_responses_for_request("../../../etc/passwd")


class TestRequestIndexAndOfflineReplay:
    def test_offline_read_by_request_manifest_id_finds_the_response(self) -> None:
        cache = _cache()
        manifest, body = _manifest_and_bytes()
        cache.store(manifest, body)
        latest = cache.read_latest_response_for_request(_REQUEST_ID)
        assert latest is not None and latest.response_manifest_id == manifest.response_manifest_id

    def test_no_responses_for_an_unknown_request_returns_none(self) -> None:
        cache = _cache()
        assert cache.read_latest_response_for_request("b" * 64) is None
        assert cache.read_responses_for_request("b" * 64) == []

    def test_same_semantic_request_receiving_different_response_over_time_is_modeled_explicitly(self) -> None:
        """The exact scenario the spec calls out: a repeated semantic
        request may legitimately receive DIFFERENT response content at a
        later `request_time` (e.g. FRED published a new observation).
        Both responses are retained; `read_latest_response_for_request`
        returns the most recently stored one; `read_responses_for_request`
        returns the full history in insertion order."""
        cache = _cache()
        manifest_v1, body_v1 = _manifest_and_bytes(body=b"vintage-1", received_time=T0)
        manifest_v2, body_v2 = _manifest_and_bytes(body=b"vintage-2", received_time=T0 + timedelta(days=1))
        assert manifest_v1.response_manifest_id != manifest_v2.response_manifest_id  # different bytes -> different identity
        cache.store(manifest_v1, body_v1)
        cache.store(manifest_v2, body_v2)

        history = cache.read_responses_for_request(_REQUEST_ID)
        assert [m.response_manifest_id for m in history] == [manifest_v1.response_manifest_id, manifest_v2.response_manifest_id]

        latest = cache.read_latest_response_for_request(_REQUEST_ID)
        assert latest is not None and latest.response_manifest_id == manifest_v2.response_manifest_id
        assert cache.read_bytes(manifest_v1.response_manifest_id) == body_v1
        assert cache.read_bytes(manifest_v2.response_manifest_id) == body_v2

    def test_storing_the_same_response_twice_does_not_duplicate_the_index(self) -> None:
        cache = _cache()
        manifest, body = _manifest_and_bytes()
        cache.store(manifest, body)
        cache.store(manifest, body)  # exact retry
        history = cache.read_responses_for_request(_REQUEST_ID)
        assert len(history) == 1


class TestConcurrentStore:
    def test_concurrent_identical_store_is_safe(self) -> None:
        import threading

        cache = _cache()
        manifest, body = _manifest_and_bytes()
        errors: list[BaseException] = []

        def _worker() -> None:
            try:
                cache.store(manifest, body)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # `experiment_lock` is FAIL-FAST (never blocks/retries), so under
        # real contention some threads legitimately lose the race and get
        # a clean `MarketDataLockError` -- the safety property under test
        # is that NOTHING ELSE ever happens (never `CacheCorruptionError`,
        # never corrupted bytes, never a duplicated index entry), not
        # that every thread silently succeeds.
        from quant_platform.core.exceptions import MarketDataLockError

        assert all(isinstance(exc, MarketDataLockError) for exc in errors), errors
        assert cache.read_bytes(manifest.response_manifest_id) == body
        assert len(cache.read_responses_for_request(_REQUEST_ID)) == 1
