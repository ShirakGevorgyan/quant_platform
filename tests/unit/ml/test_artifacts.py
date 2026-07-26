from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from quant_platform.core.exceptions import ArtifactCorruptionError, ArtifactNotFoundError, PathSecurityError
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.models import ArtifactCategory


class TestWriteAndRead:
    def test_round_trip(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        data = b"hello model bytes"
        ref = store.write_artifact(data, category=ArtifactCategory.MODEL)
        assert store.read_artifact(ref.content_hash) == data
        assert ref.size_bytes == len(data)
        assert ref.category is ArtifactCategory.MODEL

    def test_content_hash_is_sha256_of_bytes(self, tmp_path: Path) -> None:
        import hashlib

        store = MLArtifactStore(tmp_path)
        data = b"some bytes"
        ref = store.write_artifact(data, category=ArtifactCategory.METRICS)
        assert ref.content_hash == hashlib.sha256(data).hexdigest()

    def test_artifact_exists(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        ref = store.write_artifact(b"data", category=ArtifactCategory.MODEL)
        assert store.artifact_exists(ref.content_hash)
        assert not store.artifact_exists("0" * 64)

    def test_duplicate_write_is_idempotent_no_op(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        data = b"duplicate content"
        ref1 = store.write_artifact(data, category=ArtifactCategory.MODEL)
        ref2 = store.write_artifact(data, category=ArtifactCategory.MODEL)
        assert ref1 == ref2

    def test_empty_bytes_can_be_written(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        ref = store.write_artifact(b"", category=ArtifactCategory.LOGS)
        assert store.read_artifact(ref.content_hash) == b""


class TestCategoryConflict:
    def test_same_content_different_category_raises(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        data = b"identical bytes"
        store.write_artifact(data, category=ArtifactCategory.MODEL)
        with pytest.raises(ArtifactCorruptionError, match="already stored under category"):
            store.write_artifact(data, category=ArtifactCategory.METRICS)


class TestReadFailures:
    def test_unknown_hash_raises_not_found(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        with pytest.raises(ArtifactNotFoundError):
            store.read_artifact("0" * 64)

    def test_malformed_hash_raises_corruption_error(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        with pytest.raises(ArtifactCorruptionError):
            store.read_artifact("not-a-valid-hash")

    def test_tampered_content_detected(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        ref = store.write_artifact(b"original content", category=ArtifactCategory.MODEL)
        content_path = store._content_path(ref.content_hash)
        content_path.write_bytes(b"TAMPERED")
        with pytest.raises(ArtifactCorruptionError):
            store.read_artifact(ref.content_hash)

    def test_missing_metadata_sidecar_detected(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        ref = store.write_artifact(b"content", category=ArtifactCategory.MODEL)
        store._metadata_path(ref.content_hash).unlink()
        with pytest.raises(ArtifactCorruptionError, match="metadata sidecar is missing"):
            store.read_artifact(ref.content_hash)

    def test_size_mismatch_in_sidecar_detected(self, tmp_path: Path) -> None:
        import json

        store = MLArtifactStore(tmp_path)
        ref = store.write_artifact(b"content", category=ArtifactCategory.MODEL)
        metadata_path = store._metadata_path(ref.content_hash)
        raw = json.loads(metadata_path.read_text())
        raw["size_bytes"] = 99999
        metadata_path.write_text(json.dumps(raw))
        with pytest.raises(ArtifactCorruptionError, match="size_bytes"):
            store.read_artifact(ref.content_hash)


class TestPathTraversalAndSymlinkProtection:
    def test_path_traversal_like_hash_rejected(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        with pytest.raises(ArtifactCorruptionError):
            store.read_artifact("../../../etc/passwd" + "0" * 44)

    def test_content_path_stays_within_root(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        ref = store.write_artifact(b"data", category=ArtifactCategory.MODEL)
        content_path = store._content_path(ref.content_hash)
        assert tmp_path.resolve() in content_path.resolve().parents

    @pytest.mark.skipif(__import__("sys").platform == "win32", reason="symlink creation requires elevated privileges on Windows")
    def test_symlink_escape_blocked(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside_target"
        outside.mkdir(exist_ok=True)
        store_root = tmp_path / "store"
        store = MLArtifactStore(store_root)
        content_dir = store_root / "content" / "ab"
        content_dir.mkdir(parents=True, exist_ok=True)
        escaping_hash = "ab" + "0" * 62
        symlink_path = content_dir / escaping_hash
        symlink_path.symlink_to(outside)
        with pytest.raises(PathSecurityError):
            store.read_artifact(escaping_hash)


class TestRecoveryFromInterruptedWrite:
    def test_content_present_without_sidecar_is_recovered_on_next_write(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        data = b"recovered content"
        ref = store.write_artifact(data, category=ArtifactCategory.MODEL)
        # Simulate a crash between content-write and metadata-write:
        store._metadata_path(ref.content_hash).unlink()
        # Re-writing the same content (same category) should recover, not error.
        ref2 = store.write_artifact(data, category=ArtifactCategory.MODEL)
        assert ref2.content_hash == ref.content_hash
        assert store.read_artifact(ref.content_hash) == data


class TestConcurrentWrites:
    def test_concurrent_identical_writes_are_safe(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        data = b"concurrent content" * 100
        errors: list[BaseException] = []

        def write() -> None:
            try:
                store.write_artifact(data, category=ArtifactCategory.MODEL)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        expected_hash = MLArtifactStore(tmp_path).write_artifact(data, category=ArtifactCategory.MODEL).content_hash
        assert store.read_artifact(expected_hash) == data


class TestGenuineWriteFailures:
    """Distinguishes a genuine write failure from the "lost a race to an
    equivalent writer" case the store otherwise tolerates -- if the
    destination truly never gets written, the original error must still
    propagate."""

    def test_content_write_failure_cleans_up_and_reraises(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        with patch("pathlib.Path.write_bytes", side_effect=OSError("disk full")), pytest.raises(OSError, match="disk full"):
            store.write_artifact(b"data", category=ArtifactCategory.MODEL)
        assert list(tmp_path.rglob(".*.tmp")) == []

    def test_content_rename_genuine_failure_reraises_when_destination_missing(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        with patch("pathlib.Path.replace", side_effect=OSError("genuine failure, not a race")), pytest.raises(OSError, match="genuine failure"):
            store.write_artifact(b"data", category=ArtifactCategory.MODEL)

    def test_metadata_rename_genuine_failure_reraises_when_destination_missing(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        real_replace = Path.replace

        def fake_replace(self: Path, target: Path) -> Path:
            if str(self).endswith(".json.tmp") or ".json." in self.name:
                raise OSError("genuine metadata write failure")
            return real_replace(self, target)

        with patch("pathlib.Path.replace", fake_replace), pytest.raises(OSError, match="genuine metadata write failure"):
            store.write_artifact(b"data", category=ArtifactCategory.MODEL)


def test_artifact_reference_round_trips_through_store(tmp_path: Path) -> None:
    store = MLArtifactStore(tmp_path)
    ref = store.write_artifact(b"content", category=ArtifactCategory.PREDICTIONS)
    fetched = store.artifact_reference(ref.content_hash)
    assert fetched == ref
