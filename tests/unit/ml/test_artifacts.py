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


class TestDedupReVerification:
    """Milestone 5.2, Section 5: `write_artifact`'s dedup path must
    re-verify pre-existing bytes (recompute their hash, check the
    sidecar's recorded size) before trusting them -- never accept a
    hash-named path on the strength of `is_file()` alone."""

    def test_pre_existing_corrupted_content_artifact_is_detected_and_atomically_repaired(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        data = b"the genuinely correct content"
        ref = store.write_artifact(data, category=ArtifactCategory.MODEL)
        content_path = store._content_path(ref.content_hash)
        content_path.write_bytes(b"CORRUPTED BYTES, DIFFERENT LENGTH TOO")

        # Re-writing the SAME original data (same hash) must detect the
        # corruption and self-heal, not silently hand back a reference
        # to the corrupted bytes.
        ref2 = store.write_artifact(data, category=ArtifactCategory.MODEL)
        assert ref2.content_hash == ref.content_hash
        assert ref2.size_bytes == len(data)
        assert store.read_artifact(ref.content_hash) == data
        assert content_path.read_bytes() == data

    def test_pre_existing_wrong_category_artifact_fails_closed_not_repaired(self, tmp_path: Path) -> None:
        """The OTHER mismatch kind (Section 5's explicit distinction):
        category conflicts are never auto-repaired, only content/size
        corruption is -- both claims about the bytes are equally
        "valid," so this store cannot pick one silently."""
        store = MLArtifactStore(tmp_path)
        data = b"identical bytes, ambiguous category"
        store.write_artifact(data, category=ArtifactCategory.MODEL)
        with pytest.raises(ArtifactCorruptionError, match="already stored under category"):
            store.write_artifact(data, category=ArtifactCategory.METRICS)
        # And the original, untouched artifact must still read back fine.
        original_hash = store.write_artifact(data, category=ArtifactCategory.MODEL).content_hash
        assert store.read_artifact(original_hash) == data

    def test_pre_existing_semantically_mismatched_sidecar_size_is_detected_and_repaired(self, tmp_path: Path) -> None:
        """A sidecar whose `size_bytes` disagrees with the ACTUAL
        (correctly-hashing) content bytes -- metadata corruption
        distinct from content corruption, also never trusted on sight."""
        import json

        store = MLArtifactStore(tmp_path)
        data = b"content with a tampered metadata size field"
        ref = store.write_artifact(data, category=ArtifactCategory.MODEL)
        metadata_path = store._metadata_path(ref.content_hash)
        raw = json.loads(metadata_path.read_text())
        raw["size_bytes"] = 1
        metadata_path.write_text(json.dumps(raw))

        ref2 = store.write_artifact(data, category=ArtifactCategory.MODEL)
        assert ref2.size_bytes == len(data)
        # The re-verification round-trip through `read_artifact` (which
        # itself checks size consistency) must now succeed.
        assert store.read_artifact(ref.content_hash) == data

    def test_valid_matching_artifact_dedups_without_any_repair(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        data = b"perfectly valid, untampered content"
        ref1 = store.write_artifact(data, category=ArtifactCategory.MODEL)
        content_mtime_before = store._content_path(ref1.content_hash).stat().st_mtime_ns
        ref2 = store.write_artifact(data, category=ArtifactCategory.MODEL)
        assert ref1 == ref2
        # No repair/rewrite should have touched the content file at all.
        assert store._content_path(ref1.content_hash).stat().st_mtime_ns == content_mtime_before
        assert store.read_artifact(ref2.content_hash) == data

    def test_concurrent_dedup_attempts_against_a_pre_existing_artifact_are_safe(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        data = b"concurrently deduped content" * 50
        first_ref = store.write_artifact(data, category=ArtifactCategory.MODEL)
        errors: list[BaseException] = []
        results: list[object] = []

        def write() -> None:
            try:
                results.append(store.write_artifact(data, category=ArtifactCategory.MODEL))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(r == first_ref for r in results)
        assert store.read_artifact(first_ref.content_hash) == data

    def test_interrupted_replacement_is_safely_retried_on_the_next_call(self, tmp_path: Path) -> None:
        """If the atomic-rename step of a corruption REPAIR is itself
        interrupted (raises), the store must not end up in a state that
        silently looks fine -- and a subsequent call must still be able
        to detect and complete the repair."""
        store = MLArtifactStore(tmp_path)
        data = b"content that will need a repair replay"
        ref = store.write_artifact(data, category=ArtifactCategory.MODEL)
        content_path = store._content_path(ref.content_hash)
        content_path.write_bytes(b"CORRUPTED")

        with patch("pathlib.Path.replace", side_effect=OSError("interrupted mid-repair")), pytest.raises(OSError, match="interrupted mid-repair"):
            store.write_artifact(data, category=ArtifactCategory.MODEL)
        # The corruption is still present (the interrupted repair did not
        # silently "succeed") -- no tmp files were left behind either.
        assert content_path.read_bytes() == b"CORRUPTED"
        assert list(tmp_path.rglob(".*.tmp")) == []

        # A subsequent, uninterrupted call must detect and complete the repair.
        ref2 = store.write_artifact(data, category=ArtifactCategory.MODEL)
        assert ref2.content_hash == ref.content_hash
        assert store.read_artifact(ref.content_hash) == data

    def test_truncated_content_is_detected_and_repaired(self, tmp_path: Path) -> None:
        """FINAL MILESTONE 5 AUDIT, Section 5: a truncated file (fewer
        bytes than the original, a distinct corruption mode from
        `test_pre_existing_corrupted_content_artifact_is_detected_and_
        atomically_repaired`'s "different bytes, similar length" case)
        must be detected via the same content-hash recomputation, not
        merely a size check that a byte-for-byte-different-but-same-
        length corruption would also need."""
        store = MLArtifactStore(tmp_path)
        data = b"the genuinely correct and complete content, all of it"
        ref = store.write_artifact(data, category=ArtifactCategory.MODEL)
        content_path = store._content_path(ref.content_hash)
        content_path.write_bytes(data[:5])  # truncated

        ref2 = store.write_artifact(data, category=ArtifactCategory.MODEL)
        assert ref2.size_bytes == len(data)
        assert store.read_artifact(ref.content_hash) == data

    def test_wrong_sidecar_schema_version_fails_closed_loudly(self, tmp_path: Path) -> None:
        """FINAL MILESTONE 5 AUDIT, Section 5: an unsupported sidecar
        schema_version must never be silently accepted or coerced -- it
        must raise a specific, typed, loud error (`SchemaVersionError`),
        distinct from `ArtifactCorruptionError`'s content/size mismatch
        cases, so an operator/caller can tell the two failure modes apart."""
        import json

        from quant_platform.core.exceptions import SchemaVersionError

        store = MLArtifactStore(tmp_path)
        data = b"content with a sidecar from a future/unsupported schema"
        ref = store.write_artifact(data, category=ArtifactCategory.MODEL)
        metadata_path = store._metadata_path(ref.content_hash)
        raw = json.loads(metadata_path.read_text())
        raw["schema_version"] = 999
        metadata_path.write_text(json.dumps(raw))

        with pytest.raises(SchemaVersionError, match="schema_version"):
            store.write_artifact(data, category=ArtifactCategory.MODEL)

    def test_concurrent_corruption_repair_is_safe_and_a_losing_race_is_retryable(self, tmp_path: Path) -> None:
        """FINAL MILESTONE 5 AUDIT, Section 5: TWO threads racing to
        repair the SAME corruption. Per `_atomic_write_content`'s own
        documented policy (no tolerance on the repair path -- a losing
        repair race simply fails and is retried by its own caller, never
        silently guessed at), at most one thread may raise; a retry after
        any such failure must always succeed, and the final on-disk state
        must be the correct content either way."""
        store = MLArtifactStore(tmp_path)
        data = b"content two threads will race to repair" * 20
        ref = store.write_artifact(data, category=ArtifactCategory.MODEL)
        content_path = store._content_path(ref.content_hash)
        content_path.write_bytes(b"CORRUPTED" * 5)

        errors: list[BaseException] = []

        def repair() -> None:
            try:
                store.write_artifact(data, category=ArtifactCategory.MODEL)
            except OSError as exc:
                errors.append(exc)

        threads = [threading.Thread(target=repair) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Whether or not any thread lost a race (raised), a follow-up
        # call must always succeed and leave the store correctly repaired.
        store.write_artifact(data, category=ArtifactCategory.MODEL)
        assert store.read_artifact(ref.content_hash) == data


def test_artifact_reference_round_trips_through_store(tmp_path: Path) -> None:
    store = MLArtifactStore(tmp_path)
    ref = store.write_artifact(b"content", category=ArtifactCategory.PREDICTIONS)
    fetched = store.artifact_reference(ref.content_hash)
    assert fetched == ref
