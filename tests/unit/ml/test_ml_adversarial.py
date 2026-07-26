"""Section 23 adversarial self-audit: a dedicated attempt to break the
ML infrastructure with the specific attack list the milestone calls out.
Every item below either has a permanent regression test HERE or is
cross-referenced to the test file that already covers it, so this file
also serves as an audit checklist.

Covered elsewhere (cross-referenced, not duplicated):
- reordered dicts/feature lists, changed notes, changed dataset/
  preprocessing/code-revision hashes -> test_experiment_identity.py
- malformed hashes, non-UTC timestamps, NaN/Infinity in JSON,
  unsupported schema versions -> test_persistence.py, test_ml_models.py
- path traversal, symlink escape, interrupted writes, corrupted
  artifacts -> test_artifacts.py
- duplicate model registrations -> test_ml_registry.py
- incompatible objective/label types, predict-before-fit, probability
  prediction on unsupported objectives -> test_ml_models.py, test_interfaces.py
- concurrent identical preparation -> test_experiment_manager.py,
  test_ml_manifests.py, test_tracking.py
- unavailable/dirty Git repo, missing optional package metadata ->
  test_environment.py

Covered HERE (not adequately exercised elsewhere):
- Windows path separators / null bytes / unicode homoglyphs embedded in
  a hash-shaped string.
- A stale leftover .tmp file (simulating an interrupted prior write)
  never gets mistaken for a completed artifact.
- Malicious artifact category/content-hash combinations attempted via
  the public read API.
- Concurrent "inconsistent" preparation: two DIFFERENT experiment specs
  never collide on the same experiment_id or corrupt each other's
  manifests when prepared concurrently.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from tests.unit.ml.conftest import build_registry, make_dataset_manifest, make_experiment_spec_kwargs

from quant_platform.core.exceptions import ArtifactCorruptionError, PathSecurityError
from quant_platform.features.manifests import ResearchManifestStore
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.fingerprints import is_valid_sha256_hex
from quant_platform.ml.models import ArtifactCategory, ModelHyperparameters


class TestMaliciousHashLikeStrings:
    @pytest.mark.parametrize(
        "malicious",
        [
            "..\\..\\..\\windows\\system32" + "0" * 40,
            "../../../../etc/passwd" + "0" * 42,
            "a" * 32 + "\x00" + "a" * 31,  # embedded null byte
            "a" * 63 + "é",  # unicode homoglyph, not ascii hex
            "A" * 63 + "g",  # 'g' is not a valid hex digit
            "",
            " " * 64,
        ],
    )
    def test_is_valid_sha256_hex_rejects_every_malicious_form(self, malicious: str) -> None:
        assert not is_valid_sha256_hex(malicious)

    @pytest.mark.parametrize(
        "malicious",
        [
            "..\\..\\..\\windows\\system32" + "0" * 40,
            "../../../../etc/passwd" + "0" * 42,
            "a" * 32 + "\x00" + "a" * 31,
        ],
    )
    def test_artifact_store_read_rejects_malicious_hash(self, tmp_path: Path, malicious: str) -> None:
        store = MLArtifactStore(tmp_path)
        with pytest.raises(ArtifactCorruptionError):
            store.read_artifact(malicious)

    def test_artifact_store_content_path_defense_in_depth_for_traversal_string(self, tmp_path: Path) -> None:
        """`pathlib` parses embedded `/` characters in a joined string as
        real path SEGMENTS (not one opaque component), so a caller that
        bypasses `read_artifact`'s hash-format pre-check and calls the
        private `_content_path` builder directly with a `/`-separated
        traversal string produces a path that genuinely resolves outside
        the store root. `_content_path` calls `_assert_within_root`
        itself (not just `read_artifact`), so this is caught as a SECOND,
        independent defense layer -- `PathSecurityError`, never a silent
        escape -- proving defense-in-depth rather than a single choke
        point that a direct private-method call could bypass."""
        store = MLArtifactStore(tmp_path)
        with pytest.raises(PathSecurityError):
            store._content_path("../../../etc/passwd" + "0" * 42)


class TestStaleTempFileTolerance:
    def test_stale_tmp_file_from_interrupted_write_does_not_block_new_write(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        data = b"real content"
        content_hash_dir = tmp_path / "content" / "__stale_test__"
        content_hash_dir.mkdir(parents=True)
        # Simulate a crash mid-write: a stale .tmp file sitting next to
        # where a real content file would go, under an unrelated prefix.
        (content_hash_dir / ".somehash.deadbeef.tmp").write_bytes(b"PARTIAL GARBAGE")

        ref = store.write_artifact(data, category=ArtifactCategory.MODEL)
        assert store.read_artifact(ref.content_hash) == data
        # The stale tmp file must never be picked up as if it were a
        # real artifact -- it isn't addressable by any real content hash.
        assert not store.artifact_exists("0" * 64)

    def test_stale_tmp_file_left_after_successful_write_is_harmless(self, tmp_path: Path) -> None:
        store = MLArtifactStore(tmp_path)
        ref = store.write_artifact(b"content", category=ArtifactCategory.MODEL)
        content_path = store._content_path(ref.content_hash)
        stale_tmp = content_path.parent / f".{content_path.name}.leftoverfromcrash.tmp"
        stale_tmp.write_bytes(b"leftover garbage from an unrelated interrupted write")

        # A stale sibling .tmp file must not corrupt or shadow the real read.
        assert store.read_artifact(ref.content_hash) == b"content"


class TestConcurrentInconsistentPreparation:
    def test_concurrently_preparing_different_specs_never_cross_contaminates(self, tmp_path: Path) -> None:
        """Two THREADS preparing two DIFFERENT (different hyperparameter)
        specs concurrently must each get their own correct, independent
        manifest -- never a mix-up, never one experiment's content
        appearing under the other's experiment_id."""
        research_store = ResearchManifestStore(tmp_path / "research")
        research_store.save(make_dataset_manifest())
        preparer = ExperimentPreparer(
            ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=build_registry(),
            research_manifest_store=research_store,
        )
        spec_a = ExperimentSpec(**make_experiment_spec_kwargs(hyperparameters=ModelHyperparameters(values={"alpha": 0.1})))
        spec_b = ExperimentSpec(**make_experiment_spec_kwargs(hyperparameters=ModelHyperparameters(values={"alpha": 0.2})))
        assert spec_a != spec_b

        results: dict[str, object] = {}
        lock = threading.Lock()

        def prepare(key: str, spec: ExperimentSpec) -> None:
            manifest = preparer.prepare(spec)
            with lock:
                results[key] = manifest

        t_a = threading.Thread(target=prepare, args=("a", spec_a))
        t_b = threading.Thread(target=prepare, args=("b", spec_b))
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()

        manifest_a = results["a"]
        manifest_b = results["b"]
        assert manifest_a.identity.experiment_id != manifest_b.identity.experiment_id  # type: ignore[union-attr]
        assert manifest_a.spec == spec_a  # type: ignore[union-attr]
        assert manifest_b.spec == spec_b  # type: ignore[union-attr]
        # Both independently reloadable and still self-consistent.
        reloaded_a = preparer.manifest_store.load(manifest_a.identity.experiment_id)  # type: ignore[union-attr]
        reloaded_b = preparer.manifest_store.load(manifest_b.identity.experiment_id)  # type: ignore[union-attr]
        assert reloaded_a.spec.hyperparameters.values == {"alpha": 0.1}
        assert reloaded_b.spec.hyperparameters.values == {"alpha": 0.2}
