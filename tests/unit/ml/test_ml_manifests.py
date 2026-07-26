from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from tests.unit.ml.conftest import make_experiment_spec_kwargs

from quant_platform.core.exceptions import (
    ArtifactNotFoundError,
    ExperimentIdentityError,
    ExperimentStateError,
)
from quant_platform.ml.environment import capture_environment_snapshot
from quant_platform.ml.experiment_identity import compute_experiment_identity
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.manifests import MANIFEST_SCHEMA_VERSION, ExperimentManifest, ExperimentManifestStore
from quant_platform.ml.models import ExperimentStatus
from quant_platform.ml.persistence import format_utc_timestamp, utc_now


def _make_manifest(spec: ExperimentSpec | None = None, **overrides: object) -> ExperimentManifest:
    spec = spec if spec is not None else ExperimentSpec(**make_experiment_spec_kwargs())
    base: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "identity": compute_experiment_identity(spec),
        "spec": spec,
        "model_definition_fingerprint": "f" * 64,
        "status": ExperimentStatus.CREATED,
        "environment_snapshot": capture_environment_snapshot(),
        "artifact_references": (),
        "validation_report_reference": None,
        "created_at": format_utc_timestamp(utc_now()),
    }
    base.update(overrides)
    return ExperimentManifest(**base)  # type: ignore[arg-type]


class TestExperimentManifestConstruction:
    def test_valid_manifest_builds(self) -> None:
        _make_manifest()

    def test_identity_mismatch_with_spec_rejected(self) -> None:
        spec1 = ExperimentSpec(**make_experiment_spec_kwargs())
        spec2 = ExperimentSpec(**make_experiment_spec_kwargs(model_version="99"))
        with pytest.raises(ExperimentIdentityError):
            _make_manifest(spec=spec1, identity=compute_experiment_identity(spec2))

    def test_empty_model_definition_fingerprint_rejected(self) -> None:
        with pytest.raises(ValueError, match="model_definition_fingerprint"):
            _make_manifest(model_definition_fingerprint="")

    def test_non_utc_created_at_rejected(self) -> None:
        with pytest.raises(ValueError, match="not timezone-aware"):
            _make_manifest(created_at="2024-01-01T00:00:00")

    def test_completed_at_forbidden_in_non_terminal_status(self) -> None:
        with pytest.raises(ValueError, match="completed_at must be None"):
            _make_manifest(status=ExperimentStatus.CREATED, completed_at=format_utc_timestamp(utc_now()))

    def test_failure_summary_forbidden_outside_failed(self) -> None:
        with pytest.raises(ValueError, match="failure_summary"):
            _make_manifest(status=ExperimentStatus.COMPLETED, completed_at=format_utc_timestamp(utc_now()), failure_summary="x")

    def test_failure_summary_required_when_failed(self) -> None:
        with pytest.raises(ValueError, match="failure_summary is required"):
            _make_manifest(status=ExperimentStatus.FAILED, completed_at=format_utc_timestamp(utc_now()))

    def test_invalid_parent_experiment_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="parent_experiment_id"):
            _make_manifest(parent_experiment_id="not-a-hash")

    def test_round_trip(self) -> None:
        manifest = _make_manifest()
        assert ExperimentManifest.from_json_dict(manifest.to_json_dict()) == manifest


class TestExperimentManifestStoreCreate:
    def test_create_and_load(self, tmp_path: Path) -> None:
        store = ExperimentManifestStore(tmp_path)
        manifest = _make_manifest()
        store.create(manifest)
        loaded = store.load(manifest.identity.experiment_id)
        assert loaded == manifest

    def test_create_requires_created_status(self, tmp_path: Path) -> None:
        store = ExperimentManifestStore(tmp_path)
        spec = ExperimentSpec(**make_experiment_spec_kwargs())
        manifest = _make_manifest(spec=spec, status=ExperimentStatus.FAILED, completed_at=format_utc_timestamp(utc_now()), failure_summary="x")
        with pytest.raises(ExperimentStateError):
            store.create(manifest)

    def test_duplicate_create_rejected(self, tmp_path: Path) -> None:
        store = ExperimentManifestStore(tmp_path)
        manifest = _make_manifest()
        store.create(manifest)
        with pytest.raises(ExperimentStateError, match="already exists"):
            store.create(manifest)

    def test_load_missing_raises_not_found(self, tmp_path: Path) -> None:
        store = ExperimentManifestStore(tmp_path)
        with pytest.raises(ArtifactNotFoundError):
            store.load("a" * 64)

    def test_load_if_exists_returns_none_when_absent(self, tmp_path: Path) -> None:
        store = ExperimentManifestStore(tmp_path)
        assert store.load_if_exists("a" * 64) is None

    def test_exists(self, tmp_path: Path) -> None:
        store = ExperimentManifestStore(tmp_path)
        manifest = _make_manifest()
        assert not store.exists(manifest.identity.experiment_id)
        store.create(manifest)
        assert store.exists(manifest.identity.experiment_id)

    def test_list_experiment_ids(self, tmp_path: Path) -> None:
        store = ExperimentManifestStore(tmp_path)
        m1 = _make_manifest(spec=ExperimentSpec(**make_experiment_spec_kwargs(model_version="1")))
        m2 = _make_manifest(spec=ExperimentSpec(**make_experiment_spec_kwargs(model_version="2")))
        store.create(m1)
        store.create(m2)
        ids = store.list_experiment_ids()
        assert set(ids) == {m1.identity.experiment_id, m2.identity.experiment_id}
        assert ids == sorted(ids)

    def test_interrupted_create_write_leaves_no_manifest_file(self, tmp_path: Path) -> None:
        """Fault-injected (deterministic, non-flaky) proof that an
        interrupted `create()` write can never expose a truncated/partial
        manifest file: `write_json_atomic`'s temp-file-then-rename is
        broken exactly at the rename step, so the failure happens after
        the temp file is fully written but before it replaces anything."""
        from unittest.mock import patch

        store = ExperimentManifestStore(tmp_path)
        manifest = _make_manifest()

        with patch.object(Path, "replace", side_effect=OSError("simulated crash mid-write")), \
                pytest.raises(OSError, match="simulated crash"):
            store.create(manifest)

        assert not store.exists(manifest.identity.experiment_id)
        manifest_dir = store._manifest_path(manifest.identity.experiment_id).parent
        if manifest_dir.is_dir():
            assert list(manifest_dir.glob(".*.tmp")) == []


class TestExperimentManifestStoreTransitions:
    def _created(self, store: ExperimentManifestStore) -> ExperimentManifest:
        manifest = _make_manifest()
        store.create(manifest)
        return manifest

    def test_legal_transition_sequence(self, tmp_path: Path) -> None:
        store = ExperimentManifestStore(tmp_path)
        manifest = self._created(store)
        eid = manifest.identity.experiment_id
        store.transition(eid, new_status=ExperimentStatus.VALIDATING)
        store.transition(eid, new_status=ExperimentStatus.READY)
        store.transition(eid, new_status=ExperimentStatus.RUNNING)
        completed_at = format_utc_timestamp(utc_now())
        final = store.transition(eid, new_status=ExperimentStatus.COMPLETED, completed_at=completed_at)
        assert final.status is ExperimentStatus.COMPLETED
        assert final.completed_at == completed_at

    def test_illegal_transition_rejected(self, tmp_path: Path) -> None:
        store = ExperimentManifestStore(tmp_path)
        manifest = self._created(store)
        with pytest.raises(ExperimentStateError):
            store.transition(manifest.identity.experiment_id, new_status=ExperimentStatus.RUNNING)

    def test_terminal_state_cannot_be_edited_again(self, tmp_path: Path) -> None:
        store = ExperimentManifestStore(tmp_path)
        manifest = self._created(store)
        eid = manifest.identity.experiment_id
        store.transition(eid, new_status=ExperimentStatus.VALIDATING)
        store.transition(eid, new_status=ExperimentStatus.FAILED, completed_at=format_utc_timestamp(utc_now()), failure_summary="nope")
        with pytest.raises(ExperimentStateError):
            store.transition(eid, new_status=ExperimentStatus.VALIDATING)
        with pytest.raises(ExperimentStateError):
            store.transition(eid, new_status=ExperimentStatus.READY)

    def test_transition_updates_artifact_references(self, tmp_path: Path) -> None:
        from quant_platform.ml.artifacts import MLArtifactStore
        from quant_platform.ml.models import ArtifactCategory

        store = ExperimentManifestStore(tmp_path)
        artifact_store = MLArtifactStore(tmp_path)
        manifest = self._created(store)
        eid = manifest.identity.experiment_id
        ref = artifact_store.write_artifact(b"report bytes", category=ArtifactCategory.REPORT)
        store.transition(eid, new_status=ExperimentStatus.VALIDATING)
        updated = store.transition(eid, new_status=ExperimentStatus.READY, artifact_references=(ref,), validation_report_reference=ref)
        assert updated.artifact_references == (ref,)
        assert updated.validation_report_reference == ref

    def test_transition_omitting_artifact_refs_keeps_existing_ones_unchanged(self, tmp_path: Path) -> None:
        """Artifact references may only change when a caller explicitly
        passes new ones -- omitting them (`None`, the default) on a
        LATER legal transition must carry the existing values forward
        unchanged, never reset them to empty/`None`."""
        from quant_platform.ml.artifacts import MLArtifactStore
        from quant_platform.ml.models import ArtifactCategory

        store = ExperimentManifestStore(tmp_path)
        artifact_store = MLArtifactStore(tmp_path)
        manifest = self._created(store)
        eid = manifest.identity.experiment_id
        ref = artifact_store.write_artifact(b"first validation report", category=ArtifactCategory.REPORT)
        store.transition(
            eid, new_status=ExperimentStatus.VALIDATING, artifact_references=(ref,), validation_report_reference=ref,
        )
        # READY omits both artifact kwargs entirely.
        updated = store.transition(eid, new_status=ExperimentStatus.READY)
        assert updated.artifact_references == (ref,)
        assert updated.validation_report_reference == ref

    def test_transition_never_changes_scientific_bindings_identity_or_snapshot(self, tmp_path: Path) -> None:
        """Dataset/feature/label/preprocessing/model/seed/code bindings
        (all embedded in `spec`), the experiment's `identity`,
        `model_definition_fingerprint`, `environment_snapshot`, and
        `created_at` must never change across ANY status transition --
        `transition()`'s signature structurally has no parameter that
        could rewrite any of them, but this proves it end-to-end rather
        than by code inspection alone."""
        store = ExperimentManifestStore(tmp_path)
        manifest = self._created(store)
        eid = manifest.identity.experiment_id
        store.transition(eid, new_status=ExperimentStatus.VALIDATING)
        updated = store.transition(eid, new_status=ExperimentStatus.READY)

        assert updated.spec == manifest.spec
        assert updated.spec.dataset_binding == manifest.spec.dataset_binding
        assert updated.spec.feature_binding == manifest.spec.feature_binding
        assert updated.spec.label_binding == manifest.spec.label_binding
        assert updated.spec.preprocessing_binding == manifest.spec.preprocessing_binding
        assert updated.spec.split_binding == manifest.spec.split_binding
        assert updated.spec.model_name == manifest.spec.model_name
        assert updated.spec.model_version == manifest.spec.model_version
        assert updated.spec.seed_configuration == manifest.spec.seed_configuration
        assert updated.spec.code_revision_binding == manifest.spec.code_revision_binding
        assert updated.identity == manifest.identity
        assert updated.model_definition_fingerprint == manifest.model_definition_fingerprint
        assert updated.environment_snapshot == manifest.environment_snapshot
        assert updated.created_at == manifest.created_at

    def test_interrupted_transition_write_leaves_previous_manifest_intact(self, tmp_path: Path) -> None:
        """Fault-injected (deterministic, non-flaky) proof that an
        interrupted `transition()` write leaves the PREVIOUS, still-valid
        manifest fully intact and loadable -- never a half-written or
        torn file at the real path, and never a transition that "partly"
        took effect."""
        from unittest.mock import patch

        store = ExperimentManifestStore(tmp_path)
        manifest = self._created(store)
        eid = manifest.identity.experiment_id

        with patch.object(Path, "replace", side_effect=OSError("simulated crash mid-write")), \
                pytest.raises(OSError, match="simulated crash"):
            store.transition(eid, new_status=ExperimentStatus.VALIDATING)

        reloaded = store.load(eid)
        assert reloaded == manifest
        assert reloaded.status is ExperimentStatus.CREATED
        manifest_dir = store._manifest_path(eid).parent
        assert list(manifest_dir.glob(".*.tmp")) == []

    def _advance_to_terminal(self, store: ExperimentManifestStore, eid: str, terminal: ExperimentStatus) -> None:
        if terminal is ExperimentStatus.CANCELLED:
            store.transition(eid, new_status=ExperimentStatus.CANCELLED)
            return
        store.transition(eid, new_status=ExperimentStatus.VALIDATING)
        if terminal is ExperimentStatus.FAILED:
            store.transition(
                eid, new_status=ExperimentStatus.FAILED,
                completed_at=format_utc_timestamp(utc_now()), failure_summary="x",
            )
            return
        store.transition(eid, new_status=ExperimentStatus.READY)
        store.transition(eid, new_status=ExperimentStatus.RUNNING)
        store.transition(eid, new_status=ExperimentStatus.COMPLETED, completed_at=format_utc_timestamp(utc_now()))

    @pytest.mark.parametrize(
        "terminal_status", [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED]
    )
    def test_no_transition_out_of_any_terminal_status_to_any_target(
        self, tmp_path: Path, terminal_status: ExperimentStatus
    ) -> None:
        """Comprehensive version of `test_terminal_state_cannot_be_edited_
        again` above: EVERY terminal status rejects EVERY possible target
        status (including itself and including other terminal statuses),
        never just the one example already covered."""
        store = ExperimentManifestStore(tmp_path)
        manifest = self._created(store)
        eid = manifest.identity.experiment_id
        self._advance_to_terminal(store, eid, terminal_status)
        before = store.load(eid)

        for target in ExperimentStatus:
            with pytest.raises(ExperimentStateError):
                store.transition(eid, new_status=target)

        assert store.load(eid) == before


class TestManifestTamperDetection:
    def test_tampered_spec_without_recomputed_identity_raises_on_load(self, tmp_path: Path) -> None:
        store = ExperimentManifestStore(tmp_path)
        manifest = _make_manifest()
        store.create(manifest)

        manifest_path = store._manifest_path(manifest.identity.experiment_id)
        raw = json.loads(manifest_path.read_text())
        raw["spec"]["model_version"] = "999"
        manifest_path.write_text(json.dumps(raw))

        with pytest.raises(ExperimentIdentityError):
            store.load(manifest.identity.experiment_id)

    def test_invalid_experiment_id_path_rejected(self, tmp_path: Path) -> None:
        store = ExperimentManifestStore(tmp_path)
        with pytest.raises(ExperimentIdentityError):
            store.load("../../etc/passwd")


class TestManifestStoreConcurrency:
    def test_concurrent_create_of_same_manifest_is_safe(self, tmp_path: Path) -> None:
        """Exactly one concurrent `create()` call succeeds; every other
        call fails loudly (never silently corrupting the manifest, never
        succeeding twice) -- with EITHER `ExperimentLockError` (lost the
        race for the lock itself, fail-fast rather than blocking -- this
        is what a contested OR racing lock acquisition now surfaces as at
        the ML boundary, per `ml/concurrency.py`; it never leaks the
        underlying `DatasetLockError`/`PermissionError` directly) or
        `ExperimentStateError` (acquired the lock after the winner
        already created and released it, found the manifest already
        exists) -- both are legitimate "lost the race safely" outcomes.
        `historical.locking.DatasetLock` is a local advisory lock
        documented for a "single-writer-at-a-time" design target, not
        many-way simultaneous first-acquisition storms; under 4-way
        simultaneous FIRST acquisition it can rarely surface a
        pre-existing Windows-specific race in its own stale-lock reclaim
        path (`_handle_existing_lock` unlinking a lock file another thread
        is concurrently touching), which `experiment_lock` now catches
        and translates rather than letting it escape as a raw OS error.
        That underlying race is a known, documented limitation of reused
        Milestone 2 code (out of scope to fix here, "do not redesign
        M1-3") -- what THIS test actually verifies is that no matter
        which loud failure a losing thread gets, exactly one winner
        succeeds and the manifest is never corrupted, and NEITHER a raw
        `DatasetLockError` nor a raw `OSError`/`PermissionError` ever
        escapes this store's public boundary."""
        from quant_platform.core.exceptions import ExperimentLockError

        store = ExperimentManifestStore(tmp_path)
        manifest = _make_manifest()
        results: list[BaseException | None] = []
        lock = threading.Lock()

        def create() -> None:
            try:
                store.create(manifest)
                with lock:
                    results.append(None)
            except (ExperimentStateError, ExperimentLockError) as exc:
                with lock:
                    results.append(exc)

        threads = [threading.Thread(target=create) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successes = [r for r in results if r is None]
        failures = [r for r in results if r is not None]
        assert len(results) == 4, "no unexpected exception type should have escaped"
        assert len(successes) == 1
        assert len(failures) == 3
        assert store.load(manifest.identity.experiment_id) == manifest


class TestExperimentLockErrorTranslation:
    """Deterministic (non-flaky, single-threaded) proof of the ML
    boundary's lock-failure translation -- complementary to the
    concurrency stress test above, which only tolerates
    `ExperimentLockError` as ONE OF several possible outcomes under a
    genuine race. Here, the SAME thread holds the lock itself, so a
    contested acquisition is guaranteed, not probabilistic."""

    def test_create_translates_contested_lock(self, tmp_path: Path) -> None:
        from quant_platform.core.exceptions import DatasetLockError, ExperimentLockError
        from quant_platform.historical.locking import DatasetLock

        store = ExperimentManifestStore(tmp_path)
        manifest = _make_manifest()
        holder = DatasetLock(store._lock_path(manifest.identity.experiment_id))
        holder.acquire()
        try:
            with pytest.raises(ExperimentLockError) as exc_info:
                store.create(manifest)
            assert isinstance(exc_info.value.__cause__, DatasetLockError)
        finally:
            holder.release()
        # The lock being freed again means the failure never leaked a stuck lock.
        store.create(manifest)
        assert store.exists(manifest.identity.experiment_id)

    def test_transition_translates_contested_lock(self, tmp_path: Path) -> None:
        from quant_platform.core.exceptions import DatasetLockError, ExperimentLockError
        from quant_platform.historical.locking import DatasetLock

        store = ExperimentManifestStore(tmp_path)
        manifest = _make_manifest()
        store.create(manifest)
        eid = manifest.identity.experiment_id
        holder = DatasetLock(store._lock_path(eid))
        holder.acquire()
        try:
            with pytest.raises(ExperimentLockError) as exc_info:
                store.transition(eid, new_status=ExperimentStatus.VALIDATING)
            assert isinstance(exc_info.value.__cause__, DatasetLockError)
        finally:
            holder.release()
        assert store.load(eid).status is ExperimentStatus.CREATED


def test_manifest_reconstruction_yields_equivalent_domain_object(tmp_path: Path) -> None:
    store = ExperimentManifestStore(tmp_path)
    manifest = _make_manifest()
    store.create(manifest)
    loaded = store.load(manifest.identity.experiment_id)
    assert loaded.spec == manifest.spec
    assert loaded.spec.dataset_binding == manifest.spec.dataset_binding
    assert loaded.spec.feature_binding == manifest.spec.feature_binding
    assert loaded == replace(manifest)  # equality holds for an unmodified copy too
