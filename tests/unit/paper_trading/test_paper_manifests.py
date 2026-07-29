"""Milestone 7, Section 20: `PaperSessionManifest`/`PaperSessionManifestStore`.
Covers create/load/transition/resume-count against a real temp directory
(genuine file I/O, matching the established manifest-store test
convention elsewhere in this repository), the FAILED-requires-failure-
fields invariant, and concurrent-write rejection via the reused
`experiment_lock` mechanism."""

from __future__ import annotations

import threading

import pytest

from quant_platform.core.exceptions import PaperTradingManifestError, SessionLockError
from quant_platform.paper_trading.manifests import PaperSessionManifest, PaperSessionManifestStore
from quant_platform.paper_trading.models import PaperSessionStage, SessionMode

_HEX_SESSION_ID = "a" * 64


class TestPaperSessionManifestValidation:
    def _manifest(self, **overrides: object) -> PaperSessionManifest:
        defaults: dict[str, object] = {
            "schema_version": 1, "paper_session_id": _HEX_SESSION_ID, "session_mode": SessionMode.REPLAY_PAPER, "stage": PaperSessionStage.CREATED,
            "created_at": "2026-01-05T10:00:00+00:00", "updated_at": "2026-01-05T10:00:00+00:00",
        }
        defaults.update(overrides)
        return PaperSessionManifest(**defaults)  # type: ignore[arg-type]

    def test_valid_created_manifest(self) -> None:
        manifest = self._manifest()
        assert manifest.stage is PaperSessionStage.CREATED

    def test_invalid_paper_session_id_rejected(self) -> None:
        with pytest.raises(PaperTradingManifestError, match="paper_session_id"):
            self._manifest(paper_session_id="not-a-hash")

    def test_failed_without_failure_category_rejected(self) -> None:
        with pytest.raises(PaperTradingManifestError, match="failure_category"):
            self._manifest(stage=PaperSessionStage.FAILED)

    def test_failed_with_failure_fields_succeeds(self) -> None:
        manifest = self._manifest(stage=PaperSessionStage.FAILED, failure_category="execution_error", failure_stage="running")
        assert manifest.failure_category == "execution_error"

    def test_non_failed_with_failure_fields_rejected(self) -> None:
        with pytest.raises(PaperTradingManifestError, match="failure_"):
            self._manifest(failure_category="execution_error")

    def test_completed_requires_completed_at(self) -> None:
        with pytest.raises(PaperTradingManifestError, match="completed_at"):
            self._manifest(stage=PaperSessionStage.COMPLETED)

    def test_json_round_trip(self) -> None:
        manifest = self._manifest()
        assert PaperSessionManifest.from_json_dict(manifest.to_json_dict()) == manifest


class TestPaperSessionManifestStore:
    def test_create_then_load(self, tmp_path) -> None:
        store = PaperSessionManifestStore(tmp_path)
        created = store.create(paper_session_id=_HEX_SESSION_ID, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None)
        assert created.stage is PaperSessionStage.CREATED
        loaded = store.load(_HEX_SESSION_ID)
        assert loaded == created

    def test_exists_reflects_creation(self, tmp_path) -> None:
        store = PaperSessionManifestStore(tmp_path)
        assert not store.exists(_HEX_SESSION_ID)
        store.create(paper_session_id=_HEX_SESSION_ID, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None)
        assert store.exists(_HEX_SESSION_ID)

    def test_create_twice_rejected(self, tmp_path) -> None:
        store = PaperSessionManifestStore(tmp_path)
        store.create(paper_session_id=_HEX_SESSION_ID, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None)
        with pytest.raises(PaperTradingManifestError, match="already exists"):
            store.create(paper_session_id=_HEX_SESSION_ID, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None)

    def test_load_missing_manifest_rejected(self, tmp_path) -> None:
        store = PaperSessionManifestStore(tmp_path)
        with pytest.raises(PaperTradingManifestError, match="No paper session manifest"):
            store.load(_HEX_SESSION_ID)

    def test_load_if_exists_returns_none_when_missing(self, tmp_path) -> None:
        store = PaperSessionManifestStore(tmp_path)
        assert store.load_if_exists(_HEX_SESSION_ID) is None

    def test_legal_transition_updates_stage(self, tmp_path) -> None:
        store = PaperSessionManifestStore(tmp_path)
        store.create(paper_session_id=_HEX_SESSION_ID, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None)
        updated = store.transition(_HEX_SESSION_ID, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)
        assert updated.stage is PaperSessionStage.ELIGIBILITY_VERIFIED
        assert store.load(_HEX_SESSION_ID).stage is PaperSessionStage.ELIGIBILITY_VERIFIED

    def test_illegal_transition_rejected(self, tmp_path) -> None:
        store = PaperSessionManifestStore(tmp_path)
        store.create(paper_session_id=_HEX_SESSION_ID, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None)
        with pytest.raises(PaperTradingManifestError, match="Illegal"):
            store.transition(_HEX_SESSION_ID, target_stage=PaperSessionStage.RUNNING)

    def test_transition_to_failed_requires_failure_fields(self, tmp_path) -> None:
        store = PaperSessionManifestStore(tmp_path)
        store.create(paper_session_id=_HEX_SESSION_ID, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None)
        with pytest.raises(PaperTradingManifestError, match="failure_category"):
            store.transition(_HEX_SESSION_ID, target_stage=PaperSessionStage.FAILED)

    def test_transition_to_failed_with_fields_succeeds(self, tmp_path) -> None:
        store = PaperSessionManifestStore(tmp_path)
        store.create(paper_session_id=_HEX_SESSION_ID, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None)
        updated = store.transition(
            _HEX_SESSION_ID, target_stage=PaperSessionStage.FAILED, failure_category="execution_error", failure_stage="created",
            failure_event_identity="b" * 64, failure_recoverable=True, failure_safe_resume_stage="created",
        )
        assert updated.stage is PaperSessionStage.FAILED
        assert updated.failure_recoverable is True

    def test_resume_count_increments(self, tmp_path) -> None:
        store = PaperSessionManifestStore(tmp_path)
        store.create(paper_session_id=_HEX_SESSION_ID, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None)
        assert store.load(_HEX_SESSION_ID).resume_count == 0
        updated = store.bump_resume_count(_HEX_SESSION_ID)
        assert updated.resume_count == 1
        updated = store.bump_resume_count(_HEX_SESSION_ID)
        assert updated.resume_count == 2

    def test_rewind_to_earlier_non_terminal_stage_allowed(self, tmp_path) -> None:
        """Resume's own "demote on detected corruption" allowance --
        `is_legal_paper_session_transition` permits moving to any
        strictly-earlier non-terminal stage."""
        store = PaperSessionManifestStore(tmp_path)
        store.create(paper_session_id=_HEX_SESSION_ID, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None)
        store.transition(_HEX_SESSION_ID, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)
        store.transition(_HEX_SESSION_ID, target_stage=PaperSessionStage.INITIALIZED)
        rewound = store.transition(_HEX_SESSION_ID, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)
        assert rewound.stage is PaperSessionStage.ELIGIBILITY_VERIFIED

    def test_different_session_ids_are_independent(self, tmp_path) -> None:
        store = PaperSessionManifestStore(tmp_path)
        other_id = "c" * 64
        store.create(paper_session_id=_HEX_SESSION_ID, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None)
        store.create(paper_session_id=other_id, session_mode=SessionMode.SHADOW_OBSERVATION, spec_reference=None)
        store.transition(_HEX_SESSION_ID, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)
        assert store.load(other_id).stage is PaperSessionStage.CREATED


class TestConcurrentAccess:
    def test_concurrent_transition_attempts_do_not_corrupt_or_duplicate(self, tmp_path) -> None:
        """Two threads racing to transition the SAME session: the shared
        lock must serialize them -- exactly one succeeds per legal edge,
        and the manifest never ends up in a torn/inconsistent state."""
        store = PaperSessionManifestStore(tmp_path)
        store.create(paper_session_id=_HEX_SESSION_ID, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None)

        errors: list[Exception] = []

        def _bump() -> None:
            try:
                store.bump_resume_count(_HEX_SESSION_ID)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_bump) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        acceptable = (SessionLockError,)
        for error in errors:
            assert isinstance(error, acceptable), f"unexpected error type: {error!r}"

        final = store.load(_HEX_SESSION_ID)
        assert final.resume_count == (8 - len(errors))
