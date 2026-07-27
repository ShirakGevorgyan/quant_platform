"""Tests for `historical.locking.DatasetLock` -- dataset-level advisory
locking with stale-lock recovery, added during the Milestone 2
release-readiness audit."""

from __future__ import annotations

import json
import os
import threading

import pandas as pd
import pytest

from quant_platform.core.exceptions import DatasetLockError
from quant_platform.historical.locking import DatasetLock, LockInfo, dataset_lock_path


class TestBasicAcquireRelease:
    def test_acquire_creates_lock_file(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        lock = DatasetLock(lock_path)
        lock.acquire()
        assert lock_path.is_file()
        lock.release()

    def test_release_removes_lock_file(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        lock = DatasetLock(lock_path)
        lock.acquire()
        lock.release()
        assert not lock_path.is_file()

    def test_release_without_acquire_is_a_noop(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        lock = DatasetLock(lock_path)
        lock.release()  # must not raise
        assert not lock_path.is_file()

    def test_lock_file_records_pid_and_hostname(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        lock = DatasetLock(lock_path)
        lock.acquire()
        raw = json.loads(lock_path.read_text())
        assert raw["pid"] == os.getpid()
        assert raw["hostname"]
        lock.release()


class TestContextManager:
    def test_with_statement_acquires_and_releases(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        with DatasetLock(lock_path):
            assert lock_path.is_file()
        assert not lock_path.is_file()

    def test_releases_even_when_body_raises(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        with pytest.raises(ValueError, match="boom"), DatasetLock(lock_path):
            raise ValueError("boom")
        assert not lock_path.is_file()


class TestConcurrentAcquisition:
    def test_second_acquire_fails_while_first_is_held(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        first = DatasetLock(lock_path)
        first.acquire()
        second = DatasetLock(lock_path)
        with pytest.raises(DatasetLockError, match="locked by pid"):
            second.acquire()
        first.release()

    def test_lock_can_be_acquired_again_after_release(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        first = DatasetLock(lock_path)
        first.acquire()
        first.release()
        second = DatasetLock(lock_path)
        second.acquire()  # must not raise
        second.release()


class TestStaleLockRecovery:
    def test_lock_older_than_stale_after_is_reclaimed(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        old_info = LockInfo(pid=999_999, hostname="dead-host", acquired_at=pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2))
        lock_path.write_text(json.dumps(old_info.to_json_dict()))

        lock = DatasetLock(lock_path, stale_after=pd.Timedelta(hours=1))
        lock.acquire()  # must not raise -- reclaims the stale lock
        raw = json.loads(lock_path.read_text())
        assert raw["pid"] == os.getpid()
        lock.release()

    def test_lock_younger_than_stale_after_is_not_reclaimed(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        recent_info = LockInfo(pid=999_999, hostname="other-host", acquired_at=pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=5))
        lock_path.write_text(json.dumps(recent_info.to_json_dict()))

        lock = DatasetLock(lock_path, stale_after=pd.Timedelta(hours=1))
        with pytest.raises(DatasetLockError, match="locked by pid=999999"):
            lock.acquire()

    def test_corrupted_lock_file_is_treated_as_reclaimable(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        lock_path.write_text("{not valid json at all")
        lock = DatasetLock(lock_path)
        lock.acquire()  # must not raise -- corrupted lock is reclaimed, not fatal
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
        lock.release()

    def test_lock_file_with_nan_field_is_treated_as_reclaimable(self, tmp_path) -> None:
        """Milestone 4D.1: `_read_existing_lock` now reads through
        `core.json.parse_json_strict`, which rejects a bare `NaN` token at
        parse time (the old plain `json.loads` would have accepted it,
        then failed later at `int(str(raw["pid"]))` -- same ultimate
        outcome, reclaimable, just a different rejection point). Explicit
        regression test for the documented "NaN/Infinity fields ->
        reclaimable" decision in `_read_existing_lock`'s docstring."""
        lock_path = tmp_path / ".lock"
        lock_path.write_text('{"pid": NaN, "hostname": "h", "acquired_at": "2024-01-01T00:00:00+00:00"}')
        lock = DatasetLock(lock_path)
        lock.acquire()
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
        lock.release()

    def test_lock_file_with_non_object_root_is_treated_as_reclaimable(self, tmp_path) -> None:
        """Regression test: previously the except tuple in
        `_read_existing_lock` had no `TypeError`, so a non-object root
        (e.g. a JSON array) would raise UNCAUGHT out of
        `_read_existing_lock` instead of being treated as reclaimable --
        a real, now-fixed gap."""
        lock_path = tmp_path / ".lock"
        lock_path.write_text("[1, 2, 3]")
        lock = DatasetLock(lock_path)
        lock.acquire()  # must not raise
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
        lock.release()

    def test_lock_file_with_duplicate_key_is_treated_as_reclaimable(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        lock_path.write_text('{"pid": 1, "pid": 2, "hostname": "h", "acquired_at": "2024-01-01T00:00:00+00:00"}')
        lock = DatasetLock(lock_path)
        lock.acquire()  # must not raise
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
        lock.release()

    def test_lock_file_with_invalid_utf8_is_treated_as_reclaimable(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        lock_path.write_bytes(b"\xff\xfe\x00invalid utf8 \x80\x81")
        lock = DatasetLock(lock_path)
        lock.acquire()  # must not raise
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
        lock.release()

    def test_lock_file_missing_owner_field_is_treated_as_reclaimable(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        lock_path.write_text('{"hostname": "h", "acquired_at": "2024-01-01T00:00:00+00:00"}')  # no "pid"
        lock = DatasetLock(lock_path)
        lock.acquire()  # must not raise
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
        lock.release()

    def test_lock_file_with_invalid_timestamp_is_treated_as_reclaimable(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        lock_path.write_text('{"pid": 1, "hostname": "h", "acquired_at": "not-a-timestamp"}')
        lock = DatasetLock(lock_path)
        lock.acquire()  # must not raise
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
        lock.release()

    def test_truncated_lock_file_is_treated_as_reclaimable(self, tmp_path) -> None:
        lock_path = tmp_path / ".lock"
        valid = LockInfo(pid=999_999, hostname="h", acquired_at=pd.Timestamp.now(tz="UTC"))
        full_text = json.dumps(valid.to_json_dict())
        lock_path.write_text(full_text[: len(full_text) // 2])  # truncated mid-object
        lock = DatasetLock(lock_path)
        lock.acquire()  # must not raise
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
        lock.release()


class TestDatasetLockPathHelper:
    def test_lock_path_is_a_sibling_of_the_dataset_directory(self, tmp_path) -> None:
        path = dataset_lock_path(tmp_path, symbol="XAUUSD", timeframe_value="M1")
        assert path.parent == tmp_path / "canonical" / "symbol=XAUUSD" / "timeframe=M1"
        assert path.name == ".lock"

    def test_lock_path_sanitizes_symbol(self, tmp_path) -> None:
        from quant_platform.core.exceptions import DataSourceError

        with pytest.raises(DataSourceError):
            dataset_lock_path(tmp_path, symbol="../escape", timeframe_value="M1")


class TestNoDoubleAcquisitionUnderForcedInterleaving:
    """Regression tests for a genuine race found during a Milestone 4C
    release-readiness concurrency audit (not merely "pre-existing and
    dismissed" -- reproduced deterministically, root-caused, and fixed).

    ROOT CAUSE (previous implementation): the lock file was created via
    `os.open(path, O_CREAT | O_EXCL | O_WRONLY)` -- exclusive-CREATE
    atomic -- and its JSON content was written in a SEPARATE, later step.
    Between those two steps, the lock path existed but was observably
    EMPTY. A concurrent contender that lost the `open` race, fell into
    `_handle_existing_lock`, and happened to call `_read_existing_lock`
    during exactly that window would see unparseable (empty) content and
    conclude the lock was corrupted -- a reasonable conclusion from what
    it could observe, but a STALE one. If that contender then acted on
    its stale conclusion (unlink + recreate) AFTER the true holder had
    already finished writing and released its file descriptor, BOTH
    ended up with `_held = True` simultaneously. Confirmed via a forced-
    interleaving reproduction (deterministic `threading.Event`
    synchronization, not sleep timing) before this fix; the SAME
    reproduction technique is what the tests below use to prove it can
    no longer happen.

    FIX: `DatasetLock._try_publish` now writes the full JSON content to a
    private, uniquely-named temp file FIRST, then publishes it at the
    shared lock path via `os.link` -- which fails atomically with
    `FileExistsError` if that path already exists, on both POSIX and
    Windows. The shared lock path is therefore NEVER observable in a
    content-less state: it either does not exist yet, or already has its
    complete, valid content the instant it exists at all."""

    def test_reader_never_observes_a_content_less_lock_file_even_when_forced_to_read_mid_publish(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Forces the EXACT interleaving that used to cause double
        acquisition: thread A is paused (via a deterministic `threading.
        Event`, never a sleep) with its lock content already written to
        its private temp file, but BEFORE `os.link` publishes it. Thread
        B is released only once thread A reaches that paused point, and
        immediately checks the shared lock path. The old implementation
        could show an EMPTY file here; the fix must show NO file at all
        (never a content-less one) -- `_read_existing_lock` must never
        be given a chance to misdiagnose a live acquisition as
        corrupted."""
        lock_path = tmp_path / ".lock"
        real_link = os.link
        writer_paused_before_link = threading.Event()
        allow_writer_to_link = threading.Event()

        def paused_link(src: str, dst: str) -> None:
            writer_paused_before_link.set()
            assert allow_writer_to_link.wait(timeout=5), "test failed to release the writer in time"
            real_link(src, dst)

        monkeypatch.setattr(os, "link", paused_link)

        writer = DatasetLock(lock_path)
        writer_thread = threading.Thread(target=writer.acquire)
        writer_thread.start()

        assert writer_paused_before_link.wait(timeout=5), "writer did not reach its pre-link pause in time"
        # The critical assertion: at this exact instant (content already
        # written to the writer's own private temp file, but NOT YET
        # linked into the shared path), the shared lock path must not
        # exist at all -- never present-but-empty/corrupted.
        assert not lock_path.exists(), "lock path must never be observable before its content is fully published"

        allow_writer_to_link.set()
        writer_thread.join(timeout=5)
        assert not writer_thread.is_alive()

        assert lock_path.exists()
        raw = json.loads(lock_path.read_text())  # must not raise -- always fully-formed JSON once present
        assert raw["pid"] == os.getpid()
        writer.release()

    def test_exactly_one_winner_when_two_threads_link_at_the_same_instant(self, tmp_path, monkeypatch) -> None:
        """Two threads' `os.link` publish attempts are released from a
        `threading.Barrier` at (as close as the platform allows) the
        SAME instant -- the strongest concurrent-acquisition pressure
        this lock can be put under. Exactly one must succeed; the other
        must fail loudly (`DatasetLockError`), never both, never neither,
        never a silently corrupted lock file."""
        lock_path = tmp_path / ".lock"
        real_link = os.link
        barrier = threading.Barrier(2, timeout=5)

        def synchronized_link(src: str, dst: str) -> None:
            barrier.wait()
            real_link(src, dst)

        monkeypatch.setattr(os, "link", synchronized_link)

        results: list[str] = []
        results_lock = threading.Lock()

        def attempt() -> None:
            lock = DatasetLock(lock_path)
            try:
                lock.acquire()
                with results_lock:
                    results.append("acquired")
            except DatasetLockError:
                with results_lock:
                    results.append("rejected")

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not any(t.is_alive() for t in threads)
        assert sorted(results) == ["acquired", "rejected"]
        assert json.loads(lock_path.read_text())  # exactly one, fully-valid lock file remains

    def test_exactly_one_winner_when_two_threads_race_to_reclaim_a_corrupted_lock(self, tmp_path, monkeypatch) -> None:
        """Milestone 4D.1 regression: the same barrier-synchronized race
        as above, but starting from a lock file corrupted in a way ONLY
        the new `parse_json_strict`-based reader distinguishes from the
        old plain `json.loads` (a bare `NaN` token) -- proves the parser
        migration did not weaken the underlying `os.link` atomicity that
        actually adjudicates the race. Both threads independently decide
        "reclaimable" from `_read_existing_lock`, but only one may
        actually win the `os.link` publish.

        Unlike the simpler race above (starting with NO pre-existing
        lock file, so each thread calls `os.link` exactly once), starting
        from an ALREADY-corrupted file means each thread's FIRST `os.link`
        (in `acquire`'s initial `_try_publish`) always loses to the
        pre-existing file, then reclaims and retries a SECOND `os.link`
        (in `_handle_existing_lock`) -- a variable number of calls per
        thread depending on scheduling. A single shared `threading.
        Barrier` synchronizing EVERY call (not just the first) can pair a
        slow thread's first call with a fast thread's second, leaving the
        slow thread's second call without a partner until it times out.
        Scoped via `threading.local()` to only the FIRST `os.link` call
        per thread -- exactly the pattern already established for the
        optimization engine's own multi-call concurrency tests -- so only
        the genuinely simultaneous, meaningful race (both threads
        discovering and first contending with the corrupted lock) is
        forced; each thread's own retry then races via `os.link`'s own
        real atomicity, unsynchronized, which is the property under test."""
        lock_path = tmp_path / ".lock"
        lock_path.write_text('{"pid": NaN, "hostname": "h", "acquired_at": "2024-01-01T00:00:00+00:00"}')
        real_link = os.link
        barrier = threading.Barrier(2, timeout=5)
        already_synced = threading.local()

        def synchronized_link(src: str, dst: str) -> None:
            if not getattr(already_synced, "done", False):
                already_synced.done = True
                barrier.wait()
            real_link(src, dst)

        monkeypatch.setattr(os, "link", synchronized_link)

        results: list[str] = []
        results_lock = threading.Lock()

        def attempt() -> None:
            lock = DatasetLock(lock_path)
            try:
                lock.acquire()
                with results_lock:
                    results.append("acquired")
            except DatasetLockError:
                with results_lock:
                    results.append("rejected")

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not any(t.is_alive() for t in threads)
        assert sorted(results) == ["acquired", "rejected"]
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
