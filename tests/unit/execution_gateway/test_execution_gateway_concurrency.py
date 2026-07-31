"""Concurrency tests for `execution_gateway.manifests.execution_session_lock`
(Milestone 8, Section 20). `ml.concurrency.experiment_lock`'s own
concurrent-access semantics are already exhaustively tested at the source
(`tests/unit/ml/test_concurrency.py`) and reused here UNCHANGED (per
Section 20's own "reuse existing infra" instruction) -- this file's job is
narrower: proving `execution_session_lock`'s own TRANSLATION layer (a
contested acquisition becomes `ExecutionSessionLockError`, never a raw
`ExperimentLockError` or an uncaught exception) against a GENUINE, real,
concurrently-held lock, and that a REAL exception raised INSIDE the
protected block is never misclassified as a lock failure.

REGRESSION COVERAGE: a real defect was found and fixed during this
milestone's own development -- `execution_session_lock` originally caught
the broad `QuantPlatformError` (not the narrow `ExperimentLockError`),
which incorrectly re-wrapped ANY exception raised inside the `with` block
(not just genuine lock-acquisition failures) as a misleading
`ExecutionSessionLockError`, masking the real error. No regression test
existed for this fix until now."""

from __future__ import annotations

import threading
import time

import pytest

from quant_platform.core.exceptions import ExecutionGatewayArtifactError, ExecutionSessionLockError
from quant_platform.execution_gateway.manifests import execution_session_lock
from quant_platform.historical.locking import DatasetLock


class TestContestedLockAcquisitionIsTranslated:
    def test_a_lock_already_held_by_another_holder_raises_execution_session_lock_error(self, tmp_path) -> None:
        lock_path = tmp_path / "session.lock"
        holder = DatasetLock(lock_path)
        holder.acquire()
        try:
            with pytest.raises(ExecutionSessionLockError), execution_session_lock(lock_path):
                pass  # pragma: no cover -- must never be reached
        finally:
            holder.release()

    def test_lock_is_released_after_a_successful_block_and_can_be_reacquired(self, tmp_path) -> None:
        lock_path = tmp_path / "session.lock"
        with execution_session_lock(lock_path):
            pass
        # A second, independent acquisition must succeed now that the first was released.
        with execution_session_lock(lock_path):
            pass

    def test_lock_is_released_even_when_the_protected_block_raises(self, tmp_path) -> None:
        lock_path = tmp_path / "session.lock"
        try:
            with execution_session_lock(lock_path):
                raise ExecutionGatewayArtifactError("simulated failure inside the protected block")
        except ExecutionGatewayArtifactError:
            pass
        # The lock must have been released on the way out despite the exception --
        # a fresh acquisition must succeed immediately, never blocked by a leaked lock.
        with execution_session_lock(lock_path):
            pass


class TestExceptionsFromInsideTheProtectedBlockAreNeverMisclassified:
    """Regression test for the real defect described in this module's own
    docstring: a genuine domain exception raised INSIDE the `with
    execution_session_lock(...):` block must propagate as ITSELF, never
    be silently re-wrapped as a misleading `ExecutionSessionLockError`."""

    def test_a_domain_exception_raised_inside_the_block_propagates_unchanged(self, tmp_path) -> None:
        lock_path = tmp_path / "session.lock"
        raised = None
        try:
            with execution_session_lock(lock_path):
                raise ExecutionGatewayArtifactError("a genuine domain error, unrelated to locking")
        except Exception as exc:
            raised = exc
        assert isinstance(raised, ExecutionGatewayArtifactError)
        assert not isinstance(raised, ExecutionSessionLockError)


class TestConcurrentThreadsContendForTheSameLock:
    def test_two_threads_racing_for_the_same_lock_path_never_both_succeed_simultaneously(self, tmp_path) -> None:
        lock_path = tmp_path / "session.lock"
        overlap_detected = False
        currently_inside = threading.Event()
        results: list[str] = []

        def _worker(label: str) -> None:
            nonlocal overlap_detected
            try:
                with execution_session_lock(lock_path):
                    if currently_inside.is_set():
                        overlap_detected = True
                    currently_inside.set()
                    time.sleep(0.05)
                    currently_inside.clear()
                    results.append(f"{label}:acquired")
            except ExecutionSessionLockError:
                results.append(f"{label}:contested")

        t1 = threading.Thread(target=_worker, args=("t1",))
        t2 = threading.Thread(target=_worker, args=("t2",))
        t1.start()
        time.sleep(0.01)  # give t1 a head start so the race is deterministic-ish, not required for correctness
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not overlap_detected, "two threads must never simultaneously hold the same execution session lock"
        assert len(results) == 2
        assert any("acquired" in r for r in results), "at least one thread must have successfully acquired the lock"
