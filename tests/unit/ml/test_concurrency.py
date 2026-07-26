"""Direct unit tests for `ml.concurrency.experiment_lock` -- the ML
boundary adapter added during the post-4A correctness audit to stop a
raw `DatasetLockError`/`PermissionError` from `historical.locking.
DatasetLock` leaking past `manifests.py`/`tracking.py`. Complementary to
the deterministic translation tests embedded in `test_ml_manifests.py`/
`test_tracking.py`, which exercise this through those stores; this file
tests the adapter itself in isolation, including the raw-`OSError`
acquire path (the rare Windows stale-lock-reclaim race), which is hard
to trigger deterministically through a real `DatasetLock` and is
therefore covered here via a mocked `.acquire()`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from quant_platform.core.exceptions import DatasetLockError, ExperimentLockError
from quant_platform.historical.locking import DatasetLock
from quant_platform.ml.concurrency import experiment_lock


class TestHappyPath:
    def test_acquires_and_releases_normally(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".lock"
        with experiment_lock(lock_path):
            assert lock_path.is_file()
        assert not lock_path.is_file()

    def test_body_return_value_and_exceptions_pass_through_untouched(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".lock"
        with pytest.raises(ValueError, match="body failure"), experiment_lock(lock_path):
            raise ValueError("body failure")
        # Lock still released even though the body raised.
        assert not lock_path.is_file()


class TestContestedLockTranslation:
    def test_dataset_lock_error_becomes_experiment_lock_error(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".lock"
        holder = DatasetLock(lock_path)
        holder.acquire()
        try:
            with pytest.raises(ExperimentLockError) as exc_info, experiment_lock(lock_path):
                pass  # pragma: no cover - must never be reached
            assert isinstance(exc_info.value.__cause__, DatasetLockError)
        finally:
            holder.release()

    def test_raw_os_error_during_acquire_becomes_experiment_lock_error(self, tmp_path: Path) -> None:
        """The rare, documented Windows race in `DatasetLock`'s own
        stale-lock-reclaim path can surface a raw `OSError` (e.g.
        `PermissionError`) rather than its own typed `DatasetLockError`.
        Simulated here via a mocked `.acquire()` since reproducing the
        real race deterministically is not practical."""
        lock_path = tmp_path / ".lock"
        with patch.object(DatasetLock, "acquire", side_effect=PermissionError("simulated Windows reclaim race")), \
                pytest.raises(ExperimentLockError) as exc_info, \
                experiment_lock(lock_path):
            pass  # pragma: no cover - must never be reached
        assert isinstance(exc_info.value.__cause__, PermissionError)

    def test_lock_is_free_again_after_a_translated_failure(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".lock"
        holder = DatasetLock(lock_path)
        holder.acquire()
        try:
            with pytest.raises(ExperimentLockError), experiment_lock(lock_path):
                pass  # pragma: no cover - must never be reached
        finally:
            holder.release()
        # A subsequent, uncontested acquisition succeeds normally.
        with experiment_lock(lock_path):
            assert lock_path.is_file()
