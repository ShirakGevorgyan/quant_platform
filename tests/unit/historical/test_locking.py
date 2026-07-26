"""Tests for `historical.locking.DatasetLock` -- dataset-level advisory
locking with stale-lock recovery, added during the Milestone 2
release-readiness audit."""

from __future__ import annotations

import json
import os

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


class TestDatasetLockPathHelper:
    def test_lock_path_is_a_sibling_of_the_dataset_directory(self, tmp_path) -> None:
        path = dataset_lock_path(tmp_path, symbol="XAUUSD", timeframe_value="M1")
        assert path.parent == tmp_path / "canonical" / "symbol=XAUUSD" / "timeframe=M1"
        assert path.name == ".lock"

    def test_lock_path_sanitizes_symbol(self, tmp_path) -> None:
        from quant_platform.core.exceptions import DataSourceError

        with pytest.raises(DataSourceError):
            dataset_lock_path(tmp_path, symbol="../escape", timeframe_value="M1")
