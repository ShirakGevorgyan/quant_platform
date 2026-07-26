"""Tests for `historical.update_pipeline`."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone as dt_timezone

import pandas as pd
import pytest

from quant_platform.core.exceptions import DatasetLockError, UpdateConflictError
from quant_platform.core.types import Timeframe
from quant_platform.data.synthetic import SyntheticDataConfig, generate_ohlcv
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.locking import DatasetLock, dataset_lock_path
from quant_platform.historical.manifest import ManifestStore
from quant_platform.historical.models import coerce_historical_dtypes
from quant_platform.historical.update_pipeline import (
    RevisionPolicy,
    apply_incremental_update,
    determine_update_start,
)

UTC = "UTC"


def _frame(start: datetime, n: int, seed: int = 1) -> pd.DataFrame:
    sd = generate_ohlcv(SyntheticDataConfig(start=start, periods=n, timeframe=Timeframe.M1, seed=seed))
    sd = sd.rename(columns={"volume": "tick_volume"})
    sd["tick_volume"] = sd["tick_volume"].astype("int64")
    sd["real_volume"] = 0
    sd["spread"] = 15
    return coerce_historical_dtypes(sd)


@pytest.fixture
def stores(tmp_path):
    return CanonicalStore(tmp_path), ManifestStore(tmp_path)


def _common_kwargs() -> dict[str, object]:
    return {
        "symbol": "XAUUSD", "timeframe": Timeframe.M1, "source_name": "mt5", "broker": "TestBroker",
        "pipeline_version": "1.0.0", "parent_snapshot_ids": ("snap1",),
    }


class TestDetermineUpdateStart:
    def test_returns_none_when_no_canonical_data_exists(self, stores) -> None:
        cstore, _ = stores
        assert determine_update_start(cstore, symbol="XAUUSD", timeframe=Timeframe.M1) is None

    def test_returns_latest_minus_overlap(self, stores) -> None:
        cstore, mstore = stores
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 100)
        apply_incremental_update(
            cstore, mstore, df, requested_start=df["open_time"].iloc[0],
            requested_end=df["open_time"].iloc[-1] + pd.Timedelta(minutes=1), **_common_kwargs(),
        )
        start = determine_update_start(cstore, symbol="XAUUSD", timeframe=Timeframe.M1, overlap_bars=5)
        assert start == df["open_time"].iloc[-1] - 5 * pd.Timedelta(minutes=1)

    def test_rejects_negative_overlap(self, stores) -> None:
        cstore, _ = stores
        with pytest.raises(ValueError, match="non-negative"):
            determine_update_start(cstore, symbol="XAUUSD", timeframe=Timeframe.M1, overlap_bars=-1)


class TestFreshIngest:
    def test_first_update_inserts_everything(self, stores) -> None:
        cstore, mstore = stores
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 100)
        report = apply_incremental_update(
            cstore, mstore, df, requested_start=df["open_time"].iloc[0],
            requested_end=df["open_time"].iloc[-1] + pd.Timedelta(minutes=1), **_common_kwargs(),
        )
        assert report.rows_received == 100
        assert report.rows_inserted == 100
        assert report.rows_unchanged == 0
        assert report.rows_conflicting == 0
        assert report.final_row_count == 100


class TestIdempotency:
    def test_rerunning_the_identical_update_produces_the_same_manifest_version(self, stores) -> None:
        cstore, mstore = stores
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 100)
        kwargs = dict(
            requested_start=df["open_time"].iloc[0], requested_end=df["open_time"].iloc[-1] + pd.Timedelta(minutes=1),
            **_common_kwargs(),
        )
        first = apply_incremental_update(cstore, mstore, df, **kwargs)
        second = apply_incremental_update(cstore, mstore, df, **kwargs)
        assert first.manifest_version == second.manifest_version
        assert first.final_checksum == second.final_checksum
        assert second.rows_unchanged == 100
        assert second.rows_inserted == 0

    def test_rerun_does_not_create_extra_manifest_versions(self, stores) -> None:
        cstore, mstore = stores
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 100)
        kwargs = dict(
            requested_start=df["open_time"].iloc[0], requested_end=df["open_time"].iloc[-1] + pd.Timedelta(minutes=1),
            **_common_kwargs(),
        )
        apply_incremental_update(cstore, mstore, df, **kwargs)
        apply_incremental_update(cstore, mstore, df, **kwargs)
        apply_incremental_update(cstore, mstore, df, **kwargs)
        assert len(mstore.list_versions(symbol="XAUUSD", timeframe=Timeframe.M1)) == 1


class TestOverlapReconciliation:
    def test_extension_with_real_overlap_classifies_correctly(self, stores) -> None:
        cstore, mstore = stores
        df1 = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 100)
        apply_incremental_update(
            cstore, mstore, df1, requested_start=df1["open_time"].iloc[0],
            requested_end=df1["open_time"].iloc[-1] + pd.Timedelta(minutes=1), **_common_kwargs(),
        )
        tail = df1.tail(5).reset_index(drop=True)
        new_part = _frame(df1["open_time"].iloc[-1] + pd.Timedelta(minutes=1), 50, seed=99)
        df2 = pd.concat([tail, new_part]).reset_index(drop=True)
        report = apply_incremental_update(
            cstore, mstore, df2, requested_start=tail["open_time"].iloc[0],
            requested_end=df2["open_time"].iloc[-1] + pd.Timedelta(minutes=1), **_common_kwargs(),
        )
        assert report.rows_unchanged == 5
        assert report.rows_inserted == 50
        assert report.rows_conflicting == 0
        assert report.final_row_count == 150

    def test_existing_bars_outside_the_update_window_are_untouched(self, stores) -> None:
        cstore, mstore = stores
        df1 = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 100)
        apply_incremental_update(
            cstore, mstore, df1, requested_start=df1["open_time"].iloc[0],
            requested_end=df1["open_time"].iloc[-1] + pd.Timedelta(minutes=1), **_common_kwargs(),
        )
        tail = df1.tail(2).reset_index(drop=True)
        new_part = _frame(df1["open_time"].iloc[-1] + pd.Timedelta(minutes=1), 10, seed=42)
        df2 = pd.concat([tail, new_part]).reset_index(drop=True)
        apply_incremental_update(
            cstore, mstore, df2, requested_start=tail["open_time"].iloc[0],
            requested_end=df2["open_time"].iloc[-1] + pd.Timedelta(minutes=1), **_common_kwargs(),
        )
        loaded_df, _ = cstore.read_partition(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        # The first 90 bars (untouched by the second update's window) must
        # be byte-identical to what the first update wrote.
        pd.testing.assert_frame_equal(
            loaded_df.iloc[:90].reset_index(drop=True), df1.iloc[:90].reset_index(drop=True)
        )


class TestConflictingRevisions:
    def _setup_conflict(self, stores) -> tuple[CanonicalStore, ManifestStore, pd.DataFrame, pd.DataFrame]:
        cstore, mstore = stores
        df1 = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 100)
        apply_incremental_update(
            cstore, mstore, df1, requested_start=df1["open_time"].iloc[0],
            requested_end=df1["open_time"].iloc[-1] + pd.Timedelta(minutes=1), **_common_kwargs(),
        )
        conflict_df = df1.iloc[[10]].copy()
        conflict_df["close"] = conflict_df["close"] + 100.0
        return cstore, mstore, df1, conflict_df

    def test_reject_conflicts_raises_and_never_replace_by_default(self, stores) -> None:
        cstore, mstore, _, conflict_df = self._setup_conflict(stores)
        with pytest.raises(UpdateConflictError, match="conflict with already-canonicalized"):
            apply_incremental_update(
                cstore, mstore, conflict_df, requested_start=conflict_df["open_time"].iloc[0],
                requested_end=conflict_df["open_time"].iloc[0] + pd.Timedelta(minutes=1),
                revision_policy=RevisionPolicy.REJECT_CONFLICTS, **_common_kwargs(),
            )
        # Nothing must have been written -- the historical bar is unchanged.
        loaded_df, _ = cstore.read_partition(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        original_row = loaded_df[loaded_df["open_time"] == conflict_df["open_time"].iloc[0]]
        assert original_row["close"].iloc[0] != conflict_df["close"].iloc[0]

    def test_accept_newer_source_explicitly_replaces_and_reports_it(self, stores) -> None:
        cstore, mstore, _df1, conflict_df = self._setup_conflict(stores)
        report = apply_incremental_update(
            cstore, mstore, conflict_df, requested_start=conflict_df["open_time"].iloc[0],
            requested_end=conflict_df["open_time"].iloc[0] + pd.Timedelta(minutes=1),
            revision_policy=RevisionPolicy.ACCEPT_NEWER_SOURCE, **_common_kwargs(),
        )
        assert report.rows_conflicting == 1
        assert report.final_row_count == 100  # replacement, not an insertion
        loaded_df, _ = cstore.read_partition(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        revised_row = loaded_df[loaded_df["open_time"] == conflict_df["open_time"].iloc[0]]
        assert revised_row["close"].iloc[0] == conflict_df["close"].iloc[0]


class TestInputValidation:
    def test_empty_new_bars_raises(self, stores) -> None:
        cstore, mstore = stores
        empty = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 1).iloc[0:0]
        with pytest.raises(ValueError, match="must not be empty"):
            apply_incremental_update(
                cstore, mstore, empty, requested_start=pd.Timestamp("2024-01-03", tz=UTC),
                requested_end=pd.Timestamp("2024-01-04", tz=UTC), **_common_kwargs(),
            )

    def test_unsorted_new_bars_raises(self, stores) -> None:
        cstore, mstore = stores
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 10)
        shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        with pytest.raises(UpdateConflictError, match="sorted ascending"):
            apply_incremental_update(
                cstore, mstore, shuffled, requested_start=df["open_time"].iloc[0],
                requested_end=df["open_time"].iloc[-1] + pd.Timedelta(minutes=1), **_common_kwargs(),
            )


class TestCrashRecovery:
    def test_a_failed_write_leaves_the_prior_partition_valid_and_recoverable(self, stores, monkeypatch) -> None:
        cstore, mstore = stores
        df1 = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 100)
        apply_incremental_update(
            cstore, mstore, df1, requested_start=df1["open_time"].iloc[0],
            requested_end=df1["open_time"].iloc[-1] + pd.Timedelta(minutes=1), **_common_kwargs(),
        )

        def _boom(self, *args, **kwargs):
            raise OSError("simulated crash mid-write")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)
        df2 = _frame(datetime(2024, 1, 4, tzinfo=dt_timezone.utc), 50, seed=2)
        with pytest.raises(OSError, match="simulated crash"):
            apply_incremental_update(
                cstore, mstore, df2, requested_start=df2["open_time"].iloc[0],
                requested_end=df2["open_time"].iloc[-1] + pd.Timedelta(minutes=1), **_common_kwargs(),
            )
        monkeypatch.undo()

        loaded_df, _ = cstore.read_partition(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        assert len(loaded_df) == 100
        assert mstore.list_versions(symbol="XAUUSD", timeframe=Timeframe.M1) == [
            mstore.load(symbol="XAUUSD", timeframe=Timeframe.M1).version
        ]

    def test_re_running_after_a_crash_recovers_cleanly(self, stores, monkeypatch) -> None:
        cstore, mstore = stores
        df1 = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 100)
        apply_incremental_update(
            cstore, mstore, df1, requested_start=df1["open_time"].iloc[0],
            requested_end=df1["open_time"].iloc[-1] + pd.Timedelta(minutes=1), **_common_kwargs(),
        )
        df2 = _frame(datetime(2024, 1, 4, tzinfo=dt_timezone.utc), 50, seed=2)
        kwargs2 = dict(
            requested_start=df2["open_time"].iloc[0], requested_end=df2["open_time"].iloc[-1] + pd.Timedelta(minutes=1),
            **_common_kwargs(),
        )

        def _boom(self, *args, **kwargs):
            raise OSError("simulated crash mid-write")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)
        with pytest.raises(OSError):
            apply_incremental_update(cstore, mstore, df2, **kwargs2)
        monkeypatch.undo()

        # Re-running the SAME update after "fixing" the crash must succeed
        # and produce the fully-updated dataset.
        report = apply_incremental_update(cstore, mstore, df2, **kwargs2)
        assert report.final_row_count == 150


class TestConcurrencyLocking:
    """Release-readiness-audit coverage: `apply_incremental_update` holds
    a `historical.locking.DatasetLock` for the duration of the update, so
    two concurrent updates against the SAME dataset fail fast rather than
    racing on the underlying storage."""

    def test_a_concurrent_update_against_the_same_dataset_is_rejected(self, stores) -> None:
        cstore, mstore = stores
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 50)
        lock_path = dataset_lock_path(cstore.root, symbol="XAUUSD", timeframe_value="M1")

        held_lock = DatasetLock(lock_path)
        held_lock.acquire()
        try:
            with pytest.raises(DatasetLockError):
                apply_incremental_update(
                    cstore, mstore, df, requested_start=df["open_time"].iloc[0],
                    requested_end=df["open_time"].iloc[-1] + pd.Timedelta(minutes=1), **_common_kwargs(),
                )
        finally:
            held_lock.release()

    def test_the_lock_is_released_after_a_successful_update(self, stores) -> None:
        cstore, mstore = stores
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 50)
        apply_incremental_update(
            cstore, mstore, df, requested_start=df["open_time"].iloc[0],
            requested_end=df["open_time"].iloc[-1] + pd.Timedelta(minutes=1), **_common_kwargs(),
        )
        lock_path = dataset_lock_path(cstore.root, symbol="XAUUSD", timeframe_value="M1")
        assert not lock_path.is_file()

    def test_the_lock_is_released_after_a_failed_update(self, stores) -> None:
        cstore, mstore = stores
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 50)
        apply_incremental_update(
            cstore, mstore, df, requested_start=df["open_time"].iloc[0],
            requested_end=df["open_time"].iloc[-1] + pd.Timedelta(minutes=1), **_common_kwargs(),
        )
        conflict_df = df.iloc[[5]].copy()
        conflict_df["close"] = conflict_df["close"] + 100.0
        with pytest.raises(UpdateConflictError):
            apply_incremental_update(
                cstore, mstore, conflict_df, requested_start=conflict_df["open_time"].iloc[0],
                requested_end=conflict_df["open_time"].iloc[0] + pd.Timedelta(minutes=1),
                revision_policy=RevisionPolicy.REJECT_CONFLICTS, **_common_kwargs(),
            )
        lock_path = dataset_lock_path(cstore.root, symbol="XAUUSD", timeframe_value="M1")
        assert not lock_path.is_file()

    def test_different_datasets_do_not_contend_for_the_same_lock(self, stores) -> None:
        cstore, mstore = stores
        df_h1 = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 20, seed=2)
        m1_lock_path = dataset_lock_path(cstore.root, symbol="XAUUSD", timeframe_value="M1")
        held_lock = DatasetLock(m1_lock_path)
        held_lock.acquire()
        try:
            # An update against a DIFFERENT timeframe must not be blocked
            # by the M1 lock.
            report = apply_incremental_update(
                cstore, mstore, df_h1, requested_start=df_h1["open_time"].iloc[0],
                requested_end=df_h1["open_time"].iloc[-1] + pd.Timedelta(minutes=1),
                symbol="XAUUSD", timeframe=Timeframe.H1, source_name="mt5", broker="TestBroker",
                pipeline_version="1.0.0", parent_snapshot_ids=("snap1",),
            )
            assert report.rows_inserted == 20
        finally:
            held_lock.release()
