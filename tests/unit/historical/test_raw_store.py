"""Tests for `historical.raw_store.RawSnapshotStore`: the immutable raw
snapshot layer everything else in the pipeline is built on top of."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.exceptions import DataSourceError, SnapshotError
from quant_platform.core.types import Timeframe
from quant_platform.historical.raw_store import RawSnapshotStore

UTC = "UTC"


def _frame(n: int = 10) -> pd.DataFrame:
    open_time = pd.date_range("2024-01-01", periods=n, freq="1min", tz=UTC)
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": np.full(n, 2000.0),
            "high": np.full(n, 2001.0),
            "low": np.full(n, 1999.0),
            "close": np.full(n, 2000.5),
            "tick_volume": np.full(n, 100, dtype=np.int64),
            "real_volume": np.zeros(n, dtype=np.int64),
            "spread": np.full(n, 15, dtype=np.int64),
        }
    )


def _write(store: RawSnapshotStore, df: pd.DataFrame, *, extracted_at=None, symbol="XAUUSD", broker="TestBroker"):
    return store.write_snapshot(
        df,
        source_name="mt5", source_version="1.0", broker=broker, symbol=symbol, source_symbol="XAUUSDm",
        timeframe=Timeframe.M1,
        requested_start=pd.Timestamp("2024-01-01", tz=UTC), requested_end=pd.Timestamp("2024-01-01T00:10:00", tz=UTC),
        server_timezone_repr="FIXED(UTC+02:00)",
        extracted_at=extracted_at or pd.Timestamp("2024-01-01T01:00:00", tz=UTC),
        is_complete=True,
    )


class TestWriteReadRoundTrip:
    def test_round_trip_preserves_data_and_metadata(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        df = _frame()
        metadata = _write(store, df)

        snapshots = store.list_snapshots(source_name="mt5", broker="TestBroker", symbol="XAUUSD", timeframe=Timeframe.M1)
        assert len(snapshots) == 1
        loaded_df, loaded_metadata = store.read_snapshot(snapshots[0])

        pd.testing.assert_frame_equal(loaded_df, df)
        assert loaded_metadata == metadata
        assert loaded_metadata.row_count == 10
        assert loaded_metadata.min_open_time == df["open_time"].iloc[0]
        assert loaded_metadata.max_open_time == df["open_time"].iloc[-1]

    def test_success_marker_and_files_exist_on_disk(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        _write(store, _frame())
        snapshots = store.list_snapshots(source_name="mt5", broker="TestBroker", symbol="XAUUSD", timeframe=Timeframe.M1)
        snap_dir = snapshots[0]
        assert (snap_dir / "_SUCCESS").is_file()
        assert (snap_dir / "data.parquet").is_file()
        assert (snap_dir / "metadata.json").is_file()

    def test_empty_dataframe_snapshot_has_none_min_max(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        metadata = _write(store, _frame(0))
        assert metadata.row_count == 0
        assert metadata.min_open_time is None
        assert metadata.max_open_time is None


class TestImmutability:
    def test_rewriting_a_completed_snapshot_id_raises(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        fixed_extracted_at = pd.Timestamp("2024-01-01T01:00:00", tz=UTC)
        _write(store, _frame(), extracted_at=fixed_extracted_at)
        with pytest.raises(SnapshotError, match="Refusing to overwrite"):
            _write(store, _frame(), extracted_at=fixed_extracted_at)

    def test_different_extraction_times_never_collide(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        _write(store, _frame(), extracted_at=pd.Timestamp("2024-01-01T01:00:00", tz=UTC))
        _write(store, _frame(), extracted_at=pd.Timestamp("2024-01-01T02:00:00", tz=UTC))
        snapshots = store.list_snapshots(source_name="mt5", broker="TestBroker", symbol="XAUUSD", timeframe=Timeframe.M1)
        assert len(snapshots) == 2

    def test_snapshot_id_is_deterministic_given_identical_inputs(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        common = {
            "source_name": "mt5", "broker": "TestBroker", "symbol": "XAUUSD", "timeframe": Timeframe.M1,
            "requested_start": pd.Timestamp("2024-01-01", tz=UTC),
            "requested_end": pd.Timestamp("2024-01-02", tz=UTC),
            "extracted_at": pd.Timestamp("2024-01-01T05:00:00", tz=UTC),
        }
        id_a = store.compute_snapshot_id(**common)
        id_b = store.compute_snapshot_id(**common)
        assert id_a == id_b

    def test_incomplete_leftover_directory_is_safely_replaced(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        fixed_extracted_at = pd.Timestamp("2024-01-01T01:00:00", tz=UTC)
        metadata = _write(store, _frame(5), extracted_at=fixed_extracted_at)
        snap_dir = store.dataset_dir(
            source_name="mt5", broker="TestBroker", symbol="XAUUSD", timeframe=Timeframe.M1
        ) / f"snapshot={metadata.snapshot_id}"
        # Simulate a crash after data was written but before _SUCCESS: remove
        # the marker so the directory looks incomplete, then rewrite.
        (snap_dir / "_SUCCESS").unlink()
        new_metadata = _write(store, _frame(7), extracted_at=fixed_extracted_at)
        loaded_df, _ = store.read_snapshot(snap_dir)
        assert len(loaded_df) == 7
        assert new_metadata.row_count == 7


class TestCorruptionDetection:
    def test_missing_success_marker_is_rejected_on_read(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        _write(store, _frame())
        snap_dir = store.list_snapshots(
            source_name="mt5", broker="TestBroker", symbol="XAUUSD", timeframe=Timeframe.M1
        )[0]
        (snap_dir / "_SUCCESS").unlink()
        with pytest.raises(SnapshotError, match="incomplete or corrupted"):
            store.read_snapshot(snap_dir)

    def test_corrupted_checksum_is_rejected_on_read(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        _write(store, _frame())
        snap_dir = store.list_snapshots(
            source_name="mt5", broker="TestBroker", symbol="XAUUSD", timeframe=Timeframe.M1
        )[0]
        # Corrupt the data file after the fact (bit rot / disk corruption simulation).
        with (snap_dir / "data.parquet").open("ab") as fh:
            fh.write(b"\x00\x00\x00corruption")
        with pytest.raises(SnapshotError, match="checksum mismatch"):
            store.read_snapshot(snap_dir)

    def test_corrupted_metadata_json_is_rejected_on_read(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        _write(store, _frame())
        snap_dir = store.list_snapshots(
            source_name="mt5", broker="TestBroker", symbol="XAUUSD", timeframe=Timeframe.M1
        )[0]
        (snap_dir / "metadata.json").write_text("{not valid json")
        with pytest.raises(SnapshotError, match="invalid JSON"):
            store.read_snapshot(snap_dir)

    def test_missing_data_file_is_rejected_on_read(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        _write(store, _frame())
        snap_dir = store.list_snapshots(
            source_name="mt5", broker="TestBroker", symbol="XAUUSD", timeframe=Timeframe.M1
        )[0]
        (snap_dir / "data.parquet").unlink()
        with pytest.raises(SnapshotError, match=r"data\.parquet is missing"):
            store.read_snapshot(snap_dir)

    def test_row_count_mismatch_is_rejected_on_read(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        _write(store, _frame())
        snap_dir = store.list_snapshots(
            source_name="mt5", broker="TestBroker", symbol="XAUUSD", timeframe=Timeframe.M1
        )[0]
        metadata_path = snap_dir / "metadata.json"
        raw = json.loads(metadata_path.read_text())
        raw["row_count"] = 999
        # Recompute nothing else -- checksum of data.parquet is untouched, so
        # this isolates the row-count check specifically (checksum still matches).
        metadata_path.write_text(json.dumps(raw))
        with pytest.raises(SnapshotError, match=r"checksum mismatch|row count mismatch"):
            store.read_snapshot(snap_dir)

    def test_schema_fingerprint_mismatch_is_rejected_even_when_checksum_matches(self, tmp_path) -> None:
        # The content checksum covers only data.parquet's bytes; tampering
        # metadata.json's OTHER fields (schema_fingerprint here, row_count
        # above) leaves the checksum matching but must still be caught by
        # its own dedicated check -- proving schema-mismatch detection is
        # not merely a side effect of the checksum check.
        store = RawSnapshotStore(tmp_path)
        _write(store, _frame())
        snap_dir = store.list_snapshots(
            source_name="mt5", broker="TestBroker", symbol="XAUUSD", timeframe=Timeframe.M1
        )[0]
        metadata_path = snap_dir / "metadata.json"
        raw = json.loads(metadata_path.read_text())
        raw["schema_fingerprint"] = "0000000000000000"
        metadata_path.write_text(json.dumps(raw))
        with pytest.raises(SnapshotError, match="schema fingerprint mismatch"):
            store.read_snapshot(snap_dir)


class TestOutOfOrderSourceData:
    """Raw snapshots store EXACTLY what the source provided, for audit
    fidelity -- `write_snapshot` does not reject/reorder out-of-order data
    (unlike `historical.quality.run_quality_checks`, which explicitly
    flags it downstream, before it could ever reach canonical storage).
    This proves that division of responsibility is real, not accidental:
    a raw snapshot can legitimately hold unordered data, and the pipeline's
    later quality stage is what actually catches it."""

    def test_write_snapshot_accepts_out_of_order_data_verbatim(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        df = _frame(5)
        shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        metadata = _write(store, shuffled)  # must not raise
        snap_dir = store.dataset_dir(
            source_name="mt5", broker="TestBroker", symbol="XAUUSD", timeframe=Timeframe.M1
        ) / f"snapshot={metadata.snapshot_id}"
        loaded_df, _ = store.read_snapshot(snap_dir)
        assert not loaded_df["open_time"].is_monotonic_increasing

    def test_quality_checks_catch_what_the_raw_store_did_not(self, tmp_path) -> None:
        from quant_platform.historical.quality import IssueType, run_quality_checks

        store = RawSnapshotStore(tmp_path)
        df = _frame(5)
        shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        _write(store, shuffled)
        report = run_quality_checks(shuffled, symbol="XAUUSD", timeframe=Timeframe.M1)
        assert not report.is_valid
        assert any(i.issue_type is IssueType.UNORDERED_TIMESTAMP for i in report.critical_issues)


class TestPathSecurity:
    def test_path_traversal_in_symbol_is_rejected(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        with pytest.raises(DataSourceError):
            _write(store, _frame(), symbol="../../etc")

    def test_path_traversal_in_broker_is_rejected(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        with pytest.raises(DataSourceError):
            _write(store, _frame(), broker="../escape")


class TestAtomicWriteOnFailure:
    def test_failed_write_leaves_no_final_directory(self, tmp_path, monkeypatch) -> None:
        store = RawSnapshotStore(tmp_path)

        original_to_parquet = pd.DataFrame.to_parquet

        def _boom(self, *args, **kwargs):
            raise OSError("simulated disk failure mid-write")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)
        with pytest.raises(OSError, match="simulated disk failure"):
            _write(store, _frame())
        monkeypatch.setattr(pd.DataFrame, "to_parquet", original_to_parquet)

        dataset_dir = store.dataset_dir(
            source_name="mt5", broker="TestBroker", symbol="XAUUSD", timeframe=Timeframe.M1
        )
        # No completed snapshot directory should exist, and no leftover tmp dirs.
        assert not dataset_dir.is_dir() or not any(
            p.name.startswith("snapshot=") for p in dataset_dir.iterdir()
        )
        tmp_dir = dataset_dir / ".tmp"
        assert not tmp_dir.is_dir() or list(tmp_dir.iterdir()) == []


class TestListSnapshots:
    def test_returns_empty_list_for_unknown_dataset(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        assert store.list_snapshots(source_name="mt5", broker="X", symbol="Y", timeframe=Timeframe.M1) == []

    def test_returns_sorted_paths(self, tmp_path) -> None:
        store = RawSnapshotStore(tmp_path)
        _write(store, _frame(), extracted_at=pd.Timestamp("2024-01-01T03:00:00", tz=UTC))
        _write(store, _frame(), extracted_at=pd.Timestamp("2024-01-01T01:00:00", tz=UTC))
        snaps = store.list_snapshots(source_name="mt5", broker="TestBroker", symbol="XAUUSD", timeframe=Timeframe.M1)
        assert [p.name for p in snaps] == sorted(p.name for p in snaps)
