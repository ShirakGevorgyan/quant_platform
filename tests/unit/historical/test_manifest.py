"""Tests for `historical.manifest.DatasetManifest`/`ManifestStore`."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.exceptions import ManifestError
from quant_platform.core.types import Timeframe
from quant_platform.historical.manifest import DatasetManifest, ManifestStore


def _manifest(store: ManifestStore, *, row_count: int = 100, checksum: str = "abc123def456") -> DatasetManifest:
    dataset_id = store.compute_dataset_id(symbol="XAUUSD", timeframe=Timeframe.M1, source_name="mt5", broker="TestBroker")
    return DatasetManifest(
        dataset_id=dataset_id, version="PENDING", parent_snapshot_ids=("snap1", "snap2"),
        symbol="XAUUSD", source_name="mt5", broker="TestBroker", timeframe=Timeframe.M1,
        utc_start=pd.Timestamp("2024-01-01", tz="UTC"), utc_end=pd.Timestamp("2024-02-01", tz="UTC"),
        row_count=row_count, schema_fingerprint="fp123", content_checksum=checksum,
        created_at=pd.Timestamp.now(tz="UTC"), pipeline_version="1.0.0",
        normalization_settings={"server_tz": "UTC+2"}, validation_policy="STRICT",
        quality_summary={"critical": 0, "warning": 1}, repair_summary={"rows_removed": 0},
    )


class TestVersioning:
    def test_version_is_not_merely_a_date(self, tmp_path) -> None:
        store = ManifestStore(tmp_path)
        version = store.save(_manifest(store))
        assert version.startswith("000001-")
        assert "-" in version
        # The version must not parse as (or reduce to) a bare calendar date.
        with pytest.raises(ValueError):
            pd.Timestamp(version)

    def test_first_save_is_sequence_one(self, tmp_path) -> None:
        store = ManifestStore(tmp_path)
        version = store.save(_manifest(store))
        assert version.split("-")[0] == "000001"

    def test_second_distinct_save_increments_sequence(self, tmp_path) -> None:
        store = ManifestStore(tmp_path)
        v1 = store.save(_manifest(store, row_count=100, checksum="aaa"))
        v2 = store.save(_manifest(store, row_count=200, checksum="bbb"))
        assert v1.split("-")[0] == "000001"
        assert v2.split("-")[0] == "000002"
        assert v1 != v2


class TestIdempotentSave:
    def test_saving_identical_content_again_does_not_create_a_new_version(self, tmp_path) -> None:
        store = ManifestStore(tmp_path)
        manifest = _manifest(store)
        v1 = store.save(manifest)
        v_again = store.save(manifest)
        assert v1 == v_again
        assert store.list_versions(symbol="XAUUSD", timeframe=Timeframe.M1) == [v1]

    def test_saving_different_content_creates_a_new_version(self, tmp_path) -> None:
        store = ManifestStore(tmp_path)
        v1 = store.save(_manifest(store, row_count=100, checksum="aaa"))
        v2 = store.save(_manifest(store, row_count=101, checksum="bbb"))
        assert v1 != v2
        assert len(store.list_versions(symbol="XAUUSD", timeframe=Timeframe.M1)) == 2


class TestLoad:
    def test_load_without_version_returns_latest(self, tmp_path) -> None:
        store = ManifestStore(tmp_path)
        store.save(_manifest(store, row_count=100, checksum="aaa"))
        store.save(_manifest(store, row_count=200, checksum="bbb"))
        latest = store.load(symbol="XAUUSD", timeframe=Timeframe.M1)
        assert latest.row_count == 200

    def test_load_specific_version_returns_that_exact_version(self, tmp_path) -> None:
        store = ManifestStore(tmp_path)
        v1 = store.save(_manifest(store, row_count=100, checksum="aaa"))
        store.save(_manifest(store, row_count=200, checksum="bbb"))
        specific = store.load(symbol="XAUUSD", timeframe=Timeframe.M1, version=v1)
        assert specific.row_count == 100

    def test_load_missing_dataset_raises(self, tmp_path) -> None:
        store = ManifestStore(tmp_path)
        with pytest.raises(ManifestError, match="No manifest versions"):
            store.load(symbol="XAUUSD", timeframe=Timeframe.H1)

    def test_load_nonexistent_version_raises(self, tmp_path) -> None:
        store = ManifestStore(tmp_path)
        store.save(_manifest(store))
        with pytest.raises(ManifestError, match="not found"):
            store.load(symbol="XAUUSD", timeframe=Timeframe.M1, version="999999-doesnotexist")

    def test_load_rejects_a_corrupted_manifest_file(self, tmp_path) -> None:
        store = ManifestStore(tmp_path)
        version = store.save(_manifest(store))
        dataset_dir = store._dataset_dir(symbol="XAUUSD", timeframe=Timeframe.M1)
        (dataset_dir / f"{version}.json").write_text("{not valid json at all")
        with pytest.raises(ManifestError, match="corrupted"):
            store.load(symbol="XAUUSD", timeframe=Timeframe.M1, version=version)

    def test_load_latest_rejects_a_corrupted_manifest_file(self, tmp_path) -> None:
        store = ManifestStore(tmp_path)
        version = store.save(_manifest(store))
        dataset_dir = store._dataset_dir(symbol="XAUUSD", timeframe=Timeframe.M1)
        (dataset_dir / f"{version}.json").write_text("{not valid json at all")
        with pytest.raises(ManifestError, match="corrupted"):
            store.load(symbol="XAUUSD", timeframe=Timeframe.M1)


class TestRoundTripFidelity:
    def test_all_fields_survive_json_round_trip(self, tmp_path) -> None:
        store = ManifestStore(tmp_path)
        original = _manifest(store)
        version = store.save(original)
        loaded = store.load(symbol="XAUUSD", timeframe=Timeframe.M1)
        assert loaded.dataset_id == original.dataset_id
        assert loaded.version == version
        assert loaded.parent_snapshot_ids == original.parent_snapshot_ids
        assert loaded.utc_start == original.utc_start
        assert loaded.utc_end == original.utc_end
        assert loaded.normalization_settings == original.normalization_settings
        assert loaded.quality_summary == original.quality_summary
        assert loaded.repair_summary == original.repair_summary
        assert loaded.calendar_version is None
        assert loaded.resampling_config is None


class TestManifestImmutability:
    def test_writing_the_exact_same_version_file_twice_directly_raises(self, tmp_path) -> None:
        store = ManifestStore(tmp_path)
        manifest = _manifest(store)
        dataset_dir = store._dataset_dir(symbol="XAUUSD", timeframe=Timeframe.M1)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        from dataclasses import replace

        versioned = replace(manifest, version="000001-abc123def456")
        store._write_version_file(dataset_dir, "000001-abc123def456", versioned)
        with pytest.raises(ManifestError, match="immutable"):
            store._write_version_file(dataset_dir, "000001-abc123def456", versioned)


class TestDatasetIdStability:
    def test_dataset_id_is_stable_across_versions(self, tmp_path) -> None:
        store = ManifestStore(tmp_path)
        v1_manifest = _manifest(store, row_count=100, checksum="aaa")
        v2_manifest = _manifest(store, row_count=200, checksum="bbb")
        assert v1_manifest.dataset_id == v2_manifest.dataset_id

    def test_dataset_id_differs_for_different_symbols(self, tmp_path) -> None:
        store = ManifestStore(tmp_path)
        id_xau = store.compute_dataset_id(symbol="XAUUSD", timeframe=Timeframe.M1, source_name="mt5", broker="B")
        id_eur = store.compute_dataset_id(symbol="EURUSD", timeframe=Timeframe.M1, source_name="mt5", broker="B")
        assert id_xau != id_eur
