"""Unit tests for `market_data.manifests`: `DatasetKey` validation,
`DatasetManifest` identity/validation, and `DatasetManifestStore`'s
append-only version-history semantics."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_platform.core.exceptions import DatasetManifestError
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    DatasetManifestStore,
    PartitionGranularity,
    PartitioningSpec,
    create_dataset_manifest,
)

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
_SPEC = PartitioningSpec(granularity=PartitionGranularity.DAILY)


def _raw_key(**overrides: object) -> DatasetKey:
    base: dict[str, object] = {"dataset_kind": DatasetKind.RAW_MARKET_EVENTS, "instrument_id": "mt5__XAUUSD", "provider": "mt5"}
    base.update(overrides)
    return DatasetKey(**base)  # type: ignore[arg-type]


def _feature_key(**overrides: object) -> DatasetKey:
    base: dict[str, object] = {"dataset_kind": DatasetKind.DERIVED_FEATURES, "instrument_id": "mt5__XAUUSD", "feature_name": "sma_20", "feature_version": 1}
    base.update(overrides)
    return DatasetKey(**base)  # type: ignore[arg-type]


def _manifest(**overrides: object):
    base: dict[str, object] = {
        "dataset_key": _raw_key(), "schema_version": 1, "timeframe": None, "partitioning": _SPEC, "first_event_time": _T0,
        "last_event_time": _T0 + timedelta(hours=1), "event_count": 2, "ordered_partition_ids": ("a" * 64,), "raw_source_dataset_id": None,
        "semantic_digest": "b" * 64, "physical_digest": "c" * 64, "creation_time": _T0,
    }
    base.update(overrides)
    return create_dataset_manifest(**base)  # type: ignore[arg-type]


class TestDatasetKeyValidation:
    def test_raw_key_requires_provider(self) -> None:
        with pytest.raises(DatasetManifestError):
            DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="i")

    def test_raw_key_forbids_feature_fields(self) -> None:
        with pytest.raises(DatasetManifestError):
            DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="i", provider="mt5", feature_name="sma_20")

    def test_feature_key_requires_feature_name_and_version(self) -> None:
        with pytest.raises(DatasetManifestError):
            DatasetKey(dataset_kind=DatasetKind.DERIVED_FEATURES, instrument_id="i")

    def test_feature_key_forbids_provider(self) -> None:
        with pytest.raises(DatasetManifestError):
            DatasetKey(dataset_kind=DatasetKind.DERIVED_FEATURES, instrument_id="i", provider="mt5", feature_name="sma_20", feature_version=1)

    def test_round_trips_through_json(self) -> None:
        key = _raw_key()
        assert DatasetKey.from_json_dict(key.to_json_dict()) == key

    def test_storage_path_parts_are_deterministic(self) -> None:
        assert _raw_key().storage_path_parts() == ("raw_market_events", "mt5", "mt5__XAUUSD")
        assert _feature_key().storage_path_parts() == ("derived_features", "sma_20", "v1", "mt5__XAUUSD")


class TestDatasetManifestIdentity:
    def test_identical_arguments_produce_identical_ids(self) -> None:
        assert _manifest().dataset_id == _manifest().dataset_id

    def test_changed_event_count_changes_id(self) -> None:
        assert _manifest().dataset_id != _manifest(event_count=3).dataset_id

    def test_changed_partition_ids_changes_id(self) -> None:
        assert _manifest().dataset_id != _manifest(ordered_partition_ids=("d" * 64,)).dataset_id

    def test_changed_instrument_changes_id(self) -> None:
        other_key = _raw_key(instrument_id="mt5__EURUSD")
        assert _manifest().dataset_id != _manifest(dataset_key=other_key).dataset_id

    def test_operational_metadata_does_not_change_id(self) -> None:
        # creation_time and physical_digest are explicitly excluded from
        # identity -- two manifests differing ONLY in those fields must
        # produce the SAME dataset_id.
        a = _manifest(creation_time=_T0, physical_digest="c" * 64)
        b = _manifest(creation_time=_T0 + timedelta(days=365), physical_digest="f" * 64)
        assert a.dataset_id == b.dataset_id

    def test_round_trips_through_json(self) -> None:
        manifest = _manifest()
        from quant_platform.market_data.manifests import DatasetManifest

        assert DatasetManifest.from_json_dict(manifest.to_json_dict()) == manifest

    def test_empty_dataset_requires_no_event_times_or_partitions(self) -> None:
        with pytest.raises(DatasetManifestError):
            _manifest(event_count=0)  # still has first/last_event_time and partitions set -> inconsistent

    def test_empty_dataset_is_constructible(self) -> None:
        manifest = _manifest(event_count=0, first_event_time=None, last_event_time=None, ordered_partition_ids=())
        assert manifest.event_count == 0

    def test_nonempty_dataset_requires_partitions(self) -> None:
        with pytest.raises(DatasetManifestError):
            _manifest(ordered_partition_ids=())

    def test_derived_features_requires_raw_source_dataset_id(self) -> None:
        with pytest.raises(DatasetManifestError):
            _manifest(dataset_key=_feature_key(), raw_source_dataset_id=None)

    def test_raw_forbids_raw_source_dataset_id(self) -> None:
        with pytest.raises(DatasetManifestError):
            _manifest(raw_source_dataset_id="e" * 64)


class TestDatasetManifestStore:
    def test_append_and_read_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = DatasetManifestStore(Path(tmp))
            key = _raw_key()
            manifest = _manifest()
            store.append(key, manifest)
            assert store.read_current(key) == manifest

    def test_full_history_is_retained_across_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = DatasetManifestStore(Path(tmp))
            key = _raw_key()
            v1 = _manifest(event_count=2)
            v2 = _manifest(event_count=3, ordered_partition_ids=("d" * 64,))
            store.append(key, v1)
            store.append(key, v2)
            history = store.read_history(key)
            assert history == [v1, v2]
            assert store.read_current(key) == v2

    def test_identical_reappend_is_idempotent_no_new_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = DatasetManifestStore(Path(tmp))
            key = _raw_key()
            manifest = _manifest()
            store.append(key, manifest)
            store.append(key, manifest)
            assert len(store.read_history(key)) == 1

    def test_missing_dataset_reads_as_no_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = DatasetManifestStore(Path(tmp))
            assert store.read_history(_raw_key()) == []
            assert store.read_current(_raw_key()) is None

    def test_wrong_dataset_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = DatasetManifestStore(Path(tmp))
            with pytest.raises(DatasetManifestError):
                store.append(_raw_key(instrument_id="different"), _manifest())


class TestSameLogicalDataDifferentRootsGivesSameDatasetId:
    def test_two_independent_roots_produce_identical_dataset_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            store_a = DatasetManifestStore(Path(tmp_a))
            store_b = DatasetManifestStore(Path(tmp_b))
            key = _raw_key()
            manifest = _manifest()
            store_a.append(key, manifest)
            store_b.append(key, manifest)
            assert store_a.read_current(key).dataset_id == store_b.read_current(key).dataset_id  # type: ignore[union-attr]
