from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_platform.core.exceptions import DataError, PathSecurityError, ResearchDatasetError
from quant_platform.core.types import Timeframe
from quant_platform.features.manifests import (
    ResearchDatasetManifest,
    ResearchDatasetStore,
    ResearchManifestStore,
    capture_environment_metadata,
    compute_dataset_id,
)


def _splits() -> dict[str, pd.DataFrame]:
    return {
        "train": pd.DataFrame({"open_time": pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC"), "x": range(10)}),
        "test": pd.DataFrame({"open_time": pd.date_range("2024-01-01T01:00:00", periods=5, freq="1min", tz="UTC"), "x": range(5)}),
    }


class TestResearchDatasetStore:
    def test_write_and_read_round_trip(self, tmp_path) -> None:
        store = ResearchDatasetStore(tmp_path)
        content_id, checksums = store.write_artifacts("ds1", splits=_splits(), preprocessing_json={"global": {}})
        loaded = store.read_artifacts("ds1", content_id)
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded["train"], _splits()["train"])
        assert set(checksums) == {"train", "test"}

    def test_identical_rewrite_dedups_no_new_content(self, tmp_path) -> None:
        store = ResearchDatasetStore(tmp_path)
        content_id_1, _ = store.write_artifacts("ds1", splits=_splits(), preprocessing_json={"global": {}})
        content_id_2, _ = store.write_artifacts("ds1", splits=_splits(), preprocessing_json={"global": {}})
        assert content_id_1 == content_id_2

    def test_different_content_gets_different_content_id_and_old_remains_readable(self, tmp_path) -> None:
        store = ResearchDatasetStore(tmp_path)
        content_id_1, _ = store.write_artifacts("ds1", splits=_splits(), preprocessing_json={"global": {}})
        modified_splits = _splits()
        modified_splits["train"]["x"] = modified_splits["train"]["x"] + 1000
        content_id_2, _ = store.write_artifacts("ds1", splits=modified_splits, preprocessing_json={"global": {}})

        assert content_id_1 != content_id_2
        original = store.read_artifacts("ds1", content_id_1)
        assert original is not None
        assert original["train"]["x"].tolist() == list(range(10))

    def test_current_pointer_reflects_most_recent_write(self, tmp_path) -> None:
        store = ResearchDatasetStore(tmp_path)
        content_id_1, _ = store.write_artifacts("ds1", splits=_splits(), preprocessing_json={"global": {}})
        modified = _splits()
        modified["train"]["x"] = modified["train"]["x"] + 1
        content_id_2, _ = store.write_artifacts("ds1", splits=modified, preprocessing_json={"global": {}})
        assert store.current_content_id("ds1") == content_id_2
        assert content_id_1 != content_id_2

    def test_read_unknown_content_id_returns_none(self, tmp_path) -> None:
        store = ResearchDatasetStore(tmp_path)
        assert store.read_artifacts("ds1", "0" * 64) is None

    def test_corrupted_data_file_detected_on_read(self, tmp_path) -> None:
        store = ResearchDatasetStore(tmp_path)
        content_id, _ = store.write_artifacts("ds1", splits=_splits(), preprocessing_json={"global": {}})
        data_path = store.content_dir("ds1", content_id) / "train.parquet"
        data_path.write_bytes(b"not a parquet file")
        with pytest.raises(ResearchDatasetError):
            store.read_artifacts("ds1", content_id)

    def test_path_traversal_in_dataset_id_rejected(self, tmp_path) -> None:
        store = ResearchDatasetStore(tmp_path)
        with pytest.raises((PathSecurityError, DataError)):
            store.write_artifacts("../../etc", splits=_splits(), preprocessing_json={})

    def test_interrupted_write_leaves_no_partial_content_visible(self, tmp_path, monkeypatch) -> None:
        """Adversarial self-audit (Section 20 'interrupted writes'):
        simulate a crash partway through `write_artifacts` (a Parquet write
        raising mid-way) and confirm no CURRENT pointer is set and no
        partial content directory is marked `_SUCCESS` -- mirroring
        `historical.canonical_store`'s write-once, never-partially-visible
        guarantee."""
        store = ResearchDatasetStore(tmp_path)

        def exploding_to_parquet(self, *args, **kwargs):
            raise OSError("simulated crash mid-write")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", exploding_to_parquet)
        with pytest.raises(OSError, match="simulated crash"):
            store.write_artifacts("crash_ds", splits=_splits(), preprocessing_json={"global": {}})
        monkeypatch.undo()

        assert store.current_content_id("crash_ds") is None
        content_dir = tmp_path / "research_datasets" / "dataset_id=crash_ds" / "content"
        if content_dir.is_dir():
            for child in content_dir.iterdir():
                assert not (child / "_SUCCESS").is_file(), f"partial content {child} incorrectly marked complete"

        # A subsequent, non-crashing write must still succeed normally.
        content_id, _ = store.write_artifacts("crash_ds", splits=_splits(), preprocessing_json={"global": {}})
        assert store.current_content_id("crash_ds") == content_id


class TestCorruptedResearchArtifactsFailClosed:
    """Milestone 4D.1 completion: `features.manifests` now reads through
    `quant_platform.core.json.parse_json_strict` (previously plain
    `json.loads`) for both `metadata.json` and `preprocessing.json`."""

    def _written(self, tmp_path) -> tuple[ResearchDatasetStore, str]:
        store = ResearchDatasetStore(tmp_path)
        content_id, _ = store.write_artifacts("ds1", splits=_splits(), preprocessing_json={"global": {"a": 1}})
        return store, content_id

    def test_malformed_metadata_json_rejected(self, tmp_path) -> None:
        store, content_id = self._written(tmp_path)
        (store.content_dir("ds1", content_id) / "metadata.json").write_text("{not valid json")
        with pytest.raises(ResearchDatasetError, match="corrupted"):
            store.read_artifacts("ds1", content_id)

    def test_metadata_json_with_nan_metric_field_rejected(self, tmp_path) -> None:
        store, content_id = self._written(tmp_path)
        path = store.content_dir("ds1", content_id) / "metadata.json"
        raw = path.read_text()
        path.write_text(raw[:-1] + ',"bogus_metric":NaN}' if raw.endswith("}") else raw)
        with pytest.raises(ResearchDatasetError, match="corrupted"):
            store.read_artifacts("ds1", content_id)

    def test_metadata_json_non_object_root_rejected(self, tmp_path) -> None:
        store, content_id = self._written(tmp_path)
        (store.content_dir("ds1", content_id) / "metadata.json").write_text("[1, 2, 3]")
        with pytest.raises(ResearchDatasetError, match="corrupted"):
            store.read_artifacts("ds1", content_id)

    def test_metadata_json_duplicate_key_rejected(self, tmp_path) -> None:
        store, content_id = self._written(tmp_path)
        (store.content_dir("ds1", content_id) / "metadata.json").write_text('{"split_names": [], "split_names": []}')
        with pytest.raises(ResearchDatasetError, match="corrupted"):
            store.read_artifacts("ds1", content_id)

    def test_preprocessing_json_malformed_rejected(self, tmp_path) -> None:
        store, content_id = self._written(tmp_path)
        (store.content_dir("ds1", content_id) / "preprocessing.json").write_text("{not valid json")
        with pytest.raises(ResearchDatasetError, match="corrupted"):
            store.read_preprocessing("ds1", content_id)

    def test_preprocessing_json_infinity_rejected(self, tmp_path) -> None:
        store, content_id = self._written(tmp_path)
        (store.content_dir("ds1", content_id) / "preprocessing.json").write_text('{"scale": Infinity}')
        with pytest.raises(ResearchDatasetError, match="corrupted"):
            store.read_preprocessing("ds1", content_id)

    def test_preprocessing_json_non_object_root_rejected(self, tmp_path) -> None:
        store, content_id = self._written(tmp_path)
        (store.content_dir("ds1", content_id) / "preprocessing.json").write_text("[1, 2, 3]")
        with pytest.raises(ResearchDatasetError, match="must decode to a JSON object"):
            store.read_preprocessing("ds1", content_id)

    def test_preprocessing_json_invalid_utf8_rejected(self, tmp_path) -> None:
        store, content_id = self._written(tmp_path)
        (store.content_dir("ds1", content_id) / "preprocessing.json").write_bytes(b"\xff\xfe\x00bad \x80\x81")
        with pytest.raises(ResearchDatasetError, match="corrupted"):
            store.read_preprocessing("ds1", content_id)

    def test_valid_artifacts_remain_readable_after_migration(self, tmp_path) -> None:
        store, content_id = self._written(tmp_path)
        loaded = store.read_artifacts("ds1", content_id)
        assert loaded is not None
        preprocessing = store.read_preprocessing("ds1", content_id)
        assert preprocessing == {"global": {"a": 1}}


class TestResearchManifestStore:
    def _manifest(self, **overrides) -> ResearchDatasetManifest:
        base = {
            "dataset_id": "ds1", "version": "", "source_historical_dataset_id": "hist1",
            "source_historical_manifest_version": "000001-abc", "symbol": "XAUUSD", "base_timeframe": Timeframe.M1,
            "utc_start": pd.Timestamp("2024-01-01", tz="UTC"), "utc_end": pd.Timestamp("2024-02-01", tz="UTC"),
            "feature_names": ("a", "b"), "feature_versions": {"a": "1", "b": "1"}, "feature_registry_fingerprint": "fp1",
            "label_definition": {"kind": "future_return"}, "split_definition": {"strategy": "chronological"},
            "preprocessing_definition": {}, "fitted_preprocessing_fingerprint": None, "code_revision": "content:abc",
            "input_content_hashes": {"h": "1"}, "output_content_hashes": {"train": "x"}, "row_counts": {"train": 10},
            "missing_data_summary": {}, "leakage_validation_result": {"is_valid": True},
            "created_at": pd.Timestamp.now(tz="UTC"), "content_id": "contentid1",
        }
        base.update(overrides)
        return ResearchDatasetManifest(**base)

    def test_save_and_load_round_trip(self, tmp_path) -> None:
        store = ResearchManifestStore(tmp_path)
        version = store.save(self._manifest())
        loaded = store.load("ds1", version)
        assert loaded.dataset_id == "ds1"
        assert loaded.content_id == "contentid1"

    def test_identical_content_save_is_a_no_op(self, tmp_path) -> None:
        store = ResearchManifestStore(tmp_path)
        v1 = store.save(self._manifest())
        v2 = store.save(self._manifest())
        assert v1 == v2
        assert store.list_versions("ds1") == [v1]

    def test_different_content_id_produces_new_version(self, tmp_path) -> None:
        store = ResearchManifestStore(tmp_path)
        v1 = store.save(self._manifest(content_id="contentid1"))
        v2 = store.save(self._manifest(content_id="contentid2"))
        assert v1 != v2
        assert len(store.list_versions("ds1")) == 2

    def test_load_latest_by_default(self, tmp_path) -> None:
        store = ResearchManifestStore(tmp_path)
        store.save(self._manifest(content_id="contentid1"))
        v2 = store.save(self._manifest(content_id="contentid2"))
        loaded = store.load("ds1")
        assert loaded.version == v2

    def test_manifest_versions_are_immutable(self, tmp_path) -> None:
        store = ResearchManifestStore(tmp_path)
        store.save(self._manifest(content_id="contentid1"))
        # Attempting to write the identical version file path directly should
        # be refused by _write_version_file's own guard (exercised via
        # calling save() with content that maps to the same version string
        # is not directly possible without content collision, so we instead
        # confirm the file exists and is never touched by a second call).
        version = store.list_versions("ds1")[0]
        path = tmp_path / "research_datasets" / "dataset_id=ds1" / "manifests" / f"{version}.json"
        original_text = path.read_text()
        store.save(self._manifest(content_id="contentid1"))  # identical content -> no-op
        assert path.read_text() == original_text

    def test_json_round_trip_preserves_all_fields(self, tmp_path) -> None:
        manifest = self._manifest()
        restored = ResearchDatasetManifest.from_json_dict(manifest.to_json_dict())
        assert restored.dataset_id == manifest.dataset_id
        assert restored.feature_names == manifest.feature_names
        assert restored.row_counts == manifest.row_counts

    def _saved_path(self, tmp_path) -> tuple[ResearchManifestStore, Path]:
        store = ResearchManifestStore(tmp_path)
        version = store.save(self._manifest())
        path = tmp_path / "research_datasets" / "dataset_id=ds1" / "manifests" / f"{version}.json"
        return store, path

    def test_malformed_manifest_json_rejected(self, tmp_path) -> None:
        store, path = self._saved_path(tmp_path)
        path.write_text("{not valid json")
        with pytest.raises(ResearchDatasetError, match="corrupted"):
            store.load("ds1")

    def test_manifest_json_with_nan_field_rejected(self, tmp_path) -> None:
        store, path = self._saved_path(tmp_path)
        raw = path.read_text()
        path.write_text(raw.replace('"row_counts":{"train":10}', '"row_counts":{"train":NaN}'))
        with pytest.raises(ResearchDatasetError, match="corrupted"):
            store.load("ds1")

    def test_manifest_json_non_object_root_rejected(self, tmp_path) -> None:
        store, path = self._saved_path(tmp_path)
        path.write_text("[1, 2, 3]")
        with pytest.raises(ResearchDatasetError, match="corrupted"):
            store.load("ds1")

    def test_manifest_json_invalid_utf8_rejected(self, tmp_path) -> None:
        store, path = self._saved_path(tmp_path)
        path.write_bytes(b"\xff\xfe\x00invalid utf8 \x80\x81")
        with pytest.raises(ResearchDatasetError, match="corrupted"):
            store.load("ds1")

    def test_manifest_json_duplicate_key_rejected(self, tmp_path) -> None:
        store, path = self._saved_path(tmp_path)
        path.write_text('{"dataset_id": "a", "dataset_id": "b"}')
        with pytest.raises(ResearchDatasetError, match="corrupted"):
            store.load("ds1")

    def test_valid_manifest_remains_readable_after_migration(self, tmp_path) -> None:
        store, _ = self._saved_path(tmp_path)
        loaded = store.load("ds1")
        assert loaded.dataset_id == "ds1"


class TestComputeDatasetId:
    def _kwargs(self, **overrides) -> dict:
        base = {
            "symbol": "XAUUSD", "base_timeframe": Timeframe.M1, "feature_registry_fingerprint": "fp1",
            "label_definition": {"kind": "future_return", "horizon_bars": 5}, "split_definition": {"strategy": "chronological"},
            "preprocessing_definition": {"close": "standard_scale"},
        }
        base.update(overrides)
        return base

    def test_deterministic(self) -> None:
        assert compute_dataset_id(**self._kwargs()) == compute_dataset_id(**self._kwargs())

    def test_changes_with_feature_fingerprint(self) -> None:
        base_id = compute_dataset_id(**self._kwargs())
        assert compute_dataset_id(**self._kwargs(feature_registry_fingerprint="fp2")) != base_id

    def test_changes_with_label_definition(self) -> None:
        base_id = compute_dataset_id(**self._kwargs())
        assert compute_dataset_id(**self._kwargs(label_definition={"kind": "future_return", "horizon_bars": 10})) != base_id

    def test_changes_with_split_definition(self) -> None:
        base_id = compute_dataset_id(**self._kwargs())
        assert compute_dataset_id(**self._kwargs(split_definition={"strategy": "rolling_walk_forward"})) != base_id

    def test_changes_with_preprocessing_definition(self) -> None:
        base_id = compute_dataset_id(**self._kwargs())
        assert compute_dataset_id(**self._kwargs(preprocessing_definition={})) != base_id

    def test_unchanged_by_symbol_case_sensitivity_is_still_distinct(self) -> None:
        assert compute_dataset_id(**self._kwargs(symbol="xauusd")) != compute_dataset_id(**self._kwargs(symbol="XAUUSD"))


class TestEnvironmentMetadata:
    def test_captures_python_and_package_versions(self) -> None:
        env = capture_environment_metadata()
        assert "python_version" in env
        assert "pandas" in env["packages"]
