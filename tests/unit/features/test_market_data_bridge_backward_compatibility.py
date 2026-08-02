"""Backward-compatibility acceptance (Milestone 10, Phase 4D, spec
Section 22): proves the additive changes this phase made to
`features.dataset_builder`/`features.manifests`/`historical.loader` are
genuinely additive -- every pre-existing call pattern keeps working
unchanged, and `market_data_lineage` is `None` (never fabricated) for
manifests that never went through the bridge."""

from __future__ import annotations

import json

import pandas as pd

from quant_platform.core.types import Timeframe
from quant_platform.features.labels import LabelDefinition, LabelKind
from quant_platform.features.manifests import ResearchDatasetManifest, capture_environment_metadata
from quant_platform.historical.loader import DatasetLoader, HistoricalDatasetLoaderProtocol


class TestResearchDatasetBuildRequestStaysBackwardCompatible:
    def test_legacy_construction_without_market_data_lineage_still_works(self) -> None:
        from quant_platform.features.dataset_builder import ResearchDatasetBuildRequest

        # The exact pre-Phase-4D call shape: no market_data_lineage kwarg at all.
        request = ResearchDatasetBuildRequest(
            symbol="XAUUSD", base_timeframe=Timeframe.H1, start=pd.Timestamp("2024-01-01", tz="UTC"), end=pd.Timestamp("2024-01-02", tz="UTC"),
            feature_names=("sma_20",), label_definition=LabelDefinition(name="fwd_ret_5", kind=LabelKind.FUTURE_RETURN, horizon_bars=5),
            split_strategy="chronological",
        )
        assert request.market_data_lineage is None


class TestResearchDatasetManifestStaysBackwardCompatible:
    def _legacy_manifest(self) -> ResearchDatasetManifest:
        return ResearchDatasetManifest(
            dataset_id="d" * 16, version="000001-abc", source_historical_dataset_id="h" * 16, source_historical_manifest_version="000001-xyz",
            symbol="XAUUSD", base_timeframe=Timeframe.H1, utc_start=pd.Timestamp("2024-01-01", tz="UTC"), utc_end=pd.Timestamp("2024-01-02", tz="UTC"),
            feature_names=("sma_20",), feature_versions={"sma_20": "1"}, feature_registry_fingerprint="f" * 16, label_definition={},
            split_definition={}, preprocessing_definition={}, fitted_preprocessing_fingerprint=None, code_revision="abc", input_content_hashes={},
            output_content_hashes={}, row_counts={"train": 10}, missing_data_summary={}, leakage_validation_result={"is_valid": True},
            created_at=pd.Timestamp("2024-01-01", tz="UTC"),
        )

    def test_construction_without_market_data_lineage_defaults_to_none(self) -> None:
        manifest = self._legacy_manifest()
        assert manifest.market_data_lineage is None

    def test_json_round_trip_of_a_legacy_manifest(self) -> None:
        manifest = self._legacy_manifest()
        raw = manifest.to_json_dict()
        assert raw["market_data_lineage"] is None
        restored = ResearchDatasetManifest.from_json_dict(raw)
        assert restored.market_data_lineage is None
        assert restored.dataset_id == manifest.dataset_id

    def test_json_missing_the_key_entirely_still_loads(self) -> None:
        """Simulates a manifest JSON file written by a version of this
        code that predates Phase 4D entirely (key absent, not merely
        null) -- must load with market_data_lineage=None, never raise a
        KeyError, and never silently fabricate a lineage payload."""
        manifest = self._legacy_manifest()
        raw = manifest.to_json_dict()
        del raw["market_data_lineage"]
        restored = ResearchDatasetManifest.from_json_dict(raw)
        assert restored.market_data_lineage is None

    def test_canonical_json_bytes_round_trip(self) -> None:
        manifest = self._legacy_manifest()
        raw = manifest.to_json_dict()
        text = json.dumps(raw)
        reparsed = json.loads(text)
        restored = ResearchDatasetManifest.from_json_dict(reparsed)
        assert restored == manifest

    def test_environment_capture_still_works_unchanged(self) -> None:
        env = capture_environment_metadata()
        assert "python_version" in env and "packages" in env


class TestHistoricalDatasetLoaderProtocolBackwardCompatibility:
    def test_real_dataset_loader_class_still_structurally_satisfies_the_protocol(self) -> None:
        """`historical.loader.DatasetLoader` -- the ORIGINAL Milestone 2/3
        concrete class -- was never modified by Phase 4D; confirms it
        still structurally satisfies the new `HistoricalDatasetLoaderProtocol`
        Protocol added alongside it, so `ResearchDatasetBuilder`'s narrowed
        type hint remains fully backward compatible with the original caller."""
        assert hasattr(DatasetLoader, "resolve_manifest")
        assert hasattr(DatasetLoader, "load_for_engine")
        # `HistoricalDatasetLoaderProtocol` is `@runtime_checkable` and
        # declares methods only (no data attributes), so `issubclass`
        # against the class itself (not an instance) is a genuine runtime
        # structural check, not merely a static/mypy-only one.
        assert issubclass(DatasetLoader, HistoricalDatasetLoaderProtocol)
