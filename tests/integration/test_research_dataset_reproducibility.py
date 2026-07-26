"""The Milestone 3 reproducibility proof, extending Milestone 2's
content-addressed exact-reconstruction guarantee one layer up the stack:

  1. Build historical canonical dataset version V1.
  2. Build a research dataset PINNED to V1 -- fingerprint its artifacts.
  3. Revise one historical bar (`RevisionPolicy.ACCEPT_NEWER_SOURCE`),
     producing historical version V2.
  4. Rebuild the SAME research configuration, again pinned to V1 -- prove
     the result is byte-for-byte identical to step 2 (same manifest
     version, same content id, same feature/label values), even though the
     underlying historical dataset has since been revised.
  5. Build the research configuration pinned to V2 -- prove it is a
     genuinely DIFFERENT manifest version (same dataset_id, new version)
     whose feature values reflect the revision.
  6. Prove both research manifest versions remain independently loadable.

This is the concrete, end-to-end version of Milestone 3 Section 17 item 7
("rebuilding the same manifest produces identical outputs"), exercised
against a REAL historical revision rather than a synthetic no-op rebuild.
"""

from __future__ import annotations

import pandas as pd
from tests.unit.features.conftest import make_synthetic_ohlcv, seed_canonical_dataset

from quant_platform.core.types import Timeframe
from quant_platform.features.dataset_builder import ResearchDatasetBuilder, ResearchDatasetBuildRequest
from quant_platform.features.labels import LabelDefinition, LabelKind
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.technical.price import TechnicalWindows, register_core_technical_features
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader
from quant_platform.historical.manifest import ManifestStore
from quant_platform.historical.update_pipeline import RevisionPolicy, apply_incremental_update


def _registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    register_core_technical_features(
        registry, timeframe=Timeframe.M1,
        windows=TechnicalWindows(return_windows=(1, 5), momentum_windows=(10,), atr_window=14, volatility_window=20, zscore_window=20),
    )
    return registry


def _build_request(*, dataset_version: str | None, start: pd.Timestamp, end: pd.Timestamp) -> ResearchDatasetBuildRequest:
    return ResearchDatasetBuildRequest(
        symbol="XAUUSD", base_timeframe=Timeframe.M1, start=start, end=end,
        feature_names=("return_simple_1", "return_simple_5", "momentum_10", "atr_14", "rolling_zscore_close_20"),
        label_definition=LabelDefinition(name="fut_ret_5", kind=LabelKind.FUTURE_RETURN, horizon_bars=5),
        split_strategy="chronological",
        split_params={"train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 5, "embargo_bars": 5},
        dataset_version=dataset_version,
    )


class TestResearchDatasetReproducibilityAcrossHistoricalRevision:
    def test_v1_pinned_research_dataset_is_byte_identical_after_a_later_revision(self, tmp_path) -> None:
        historical_root = tmp_path / "historical"
        research_root = tmp_path / "research"

        df_v1 = make_synthetic_ohlcv(3000, seed=42)
        seed_canonical_dataset(historical_root, df_v1)

        canonical_store = CanonicalStore(historical_root)
        manifest_store = ManifestStore(historical_root)
        loader = DatasetLoader(canonical_store, manifest_store)
        historical_manifest_v1 = manifest_store.load(symbol="XAUUSD", timeframe=Timeframe.M1)
        v1_version = historical_manifest_v1.version

        registry = _registry()
        research_store = ResearchDatasetStore(research_root)
        research_manifest_store = ResearchManifestStore(research_root)
        builder = ResearchDatasetBuilder(
            historical_loader=loader, registry=registry, research_store=research_store,
            manifest_store=research_manifest_store,
        )

        start = df_v1["open_time"].iloc[0]
        end = df_v1["open_time"].iloc[-1] + Timeframe.M1.duration

        research_manifest_r1 = builder.build(_build_request(dataset_version=v1_version, start=start, end=end))
        r1_splits = research_store.read_artifacts(research_manifest_r1.dataset_id, research_manifest_r1.content_id)
        assert r1_splits is not None

        # --- Step 3: revise one historical bar (shift all four OHLC prices
        # together to preserve OHLC validity, mirroring the exact technique
        # used in the Milestone 2 reproducibility audit). ---
        revised_row = df_v1.iloc[[1500]].copy()
        for col in ("open", "high", "low", "close"):
            revised_row[col] = revised_row[col] + 500.0
        update_report = apply_incremental_update(
            canonical_store, manifest_store, revised_row, symbol="XAUUSD", timeframe=Timeframe.M1,
            source_name="synthetic", broker="test", pipeline_version="test", parent_snapshot_ids=(),
            requested_start=revised_row["open_time"].iloc[0], requested_end=revised_row["open_time"].iloc[0] + Timeframe.M1.duration,
            revision_policy=RevisionPolicy.ACCEPT_NEWER_SOURCE,
        )
        assert update_report.rows_conflicting == 1
        historical_manifest_v2 = manifest_store.load(symbol="XAUUSD", timeframe=Timeframe.M1)
        v2_version = historical_manifest_v2.version
        assert v2_version != v1_version

        # --- Step 4: rebuild pinned to V1 -- must reproduce R1 exactly. ---
        research_manifest_r1_rebuilt = builder.build(_build_request(dataset_version=v1_version, start=start, end=end))
        assert research_manifest_r1_rebuilt.dataset_id == research_manifest_r1.dataset_id
        assert research_manifest_r1_rebuilt.version == research_manifest_r1.version
        assert research_manifest_r1_rebuilt.content_id == research_manifest_r1.content_id

        r1_rebuilt_splits = research_store.read_artifacts(
            research_manifest_r1_rebuilt.dataset_id, research_manifest_r1_rebuilt.content_id
        )
        assert r1_rebuilt_splits is not None
        for split_name in r1_splits:
            pd.testing.assert_frame_equal(r1_splits[split_name], r1_rebuilt_splits[split_name])

        # --- Step 5: build pinned to V2 -- a genuinely different version, same dataset_id. ---
        research_manifest_r2 = builder.build(_build_request(dataset_version=v2_version, start=start, end=end))
        assert research_manifest_r2.dataset_id == research_manifest_r1.dataset_id
        assert research_manifest_r2.version != research_manifest_r1.version
        assert research_manifest_r2.content_id != research_manifest_r1.content_id

        r2_splits = research_store.read_artifacts(research_manifest_r2.dataset_id, research_manifest_r2.content_id)
        assert r2_splits is not None
        # at least one split's feature values must differ, since the revision
        # perturbs return/momentum/atr/zscore features computed around row 1500
        any_split_differs = any(
            not r1_splits[name].drop(columns=["open_time"]).equals(r2_splits[name].drop(columns=["open_time"]))
            for name in r1_splits
        )
        assert any_split_differs

        # --- Step 6: both versions remain independently loadable. ---
        reloaded_r1 = research_manifest_store.load(research_manifest_r1.dataset_id, version=research_manifest_r1.version)
        reloaded_r2 = research_manifest_store.load(research_manifest_r1.dataset_id, version=research_manifest_r2.version)
        assert reloaded_r1.content_id == research_manifest_r1.content_id
        assert reloaded_r2.content_id == research_manifest_r2.content_id
        assert research_store.read_artifacts(reloaded_r1.dataset_id, reloaded_r1.content_id) is not None
        assert research_store.read_artifacts(reloaded_r2.dataset_id, reloaded_r2.content_id) is not None

    def test_rebuilding_the_same_manifest_with_no_underlying_change_is_a_pure_no_op(self, tmp_path) -> None:
        """Section 17 item 7 in its simplest form: with nothing revised at
        all, rebuilding twice must produce the exact same manifest version
        (no redundant version minted) and byte-identical artifacts."""
        historical_root = tmp_path / "historical"
        research_root = tmp_path / "research"
        df = make_synthetic_ohlcv(1000, seed=7)
        seed_canonical_dataset(historical_root, df)

        canonical_store = CanonicalStore(historical_root)
        manifest_store = ManifestStore(historical_root)
        loader = DatasetLoader(canonical_store, manifest_store)

        registry = _registry()
        research_store = ResearchDatasetStore(research_root)
        research_manifest_store = ResearchManifestStore(research_root)
        builder = ResearchDatasetBuilder(
            historical_loader=loader, registry=registry, research_store=research_store,
            manifest_store=research_manifest_store,
        )
        start, end = df["open_time"].iloc[0], df["open_time"].iloc[-1] + Timeframe.M1.duration

        manifest_a = builder.build(_build_request(dataset_version=None, start=start, end=end))
        manifest_b = builder.build(_build_request(dataset_version=None, start=start, end=end))

        assert manifest_a.version == manifest_b.version
        assert manifest_a.content_id == manifest_b.content_id
        assert research_manifest_store.list_versions(manifest_a.dataset_id) == [manifest_a.version]
