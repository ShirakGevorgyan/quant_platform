"""Milestone 11 Phase 1, Part 2: truncation invariance, replay
invariance, and determinism proofs. Each test exercises the REAL
`ResearchDatasetBuilder`/`DatasetQualificationEngine` pipeline end to
end -- no shortcuts, no mocked storage."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from quant_platform.core.types import Timeframe
from quant_platform.features.dataset_builder import ResearchDatasetBuilder
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.qualification.engine import DatasetQualificationEngine
from quant_platform.qualification.verifier import verify_no_future_leakage


class TestTruncationInvariance:
    """"Removing all future rows must never change qualification before
    cutoff" -- proven at two levels: (1) the row-level, order-sensitive
    checks (`verify_no_future_leakage`, duplicate/monotonicity checks)
    produce IDENTICAL findings for a fixed row window regardless of
    whether rows after that window exist in the frame passed in; (2) a
    REAL rebuild with a shorter requested date range reproduces the
    exact same `open_time`/feature values for the rows it shares with
    the full build (proving feature computation itself never looks
    ahead of a row's own position)."""

    def test_row_level_leakage_check_is_unaffected_by_appending_future_rows(self, qualified_manifest, research_store) -> None:
        from quant_platform.qualification.verifier import QualificationVerifier

        facts = QualificationVerifier().verify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        train_df = (facts.artifacts.splits or {})["train"]
        cutoff_index = len(train_df) // 2
        prefix = train_df.iloc[:cutoff_index].reset_index(drop=True)

        prefix_only_ok, prefix_only_messages = verify_no_future_leakage(prefix, split_name="train")
        full_ok, full_messages = verify_no_future_leakage(train_df, split_name="train")
        # Both clean-fixture runs should agree there is no leakage -- the invariant under test is that
        # ADDING future rows does not retroactively change what was found true of the earlier rows.
        assert prefix_only_ok == full_ok is True
        assert prefix_only_messages == full_messages == ()

    def test_monotonicity_of_a_prefix_is_unaffected_by_future_rows(self, qualified_manifest, research_store) -> None:
        from quant_platform.qualification.verifier import QualificationVerifier

        facts = QualificationVerifier().verify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        train_df = (facts.artifacts.splits or {})["train"]
        cutoff_index = len(train_df) // 2
        prefix = train_df.iloc[:cutoff_index]
        assert prefix["open_time"].is_monotonic_increasing == train_df["open_time"].is_monotonic_increasing is True

    def test_real_rebuild_with_a_shorter_date_range_reproduces_identical_prefix_rows(
        self, tmp_path, seeded_loader, synthetic_m1_df, trend_registry_factory, build_request_factory,
    ) -> None:
        research_store = ResearchDatasetStore(tmp_path / "research_full")
        builder = ResearchDatasetBuilder(
            historical_loader=seeded_loader, registry=trend_registry_factory(), research_store=research_store,
            manifest_store=ResearchManifestStore(tmp_path / "research_full"),
        )
        full_manifest = builder.build(build_request_factory())
        full_splits = research_store.read_artifacts(full_manifest.dataset_id, full_manifest.content_id)
        assert full_splits is not None
        full_all = pd.concat(full_splits.values(), ignore_index=True).sort_values("open_time").reset_index(drop=True)

        cutoff = synthetic_m1_df["open_time"].iloc[999]
        truncated_research_store = ResearchDatasetStore(tmp_path / "research_truncated")
        truncated_builder = ResearchDatasetBuilder(
            historical_loader=seeded_loader, registry=trend_registry_factory(), research_store=truncated_research_store,
            manifest_store=ResearchManifestStore(tmp_path / "research_truncated"),
        )
        truncated_manifest = truncated_builder.build(build_request_factory(end=cutoff + Timeframe.M1.duration))
        truncated_splits = truncated_research_store.read_artifacts(truncated_manifest.dataset_id, truncated_manifest.content_id)
        assert truncated_splits is not None
        truncated_all = pd.concat(truncated_splits.values(), ignore_index=True).sort_values("open_time").reset_index(drop=True)

        shared_open_times = truncated_all["open_time"]
        full_prefix = full_all[full_all["open_time"].isin(shared_open_times)].sort_values("open_time").reset_index(drop=True)
        truncated_sorted = truncated_all.sort_values("open_time").reset_index(drop=True)
        assert len(full_prefix) == len(truncated_sorted)
        pd.testing.assert_series_equal(full_prefix["trend"], truncated_sorted["trend"], check_names=False)
        pd.testing.assert_series_equal(full_prefix["open_time"], truncated_sorted["open_time"], check_names=False)


class TestReplayInvariance:
    """Replay -> qualify -> destroy artifacts -> replay -> qualify:
    everything identical. Proves the qualification decision depends
    entirely on the deterministic (source, recipe) pair, never on
    whatever happened to be sitting on disk from a previous build."""

    def test_rebuilding_after_destroying_artifacts_reproduces_an_identical_report(
        self, tmp_path, seeded_loader, trend_registry_factory, build_request_factory,
    ) -> None:
        research_root = tmp_path / "research"
        research_store = ResearchDatasetStore(research_root)
        manifest_store = ResearchManifestStore(research_root)
        builder = ResearchDatasetBuilder(historical_loader=seeded_loader, registry=trend_registry_factory(), research_store=research_store, manifest_store=manifest_store)

        manifest_1 = builder.build(build_request_factory())
        report_1 = DatasetQualificationEngine().qualify(manifest_1, research_store, required_feature_names=frozenset({"trend"}))

        content_dir = research_store.content_dir(manifest_1.dataset_id, manifest_1.content_id)
        assert content_dir.is_dir()
        shutil.rmtree(content_dir)
        assert not content_dir.exists()

        manifest_2 = builder.build(build_request_factory())
        report_2 = DatasetQualificationEngine().qualify(manifest_2, research_store, required_feature_names=frozenset({"trend"}))

        assert manifest_1.dataset_id == manifest_2.dataset_id
        assert manifest_1.content_id == manifest_2.content_id
        assert report_1.decision.decision == report_2.decision.decision
        assert report_1.decision.overall_score == report_2.decision.overall_score
        assert [r.score for r in report_1.dimension_results] == [r.score for r in report_2.dimension_results]
        assert report_1.all_blocking_failures == report_2.all_blocking_failures


_DETERMINISM_SCRIPT = textwrap.dedent("""
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from quant_platform.core.types import Timeframe
    from quant_platform.features.dataset_builder import ResearchDatasetBuilder, ResearchDatasetBuildRequest
    from quant_platform.features.interfaces import FeatureDefinition
    from quant_platform.features.labels import LabelDefinition, LabelKind
    from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
    from quant_platform.features.models import FeatureCategory, FeatureSpec
    from quant_platform.features.registry import FeatureRegistry
    from quant_platform.historical import PIPELINE_VERSION
    from quant_platform.historical.canonical_store import CanonicalStore
    from quant_platform.historical.loader import DatasetLoader
    from quant_platform.historical.manifest import ManifestStore
    from quant_platform.historical.models import RAW_HISTORICAL_COLUMNS
    from quant_platform.historical.update_pipeline import apply_incremental_update
    from quant_platform.qualification.engine import DatasetQualificationEngine

    tmp_root = Path(sys.argv[1])

    rng = np.random.default_rng(1)
    n = 300
    open_time = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    returns = rng.normal(0, 0.0005, size=n)
    close = 2000.0 * np.cumprod(1 + returns)
    open_ = np.roll(close, 1)
    open_[0] = 2000.0
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0002, size=n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0002, size=n)))
    tick_volume = rng.integers(10, 1000, size=n)
    real_volume = np.zeros(n, dtype=np.int64)
    spread = rng.integers(1, 30, size=n)
    df = pd.DataFrame({
        "open_time": open_time, "open": open_, "high": high, "low": low, "close": close,
        "tick_volume": tick_volume, "real_volume": real_volume, "spread": spread,
    })[list(RAW_HISTORICAL_COLUMNS)]

    canonical_store = CanonicalStore(tmp_root)
    manifest_store = ManifestStore(tmp_root)
    apply_incremental_update(
        canonical_store, manifest_store, df, symbol="XAUUSD", timeframe=Timeframe.M1, source_name="synthetic",
        broker="test", pipeline_version=PIPELINE_VERSION, parent_snapshot_ids=(),
        requested_start=df["open_time"].iloc[0], requested_end=df["open_time"].iloc[-1] + Timeframe.M1.duration,
    )
    loader = DatasetLoader(canonical_store, manifest_store)

    registry = FeatureRegistry()
    spec = FeatureSpec(
        name="trend", version="1", description="row index as a float", category=FeatureCategory.PRICE,
        required_inputs=(), source_symbols=(), source_timeframe=Timeframe.M1, output_dtype="float64",
        lookback_bars=0, warmup_bars=0,
    )
    registry.register(FeatureDefinition(spec=spec, compute=lambda ctx: pd.Series(np.arange(len(ctx.base_df), dtype="float64"))))

    research_store = ResearchDatasetStore(tmp_root / "research")
    research_manifest_store = ResearchManifestStore(tmp_root / "research")
    builder = ResearchDatasetBuilder(historical_loader=loader, registry=registry, research_store=research_store, manifest_store=research_manifest_store)
    request = ResearchDatasetBuildRequest(
        symbol="XAUUSD", base_timeframe=Timeframe.M1, start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-01", tz="UTC") + Timeframe.M1.duration * n,
        feature_names=("trend",), label_definition=LabelDefinition(name="fut", kind=LabelKind.FUTURE_RETURN, horizon_bars=5),
        split_strategy="chronological", split_params={"train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 5, "embargo_bars": 5},
    )
    manifest = builder.build(request)
    report = DatasetQualificationEngine().qualify(manifest, research_store, required_feature_names=frozenset({"trend"}))
    print(json.dumps(report.to_json_dict(), sort_keys=True))
""")


class TestDeterminism:
    """Run qualification multiple times with different PYTHONHASHSEED
    values and different filesystem roots/temporary directories: results
    identical. Run as separate subprocesses (not just repeated in-
    process calls) so PYTHONHASHSEED -- read once at interpreter start-up
    -- actually varies between runs."""

    @pytest.mark.parametrize("pythonhashseed", ["0", "1", "random"])
    def test_qualification_report_is_identical_across_hashseeds_and_filesystem_roots(self, tmp_path, pythonhashseed: str) -> None:
        script_path = tmp_path / "determinism_script.py"
        script_path.write_text(_DETERMINISM_SCRIPT, encoding="utf-8")

        baseline_root = tmp_path / "baseline_root"
        baseline_root.mkdir()
        baseline = subprocess.run(
            [sys.executable, str(script_path), str(baseline_root)], capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        baseline_report = json.loads(baseline.stdout)

        varied_root = tmp_path / f"varied_root_{pythonhashseed}"
        varied_root.mkdir()
        varied = subprocess.run(
            [sys.executable, str(script_path), str(varied_root)], capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": pythonhashseed},
        )
        varied_report = json.loads(varied.stdout)

        # generated_at legitimately differs (wall-clock) -- strip it before comparing everything else.
        baseline_report.pop("generated_at"), varied_report.pop("generated_at")
        baseline_report["decision"].pop("generated_at"), varied_report["decision"].pop("generated_at")
        assert baseline_report == varied_report
