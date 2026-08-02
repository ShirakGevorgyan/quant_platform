"""Milestone 11, Phase 2, Part 1: repeat determinism tests. Runs the
full `FeatureDiscoveryEngine.discover` pipeline as a fresh subprocess
with different `PYTHONHASHSEED` values, each pointed at its own,
never-shared filesystem root/temp directory, and asserts the resulting
`FeatureDiscoveryReport` (with the legitimately wall-clock-dependent
`evaluation_time` field stripped) is byte-identical across every run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

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
    from quant_platform.feature_discovery.engine import FeatureDiscoveryEngine

    tmp_root = Path(sys.argv[1])

    rng = np.random.default_rng(1)
    n = 400
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
    trend_spec = FeatureSpec(
        name="trend", version="1", description="row index as a float", category=FeatureCategory.PRICE,
        required_inputs=(), source_symbols=(), source_timeframe=Timeframe.M1, output_dtype="float64",
        lookback_bars=0, warmup_bars=0,
    )
    registry.register(FeatureDefinition(spec=trend_spec, compute=lambda ctx: pd.Series(np.arange(len(ctx.base_df), dtype="float64"))))
    const_spec = FeatureSpec(
        name="const", version="1", description="constant feature", category=FeatureCategory.PRICE,
        required_inputs=(), source_symbols=(), source_timeframe=Timeframe.M1, output_dtype="float64",
        lookback_bars=0, warmup_bars=0,
    )
    registry.register(FeatureDefinition(spec=const_spec, compute=lambda ctx: pd.Series(np.full(len(ctx.base_df), 7.0))))

    research_store = ResearchDatasetStore(tmp_root / "research")
    research_manifest_store = ResearchManifestStore(tmp_root / "research")
    builder = ResearchDatasetBuilder(historical_loader=loader, registry=registry, research_store=research_store, manifest_store=research_manifest_store)
    request = ResearchDatasetBuildRequest(
        symbol="XAUUSD", base_timeframe=Timeframe.M1, start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-01", tz="UTC") + Timeframe.M1.duration * n,
        feature_names=("trend", "const"), label_definition=LabelDefinition(name="fut", kind=LabelKind.FUTURE_RETURN, horizon_bars=5),
        split_strategy="chronological", split_params={"train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 5, "embargo_bars": 5},
    )
    manifest = builder.build(request)
    report = FeatureDiscoveryEngine().discover(manifest, research_store)
    print(json.dumps(report.to_json_dict(), sort_keys=True))
""")


class TestRepeatDeterminism:
    @pytest.mark.parametrize("pythonhashseed", ["0", "1", "random"])
    def test_report_is_identical_across_hashseeds_and_filesystem_roots(self, tmp_path, pythonhashseed: str) -> None:
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

        baseline_report.pop("evaluation_time")
        varied_report.pop("evaluation_time")
        assert baseline_report == varied_report


class TestRepeatInProcess:
    """Repeat discovery/verification/reconciliation 10 times in-process,
    the quality-gate-style repetition proof (mirrors `qualification`'s
    own `test_qualification_repetition_gates.py`)."""

    def test_ten_repeated_discover_calls_are_identical(self, discovered_manifest, research_store) -> None:
        from quant_platform.feature_discovery.engine import FeatureDiscoveryEngine

        engine = FeatureDiscoveryEngine()
        reports = []
        for _ in range(10):
            raw = engine.discover(discovered_manifest, research_store).to_json_dict()
            raw.pop("evaluation_time")
            reports.append(raw)
        assert all(r == reports[0] for r in reports)

    def test_ten_repeated_verify_calls_are_identical(self, discovered_manifest, research_store) -> None:
        from quant_platform.feature_discovery.engine import FeatureDiscoveryEngine
        from quant_platform.feature_discovery.verification import FeatureDiscoveryVerifier

        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        verifier = FeatureDiscoveryVerifier()
        results = []
        for _ in range(10):
            raw = verifier.verify(report, discovered_manifest, research_store).to_json_dict()
            raw.pop("generated_at")
            raw["reconciliation"] = {k: v for k, v in raw["reconciliation"].items() if k != "generated_at"}
            results.append(raw)
        assert all(r == results[0] for r in results)
        assert results[0]["verified"] is True
