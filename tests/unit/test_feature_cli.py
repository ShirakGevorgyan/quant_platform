from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv, seed_canonical_dataset

from quant_platform.feature_cli import build_parser, main


def _write_config(tmp_path: Path, *, historical_root: Path, research_root: Path, extra: dict | None = None) -> Path:
    config = {
        "symbol": "XAUUSD",
        "base_timeframe": "M1",
        "start": "2024-01-01T00:00:00Z",
        "end": (pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(minutes=2000)).isoformat(),
        "historical_storage_root": str(historical_root),
        "research_storage_root": str(research_root),
        "technical": {"return_windows": [1, 5], "momentum_windows": [10], "atr_window": 14},
        "temporal": {"enabled": True},
        "label": {"name": "fut", "kind": "future_return", "horizon_bars": 5},
        "split": {"strategy": "chronological", "train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 5, "embargo_bars": 5},
        "preprocessing": {"transforms": {"return_simple_1": "standard_scale"}},
    }
    if extra:
        config.update(extra)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def seeded_config(tmp_path) -> Path:
    historical_root = tmp_path / "data"
    research_root = tmp_path / "research"
    df = make_synthetic_ohlcv(2000, seed=1)
    seed_canonical_dataset(historical_root, df)
    return _write_config(tmp_path, historical_root=historical_root, research_root=research_root)


class TestBuildParser:
    def test_all_seven_commands_registered(self) -> None:
        parser = build_parser()
        subparsers_action = next(a for a in parser._actions if a.dest == "command")
        assert set(subparsers_action.choices) == {
            "list-features", "describe-feature", "build-research-dataset", "validate-research-dataset",
            "inspect-lineage", "inspect-dataset-manifest", "compare-feature-drift",
        }


class TestListFeatures:
    def test_prints_registered_feature_names(self, seeded_config, capsys) -> None:
        rc = main(["list-features", "--config", str(seeded_config)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "return_simple_1" in out
        assert "hour_of_day" in out


class TestDescribeFeature:
    def test_prints_feature_metadata(self, seeded_config, capsys) -> None:
        rc = main(["describe-feature", "--config", str(seeded_config), "--name", "return_simple_1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "category: price" in out

    def test_unknown_feature_name_fails_actionably(self, seeded_config, capsys) -> None:
        rc = main(["describe-feature", "--config", str(seeded_config), "--name", "does_not_exist"])
        assert rc == 1
        assert "ERROR" in capsys.readouterr().err


class TestBuildAndInspect:
    def test_build_then_inspect_manifest_then_validate(self, seeded_config, capsys) -> None:
        rc = main(["build-research-dataset", "--config", str(seeded_config)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Research dataset built" in out

        config = json.loads(seeded_config.read_text())
        # Find the dataset_id via directory listing since the CLI does not
        # (yet) print raw JSON -- read it back from the store directly.
        research_root = Path(config["research_storage_root"])
        dataset_dir = next((research_root / "research_datasets").iterdir())
        dataset_id = dataset_dir.name.removeprefix("dataset_id=")

        rc = main(["inspect-dataset-manifest", "--config", str(seeded_config), "--dataset-id", dataset_id])
        assert rc == 0
        manifest_out = capsys.readouterr().out
        assert "dataset_id" in manifest_out

        rc = main(["validate-research-dataset", "--config", str(seeded_config), "--dataset-id", dataset_id])
        assert rc in (0, 2)
        capsys.readouterr()

        rc = main(["inspect-lineage", "--config", str(seeded_config), "--dataset-id", dataset_id])
        assert rc == 0
        lineage_out = capsys.readouterr().out
        assert "Feature lineage report" in lineage_out

        rc = main(
            ["compare-feature-drift", "--config", str(seeded_config), "--dataset-id", dataset_id,
             "--reference", "train", "--comparison", "test"]
        )
        assert rc == 0
        drift_out = capsys.readouterr().out
        assert "Drift report" in drift_out

    def test_compare_drift_unknown_split_fails_actionably(self, seeded_config, capsys) -> None:
        main(["build-research-dataset", "--config", str(seeded_config)])
        capsys.readouterr()
        config = json.loads(seeded_config.read_text())
        research_root = Path(config["research_storage_root"])
        dataset_dir = next((research_root / "research_datasets").iterdir())
        dataset_id = dataset_dir.name.removeprefix("dataset_id=")

        rc = main(
            ["compare-feature-drift", "--config", str(seeded_config), "--dataset-id", dataset_id,
             "--reference", "train", "--comparison", "does_not_exist"]
        )
        assert rc == 1
        assert "ERROR" in capsys.readouterr().err

    def test_inspect_manifest_unknown_dataset_id_fails_actionably(self, seeded_config, capsys) -> None:
        rc = main(["inspect-dataset-manifest", "--config", str(seeded_config), "--dataset-id", "nonexistent"])
        assert rc == 1
        assert "ERROR" in capsys.readouterr().err
