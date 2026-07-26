from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv, seed_canonical_dataset

from quant_platform.feature_cli import main as feature_main
from quant_platform.ml_cli import build_parser, main


def _build_research_dataset(tmp_path: Path) -> tuple[str, str, Path, Path]:
    historical_root = tmp_path / "data"
    research_root = tmp_path / "research"
    df = make_synthetic_ohlcv(2000, seed=1)
    seed_canonical_dataset(historical_root, df)

    feature_config = {
        "symbol": "XAUUSD", "base_timeframe": "M1",
        "start": "2024-01-01T00:00:00Z",
        "end": (pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(minutes=2000)).isoformat(),
        "historical_storage_root": str(historical_root), "research_storage_root": str(research_root),
        "technical": {"return_windows": [1, 5], "momentum_windows": [10], "atr_window": 14},
        "temporal": {"enabled": True},
        "label": {"name": "fut", "kind": "future_return", "horizon_bars": 5},
        "split": {"strategy": "chronological", "train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 5, "embargo_bars": 5},
        "preprocessing": {"transforms": {"return_simple_1": "standard_scale"}},
    }
    feature_config_path = tmp_path / "feature_config.json"
    feature_config_path.write_text(json.dumps(feature_config))

    rc = feature_main(["build-research-dataset", "--config", str(feature_config_path)])
    assert rc == 0
    from quant_platform.features.manifests import ResearchManifestStore

    # Recover dataset_id/version by scanning the store rather than parsing stdout.
    store = ResearchManifestStore(research_root)
    dataset_dirs = list((research_root / "research_datasets").glob("dataset_id=*"))
    assert len(dataset_dirs) == 1
    dataset_id = dataset_dirs[0].name.removeprefix("dataset_id=")
    manifest = store.load(dataset_id)
    return dataset_id, manifest.version, research_root, historical_root


def _write_ml_config(tmp_path: Path, *, dataset_id: str, version: str, research_root: Path, ml_root: Path, **overrides: object) -> Path:
    config: dict[str, object] = {
        "ml_artifacts_root": str(ml_root),
        "dataset": {"dataset_id": dataset_id, "manifest_version": version, "research_storage_root": str(research_root), "feature_names": None},
        "label": {"name": "fut", "kind": "future_return", "horizon_bars": 5, "label_type": "continuous", "params": {}},
        "split": {"strategy": "chronological", "params": {}},
        "model": {"name": "constant_test_model", "version": "1", "objective": "regression", "hyperparameters": {}},
        "seeds": {"master_seed": 42},
        "primary_metric": "rmse", "environment_requirements": {}, "tags": [], "notes": "cli test",
    }
    config.update(overrides)
    path = tmp_path / "ml_config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def ml_config(tmp_path: Path) -> Path:
    dataset_id, version, research_root, _ = _build_research_dataset(tmp_path)
    return _write_ml_config(tmp_path, dataset_id=dataset_id, version=version, research_root=research_root, ml_root=tmp_path / "ml_artifacts")


class TestBuildParser:
    def test_all_eight_commands_registered(self) -> None:
        parser = build_parser()
        subparsers_action = next(a for a in parser._actions if a.dest == "command")
        assert set(subparsers_action.choices) == {
            "list-model-definitions", "describe-model-definition", "prepare-experiment", "validate-experiment",
            "inspect-experiment", "inspect-experiment-manifest", "verify-artifact", "list-experiment-events",
        }


class TestListAndDescribeModelDefinitions:
    def test_list_model_definitions(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["list-model-definitions"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "constant_test_model@1" in out
        assert "TEST-ONLY" in out

    def test_describe_model_definition(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["describe-model-definition", "--name", "constant_test_model", "--version", "1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "fingerprint:" in out

    def test_describe_unknown_model_fails_actionably(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["describe-model-definition", "--name", "nonexistent", "--version", "1"])
        assert rc == 1
        assert "ERROR" in capsys.readouterr().err


class TestValidateAndPrepareExperiment:
    def test_validate_experiment_dry_run_does_not_create_manifest(self, ml_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["validate-experiment", "--config", str(ml_config)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "is_ready: True" in out

        config_data = json.loads(ml_config.read_text())
        ml_root = Path(config_data["ml_artifacts_root"])
        experiments_dir = ml_root / "experiments"
        assert not experiments_dir.is_dir() or not list(experiments_dir.iterdir())

    def test_prepare_experiment_succeeds(self, ml_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["prepare-experiment", "--config", str(ml_config)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "status: ready" in out
        assert "experiment_id:" in out

    def test_prepare_experiment_is_idempotent(self, ml_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc1 = main(["prepare-experiment", "--config", str(ml_config)])
        out1 = capsys.readouterr().out
        rc2 = main(["prepare-experiment", "--config", str(ml_config)])
        out2 = capsys.readouterr().out
        assert rc1 == rc2 == 0
        assert out1 == out2


class TestInspectionCommands:
    def _prepare(self, ml_config: Path, capsys: pytest.CaptureFixture[str]) -> str:
        main(["prepare-experiment", "--config", str(ml_config)])
        out = capsys.readouterr().out
        return out.split("experiment_id: ")[1].split("\n")[0].strip()

    def test_inspect_experiment_markdown(self, ml_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config, capsys)
        rc = main(["inspect-experiment", "--config", str(ml_config), "--experiment-id", experiment_id])
        assert rc == 0
        out = capsys.readouterr().out
        assert "# Experiment Preparation Report" in out

    def test_inspect_experiment_json(self, ml_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config, capsys)
        rc = main(["inspect-experiment", "--config", str(ml_config), "--experiment-id", experiment_id, "--format", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["experiment_id"] == experiment_id

    def test_inspect_experiment_manifest(self, ml_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config, capsys)
        rc = main(["inspect-experiment-manifest", "--config", str(ml_config), "--experiment-id", experiment_id])
        assert rc == 0
        out = capsys.readouterr().out
        assert "schema_version" in out

    def test_list_experiment_events(self, ml_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config, capsys)
        rc = main(["list-experiment-events", "--config", str(ml_config), "--experiment-id", experiment_id])
        assert rc == 0
        out = capsys.readouterr().out
        assert "experiment_created" in out
        assert "validation_passed" in out

    def test_list_experiment_events_unknown_id_fails(self, ml_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["list-experiment-events", "--config", str(ml_config), "--experiment-id", "a" * 64])
        assert rc == 1
        assert "No events recorded" in capsys.readouterr().err

    def test_verify_artifact(self, ml_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config, capsys)
        from quant_platform.ml.manifests import ExperimentManifestStore

        config_data = json.loads(ml_config.read_text())
        manifest = ExperimentManifestStore(config_data["ml_artifacts_root"]).load(experiment_id)
        assert manifest.validation_report_reference is not None
        rc = main(["verify-artifact", "--config", str(ml_config), "--content-hash", manifest.validation_report_reference.content_hash])
        assert rc == 0
        out = capsys.readouterr().out
        assert "verified: OK" in out

    def test_verify_artifact_unknown_hash_fails_actionably(self, ml_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["verify-artifact", "--config", str(ml_config), "--content-hash", "0" * 64])
        assert rc == 1
        assert "ERROR" in capsys.readouterr().err
