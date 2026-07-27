"""Milestone 4D: CLI tests for the 9 optimization commands added to the
shared `quant_platform.ml_cli` parser. Mirrors `tests/unit/test_ml_cli.py`'s
exact conventions -- `main([...])` invoked in-process (never a subprocess)
with `capsys` capturing stdout/stderr, a real (constant-model) research
dataset/experiment built once per test via the standard config-JSON path,
never mocked."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv, seed_canonical_dataset

from quant_platform.feature_cli import main as feature_main
from quant_platform.features.manifests import ResearchManifestStore
from quant_platform.ml_cli import build_parser, main


def _build_research_dataset(tmp_path: Path) -> tuple[str, str, Path, Path]:
    historical_root = tmp_path / "data"
    research_root = tmp_path / "research"
    df = make_synthetic_ohlcv(2000, seed=1)
    seed_canonical_dataset(historical_root, df)

    feature_config: dict[str, object] = {
        "symbol": "XAUUSD", "base_timeframe": "M1", "start": "2024-01-01T00:00:00Z",
        "end": (pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(minutes=2000)).isoformat(),
        "historical_storage_root": str(historical_root), "research_storage_root": str(research_root),
        "technical": {"return_windows": [1, 5], "momentum_windows": [10], "atr_window": 14},
        "temporal": {"enabled": True},
        "label": {"name": "fut", "kind": "future_return", "horizon_bars": 5},
        "split": {"strategy": "chronological", "train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 5, "embargo_bars": 5},
    }
    feature_config_path = tmp_path / "feature_config.json"
    feature_config_path.write_text(json.dumps(feature_config))
    rc = feature_main(["build-research-dataset", "--config", str(feature_config_path)])
    assert rc == 0

    store = ResearchManifestStore(research_root)
    dataset_dirs = list((research_root / "research_datasets").glob("dataset_id=*"))
    assert len(dataset_dirs) == 1
    dataset_id = dataset_dirs[0].name.removeprefix("dataset_id=")
    manifest = store.load(dataset_id)
    return dataset_id, manifest.version, research_root, historical_root


def _write_ml_config(tmp_path: Path, *, dataset_id: str, version: str, research_root: Path, ml_root: Path) -> Path:
    config: dict[str, object] = {
        "ml_artifacts_root": str(ml_root),
        "dataset": {"dataset_id": dataset_id, "manifest_version": version, "research_storage_root": str(research_root), "feature_names": None},
        "label": {"name": "fut", "kind": "future_return", "horizon_bars": 5, "label_type": "continuous", "params": {}},
        "split": {"strategy": "expanding_walk_forward", "params": {"n_splits": 2, "test_size": 150, "purge_bars": 5, "embargo_bars": 2}},
        "model": {"name": "constant_test_model", "version": "1", "objective": "regression", "hyperparameters": {}},
        "seeds": {"master_seed": 42},
        "primary_metric": "rmse", "environment_requirements": {}, "tags": [], "notes": "optimization cli test",
    }
    path = tmp_path / "ml_config.json"
    path.write_text(json.dumps(config))
    return path


def _write_optimization_config(tmp_path: Path, *, experiment_config_path: Path, ml_root: Path, max_trials: int = 2) -> Path:
    config: dict[str, object] = {
        "ml_artifacts_root": str(ml_root), "experiment_config_path": str(experiment_config_path),
        "model_name": "constant_test_model", "model_version": "1", "primary_metric": "rmse",
        "inner_split": {"strategy": "expanding_walk_forward", "n_splits": 2, "test_size_fraction": 0.2},
        "feature_selection": {"strategy": "none"},
        "search_space": {"use_default_for_model": False, "parameters": [{"kind": "fixed", "name": "alpha", "value": 0.1}]},
        "sampler": "tpe", "pruning": {"kind": "none"}, "early_stopping": {"enabled": False},
        "max_trials": max_trials, "min_successful_inner_folds": 1, "seeds": {"master_seed": 7},
        "tags": [], "notes": "optimization cli test",
    }
    path = tmp_path / "optimization_config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def optimization_config(tmp_path: Path) -> Path:
    """A prepared parent experiment plus a matching `OptimizationConfig`
    JSON pointing at it -- `optimize` requires the parent experiment to
    already be READY (see `cmd_optimize`'s own docstring)."""
    dataset_id, version, research_root, _ = _build_research_dataset(tmp_path)
    ml_config_path = _write_ml_config(tmp_path, dataset_id=dataset_id, version=version, research_root=research_root, ml_root=tmp_path / "ml_artifacts")
    rc = main(["prepare-experiment", "--config", str(ml_config_path)])
    assert rc == 0
    return _write_optimization_config(tmp_path, experiment_config_path=ml_config_path, ml_root=tmp_path / "ml_artifacts")


def _run_optimize(config_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    rc = main(["optimize", "--config", str(config_path)])
    out = capsys.readouterr().out
    return rc, out


def _extract_optimization_id(out: str) -> str:
    return out.split("optimization_id: ")[1].split("\n")[0].strip()


class TestBuildParserIncludesOptimizationCommands:
    def test_all_nine_optimization_commands_registered(self) -> None:
        parser = build_parser()
        subparsers_action = next(a for a in parser._actions if a.dest == "command")
        assert {
            "optimize", "resume-optimization", "inspect-optimization", "list-trials", "inspect-trial",
            "verify-optimization", "compare-optimization-candidates", "feature-stability", "hyperparameter-stability",
        }.issubset(set(subparsers_action.choices))


class TestOptimizeCommand:
    def test_optimize_completes_and_reports_stage(self, optimization_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc, out = _run_optimize(optimization_config, capsys)
        assert rc == 0
        assert "stage: completed" in out
        assert "optimization_id:" in out
        assert "winning_trial_by_outer_fold:" in out

    def test_optimize_is_idempotent(self, optimization_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc1, out1 = _run_optimize(optimization_config, capsys)
        rc2, out2 = _run_optimize(optimization_config, capsys)
        assert rc1 == rc2 == 0
        assert _extract_optimization_id(out1) == _extract_optimization_id(out2)


class TestResumeOptimizationCommand:
    def test_resuming_a_completed_optimization_fails_actionably(self, optimization_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_optimize(optimization_config, capsys)
        optimization_id = _extract_optimization_id(out)
        rc = main(["resume-optimization", "--config", str(optimization_config), "--optimization-id", optimization_id])
        assert rc == 1
        assert "ERROR" in capsys.readouterr().err

    def test_resuming_an_unknown_optimization_id_fails_actionably(self, optimization_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["resume-optimization", "--config", str(optimization_config), "--optimization-id", "a" * 64])
        assert rc == 1
        assert "ERROR" in capsys.readouterr().err


class TestInspectOptimizationCommand:
    def test_inspect_markdown(self, optimization_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_optimize(optimization_config, capsys)
        optimization_id = _extract_optimization_id(out)
        rc = main(["inspect-optimization", "--config", str(optimization_config), "--optimization-id", optimization_id])
        assert rc == 0
        markdown = capsys.readouterr().out
        assert "#" in markdown  # a real markdown report was rendered

    def test_inspect_json(self, optimization_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_optimize(optimization_config, capsys)
        optimization_id = _extract_optimization_id(out)
        rc = main(["inspect-optimization", "--config", str(optimization_config), "--optimization-id", optimization_id, "--format", "json"])
        assert rc == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["optimization_id"] == optimization_id


class TestListAndInspectTrialCommands:
    def test_list_trials_for_outer_fold_zero(self, optimization_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_optimize(optimization_config, capsys)
        optimization_id = _extract_optimization_id(out)
        rc = main(["list-trials", "--config", str(optimization_config), "--optimization-id", optimization_id, "--outer-fold-index", "0"])
        assert rc == 0
        trial_lines = capsys.readouterr().out
        assert "trial 0: status=" in trial_lines

    def test_list_trials_for_unknown_outer_fold_fails_actionably(self, optimization_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_optimize(optimization_config, capsys)
        optimization_id = _extract_optimization_id(out)
        rc = main(["list-trials", "--config", str(optimization_config), "--optimization-id", optimization_id, "--outer-fold-index", "99"])
        assert rc == 1
        assert "No trials recorded" in capsys.readouterr().err

    def test_inspect_trial_prints_full_trial_result(self, optimization_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_optimize(optimization_config, capsys)
        optimization_id = _extract_optimization_id(out)
        rc = main([
            "inspect-trial", "--config", str(optimization_config), "--optimization-id", optimization_id,
            "--outer-fold-index", "0", "--trial-number", "0",
        ])
        assert rc == 0
        trial_out = capsys.readouterr().out
        assert "status: completed" in trial_out or "status: " in trial_out
        assert "sampled_hyperparameters:" in trial_out

    def test_inspect_unknown_trial_number_fails_actionably(self, optimization_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_optimize(optimization_config, capsys)
        optimization_id = _extract_optimization_id(out)
        rc = main([
            "inspect-trial", "--config", str(optimization_config), "--optimization-id", optimization_id,
            "--outer-fold-index", "0", "--trial-number", "999",
        ])
        assert rc == 1
        assert "No trial recorded" in capsys.readouterr().err


class TestVerifyOptimizationCommand:
    def test_verify_a_completed_optimization_is_ready(self, optimization_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_optimize(optimization_config, capsys)
        optimization_id = _extract_optimization_id(out)
        rc = main(["verify-optimization", "--config", str(optimization_config), "--optimization-id", optimization_id])
        assert rc == 0
        assert "is_ready: True" in capsys.readouterr().out


class TestCompareOptimizationCandidatesCommand:
    def test_prints_ranking_table_for_outer_fold(self, optimization_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_optimize(optimization_config, capsys)
        optimization_id = _extract_optimization_id(out)
        rc = main([
            "compare-optimization-candidates", "--config", str(optimization_config),
            "--optimization-id", optimization_id, "--outer-fold-index", "0",
        ])
        assert rc == 0
        table_out = capsys.readouterr().out
        assert "primary_metric: rmse" in table_out
        assert "rank 1: trial=" in table_out


class TestFeatureAndHyperparameterStabilityCommands:
    def test_feature_stability_report(self, optimization_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_optimize(optimization_config, capsys)
        optimization_id = _extract_optimization_id(out)
        rc = main(["feature-stability", "--config", str(optimization_config), "--optimization-id", optimization_id])
        assert rc == 0
        assert "total_evaluations:" in capsys.readouterr().out

    def test_hyperparameter_stability_report(self, optimization_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_optimize(optimization_config, capsys)
        optimization_id = _extract_optimization_id(out)
        rc = main(["hyperparameter-stability", "--config", str(optimization_config), "--optimization-id", optimization_id])
        assert rc == 0
        assert "trial_score_dispersion:" in capsys.readouterr().out
