from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.unit.features.conftest import make_synthetic_ohlcv, seed_canonical_dataset

from quant_platform.feature_cli import main as feature_main
from quant_platform.ml_cli import build_parser, main


def _build_research_dataset(
    tmp_path: Path, *, preprocessing: dict[str, object] | None = None,
) -> tuple[str, str, Path, Path]:
    """`preprocessing` defaults to omitted (no fitted transforms) -- the
    ONLY dataset shape Milestone 4B's execution engine can safely
    re-split (see `execution.runner.
    assert_preprocessing_is_safe_for_execution`). Pass an explicit
    `preprocessing` dict only from a test that specifically means to
    build an UNSAFE (globally fitted) dataset to prove that check
    rejects it -- see `TestPreprocessingSafety`."""
    historical_root = tmp_path / "data"
    research_root = tmp_path / "research"
    df = make_synthetic_ohlcv(2000, seed=1)
    seed_canonical_dataset(historical_root, df)

    feature_config: dict[str, object] = {
        "symbol": "XAUUSD", "base_timeframe": "M1",
        "start": "2024-01-01T00:00:00Z",
        "end": (pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(minutes=2000)).isoformat(),
        "historical_storage_root": str(historical_root), "research_storage_root": str(research_root),
        "technical": {"return_windows": [1, 5], "momentum_windows": [10], "atr_window": 14},
        "temporal": {"enabled": True},
        "label": {"name": "fut", "kind": "future_return", "horizon_bars": 5},
        "split": {"strategy": "chronological", "train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 5, "embargo_bars": 5},
    }
    if preprocessing is not None:
        feature_config["preprocessing"] = preprocessing
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


@pytest.fixture
def ml_config_walk_forward(tmp_path: Path) -> Path:
    """A walk-forward-strategy variant of `ml_config`, used by the
    Milestone 4B execution commands (`execute`/`resume`/etc.) -- the
    default `ml_config` fixture's `chronological` strategy has no
    concept of folds at all, since `execution.splitters` never dispatches
    on it (see `build_folds_from_split_binding`)."""
    dataset_id, version, research_root, _ = _build_research_dataset(tmp_path)
    return _write_ml_config(
        tmp_path, dataset_id=dataset_id, version=version, research_root=research_root, ml_root=tmp_path / "ml_artifacts",
        split={"strategy": "expanding_walk_forward", "params": {"n_splits": 3, "test_size": 100, "purge_bars": 5, "embargo_bars": 2}},
    )


class TestBuildParser:
    def test_all_seventy_commands_registered(self) -> None:
        """19 Milestone 4A-4C commands, 9 Milestone 4D optimization
        commands, 8 Milestone 4E calibration commands, 8 Milestone 5
        backtest commands, 2 Milestone 5.2 Section 6 lock-recovery
        commands, 10 Milestone 6 robustness/promotion commands, and 14
        Milestone 7 paper-trading/shadow-execution commands, all
        registered on the one shared `ml_cli.py` parser. No command name
        anywhere in this set resembles live order transmission -- see
        `test_safety_scan.py::TestCliHasNoLiveCommands` for the explicit
        negative check (`run-live`/`submit-live-order`/`connect-broker`/
        `execute-mt5`/`deploy-live`)."""
        parser = build_parser()
        subparsers_action = next(a for a in parser._actions if a.dest == "command")
        assert set(subparsers_action.choices) == {
            "list-model-definitions", "describe-model-definition", "prepare-experiment", "validate-experiment",
            "inspect-experiment", "inspect-experiment-manifest", "verify-artifact", "list-experiment-events",
            "execute", "resume", "inspect-execution", "inspect-fold", "list-folds", "verify-execution",
            "list-models", "inspect-model", "validate-model", "train", "compare",
            "optimize", "resume-optimization", "inspect-optimization", "list-trials", "inspect-trial",
            "verify-optimization", "compare-optimization-candidates", "feature-stability", "hyperparameter-stability",
            "create-calibration-spec", "run-calibration", "resume-calibration", "inspect-calibration",
            "report-calibration", "inspect-calibration-fold", "verify-calibration", "compare-calibration",
            "create-backtest-spec", "run-backtest", "resume-backtest", "inspect-backtest",
            "report-backtest", "inspect-backtest-fold", "verify-backtest", "compare-backtests",
            "inspect-backtest-lock", "recover-backtest-lock",
            "create-robustness-spec", "run-robustness", "resume-robustness", "inspect-robustness",
            "report-robustness", "verify-robustness", "compare-robustness", "inspect-promotion-decision",
            "inspect-strategy-family", "compare-strategy-candidates",
            "create-paper-trading-spec", "run-paper-session", "resume-paper-session", "pause-paper-session",
            "inspect-paper-session", "report-paper-session", "verify-paper-session", "compare-paper-to-backtest",
            "inspect-paper-orders", "inspect-paper-fills", "inspect-paper-risk-events", "inspect-paper-reconciliation",
            "run-shadow-session", "report-shadow-session",
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


class TestExecutionCommands:
    def _prepare(self, ml_config: Path, capsys: pytest.CaptureFixture[str]) -> str:
        rc = main(["prepare-experiment", "--config", str(ml_config)])
        assert rc == 0
        out = capsys.readouterr().out
        return out.split("experiment_id: ")[1].split("\n")[0].strip()

    def test_execute_runs_all_folds_to_completion(self, ml_config_walk_forward: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config_walk_forward, capsys)
        rc = main(["execute", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        assert rc == 0
        out = capsys.readouterr().out
        assert "overall_status: completed" in out
        assert "completed_folds: [0, 1, 2]" in out

    def test_execute_is_idempotent(self, ml_config_walk_forward: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config_walk_forward, capsys)
        rc1 = main(["execute", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        capsys.readouterr()
        rc2 = main(["execute", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        out2 = capsys.readouterr().out
        assert rc1 == rc2 == 0
        assert "idempotent_no_op: True" in out2

    def test_resume_without_prior_execution_fails_actionably(self, ml_config_walk_forward: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config_walk_forward, capsys)
        rc = main(["resume", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        assert rc == 1
        assert "ERROR" in capsys.readouterr().err

    def test_inspect_execution_markdown(self, ml_config_walk_forward: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config_walk_forward, capsys)
        main(["execute", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        capsys.readouterr()
        rc = main(["inspect-execution", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        assert rc == 0
        out = capsys.readouterr().out
        assert "# Execution Report" in out

    def test_inspect_execution_json(self, ml_config_walk_forward: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config_walk_forward, capsys)
        main(["execute", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        capsys.readouterr()
        rc = main(["inspect-execution", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id, "--format", "json"])
        assert rc == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["experiment_id"] == experiment_id
        assert parsed["stage"] == "completed"

    def test_list_folds(self, ml_config_walk_forward: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config_walk_forward, capsys)
        main(["execute", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        capsys.readouterr()
        rc = main(["list-folds", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        assert rc == 0
        out = capsys.readouterr().out
        assert "fold 0: completed" in out
        assert "fold 1: completed" in out
        assert "fold 2: completed" in out

    def test_inspect_fold(self, ml_config_walk_forward: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config_walk_forward, capsys)
        main(["execute", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        capsys.readouterr()
        rc = main(["inspect-fold", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id, "--fold-index", "0"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "status: completed" in out
        assert "fold_index: 0" in out

    def test_inspect_fold_unknown_index_fails_actionably(self, ml_config_walk_forward: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config_walk_forward, capsys)
        main(["execute", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        capsys.readouterr()
        rc = main(["inspect-fold", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id, "--fold-index", "99"])
        assert rc == 1
        assert "No fold result" in capsys.readouterr().err

    def test_verify_execution(self, ml_config_walk_forward: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config_walk_forward, capsys)
        main(["execute", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        capsys.readouterr()
        rc = main(["verify-execution", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "is_ready: True" in out
        assert "all_fold_results_verified" in out
        assert "aggregate_verified" in out
        assert "timeline_verified" in out
        assert "experiment_status_compatible" in out

    def test_verify_execution_before_any_run_fails_actionably(self, ml_config_walk_forward: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config_walk_forward, capsys)
        rc = main(["verify-execution", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        assert rc == 1
        assert "ERROR" in capsys.readouterr().err


class TestCliBehaviorOnCorruptedDurableArtifacts:
    """Milestone 4D.1, Section 12: corrupted durable JSON must produce a
    non-zero exit code, a concise domain-specific stderr message, and NO
    Python traceback under normal CLI operation -- never a partial,
    misleading report."""

    def _prepare_and_execute(self, ml_config: Path, capsys: pytest.CaptureFixture[str]) -> str:
        rc = main(["prepare-experiment", "--config", str(ml_config)])
        assert rc == 0
        experiment_id = capsys.readouterr().out.split("experiment_id: ")[1].split("\n")[0].strip()
        rc = main(["execute", "--config", str(ml_config), "--experiment-id", experiment_id])
        assert rc == 0
        capsys.readouterr()
        return experiment_id

    def _corrupt_fold_0_result(self, ml_config: Path, experiment_id: str, *, payload: bytes) -> None:
        from quant_platform.execution.manifests import ExecutionManifestStore
        from quant_platform.ml.artifacts import MLArtifactStore
        from quant_platform.ml.models import ArtifactCategory

        config_data = json.loads(ml_config.read_text())
        artifact_store = MLArtifactStore(config_data["ml_artifacts_root"])
        bad_ref = artifact_store.write_artifact(payload, category=ArtifactCategory.FOLD_RESULT)
        manifest_store = ExecutionManifestStore(config_data["ml_artifacts_root"])
        manifest_path = manifest_store._manifest_path(experiment_id)
        raw = json.loads(manifest_path.read_text())
        raw["fold_result_references"]["0"] = {
            "category": "fold_result", "content_hash": bad_ref.content_hash,
            "size_bytes": len(payload), "created_at": "2024-01-01T00:00:00+00:00",
        }
        manifest_path.write_text(json.dumps(raw))

    def test_verify_execution_on_malformed_fold_result_fails_cleanly_no_traceback(
        self, ml_config_walk_forward: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        experiment_id = self._prepare_and_execute(ml_config_walk_forward, capsys)
        self._corrupt_fold_0_result(ml_config_walk_forward, experiment_id, payload=b"{not valid json")

        rc = main(["verify-execution", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        out, err = capsys.readouterr()
        assert rc == 2, "verify-execution itself does not raise for corruption -- it REPORTS it, then exits 2 (not-ready)"
        assert "is_ready: False" in out
        assert "fold_result_unverifiable" in out
        assert "Traceback" not in out and "Traceback" not in err

    def test_verify_execution_on_nan_poisoned_fold_result_fails_cleanly_no_traceback(
        self, ml_config_walk_forward: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        experiment_id = self._prepare_and_execute(ml_config_walk_forward, capsys)
        poisoned = (
            b'{"schema_version": 1, "fold_index": 0, "train_start": "2024-01-01T00:00:00+00:00", '
            b'"train_end": "2024-01-01T00:00:00+00:00", "test_start": "2024-01-01T00:00:00+00:00", '
            b'"test_end": "2024-01-01T00:00:00+00:00", "train_size": 1, "test_size": 1, "status": "completed", '
            b'"duration_seconds": NaN, "validation_size": 0, "artifact_references": [], "metrics": {}, '
            b'"failure_reason": null}'
        )
        self._corrupt_fold_0_result(ml_config_walk_forward, experiment_id, payload=poisoned)

        rc = main(["verify-execution", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        out, err = capsys.readouterr()
        assert rc == 2
        assert "is_ready: False" in out
        assert "fold_result_unverifiable" in out
        assert "Traceback" not in out and "Traceback" not in err

    def test_inspect_fold_on_malformed_fold_result_returns_domain_error_no_traceback(
        self, ml_config_walk_forward: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        experiment_id = self._prepare_and_execute(ml_config_walk_forward, capsys)
        self._corrupt_fold_0_result(ml_config_walk_forward, experiment_id, payload=b"{not valid json")

        rc = main(["inspect-fold", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id, "--fold-index", "0"])
        out, err = capsys.readouterr()
        assert rc == 1
        assert "ERROR" in err
        assert "Traceback" not in out and "Traceback" not in err

    def test_resume_on_completed_execution_with_corrupted_summary_fails_cleanly_no_traceback(
        self, ml_config_walk_forward: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The idempotent-resume path (`ExecutionRunner._load_existing_
        aggregate`) specifically -- exercised via `execute` called a
        SECOND time on an already-COMPLETED execution, exactly like
        `test_execute_is_idempotent`, except the EXECUTION_SUMMARY
        artifact has been corrupted in between."""
        from quant_platform.execution.manifests import ExecutionManifestStore
        from quant_platform.ml.artifacts import MLArtifactStore
        from quant_platform.ml.models import ArtifactCategory

        experiment_id = self._prepare_and_execute(ml_config_walk_forward, capsys)
        config_data = json.loads(ml_config_walk_forward.read_text())
        artifact_store = MLArtifactStore(config_data["ml_artifacts_root"])
        bad_ref = artifact_store.write_artifact(b"{not valid json", category=ArtifactCategory.EXECUTION_SUMMARY)
        manifest_store = ExecutionManifestStore(config_data["ml_artifacts_root"])
        manifest_path = manifest_store._manifest_path(experiment_id)
        raw = json.loads(manifest_path.read_text())
        for ref in raw["artifact_references"]:
            if ref["category"] == "execution_summary":
                ref["content_hash"] = bad_ref.content_hash
        manifest_path.write_text(json.dumps(raw))

        rc = main(["execute", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        out, err = capsys.readouterr()
        assert rc == 1
        assert "ERROR" in err
        assert "Traceback" not in out and "Traceback" not in err


class TestCliJsonOutputNeverContainsNonFiniteTokens:
    def test_inspect_execution_json_output_is_standards_compliant(
        self, ml_config_walk_forward: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["prepare-experiment", "--config", str(ml_config_walk_forward)])
        assert rc == 0
        experiment_id = capsys.readouterr().out.split("experiment_id: ")[1].split("\n")[0].strip()
        main(["execute", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id])
        capsys.readouterr()

        rc = main([
            "inspect-execution", "--config", str(ml_config_walk_forward), "--experiment-id", experiment_id, "--format", "json",
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "NaN" not in out
        assert "Infinity" not in out
        json.loads(out)  # standard `json` module: would itself reject a bare NaN/Infinity token


_WALK_FORWARD_SPLIT: dict[str, object] = {
    "strategy": "expanding_walk_forward",
    "params": {"n_splits": 3, "test_size": 100, "purge_bars": 5, "embargo_bars": 2},
}


@pytest.fixture
def ml_config_lightgbm(tmp_path: Path) -> Path:
    """Milestone 4C: the SAME walk-forward dataset `ml_config_walk_forward`
    uses, bound to a REAL model (`lightgbm`) instead of the test-only
    one. Written into its own subdirectory (not directly under
    `tmp_path`) so it can coexist with `ml_config_baseline`'s config
    file, which otherwise collides on the shared `tmp_path /
    "ml_config.json"` path `_write_ml_config` always uses."""
    config_dir = tmp_path / "candidate"
    config_dir.mkdir(parents=True, exist_ok=True)
    dataset_id, version, research_root, _ = _build_research_dataset(config_dir)
    return _write_ml_config(
        config_dir, dataset_id=dataset_id, version=version, research_root=research_root, ml_root=tmp_path / "ml_artifacts",
        split=_WALK_FORWARD_SPLIT,
        model={"name": "lightgbm", "version": "1", "objective": "regression", "hyperparameters": {"num_boost_round": 10, "num_leaves": 7}},
    )


@pytest.fixture
def ml_config_baseline(tmp_path: Path) -> Path:
    """Milestone 4C: an INDEPENDENT dataset copy (own subdirectory, same
    synthetic generation seed so its content is identical) but the SAME
    shared `ml_artifacts_root` as `ml_config_lightgbm` -- `compare`
    resolves both the candidate and every baseline experiment id from a
    single `--config`'s `ml_artifacts_root`, so both configs must agree
    on that root even though their dataset-build directories differ."""
    config_dir = tmp_path / "baseline"
    config_dir.mkdir(parents=True, exist_ok=True)
    dataset_id, version, research_root, _ = _build_research_dataset(config_dir)
    return _write_ml_config(
        config_dir, dataset_id=dataset_id, version=version, research_root=research_root, ml_root=tmp_path / "ml_artifacts",
        split=_WALK_FORWARD_SPLIT,
        model={"name": "dummy_mean_regressor", "version": "1", "objective": "regression", "hyperparameters": {}},
    )


class TestListAndInspectModels:
    def test_list_models_shows_every_real_model_and_the_test_model(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["list-models"])
        assert rc == 0
        out = capsys.readouterr().out
        for name in ("lightgbm@1", "xgboost@1", "catboost@1", "logistic_regression@1", "elastic_net@1",
                     "constant_predictor@1", "random_predictor@1", "majority_predictor@1", "dummy_mean_regressor@1",
                     "constant_test_model@1"):
            assert name in out

    def test_inspect_model_prints_every_capability_field(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["inspect-model", "--name", "catboost", "--version", "1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "supports_categorical_features: True" in out
        assert "library_name: catboost" in out
        assert "seed_usage:" in out
        assert "required_preprocessing:" in out

    def test_inspect_unknown_model_fails_actionably(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["inspect-model", "--name", "no_such_model", "--version", "1"])
        assert rc == 1
        assert "ERROR" in capsys.readouterr().err


class TestValidateModel:
    def test_validate_model_ready_for_lightgbm(self, ml_config_lightgbm: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["validate-model", "--config", str(ml_config_lightgbm)])
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "is_ready: True" in out
        assert "all_features_numeric" in out

    def test_validate_model_before_any_prepare_still_works(self, ml_config_lightgbm: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """`validate-model` writes nothing and needs no prior
        `prepare-experiment` call -- a pure dry-run against the dataset."""
        rc = main(["validate-model", "--config", str(ml_config_lightgbm)])
        assert rc == 0


class TestTrainCommand:
    def _prepare(self, ml_config: Path, capsys: pytest.CaptureFixture[str]) -> str:
        rc = main(["prepare-experiment", "--config", str(ml_config)])
        assert rc == 0
        out = capsys.readouterr().out
        return out.split("experiment_id: ")[1].split("\n")[0].strip()

    def test_train_runs_all_folds_with_real_model_and_completes(self, ml_config_lightgbm: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config_lightgbm, capsys)
        rc = main(["train", "--config", str(ml_config_lightgbm), "--experiment-id", experiment_id])
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "overall_status: completed" in out

    def test_train_persists_real_metrics_readable_via_inspect_fold(self, ml_config_lightgbm: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config_lightgbm, capsys)
        main(["train", "--config", str(ml_config_lightgbm), "--experiment-id", experiment_id])
        capsys.readouterr()
        rc = main(["inspect-fold", "--config", str(ml_config_lightgbm), "--experiment-id", experiment_id, "--fold-index", "0"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "metrics:" in out
        assert "mae" in out

    def test_train_is_idempotent(self, ml_config_lightgbm: Path, capsys: pytest.CaptureFixture[str]) -> None:
        experiment_id = self._prepare(ml_config_lightgbm, capsys)
        main(["train", "--config", str(ml_config_lightgbm), "--experiment-id", experiment_id])
        capsys.readouterr()
        rc = main(["train", "--config", str(ml_config_lightgbm), "--experiment-id", experiment_id])
        out = capsys.readouterr().out
        assert rc == 0
        assert "idempotent_no_op: True" in out

    def test_execute_still_works_for_a_real_model_without_metrics(self, ml_config_lightgbm: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """`execute` (unchanged since Milestone 4B, `DeterministicFoldExecutor`)
        still successfully runs a REAL model end to end -- just without
        computing metrics/probabilities/training-metadata."""
        experiment_id = self._prepare(ml_config_lightgbm, capsys)
        rc = main(["execute", "--config", str(ml_config_lightgbm), "--experiment-id", experiment_id])
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "overall_status: completed" in out
        rc = main(["inspect-fold", "--config", str(ml_config_lightgbm), "--experiment-id", experiment_id, "--fold-index", "0"])
        fold_out = capsys.readouterr().out
        assert "metrics: {}" in fold_out


class TestCompareCommand:
    def _prepare_and_train(self, ml_config: Path, capsys: pytest.CaptureFixture[str]) -> str:
        rc = main(["prepare-experiment", "--config", str(ml_config)])
        assert rc == 0
        out = capsys.readouterr().out
        experiment_id = out.split("experiment_id: ")[1].split("\n")[0].strip()
        rc = main(["train", "--config", str(ml_config), "--experiment-id", experiment_id])
        assert rc == 0
        capsys.readouterr()
        return experiment_id

    def test_compare_reports_a_result_and_a_definite_exit_code(
        self, ml_config_lightgbm: Path, ml_config_baseline: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        candidate_id = self._prepare_and_train(ml_config_lightgbm, capsys)
        baseline_id = self._prepare_and_train(ml_config_baseline, capsys)
        rc = main([
            "compare", "--config", str(ml_config_lightgbm),
            "--candidate-experiment-id", candidate_id, "--baseline-experiment-id", baseline_id,
            "--primary-metric", "mae",
        ])
        out = capsys.readouterr().out
        assert rc in (0, 2)
        assert "outperforms_all_baselines (mae):" in out
        assert "mae:" in out

    def test_compare_before_training_fails_actionably(self, ml_config_lightgbm: Path, ml_config_baseline: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["prepare-experiment", "--config", str(ml_config_lightgbm)])
        out = capsys.readouterr().out
        candidate_id = out.split("experiment_id: ")[1].split("\n")[0].strip()
        rc = main(["prepare-experiment", "--config", str(ml_config_baseline)])
        out = capsys.readouterr().out
        baseline_id = out.split("experiment_id: ")[1].split("\n")[0].strip()

        rc = main([
            "compare", "--config", str(ml_config_lightgbm),
            "--candidate-experiment-id", candidate_id, "--baseline-experiment-id", baseline_id,
            "--primary-metric", "mae",
        ])
        assert rc == 1
        assert "ERROR" in capsys.readouterr().err
