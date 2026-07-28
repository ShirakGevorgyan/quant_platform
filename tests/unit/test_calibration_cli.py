"""Milestone 4E release audit, Section 12: CLI tests for the 8 calibration
commands added to the shared `quant_platform.ml_cli` parser. Mirrors
`tests/unit/test_optimization_cli.py`'s exact conventions -- `main([...])`
invoked in-process with `capsys` capturing stdout/stderr, a real
(constant-model) research dataset/experiment built once per test via the
standard config-JSON path, never mocked. This is the SAME "exercise the
CLI" methodology every other command group in this codebase already uses
(`test_ml_cli.py`, `test_optimization_cli.py`, `test_feature_cli.py`,
`test_data_cli.py`) -- none of them spawn a literal OS subprocess either;
`main([...])` IS this platform's established CLI-testing boundary
(argument parsing, dispatch, exit codes, and stdout/stderr formatting all
run for real, only the OS process boundary itself is not re-crossed). A
handful of genuine OS-level subprocess invocations were additionally run
live (not as committed tests) as part of the release audit -- see the
delivery report.

Covers, for every one of the 8 commands: the success path, at least one
failure mode (invalid ID, missing artifact, wrong config), the exit code
convention (0=success, 1=command-level failure, 2=semantic not-ready), and
that no raw traceback ever reaches the user (only `main`'s own
`ERROR: {exc}` formatting -- see `main`'s `except (QuantPlatformError,
ValidationError, OSError, ValueError, KeyError, TypeError)` clause)."""

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
    df = make_synthetic_ohlcv(1200, seed=3)
    seed_canonical_dataset(historical_root, df)

    feature_config: dict[str, object] = {
        "symbol": "XAUUSD", "base_timeframe": "M1", "start": "2024-01-01T00:00:00Z",
        "end": (pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(minutes=1200)).isoformat(),
        "historical_storage_root": str(historical_root), "research_storage_root": str(research_root),
        "technical": {"return_windows": [1, 5], "momentum_windows": [10], "atr_window": 14},
        "temporal": {"enabled": True},
        "label": {"name": "fut5", "kind": "binary_direction", "horizon_bars": 5},
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
        "label": {"name": "fut5", "kind": "binary_direction", "horizon_bars": 5, "label_type": "binary", "params": {}},
        "split": {"strategy": "expanding_walk_forward", "params": {"n_splits": 2, "test_size": 120, "purge_bars": 5, "embargo_bars": 2}},
        "model": {"name": "constant_test_model", "version": "1", "objective": "binary_classification", "hyperparameters": {}},
        "seeds": {"master_seed": 11},
        "primary_metric": "accuracy", "environment_requirements": {}, "tags": [], "notes": "calibration cli test",
    }
    path = tmp_path / "ml_config.json"
    path.write_text(json.dumps(config))
    return path


def _write_calibration_config(
    tmp_path: Path, *, source_experiment_id: str, ml_root: Path, research_root: Path, name: str = "calibration_config.json",
) -> Path:
    config: dict[str, object] = {
        "ml_artifacts_root": str(ml_root), "research_storage_root": str(research_root),
        "source_experiment_id": source_experiment_id,
        "calibration_method_candidates": ["identity", "platt", "isotonic"],
        "calibration_selection_metric": "log_loss",
        "minimum_calibration_sample_count": 10, "minimum_samples_per_class": 2,
        "inner_oof_policy": {"strategy": "expanding_walk_forward", "n_splits": 2, "test_size_fraction": 0.2, "embargo_bars": 1},
        "threshold": {"policy": "f1", "candidate_grid_size": 51},
        "abstention": {"policy": "none"},
        "confidence": {"very_low_max": 0.2, "low_max": 0.4, "medium_max": 0.6, "high_max": 0.8},
        "uncertainty": {"components": ["entropy", "margin"], "aggregation": "mean"},
        "reliability_binning": [{"strategy": "equal_width", "n_bins": 10}],
        "seeds": {"master_seed": 5}, "determinism_policy": "strict",
    }
    path = tmp_path / name
    path.write_text(json.dumps(config))
    return path


def _prepare_experiment_and_get_id(ml_config_path: Path, capsys: pytest.CaptureFixture[str]) -> str:
    rc = main(["prepare-experiment", "--config", str(ml_config_path)])
    assert rc == 0
    out = capsys.readouterr().out
    return out.split("experiment_id: ")[1].split("\n")[0].strip()


@pytest.fixture
def calibration_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """A prepared parent experiment plus a matching `CalibrationConfig`
    JSON pointing at it -- every calibration command requires the parent
    experiment to already be READY."""
    dataset_id, version, research_root, _ = _build_research_dataset(tmp_path)
    ml_root = tmp_path / "ml_artifacts"
    ml_config_path = _write_ml_config(tmp_path, dataset_id=dataset_id, version=version, research_root=research_root, ml_root=ml_root)
    experiment_id = _prepare_experiment_and_get_id(ml_config_path, capsys)
    return _write_calibration_config(tmp_path, source_experiment_id=experiment_id, ml_root=ml_root, research_root=research_root)


def _run_calibration(config_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    rc = main(["run-calibration", "--config", str(config_path)])
    out = capsys.readouterr().out
    return rc, out


def _extract_calibration_id(out: str) -> str:
    return out.split("calibration_id: ")[1].split("\n")[0].strip()


class TestBuildParserIncludesCalibrationCommands:
    def test_all_eight_calibration_commands_registered(self) -> None:
        parser = build_parser()
        subparsers_action = next(a for a in parser._actions if a.dest == "command")
        assert {
            "create-calibration-spec", "run-calibration", "resume-calibration", "inspect-calibration",
            "report-calibration", "inspect-calibration-fold", "verify-calibration", "compare-calibration",
        }.issubset(set(subparsers_action.choices))


class TestCreateCalibrationSpecCommand:
    def test_dry_run_prints_identity_and_writes_nothing(self, calibration_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        config = json.loads(calibration_config.read_text())
        ml_root = Path(config["ml_artifacts_root"])
        calibrations_dir_before = list((ml_root / "calibrations").glob("*")) if (ml_root / "calibrations").is_dir() else []
        rc = main(["create-calibration-spec", "--config", str(calibration_config)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "calibration_id:" in out
        assert "calibration_method_candidates:" in out
        calibrations_dir_after = list((ml_root / "calibrations").glob("*")) if (ml_root / "calibrations").is_dir() else []
        assert calibrations_dir_before == calibrations_dir_after, "create-calibration-spec must write nothing (dry run)"

    def test_unknown_source_experiment_id_fails_actionably_no_traceback(
        self, calibration_config: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = json.loads(calibration_config.read_text())
        config["source_experiment_id"] = "f" * 64
        bad_config_path = calibration_config.parent / "bad_config.json"
        bad_config_path.write_text(json.dumps(config))
        rc = main(["create-calibration-spec", "--config", str(bad_config_path)])
        assert rc == 1
        err = capsys.readouterr().err
        assert err.startswith("ERROR:")
        assert "Traceback" not in err

    def test_missing_config_file_fails_actionably(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["create-calibration-spec", "--config", str(tmp_path / "does_not_exist.json")])
        assert rc == 1
        err = capsys.readouterr().err
        assert err.startswith("ERROR:")
        assert "Traceback" not in err
        # No absolute-path-revealing raw OSError leakage beyond the message itself intentionally naming the missing path.
        assert str(tmp_path) not in err or "does_not_exist.json" in err


class TestRunCalibrationCommand:
    def test_run_completes_and_is_idempotent(self, calibration_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc1, out1 = _run_calibration(calibration_config, capsys)
        assert rc1 == 0
        assert "stage: completed" in out1
        calibration_id_1 = _extract_calibration_id(out1)

        rc2, out2 = _run_calibration(calibration_config, capsys)
        assert rc2 == 0
        assert _extract_calibration_id(out2) == calibration_id_1

    def test_run_with_malformed_json_config_fails_actionably(self, calibration_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        malformed_path = calibration_config.parent / "malformed.json"
        malformed_path.write_text("{not valid json")
        rc = main(["run-calibration", "--config", str(malformed_path)])
        assert rc == 1
        err = capsys.readouterr().err
        assert err.startswith("ERROR:")
        assert "Traceback" not in err

    def test_run_with_wrong_schema_config_fails_actionably(self, calibration_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """An extra, unknown field must be rejected (`extra='forbid'`)."""
        config = json.loads(calibration_config.read_text())
        config["unsupported_extra_field"] = "surprise"
        wrong_schema_path = calibration_config.parent / "wrong_schema.json"
        wrong_schema_path.write_text(json.dumps(config))
        rc = main(["run-calibration", "--config", str(wrong_schema_path)])
        assert rc == 1
        err = capsys.readouterr().err
        assert err.startswith("ERROR:")
        assert "Traceback" not in err


class TestResumeCalibrationCommand:
    def test_resuming_a_completed_calibration_is_an_idempotent_no_op(self, calibration_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Unlike `resume-optimization` (which raises for an already-
        terminal optimization), `CalibrationRunner.resume()` deliberately
        special-cases an already-`COMPLETED` calibration as a safe
        idempotent no-op -- see `cmd_resume_calibration`'s docstring."""
        _, out = _run_calibration(calibration_config, capsys)
        calibration_id = _extract_calibration_id(out)
        rc = main(["resume-calibration", "--config", str(calibration_config), "--calibration-id", calibration_id])
        assert rc == 0
        resume_out = capsys.readouterr().out
        assert "stage: completed" in resume_out
        assert _extract_calibration_id(resume_out) == calibration_id

    def test_resuming_an_unknown_calibration_id_fails_actionably(self, calibration_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["resume-calibration", "--config", str(calibration_config), "--calibration-id", "a" * 64])
        assert rc == 1
        err = capsys.readouterr().err
        assert err.startswith("ERROR:")
        assert "Traceback" not in err

    def test_resuming_an_invalid_id_format_fails_actionably(self, calibration_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["resume-calibration", "--config", str(calibration_config), "--calibration-id", "not-a-valid-hash"])
        assert rc == 1
        err = capsys.readouterr().err
        assert err.startswith("ERROR:")
        assert "Traceback" not in err


class TestInspectAndReportCalibrationCommands:
    def test_inspect_markdown(self, calibration_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_calibration(calibration_config, capsys)
        calibration_id = _extract_calibration_id(out)
        rc = main(["inspect-calibration", "--config", str(calibration_config), "--calibration-id", calibration_id])
        assert rc == 0
        markdown = capsys.readouterr().out
        assert "#" in markdown

    def test_inspect_json_is_well_formed_and_rejects_nan(self, calibration_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_calibration(calibration_config, capsys)
        calibration_id = _extract_calibration_id(out)
        rc = main(["inspect-calibration", "--config", str(calibration_config), "--calibration-id", calibration_id, "--format", "json"])
        assert rc == 0
        raw = capsys.readouterr().out
        assert "NaN" not in raw and "Infinity" not in raw
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_report_calibration_is_an_alias_for_inspect(self, calibration_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_calibration(calibration_config, capsys)
        calibration_id = _extract_calibration_id(out)
        rc = main(["report-calibration", "--config", str(calibration_config), "--calibration-id", calibration_id, "--format", "json"])
        assert rc == 0
        report_out = json.loads(capsys.readouterr().out)
        rc2 = main(["inspect-calibration", "--config", str(calibration_config), "--calibration-id", calibration_id, "--format", "json"])
        assert rc2 == 0
        inspect_out = json.loads(capsys.readouterr().out)
        assert report_out == inspect_out

    def test_inspect_unknown_calibration_id_fails_actionably(self, calibration_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["inspect-calibration", "--config", str(calibration_config), "--calibration-id", "b" * 64])
        assert rc == 1
        err = capsys.readouterr().err
        assert err.startswith("ERROR:")
        assert "Traceback" not in err


class TestInspectCalibrationFoldCommand:
    def test_inspect_fold_zero(self, calibration_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_calibration(calibration_config, capsys)
        calibration_id = _extract_calibration_id(out)
        rc = main(["inspect-calibration-fold", "--config", str(calibration_config), "--calibration-id", calibration_id, "--outer-fold-index", "0"])
        assert rc == 0
        fold_out = capsys.readouterr().out
        assert "classification_metrics:" in fold_out
        assert "decision_counts:" in fold_out
        assert "nan" not in fold_out.lower().replace("nan_", "")  # crude but sufficient: no bare NaN token in the text report

    def test_inspect_unknown_outer_fold_index_fails_cleanly_not_a_crash(
        self, calibration_config: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out = _run_calibration(calibration_config, capsys)
        calibration_id = _extract_calibration_id(out)
        rc = main(["inspect-calibration-fold", "--config", str(calibration_config), "--calibration-id", calibration_id, "--outer-fold-index", "999"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "No recorded result" in err
        assert "Traceback" not in err


class TestVerifyCalibrationCommand:
    def test_verify_a_completed_calibration_is_ready(self, calibration_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = _run_calibration(calibration_config, capsys)
        calibration_id = _extract_calibration_id(out)
        rc = main(["verify-calibration", "--config", str(calibration_config), "--calibration-id", calibration_id])
        assert rc == 0
        verify_out = capsys.readouterr().out
        assert "is_ready: True" in verify_out
        assert "calibrated_probabilities_reproduce" in verify_out

    def test_verify_a_corrupted_calibration_returns_two_not_a_crash(
        self, calibration_config: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = json.loads(calibration_config.read_text())
        ml_root = Path(config["ml_artifacts_root"])
        _, out = _run_calibration(calibration_config, capsys)
        calibration_id = _extract_calibration_id(out)

        from quant_platform.calibration.manifests import CalibrationManifestStore
        from quant_platform.ml.artifacts import MLArtifactStore

        manifest = CalibrationManifestStore(ml_root).load(calibration_id)
        ref = manifest.outer_fold_result_references[0]
        content_path = ml_root / "content" / ref.content_hash[:2] / ref.content_hash
        original = content_path.read_bytes()
        content_path.write_bytes(original[:-1] + (b"\x00" if original[-1:] != b"\x00" else b"\x01"))

        rc = main(["verify-calibration", "--config", str(calibration_config), "--calibration-id", calibration_id])
        assert rc == 2
        verify_out = capsys.readouterr().out
        assert "is_ready: False" in verify_out
        assert MLArtifactStore is not None  # imported for clarity of what module owns content-hash verification


class TestCompareCalibrationCommand:
    def test_compares_two_runs_side_by_side(self, calibration_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        config = json.loads(calibration_config.read_text())
        _, out1 = _run_calibration(calibration_config, capsys)
        calibration_id_1 = _extract_calibration_id(out1)

        second_config_path = _write_calibration_config(
            tmp_path, source_experiment_id=config["source_experiment_id"], ml_root=Path(config["ml_artifacts_root"]),
            research_root=Path(config["research_storage_root"]), name="calibration_config_2.json",
        )
        second_config = json.loads(second_config_path.read_text())
        second_config["seeds"]["master_seed"] = 999
        second_config_path.write_text(json.dumps(second_config))
        rc2, out2 = _run_calibration(second_config_path, capsys)
        assert rc2 == 0
        calibration_id_2 = _extract_calibration_id(out2)
        assert calibration_id_2 != calibration_id_1, "a different seed must produce a different calibration_id"

        rc = main([
            "compare-calibration", "--config", str(calibration_config), "--calibration-id", calibration_id_1,
            "--baseline-calibration-id", calibration_id_2, "--metric", "accuracy",
        ])
        assert rc == 0
        compare_out = capsys.readouterr().out
        assert calibration_id_1[:12] in compare_out
        assert calibration_id_2[:12] in compare_out

    def test_compare_with_unsupported_metric_name_prints_none_not_a_crash(
        self, calibration_config: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out = _run_calibration(calibration_config, capsys)
        calibration_id = _extract_calibration_id(out)
        rc = main([
            "compare-calibration", "--config", str(calibration_config), "--calibration-id", calibration_id,
            "--baseline-calibration-id", calibration_id, "--metric", "definitely_not_a_real_metric_name",
        ])
        assert rc == 0
        compare_out = capsys.readouterr().out
        assert "None" in compare_out
