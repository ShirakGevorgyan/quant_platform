"""Milestone 6, Section 16: exercises the 10 `quant_platform.ml_cli`
robustness commands via REAL `subprocess.run` OS process launches --
`python -m quant_platform.ml_cli ...`, a genuinely separate interpreter
process each time (mirrors `test_backtesting_cli_subprocess.py`'s own
convention and reasoning for why this milestone specifically wants the
heavier, real-subprocess proof rather than in-process `main([...])`
calls).

SCOPE: `run-robustness`/`resume-robustness` themselves are NOT exercised
via subprocess here -- the full pipeline (bootstrap, sensitivity, stress,
regime analysis, independent verification) is expensive (the Section 17
acceptance workflow alone takes several minutes even in-process); running
it a second time as a subprocess would roughly double this test suite's
wall-clock cost for no additional boundary coverage (the CLI's own
argument-parsing/dispatch/config-loading path is identical for
`run-robustness` and `create-robustness-spec`, and the latter already
proves it end to end). This file instead proves: the dry-run
`create-robustness-spec` path against a REAL, valid `source_backtest_id`;
every command's clean, non-traceback, correctly-coded failure path
against config/id/hash inputs that legitimately do not exist; and that
`--help` enumerates all 10 commands. `run-robustness`'s own forward-
pipeline correctness is proven by `test_robustness_real_model_
acceptance.py`, in-process, once."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from tests.integration.test_backtesting_engine import _build_ready_setup

_TIMEOUT = 120


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "quant_platform.ml_cli", *args], capture_output=True, text=True, timeout=_TIMEOUT)


def _assert_no_traceback(result: subprocess.CompletedProcess[str]) -> None:
    assert "Traceback (most recent call last)" not in result.stdout
    assert "Traceback (most recent call last)" not in result.stderr


def _assert_no_nan_or_infinity(text: str) -> None:
    assert "NaN" not in text
    assert "Infinity" not in text


@pytest.fixture(scope="module")
def cli_environment(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp_path = tmp_path_factory.mktemp("robustness_cli_subprocess")
    spec, runner, ml_artifacts_root, _research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)
    outcome = runner.run(spec)
    from quant_platform.backtesting.models import BacktestStage

    assert outcome.manifest.stage is BacktestStage.COMPLETED, outcome.manifest.failure_summary

    robustness_config = {
        "ml_artifacts_root": str(ml_artifacts_root), "historical_storage_root": str(historical_root),
        "research_storage_root": str(research_store.root), "source_backtest_id": outcome.manifest.backtest_id,
        "bootstrap_repetitions": 100,
    }
    config_path = tmp_path / "robustness_config.json"
    config_path.write_text(json.dumps(robustness_config), encoding="utf-8")

    from quant_platform.config.robustness_schemas import RobustnessConfig
    from quant_platform.robustness.specs import compute_robustness_identity

    resolved = RobustnessConfig.model_validate_json(config_path.read_text()).build(source_backtest_spec=spec)
    expected_robustness_id = compute_robustness_identity(resolved).robustness_id

    return {"config_path": config_path, "ml_artifacts_root": ml_artifacts_root, "expected_robustness_id": expected_robustness_id, "tmp_path": tmp_path}


class TestHelpEnumeratesAllRobustnessCommands:
    def test_help_lists_all_ten_commands(self) -> None:
        result = _run_cli("--help")
        assert result.returncode == 0
        for command in (
            "create-robustness-spec", "run-robustness", "resume-robustness", "inspect-robustness", "report-robustness",
            "verify-robustness", "compare-robustness", "inspect-promotion-decision", "inspect-strategy-family", "compare-strategy-candidates",
        ):
            assert command in result.stdout, f"{command!r} missing from --help output"


class TestCreateRobustnessSpec:
    def test_success_prints_the_expected_deterministic_robustness_id(self, cli_environment: dict) -> None:
        result = _run_cli("create-robustness-spec", "--config", str(cli_environment["config_path"]))
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        _assert_no_nan_or_infinity(result.stdout)
        assert f"robustness_id: {cli_environment['expected_robustness_id']}" in result.stdout

    def test_failure_missing_config_file(self, tmp_path: Path) -> None:
        result = _run_cli("create-robustness-spec", "--config", str(tmp_path / "does_not_exist.json"))
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_failure_malformed_config_json(self, tmp_path: Path) -> None:
        bad_config_path = tmp_path / "malformed.json"
        bad_config_path.write_text("{not valid json", encoding="utf-8")
        result = _run_cli("create-robustness-spec", "--config", str(bad_config_path))
        assert result.returncode == 1
        _assert_no_traceback(result)

    def test_failure_nonexistent_source_backtest_id(self, tmp_path: Path, cli_environment: dict) -> None:
        bad_config = {
            "ml_artifacts_root": str(cli_environment["ml_artifacts_root"]), "historical_storage_root": str(cli_environment["ml_artifacts_root"]),
            "research_storage_root": str(cli_environment["ml_artifacts_root"]), "source_backtest_id": "f" * 64,
        }
        bad_config_path = tmp_path / "bad_source.json"
        bad_config_path.write_text(json.dumps(bad_config), encoding="utf-8")
        result = _run_cli("create-robustness-spec", "--config", str(bad_config_path))
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr


class TestUnknownRobustnessIdFailsCleanlyAcrossEveryInspectionCommand:
    """Every inspection/verification command must fail with a clean,
    actionable, non-traceback error (never a raw Python stack) when
    asked about a `--robustness-id` that legitimately does not exist."""

    _UNKNOWN_ROBUSTNESS_ID = "e" * 64

    @pytest.mark.parametrize("command", ["inspect-robustness", "report-robustness"])
    def test_inspect_and_report(self, cli_environment: dict, command: str) -> None:
        result = _run_cli(command, "--config", str(cli_environment["config_path"]), "--robustness-id", self._UNKNOWN_ROBUSTNESS_ID)
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_resume(self, cli_environment: dict) -> None:
        result = _run_cli("resume-robustness", "--config", str(cli_environment["config_path"]), "--robustness-id", self._UNKNOWN_ROBUSTNESS_ID)
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_verify(self, cli_environment: dict) -> None:
        result = _run_cli("verify-robustness", "--config", str(cli_environment["config_path"]), "--robustness-id", self._UNKNOWN_ROBUSTNESS_ID)
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_compare(self, cli_environment: dict) -> None:
        result = _run_cli(
            "compare-robustness", "--config", str(cli_environment["config_path"]), "--robustness-id", self._UNKNOWN_ROBUSTNESS_ID,
            "--baseline-robustness-id", "d" * 64,
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_inspect_promotion_decision(self, cli_environment: dict) -> None:
        result = _run_cli("inspect-promotion-decision", "--config", str(cli_environment["config_path"]), "--robustness-id", self._UNKNOWN_ROBUSTNESS_ID)
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_compare_strategy_candidates(self, cli_environment: dict) -> None:
        result = _run_cli("compare-strategy-candidates", "--config", str(cli_environment["config_path"]), "--robustness-id", self._UNKNOWN_ROBUSTNESS_ID)
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr


class TestInspectStrategyFamilyUnknownContentHash:
    def test_failure_nonexistent_content_hash(self, cli_environment: dict) -> None:
        result = _run_cli("inspect-strategy-family", "--config", str(cli_environment["config_path"]), "--content-hash", "c" * 64)
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr
