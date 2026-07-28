"""Milestone 5.1, Section 8: exercises all 8 `quant_platform.ml_cli`
backtest commands (`create-backtest-spec`, `run-backtest`, `resume-
backtest`, `inspect-backtest`, `report-backtest`, `inspect-backtest-fold`,
`verify-backtest`, `compare-backtests`) via REAL `subprocess.run` OS
process launches -- `python -m quant_platform.ml_cli ...`, a genuinely
separate interpreter process each time, NOT `main([...])` invoked
in-process (this platform's usual CLI-testing convention -- see `tests/
unit/test_calibration_cli.py`'s own docstring for why that convention is
normally sufficient and why this milestone specifically asks for the
heavier, real-subprocess proof this file provides instead).

The expensive real dataset/experiment/execution/calibration pipeline is
built ONCE per test session (via the existing, already-verified
`_build_ready_setup` helper, reused exactly as `test_backtesting_
throughput.py::TestBacktestRunnerThroughput` already reuses it) --
building it fresh via CLI subprocesses for every command would be pure
overhead unrelated to what this file actually verifies (the CLI
boundary itself: argument parsing, dispatch, exit codes, stdout/stderr
formatting, JSON validity, error handling -- never leakage-safety, which
is what the rest of the backtesting test suite already proves)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from tests.integration.test_backtesting_engine import _build_ready_setup

_TIMEOUT = 120


def _config_json_from_spec(spec, *, ml_artifacts_root: Path, historical_storage_root: Path, research_storage_root: Path) -> dict:
    return {
        "ml_artifacts_root": str(ml_artifacts_root), "historical_storage_root": str(historical_storage_root),
        "research_storage_root": str(research_storage_root), "source_calibration_id": spec.source_calibration_id,
        "decision_timestamp_policy": spec.decision_timestamp_policy.value,
        "signal_mapping": {
            "kind": spec.signal_mapping.kind.value, "probability_band_long_min": spec.signal_mapping.probability_band_long_min,
            "probability_band_short_max": spec.signal_mapping.probability_band_short_max, "confidence_floor": spec.signal_mapping.confidence_floor,
            "uncertainty_ceiling": spec.signal_mapping.uncertainty_ceiling,
        },
        "position_mode": spec.position_mode.value,
        "entry": {"kind": spec.entry_spec.kind.value, "delay_bars": spec.entry_spec.delay_bars, "allow_same_bar_close": spec.entry_spec.allow_same_bar_close},
        "exit": {"kind": spec.exit_spec.kind.value, "holding_period_bars": spec.exit_spec.holding_period_bars, "final_trade_policy": spec.exit_spec.final_trade_policy.value},
        "overlap_policy": spec.overlap_policy.value, "price_basis": spec.price_basis.value,
        "spread": {"kind": spec.spread_spec.kind.value, "price_units": spec.spread_spec.price_units, "basis_points": spec.spread_spec.basis_points},
        "commission": {"kind": spec.commission_spec.kind.value, "per_side_basis_points": spec.commission_spec.per_side_basis_points, "fixed_per_trade": spec.commission_spec.fixed_per_trade},
        "slippage": {"kind": spec.slippage_spec.kind.value, "basis_points": spec.slippage_spec.basis_points, "price_units": spec.slippage_spec.price_units},
        "financing": {"kind": spec.financing_spec.kind.value, "daily_basis_points": spec.financing_spec.daily_basis_points},
        "return_calculation_policy": spec.return_calculation_policy.value, "compounding_policy": spec.compounding_policy.value,
        "initial_notional": spec.initial_notional, "respect_calibration_abstention": spec.respect_calibration_abstention,
        "exposure_cap": spec.exposure_cap, "annual_risk_free_rate": spec.annual_risk_free_rate,
        "minimum_trades_for_valid_fold": spec.minimum_trades_for_valid_fold, "timestamp_column": spec.timestamp_column,
        "seed": spec.seed, "determinism_policy": spec.determinism_policy.value,
    }


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
    tmp_path = tmp_path_factory.mktemp("backtest_cli_subprocess")
    spec, _runner, ml_artifacts_root, _research_manifest_store, research_store, historical_root = _build_ready_setup(tmp_path)
    config = _config_json_from_spec(spec, ml_artifacts_root=ml_artifacts_root, historical_storage_root=historical_root, research_storage_root=research_store.root)
    config_path = tmp_path / "backtest_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    from quant_platform.backtesting.specs import compute_backtest_identity

    expected_backtest_id = compute_backtest_identity(spec).backtest_id

    # Bring the backtest to COMPLETED via a real `run-backtest` subprocess
    # here in fixture setup (not left to whichever test class happens to
    # run first) -- every test in this module can then assume a genuinely
    # completed backtest exists, regardless of test selection/order.
    setup_result = _run_cli("run-backtest", "--config", str(config_path))
    assert setup_result.returncode == 0, f"fixture setup: run-backtest failed: {setup_result.stderr}"
    assert "stage: completed" in setup_result.stdout

    return {
        "config_path": config_path, "ml_artifacts_root": ml_artifacts_root, "expected_backtest_id": expected_backtest_id,
        "tmp_path": tmp_path,
    }


class TestCreateBacktestSpec:
    def test_success_prints_the_expected_deterministic_backtest_id(self, cli_environment: dict) -> None:
        result = _run_cli("create-backtest-spec", "--config", str(cli_environment["config_path"]))
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        assert f"backtest_id: {cli_environment['expected_backtest_id']}" in result.stdout

    def test_failure_missing_config_file(self, tmp_path: Path) -> None:
        result = _run_cli("create-backtest-spec", "--config", str(tmp_path / "does_not_exist.json"))
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr


class TestRunBacktest:
    def test_success_runs_to_completion_and_is_idempotent(self, cli_environment: dict) -> None:
        result = _run_cli("run-backtest", "--config", str(cli_environment["config_path"]))
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        assert f"backtest_id: {cli_environment['expected_backtest_id']}" in result.stdout
        assert "stage: completed" in result.stdout

        # Idempotent re-run via the SAME CLI command (a second real subprocess).
        second = _run_cli("run-backtest", "--config", str(cli_environment["config_path"]))
        assert second.returncode == 0, second.stderr
        assert "stage: completed" in second.stdout

    def test_failure_malformed_config_json(self, tmp_path: Path) -> None:
        bad_config_path = tmp_path / "malformed.json"
        bad_config_path.write_text("{not valid json", encoding="utf-8")
        result = _run_cli("run-backtest", "--config", str(bad_config_path))
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr


class TestInspectBacktest:
    def test_success_text_format(self, cli_environment: dict) -> None:
        result = _run_cli("inspect-backtest", "--config", str(cli_environment["config_path"]), "--backtest-id", cli_environment["expected_backtest_id"])
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        assert "Backtest Report" in result.stdout
        _assert_no_nan_or_infinity(result.stdout)

    def test_success_json_format_is_valid_and_finite(self, cli_environment: dict) -> None:
        result = _run_cli("inspect-backtest", "--config", str(cli_environment["config_path"]), "--backtest-id", cli_environment["expected_backtest_id"], "--format", "json")
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        payload = json.loads(result.stdout)  # raises if not valid JSON
        assert payload["backtest_id"] == cli_environment["expected_backtest_id"]
        assert payload["stage"] == "completed"
        assert payload["stitched_equity"] is not None
        _assert_no_nan_or_infinity(result.stdout)
        # No accidental absolute local filesystem path leakage into the report body.
        assert str(cli_environment["tmp_path"]) not in result.stdout

    def test_failure_unknown_backtest_id(self, cli_environment: dict) -> None:
        result = _run_cli("inspect-backtest", "--config", str(cli_environment["config_path"]), "--backtest-id", "f" * 64)
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr


class TestReportBacktest:
    def test_success_is_an_alias_for_inspect(self, cli_environment: dict) -> None:
        result = _run_cli("report-backtest", "--config", str(cli_environment["config_path"]), "--backtest-id", cli_environment["expected_backtest_id"], "--format", "json")
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        payload = json.loads(result.stdout)
        assert payload["backtest_id"] == cli_environment["expected_backtest_id"]

    def test_failure_unknown_backtest_id(self, cli_environment: dict) -> None:
        result = _run_cli("report-backtest", "--config", str(cli_environment["config_path"]), "--backtest-id", "e" * 64)
        assert result.returncode == 1
        _assert_no_traceback(result)


class TestInspectBacktestFold:
    def test_success_prints_fold_zero(self, cli_environment: dict) -> None:
        result = _run_cli("inspect-backtest-fold", "--config", str(cli_environment["config_path"]), "--backtest-id", cli_environment["expected_backtest_id"], "--outer-fold-index", "0")
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        assert "outer_fold_index: 0" in result.stdout
        _assert_no_nan_or_infinity(result.stdout)

    def test_failure_unknown_outer_fold_index(self, cli_environment: dict) -> None:
        result = _run_cli("inspect-backtest-fold", "--config", str(cli_environment["config_path"]), "--backtest-id", cli_environment["expected_backtest_id"], "--outer-fold-index", "999")
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "No recorded result" in result.stderr


class TestVerifyBacktest:
    def test_success_reports_ready(self, cli_environment: dict) -> None:
        result = _run_cli("verify-backtest", "--config", str(cli_environment["config_path"]), "--backtest-id", cli_environment["expected_backtest_id"])
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        assert "is_ready: True" in result.stdout
        assert any("] stitched_equity_reproduces:" in line for line in result.stdout.splitlines())

    def test_failure_on_corrupted_artifact_returns_exit_code_2_not_a_traceback(self, cli_environment: dict) -> None:
        """A semantic failure (not-ready) must return exit code 2 (per
        `cmd_verify_backtest`'s own documented convention), distinct from
        1 (command-level failure) -- and must never leak a raw traceback
        even though the underlying artifact is corrupted."""
        ml_artifacts_root = cli_environment["ml_artifacts_root"]
        backtest_id = cli_environment["expected_backtest_id"]

        from quant_platform.backtesting.manifests import BacktestManifestStore

        manifest = BacktestManifestStore(ml_artifacts_root).load(backtest_id)
        ref = manifest.outer_fold_result_references[0]
        content_path = ml_artifacts_root / "content" / ref.content_hash[:2] / ref.content_hash
        original = content_path.read_bytes()
        content_path.write_bytes(original[:-1] + (b"\x00" if original[-1:] != b"\x00" else b"\x01"))
        try:
            result = _run_cli("verify-backtest", "--config", str(cli_environment["config_path"]), "--backtest-id", backtest_id)
            assert result.returncode == 2, result.stdout
            _assert_no_traceback(result)
            assert "is_ready: False" in result.stdout
        finally:
            content_path.write_bytes(original)  # restore for subsequent tests in this module


class TestCompareBacktests:
    def test_success_compares_a_backtest_against_itself(self, cli_environment: dict) -> None:
        """No second completed backtest is built solely for this test
        (subprocess/dataset overhead) -- comparing the SAME backtest_id
        against itself still exercises the full command (parsing,
        loading both sides, per-fold side-by-side printing) end to end."""
        backtest_id = cli_environment["expected_backtest_id"]
        result = _run_cli(
            "compare-backtests", "--config", str(cli_environment["config_path"]), "--backtest-id", backtest_id,
            "--baseline-backtest-id", backtest_id, "--metric", "total_net_return",
        )
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        assert "outer_fold 0" in result.stdout
        _assert_no_nan_or_infinity(result.stdout)

    def test_failure_unknown_baseline_backtest_id(self, cli_environment: dict) -> None:
        backtest_id = cli_environment["expected_backtest_id"]
        result = _run_cli(
            "compare-backtests", "--config", str(cli_environment["config_path"]), "--backtest-id", backtest_id,
            "--baseline-backtest-id", "d" * 64, "--metric", "total_net_return",
        )
        assert result.returncode == 1
        _assert_no_traceback(result)


class TestResumeBacktest:
    def test_success_on_an_already_completed_backtest_is_an_idempotent_no_op(self, cli_environment: dict) -> None:
        """`resume-backtest` on an already-COMPLETED backtest is a safe,
        documented no-op (mirrors `run-backtest`'s own idempotency)."""
        result = _run_cli("resume-backtest", "--config", str(cli_environment["config_path"]), "--backtest-id", cli_environment["expected_backtest_id"])
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        assert "stage: completed" in result.stdout


class TestBacktestLockCommands:
    """Milestone 5.2, Section 6, exercised through REAL CLI subprocesses
    (matching this file's own convention for the other 8 backtest
    commands)."""

    def _lock_path(self, cli_environment: dict) -> Path:
        return cli_environment["ml_artifacts_root"] / "backtests" / cli_environment["expected_backtest_id"] / ".backtest_run.lock"

    def test_inspect_lock_on_a_completed_backtest_shows_no_lock(self, cli_environment: dict) -> None:
        """The lock is released once `run-backtest` completes -- a
        completed backtest's lock diagnostics must show nothing held."""
        result = _run_cli("inspect-backtest-lock", "--config", str(cli_environment["config_path"]), "--backtest-id", cli_environment["expected_backtest_id"])
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        assert "lock_exists: False" in result.stdout
        assert "manifest_exists: True" in result.stdout
        assert "manifest_backtest_id_matches: True" in result.stdout

    def test_recover_lock_refuses_a_fresh_abandoned_lock_without_force(self, cli_environment: dict) -> None:
        import json as json_module

        import pandas as pd

        from quant_platform.historical.locking import LockInfo

        lock_path = self._lock_path(cli_environment)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fresh = LockInfo(pid=999_999, hostname="cli-subprocess-test-host", acquired_at=pd.Timestamp.now(tz="UTC"))
        lock_path.write_text(json_module.dumps(fresh.to_json_dict()))
        try:
            result = _run_cli("recover-backtest-lock", "--config", str(cli_environment["config_path"]), "--backtest-id", cli_environment["expected_backtest_id"])
            assert result.returncode == 1
            _assert_no_traceback(result)
            assert "refusing to steal" in result.stderr
            assert lock_path.is_file()
        finally:
            lock_path.unlink(missing_ok=True)

    def test_recover_lock_reclaims_a_stale_abandoned_lock_and_inspect_confirms_it(self, cli_environment: dict) -> None:
        import json as json_module

        import pandas as pd

        from quant_platform.historical.locking import LockInfo

        lock_path = self._lock_path(cli_environment)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        stale = LockInfo(pid=999_999, hostname="cli-subprocess-test-host", acquired_at=pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2))
        lock_path.write_text(json_module.dumps(stale.to_json_dict()))

        result = _run_cli("recover-backtest-lock", "--config", str(cli_environment["config_path"]), "--backtest-id", cli_environment["expected_backtest_id"])
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        assert "is_stale_by_age: True" in result.stdout
        assert "recovered: yes" in result.stdout
        assert not lock_path.is_file()

        follow_up = _run_cli("inspect-backtest-lock", "--config", str(cli_environment["config_path"]), "--backtest-id", cli_environment["expected_backtest_id"])
        assert follow_up.returncode == 0, follow_up.stderr
        assert "lock_exists: False" in follow_up.stdout

    def test_recover_lock_with_force_reclaims_a_fresh_lock(self, cli_environment: dict) -> None:
        import json as json_module

        import pandas as pd

        from quant_platform.historical.locking import LockInfo

        lock_path = self._lock_path(cli_environment)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fresh = LockInfo(pid=999_999, hostname="cli-subprocess-test-host", acquired_at=pd.Timestamp.now(tz="UTC"))
        lock_path.write_text(json_module.dumps(fresh.to_json_dict()))

        result = _run_cli("recover-backtest-lock", "--config", str(cli_environment["config_path"]), "--backtest-id", cli_environment["expected_backtest_id"], "--force")
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        assert "recovered: yes" in result.stdout
        assert not lock_path.is_file()

    def test_failure_unknown_backtest_id(self, cli_environment: dict) -> None:
        result = _run_cli("resume-backtest", "--config", str(cli_environment["config_path"]), "--backtest-id", "c" * 64)
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr
