"""Milestone 8, Section 32/34: exercises the 16 `quant_platform.ml_cli`
execution-gateway commands via REAL `subprocess.run` OS process launches --
mirrors `test_paper_trading_cli_subprocess.py`'s own convention and
reasoning exactly.

SCOPE: `run-dummy-execution-session`/`resume-execution-session` themselves
are NOT exercised end-to-end via subprocess here -- doing so needs a REAL,
independently-verified Milestone 6/7 chain (a promoted candidate and a
COMPLETED, verified paper session), which is expensive to construct and is
already exercised once, in-process, by
`test_execution_gateway_acceptance.py`. This file instead proves: the
dry-run `create-execution-gateway-spec` path (needs no real lookups --
`ExecutionGatewayConfigSchema.build()` performs no artifact loading at
all); every command's clean, non-traceback, correctly-coded failure path
against config/id inputs that legitimately do not exist; `--help`
enumerates all 16 commands and no forbidden live-trading command exists;
inspection row-limit enforcement; and, separately, a genuinely clean,
non-editable install of this project into an isolated virtual environment
(Section 32's own explicit requirement) followed by the same wiring/error-
path smoke checks run against THAT venv's own interpreter -- proving the
CLI works when installed for real, not merely importable from this
repository's editable checkout."""

from __future__ import annotations

import json
import subprocess
import sys
import venv
from pathlib import Path

import pytest

_TIMEOUT = 60
_INSTALL_TIMEOUT = 600
_HEX_A = "a" * 64
_HEX_B = "b" * 64
_HEX_C = "c" * 64
_HEX_D = "d" * 64
_UNKNOWN_EXECUTION_SESSION_ID = "f" * 64
_MISMATCHED_EXECUTION_SESSION_ID = "7" * 64

_ALL_SIXTEEN_COMMANDS = (
    "create-execution-gateway-spec", "run-dummy-execution-session", "resume-execution-session", "pause-execution-session",
    "inspect-execution-session", "inspect-execution-intents", "inspect-execution-commands", "inspect-execution-orders",
    "inspect-execution-fills", "inspect-broker-events", "inspect-execution-health", "inspect-execution-reconciliation",
    "report-execution-session", "verify-execution-session", "replay-execution-session", "compare-execution-to-paper",
)
_FORBIDDEN_COMMAND_NAMES = ("run-live", "submit-live-order", "connect-mt5", "connect-broker", "deploy-live", "execute-mt5")


def _run_cli(*args: str, python: str | None = None) -> subprocess.CompletedProcess[str]:
    executable = python or sys.executable
    return subprocess.run([executable, "-m", "quant_platform.ml_cli", *args], capture_output=True, text=True, timeout=_TIMEOUT)


def _assert_no_traceback(result: subprocess.CompletedProcess[str]) -> None:
    assert "Traceback (most recent call last)" not in result.stdout
    assert "Traceback (most recent call last)" not in result.stderr


def _assert_no_nan_or_infinity(text: str) -> None:
    assert "NaN" not in text
    assert "Infinity" not in text


def _minimal_config_dict(ml_artifacts_root: Path) -> dict:
    return {
        "ml_artifacts_root": str(ml_artifacts_root), "research_storage_root": str(ml_artifacts_root), "historical_storage_root": str(ml_artifacts_root),
        "execution_mode": "test_only", "adapter_kind": "deterministic_dummy",
        "paper_session_id": _HEX_A, "paper_trading_spec_id": _HEX_B, "promotion_decision_id": _HEX_C, "instrument_spec_id": _HEX_D,
    }


@pytest.fixture(scope="module")
def cli_environment(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp_path = tmp_path_factory.mktemp("execution_gateway_cli_subprocess")
    ml_artifacts_root = tmp_path / "ml_artifacts"
    config_path = tmp_path / "execution_gateway_config.json"
    config_path.write_text(json.dumps(_minimal_config_dict(ml_artifacts_root)), encoding="utf-8")

    from quant_platform.config.execution_gateway_schemas import ExecutionGatewayConfigSchema
    from quant_platform.execution_gateway.specs import compute_execution_gateway_spec_id

    resolved = ExecutionGatewayConfigSchema.model_validate_json(config_path.read_text()).build()
    expected_execution_session_id = compute_execution_gateway_spec_id(resolved).execution_gateway_spec_id

    missing_replay_source = tmp_path / "does_not_exist.jsonl"

    return {
        "config_path": config_path, "ml_artifacts_root": ml_artifacts_root, "expected_execution_session_id": expected_execution_session_id,
        "missing_replay_source": missing_replay_source, "tmp_path": tmp_path,
    }


class TestHelpEnumeratesAllExecutionGatewayCommands:
    def test_help_lists_all_sixteen_commands(self) -> None:
        result = _run_cli("--help")
        assert result.returncode == 0
        for command in _ALL_SIXTEEN_COMMANDS:
            assert command in result.stdout, f"{command!r} missing from --help output"

    def test_help_contains_no_live_trading_command(self) -> None:
        result = _run_cli("--help")
        for forbidden in _FORBIDDEN_COMMAND_NAMES:
            assert forbidden not in result.stdout


class TestCreateExecutionGatewaySpec:
    def test_success_prints_the_expected_deterministic_execution_session_id(self, cli_environment: dict) -> None:
        result = _run_cli("create-execution-gateway-spec", "--config", str(cli_environment["config_path"]))
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        _assert_no_nan_or_infinity(result.stdout)
        assert f"execution_gateway_spec_id: {cli_environment['expected_execution_session_id']}" in result.stdout
        assert "execution_mode: test_only" in result.stdout
        assert "adapter_kind: deterministic_dummy" in result.stdout
        assert "No order is ever sent to any real broker" in result.stdout

    def test_failure_missing_config_file(self, tmp_path: Path) -> None:
        result = _run_cli("create-execution-gateway-spec", "--config", str(tmp_path / "does_not_exist.json"))
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_failure_malformed_config_json(self, tmp_path: Path) -> None:
        bad_config_path = tmp_path / "malformed.json"
        bad_config_path.write_text("{not valid json", encoding="utf-8")
        result = _run_cli("create-execution-gateway-spec", "--config", str(bad_config_path))
        assert result.returncode == 1
        _assert_no_traceback(result)

    def test_failure_invalid_identity_hex(self, tmp_path: Path, cli_environment: dict) -> None:
        bad_config = _minimal_config_dict(cli_environment["ml_artifacts_root"])
        bad_config["paper_session_id"] = "not_a_valid_hex_digest"
        bad_config_path = tmp_path / "bad_identity.json"
        bad_config_path.write_text(json.dumps(bad_config), encoding="utf-8")
        result = _run_cli("create-execution-gateway-spec", "--config", str(bad_config_path))
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_failure_unknown_field_rejected(self, tmp_path: Path, cli_environment: dict) -> None:
        """`extra="forbid"` (Section 4/29) -- a broker-credential-shaped
        or otherwise unknown field must be rejected, never silently
        ignored."""
        bad_config = _minimal_config_dict(cli_environment["ml_artifacts_root"])
        bad_config["api_key"] = "sk-should-not-exist"
        bad_config_path = tmp_path / "credential_shaped.json"
        bad_config_path.write_text(json.dumps(bad_config), encoding="utf-8")
        result = _run_cli("create-execution-gateway-spec", "--config", str(bad_config_path))
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr


class TestUnknownExecutionSessionIdFailsCleanlyAcrossEveryInspectionCommand:
    """Every inspection/verification/reconciliation command must fail with
    a clean, actionable, non-traceback error (never a raw Python stack)
    when asked about an `--execution-session-id` that legitimately does
    not exist."""

    @pytest.mark.parametrize(
        "command",
        [
            "inspect-execution-session", "inspect-execution-intents", "inspect-execution-commands", "inspect-execution-orders",
            "inspect-execution-fills", "inspect-broker-events", "inspect-execution-health", "inspect-execution-reconciliation",
            "report-execution-session", "verify-execution-session", "pause-execution-session",
        ],
    )
    def test_unknown_session_id(self, cli_environment: dict, command: str) -> None:
        result = _run_cli(command, "--config", str(cli_environment["config_path"]), "--execution-session-id", _UNKNOWN_EXECUTION_SESSION_ID)
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_compare_execution_to_paper_unknown_session_id(self, cli_environment: dict) -> None:
        result = _run_cli(
            "compare-execution-to-paper", "--config", str(cli_environment["config_path"]), "--execution-session-id", _UNKNOWN_EXECUTION_SESSION_ID,
            "--paper-session-id", _HEX_A,
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr


class TestRunCommandsFailCleanlyOnMissingReplaySource:
    """`run-dummy-execution-session`/`resume-execution-session`/`replay-
    execution-session` all load `--replay-source` before touching the
    (expensive, not exercised here) real paper-bridge resolution path -- a
    missing file must fail cleanly, never with a traceback."""

    def test_run_dummy_execution_session(self, cli_environment: dict) -> None:
        result = _run_cli(
            "run-dummy-execution-session", "--config", str(cli_environment["config_path"]),
            "--replay-source", str(cli_environment["missing_replay_source"]),
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_resume_execution_session_nonexistent(self, cli_environment: dict) -> None:
        result = _run_cli(
            "resume-execution-session", "--config", str(cli_environment["config_path"]), "--execution-session-id", _UNKNOWN_EXECUTION_SESSION_ID,
            "--replay-source", str(cli_environment["missing_replay_source"]),
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_replay_execution_session(self, cli_environment: dict, tmp_path: Path) -> None:
        result = _run_cli(
            "replay-execution-session", "--config", str(cli_environment["config_path"]),
            "--replay-source", str(cli_environment["missing_replay_source"]), "--replay-storage-root", str(tmp_path / "replay_root"),
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr


class TestResumeRefusesMismatchedSessionIdentity:
    """Mirrors the Milestone 7 release-audit's `resume-paper-session`
    identity-binding fix: `resume-execution-session` must refuse to
    operate on a `--execution-session-id` that does not match what
    `--config` resolves to, before ever touching `--replay-source`."""

    def test_resume_with_mismatched_execution_session_id_fails_cleanly(self, cli_environment: dict) -> None:
        from quant_platform.execution_gateway.manifests import ExecutionSessionManifestStore
        from quant_platform.execution_gateway.models import ExecutionMode

        assert cli_environment["expected_execution_session_id"] != _MISMATCHED_EXECUTION_SESSION_ID
        ExecutionSessionManifestStore(cli_environment["ml_artifacts_root"]).create(
            execution_session_id=_MISMATCHED_EXECUTION_SESSION_ID, execution_gateway_spec_id=_MISMATCHED_EXECUTION_SESSION_ID,
            paper_session_id=_HEX_A, adapter_id="dummy-broker-1", execution_mode=ExecutionMode.TEST_ONLY,
        )

        result = _run_cli(
            "resume-execution-session", "--config", str(cli_environment["config_path"]), "--execution-session-id", _MISMATCHED_EXECUTION_SESSION_ID,
            "--replay-source", str(cli_environment["missing_replay_source"]),
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr
        assert _MISMATCHED_EXECUTION_SESSION_ID in result.stderr
        assert cli_environment["expected_execution_session_id"] in result.stderr


def _seed_ledger_with_n_intents(cli_environment: dict, execution_session_id: str, *, intent_count: int) -> None:
    from datetime import datetime, timezone

    from quant_platform.execution_gateway.manifests import ExecutionSessionManifestStore
    from quant_platform.execution_gateway.models import ExecutionLedgerEntryKind, ExecutionMode
    from quant_platform.execution_gateway.persistence import (
        ExecutionSessionEventStore,
        create_execution_ledger_entry,
    )

    manifest_store = ExecutionSessionManifestStore(cli_environment["ml_artifacts_root"])
    event_store = ExecutionSessionEventStore(cli_environment["ml_artifacts_root"])
    manifest_store.create(
        execution_session_id=execution_session_id, execution_gateway_spec_id=execution_session_id, paper_session_id=_HEX_A,
        adapter_id="dummy-broker-1", execution_mode=ExecutionMode.TEST_ONLY,
    )
    event_time = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
    previous_hash = None
    for seq in range(intent_count):
        payload = {"execution_intent_id": f"{seq:064x}", "instrument_id": "X", "side": "buy", "quantity": "1"}
        entry = create_execution_ledger_entry(
            execution_session_id=execution_session_id, entry_sequence=seq, entry_kind=ExecutionLedgerEntryKind.EXECUTION_INTENT_ACCEPTED,
            payload=payload, event_time=event_time, previous_entry_hash=previous_hash,
        )
        persisted = event_store.append(execution_session_id, entry)
        previous_hash = persisted.entry_id


class TestInspectionRowLimitsEnforced:
    """Mirrors the Milestone 7 release-audit's row-limit fix -- every
    inspection command must cap its output at `--limit` (default 200),
    never print an unbounded number of rows."""

    def test_inspect_execution_intents_truncates_at_the_requested_limit(self, cli_environment: dict) -> None:
        execution_session_id = "1" * 64
        _seed_ledger_with_n_intents(cli_environment, execution_session_id, intent_count=10)

        result = _run_cli(
            "inspect-execution-intents", "--config", str(cli_environment["config_path"]), "--execution-session-id", execution_session_id, "--limit", "3",
        )
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        assert result.stdout.count("execution_intent_id=") == 3
        assert "7 additional intent(s) not shown" in result.stdout

    def test_inspect_execution_intents_negative_limit_fails_cleanly(self, cli_environment: dict) -> None:
        execution_session_id = "2" * 64
        _seed_ledger_with_n_intents(cli_environment, execution_session_id, intent_count=1)

        result = _run_cli(
            "inspect-execution-intents", "--config", str(cli_environment["config_path"]), "--execution-session-id", execution_session_id, "--limit", "-1",
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr


class TestNoLiveTradingLanguageAnywhereInOutput:
    def test_every_command_help_is_free_of_live_claims(self) -> None:
        for command in _ALL_SIXTEEN_COMMANDS:
            result = _run_cli(command, "--help")
            assert result.returncode == 0
            lowered = result.stdout.lower()
            for forbidden_phrase in ("live trading", "real broker", "real money", "mt5", "fxpro"):
                assert forbidden_phrase not in lowered, f"{command} --help unexpectedly mentions {forbidden_phrase!r}"


# --------------------------------------------------------------------------
# Section 32: clean-install test. A genuinely isolated virtual environment
# (confirmed via `pyvenv.cfg`'s own `include-system-site-packages = false`),
# a NON-EDITABLE `pip install .` of this project into it, then the same
# wiring/error-path smoke checks run against THAT interpreter -- proving
# the CLI genuinely works once installed, not merely importable from this
# repository's own editable checkout. Scoped `class` so the (expensive)
# venv creation and install happen exactly once for every test in this
# class, mirroring `cli_environment`'s own module-scoped-fixture reuse
# above.
# --------------------------------------------------------------------------


class TestCleanVenvInstall:
    @staticmethod
    @pytest.fixture(scope="class")
    def clean_venv_python(tmp_path_factory: pytest.TempPathFactory) -> str:
        venv_dir = tmp_path_factory.mktemp("execution_gateway_clean_venv") / "venv"
        venv.EnvBuilder(with_pip=True).create(venv_dir)

        cfg_text = (venv_dir / "pyvenv.cfg").read_text(encoding="utf-8")
        assert "include-system-site-packages = false" in cfg_text, f"clean venv is not isolated from system site-packages: {cfg_text}"

        python = venv_dir / "Scripts" / "python.exe" if sys.platform == "win32" else venv_dir / "bin" / "python"
        assert python.is_file(), f"expected venv interpreter not found at {python}"

        project_root = Path(__file__).resolve().parents[2]
        assert (project_root / "pyproject.toml").is_file(), f"expected project root (with pyproject.toml) at {project_root}"

        result = subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", "--no-input", str(project_root)],
            capture_output=True, text=True, timeout=_INSTALL_TIMEOUT,
        )
        assert result.returncode == 0, f"non-editable install into clean venv failed:\nstdout={result.stdout}\nstderr={result.stderr}"

        show_result = subprocess.run([str(python), "-m", "pip", "show", "quant-platform"], capture_output=True, text=True, timeout=_TIMEOUT)
        assert show_result.returncode == 0
        assert str(venv_dir) in show_result.stdout, "installed package Location does not point inside the clean venv -- install did not land where expected"

        return str(python)

    def test_help_lists_all_sixteen_commands(self, clean_venv_python: str) -> None:
        result = _run_cli("--help", python=clean_venv_python)
        assert result.returncode == 0, result.stderr
        for command in _ALL_SIXTEEN_COMMANDS:
            assert command in result.stdout, f"{command!r} missing from --help output in the clean venv"

    def test_help_contains_no_live_trading_command(self, clean_venv_python: str) -> None:
        result = _run_cli("--help", python=clean_venv_python)
        for forbidden in _FORBIDDEN_COMMAND_NAMES:
            assert forbidden not in result.stdout

    def test_create_execution_gateway_spec_succeeds(self, clean_venv_python: str, cli_environment: dict) -> None:
        result = _run_cli("create-execution-gateway-spec", "--config", str(cli_environment["config_path"]), python=clean_venv_python)
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        assert f"execution_gateway_spec_id: {cli_environment['expected_execution_session_id']}" in result.stdout
        assert "No order is ever sent to any real broker" in result.stdout

    def test_inspect_execution_session_unknown_id_fails_cleanly(self, clean_venv_python: str, cli_environment: dict) -> None:
        result = _run_cli(
            "inspect-execution-session", "--config", str(cli_environment["config_path"]), "--execution-session-id", _UNKNOWN_EXECUTION_SESSION_ID,
            python=clean_venv_python,
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_verify_execution_session_unknown_id_fails_cleanly(self, clean_venv_python: str, cli_environment: dict) -> None:
        result = _run_cli(
            "verify-execution-session", "--config", str(cli_environment["config_path"]), "--execution-session-id", _UNKNOWN_EXECUTION_SESSION_ID,
            python=clean_venv_python,
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_report_execution_session_unknown_id_fails_cleanly(self, clean_venv_python: str, cli_environment: dict) -> None:
        result = _run_cli(
            "report-execution-session", "--config", str(cli_environment["config_path"]), "--execution-session-id", _UNKNOWN_EXECUTION_SESSION_ID,
            python=clean_venv_python,
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_run_dummy_execution_session_missing_replay_source_fails_cleanly(self, clean_venv_python: str, cli_environment: dict) -> None:
        result = _run_cli(
            "run-dummy-execution-session", "--config", str(cli_environment["config_path"]),
            "--replay-source", str(cli_environment["missing_replay_source"]), python=clean_venv_python,
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_negative_limit_fails_cleanly(self, clean_venv_python: str, cli_environment: dict) -> None:
        result = _run_cli(
            "inspect-execution-intents", "--config", str(cli_environment["config_path"]), "--execution-session-id", _UNKNOWN_EXECUTION_SESSION_ID,
            "--limit", "-1", python=clean_venv_python,
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr
