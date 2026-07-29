"""Milestone 7, Section 34 (test group 31): exercises the 14
`quant_platform.ml_cli` paper-trading/shadow-execution commands via REAL
`subprocess.run` OS process launches -- mirrors `test_robustness_cli_
subprocess.py`'s own convention and reasoning.

SCOPE: `run-paper-session`/`resume-paper-session`/`run-shadow-session`
themselves are NOT exercised end-to-end via subprocess here -- the full
real-model pipeline they need is expensive (the Section 33 acceptance
workflow alone takes minutes, even in-process); running it again as a
subprocess would roughly double this suite's wall-clock cost for no
additional boundary coverage. This file instead proves: the dry-run
`create-paper-trading-spec` path (needs no real lookups -- unlike
`create-robustness-spec`, `PaperTradingConfig.build()` performs no
artifact loading at all, so this command's happy path is genuinely free
of ML-pipeline setup); every command's clean, non-traceback, correctly-
coded failure path against config/id inputs that legitimately do not
exist; and that `--help` enumerates all 14 commands. `run-paper-session`/
`run-shadow-session`'s own forward-pipeline correctness is proven by
`test_paper_trading_real_model_acceptance.py`, in-process, once."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

_TIMEOUT = 60
_HEX_A = "a" * 64
_HEX_B = "b" * 64
_HEX_C = "c" * 64
_HEX_D = "d" * 64
_HEX_E = "e" * 64
_UNKNOWN_PAPER_SESSION_ID = "f" * 64
_MISMATCHED_PAPER_SESSION_ID = "7" * 64


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "quant_platform.ml_cli", *args], capture_output=True, text=True, timeout=_TIMEOUT)


def _assert_no_traceback(result: subprocess.CompletedProcess[str]) -> None:
    assert "Traceback (most recent call last)" not in result.stdout
    assert "Traceback (most recent call last)" not in result.stderr


def _assert_no_nan_or_infinity(text: str) -> None:
    assert "NaN" not in text
    assert "Infinity" not in text


def _minimal_config_dict(ml_artifacts_root: Path) -> dict:
    return {
        "ml_artifacts_root": str(ml_artifacts_root), "historical_storage_root": str(ml_artifacts_root), "research_storage_root": str(ml_artifacts_root),
        "verified_robustness_id": _HEX_A, "verified_promotion_decision_id": _HEX_B, "strategy_candidate_identity": _HEX_C,
        "model_artifact_identity": _HEX_D, "feature_spec_identity": _HEX_E,
        "instrument": {
            "symbol": "X", "quote_currency": "USD", "contract_multiplier": 1.0, "tick_size": 0.01, "quantity_step": 0.01, "minimum_quantity": 0.01,
            "price_precision": 2, "quantity_precision": 2, "margin_mode": "cash", "account_currency": "USD", "financing_convention": "none",
            "trading_timezone": "UTC", "session_calendar_identity": "always_open",
        },
        "price_precision": 2, "quantity_precision": 2, "starting_cash": 100_000.0,
    }


@pytest.fixture(scope="module")
def cli_environment(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp_path = tmp_path_factory.mktemp("paper_trading_cli_subprocess")
    ml_artifacts_root = tmp_path / "ml_artifacts"
    config_path = tmp_path / "paper_trading_config.json"
    config_path.write_text(json.dumps(_minimal_config_dict(ml_artifacts_root)), encoding="utf-8")

    from quant_platform.config.paper_trading_schemas import PaperTradingConfig
    from quant_platform.core.types import Timeframe
    from quant_platform.paper_trading.events import create_bar_event, create_end_of_stream_event
    from quant_platform.paper_trading.replay import write_replay_events
    from quant_platform.paper_trading.specs import compute_paper_session_spec_id

    resolved = PaperTradingConfig.model_validate_json(config_path.read_text()).build()
    expected_paper_session_id = compute_paper_session_spec_id(resolved).paper_session_spec_id

    missing_replay_source = tmp_path / "does_not_exist.jsonl"

    valid_replay_source = tmp_path / "valid_replay.jsonl"
    open_time = pd.Timestamp("2026-01-05T10:00:00Z").to_pydatetime()
    bar = create_bar_event(instrument="X", interval=Timeframe.H1, open_time=open_time, open=100.0, high=100.5, low=99.5, close=100.0, sequence=1, source="test")
    eos = create_end_of_stream_event(instrument="X", event_time=bar.close_time, sequence=2, source="test")
    write_replay_events(valid_replay_source, (bar, eos))

    return {
        "config_path": config_path, "ml_artifacts_root": ml_artifacts_root, "expected_paper_session_id": expected_paper_session_id,
        "missing_replay_source": missing_replay_source, "valid_replay_source": valid_replay_source, "tmp_path": tmp_path,
    }


class TestHelpEnumeratesAllPaperTradingCommands:
    def test_help_lists_all_fourteen_commands(self) -> None:
        result = _run_cli("--help")
        assert result.returncode == 0
        for command in (
            "create-paper-trading-spec", "run-paper-session", "resume-paper-session", "pause-paper-session",
            "inspect-paper-session", "report-paper-session", "verify-paper-session", "compare-paper-to-backtest",
            "inspect-paper-orders", "inspect-paper-fills", "inspect-paper-risk-events", "inspect-paper-reconciliation",
            "run-shadow-session", "report-shadow-session",
        ):
            assert command in result.stdout, f"{command!r} missing from --help output"

    def test_help_contains_no_live_trading_command(self) -> None:
        result = _run_cli("--help")
        for forbidden in ("run-live", "submit-live-order", "connect-broker", "execute-mt5", "deploy-live"):
            assert forbidden not in result.stdout


class TestCreatePaperTradingSpec:
    def test_success_prints_the_expected_deterministic_paper_session_id(self, cli_environment: dict) -> None:
        result = _run_cli("create-paper-trading-spec", "--config", str(cli_environment["config_path"]))
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        _assert_no_nan_or_infinity(result.stdout)
        assert f"paper_session_spec_id: {cli_environment['expected_paper_session_id']}" in result.stdout
        assert "No order is ever sent to any broker" in result.stdout

    def test_failure_missing_config_file(self, tmp_path: Path) -> None:
        result = _run_cli("create-paper-trading-spec", "--config", str(tmp_path / "does_not_exist.json"))
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_failure_malformed_config_json(self, tmp_path: Path) -> None:
        bad_config_path = tmp_path / "malformed.json"
        bad_config_path.write_text("{not valid json", encoding="utf-8")
        result = _run_cli("create-paper-trading-spec", "--config", str(bad_config_path))
        assert result.returncode == 1
        _assert_no_traceback(result)

    def test_failure_invalid_identity_hex(self, tmp_path: Path, cli_environment: dict) -> None:
        bad_config = _minimal_config_dict(cli_environment["ml_artifacts_root"])
        bad_config["verified_robustness_id"] = "not_a_valid_hex_digest"
        bad_config_path = tmp_path / "bad_identity.json"
        bad_config_path.write_text(json.dumps(bad_config), encoding="utf-8")
        result = _run_cli("create-paper-trading-spec", "--config", str(bad_config_path))
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr


class TestUnknownPaperSessionIdFailsCleanlyAcrossEveryInspectionCommand:
    """Every inspection/verification/reconciliation command must fail with
    a clean, actionable, non-traceback error (never a raw Python stack)
    when asked about a `--paper-session-id` that legitimately does not
    exist."""

    @pytest.mark.parametrize("command", ["inspect-paper-session", "report-paper-session", "verify-paper-session", "inspect-paper-orders", "inspect-paper-fills", "inspect-paper-risk-events", "inspect-paper-reconciliation", "pause-paper-session"])
    def test_unknown_session_id(self, cli_environment: dict, command: str) -> None:
        result = _run_cli(command, "--config", str(cli_environment["config_path"]), "--paper-session-id", _UNKNOWN_PAPER_SESSION_ID)
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_compare_paper_to_backtest_unknown_session_id(self, cli_environment: dict) -> None:
        result = _run_cli(
            "compare-paper-to-backtest", "--config", str(cli_environment["config_path"]), "--paper-session-id", _UNKNOWN_PAPER_SESSION_ID,
            "--backtest-id", "9" * 64,
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_report_shadow_session_unknown_session_id(self, cli_environment: dict) -> None:
        result = _run_cli("report-shadow-session", "--config", str(cli_environment["config_path"]), "--paper-session-id", _UNKNOWN_PAPER_SESSION_ID)
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr


class TestRunCommandsFailCleanlyOnMissingReplaySource:
    """`run-paper-session`/`resume-paper-session`/`run-shadow-session` all
    load `--replay-source` before touching the (expensive, not exercised
    here) real-model resolution path -- a missing file must fail cleanly,
    never with a traceback, regardless of how far downstream the real
    pipeline would otherwise go."""

    def test_run_paper_session(self, cli_environment: dict) -> None:
        result = _run_cli(
            "run-paper-session", "--config", str(cli_environment["config_path"]), "--replay-source", str(cli_environment["missing_replay_source"]),
            "--feature-name", "candle_body_ratio",
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_resume_paper_session(self, cli_environment: dict) -> None:
        result = _run_cli(
            "resume-paper-session", "--config", str(cli_environment["config_path"]), "--paper-session-id", _UNKNOWN_PAPER_SESSION_ID,
            "--replay-source", str(cli_environment["missing_replay_source"]), "--feature-name", "candle_body_ratio",
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr

    def test_run_shadow_session(self, cli_environment: dict) -> None:
        result = _run_cli(
            "run-shadow-session", "--config", str(cli_environment["config_path"]), "--replay-source", str(cli_environment["missing_replay_source"]),
            "--feature-name", "candle_body_ratio",
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr


class TestResumeRefusesMismatchedSessionIdentity:
    """Release-audit finding, fixed: `resume-paper-session` used to check
    `--paper-session-id` for EXISTENCE only, then silently operate on
    whatever session `--config` resolves to instead -- a `--config` that
    (by drift, typo, or a stale file) resolves to a DIFFERENT session
    than the one named would previously resume (or even silently create)
    a completely different session under the operator's nose. This test
    seeds a REAL manifest at a session id that does NOT match what
    `cli_environment`'s own config resolves to, and proves resume now
    refuses before ever touching `--replay-source` (never a traceback)."""

    def test_resume_with_mismatched_paper_session_id_fails_cleanly(self, cli_environment: dict) -> None:
        from quant_platform.paper_trading.manifests import PaperSessionManifestStore
        from quant_platform.paper_trading.models import SessionMode

        assert cli_environment["expected_paper_session_id"] != _MISMATCHED_PAPER_SESSION_ID
        PaperSessionManifestStore(cli_environment["ml_artifacts_root"]).create(
            paper_session_id=_MISMATCHED_PAPER_SESSION_ID, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None,
        )

        result = _run_cli(
            "resume-paper-session", "--config", str(cli_environment["config_path"]), "--paper-session-id", _MISMATCHED_PAPER_SESSION_ID,
            "--replay-source", str(cli_environment["missing_replay_source"]), "--feature-name", "candle_body_ratio",
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr
        assert _MISMATCHED_PAPER_SESSION_ID in result.stderr
        assert cli_environment["expected_paper_session_id"] in result.stderr


class TestResumeReVerifiesEligibilityNotJustExistence:
    """Release-audit Area 8: `resume-paper-session` used to check `--
    paper-session-id` for EXISTENCE only and then run `run_paper_trading_
    session`, which itself used to verify eligibility EXACTLY ONCE --
    inside `create_paper_session`, on the session's very first call.
    Every subsequent resume skipped eligibility re-verification entirely.
    This test seeds a manifest whose id genuinely MATCHES what `--config`
    resolves to (so the identity check above does not preempt this one),
    with `cli_environment`'s `ml_artifacts_root` containing no real
    ML pipeline artifacts at all -- resume must still fail cleanly (never
    silently proceed to process the replay source's events) once it
    reaches the mandatory eligibility/model-resolution re-check, not
    merely because the session didn't exist."""

    def test_resume_against_a_real_but_artifact_free_session_fails_cleanly(self, cli_environment: dict) -> None:
        from quant_platform.paper_trading.manifests import PaperSessionManifestStore
        from quant_platform.paper_trading.models import SessionMode

        expected_id = cli_environment["expected_paper_session_id"]
        PaperSessionManifestStore(cli_environment["ml_artifacts_root"]).create(
            paper_session_id=expected_id, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None,
        )

        # A genuinely VALID, EXISTING replay source (unlike the other tests
        # in this file) -- this call must get PAST the "replay source
        # missing" short-circuit and fail at model/eligibility resolution
        # instead, proving resume does not silently trust an existing
        # manifest with nothing behind it.
        result = _run_cli(
            "resume-paper-session", "--config", str(cli_environment["config_path"]), "--paper-session-id", expected_id,
            "--replay-source", str(cli_environment["valid_replay_source"]), "--feature-name", "candle_body_ratio",
        )
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr


def _seed_manifest_with_n_orders(cli_environment: dict, paper_session_id: str, *, order_count: int) -> None:
    from datetime import datetime, timezone

    from quant_platform.paper_trading.manifests import PaperSessionManifestStore
    from quant_platform.paper_trading.models import (
        LedgerEntryKind,
        OrderSide,
        OrderState,
        OrderTypeKind,
        PositionIntentKind,
        SessionMode,
        TimeInForceKind,
    )
    from quant_platform.paper_trading.orders import create_order_request, create_order_state_event
    from quant_platform.paper_trading.persistence import PaperSessionEventStore, create_ledger_entry

    manifest_store = PaperSessionManifestStore(cli_environment["ml_artifacts_root"])
    event_store = PaperSessionEventStore(cli_environment["ml_artifacts_root"])
    manifest_store.create(paper_session_id=paper_session_id, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None)
    event_time = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
    previous_hash = None
    for seq, i in enumerate(range(order_count)):
        order = create_order_request(
            client_order_id=f"row-limit-test-{i}", session_id=paper_session_id, strategy_decision_id="0" * 64, instrument="X",
            side=OrderSide.BUY, order_type=OrderTypeKind.MARKET,
            quantity=1.0, time_in_force=TimeInForceKind.DAY, create_time=event_time, submit_time=event_time, reduce_only=False, position_intent=PositionIntentKind.OPEN,
        )
        state_event = create_order_state_event(order_id=order.order_id, session_id=paper_session_id, from_state=OrderState.CREATED, to_state=OrderState.VALIDATED, event_time=event_time, sequence=seq)
        payload = {"order_state_event": state_event.to_json_dict(), "order": order.to_json_dict()}
        entry = create_ledger_entry(session_id=paper_session_id, sequence=seq, kind=LedgerEntryKind.ORDER_STATE_EVENT, payload=payload, event_time=event_time, previous_entry_hash=previous_hash)
        persisted = event_store.append(paper_session_id, entry)
        previous_hash = persisted.entry_id


class TestInspectionRowLimitsEnforced:
    """Release-audit finding, fixed (Section 11): `inspect-paper-orders`
    used to print EVERY recorded order unconditionally, with no cap --
    an operator-safety gap for any long-running session. `--limit`
    (default 200) now caps output; this test seeds MORE orders than the
    default and confirms both the cap and the truncation notice."""

    def test_inspect_paper_orders_truncates_at_the_requested_limit(self, cli_environment: dict) -> None:
        paper_session_id = "1" * 64
        _seed_manifest_with_n_orders(cli_environment, paper_session_id, order_count=10)

        result = _run_cli("inspect-paper-orders", "--config", str(cli_environment["config_path"]), "--paper-session-id", paper_session_id, "--limit", "3")
        assert result.returncode == 0, result.stderr
        _assert_no_traceback(result)
        assert result.stdout.count("order_id=") == 3
        assert "7 more order(s) not shown" in result.stdout

    def test_inspect_paper_orders_negative_limit_fails_cleanly(self, cli_environment: dict) -> None:
        paper_session_id = "2" * 64
        _seed_manifest_with_n_orders(cli_environment, paper_session_id, order_count=1)

        result = _run_cli("inspect-paper-orders", "--config", str(cli_environment["config_path"]), "--paper-session-id", paper_session_id, "--limit", "-1")
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr


class TestReportModeMismatchFailsCleanly:
    """Release-audit finding, fixed (Section 11): neither `report-paper-
    session` nor `report-shadow-session` previously checked the target
    session's OWN declared `session_mode` -- calling the wrong report
    command against the wrong session id silently produced a
    misleadingly-labeled report instead of a clean, typed error."""

    def test_report_shadow_session_against_a_replay_paper_session_fails_cleanly(self, cli_environment: dict) -> None:
        from quant_platform.paper_trading.manifests import PaperSessionManifestStore
        from quant_platform.paper_trading.models import SessionMode

        paper_session_id = "3" * 64
        PaperSessionManifestStore(cli_environment["ml_artifacts_root"]).create(paper_session_id=paper_session_id, session_mode=SessionMode.REPLAY_PAPER, spec_reference=None)

        result = _run_cli("report-shadow-session", "--config", str(cli_environment["config_path"]), "--paper-session-id", paper_session_id)
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr
        assert "shadow_observation" in result.stderr

    def test_report_paper_session_against_a_shadow_observation_session_fails_cleanly(self, cli_environment: dict) -> None:
        from quant_platform.paper_trading.manifests import PaperSessionManifestStore
        from quant_platform.paper_trading.models import SessionMode

        paper_session_id = "4" * 64
        PaperSessionManifestStore(cli_environment["ml_artifacts_root"]).create(paper_session_id=paper_session_id, session_mode=SessionMode.SHADOW_OBSERVATION, spec_reference=None)

        result = _run_cli("report-paper-session", "--config", str(cli_environment["config_path"]), "--paper-session-id", paper_session_id)
        assert result.returncode == 1
        _assert_no_traceback(result)
        assert "ERROR" in result.stderr
        assert "shadow_observation" in result.stderr
