"""Unit tests for `portfolio_risk.replay`: replaying the identical
operation sequence into independent, fresh stores -- across different
filesystem roots, different operational (verification-time) labels,
different `PYTHONHASHSEED` values in separate OS processes, and no
dependence on real wall-clock time -- always yields an identical
`PortfolioRiskReplayResult`."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import PortfolioRiskVerificationError
from quant_platform.portfolio_risk.authorization import create_risk_authorization
from quant_platform.portfolio_risk.ledger import PortfolioRiskLedgerStore
from quant_platform.portfolio_risk.lifecycle import (
    consume_authorization,
    record_authorization_issuance,
    reserve_authorization,
)
from quant_platform.portfolio_risk.models import RiskDecisionKind
from quant_platform.portfolio_risk.replay import assert_replay_deterministic, compute_replay_result

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _authorization(**overrides: object):
    base: dict[str, object] = {
        "execution_intent_id": "1" * 64, "execution_session_id": "2" * 64, "portfolio_id": "p1", "portfolio_snapshot_id": "3" * 64,
        "price_snapshot_id": "4" * 64, "risk_policy_id": "5" * 64, "risk_decision_id": "6" * 64, "decision_kind": RiskDecisionKind.APPROVED,
        "evaluated_quantity": Decimal("1000"), "evaluated_price": Decimal("1.10"), "authorization_sequence": 0, "event_time": _T0,
    }
    base.update(overrides)
    return create_risk_authorization(**base)  # type: ignore[arg-type]


def _run_scenario(store: PortfolioRiskLedgerStore) -> None:
    authorization = _authorization()
    record_authorization_issuance(store, authorization, event_time=_T0)
    kwargs: dict[str, object] = {
        "execution_intent_id": authorization.execution_intent_id, "execution_session_id": authorization.execution_session_id,
        "portfolio_id": authorization.portfolio_id, "portfolio_snapshot_id": authorization.portfolio_snapshot_id,
        "price_snapshot_id": authorization.price_snapshot_id, "risk_policy_id": authorization.risk_policy_id,
        "quantity": authorization.evaluated_quantity, "price": authorization.evaluated_price, "consumption_identity": "use-1",
        "evaluation_time": _T0,
    }
    reserve_authorization(store, authorization, **kwargs)  # type: ignore[arg-type]
    consume_authorization(store, authorization, **kwargs)  # type: ignore[arg-type]


class TestFreshStoresProduceIdenticalResults:
    def test_two_independent_temp_roots_replay_identically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            store_a = PortfolioRiskLedgerStore(tmp_a)
            store_b = PortfolioRiskLedgerStore(tmp_b)
            _run_scenario(store_a)
            _run_scenario(store_b)
            result_a = compute_replay_result(portfolio_id="p1", store=store_a, verification_time=_T0)
            result_b = compute_replay_result(portfolio_id="p1", store=store_b, verification_time=_T0)
            assert_replay_deterministic(result_a, result_b)
            assert result_a == result_b

    def test_deeply_nested_vs_shallow_root_paths_replay_identically(self) -> None:
        # The storage root path itself is a purely operational detail --
        # its depth/shape must never leak into the economic outcome.
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            shallow_root = tmp_a
            nested_root = tempfile.mkdtemp(dir=tmp_b)
            nested_root = tempfile.mkdtemp(dir=nested_root)
            store_a = PortfolioRiskLedgerStore(shallow_root)
            store_b = PortfolioRiskLedgerStore(nested_root)
            _run_scenario(store_a)
            _run_scenario(store_b)
            result_a = compute_replay_result(portfolio_id="p1", store=store_a, verification_time=_T0)
            result_b = compute_replay_result(portfolio_id="p1", store=store_b, verification_time=_T0)
            assert_replay_deterministic(result_a, result_b)


class TestOperationalLabelsDoNotAffectReplayResult:
    def test_a_different_verification_time_label_does_not_change_the_result(self) -> None:
        # `verification_time` is the one caller-supplied "operational label"
        # `compute_replay_result` accepts. Two wildly different values (one
        # historical, one far-future) must produce an identical result --
        # proving nothing about the ledger's own economic content depends
        # on when verification happens to be run.
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            store_a = PortfolioRiskLedgerStore(tmp_a)
            store_b = PortfolioRiskLedgerStore(tmp_b)
            _run_scenario(store_a)
            _run_scenario(store_b)
            result_a = compute_replay_result(portfolio_id="p1", store=store_a, verification_time=_T0)
            result_b = compute_replay_result(portfolio_id="p1", store=store_b, verification_time=_T0 + timedelta(days=3650))
            assert_replay_deterministic(result_a, result_b)


class TestNoWallClockDependence:
    def test_repeated_computation_on_the_same_store_never_changes(self) -> None:
        # Nothing in this module reads the real system clock -- calling it
        # twice, with real time having elapsed between calls, must yield
        # byte-identical results both times.
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioRiskLedgerStore(tmp)
            _run_scenario(store)
            first = compute_replay_result(portfolio_id="p1", store=store, verification_time=_T0)
            second = compute_replay_result(portfolio_id="p1", store=store, verification_time=_T0)
            assert first == second


class TestReplayDivergenceIsDetected:
    def test_a_genuinely_different_ledger_raises_on_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            store_a = PortfolioRiskLedgerStore(tmp_a)
            store_b = PortfolioRiskLedgerStore(tmp_b)
            _run_scenario(store_a)
            _run_scenario(store_b)
            # Only store_b receives a SECOND, distinct authorization -- an
            # actual divergence, not merely a different label.
            second = _authorization(execution_intent_id="7" * 64, risk_decision_id="8" * 64)
            record_authorization_issuance(store_b, second, event_time=_T0)

            result_a = compute_replay_result(portfolio_id="p1", store=store_a, verification_time=_T0)
            result_b = compute_replay_result(portfolio_id="p1", store=store_b, verification_time=_T0)
            with pytest.raises(PortfolioRiskVerificationError):
                assert_replay_deterministic(result_a, result_b)


_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import sys
    from datetime import datetime, timezone
    from decimal import Decimal

    from quant_platform.portfolio_risk.authorization import create_risk_authorization
    from quant_platform.portfolio_risk.ledger import PortfolioRiskLedgerStore
    from quant_platform.portfolio_risk.lifecycle import consume_authorization, record_authorization_issuance, reserve_authorization
    from quant_platform.portfolio_risk.models import RiskDecisionKind
    from quant_platform.portfolio_risk.replay import compute_replay_result

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    authorization = create_risk_authorization(
        execution_intent_id="1" * 64, execution_session_id="2" * 64, portfolio_id="p1", portfolio_snapshot_id="3" * 64,
        price_snapshot_id="4" * 64, risk_policy_id="5" * 64, risk_decision_id="6" * 64, decision_kind=RiskDecisionKind.APPROVED,
        evaluated_quantity=Decimal("1000"), evaluated_price=Decimal("1.10"), authorization_sequence=0, event_time=t0,
    )
    store = PortfolioRiskLedgerStore(sys.argv[1])
    record_authorization_issuance(store, authorization, event_time=t0)
    kwargs = dict(
        execution_intent_id=authorization.execution_intent_id, execution_session_id=authorization.execution_session_id,
        portfolio_id=authorization.portfolio_id, portfolio_snapshot_id=authorization.portfolio_snapshot_id,
        price_snapshot_id=authorization.price_snapshot_id, risk_policy_id=authorization.risk_policy_id,
        quantity=authorization.evaluated_quantity, price=authorization.evaluated_price, consumption_identity="use-1",
        evaluation_time=t0,
    )
    reserve_authorization(store, authorization, **kwargs)
    consume_authorization(store, authorization, **kwargs)
    result = compute_replay_result(portfolio_id="p1", store=store, verification_time=t0)
    print(result.semantic_digest)
    print(",".join(result.authorization_ids))
    """
)


def _run_in_subprocess(root: str, *, hashseed: str) -> tuple[str, str]:
    import os

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    completed = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT, root], env=env, capture_output=True, text=True, timeout=60, check=True,
    )
    lines = completed.stdout.strip().splitlines()
    return lines[0], lines[1]


class TestPythonHashSeedIndependence:
    def test_semantic_digest_is_stable_across_separate_processes_with_different_hash_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            digest_a, ids_a = _run_in_subprocess(tmp_a, hashseed="0")
            digest_b, ids_b = _run_in_subprocess(tmp_b, hashseed="4294967295")
            assert digest_a == digest_b
            assert ids_a == ids_b

    def test_subprocess_digest_matches_in_process_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            digest_subprocess, _ = _run_in_subprocess(tmp_a, hashseed="12345")
            store = PortfolioRiskLedgerStore(tmp_b)
            _run_scenario(store)
            result_in_process = compute_replay_result(portfolio_id="p1", store=store, verification_time=_T0)
            assert digest_subprocess == result_in_process.semantic_digest
