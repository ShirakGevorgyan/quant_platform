"""Unit tests for `portfolio_risk.evaluator`: full integration
(APPROVED/DENIED/HALTED aggregation), determinism/identity, and the
adversarial scenarios Phase 2 requires."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import RiskEvaluationError
from quant_platform.portfolio_risk.decisions import create_risk_evaluation_request
from quant_platform.portfolio_risk.evaluator import _aggregate_decision_kind, evaluate_risk
from quant_platform.portfolio_risk.models import (
    OrderSide,
    RiskCheckSeverity,
    RiskDecisionKind,
    RiskDenialReason,
)
from quant_platform.portfolio_risk.snapshots import (
    PortfolioSnapshot,
    PositionSnapshot,
    create_portfolio_snapshot,
    create_price_snapshot,
)
from quant_platform.portfolio_risk.specs import (
    PortfolioRiskPolicy,
    PortfolioRiskSpec,
    compute_portfolio_risk_spec_id,
)

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_SHA_INTENT = "1" * 64
_SHA_SESSION = "2" * 64


def _policy(**overrides: object) -> PortfolioRiskPolicy:
    base: dict[str, object] = {
        "max_order_notional": Decimal("50000"), "max_position_notional": Decimal("50000"), "max_instrument_gross_exposure": Decimal("50000"),
        "max_strategy_gross_exposure": Decimal("50000"), "max_portfolio_gross_exposure": Decimal("100000"), "max_portfolio_net_exposure": Decimal("100000"),
        "max_concentration_fraction": Decimal("1"), "max_leverage": Decimal("5"), "max_daily_realized_loss": Decimal("10000"),
        "max_total_loss": Decimal("20000"), "max_drawdown_fraction": Decimal("0.5"), "max_consecutive_losses": 5, "minimum_cash_buffer": Decimal("1000"),
        "maximum_price_age": 60, "maximum_portfolio_snapshot_age": 60, "allow_reduce_only_during_halt": True,
    }
    base.update(overrides)
    return PortfolioRiskPolicy(**base)  # type: ignore[arg-type]


def _spec(**overrides: object) -> PortfolioRiskSpec:
    return PortfolioRiskSpec(schema_version=1, policy=_policy(**overrides))


def _price(**overrides: object):
    base: dict[str, object] = {
        "instrument_id": "EURUSD", "bid": Decimal("1.0995"), "ask": Decimal("1.1005"), "reference_price": Decimal("1.10"), "event_time": _T0,
        "source_event_id": "e1",
    }
    base.update(overrides)
    return create_price_snapshot(**base)  # type: ignore[arg-type]


def _portfolio(*, positions: tuple[PositionSnapshot, ...] = (), cash: Decimal = Decimal("100000"), **overrides: object) -> PortfolioSnapshot:
    marked_value = sum((p.market_value for p in positions), start=Decimal(0))
    unrealized = sum((p.unrealized_pnl for p in positions), start=Decimal(0))
    base: dict[str, object] = {
        "portfolio_id": "p1", "event_time": _T0, "cash": cash, "equity": cash + marked_value, "realized_pnl": Decimal("0"),
        "unrealized_pnl": unrealized, "peak_equity": cash + marked_value, "daily_start_equity": cash, "positions": positions,
        "source_execution_session_id": None,
    }
    base.update(overrides)
    return create_portfolio_snapshot(**base)  # type: ignore[arg-type]


def _request(*, portfolio: PortfolioSnapshot, price, spec: PortfolioRiskSpec, **overrides: object):
    base: dict[str, object] = {
        "execution_intent_id": _SHA_INTENT, "execution_session_id": _SHA_SESSION, "portfolio_id": portfolio.portfolio_id, "strategy_id": "s1",
        "instrument_id": price.instrument_id, "side": OrderSide.BUY, "quantity": Decimal("1000"), "portfolio_snapshot_id": portfolio.snapshot_id,
        "price_snapshot_id": price.price_snapshot_id, "risk_policy_id": compute_portfolio_risk_spec_id(spec).portfolio_risk_spec_id,
        "reduce_only": False, "requested_sequence": 0, "event_time": _T0,
    }
    base.update(overrides)
    return create_risk_evaluation_request(**base)  # type: ignore[arg-type]


def _evaluate(*, portfolio: PortfolioSnapshot, price, spec: PortfolioRiskSpec, request=None, **kwargs: object):
    req = request if request is not None else _request(portfolio=portfolio, price=price, spec=spec)
    base: dict[str, object] = {
        "request": req, "portfolio": portfolio, "price": price, "spec": spec, "evaluation_time": _T0, "portfolio_halted": False,
        "consecutive_losses": 0, "contract_multiplier": Decimal("1"), "decision_sequence": 0,
    }
    base.update(kwargs)
    return evaluate_risk(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Full integration: APPROVED / DENIED / HALTED aggregation
# --------------------------------------------------------------------------
class TestApprovedPath:
    def test_ordinary_order_within_all_limits_is_approved(self) -> None:
        portfolio = _portfolio()
        price = _price()
        spec = _spec()
        outcome = _evaluate(portfolio=portfolio, price=price, spec=spec)
        assert outcome.decision.kind is RiskDecisionKind.APPROVED
        assert outcome.decision.denial_reasons == ()
        assert len(outcome.decision.check_results) == 18
        assert all(c.passed for c in outcome.decision.check_results)

    def test_approved_produces_position_size_proposal_and_capital_allocation(self) -> None:
        portfolio = _portfolio()
        price = _price()
        spec = _spec()
        outcome = _evaluate(portfolio=portfolio, price=price, spec=spec)
        assert outcome.position_size_proposal is not None
        assert outcome.position_size_proposal.proposed_quantity == Decimal("1000")
        assert outcome.capital_allocation is not None
        assert outcome.capital_allocation.utilized_capital <= outcome.capital_allocation.allocated_capital


class TestDeniedPath:
    def test_order_notional_over_limit_is_denied(self) -> None:
        portfolio = _portfolio()
        price = _price()
        spec = _spec(max_order_notional=Decimal("100"))
        outcome = _evaluate(portfolio=portfolio, price=price, spec=spec)
        assert outcome.decision.kind is RiskDecisionKind.DENIED
        assert RiskDenialReason.ORDER_NOTIONAL_LIMIT_EXCEEDED in outcome.decision.denial_reasons
        assert outcome.position_size_proposal is None
        assert outcome.capital_allocation is None

    def test_denied_still_reports_all_eighteen_checks(self) -> None:
        portfolio = _portfolio()
        price = _price()
        spec = _spec(max_order_notional=Decimal("1"))
        outcome = _evaluate(portfolio=portfolio, price=price, spec=spec)
        assert len(outcome.decision.check_results) == 18


class TestHaltedPath:
    def test_pre_existing_halt_propagates(self) -> None:
        portfolio = _portfolio()
        price = _price()
        spec = _spec()
        outcome = _evaluate(portfolio=portfolio, price=price, spec=spec, portfolio_halted=True)
        assert outcome.decision.kind is RiskDecisionKind.HALTED
        assert RiskDenialReason.PORTFOLIO_HALTED in outcome.decision.denial_reasons

    def test_severe_drawdown_triggers_halt_not_mere_denial(self) -> None:
        portfolio = _portfolio(cash=Decimal("50000"), equity=Decimal("50000"), peak_equity=Decimal("100000"), daily_start_equity=Decimal("100000"))
        price = _price()
        spec = _spec(max_drawdown_fraction=Decimal("0.3"))
        outcome = _evaluate(portfolio=portfolio, price=price, spec=spec)
        assert outcome.decision.kind is RiskDecisionKind.HALTED
        assert RiskDenialReason.DRAWDOWN_LIMIT_EXCEEDED in outcome.decision.denial_reasons

    def test_halt_takes_priority_over_a_simultaneous_deny(self) -> None:
        # Both a DENY-severity breach (tiny order notional limit) and a
        # HALT-severity breach (halted portfolio) are present -- the
        # overall decision must be HALTED, the more severe of the two.
        portfolio = _portfolio()
        price = _price()
        spec = _spec(max_order_notional=Decimal("1"))
        outcome = _evaluate(portfolio=portfolio, price=price, spec=spec, portfolio_halted=True)
        assert outcome.decision.kind is RiskDecisionKind.HALTED


class TestReduceOnlyDuringHalt:
    def test_reduce_only_order_still_gets_evaluated_as_halted_when_allow_flag_is_false(self) -> None:
        position = PositionSnapshot(
            instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("1000"), average_entry_price=Decimal("1.10"),
            mark_price=Decimal("1.10"), unrealized_pnl=Decimal("0"), realized_pnl=Decimal("0"), contract_multiplier=Decimal("1"),
        )
        portfolio = _portfolio(positions=(position,), cash=Decimal("98900"))
        price = _price()
        spec = _spec(allow_reduce_only_during_halt=False)
        request = _request(portfolio=portfolio, price=price, spec=spec, side=OrderSide.SELL, quantity=Decimal("500"), reduce_only=True)
        outcome = _evaluate(portfolio=portfolio, price=price, spec=spec, request=request, portfolio_halted=True)
        # portfolio_halted check itself doesn't consult allow_reduce_only_during_halt
        # in Phase 2 (see architecture doc's Known Limitations) -- the halt
        # is reported regardless; a future phase's dispatch gate is
        # responsible for consulting the flag before actually blocking.
        assert outcome.decision.kind is RiskDecisionKind.HALTED


# --------------------------------------------------------------------------
# Determinism and identity
# --------------------------------------------------------------------------
class TestDeterminism:
    def test_identical_inputs_produce_byte_identical_serialization_and_ids(self) -> None:
        portfolio = _portfolio()
        price = _price()
        spec = _spec()
        outcome_a = _evaluate(portfolio=portfolio, price=price, spec=spec)
        outcome_b = _evaluate(portfolio=portfolio, price=price, spec=spec)
        assert outcome_a.decision.risk_decision_id == outcome_b.decision.risk_decision_id
        assert json.dumps(outcome_a.decision.to_json_dict(), sort_keys=True) == json.dumps(outcome_b.decision.to_json_dict(), sort_keys=True)

    def test_shuffled_position_construction_order_does_not_change_the_decision(self) -> None:
        a = PositionSnapshot(instrument_id="AAA", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("100"), average_entry_price=Decimal("10"), mark_price=Decimal("10"), unrealized_pnl=Decimal("0"), realized_pnl=Decimal("0"), contract_multiplier=Decimal("1"))
        b = PositionSnapshot(instrument_id="BBB", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("100"), average_entry_price=Decimal("10"), mark_price=Decimal("10"), unrealized_pnl=Decimal("0"), realized_pnl=Decimal("0"), contract_multiplier=Decimal("1"))
        portfolio_forward = _portfolio(positions=(a, b), cash=Decimal("98000"))
        portfolio_backward = _portfolio(positions=(b, a), cash=Decimal("98000"))
        assert portfolio_forward.snapshot_id == portfolio_backward.snapshot_id
        price = _price(instrument_id="AAA")
        spec = _spec()
        outcome_forward = _evaluate(portfolio=portfolio_forward, price=price, spec=spec, request=_request(portfolio=portfolio_forward, price=price, spec=spec, instrument_id="AAA"))
        outcome_backward = _evaluate(portfolio=portfolio_backward, price=price, spec=spec, request=_request(portfolio=portfolio_backward, price=price, spec=spec, instrument_id="AAA"))
        assert outcome_forward.decision.risk_decision_id == outcome_backward.decision.risk_decision_id

    def test_economic_field_change_does_change_identity(self) -> None:
        portfolio = _portfolio()
        price = _price()
        spec = _spec()
        outcome_a = _evaluate(portfolio=portfolio, price=price, spec=spec, request=_request(portfolio=portfolio, price=price, spec=spec, quantity=Decimal("1000")))
        outcome_b = _evaluate(portfolio=portfolio, price=price, spec=spec, request=_request(portfolio=portfolio, price=price, spec=spec, quantity=Decimal("2000")))
        assert outcome_a.decision.risk_decision_id != outcome_b.decision.risk_decision_id

    def test_independently_reconstructed_identical_scenario_yields_identical_id(self) -> None:
        # Two COMPLETELY separate object graphs, built from scratch, that
        # are economically identical -- proves no incidental Python
        # object-identity/construction-order "operational metadata" leaks
        # into the decision id.
        def build() -> str:
            portfolio = _portfolio()
            price = _price()
            spec = _spec()
            return _evaluate(portfolio=portfolio, price=price, spec=spec).decision.risk_decision_id

        assert build() == build()

    def test_no_wall_clock_dependence_across_a_real_delay(self) -> None:
        import time

        portfolio = _portfolio()
        price = _price()
        spec = _spec()
        outcome_a = _evaluate(portfolio=portfolio, price=price, spec=spec)
        time.sleep(0.05)
        outcome_b = _evaluate(portfolio=portfolio, price=price, spec=spec)
        assert outcome_a.decision.risk_decision_id == outcome_b.decision.risk_decision_id

    def test_identity_stable_under_pythonhashseed(self) -> None:
        import os
        import subprocess
        import sys

        script = (
            "from decimal import Decimal\n"
            "from datetime import datetime, timezone\n"
            "from quant_platform.portfolio_risk.models import OrderSide\n"
            "from quant_platform.portfolio_risk.snapshots import create_portfolio_snapshot, create_price_snapshot\n"
            "from quant_platform.portfolio_risk.specs import PortfolioRiskPolicy, PortfolioRiskSpec, compute_portfolio_risk_spec_id\n"
            "from quant_platform.portfolio_risk.decisions import create_risk_evaluation_request\n"
            "from quant_platform.portfolio_risk.evaluator import evaluate_risk\n"
            "T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)\n"
            "portfolio = create_portfolio_snapshot(portfolio_id='p1', event_time=T0, cash=Decimal('100000'), equity=Decimal('100000'), "
            "realized_pnl=Decimal('0'), unrealized_pnl=Decimal('0'), peak_equity=Decimal('100000'), daily_start_equity=Decimal('100000'), "
            "positions=(), source_execution_session_id=None)\n"
            "price = create_price_snapshot(instrument_id='EURUSD', bid=Decimal('1.0995'), ask=Decimal('1.1005'), reference_price=Decimal('1.10'), "
            "event_time=T0, source_event_id='e1')\n"
            "policy = PortfolioRiskPolicy(max_order_notional=Decimal('50000'), max_position_notional=Decimal('50000'), "
            "max_instrument_gross_exposure=Decimal('50000'), max_strategy_gross_exposure=Decimal('50000'), "
            "max_portfolio_gross_exposure=Decimal('100000'), max_portfolio_net_exposure=Decimal('100000'), "
            "max_concentration_fraction=Decimal('1'), max_leverage=Decimal('5'), max_daily_realized_loss=Decimal('10000'), "
            "max_total_loss=Decimal('20000'), max_drawdown_fraction=Decimal('0.5'), max_consecutive_losses=5, "
            "minimum_cash_buffer=Decimal('1000'), maximum_price_age=60, maximum_portfolio_snapshot_age=60, allow_reduce_only_during_halt=True)\n"
            "spec = PortfolioRiskSpec(schema_version=1, policy=policy)\n"
            "policy_id = compute_portfolio_risk_spec_id(spec).portfolio_risk_spec_id\n"
            "request = create_risk_evaluation_request(execution_intent_id='1'*64, execution_session_id='2'*64, portfolio_id='p1', "
            "strategy_id='s1', instrument_id='EURUSD', side=OrderSide.BUY, quantity=Decimal('1000'), portfolio_snapshot_id=portfolio.snapshot_id, "
            "price_snapshot_id=price.price_snapshot_id, risk_policy_id=policy_id, reduce_only=False, requested_sequence=0, event_time=T0)\n"
            "outcome = evaluate_risk(request=request, portfolio=portfolio, price=price, spec=spec, evaluation_time=T0, portfolio_halted=False, "
            "consecutive_losses=0, contract_multiplier=Decimal('1'), decision_sequence=0)\n"
            "print(outcome.decision.risk_decision_id)\n"
        )
        results = set()
        for seed in ("0", "1", "42"):
            proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": seed})
            results.add(proc.stdout.strip())
        assert len(results) == 1


# --------------------------------------------------------------------------
# Adversarial
# --------------------------------------------------------------------------
class TestAdversarialForgedIdentity:
    def test_tampered_portfolio_snapshot_is_rejected(self) -> None:
        portfolio = _portfolio()
        price = _price()
        spec = _spec()
        # Round-trip through JSON with a hand-tampered realized_pnl figure
        # (deliberately chosen because Phase 1's own accounting-
        # reconciliation invariant does NOT cross-validate realized_pnl
        # against anything else -- see snapshots.py's module docstring --
        # so this tamper survives __post_init__ and reaches the
        # forged-identity check specifically, rather than being caught
        # earlier by the unrelated accounting-reconciliation check), but
        # KEEPS the original (now stale/forged) snapshot_id -- exactly the
        # "presented a forged/tampered snapshot" adversarial scenario.
        tampered_raw = portfolio.to_json_dict()
        tampered_raw["realized_pnl"] = "999999999"
        tampered_portfolio = PortfolioSnapshot.from_json_dict(tampered_raw)
        assert tampered_portfolio.snapshot_id == portfolio.snapshot_id  # id was NOT recomputed
        request = _request(portfolio=portfolio, price=price, spec=spec)  # references the ORIGINAL id
        with pytest.raises(RiskEvaluationError):
            _evaluate(portfolio=tampered_portfolio, price=price, spec=spec, request=request)

    def test_tampered_price_snapshot_is_rejected(self) -> None:
        from quant_platform.portfolio_risk.snapshots import PriceSnapshot

        portfolio = _portfolio()
        price = _price()
        spec = _spec()
        tampered_raw = price.to_json_dict()
        tampered_raw["ask"] = "999.0"
        tampered_price = PriceSnapshot.from_json_dict(tampered_raw)
        request = _request(portfolio=portfolio, price=price, spec=spec)
        with pytest.raises(RiskEvaluationError):
            _evaluate(portfolio=portfolio, price=tampered_price, spec=spec, request=request)


class TestAdversarialCrossInstrumentPrice:
    def test_price_for_a_different_instrument_than_requested_is_rejected(self) -> None:
        portfolio = _portfolio()
        price = _price(instrument_id="GBPUSD")
        spec = _spec()
        request = _request(portfolio=portfolio, price=price, spec=spec, instrument_id="EURUSD")
        with pytest.raises(RiskEvaluationError):
            _evaluate(portfolio=portfolio, price=price, spec=spec, request=request)


class TestAdversarialCrossPortfolioMismatch:
    def test_portfolio_id_mismatch_is_rejected(self) -> None:
        portfolio = _portfolio(portfolio_id="portfolio-A")
        price = _price()
        spec = _spec()
        request = _request(portfolio=portfolio, price=price, spec=spec, portfolio_id="portfolio-B")
        with pytest.raises(RiskEvaluationError):
            _evaluate(portfolio=portfolio, price=price, spec=spec, request=request)


class TestAdversarialCrossPolicyMismatch:
    def test_wrong_policy_id_in_request_is_rejected(self) -> None:
        portfolio = _portfolio()
        price = _price()
        spec = _spec()
        request = _request(portfolio=portfolio, price=price, spec=spec, risk_policy_id="f" * 64)
        with pytest.raises(RiskEvaluationError):
            _evaluate(portfolio=portfolio, price=price, spec=spec, request=request)


class TestAdversarialStaleData:
    def test_stale_price_denies(self) -> None:
        portfolio = _portfolio()
        price = _price(event_time=_T0 - timedelta(seconds=120))
        spec = _spec(maximum_price_age=60)
        outcome = _evaluate(portfolio=portfolio, price=price, spec=spec)
        assert outcome.decision.kind is RiskDecisionKind.DENIED
        assert RiskDenialReason.STALE_PRICE in outcome.decision.denial_reasons

    def test_stale_portfolio_snapshot_denies(self) -> None:
        portfolio = _portfolio(event_time=_T0 - timedelta(seconds=120))
        price = _price()
        spec = _spec(maximum_portfolio_snapshot_age=60)
        outcome = _evaluate(portfolio=portfolio, price=price, spec=spec)
        assert outcome.decision.kind is RiskDecisionKind.DENIED
        assert RiskDenialReason.STALE_PORTFOLIO_SNAPSHOT in outcome.decision.denial_reasons


class TestAdversarialIncoherentPortfolioAccounting:
    def test_portfolio_snapshot_cannot_be_constructed_incoherently(self) -> None:
        # Phase 1's own construction-time invariant makes this
        # structurally impossible to even reach the evaluator -- verified
        # here at the integration boundary.
        from quant_platform.core.exceptions import PortfolioSnapshotValidationError

        with pytest.raises(PortfolioSnapshotValidationError):
            _portfolio(equity=Decimal("999999"))


class TestAdversarialNaNInfinityDecimal:
    def test_check_result_rejects_nan_measured_value(self) -> None:
        from quant_platform.core.exceptions import RiskEvaluationError as CheckError
        from quant_platform.portfolio_risk.decisions import RiskCheckResult

        with pytest.raises(CheckError):
            RiskCheckResult(check_identity="x", measured_value=Decimal("NaN"), limit_value=Decimal("1"), passed=True, severity=RiskCheckSeverity.INFO, denial_reason=None)

    def test_check_result_rejects_infinite_limit_value(self) -> None:
        from quant_platform.portfolio_risk.decisions import RiskCheckResult

        with pytest.raises(RiskEvaluationError):
            RiskCheckResult(check_identity="x", measured_value=Decimal("1"), limit_value=Decimal("Infinity"), passed=True, severity=RiskCheckSeverity.INFO, denial_reason=None)

    def test_policy_rejects_nan_limit(self) -> None:
        from quant_platform.core.exceptions import PortfolioRiskPolicyError

        with pytest.raises(PortfolioRiskPolicyError):
            _policy(max_order_notional=Decimal("NaN"))


class TestAdversarialDuplicatePositions:
    def test_duplicate_instrument_strategy_identity_cannot_be_constructed(self) -> None:
        from quant_platform.core.exceptions import PortfolioSnapshotValidationError

        a = PositionSnapshot(instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("100"), average_entry_price=Decimal("1"), mark_price=Decimal("1"), unrealized_pnl=Decimal("0"), realized_pnl=Decimal("0"), contract_multiplier=Decimal("1"))
        b = PositionSnapshot(instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("100"), average_entry_price=Decimal("1"), mark_price=Decimal("1"), unrealized_pnl=Decimal("0"), realized_pnl=Decimal("0"), contract_multiplier=Decimal("1"))
        with pytest.raises(PortfolioSnapshotValidationError):
            _portfolio(positions=(a, b), cash=Decimal("100000"), unrealized_pnl=Decimal("0"))


class TestAdversarialNegativeOrZeroPrices:
    def test_price_snapshot_rejects_zero_and_negative(self) -> None:
        from quant_platform.core.exceptions import StalePriceError

        with pytest.raises(StalePriceError):
            _price(bid=Decimal("0"))
        with pytest.raises(StalePriceError):
            _price(ask=Decimal("-1"))


class TestAdversarialUnsupportedSide:
    def test_unknown_side_value_is_rejected_at_parse_time(self) -> None:
        with pytest.raises(ValueError):
            OrderSide("hold")


class TestAdversarialReduceOnlyCrossingThroughZero:
    def test_reduce_only_that_would_cross_through_zero_is_denied(self) -> None:
        position = PositionSnapshot(instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("1000"), average_entry_price=Decimal("1.10"), mark_price=Decimal("1.10"), unrealized_pnl=Decimal("0"), realized_pnl=Decimal("0"), contract_multiplier=Decimal("1"))
        portfolio = _portfolio(positions=(position,), cash=Decimal("98900"))
        price = _price()
        spec = _spec()
        request = _request(portfolio=portfolio, price=price, spec=spec, side=OrderSide.SELL, quantity=Decimal("2000"), reduce_only=True)
        outcome = _evaluate(portfolio=portfolio, price=price, spec=spec, request=request)
        assert outcome.decision.kind is RiskDecisionKind.DENIED
        assert RiskDenialReason.INCOHERENT_EVALUATION_STATE in outcome.decision.denial_reasons

    def test_reduce_only_opening_a_new_position_is_denied(self) -> None:
        portfolio = _portfolio()
        price = _price()
        spec = _spec()
        request = _request(portfolio=portfolio, price=price, spec=spec, side=OrderSide.BUY, quantity=Decimal("1000"), reduce_only=True)
        outcome = _evaluate(portfolio=portfolio, price=price, spec=spec, request=request)
        assert outcome.decision.kind is RiskDecisionKind.DENIED
        assert RiskDenialReason.INCOHERENT_EVALUATION_STATE in outcome.decision.denial_reasons


class TestAdversarialCannotApproveAfterOneCheckFails:
    def test_a_single_failing_check_forces_non_approval(self) -> None:
        portfolio = _portfolio()
        price = _price()
        spec = _spec(max_leverage=Decimal("0.0001"))
        outcome = _evaluate(portfolio=portfolio, price=price, spec=spec)
        assert outcome.decision.kind is not RiskDecisionKind.APPROVED


class TestAdversarialCannotApproveWhenACheckWasNotExecuted:
    def test_aggregate_raises_when_a_required_check_is_missing(self) -> None:
        from quant_platform.portfolio_risk import checks
        from quant_platform.portfolio_risk.decisions import RiskCheckResult

        all_passing = [
            RiskCheckResult(check_identity=identity, measured_value=Decimal("0"), limit_value=Decimal("1"), passed=True, severity=RiskCheckSeverity.INFO, denial_reason=None)
            for identity in checks.CHECK_ORDER
        ]
        incomplete = tuple(all_passing[:-1])  # drop the last required check
        with pytest.raises(RiskEvaluationError):
            _aggregate_decision_kind(incomplete)

    def test_aggregate_succeeds_with_all_eighteen_present(self) -> None:
        from quant_platform.portfolio_risk import checks
        from quant_platform.portfolio_risk.decisions import RiskCheckResult

        all_passing = tuple(
            RiskCheckResult(check_identity=identity, measured_value=Decimal("0"), limit_value=Decimal("1"), passed=True, severity=RiskCheckSeverity.INFO, denial_reason=None)
            for identity in checks.CHECK_ORDER
        )
        kind, reasons = _aggregate_decision_kind(all_passing)
        assert kind is RiskDecisionKind.APPROVED
        assert reasons == ()

    def test_duplicate_check_identity_with_a_missing_one_is_rejected(self) -> None:
        from quant_platform.portfolio_risk import checks
        from quant_platform.portfolio_risk.decisions import RiskCheckResult

        results = [
            RiskCheckResult(check_identity=identity, measured_value=Decimal("0"), limit_value=Decimal("1"), passed=True, severity=RiskCheckSeverity.INFO, denial_reason=None)
            for identity in checks.CHECK_ORDER
        ]
        results[-1] = results[0]  # duplicate the first identity, dropping the last
        with pytest.raises(RiskEvaluationError):
            _aggregate_decision_kind(tuple(results))
