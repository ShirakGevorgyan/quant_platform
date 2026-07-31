"""Unit tests for `execution_gateway.paper_bridge`'s standalone domain
objects (Milestone 8, Section 5/6): `ExecutionIntent`/`ExecutionAuthorization`
construction, validation, and content-addressed identity. The full
`execution_intent_from_paper_order` bridge (which requires a genuine,
independently verified Milestone 7 paper session) is covered end-to-end
by the acceptance workflow (`tests/integration/
test_execution_gateway_acceptance.py`), not here."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import ExecutionIntentError
from quant_platform.execution_gateway.models import (
    AuthorizationMode,
    OrderSide,
    OrderTypeKind,
    TimeInForceKind,
)
from quant_platform.execution_gateway.paper_bridge import (
    ExecutionAuthorization,
    ExecutionIntent,
    create_execution_authorization,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64


def _intent(**overrides: object) -> ExecutionIntent:
    base: dict[str, object] = {
        "execution_intent_id": "0" * 64, "execution_session_id": _SHA_A, "paper_session_id": _SHA_B, "source_decision_id": _SHA_C,
        "source_paper_order_id": _SHA_D, "instrument_id": "EURUSD", "side": OrderSide.BUY, "quantity": Decimal("1.5"), "order_type": OrderTypeKind.MARKET,
        "limit_price": None, "stop_price": None, "time_in_force": TimeInForceKind.DAY, "reduce_only": False, "close_position": False,
        "strategy_candidate_id": _SHA_A, "model_artifact_id": _SHA_B, "execution_bridge_authorization_id": _SHA_E, "portfolio_risk_authorization_id": None,
        "source_event_id": None, "source_event_time": "2026-01-01T00:00:00+00:00", "created_sequence": 0, "contract_multiplier": Decimal("100"),
        "identity_version": 2,
    }
    base.update(overrides)
    return ExecutionIntent(**base)  # type: ignore[arg-type]


def _authorization(**overrides: object) -> ExecutionAuthorization:
    return create_execution_authorization(
        paper_session_id=overrides.get("paper_session_id", _SHA_A),  # type: ignore[arg-type]
        paper_order_id=overrides.get("paper_order_id", _SHA_B),  # type: ignore[arg-type]
        execution_gateway_spec_id=overrides.get("execution_gateway_spec_id", _SHA_C),  # type: ignore[arg-type]
        authorized_quantity=overrides.get("authorized_quantity", Decimal("1")),  # type: ignore[arg-type]
        authorized_side=overrides.get("authorized_side", OrderSide.BUY),  # type: ignore[arg-type]
        issued_sequence=overrides.get("issued_sequence", 0),  # type: ignore[arg-type]
        source_verification_id=overrides.get("source_verification_id", _SHA_D),  # type: ignore[arg-type]
    )


class TestExecutionIntentValidation:
    def test_valid_market_intent_constructs(self) -> None:
        intent = _intent()
        assert intent.order_type is OrderTypeKind.MARKET

    def test_market_rejects_limit_price(self) -> None:
        with pytest.raises(ExecutionIntentError):
            _intent(limit_price=Decimal("1.1"))

    def test_market_rejects_stop_price(self) -> None:
        with pytest.raises(ExecutionIntentError):
            _intent(stop_price=Decimal("1.1"))

    def test_limit_requires_limit_price(self) -> None:
        with pytest.raises(ExecutionIntentError):
            _intent(order_type=OrderTypeKind.LIMIT)

    def test_limit_rejects_stop_price(self) -> None:
        with pytest.raises(ExecutionIntentError):
            _intent(order_type=OrderTypeKind.LIMIT, limit_price=Decimal("1.1"), stop_price=Decimal("1.2"))

    def test_stop_requires_stop_price(self) -> None:
        with pytest.raises(ExecutionIntentError):
            _intent(order_type=OrderTypeKind.STOP)

    def test_stop_rejects_limit_price(self) -> None:
        with pytest.raises(ExecutionIntentError):
            _intent(order_type=OrderTypeKind.STOP, stop_price=Decimal("1.2"), limit_price=Decimal("1.1"))

    def test_rejects_non_positive_quantity(self) -> None:
        with pytest.raises(ExecutionIntentError):
            _intent(quantity=Decimal("0"))

    def test_rejects_non_positive_contract_multiplier(self) -> None:
        with pytest.raises(ExecutionIntentError):
            _intent(contract_multiplier=Decimal("-1"))

    def test_close_position_requires_reduce_only(self) -> None:
        with pytest.raises(ExecutionIntentError):
            _intent(close_position=True, reduce_only=False)

    def test_close_position_with_reduce_only_is_valid(self) -> None:
        intent = _intent(close_position=True, reduce_only=True)
        assert intent.close_position is True

    def test_rejects_non_sha256_execution_session_id(self) -> None:
        with pytest.raises(ExecutionIntentError):
            _intent(execution_session_id="not-a-hash")

    def test_rejects_negative_created_sequence(self) -> None:
        with pytest.raises(ExecutionIntentError):
            _intent(created_sequence=-1)

    def test_round_trips_through_json(self) -> None:
        intent = _intent()
        restored = ExecutionIntent.from_json_dict(intent.to_json_dict())
        assert restored.to_json_dict() == intent.to_json_dict()


class TestExecutionAuthorizationValidation:
    def test_valid_authorization_constructs(self) -> None:
        auth = _authorization()
        assert auth.authorization_mode is AuthorizationMode.TEST_ONLY_DUMMY_EXECUTION

    def test_rejects_non_positive_authorized_quantity(self) -> None:
        with pytest.raises(ExecutionIntentError):
            ExecutionAuthorization(
                execution_authorization_id="0" * 64, authorization_mode=AuthorizationMode.TEST_ONLY_DUMMY_EXECUTION, paper_session_id=_SHA_A,
                paper_order_id=_SHA_B, execution_gateway_spec_id=_SHA_C, authorized_quantity=Decimal("0"), authorized_side=OrderSide.BUY,
                issued_sequence=0, source_verification_id=_SHA_D,
            )

    def test_rejects_negative_issued_sequence(self) -> None:
        with pytest.raises(ExecutionIntentError):
            ExecutionAuthorization(
                execution_authorization_id="0" * 64, authorization_mode=AuthorizationMode.TEST_ONLY_DUMMY_EXECUTION, paper_session_id=_SHA_A,
                paper_order_id=_SHA_B, execution_gateway_spec_id=_SHA_C, authorized_quantity=Decimal("1"), authorized_side=OrderSide.BUY,
                issued_sequence=-1, source_verification_id=_SHA_D,
            )

    def test_round_trips_through_json(self) -> None:
        auth = _authorization()
        restored = ExecutionAuthorization.from_json_dict(auth.to_json_dict())
        assert restored.to_json_dict() == auth.to_json_dict()


class TestIntentAndAuthorizationIdentity:
    def test_intent_identity_is_deterministic(self) -> None:
        assert _intent().execution_intent_id == _intent().execution_intent_id

    def test_intent_identity_changes_with_quantity(self) -> None:
        assert _intent(execution_intent_id="1" * 64).to_json_dict() != _intent(execution_intent_id="1" * 64, quantity=Decimal("2")).to_json_dict()

    def test_authorization_identity_is_deterministic(self) -> None:
        assert _authorization().execution_authorization_id == _authorization().execution_authorization_id

    def test_authorization_identity_changes_with_quantity(self) -> None:
        a = _authorization(authorized_quantity=Decimal("1"))
        b = _authorization(authorized_quantity=Decimal("2"))
        assert a.execution_authorization_id != b.execution_authorization_id

    def test_authorization_identity_changes_with_paper_order_id(self) -> None:
        a = _authorization(paper_order_id=_SHA_A)
        b = _authorization(paper_order_id=_SHA_E)
        assert a.execution_authorization_id != b.execution_authorization_id

    def test_forged_authorization_id_is_detectable_by_recomputation(self) -> None:
        genuine = _authorization()
        forged = replace(genuine, execution_authorization_id="f" * 64)
        from quant_platform.execution_gateway.identity import compute_content_id
        from quant_platform.execution_gateway.paper_bridge import EXECUTION_AUTHORIZATION_KIND

        recomputed = compute_content_id(EXECUTION_AUTHORIZATION_KIND, forged.to_identity_payload())
        assert recomputed != forged.execution_authorization_id
        assert recomputed == genuine.execution_authorization_id
