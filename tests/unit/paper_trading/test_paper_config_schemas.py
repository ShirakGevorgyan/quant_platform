"""Milestone 7, Section 29: `config.paper_trading_schemas.PaperTradingConfig`.
Covers unknown-field rejection, finite/positive/enum validation, the two
cross-field checks (`instrument.maximum_quantity >= minimum_quantity`,
`bar_interval` required-for-bar/forbidden-for-quote), identity-hex
validation, and a full `.build()` round-trip producing a genuinely valid
`PaperTradingSpec` (deterministic identity computable without error)."""

from __future__ import annotations

import math

import pydantic
import pytest

from quant_platform.config.paper_trading_schemas import PaperTradingConfig
from quant_platform.paper_trading.models import SessionMode
from quant_platform.paper_trading.specs import compute_paper_session_spec_id

_HEX_A = "a" * 64
_HEX_B = "b" * 64
_HEX_C = "c" * 64
_HEX_D = "d" * 64
_HEX_E = "e" * 64

_MINIMAL_INSTRUMENT: dict[str, object] = {
    "symbol": "X", "quote_currency": "USD", "contract_multiplier": 1.0, "tick_size": 0.01, "quantity_step": 0.01, "minimum_quantity": 0.01,
    "price_precision": 2, "quantity_precision": 2, "margin_mode": "cash", "account_currency": "USD", "financing_convention": "none",
    "trading_timezone": "UTC", "session_calendar_identity": "always_open",
}


def _minimal_config(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "ml_artifacts_root": "/tmp/ml_artifacts", "historical_storage_root": "/tmp/historical", "research_storage_root": "/tmp/research",
        "verified_robustness_id": _HEX_A, "verified_promotion_decision_id": _HEX_B, "strategy_candidate_identity": _HEX_C,
        "model_artifact_identity": _HEX_D, "feature_spec_identity": _HEX_E, "instrument": dict(_MINIMAL_INSTRUMENT),
        "price_precision": 2, "quantity_precision": 2, "starting_cash": 100_000.0,
    }
    defaults.update(overrides)
    return defaults


class TestPaperTradingConfigHappyPath:
    def test_minimal_config_builds_a_valid_spec(self) -> None:
        config = PaperTradingConfig(**_minimal_config())
        spec = config.build()
        assert spec.session_mode is SessionMode.REPLAY_PAPER
        assert spec.instrument.symbol == "X"
        assert spec.starting_cash == 100_000.0
        # Deterministic identity must compute without error -- proves the
        # built spec is genuinely well-formed, not merely "constructed."
        identity = compute_paper_session_spec_id(spec)
        assert identity.paper_session_spec_id

    def test_defaults_match_documented_fail_closed_choices(self) -> None:
        config = PaperTradingConfig(**_minimal_config())
        assert config.session_mode == "replay_paper"
        assert config.fill_policy.partial_fill_policy == "full_fill_only"
        assert config.position_policy.single_instrument_only is True
        assert config.execution_policy.bar_ambiguity_policy == "worst_case"

    def test_shadow_observation_mode_builds(self) -> None:
        config = PaperTradingConfig(**_minimal_config(session_mode="shadow_observation"))
        spec = config.build()
        assert spec.session_mode is SessionMode.SHADOW_OBSERVATION

    def test_quote_mode_with_no_bar_interval_builds(self) -> None:
        config = PaperTradingConfig(**_minimal_config(market_event_mode="quote", bar_interval=None))
        spec = config.build()
        assert spec.bar_interval is None

    def test_full_override_of_every_nested_policy_builds(self) -> None:
        config = PaperTradingConfig(**_minimal_config(
            order_policy={"close_before_reverse": False, "cooldown_bars": 2, "maximum_orders_per_event": 3, "maximum_order_rate_per_window": 10, "order_rate_window_events": 50},
            spread={"kind": "fixed_price_units", "price_units": 0.3}, slippage={"kind": "fixed_basis_points", "basis_points": 1.0},
            commission={"kind": "per_side_basis_points", "per_side_basis_points": 2.0},
            long_financing={"kind": "fixed_daily_basis_points", "daily_basis_points": 1.5}, short_financing={"kind": "fixed_daily_basis_points", "daily_basis_points": -0.5},
            latency_policy={"decision_to_submit_ms": 50, "submit_to_accept_ms": 50, "accept_to_fill_eligible_ms": 50},
            risk_limits={"maximum_absolute_position": 10.0, "maximum_drawdown_fraction": 0.2},
            starting_positions=[{"instrument_symbol": "X", "signed_quantity": 5.0, "average_entry_price": 101.0}],
        ))
        spec = config.build()
        assert spec.order_policy.cooldown_bars == 2
        assert spec.spread_policy.price_units == 0.3
        assert spec.financing_policy.long_financing.daily_basis_points == 1.5
        assert spec.financing_policy.short_financing.daily_basis_points == -0.5
        assert spec.risk_limits.maximum_absolute_position == 10.0
        assert len(spec.starting_positions) == 1
        assert spec.starting_positions[0].signed_quantity == 5.0

    def test_starting_positions_omitted_and_explicit_empty_list_are_equivalent(self) -> None:
        omitted = PaperTradingConfig(**_minimal_config()).build()
        explicit_empty = PaperTradingConfig(**_minimal_config(starting_positions=[])).build()
        assert omitted.starting_positions == explicit_empty.starting_positions == ()


class TestPaperTradingConfigRejectsUnknownFields:
    def test_unknown_top_level_field_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            PaperTradingConfig(**_minimal_config(broker_api_key="secret"))

    def test_unknown_nested_instrument_field_rejected(self) -> None:
        instrument = dict(_MINIMAL_INSTRUMENT)
        instrument["broker_endpoint_url"] = "https://example.com/orders"
        with pytest.raises(pydantic.ValidationError):
            PaperTradingConfig(**_minimal_config(instrument=instrument))


class TestPaperTradingConfigValidation:
    def test_live_session_mode_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            PaperTradingConfig(**_minimal_config(session_mode="live"))

    @pytest.mark.parametrize("bad_hex", ["not_hex", "a" * 63, "A" * 64, ""])
    def test_malformed_identity_rejected(self, bad_hex: str) -> None:
        with pytest.raises(pydantic.ValidationError):
            PaperTradingConfig(**_minimal_config(verified_robustness_id=bad_hex))

    def test_negative_starting_cash_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            PaperTradingConfig(**_minimal_config(starting_cash=-1.0))

    def test_infinite_starting_cash_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            PaperTradingConfig(**_minimal_config(starting_cash=math.inf))

    def test_nan_starting_cash_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            PaperTradingConfig(**_minimal_config(starting_cash=math.nan))

    def test_negative_commission_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            PaperTradingConfig(**_minimal_config(commission={"kind": "fixed_per_trade", "fixed_per_trade": -1.0}))

    def test_instrument_maximum_below_minimum_quantity_rejected(self) -> None:
        instrument = dict(_MINIMAL_INSTRUMENT)
        instrument["minimum_quantity"] = 10.0
        instrument["maximum_quantity"] = 1.0
        with pytest.raises(pydantic.ValidationError, match="maximum_quantity"):
            PaperTradingConfig(**_minimal_config(instrument=instrument))

    def test_bar_mode_without_bar_interval_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="bar_interval"):
            PaperTradingConfig(**_minimal_config(market_event_mode="bar", bar_interval=None))

    def test_quote_mode_with_bar_interval_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="bar_interval"):
            PaperTradingConfig(**_minimal_config(market_event_mode="quote", bar_interval="H1"))

    def test_unsupported_bar_interval_literal_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            PaperTradingConfig(**_minimal_config(bar_interval="W1"))

    def test_calibration_artifact_identity_omitted_is_fine(self) -> None:
        config = PaperTradingConfig(**_minimal_config())
        assert config.calibration_artifact_identity is None
        spec = config.build()
        assert spec.calibration_artifact_identity is None
