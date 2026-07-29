"""Milestone 7, Section 3: `PaperTradingSpec`/`compute_paper_session_spec_id`
identity determinism and immutable-spec validation. Two independently
constructed specs with identical field values must produce the same
`paper_session_spec_id`; any result-affecting field change must change it.
Mirrors `tests/unit/robustness/test_specs_and_identity.py`'s structure,
including its release-audit-derived regression classes (order-preserving
`to_json_dict` versus canonicalizing `to_identity_payload`)."""

from __future__ import annotations

import pytest

from quant_platform.backtesting.models import (
    CommissionModelKind,
    FinancingModelKind,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.specs import CommissionSpec, FinancingSpec, SlippageSpec, SpreadSpec
from quant_platform.core.exceptions import PaperTradingSpecError
from quant_platform.core.types import Timeframe
from quant_platform.paper_trading.models import (
    BarAmbiguityPolicyKind,
    ClockMode,
    MarketEventMode,
    MarkFieldKind,
    PartialFillPolicyKind,
    SessionMode,
)
from quant_platform.paper_trading.specs import (
    DEFAULT_EXECUTION_POLICY,
    DEFAULT_POSITION_POLICY,
    DEFAULT_RISK_LIMITS,
    DEFAULT_SESSION_BOUNDARY_POLICY,
    ExecutionPolicySpec,
    FillPolicySpec,
    FinancingPolicySpec,
    InstrumentSpec,
    LatencyPolicySpec,
    LiquidityPolicySpec,
    OrderPolicySpec,
    PaperTradingSpec,
    PositionPolicySpec,
    RiskLimitsSpec,
    StartingPositionSpec,
    compute_paper_session_spec_id,
)

_HEX_A = "a" * 64
_HEX_B = "b" * 64
_HEX_C = "c" * 64
_HEX_D = "d" * 64
_HEX_E = "e" * 64


def _instrument(**overrides: object) -> InstrumentSpec:
    defaults: dict[str, object] = {
        "symbol": "HYPOTHETICAL_XAU", "base_currency": "XAU", "quote_currency": "USD", "contract_multiplier": 100.0,
        "tick_size": 0.01, "tick_value": 1.0, "quantity_step": 0.01, "minimum_quantity": 0.01, "maximum_quantity": 100.0,
        "price_precision": 2, "quantity_precision": 2, "margin_mode": "hypothetical_margin_mode", "account_currency": "USD",
        "financing_convention": "hypothetical_daily_swap", "trading_timezone": "UTC", "session_calendar_identity": "hypothetical_247",
    }
    defaults.update(overrides)
    return InstrumentSpec(**defaults)  # type: ignore[arg-type]


def _spec(**overrides: object) -> PaperTradingSpec:
    defaults: dict[str, object] = {
        "schema_version": 1, "verified_robustness_id": _HEX_A, "verified_promotion_decision_id": _HEX_B,
        "strategy_candidate_identity": _HEX_C, "model_artifact_identity": _HEX_D, "calibration_artifact_identity": _HEX_E,
        "feature_spec_identity": _HEX_A, "instrument": _instrument(), "price_precision": 2, "quantity_precision": 2,
        "session_mode": SessionMode.REPLAY_PAPER, "market_event_mode": MarketEventMode.BAR, "bar_interval": Timeframe.H1,
        "clock_mode": ClockMode.REPLAY, "starting_cash": 100_000.0, "starting_positions": (),
        "order_policy": OrderPolicySpec(close_before_reverse=True, cooldown_bars=0, maximum_orders_per_event=5, maximum_order_rate_per_window=10, order_rate_window_events=20),
        "execution_policy": DEFAULT_EXECUTION_POLICY, "fill_policy": FillPolicySpec(partial_fill_policy=PartialFillPolicyKind.FULL_FILL_ONLY),
        "spread_policy": SpreadSpec(kind=SpreadModelKind.FIXED_PRICE_UNITS, price_units=0.3),
        "slippage_policy": SlippageSpec(kind=SlippageModelKind.FIXED_BASIS_POINTS, basis_points=1.0),
        "commission_policy": CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=2.0),
        "financing_policy": FinancingPolicySpec(
            long_financing=FinancingSpec(kind=FinancingModelKind.FIXED_DAILY_BASIS_POINTS, daily_basis_points=1.5),
            short_financing=FinancingSpec(kind=FinancingModelKind.FIXED_DAILY_BASIS_POINTS, daily_basis_points=-0.5),
        ),
        "latency_policy": LatencyPolicySpec(decision_to_submit_ms=50, submit_to_accept_ms=50, accept_to_fill_eligible_ms=50),
        "liquidity_policy": LiquidityPolicySpec(trust_disclosed_size=False), "position_policy": DEFAULT_POSITION_POLICY,
        "risk_limits": DEFAULT_RISK_LIMITS, "session_boundary_policy": DEFAULT_SESSION_BOUNDARY_POLICY, "seed": 0,
    }
    defaults.update(overrides)
    return PaperTradingSpec(**defaults)  # type: ignore[arg-type]


class TestPaperSessionSpecIdentityDeterminism:
    def test_identical_specs_produce_identical_id(self) -> None:
        assert compute_paper_session_spec_id(_spec()).paper_session_spec_id == compute_paper_session_spec_id(_spec()).paper_session_spec_id

    def test_json_round_trip_preserves_identity(self) -> None:
        spec = _spec()
        roundtripped = PaperTradingSpec.from_json_dict(spec.to_json_dict())
        assert compute_paper_session_spec_id(spec).paper_session_spec_id == compute_paper_session_spec_id(roundtripped).paper_session_spec_id

    def test_schema_version_does_not_affect_identity(self) -> None:
        payload_a = _spec().to_identity_payload()
        payload_b = _spec(schema_version=1).to_identity_payload()
        assert payload_a == payload_b
        assert "schema_version" not in payload_a

    @pytest.mark.parametrize(
        ("label", "override"),
        [
            ("verified_robustness_id", {"verified_robustness_id": "1" * 64}),
            ("verified_promotion_decision_id", {"verified_promotion_decision_id": "2" * 64}),
            ("strategy_candidate_identity", {"strategy_candidate_identity": "3" * 64}),
            ("model_artifact_identity", {"model_artifact_identity": "4" * 64}),
            ("calibration_artifact_identity", {"calibration_artifact_identity": None}),
            ("feature_spec_identity", {"feature_spec_identity": "5" * 64}),
            ("instrument_symbol", {"instrument": _instrument(symbol="OTHER_SYMBOL")}),
            ("instrument_tick_size", {"instrument": _instrument(tick_size=0.02)}),
            ("price_precision", {"price_precision": 3}),
            ("quantity_precision", {"quantity_precision": 3}),
            ("session_mode", {"session_mode": SessionMode.SHADOW_OBSERVATION}),
            ("market_event_mode_and_interval", {"market_event_mode": MarketEventMode.QUOTE, "bar_interval": None}),
            ("bar_interval", {"bar_interval": Timeframe.M15}),
            ("clock_mode", {"clock_mode": ClockMode.MANUAL_TEST}),
            ("starting_cash", {"starting_cash": 50_000.0}),
            ("starting_positions", {"starting_positions": (StartingPositionSpec(instrument_symbol="HYPOTHETICAL_XAU", signed_quantity=1.0, average_entry_price=1900.0),)}),
            ("order_policy", {"order_policy": OrderPolicySpec(close_before_reverse=False, cooldown_bars=0, maximum_orders_per_event=5, maximum_order_rate_per_window=10, order_rate_window_events=20)}),
            ("execution_policy", {"execution_policy": ExecutionPolicySpec(bar_ambiguity_policy=BarAmbiguityPolicyKind.BEST_CASE, mark_field=MarkFieldKind.CLOSE)}),
            ("fill_policy", {"fill_policy": FillPolicySpec(partial_fill_policy=PartialFillPolicyKind.DETERMINISTIC_PARTIAL)}),
            ("spread_policy", {"spread_policy": SpreadSpec(kind=SpreadModelKind.FIXED_PRICE_UNITS, price_units=0.5)}),
            ("slippage_policy", {"slippage_policy": SlippageSpec(kind=SlippageModelKind.FIXED_BASIS_POINTS, basis_points=2.0)}),
            ("commission_policy", {"commission_policy": CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=3.0)}),
            ("financing_policy", {"financing_policy": FinancingPolicySpec(
                long_financing=FinancingSpec(kind=FinancingModelKind.FIXED_DAILY_BASIS_POINTS, daily_basis_points=2.5),
                short_financing=FinancingSpec(kind=FinancingModelKind.FIXED_DAILY_BASIS_POINTS, daily_basis_points=-0.5),
            )}),
            ("latency_policy", {"latency_policy": LatencyPolicySpec(decision_to_submit_ms=100, submit_to_accept_ms=50, accept_to_fill_eligible_ms=50)}),
            ("liquidity_policy", {"liquidity_policy": LiquidityPolicySpec(trust_disclosed_size=True)}),
            ("position_policy", {"position_policy": PositionPolicySpec(single_instrument_only=True, reduce_only_default=True)}),
            ("risk_limits", {"risk_limits": RiskLimitsSpec(
                maximum_signed_position=10.0, maximum_absolute_position=None, maximum_gross_exposure=None, maximum_order_quantity=None,
                maximum_order_notional=None, maximum_turnover=None, maximum_daily_loss=None, maximum_drawdown_fraction=None,
                maximum_realized_loss=None, maximum_unrealized_loss=None, maximum_rejected_order_count=None,
                maximum_consecutive_execution_failures=None, maximum_stale_data_seconds=None, maximum_reconciliation_discrepancy=1e-6,
            )}),
            ("seed", {"seed": 1}),
        ],
        ids=lambda p: p if isinstance(p, str) else "override",
    )
    def test_changing_any_identity_relevant_field_changes_id(self, label: str, override: dict[str, object]) -> None:
        baseline = compute_paper_session_spec_id(_spec()).paper_session_spec_id
        changed = compute_paper_session_spec_id(_spec(**override)).paper_session_spec_id
        assert baseline != changed, f"changing {label!r} did not change paper_session_spec_id"


class TestStartingPositionsCanonicalizedForIdentityOnly:
    """Mirrors the Milestone 6 release-audit lesson directly: `starting_
    positions` is semantically an unordered set (uniqueness of `instrument_
    symbol` already enforced in `__post_init__`), so `to_identity_payload`
    sorts it, but `to_json_dict` (the durable, round-tripped form) must
    preserve caller-declared order exactly."""

    def _two_positions(self) -> tuple[StartingPositionSpec, StartingPositionSpec]:
        return (
            StartingPositionSpec(instrument_symbol="AAA", signed_quantity=1.0, average_entry_price=10.0),
            StartingPositionSpec(instrument_symbol="BBB", signed_quantity=-2.0, average_entry_price=20.0),
        )

    def _spec_with_positions(self, positions: tuple[StartingPositionSpec, ...]) -> PaperTradingSpec:
        return _spec(position_policy=PositionPolicySpec(single_instrument_only=False, reduce_only_default=False), starting_positions=positions)

    def test_reordered_starting_positions_produce_identical_id(self) -> None:
        forward, backward = self._two_positions()
        forward_id = compute_paper_session_spec_id(self._spec_with_positions((forward, backward))).paper_session_spec_id
        reversed_id = compute_paper_session_spec_id(self._spec_with_positions((backward, forward))).paper_session_spec_id
        assert forward_id == reversed_id

    def test_to_json_dict_preserves_declared_order(self) -> None:
        forward, backward = self._two_positions()
        spec = self._spec_with_positions((backward, forward))
        symbols = [p["instrument_symbol"] for p in spec.to_json_dict()["starting_positions"]]  # type: ignore[index]
        assert symbols == ["BBB", "AAA"]

    def test_full_round_trip_preserves_declared_order(self) -> None:
        forward, backward = self._two_positions()
        spec = self._spec_with_positions((backward, forward))
        roundtripped = PaperTradingSpec.from_json_dict(spec.to_json_dict())
        assert [p.instrument_symbol for p in roundtripped.starting_positions] == ["BBB", "AAA"]


class TestInstrumentSpecValidation:
    def test_empty_symbol_rejected(self) -> None:
        with pytest.raises(PaperTradingSpecError, match="symbol"):
            _instrument(symbol="")

    def test_non_positive_contract_multiplier_rejected(self) -> None:
        with pytest.raises(PaperTradingSpecError, match="contract_multiplier"):
            _instrument(contract_multiplier=0.0)

    def test_non_finite_tick_size_rejected(self) -> None:
        with pytest.raises(PaperTradingSpecError, match="tick_size"):
            _instrument(tick_size=float("nan"))

    def test_maximum_quantity_below_minimum_rejected(self) -> None:
        with pytest.raises(PaperTradingSpecError, match="maximum_quantity"):
            _instrument(minimum_quantity=10.0, maximum_quantity=1.0)

    def test_minimum_quantity_not_multiple_of_step_rejected(self) -> None:
        with pytest.raises(PaperTradingSpecError, match="quantity_step"):
            _instrument(quantity_step=0.03, minimum_quantity=0.1)

    def test_no_hardcoded_xauusd_values_required(self) -> None:
        """Documentation-of-intent test: a completely different, non-XAUUSD
        instrument must construct without any special-casing anywhere."""
        eurusd_like = _instrument(symbol="EURUSD_LIKE", base_currency="EUR", quote_currency="USD", contract_multiplier=100_000.0, tick_size=0.00001, tick_value=None)
        assert eurusd_like.symbol == "EURUSD_LIKE"


class TestRiskLimitsSpecValidation:
    def test_drawdown_fraction_out_of_range_rejected(self) -> None:
        with pytest.raises(PaperTradingSpecError, match="maximum_drawdown_fraction"):
            RiskLimitsSpec(
                maximum_signed_position=None, maximum_absolute_position=None, maximum_gross_exposure=None, maximum_order_quantity=None,
                maximum_order_notional=None, maximum_turnover=None, maximum_daily_loss=None, maximum_drawdown_fraction=1.5,
                maximum_realized_loss=None, maximum_unrealized_loss=None, maximum_rejected_order_count=None,
                maximum_consecutive_execution_failures=None, maximum_stale_data_seconds=None, maximum_reconciliation_discrepancy=1e-6,
            )

    def test_negative_reconciliation_discrepancy_rejected(self) -> None:
        with pytest.raises(PaperTradingSpecError, match="maximum_reconciliation_discrepancy"):
            RiskLimitsSpec(
                maximum_signed_position=None, maximum_absolute_position=None, maximum_gross_exposure=None, maximum_order_quantity=None,
                maximum_order_notional=None, maximum_turnover=None, maximum_daily_loss=None, maximum_drawdown_fraction=None,
                maximum_realized_loss=None, maximum_unrealized_loss=None, maximum_rejected_order_count=None,
                maximum_consecutive_execution_failures=None, maximum_stale_data_seconds=None, maximum_reconciliation_discrepancy=-1.0,
            )


class TestPaperTradingSpecValidation:
    def test_bar_interval_required_for_bar_mode(self) -> None:
        with pytest.raises(PaperTradingSpecError, match="bar_interval"):
            _spec(market_event_mode=MarketEventMode.BAR, bar_interval=None)

    def test_bar_interval_forbidden_for_quote_mode(self) -> None:
        with pytest.raises(PaperTradingSpecError, match="bar_interval"):
            _spec(market_event_mode=MarketEventMode.QUOTE, bar_interval=Timeframe.H1)

    def test_invalid_robustness_id_rejected(self) -> None:
        with pytest.raises(PaperTradingSpecError, match="verified_robustness_id"):
            _spec(verified_robustness_id="not-a-hex-digest")

    def test_negative_starting_cash_rejected(self) -> None:
        with pytest.raises(PaperTradingSpecError, match="starting_cash"):
            _spec(starting_cash=-1.0)

    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(PaperTradingSpecError, match="seed"):
            _spec(seed=-1)

    def test_duplicate_starting_position_symbols_rejected(self) -> None:
        dup = (
            StartingPositionSpec(instrument_symbol="AAA", signed_quantity=1.0, average_entry_price=10.0),
            StartingPositionSpec(instrument_symbol="AAA", signed_quantity=2.0, average_entry_price=11.0),
        )
        with pytest.raises(PaperTradingSpecError, match="repeat"):
            _spec(position_policy=PositionPolicySpec(single_instrument_only=False, reduce_only_default=False), starting_positions=dup)

    def test_single_instrument_only_rejects_foreign_starting_position(self) -> None:
        foreign = (StartingPositionSpec(instrument_symbol="NOT_THE_SESSION_INSTRUMENT", signed_quantity=1.0, average_entry_price=10.0),)
        with pytest.raises(PaperTradingSpecError, match="single_instrument_only"):
            _spec(starting_positions=foreign)

    def test_single_instrument_only_allows_own_instrument_starting_position(self) -> None:
        own = (StartingPositionSpec(instrument_symbol="HYPOTHETICAL_XAU", signed_quantity=1.0, average_entry_price=10.0),)
        spec = _spec(starting_positions=own)
        assert spec.starting_positions == own

    def test_no_live_mode_member_exists(self) -> None:
        """`SessionMode` (Section 0.8/19) must have exactly the three
        documented members -- there is no way to even attempt constructing
        a `LIVE` session."""
        assert {m.value for m in SessionMode} == {"replay_paper", "forward_paper", "shadow_observation"}

    def test_live_like_session_mode_string_rejected_defensively(self) -> None:
        """`SessionMode` cannot literally contain "live" (enforced above),
        but `_reject_live_like` is exercised directly as defense-in-depth
        documentation should the enum ever grow a careless member."""
        from quant_platform.paper_trading.specs import _reject_live_like

        with pytest.raises(PaperTradingSpecError, match="LIVE"):
            _reject_live_like("live", field_name="test_field")


class TestUnknownFieldAndNonFiniteRejection:
    def test_unknown_top_level_field_is_not_silently_ignored_by_constructor(self) -> None:
        """`PaperTradingSpec` is a slotted dataclass -- passing an unknown
        keyword raises `TypeError` at construction, which is the strongest
        possible form of unknown-field rejection (impossible to construct
        at all, not merely rejected after the fact)."""
        with pytest.raises(TypeError):
            _spec(unknown_field_that_does_not_exist=123)  # type: ignore[call-arg]

    def test_non_finite_starting_cash_rejected(self) -> None:
        with pytest.raises(PaperTradingSpecError):
            _spec(starting_cash=float("inf"))

    def test_non_finite_risk_limit_rejected(self) -> None:
        with pytest.raises(PaperTradingSpecError):
            _spec(risk_limits=RiskLimitsSpec(
                maximum_signed_position=float("nan"), maximum_absolute_position=None, maximum_gross_exposure=None, maximum_order_quantity=None,
                maximum_order_notional=None, maximum_turnover=None, maximum_daily_loss=None, maximum_drawdown_fraction=None,
                maximum_realized_loss=None, maximum_unrealized_loss=None, maximum_rejected_order_count=None,
                maximum_consecutive_execution_failures=None, maximum_stale_data_seconds=None, maximum_reconciliation_discrepancy=1e-6,
            ))


class TestExplicitVersusOmittedDefaults:
    def test_explicit_empty_starting_positions_equivalent_to_omitted(self) -> None:
        explicit = _spec(starting_positions=())
        raw = dict(explicit.to_json_dict())
        del raw["starting_positions"]
        restored = PaperTradingSpec.from_json_dict(raw)
        assert restored.starting_positions == ()
        assert compute_paper_session_spec_id(explicit).paper_session_spec_id == compute_paper_session_spec_id(restored).paper_session_spec_id
