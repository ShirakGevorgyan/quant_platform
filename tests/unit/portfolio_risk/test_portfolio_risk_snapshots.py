"""Unit tests for `portfolio_risk.snapshots`: `PriceSnapshot`,
`PositionSnapshot`, `PortfolioSnapshot`, and `ExposureSnapshot`."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import (
    PortfolioSnapshotValidationError,
    StalePortfolioSnapshotError,
    StalePriceError,
)
from quant_platform.portfolio_risk.models import OrderSide
from quant_platform.portfolio_risk.snapshots import (
    ExposureScopeKind,
    ExposureSnapshot,
    PortfolioSnapshot,
    PositionSnapshot,
    PriceSnapshot,
    compute_instrument_exposure,
    compute_portfolio_exposure,
    compute_strategy_exposure,
    create_portfolio_snapshot,
    create_price_snapshot,
    is_portfolio_snapshot_stale,
    is_price_stale,
)

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _price(**overrides: object) -> PriceSnapshot:
    base: dict[str, object] = {
        "instrument_id": "EURUSD", "bid": Decimal("1.1000"), "ask": Decimal("1.1002"), "reference_price": Decimal("1.1001"),
        "event_time": _T0, "source_event_id": "market-event-1",
    }
    base.update(overrides)
    return create_price_snapshot(**base)  # type: ignore[arg-type]


def _position(**overrides: object) -> PositionSnapshot:
    base: dict[str, object] = {
        "instrument_id": "EURUSD", "strategy_id": "strategy-a", "side": OrderSide.BUY, "quantity": Decimal("1000"),
        "average_entry_price": Decimal("1.1000"), "mark_price": Decimal("1.1050"), "unrealized_pnl": Decimal("5.00"),
        "realized_pnl": Decimal("0"), "contract_multiplier": Decimal("1"),
    }
    base.update(overrides)
    return PositionSnapshot(**base)  # type: ignore[arg-type]


def _portfolio(*, positions: tuple[PositionSnapshot, ...] = (), cash: Decimal = Decimal("100000"), **overrides: object) -> PortfolioSnapshot:
    marked_value = sum((p.market_value for p in positions), start=Decimal(0))
    unrealized = sum((p.unrealized_pnl for p in positions), start=Decimal(0))
    base: dict[str, object] = {
        "portfolio_id": "portfolio-1", "event_time": _T0, "cash": cash, "equity": cash + marked_value, "realized_pnl": Decimal("0"),
        "unrealized_pnl": unrealized, "peak_equity": cash + marked_value, "daily_start_equity": cash, "positions": positions,
        "source_execution_session_id": None,
    }
    base.update(overrides)
    return create_portfolio_snapshot(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# PriceSnapshot
# --------------------------------------------------------------------------
class TestPriceSnapshotValidConstruction:
    def test_default_constructs(self) -> None:
        price = _price()
        assert price.instrument_id == "EURUSD"
        assert len(price.price_snapshot_id) == 64

    def test_source_event_id_may_be_none(self) -> None:
        price = _price(source_event_id=None)
        assert price.source_event_id is None


class TestPriceSnapshotInvariants:
    def test_bid_greater_than_ask_rejected(self) -> None:
        with pytest.raises(StalePriceError):
            _price(bid=Decimal("1.2"), ask=Decimal("1.1"))

    def test_bid_equal_ask_accepted(self) -> None:
        price = _price(bid=Decimal("1.10"), ask=Decimal("1.10"))
        assert price.bid == price.ask

    @pytest.mark.parametrize("field_name", ["bid", "ask", "reference_price"])
    def test_non_positive_price_rejected(self, field_name: str) -> None:
        with pytest.raises(StalePriceError):
            _price(**{field_name: Decimal("0")})
        with pytest.raises(StalePriceError):
            _price(**{field_name: Decimal("-1")})

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(StalePriceError):
            _price(event_time=datetime(2026, 1, 1))

    def test_empty_instrument_id_rejected(self) -> None:
        with pytest.raises(StalePriceError):
            _price(instrument_id="")

    def test_empty_source_event_id_string_rejected(self) -> None:
        with pytest.raises(StalePriceError):
            _price(source_event_id="")

    def test_reference_price_is_never_auto_derived(self) -> None:
        # A reference price far outside [bid, ask] is still accepted --
        # this class does not silently override an explicitly-supplied
        # reference price with (bid+ask)/2 or any other derivation.
        price = _price(bid=Decimal("1.10"), ask=Decimal("1.11"), reference_price=Decimal("2.00"))
        assert price.reference_price == Decimal("2.00")


class TestPriceSnapshotIdentity:
    def test_deterministic(self) -> None:
        a = _price().price_snapshot_id
        b = _price().price_snapshot_id
        assert a == b

    def test_source_event_id_participates_in_identity(self) -> None:
        a = _price(source_event_id="event-a").price_snapshot_id
        b = _price(source_event_id="event-b").price_snapshot_id
        assert a != b

    def test_event_time_participates_in_identity(self) -> None:
        a = _price(event_time=_T0).price_snapshot_id
        b = _price(event_time=_T0 + timedelta(seconds=1)).price_snapshot_id
        assert a != b

    def test_reference_price_participates_in_identity(self) -> None:
        a = _price(reference_price=Decimal("1.1001")).price_snapshot_id
        b = _price(reference_price=Decimal("1.1002")).price_snapshot_id
        assert a != b


class TestPriceSnapshotRoundTrip:
    def test_round_trips_through_json(self) -> None:
        price = _price()
        restored = PriceSnapshot.from_json_dict(price.to_json_dict())
        assert restored.to_json_dict() == price.to_json_dict()


class TestPriceStaleness:
    def test_not_configured_never_stale(self) -> None:
        price = _price(event_time=_T0)
        assert is_price_stale(price, reference_time=_T0 + timedelta(days=999), maximum_age_seconds=None) is False

    def test_within_bound_not_stale(self) -> None:
        price = _price(event_time=_T0)
        assert is_price_stale(price, reference_time=_T0 + timedelta(seconds=10), maximum_age_seconds=30) is False

    def test_exactly_at_bound_not_stale(self) -> None:
        price = _price(event_time=_T0)
        assert is_price_stale(price, reference_time=_T0 + timedelta(seconds=30), maximum_age_seconds=30) is False

    def test_beyond_bound_is_stale(self) -> None:
        price = _price(event_time=_T0)
        assert is_price_stale(price, reference_time=_T0 + timedelta(seconds=31), maximum_age_seconds=30) is True

    def test_reference_time_before_event_time_raises(self) -> None:
        price = _price(event_time=_T0)
        with pytest.raises(StalePriceError):
            is_price_stale(price, reference_time=_T0 - timedelta(seconds=1), maximum_age_seconds=30)

    def test_naive_reference_time_rejected(self) -> None:
        price = _price(event_time=_T0)
        with pytest.raises(StalePriceError):
            is_price_stale(price, reference_time=datetime(2026, 1, 1), maximum_age_seconds=30)


# --------------------------------------------------------------------------
# PositionSnapshot
# --------------------------------------------------------------------------
class TestPositionSnapshotInvariants:
    def test_default_constructs(self) -> None:
        position = _position()
        assert position.signed_quantity == Decimal("1000")

    def test_short_side_negates_signed_quantity(self) -> None:
        position = _position(side=OrderSide.SELL, quantity=Decimal("1000"), average_entry_price=Decimal("1.1050"), mark_price=Decimal("1.1000"), unrealized_pnl=Decimal("5.00"))
        assert position.signed_quantity == Decimal("-1000")

    def test_non_positive_quantity_rejected(self) -> None:
        with pytest.raises(PortfolioSnapshotValidationError):
            _position(quantity=Decimal("0"))
        with pytest.raises(PortfolioSnapshotValidationError):
            _position(quantity=Decimal("-1"))

    def test_unrealized_pnl_must_reconcile_with_mark_and_entry(self) -> None:
        # 1000 * (1.1050 - 1.1000) * 1 == 5.00 (correct, from _position's
        # own defaults); corrupting it must be rejected.
        with pytest.raises(PortfolioSnapshotValidationError):
            _position(unrealized_pnl=Decimal("999"))

    def test_unrealized_pnl_reconciles_for_short_position(self) -> None:
        # -1000 * (1.0950 - 1.1000) * 1 == 5.00
        position = _position(side=OrderSide.SELL, average_entry_price=Decimal("1.1000"), mark_price=Decimal("1.0950"), unrealized_pnl=Decimal("5.00"))
        assert position.unrealized_pnl == Decimal("5.00")

    def test_market_value_uses_signed_quantity(self) -> None:
        position = _position(
            quantity=Decimal("1000"), average_entry_price=Decimal("1.10"), mark_price=Decimal("1.10"), unrealized_pnl=Decimal("0"),
            contract_multiplier=Decimal("1"),
        )
        assert position.market_value == Decimal("1000") * Decimal("1.10")

    def test_empty_instrument_id_rejected(self) -> None:
        with pytest.raises(PortfolioSnapshotValidationError):
            _position(instrument_id="")

    def test_empty_strategy_id_rejected(self) -> None:
        with pytest.raises(PortfolioSnapshotValidationError):
            _position(strategy_id="")

    def test_non_positive_contract_multiplier_rejected(self) -> None:
        with pytest.raises(PortfolioSnapshotValidationError):
            _position(contract_multiplier=Decimal("0"))


class TestPositionSnapshotRoundTrip:
    def test_round_trips_through_json(self) -> None:
        position = _position()
        restored = PositionSnapshot.from_json_dict(position.to_json_dict())
        assert restored.to_json_dict() == position.to_json_dict()


# --------------------------------------------------------------------------
# PortfolioSnapshot
# --------------------------------------------------------------------------
class TestPortfolioSnapshotAccountingReconciliation:
    def test_flat_portfolio_equity_equals_cash(self) -> None:
        portfolio = _portfolio(positions=())
        assert portfolio.equity == portfolio.cash

    def test_equity_reconciles_with_cash_plus_marked_position_value(self) -> None:
        position = _position()
        portfolio = _portfolio(positions=(position,))
        assert portfolio.equity == portfolio.cash + position.market_value

    def test_equity_not_reconciling_is_rejected(self) -> None:
        position = _position()
        with pytest.raises(PortfolioSnapshotValidationError):
            _portfolio(positions=(position,), equity=Decimal("1"))

    def test_unrealized_pnl_must_equal_sum_of_open_positions(self) -> None:
        position = _position()
        with pytest.raises(PortfolioSnapshotValidationError):
            _portfolio(positions=(position,), unrealized_pnl=Decimal("999"))

    def test_realized_pnl_is_not_cross_validated_against_positions(self) -> None:
        # realized_pnl reflects history from positions that may have
        # since gone flat and been removed -- it is independently
        # trusted, never derived from currently-open positions.
        position = _position(realized_pnl=Decimal("0"))
        portfolio = _portfolio(positions=(position,), realized_pnl=Decimal("12345.67"))
        assert portfolio.realized_pnl == Decimal("12345.67")

    def test_peak_equity_below_equity_rejected(self) -> None:
        with pytest.raises(PortfolioSnapshotValidationError):
            _portfolio(peak_equity=Decimal("1"), equity=Decimal("100000"))

    def test_peak_equity_equal_to_equity_accepted(self) -> None:
        portfolio = _portfolio(peak_equity=Decimal("100000"), equity=Decimal("100000"))
        assert portfolio.peak_equity == portfolio.equity

    def test_drawdown_fraction_is_derived_not_stored(self) -> None:
        portfolio = _portfolio(cash=Decimal("100000"), peak_equity=Decimal("120000"), equity=Decimal("100000"), unrealized_pnl=Decimal("0"))
        assert portfolio.drawdown_fraction == (Decimal("120000") - Decimal("100000")) / Decimal("120000")

    def test_drawdown_fraction_zero_when_at_peak(self) -> None:
        portfolio = _portfolio(peak_equity=Decimal("100000"), equity=Decimal("100000"))
        assert portfolio.drawdown_fraction == Decimal("0")

    def test_naive_event_time_rejected(self) -> None:
        with pytest.raises(PortfolioSnapshotValidationError):
            _portfolio(event_time=datetime(2026, 1, 1))

    def test_empty_portfolio_id_rejected(self) -> None:
        with pytest.raises(PortfolioSnapshotValidationError):
            _portfolio(portfolio_id="")


class TestPortfolioSnapshotDuplicatePositions:
    def test_two_distinct_identities_accepted(self) -> None:
        a = _position(instrument_id="EURUSD", strategy_id="strategy-a")
        b = _position(instrument_id="EURUSD", strategy_id="strategy-b")
        portfolio = _portfolio(positions=(a, b))
        assert len(portfolio.positions) == 2

    def test_duplicate_instrument_and_strategy_identity_rejected(self) -> None:
        a = _position(instrument_id="EURUSD", strategy_id="strategy-a", unrealized_pnl=Decimal("5.00"))
        b = _position(instrument_id="EURUSD", strategy_id="strategy-a", unrealized_pnl=Decimal("5.00"))
        with pytest.raises(PortfolioSnapshotValidationError):
            _portfolio(positions=(a, b), unrealized_pnl=Decimal("10.00"))

    def test_same_instrument_different_strategy_is_not_a_duplicate(self) -> None:
        a = _position(instrument_id="EURUSD", strategy_id="strategy-a")
        b = _position(instrument_id="EURUSD", strategy_id="strategy-b")
        # Constructs without raising -- confirms distinct strategy_id
        # alone is sufficient disambiguation.
        assert _portfolio(positions=(a, b)) is not None

    def test_position_for_looks_up_by_composite_identity(self) -> None:
        a = _position(instrument_id="EURUSD", strategy_id="strategy-a")
        portfolio = _portfolio(positions=(a,))
        assert portfolio.position_for(instrument_id="EURUSD", strategy_id="strategy-a") is a
        assert portfolio.position_for(instrument_id="EURUSD", strategy_id="strategy-b") is None


class TestPortfolioSnapshotRoundTrip:
    def test_round_trips_through_json_with_positions(self) -> None:
        position = _position()
        portfolio = _portfolio(positions=(position,))
        restored = PortfolioSnapshot.from_json_dict(portfolio.to_json_dict())
        assert restored.to_json_dict() == portfolio.to_json_dict()


class TestPortfolioSnapshotIdentity:
    def test_deterministic(self) -> None:
        a = _portfolio().snapshot_id
        b = _portfolio().snapshot_id
        assert a == b

    def test_canonical_ordering_of_positions(self) -> None:
        a = _position(instrument_id="AAA", strategy_id="s1")
        b = _position(instrument_id="BBB", strategy_id="s1")
        forward = _portfolio(positions=(a, b))
        backward = _portfolio(positions=(b, a))
        assert forward.snapshot_id == backward.snapshot_id

    def test_changing_a_position_changes_identity(self) -> None:
        a = _position(instrument_id="EURUSD", strategy_id="strategy-a", quantity=Decimal("1000"))
        b = _position(instrument_id="EURUSD", strategy_id="strategy-a", quantity=Decimal("2000"), unrealized_pnl=Decimal("10.00"))
        cash = Decimal("100000")
        id_a = _portfolio(positions=(a,), cash=cash).snapshot_id
        id_b = _portfolio(positions=(b,), cash=cash).snapshot_id
        assert id_a != id_b

    def test_source_execution_session_id_participates_in_identity(self) -> None:
        sha_a = "a" * 64
        sha_b = "b" * 64
        id_a = _portfolio(source_execution_session_id=sha_a).snapshot_id
        id_b = _portfolio(source_execution_session_id=sha_b).snapshot_id
        assert id_a != id_b

    def test_source_execution_session_id_must_be_sha256_when_present(self) -> None:
        with pytest.raises(PortfolioSnapshotValidationError):
            _portfolio(source_execution_session_id="not-a-hash")


class TestPortfolioSnapshotStaleness:
    def test_not_configured_never_stale(self) -> None:
        portfolio = _portfolio(event_time=_T0)
        assert is_portfolio_snapshot_stale(portfolio, reference_time=_T0 + timedelta(days=999), maximum_age_seconds=None) is False

    def test_beyond_bound_is_stale(self) -> None:
        portfolio = _portfolio(event_time=_T0)
        assert is_portfolio_snapshot_stale(portfolio, reference_time=_T0 + timedelta(seconds=61), maximum_age_seconds=60) is True

    def test_within_bound_not_stale(self) -> None:
        portfolio = _portfolio(event_time=_T0)
        assert is_portfolio_snapshot_stale(portfolio, reference_time=_T0 + timedelta(seconds=59), maximum_age_seconds=60) is False

    def test_reference_time_before_event_time_raises(self) -> None:
        portfolio = _portfolio(event_time=_T0)
        with pytest.raises(StalePortfolioSnapshotError):
            is_portfolio_snapshot_stale(portfolio, reference_time=_T0 - timedelta(seconds=1), maximum_age_seconds=60)


# --------------------------------------------------------------------------
# ExposureSnapshot
# --------------------------------------------------------------------------
class TestExposureSnapshotDerivation:
    def test_gross_exposure_is_never_negative(self) -> None:
        long_position = _position(instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("1000"))
        short_position = _position(instrument_id="GBPUSD", strategy_id="s1", side=OrderSide.SELL, quantity=Decimal("500"), average_entry_price=Decimal("1.30"), mark_price=Decimal("1.31"), unrealized_pnl=Decimal("-5.00"))
        portfolio = _portfolio(positions=(long_position, short_position), cash=Decimal("100000"))
        exposure = compute_portfolio_exposure(portfolio)
        assert exposure.gross_exposure >= 0

    def test_gross_exposure_sums_absolute_values_net_can_partially_offset(self) -> None:
        long_position = _position(instrument_id="EURUSD", strategy_id="s1", side=OrderSide.BUY, quantity=Decimal("1000"), mark_price=Decimal("1.10"), average_entry_price=Decimal("1.10"), unrealized_pnl=Decimal("0"))
        short_position = _position(instrument_id="GBPUSD", strategy_id="s1", side=OrderSide.SELL, quantity=Decimal("1000"), mark_price=Decimal("1.10"), average_entry_price=Decimal("1.10"), unrealized_pnl=Decimal("0"))
        portfolio = _portfolio(positions=(long_position, short_position), cash=Decimal("100000"))
        exposure = compute_portfolio_exposure(portfolio)
        assert exposure.gross_exposure == Decimal("1000") * Decimal("1.10") * 2
        assert exposure.net_exposure == Decimal("0")

    def test_instrument_scope_only_includes_matching_instrument(self) -> None:
        a = _position(instrument_id="EURUSD", strategy_id="s1")
        b = _position(instrument_id="GBPUSD", strategy_id="s1", average_entry_price=Decimal("1.30"), mark_price=Decimal("1.30"), unrealized_pnl=Decimal("0"))
        portfolio = _portfolio(positions=(a, b), cash=Decimal("100000"))
        exposure = compute_instrument_exposure(portfolio, instrument_id="EURUSD")
        assert exposure.scope_kind is ExposureScopeKind.INSTRUMENT
        assert exposure.gross_exposure == abs(a.market_value)

    def test_strategy_scope_only_includes_matching_strategy(self) -> None:
        a = _position(instrument_id="EURUSD", strategy_id="strategy-a")
        b = _position(instrument_id="GBPUSD", strategy_id="strategy-b", average_entry_price=Decimal("1.30"), mark_price=Decimal("1.30"), unrealized_pnl=Decimal("0"))
        portfolio = _portfolio(positions=(a, b), cash=Decimal("100000"))
        exposure = compute_strategy_exposure(portfolio, strategy_id="strategy-a")
        assert exposure.gross_exposure == abs(a.market_value)

    def test_empty_portfolio_has_zero_exposure(self) -> None:
        exposure = compute_portfolio_exposure(_portfolio(positions=()))
        assert exposure.gross_exposure == Decimal("0")
        assert exposure.net_exposure == Decimal("0")

    def test_net_exposure_cannot_exceed_gross_exposure_construction_invariant(self) -> None:
        with pytest.raises(PortfolioSnapshotValidationError):
            ExposureSnapshot(scope_kind=ExposureScopeKind.PORTFOLIO, scope_id=None, gross_exposure=Decimal("10"), net_exposure=Decimal("20"))

    def test_negative_gross_exposure_rejected(self) -> None:
        with pytest.raises(PortfolioSnapshotValidationError):
            ExposureSnapshot(scope_kind=ExposureScopeKind.PORTFOLIO, scope_id=None, gross_exposure=Decimal("-1"), net_exposure=Decimal("0"))

    def test_portfolio_scope_requires_none_scope_id(self) -> None:
        with pytest.raises(PortfolioSnapshotValidationError):
            ExposureSnapshot(scope_kind=ExposureScopeKind.PORTFOLIO, scope_id="something", gross_exposure=Decimal("0"), net_exposure=Decimal("0"))

    def test_instrument_scope_requires_non_empty_scope_id(self) -> None:
        with pytest.raises(PortfolioSnapshotValidationError):
            ExposureSnapshot(scope_kind=ExposureScopeKind.INSTRUMENT, scope_id=None, gross_exposure=Decimal("0"), net_exposure=Decimal("0"))
