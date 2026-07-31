"""Portfolio and price snapshot models for `quant_platform.portfolio_risk`
(Milestone 9, Phase 1). Every timestamp here is CALLER-SUPPLIED -- there
is no internal `datetime.now()`/`utc_now()` call anywhere in this module,
and no economic or staleness decision ever depends on one (Phase 1's own
explicit safety scope).

EXACT RECONCILIATION FORMULA (mirrors `paper_trading.portfolio`'s own
documented formula, Milestone 7 Section 13, adapted to this package's
point-in-time snapshot shape and restated here verbatim for this
package's own accounting model):

    equity = cash + sum(position.market_value for position in positions)

`positions` holds only NON-FLAT positions (Phase 1's own documented
simplification: a flat/closed position simply is not represented, unlike
`paper_trading.accounting.PositionState`, which retains a flat position's
accumulated history). There is no separate `liabilities`/`accrued_costs`
term in Phase 1 -- a fully cash-settled model with no transaction-cost
line item, the same documented simplification `paper_trading.portfolio`
and `execution_gateway.dispatcher` already carry (see each package's own
architecture doc). `unrealized_pnl` at the portfolio level MUST equal the
sum of every open position's own `unrealized_pnl` (unrealized pnl only
exists for currently-open positions, so this IS always fully derivable).
`realized_pnl` at the portfolio level is NOT cross-validated against
`positions` -- realized pnl accrues from CLOSED positions too, which are
not retained in this point-in-time snapshot's `positions` collection, so
the portfolio-level figure is independently caller-supplied, trusted
history, not a derived quantity.

`drawdown_fraction` is deliberately a DERIVED PROPERTY, never a stored
field -- Section requirement: "drawdown must never be independently
trusted when it can be derived." The same principle applies to exposure:
`ExposureSnapshot` is always computed via `compute_portfolio_exposure`/
`compute_instrument_exposure`/`compute_strategy_exposure`, never stored
redundantly on `PortfolioSnapshot` itself."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

import pandas as pd

from quant_platform.core.exceptions import (
    PortfolioSnapshotValidationError,
    StalePortfolioSnapshotError,
    StalePriceError,
)
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp
from quant_platform.portfolio_risk.identity import (
    compute_content_id,
    decimal_to_json,
    is_valid_sha256_hex,
    parse_decimal,
)
from quant_platform.portfolio_risk.models import OrderSide

PORTFOLIO_SNAPSHOT_KIND = "portfolio_snapshot"
PRICE_SNAPSHOT_KIND = "price_snapshot"


def _require_tz_aware(ts: datetime, *, field_name: str, error_cls: type[Exception] = PortfolioSnapshotValidationError) -> None:
    if ts.tzinfo is None:
        raise error_cls(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _serialize_timestamp(ts: datetime, *, field_name: str, error_cls: type[Exception] = PortfolioSnapshotValidationError) -> str:
    _require_tz_aware(ts, field_name=field_name, error_cls=error_cls)
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise error_cls(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str, error_cls: type[Exception] = PortfolioSnapshotValidationError) -> datetime:
    if not isinstance(value, str):
        raise error_cls(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise error_cls(f"{field_name}: {exc}") from exc


def _require_sha256(value: str, *, field_name: str, error_cls: type[Exception] = PortfolioSnapshotValidationError) -> None:
    if not is_valid_sha256_hex(value):
        raise error_cls(f"{field_name} must be a 64-character lowercase hex SHA-256 digest, got {value!r}")


def _positive_decimal(value: Decimal, *, field_name: str, error_cls: type[Exception] = PortfolioSnapshotValidationError) -> None:
    if not value.is_finite() or value <= 0:
        raise error_cls(f"{field_name} must be finite and > 0, got {value!r}")


def _finite_decimal(value: Decimal, *, field_name: str, error_cls: type[Exception] = PortfolioSnapshotValidationError) -> None:
    if not value.is_finite():
        raise error_cls(f"{field_name} must be finite, got {value!r}")


# --------------------------------------------------------------------------
# PriceSnapshot
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PriceSnapshot:
    price_snapshot_id: str
    instrument_id: str
    bid: Decimal
    ask: Decimal
    reference_price: Decimal
    """Always EXPLICITLY supplied -- never auto-derived as `(bid + ask) /
    2` or similar. A caller may choose to pass the midpoint, the last
    trade price, or any other economically meaningful reference; this
    class does not assume one."""
    event_time: datetime
    source_event_id: str | None

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise StalePriceError("PriceSnapshot.instrument_id must not be empty")
        for field_name, value in (("bid", self.bid), ("ask", self.ask), ("reference_price", self.reference_price)):
            _positive_decimal(value, field_name=f"PriceSnapshot.{field_name}", error_cls=StalePriceError)
        if self.bid > self.ask:
            raise StalePriceError(f"PriceSnapshot.bid ({self.bid!r}) must be <= PriceSnapshot.ask ({self.ask!r})")
        _require_tz_aware(self.event_time, field_name="PriceSnapshot.event_time", error_cls=StalePriceError)
        if self.source_event_id is not None and not self.source_event_id:
            raise StalePriceError("PriceSnapshot.source_event_id must not be an empty string when present")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "price_snapshot_id": self.price_snapshot_id, "instrument_id": self.instrument_id, "bid": decimal_to_json(self.bid),
            "ask": decimal_to_json(self.ask), "reference_price": decimal_to_json(self.reference_price),
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time", error_cls=StalePriceError),
            "source_event_id": self.source_event_id,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["price_snapshot_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PriceSnapshot:
        return cls(
            price_snapshot_id=str(raw["price_snapshot_id"]), instrument_id=str(raw["instrument_id"]),
            bid=parse_decimal(raw["bid"], field_name="bid"), ask=parse_decimal(raw["ask"], field_name="ask"),
            reference_price=parse_decimal(raw["reference_price"], field_name="reference_price"),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time", error_cls=StalePriceError),
            source_event_id=(None if raw.get("source_event_id") is None else str(raw["source_event_id"])),
        )


def create_price_snapshot(
    *, instrument_id: str, bid: Decimal, ask: Decimal, reference_price: Decimal, event_time: datetime, source_event_id: str | None,
) -> PriceSnapshot:
    provisional = PriceSnapshot(
        price_snapshot_id="0" * 64, instrument_id=instrument_id, bid=bid, ask=ask, reference_price=reference_price, event_time=event_time,
        source_event_id=source_event_id,
    )
    price_snapshot_id = compute_content_id(PRICE_SNAPSHOT_KIND, provisional.to_identity_payload())
    return PriceSnapshot(
        price_snapshot_id=price_snapshot_id, instrument_id=instrument_id, bid=bid, ask=ask, reference_price=reference_price,
        event_time=event_time, source_event_id=source_event_id,
    )


def is_price_stale(price: PriceSnapshot, *, reference_time: datetime, maximum_age_seconds: int | None) -> bool:
    """`maximum_age_seconds=None` means "not configured" -- never
    considered stale by this check (mirrors `PortfolioRiskPolicy`'s own
    "`None` = not checked" convention). Comparison is exact `timedelta`
    arithmetic -- no float duration is ever computed."""
    if maximum_age_seconds is None:
        return False
    _require_tz_aware(reference_time, field_name="reference_time", error_cls=StalePriceError)
    age = reference_time - price.event_time
    if age < timedelta(0):
        raise StalePriceError(f"reference_time {reference_time!r} precedes PriceSnapshot.event_time {price.event_time!r}")
    return age > timedelta(seconds=maximum_age_seconds)


# --------------------------------------------------------------------------
# PositionSnapshot -- nested value object, no independent content id
# (identified implicitly by its (instrument_id, strategy_id) position
# within the parent PortfolioSnapshot.positions collection, exactly as
# `paper_trading.accounting.PositionState` has no independent id either).
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    instrument_id: str
    strategy_id: str
    side: OrderSide
    quantity: Decimal
    """Always the POSITIVE magnitude -- direction lives in `side`. A flat
    position is not represented at all (Phase 1's documented
    simplification -- see module docstring); `quantity` must be > 0."""
    average_entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    contract_multiplier: Decimal

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise PortfolioSnapshotValidationError("PositionSnapshot.instrument_id must not be empty")
        if not self.strategy_id:
            raise PortfolioSnapshotValidationError("PositionSnapshot.strategy_id must not be empty")
        for field_name, value in (
            ("quantity", self.quantity), ("average_entry_price", self.average_entry_price), ("mark_price", self.mark_price),
            ("contract_multiplier", self.contract_multiplier),
        ):
            _positive_decimal(value, field_name=f"PositionSnapshot.{field_name}")
        for field_name, value in (("unrealized_pnl", self.unrealized_pnl), ("realized_pnl", self.realized_pnl)):
            _finite_decimal(value, field_name=f"PositionSnapshot.{field_name}")
        expected_unrealized_pnl = self.signed_quantity * (self.mark_price - self.average_entry_price) * self.contract_multiplier
        if self.unrealized_pnl != expected_unrealized_pnl:
            raise PortfolioSnapshotValidationError(
                f"PositionSnapshot.unrealized_pnl ({self.unrealized_pnl!r}) does not reconcile with signed_quantity * "
                f"(mark_price - average_entry_price) * contract_multiplier ({expected_unrealized_pnl!r}) for instrument "
                f"{self.instrument_id!r}/strategy {self.strategy_id!r}"
            )

    @property
    def signed_quantity(self) -> Decimal:
        return self.quantity if self.side is OrderSide.BUY else -self.quantity

    @property
    def market_value(self) -> Decimal:
        return self.signed_quantity * self.mark_price * self.contract_multiplier

    def to_json_dict(self) -> dict[str, object]:
        return {
            "instrument_id": self.instrument_id, "strategy_id": self.strategy_id, "side": self.side.value,
            "quantity": decimal_to_json(self.quantity), "average_entry_price": decimal_to_json(self.average_entry_price),
            "mark_price": decimal_to_json(self.mark_price), "unrealized_pnl": decimal_to_json(self.unrealized_pnl),
            "realized_pnl": decimal_to_json(self.realized_pnl), "contract_multiplier": decimal_to_json(self.contract_multiplier),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PositionSnapshot:
        return cls(
            instrument_id=str(raw["instrument_id"]), strategy_id=str(raw["strategy_id"]), side=OrderSide(raw["side"]),
            quantity=parse_decimal(raw["quantity"], field_name="quantity"),
            average_entry_price=parse_decimal(raw["average_entry_price"], field_name="average_entry_price"),
            mark_price=parse_decimal(raw["mark_price"], field_name="mark_price"),
            unrealized_pnl=parse_decimal(raw["unrealized_pnl"], field_name="unrealized_pnl"),
            realized_pnl=parse_decimal(raw["realized_pnl"], field_name="realized_pnl"),
            contract_multiplier=parse_decimal(raw["contract_multiplier"], field_name="contract_multiplier"),
        )


# --------------------------------------------------------------------------
# PortfolioSnapshot
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    portfolio_id: str
    snapshot_id: str
    event_time: datetime
    cash: Decimal
    equity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    peak_equity: Decimal
    daily_start_equity: Decimal
    positions: tuple[PositionSnapshot, ...]
    source_execution_session_id: str | None

    def __post_init__(self) -> None:
        if not self.portfolio_id:
            raise PortfolioSnapshotValidationError("PortfolioSnapshot.portfolio_id must not be empty")
        _require_tz_aware(self.event_time, field_name="PortfolioSnapshot.event_time")
        for field_name, value in (
            ("cash", self.cash), ("equity", self.equity), ("realized_pnl", self.realized_pnl), ("unrealized_pnl", self.unrealized_pnl),
            ("peak_equity", self.peak_equity), ("daily_start_equity", self.daily_start_equity),
        ):
            _finite_decimal(value, field_name=f"PortfolioSnapshot.{field_name}")
        if self.peak_equity < self.equity:
            raise PortfolioSnapshotValidationError(f"PortfolioSnapshot.peak_equity ({self.peak_equity!r}) must be >= equity ({self.equity!r})")

        seen_identities: set[tuple[str, str]] = set()
        for position in self.positions:
            identity = (position.instrument_id, position.strategy_id)
            if identity in seen_identities:
                raise PortfolioSnapshotValidationError(
                    f"PortfolioSnapshot.positions contains a duplicate (instrument_id, strategy_id) identity: {identity!r}"
                )
            seen_identities.add(identity)

        expected_unrealized_pnl = sum((p.unrealized_pnl for p in self.positions), start=Decimal(0))
        if self.unrealized_pnl != expected_unrealized_pnl:
            raise PortfolioSnapshotValidationError(
                f"PortfolioSnapshot.unrealized_pnl ({self.unrealized_pnl!r}) does not equal the sum of every open "
                f"position's own unrealized_pnl ({expected_unrealized_pnl!r})"
            )

        marked_position_value = sum((p.market_value for p in self.positions), start=Decimal(0))
        expected_equity = self.cash + marked_position_value
        if self.equity != expected_equity:
            raise PortfolioSnapshotValidationError(
                f"PortfolioSnapshot.equity ({self.equity!r}) does not reconcile with cash + marked position value "
                f"({expected_equity!r}) -- see module docstring for the exact reconciliation formula"
            )

        if self.source_execution_session_id is not None:
            _require_sha256(self.source_execution_session_id, field_name="PortfolioSnapshot.source_execution_session_id")

    def position_for(self, *, instrument_id: str, strategy_id: str) -> PositionSnapshot | None:
        for position in self.positions:
            if position.instrument_id == instrument_id and position.strategy_id == strategy_id:
                return position
        return None

    @property
    def drawdown_fraction(self) -> Decimal:
        """Always DERIVED, never independently stored/trusted (Section
        requirement). Mirrors `paper_trading.portfolio.PortfolioState.
        drawdown_fraction`'s identical formula and zero-guard."""
        if self.peak_equity <= 0:
            return Decimal(0)
        return max(Decimal(0), (self.peak_equity - self.equity) / self.peak_equity)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "portfolio_id": self.portfolio_id, "snapshot_id": self.snapshot_id,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "cash": decimal_to_json(self.cash),
            "equity": decimal_to_json(self.equity), "realized_pnl": decimal_to_json(self.realized_pnl),
            "unrealized_pnl": decimal_to_json(self.unrealized_pnl), "peak_equity": decimal_to_json(self.peak_equity),
            "daily_start_equity": decimal_to_json(self.daily_start_equity),
            "positions": [p.to_json_dict() for p in self.positions], "source_execution_session_id": self.source_execution_session_id,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["snapshot_id"]
        # `positions` is a genuinely UNORDERED collection from a domain
        # perspective -- two snapshots holding the same set of positions
        # built in different construction order must produce the SAME
        # identity. Canonical order: sorted by (instrument_id, strategy_id),
        # exactly mirroring `DummyBrokerScenarioSpec.to_identity_payload`'s
        # sort-by-key convention.
        payload["positions"] = [p.to_json_dict() for p in sorted(self.positions, key=lambda p: (p.instrument_id, p.strategy_id))]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PortfolioSnapshot:
        from quant_platform.ml.persistence import as_json_dict, as_json_list

        positions_raw = as_json_list(raw.get("positions") or [], field_name="positions")
        return cls(
            portfolio_id=str(raw["portfolio_id"]), snapshot_id=str(raw["snapshot_id"]),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), cash=parse_decimal(raw["cash"], field_name="cash"),
            equity=parse_decimal(raw["equity"], field_name="equity"), realized_pnl=parse_decimal(raw["realized_pnl"], field_name="realized_pnl"),
            unrealized_pnl=parse_decimal(raw["unrealized_pnl"], field_name="unrealized_pnl"),
            peak_equity=parse_decimal(raw["peak_equity"], field_name="peak_equity"),
            daily_start_equity=parse_decimal(raw["daily_start_equity"], field_name="daily_start_equity"),
            positions=tuple(PositionSnapshot.from_json_dict(as_json_dict(p, field_name="positions[]")) for p in positions_raw),
            source_execution_session_id=(None if raw.get("source_execution_session_id") is None else str(raw["source_execution_session_id"])),
        )


def create_portfolio_snapshot(
    *, portfolio_id: str, event_time: datetime, cash: Decimal, equity: Decimal, realized_pnl: Decimal, unrealized_pnl: Decimal,
    peak_equity: Decimal, daily_start_equity: Decimal, positions: tuple[PositionSnapshot, ...], source_execution_session_id: str | None,
) -> PortfolioSnapshot:
    provisional = PortfolioSnapshot(
        portfolio_id=portfolio_id, snapshot_id="0" * 64, event_time=event_time, cash=cash, equity=equity, realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl, peak_equity=peak_equity, daily_start_equity=daily_start_equity, positions=positions,
        source_execution_session_id=source_execution_session_id,
    )
    snapshot_id = compute_content_id(PORTFOLIO_SNAPSHOT_KIND, provisional.to_identity_payload())
    return PortfolioSnapshot(
        portfolio_id=portfolio_id, snapshot_id=snapshot_id, event_time=event_time, cash=cash, equity=equity, realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl, peak_equity=peak_equity, daily_start_equity=daily_start_equity, positions=positions,
        source_execution_session_id=source_execution_session_id,
    )


def is_portfolio_snapshot_stale(snapshot: PortfolioSnapshot, *, reference_time: datetime, maximum_age_seconds: int | None) -> bool:
    if maximum_age_seconds is None:
        return False
    _require_tz_aware(reference_time, field_name="reference_time", error_cls=StalePortfolioSnapshotError)
    age = reference_time - snapshot.event_time
    if age < timedelta(0):
        raise StalePortfolioSnapshotError(f"reference_time {reference_time!r} precedes PortfolioSnapshot.event_time {snapshot.event_time!r}")
    return age > timedelta(seconds=maximum_age_seconds)


# --------------------------------------------------------------------------
# ExposureSnapshot -- always DERIVED, never stored on PortfolioSnapshot.
# --------------------------------------------------------------------------
class ExposureScopeKind(Enum):
    INSTRUMENT = "instrument"
    STRATEGY = "strategy"
    PORTFOLIO = "portfolio"


@dataclass(frozen=True, slots=True)
class ExposureSnapshot:
    scope_kind: ExposureScopeKind
    scope_id: str | None
    """The `instrument_id`/`strategy_id` this exposure is scoped to; `None`
    when `scope_kind is PORTFOLIO`."""
    gross_exposure: Decimal
    net_exposure: Decimal

    def __post_init__(self) -> None:
        if self.scope_kind is ExposureScopeKind.PORTFOLIO:
            if self.scope_id is not None:
                raise PortfolioSnapshotValidationError("ExposureSnapshot.scope_id must be None when scope_kind is PORTFOLIO")
        elif not self.scope_id:
            raise PortfolioSnapshotValidationError(f"ExposureSnapshot.scope_id must not be empty when scope_kind is {self.scope_kind.value!r}")
        _finite_decimal(self.gross_exposure, field_name="ExposureSnapshot.gross_exposure")
        _finite_decimal(self.net_exposure, field_name="ExposureSnapshot.net_exposure")
        if self.gross_exposure < 0:
            raise PortfolioSnapshotValidationError(f"ExposureSnapshot.gross_exposure must be >= 0, got {self.gross_exposure!r}")
        if abs(self.net_exposure) > self.gross_exposure:
            raise PortfolioSnapshotValidationError(
                f"ExposureSnapshot.net_exposure magnitude ({self.net_exposure!r}) cannot exceed gross_exposure ({self.gross_exposure!r})"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "scope_kind": self.scope_kind.value, "scope_id": self.scope_id, "gross_exposure": decimal_to_json(self.gross_exposure),
            "net_exposure": decimal_to_json(self.net_exposure),
        }


def _matching_positions(portfolio: PortfolioSnapshot, *, scope_kind: ExposureScopeKind, scope_id: str | None) -> tuple[PositionSnapshot, ...]:
    if scope_kind is ExposureScopeKind.PORTFOLIO:
        return portfolio.positions
    if scope_kind is ExposureScopeKind.INSTRUMENT:
        return tuple(p for p in portfolio.positions if p.instrument_id == scope_id)
    return tuple(p for p in portfolio.positions if p.strategy_id == scope_id)


def compute_exposure_snapshot(portfolio: PortfolioSnapshot, *, scope_kind: ExposureScopeKind, scope_id: str | None) -> ExposureSnapshot:
    """Pure derivation from `portfolio.positions` -- never trusts an
    independently stored exposure figure. `gross_exposure` sums the
    ABSOLUTE market value of every matching position (always >= 0 by
    construction); `net_exposure` sums the SIGNED market value."""
    matching = _matching_positions(portfolio, scope_kind=scope_kind, scope_id=scope_id)
    gross_exposure = sum((abs(p.market_value) for p in matching), start=Decimal(0))
    net_exposure = sum((p.market_value for p in matching), start=Decimal(0))
    return ExposureSnapshot(scope_kind=scope_kind, scope_id=scope_id, gross_exposure=gross_exposure, net_exposure=net_exposure)


def compute_portfolio_exposure(portfolio: PortfolioSnapshot) -> ExposureSnapshot:
    return compute_exposure_snapshot(portfolio, scope_kind=ExposureScopeKind.PORTFOLIO, scope_id=None)


def compute_instrument_exposure(portfolio: PortfolioSnapshot, *, instrument_id: str) -> ExposureSnapshot:
    return compute_exposure_snapshot(portfolio, scope_kind=ExposureScopeKind.INSTRUMENT, scope_id=instrument_id)


def compute_strategy_exposure(portfolio: PortfolioSnapshot, *, strategy_id: str) -> ExposureSnapshot:
    return compute_exposure_snapshot(portfolio, scope_kind=ExposureScopeKind.STRATEGY, scope_id=strategy_id)


__all__ = [
    "PORTFOLIO_SNAPSHOT_KIND",
    "PRICE_SNAPSHOT_KIND",
    "ExposureScopeKind",
    "ExposureSnapshot",
    "PortfolioSnapshot",
    "PositionSnapshot",
    "PriceSnapshot",
    "compute_exposure_snapshot",
    "compute_instrument_exposure",
    "compute_portfolio_exposure",
    "compute_strategy_exposure",
    "create_portfolio_snapshot",
    "create_price_snapshot",
    "is_portfolio_snapshot_stale",
    "is_price_stale",
]
