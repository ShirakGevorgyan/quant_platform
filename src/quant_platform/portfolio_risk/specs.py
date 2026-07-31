"""Immutable specifications for `quant_platform.portfolio_risk`
(Milestone 9, Phase 1). Every spec here is a frozen, slotted dataclass
with an explicit `__post_init__` validator, following
`execution_gateway.specs`'s (and, one layer further down,
`paper_trading.specs`'s) identical convention exactly: `to_json_dict()`
always preserves caller-declared field order (the durable, round-tripped
representation); `to_identity_payload()` is where any genuinely UNORDERED
collection would be canonicalized -- there is none in this milestone's
Phase 1 policy, so the two payloads are currently identical apart from
`schema_version` (excluded from identity, exactly like
`ExecutionGatewaySpec`).

`PortfolioRiskPolicy` is every configurable limit this milestone's
Section-required list defines (Phase 1: the fields exist and validate;
NO evaluator in this package reads them yet). Every limit field is
`Decimal | None` (or `int | None` for the two age bounds and the
consecutive-losses count) -- `None` means "not configured", mirroring
`paper_trading.specs.RiskLimitsSpec`'s identical convention exactly: a
limit that is not configured is simply never checked, it is never
silently treated as "no limit" in a way that differs from an explicit
absence.

`PortfolioRiskSpec` is the thin, top-level, content-addressed wrapper
around one `PortfolioRiskPolicy` -- deliberately NOT bound to a specific
`portfolio_id` (a single spec/policy is reusable across many portfolios;
`RiskAuthorization` binds `portfolio_id` and `risk_policy_id`
independently), exactly mirroring how `ExecutionGatewaySpec` wraps
several independently-defined sub-policies rather than folding their
fields directly into one flat dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from quant_platform.core.exceptions import PortfolioRiskPolicyError
from quant_platform.portfolio_risk.identity import compute_content_id, decimal_to_json, parse_decimal
from quant_platform.portfolio_risk.models import PORTFOLIO_RISK_SPEC_SCHEMA_VERSION

PORTFOLIO_RISK_SPEC_KIND = "portfolio_risk_spec"


def _finite_decimal(value: Decimal, *, field_name: str) -> None:
    if not value.is_finite():
        raise PortfolioRiskPolicyError(f"{field_name} must be finite, got {value!r}")


def _positive_decimal(value: Decimal, *, field_name: str) -> None:
    _finite_decimal(value, field_name=field_name)
    if value <= 0:
        raise PortfolioRiskPolicyError(f"{field_name} must be > 0, got {value!r}")


def _non_negative_decimal(value: Decimal, *, field_name: str) -> None:
    _finite_decimal(value, field_name=field_name)
    if value < 0:
        raise PortfolioRiskPolicyError(f"{field_name} must be >= 0, got {value!r}")


def _fraction_in_unit_interval(value: Decimal, *, field_name: str) -> None:
    _finite_decimal(value, field_name=field_name)
    if not (0 < value <= 1):
        raise PortfolioRiskPolicyError(f"{field_name} must be in (0, 1], got {value!r}")


def _positive_int(value: int, *, field_name: str) -> None:
    if value < 1:
        raise PortfolioRiskPolicyError(f"{field_name} must be >= 1, got {value!r}")


def _non_negative_int(value: int, *, field_name: str) -> None:
    if value < 0:
        raise PortfolioRiskPolicyError(f"{field_name} must be >= 0, got {value!r}")


# --------------------------------------------------------------------------
# PortfolioRiskPolicy
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PortfolioRiskPolicy:
    """Every field is optional (`None` = not configured = never checked by
    a future evaluator, exactly mirroring `paper_trading.specs.
    RiskLimitsSpec`) except `allow_reduce_only_during_halt`, which is a
    mandatory, always-explicit behavioral switch: whether a reduce-only
    order may still pass while the portfolio is HALTED. Both `True` and
    `False` are legitimate operator choices for that field -- this class
    does not constrain its value, only requires it be stated."""

    max_order_notional: Decimal | None
    max_position_notional: Decimal | None
    max_instrument_gross_exposure: Decimal | None
    max_strategy_gross_exposure: Decimal | None
    max_portfolio_gross_exposure: Decimal | None
    max_portfolio_net_exposure: Decimal | None
    """Bounds the MAGNITUDE of net exposure (`abs(net_exposure) <=
    max_portfolio_net_exposure`) -- net exposure itself may be negative
    (net short); this limit is always a non-negative bound on its
    absolute value."""
    max_concentration_fraction: Decimal | None
    """Largest fraction of gross portfolio exposure any single instrument
    may represent, in (0, 1]."""
    max_leverage: Decimal | None
    """Bounds `gross_exposure / equity`."""
    max_daily_realized_loss: Decimal | None
    """A loss-MAGNITUDE bound: `-realized_pnl_today <=
    max_daily_realized_loss` (a positive limit bounding a negative pnl)."""
    max_total_loss: Decimal | None
    """A loss-magnitude bound against cumulative realized+unrealized pnl
    since the portfolio's own inception, analogous to `max_daily_realized_
    loss` but not reset daily."""
    max_drawdown_fraction: Decimal | None
    """`(peak_equity - equity) / peak_equity`, in (0, 1]."""
    max_consecutive_losses: int | None
    minimum_cash_buffer: Decimal | None
    """A FLOOR, not a ceiling -- `cash >= minimum_cash_buffer`. Zero is a
    legitimate, explicit floor (no buffer required beyond solvency)."""
    maximum_price_age: int | None
    """Whole seconds. Compared against a caller-supplied reference time
    minus `PriceSnapshot.event_time` via `timedelta`, never a float
    duration."""
    maximum_portfolio_snapshot_age: int | None
    """Whole seconds, same convention as `maximum_price_age`."""
    allow_reduce_only_during_halt: bool

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_order_notional", self.max_order_notional), ("max_position_notional", self.max_position_notional),
            ("max_instrument_gross_exposure", self.max_instrument_gross_exposure),
            ("max_strategy_gross_exposure", self.max_strategy_gross_exposure),
            ("max_portfolio_gross_exposure", self.max_portfolio_gross_exposure),
            ("max_portfolio_net_exposure", self.max_portfolio_net_exposure), ("max_leverage", self.max_leverage),
            ("max_daily_realized_loss", self.max_daily_realized_loss), ("max_total_loss", self.max_total_loss),
        ):
            if value is not None:
                _positive_decimal(value, field_name=f"PortfolioRiskPolicy.{field_name}")
        for field_name, fraction in (
            ("max_concentration_fraction", self.max_concentration_fraction),
            ("max_drawdown_fraction", self.max_drawdown_fraction),
        ):
            if fraction is not None:
                _fraction_in_unit_interval(fraction, field_name=f"PortfolioRiskPolicy.{field_name}")
        if self.max_consecutive_losses is not None:
            _positive_int(self.max_consecutive_losses, field_name="PortfolioRiskPolicy.max_consecutive_losses")
        if self.minimum_cash_buffer is not None:
            _non_negative_decimal(self.minimum_cash_buffer, field_name="PortfolioRiskPolicy.minimum_cash_buffer")
        if self.maximum_price_age is not None:
            _non_negative_int(self.maximum_price_age, field_name="PortfolioRiskPolicy.maximum_price_age")
        if self.maximum_portfolio_snapshot_age is not None:
            _non_negative_int(self.maximum_portfolio_snapshot_age, field_name="PortfolioRiskPolicy.maximum_portfolio_snapshot_age")

    def to_json_dict(self) -> dict[str, object]:
        def _opt_decimal(value: Decimal | None) -> str | None:
            return None if value is None else decimal_to_json(value)

        return {
            "max_order_notional": _opt_decimal(self.max_order_notional), "max_position_notional": _opt_decimal(self.max_position_notional),
            "max_instrument_gross_exposure": _opt_decimal(self.max_instrument_gross_exposure),
            "max_strategy_gross_exposure": _opt_decimal(self.max_strategy_gross_exposure),
            "max_portfolio_gross_exposure": _opt_decimal(self.max_portfolio_gross_exposure),
            "max_portfolio_net_exposure": _opt_decimal(self.max_portfolio_net_exposure),
            "max_concentration_fraction": _opt_decimal(self.max_concentration_fraction), "max_leverage": _opt_decimal(self.max_leverage),
            "max_daily_realized_loss": _opt_decimal(self.max_daily_realized_loss), "max_total_loss": _opt_decimal(self.max_total_loss),
            "max_drawdown_fraction": _opt_decimal(self.max_drawdown_fraction), "max_consecutive_losses": self.max_consecutive_losses,
            "minimum_cash_buffer": _opt_decimal(self.minimum_cash_buffer), "maximum_price_age": self.maximum_price_age,
            "maximum_portfolio_snapshot_age": self.maximum_portfolio_snapshot_age,
            "allow_reduce_only_during_halt": self.allow_reduce_only_during_halt,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PortfolioRiskPolicy:
        def _opt_decimal(value: object, *, field_name: str) -> Decimal | None:
            return None if value is None else parse_decimal(value, field_name=field_name)

        def _opt_int(value: object) -> int | None:
            return None if value is None else int(str(value))

        return cls(
            max_order_notional=_opt_decimal(raw.get("max_order_notional"), field_name="max_order_notional"),
            max_position_notional=_opt_decimal(raw.get("max_position_notional"), field_name="max_position_notional"),
            max_instrument_gross_exposure=_opt_decimal(raw.get("max_instrument_gross_exposure"), field_name="max_instrument_gross_exposure"),
            max_strategy_gross_exposure=_opt_decimal(raw.get("max_strategy_gross_exposure"), field_name="max_strategy_gross_exposure"),
            max_portfolio_gross_exposure=_opt_decimal(raw.get("max_portfolio_gross_exposure"), field_name="max_portfolio_gross_exposure"),
            max_portfolio_net_exposure=_opt_decimal(raw.get("max_portfolio_net_exposure"), field_name="max_portfolio_net_exposure"),
            max_concentration_fraction=_opt_decimal(raw.get("max_concentration_fraction"), field_name="max_concentration_fraction"),
            max_leverage=_opt_decimal(raw.get("max_leverage"), field_name="max_leverage"),
            max_daily_realized_loss=_opt_decimal(raw.get("max_daily_realized_loss"), field_name="max_daily_realized_loss"),
            max_total_loss=_opt_decimal(raw.get("max_total_loss"), field_name="max_total_loss"),
            max_drawdown_fraction=_opt_decimal(raw.get("max_drawdown_fraction"), field_name="max_drawdown_fraction"),
            max_consecutive_losses=_opt_int(raw.get("max_consecutive_losses")),
            minimum_cash_buffer=_opt_decimal(raw.get("minimum_cash_buffer"), field_name="minimum_cash_buffer"),
            maximum_price_age=_opt_int(raw.get("maximum_price_age")),
            maximum_portfolio_snapshot_age=_opt_int(raw.get("maximum_portfolio_snapshot_age")),
            allow_reduce_only_during_halt=bool(raw["allow_reduce_only_during_halt"]),
        )


# --------------------------------------------------------------------------
# PortfolioRiskSpec -- top-level, content-addressed
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PortfolioRiskSpec:
    schema_version: int
    policy: PortfolioRiskPolicy

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "policy": self.policy.to_json_dict()}

    def to_identity_payload(self) -> dict[str, object]:
        payload = self.to_json_dict()
        del payload["schema_version"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PortfolioRiskSpec:
        from quant_platform.ml.persistence import as_json_dict, require_schema_version

        require_schema_version(raw, supported=PORTFOLIO_RISK_SPEC_SCHEMA_VERSION, context="PortfolioRiskSpec")
        return cls(
            schema_version=PORTFOLIO_RISK_SPEC_SCHEMA_VERSION,
            policy=PortfolioRiskPolicy.from_json_dict(as_json_dict(raw["policy"], field_name="policy")),
        )


@dataclass(frozen=True, slots=True)
class PortfolioRiskSpecIdentity:
    schema_version: int
    portfolio_risk_spec_id: str

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "portfolio_risk_spec_id": self.portfolio_risk_spec_id}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PortfolioRiskSpecIdentity:
        from quant_platform.ml.persistence import require_schema_version

        require_schema_version(raw, supported=1, context="PortfolioRiskSpecIdentity")
        return cls(schema_version=1, portfolio_risk_spec_id=str(raw["portfolio_risk_spec_id"]))


def compute_portfolio_risk_spec_id(spec: PortfolioRiskSpec) -> PortfolioRiskSpecIdentity:
    portfolio_risk_spec_id = compute_content_id(PORTFOLIO_RISK_SPEC_KIND, spec.to_identity_payload())
    return PortfolioRiskSpecIdentity(schema_version=1, portfolio_risk_spec_id=portfolio_risk_spec_id)


def verify_portfolio_risk_spec_identity(spec: PortfolioRiskSpec, expected_id: str) -> bool:
    """Pure recomputation-and-compare -- never trusts a caller-supplied id
    without recomputing it fresh from `spec`."""
    return compute_portfolio_risk_spec_id(spec).portfolio_risk_spec_id == expected_id


__all__ = [
    "PORTFOLIO_RISK_SPEC_KIND",
    "PortfolioRiskPolicy",
    "PortfolioRiskSpec",
    "PortfolioRiskSpecIdentity",
    "compute_portfolio_risk_spec_id",
    "verify_portfolio_risk_spec_identity",
]
