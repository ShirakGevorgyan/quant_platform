"""Futures contract metadata and continuous-series policy (Milestone 10,
Phase 4C).

A continuous futures series is NOT a raw market fact -- it is a
DERIVED, policy-dependent construction (which contract is "active" on
any given day, and how the series behaves across a roll). This module
models both halves honestly: `FuturesContractMetadata` for one
SPECIFIC, individually identified contract (rejected outright if
result-critical fields are missing -- a provider that cannot supply
this must be classified `PROVIDER_NATIVE_CONTINUOUS` instead of a
policy asserting individual-contract knowledge it does not have), and
`ContinuationPolicy` for how (or whether) this platform stitches
contracts together. Every continuous value must preserve roll
provenance; nothing here silently stitches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum

from quant_platform.core.exceptions import ContinuationPolicyError, FuturesContractError
from quant_platform.market_data.identity import compute_content_id, require_non_empty

__all__ = [
    "CONTINUATION_POLICY_KIND",
    "FUTURES_CONTRACT_METADATA_KIND",
    "ContinuationPolicy",
    "ContinuationPolicyKind",
    "FuturesContractMetadata",
    "RollProvenance",
    "create_continuation_policy",
    "create_futures_contract_metadata",
]

FUTURES_CONTRACT_METADATA_KIND = "cross_asset_futures_contract_metadata"
CONTINUATION_POLICY_KIND = "cross_asset_continuation_policy"


class ContinuationPolicyKind(Enum):
    PROVIDER_NATIVE_CONTINUOUS = "provider_native_continuous"
    """The provider itself already generates the continuous series; this
    platform preserves it AS SUCH, treating the provider's own roll
    methodology as an external, disclosed limitation -- never
    reverse-engineering or asserting individual-contract knowledge."""

    FRONT_MONTH_NO_BACK_ADJUSTMENT = "front_month_no_back_adjustment"
    """Always the current front-month contract, switched at roll with NO
    price adjustment -- the series has real, visible price jumps at
    every roll."""

    ROLL_ON_FIXED_DAYS_BEFORE_EXPIRY = "roll_on_fixed_days_before_expiry"
    """Rolls to the next contract a fixed, configured number of calendar
    days before the active contract's expiry."""

    ROLL_ON_VOLUME_CROSSOVER = "roll_on_volume_crossover"
    """Rolls when the next contract's traded volume first exceeds the
    active contract's -- requires per-contract volume evidence."""

    BACK_ADJUSTED_DIFFERENCE = "back_adjusted_difference"
    """Historical prices before each roll are shifted by a constant
    ADDITIVE difference so the series has no roll-induced price jump --
    requires the roll-date price difference to be recorded as
    provenance."""

    RATIO_ADJUSTED = "ratio_adjusted"
    """Historical prices before each roll are scaled by a constant
    MULTIPLICATIVE ratio -- requires the roll-date ratio to be recorded
    as provenance."""


_KINDS_REQUIRING_ADJUSTMENT_EVIDENCE = frozenset({ContinuationPolicyKind.BACK_ADJUSTED_DIFFERENCE, ContinuationPolicyKind.RATIO_ADJUSTED})


@dataclass(frozen=True, slots=True)
class FuturesContractMetadata:
    futures_contract_metadata_id: str
    root_symbol: str
    full_contract_symbol: str
    exchange: str
    expiry: date
    first_notice_date: date | None
    last_trade_date: date | None
    contract_month: int
    contract_year: int
    contract_multiplier: Decimal
    quote_unit: str
    currency: str
    tick_size: Decimal
    session_timezone_key: str
    source_metadata_note: str
    """Documentation only, excluded from identity."""

    def __post_init__(self) -> None:
        require_non_empty(self.root_symbol, field_name="FuturesContractMetadata.root_symbol")
        require_non_empty(self.full_contract_symbol, field_name="FuturesContractMetadata.full_contract_symbol")
        require_non_empty(self.exchange, field_name="FuturesContractMetadata.exchange")
        require_non_empty(self.quote_unit, field_name="FuturesContractMetadata.quote_unit")
        require_non_empty(self.currency, field_name="FuturesContractMetadata.currency")
        require_non_empty(self.session_timezone_key, field_name="FuturesContractMetadata.session_timezone_key")
        if not (1 <= self.contract_month <= 12):
            raise FuturesContractError(f"FuturesContractMetadata.contract_month must be in [1, 12], got {self.contract_month}")
        if self.contract_year < 1970:
            raise FuturesContractError(f"FuturesContractMetadata.contract_year must be >= 1970, got {self.contract_year}")
        if self.contract_multiplier <= 0:
            raise FuturesContractError(f"FuturesContractMetadata.contract_multiplier must be > 0, got {self.contract_multiplier}")
        if self.tick_size <= 0:
            raise FuturesContractError(f"FuturesContractMetadata.tick_size must be > 0, got {self.tick_size}")
        if self.first_notice_date is not None and self.first_notice_date > self.expiry:
            raise FuturesContractError(f"FuturesContractMetadata.first_notice_date ({self.first_notice_date}) must be <= expiry ({self.expiry})")
        if self.last_trade_date is not None and self.last_trade_date > self.expiry:
            raise FuturesContractError(f"FuturesContractMetadata.last_trade_date ({self.last_trade_date}) must be <= expiry ({self.expiry})")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": FUTURES_CONTRACT_METADATA_KIND, "futures_contract_metadata_id": self.futures_contract_metadata_id,
            "root_symbol": self.root_symbol, "full_contract_symbol": self.full_contract_symbol, "exchange": self.exchange,
            "expiry": self.expiry.isoformat(), "first_notice_date": (None if self.first_notice_date is None else self.first_notice_date.isoformat()),
            "last_trade_date": (None if self.last_trade_date is None else self.last_trade_date.isoformat()),
            "contract_month": self.contract_month, "contract_year": self.contract_year, "contract_multiplier": str(self.contract_multiplier),
            "quote_unit": self.quote_unit, "currency": self.currency, "tick_size": str(self.tick_size),
            "session_timezone_key": self.session_timezone_key, "source_metadata_note": self.source_metadata_note,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["futures_contract_metadata_id"]
        del payload["source_metadata_note"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FuturesContractMetadata:
        raw_fnd = raw.get("first_notice_date")
        raw_ltd = raw.get("last_trade_date")
        return cls(
            futures_contract_metadata_id=str(raw["futures_contract_metadata_id"]), root_symbol=str(raw["root_symbol"]),
            full_contract_symbol=str(raw["full_contract_symbol"]), exchange=str(raw["exchange"]), expiry=date.fromisoformat(str(raw["expiry"])),
            first_notice_date=(None if raw_fnd is None else date.fromisoformat(str(raw_fnd))),
            last_trade_date=(None if raw_ltd is None else date.fromisoformat(str(raw_ltd))),
            contract_month=int(str(raw["contract_month"])), contract_year=int(str(raw["contract_year"])),
            contract_multiplier=Decimal(str(raw["contract_multiplier"])), quote_unit=str(raw["quote_unit"]), currency=str(raw["currency"]),
            tick_size=Decimal(str(raw["tick_size"])), session_timezone_key=str(raw["session_timezone_key"]),
            source_metadata_note=str(raw.get("source_metadata_note", "")),
        )


def create_futures_contract_metadata(
    *, root_symbol: str, full_contract_symbol: str, exchange: str, expiry: date, contract_month: int, contract_year: int,
    contract_multiplier: Decimal, quote_unit: str, currency: str, tick_size: Decimal, session_timezone_key: str,
    first_notice_date: date | None = None, last_trade_date: date | None = None, source_metadata_note: str = "",
) -> FuturesContractMetadata:
    provisional = FuturesContractMetadata(
        futures_contract_metadata_id="0" * 64, root_symbol=root_symbol, full_contract_symbol=full_contract_symbol, exchange=exchange,
        expiry=expiry, first_notice_date=first_notice_date, last_trade_date=last_trade_date, contract_month=contract_month,
        contract_year=contract_year, contract_multiplier=contract_multiplier, quote_unit=quote_unit, currency=currency, tick_size=tick_size,
        session_timezone_key=session_timezone_key, source_metadata_note=source_metadata_note,
    )
    futures_contract_metadata_id = compute_content_id(FUTURES_CONTRACT_METADATA_KIND, provisional.to_identity_payload())
    return FuturesContractMetadata(
        futures_contract_metadata_id=futures_contract_metadata_id, root_symbol=root_symbol, full_contract_symbol=full_contract_symbol,
        exchange=exchange, expiry=expiry, first_notice_date=first_notice_date, last_trade_date=last_trade_date, contract_month=contract_month,
        contract_year=contract_year, contract_multiplier=contract_multiplier, quote_unit=quote_unit, currency=currency, tick_size=tick_size,
        session_timezone_key=session_timezone_key, source_metadata_note=source_metadata_note,
    )


@dataclass(frozen=True, slots=True)
class ContinuationPolicy:
    continuation_policy_id: str
    kind: ContinuationPolicyKind
    policy_version: int
    roll_days_before_expiry: int | None
    """Required (and only meaningful) for `ROLL_ON_FIXED_DAYS_BEFORE_
    EXPIRY`."""
    notes: str

    def __post_init__(self) -> None:
        if self.policy_version < 1:
            raise ContinuationPolicyError(f"ContinuationPolicy.policy_version must be >= 1, got {self.policy_version}")
        if self.kind is ContinuationPolicyKind.ROLL_ON_FIXED_DAYS_BEFORE_EXPIRY:
            if self.roll_days_before_expiry is None or self.roll_days_before_expiry < 0:
                raise ContinuationPolicyError("ContinuationPolicy.roll_days_before_expiry must be >= 0 for ROLL_ON_FIXED_DAYS_BEFORE_EXPIRY")
        elif self.roll_days_before_expiry is not None:
            raise ContinuationPolicyError(f"ContinuationPolicy.roll_days_before_expiry must be None for {self.kind.value}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": CONTINUATION_POLICY_KIND, "continuation_policy_id": self.continuation_policy_id, "continuation_kind": self.kind.value,
            "policy_version": self.policy_version, "roll_days_before_expiry": self.roll_days_before_expiry, "notes": self.notes,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["continuation_policy_id"]
        del payload["notes"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ContinuationPolicy:
        raw_days = raw.get("roll_days_before_expiry")
        return cls(
            continuation_policy_id=str(raw["continuation_policy_id"]), kind=ContinuationPolicyKind(raw["continuation_kind"]),
            policy_version=int(str(raw["policy_version"])), roll_days_before_expiry=(None if raw_days is None else int(str(raw_days))),
            notes=str(raw.get("notes", "")),
        )


def create_continuation_policy(
    *, kind: ContinuationPolicyKind, policy_version: int = 1, roll_days_before_expiry: int | None = None, notes: str = "",
) -> ContinuationPolicy:
    provisional = ContinuationPolicy(
        continuation_policy_id="0" * 64, kind=kind, policy_version=policy_version, roll_days_before_expiry=roll_days_before_expiry, notes=notes,
    )
    continuation_policy_id = compute_content_id(CONTINUATION_POLICY_KIND, provisional.to_identity_payload())
    return ContinuationPolicy(
        continuation_policy_id=continuation_policy_id, kind=kind, policy_version=policy_version,
        roll_days_before_expiry=roll_days_before_expiry, notes=notes,
    )


@dataclass(frozen=True, slots=True)
class RollProvenance:
    """Attached to every bar of a continuous series -- see
    `market_record.MarketDriverBar.roll_provenance`. Never optional for
    a bar whose `instrument_form` is `PROVIDER_CONTINUOUS_FUTURES` under
    any kind OTHER than `PROVIDER_NATIVE_CONTINUOUS` (which has no
    per-bar roll evidence of its own to preserve -- see
    `ContinuationPolicyKind.PROVIDER_NATIVE_CONTINUOUS`'s own
    docstring)."""

    active_contract_symbol: str
    prior_contract_symbol: str | None
    next_contract_symbol: str | None
    roll_timestamp: str | None
    """ISO date/datetime text of the roll event that produced the
    CURRENTLY active contract segment -- `None` if this bar predates the
    series' first roll."""
    adjustment_amount: Decimal | None
    """Populated only for `BACK_ADJUSTED_DIFFERENCE`."""
    adjustment_ratio: Decimal | None
    """Populated only for `RATIO_ADJUSTED`."""
    continuation_policy_id: str

    def __post_init__(self) -> None:
        require_non_empty(self.active_contract_symbol, field_name="RollProvenance.active_contract_symbol")
        require_non_empty(self.continuation_policy_id, field_name="RollProvenance.continuation_policy_id")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "active_contract_symbol": self.active_contract_symbol, "prior_contract_symbol": self.prior_contract_symbol,
            "next_contract_symbol": self.next_contract_symbol, "roll_timestamp": self.roll_timestamp,
            "adjustment_amount": (None if self.adjustment_amount is None else str(self.adjustment_amount)),
            "adjustment_ratio": (None if self.adjustment_ratio is None else str(self.adjustment_ratio)),
            "continuation_policy_id": self.continuation_policy_id,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RollProvenance:
        raw_amount = raw.get("adjustment_amount")
        raw_ratio = raw.get("adjustment_ratio")
        return cls(
            active_contract_symbol=str(raw["active_contract_symbol"]),
            prior_contract_symbol=(None if raw.get("prior_contract_symbol") is None else str(raw["prior_contract_symbol"])),
            next_contract_symbol=(None if raw.get("next_contract_symbol") is None else str(raw["next_contract_symbol"])),
            roll_timestamp=(None if raw.get("roll_timestamp") is None else str(raw["roll_timestamp"])),
            adjustment_amount=(None if raw_amount is None else Decimal(str(raw_amount))),
            adjustment_ratio=(None if raw_ratio is None else Decimal(str(raw_ratio))), continuation_policy_id=str(raw["continuation_policy_id"]),
        )


def require_adjustment_evidence(policy: ContinuationPolicy, provenance: RollProvenance) -> None:
    """Structural guard: a `BACK_ADJUSTED_DIFFERENCE`/`RATIO_ADJUSTED`
    continuation must never produce a bar without its own adjustment
    evidence -- called by `market_normalization.py` for every continuous
    bar it builds."""
    if policy.kind not in _KINDS_REQUIRING_ADJUSTMENT_EVIDENCE:
        return
    if policy.kind is ContinuationPolicyKind.BACK_ADJUSTED_DIFFERENCE and provenance.adjustment_amount is None:
        raise ContinuationPolicyError("BACK_ADJUSTED_DIFFERENCE requires RollProvenance.adjustment_amount")
    if policy.kind is ContinuationPolicyKind.RATIO_ADJUSTED and provenance.adjustment_ratio is None:
        raise ContinuationPolicyError("RATIO_ADJUSTED requires RollProvenance.adjustment_ratio")
