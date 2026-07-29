"""Normalized, broker-neutral market-event model (Milestone 7, Section 5).
Every event type below is immutable, carries a deterministic, content-
addressed `event_id` (computed by the module-level `create_*` factory via
`identity.compute_content_id`, the same building block `specs.py` uses for
`paper_session_spec_id`), and round-trips losslessly through `to_json_dict`/
`from_json_dict` for the event ledger (`persistence.py`).

`CorporateOrInstrumentAdjustmentEvent` is deliberately NOT implemented --
Section 5 permits it "only if generically supported," and no generic
corporate-action model exists anywhere in this repository to reuse; adding
one from scratch is out of this milestone's scope (documented limitation,
not a corner cut on a hard requirement).

ORDERING DISCIPLINE (Section 5): this module validates a SINGLE event's
own internal consistency only. It never reorders, deduplicates, or drops
anything -- cross-event sequence/ordering enforcement (`MarketEventOrderError`)
is `market_data.py`'s job for a live/forward stream, and `replay.py`'s job
(with explicit `NORMALIZED_FROM_SOURCE` disclosure) for a bounded replay
source prepared before a session starts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from quant_platform.core.exceptions import MarketEventError
from quant_platform.core.time_utils import compute_close_time
from quant_platform.core.types import Timeframe
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp
from quant_platform.paper_trading.identity import compute_content_id
from quant_platform.paper_trading.models import MarketEventKind, MarketEventQualityFlagKind


def _finite_positive(value: float, *, field_name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise MarketEventError(f"{field_name} must be finite and > 0, got {value!r}")


def _non_negative_if_present(value: float | None, *, field_name: str) -> None:
    if value is not None and (not math.isfinite(value) or value < 0.0):
        raise MarketEventError(f"{field_name} must be finite and >= 0 when present, got {value!r}")


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise MarketEventError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _require_non_negative_sequence(sequence: int, *, field_name: str = "sequence") -> None:
    if sequence < 0:
        raise MarketEventError(f"{field_name} must be >= 0, got {sequence}")


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not value:
        raise MarketEventError(f"{field_name} must not be empty")


def _serialize_timestamp(ts: datetime, *, field_name: str) -> str:
    try:
        return format_utc_timestamp(pd.Timestamp(ts))
    except ValueError as exc:
        raise MarketEventError(f"{field_name}: {exc}") from exc


def _deserialize_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise MarketEventError(f"{field_name} must be a string, got {type(value).__name__}")
    try:
        return parse_utc_timestamp(value).to_pydatetime()
    except ValueError as exc:
        raise MarketEventError(f"{field_name}: {exc}") from exc


def _quality_flags_to_json(flags: tuple[MarketEventQualityFlagKind, ...]) -> list[str]:
    """Declared order preserved -- `quality_flags` is a small, semantically
    UNORDERED set (uniqueness enforced below), but `to_json_dict` is the
    durable round-tripped form and must never silently reorder it; only
    the `*_identity_payload` counterpart sorts it."""
    return [f.value for f in flags]


def _validate_quality_flags(flags: tuple[MarketEventQualityFlagKind, ...], *, field_name: str = "quality_flags") -> None:
    if len(set(flags)) != len(flags):
        raise MarketEventError(f"{field_name} must not repeat a flag")


def _sorted_quality_flag_values(flags: tuple[MarketEventQualityFlagKind, ...]) -> list[str]:
    return sorted(f.value for f in flags)


def _parse_quality_flags(raw: object, *, field_name: str = "quality_flags") -> tuple[MarketEventQualityFlagKind, ...]:
    if not isinstance(raw, list):
        raise MarketEventError(f"{field_name} must be a JSON array, got {type(raw).__name__}")
    return tuple(MarketEventQualityFlagKind(v) for v in raw)


# --------------------------------------------------------------------------
# QuoteEvent
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class QuoteEvent:
    event_id: str
    instrument: str
    event_time: datetime
    receive_time: datetime | None
    sequence: int
    bid: float
    ask: float
    bid_size: float | None
    ask_size: float | None
    source: str
    source_event_identity: str | None
    quality_flags: tuple[MarketEventQualityFlagKind, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.instrument, field_name="QuoteEvent.instrument")
        _require_tz_aware(self.event_time, field_name="QuoteEvent.event_time")
        if self.receive_time is not None:
            _require_tz_aware(self.receive_time, field_name="QuoteEvent.receive_time")
            if self.receive_time < self.event_time:
                raise MarketEventError(f"QuoteEvent.receive_time ({self.receive_time}) must be >= event_time ({self.event_time})")
        _require_non_negative_sequence(self.sequence)
        _finite_positive(self.bid, field_name="QuoteEvent.bid")
        _finite_positive(self.ask, field_name="QuoteEvent.ask")
        if self.ask < self.bid:
            raise MarketEventError(f"QuoteEvent.ask ({self.ask}) must be >= bid ({self.bid})")
        _non_negative_if_present(self.bid_size, field_name="QuoteEvent.bid_size")
        _non_negative_if_present(self.ask_size, field_name="QuoteEvent.ask_size")
        _require_non_empty(self.source, field_name="QuoteEvent.source")
        _validate_quality_flags(self.quality_flags)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": MarketEventKind.QUOTE.value, "event_id": self.event_id, "instrument": self.instrument,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"),
            "receive_time": (None if self.receive_time is None else _serialize_timestamp(self.receive_time, field_name="receive_time")),
            "sequence": self.sequence, "bid": self.bid, "ask": self.ask, "bid_size": self.bid_size, "ask_size": self.ask_size,
            "source": self.source, "source_event_identity": self.source_event_identity,
            "quality_flags": _quality_flags_to_json(self.quality_flags),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        payload["quality_flags"] = _sorted_quality_flag_values(self.quality_flags)
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> QuoteEvent:
        return cls(
            event_id=str(raw["event_id"]), instrument=str(raw["instrument"]),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"),
            receive_time=(None if raw.get("receive_time") is None else _deserialize_timestamp(raw["receive_time"], field_name="receive_time")),
            sequence=int(str(raw["sequence"])), bid=float(str(raw["bid"])), ask=float(str(raw["ask"])),
            bid_size=(None if raw.get("bid_size") is None else float(str(raw["bid_size"]))),
            ask_size=(None if raw.get("ask_size") is None else float(str(raw["ask_size"]))),
            source=str(raw["source"]), source_event_identity=(None if raw.get("source_event_identity") is None else str(raw["source_event_identity"])),
            quality_flags=_parse_quality_flags(raw.get("quality_flags", [])),
        )


def create_quote_event(
    *, instrument: str, event_time: datetime, sequence: int, bid: float, ask: float, source: str,
    receive_time: datetime | None = None, bid_size: float | None = None, ask_size: float | None = None,
    source_event_identity: str | None = None, quality_flags: tuple[MarketEventQualityFlagKind, ...] = (),
) -> QuoteEvent:
    """The only supported way to mint a fresh `QuoteEvent` -- computes its
    deterministic `event_id` from every other field before constructing
    it, so two calls with identical arguments always produce byte-
    identical events."""
    provisional = QuoteEvent(
        event_id="0" * 64, instrument=instrument, event_time=event_time, receive_time=receive_time, sequence=sequence, bid=bid, ask=ask,
        bid_size=bid_size, ask_size=ask_size, source=source, source_event_identity=source_event_identity, quality_flags=quality_flags,
    )
    event_id = compute_content_id(MarketEventKind.QUOTE.value, provisional.to_identity_payload())
    return QuoteEvent(
        event_id=event_id, instrument=instrument, event_time=event_time, receive_time=receive_time, sequence=sequence, bid=bid, ask=ask,
        bid_size=bid_size, ask_size=ask_size, source=source, source_event_identity=source_event_identity, quality_flags=quality_flags,
    )


# --------------------------------------------------------------------------
# BarEvent
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BarEvent:
    event_id: str
    instrument: str
    interval: Timeframe
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    bid_close: float | None
    ask_close: float | None
    is_complete: bool
    sequence: int
    source: str
    source_event_identity: str | None
    quality_flags: tuple[MarketEventQualityFlagKind, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.instrument, field_name="BarEvent.instrument")
        _require_tz_aware(self.open_time, field_name="BarEvent.open_time")
        _require_tz_aware(self.close_time, field_name="BarEvent.close_time")
        expected_close_time = compute_close_time(self.open_time, self.interval)
        if self.close_time != expected_close_time:
            raise MarketEventError(f"BarEvent.close_time ({self.close_time}) must equal open_time + interval duration ({expected_close_time})")
        for field_name, value in (("open", self.open), ("high", self.high), ("low", self.low), ("close", self.close)):
            _finite_positive(value, field_name=f"BarEvent.{field_name}")
        if self.high < max(self.open, self.close, self.low):
            raise MarketEventError(f"BarEvent.high ({self.high}) must be >= max(open, close, low)")
        if self.low > min(self.open, self.close, self.high):
            raise MarketEventError(f"BarEvent.low ({self.low}) must be <= min(open, close, high)")
        _non_negative_if_present(self.volume, field_name="BarEvent.volume")
        if self.bid_close is not None:
            _finite_positive(self.bid_close, field_name="BarEvent.bid_close")
        if self.ask_close is not None:
            _finite_positive(self.ask_close, field_name="BarEvent.ask_close")
        if self.bid_close is not None and self.ask_close is not None and self.ask_close < self.bid_close:
            raise MarketEventError(f"BarEvent.ask_close ({self.ask_close}) must be >= bid_close ({self.bid_close})")
        _require_non_negative_sequence(self.sequence)
        _require_non_empty(self.source, field_name="BarEvent.source")
        _validate_quality_flags(self.quality_flags)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": MarketEventKind.BAR.value, "event_id": self.event_id, "instrument": self.instrument, "interval": self.interval.value,
            "open_time": _serialize_timestamp(self.open_time, field_name="open_time"),
            "close_time": _serialize_timestamp(self.close_time, field_name="close_time"),
            "open": self.open, "high": self.high, "low": self.low, "close": self.close, "volume": self.volume,
            "bid_close": self.bid_close, "ask_close": self.ask_close, "is_complete": self.is_complete, "sequence": self.sequence,
            "source": self.source, "source_event_identity": self.source_event_identity,
            "quality_flags": _quality_flags_to_json(self.quality_flags),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        payload["quality_flags"] = _sorted_quality_flag_values(self.quality_flags)
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BarEvent:
        return cls(
            event_id=str(raw["event_id"]), instrument=str(raw["instrument"]), interval=Timeframe(raw["interval"]),
            open_time=_deserialize_timestamp(raw["open_time"], field_name="open_time"),
            close_time=_deserialize_timestamp(raw["close_time"], field_name="close_time"),
            open=float(str(raw["open"])), high=float(str(raw["high"])), low=float(str(raw["low"])), close=float(str(raw["close"])),
            volume=(None if raw.get("volume") is None else float(str(raw["volume"]))),
            bid_close=(None if raw.get("bid_close") is None else float(str(raw["bid_close"]))),
            ask_close=(None if raw.get("ask_close") is None else float(str(raw["ask_close"]))),
            is_complete=bool(raw["is_complete"]), sequence=int(str(raw["sequence"])), source=str(raw["source"]),
            source_event_identity=(None if raw.get("source_event_identity") is None else str(raw["source_event_identity"])),
            quality_flags=_parse_quality_flags(raw.get("quality_flags", [])),
        )


def create_bar_event(
    *, instrument: str, interval: Timeframe, open_time: datetime, open: float, high: float, low: float, close: float,
    sequence: int, source: str, volume: float | None = None, bid_close: float | None = None, ask_close: float | None = None,
    is_complete: bool = True, source_event_identity: str | None = None, quality_flags: tuple[MarketEventQualityFlagKind, ...] = (),
) -> BarEvent:
    close_time = compute_close_time(open_time, interval)
    provisional = BarEvent(
        event_id="0" * 64, instrument=instrument, interval=interval, open_time=open_time, close_time=close_time, open=open, high=high,
        low=low, close=close, volume=volume, bid_close=bid_close, ask_close=ask_close, is_complete=is_complete, sequence=sequence,
        source=source, source_event_identity=source_event_identity, quality_flags=quality_flags,
    )
    event_id = compute_content_id(MarketEventKind.BAR.value, provisional.to_identity_payload())
    return BarEvent(
        event_id=event_id, instrument=instrument, interval=interval, open_time=open_time, close_time=close_time, open=open, high=high,
        low=low, close=close, volume=volume, bid_close=bid_close, ask_close=ask_close, is_complete=is_complete, sequence=sequence,
        source=source, source_event_identity=source_event_identity, quality_flags=quality_flags,
    )


# --------------------------------------------------------------------------
# Session / halt / resume / financing / end-of-stream markers -- all share
# the same small (event_id, instrument, event_time, sequence, source,
# source_event_identity) shape, plus a per-kind extra field where needed.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SessionOpenEvent:
    event_id: str
    instrument: str
    event_time: datetime
    sequence: int
    source: str
    source_event_identity: str | None

    def __post_init__(self) -> None:
        _require_non_empty(self.instrument, field_name="SessionOpenEvent.instrument")
        _require_tz_aware(self.event_time, field_name="SessionOpenEvent.event_time")
        _require_non_negative_sequence(self.sequence)
        _require_non_empty(self.source, field_name="SessionOpenEvent.source")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": MarketEventKind.SESSION_OPEN.value, "event_id": self.event_id, "instrument": self.instrument,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "sequence": self.sequence,
            "source": self.source, "source_event_identity": self.source_event_identity,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SessionOpenEvent:
        return cls(
            event_id=str(raw["event_id"]), instrument=str(raw["instrument"]),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), sequence=int(str(raw["sequence"])),
            source=str(raw["source"]), source_event_identity=(None if raw.get("source_event_identity") is None else str(raw["source_event_identity"])),
        )


def create_session_open_event(*, instrument: str, event_time: datetime, sequence: int, source: str, source_event_identity: str | None = None) -> SessionOpenEvent:
    provisional = SessionOpenEvent(event_id="0" * 64, instrument=instrument, event_time=event_time, sequence=sequence, source=source, source_event_identity=source_event_identity)
    event_id = compute_content_id(MarketEventKind.SESSION_OPEN.value, provisional.to_identity_payload())
    return SessionOpenEvent(event_id=event_id, instrument=instrument, event_time=event_time, sequence=sequence, source=source, source_event_identity=source_event_identity)


@dataclass(frozen=True, slots=True)
class SessionCloseEvent:
    event_id: str
    instrument: str
    event_time: datetime
    sequence: int
    source: str
    source_event_identity: str | None

    def __post_init__(self) -> None:
        _require_non_empty(self.instrument, field_name="SessionCloseEvent.instrument")
        _require_tz_aware(self.event_time, field_name="SessionCloseEvent.event_time")
        _require_non_negative_sequence(self.sequence)
        _require_non_empty(self.source, field_name="SessionCloseEvent.source")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": MarketEventKind.SESSION_CLOSE.value, "event_id": self.event_id, "instrument": self.instrument,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "sequence": self.sequence,
            "source": self.source, "source_event_identity": self.source_event_identity,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SessionCloseEvent:
        return cls(
            event_id=str(raw["event_id"]), instrument=str(raw["instrument"]),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), sequence=int(str(raw["sequence"])),
            source=str(raw["source"]), source_event_identity=(None if raw.get("source_event_identity") is None else str(raw["source_event_identity"])),
        )


def create_session_close_event(*, instrument: str, event_time: datetime, sequence: int, source: str, source_event_identity: str | None = None) -> SessionCloseEvent:
    provisional = SessionCloseEvent(event_id="0" * 64, instrument=instrument, event_time=event_time, sequence=sequence, source=source, source_event_identity=source_event_identity)
    event_id = compute_content_id(MarketEventKind.SESSION_CLOSE.value, provisional.to_identity_payload())
    return SessionCloseEvent(event_id=event_id, instrument=instrument, event_time=event_time, sequence=sequence, source=source, source_event_identity=source_event_identity)


@dataclass(frozen=True, slots=True)
class TradingHaltEvent:
    event_id: str
    instrument: str
    event_time: datetime
    sequence: int
    reason: str
    source: str
    source_event_identity: str | None

    def __post_init__(self) -> None:
        _require_non_empty(self.instrument, field_name="TradingHaltEvent.instrument")
        _require_tz_aware(self.event_time, field_name="TradingHaltEvent.event_time")
        _require_non_negative_sequence(self.sequence)
        _require_non_empty(self.reason, field_name="TradingHaltEvent.reason")
        _require_non_empty(self.source, field_name="TradingHaltEvent.source")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": MarketEventKind.TRADING_HALT.value, "event_id": self.event_id, "instrument": self.instrument,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "sequence": self.sequence,
            "reason": self.reason, "source": self.source, "source_event_identity": self.source_event_identity,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> TradingHaltEvent:
        return cls(
            event_id=str(raw["event_id"]), instrument=str(raw["instrument"]),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), sequence=int(str(raw["sequence"])),
            reason=str(raw["reason"]), source=str(raw["source"]),
            source_event_identity=(None if raw.get("source_event_identity") is None else str(raw["source_event_identity"])),
        )


def create_trading_halt_event(*, instrument: str, event_time: datetime, sequence: int, reason: str, source: str, source_event_identity: str | None = None) -> TradingHaltEvent:
    provisional = TradingHaltEvent(event_id="0" * 64, instrument=instrument, event_time=event_time, sequence=sequence, reason=reason, source=source, source_event_identity=source_event_identity)
    event_id = compute_content_id(MarketEventKind.TRADING_HALT.value, provisional.to_identity_payload())
    return TradingHaltEvent(event_id=event_id, instrument=instrument, event_time=event_time, sequence=sequence, reason=reason, source=source, source_event_identity=source_event_identity)


@dataclass(frozen=True, slots=True)
class TradingResumeEvent:
    event_id: str
    instrument: str
    event_time: datetime
    sequence: int
    source: str
    source_event_identity: str | None

    def __post_init__(self) -> None:
        _require_non_empty(self.instrument, field_name="TradingResumeEvent.instrument")
        _require_tz_aware(self.event_time, field_name="TradingResumeEvent.event_time")
        _require_non_negative_sequence(self.sequence)
        _require_non_empty(self.source, field_name="TradingResumeEvent.source")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": MarketEventKind.TRADING_RESUME.value, "event_id": self.event_id, "instrument": self.instrument,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "sequence": self.sequence,
            "source": self.source, "source_event_identity": self.source_event_identity,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> TradingResumeEvent:
        return cls(
            event_id=str(raw["event_id"]), instrument=str(raw["instrument"]),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), sequence=int(str(raw["sequence"])),
            source=str(raw["source"]), source_event_identity=(None if raw.get("source_event_identity") is None else str(raw["source_event_identity"])),
        )


def create_trading_resume_event(*, instrument: str, event_time: datetime, sequence: int, source: str, source_event_identity: str | None = None) -> TradingResumeEvent:
    provisional = TradingResumeEvent(event_id="0" * 64, instrument=instrument, event_time=event_time, sequence=sequence, source=source, source_event_identity=source_event_identity)
    event_id = compute_content_id(MarketEventKind.TRADING_RESUME.value, provisional.to_identity_payload())
    return TradingResumeEvent(event_id=event_id, instrument=instrument, event_time=event_time, sequence=sequence, source=source, source_event_identity=source_event_identity)


@dataclass(frozen=True, slots=True)
class FinancingEvent:
    """A session-boundary trigger marker only -- it signals "apply
    financing now"; the actual amount is computed by `accounting.py` from
    `FinancingSpec` at the moment this event is processed, never carried
    as a pre-computed value here (single source of truth for the
    formula)."""

    event_id: str
    instrument: str
    event_time: datetime
    sequence: int
    source: str
    source_event_identity: str | None

    def __post_init__(self) -> None:
        _require_non_empty(self.instrument, field_name="FinancingEvent.instrument")
        _require_tz_aware(self.event_time, field_name="FinancingEvent.event_time")
        _require_non_negative_sequence(self.sequence)
        _require_non_empty(self.source, field_name="FinancingEvent.source")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": MarketEventKind.FINANCING.value, "event_id": self.event_id, "instrument": self.instrument,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "sequence": self.sequence,
            "source": self.source, "source_event_identity": self.source_event_identity,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FinancingEvent:
        return cls(
            event_id=str(raw["event_id"]), instrument=str(raw["instrument"]),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), sequence=int(str(raw["sequence"])),
            source=str(raw["source"]), source_event_identity=(None if raw.get("source_event_identity") is None else str(raw["source_event_identity"])),
        )


def create_financing_event(*, instrument: str, event_time: datetime, sequence: int, source: str, source_event_identity: str | None = None) -> FinancingEvent:
    provisional = FinancingEvent(event_id="0" * 64, instrument=instrument, event_time=event_time, sequence=sequence, source=source, source_event_identity=source_event_identity)
    event_id = compute_content_id(MarketEventKind.FINANCING.value, provisional.to_identity_payload())
    return FinancingEvent(event_id=event_id, instrument=instrument, event_time=event_time, sequence=sequence, source=source, source_event_identity=source_event_identity)


@dataclass(frozen=True, slots=True)
class EndOfStreamEvent:
    """Deterministic marker for the end of a BOUNDED replay/forward
    source -- `runner.py` transitions `RUNNING` -> `END_OF_STREAM` on
    consuming this and never invents one on its own (Section 37: "must
    not require loading an unlimited forward stream into memory," so a
    `FORWARD_PAPER` session may legitimately run without ever seeing one
    until the external stream explicitly signals completion)."""

    event_id: str
    instrument: str
    event_time: datetime
    sequence: int
    source: str

    def __post_init__(self) -> None:
        _require_non_empty(self.instrument, field_name="EndOfStreamEvent.instrument")
        _require_tz_aware(self.event_time, field_name="EndOfStreamEvent.event_time")
        _require_non_negative_sequence(self.sequence)
        _require_non_empty(self.source, field_name="EndOfStreamEvent.source")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": MarketEventKind.END_OF_STREAM.value, "event_id": self.event_id, "instrument": self.instrument,
            "event_time": _serialize_timestamp(self.event_time, field_name="event_time"), "sequence": self.sequence, "source": self.source,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> EndOfStreamEvent:
        return cls(
            event_id=str(raw["event_id"]), instrument=str(raw["instrument"]),
            event_time=_deserialize_timestamp(raw["event_time"], field_name="event_time"), sequence=int(str(raw["sequence"])), source=str(raw["source"]),
        )


def create_end_of_stream_event(*, instrument: str, event_time: datetime, sequence: int, source: str) -> EndOfStreamEvent:
    provisional = EndOfStreamEvent(event_id="0" * 64, instrument=instrument, event_time=event_time, sequence=sequence, source=source)
    event_id = compute_content_id(MarketEventKind.END_OF_STREAM.value, provisional.to_identity_payload())
    return EndOfStreamEvent(event_id=event_id, instrument=instrument, event_time=event_time, sequence=sequence, source=source)


MarketEvent = QuoteEvent | BarEvent | SessionOpenEvent | SessionCloseEvent | TradingHaltEvent | TradingResumeEvent | FinancingEvent | EndOfStreamEvent

_EVENT_CLASSES_BY_KIND: dict[str, type[MarketEvent]] = {
    MarketEventKind.QUOTE.value: QuoteEvent,
    MarketEventKind.BAR.value: BarEvent,
    MarketEventKind.SESSION_OPEN.value: SessionOpenEvent,
    MarketEventKind.SESSION_CLOSE.value: SessionCloseEvent,
    MarketEventKind.TRADING_HALT.value: TradingHaltEvent,
    MarketEventKind.TRADING_RESUME.value: TradingResumeEvent,
    MarketEventKind.FINANCING.value: FinancingEvent,
    MarketEventKind.END_OF_STREAM.value: EndOfStreamEvent,
}


def market_event_from_json_dict(raw: dict[str, object]) -> MarketEvent:
    """Dispatch on the persisted `"kind"` discriminator -- the single
    entry point `persistence.py`'s ledger reader and `replay.py`'s source
    reader use to reconstruct a `MarketEvent` of the correct concrete
    type."""
    kind = raw.get("kind")
    if kind not in _EVENT_CLASSES_BY_KIND:
        raise MarketEventError(f"Unknown market event kind {kind!r}")
    return _EVENT_CLASSES_BY_KIND[str(kind)].from_json_dict(raw)


def market_event_id(event: MarketEvent) -> str:
    return event.event_id


def market_event_time(event: MarketEvent) -> datetime:
    """The one event-time concept every kind exposes, for `market_data.
    py`'s cross-event ordering check. `BarEvent` has no single `event_time`
    field (it has `open_time`/`close_time` instead) -- its CLOSE time is
    the correct value here, since that is "the instant a bar's data is
    actually fully known" (`core.time_utils.compute_close_time`'s own
    docstring) and therefore the instant the bar may correctly participate
    in event-time ordering against other events."""
    if isinstance(event, BarEvent):
        return event.close_time
    return event.event_time


def market_event_sequence(event: MarketEvent) -> int:
    return event.sequence


__all__ = [
    "BarEvent",
    "EndOfStreamEvent",
    "FinancingEvent",
    "MarketEvent",
    "QuoteEvent",
    "SessionCloseEvent",
    "SessionOpenEvent",
    "TradingHaltEvent",
    "TradingResumeEvent",
    "create_bar_event",
    "create_end_of_stream_event",
    "create_financing_event",
    "create_quote_event",
    "create_session_close_event",
    "create_session_open_event",
    "create_trading_halt_event",
    "create_trading_resume_event",
    "market_event_from_json_dict",
    "market_event_id",
    "market_event_sequence",
    "market_event_time",
]
