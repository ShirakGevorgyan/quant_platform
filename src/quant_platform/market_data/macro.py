"""Immutable macro/economic-calendar events and their append-only store
(Milestone 10, Phase 1).

A `MacroEvent` represents one published data-series release (e.g. a
non-farm-payrolls print, a CPI reading, a central-bank rate decision):
`event_time` is the release's official timestamp -- the instant the value
BECOMES known -- never a later "we noticed it" time (that is
`arrival_time`'s job). This distinction matters because macro features
are exactly the kind of input `core.exceptions.PointInTimeViolationError`
already warns about ("a macro value before its release timestamp"): a
downstream consumer must only treat a `MacroEvent` as available at or
after its own `event_time`, which `is_macro_event_available_at` makes
explicit and checkable rather than left to informal convention.

`MacroEvent` is partitioned by `series_id` (e.g. `"US_NFP"`), not
`instrument_id` -- a macro release is not intrinsically tied to one
instrument the way a tick/quote/trade/candle is; which instruments treat
it as relevant is a downstream (feature-generation) policy decision, not
part of this event's own identity."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from quant_platform.core.exceptions import (
    ExperimentLockError,
    MacroDataError,
    MarketDataLockError,
    MarketDataPersistenceError,
)
from quant_platform.core.json import canonical_json_bytes
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    parse_decimal,
    require_non_empty,
    require_non_negative_sequence,
    require_tz_aware,
    serialize_timestamp,
)
from quant_platform.ml.concurrency import experiment_lock
from quant_platform.ml.persistence import as_json_dict, parse_json_strict

__all__ = ["MACRO_EVENT_KIND", "MacroEvent", "MacroEventStore", "create_macro_event", "is_macro_event_available_at"]

MACRO_EVENT_KIND = "macro_event"
_PAYLOAD_KEYS = ("value", "previous_value", "unit")


def _payload_to_json(*, value: Decimal, previous_value: Decimal | None, unit: str | None) -> dict[str, object]:
    return {"value": str(value), "previous_value": (None if previous_value is None else str(previous_value)), "unit": unit}


def _parse_payload(payload: dict[str, object]) -> tuple[Decimal, Decimal | None, str | None]:
    extra_keys = set(payload.keys()) - set(_PAYLOAD_KEYS)
    if extra_keys:
        raise MacroDataError(f"MacroEvent.payload has unexpected key(s): {sorted(extra_keys)}")
    missing_keys = set(_PAYLOAD_KEYS) - set(payload.keys())
    if missing_keys:
        raise MacroDataError(f"MacroEvent.payload is missing required key(s): {sorted(missing_keys)}")
    value = parse_decimal(payload["value"], field_name="MacroEvent.payload['value']")
    raw_previous = payload["previous_value"]
    previous_value = None if raw_previous is None else parse_decimal(raw_previous, field_name="MacroEvent.payload['previous_value']")
    raw_unit = payload["unit"]
    unit = None if raw_unit is None else str(raw_unit)
    return value, previous_value, unit


@dataclass(frozen=True, slots=True)
class MacroEvent:
    event_id: str
    series_id: str
    provider: str
    event_time: datetime
    """The series' official release timestamp."""
    arrival_time: datetime
    sequence: int
    source_event_id: str | None
    payload: dict[str, object]

    def __post_init__(self) -> None:
        require_non_empty(self.series_id, field_name="MacroEvent.series_id")
        require_non_empty(self.provider, field_name="MacroEvent.provider")
        require_tz_aware(self.event_time, field_name="MacroEvent.event_time")
        require_tz_aware(self.arrival_time, field_name="MacroEvent.arrival_time")
        if self.arrival_time < self.event_time:
            raise MacroDataError(f"MacroEvent.arrival_time ({self.arrival_time}) must be >= event_time ({self.event_time})")
        require_non_negative_sequence(self.sequence)
        _parse_payload(self.payload)

    @property
    def value(self) -> Decimal:
        return _parse_payload(self.payload)[0]

    @property
    def previous_value(self) -> Decimal | None:
        return _parse_payload(self.payload)[1]

    @property
    def unit(self) -> str | None:
        return _parse_payload(self.payload)[2]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": MACRO_EVENT_KIND, "event_id": self.event_id, "series_id": self.series_id, "provider": self.provider,
            "event_time": serialize_timestamp(self.event_time, field_name="event_time"),
            "arrival_time": serialize_timestamp(self.arrival_time, field_name="arrival_time"), "sequence": self.sequence,
            "source_event_id": self.source_event_id, "payload": self.payload,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["event_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> MacroEvent:
        return cls(
            event_id=str(raw["event_id"]), series_id=str(raw["series_id"]), provider=str(raw["provider"]),
            event_time=deserialize_timestamp(raw["event_time"], field_name="event_time"),
            arrival_time=deserialize_timestamp(raw["arrival_time"], field_name="arrival_time"), sequence=int(str(raw["sequence"])),
            source_event_id=(None if raw.get("source_event_id") is None else str(raw["source_event_id"])),
            payload=as_json_dict(raw["payload"], field_name="payload"),
        )


def create_macro_event(
    *, series_id: str, provider: str, event_time: datetime, sequence: int, value: Decimal, previous_value: Decimal | None = None,
    unit: str | None = None, arrival_time: datetime | None = None, source_event_id: str | None = None,
) -> MacroEvent:
    resolved_arrival_time = event_time if arrival_time is None else arrival_time
    payload = _payload_to_json(value=value, previous_value=previous_value, unit=unit)
    provisional = MacroEvent(
        event_id="0" * 64, series_id=series_id, provider=provider, event_time=event_time, arrival_time=resolved_arrival_time,
        sequence=sequence, source_event_id=source_event_id, payload=payload,
    )
    event_id = compute_content_id(MACRO_EVENT_KIND, provisional.to_identity_payload())
    return MacroEvent(
        event_id=event_id, series_id=series_id, provider=provider, event_time=event_time, arrival_time=resolved_arrival_time,
        sequence=sequence, source_event_id=source_event_id, payload=payload,
    )


def is_macro_event_available_at(event: MacroEvent, as_of: datetime) -> bool:
    """Point-in-time availability: `event` may only be used by a
    computation attributed to `as_of` when `as_of >= event.event_time`.
    Never based on `arrival_time` -- see module docstring."""
    require_tz_aware(as_of, field_name="as_of")
    return as_of >= event.event_time


@contextmanager
def macro_data_lock(lock_path: Path) -> Iterator[None]:
    try:
        with experiment_lock(lock_path):
            yield
    except ExperimentLockError as exc:
        raise MarketDataLockError(f"Could not acquire macro event lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
    except OSError as exc:
        raise MarketDataLockError(f"Macro event lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc


class MacroEventStore:
    """Storage layout: `{storage_root}/macro_events/{provider}/{series_id}/
    events.jsonl` -- mirrors `events.MarketEventStore` exactly, partitioned
    by `series_id` instead of `instrument_id`."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _partition_dir(self, provider: str, series_id: str) -> Path:
        return self._root / "macro_events" / provider / series_id

    def _events_path(self, provider: str, series_id: str) -> Path:
        return self._partition_dir(provider, series_id) / "events.jsonl"

    def _lock_path(self, provider: str, series_id: str) -> Path:
        return self._partition_dir(provider, series_id) / ".events.lock"

    def read_events(self, provider: str, series_id: str) -> list[MacroEvent]:
        path = self._events_path(provider, series_id)
        if not path.is_file():
            return []
        events: list[MacroEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted macro event line for {provider}/{series_id}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted macro event line for {provider}/{series_id}: expected a JSON object")
            events.append(MacroEvent.from_json_dict(raw))
        return events

    def append(self, event: MacroEvent) -> MacroEvent:
        provider, series_id = event.provider, event.series_id
        lock_path = self._lock_path(provider, series_id)
        self._partition_dir(provider, series_id).mkdir(parents=True, exist_ok=True)
        with macro_data_lock(lock_path):
            existing_events = self.read_events(provider, series_id)
            if event.sequence < len(existing_events):
                existing = existing_events[event.sequence]
                if existing.event_id == event.event_id:
                    return existing
                raise MarketDataPersistenceError(
                    f"Conflicting append at sequence {event.sequence} for {provider}/{series_id}: existing event_id "
                    f"{existing.event_id!r} != new event_id {event.event_id!r}"
                )
            if event.sequence > len(existing_events):
                raise MarketDataPersistenceError(f"Event sequence {event.sequence} leaves a gap for {provider}/{series_id} (expected {len(existing_events)})")
            path = self._events_path(provider, series_id)
            with path.open("ab") as handle:
                handle.write(canonical_json_bytes(event.to_json_dict()))
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        return event

    def next_sequence(self, provider: str, series_id: str) -> int:
        return len(self.read_events(provider, series_id))
