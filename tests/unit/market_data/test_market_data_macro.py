"""Unit tests for `market_data.macro`: `MacroEvent` construction/identity,
point-in-time availability, and `MacroEventStore`'s append-only semantics."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from quant_platform.core.exceptions import MacroDataError, MarketDataError, MarketDataPersistenceError
from quant_platform.market_data.macro import MacroEventStore, create_macro_event, is_macro_event_available_at

_T0 = datetime(2026, 1, 5, 13, 30, tzinfo=timezone.utc)


def _event(**overrides: object):
    base: dict[str, object] = {"series_id": "US_NFP", "provider": "econcal", "event_time": _T0, "sequence": 0, "value": Decimal("200000")}
    base.update(overrides)
    return create_macro_event(**base)  # type: ignore[arg-type]


class TestMacroEventConstruction:
    def test_round_trips_through_json(self) -> None:
        event = _event(previous_value=Decimal("190000"), unit="jobs")
        assert type(event).from_json_dict(event.to_json_dict()) == event

    def test_identical_arguments_produce_identical_ids(self) -> None:
        assert _event().event_id == _event().event_id

    def test_different_series_id_changes_the_id(self) -> None:
        assert _event().event_id != _event(series_id="US_CPI").event_id

    def test_arrival_time_before_event_time_is_rejected(self) -> None:
        with pytest.raises(MacroDataError):
            _event(arrival_time=_T0 - timedelta(minutes=1))

    def test_empty_series_id_is_rejected(self) -> None:
        with pytest.raises(MarketDataError):
            _event(series_id="")


class TestPointInTimeAvailability:
    def test_not_available_before_release(self) -> None:
        event = _event()
        assert is_macro_event_available_at(event, _T0 - timedelta(seconds=1)) is False

    def test_available_at_or_after_release(self) -> None:
        event = _event()
        assert is_macro_event_available_at(event, _T0) is True
        assert is_macro_event_available_at(event, _T0 + timedelta(days=1)) is True


class TestMacroEventStore:
    def test_append_and_read_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MacroEventStore(Path(tmp))
            event = _event(sequence=0)
            store.append(event)
            assert store.read_events("econcal", "US_NFP") == [event]

    def test_identical_reappend_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MacroEventStore(Path(tmp))
            event = _event(sequence=0)
            store.append(event)
            store.append(event)
            assert len(store.read_events("econcal", "US_NFP")) == 1

    def test_conflicting_append_at_same_sequence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MacroEventStore(Path(tmp))
            store.append(_event(sequence=0))
            with pytest.raises(MarketDataPersistenceError):
                store.append(_event(sequence=0, value=Decimal("999999")))

    def test_sequence_gap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MacroEventStore(Path(tmp))
            with pytest.raises(MarketDataPersistenceError):
                store.append(_event(sequence=1))

    def test_different_series_partitions_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MacroEventStore(Path(tmp))
            store.append(_event(series_id="US_NFP", sequence=0))
            store.append(_event(series_id="US_CPI", sequence=0))
            assert len(store.read_events("econcal", "US_NFP")) == 1
            assert len(store.read_events("econcal", "US_CPI")) == 1
