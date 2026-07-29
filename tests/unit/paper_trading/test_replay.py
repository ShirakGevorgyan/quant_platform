"""Milestone 7, Section 32: bounded deterministic replay input. Covers
schema validation, chronological order, sequence continuity/strict
monotonicity, duplicate rejection, mixed-instrument rejection, corrupted-
file rejection, the mandatory single trailing `EndOfStreamEvent`, and
deterministic source identity."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_platform.core.exceptions import MarketEventError, MarketEventOrderError
from quant_platform.core.types import Timeframe
from quant_platform.paper_trading.events import create_bar_event, create_end_of_stream_event
from quant_platform.paper_trading.replay import (
    compute_replay_source_identity,
    load_replay_events,
    validate_replay_sequence,
    write_replay_events,
)

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)


def _bar(instrument: str, *, hour_offset: int, close: float, sequence: int) -> object:
    open_time = _T0 + timedelta(hours=hour_offset)
    return create_bar_event(instrument=instrument, interval=Timeframe.H1, open_time=open_time, open=close, high=close + 0.5, low=close - 0.5, close=close, sequence=sequence, source="test")


def _valid_sequence(n: int = 3) -> tuple:
    bars = tuple(_bar("X", hour_offset=i, close=100.0 + i, sequence=i + 1) for i in range(n))
    eos = create_end_of_stream_event(instrument="X", event_time=_T0 + timedelta(hours=n), sequence=n + 1, source="test")
    return (*bars, eos)


class TestValidateReplaySequenceHappyPath:
    def test_valid_sequence_passes(self) -> None:
        validate_replay_sequence(_valid_sequence())

    def test_empty_sequence_is_a_no_op(self) -> None:
        validate_replay_sequence(())


class TestValidateReplaySequenceTampering:
    def test_missing_end_of_stream_event_rejected(self) -> None:
        events = _valid_sequence()[:-1]
        with pytest.raises(MarketEventOrderError, match="EndOfStreamEvent"):
            validate_replay_sequence(events)

    def test_end_of_stream_event_not_last_rejected(self) -> None:
        bar0 = _bar("X", hour_offset=0, close=100.0, sequence=1)
        eos = create_end_of_stream_event(instrument="X", event_time=_T0 + timedelta(hours=1), sequence=2, source="test")
        bar2 = _bar("X", hour_offset=2, close=101.0, sequence=3)
        bar3 = _bar("X", hour_offset=3, close=102.0, sequence=4)
        events = (bar0, eos, bar2, bar3)
        with pytest.raises(MarketEventOrderError, match="exactly one EndOfStreamEvent"):
            validate_replay_sequence(events)

    def test_duplicate_event_id_rejected(self) -> None:
        events = _valid_sequence()
        tampered = (*events[:-1], events[-2], events[-1])
        with pytest.raises(MarketEventOrderError, match="duplicate event_id"):
            validate_replay_sequence(tampered)

    def test_duplicate_sequence_number_rejected(self) -> None:
        bar_a = _bar("X", hour_offset=0, close=100.0, sequence=1)
        bar_b = _bar("X", hour_offset=1, close=101.0, sequence=1)
        eos = create_end_of_stream_event(instrument="X", event_time=_T0 + timedelta(hours=2), sequence=2, source="test")
        with pytest.raises(MarketEventOrderError, match="duplicate sequence"):
            validate_replay_sequence((bar_a, bar_b, eos))

    def test_non_increasing_sequence_rejected(self) -> None:
        bar_a = _bar("X", hour_offset=0, close=100.0, sequence=5)
        bar_b = _bar("X", hour_offset=1, close=101.0, sequence=3)
        eos = create_end_of_stream_event(instrument="X", event_time=_T0 + timedelta(hours=2), sequence=6, source="test")
        with pytest.raises(MarketEventOrderError, match="strictly increasing"):
            validate_replay_sequence((bar_a, bar_b, eos))

    def test_non_contiguous_sequence_rejected_when_required(self) -> None:
        bar_a = _bar("X", hour_offset=0, close=100.0, sequence=1)
        bar_b = _bar("X", hour_offset=1, close=101.0, sequence=5)
        eos = create_end_of_stream_event(instrument="X", event_time=_T0 + timedelta(hours=2), sequence=6, source="test")
        with pytest.raises(MarketEventOrderError, match="not contiguous"):
            validate_replay_sequence((bar_a, bar_b, eos), require_contiguous_sequence=True)

    def test_non_contiguous_sequence_allowed_by_default(self) -> None:
        bar_a = _bar("X", hour_offset=0, close=100.0, sequence=1)
        bar_b = _bar("X", hour_offset=1, close=101.0, sequence=5)
        eos = create_end_of_stream_event(instrument="X", event_time=_T0 + timedelta(hours=2), sequence=6, source="test")
        validate_replay_sequence((bar_a, bar_b, eos))

    def test_mixed_instruments_rejected_by_default(self) -> None:
        bar_a = _bar("X", hour_offset=0, close=100.0, sequence=1)
        bar_b = _bar("Y", hour_offset=1, close=101.0, sequence=2)
        eos = create_end_of_stream_event(instrument="X", event_time=_T0 + timedelta(hours=2), sequence=3, source="test")
        with pytest.raises(MarketEventOrderError, match="mixes multiple instruments"):
            validate_replay_sequence((bar_a, bar_b, eos))

    def test_mixed_instruments_allowed_when_explicit(self) -> None:
        bar_a = _bar("X", hour_offset=0, close=100.0, sequence=1)
        bar_b = _bar("Y", hour_offset=1, close=101.0, sequence=2)
        eos = create_end_of_stream_event(instrument="X", event_time=_T0 + timedelta(hours=2), sequence=3, source="test")
        validate_replay_sequence((bar_a, bar_b, eos), allow_mixed_instruments=True)

    def test_out_of_chronological_order_rejected(self) -> None:
        bar_a = _bar("X", hour_offset=5, close=100.0, sequence=1)
        bar_b = _bar("X", hour_offset=1, close=101.0, sequence=2)
        eos = create_end_of_stream_event(instrument="X", event_time=_T0 + timedelta(hours=6), sequence=3, source="test")
        with pytest.raises(MarketEventOrderError, match="not in chronological order"):
            validate_replay_sequence((bar_a, bar_b, eos))


class TestLoadReplayEventsFromFile:
    def test_round_trips_a_written_sequence(self, tmp_path) -> None:
        events = _valid_sequence()
        path = tmp_path / "source.jsonl"
        write_replay_events(path, events)
        loaded = load_replay_events(path)
        assert loaded == events

    def test_missing_file_raises_market_event_error(self, tmp_path) -> None:
        with pytest.raises(MarketEventError, match="not found"):
            load_replay_events(tmp_path / "does_not_exist.jsonl")

    def test_corrupted_json_line_raises_market_event_error(self, tmp_path) -> None:
        path = tmp_path / "corrupted.jsonl"
        path.write_text("{not valid json\n", encoding="utf-8")
        with pytest.raises(MarketEventError, match="corrupted JSON"):
            load_replay_events(path)

    def test_unknown_event_kind_raises_market_event_error(self, tmp_path) -> None:
        path = tmp_path / "unknown_kind.jsonl"
        path.write_text('{"kind": "not_a_real_kind"}\n', encoding="utf-8")
        with pytest.raises(MarketEventError):
            load_replay_events(path)

    def test_blank_lines_are_skipped(self, tmp_path) -> None:
        events = _valid_sequence()
        path = tmp_path / "with_blanks.jsonl"
        write_replay_events(path, events)
        original = path.read_text(encoding="utf-8")
        path.write_text("\n\n" + original + "\n\n", encoding="utf-8")
        loaded = load_replay_events(path)
        assert loaded == events

    def test_cross_event_validation_still_runs_on_loaded_file(self, tmp_path) -> None:
        events = _valid_sequence()[:-1]  # missing EndOfStreamEvent
        path = tmp_path / "incomplete.jsonl"
        write_replay_events(path, events)
        with pytest.raises(MarketEventOrderError, match="EndOfStreamEvent"):
            load_replay_events(path)


class TestReplaySourceIdentity:
    def test_deterministic_across_calls(self) -> None:
        events = _valid_sequence()
        assert compute_replay_source_identity(events) == compute_replay_source_identity(events)

    def test_differs_for_different_sequences(self) -> None:
        assert compute_replay_source_identity(_valid_sequence(3)) != compute_replay_source_identity(_valid_sequence(4))

    def test_stable_across_file_round_trip(self, tmp_path) -> None:
        events = _valid_sequence()
        path = tmp_path / "source.jsonl"
        write_replay_events(path, events)
        loaded = load_replay_events(path)
        assert compute_replay_source_identity(events) == compute_replay_source_identity(loaded)
