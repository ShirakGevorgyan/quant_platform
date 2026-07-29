"""Event clocks (Milestone 7, Section 6). Every timing decision inside
`paper_trading` goes through one of the three clocks below or the pure
latency-arithmetic functions at the bottom of this module -- NEVER through
a direct `datetime.now()`/`time.time()` call. `ReplayClock` and
`ManualTestClock` are fully deterministic and never touch the wall clock
at all (a hard requirement for byte-identical `REPLAY_PAPER` replay);
`ForwardEventClock` is the one place a real wall-clock read is permitted,
and only for a diagnostic `engine_processing_time`, never for a decision
input.

This module distinguishes all 7 of Section 6's time concepts:
  1. market event time      -- `events.market_event_time(event)`
  2. event receive time     -- `QuoteEvent.receive_time` (event-carried)
  3. engine processing time -- `Clock.engine_processing_time()`
  4. strategy decision time -- `decision_time_for(event_time)`
  5. order submission time  -- `submission_time_for(decision_time, ...)`
  6. broker acceptance time -- `acceptance_time_for(submission_time, ...)`
  7. fill-eligible time     -- `fill_eligible_time_for(acceptance_time, ...)`
Items 4-7 are pure arithmetic on whatever event time drove them -- latency
is always an arithmetic time-delta applied to event time, never a real
`sleep`, which is what makes `REPLAY_PAPER` reproducible independent of
how fast or slow the host machine actually runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol

from quant_platform.core.exceptions import ClockError
from quant_platform.paper_trading.specs import LatencyPolicySpec


def _require_tz_aware(ts: datetime, *, field_name: str) -> None:
    if ts.tzinfo is None:
        raise ClockError(f"{field_name} must be timezone-aware, got naive datetime {ts!r}")


def _real_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Clock(Protocol):
    """The common interface every clock implementation exposes --
    `runner.py`/`execution.py`/`strategy.py` depend on this Protocol, never
    on a concrete clock class, so swapping `REPLAY_PAPER` for `FORWARD_
    PAPER` never touches decision logic."""

    def now(self) -> datetime: ...
    def engine_processing_time(self) -> datetime: ...
    def advance_to(self, event_time: datetime, *, sequence: int) -> None: ...


# --------------------------------------------------------------------------
# ReplayClock
# --------------------------------------------------------------------------
@dataclass
class ReplayClock:
    """Fully deterministic -- driven entirely by the `event_time` of
    whatever market event `advance_to` is given, and NEVER reads the real
    wall clock. Supports pause/single-step/run-to-end/run-until-timestamp
    exactly per Section 6, all as pure state transitions (no timers, no
    threads)."""

    _current_time: datetime | None = field(default=None, init=False)
    _last_sequence: int | None = field(default=None, init=False)
    _paused: bool = field(default=False, init=False)
    _single_step: bool = field(default=False, init=False)
    _run_until: datetime | None = field(default=None, init=False)

    def now(self) -> datetime:
        if self._current_time is None:
            raise ClockError("ReplayClock.now() called before the first advance_to()")
        return self._current_time

    def engine_processing_time(self) -> datetime:
        """In REPLAY_PAPER, processing is entirely event-time-driven --
        there is no separate wall-clock component, so this coincides
        exactly with the current event time."""
        return self.now()

    def advance_to(self, event_time: datetime, *, sequence: int) -> None:
        _require_tz_aware(event_time, field_name="event_time")
        if self._current_time is not None and event_time < self._current_time:
            raise ClockError(f"ReplayClock: backward timestamp {event_time} < current {self._current_time} -- a replay/forward stream may never be silently reordered")
        if self._last_sequence is not None and sequence <= self._last_sequence:
            raise ClockError(f"ReplayClock: non-increasing sequence {sequence} <= last {self._last_sequence} (duplicate or out-of-order delivery)")
        self._current_time = event_time
        self._last_sequence = sequence
        if self._single_step:
            self._paused = True

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def set_single_step(self, enabled: bool) -> None:
        self._single_step = enabled

    def set_run_until(self, timestamp: datetime | None) -> None:
        if timestamp is not None:
            _require_tz_aware(timestamp, field_name="run_until timestamp")
        self._run_until = timestamp

    def should_stop_before(self, event_time: datetime) -> bool:
        """`True` if consuming an event at `event_time` would cross the
        configured run-until boundary -- the runner checks this BEFORE
        calling `advance_to`, so a `run_until` exactly on an event's own
        time still processes that event (inclusive boundary)."""
        return self._run_until is not None and event_time > self._run_until


# --------------------------------------------------------------------------
# ForwardEventClock
# --------------------------------------------------------------------------
@dataclass
class ForwardEventClock:
    """Driven entirely by whatever event is handed to `advance_to` --
    never hidden polling. `engine_processing_time()` is the ONE place in
    this whole package a real wall-clock read is permitted (via the
    injectable `now_fn`, defaulting to the real clock) -- purely
    diagnostic (e.g. "how much wall-clock latency did we actually
    observe"), never a decision input, and never exercised by a
    `REPLAY_PAPER` session at all."""

    now_fn: Callable[[], datetime] = field(default=_real_utc_now)
    _current_time: datetime | None = field(default=None, init=False)
    _last_sequence: int | None = field(default=None, init=False)

    def now(self) -> datetime:
        if self._current_time is None:
            raise ClockError("ForwardEventClock.now() called before the first advance_to()")
        return self._current_time

    def engine_processing_time(self) -> datetime:
        return self.now_fn()

    def advance_to(self, event_time: datetime, *, sequence: int) -> None:
        _require_tz_aware(event_time, field_name="event_time")
        if self._current_time is not None and event_time < self._current_time:
            raise ClockError(f"ForwardEventClock: backward timestamp {event_time} < current {self._current_time} -- a live/forward stream may never be silently reordered")
        if self._last_sequence is not None and sequence <= self._last_sequence:
            raise ClockError(f"ForwardEventClock: non-increasing sequence {sequence} <= last {self._last_sequence} (duplicate or out-of-order delivery)")
        self._current_time = event_time
        self._last_sequence = sequence


# --------------------------------------------------------------------------
# ManualTestClock
# --------------------------------------------------------------------------
@dataclass
class ManualTestClock:
    """Deliberately permissive (no monotonicity enforcement) -- exists so
    OTHER modules' unit tests can drive an arbitrary, hand-picked sequence
    of times (including intentionally backward ones, to test THEIR OWN
    rejection logic) without constructing a full `ReplayClock`."""

    start_time: datetime

    def __post_init__(self) -> None:
        _require_tz_aware(self.start_time, field_name="start_time")
        self._current_time: datetime = self.start_time

    def now(self) -> datetime:
        return self._current_time

    def engine_processing_time(self) -> datetime:
        return self._current_time

    def advance_to(self, event_time: datetime, *, sequence: int = 0) -> None:  # noqa: ARG002 -- deliberately unvalidated, see class docstring
        _require_tz_aware(event_time, field_name="event_time")
        self._current_time = event_time

    def set_time(self, new_time: datetime) -> None:
        _require_tz_aware(new_time, field_name="new_time")
        self._current_time = new_time


# --------------------------------------------------------------------------
# Latency arithmetic (Section 6 items 4-7) -- pure functions of a time and
# a `LatencyPolicySpec`, independent of which `Clock` implementation is
# driving the session.
# --------------------------------------------------------------------------
def decision_time_for(event_time: datetime) -> datetime:
    """A strategy decision is made at exactly the event time that
    triggered it -- no artificial latency between an event arriving and a
    decision being formed; all configured latency applies AFTER the
    decision (submission/acceptance/fill), per Section 6/10.4."""
    return event_time


def submission_time_for(decision_time: datetime, latency_policy: LatencyPolicySpec) -> datetime:
    return decision_time + timedelta(milliseconds=latency_policy.decision_to_submit_ms)


def acceptance_time_for(submission_time: datetime, latency_policy: LatencyPolicySpec) -> datetime:
    return submission_time + timedelta(milliseconds=latency_policy.submit_to_accept_ms)


def fill_eligible_time_for(acceptance_time: datetime, latency_policy: LatencyPolicySpec) -> datetime:
    """The EARLIEST time a fill may occur -- the actual fill time used by
    `execution.py` is `max(this value, the market event time that actually
    satisfies the order's fill condition)`, since a fill can never occur
    before the market data that justifies it exists."""
    return acceptance_time + timedelta(milliseconds=latency_policy.accept_to_fill_eligible_ms)


__all__ = [
    "Clock",
    "ForwardEventClock",
    "ManualTestClock",
    "ReplayClock",
    "acceptance_time_for",
    "decision_time_for",
    "fill_eligible_time_for",
    "submission_time_for",
]
