"""Milestone 7, Section 6: `ReplayClock`/`ForwardEventClock`/
`ManualTestClock` and the pure latency-arithmetic functions. Explicitly
covers the spec's named edge cases: daylight-saving transitions, equal
timestamps, gaps, backward timestamps, and duplicate sequence numbers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from quant_platform.core.exceptions import ClockError
from quant_platform.paper_trading.clock import (
    ForwardEventClock,
    ManualTestClock,
    ReplayClock,
    acceptance_time_for,
    decision_time_for,
    fill_eligible_time_for,
    submission_time_for,
)
from quant_platform.paper_trading.specs import LatencyPolicySpec

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)


class TestReplayClockBasics:
    def test_now_before_any_advance_raises(self) -> None:
        clock = ReplayClock()
        with pytest.raises(ClockError, match="before"):
            clock.now()

    def test_advance_to_sets_now(self) -> None:
        clock = ReplayClock()
        clock.advance_to(_T0, sequence=1)
        assert clock.now() == _T0

    def test_engine_processing_time_matches_now_in_replay(self) -> None:
        clock = ReplayClock()
        clock.advance_to(_T0, sequence=1)
        assert clock.engine_processing_time() == clock.now()

    def test_naive_timestamp_rejected(self) -> None:
        clock = ReplayClock()
        with pytest.raises(ClockError, match="timezone-aware"):
            clock.advance_to(datetime(2026, 1, 5, 10, 0, 0), sequence=1)

    def test_equal_timestamps_with_incrementing_sequence_allowed(self) -> None:
        """Two events at the identical instant (tie broken by sequence) is
        legal -- e.g. a bar close and a session marker firing together."""
        clock = ReplayClock()
        clock.advance_to(_T0, sequence=1)
        clock.advance_to(_T0, sequence=2)
        assert clock.now() == _T0

    def test_gap_between_events_allowed(self) -> None:
        clock = ReplayClock()
        clock.advance_to(_T0, sequence=1)
        clock.advance_to(_T0 + timedelta(days=3), sequence=2)
        assert clock.now() == _T0 + timedelta(days=3)

    def test_backward_timestamp_rejected(self) -> None:
        clock = ReplayClock()
        clock.advance_to(_T0, sequence=1)
        with pytest.raises(ClockError, match="backward"):
            clock.advance_to(_T0 - timedelta(seconds=1), sequence=2)

    def test_duplicate_sequence_number_rejected(self) -> None:
        clock = ReplayClock()
        clock.advance_to(_T0, sequence=1)
        with pytest.raises(ClockError, match="sequence"):
            clock.advance_to(_T0 + timedelta(seconds=1), sequence=1)

    def test_non_increasing_sequence_rejected_even_with_later_time(self) -> None:
        clock = ReplayClock()
        clock.advance_to(_T0, sequence=5)
        with pytest.raises(ClockError, match="sequence"):
            clock.advance_to(_T0 + timedelta(seconds=1), sequence=3)

    def test_daylight_saving_transition_ordering_is_correct(self) -> None:
        """US DST spring-forward: 2026-03-08 02:00 America/New_York jumps
        to 03:00. Two tz-aware instants straddling the transition must
        still compare correctly in absolute (UTC) terms -- proving this
        package's exclusive use of tz-aware datetimes makes it immune to
        wall-clock DST ambiguity by construction."""
        ny = ZoneInfo("America/New_York")
        before_transition = datetime(2026, 3, 8, 1, 30, 0, tzinfo=ny)
        after_transition = datetime(2026, 3, 8, 3, 30, 0, tzinfo=ny)
        assert after_transition > before_transition
        clock = ReplayClock()
        clock.advance_to(before_transition, sequence=1)
        clock.advance_to(after_transition, sequence=2)
        assert clock.now() == after_transition


class TestReplayClockPauseSingleStepRunUntil:
    def test_pause_and_resume(self) -> None:
        clock = ReplayClock()
        assert not clock.is_paused
        clock.pause()
        assert clock.is_paused
        clock.resume()
        assert not clock.is_paused

    def test_single_step_auto_pauses_after_each_advance(self) -> None:
        clock = ReplayClock()
        clock.set_single_step(True)
        clock.advance_to(_T0, sequence=1)
        assert clock.is_paused
        clock.resume()
        clock.advance_to(_T0 + timedelta(seconds=1), sequence=2)
        assert clock.is_paused

    def test_run_to_end_never_pauses_without_single_step(self) -> None:
        clock = ReplayClock()
        for i in range(5):
            clock.advance_to(_T0 + timedelta(seconds=i), sequence=i + 1)
        assert not clock.is_paused

    def test_run_until_timestamp_boundary_is_inclusive(self) -> None:
        clock = ReplayClock()
        clock.set_run_until(_T0)
        assert clock.should_stop_before(_T0) is False
        assert clock.should_stop_before(_T0 + timedelta(seconds=1)) is True

    def test_no_run_until_never_stops(self) -> None:
        clock = ReplayClock()
        assert clock.should_stop_before(_T0 + timedelta(days=365)) is False


class TestForwardEventClock:
    def test_advance_to_sets_now(self) -> None:
        clock = ForwardEventClock()
        clock.advance_to(_T0, sequence=1)
        assert clock.now() == _T0

    def test_engine_processing_time_uses_injected_now_fn(self) -> None:
        fixed_wall_clock_time = _T0 + timedelta(minutes=5)
        clock = ForwardEventClock(now_fn=lambda: fixed_wall_clock_time)
        clock.advance_to(_T0, sequence=1)
        assert clock.engine_processing_time() == fixed_wall_clock_time
        assert clock.engine_processing_time() != clock.now()

    def test_backward_timestamp_rejected(self) -> None:
        clock = ForwardEventClock()
        clock.advance_to(_T0, sequence=1)
        with pytest.raises(ClockError, match="backward"):
            clock.advance_to(_T0 - timedelta(seconds=1), sequence=2)

    def test_duplicate_sequence_rejected(self) -> None:
        clock = ForwardEventClock()
        clock.advance_to(_T0, sequence=1)
        with pytest.raises(ClockError, match="sequence"):
            clock.advance_to(_T0, sequence=1)


class TestManualTestClock:
    def test_starts_at_given_time(self) -> None:
        clock = ManualTestClock(start_time=_T0)
        assert clock.now() == _T0

    def test_naive_start_time_rejected(self) -> None:
        with pytest.raises(ClockError, match="timezone-aware"):
            ManualTestClock(start_time=datetime(2026, 1, 5, 10, 0, 0))

    def test_set_time_allows_arbitrary_backward_jump(self) -> None:
        """Deliberately permissive -- other modules' tests use this to
        exercise THEIR OWN backward-timestamp rejection logic."""
        clock = ManualTestClock(start_time=_T0)
        clock.set_time(_T0 - timedelta(days=1))
        assert clock.now() == _T0 - timedelta(days=1)

    def test_advance_to_ignores_sequence_validation(self) -> None:
        clock = ManualTestClock(start_time=_T0)
        clock.advance_to(_T0, sequence=1)
        clock.advance_to(_T0, sequence=1)
        assert clock.now() == _T0


class TestLatencyArithmetic:
    def _policy(self, *, decision_to_submit_ms: int = 10, submit_to_accept_ms: int = 20, accept_to_fill_eligible_ms: int = 30) -> LatencyPolicySpec:
        return LatencyPolicySpec(decision_to_submit_ms=decision_to_submit_ms, submit_to_accept_ms=submit_to_accept_ms, accept_to_fill_eligible_ms=accept_to_fill_eligible_ms)

    def test_decision_time_equals_event_time(self) -> None:
        assert decision_time_for(_T0) == _T0

    def test_submission_time_adds_decision_to_submit_latency(self) -> None:
        assert submission_time_for(_T0, self._policy(decision_to_submit_ms=50)) == _T0 + timedelta(milliseconds=50)

    def test_acceptance_time_adds_submit_to_accept_latency(self) -> None:
        submission_time = _T0 + timedelta(milliseconds=50)
        assert acceptance_time_for(submission_time, self._policy(submit_to_accept_ms=75)) == submission_time + timedelta(milliseconds=75)

    def test_fill_eligible_time_adds_accept_to_fill_latency(self) -> None:
        acceptance_time = _T0 + timedelta(milliseconds=125)
        assert fill_eligible_time_for(acceptance_time, self._policy(accept_to_fill_eligible_ms=100)) == acceptance_time + timedelta(milliseconds=100)

    def test_zero_latency_policy_is_a_no_op_chain(self) -> None:
        zero_policy = self._policy(decision_to_submit_ms=0, submit_to_accept_ms=0, accept_to_fill_eligible_ms=0)
        decision_time = decision_time_for(_T0)
        submission_time = submission_time_for(decision_time, zero_policy)
        acceptance_time = acceptance_time_for(submission_time, zero_policy)
        fill_time = fill_eligible_time_for(acceptance_time, zero_policy)
        assert decision_time == submission_time == acceptance_time == fill_time == _T0

    def test_full_chain_is_monotonically_non_decreasing(self) -> None:
        policy = self._policy()
        decision_time = decision_time_for(_T0)
        submission_time = submission_time_for(decision_time, policy)
        acceptance_time = acceptance_time_for(submission_time, policy)
        fill_time = fill_eligible_time_for(acceptance_time, policy)
        assert decision_time <= submission_time <= acceptance_time <= fill_time
