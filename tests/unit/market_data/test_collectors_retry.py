"""Retry policy tests (Milestone 10, Phase 4A) -- `classify_failure`/
`plan_next_wait_seconds`/`parse_retry_after` are PURE (no sleep, no
wall-clock read), so every test here runs in microseconds."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_platform.core.exceptions import CollectorError
from quant_platform.market_data.collectors.retry import (
    DEFAULT_RETRYABLE_STATUSES,
    RetryFailureKind,
    RetryOutcome,
    classify_failure,
    create_retry_policy,
    parse_retry_after,
    plan_next_wait_seconds,
)

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _policy(**overrides):
    defaults = {"max_attempts": 3, "backoff_schedule_seconds": (1.0, 2.0)}
    defaults.update(overrides)
    return create_retry_policy(**defaults)


class TestRetryPolicyIdentity:
    def test_deterministic_id_for_identical_construction(self) -> None:
        assert _policy().retry_policy_id == _policy().retry_policy_id

    def test_max_attempts_change_changes_id(self) -> None:
        assert _policy(max_attempts=3).retry_policy_id != _policy(max_attempts=4, backoff_schedule_seconds=(1.0, 2.0, 3.0)).retry_policy_id

    def test_backoff_schedule_change_changes_id(self) -> None:
        assert _policy(backoff_schedule_seconds=(1.0, 2.0)).retry_policy_id != _policy(backoff_schedule_seconds=(5.0, 5.0)).retry_policy_id

    def test_retryable_statuses_change_changes_id(self) -> None:
        a = _policy(retryable_statuses=frozenset({500}))
        b = _policy(retryable_statuses=frozenset({500, 503}))
        assert a.retry_policy_id != b.retry_policy_id

    def test_respect_retry_after_change_changes_id(self) -> None:
        a = _policy(respect_retry_after=True)
        b = _policy(respect_retry_after=False)
        assert a.retry_policy_id != b.retry_policy_id

    def test_round_trip_through_json(self) -> None:
        policy = _policy()
        from quant_platform.market_data.collectors.retry import RetryPolicy

        restored = RetryPolicy.from_json_dict(policy.to_json_dict())
        assert restored == policy


class TestRetryPolicyConstruction:
    def test_max_attempts_below_one_is_rejected(self) -> None:
        with pytest.raises(CollectorError):
            create_retry_policy(max_attempts=0, backoff_schedule_seconds=())

    def test_insufficient_backoff_entries_is_rejected(self) -> None:
        with pytest.raises(CollectorError):
            create_retry_policy(max_attempts=3, backoff_schedule_seconds=(1.0,))

    def test_negative_backoff_entry_is_rejected(self) -> None:
        with pytest.raises(CollectorError):
            create_retry_policy(max_attempts=2, backoff_schedule_seconds=(-1.0,))

    def test_cannot_configure_never_retry_status_as_retryable(self) -> None:
        for status in (400, 401, 403):
            with pytest.raises(CollectorError):
                create_retry_policy(max_attempts=2, backoff_schedule_seconds=(1.0,), retryable_statuses=frozenset({status}))


class TestClassifyFailure:
    def test_connect_timeout_is_always_retryable(self) -> None:
        assert classify_failure(kind=RetryFailureKind.CONNECT_TIMEOUT, status_code=None, policy=_policy()) is RetryOutcome.RETRY

    def test_read_timeout_is_always_retryable(self) -> None:
        assert classify_failure(kind=RetryFailureKind.READ_TIMEOUT, status_code=None, policy=_policy()) is RetryOutcome.RETRY

    def test_malformed_response_is_retryable(self) -> None:
        assert classify_failure(kind=RetryFailureKind.MALFORMED_RESPONSE, status_code=None, policy=_policy()) is RetryOutcome.RETRY

    def test_integrity_failure_is_retryable(self) -> None:
        assert classify_failure(kind=RetryFailureKind.INTEGRITY_FAILURE, status_code=None, policy=_policy()) is RetryOutcome.RETRY

    @pytest.mark.parametrize("status", sorted(DEFAULT_RETRYABLE_STATUSES))
    def test_default_retryable_statuses_are_retried(self, status: int) -> None:
        assert classify_failure(kind=RetryFailureKind.HTTP_STATUS, status_code=status, policy=_policy()) is RetryOutcome.RETRY

    @pytest.mark.parametrize("status", [400, 401, 403])
    def test_permanent_client_errors_are_never_retried(self, status: int) -> None:
        assert classify_failure(kind=RetryFailureKind.HTTP_STATUS, status_code=status, policy=_policy()) is RetryOutcome.STOP

    def test_unlisted_status_is_not_retried(self) -> None:
        assert classify_failure(kind=RetryFailureKind.HTTP_STATUS, status_code=404, policy=_policy()) is RetryOutcome.STOP

    def test_unknown_series_style_error_treated_as_non_retryable_via_4xx_status(self) -> None:
        # A 404 (unknown series id, in FRED's real semantics) is not in
        # DEFAULT_RETRYABLE_STATUSES, so is correctly non-retryable.
        assert classify_failure(kind=RetryFailureKind.HTTP_STATUS, status_code=404, policy=_policy()) is RetryOutcome.STOP


class TestParseRetryAfter:
    def test_digit_string_is_parsed_as_seconds(self) -> None:
        assert parse_retry_after("120", now=T0) == 120.0

    def test_http_date_is_parsed_relative_to_now(self) -> None:
        future = T0 + timedelta(seconds=90)
        header = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
        result = parse_retry_after(header, now=T0)
        assert result is not None
        assert abs(result - 90.0) < 1.0

    def test_past_http_date_clamps_to_zero(self) -> None:
        past = T0 - timedelta(seconds=90)
        header = past.strftime("%a, %d %b %Y %H:%M:%S GMT")
        assert parse_retry_after(header, now=T0) == 0.0

    def test_malformed_value_returns_none(self) -> None:
        assert parse_retry_after("not-a-valid-value", now=T0) is None

    def test_empty_value_returns_none(self) -> None:
        assert parse_retry_after("", now=T0) is None
        assert parse_retry_after("   ", now=T0) is None

    def test_negative_digit_string_is_not_treated_as_digits(self) -> None:
        # "-5" is not `.isdigit()`, so it must fall through to the
        # HTTP-date parser and fail closed to None, never a negative wait.
        assert parse_retry_after("-5", now=T0) is None


class TestPlanNextWaitSeconds:
    def test_retry_after_takes_precedence_over_backoff_schedule(self) -> None:
        policy = _policy(respect_retry_after=True)
        assert plan_next_wait_seconds(policy, attempt_number=1, retry_after_seconds=42.0) == 42.0

    def test_backoff_schedule_used_when_retry_after_disrespected(self) -> None:
        policy = _policy(respect_retry_after=False)
        assert plan_next_wait_seconds(policy, attempt_number=1, retry_after_seconds=42.0) == 1.0

    def test_backoff_schedule_used_when_no_retry_after(self) -> None:
        policy = _policy()
        assert plan_next_wait_seconds(policy, attempt_number=1, retry_after_seconds=None) == 1.0
        assert plan_next_wait_seconds(policy, attempt_number=2, retry_after_seconds=None) == 2.0

    def test_out_of_range_attempt_number_raises(self) -> None:
        policy = _policy()
        with pytest.raises(CollectorError):
            plan_next_wait_seconds(policy, attempt_number=99, retry_after_seconds=None)

    def test_deterministic_attempt_sequence_no_real_sleep(self) -> None:
        """Confirms these functions are genuinely pure: calling them many
        times with identical inputs, with no `time.sleep` anywhere in
        this test, produces identical, immediate results."""
        policy = _policy()
        results = [plan_next_wait_seconds(policy, attempt_number=1, retry_after_seconds=None) for _ in range(1000)]
        assert results == [1.0] * 1000
