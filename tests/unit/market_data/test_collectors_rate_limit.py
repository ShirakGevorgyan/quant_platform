"""Rate-limit tests (Milestone 10, Phase 4A) -- pure immutable
token-bucket model, caller supplies `now`, no global singleton, no
sleeping anywhere in this file."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import CollectorError
from quant_platform.market_data.collectors.rate_limit import (
    RateLimitPolicy,
    create_rate_limit_policy,
    initial_bucket_state,
    seconds_until_available,
    try_acquire,
)

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _policy(max_tokens: str = "5", refill_rate: str = "1") -> RateLimitPolicy:
    return create_rate_limit_policy(max_tokens=Decimal(max_tokens), refill_rate_per_second=Decimal(refill_rate))


class TestRateLimitPolicyIdentity:
    def test_deterministic_id(self) -> None:
        assert _policy().rate_limit_policy_id == _policy().rate_limit_policy_id

    def test_max_tokens_change_changes_id(self) -> None:
        assert _policy(max_tokens="5").rate_limit_policy_id != _policy(max_tokens="10").rate_limit_policy_id

    def test_refill_rate_change_changes_id(self) -> None:
        assert _policy(refill_rate="1").rate_limit_policy_id != _policy(refill_rate="2").rate_limit_policy_id

    def test_round_trip_through_json(self) -> None:
        policy = _policy()
        assert RateLimitPolicy.from_json_dict(policy.to_json_dict()) == policy


class TestRateLimitPolicyConstruction:
    def test_non_positive_max_tokens_rejected(self) -> None:
        with pytest.raises(CollectorError):
            create_rate_limit_policy(max_tokens=Decimal(0), refill_rate_per_second=Decimal(1))

    def test_non_positive_refill_rate_rejected(self) -> None:
        with pytest.raises(CollectorError):
            create_rate_limit_policy(max_tokens=Decimal(1), refill_rate_per_second=Decimal(0))


class TestTryAcquire:
    def test_available_token_is_acquired(self) -> None:
        policy = _policy()
        state = initial_bucket_state(policy, now=T0)
        acquired, new_state = try_acquire(state, policy, now=T0)
        assert acquired
        assert new_state.tokens == Decimal(4)

    def test_unavailable_token_after_bucket_drained(self) -> None:
        policy = _policy(max_tokens="1", refill_rate="0.001")
        state = initial_bucket_state(policy, now=T0)
        acquired1, state = try_acquire(state, policy, now=T0)
        assert acquired1
        acquired2, state = try_acquire(state, policy, now=T0)
        assert not acquired2
        assert state.tokens == Decimal(0)

    def test_deterministic_refill_after_elapsed_time(self) -> None:
        policy = _policy(max_tokens="5", refill_rate="1")
        state = initial_bucket_state(policy, now=T0)
        for _ in range(5):
            acquired, state = try_acquire(state, policy, now=T0)
            assert acquired
        acquired, state = try_acquire(state, policy, now=T0)
        assert not acquired
        later = T0 + timedelta(seconds=3)
        acquired, state = try_acquire(state, policy, now=later)
        assert acquired
        # 3 seconds elapsed at 1 token/sec => 3 tokens refilled, 1 consumed -> 2 remain
        assert state.tokens == Decimal(2)

    def test_refill_is_capped_at_max_tokens(self) -> None:
        policy = _policy(max_tokens="5", refill_rate="1")
        state = initial_bucket_state(policy, now=T0)
        much_later = T0 + timedelta(hours=1)
        acquired, state = try_acquire(state, policy, now=much_later)
        assert acquired
        assert state.tokens == Decimal(4)  # capped at max_tokens (5), minus the 1 acquired

    def test_backward_time_movement_is_rejected(self) -> None:
        policy = _policy()
        state = initial_bucket_state(policy, now=T0)
        earlier = T0 - timedelta(seconds=1)
        with pytest.raises(CollectorError):
            try_acquire(state, policy, now=earlier)

    def test_zero_or_negative_tokens_needed_is_rejected(self) -> None:
        policy = _policy()
        state = initial_bucket_state(policy, now=T0)
        with pytest.raises(CollectorError):
            try_acquire(state, policy, now=T0, tokens_needed=Decimal(0))

    def test_rate_limit_state_never_influences_semantic_identity(self) -> None:
        """A rate limit is purely operational -- `TokenBucketState` has no
        `to_identity_payload`/content-id method at all, and this asserts
        it structurally: exhausting or refilling the bucket has zero
        bearing on any request/response/source-manifest identity, which
        this module never even receives as an argument."""
        assert not hasattr(initial_bucket_state(_policy(), now=T0), "to_identity_payload")


class TestSecondsUntilAvailable:
    def test_zero_when_already_available(self) -> None:
        policy = _policy()
        state = initial_bucket_state(policy, now=T0)
        assert seconds_until_available(state, policy) == Decimal(0)

    def test_positive_when_drained(self) -> None:
        policy = _policy(max_tokens="1", refill_rate="0.5")
        state = initial_bucket_state(policy, now=T0)
        _, state = try_acquire(state, policy, now=T0)
        assert seconds_until_available(state, policy) == Decimal(2)  # need 1 token at 0.5/sec


class TestConcurrentAccess:
    """`TokenBucketState` is immutable/pure -- concurrent callers reading
    a SHARED, already-computed state never observe a torn/partial value
    (there is nothing to tear); this proves reading the same state from
    many threads is trivially safe, matching the module's own "no
    opinion on caller thread-safety for read-modify-write" documented
    scope."""

    def test_many_threads_reading_the_same_state_see_consistent_results(self) -> None:
        policy = _policy()
        state = initial_bucket_state(policy, now=T0)
        results: list[bool] = []
        lock = threading.Lock()

        def _worker() -> None:
            acquired, _ = try_acquire(state, policy, now=T0)
            with lock:
                results.append(acquired)

        threads = [threading.Thread(target=_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(results)  # every thread independently derives the same (correct) decision from the same immutable input
