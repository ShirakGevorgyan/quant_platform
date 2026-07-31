"""Deterministic collector rate limiting (Milestone 10, Phase 4A) -- a
pure token-bucket model. Every function here is a pure function of an
explicit, caller-supplied `now: datetime` and an immutable
`TokenBucketState` (returning a NEW state, never mutating in place), so
unit tests never sleep and never depend on wall-clock timing. There is
no global mutable singleton: a caller (a single collector instance, or
several running concurrently) owns its own `TokenBucketState` value and
is responsible for its own thread-safety if that value is genuinely
shared across threads (e.g. guarding read-modify-write with its own
lock) -- this module intentionally has no opinion on that, exactly like
`backfill.py`'s pure planner has no opinion on I/O. Rate-limit state
never participates in any semantic dataset/request/response identity --
it is purely an operational throttle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from quant_platform.core.exceptions import CollectorError
from quant_platform.market_data.identity import compute_content_id, require_tz_aware

__all__ = [
    "RATE_LIMIT_POLICY_KIND",
    "RateLimitPolicy",
    "TokenBucketState",
    "create_rate_limit_policy",
    "initial_bucket_state",
    "seconds_until_available",
    "try_acquire",
]

RATE_LIMIT_POLICY_KIND = "collector_rate_limit_policy"


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    rate_limit_policy_id: str
    max_tokens: Decimal
    refill_rate_per_second: Decimal

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise CollectorError(f"RateLimitPolicy.max_tokens must be > 0, got {self.max_tokens}")
        if self.refill_rate_per_second <= 0:
            raise CollectorError(f"RateLimitPolicy.refill_rate_per_second must be > 0, got {self.refill_rate_per_second}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": RATE_LIMIT_POLICY_KIND, "rate_limit_policy_id": self.rate_limit_policy_id,
            "max_tokens": str(self.max_tokens), "refill_rate_per_second": str(self.refill_rate_per_second),
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["rate_limit_policy_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RateLimitPolicy:
        return cls(
            rate_limit_policy_id=str(raw["rate_limit_policy_id"]), max_tokens=Decimal(str(raw["max_tokens"])),
            refill_rate_per_second=Decimal(str(raw["refill_rate_per_second"])),
        )


def create_rate_limit_policy(*, max_tokens: Decimal, refill_rate_per_second: Decimal) -> RateLimitPolicy:
    provisional = RateLimitPolicy(rate_limit_policy_id="0" * 64, max_tokens=max_tokens, refill_rate_per_second=refill_rate_per_second)
    rate_limit_policy_id = compute_content_id(RATE_LIMIT_POLICY_KIND, provisional.to_identity_payload())
    return RateLimitPolicy(rate_limit_policy_id=rate_limit_policy_id, max_tokens=max_tokens, refill_rate_per_second=refill_rate_per_second)


@dataclass(frozen=True, slots=True)
class TokenBucketState:
    tokens: Decimal
    last_refill_time: datetime

    def __post_init__(self) -> None:
        require_tz_aware(self.last_refill_time, field_name="TokenBucketState.last_refill_time")
        if self.tokens < 0:
            raise CollectorError(f"TokenBucketState.tokens must be >= 0, got {self.tokens}")


def initial_bucket_state(policy: RateLimitPolicy, *, now: datetime) -> TokenBucketState:
    return TokenBucketState(tokens=policy.max_tokens, last_refill_time=now)


def _refill(state: TokenBucketState, policy: RateLimitPolicy, *, now: datetime) -> TokenBucketState:
    require_tz_aware(now, field_name="now")
    if now < state.last_refill_time:
        raise CollectorError(f"now ({now}) must be >= state.last_refill_time ({state.last_refill_time}) -- time must move forward")
    elapsed_seconds = Decimal(str((now - state.last_refill_time).total_seconds()))
    refilled_tokens = min(policy.max_tokens, state.tokens + elapsed_seconds * policy.refill_rate_per_second)
    return TokenBucketState(tokens=refilled_tokens, last_refill_time=now)


def try_acquire(state: TokenBucketState, policy: RateLimitPolicy, *, now: datetime, tokens_needed: Decimal = Decimal(1)) -> tuple[bool, TokenBucketState]:
    """Pure: refills `state` up to `now`, then deducts `tokens_needed` if
    enough are available. Returns `(True, new_state_with_deduction)` on
    success, `(False, refilled_state_no_deduction)` on failure -- the
    caller can inspect the returned (refilled but undeducted) state to
    decide whether to wait (`seconds_until_available`) or fail closed
    (`RateLimitUnavailableError`)."""
    if tokens_needed <= 0:
        raise CollectorError(f"tokens_needed must be > 0, got {tokens_needed}")
    refreshed = _refill(state, policy, now=now)
    if refreshed.tokens >= tokens_needed:
        return True, TokenBucketState(tokens=refreshed.tokens - tokens_needed, last_refill_time=refreshed.last_refill_time)
    return False, refreshed


def seconds_until_available(state: TokenBucketState, policy: RateLimitPolicy, *, tokens_needed: Decimal = Decimal(1)) -> Decimal:
    """Pure: seconds from `state.last_refill_time` until `tokens_needed`
    would be available, assuming no other acquisition happens meanwhile.
    Zero if already available."""
    if state.tokens >= tokens_needed:
        return Decimal(0)
    deficit = tokens_needed - state.tokens
    return deficit / policy.refill_rate_per_second
