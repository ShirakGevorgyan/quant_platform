"""Deterministic retry POLICY and decision logic (Milestone 10, Phase
4A) -- deliberately separated from transport EXECUTION (never calls a
transport, never sleeps, never reads the wall clock): `classify_failure`
and `plan_next_wait_seconds` are pure functions over explicit inputs, so
every retry-decision test runs in microseconds with zero timing
dependency. The actual attempt LOOP (which does call a transport, and
does -- in real, non-test use -- sleep between attempts via an
injectable `sleep_fn`) lives in `fred.py`, the one place that genuinely
needs to coordinate transport + retry + rate-limit together."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import Enum

from quant_platform.core.exceptions import CollectorError
from quant_platform.market_data.identity import compute_content_id

__all__ = [
    "RETRY_POLICY_KIND",
    "RetryAttemptRecord",
    "RetryOutcome",
    "RetryPolicy",
    "classify_failure",
    "create_retry_policy",
    "parse_retry_after",
    "plan_next_wait_seconds",
]

RETRY_POLICY_KIND = "collector_retry_policy"

_NEVER_RETRY_STATUSES = frozenset({400, 401, 403})
"""Permanent client errors -- retrying can never succeed without a
DIFFERENT request, so these are never retryable regardless of policy
configuration (a caller cannot configure `RetryPolicy` to retry them)."""

DEFAULT_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class RetryOutcome(Enum):
    RETRY = "retry"
    STOP = "stop"


class RetryFailureKind(Enum):
    """What kind of failure occurred, for `classify_failure` -- distinct
    from an HTTP status code, since a connect/read timeout or a
    malformed/corrupted response never reaches "has a status code" at
    all."""

    CONNECT_TIMEOUT = "connect_timeout"
    READ_TIMEOUT = "read_timeout"
    HTTP_STATUS = "http_status"
    MALFORMED_RESPONSE = "malformed_response"
    INTEGRITY_FAILURE = "integrity_failure"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    retry_policy_id: str
    max_attempts: int
    backoff_schedule_seconds: tuple[float, ...]
    """Explicit wait, in seconds, before attempt `i+2` -- i.e.
    `backoff_schedule_seconds[0]` is the wait before the SECOND attempt.
    Must have at least `max_attempts - 1` entries. No jitter is ever
    applied implicitly; a caller wanting jitter supplies it as part of
    THESE explicit values (deterministically, e.g. precomputed from a
    seeded source) -- see module docstring's own "no sleeping in pure
    tests" discipline, which an implicit random jitter would violate."""
    retryable_statuses: frozenset[int]
    respect_retry_after: bool

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise CollectorError(f"RetryPolicy.max_attempts must be >= 1, got {self.max_attempts}")
        if len(self.backoff_schedule_seconds) < self.max_attempts - 1:
            raise CollectorError(
                f"RetryPolicy.backoff_schedule_seconds must have at least max_attempts-1={self.max_attempts - 1} "
                f"entries, got {len(self.backoff_schedule_seconds)}"
            )
        if any(w < 0 for w in self.backoff_schedule_seconds):
            raise CollectorError(f"RetryPolicy.backoff_schedule_seconds must be all >= 0, got {self.backoff_schedule_seconds}")
        overlap = self.retryable_statuses & _NEVER_RETRY_STATUSES
        if overlap:
            raise CollectorError(f"RetryPolicy.retryable_statuses must not include permanent client-error status(es): {sorted(overlap)}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": RETRY_POLICY_KIND, "retry_policy_id": self.retry_policy_id, "max_attempts": self.max_attempts,
            "backoff_schedule_seconds": [str(w) for w in self.backoff_schedule_seconds],
            "retryable_statuses": sorted(self.retryable_statuses), "respect_retry_after": self.respect_retry_after,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["retry_policy_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RetryPolicy:
        from quant_platform.ml.persistence import as_json_list

        return cls(
            retry_policy_id=str(raw["retry_policy_id"]), max_attempts=int(str(raw["max_attempts"])),
            backoff_schedule_seconds=tuple(float(str(w)) for w in as_json_list(raw["backoff_schedule_seconds"], field_name="backoff_schedule_seconds")),
            retryable_statuses=frozenset(int(str(s)) for s in as_json_list(raw["retryable_statuses"], field_name="retryable_statuses")),
            respect_retry_after=bool(raw["respect_retry_after"]),
        )


def create_retry_policy(
    *, max_attempts: int, backoff_schedule_seconds: tuple[float, ...], retryable_statuses: frozenset[int] = DEFAULT_RETRYABLE_STATUSES,
    respect_retry_after: bool = True,
) -> RetryPolicy:
    provisional = RetryPolicy(
        retry_policy_id="0" * 64, max_attempts=max_attempts, backoff_schedule_seconds=backoff_schedule_seconds,
        retryable_statuses=retryable_statuses, respect_retry_after=respect_retry_after,
    )
    retry_policy_id = compute_content_id(RETRY_POLICY_KIND, provisional.to_identity_payload())
    return RetryPolicy(
        retry_policy_id=retry_policy_id, max_attempts=max_attempts, backoff_schedule_seconds=backoff_schedule_seconds,
        retryable_statuses=retryable_statuses, respect_retry_after=respect_retry_after,
    )


def classify_failure(*, kind: RetryFailureKind, status_code: int | None, policy: RetryPolicy) -> RetryOutcome:
    """Pure decision: given what kind of failure just happened (and, for
    `HTTP_STATUS`, which status), should this be retried under `policy`?
    A permanent client error (400/401/403) is NEVER retryable, regardless
    of `policy.retryable_statuses` (enforced structurally --
    `RetryPolicy.__post_init__` already refuses to construct a policy
    that tries to make one retryable)."""
    if kind is not RetryFailureKind.HTTP_STATUS:
        return RetryOutcome.RETRY
    assert status_code is not None
    if status_code in _NEVER_RETRY_STATUSES:
        return RetryOutcome.STOP
    return RetryOutcome.RETRY if status_code in policy.retryable_statuses else RetryOutcome.STOP


def parse_retry_after(value: str, *, now: datetime) -> float | None:
    """Strict `Retry-After` parsing: either delta-seconds (all ASCII
    digits) or an HTTP-date (RFC 7231). Returns `None` for anything else
    (malformed input fails closed to "absent" -- the caller's own
    `backoff_schedule_seconds` is the fallback, never a guessed value)."""
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.isdigit():
        return float(stripped)
    try:
        parsed = parsedate_to_datetime(stripped)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max((parsed - now).total_seconds(), 0.0)


def plan_next_wait_seconds(policy: RetryPolicy, *, attempt_number: int, retry_after_seconds: float | None) -> float:
    """`attempt_number` is the attempt that JUST FAILED (1-based); returns
    the wait, in seconds, before the NEXT attempt. Pure and deterministic
    -- no sleeping, no wall-clock read. `Retry-After` (when present and
    `policy.respect_retry_after`) takes precedence over the configured
    backoff schedule, per HTTP semantics."""
    if policy.respect_retry_after and retry_after_seconds is not None:
        return max(retry_after_seconds, 0.0)
    index = attempt_number - 1
    if index < 0 or index >= len(policy.backoff_schedule_seconds):
        raise CollectorError(f"no backoff_schedule_seconds entry configured for attempt_number={attempt_number}")
    return policy.backoff_schedule_seconds[index]


@dataclass(frozen=True, slots=True)
class RetryAttemptRecord:
    """One row of a deterministic, secret-free retry report. The caller
    assembling these is responsible for never putting a secret into
    `detail` (mirrors every other collector artifact's own rule)."""

    attempt_number: int
    outcome: str
    status_code: int | None
    failure_kind: str | None
    wait_seconds_before_next: float | None
    detail: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "attempt_number": self.attempt_number, "outcome": self.outcome, "status_code": self.status_code,
            "failure_kind": self.failure_kind, "wait_seconds_before_next": self.wait_seconds_before_next, "detail": self.detail,
        }
