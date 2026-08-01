"""Revision/vintage policy (Milestone 10, Phase 4B) -- how a curated
series' possibly-revised FRED observations map onto point-in-time-safe
(or explicitly NOT point-in-time-safe) request semantics.

FRED's real ALFRED API expresses "which vintage(s) of a value" via
`output_type` (1=by real-time period [default], 2=by vintage date/ALL
observations, 3=by vintage date/new+revised only, 4=initial release
only) and `realtime_start`/`realtime_end` (a window of "as officially
known between these two dates"), NOT via a single dedicated parameter
-- `resolve_fred_request_overrides` is the one place that FRED-specific
mapping lives, so every other module reasons in terms of the four
named policy KINDS below, never raw FRED parameter names."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from quant_platform.core.exceptions import RevisionPolicyError
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_tz_aware,
    serialize_timestamp,
)

__all__ = [
    "REVISION_POLICY_KIND",
    "RevisionPolicy",
    "RevisionPolicyKind",
    "create_revision_policy",
    "resolve_fred_request_overrides",
]

REVISION_POLICY_KIND = "curated_revision_policy"


class RevisionPolicyKind(Enum):
    LATEST_AVAILABLE = "latest_available"
    """Whatever FRED currently reports -- useful for descriptive
    research but NOT automatically point-in-time-safe (see module
    docstring's own point-in-time discussion in `availability.py`)."""

    FIRST_RELEASE_ONLY = "first_release_only"
    """FRED's `output_type=4` -- the INITIAL release of each
    observation, never a later revision."""

    AS_OF_REALTIME_DATE = "as_of_realtime_date"
    """Reconstructs "what was officially known as of this EXPLICIT
    historical date" via `realtime_start=realtime_end=as_of_realtime_date`."""

    VINTAGE_SERIES = "vintage_series"
    """Retains EVERY distinct revision/vintage as its own source fact
    (`output_type=2`) -- never collapses them."""


@dataclass(frozen=True, slots=True)
class RevisionPolicy:
    revision_policy_id: str
    kind: RevisionPolicyKind
    as_of_realtime_date: datetime | None
    revision_policy_version: int

    def __post_init__(self) -> None:
        if self.revision_policy_version < 1:
            raise RevisionPolicyError(f"RevisionPolicy.revision_policy_version must be >= 1, got {self.revision_policy_version}")
        if self.kind is RevisionPolicyKind.AS_OF_REALTIME_DATE:
            if self.as_of_realtime_date is None:
                raise RevisionPolicyError("RevisionPolicy.as_of_realtime_date is required when kind is AS_OF_REALTIME_DATE")
            require_tz_aware(self.as_of_realtime_date, field_name="RevisionPolicy.as_of_realtime_date")
        elif self.as_of_realtime_date is not None:
            raise RevisionPolicyError(f"RevisionPolicy.as_of_realtime_date must be None when kind is {self.kind.value!r} -- invalid combination")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": REVISION_POLICY_KIND, "revision_policy_id": self.revision_policy_id, "policy_kind": self.kind.value,
            "as_of_realtime_date": (None if self.as_of_realtime_date is None else serialize_timestamp(self.as_of_realtime_date, field_name="as_of_realtime_date")),
            "revision_policy_version": self.revision_policy_version,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["revision_policy_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RevisionPolicy:
        raw_as_of = raw.get("as_of_realtime_date")
        return cls(
            revision_policy_id=str(raw["revision_policy_id"]), kind=RevisionPolicyKind(raw["policy_kind"]),
            as_of_realtime_date=(None if raw_as_of is None else deserialize_timestamp(raw_as_of, field_name="as_of_realtime_date")),
            revision_policy_version=int(str(raw["revision_policy_version"])),
        )


def create_revision_policy(
    *, kind: RevisionPolicyKind, as_of_realtime_date: datetime | None = None, revision_policy_version: int = 1,
) -> RevisionPolicy:
    provisional = RevisionPolicy(revision_policy_id="0" * 64, kind=kind, as_of_realtime_date=as_of_realtime_date, revision_policy_version=revision_policy_version)
    revision_policy_id = compute_content_id(REVISION_POLICY_KIND, provisional.to_identity_payload())
    return RevisionPolicy(revision_policy_id=revision_policy_id, kind=kind, as_of_realtime_date=as_of_realtime_date, revision_policy_version=revision_policy_version)


def resolve_fred_request_overrides(policy: RevisionPolicy) -> dict[str, object]:
    """Pure: the exact `realtime_start`/`realtime_end`/`output_type`
    FRED request-parameter overrides this policy implies -- passed
    straight through to `fred.build_fred_request_manifest`'s own
    matching kwargs. Never reads the wall clock; `AS_OF_REALTIME_DATE`
    uses only the policy's own explicit `as_of_realtime_date`."""
    if policy.kind is RevisionPolicyKind.LATEST_AVAILABLE:
        return {"realtime_start": None, "realtime_end": None, "output_type": None}
    if policy.kind is RevisionPolicyKind.FIRST_RELEASE_ONLY:
        return {"realtime_start": None, "realtime_end": None, "output_type": 4}
    if policy.kind is RevisionPolicyKind.AS_OF_REALTIME_DATE:
        assert policy.as_of_realtime_date is not None
        return {"realtime_start": policy.as_of_realtime_date, "realtime_end": policy.as_of_realtime_date, "output_type": None}
    if policy.kind is RevisionPolicyKind.VINTAGE_SERIES:
        return {"realtime_start": None, "realtime_end": None, "output_type": 2}
    raise RevisionPolicyError(f"unsupported RevisionPolicyKind: {policy.kind!r}")
