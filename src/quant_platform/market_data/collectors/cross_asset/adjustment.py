"""Price-adjustment policy (Milestone 10, Phase 4C).

For spot/index/futures data, adjustment is typically `NOT_APPLICABLE` --
there is no corporate action to adjust for. For equities/ETFs, raw
(as-traded) OHLC and adjusted-close semantics must never be mixed within
one series: a provider's own "adjusted close" column is either verified
against known corporate-action evidence or classified honestly as
`PROVIDER_ADJUSTED_UNVERIFIED` (never silently trusted as equivalent to
a verified split/dividend adjustment). This module performs NO
corporate-action arithmetic itself -- splits and dividends are never
silently applied; a policy CHANGE always changes dataset identity (see
`AdjustmentPolicy.adjustment_policy_id`)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from quant_platform.core.exceptions import AdjustmentPolicyError
from quant_platform.market_data.identity import compute_content_id

__all__ = [
    "ADJUSTMENT_POLICY_KIND",
    "AdjustmentPolicy",
    "AdjustmentPolicyKind",
    "create_adjustment_policy",
    "require_equity_like_adjustment",
]

ADJUSTMENT_POLICY_KIND = "cross_asset_adjustment_policy"


class AdjustmentPolicyKind(Enum):
    RAW_UNADJUSTED = "raw_unadjusted"
    """As-traded OHLC, no adjustment of any kind -- the only policy this
    phase's own `providers/alpha_vantage.py` adapter produces (its one
    verified endpoint, `TIME_SERIES_DAILY`, is explicitly documented as
    raw/as-traded)."""

    SPLIT_ADJUSTED = "split_adjusted"
    """Adjusted for splits only, verified against known corporate-action
    evidence -- not produced by this phase's shipped provider adapter."""

    TOTAL_RETURN_ADJUSTED = "total_return_adjusted"
    """Adjusted for both splits and dividends (a total-return series) --
    not produced by this phase's shipped provider adapter."""

    PROVIDER_ADJUSTED_UNVERIFIED = "provider_adjusted_unverified"
    """The provider CLAIMS an adjustment was applied, but this platform
    has not independently verified it against corporate-action evidence
    -- honest, not a silent upgrade to `SPLIT_ADJUSTED`/`TOTAL_RETURN_
    ADJUSTED`."""

    NOT_APPLICABLE = "not_applicable"
    """Spot, cash index, and futures data -- there is no corporate
    action to adjust for."""


_EQUITY_LIKE_KINDS = frozenset({
    AdjustmentPolicyKind.RAW_UNADJUSTED, AdjustmentPolicyKind.SPLIT_ADJUSTED,
    AdjustmentPolicyKind.TOTAL_RETURN_ADJUSTED, AdjustmentPolicyKind.PROVIDER_ADJUSTED_UNVERIFIED,
})


@dataclass(frozen=True, slots=True)
class AdjustmentPolicy:
    adjustment_policy_id: str
    kind: AdjustmentPolicyKind
    policy_version: int
    notes: str
    """Documentation only, excluded from identity."""

    def __post_init__(self) -> None:
        if self.policy_version < 1:
            raise AdjustmentPolicyError(f"AdjustmentPolicy.policy_version must be >= 1, got {self.policy_version}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": ADJUSTMENT_POLICY_KIND, "adjustment_policy_id": self.adjustment_policy_id,
            "adjustment_kind": self.kind.value, "policy_version": self.policy_version, "notes": self.notes,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["adjustment_policy_id"]
        del payload["notes"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> AdjustmentPolicy:
        return cls(
            adjustment_policy_id=str(raw["adjustment_policy_id"]), kind=AdjustmentPolicyKind(raw["adjustment_kind"]),
            policy_version=int(str(raw["policy_version"])), notes=str(raw.get("notes", "")),
        )


def create_adjustment_policy(*, kind: AdjustmentPolicyKind, policy_version: int = 1, notes: str = "") -> AdjustmentPolicy:
    provisional = AdjustmentPolicy(adjustment_policy_id="0" * 64, kind=kind, policy_version=policy_version, notes=notes)
    adjustment_policy_id = compute_content_id(ADJUSTMENT_POLICY_KIND, provisional.to_identity_payload())
    return AdjustmentPolicy(adjustment_policy_id=adjustment_policy_id, kind=kind, policy_version=policy_version, notes=notes)


def require_equity_like_adjustment(kind: AdjustmentPolicyKind, *, instrument_form: str) -> None:
    """`registry.py`'s own guard: an equity/ETF mapping without ANY
    adjustment policy is rejected at construction (spec Section 6); this
    helper distinguishes "an equity-shaped policy was declared" from
    "NOT_APPLICABLE was declared for something that is not spot/index/
    futures", which callers pass `instrument_form` for."""
    if kind not in _EQUITY_LIKE_KINDS:
        raise AdjustmentPolicyError(f"instrument_form={instrument_form!r} requires an equity-like adjustment policy kind, got {kind.value!r}")
