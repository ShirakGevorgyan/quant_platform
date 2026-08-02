"""Incremental rebuild planning (Milestone 10, Phase 4D, spec Section 18)
-- a PURE function: no I/O, no `market_data`/`features` store reads of its
own. Every input is a value the caller has already resolved (an existing
`ResearchDatasetManifest`'s recorded `market_data_lineage`, the newly
proposed bindings/recipe, and OPTIONAL `SourceChangeEvidence` the caller
obtained separately by reading `market_data.manifests.DatasetManifest`/
`collectors.curated.datasets.ComponentDatasetManifest`/`collectors.
cross_asset.datasets.ComponentMarketDatasetManifest` history for the
affected source).

CONSERVATIVE BY CONSTRUCTION: a bare binding-id comparison alone (pinned
`dataset_id`/`component_manifest_id` old vs. new) can prove "this source's
content did or did not change," but never WHY it changed -- a content
hash carries no diff. Distinguishing a safe append-only extension from a
historical correction therefore requires the CALLER to supply
`SourceChangeEvidence` (old/new `first_covered_time`/`last_covered_time`/
`observation_count`) for that source; without it, this planner always
recommends the conservative, more-expensive-but-safe option
(`FULL_REBUILD_REQUIRED` for a genuinely changed required source) rather
than guessing. This mirrors spec Section 18's own worked examples: "new
future base candles may allow append-only extension" (detectable: same
`first_covered_time`, larger `last_covered_time`/count) vs. "corrected
cross-asset candle requires recomputation from the affected alignment
coordinate" (only detectable with per-record evidence this planner does
not fabricate)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from quant_platform.core.exceptions import RebuildPlanError
from quant_platform.market_data.identity import compute_content_id

__all__ = [
    "REBUILD_PLAN_KIND",
    "RebuildPlan",
    "RebuildPlanKind",
    "SourceChangeEvidence",
    "plan_rebuild",
]

REBUILD_PLAN_KIND = "market_data_bridge_rebuild_plan"


class RebuildPlanKind(Enum):
    NO_OP = "no_op"
    APPEND_ONLY_SAFE_EXTENSION = "append_only_safe_extension"
    PARTIAL_RECOMPUTATION_REQUIRED = "partial_recomputation_required"
    FULL_REBUILD_REQUIRED = "full_rebuild_required"


@dataclass(frozen=True, slots=True)
class SourceChangeEvidence:
    """Caller-supplied, independently-obtained summary of how one
    source's coverage changed between the manifest's recorded binding and
    the newly proposed one -- see module docstring. `None` fields mean
    "no data at all" (e.g. a brand-new optional source)."""

    source_kind: str
    source_name: str
    old_first_covered_time: pd.Timestamp | None
    old_last_covered_time: pd.Timestamp | None
    old_observation_count: int
    new_first_covered_time: pd.Timestamp | None
    new_last_covered_time: pd.Timestamp | None
    new_observation_count: int

    @property
    def is_append_only(self) -> bool:
        """`True` only if the covered range's START is unchanged, the END
        moved forward or stayed put, and the count did not decrease -- the
        narrowest, safest signal an append actually happened with nothing
        removed or altered in the already-covered range. Any ambiguity
        (e.g. count increased but the start also moved) is `False`.

        REQUIRES STRICT GROWTH in at least one of `last_covered_time`/
        `observation_count` -- not merely `>=`. `_consider` only reaches
        this property when the component id itself changed, so identical
        `first`/`last`/`count` summary stats alongside a changed id means
        "something changed but nothing observably extended," which is
        exactly the signature of a same-range, same-count CONTENT
        revision (e.g. a corrected value for an already-covered period)
        -- unsafe to classify as append-only. An earlier `>=`-based
        version of this check missed exactly this case; see the
        adversarial regression test covering it."""
        if self.old_first_covered_time is None or self.new_first_covered_time is None:
            return False
        if self.new_first_covered_time != self.old_first_covered_time:
            return False
        if self.new_last_covered_time is None:
            return False
        if self.old_last_covered_time is not None and self.new_last_covered_time < self.old_last_covered_time:
            return False
        if self.new_observation_count < self.old_observation_count:
            return False
        grew_temporally = self.old_last_covered_time is None or self.new_last_covered_time > self.old_last_covered_time
        grew_in_count = self.new_observation_count > self.old_observation_count
        return grew_temporally or grew_in_count

    @property
    def affected_from(self) -> pd.Timestamp | None:
        """The earliest instant recomputation must cover under
        `PARTIAL_RECOMPUTATION_REQUIRED` -- the earlier of the two covered
        starts (a correction could, in principle, touch any point from
        there forward; this planner does not narrow further than that
        without row-level evidence it was not given)."""
        candidates = [t for t in (self.old_first_covered_time, self.new_first_covered_time) if t is not None]
        return min(candidates) if candidates else None


@dataclass(frozen=True, slots=True)
class RebuildPlan:
    plan_id: str
    kind: RebuildPlanKind
    reason_codes: tuple[str, ...]
    affected_source_names: tuple[str, ...]
    required_warmup_from: pd.Timestamp | None
    expected_output_dataset_id: str | None
    """The existing manifest's own `dataset_id` if the recipe (feature
    registry/label/split/preprocessing) is unchanged -- a rebuild always
    lands on the SAME `dataset_id` (just possibly a new `version`) unless
    the recipe itself changed, in which case this is `None` (a
    genuinely different dataset, not a "rebuild" of the same one)."""


def _plan_id(kind: RebuildPlanKind, reason_codes: tuple[str, ...], affected: tuple[str, ...]) -> str:
    return compute_content_id(
        REBUILD_PLAN_KIND, {"kind": kind.value, "reason_codes": sorted(reason_codes), "affected_source_names": sorted(affected)}
    )


def plan_rebuild(
    *, existing_lineage: dict[str, object] | None, existing_dataset_id: str | None, recipe_unchanged: bool,
    new_base_pinned_dataset_id: str, old_base_pinned_dataset_id: str | None,
    new_macro_component_ids: dict[str, str], old_macro_component_ids: dict[str, str],
    new_cross_asset_component_ids: dict[str, str], old_cross_asset_component_ids: dict[str, str],
    evidence_by_source_name: dict[str, SourceChangeEvidence] | None = None,
) -> RebuildPlan:
    """Pure. `old_*`/`existing_*` arguments should come from the prior
    `ResearchDatasetManifest.market_data_lineage` (via
    `lineage.build_market_data_lineage`'s own binding JSON shape) -- never
    a fresh `market_data` read (that would make this "planning" a second,
    redundant verification pass; verification is `verification.py`'s own,
    separate job)."""
    evidence_by_source_name = evidence_by_source_name or {}
    reason_codes: list[str] = []
    affected: list[str] = []

    if existing_lineage is None or existing_dataset_id is None:
        return RebuildPlan(
            plan_id=_plan_id(RebuildPlanKind.FULL_REBUILD_REQUIRED, ("no_prior_market_data_lineage",), ()),
            kind=RebuildPlanKind.FULL_REBUILD_REQUIRED, reason_codes=("no_prior_market_data_lineage",),
            affected_source_names=(), required_warmup_from=None, expected_output_dataset_id=None,
        )

    if not recipe_unchanged:
        return RebuildPlan(
            plan_id=_plan_id(RebuildPlanKind.FULL_REBUILD_REQUIRED, ("recipe_changed",), ()),
            kind=RebuildPlanKind.FULL_REBUILD_REQUIRED, reason_codes=("recipe_changed",),
            affected_source_names=(), required_warmup_from=None, expected_output_dataset_id=None,
        )

    worst = RebuildPlanKind.NO_OP
    earliest_affected: pd.Timestamp | None = None

    def _consider(source_name: str, old_id: str | None, new_id: str) -> None:
        nonlocal worst, earliest_affected
        if old_id == new_id:
            return
        affected.append(source_name)
        evidence = evidence_by_source_name.get(source_name)
        if evidence is not None and evidence.is_append_only:
            reason_codes.append(f"{source_name}:append_only_extension")
            if worst is RebuildPlanKind.NO_OP:
                worst = RebuildPlanKind.APPEND_ONLY_SAFE_EXTENSION
        elif evidence is not None:
            reason_codes.append(f"{source_name}:partial_recomputation")
            if worst in (RebuildPlanKind.NO_OP, RebuildPlanKind.APPEND_ONLY_SAFE_EXTENSION):
                worst = RebuildPlanKind.PARTIAL_RECOMPUTATION_REQUIRED
            if evidence.affected_from is not None:
                earliest_affected = evidence.affected_from if earliest_affected is None else min(earliest_affected, evidence.affected_from)
        else:
            reason_codes.append(f"{source_name}:changed_without_evidence")
            worst = RebuildPlanKind.FULL_REBUILD_REQUIRED

    _consider("base", old_base_pinned_dataset_id, new_base_pinned_dataset_id)
    for name in sorted(set(new_macro_component_ids) | set(old_macro_component_ids)):
        _consider(name, old_macro_component_ids.get(name), new_macro_component_ids.get(name, ""))
    for name in sorted(set(new_cross_asset_component_ids) | set(old_cross_asset_component_ids)):
        _consider(name, old_cross_asset_component_ids.get(name), new_cross_asset_component_ids.get(name, ""))

    if not reason_codes:
        reason_codes.append("no_bound_source_changed")

    plan = RebuildPlan(
        plan_id=_plan_id(worst, tuple(reason_codes), tuple(affected)), kind=worst, reason_codes=tuple(reason_codes),
        affected_source_names=tuple(sorted(set(affected))),
        required_warmup_from=(earliest_affected if worst is RebuildPlanKind.PARTIAL_RECOMPUTATION_REQUIRED else None),
        expected_output_dataset_id=existing_dataset_id,
    )
    if plan.kind is RebuildPlanKind.PARTIAL_RECOMPUTATION_REQUIRED and plan.required_warmup_from is None:
        raise RebuildPlanError(
            "Internal inconsistency: PARTIAL_RECOMPUTATION_REQUIRED plan has no required_warmup_from -- a "
            "partial-recomputation plan must always name where recomputation begins.",
            context={"plan_id": plan.plan_id},
        )
    return plan
