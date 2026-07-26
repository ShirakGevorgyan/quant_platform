"""Explicit, auditable repair/quarantine policy for historical bars flagged
by `historical.quality.run_quality_checks`.

Every action this module can take is deliberately narrow and enumerable:

  * Exact (byte/field-identical) duplicate rows may always be removed --
    zero information is lost, since the surviving row is indistinguishable
    from the one discarded.
  * Sorting by `open_time` happens ONLY when explicitly requested
    (`allow_sort=True`), and is itself recorded as a repair step -- never
    silently applied as a side effect of some other operation.
  * Rows carrying a CRITICAL quality issue are handled per an explicit
    `SeverityPolicy`: reject the whole batch, retain everything with the
    issues merely noted, or quarantine exactly the affected rows into a
    separate, still-fully-inspectable `quarantined` frame.
  * Nothing here ever repairs a PRICE. There is no interpolation, no
    forward-fill, no synthetic candle construction anywhere in this
    module -- an unexplained gap stays a gap; a bad bar is removed or
    quarantined, never patched with a guessed value.

Every call produces a `RepairLineage`: the transformation name/version,
its parameters, every step taken (with exact row counts and the complete
set of affected row positions -- never hidden or approximated), and a
content checksum of the resulting dataset. "Never hide a repair" is
enforced structurally here: there is no code path that removes or changes a
row without appending a `RepairStep` describing it.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import QuarantineError
from quant_platform.historical.quality import QualityReport

logger = logging.getLogger(__name__)

_TRANSFORMATION_NAME = "historical.repair.apply_repair_policy"
_TRANSFORMATION_VERSION = "1.0.0"


class SeverityPolicy(Enum):
    """How to handle rows carrying a CRITICAL quality issue. Defaults to
    `STRICT` wherever a caller does not explicitly choose otherwise (see
    `historical.quality`'s Section F requirement: "defaulting to strict
    for critical issues")."""

    STRICT = "STRICT"
    """Reject the entire batch (raise `QuarantineError`) if ANY row has a
    critical issue. The safest default: a caller must opt into a weaker
    policy deliberately."""
    WARN_ONLY = "WARN_ONLY"
    """Retain every row unchanged; critical issues are recorded in the
    lineage but nothing is removed. Intended for exploratory/diagnostic
    use, never for building a canonical dataset silently."""
    QUARANTINE = "QUARANTINE"
    """Remove exactly the rows carrying a critical issue into a separate,
    still-fully-inspectable `quarantined` frame; everything else proceeds."""


class RepairAction(Enum):
    EXACT_DUPLICATE_REMOVED = "EXACT_DUPLICATE_REMOVED"
    SORTED = "SORTED"
    QUARANTINED = "QUARANTINED"
    RETAINED_WITH_WARNING = "RETAINED_WITH_WARNING"


@dataclass(frozen=True, slots=True)
class RepairStep:
    action: RepairAction
    reason: str
    rows_removed: int = 0
    rows_changed: int = 0
    affected_row_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class RepairLineage:
    """The complete audit trail for one `apply_repair_policy` call."""

    input_snapshot_id: str | None
    transformation_name: str
    transformation_version: str
    parameters: dict[str, object]
    steps: tuple[RepairStep, ...]
    rows_in: int
    rows_out: int
    rows_quarantined: int
    resulting_checksum: str
    performed_at: pd.Timestamp


@dataclass(frozen=True, slots=True)
class RepairResult:
    data: pd.DataFrame
    quarantined: pd.DataFrame
    lineage: RepairLineage


def _content_checksum(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return hashlib.sha256(b"").hexdigest()
    row_hashes = pd.util.hash_pandas_object(df, index=False)
    return hashlib.sha256(row_hashes.to_numpy().tobytes()).hexdigest()


def apply_repair_policy(
    df: pd.DataFrame,
    report: QualityReport,
    *,
    policy: SeverityPolicy = SeverityPolicy.STRICT,
    allow_sort: bool = False,
    allow_exact_duplicate_removal: bool = True,
    input_snapshot_id: str | None = None,
) -> RepairResult:
    """Apply `policy` to `df` using the findings in `report` (which MUST
    have been computed from this exact `df` -- see the positional-index
    contract note below) and return the resulting `RepairResult`.

    Positional-index contract: `report.critical_issues[*].affected_row_indices`
    are POSITIONAL indices into `df` as originally passed in (i.e. `df.iloc[i]`,
    not `df.loc[i]`). This function never reindexes `df` mid-computation for
    exactly that reason -- every step below operates on boolean masks aligned
    to `df`'s ORIGINAL row positions, and only slices `df` once, at the end,
    to build the returned frames. Reindexing partway through (e.g. immediately
    after removing exact duplicates) would silently desynchronize any
    still-pending position-based lookups from `report` -- a real ordering
    hazard identified and deliberately designed around while building this
    function, not merely a defensive precaution.
    """
    started_at = time.perf_counter()
    rows_in = len(df)
    performed_at = pd.Timestamp.now(tz="UTC")
    steps: list[RepairStep] = []
    parameters: dict[str, object] = {
        "policy": policy.value, "allow_sort": allow_sort,
        "allow_exact_duplicate_removal": allow_exact_duplicate_removal,
    }

    keep_mask = np.ones(rows_in, dtype=bool)
    quarantine_mask = np.zeros(rows_in, dtype=bool)

    if allow_exact_duplicate_removal and rows_in > 0:
        exact_dup = df.duplicated(keep="first").to_numpy()
        if exact_dup.any():
            keep_mask &= ~exact_dup
            positions = tuple(int(i) for i in np.flatnonzero(exact_dup))
            steps.append(
                RepairStep(
                    RepairAction.EXACT_DUPLICATE_REMOVED,
                    "byte/field-identical duplicate row(s) removed (zero information loss)",
                    rows_removed=len(positions), affected_row_indices=positions,
                )
            )

    critical_positions: set[int] = set()
    for issue in report.critical_issues:
        critical_positions.update(issue.affected_row_indices)
    # Rows already removed as exact duplicates above must not also be
    # counted against the policy below -- e.g. a duplicate-timestamp
    # critical issue whose flagged row was exactly the copy just removed
    # is already resolved, not a remaining conflict.
    critical_positions = {p for p in critical_positions if keep_mask[p]}

    if critical_positions:
        sorted_positions = tuple(sorted(critical_positions))
        if policy is SeverityPolicy.STRICT:
            logger.warning(
                "repair policy rejected batch: rows_in=%d critical_row_count=%d policy=%s duration_s=%.3f",
                rows_in, len(critical_positions), policy.value, time.perf_counter() - started_at,
            )
            raise QuarantineError(
                f"{len(critical_positions)} row(s) carry a critical quality issue and the "
                "STRICT policy rejects the entire batch rather than silently proceeding. "
                "Re-run with policy=QUARANTINE or WARN_ONLY to make an explicit, "
                "different choice.",
                context={"critical_row_count": len(critical_positions), "policy": policy.value},
            )
        if policy is SeverityPolicy.WARN_ONLY:
            steps.append(
                RepairStep(
                    RepairAction.RETAINED_WITH_WARNING,
                    f"{len(critical_positions)} row(s) carry a critical quality issue but "
                    "WARN_ONLY policy retains them unchanged",
                    affected_row_indices=sorted_positions,
                )
            )
        elif policy is SeverityPolicy.QUARANTINE:
            for position in sorted_positions:
                quarantine_mask[position] = True
                keep_mask[position] = False
            steps.append(
                RepairStep(
                    RepairAction.QUARANTINED,
                    f"{len(critical_positions)} row(s) carrying a critical quality issue "
                    "moved to quarantine",
                    rows_removed=len(sorted_positions), affected_row_indices=sorted_positions,
                )
            )

    working = df.loc[keep_mask].reset_index(drop=True)
    quarantined = df.loc[quarantine_mask].reset_index(drop=True)

    if allow_sort and len(working) > 1 and not working["open_time"].is_monotonic_increasing:
        working = working.sort_values("open_time", kind="stable").reset_index(drop=True)
        steps.append(
            RepairStep(
                RepairAction.SORTED,
                "rows sorted by open_time (allow_sort=True was explicitly set)",
                rows_changed=len(working),
            )
        )

    lineage = RepairLineage(
        input_snapshot_id=input_snapshot_id,
        transformation_name=_TRANSFORMATION_NAME,
        transformation_version=_TRANSFORMATION_VERSION,
        parameters=parameters,
        steps=tuple(steps),
        rows_in=rows_in,
        rows_out=len(working),
        rows_quarantined=len(quarantined),
        resulting_checksum=_content_checksum(working),
        performed_at=performed_at,
    )
    logger.info(
        "repair policy applied: policy=%s rows_in=%d rows_out=%d rows_quarantined=%d steps=%d "
        "checksum=%s duration_s=%.3f",
        policy.value, rows_in, len(working), len(quarantined), len(steps),
        lineage.resulting_checksum[:12], time.perf_counter() - started_at,
    )
    return RepairResult(data=working, quarantined=quarantined, lineage=lineage)


__all__ = [
    "RepairAction",
    "RepairLineage",
    "RepairResult",
    "RepairStep",
    "SeverityPolicy",
    "apply_repair_policy",
]
