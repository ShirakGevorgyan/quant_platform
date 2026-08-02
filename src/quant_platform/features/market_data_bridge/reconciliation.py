"""Reconciliation (Milestone 10, Phase 4D, spec Section 19): the
NON-RAISING counterpart to `*_adapter.py`'s fail-closed `verify_*`
functions -- runs the same underlying checks (plus a handful of
alignment-level PIT re-checks the adapters have no reason to run
themselves, since they already refuse leaky input by construction) and
reports every finding as a structured `ReconciliationIssue` instead of
raising on the first one. Useful for an audit/diagnostic pass over many
bindings, or a fixture-acceptance workflow that wants to assert ZERO
issues rather than merely "it didn't raise."

`ReconciliationIssueCode` enumerates every finding spec Section 19 names.
Not every code is independently derivable from data this bridge already
holds without a second `market_data` read the caller must supply
explicitly (documented per-function below) -- this module never
FABRICATES a finding it cannot actually support with evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from quant_platform.core.exceptions import BridgeReconciliationError, SourceVerificationError
from quant_platform.core.types import Timeframe
from quant_platform.features.alignment import align_higher_timeframe, as_of_join_external
from quant_platform.features.manifests import ResearchDatasetManifest
from quant_platform.features.market_data_bridge.base_asset_adapter import verify_base_asset_binding
from quant_platform.features.market_data_bridge.bindings import (
    BaseAssetDatasetBinding,
    CrossAssetDatasetBinding,
    MacroDatasetBinding,
)
from quant_platform.features.market_data_bridge.coverage import CoverageReport
from quant_platform.features.market_data_bridge.cross_asset_adapter import verify_cross_asset_binding
from quant_platform.features.market_data_bridge.lineage import build_market_data_lineage
from quant_platform.features.market_data_bridge.macro_adapter import verify_macro_binding
from quant_platform.market_data.collectors.cross_asset.datasets import ComponentMarketDatasetManifestStore
from quant_platform.market_data.collectors.cross_asset.market_record import MarketDriverBarStore
from quant_platform.market_data.collectors.curated.datasets import ComponentDatasetManifestStore
from quant_platform.market_data.collectors.curated.macro_observation import CuratedObservationStore
from quant_platform.market_data.repository import MarketDataRepository

__all__ = [
    "ReconciliationIssue",
    "ReconciliationIssueCode",
    "ReconciliationReport",
    "reconcile_binding_source",
    "reconcile_manifest_lineage",
    "reconcile_no_pre_availability_macro_leakage",
    "reconcile_no_pre_close_cross_asset_leakage",
    "reconcile_output_range",
]


class ReconciliationIssueCode(Enum):
    MISSING_COMPONENT = "missing_component"
    WRONG_COMPONENT_VERSION = "wrong_component_version"
    SEMANTIC_DIGEST_MISMATCH = "semantic_digest_mismatch"
    DUPLICATE_ALIGNED_COORDINATE = "duplicate_aligned_coordinate"
    MACRO_VALUE_VISIBLE_BEFORE_AVAILABILITY = "macro_value_visible_before_availability"
    CROSS_ASSET_CANDLE_VISIBLE_BEFORE_CLOSE = "cross_asset_candle_visible_before_close"
    MANIFEST_LINEAGE_MISMATCH = "manifest_lineage_mismatch"
    OUTPUT_ROW_OUTSIDE_SAFE_RANGE = "output_row_outside_safe_range"


@dataclass(frozen=True, slots=True)
class ReconciliationIssue:
    code: ReconciliationIssueCode
    message: str
    context: dict[str, object]


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    issues: tuple[ReconciliationIssue, ...]

    @property
    def is_clean(self) -> bool:
        return not self.issues


def _issue_from_verification_error(exc: SourceVerificationError) -> ReconciliationIssue:
    message = str(exc)
    code = (
        ReconciliationIssueCode.SEMANTIC_DIGEST_MISMATCH if "semantic digest" in message
        else ReconciliationIssueCode.WRONG_COMPONENT_VERSION if "does not match the CURRENT" in message
        else ReconciliationIssueCode.MISSING_COMPONENT if "No " in message and "exists" in message
        else ReconciliationIssueCode.DUPLICATE_ALIGNED_COORDINATE if "Conflicting" in message
        else ReconciliationIssueCode.MISSING_COMPONENT
    )
    return ReconciliationIssue(code=code, message=message, context=dict(exc.context))


def reconcile_binding_source(
    *, base: tuple[MarketDataRepository, BaseAssetDatasetBinding] | None = None,
    macro: tuple[CuratedObservationStore, ComponentDatasetManifestStore, MacroDatasetBinding] | None = None,
    cross_asset: tuple[MarketDriverBarStore, ComponentMarketDatasetManifestStore, CrossAssetDatasetBinding] | None = None,
) -> ReconciliationReport:
    """Runs exactly one of the three `verify_*_binding` functions
    (whichever tuple is supplied) and converts a raised
    `SourceVerificationError` into a reported issue instead of
    propagating it. Exactly one of `base`/`macro`/`cross_asset` must be
    supplied."""
    supplied = [x for x in (base, macro, cross_asset) if x is not None]
    if len(supplied) != 1:
        raise BridgeReconciliationError("reconcile_binding_source requires exactly one of base/macro/cross_asset")
    try:
        if base is not None:
            verify_base_asset_binding(*base)
        elif macro is not None:
            verify_macro_binding(*macro)
        elif cross_asset is not None:
            verify_cross_asset_binding(*cross_asset)
    except SourceVerificationError as exc:
        return ReconciliationReport(issues=(_issue_from_verification_error(exc),))
    return ReconciliationReport(issues=())


def reconcile_no_pre_availability_macro_leakage(
    base_availability_times: pd.Series, macro_df: pd.DataFrame, *, source_name: str,
) -> ReconciliationReport:
    """Independently re-derives the as-of join's own output and asserts,
    row by row, that the selected `release_time` never exceeds that row's
    own availability instant -- a standalone assertion of the exact
    invariant `features.alignment.as_of_join_external`'s backward
    `merge_asof` is supposed to guarantee by construction, re-checked here
    rather than merely trusted."""
    joined = as_of_join_external(base_availability_times, macro_df, value_column="value", output_name="level")
    base_times = pd.Series(base_availability_times).reset_index(drop=True)
    violations = joined["level_release_time"].notna() & (joined["level_release_time"] > base_times)
    if violations.any():
        bad_index = int(violations.idxmax())
        return ReconciliationReport(
            issues=(
                ReconciliationIssue(
                    code=ReconciliationIssueCode.MACRO_VALUE_VISIBLE_BEFORE_AVAILABILITY,
                    message=(
                        f"macro source {source_name!r}: row {bad_index} shows release_time "
                        f"{joined['level_release_time'].iloc[bad_index]} after its own availability instant "
                        f"{base_times.iloc[bad_index]}"
                    ),
                    context={"source_name": source_name, "row_index": bad_index},
                ),
            )
        )
    return ReconciliationReport(issues=())


def reconcile_no_pre_close_cross_asset_leakage(
    base_availability_times: pd.Series, cross_asset_df: pd.DataFrame, *, source_name: str, timeframe: Timeframe,
) -> ReconciliationReport:
    """The cross-asset analogue of `reconcile_no_pre_availability_macro_leakage`
    -- re-derives `align_higher_timeframe`'s own output and asserts no
    revealed bar's close time exceeds the base row's availability instant."""
    aligned = align_higher_timeframe(base_availability_times, cross_asset_df, timeframe)
    prefix = f"htf_{timeframe.value}_"
    base_times = pd.Series(base_availability_times).reset_index(drop=True)
    close_times = aligned[f"{prefix}close_time"]
    violations = close_times.notna() & (close_times > base_times)
    if violations.any():
        bad_index = int(violations.idxmax())
        return ReconciliationReport(
            issues=(
                ReconciliationIssue(
                    code=ReconciliationIssueCode.CROSS_ASSET_CANDLE_VISIBLE_BEFORE_CLOSE,
                    message=(
                        f"cross-asset source {source_name!r}: row {bad_index} shows a revealed bar close_time "
                        f"{close_times.iloc[bad_index]} after its own availability instant {base_times.iloc[bad_index]}"
                    ),
                    context={"source_name": source_name, "row_index": bad_index},
                ),
            )
        )
    return ReconciliationReport(issues=())


def reconcile_manifest_lineage(
    manifest: ResearchDatasetManifest, *, base_binding: BaseAssetDatasetBinding, macro_bindings: dict[str, MacroDatasetBinding],
    cross_asset_bindings: dict[str, CrossAssetDatasetBinding], coverage_report: CoverageReport,
) -> ReconciliationReport:
    """Recomputes `lineage.build_market_data_lineage` from the SAME
    bindings a caller claims were used and requires an exact match against
    `manifest.market_data_lineage` -- detects a manifest whose recorded
    lineage was tampered with (or simply does not match what the caller
    now presents as the source of truth)."""
    if manifest.market_data_lineage is None:
        return ReconciliationReport(
            issues=(
                ReconciliationIssue(
                    code=ReconciliationIssueCode.MANIFEST_LINEAGE_MISMATCH,
                    message="manifest.market_data_lineage is None but a market_data-backed reconciliation was requested",
                    context={"dataset_id": manifest.dataset_id, "version": manifest.version},
                ),
            )
        )
    expected = build_market_data_lineage(
        base_binding=base_binding, macro_bindings=macro_bindings, cross_asset_bindings=cross_asset_bindings, coverage_report=coverage_report
    )
    if expected != manifest.market_data_lineage:
        return ReconciliationReport(
            issues=(
                ReconciliationIssue(
                    code=ReconciliationIssueCode.MANIFEST_LINEAGE_MISMATCH,
                    message="recomputed market_data_lineage does not match the manifest's own recorded lineage",
                    context={"dataset_id": manifest.dataset_id, "version": manifest.version},
                ),
            )
        )
    return ReconciliationReport(issues=())


def reconcile_output_range(manifest: ResearchDatasetManifest, coverage_report: CoverageReport) -> ReconciliationReport:
    if manifest.utc_start < coverage_report.safe_start or manifest.utc_end > coverage_report.safe_end:
        return ReconciliationReport(
            issues=(
                ReconciliationIssue(
                    code=ReconciliationIssueCode.OUTPUT_ROW_OUTSIDE_SAFE_RANGE,
                    message=(
                        f"manifest range [{manifest.utc_start}, {manifest.utc_end}] extends beyond the coverage-"
                        f"evaluated safe range [{coverage_report.safe_start}, {coverage_report.safe_end}]"
                    ),
                    context={
                        "manifest_start": str(manifest.utc_start), "manifest_end": str(manifest.utc_end),
                        "safe_start": str(coverage_report.safe_start), "safe_end": str(coverage_report.safe_end),
                    },
                ),
            )
        )
    return ReconciliationReport(issues=())
