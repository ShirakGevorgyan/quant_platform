"""Assembles the `market_data_lineage` payload
`features.manifests.ResearchDatasetManifest.market_data_lineage` stores,
and the matching `input_content_hashes` entry that gives it a single,
cheap-to-compare content-addressed fingerprint (spec Section 16's
"research manifest binds... source-version pinning").

Binds, directly or transitively, every field spec Section 16 requires:
base dataset id, macro universe id + exact macro component ids,
cross-asset universe id + exact cross-asset component ids, provider
mappings, proxy/instrument forms, revision/availability/session/
adjustment/continuation policies, alignment policy ids (the base-asset
availability policy, and -- via `CrossAssetDatasetBinding`/
`MacroDatasetBinding` themselves -- each source's own), source coverage
result, and (added by the caller in `request.py`, which has direct
access to them) feature registry fingerprint/order/label/split/
preprocessing already live on the base `ResearchDatasetManifest` fields
this is additive to -- this payload deliberately does not repeat them.
"""

from __future__ import annotations

from quant_platform.features.market_data_bridge.bindings import (
    BaseAssetDatasetBinding,
    CrossAssetDatasetBinding,
    MacroDatasetBinding,
)
from quant_platform.features.market_data_bridge.coverage import CoverageReport
from quant_platform.market_data.identity import compute_content_id

__all__ = [
    "MARKET_DATA_LINEAGE_CONTENT_KIND",
    "MARKET_DATA_LINEAGE_SCHEMA_VERSION",
    "build_market_data_lineage",
    "lineage_content_id",
]

MARKET_DATA_LINEAGE_SCHEMA_VERSION = 1
MARKET_DATA_LINEAGE_CONTENT_KIND = "market_data_bridge_lineage"


def _coverage_json(decision: CoverageReport | None) -> dict[str, object] | None:
    if decision is None:
        return None
    return {
        "safe_start": decision.safe_start.isoformat(), "safe_end": decision.safe_end.isoformat(),
        "trimmed": decision.trimmed, "trim_reason": decision.trim_reason,
        "findings": [
            {
                "source_kind": f.source_kind, "source_name": f.source_name, "required": f.required,
                "status": f.status, "coverage_fraction": f.coverage_fraction,
            }
            for f in decision.findings
        ],
    }


def build_market_data_lineage(
    *, base_binding: BaseAssetDatasetBinding, macro_bindings: dict[str, MacroDatasetBinding],
    cross_asset_bindings: dict[str, CrossAssetDatasetBinding], coverage_report: CoverageReport | None = None,
) -> dict[str, object]:
    """Pure, deterministic assembly -- no I/O, no market_data reads (the
    caller has already resolved/verified every binding via the
    `*_adapter.py` modules before calling this). Every field here is
    JSON-safe and stable-ordered (`macro_bindings`/`cross_asset_bindings`
    dicts are re-sorted by key), so two calls with the same binding set
    always produce byte-identical output regardless of caller dict
    iteration order."""
    payload: dict[str, object] = {
        "schema_version": MARKET_DATA_LINEAGE_SCHEMA_VERSION,
        "base_asset_binding": base_binding.to_json_dict(),
        "macro_bindings": {name: b.to_json_dict() for name, b in sorted(macro_bindings.items())},
        "cross_asset_bindings": {name: b.to_json_dict() for name, b in sorted(cross_asset_bindings.items())},
        "coverage_decision": _coverage_json(coverage_report),
    }
    return payload


def lineage_content_id(lineage: dict[str, object]) -> str:
    """The single, cheap-to-compare fingerprint bound into
    `ResearchDatasetManifest.input_content_hashes["market_data_lineage_content_id"]`
    -- a content hash of the exact same payload stored in full under
    `market_data_lineage`. Two research dataset builds differ in this
    fingerprint if and only if they differ in `market_data_lineage`,
    which `ResearchManifestStore.save`'s content-duplicate detection
    already keys off of `_identity_fields()` (which includes both
    fields) -- this fingerprint exists purely so a caller (`reports.py`,
    `rebuild_planner.py`) can compare two lineages with one string
    equality check instead of a deep dict diff."""
    return compute_content_id(MARKET_DATA_LINEAGE_CONTENT_KIND, lineage)
