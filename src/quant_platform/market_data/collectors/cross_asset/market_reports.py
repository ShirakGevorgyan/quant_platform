"""Deterministic reporting for the curated cross-asset universe
(Milestone 10, Phase 4C, spec Section 28) -- mirrors `curated.reports`'s
own convention exactly: every function wraps an object this package
already produces into a stable, deterministic dict, never re-deriving
new facts of its own. `generate_quarantine_summary_report`/
`generate_provenance_summary_report` are RE-EXPORTED UNCHANGED from
`collectors.reports` (already `DatasetKey`-generic).

NO REPORT HERE MAY EVER CONTAIN A PROVIDER API KEY OR A COMPLETE REQUEST
URL WITH SECRETS: every dict is assembled purely from objects that are
themselves structurally secret-free (`CollectorRequestManifest`/
`CollectorResponseManifest`, `CuratedMarketDriverRegistry`,
`ProviderSymbolMapping`, `MarketDriverBar`, ...); no function here ever
accepts a raw credential as an argument."""

from __future__ import annotations

from quant_platform.market_data.collectors.cross_asset.datasets import (
    CombinedCrossAssetManifest,
    ComponentMarketDatasetManifest,
)
from quant_platform.market_data.collectors.cross_asset.gap_policy import GapAnalysisReport
from quant_platform.market_data.collectors.cross_asset.market_backfill import MarketBackfillSpec
from quant_platform.market_data.collectors.cross_asset.market_orchestration import (
    CrossAssetIngestionReport,
    MappingOutcome,
)
from quant_platform.market_data.collectors.cross_asset.protocols import MarketCollectorCapabilities
from quant_platform.market_data.collectors.cross_asset.registry import CuratedMarketDriverRegistry
from quant_platform.market_data.collectors.cross_asset.symbol_mapping import ProviderSymbolMapping
from quant_platform.market_data.collectors.cross_asset.update_plan import CrossAssetUpdatePlan
from quant_platform.market_data.collectors.reports import (
    generate_provenance_summary_report,
    generate_quarantine_summary_report,
)
from quant_platform.ml.models import ValidationReport

__all__ = [
    "generate_capability_assessment_report",
    "generate_combined_cross_asset_report",
    "generate_component_market_dataset_report",
    "generate_cross_asset_backfill_plan_report",
    "generate_cross_asset_ingestion_report",
    "generate_cross_asset_reconciliation_report",
    "generate_cross_asset_registry_report",
    "generate_cross_asset_update_plan_report",
    "generate_cross_asset_verification_report",
    "generate_gap_analysis_report",
    "generate_mapping_collection_report",
    "generate_provenance_summary_report",
    "generate_quarantine_summary_report",
    "generate_symbol_mapping_report",
]


def generate_cross_asset_registry_report(registry: CuratedMarketDriverRegistry) -> dict[str, object]:
    return {
        "registry_id": registry.registry_id, "registry_version": registry.registry_version, "driver_count": len(registry.specs),
        "enabled_driver_ids": list(registry.enabled_driver_ids()), "required_driver_ids": list(registry.required_driver_ids()),
        "drivers_by_tier": {
            tier: sorted(s.canonical_driver_id for s in registry.specs if s.tier.value == tier)
            for tier in sorted({s.tier.value for s in registry.specs})
        },
    }


def generate_symbol_mapping_report(mapping: ProviderSymbolMapping) -> dict[str, object]:
    return mapping.to_json_dict()


def generate_capability_assessment_report(provider: str, capabilities: MarketCollectorCapabilities) -> dict[str, object]:
    return {
        "provider": provider, "candles_supported": capabilities.candles_supported, "quotes_supported": capabilities.quotes_supported,
        "trades_supported": capabilities.trades_supported, "adjusted_data_supported": capabilities.adjusted_data_supported,
        "unadjusted_data_supported": capabilities.unadjusted_data_supported, "corporate_actions_supported": capabilities.corporate_actions_supported,
        "futures_contracts_supported": capabilities.futures_contracts_supported, "continuous_futures_supported": capabilities.continuous_futures_supported,
        "pagination_supported": capabilities.pagination_supported, "anonymous_access_supported": capabilities.anonymous_access_supported,
        "runtime_credential_required": capabilities.runtime_credential_required, "max_interval_days_per_request": capabilities.max_interval_days_per_request,
        "max_rows_per_page": capabilities.max_rows_per_page, "supported_granularities": list(capabilities.supported_granularities),
        "supported_instrument_forms": [f.value for f in capabilities.supported_instrument_forms],
    }


def generate_cross_asset_backfill_plan_report(spec: MarketBackfillSpec) -> dict[str, object]:
    return spec.to_json_dict()


def generate_mapping_collection_report(outcome: MappingOutcome) -> dict[str, object]:
    return {
        "mapping_id": outcome.mapping_id, "canonical_driver_id": outcome.canonical_driver_id, "succeeded": outcome.succeeded,
        "failure_reason": outcome.failure_reason, "parsed_row_count": outcome.parsed_row_count, "valid_row_count": outcome.valid_row_count,
        "quarantined_row_count": outcome.quarantined_row_count, "committed_bar_count": outcome.committed_bar_count,
        "component_manifest_id": outcome.component_manifest_id,
    }


def generate_cross_asset_ingestion_report(report: CrossAssetIngestionReport) -> dict[str, object]:
    return {
        "operation_id": report.operation_id, "target_dataset_namespace": report.target_dataset_namespace, "backfill_plan_id": report.backfill_plan_id,
        "stage": report.stage.value, "completeness_status": report.completeness_status,
        "mapping_outcomes": [generate_mapping_collection_report(o) for o in report.mapping_outcomes],
        "combined_manifest_id": report.combined_manifest_id, "is_dry_run": report.is_dry_run,
    }


def generate_component_market_dataset_report(manifest: ComponentMarketDatasetManifest) -> dict[str, object]:
    return manifest.to_json_dict()


def generate_combined_cross_asset_report(manifest: CombinedCrossAssetManifest) -> dict[str, object]:
    return manifest.to_json_dict()


def generate_gap_analysis_report(report: GapAnalysisReport) -> dict[str, object]:
    return report.to_json_dict()


def generate_cross_asset_update_plan_report(plan: CrossAssetUpdatePlan) -> dict[str, object]:
    result = plan.to_json_dict()
    result["mappings_requiring_update"] = list(plan.mappings_requiring_update())
    result["is_exact_no_op"] = plan.is_exact_no_op()
    return result


def _summarize_validation_report(report: ValidationReport, *, scope: str) -> dict[str, object]:
    return {
        "scope": scope, "generated_at": report.generated_at, "critical_count": len(report.criticals), "warning_count": len(report.warnings),
        "total_issue_count": len(report.issues), "issues": [i.to_json_dict() for i in report.issues],
    }


def generate_cross_asset_reconciliation_report(*, report: ValidationReport, target_dataset_namespace: str) -> dict[str, object]:
    return _summarize_validation_report(report, scope=target_dataset_namespace)


def generate_cross_asset_verification_report(*, report: ValidationReport, target_dataset_namespace: str) -> dict[str, object]:
    return _summarize_validation_report(report, scope=target_dataset_namespace)
