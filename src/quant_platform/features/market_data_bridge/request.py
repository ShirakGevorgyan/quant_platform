"""The bridge's single orchestration entry point: `MarketDataResearchDatasetRequest`
pins every market_data binding + alignment/coverage policy a research
dataset build needs, and `build_research_dataset_from_market_data` wires
them into the REAL, UNMODIFIED `features.dataset_builder.
ResearchDatasetBuilder` -- never a second builder (spec's explicit
prohibition).

FIXED ORCHESTRATION ORDER (mirrors `dataset_builder.py`'s own module
docstring's fixed-order leakage guarantee, one level up):

  1. Resolve + verify every macro/cross-asset binding into aligned-ready
     `pandas.DataFrame`s (`macro_adapter`/`cross_asset_adapter`).
  2. Resolve + verify the base-asset binding directly ONCE here (for
     coverage evaluation only -- `ResearchDatasetBuilder.build()` below
     resolves it a SECOND time internally via `MarketDataBaseAssetLoader`;
     a deliberate, documented redundant read rather than modifying
     `dataset_builder.py`'s own fixed build order -- see module docstring
     of `base_asset_adapter.py` for why this stays cheap and always
     re-verifies rather than caching a possibly-stale read).
  3. Evaluate source coverage (`coverage.evaluate_source_coverage`) over
     the ORIGINALLY requested range; under `TRIM_TO_COMMON_SAFE_RANGE`,
     narrow the request to the reported safe range.
  4. Assemble `market_data_lineage` (`lineage.build_market_data_lineage`)
     and its content fingerprint, and thread BOTH into the (possibly
     range-narrowed) `ResearchDatasetBuildRequest` -- `market_data_lineage`
     directly, and the fingerprint via `aux_input_content_hashes["market_data_lineage_content_id"]`.
  5. Call the real `ResearchDatasetBuilder.build()` -- unchanged,
     untouched, doing everything it already does (feature computation,
     labeling, splitting, preprocessing, validation, manifest writing).

Coverage evaluation and lineage assembly never see label values (they run
entirely on source `open_time`/`release_time` coverage, before
`ResearchDatasetBuilder.build()` ever computes a label) -- preserving
spec Section 15's "source trimming/coverage decisions must not use label
values" requirement structurally, not by convention.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import pandas as pd

from quant_platform.core.types import Timeframe
from quant_platform.features.dataset_builder import (
    ResearchDatasetBuilder,
    ResearchDatasetBuildRequest,
)
from quant_platform.features.manifests import (
    ResearchDatasetManifest,
    ResearchDatasetStore,
    ResearchManifestStore,
)
from quant_platform.features.market_data_bridge.base_asset_adapter import (
    MarketDataBaseAssetLoader,
    resolve_base_asset_dataframe,
)
from quant_platform.features.market_data_bridge.bindings import (
    BaseAssetDatasetBinding,
    CrossAssetDatasetBinding,
    MacroDatasetBinding,
)
from quant_platform.features.market_data_bridge.coverage import (
    CoverageReport,
    SourceCoveragePolicy,
    evaluate_source_coverage,
)
from quant_platform.features.market_data_bridge.cross_asset_adapter import resolve_cross_asset_dataframe
from quant_platform.features.market_data_bridge.lineage import build_market_data_lineage, lineage_content_id
from quant_platform.features.market_data_bridge.macro_adapter import resolve_macro_dataframe
from quant_platform.features.registry import FeatureRegistry
from quant_platform.market_data.collectors.cross_asset.datasets import ComponentMarketDatasetManifestStore
from quant_platform.market_data.collectors.cross_asset.market_record import MarketDriverBarStore
from quant_platform.market_data.collectors.curated.datasets import ComponentDatasetManifestStore
from quant_platform.market_data.collectors.curated.macro_observation import CuratedObservationStore
from quant_platform.market_data.repository import MarketDataRepository

__all__ = [
    "CrossAssetRepository",
    "MacroRepository",
    "MarketDataResearchDatasetRequest",
    "build_research_dataset_from_market_data",
]


@dataclass(frozen=True, slots=True)
class MacroRepository:
    observation_store: CuratedObservationStore
    manifest_store: ComponentDatasetManifestStore


@dataclass(frozen=True, slots=True)
class CrossAssetRepository:
    bar_store: MarketDriverBarStore
    manifest_store: ComponentMarketDatasetManifestStore


@dataclass(frozen=True, slots=True)
class MarketDataResearchDatasetRequest:
    base_binding: BaseAssetDatasetBinding
    macro_bindings: dict[str, MacroDatasetBinding]
    """Keyed by the `source_name` that becomes both the
    `FeatureContext.macro_data` dict key and (by M3 convention) the
    `macro_{source_name}_*` feature-name prefix."""
    cross_asset_bindings: dict[str, CrossAssetDatasetBinding]
    """Keyed by `cross_asset_symbol` -- the `FeatureContext.cross_asset_data`
    dict key `features.cross_asset.cross_asset.register_cross_asset_features`
    was registered with."""
    coverage_policy: SourceCoveragePolicy
    build_request: ResearchDatasetBuildRequest
    higher_timeframe_data: Mapping[Timeframe, pd.DataFrame] | None = None
    """Genuine same-instrument higher-timeframe data, passed straight
    through to `ResearchDatasetBuilder` unchanged -- out of this bridge's
    scope to derive (spec Section 8: reuse M3's own higher-timeframe
    support unchanged); a caller wanting market_data-backed HTF data can
    resolve it via a second `base_asset_adapter.resolve_base_asset_dataframe`
    call at the coarser timeframe and pass the result here."""


def build_research_dataset_from_market_data(
    *, market_data_repository: MarketDataRepository, macro_repository: MacroRepository, cross_asset_repository: CrossAssetRepository,
    registry: FeatureRegistry, research_store: ResearchDatasetStore, manifest_store: ResearchManifestStore,
    request: MarketDataResearchDatasetRequest,
) -> tuple[ResearchDatasetManifest, CoverageReport]:
    macro_frames = {
        name: resolve_macro_dataframe(macro_repository.observation_store, macro_repository.manifest_store, binding)
        for name, binding in sorted(request.macro_bindings.items())
    }
    cross_asset_frames = {
        name: resolve_cross_asset_dataframe(cross_asset_repository.bar_store, cross_asset_repository.manifest_store, binding)
        for name, binding in sorted(request.cross_asset_bindings.items())
    }

    base_df = resolve_base_asset_dataframe(
        market_data_repository, request.base_binding, start=request.build_request.start, end=request.build_request.end
    )
    coverage_report = evaluate_source_coverage(
        base_df=base_df, base_timeframe=request.base_binding.timeframe, macro_frames=macro_frames,
        macro_bindings=request.macro_bindings, cross_asset_frames=cross_asset_frames, cross_asset_bindings=request.cross_asset_bindings,
        requested_start=request.build_request.start, requested_end=request.build_request.end, policy=request.coverage_policy,
    )

    lineage = build_market_data_lineage(
        base_binding=request.base_binding, macro_bindings=request.macro_bindings, cross_asset_bindings=request.cross_asset_bindings,
        coverage_report=coverage_report,
    )
    lineage_id = lineage_content_id(lineage)

    effective_request = replace(
        request.build_request, start=coverage_report.safe_start, end=coverage_report.safe_end,
        aux_input_content_hashes={**request.build_request.aux_input_content_hashes, "market_data_lineage_content_id": lineage_id},
        market_data_lineage=lineage,
    )

    base_loader = MarketDataBaseAssetLoader(market_data_repository, request.base_binding)
    builder = ResearchDatasetBuilder(
        historical_loader=base_loader, registry=registry, research_store=research_store, manifest_store=manifest_store,
        higher_timeframe_data=request.higher_timeframe_data, cross_asset_data=cross_asset_frames, macro_data=macro_frames,
    )
    manifest = builder.build(effective_request)
    return manifest, coverage_report
