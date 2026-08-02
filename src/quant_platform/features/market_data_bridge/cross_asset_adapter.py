"""Resolves a `bindings.CrossAssetDatasetBinding` into the OHLCV
`pandas.DataFrame` shape `features.cross_asset.cross_asset.
register_cross_asset_features` expects as one entry of
`FeatureContext.cross_asset_data` (keyed by `cross_asset_symbol`) --
reads Milestone 10 Phase 4C's cross-asset layer (`collectors.
cross_asset.market_record.MarketDriverBarStore` + `collectors.
cross_asset.datasets.ComponentMarketDatasetManifestStore`).

THE AVAILABILITY-SHIFT ADAPTER (the one genuinely new alignment idea in
this module, spec Section 8's "if conventions differ, implement an
explicit adapter and document the conversion"). `features.alignment.
align_higher_timeframe` -- the ONLY alignment primitive this bridge
reuses for cross-asset data, exactly like `features.cross_asset.
cross_asset` itself already does for its own feature family -- derives
its reveal instant internally as `open_time + timeframe.duration`; it has
no parameter for an externally supplied `availability_time`. A
`MarketDriverBar`, on the other hand, carries its OWN resolved
`availability_time`, which is `>= close_time` and MAY be materially later
than a naive `open_time + timeframe.duration` guess (e.g. Alpha Vantage
daily ETF bars under `CLOSE_PLUS_CONSERVATIVE_DELAY`, next-day
availability). To make `align_higher_timeframe`'s own close-time-based
reveal rule land on the bar's TRUE `availability_time` without touching
that shared, already-tested M3 primitive, this module emits a SYNTHETIC
`open_time` for each row:

    synthetic_open_time := availability_time - cross_asset_timeframe.duration

so that `align_higher_timeframe`'s internal `synthetic_open_time +
cross_asset_timeframe.duration == availability_time` exactly. The OHLCV
VALUES stored under that synthetic row are unchanged (the bar's real
open/high/low/close/volume) -- only the row's timestamp coordinate is
shifted, and only for the purpose of feeding this one alignment function
correctly. `resolve_cross_asset_dataframe` documents this explicitly and
`verification.py`'s truncation-invariance proof exercises it end-to-end.
"""

from __future__ import annotations

import pandas as pd

from quant_platform.core.exceptions import SourceVerificationError
from quant_platform.core.types import OHLCV_COLUMNS
from quant_platform.features.market_data_bridge.bindings import CrossAssetDatasetBinding
from quant_platform.market_data.collectors.cross_asset.datasets import ComponentMarketDatasetManifestStore
from quant_platform.market_data.collectors.cross_asset.market_record import (
    MarketDriverBar,
    MarketDriverBarStore,
)
from quant_platform.market_data.identity import compute_content_id

__all__ = ["resolve_cross_asset_dataframe", "verify_cross_asset_binding"]


def verify_cross_asset_binding(
    bar_store: MarketDriverBarStore, manifest_store: ComponentMarketDatasetManifestStore, binding: CrossAssetDatasetBinding,
) -> list[MarketDriverBar]:
    """Independently re-verifies `binding` against the CURRENT durable
    cross-asset repository state: the pinned `component_manifest_id` must
    equal the store's current manifest for `binding.mapping_id`, AND a
    live `MarketDriverBarStore` read must reproduce that manifest's own
    `semantic_digest` (via the exact same `compute_content_id(
    "cross_asset_component_semantic_digest", ...)` formula `collectors.
    cross_asset.datasets.create_component_market_dataset_manifest` itself
    uses). Also rejects a same-`open_time` conflicting-bar pair (two
    DIFFERENT bars claiming the same coordinate) the same way
    `base_asset_adapter.verify_base_asset_binding` does for `Candle`s --
    `ComponentMarketDatasetManifest`'s own `conflicting_coordinate_count`
    is a CALLER-supplied count Phase 4C's own backfill pipeline computes
    at write time, not something this read-only bridge can independently
    trust without re-checking the live bar set itself."""
    current = manifest_store.read_current(binding.mapping_id)
    if current is None:
        raise SourceVerificationError(
            f"No cross-asset component manifest exists for mapping_id={binding.mapping_id!r}",
            context={"mapping_id": binding.mapping_id},
        )
    if current.component_manifest_id != binding.component_manifest_id:
        raise SourceVerificationError(
            f"CrossAssetDatasetBinding.component_manifest_id={binding.component_manifest_id!r} does not match "
            f"the CURRENT component manifest id={current.component_manifest_id!r} for mapping_id="
            f"{binding.mapping_id!r} -- re-pin this binding to the current id as a deliberate, explicit action; "
            "this bridge never silently substitutes newer data for a stale pin.",
            context={"pinned": binding.component_manifest_id, "current": current.component_manifest_id},
        )
    bars = bar_store.read_bars(binding.provider, binding.canonical_driver_id, binding.instrument_form)
    recomputed_digest = compute_content_id(
        "cross_asset_component_semantic_digest", {"bar_ids": sorted(b.bar_id for b in bars)}
    )
    if recomputed_digest != current.semantic_digest:
        raise SourceVerificationError(
            "Recomputed semantic digest of a live bar-store read does not match the current manifest's own "
            "recorded semantic_digest -- the durable bar store and its manifest have diverged; refusing to "
            "build a research dataset from unverifiable cross-asset data.",
            context={
                "mapping_id": binding.mapping_id, "manifest_semantic_digest": current.semantic_digest,
                "recomputed_semantic_digest": recomputed_digest,
            },
        )
    if not bars:
        raise SourceVerificationError(
            f"Binding {binding.binding_id} resolved zero MarketDriverBar events for mapping_id={binding.mapping_id!r}",
            context={"mapping_id": binding.mapping_id},
        )

    ordered = sorted(bars, key=lambda b: (b.open_time, b.bar_id))
    by_open_time: dict[pd.Timestamp, MarketDriverBar] = {}
    for bar in ordered:
        open_time = pd.Timestamp(bar.open_time)
        prior = by_open_time.get(open_time)
        if prior is not None and prior.bar_id != bar.bar_id:
            raise SourceVerificationError(
                f"Conflicting MarketDriverBar events at open_time={open_time} for mapping_id="
                f"{binding.mapping_id!r}: bar_id {prior.bar_id!r} vs {bar.bar_id!r} claim the same open "
                "coordinate with different content -- this bridge never silently picks one; the underlying "
                "repository must be reconciled before this binding can be resolved.",
                context={"open_time": str(open_time), "bar_id_a": prior.bar_id, "bar_id_b": bar.bar_id},
            )
        if prior is None:
            by_open_time[open_time] = bar
    deduped = sorted(by_open_time.values(), key=lambda b: b.open_time)
    return deduped


def resolve_cross_asset_dataframe(
    bar_store: MarketDriverBarStore, manifest_store: ComponentMarketDatasetManifestStore, binding: CrossAssetDatasetBinding,
) -> pd.DataFrame:
    """Verified, deterministically ordered `core.types.OHLCV_COLUMNS`
    frame for `binding`, with `open_time` replaced by the SYNTHETIC
    availability-shifted coordinate described in the module docstring --
    the single, documented Decimal -> float64 boundary crossing plus the
    one documented timestamp adjustment for cross-asset data. Never
    exposes a bar before its own true `availability_time` (which is
    itself always `>= close_time`, enforced by `MarketDriverBar.
    __post_init__` -- so an incomplete/not-yet-closed bar can never enter
    this frame at all)."""
    bars = verify_cross_asset_binding(bar_store, manifest_store, binding)
    duration = binding.timeframe.duration
    rows: list[dict[str, object]] = []
    for bar in bars:
        synthetic_open_time = pd.Timestamp(bar.availability_time) - duration
        volume = bar.volume
        rows.append(
            {
                "open_time": synthetic_open_time, "open": float(bar.open), "high": float(bar.high), "low": float(bar.low),
                "close": float(bar.close), "volume": (0.0 if volume is None else float(volume)),
            }
        )
    df = pd.DataFrame(rows, columns=list(OHLCV_COLUMNS))
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time", kind="mergesort").reset_index(drop=True)
    if df["open_time"].duplicated().any():
        raise SourceVerificationError(
            f"resolve_cross_asset_dataframe: two bars for mapping_id={binding.mapping_id!r} resolved to the "
            "same synthetic open_time after the availability shift -- this can happen if the binding's "
            "availability_policy_id does not actually resolve availability_time to a strictly-increasing "
            "sequence with >= 1 timeframe-duration spacing; refusing to hand `align_higher_timeframe` "
            "ambiguous input.",
            context={"mapping_id": binding.mapping_id},
        )
    return df
