"""Resolves a `bindings.BaseAssetDatasetBinding` into an actual OHLCV
`pandas.DataFrame`, playing the same role for `market_data`-backed
base-asset data that `historical.loader.DatasetLoader` plays for the
original Milestone 2/3 pipeline.

`MarketDataBaseAssetLoader` implements `historical.loader.
HistoricalDatasetLoaderProtocol` structurally (never inherits from
`DatasetLoader`) -- it is a drop-in `historical_loader=` for the
UNMODIFIED `features.dataset_builder.ResearchDatasetBuilder`. No feature
computation, labeling, splitting, or manifest-writing logic lives here;
this module's only job is producing a verified, point-in-time-safe
`core.types.OHLCV_COLUMNS` frame plus a `HistoricalManifestLike`-shaped
identity object.

VERIFICATION, NOT TRUST. `verify_base_asset_binding` never simply reads
`market_data.events.MarketEventStore.read_events` and hands the result
back -- it independently re-derives the current `market_data.manifests.
DatasetManifest` for `(RAW_MARKET_EVENTS, provider, canonical_instrument_id)`
and requires the binding's pinned `dataset_id` to equal it EXACTLY. This
is deliberately fail-closed rather than fail-open: `market_data.
partitions.PartitionStore` is current-version-only storage (see its own
module docstring), so an OLDER, superseded `dataset_id` can no longer be
reconstructed byte-for-byte once its partitions have been rebuilt by a
later commit -- there is no market_data API to selectively read "exactly
the events that made up manifest version N" once N is no longer current.
A caller whose pin has gone stale must explicitly re-pin (a deliberate
action visible in the resulting research dataset's own lineage/identity),
never silently receive whatever the repository has since become.

AVAILABILITY POLICY FOR THE BASE ASSET. `market_data.candles.Candle`
carries no `availability_time` field (unlike the richer Phase 4B/4C
records) -- this bridge's own explicit, honest policy
(`BaseAssetDatasetBinding.availability_policy_id ==
"close_time_as_availability"`, the only policy this module implements) is
that a candle becomes visible exactly at its own `close_time`
(`open_time + timeframe.duration`). This is not a bridge-invented rule:
it is EXACTLY `features.engine.FeatureEngine.compute`'s own availability-
instant derivation for the base timeframe (`base_reset["open_time"] +
timeframe.duration`, see that module) -- this adapter changes nothing
about how the base timeframe's own availability is computed, it only
supplies the `open_time`/OHLCV values the engine already derives that
instant from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from quant_platform.core.exceptions import SourceVerificationError
from quant_platform.core.types import OHLCV_COLUMNS
from quant_platform.features.market_data_bridge.bindings import BaseAssetDatasetBinding
from quant_platform.historical.loader import HistoricalManifestLike, LoadRequest
from quant_platform.market_data.candles import Candle
from quant_platform.market_data.identity import compute_content_id
from quant_platform.market_data.ingestion import semantic_digest_for_raw_events
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.repository import MarketDataRepository

__all__ = [
    "BASE_ASSET_CONTENT_CHECKSUM_KIND",
    "MarketDataBaseAssetLoader",
    "ResolvedHistoricalManifest",
    "resolve_base_asset_dataframe",
    "verify_base_asset_binding",
]

BASE_ASSET_CONTENT_CHECKSUM_KIND = "market_data_bridge_base_asset_content_checksum"


@dataclass(frozen=True, slots=True)
class ResolvedHistoricalManifest:
    """Satisfies `historical.loader.HistoricalManifestLike`
    (`.dataset_id`/`.version`/`.content_checksum`) -- the market_data-
    backed analogue of a real `historical.manifest.DatasetManifest`,
    minted entirely from the base-asset binding's OWN pinned identity and
    the verified event set, never fabricated."""

    dataset_id: str
    version: str
    content_checksum: str


def _dataset_key(binding: BaseAssetDatasetBinding) -> DatasetKey:
    return DatasetKey(
        dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id=binding.canonical_instrument_id, provider=binding.provider
    )


def verify_base_asset_binding(repository: MarketDataRepository, binding: BaseAssetDatasetBinding) -> list[Candle]:
    """Independently re-verifies `binding` against the CURRENT durable
    `market_data` repository state and returns exactly the pinned
    `Candle` set for `binding.timeframe`, deterministically ordered by
    `(event_time, event_id)` with same-coordinate conflicts rejected.
    Raises `SourceVerificationError` (never silently substitutes) if the
    repository does not durably hold the pinned version -- see module
    docstring."""
    dataset_key = _dataset_key(binding)
    current_manifest = repository.manifest_store.read_current(dataset_key)
    if current_manifest is None:
        raise SourceVerificationError(
            f"No market_data DatasetManifest exists for provider={binding.provider!r} "
            f"instrument_id={binding.canonical_instrument_id!r} -- nothing has ever been durably ingested "
            "for this base-asset binding.",
            context={"provider": binding.provider, "instrument_id": binding.canonical_instrument_id},
        )
    if current_manifest.dataset_id != binding.pinned_dataset_id:
        raise SourceVerificationError(
            f"BaseAssetDatasetBinding.pinned_dataset_id={binding.pinned_dataset_id!r} does not match the "
            f"CURRENT market_data manifest dataset_id={current_manifest.dataset_id!r} for "
            f"{binding.provider}/{binding.canonical_instrument_id}. The repository has advanced past this "
            "pinned version (or the pin was never valid) -- re-pin this binding to the current dataset_id "
            "as a deliberate, explicit action; this bridge never silently substitutes the current state for "
            "a stale pin.",
            context={
                "pinned_dataset_id": binding.pinned_dataset_id, "current_dataset_id": current_manifest.dataset_id,
                "provider": binding.provider, "instrument_id": binding.canonical_instrument_id,
            },
        )

    all_events = repository.event_store.read_events(binding.provider, binding.canonical_instrument_id)
    recomputed_digest = semantic_digest_for_raw_events(all_events)
    if recomputed_digest != current_manifest.semantic_digest:
        raise SourceVerificationError(
            "Recomputed semantic digest of the live event-store read does not match the current manifest's "
            "own recorded semantic_digest -- the durable event store and its manifest have diverged; refusing "
            "to build a research dataset from unverifiable base-asset data.",
            context={
                "provider": binding.provider, "instrument_id": binding.canonical_instrument_id,
                "manifest_semantic_digest": current_manifest.semantic_digest, "recomputed_semantic_digest": recomputed_digest,
            },
        )

    if binding.expected_event_kind != "candle":
        raise SourceVerificationError(
            f"BaseAssetDatasetBinding.expected_event_kind={binding.expected_event_kind!r} is not supported -- "
            "this bridge only resolves 'candle' base-asset timelines.",
            context={"expected_event_kind": binding.expected_event_kind},
        )

    candles = [
        e for e in all_events
        if isinstance(e, Candle) and e.instrument_id == binding.canonical_instrument_id and e.timeframe is binding.timeframe
    ]
    if not candles:
        raise SourceVerificationError(
            f"Binding {binding.binding_id} resolved zero Candle events for "
            f"{binding.provider}/{binding.canonical_instrument_id}/{binding.timeframe.value} -- the pinned "
            "dataset_id is verified, but no matching candles exist at that timeframe.",
            context={"provider": binding.provider, "instrument_id": binding.canonical_instrument_id, "timeframe": binding.timeframe.value},
        )

    ordered = sorted(candles, key=lambda c: (c.event_time, c.event_id))
    by_open_time: dict[pd.Timestamp, Candle] = {}
    for candle in ordered:
        open_time = pd.Timestamp(candle.event_time)
        prior = by_open_time.get(open_time)
        if prior is not None and prior.event_id != candle.event_id:
            raise SourceVerificationError(
                f"Conflicting Candle events at open_time={open_time} for "
                f"{binding.provider}/{binding.canonical_instrument_id}/{binding.timeframe.value}: event_id "
                f"{prior.event_id!r} vs {candle.event_id!r} claim the same open coordinate with different "
                "content -- this bridge never silently picks one; the underlying repository must be "
                "reconciled before this binding can be resolved.",
                context={"open_time": str(open_time), "event_id_a": prior.event_id, "event_id_b": candle.event_id},
            )
        if prior is None:
            by_open_time[open_time] = candle
    deduped = sorted(by_open_time.values(), key=lambda c: c.event_time)
    return deduped


def resolve_base_asset_dataframe(
    repository: MarketDataRepository, binding: BaseAssetDatasetBinding, *, start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    """Verified, deterministically ordered `core.types.OHLCV_COLUMNS`
    frame for `[start, end)`, in UTC, strictly ascending, no duplicate
    `open_time`. The single, documented Decimal -> float64 boundary
    crossing for base-asset data (mirrors `historical.loader.
    DatasetLoader.load_for_engine`'s own analogous projection)."""
    if start.tzinfo is None or end.tzinfo is None:
        raise SourceVerificationError("resolve_base_asset_dataframe: start/end must be timezone-aware")
    candles = verify_base_asset_binding(repository, binding)
    rows: list[dict[str, object]] = []
    for candle in candles:
        open_time = pd.Timestamp(candle.event_time)
        if open_time < start or open_time >= end:
            continue
        volume = candle.volume
        rows.append(
            {
                "open_time": open_time, "open": float(candle.open), "high": float(candle.high), "low": float(candle.low),
                "close": float(candle.close), "volume": (0.0 if volume is None else float(volume)),
            }
        )
    df = pd.DataFrame(rows, columns=list(OHLCV_COLUMNS))
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.reset_index(drop=True)


class MarketDataBaseAssetLoader:
    """Implements `historical.loader.HistoricalDatasetLoaderProtocol` --
    a drop-in `historical_loader=` for the unmodified
    `features.dataset_builder.ResearchDatasetBuilder`, backed by one
    pinned `BaseAssetDatasetBinding` instead of `historical.loader.
    DatasetLoader`'s `CanonicalStore`/`ManifestStore` pair."""

    def __init__(self, repository: MarketDataRepository, binding: BaseAssetDatasetBinding) -> None:
        self._repository = repository
        self._binding = binding

    @property
    def binding(self) -> BaseAssetDatasetBinding:
        return self._binding

    def _check_request(self, request: LoadRequest) -> None:
        if request.symbol != self._binding.canonical_instrument_id:
            raise SourceVerificationError(
                f"LoadRequest.symbol={request.symbol!r} does not match this loader's bound instrument "
                f"{self._binding.canonical_instrument_id!r}",
                context={"requested_symbol": request.symbol, "bound_instrument_id": self._binding.canonical_instrument_id},
            )
        if request.timeframe is not self._binding.timeframe:
            raise SourceVerificationError(
                f"LoadRequest.timeframe={request.timeframe.value!r} does not match this loader's bound "
                f"timeframe {self._binding.timeframe.value!r}",
                context={"requested_timeframe": request.timeframe.value, "bound_timeframe": self._binding.timeframe.value},
            )
        if request.dataset_version is not None and request.dataset_version != self._binding.pinned_dataset_id:
            raise SourceVerificationError(
                f"LoadRequest.dataset_version={request.dataset_version!r} does not match this loader's bound "
                f"pinned_dataset_id {self._binding.pinned_dataset_id!r} -- a `MarketDataBaseAssetLoader` "
                "always serves exactly the one dataset_id its binding pins; leave dataset_version=None to "
                "use it implicitly.",
                context={"requested_dataset_version": request.dataset_version, "bound_dataset_id": self._binding.pinned_dataset_id},
            )

    def resolve_manifest(self, request: LoadRequest) -> HistoricalManifestLike:
        self._check_request(request)
        candles = verify_base_asset_binding(self._repository, self._binding)
        content_checksum = compute_content_id(
            BASE_ASSET_CONTENT_CHECKSUM_KIND, {"event_ids": sorted(c.event_id for c in candles)}
        )
        return ResolvedHistoricalManifest(
            dataset_id=self._binding.pinned_dataset_id, version=self._binding.binding_id, content_checksum=content_checksum
        )

    def load_for_engine(
        self, request: LoadRequest, *, volume_source: Literal["tick_volume", "real_volume"] = "tick_volume"  # noqa: ARG002
    ) -> pd.DataFrame:
        """`volume_source` is accepted only for `HistoricalDatasetLoaderProtocol`
        structural compatibility -- `market_data.candles.Candle` carries a
        single `volume` field with no tick/real-volume distinction, so
        this parameter is always a documented no-op here."""
        self._check_request(request)
        return resolve_base_asset_dataframe(self._repository, self._binding, start=request.start, end=request.end)
