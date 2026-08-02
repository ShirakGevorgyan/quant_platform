"""Reproducible dataset loading -- the read-side boundary between the
historical data pipeline and everything that consumes it (research
notebooks, and directly, `multiframe.cursor.TimeframeCursor` /
`engine.backtest_engine.BacktestEngine`).

A `LoadRequest` pins down exactly what is being asked for: dataset version
(or explicitly "latest"), symbol, timeframe, UTC date range, a minimum
acceptable quality level, and optionally a column subset.

**Exact version reconstruction.** `DatasetLoader` never reads "whatever is
currently in canonical storage" -- it always resolves the requested (or
latest) `DatasetManifest` first, then reads EXACTLY the content-addressed
partition blobs that manifest recorded (`DatasetManifest.
partition_content_ids`, via `CanonicalStore.read_partition_by_content_id`).
Because those blobs are immutable and content-addressed (see
`historical.canonical_store`'s module docstring), this means loading an
OLD `dataset_version` months after the dataset has been incrementally
updated or revised still returns the byte-for-byte data that version
actually described -- not whatever the dataset has since become. This is
what makes a backtest run against a pinned dataset version reproducible.

`load_for_engine` is the explicit, documented boundary crossing from this
pipeline's wider canonical schema (`historical.models.RAW_HISTORICAL_COLUMNS`
-- `tick_volume`/`real_volume`/`spread` kept distinct) down to
`core.types.OHLCV_COLUMNS` (a single `volume` column), which is what
`TimeframeCursor`/`BacktestEngine` are typed against. Which volume field
becomes "volume" is a caller-visible, explicit parameter (default
`tick_volume`, since that's what's actually populated for spot/CFD XAUUSD
on most retail MT5 brokers -- see `historical.models` module docstring) --
never an implicit, silent choice.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

from quant_platform.core.exceptions import DataQualityError, ManifestError, SnapshotError
from quant_platform.core.types import OHLCV_COLUMNS, Timeframe
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.manifest import DatasetManifest, ManifestStore
from quant_platform.historical.models import RAW_HISTORICAL_COLUMNS
from quant_platform.historical.timezones import require_utc

logger = logging.getLogger(__name__)

RequiredQuality = Literal["strict", "lenient"]


@dataclass(frozen=True, slots=True)
class LoadRequest:
    symbol: str
    timeframe: Timeframe
    start: pd.Timestamp
    end: pd.Timestamp
    dataset_version: str | None = None
    """`None` resolves to the dataset's latest manifest version."""
    required_quality: RequiredQuality = "strict"
    """`"strict"` (default) refuses to load a dataset whose manifest
    recorded ANY critical quality issue at build time; `"lenient"` loads
    regardless, trusting the caller to have already inspected
    `manifest.quality_summary` themselves."""
    columns: tuple[str, ...] | None = None
    """`None` returns every `RAW_HISTORICAL_COLUMNS` column."""

    def __post_init__(self) -> None:
        require_utc(pd.Series([self.start, self.end]), context="LoadRequest")
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be after start ({self.start})")
        if self.columns is not None:
            unknown = set(self.columns) - set(RAW_HISTORICAL_COLUMNS)
            if unknown:
                raise ValueError(f"Unknown column(s) requested: {sorted(unknown)}")


class DatasetLoader:
    def __init__(self, canonical_store: CanonicalStore, manifest_store: ManifestStore) -> None:
        self._canonical_store = canonical_store
        self._manifest_store = manifest_store

    def resolve_manifest(self, request: LoadRequest) -> DatasetManifest:
        manifest = self._manifest_store.load(
            symbol=request.symbol, timeframe=request.timeframe, version=request.dataset_version
        )
        if request.required_quality == "strict":
            critical_count = int(str(manifest.quality_summary.get("critical", 0) or 0))
            if critical_count > 0:
                raise DataQualityError(
                    f"Manifest version {manifest.version} for {request.symbol}/{request.timeframe.value} "
                    f"recorded {critical_count} critical quality issue(s) at build time; "
                    "required_quality='strict' refuses to load it. Pass required_quality='lenient' to "
                    "load it anyway.",
                    context={"symbol": request.symbol, "timeframe": request.timeframe.value,
                              "version": manifest.version, "critical_count": critical_count},
                )
        return manifest

    def load(self, request: LoadRequest) -> pd.DataFrame:
        """Resolve `request` against the manifest and canonical storage,
        returning a fresh, defensively-copied `RAW_HISTORICAL_COLUMNS`
        (or the requested subset) DataFrame in strict chronological order
        with no duplicate timestamps. Always reads EXACTLY the content-
        addressed partitions the resolved manifest recorded -- see module
        docstring for why this is what makes a pinned `dataset_version`
        reproducible rather than silently drifting with later updates."""
        started_at = time.perf_counter()
        manifest = self.resolve_manifest(request)
        df = self._read_manifest_range(manifest, start=request.start, end=request.end)

        require_utc(df["open_time"], context="DatasetLoader.load")
        if not df["open_time"].is_monotonic_increasing:
            raise ManifestError(
                "Loaded data is not in strict chronological order",
                context={"symbol": request.symbol, "timeframe": request.timeframe.value},
            )
        if df["open_time"].duplicated().any():
            raise ManifestError(
                "Loaded data contains duplicate open_time values",
                context={"symbol": request.symbol, "timeframe": request.timeframe.value},
            )

        if request.columns is not None:
            df = df[list(request.columns)]

        result = df.reset_index(drop=True).copy()
        logger.info(
            "dataset load complete: symbol=%s timeframe=%s range=[%s, %s) manifest_version=%s rows=%d "
            "duration_s=%.3f",
            request.symbol, request.timeframe.value, request.start, request.end, manifest.version,
            len(result), time.perf_counter() - started_at,
        )
        return result

    def _read_manifest_range(self, manifest: DatasetManifest, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Read `[start, end)` using EXACTLY the content ids `manifest`
        recorded per year -- never `CanonicalStore.read_range` (which
        follows `CURRENT`, i.e. "whatever the dataset has become by now").
        A manifest with no recorded years (e.g. one saved before this
        field existed) yields an empty result rather than silently falling
        back to reading current data under a stale label."""
        relevant_years = sorted(
            y for y in manifest.partition_content_ids if start.year <= y <= end.year
        )
        frames: list[pd.DataFrame] = []
        for year in relevant_years:
            content_id = manifest.partition_content_ids[year]
            try:
                loaded = self._canonical_store.read_partition_by_content_id(
                    symbol=manifest.symbol, timeframe=manifest.timeframe, year=year, content_id=content_id,
                )
            except SnapshotError as exc:
                raise ManifestError(
                    f"Canonical data for {manifest.symbol}/{manifest.timeframe.value} (manifest version "
                    f"{manifest.version}, year {year}, content {content_id!r}) failed verification "
                    f"during load: {exc}",
                    context={"symbol": manifest.symbol, "timeframe": manifest.timeframe.value, "year": year},
                ) from exc
            if loaded is None:
                raise ManifestError(
                    f"Manifest version {manifest.version} for {manifest.symbol}/{manifest.timeframe.value} "
                    f"references content {content_id!r} for year {year}, but that content no longer "
                    "exists in canonical storage -- it should never be possible to remove a "
                    "content-addressed partition once written; this indicates out-of-band tampering "
                    "with the storage root.",
                    context={"symbol": manifest.symbol, "timeframe": manifest.timeframe.value, "year": year,
                              "content_id": content_id},
                )
            df, _ = loaded
            mask = (df["open_time"] >= start) & (df["open_time"] < end)
            frames.append(df.loc[mask])

        if not frames:
            return pd.DataFrame({col: pd.Series([], dtype=object) for col in RAW_HISTORICAL_COLUMNS})
        return pd.concat(frames, ignore_index=True)

    def load_for_engine(
        self, request: LoadRequest, *, volume_source: Literal["tick_volume", "real_volume"] = "tick_volume"
    ) -> pd.DataFrame:
        """Load, then project down to `core.types.OHLCV_COLUMNS` for
        direct use by `TimeframeCursor`/`BacktestEngine`. `volume_source`
        picks which canonical volume field becomes the engine's single
        `volume` column -- an explicit choice, never a silent default
        buried in behavior (see module docstring)."""
        full_request = request if request.columns is None else LoadRequest(
            symbol=request.symbol, timeframe=request.timeframe, start=request.start, end=request.end,
            dataset_version=request.dataset_version, required_quality=request.required_quality, columns=None,
        )
        df = self.load(full_request)
        projected = df[["open_time", "open", "high", "low", "close"]].copy()
        projected["volume"] = df[volume_source]
        return projected[list(OHLCV_COLUMNS)]


@runtime_checkable
class HistoricalManifestLike(Protocol):
    """Structural subset of `historical.manifest.DatasetManifest` that
    `features.dataset_builder.ResearchDatasetBuilder.build` actually reads
    off a resolved manifest -- exactly `.dataset_id`, `.version`, and
    `.content_checksum` (see that module's `build()` body). A caller
    substituting a non-`historical`-backed base-asset source (Milestone 10
    Phase 4D's `features.market_data_bridge.base_asset_adapter`) only
    needs to satisfy this narrow shape, never construct a full real
    `DatasetManifest`. Declared via `@property` (read-only) rather than
    plain attributes -- `ResearchDatasetBuilder.build()` only ever READS
    these three fields, and a read-only Protocol lets an immutable
    (`frozen=True`) dataclass implementation (e.g. `market_data_bridge.
    base_asset_adapter.ResolvedHistoricalManifest`) satisfy it; a
    plain-attribute Protocol declaration requires a SETTABLE attribute,
    which mypy correctly refuses to match against a frozen dataclass's
    field."""

    @property
    def dataset_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def content_checksum(self) -> str: ...


@runtime_checkable
class HistoricalDatasetLoaderProtocol(Protocol):
    """Structural interface `ResearchDatasetBuilder` actually depends on --
    exactly the 2 methods its `build()` calls (`resolve_manifest`,
    `load_for_engine`). The concrete `DatasetLoader` above already
    satisfies this unchanged (added narrowly so `ResearchDatasetBuilder.
    __init__`'s type hint can accept it -- a type-hint-only, zero-runtime-
    behavior change); it exists so a non-`historical`-backed base-asset
    source can be substituted at the SAME call site without a second
    `ResearchDatasetBuilder` or any change to its leak-safety guarantees --
    see `features.market_data_bridge.base_asset_adapter` (Milestone 10
    Phase 4D)."""

    def resolve_manifest(self, request: LoadRequest) -> HistoricalManifestLike: ...

    def load_for_engine(
        self, request: LoadRequest, *, volume_source: Literal["tick_volume", "real_volume"] = "tick_volume"
    ) -> pd.DataFrame: ...


__all__ = [
    "DatasetLoader",
    "HistoricalDatasetLoaderProtocol",
    "HistoricalManifestLike",
    "LoadRequest",
    "RequiredQuality",
]
