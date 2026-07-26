"""Idempotent incremental updates to a canonical dataset.

An update never blindly appends: it re-requests a small OVERLAPPING window
at the tail of what's already canonicalized (see `determine_update_start`),
then reconciles every bar in that overlap against what's already stored --
classifying each as an EXACT match (nothing to do), a genuinely NEW bar
(insert), or a CONFLICTING revision (same `open_time`, different OHLCV).
Conflicts are never silently resolved: the default `RevisionPolicy.
REJECT_CONFLICTS` raises `UpdateConflictError` rather than picking a
winner, and even the opt-in `ACCEPT_NEWER_SOURCE` policy that explicitly
allows replacing a historical bar records exactly how many bars were
revised in the returned `UpdateReport` -- "never silently replace historical
bars" is enforced by making replacement an explicit, reported, opt-in
choice, not by pretending revisions can't happen.

Idempotency falls out of composing already-idempotent lower layers rather
than needing its own special-case logic: re-running an update with
identical input reclassifies every overlapping bar as "unchanged" and finds
zero new rows, so the merged partition is byte-identical to what's already
stored; `CanonicalStore.write_partition` then writes the same bytes (same
checksum), and `ManifestStore.save` recognizes the resulting manifest as a
content-duplicate of the current latest version and mints no new version
(see `historical.manifest`). Every step re-running the same update touches
is idempotent on its own, which composes into the pipeline being idempotent
as a whole -- this is a deliberate design property, not a coincidence, and
is directly exercised by `tests/unit/historical/test_update_pipeline.py::TestIdempotency`.

Crash recovery: each year-partition is written atomically (temp dir +
`os.replace`, see `historical.canonical_store`). An update spanning
multiple years is a SEQUENCE of independent atomic partition writes, not a
single cross-partition transaction -- a crash between two partition writes
leaves the ones already written valid and complete, and the ones not yet
reached untouched; simply re-running the same update is safe and idempotent
(per the paragraph above) and will finish the remaining partitions. This is
a deliberate, scale-appropriate choice for a single-symbol research
platform, not an oversight -- true multi-partition atomicity would require
machinery (e.g. a WAL or two-phase commit) this platform's data volumes do
not justify.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from quant_platform.core.exceptions import UpdateConflictError
from quant_platform.core.types import Timeframe
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.code_revision import capture_code_revision
from quant_platform.historical.locking import DatasetLock, dataset_lock_path
from quant_platform.historical.manifest import DatasetManifest, ManifestStore
from quant_platform.historical.models import (
    RAW_HISTORICAL_COLUMNS,
    coerce_historical_dtypes,
    schema_fingerprint,
    validate_historical_schema,
)

logger = logging.getLogger(__name__)

_VALUE_COLUMNS = [c for c in RAW_HISTORICAL_COLUMNS if c != "open_time"]


class RevisionPolicy(Enum):
    REJECT_CONFLICTS = "REJECT_CONFLICTS"
    """Raise `UpdateConflictError` if any re-fetched bar conflicts with an
    already-canonicalized bar at the same `open_time`. The safe default."""
    ACCEPT_NEWER_SOURCE = "ACCEPT_NEWER_SOURCE"
    """Explicitly accept the newly fetched source data as an authoritative
    revision, replacing the conflicting historical bar(s). Must be chosen
    deliberately by a caller who has a specific reason to trust source
    revisions (e.g. a broker that is known to correct bars after the fact)
    -- every replacement is still counted and reported."""


@dataclass(frozen=True, slots=True)
class UpdateReport:
    symbol: str
    timeframe: Timeframe
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    rows_received: int
    rows_unchanged: int
    rows_conflicting: int
    rows_inserted: int
    final_row_count: int
    final_checksum: str
    manifest_version: str
    performed_at: pd.Timestamp


def determine_update_start(
    canonical_store: CanonicalStore, *, symbol: str, timeframe: Timeframe, overlap_bars: int = 5
) -> pd.Timestamp | None:
    """The UTC instant an incremental update should re-request FROM: the
    latest canonical bar's `open_time` minus `overlap_bars` bar-widths, so
    the re-fetch overlaps a small tail window of already-stored data and
    can therefore detect a source-side revision to those bars. Returns
    `None` if no canonical data exists yet (the caller needs a full
    historical backfill, not an incremental update)."""
    if overlap_bars < 0:
        raise ValueError(f"overlap_bars must be non-negative, got {overlap_bars}")
    years = canonical_store.list_years(symbol=symbol, timeframe=timeframe)
    if not years:
        return None
    loaded = canonical_store.read_partition(symbol=symbol, timeframe=timeframe, year=years[-1])
    if loaded is None or len(loaded[0]) == 0:
        return None
    df, _ = loaded
    latest_open_time = pd.Timestamp(df["open_time"].max())
    return latest_open_time - overlap_bars * timeframe.duration


def apply_incremental_update(
    canonical_store: CanonicalStore,
    manifest_store: ManifestStore,
    new_bars: pd.DataFrame,
    *,
    symbol: str,
    timeframe: Timeframe,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    source_name: str,
    broker: str,
    parent_snapshot_ids: tuple[str, ...],
    pipeline_version: str,
    revision_policy: RevisionPolicy = RevisionPolicy.REJECT_CONFLICTS,
    normalization_settings: dict[str, object] | None = None,
    validation_policy: str = "STRICT",
    quality_summary: dict[str, object] | None = None,
    repair_summary: dict[str, object] | None = None,
    calendar_version: str | None = None,
    resampling_config: dict[str, object] | None = None,
    code_revision: str | None = None,
    reproducibility_seed: int | None = None,
) -> UpdateReport:
    """Reconcile `new_bars` (already normalized/validated/repaired --
    this function performs merge/reconciliation, not raw data cleaning)
    against the existing canonical dataset, write the affected
    year-partitions atomically, and save a new dataset manifest version.

    Also used to write a DERIVED (resampled) dataset's canonical partitions
    and manifest: pass `resampling_config` (e.g.
    `{"source_timeframe": "M1", "policy": "REJECT_INCOMPLETE"}`) so the
    saved manifest records that this dataset was produced by
    `historical.resampling.resample_ohlcv` rather than ingested directly
    from a source -- the reconciliation logic above is identical either
    way (a resampled dataset can still be incrementally extended as new
    source bars arrive and get re-resampled).

    `code_revision`, if not given, is auto-captured via
    `historical.code_revision.capture_code_revision` (a Git commit hash if
    one exists, otherwise a deterministic content hash of the pipeline's
    own source -- never silently absent just because this repository has
    no commits yet). The whole reconciliation-and-write is performed under
    a `historical.locking.DatasetLock` scoped to this `(symbol, timeframe)`,
    so two concurrent `apply_incremental_update` calls against the same
    dataset fail fast (or wait out a stale lock) rather than racing.
    """
    started_at = time.perf_counter()
    validate_historical_schema(new_bars, context="apply_incremental_update")
    if len(new_bars) == 0:
        raise ValueError("new_bars must not be empty")
    if not new_bars["open_time"].is_monotonic_increasing or new_bars["open_time"].duplicated().any():
        raise UpdateConflictError("new_bars must be sorted ascending with no duplicate open_time values")

    lock_path = dataset_lock_path(canonical_store.root, symbol=symbol, timeframe_value=timeframe.value)
    with DatasetLock(lock_path):
        return _apply_incremental_update_locked(
            canonical_store, manifest_store, new_bars,
            symbol=symbol, timeframe=timeframe, requested_start=requested_start, requested_end=requested_end,
            source_name=source_name, broker=broker, parent_snapshot_ids=parent_snapshot_ids,
            pipeline_version=pipeline_version, revision_policy=revision_policy,
            normalization_settings=normalization_settings, validation_policy=validation_policy,
            quality_summary=quality_summary, repair_summary=repair_summary, calendar_version=calendar_version,
            resampling_config=resampling_config,
            code_revision=code_revision if code_revision is not None else capture_code_revision(),
            reproducibility_seed=reproducibility_seed, started_at=started_at,
        )


def _apply_incremental_update_locked(
    canonical_store: CanonicalStore,
    manifest_store: ManifestStore,
    new_bars: pd.DataFrame,
    *,
    symbol: str,
    timeframe: Timeframe,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    source_name: str,
    broker: str,
    parent_snapshot_ids: tuple[str, ...],
    pipeline_version: str,
    revision_policy: RevisionPolicy,
    normalization_settings: dict[str, object] | None,
    validation_policy: str,
    quality_summary: dict[str, object] | None,
    repair_summary: dict[str, object] | None,
    calendar_version: str | None,
    resampling_config: dict[str, object] | None,
    code_revision: str | None,
    reproducibility_seed: int | None,
    started_at: float,
) -> UpdateReport:
    """The actual reconciliation logic, always called with the dataset
    lock already held -- see `apply_incremental_update`."""
    performed_at = pd.Timestamp.now(tz="UTC")
    years = sorted({int(y) for y in new_bars["open_time"].dt.year})

    rows_unchanged = 0
    rows_conflicting = 0
    rows_inserted = 0

    for year in years:
        year_start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        year_end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
        new_slice = new_bars.loc[
            (new_bars["open_time"] >= year_start) & (new_bars["open_time"] < year_end)
        ].reset_index(drop=True)
        if len(new_slice) == 0:
            continue

        existing_loaded = canonical_store.read_partition(symbol=symbol, timeframe=timeframe, year=year)
        existing_df = existing_loaded[0] if existing_loaded is not None else new_slice.iloc[0:0].copy()

        existing_indexed = existing_df.set_index("open_time")
        new_indexed = new_slice.set_index("open_time")

        common_index = existing_indexed.index.intersection(new_indexed.index)
        new_only_index = new_indexed.index.difference(existing_indexed.index)

        if len(common_index) > 0:
            existing_common = existing_indexed.loc[common_index, _VALUE_COLUMNS]
            new_common = new_indexed.loc[common_index, _VALUE_COLUMNS]
            differs = (existing_common.to_numpy() != new_common.to_numpy()).any(axis=1)
            conflicting_index = common_index[differs]
            unchanged_index = common_index[~differs]
        else:
            conflicting_index = common_index[:0]
            unchanged_index = common_index[:0]

        rows_unchanged += len(unchanged_index)
        rows_inserted += len(new_only_index)

        if len(conflicting_index) > 0:
            if revision_policy is RevisionPolicy.REJECT_CONFLICTS:
                raise UpdateConflictError(
                    f"{len(conflicting_index)} bar(s) in year {year} conflict with already-"
                    "canonicalized data (same open_time, different OHLCV). REJECT_CONFLICTS "
                    "policy refuses to silently replace historical bars -- re-run with "
                    "revision_policy=ACCEPT_NEWER_SOURCE to explicitly accept the new source "
                    "data as a deliberate revision.",
                    context={"symbol": symbol, "timeframe": timeframe.value, "year": year,
                              "conflict_count": len(conflicting_index)},
                )
            rows_conflicting += len(conflicting_index)

        if revision_policy is RevisionPolicy.ACCEPT_NEWER_SOURCE and len(conflicting_index) > 0:
            kept_existing = existing_indexed.drop(index=conflicting_index)
            rows_to_add_index = new_only_index.union(conflicting_index)
        else:
            kept_existing = existing_indexed
            rows_to_add_index = new_only_index

        rows_to_add = new_indexed.loc[rows_to_add_index]
        merged = pd.concat([kept_existing, rows_to_add]).sort_index()
        merged = merged.reset_index().rename(columns={"index": "open_time"})
        merged = coerce_historical_dtypes(merged)

        if merged["open_time"].duplicated().any():
            raise UpdateConflictError(
                f"Internal reconciliation error: merged partition for year {year} contains "
                "duplicate open_time values after merge -- this should be structurally "
                "impossible given the index-based union above.",
                context={"symbol": symbol, "timeframe": timeframe.value, "year": year},
            )

        canonical_store.write_partition(merged, symbol=symbol, timeframe=timeframe, year=year)
        logger.info(
            "Incremental update wrote partition: symbol=%s timeframe=%s year=%d rows=%d",
            symbol, timeframe.value, year, len(merged),
        )

    summary = _summarize_dataset(canonical_store, symbol=symbol, timeframe=timeframe)

    manifest = DatasetManifest(
        dataset_id=manifest_store.compute_dataset_id(
            symbol=symbol, timeframe=timeframe, source_name=source_name, broker=broker
        ),
        version="PENDING",
        parent_snapshot_ids=parent_snapshot_ids,
        symbol=symbol, source_name=source_name, broker=broker, timeframe=timeframe,
        utc_start=summary.utc_start, utc_end=summary.utc_end,
        row_count=summary.row_count,
        schema_fingerprint=schema_fingerprint(new_bars),
        content_checksum=summary.combined_checksum,
        created_at=performed_at,
        pipeline_version=pipeline_version,
        normalization_settings=normalization_settings or {},
        validation_policy=validation_policy,
        quality_summary=quality_summary or {},
        repair_summary=repair_summary or {},
        calendar_version=calendar_version,
        resampling_config=resampling_config,
        code_revision=code_revision,
        reproducibility_seed=reproducibility_seed,
        partition_content_ids=summary.partition_content_ids,
    )
    manifest_version = manifest_store.save(manifest)

    logger.info(
        "incremental update complete: symbol=%s timeframe=%s range=[%s, %s) received=%d unchanged=%d "
        "conflicting=%d inserted=%d final_row_count=%d manifest_version=%s duration_s=%.3f",
        symbol, timeframe.value, requested_start, requested_end, len(new_bars), rows_unchanged,
        rows_conflicting, rows_inserted, summary.row_count, manifest_version, time.perf_counter() - started_at,
    )

    return UpdateReport(
        symbol=symbol, timeframe=timeframe, requested_start=requested_start, requested_end=requested_end,
        rows_received=len(new_bars), rows_unchanged=rows_unchanged, rows_conflicting=rows_conflicting,
        rows_inserted=rows_inserted, final_row_count=summary.row_count, final_checksum=summary.combined_checksum,
        manifest_version=manifest_version, performed_at=performed_at,
    )


@dataclass(frozen=True, slots=True)
class _DatasetSummary:
    row_count: int
    combined_checksum: str
    utc_start: pd.Timestamp
    utc_end: pd.Timestamp
    partition_content_ids: dict[int, str]


def _summarize_dataset(canonical_store: CanonicalStore, *, symbol: str, timeframe: Timeframe) -> _DatasetSummary:
    """Row count, a combined content checksum, the overall UTC date range,
    AND the exact content id CURRENT for each year -- across EVERY
    year-partition of a dataset (not just the ones an update happened to
    touch), since a manifest describes the whole dataset's state, not one
    update's delta. `partition_content_ids` is what makes that exact state
    reconstructable later regardless of what CURRENT points at by then --
    see `historical.canonical_store` and `historical.loader`."""
    years = canonical_store.list_years(symbol=symbol, timeframe=timeframe)
    total_rows = 0
    partition_checksums: list[str] = []
    partition_content_ids: dict[int, str] = {}
    overall_min: pd.Timestamp | None = None
    overall_max: pd.Timestamp | None = None
    for year in years:
        loaded = canonical_store.read_partition(symbol=symbol, timeframe=timeframe, year=year)
        if loaded is None:
            continue
        _, metadata = loaded
        total_rows += metadata.row_count
        partition_checksums.append(f"{year}:{metadata.content_checksum}")
        partition_content_ids[year] = metadata.content_id
        if metadata.min_open_time is not None:
            overall_min = metadata.min_open_time if overall_min is None else min(overall_min, metadata.min_open_time)
        if metadata.max_open_time is not None:
            overall_max = metadata.max_open_time if overall_max is None else max(overall_max, metadata.max_open_time)

    combined_checksum = hashlib.sha256(",".join(partition_checksums).encode("utf-8")).hexdigest()
    default_start = pd.Timestamp(0, tz="UTC")
    return _DatasetSummary(
        row_count=total_rows, combined_checksum=combined_checksum,
        utc_start=overall_min or default_start, utc_end=overall_max or default_start,
        partition_content_ids=partition_content_ids,
    )


__all__ = ["RevisionPolicy", "UpdateReport", "apply_incremental_update", "determine_update_start"]
