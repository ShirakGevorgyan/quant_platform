"""Canonical Parquet storage for cleaned, validated historical bars.

**Content-addressed, immutable partition storage** (redesigned during the
Milestone 2 release-readiness audit -- see the audit report's "Architecture
changes" section for the reproducibility gap this closes). Layout:

    <storage_root>/canonical/symbol=<symbol>/timeframe=<timeframe>/
        year=<YYYY>/
            content/<sha256>/
                data.parquet     -- `historical.models.RAW_HISTORICAL_COLUMNS`
                metadata.json    -- `PartitionMetadata`
                _SUCCESS
            CURRENT              -- one line: the sha256 of the content
                                     directory that is "current" for casual
                                     (non-version-pinned) reads

Every `content/<sha256>/` directory is written EXACTLY ONCE and never
modified or deleted afterward -- `write_partition` never overwrites a
partition's data, it writes new content under its own content-address (a
sha256 of the Parquet bytes, doubling as `PartitionMetadata.content_checksum`)
and then atomically repoints `CURRENT` at it. Two consequences fall out of
this for free:

1. **Copy-on-write / deduplication.** If a year's data is unchanged between
   two dataset versions (the common case -- only the most recent year
   actually changes on a routine incremental update), the SAME content
   directory is reused; nothing is rewritten or duplicated on disk.
2. **Exact historical reconstruction.** A `DatasetManifest` records, per
   year, exactly which content-address was current when that manifest
   version was saved (`DatasetManifest.partition_content_ids`). Because
   that content directory is never touched again, `DatasetLoader` can
   reconstruct ANY past dataset version byte-for-byte by reading the
   content ids that specific manifest recorded, via
   `read_partition_by_content_id` -- not just "whatever is current now".
   This is what makes a backtest run against dataset version N
   reproducible months later even after the dataset has since been
   incrementally updated or revised multiple times.

Why yearly partitions rather than one monolithic file per symbol/timeframe
(which is what the existing `data.sources.parquet_source.ParquetDataSource`
already uses, and which pyarrow's row-group statistics alone make
efficient to range-query): a single ever-growing file makes every
incremental update rewrite the ENTIRE history to add one more day of bars.
Yearly partitions bound that cost to "rewrite at most the current year"
for the common case, while still being coarse enough not to create
thousands of tiny files for a single-symbol research platform. Predicate-
friendly range reads fall out of this for free: `read_range` only opens
the year-partitions that overlap the requested window.

Concurrency: writing a content directory is safe under concurrent writers
by construction (two writers computing the same content race harmlessly to
the same immutable destination; two writers computing different content
for the same year never collide, since each gets its own content-address).
The only remaining race is `CURRENT` itself, which two concurrent writers
could set to either result -- both remain fully valid and reconstructable,
so the worst case is "which one is casually current" being nondeterministic
under a true write race, never corruption. `historical.locking.DatasetLock`
provides real mutual exclusion for the normal single-writer-at-a-time case;
see `historical/update_pipeline.py`.
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from quant_platform.core.exceptions import PathSecurityError, SnapshotError
from quant_platform.core.json import canonical_json_bytes, parse_json_strict
from quant_platform.core.types import Timeframe
from quant_platform.data.interfaces import DataSource
from quant_platform.historical.models import (
    schema_fingerprint,
    sha256_file,
    validate_historical_schema,
)
from quant_platform.historical.timezones import require_utc

logger = logging.getLogger(__name__)

_SUCCESS_MARKER = "_SUCCESS"
_METADATA_FILE = "metadata.json"
_DATA_FILE = "data.parquet"
_CURRENT_POINTER_FILE = "CURRENT"
_CONTENT_DIR_NAME = "content"
CompressionCodec = Literal["snappy", "gzip", "brotli", "lz4", "zstd"]
DEFAULT_COMPRESSION: CompressionCodec = "zstd"
"""zstd gives a materially better compression ratio than snappy on
repetitive OHLCV-shaped float/int columns at a modest, acceptable CPU cost
for a batch (not per-request-latency-sensitive) storage write -- see
Section R for measured write/read throughput. Configurable per
`CanonicalStore` instance for callers with different tradeoffs."""


@dataclass(frozen=True, slots=True)
class PartitionMetadata:
    symbol: str
    timeframe: Timeframe
    year: int
    row_count: int
    schema_fingerprint: str
    content_checksum: str
    min_open_time: pd.Timestamp | None
    max_open_time: pd.Timestamp | None
    written_at: pd.Timestamp
    compression: str

    @property
    def content_id(self) -> str:
        """The content-address this partition is stored under -- an alias
        for `content_checksum`, which already uniquely identifies the
        content and doubles as the directory name under `content/`."""
        return self.content_checksum

    def to_json_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "year": self.year,
            "row_count": self.row_count,
            "schema_fingerprint": self.schema_fingerprint,
            "content_checksum": self.content_checksum,
            "min_open_time": self.min_open_time.isoformat() if self.min_open_time is not None else None,
            "max_open_time": self.max_open_time.isoformat() if self.max_open_time is not None else None,
            "written_at": self.written_at.isoformat(),
            "compression": self.compression,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PartitionMetadata:
        def _ts(value: object) -> pd.Timestamp | None:
            return None if value is None else pd.Timestamp(str(value))

        return cls(
            symbol=str(raw["symbol"]),
            timeframe=Timeframe(raw["timeframe"]),
            year=int(str(raw["year"])),
            row_count=int(str(raw["row_count"])),
            schema_fingerprint=str(raw["schema_fingerprint"]),
            content_checksum=str(raw["content_checksum"]),
            min_open_time=_ts(raw["min_open_time"]),
            max_open_time=_ts(raw["max_open_time"]),
            written_at=pd.Timestamp(str(raw["written_at"])),
            compression=str(raw["compression"]),
        )


class CanonicalStore:
    def __init__(self, storage_root: Path | str, *, compression: CompressionCodec = DEFAULT_COMPRESSION) -> None:
        self._root = Path(storage_root).resolve()
        self._compression = compression

    @property
    def root(self) -> Path:
        return self._root

    def partition_dir(self, *, symbol: str, timeframe: Timeframe, year: int) -> Path:
        """The YEAR directory (containing `content/` and `CURRENT`) -- not
        a single partition's data location anymore. Use
        `content_dir(..., content_id=...)` for a specific immutable blob's
        location."""
        safe_symbol = DataSource.sanitize_identifier(symbol, field_name="symbol")
        path = self._root / "canonical" / f"symbol={safe_symbol}" / f"timeframe={timeframe.value}" / f"year={year:04d}"
        self._assert_within_root(path)
        return path

    def content_dir(self, *, symbol: str, timeframe: Timeframe, year: int, content_id: str) -> Path:
        """The immutable, content-addressed location of one specific
        partition blob. Never written to more than once; never deleted."""
        safe_content_id = DataSource.sanitize_identifier(content_id, field_name="content_id")
        path = self.partition_dir(symbol=symbol, timeframe=timeframe, year=year) / _CONTENT_DIR_NAME / safe_content_id
        self._assert_within_root(path)
        return path

    def _current_pointer_path(self, *, symbol: str, timeframe: Timeframe, year: int) -> Path:
        return self.partition_dir(symbol=symbol, timeframe=timeframe, year=year) / _CURRENT_POINTER_FILE

    def _assert_within_root(self, path: Path) -> None:
        resolved = path.resolve()
        if self._root not in resolved.parents and resolved != self._root:
            raise PathSecurityError(
                f"Resolved path {resolved} escapes storage root {self._root}",
                context={"path": str(path), "root": str(self._root)},
            )

    def write_partition(self, df: pd.DataFrame, *, symbol: str, timeframe: Timeframe, year: int) -> PartitionMetadata:
        """Write `df` as an immutable, content-addressed partition blob for
        `year` and atomically repoint `CURRENT` at it. If a content
        directory with the exact same bytes already exists (identical
        content re-submitted, e.g. an idempotent re-run), nothing new is
        written -- the existing blob is reused (content-addressed dedup).
        `df` must contain only bars whose `open_time` falls within that UTC
        calendar year -- a caller contract, not something this function
        silently filters."""
        started_at = time.perf_counter()
        validate_historical_schema(df, context="CanonicalStore.write_partition")
        if len(df) > 0:
            year_start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
            year_end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
            out_of_range = (df["open_time"] < year_start) | (df["open_time"] >= year_end)
            if out_of_range.any():
                raise ValueError(
                    f"write_partition(year={year}): {int(out_of_range.sum())} row(s) fall outside "
                    f"[{year_start}, {year_end}); the caller must partition data by UTC year before "
                    "calling this method."
                )

        partition_dir = self.partition_dir(symbol=symbol, timeframe=timeframe, year=year)
        tmp_parent = partition_dir / ".tmp"
        tmp_parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_parent / uuid.uuid4().hex
        tmp_dir.mkdir(parents=False, exist_ok=False)
        try:
            data_path = tmp_dir / _DATA_FILE
            df.to_parquet(data_path, index=False, compression=self._compression)
            checksum = sha256_file(data_path)
            min_open_time = pd.Timestamp(df["open_time"].min()) if len(df) else None
            max_open_time = pd.Timestamp(df["open_time"].max()) if len(df) else None
            metadata = PartitionMetadata(
                symbol=symbol, timeframe=timeframe, year=year, row_count=len(df),
                schema_fingerprint=schema_fingerprint(df), content_checksum=checksum,
                min_open_time=min_open_time, max_open_time=max_open_time,
                written_at=pd.Timestamp.now(tz="UTC"), compression=self._compression,
            )
            (tmp_dir / _METADATA_FILE).write_bytes(canonical_json_bytes(metadata.to_json_dict()))
            (tmp_dir / _SUCCESS_MARKER).write_text("")
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        final_content_dir = self.content_dir(symbol=symbol, timeframe=timeframe, year=year, content_id=checksum)
        if (final_content_dir / _SUCCESS_MARKER).is_file():
            # Content-addressed dedup: this exact content is already stored
            # immutably (e.g. an idempotent re-run, or unchanged data being
            # rewritten as part of a multi-year update). Discard the
            # redundant write rather than touching the existing blob.
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info(
                "Canonical content already stored (dedup): symbol=%s timeframe=%s year=%d checksum=%s",
                symbol, timeframe.value, year, checksum[:12],
            )
        else:
            final_content_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                Path(tmp_dir).replace(final_content_dir)
            except OSError:
                # Lost a race with a concurrent writer submitting the exact
                # same content -- both are byte-identical by definition
                # (same checksum), so the existing one is equally valid.
                shutil.rmtree(tmp_dir, ignore_errors=True)
                if not (final_content_dir / _SUCCESS_MARKER).is_file():
                    raise

        self._set_current(symbol=symbol, timeframe=timeframe, year=year, content_id=checksum)

        logger.info(
            "Canonical partition written: symbol=%s timeframe=%s year=%d rows=%d checksum=%s duration_s=%.3f",
            symbol, timeframe.value, year, len(df), checksum[:12], time.perf_counter() - started_at,
        )
        return metadata

    def _set_current(self, *, symbol: str, timeframe: Timeframe, year: int, content_id: str) -> None:
        pointer_path = self._current_pointer_path(symbol=symbol, timeframe=timeframe, year=year)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = pointer_path.parent / f".{_CURRENT_POINTER_FILE}.{uuid.uuid4().hex}.tmp"
        tmp_path.write_text(content_id)
        tmp_path.replace(pointer_path)

    def current_content_id(self, *, symbol: str, timeframe: Timeframe, year: int) -> str | None:
        """The content id `CURRENT` points at for `year`, or `None` if
        nothing has ever been written for it."""
        pointer_path = self._current_pointer_path(symbol=symbol, timeframe=timeframe, year=year)
        if not pointer_path.is_file():
            return None
        return pointer_path.read_text().strip()

    def read_partition(self, *, symbol: str, timeframe: Timeframe, year: int) -> tuple[pd.DataFrame, PartitionMetadata] | None:
        """Read whatever `CURRENT` points at for `year` -- the casual,
        non-version-pinned read. Use `read_partition_by_content_id` to read
        an exact historical blob regardless of what is current now."""
        content_id = self.current_content_id(symbol=symbol, timeframe=timeframe, year=year)
        if content_id is None:
            return None
        result = self.read_partition_by_content_id(symbol=symbol, timeframe=timeframe, year=year, content_id=content_id)
        if result is None:
            raise SnapshotError(
                f"CURRENT pointer for {symbol}/{timeframe.value}/year={year} references content "
                f"{content_id!r}, but that content directory does not exist",
                context={"symbol": symbol, "timeframe": timeframe.value, "year": year, "content_id": content_id},
            )
        return result

    def read_partition_by_content_id(
        self, *, symbol: str, timeframe: Timeframe, year: int, content_id: str
    ) -> tuple[pd.DataFrame, PartitionMetadata] | None:
        """Read one SPECIFIC immutable content-addressed partition blob
        directly, bypassing `CURRENT` entirely -- this is what lets a
        `DatasetManifest` reconstruct exactly the data it recorded, even
        long after `CURRENT` has moved on to a newer revision. Returns
        `None` if that exact content directory does not exist (never
        expected given write-once immutability, but a manifest referencing
        content that was never written -- or was somehow removed out of
        band -- must be distinguishable from a checksum/schema failure)."""
        content_dir = self.content_dir(symbol=symbol, timeframe=timeframe, year=year, content_id=content_id)
        if not content_dir.is_dir():
            return None
        return self._read_content_dir(content_dir)

    def _read_content_dir(self, content_dir: Path) -> tuple[pd.DataFrame, PartitionMetadata]:
        if not (content_dir / _SUCCESS_MARKER).is_file():
            raise SnapshotError(
                "Canonical partition is incomplete or corrupted: missing _SUCCESS marker",
                context={"path": str(content_dir)},
            )

        metadata_path = content_dir / _METADATA_FILE
        if not metadata_path.is_file():
            raise SnapshotError("Canonical partition metadata.json is missing", context={"path": str(content_dir)})
        try:
            raw_metadata = parse_json_strict(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(raw_metadata, dict):
                raise ValueError(f"expected a JSON object, got {type(raw_metadata).__name__}")
            metadata = PartitionMetadata.from_json_dict(raw_metadata)
        except (UnicodeDecodeError, KeyError, ValueError, TypeError) as exc:
            raise SnapshotError(
                f"Canonical partition metadata.json is corrupted: {exc}", context={"path": str(content_dir)}
            ) from exc

        data_path = content_dir / _DATA_FILE
        if not data_path.is_file():
            raise SnapshotError("Canonical partition data.parquet is missing", context={"path": str(content_dir)})

        actual_checksum = sha256_file(data_path)
        if actual_checksum != metadata.content_checksum:
            raise SnapshotError(
                "Canonical partition checksum mismatch: data is corrupted",
                context={
                    "path": str(content_dir), "expected": metadata.content_checksum, "actual": actual_checksum,
                },
            )

        df = pd.read_parquet(data_path)
        validate_historical_schema(df, context="CanonicalStore.read_partition")
        actual_fingerprint = schema_fingerprint(df)
        if actual_fingerprint != metadata.schema_fingerprint:
            raise SnapshotError(
                "Canonical partition schema fingerprint mismatch",
                context={
                    "path": str(content_dir), "expected": metadata.schema_fingerprint, "actual": actual_fingerprint,
                },
            )
        if len(df) != metadata.row_count:
            raise SnapshotError(
                f"Canonical partition row count mismatch: metadata says {metadata.row_count}, got {len(df)}",
                context={"path": str(content_dir)},
            )
        return df, metadata

    def list_years(self, *, symbol: str, timeframe: Timeframe) -> list[int]:
        """Years with a CURRENT partition available for casual reads.
        A year whose content was written but whose `CURRENT` pointer was
        never set (e.g. a crash between the two steps) is deliberately
        excluded here -- it is not "available" for a normal read, though
        its content remains reconstructable by exact content id if any
        manifest ever recorded it."""
        safe_symbol = DataSource.sanitize_identifier(symbol, field_name="symbol")
        dataset_dir = self._root / "canonical" / f"symbol={safe_symbol}" / f"timeframe={timeframe.value}"
        if not dataset_dir.is_dir():
            return []
        years = []
        for child in dataset_dir.iterdir():
            if not (child.is_dir() and child.name.startswith("year=")):
                continue
            if not (child / _CURRENT_POINTER_FILE).is_file():
                continue
            try:
                years.append(int(child.name.removeprefix("year=")))
            except ValueError:
                continue
        return sorted(years)

    def read_range(
        self, *, symbol: str, timeframe: Timeframe, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Read only the bars in `[start, end)` from CURRENT data, opening
        only the year-partitions that overlap the range."""
        require_utc(pd.Series([start, end]), context="CanonicalStore.read_range")
        if end <= start:
            raise ValueError(f"end ({end}) must be after start ({start})")

        relevant_years = [y for y in self.list_years(symbol=symbol, timeframe=timeframe) if y >= start.year and y <= end.year]
        frames: list[pd.DataFrame] = []
        for year in relevant_years:
            loaded = self.read_partition(symbol=symbol, timeframe=timeframe, year=year)
            if loaded is None:
                continue
            df, _ = loaded
            mask = (df["open_time"] >= start) & (df["open_time"] < end)
            frames.append(df.loc[mask])

        if not frames:
            from quant_platform.historical.models import RAW_HISTORICAL_COLUMNS

            return pd.DataFrame({col: pd.Series([], dtype=object) for col in RAW_HISTORICAL_COLUMNS})

        combined = pd.concat(frames, ignore_index=True)
        if not combined["open_time"].is_monotonic_increasing:
            raise SnapshotError(
                "CanonicalStore.read_range: combined result across partitions is not "
                "monotonically increasing -- overlapping or misassigned partitions detected",
                context={"symbol": symbol, "timeframe": timeframe.value},
            )
        return combined


__all__ = ["DEFAULT_COMPRESSION", "CanonicalStore", "CompressionCodec", "PartitionMetadata"]
