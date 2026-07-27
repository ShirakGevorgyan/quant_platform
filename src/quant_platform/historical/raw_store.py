"""Immutable raw snapshot store.

Every historical-data extraction is persisted here EXACTLY as received
(post schema/dtype normalization, pre validation/repair) before any further
processing happens. A raw snapshot, once written, is never mutated or
overwritten -- it is the platform's permanent record of "what the source
actually said at extraction time," which is what makes every later stage
(validation, repair, canonical storage, resampling) fully re-derivable and
auditable: if a downstream bug is ever found, the fix can be re-run from
the untouched raw snapshots rather than from data that may already have
been silently altered by the very code being fixed.

Layout (adapted to this repository's existing convention of a configurable
storage root rather than a hardcoded path -- see `config.schemas` for how
it is configured):

    <storage_root>/raw/source=<source>/broker=<broker>/symbol=<symbol>/
        timeframe=<timeframe>/snapshot=<snapshot_id>/
            data.parquet     -- the raw bars, `historical.models.RAW_HISTORICAL_COLUMNS`
            metadata.json    -- `SnapshotMetadata`, human-inspectable
            _SUCCESS         -- empty marker written LAST; its absence means
                                 the snapshot is incomplete/corrupted and
                                 must never be trusted (see `read_snapshot`)

A snapshot's three files are written to a temporary sibling directory and
promoted into place with a single `os.replace()` directory rename -- the
one filesystem operation that is atomic as a whole, so no external reader
can ever observe a partially-written snapshot (missing `_SUCCESS` is
therefore a reliable, not just probable, signal of an interrupted write or
later corruption, not a race condition to work around).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True, slots=True)
class SnapshotMetadata:
    """Everything needed to audit, reproduce, or invalidate a raw snapshot
    without touching `data.parquet` itself."""

    snapshot_id: str
    source_name: str
    source_version: str
    broker: str
    symbol: str
    source_symbol: str
    timeframe: Timeframe
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    server_timezone_repr: str
    extracted_at: pd.Timestamp
    row_count: int
    schema_fingerprint: str
    content_checksum: str
    min_open_time: pd.Timestamp | None
    max_open_time: pd.Timestamp | None
    is_complete: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "source_name": self.source_name,
            "source_version": self.source_version,
            "broker": self.broker,
            "symbol": self.symbol,
            "source_symbol": self.source_symbol,
            "timeframe": self.timeframe.value,
            "requested_start": self.requested_start.isoformat(),
            "requested_end": self.requested_end.isoformat(),
            "server_timezone_repr": self.server_timezone_repr,
            "extracted_at": self.extracted_at.isoformat(),
            "row_count": self.row_count,
            "schema_fingerprint": self.schema_fingerprint,
            "content_checksum": self.content_checksum,
            "min_open_time": self.min_open_time.isoformat() if self.min_open_time is not None else None,
            "max_open_time": self.max_open_time.isoformat() if self.max_open_time is not None else None,
            "is_complete": self.is_complete,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SnapshotMetadata:
        def _ts(value: object) -> pd.Timestamp | None:
            return None if value is None else pd.Timestamp(str(value))

        return cls(
            snapshot_id=str(raw["snapshot_id"]),
            source_name=str(raw["source_name"]),
            source_version=str(raw["source_version"]),
            broker=str(raw["broker"]),
            symbol=str(raw["symbol"]),
            source_symbol=str(raw["source_symbol"]),
            timeframe=Timeframe(raw["timeframe"]),
            requested_start=pd.Timestamp(str(raw["requested_start"])),
            requested_end=pd.Timestamp(str(raw["requested_end"])),
            server_timezone_repr=str(raw["server_timezone_repr"]),
            extracted_at=pd.Timestamp(str(raw["extracted_at"])),
            row_count=int(str(raw["row_count"])),
            schema_fingerprint=str(raw["schema_fingerprint"]),
            content_checksum=str(raw["content_checksum"]),
            min_open_time=_ts(raw["min_open_time"]),
            max_open_time=_ts(raw["max_open_time"]),
            is_complete=bool(raw["is_complete"]),
        )


class RawSnapshotStore:
    """Writes and reads immutable raw snapshots under `storage_root`."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def compute_snapshot_id(
        self,
        *,
        source_name: str,
        broker: str,
        symbol: str,
        timeframe: Timeframe,
        requested_start: pd.Timestamp,
        requested_end: pd.Timestamp,
        extracted_at: pd.Timestamp,
    ) -> str:
        """Deterministic given its inputs, and collision-safe across
        distinct ingestion runs of the identical logical range because
        `extracted_at` (which necessarily differs between any two separate
        extraction calls) is part of both the readable prefix and the hash
        -- two runs of the same [start, end) range can never produce the
        same snapshot id, so neither can ever be mistaken for or silently
        overwrite the other."""
        start_compact = requested_start.tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")
        end_compact = requested_end.tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")
        extracted_compact = extracted_at.tz_convert("UTC").strftime("%Y%m%dT%H%M%S%fZ")
        signature = (
            f"{source_name}|{broker}|{symbol}|{timeframe.value}|"
            f"{requested_start.isoformat()}|{requested_end.isoformat()}|{extracted_at.isoformat()}"
        )
        short_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:8]
        return f"{start_compact}_{end_compact}__{extracted_compact}_{short_hash}"

    def dataset_dir(self, *, source_name: str, broker: str, symbol: str, timeframe: Timeframe) -> Path:
        safe_source = DataSource.sanitize_identifier(source_name, field_name="source_name")
        safe_broker = DataSource.sanitize_identifier(broker, field_name="broker")
        safe_symbol = DataSource.sanitize_identifier(symbol, field_name="symbol")
        path = (
            self._root / "raw" / f"source={safe_source}" / f"broker={safe_broker}"
            / f"symbol={safe_symbol}" / f"timeframe={timeframe.value}"
        )
        self._assert_within_root(path)
        return path

    def _assert_within_root(self, path: Path) -> None:
        resolved = path.resolve()
        if self._root not in resolved.parents and resolved != self._root:
            raise PathSecurityError(
                f"Resolved path {resolved} escapes storage root {self._root}",
                context={"path": str(path), "root": str(self._root)},
            )

    def write_snapshot(
        self,
        df: pd.DataFrame,
        *,
        source_name: str,
        source_version: str,
        broker: str,
        symbol: str,
        source_symbol: str,
        timeframe: Timeframe,
        requested_start: pd.Timestamp,
        requested_end: pd.Timestamp,
        server_timezone_repr: str,
        extracted_at: pd.Timestamp,
        is_complete: bool,
    ) -> SnapshotMetadata:
        started_at = time.perf_counter()
        validate_historical_schema(df, context="RawSnapshotStore.write_snapshot")
        require_utc(pd.Series([requested_start, requested_end, extracted_at]), context="RawSnapshotStore.write_snapshot")

        snapshot_id = self.compute_snapshot_id(
            source_name=source_name, broker=broker, symbol=symbol, timeframe=timeframe,
            requested_start=requested_start, requested_end=requested_end, extracted_at=extracted_at,
        )
        dataset_dir = self.dataset_dir(source_name=source_name, broker=broker, symbol=symbol, timeframe=timeframe)
        final_dir = dataset_dir / f"snapshot={snapshot_id}"
        self._assert_within_root(final_dir)

        if final_dir.exists():
            if (final_dir / _SUCCESS_MARKER).exists():
                raise SnapshotError(
                    "Refusing to overwrite a completed, immutable raw snapshot",
                    context={"snapshot_id": snapshot_id, "path": str(final_dir)},
                )
            logger.warning(
                "Removing incomplete leftover snapshot directory before rewrite: %s", final_dir
            )
            shutil.rmtree(final_dir)

        tmp_parent = dataset_dir / ".tmp"
        tmp_parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_parent / f"{snapshot_id}-{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=False, exist_ok=False)
        try:
            data_path = tmp_dir / _DATA_FILE
            df.to_parquet(data_path, index=False)
            checksum = sha256_file(data_path)

            min_open_time = pd.Timestamp(df["open_time"].min()) if len(df) else None
            max_open_time = pd.Timestamp(df["open_time"].max()) if len(df) else None

            metadata = SnapshotMetadata(
                snapshot_id=snapshot_id, source_name=source_name, source_version=source_version,
                broker=broker, symbol=symbol, source_symbol=source_symbol, timeframe=timeframe,
                requested_start=requested_start, requested_end=requested_end,
                server_timezone_repr=server_timezone_repr, extracted_at=extracted_at,
                row_count=len(df), schema_fingerprint=schema_fingerprint(df),
                content_checksum=checksum, min_open_time=min_open_time, max_open_time=max_open_time,
                is_complete=is_complete,
            )
            (tmp_dir / _METADATA_FILE).write_bytes(canonical_json_bytes(metadata.to_json_dict()))
            (tmp_dir / _SUCCESS_MARKER).write_text("")
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        Path(tmp_dir).replace(final_dir)

        logger.info(
            "Raw snapshot written: id=%s source=%s broker=%s symbol=%s timeframe=%s range=[%s, %s) "
            "rows=%d complete=%s checksum=%s duration_s=%.3f",
            snapshot_id, source_name, broker, symbol, timeframe.value, requested_start, requested_end,
            len(df), is_complete, checksum[:12], time.perf_counter() - started_at,
        )
        return metadata

    def read_snapshot(self, snapshot_dir: Path) -> tuple[pd.DataFrame, SnapshotMetadata]:
        self._assert_within_root(snapshot_dir)

        if not (snapshot_dir / _SUCCESS_MARKER).is_file():
            raise SnapshotError(
                "Snapshot is incomplete or corrupted: missing _SUCCESS completion marker. "
                "This snapshot must never be trusted or used as input to any later stage.",
                context={"path": str(snapshot_dir)},
            )

        metadata_path = snapshot_dir / _METADATA_FILE
        if not metadata_path.is_file():
            raise SnapshotError("Snapshot metadata.json is missing", context={"path": str(snapshot_dir)})
        try:
            raw_metadata = parse_json_strict(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(raw_metadata, dict):
                raise ValueError(f"expected a JSON object, got {type(raw_metadata).__name__}")
            metadata = SnapshotMetadata.from_json_dict(raw_metadata)
        except (UnicodeDecodeError, KeyError, ValueError, TypeError) as exc:
            raise SnapshotError(
                f"Snapshot metadata.json is corrupted (invalid JSON or wrong structure): {exc}",
                context={"path": str(snapshot_dir)},
            ) from exc

        data_path = snapshot_dir / _DATA_FILE
        if not data_path.is_file():
            raise SnapshotError("Snapshot data.parquet is missing", context={"path": str(snapshot_dir)})

        actual_checksum = sha256_file(data_path)
        if actual_checksum != metadata.content_checksum:
            raise SnapshotError(
                "Snapshot content checksum mismatch: data.parquet does not match the checksum "
                "recorded at write time. The raw snapshot is corrupted and must not be used.",
                context={
                    "path": str(snapshot_dir),
                    "expected_checksum": metadata.content_checksum,
                    "actual_checksum": actual_checksum,
                },
            )

        df = pd.read_parquet(data_path)
        validate_historical_schema(df, context="RawSnapshotStore.read_snapshot")

        actual_fingerprint = schema_fingerprint(df)
        if actual_fingerprint != metadata.schema_fingerprint:
            raise SnapshotError(
                "Snapshot schema fingerprint mismatch: the persisted data's schema has drifted "
                "from what was recorded at write time.",
                context={
                    "path": str(snapshot_dir),
                    "expected_fingerprint": metadata.schema_fingerprint,
                    "actual_fingerprint": actual_fingerprint,
                },
            )
        if len(df) != metadata.row_count:
            raise SnapshotError(
                f"Snapshot row count mismatch: metadata says {metadata.row_count}, "
                f"data.parquet has {len(df)} rows.",
                context={"path": str(snapshot_dir)},
            )

        return df, metadata

    def list_snapshots(self, *, source_name: str, broker: str, symbol: str, timeframe: Timeframe) -> list[Path]:
        dataset_dir = self.dataset_dir(source_name=source_name, broker=broker, symbol=symbol, timeframe=timeframe)
        if not dataset_dir.is_dir():
            return []
        return sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("snapshot="))


__all__ = ["RawSnapshotStore", "SnapshotMetadata"]
