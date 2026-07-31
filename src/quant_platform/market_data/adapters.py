"""Source-neutral historical adapter contract (Milestone 10, Phase 3).

An adapter's ONLY job is to expose enough information to identify a
source and iterate its raw records deterministically -- it never decides
final repository identity, never normalizes a value, never validates a
row beyond what is needed to read it at all, and never touches the
repository, provenance, or checkpoint stores. `HistoricalSourceAdapter`
is a `Protocol`, not a base class, so `csv_adapter.CsvCandleAdapter`,
`jsonl_adapter.JsonlMarketEventAdapter`, and `InMemorySourceAdapter`
(below) share no inheritance -- only this shape.

Adapter output is `RawSourceRecord`: untyped TEXT fields exactly as
read, never a prematurely-trusted `MarketEvent`. Everything downstream
of an adapter -- normalization, mapping resolution, validation, canonical
event construction, repository ingestion, checkpointing, reporting --
belongs to `orchestration.py`, never to the adapter itself."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from quant_platform.market_data.identity import compute_content_id, require_non_empty
from quant_platform.market_data.source_manifests import RecordKind, SourceKind

__all__ = [
    "RAW_SOURCE_RECORD_KIND",
    "HistoricalSourceAdapter",
    "InMemorySourceAdapter",
    "RawSourceRecord",
    "SourceRowCoordinate",
    "create_in_memory_adapter",
]

RAW_SOURCE_RECORD_KIND = "raw_source_record"


@dataclass(frozen=True, slots=True)
class RawSourceRecord:
    row_index: int
    """0-based physical position of this record within the source (data
    records only -- a CSV header row is never assigned an index)."""
    raw_fields: dict[str, str]
    """Untyped TEXT fields exactly as read -- e.g. `{"open": "2000.5"}`,
    never `{"open": Decimal("2000.5")}`. Typed parsing is
    `source_normalization.py`'s job, invoked by the orchestration layer,
    never the adapter."""
    raw_text: str
    """A deterministic textual representation of this record, used for
    `record_digest()` -- for CSV, the delimiter-joined parsed field
    values (not necessarily byte-identical to the original line's exact
    quoting/whitespace, which the `csv` module does not preserve); for
    JSON Lines, the exact original line."""

    def __post_init__(self) -> None:
        if self.row_index < 0:
            raise ValueError(f"RawSourceRecord.row_index must be >= 0, got {self.row_index}")

    def record_digest(self) -> str:
        return compute_content_id(RAW_SOURCE_RECORD_KIND, {"raw_text": self.raw_text})


@dataclass(frozen=True, slots=True)
class SourceRowCoordinate:
    """Binds a `RawSourceRecord.row_index` to the specific
    `SourceManifest` it belongs to -- used by `provenance.py`/
    `quarantine.py`, never held on `RawSourceRecord` itself (an adapter
    does not know its own `source_manifest_id` while iterating; that id
    is computed by the caller from the adapter's own `content_digest()`/
    `byte_size()` before or after iteration, not during)."""

    source_manifest_id: str
    row_index: int

    def __post_init__(self) -> None:
        require_non_empty(self.source_manifest_id, field_name="SourceRowCoordinate.source_manifest_id")
        if self.row_index < 0:
            raise ValueError(f"SourceRowCoordinate.row_index must be >= 0, got {self.row_index}")

    def to_json_dict(self) -> dict[str, object]:
        return {"source_manifest_id": self.source_manifest_id, "row_index": self.row_index}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SourceRowCoordinate:
        return cls(source_manifest_id=str(raw["source_manifest_id"]), row_index=int(str(raw["row_index"])))


class HistoricalSourceAdapter(Protocol):
    def source_kind(self) -> SourceKind: ...
    def source_schema_version(self) -> int: ...
    def record_kind(self) -> RecordKind: ...
    def content_digest(self) -> str: ...
    def byte_size(self) -> int: ...
    def describe(self) -> dict[str, object]: ...
    def iter_records(self) -> Iterator[RawSourceRecord]: ...


@dataclass(frozen=True, slots=True)
class InMemorySourceAdapter:
    """Deterministic in-memory adapter for tests -- implements the same
    protocol as `CsvCandleAdapter`/`JsonlMarketEventAdapter` via
    structural typing, with zero file I/O. `create_in_memory_adapter`
    is the supported construction path (mirrors every other `create_*`
    factory in this repository)."""

    _source_kind: SourceKind
    _source_schema_version: int
    _record_kind: RecordKind
    _records: tuple[RawSourceRecord, ...]
    _content_digest: str
    _byte_size: int
    _metadata: dict[str, object]

    def source_kind(self) -> SourceKind:
        return self._source_kind

    def source_schema_version(self) -> int:
        return self._source_schema_version

    def record_kind(self) -> RecordKind:
        return self._record_kind

    def content_digest(self) -> str:
        return self._content_digest

    def byte_size(self) -> int:
        return self._byte_size

    def describe(self) -> dict[str, object]:
        return dict(self._metadata)

    def iter_records(self) -> Iterator[RawSourceRecord]:
        return iter(self._records)


def create_in_memory_adapter(
    *, source_schema_version: int, record_kind: RecordKind, rows: list[dict[str, str]], metadata: dict[str, object] | None = None,
) -> InMemorySourceAdapter:
    records = tuple(
        RawSourceRecord(row_index=i, raw_fields=dict(row), raw_text=",".join(f"{k}={v}" for k, v in sorted(row.items())))
        for i, row in enumerate(rows)
    )
    content_digest = compute_content_id("in_memory_source_content", {"records": [r.raw_text for r in records]})
    byte_size = sum(len(r.raw_text.encode("utf-8")) for r in records)
    return InMemorySourceAdapter(
        _source_kind=SourceKind.IN_MEMORY, _source_schema_version=source_schema_version, _record_kind=record_kind, _records=records,
        _content_digest=content_digest, _byte_size=byte_size, _metadata=(metadata or {}),
    )
