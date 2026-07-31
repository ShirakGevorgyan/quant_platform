"""CSV candle adapter with configurable column mapping (Milestone 10,
Phase 3). Reads a local CSV file ONCE at construction time -- the whole
file's bytes are hashed for `content_digest()` and every row is parsed
into an immutable `RawSourceRecord` up front, exactly like
`InMemorySourceAdapter` -- so `iter_records()` is repeatable and the
adapter's semantic content no longer depends on the file still existing
at the same path (only the ORIGINAL read did local filesystem I/O; no
network I/O ever). `CsvColumnMapping` is itself a small content-addressed
spec (mirrors `mappings.py`'s pattern) so that changing which CSV column
means "close" changes `header_mapping_id`, and therefore
`SourceManifest.source_manifest_id`, exactly as the specification
requires ("mapping changes must change identity").

Column PRESENCE (a whole-file schema concern -- a CSV has one header for
every row) is validated once, here, at construction: a missing declared
column, an ambiguous duplicate header name, or -- under
`strict_columns=True` -- an undeclared extra column, all raise
`SourceAdapterError` immediately, since none of those can be safely read
into a stable `raw_fields` mapping at all. A ragged data row (wrong
field COUNT relative to the header) is the same kind of structural
read failure. Everything else -- is this row's `close` a valid Decimal,
is its timestamp parseable, is its symbol mapped -- is deliberately left
untouched here; that is `source_normalization.py`/`orchestration.py`'s
job, never the adapter's (see `adapters.py`'s own docstring)."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from quant_platform.core.exceptions import SourceAdapterError
from quant_platform.market_data.adapters import RawSourceRecord
from quant_platform.market_data.identity import compute_content_id, require_non_empty
from quant_platform.market_data.source_manifests import RecordKind, SourceKind, compute_content_digest

__all__ = [
    "CSV_COLUMN_MAPPING_KIND",
    "CsvCandleAdapter",
    "CsvColumnMapping",
    "create_csv_column_mapping",
    "read_csv_candle_adapter",
]

CSV_COLUMN_MAPPING_KIND = "csv_column_mapping"

_REQUIRED_FIELDS = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass(frozen=True, slots=True)
class CsvColumnMapping:
    mapping_id: str
    timestamp_column: str
    open_column: str
    high_column: str
    low_column: str
    close_column: str
    volume_column: str
    symbol_column: str | None = None
    provider_column: str | None = None
    sequence_column: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("timestamp_column", self.timestamp_column),
            ("open_column", self.open_column),
            ("high_column", self.high_column),
            ("low_column", self.low_column),
            ("close_column", self.close_column),
            ("volume_column", self.volume_column),
        ):
            require_non_empty(value, field_name=f"CsvColumnMapping.{field_name}")
        declared = self.declared_columns()
        if len(set(declared)) != len(declared):
            raise SourceAdapterError(f"CsvColumnMapping declares the same CSV column name more than once: {declared}")

    def declared_columns(self) -> tuple[str, ...]:
        columns = [self.timestamp_column, self.open_column, self.high_column, self.low_column, self.close_column, self.volume_column]
        for optional in (self.symbol_column, self.provider_column, self.sequence_column):
            if optional is not None:
                columns.append(optional)
        return tuple(columns)

    def field_names_by_column(self) -> dict[str, str]:
        """Maps CSV header name -> canonical `RawSourceRecord.raw_fields` key."""
        mapping = {
            self.timestamp_column: "timestamp",
            self.open_column: "open",
            self.high_column: "high",
            self.low_column: "low",
            self.close_column: "close",
            self.volume_column: "volume",
        }
        if self.symbol_column is not None:
            mapping[self.symbol_column] = "symbol"
        if self.provider_column is not None:
            mapping[self.provider_column] = "provider"
        if self.sequence_column is not None:
            mapping[self.sequence_column] = "sequence"
        return mapping

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": CSV_COLUMN_MAPPING_KIND,
            "mapping_id": self.mapping_id,
            "timestamp_column": self.timestamp_column,
            "open_column": self.open_column,
            "high_column": self.high_column,
            "low_column": self.low_column,
            "close_column": self.close_column,
            "volume_column": self.volume_column,
            "symbol_column": self.symbol_column,
            "provider_column": self.provider_column,
            "sequence_column": self.sequence_column,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["mapping_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CsvColumnMapping:
        return cls(
            mapping_id=str(raw["mapping_id"]),
            timestamp_column=str(raw["timestamp_column"]),
            open_column=str(raw["open_column"]),
            high_column=str(raw["high_column"]),
            low_column=str(raw["low_column"]),
            close_column=str(raw["close_column"]),
            volume_column=str(raw["volume_column"]),
            symbol_column=(None if raw.get("symbol_column") is None else str(raw["symbol_column"])),
            provider_column=(None if raw.get("provider_column") is None else str(raw["provider_column"])),
            sequence_column=(None if raw.get("sequence_column") is None else str(raw["sequence_column"])),
        )


def create_csv_column_mapping(
    *,
    timestamp_column: str,
    open_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    volume_column: str,
    symbol_column: str | None = None,
    provider_column: str | None = None,
    sequence_column: str | None = None,
) -> CsvColumnMapping:
    provisional = CsvColumnMapping(
        mapping_id="0" * 64, timestamp_column=timestamp_column, open_column=open_column, high_column=high_column,
        low_column=low_column, close_column=close_column, volume_column=volume_column, symbol_column=symbol_column,
        provider_column=provider_column, sequence_column=sequence_column,
    )
    mapping_id = compute_content_id(CSV_COLUMN_MAPPING_KIND, provisional.to_identity_payload())
    return CsvColumnMapping(
        mapping_id=mapping_id, timestamp_column=timestamp_column, open_column=open_column, high_column=high_column,
        low_column=low_column, close_column=close_column, volume_column=volume_column, symbol_column=symbol_column,
        provider_column=provider_column, sequence_column=sequence_column,
    )


@dataclass(frozen=True, slots=True)
class CsvCandleAdapter:
    _column_mapping: CsvColumnMapping
    _source_schema_version: int
    _content_digest: str
    _byte_size: int
    _records: tuple[RawSourceRecord, ...]
    _metadata: dict[str, object]

    def source_kind(self) -> SourceKind:
        return SourceKind.CSV_CANDLES

    def source_schema_version(self) -> int:
        return self._source_schema_version

    def record_kind(self) -> RecordKind:
        return RecordKind.CANDLE

    def content_digest(self) -> str:
        return self._content_digest

    def byte_size(self) -> int:
        return self._byte_size

    def describe(self) -> dict[str, object]:
        return dict(self._metadata)

    def iter_records(self) -> Iterator[RawSourceRecord]:
        return iter(self._records)


def read_csv_candle_adapter(
    path: Path,
    *,
    column_mapping: CsvColumnMapping,
    source_schema_version: int = 1,
    encoding: str = "utf-8",
    delimiter: str = ",",
    strict_columns: bool = True,
) -> CsvCandleAdapter:
    """Reads `path` ONCE (local filesystem I/O only -- never network) and
    materializes every row into an immutable `RawSourceRecord`.
    `content_digest()`/`byte_size()` are computed from the raw bytes
    alone -- `path` itself never participates in adapter identity, only
    in the caller-supplied `SourceManifest.source_label`/`describe()`
    metadata (see module docstring: "filesystem root must not affect
    semantic identity")."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SourceAdapterError(f"Could not read CSV source {path}: {exc}") from exc

    content_digest = compute_content_digest(data)
    byte_size = len(data)

    try:
        text = data.decode(encoding)
    except UnicodeDecodeError as exc:
        raise SourceAdapterError(f"CSV source {path} does not match declared encoding {encoding!r}: {exc}") from exc
    if text.startswith("﻿"):
        text = text[1:]

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        raw_header = next(reader)
    except StopIteration:
        raise SourceAdapterError(f"CSV source {path} has no header row") from None
    header = tuple(h.strip() for h in raw_header)
    if len(set(header)) != len(header):
        raise SourceAdapterError(f"CSV header contains duplicate column name(s): {header}")

    field_by_column = column_mapping.field_names_by_column()
    missing = [c for c in column_mapping.declared_columns() if c not in header]
    if missing:
        raise SourceAdapterError(f"CSV header {header} is missing declared column(s): {missing}")
    if strict_columns:
        extra = [c for c in header if c not in field_by_column]
        if extra:
            raise SourceAdapterError(f"CSV header {header} contains column(s) not declared in the column mapping (strict_columns=True): {extra}")

    records: list[RawSourceRecord] = []
    for row_index, row in enumerate(reader):
        if len(row) != len(header):
            raise SourceAdapterError(
                f"CSV source {path} row {row_index} has {len(row)} field(s), expected {len(header)} to match the header {header}"
            )
        raw_fields: dict[str, str] = {}
        for column_name, value in zip(header, row, strict=True):
            canonical_name = field_by_column.get(column_name)
            if canonical_name is not None:
                raw_fields[canonical_name] = value
            elif not strict_columns:
                raw_fields[column_name] = value
        raw_text = delimiter.join(row)
        records.append(RawSourceRecord(row_index=row_index, raw_fields=raw_fields, raw_text=raw_text))

    metadata: dict[str, object] = {
        "source_label": path.name,
        "row_count": len(records),
        "header": list(header),
        "column_mapping_id": column_mapping.mapping_id,
        "encoding": encoding,
        "delimiter": delimiter,
        "strict_columns": strict_columns,
    }
    return CsvCandleAdapter(
        _column_mapping=column_mapping, _source_schema_version=source_schema_version, _content_digest=content_digest,
        _byte_size=byte_size, _records=tuple(records), _metadata=metadata,
    )
