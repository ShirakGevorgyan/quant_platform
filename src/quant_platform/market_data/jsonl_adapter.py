"""JSON Lines market-event adapter (Milestone 10, Phase 3). Every line
must decode -- via `core.json.parse_json_strict` (rejects NaN/Infinity
tokens and duplicate object keys; never `pickle`, never `eval`) -- to a
flat JSON OBJECT whose fields are drawn from a fixed, versioned allow-
list for the adapter's declared `record_kind` (`schema_for_record_kind`
below), with every field value either a JSON string or JSON `null` --
never a JSON number, bool, list, or nested object. Financial/text values
are NEVER accepted as JSON numbers: a source emitting `"open": 2000.5`
as a bare JSON number (rather than `"open": "2000.5"`) is rejected
outright, since `json.loads` would otherwise hand back a Python `float`
and there is no way to recover the source's original exact decimal text
from it -- exactly the float-parsing hazard `source_normalization.py`'s
own docstring refuses to risk, pushed back to the earliest possible
boundary. This is "no permissive arbitrary-object deserialization": a
line is either exactly one of the small number of known-safe shapes, or
the whole read fails closed.

Each line also carries an explicit `"kind"` field that must equal the
adapter's own declared `record_kind` -- a defensive, self-describing
check against silently mixing event kinds within one file (mirrors this
package's own `to_json_dict()` convention, which always tags `"kind"`).

Like `csv_adapter.py`, a schema violation on any single line (malformed
JSON, wrong top-level shape, undeclared field, non-string field value, a
missing required field, or a `kind` mismatch) fails the WHOLE read
immediately (`SourceAdapterError`) rather than skipping just that line --
each line is fully self-describing (unlike a positional CSV row), but
this adapter still treats "can this be read into a `RawSourceRecord` at
all" as a single, whole-file-scoped, adapter-level concern, symmetric
with `csv_adapter.py`. Per-value SEMANTIC validation (is this timestamp
text actually parseable, is this symbol actually mapped, ...) remains
strictly downstream, in `source_normalization.py`/`orchestration.py` --
never here (see `adapters.py`'s own docstring)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from quant_platform.core.exceptions import SourceAdapterError
from quant_platform.core.json import parse_json_strict
from quant_platform.market_data.adapters import RawSourceRecord
from quant_platform.market_data.source_manifests import RecordKind, SourceKind, compute_content_digest

__all__ = [
    "JsonlMarketEventAdapter",
    "read_jsonl_market_event_adapter",
    "schema_for_record_kind",
]

_ENVELOPE_REQUIRED: tuple[str, ...] = ("kind", "timestamp", "symbol")
_ENVELOPE_OPTIONAL: tuple[str, ...] = ("provider", "sequence", "source_event_id")

# Field names deliberately mirror each typed event's own envelope/payload
# naming (`candles.py`/`ticks.py`/`events.py`) so a downstream normalizer
# can resolve them identically regardless of which adapter produced the
# raw record -- but these are SOURCE fields (e.g. `"symbol"`, never
# `"instrument_id"`; `"timeframe"` as a raw source label, never a
# resolved `Timeframe`), resolved only later via `mappings.py`.
_RECORD_KIND_SCHEMAS: dict[RecordKind, tuple[tuple[str, ...], tuple[str, ...]]] = {
    RecordKind.CANDLE: (("open", "high", "low", "close", "timeframe"), ("volume",)),
    RecordKind.TICK: (("price",), ("volume",)),
    RecordKind.QUOTE: (("bid", "ask"), ("bid_size", "ask_size")),
    RecordKind.TRADE: (("price", "size"), ("side",)),
}


def schema_for_record_kind(record_kind: RecordKind) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Returns `(required_fields, optional_fields)` for `record_kind`.
    `required_fields` always includes the shared envelope fields
    (`"kind"`, `"timestamp"`, `"symbol"`) plus this kind's own payload
    fields; `optional_fields` likewise always includes the shared
    optional envelope fields (`"provider"`, `"sequence"`,
    `"source_event_id"`)."""
    kind_required, kind_optional = _RECORD_KIND_SCHEMAS[record_kind]
    return _ENVELOPE_REQUIRED + kind_required, _ENVELOPE_OPTIONAL + kind_optional


@dataclass(frozen=True, slots=True)
class JsonlMarketEventAdapter:
    _record_kind: RecordKind
    _source_schema_version: int
    _content_digest: str
    _byte_size: int
    _records: tuple[RawSourceRecord, ...]
    _metadata: dict[str, object]

    def source_kind(self) -> SourceKind:
        return SourceKind.JSONL_MARKET_EVENTS

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


def read_jsonl_market_event_adapter(
    path: Path,
    *,
    record_kind: RecordKind,
    source_schema_version: int = 1,
    encoding: str = "utf-8",
) -> JsonlMarketEventAdapter:
    """Reads `path` ONCE (local filesystem I/O only -- never network) and
    materializes every line into an immutable `RawSourceRecord`, exactly
    like `csv_adapter.read_csv_candle_adapter`. `content_digest()`/
    `byte_size()` are computed from the raw bytes alone -- `path` itself
    never participates in adapter identity."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SourceAdapterError(f"Could not read JSONL source {path}: {exc}") from exc

    content_digest = compute_content_digest(data)
    byte_size = len(data)

    try:
        text = data.decode(encoding)
    except UnicodeDecodeError as exc:
        raise SourceAdapterError(f"JSONL source {path} does not match declared encoding {encoding!r}: {exc}") from exc
    if text.startswith("﻿"):
        text = text[1:]

    lines = text.splitlines()
    if lines and lines[-1].strip() == "":
        lines = lines[:-1]

    required_fields, optional_fields = schema_for_record_kind(record_kind)
    allowed_fields = set(required_fields) | set(optional_fields)

    records: list[RawSourceRecord] = []
    for row_index, line in enumerate(lines):
        if line.strip() == "":
            raise SourceAdapterError(f"JSONL source {path} line {row_index} is blank -- only a single trailing blank line at end-of-file is tolerated")
        try:
            parsed = parse_json_strict(line)
        except ValueError as exc:
            raise SourceAdapterError(f"JSONL source {path} line {row_index} is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SourceAdapterError(f"JSONL source {path} line {row_index} must decode to a JSON object, got {type(parsed).__name__}")

        raw_fields: dict[str, str] = {}
        for key, value in parsed.items():
            if key not in allowed_fields:
                raise SourceAdapterError(
                    f"JSONL source {path} line {row_index} has an undeclared field {key!r} for record_kind={record_kind.value!r} "
                    f"(allowed: {sorted(allowed_fields)})"
                )
            if value is None:
                continue
            if not isinstance(value, str):
                raise SourceAdapterError(
                    f"JSONL source {path} line {row_index} field {key!r} must be a JSON string or null, got {type(value).__name__} -- "
                    "financial/text values are never accepted as JSON numbers"
                )
            raw_fields[key] = value

        missing = [f for f in required_fields if f not in raw_fields]
        if missing:
            raise SourceAdapterError(f"JSONL source {path} line {row_index} is missing required field(s): {missing}")
        if raw_fields["kind"] != record_kind.value:
            raise SourceAdapterError(
                f"JSONL source {path} line {row_index} has kind={raw_fields['kind']!r}, expected {record_kind.value!r} for this adapter"
            )

        records.append(RawSourceRecord(row_index=row_index, raw_fields=raw_fields, raw_text=line))

    metadata: dict[str, object] = {
        "source_label": path.name,
        "row_count": len(records),
        "record_kind": record_kind.value,
        "encoding": encoding,
    }
    return JsonlMarketEventAdapter(
        _record_kind=record_kind, _source_schema_version=source_schema_version, _content_digest=content_digest,
        _byte_size=byte_size, _records=tuple(records), _metadata=metadata,
    )
