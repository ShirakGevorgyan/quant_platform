"""Immutable, self-verifying source manifests (Milestone 10, Phase 3).

A `SourceManifest` binds a CONTENT DIGEST -- never a mutable filesystem
path -- as the one true anchor of "which source". Two manifests
constructed from byte-identical source content, the same mapping specs,
and the same declared metadata always produce the IDENTICAL
`source_manifest_id`, regardless of which directory the file happened to
live in when read; changing any of that content (the bytes themselves,
the instrument/timeframe mapping, the timezone policy, the column
mapping) always changes it. `creation_time` (a caller-supplied
OPERATIONAL label, exactly like every other "recorded_time"/
"creation_time" field elsewhere in this repository) and `row_count` (a
DERIVED, purely observational count -- 100% determined by
`content_digest` already, so it adds no new information to identity) are
both excluded from the identity payload.

`source_label` (the declared logical name, e.g. `"XAUUSD_M1_2024.csv"`)
IS part of identity -- unlike a raw filesystem path (never stored on this
object at all), a caller-declared label is a semantic assertion ("this
is the file I intend it to be"), not an incidental operational detail."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from quant_platform.core.exceptions import SourceManifestError
from quant_platform.core.json import sha256_hex_bytes
from quant_platform.core.types import Timeframe
from quant_platform.historical.timezones import FixedOffsetTimezone, NamedZoneTimezone
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)
from quant_platform.market_data.source_normalization import TimestampParsingPolicy

__all__ = [
    "SOURCE_MANIFEST_KIND",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "RecordKind",
    "SourceKind",
    "SourceManifest",
    "compute_content_digest",
    "compute_timestamp_policy_id",
    "create_source_manifest",
]

SOURCE_MANIFEST_KIND = "source_manifest"
SOURCE_MANIFEST_SCHEMA_VERSION = 1


class SourceKind(Enum):
    CSV_CANDLES = "csv_candles"
    JSONL_MARKET_EVENTS = "jsonl_market_events"
    IN_MEMORY = "in_memory"


class RecordKind(Enum):
    CANDLE = "candle"
    TICK = "tick"
    QUOTE = "quote"
    TRADE = "trade"


def compute_content_digest(data: bytes) -> str:
    return sha256_hex_bytes(data)


def _source_timezone_to_json(tz: FixedOffsetTimezone | NamedZoneTimezone | None) -> dict[str, object] | None:
    if tz is None:
        return None
    if isinstance(tz, FixedOffsetTimezone):
        return {"kind": "fixed_offset", "offset_seconds": str(Decimal(tz.offset.total_seconds())), "name": tz.name}
    return {"kind": "named_zone", "key": tz.key}


def compute_timestamp_policy_id(policy: TimestampParsingPolicy) -> str:
    payload: dict[str, object] = {"formats": list(policy.formats), "source_timezone": _source_timezone_to_json(policy.source_timezone)}
    return compute_content_id("timestamp_parsing_policy", payload)


@dataclass(frozen=True, slots=True)
class SourceManifest:
    source_manifest_id: str
    source_name: str
    source_kind: SourceKind
    source_schema_version: int
    record_kind: RecordKind
    source_label: str
    content_digest: str
    byte_size: int
    encoding: str
    delimiter: str | None
    header_mapping_id: str | None
    instrument_mapping_id: str
    timeframe_mapping_id: str | None
    timezone_policy_id: str
    unit_normalization_version: int
    expected_timeframe: Timeframe | None
    expected_start: datetime | None
    expected_end: datetime | None
    creation_time: datetime
    row_count: int | None

    def __post_init__(self) -> None:
        require_non_empty(self.source_name, field_name="SourceManifest.source_name")
        require_non_empty(self.source_label, field_name="SourceManifest.source_label")
        require_non_empty(self.encoding, field_name="SourceManifest.encoding")
        if self.source_schema_version < 1:
            raise SourceManifestError(f"SourceManifest.source_schema_version must be >= 1, got {self.source_schema_version}")
        if self.byte_size < 0:
            raise SourceManifestError(f"SourceManifest.byte_size must be >= 0, got {self.byte_size}")
        if self.unit_normalization_version < 1:
            raise SourceManifestError(f"SourceManifest.unit_normalization_version must be >= 1, got {self.unit_normalization_version}")
        if self.row_count is not None and self.row_count < 0:
            raise SourceManifestError(f"SourceManifest.row_count must be >= 0 or None, got {self.row_count}")
        require_tz_aware(self.creation_time, field_name="SourceManifest.creation_time")
        if self.expected_start is not None:
            require_tz_aware(self.expected_start, field_name="SourceManifest.expected_start")
        if self.expected_end is not None:
            require_tz_aware(self.expected_end, field_name="SourceManifest.expected_end")
        if self.expected_start is not None and self.expected_end is not None and self.expected_end < self.expected_start:
            raise SourceManifestError(f"SourceManifest.expected_end ({self.expected_end}) must be >= expected_start ({self.expected_start})")
        if len(self.content_digest) != 64:
            raise SourceManifestError(f"SourceManifest.content_digest must be a 64-char sha256 hex digest, got {self.content_digest!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": SOURCE_MANIFEST_KIND, "source_manifest_id": self.source_manifest_id, "source_name": self.source_name,
            "source_kind": self.source_kind.value, "source_schema_version": self.source_schema_version, "record_kind": self.record_kind.value,
            "source_label": self.source_label, "content_digest": self.content_digest, "byte_size": self.byte_size,
            "encoding": self.encoding, "delimiter": self.delimiter, "header_mapping_id": self.header_mapping_id,
            "instrument_mapping_id": self.instrument_mapping_id, "timeframe_mapping_id": self.timeframe_mapping_id,
            "timezone_policy_id": self.timezone_policy_id, "unit_normalization_version": self.unit_normalization_version,
            "expected_timeframe": (None if self.expected_timeframe is None else self.expected_timeframe.value),
            "expected_start": (None if self.expected_start is None else serialize_timestamp(self.expected_start, field_name="expected_start")),
            "expected_end": (None if self.expected_end is None else serialize_timestamp(self.expected_end, field_name="expected_end")),
            "creation_time": serialize_timestamp(self.creation_time, field_name="creation_time"), "row_count": self.row_count,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        for key in ("source_manifest_id", "creation_time", "row_count"):
            del payload[key]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SourceManifest:
        raw_expected_timeframe = raw.get("expected_timeframe")
        raw_expected_start = raw.get("expected_start")
        raw_expected_end = raw.get("expected_end")
        raw_row_count = raw.get("row_count")
        return cls(
            source_manifest_id=str(raw["source_manifest_id"]), source_name=str(raw["source_name"]), source_kind=SourceKind(raw["source_kind"]),
            source_schema_version=int(str(raw["source_schema_version"])), record_kind=RecordKind(raw["record_kind"]),
            source_label=str(raw["source_label"]), content_digest=str(raw["content_digest"]), byte_size=int(str(raw["byte_size"])),
            encoding=str(raw["encoding"]), delimiter=(None if raw.get("delimiter") is None else str(raw["delimiter"])),
            header_mapping_id=(None if raw.get("header_mapping_id") is None else str(raw["header_mapping_id"])),
            instrument_mapping_id=str(raw["instrument_mapping_id"]),
            timeframe_mapping_id=(None if raw.get("timeframe_mapping_id") is None else str(raw["timeframe_mapping_id"])),
            timezone_policy_id=str(raw["timezone_policy_id"]), unit_normalization_version=int(str(raw["unit_normalization_version"])),
            expected_timeframe=(None if raw_expected_timeframe is None else Timeframe(raw_expected_timeframe)),
            expected_start=(None if raw_expected_start is None else deserialize_timestamp(raw_expected_start, field_name="expected_start")),
            expected_end=(None if raw_expected_end is None else deserialize_timestamp(raw_expected_end, field_name="expected_end")),
            creation_time=deserialize_timestamp(raw["creation_time"], field_name="creation_time"),
            row_count=(None if raw_row_count is None else int(str(raw_row_count))),
        )


def create_source_manifest(
    *, source_name: str, source_kind: SourceKind, source_schema_version: int, record_kind: RecordKind, source_label: str,
    content_digest: str, byte_size: int, encoding: str, instrument_mapping_id: str, timezone_policy_id: str,
    unit_normalization_version: int, creation_time: datetime, delimiter: str | None = None, header_mapping_id: str | None = None,
    timeframe_mapping_id: str | None = None, expected_timeframe: Timeframe | None = None, expected_start: datetime | None = None,
    expected_end: datetime | None = None, row_count: int | None = None,
) -> SourceManifest:
    provisional = SourceManifest(
        source_manifest_id="0" * 64, source_name=source_name, source_kind=source_kind, source_schema_version=source_schema_version,
        record_kind=record_kind, source_label=source_label, content_digest=content_digest, byte_size=byte_size, encoding=encoding,
        delimiter=delimiter, header_mapping_id=header_mapping_id, instrument_mapping_id=instrument_mapping_id,
        timeframe_mapping_id=timeframe_mapping_id, timezone_policy_id=timezone_policy_id,
        unit_normalization_version=unit_normalization_version, expected_timeframe=expected_timeframe, expected_start=expected_start,
        expected_end=expected_end, creation_time=creation_time, row_count=row_count,
    )
    source_manifest_id = compute_content_id(SOURCE_MANIFEST_KIND, provisional.to_identity_payload())
    return SourceManifest(
        source_manifest_id=source_manifest_id, source_name=source_name, source_kind=source_kind, source_schema_version=source_schema_version,
        record_kind=record_kind, source_label=source_label, content_digest=content_digest, byte_size=byte_size, encoding=encoding,
        delimiter=delimiter, header_mapping_id=header_mapping_id, instrument_mapping_id=instrument_mapping_id,
        timeframe_mapping_id=timeframe_mapping_id, timezone_policy_id=timezone_policy_id,
        unit_normalization_version=unit_normalization_version, expected_timeframe=expected_timeframe, expected_start=expected_start,
        expected_end=expected_end, creation_time=creation_time, row_count=row_count,
    )
