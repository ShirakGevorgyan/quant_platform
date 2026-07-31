"""Unit tests for `market_data.source_normalization` and
`market_data.adapters` (Milestone 10, Phase 3): timestamp/Decimal
normalization determinism and fail-closed rules, plus the source-neutral
adapter contract (`RawSourceRecord`, `SourceRowCoordinate`,
`InMemorySourceAdapter`)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from quant_platform.core.exceptions import HistoricalIngestionError, TimezoneError
from quant_platform.historical.timezones import FixedOffsetTimezone, NamedZoneTimezone
from quant_platform.market_data.adapters import (
    RawSourceRecord,
    SourceRowCoordinate,
    create_in_memory_adapter,
)
from quant_platform.market_data.source_manifests import RecordKind, SourceKind
from quant_platform.market_data.source_normalization import (
    TimestampParsingPolicy,
    normalize_signed_zero,
    normalize_volume,
    parse_source_decimal,
    parse_source_timestamp,
)


class TestTimestampParsingPolicy:
    def test_empty_formats_rejected(self) -> None:
        with pytest.raises(HistoricalIngestionError):
            TimestampParsingPolicy(formats=(), source_timezone=None)


class TestParseSourceTimestamp:
    def test_explicit_offset_format_converts_to_utc(self) -> None:
        policy = TimestampParsingPolicy(formats=("%Y-%m-%dT%H:%M:%S%z",), source_timezone=None)
        result = parse_source_timestamp("2024-01-01T00:00:00+02:00", policy=policy)
        assert result == datetime(2023, 12, 31, 22, 0, tzinfo=timezone.utc)
        assert result.tzinfo is not None

    def test_naive_timestamp_without_policy_fails_closed(self) -> None:
        policy = TimestampParsingPolicy(formats=("%Y-%m-%d %H:%M:%S",), source_timezone=None)
        with pytest.raises(TimezoneError):
            parse_source_timestamp("2024-01-01 00:00:00", policy=policy)

    def test_naive_timestamp_with_fixed_offset_policy_localizes(self) -> None:
        policy = TimestampParsingPolicy(formats=("%Y-%m-%d %H:%M:%S",), source_timezone=FixedOffsetTimezone(offset=timedelta(hours=2), name="broker+2"))
        result = parse_source_timestamp("2024-01-01 02:00:00", policy=policy)
        assert result == datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

    def test_malformed_timestamp_rejected(self) -> None:
        policy = TimestampParsingPolicy(formats=("%Y-%m-%d",), source_timezone=None)
        with pytest.raises(HistoricalIngestionError):
            parse_source_timestamp("not-a-date", policy=policy)

    def test_empty_timestamp_rejected(self) -> None:
        policy = TimestampParsingPolicy(formats=("%Y-%m-%d",), source_timezone=None)
        with pytest.raises(HistoricalIngestionError):
            parse_source_timestamp("", policy=policy)

    def test_dst_ambiguous_local_time_fails_closed(self) -> None:
        # 2026-10-25 02:30 is inside the Europe/Berlin fall-back DST transition.
        policy = TimestampParsingPolicy(formats=("%Y-%m-%d %H:%M:%S",), source_timezone=NamedZoneTimezone(key="Europe/Berlin"))
        with pytest.raises(TimezoneError):
            parse_source_timestamp("2026-10-25 02:30:00", policy=policy)

    def test_first_matching_format_wins(self) -> None:
        policy = TimestampParsingPolicy(formats=("%Y-%m-%d", "%Y/%m/%d"), source_timezone=NamedZoneTimezone(key="UTC"))
        result = parse_source_timestamp("2024-01-01", policy=policy)
        assert result == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_no_locale_dependent_auto_parsing(self) -> None:
        # A format string not present in `formats` must never silently parse.
        policy = TimestampParsingPolicy(formats=("%Y-%m-%d",), source_timezone=None)
        with pytest.raises(HistoricalIngestionError):
            parse_source_timestamp("01/02/2024", policy=policy)


class TestParseSourceDecimal:
    def test_never_parses_through_float(self) -> None:
        # 0.1 is not exactly representable in binary float; a Decimal
        # parse from the exact source text must preserve it exactly.
        value = parse_source_decimal("0.1", field_name="price")
        assert value == Decimal("0.1")
        assert str(value) == "0.1"

    def test_rejects_nan(self) -> None:
        with pytest.raises(HistoricalIngestionError):
            parse_source_decimal("nan", field_name="price")

    def test_rejects_infinity(self) -> None:
        with pytest.raises(HistoricalIngestionError):
            parse_source_decimal("inf", field_name="price")

    def test_rejects_comma_thousands_separator(self) -> None:
        with pytest.raises(HistoricalIngestionError):
            parse_source_decimal("2,000.5", field_name="price")

    def test_rejects_malformed_text(self) -> None:
        with pytest.raises(HistoricalIngestionError):
            parse_source_decimal("not_a_number", field_name="price")

    def test_normalizes_signed_zero(self) -> None:
        value = parse_source_decimal("-0.00", field_name="price")
        assert value == 0
        assert str(value) == "0"

    def test_strips_surrounding_whitespace(self) -> None:
        assert parse_source_decimal("  2000.5  ", field_name="price") == Decimal("2000.5")


class TestNormalizeSignedZero:
    def test_negative_zero_becomes_positive_zero(self) -> None:
        assert str(normalize_signed_zero(Decimal("-0"))) == "0"

    def test_nonzero_values_pass_through_unchanged(self) -> None:
        assert normalize_signed_zero(Decimal("5.5")) == Decimal("5.5")


class TestNormalizeVolume:
    def test_negative_volume_rejected(self) -> None:
        with pytest.raises(HistoricalIngestionError):
            normalize_volume("-5", field_name="volume")

    def test_unit_scale_applied(self) -> None:
        assert normalize_volume("10", unit_scale=Decimal(100), field_name="volume") == Decimal("1000")

    def test_zero_volume_allowed(self) -> None:
        assert normalize_volume("0", field_name="volume") == Decimal("0")


class TestRawSourceRecord:
    def test_negative_row_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="row_index"):
            RawSourceRecord(row_index=-1, raw_fields={}, raw_text="")

    def test_record_digest_is_deterministic(self) -> None:
        r1 = RawSourceRecord(row_index=0, raw_fields={"a": "1"}, raw_text="a=1")
        r2 = RawSourceRecord(row_index=0, raw_fields={"a": "1"}, raw_text="a=1")
        assert r1.record_digest() == r2.record_digest()

    def test_record_digest_depends_only_on_raw_text(self) -> None:
        r1 = RawSourceRecord(row_index=0, raw_fields={"a": "1"}, raw_text="same")
        r2 = RawSourceRecord(row_index=99, raw_fields={"different": "fields"}, raw_text="same")
        assert r1.record_digest() == r2.record_digest()

    def test_different_raw_text_different_digest(self) -> None:
        r1 = RawSourceRecord(row_index=0, raw_fields={}, raw_text="a")
        r2 = RawSourceRecord(row_index=0, raw_fields={}, raw_text="b")
        assert r1.record_digest() != r2.record_digest()


class TestSourceRowCoordinate:
    def test_empty_source_manifest_id_rejected(self) -> None:
        with pytest.raises(Exception, match="source_manifest_id"):
            SourceRowCoordinate(source_manifest_id="", row_index=0)

    def test_negative_row_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="row_index"):
            SourceRowCoordinate(source_manifest_id="a" * 64, row_index=-1)

    def test_round_trip(self) -> None:
        coord = SourceRowCoordinate(source_manifest_id="a" * 64, row_index=5)
        assert SourceRowCoordinate.from_json_dict(coord.to_json_dict()) == coord


class TestInMemorySourceAdapter:
    def test_deterministic_content_digest(self) -> None:
        rows = [{"timestamp": "2024-01-01", "price": "1"}]
        a1 = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.TICK, rows=rows)
        a2 = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.TICK, rows=rows)
        assert a1.content_digest() == a2.content_digest()

    def test_changed_row_changes_digest(self) -> None:
        a1 = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.TICK, rows=[{"price": "1"}])
        a2 = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.TICK, rows=[{"price": "2"}])
        assert a1.content_digest() != a2.content_digest()

    def test_iter_records_assigns_sequential_row_indices(self) -> None:
        adapter = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.TICK, rows=[{"a": "1"}, {"a": "2"}, {"a": "3"}])
        indices = [r.row_index for r in adapter.iter_records()]
        assert indices == [0, 1, 2]

    def test_source_kind_is_in_memory(self) -> None:
        adapter = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.CANDLE, rows=[])
        assert adapter.source_kind() is SourceKind.IN_MEMORY

    def test_describe_returns_provided_metadata(self) -> None:
        adapter = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.TICK, rows=[], metadata={"note": "fixture"})
        assert adapter.describe() == {"note": "fixture"}

    def test_iter_records_repeatable(self) -> None:
        adapter = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.TICK, rows=[{"a": "1"}])
        first_pass = [r.record_digest() for r in adapter.iter_records()]
        second_pass = [r.record_digest() for r in adapter.iter_records()]
        assert first_pass == second_pass
