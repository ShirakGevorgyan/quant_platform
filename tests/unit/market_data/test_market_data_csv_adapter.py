"""Unit tests for `market_data.csv_adapter` (Milestone 10, Phase 3): CSV
candle adapter column mapping, content identity, and structural
read-time failure modes."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_platform.core.exceptions import SourceAdapterError
from quant_platform.market_data.csv_adapter import (
    CsvColumnMapping,
    create_csv_column_mapping,
    read_csv_candle_adapter,
)
from quant_platform.market_data.source_manifests import RecordKind, SourceKind


def _write(dir_path: Path, name: str, text: str) -> Path:
    path = dir_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def mapping() -> CsvColumnMapping:
    return create_csv_column_mapping(
        timestamp_column="time", open_column="o", high_column="h", low_column="l", close_column="c", volume_column="v", symbol_column="sym",
    )


class TestCsvColumnMapping:
    def test_deterministic_mapping_id(self, mapping: CsvColumnMapping) -> None:
        other = create_csv_column_mapping(
            timestamp_column="time", open_column="o", high_column="h", low_column="l", close_column="c", volume_column="v", symbol_column="sym",
        )
        assert mapping.mapping_id == other.mapping_id

    def test_changed_column_changes_id(self, mapping: CsvColumnMapping) -> None:
        other = create_csv_column_mapping(timestamp_column="time", open_column="o", high_column="h", low_column="l", close_column="c", volume_column="v", symbol_column="ticker")
        assert mapping.mapping_id != other.mapping_id

    def test_duplicate_mapped_column_rejected(self) -> None:
        with pytest.raises(SourceAdapterError):
            create_csv_column_mapping(timestamp_column="time", open_column="o", high_column="o", low_column="l", close_column="c", volume_column="v")

    def test_round_trip(self, mapping: CsvColumnMapping) -> None:
        assert CsvColumnMapping.from_json_dict(mapping.to_json_dict()) == mapping


class TestReadCsvCandleAdapter:
    def test_happy_path_parses_rows(self, mapping: CsvColumnMapping, tmp_path: Path) -> None:
        path = _write(tmp_path, "good.csv", "time,o,h,l,c,v,sym\n2024-01-01T00:00:00Z,2000.5,2001.0,1999.5,2000.0,100,XAUUSD\n")
        adapter = read_csv_candle_adapter(path, column_mapping=mapping)
        assert adapter.source_kind() is SourceKind.CSV_CANDLES
        assert adapter.record_kind() is RecordKind.CANDLE
        records = list(adapter.iter_records())
        assert len(records) == 1
        assert records[0].raw_fields == {"timestamp": "2024-01-01T00:00:00Z", "open": "2000.5", "high": "2001.0", "low": "1999.5", "close": "2000.0", "volume": "100", "symbol": "XAUUSD"}

    def test_content_digest_independent_of_path(self, mapping: CsvColumnMapping, tmp_path: Path) -> None:
        content = "time,o,h,l,c,v\n2024-01-01T00:00:00Z,1,2,0.5,1.5,10\n"
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        dir_b = tmp_path / "b" / "nested"
        dir_b.mkdir(parents=True)
        mapping_no_sym = create_csv_column_mapping(timestamp_column="time", open_column="o", high_column="h", low_column="l", close_column="c", volume_column="v")
        path_a = _write(dir_a, "one.csv", content)
        path_b = _write(dir_b, "different_name.csv", content)
        adapter_a = read_csv_candle_adapter(path_a, column_mapping=mapping_no_sym)
        adapter_b = read_csv_candle_adapter(path_b, column_mapping=mapping_no_sym)
        assert adapter_a.content_digest() == adapter_b.content_digest()

    def test_missing_required_column_rejected(self, mapping: CsvColumnMapping, tmp_path: Path) -> None:
        path = _write(tmp_path, "missing.csv", "time,o,h,l,c,sym\n2024-01-01T00:00:00Z,1,2,0.5,1.5,XAUUSD\n")
        with pytest.raises(SourceAdapterError):
            read_csv_candle_adapter(path, column_mapping=mapping)

    def test_extra_column_rejected_under_strict(self, mapping: CsvColumnMapping, tmp_path: Path) -> None:
        path = _write(tmp_path, "extra.csv", "time,o,h,l,c,v,sym,extra\n2024-01-01T00:00:00Z,1,2,0.5,1.5,10,XAUUSD,zzz\n")
        with pytest.raises(SourceAdapterError):
            read_csv_candle_adapter(path, column_mapping=mapping, strict_columns=True)

    def test_extra_column_allowed_under_lenient(self, mapping: CsvColumnMapping, tmp_path: Path) -> None:
        path = _write(tmp_path, "extra.csv", "time,o,h,l,c,v,sym,extra\n2024-01-01T00:00:00Z,1,2,0.5,1.5,10,XAUUSD,zzz\n")
        adapter = read_csv_candle_adapter(path, column_mapping=mapping, strict_columns=False)
        record = next(iter(adapter.iter_records()))
        assert record.raw_fields["extra"] == "zzz"

    def test_ragged_row_rejected(self, mapping: CsvColumnMapping, tmp_path: Path) -> None:
        path = _write(tmp_path, "ragged.csv", "time,o,h,l,c,v,sym\n2024-01-01T00:00:00Z,1,2,0.5,1.5,10\n")
        with pytest.raises(SourceAdapterError):
            read_csv_candle_adapter(path, column_mapping=mapping)

    def test_duplicate_header_column_rejected(self, mapping: CsvColumnMapping, tmp_path: Path) -> None:
        path = _write(tmp_path, "dup.csv", "time,o,h,l,c,v,v\n2024-01-01T00:00:00Z,1,2,0.5,1.5,10,10\n")
        with pytest.raises(SourceAdapterError):
            read_csv_candle_adapter(path, column_mapping=mapping)

    def test_bom_handling(self, tmp_path: Path) -> None:
        path = tmp_path / "bom.csv"
        path.write_bytes(b"\xef\xbb\xbf" + b"time,o,h,l,c,v\n2024-01-01T00:00:00Z,1,2,0.5,1.5,10\n")
        mapping_no_sym = create_csv_column_mapping(timestamp_column="time", open_column="o", high_column="h", low_column="l", close_column="c", volume_column="v")
        adapter = read_csv_candle_adapter(path, column_mapping=mapping_no_sym)
        record = next(iter(adapter.iter_records()))
        assert record.raw_fields["timestamp"] == "2024-01-01T00:00:00Z"

    def test_empty_file_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "empty.csv", "")
        mapping_no_sym = create_csv_column_mapping(timestamp_column="time", open_column="o", high_column="h", low_column="l", close_column="c", volume_column="v")
        with pytest.raises(SourceAdapterError):
            read_csv_candle_adapter(path, column_mapping=mapping_no_sym)

    def test_quoted_field_handling(self, mapping: CsvColumnMapping, tmp_path: Path) -> None:
        path = _write(tmp_path, "quoted.csv", 'time,o,h,l,c,v,sym\n"2024-01-01T00:00:00Z",1,2,0.5,1.5,10,"XAU,USD"\n')
        adapter = read_csv_candle_adapter(path, column_mapping=mapping)
        record = next(iter(adapter.iter_records()))
        assert record.raw_fields["symbol"] == "XAU,USD"

    def test_nonexistent_file_raises_source_adapter_error(self, mapping: CsvColumnMapping, tmp_path: Path) -> None:
        with pytest.raises(SourceAdapterError):
            read_csv_candle_adapter(tmp_path / "does_not_exist.csv", column_mapping=mapping)

    def test_deterministic_repeated_read(self, mapping: CsvColumnMapping, tmp_path: Path) -> None:
        path = _write(tmp_path, "good.csv", "time,o,h,l,c,v,sym\n2024-01-01T00:00:00Z,1,2,0.5,1.5,10,XAUUSD\n")
        a1 = read_csv_candle_adapter(path, column_mapping=mapping)
        a2 = read_csv_candle_adapter(path, column_mapping=mapping)
        assert a1.content_digest() == a2.content_digest()
        assert next(iter(a1.iter_records())).record_digest() == next(iter(a2.iter_records())).record_digest()
