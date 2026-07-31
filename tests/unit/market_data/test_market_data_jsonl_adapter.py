"""Unit tests for `market_data.jsonl_adapter` (Milestone 10, Phase 3):
JSON Lines market-event adapter strict schema enforcement per
`RecordKind`, and content identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_platform.core.exceptions import SourceAdapterError
from quant_platform.market_data.jsonl_adapter import read_jsonl_market_event_adapter, schema_for_record_kind
from quant_platform.market_data.source_manifests import RecordKind, SourceKind

_CANDLE_LINE = {
    "kind": "candle", "timestamp": "2024-01-01T00:00:00Z", "symbol": "XAUUSD", "open": "2000.5", "high": "2001.0",
    "low": "1999.5", "close": "2000.0", "volume": "100", "timeframe": "M1",
}


def _write_lines(path: Path, lines: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


class TestSchemaForRecordKind:
    def test_candle_schema(self) -> None:
        required, optional = schema_for_record_kind(RecordKind.CANDLE)
        assert set(required) == {"kind", "timestamp", "symbol", "open", "high", "low", "close", "timeframe"}
        assert set(optional) == {"provider", "sequence", "source_event_id", "volume"}

    def test_tick_schema(self) -> None:
        required, optional = schema_for_record_kind(RecordKind.TICK)
        assert set(required) == {"kind", "timestamp", "symbol", "price"}
        assert "volume" in optional

    def test_quote_schema(self) -> None:
        required, _optional = schema_for_record_kind(RecordKind.QUOTE)
        assert {"bid", "ask"} <= set(required)

    def test_trade_schema(self) -> None:
        required, optional = schema_for_record_kind(RecordKind.TRADE)
        assert {"price", "size"} <= set(required)
        assert "side" in optional


class TestReadJsonlMarketEventAdapter:
    def test_happy_path(self, tmp_path: Path) -> None:
        path = _write_lines(tmp_path / "candles.jsonl", [_CANDLE_LINE])
        adapter = read_jsonl_market_event_adapter(path, record_kind=RecordKind.CANDLE)
        assert adapter.source_kind() is SourceKind.JSONL_MARKET_EVENTS
        assert adapter.record_kind() is RecordKind.CANDLE
        records = list(adapter.iter_records())
        assert len(records) == 1
        assert records[0].raw_fields["open"] == "2000.5"

    def test_optional_field_omitted_is_absent(self, tmp_path: Path) -> None:
        line = dict(_CANDLE_LINE)
        del line["volume"]
        path = _write_lines(tmp_path / "candles.jsonl", [line])
        adapter = read_jsonl_market_event_adapter(path, record_kind=RecordKind.CANDLE)
        assert "volume" not in next(iter(adapter.iter_records())).raw_fields

    def test_json_number_field_rejected(self, tmp_path: Path) -> None:
        line = dict(_CANDLE_LINE)
        line["open"] = 2000.5  # actual JSON number, not string
        path = _write_lines(tmp_path / "bad.jsonl", [line])
        with pytest.raises(SourceAdapterError):
            read_jsonl_market_event_adapter(path, record_kind=RecordKind.CANDLE)

    def test_missing_required_field_rejected(self, tmp_path: Path) -> None:
        line = dict(_CANDLE_LINE)
        del line["close"]
        path = _write_lines(tmp_path / "missing.jsonl", [line])
        with pytest.raises(SourceAdapterError):
            read_jsonl_market_event_adapter(path, record_kind=RecordKind.CANDLE)

    def test_undeclared_field_rejected(self, tmp_path: Path) -> None:
        line = dict(_CANDLE_LINE)
        line["unexpected_field"] = "zzz"
        path = _write_lines(tmp_path / "extra.jsonl", [line])
        with pytest.raises(SourceAdapterError):
            read_jsonl_market_event_adapter(path, record_kind=RecordKind.CANDLE)

    def test_kind_mismatch_rejected(self, tmp_path: Path) -> None:
        line = dict(_CANDLE_LINE)
        line["kind"] = "tick"
        path = _write_lines(tmp_path / "mismatch.jsonl", [line])
        with pytest.raises(SourceAdapterError):
            read_jsonl_market_event_adapter(path, record_kind=RecordKind.CANDLE)

    def test_malformed_json_line_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "malformed.jsonl"
        path.write_text("{not valid json\n", encoding="utf-8")
        with pytest.raises(SourceAdapterError):
            read_jsonl_market_event_adapter(path, record_kind=RecordKind.CANDLE)

    def test_duplicate_json_key_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "dupkey.jsonl"
        path.write_text('{"kind":"candle","kind":"candle","timestamp":"x","symbol":"y","open":"1","high":"1","low":"1","close":"1","timeframe":"M1"}\n', encoding="utf-8")
        with pytest.raises(SourceAdapterError):
            read_jsonl_market_event_adapter(path, record_kind=RecordKind.CANDLE)

    def test_nan_token_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "nan.jsonl"
        path.write_text('{"kind":"candle","timestamp":"x","symbol":"y","open":NaN,"high":"1","low":"1","close":"1","timeframe":"M1"}\n', encoding="utf-8")
        with pytest.raises(SourceAdapterError):
            read_jsonl_market_event_adapter(path, record_kind=RecordKind.CANDLE)

    def test_non_object_top_level_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "array.jsonl"
        path.write_text("[1,2,3]\n", encoding="utf-8")
        with pytest.raises(SourceAdapterError):
            read_jsonl_market_event_adapter(path, record_kind=RecordKind.CANDLE)

    def test_blank_line_in_middle_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "blank_middle.jsonl"
        path.write_text(json.dumps(_CANDLE_LINE) + "\n\n" + json.dumps(_CANDLE_LINE) + "\n", encoding="utf-8")
        with pytest.raises(SourceAdapterError):
            read_jsonl_market_event_adapter(path, record_kind=RecordKind.CANDLE)

    def test_trailing_blank_line_at_eof_is_tolerated(self, tmp_path: Path) -> None:
        path = tmp_path / "trailing_blank.jsonl"
        path.write_text(json.dumps(_CANDLE_LINE) + "\n\n", encoding="utf-8")
        adapter = read_jsonl_market_event_adapter(path, record_kind=RecordKind.CANDLE)
        assert len(list(adapter.iter_records())) == 1

    def test_tick_quote_trade_schemas_readable(self, tmp_path: Path) -> None:
        tick_path = _write_lines(tmp_path / "tick.jsonl", [{"kind": "tick", "timestamp": "x", "symbol": "y", "price": "2000.5"}])
        tick_adapter = read_jsonl_market_event_adapter(tick_path, record_kind=RecordKind.TICK)
        assert next(iter(tick_adapter.iter_records())).raw_fields["price"] == "2000.5"

        quote_path = _write_lines(tmp_path / "quote.jsonl", [{"kind": "quote", "timestamp": "x", "symbol": "y", "bid": "1", "ask": "2"}])
        quote_adapter = read_jsonl_market_event_adapter(quote_path, record_kind=RecordKind.QUOTE)
        assert next(iter(quote_adapter.iter_records())).raw_fields["bid"] == "1"

        trade_path = _write_lines(tmp_path / "trade.jsonl", [{"kind": "trade", "timestamp": "x", "symbol": "y", "price": "1", "size": "2"}])
        trade_adapter = read_jsonl_market_event_adapter(trade_path, record_kind=RecordKind.TRADE)
        assert next(iter(trade_adapter.iter_records())).raw_fields["size"] == "2"

    def test_content_digest_deterministic(self, tmp_path: Path) -> None:
        path = _write_lines(tmp_path / "candles.jsonl", [_CANDLE_LINE])
        a1 = read_jsonl_market_event_adapter(path, record_kind=RecordKind.CANDLE)
        a2 = read_jsonl_market_event_adapter(path, record_kind=RecordKind.CANDLE)
        assert a1.content_digest() == a2.content_digest()
