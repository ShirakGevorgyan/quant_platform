"""FRED JSON/CSV parsing tests (Milestone 10, Phase 4A) -- strict,
text-only parsing (never through float), FRED's `"."` missing-value
convention, and strict-mode rejection of malformed/undeclared shapes."""

from __future__ import annotations

import json

import pytest

from quant_platform.core.exceptions import MalformedFredResponseError, UnsupportedFredSchemaError
from quant_platform.market_data.collectors.fred_schemas import (
    FRED_MISSING_VALUE_TEXT,
    is_missing_value,
    parse_fred_csv_response,
    parse_fred_json_response,
)


def _json_bytes(observations: list[dict]) -> bytes:
    return json.dumps({"observations": observations}).encode("utf-8")


class TestIsMissingValue:
    def test_dot_is_missing(self) -> None:
        assert is_missing_value(FRED_MISSING_VALUE_TEXT)
        assert is_missing_value(".")

    def test_dot_with_surrounding_whitespace_is_missing(self) -> None:
        assert is_missing_value("  .  ")

    def test_real_value_is_not_missing(self) -> None:
        assert not is_missing_value("4.02")

    def test_empty_string_is_not_the_fred_missing_marker(self) -> None:
        assert not is_missing_value("")


class TestParseFredJsonResponse:
    def test_valid_observations_are_parsed(self) -> None:
        body = _json_bytes([
            {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"},
            {"date": "2024-01-03", "value": "4.05", "realtime_start": "2024-01-03", "realtime_end": "9999-12-31"},
        ])
        observations = parse_fred_json_response(body, series_id="DGS10")
        assert len(observations) == 2
        assert observations[0].observation_date == "2024-01-02"
        assert observations[0].value_text == "4.02"
        assert observations[0].series_id == "DGS10"
        assert observations[0].row_index == 0
        assert observations[1].row_index == 1

    def test_missing_value_marker_is_preserved_as_text(self) -> None:
        body = _json_bytes([{"date": "2024-01-02", "value": "."}])
        observations = parse_fred_json_response(body, series_id="DGS10")
        assert observations[0].value_text == "."
        assert is_missing_value(observations[0].value_text)

    def test_observations_without_realtime_fields_parse_with_none(self) -> None:
        body = _json_bytes([{"date": "2024-01-02", "value": "4.02"}])
        observations = parse_fred_json_response(body, series_id="DGS10")
        assert observations[0].realtime_start is None
        assert observations[0].realtime_end is None

    def test_malformed_json_is_rejected(self) -> None:
        with pytest.raises(MalformedFredResponseError):
            parse_fred_json_response(b"{not valid json", series_id="DGS10")

    def test_non_object_top_level_is_rejected(self) -> None:
        with pytest.raises(MalformedFredResponseError):
            parse_fred_json_response(b"[1, 2, 3]", series_id="DGS10")

    def test_missing_observations_key_is_unsupported_schema(self) -> None:
        with pytest.raises(UnsupportedFredSchemaError):
            parse_fred_json_response(json.dumps({"not_observations": []}).encode(), series_id="DGS10")

    def test_observations_not_a_list_is_rejected(self) -> None:
        with pytest.raises(MalformedFredResponseError):
            parse_fred_json_response(json.dumps({"observations": "nope"}).encode(), series_id="DGS10")

    def test_observation_not_an_object_is_rejected(self) -> None:
        with pytest.raises(MalformedFredResponseError):
            parse_fred_json_response(json.dumps({"observations": ["not-a-dict"]}).encode(), series_id="DGS10")

    def test_missing_required_key_is_rejected(self) -> None:
        with pytest.raises(MalformedFredResponseError):
            parse_fred_json_response(json.dumps({"observations": [{"date": "2024-01-02"}]}).encode(), series_id="DGS10")

    def test_undeclared_key_under_strict_mode_is_rejected(self) -> None:
        body = json.dumps({"observations": [{"date": "2024-01-02", "value": "4.02", "unexpected_field": "x"}]}).encode()
        with pytest.raises(MalformedFredResponseError):
            parse_fred_json_response(body, series_id="DGS10")

    def test_json_number_for_value_is_rejected_never_parsed_as_float(self) -> None:
        """The single most important strict-parsing rule: FRED's `value`
        MUST be a JSON string, never coerced from a JSON number (which
        would silently round-trip through float)."""
        body = json.dumps({"observations": [{"date": "2024-01-02", "value": 4.02}]}).encode()
        with pytest.raises(MalformedFredResponseError):
            parse_fred_json_response(body, series_id="DGS10")

    def test_json_number_for_date_is_rejected(self) -> None:
        body = json.dumps({"observations": [{"date": 20240102, "value": "4.02"}]}).encode()
        with pytest.raises(MalformedFredResponseError):
            parse_fred_json_response(body, series_id="DGS10")

    def test_non_string_realtime_start_is_rejected(self) -> None:
        body = json.dumps({"observations": [{"date": "2024-01-02", "value": "4.02", "realtime_start": 20240102}]}).encode()
        with pytest.raises(MalformedFredResponseError):
            parse_fred_json_response(body, series_id="DGS10")

    def test_duplicate_observation_dates_pass_through_parsing(self) -> None:
        """Deduplication is deferred to a later layer (normalization/
        reconciliation) -- the parser itself just reports rows as-is."""
        body = _json_bytes([
            {"date": "2024-01-02", "value": "4.02"},
            {"date": "2024-01-02", "value": "4.03"},
        ])
        observations = parse_fred_json_response(body, series_id="DGS10")
        assert len(observations) == 2
        assert observations[0].observation_date == observations[1].observation_date

    def test_conflicting_vintage_rows_pass_through_parsing(self) -> None:
        body = _json_bytes([
            {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"},
            {"date": "2024-01-02", "value": "4.05", "realtime_start": "2024-06-01"},
        ])
        observations = parse_fred_json_response(body, series_id="DGS10")
        assert len(observations) == 2
        assert observations[0].value_text != observations[1].value_text

    def test_empty_observations_list_is_valid(self) -> None:
        assert parse_fred_json_response(_json_bytes([]), series_id="DGS10") == []

    def test_wrong_encoding_bytes_are_rejected(self) -> None:
        with pytest.raises(MalformedFredResponseError):
            parse_fred_json_response(b"\xff\xfe\x00\x01", series_id="DGS10")

    def test_metadata_mismatch_series_id_is_caller_supplied_not_validated_against_body(self) -> None:
        """FRED's JSON schema itself carries no per-row series_id --
        `series_id` is a caller-supplied label attached to every parsed
        row; a caller requesting the wrong series against a cached
        response for a DIFFERENT series is a metadata-linkage concern
        the SOURCE MANIFEST layer catches (content digest binding), not
        this parser."""
        body = _json_bytes([{"date": "2024-01-02", "value": "4.02"}])
        observations = parse_fred_json_response(body, series_id="CPIAUCSL")
        assert observations[0].series_id == "CPIAUCSL"


class TestParseFredCsvResponse:
    def test_valid_csv_is_parsed(self) -> None:
        body = b"DATE,DGS10\n2024-01-02,4.02\n2024-01-03,4.05\n"
        observations = parse_fred_csv_response(body, series_id="DGS10")
        assert len(observations) == 2
        assert observations[0].observation_date == "2024-01-02"
        assert observations[0].value_text == "4.02"
        assert observations[0].realtime_start is None

    def test_missing_value_marker_in_csv(self) -> None:
        body = b"DATE,DGS10\n2024-01-02,.\n"
        observations = parse_fred_csv_response(body, series_id="DGS10")
        assert is_missing_value(observations[0].value_text)

    def test_wrong_header_is_rejected(self) -> None:
        body = b"NOT_DATE,DGS10\n2024-01-02,4.02\n"
        with pytest.raises(UnsupportedFredSchemaError):
            parse_fred_csv_response(body, series_id="DGS10")

    def test_wrong_column_count_header_is_rejected(self) -> None:
        body = b"DATE,DGS10,EXTRA\n2024-01-02,4.02,x\n"
        with pytest.raises(UnsupportedFredSchemaError):
            parse_fred_csv_response(body, series_id="DGS10")

    def test_empty_response_has_no_header_and_is_rejected(self) -> None:
        with pytest.raises(MalformedFredResponseError):
            parse_fred_csv_response(b"", series_id="DGS10")

    def test_header_only_csv_is_valid_and_empty(self) -> None:
        assert parse_fred_csv_response(b"DATE,DGS10\n", series_id="DGS10") == []

    def test_row_with_wrong_field_count_is_rejected(self) -> None:
        body = b"DATE,DGS10\n2024-01-02,4.02,extra\n"
        with pytest.raises(MalformedFredResponseError):
            parse_fred_csv_response(body, series_id="DGS10")

    def test_byte_order_mark_is_tolerated(self) -> None:
        body = "﻿DATE,DGS10\n2024-01-02,4.02\n".encode()
        observations = parse_fred_csv_response(body, series_id="DGS10")
        assert len(observations) == 1

    def test_header_match_is_case_insensitive(self) -> None:
        body = b"date,dgs10\n2024-01-02,4.02\n"
        observations = parse_fred_csv_response(body, series_id="DGS10")
        assert len(observations) == 1
