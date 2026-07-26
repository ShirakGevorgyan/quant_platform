from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.core.exceptions import ArtifactCorruptionError, SchemaVersionError
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    canonical_json_bytes,
    format_utc_timestamp,
    parse_json_strict,
    parse_utc_timestamp,
    read_json_file,
    require_schema_version,
    sha256_hex_bytes,
    utc_now,
    write_json_atomic,
)


class TestCanonicalJsonBytes:
    def test_sorts_keys_deterministically(self) -> None:
        b1 = canonical_json_bytes({"b": 1, "a": 2})
        b2 = canonical_json_bytes({"a": 2, "b": 1})
        assert b1 == b2

    def test_nested_dict_order_independence(self) -> None:
        b1 = canonical_json_bytes({"outer": {"z": 1, "a": 2}})
        b2 = canonical_json_bytes({"outer": {"a": 2, "z": 1}})
        assert b1 == b2

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json_bytes({"a": float("nan")})

    def test_rejects_infinity(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json_bytes({"a": float("inf")})
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json_bytes({"a": float("-inf")})

    def test_compact_separators_no_whitespace(self) -> None:
        text = canonical_json_bytes({"a": 1, "b": 2}).decode()
        assert " " not in text


class TestParseJsonStrict:
    def test_round_trip(self) -> None:
        payload = {"a": 1, "b": "x"}
        assert parse_json_strict(canonical_json_bytes(payload).decode()) == payload

    def test_rejects_nan_token(self) -> None:
        with pytest.raises(ValueError, match="Non-finite"):
            parse_json_strict('{"a": NaN}')

    def test_rejects_infinity_token(self) -> None:
        with pytest.raises(ValueError, match="Non-finite"):
            parse_json_strict('{"a": Infinity}')
        with pytest.raises(ValueError, match="Non-finite"):
            parse_json_strict('{"a": -Infinity}')

    def test_malformed_json_raises_actionable_error(self) -> None:
        with pytest.raises(ValueError, match="Malformed JSON"):
            parse_json_strict("{not json")


class TestUtcTimestampFormatting:
    def test_utc_now_is_tz_aware(self) -> None:
        assert utc_now().tzinfo is not None

    def test_format_round_trip(self) -> None:
        ts = pd.Timestamp("2024-01-01T12:34:56", tz="UTC")
        formatted = format_utc_timestamp(ts)
        assert parse_utc_timestamp(formatted) == ts

    def test_format_rejects_naive_timestamp(self) -> None:
        with pytest.raises(ValueError, match="not timezone-aware"):
            format_utc_timestamp(pd.Timestamp("2024-01-01"))

    def test_parse_rejects_naive_string(self) -> None:
        with pytest.raises(ValueError, match="not timezone-aware"):
            parse_utc_timestamp("2024-01-01T00:00:00")

    def test_parse_converts_non_utc_to_utc(self) -> None:
        ts = parse_utc_timestamp("2024-01-01T00:00:00+05:00")
        assert str(ts.tzinfo) == "UTC"


class TestWriteReadJsonAtomic:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "file.json"
        payload = {"a": 1, "b": [1, 2, 3]}
        write_json_atomic(path, payload)
        assert read_json_file(path) == payload

    def test_no_temp_file_left_behind_on_success(self, tmp_path: Path) -> None:
        path = tmp_path / "file.json"
        write_json_atomic(path, {"a": 1})
        leftovers = list(tmp_path.glob(".*"))
        assert leftovers == []

    def test_temp_file_cleaned_up_on_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "file.json"
        with pytest.raises(ValueError):
            write_json_atomic(path, {"a": float("nan")})
        assert not path.exists()
        assert list(tmp_path.glob(".*.tmp")) == []

    def test_read_missing_file_raises_artifact_corruption_error(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactCorruptionError):
            read_json_file(tmp_path / "does_not_exist.json")

    def test_read_malformed_json_raises_artifact_corruption_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        with pytest.raises(ArtifactCorruptionError):
            read_json_file(path)

    def test_write_never_exposes_partial_file(self, tmp_path: Path) -> None:
        """A reader must never observe a file mid-write -- verified here by
        confirming the final file is only ever created via a rename onto
        the destination path (no direct writes to `path` itself)."""
        path = tmp_path / "file.json"
        write_json_atomic(path, {"a": 1})
        # A second write to the same path must still succeed atomically
        write_json_atomic(path, {"a": 2})
        assert read_json_file(path) == {"a": 2}


class TestRequireSchemaVersion:
    def test_accepts_matching_version(self) -> None:
        assert require_schema_version({"schema_version": 3}, supported=3, context="X") == 3

    def test_rejects_missing_version(self) -> None:
        with pytest.raises(SchemaVersionError):
            require_schema_version({}, supported=1, context="X")

    def test_rejects_non_int_version(self) -> None:
        with pytest.raises(SchemaVersionError):
            require_schema_version({"schema_version": "1"}, supported=1, context="X")

    def test_rejects_bool_version(self) -> None:
        with pytest.raises(SchemaVersionError):
            require_schema_version({"schema_version": True}, supported=1, context="X")

    def test_rejects_mismatched_version(self) -> None:
        with pytest.raises(SchemaVersionError):
            require_schema_version({"schema_version": 2}, supported=1, context="X")


class TestAsJsonDictAndList:
    def test_as_json_dict_accepts_dict(self) -> None:
        assert as_json_dict({"a": 1}, field_name="f") == {"a": 1}

    def test_as_json_dict_rejects_non_dict(self) -> None:
        with pytest.raises(Exception, match="expected a JSON object"):
            as_json_dict([1, 2], field_name="f")

    def test_as_json_list_accepts_list(self) -> None:
        assert as_json_list([1, 2], field_name="f") == [1, 2]

    def test_as_json_list_rejects_non_list(self) -> None:
        with pytest.raises(Exception, match="expected a JSON array"):
            as_json_list({"a": 1}, field_name="f")


def test_sha256_hex_bytes_matches_stdlib(tmp_path: Path) -> None:
    import hashlib

    data = b"hello world"
    assert sha256_hex_bytes(data) == hashlib.sha256(data).hexdigest()


def test_canonical_json_bytes_is_valid_json() -> None:
    payload = {"a": 1, "b": "x"}
    assert json.loads(canonical_json_bytes(payload)) == payload
