"""Unit tests for `market_data.source_manifests` and `market_data.mappings`
(Milestone 10, Phase 3): source manifest identity rules, and instrument/
timeframe mapping resolution and identity."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.core.exceptions import InstrumentMappingError, SourceManifestError, TimeframeMappingError
from quant_platform.core.types import Timeframe
from quant_platform.market_data.mappings import (
    InstrumentMappingEntry,
    TimeframeMappingEntry,
    create_instrument_mapping_spec,
    create_timeframe_mapping_spec,
    default_timeframe_mapping_spec,
    resolve_instrument_id,
    resolve_timeframe,
)
from quant_platform.market_data.source_manifests import (
    RecordKind,
    SourceKind,
    SourceManifest,
    compute_timestamp_policy_id,
    create_source_manifest,
)
from quant_platform.market_data.source_normalization import TimestampParsingPolicy

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)


def _manifest(**overrides: object) -> SourceManifest:
    kwargs: dict[str, object] = {
        "source_name": "test_source", "source_kind": SourceKind.CSV_CANDLES, "source_schema_version": 1,
        "record_kind": RecordKind.CANDLE, "source_label": "xauusd.csv", "content_digest": "a" * 64, "byte_size": 100,
        "encoding": "utf-8", "instrument_mapping_id": "b" * 64, "timezone_policy_id": "c" * 64,
        "unit_normalization_version": 1, "creation_time": _T0,
    }
    kwargs.update(overrides)
    return create_source_manifest(**kwargs)  # type: ignore[arg-type]


class TestSourceManifestIdentity:
    def test_same_content_different_creation_time_same_id(self) -> None:
        m1 = _manifest(creation_time=_T0)
        m2 = _manifest(creation_time=datetime(2030, 1, 1, tzinfo=timezone.utc))
        assert m1.source_manifest_id == m2.source_manifest_id

    def test_changed_content_digest_changes_id(self) -> None:
        m1 = _manifest(content_digest="a" * 64)
        m2 = _manifest(content_digest="d" * 64)
        assert m1.source_manifest_id != m2.source_manifest_id

    def test_changed_timezone_policy_id_changes_id(self) -> None:
        m1 = _manifest(timezone_policy_id="c" * 64)
        m2 = _manifest(timezone_policy_id="e" * 64)
        assert m1.source_manifest_id != m2.source_manifest_id

    def test_changed_instrument_mapping_id_changes_id(self) -> None:
        m1 = _manifest(instrument_mapping_id="b" * 64)
        m2 = _manifest(instrument_mapping_id="f" * 64)
        assert m1.source_manifest_id != m2.source_manifest_id

    def test_row_count_excluded_from_identity(self) -> None:
        m1 = _manifest(row_count=None)
        m2 = _manifest(row_count=42)
        assert m1.source_manifest_id == m2.source_manifest_id

    def test_round_trips_through_json(self) -> None:
        m1 = _manifest()
        rt = SourceManifest.from_json_dict(m1.to_json_dict())
        assert rt == m1

    def test_source_label_participates_in_identity(self) -> None:
        m1 = _manifest(source_label="a.csv")
        m2 = _manifest(source_label="b.csv")
        assert m1.source_manifest_id != m2.source_manifest_id

    def test_expected_end_before_expected_start_rejected(self) -> None:
        with pytest.raises(SourceManifestError):
            _manifest(expected_start=_T0, expected_end=_T0.replace(year=2020))

    def test_invalid_content_digest_length_rejected(self) -> None:
        with pytest.raises(SourceManifestError):
            _manifest(content_digest="short")

    def test_non_positive_schema_version_rejected(self) -> None:
        with pytest.raises(SourceManifestError):
            _manifest(source_schema_version=0)


class TestTimestampPolicyId:
    def test_deterministic(self) -> None:
        p1 = TimestampParsingPolicy(formats=("%Y-%m-%d",), source_timezone=None)
        p2 = TimestampParsingPolicy(formats=("%Y-%m-%d",), source_timezone=None)
        assert compute_timestamp_policy_id(p1) == compute_timestamp_policy_id(p2)

    def test_different_formats_different_id(self) -> None:
        p1 = TimestampParsingPolicy(formats=("%Y-%m-%d",), source_timezone=None)
        p2 = TimestampParsingPolicy(formats=("%Y/%m/%d",), source_timezone=None)
        assert compute_timestamp_policy_id(p1) != compute_timestamp_policy_id(p2)


class TestInstrumentMapping:
    def test_exact_provider_match_takes_precedence_over_wildcard(self) -> None:
        spec = create_instrument_mapping_spec(
            mapping_version=1,
            entries=(
                InstrumentMappingEntry(source_symbol="GC", instrument_id="GENERIC_GOLD", provider=None),
                InstrumentMappingEntry(source_symbol="GC", instrument_id="COMEX_GOLD_FUTURE", provider="cme"),
            ),
        )
        assert resolve_instrument_id(spec, source_symbol="GC", provider="cme") == "COMEX_GOLD_FUTURE"
        assert resolve_instrument_id(spec, source_symbol="GC", provider="other_provider") == "GENERIC_GOLD"

    def test_unmapped_symbol_fails_closed(self) -> None:
        spec = create_instrument_mapping_spec(mapping_version=1, entries=(InstrumentMappingEntry(source_symbol="XAUUSD", instrument_id="XAUUSD"),))
        with pytest.raises(InstrumentMappingError):
            resolve_instrument_id(spec, source_symbol="UNKNOWN", provider="any")

    def test_duplicate_source_symbol_provider_pair_rejected_at_construction(self) -> None:
        with pytest.raises(InstrumentMappingError):
            create_instrument_mapping_spec(
                mapping_version=1,
                entries=(
                    InstrumentMappingEntry(source_symbol="XAUUSD", instrument_id="A"),
                    InstrumentMappingEntry(source_symbol="XAUUSD", instrument_id="B"),
                ),
            )

    def test_mapping_change_produces_new_mapping_id(self) -> None:
        spec1 = create_instrument_mapping_spec(mapping_version=1, entries=(InstrumentMappingEntry(source_symbol="XAUUSD", instrument_id="XAUUSD"),))
        spec2 = create_instrument_mapping_spec(mapping_version=1, entries=(InstrumentMappingEntry(source_symbol="XAUUSD", instrument_id="XAUUSD_V2"),))
        assert spec1.mapping_id != spec2.mapping_id

    def test_entry_order_does_not_affect_mapping_id(self) -> None:
        entries_a = (InstrumentMappingEntry(source_symbol="A", instrument_id="1"), InstrumentMappingEntry(source_symbol="B", instrument_id="2"))
        entries_b = (InstrumentMappingEntry(source_symbol="B", instrument_id="2"), InstrumentMappingEntry(source_symbol="A", instrument_id="1"))
        spec_a = create_instrument_mapping_spec(mapping_version=1, entries=entries_a)
        spec_b = create_instrument_mapping_spec(mapping_version=1, entries=entries_b)
        assert spec_a.mapping_id == spec_b.mapping_id

    def test_round_trip(self) -> None:
        spec = create_instrument_mapping_spec(mapping_version=1, entries=(InstrumentMappingEntry(source_symbol="XAUUSD", instrument_id="XAUUSD", provider="mt5"),))
        rt = type(spec).from_json_dict(spec.to_json_dict())
        assert rt == spec

    def test_non_positive_mapping_version_rejected(self) -> None:
        with pytest.raises(InstrumentMappingError):
            create_instrument_mapping_spec(mapping_version=0, entries=())


class TestTimeframeMapping:
    def test_resolves_known_aliases(self) -> None:
        spec = default_timeframe_mapping_spec()
        assert resolve_timeframe(spec, source_label="1h") is Timeframe.H1
        assert resolve_timeframe(spec, source_label="H1") is Timeframe.H1
        assert resolve_timeframe(spec, source_label="1d") is Timeframe.D1

    def test_unknown_timeframe_fails_closed(self) -> None:
        spec = default_timeframe_mapping_spec()
        with pytest.raises(TimeframeMappingError):
            resolve_timeframe(spec, source_label="not_a_real_timeframe")

    def test_duplicate_source_label_rejected_at_construction(self) -> None:
        with pytest.raises(TimeframeMappingError):
            create_timeframe_mapping_spec(
                mapping_version=1,
                entries=(TimeframeMappingEntry(source_label="M1", timeframe=Timeframe.M1), TimeframeMappingEntry(source_label="M1", timeframe=Timeframe.M5)),
            )

    def test_mapping_change_produces_new_mapping_id(self) -> None:
        spec1 = create_timeframe_mapping_spec(mapping_version=1, entries=(TimeframeMappingEntry(source_label="M1", timeframe=Timeframe.M1),))
        spec2 = create_timeframe_mapping_spec(mapping_version=1, entries=(TimeframeMappingEntry(source_label="M1", timeframe=Timeframe.M5),))
        assert spec1.mapping_id != spec2.mapping_id

    def test_round_trip(self) -> None:
        spec = default_timeframe_mapping_spec()
        rt = type(spec).from_json_dict(spec.to_json_dict())
        assert rt == spec
