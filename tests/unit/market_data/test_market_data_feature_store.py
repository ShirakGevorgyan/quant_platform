"""Unit tests for `market_data.feature_store`: `FeatureRecord` identity/
validation and `FeatureStore`'s append-only, no-overwrite, idempotent,
deterministically-ordered semantics."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from quant_platform.core.exceptions import FeatureStoreError
from quant_platform.core.types import Timeframe
from quant_platform.market_data.feature_store import FeatureStore, create_feature_record

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)


def _record(**overrides: object):
    base: dict[str, object] = {"feature_name": "sma_20", "feature_version": 1, "instrument_id": "mt5__XAUUSD", "timestamp": _T0, "timeframe": Timeframe.H1, "value": Decimal("2000.5")}
    base.update(overrides)
    return create_feature_record(**base)  # type: ignore[arg-type]


class TestFeatureRecordConstruction:
    def test_round_trips_through_json(self) -> None:
        record = _record(metadata={"window": 20})
        assert type(record).from_json_dict(record.to_json_dict()) == record

    def test_identical_arguments_produce_identical_ids(self) -> None:
        assert _record().feature_id == _record().feature_id

    def test_different_value_changes_the_id(self) -> None:
        assert _record().feature_id != _record(value=Decimal("2000.6")).feature_id

    def test_feature_version_below_one_is_rejected(self) -> None:
        with pytest.raises(FeatureStoreError):
            _record(feature_version=0)

    def test_none_timeframe_is_allowed(self) -> None:
        record = _record(timeframe=None)
        assert type(record).from_json_dict(record.to_json_dict()) == record


class TestFeatureStoreAppendOnly:
    def test_append_and_read_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            record = _record()
            store.append(record)
            assert store.read_records("sma_20", 1, "mt5__XAUUSD") == [record]

    def test_identical_reappend_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            record = _record()
            store.append(record)
            store.append(record)
            assert len(store.read_records("sma_20", 1, "mt5__XAUUSD")) == 1

    def test_conflicting_value_at_same_timestamp_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            store.append(_record())
            with pytest.raises(FeatureStoreError):
                store.append(_record(value=Decimal("1999.9")))

    def test_out_of_chronological_order_appends_are_accepted(self) -> None:
        # Unlike MarketEventStore's positional sequence check, FeatureStore
        # is keyed by economic coordinate -- a later timestamp may be
        # appended before an earlier one (e.g. a backfill) without error.
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            later = _record(timestamp=_T0 + timedelta(hours=1))
            earlier = _record(timestamp=_T0)
            store.append(later)
            store.append(earlier)
            records = store.read_records("sma_20", 1, "mt5__XAUUSD")
            assert [r.timestamp for r in records] == [_T0, _T0 + timedelta(hours=1)]

    def test_read_records_is_sorted_by_timestamp_regardless_of_append_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            for hour in (3, 1, 2, 0):
                store.append(_record(timestamp=_T0 + timedelta(hours=hour)))
            records = store.read_records("sma_20", 1, "mt5__XAUUSD")
            assert [r.timestamp for r in records] == [_T0 + timedelta(hours=h) for h in range(4)]

    def test_different_feature_versions_are_independent_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            store.append(_record(feature_version=1))
            store.append(_record(feature_version=2, value=Decimal("9999")))
            assert len(store.read_records("sma_20", 1, "mt5__XAUUSD")) == 1
            assert len(store.read_records("sma_20", 2, "mt5__XAUUSD")) == 1

    def test_missing_partition_reads_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            assert store.read_records("does_not_exist", 1, "mt5__XAUUSD") == []
