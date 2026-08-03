from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelRecordConflictError, LabelRequestError
from quant_platform.core.types import Timeframe
from quant_platform.labels.builder import LabelBuilder, LabelDefinition
from quant_platform.labels.next_return import build_next_return_specification, generate_next_return_labels
from quant_platform.labels.pricing import PriceBasis
from quant_platform.labels.records import (
    LabelRecord,
    LabelRecordLedger,
    compute_content_hash,
    compute_label_id,
    materialize_label_records,
)


@pytest.fixture
def next_return_bundle(ohlcv_source_data: pd.DataFrame):
    spec = build_next_return_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1")
    definition = LabelDefinition(specification=spec, generate=generate_next_return_labels)
    return LabelBuilder().build(definition, ohlcv_source_data, source_content_id="src1")


class TestMaterializeLabelRecords:
    def test_one_record_per_row(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        records = materialize_label_records(next_return_bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        assert len(records) == len(ohlcv_source_data)

    def test_every_record_carries_the_eight_required_fields(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        records = materialize_label_records(next_return_bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        record = records[0]
        assert record.label_id
        assert record.label_specification_id == next_return_bundle.specification.label_specification_id
        assert record.dataset_id == "ds1"
        assert record.row_identity
        assert record.event_time
        assert record.availability_time
        assert record.generation_version == next_return_bundle.specification.generation_version
        assert record.content_hash

    def test_event_time_is_bar_close_time(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        records = materialize_label_records(next_return_bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        expected = (ohlcv_source_data["open_time"].iloc[0] + Timeframe.M1.duration).isoformat()
        assert records[0].event_time == expected

    def test_availability_time_is_event_time_plus_horizon(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        records = materialize_label_records(next_return_bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        record = records[0]
        expected = pd.Timestamp(record.event_time) + Timeframe.M1.duration * 5
        assert pd.Timestamp(record.availability_time) == expected

    def test_missing_open_time_column_rejected(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        stripped = ohlcv_source_data.drop(columns=["open_time"])
        with pytest.raises(LabelRequestError):
            materialize_label_records(next_return_bundle, stripped, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)

    def test_records_are_self_consistent(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        records = materialize_label_records(next_return_bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        for record in records:
            consistent, issues = record.verify_self_consistency()
            assert consistent is True
            assert issues == ()

    def test_nan_values_are_encoded_as_none(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        records = materialize_label_records(next_return_bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        assert records[-1].value is None  # trailing NaN row


class TestComputeLabelIdAndContentHash:
    def test_label_id_deterministic(self) -> None:
        assert compute_label_id("spec-1", "row-1") == compute_label_id("spec-1", "row-1")

    def test_label_id_independent_of_value(self) -> None:
        label_id = compute_label_id("spec-1", "row-1")
        assert compute_content_hash(label_id, 1.0) != compute_content_hash(label_id, 2.0)

    def test_content_hash_changes_with_value_but_label_id_does_not(self) -> None:
        label_id = compute_label_id("spec-1", "row-1")
        hash_a = compute_content_hash(label_id, 1.0)
        hash_b = compute_content_hash(label_id, None)
        assert hash_a != hash_b


class TestLabelRecordJsonRoundTrip:
    def test_round_trip(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        records = materialize_label_records(next_return_bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        restored = LabelRecord.from_json_dict(records[0].to_json_dict())
        assert restored == records[0]


class TestLabelRecordLedger:
    def test_commit_then_recover_returns_empty(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        records = materialize_label_records(next_return_bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        ledger = LabelRecordLedger()
        ledger.commit(records)
        assert ledger.recover(records) == ()
        assert ledger.committed_count(next_return_bundle.specification.label_specification_id) == len(records)

    def test_double_commit_raises_and_never_overwrites(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        records = materialize_label_records(next_return_bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        ledger = LabelRecordLedger()
        ledger.commit(records)
        with pytest.raises(LabelRecordConflictError):
            ledger.commit(records)
        # still exactly one commit's worth -- the failed re-commit did not partially apply
        assert ledger.committed_count(next_return_bundle.specification.label_specification_id) == len(records)

    def test_partial_commit_then_recover_returns_only_missing(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        records = materialize_label_records(next_return_bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        ledger = LabelRecordLedger()
        already_committed, not_yet_committed = records[:10], records[10:]
        ledger.commit(already_committed)
        recovered = ledger.recover(records)
        assert recovered == not_yet_committed

    def test_atomic_commit_rejects_batch_with_any_conflict(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        records = materialize_label_records(next_return_bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        ledger = LabelRecordLedger()
        ledger.commit(records[:1])
        with pytest.raises(LabelRecordConflictError):
            ledger.commit(records[:5])  # records[0] conflicts -- the whole batch must be refused
        # records[1:5] must NOT have been committed by the refused batch
        assert ledger.committed_count(next_return_bundle.specification.label_specification_id) == 1
