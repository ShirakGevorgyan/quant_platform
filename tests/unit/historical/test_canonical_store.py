"""Tests for `historical.canonical_store.CanonicalStore`."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone as dt_timezone

import pandas as pd
import pytest

from quant_platform.core.exceptions import DataSourceError, SnapshotError
from quant_platform.core.types import Timeframe
from quant_platform.data.synthetic import SyntheticDataConfig, generate_ohlcv
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.models import coerce_historical_dtypes

UTC = "UTC"


def _frame(start: datetime, n: int, seed: int = 1) -> pd.DataFrame:
    sd = generate_ohlcv(SyntheticDataConfig(start=start, periods=n, timeframe=Timeframe.M1, seed=seed))
    sd = sd.rename(columns={"volume": "tick_volume"})
    sd["tick_volume"] = sd["tick_volume"].astype("int64")
    sd["real_volume"] = 0
    sd["spread"] = 15
    return coerce_historical_dtypes(sd)


class TestWriteReadRoundTrip:
    def test_round_trip_preserves_data(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 100)
        store.write_partition(df, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        loaded = store.read_partition(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        assert loaded is not None
        loaded_df, metadata = loaded
        pd.testing.assert_frame_equal(loaded_df, df)
        assert metadata.row_count == 100
        assert metadata.compression == "zstd"

    def test_missing_partition_returns_none(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        assert store.read_partition(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024) is None

    def test_empty_dataframe_partition_round_trips(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 1).iloc[0:0]
        metadata = store.write_partition(df, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        assert metadata.row_count == 0
        assert metadata.min_open_time is None


class TestPartitionBoundaryEnforcement:
    def test_rejects_rows_outside_the_declared_year(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        df = _frame(datetime(2023, 12, 31, 23, 0, tzinfo=dt_timezone.utc), 120)
        with pytest.raises(ValueError, match="fall outside"):
            store.write_partition(df, symbol="XAUUSD", timeframe=Timeframe.M1, year=2023)

    def test_accepts_rows_fully_within_the_declared_year(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        df = _frame(datetime(2023, 12, 31, 20, 0, tzinfo=dt_timezone.utc), 60)
        metadata = store.write_partition(df, symbol="XAUUSD", timeframe=Timeframe.M1, year=2023)
        assert metadata.row_count == 60


class TestFullPartitionReplace:
    def test_rewriting_a_partition_fully_replaces_it(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        first = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 100)
        store.write_partition(first, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        second = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 50, seed=2)
        store.write_partition(second, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        loaded_df, metadata = store.read_partition(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        assert metadata.row_count == 50
        pd.testing.assert_frame_equal(loaded_df, second)


class TestCorruptionDetection:
    def test_missing_success_marker_is_rejected(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 20)
        metadata = store.write_partition(df, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        content_dir = store.content_dir(
            symbol="XAUUSD", timeframe=Timeframe.M1, year=2024, content_id=metadata.content_id
        )
        (content_dir / "_SUCCESS").unlink()
        with pytest.raises(SnapshotError, match="incomplete or corrupted"):
            store.read_partition(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)

    def test_corrupted_data_file_is_rejected(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 20)
        metadata = store.write_partition(df, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        content_dir = store.content_dir(
            symbol="XAUUSD", timeframe=Timeframe.M1, year=2024, content_id=metadata.content_id
        )
        with (content_dir / "data.parquet").open("ab") as fh:
            fh.write(b"corruption")
        with pytest.raises(SnapshotError, match="checksum mismatch"):
            store.read_partition(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)


class TestPathSecurity:
    def test_path_traversal_in_symbol_is_rejected(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 5)
        with pytest.raises(DataSourceError):
            store.write_partition(df, symbol="../escape", timeframe=Timeframe.M1, year=2024)


class TestReadRange:
    def test_reads_only_overlapping_years_and_filters_within_them(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        df_2023 = _frame(datetime(2023, 12, 31, 20, 0, tzinfo=dt_timezone.utc), 60)
        df_2024 = _frame(datetime(2024, 1, 1, 0, 0, tzinfo=dt_timezone.utc), 120)
        store.write_partition(df_2023, symbol="XAUUSD", timeframe=Timeframe.M1, year=2023)
        store.write_partition(df_2024, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)

        combined = store.read_range(
            symbol="XAUUSD", timeframe=Timeframe.M1,
            start=pd.Timestamp("2023-12-31T20:30", tz=UTC), end=pd.Timestamp("2024-01-01T01:00", tz=UTC),
        )
        assert combined["open_time"].is_monotonic_increasing
        assert combined["open_time"].iloc[0] == pd.Timestamp("2023-12-31T20:30", tz=UTC)
        assert combined["open_time"].iloc[-1] == pd.Timestamp("2024-01-01T00:59", tz=UTC)
        assert len(combined) == 90  # hand-computed: 30 bars from 2023 tail + 60 bars from 2024 head

    def test_unknown_symbol_returns_empty_frame_with_correct_schema(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        result = store.read_range(
            symbol="NOPE", timeframe=Timeframe.M1,
            start=pd.Timestamp("2024-01-01", tz=UTC), end=pd.Timestamp("2024-01-02", tz=UTC),
        )
        assert len(result) == 0
        assert list(result.columns) == [
            "open_time", "open", "high", "low", "close", "tick_volume", "real_volume", "spread",
        ]

    def test_rejects_end_not_after_start(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        with pytest.raises(ValueError, match="must be after"):
            store.read_range(
                symbol="XAUUSD", timeframe=Timeframe.M1,
                start=pd.Timestamp("2024-01-02", tz=UTC), end=pd.Timestamp("2024-01-01", tz=UTC),
            )


class TestContentAddressedImmutability:
    """Release-readiness-audit coverage for the redesigned storage layer:
    every partition write is content-addressed and immutable -- writing
    DIFFERENT content for the same year never destroys the old blob, and
    it remains byte-for-byte reconstructable by content id forever, even
    after `CURRENT` has moved on. This is what makes exact historical
    dataset-version reconstruction possible (see `historical.loader`)."""

    def test_writing_different_content_preserves_the_old_blob(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        original = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 100, seed=1)
        original_meta = store.write_partition(original, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)

        revised = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 150, seed=2)
        revised_meta = store.write_partition(revised, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        assert revised_meta.content_id != original_meta.content_id

        # CURRENT now serves the revised data...
        current_df, _ = store.read_partition(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        assert len(current_df) == 150

        # ...but the ORIGINAL content is still there, untouched, forever.
        old_df, _ = store.read_partition_by_content_id(
            symbol="XAUUSD", timeframe=Timeframe.M1, year=2024, content_id=original_meta.content_id
        )
        pd.testing.assert_frame_equal(old_df, original)

    def test_rewriting_identical_content_is_deduplicated(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 50)
        meta_a = store.write_partition(df, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        content_dir = store.content_dir(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024, content_id=meta_a.content_id)
        written_at_first = (content_dir / "metadata.json").read_text()

        meta_b = store.write_partition(df, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        assert meta_b.content_id == meta_a.content_id
        # The existing blob was reused, not rewritten (same metadata.json bytes).
        assert (content_dir / "metadata.json").read_text() == written_at_first

    def test_current_pointer_can_be_flipped_back_to_an_earlier_content_id(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        original = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 50, seed=1)
        original_meta = store.write_partition(original, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        store.write_partition(_frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 75, seed=2), symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)

        # Re-submitting the ORIGINAL content moves CURRENT back to it.
        store.write_partition(original, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        current_id = store.current_content_id(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        assert current_id == original_meta.content_id

    def test_content_id_matches_content_checksum(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 20)
        metadata = store.write_partition(df, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        assert metadata.content_id == metadata.content_checksum

    def test_read_partition_by_content_id_returns_none_for_unknown_content(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 5)
        store.write_partition(df, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        result = store.read_partition_by_content_id(
            symbol="XAUUSD", timeframe=Timeframe.M1, year=2024, content_id="0" * 64
        )
        assert result is None

    def test_year_with_content_but_no_current_pointer_is_excluded_from_list_years(self, tmp_path) -> None:
        store = CanonicalStore(tmp_path)
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 5)
        metadata = store.write_partition(df, symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        pointer_path = store.partition_dir(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024) / "CURRENT"
        pointer_path.unlink()
        assert store.list_years(symbol="XAUUSD", timeframe=Timeframe.M1) == []
        # But the content itself is still there, reconstructable by id.
        loaded = store.read_partition_by_content_id(
            symbol="XAUUSD", timeframe=Timeframe.M1, year=2024, content_id=metadata.content_id
        )
        assert loaded is not None
