"""Tests for `historical.loader.DatasetLoader`."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone as dt_timezone

import pandas as pd
import pytest

from quant_platform.core.exceptions import DataQualityError, ManifestError
from quant_platform.core.types import OHLCV_COLUMNS, Timeframe
from quant_platform.data.synthetic import SyntheticDataConfig, generate_ohlcv
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader, LoadRequest
from quant_platform.historical.manifest import ManifestStore
from quant_platform.historical.models import coerce_historical_dtypes
from quant_platform.historical.update_pipeline import apply_incremental_update

UTC = "UTC"


def _frame(start: datetime, n: int, seed: int = 1) -> pd.DataFrame:
    sd = generate_ohlcv(SyntheticDataConfig(start=start, periods=n, timeframe=Timeframe.M1, seed=seed))
    sd = sd.rename(columns={"volume": "tick_volume"})
    sd["tick_volume"] = sd["tick_volume"].astype("int64")
    sd["real_volume"] = 0
    sd["spread"] = 15
    return coerce_historical_dtypes(sd)


@pytest.fixture
def loaded_dataset(tmp_path):
    cstore = CanonicalStore(tmp_path)
    mstore = ManifestStore(tmp_path)
    df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 200)
    apply_incremental_update(
        cstore, mstore, df, symbol="XAUUSD", timeframe=Timeframe.M1, source_name="mt5", broker="B",
        pipeline_version="1.0.0", parent_snapshot_ids=(),
        requested_start=df["open_time"].iloc[0], requested_end=df["open_time"].iloc[-1] + pd.Timedelta(minutes=1),
        quality_summary={"critical": 0, "warning": 1},
    )
    return DatasetLoader(cstore, mstore), df


class TestLoadRequestValidation:
    def test_rejects_naive_start_and_end(self) -> None:
        with pytest.raises(Exception, match="naive"):
            LoadRequest(symbol="X", timeframe=Timeframe.M1, start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-02"))

    def test_rejects_mixed_naive_and_aware(self) -> None:
        with pytest.raises(Exception, match="datetime64"):
            LoadRequest(symbol="X", timeframe=Timeframe.M1, start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-02", tz=UTC))

    def test_rejects_end_not_after_start(self) -> None:
        with pytest.raises(ValueError, match="must be after"):
            LoadRequest(
                symbol="X", timeframe=Timeframe.M1,
                start=pd.Timestamp("2024-01-02", tz=UTC), end=pd.Timestamp("2024-01-01", tz=UTC),
            )

    def test_rejects_unknown_columns(self) -> None:
        with pytest.raises(ValueError, match="Unknown column"):
            LoadRequest(
                symbol="X", timeframe=Timeframe.M1,
                start=pd.Timestamp("2024-01-01", tz=UTC), end=pd.Timestamp("2024-01-02", tz=UTC),
                columns=("not_a_real_column",),
            )


class TestLoad:
    def test_loads_exactly_the_requested_range(self, loaded_dataset) -> None:
        loader, df = loaded_dataset
        request = LoadRequest(
            symbol="XAUUSD", timeframe=Timeframe.M1, start=df["open_time"].iloc[10], end=df["open_time"].iloc[50]
        )
        result = loader.load(request)
        assert len(result) == 40
        assert list(result.columns) == [
            "open_time", "open", "high", "low", "close", "tick_volume", "real_volume", "spread",
        ]

    def test_result_is_in_strict_chronological_order(self, loaded_dataset) -> None:
        loader, df = loaded_dataset
        request = LoadRequest(symbol="XAUUSD", timeframe=Timeframe.M1, start=df["open_time"].iloc[0], end=df["open_time"].iloc[-1])
        result = loader.load(request)
        assert result["open_time"].is_monotonic_increasing
        assert not result["open_time"].duplicated().any()

    def test_column_subset(self, loaded_dataset) -> None:
        loader, df = loaded_dataset
        request = LoadRequest(
            symbol="XAUUSD", timeframe=Timeframe.M1, start=df["open_time"].iloc[10], end=df["open_time"].iloc[50],
            columns=("open_time", "close"),
        )
        result = loader.load(request)
        assert list(result.columns) == ["open_time", "close"]

    def test_returned_frame_is_a_defensive_copy(self, loaded_dataset) -> None:
        loader, df = loaded_dataset
        request = LoadRequest(symbol="XAUUSD", timeframe=Timeframe.M1, start=df["open_time"].iloc[10], end=df["open_time"].iloc[50])
        result = loader.load(request)
        result.loc[0, "close"] = -999.0
        reloaded = loader.load(request)
        assert reloaded["close"].iloc[0] != -999.0


class TestQualityGate:
    def test_strict_rejects_a_manifest_with_recorded_critical_issues(self, tmp_path) -> None:
        cstore = CanonicalStore(tmp_path)
        mstore = ManifestStore(tmp_path)
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 50)
        apply_incremental_update(
            cstore, mstore, df, symbol="XAUUSD", timeframe=Timeframe.H1, source_name="mt5", broker="B",
            pipeline_version="1.0.0", parent_snapshot_ids=(),
            requested_start=df["open_time"].iloc[0], requested_end=df["open_time"].iloc[-1] + pd.Timedelta(minutes=1),
            quality_summary={"critical": 2, "warning": 0},
        )
        loader = DatasetLoader(cstore, mstore)
        request = LoadRequest(
            symbol="XAUUSD", timeframe=Timeframe.H1, start=df["open_time"].iloc[0], end=df["open_time"].iloc[-1]
        )
        with pytest.raises(DataQualityError, match="critical quality issue"):
            loader.load(request)

    def test_lenient_loads_it_anyway(self, tmp_path) -> None:
        cstore = CanonicalStore(tmp_path)
        mstore = ManifestStore(tmp_path)
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 50)
        apply_incremental_update(
            cstore, mstore, df, symbol="XAUUSD", timeframe=Timeframe.H1, source_name="mt5", broker="B",
            pipeline_version="1.0.0", parent_snapshot_ids=(),
            requested_start=df["open_time"].iloc[0], requested_end=df["open_time"].iloc[-1] + pd.Timedelta(minutes=1),
            quality_summary={"critical": 2, "warning": 0},
        )
        loader = DatasetLoader(cstore, mstore)
        request = LoadRequest(
            symbol="XAUUSD", timeframe=Timeframe.H1, start=df["open_time"].iloc[0], end=df["open_time"].iloc[-1],
            required_quality="lenient",
        )
        result = loader.load(request)
        assert len(result) == 49


class TestVersionReproducibility:
    """Release-readiness-audit regression: `CanonicalStore` now stores
    immutable, content-addressed partition blobs (see
    `historical.canonical_store`'s module docstring), and every
    `DatasetManifest` records exactly which blob was current, per year, at
    save time. This proves an OLDER `dataset_version`, requested explicitly
    after the dataset has since been revised, is reconstructed byte-for-
    byte from what it actually recorded -- never silently served as
    whatever the dataset has since become."""

    def test_loading_the_current_version_explicitly_still_works(self, tmp_path) -> None:
        cstore = CanonicalStore(tmp_path)
        mstore = ManifestStore(tmp_path)
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 50)
        report = apply_incremental_update(
            cstore, mstore, df, symbol="XAUUSD", timeframe=Timeframe.M1, source_name="mt5", broker="B",
            pipeline_version="1.0.0", parent_snapshot_ids=(),
            requested_start=df["open_time"].iloc[0], requested_end=df["open_time"].iloc[-1] + pd.Timedelta(minutes=1),
        )
        loader = DatasetLoader(cstore, mstore)
        request = LoadRequest(
            symbol="XAUUSD", timeframe=Timeframe.M1, start=df["open_time"].iloc[0], end=df["open_time"].iloc[-1],
            dataset_version=report.manifest_version,
        )
        result = loader.load(request)  # must not raise
        assert len(result) == 49

    def test_requesting_an_older_version_after_a_revision_reconstructs_the_original_data(self, tmp_path) -> None:
        from quant_platform.historical.update_pipeline import RevisionPolicy

        cstore = CanonicalStore(tmp_path)
        mstore = ManifestStore(tmp_path)
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 50)
        common = {
            "symbol": "XAUUSD", "timeframe": Timeframe.M1, "source_name": "mt5", "broker": "B",
            "pipeline_version": "1.0.0", "parent_snapshot_ids": (),
        }
        first_report = apply_incremental_update(
            cstore, mstore, df,
            requested_start=df["open_time"].iloc[0], requested_end=df["open_time"].iloc[-1] + pd.Timedelta(minutes=1),
            **common,
        )
        v1 = first_report.manifest_version
        loader = DatasetLoader(cstore, mstore)
        v1_request = LoadRequest(
            symbol="XAUUSD", timeframe=Timeframe.M1, start=df["open_time"].iloc[0], end=df["open_time"].iloc[-1],
            dataset_version=v1,
        )
        v1_before_revision = loader.load(v1_request)

        conflict_df = df.iloc[[10]].copy()
        conflict_df["close"] = conflict_df["close"] + 100.0
        second_report = apply_incremental_update(
            cstore, mstore, conflict_df,
            requested_start=conflict_df["open_time"].iloc[0], requested_end=conflict_df["open_time"].iloc[0] + pd.Timedelta(minutes=1),
            revision_policy=RevisionPolicy.ACCEPT_NEWER_SOURCE, **common,
        )
        v2 = second_report.manifest_version
        assert v1 != v2

        # Reloading v1 explicitly, AFTER the revision, must give back the
        # exact original (unrevised) data -- byte-identical to what v1
        # returned before the revision ever happened.
        v1_after_revision = loader.load(v1_request)
        pd.testing.assert_frame_equal(v1_before_revision, v1_after_revision)
        original_close = df["close"].iloc[10]
        reloaded_close = v1_after_revision.loc[
            v1_after_revision["open_time"] == df["open_time"].iloc[10], "close"
        ].iloc[0]
        assert reloaded_close == original_close

        # The latest (unpinned) load must reflect the revision.
        latest = loader.load(LoadRequest(
            symbol="XAUUSD", timeframe=Timeframe.M1, start=df["open_time"].iloc[0], end=df["open_time"].iloc[-1],
        ))
        latest_close = latest.loc[latest["open_time"] == df["open_time"].iloc[10], "close"].iloc[0]
        assert latest_close == conflict_df["close"].iloc[0]
        assert latest_close != original_close

    def test_loading_without_specifying_a_version_always_succeeds(self, tmp_path) -> None:
        cstore = CanonicalStore(tmp_path)
        mstore = ManifestStore(tmp_path)
        df = _frame(datetime(2024, 1, 3, tzinfo=dt_timezone.utc), 50)
        apply_incremental_update(
            cstore, mstore, df, symbol="XAUUSD", timeframe=Timeframe.M1, source_name="mt5", broker="B",
            pipeline_version="1.0.0", parent_snapshot_ids=(),
            requested_start=df["open_time"].iloc[0], requested_end=df["open_time"].iloc[-1] + pd.Timedelta(minutes=1),
        )
        loader = DatasetLoader(cstore, mstore)
        request = LoadRequest(
            symbol="XAUUSD", timeframe=Timeframe.M1, start=df["open_time"].iloc[0], end=df["open_time"].iloc[-1],
        )
        result = loader.load(request)  # must not raise
        assert len(result) == 49


class TestMissingDataset:
    def test_raises_manifest_error_for_unknown_dataset(self, tmp_path) -> None:
        cstore = CanonicalStore(tmp_path)
        mstore = ManifestStore(tmp_path)
        loader = DatasetLoader(cstore, mstore)
        request = LoadRequest(
            symbol="NOPE", timeframe=Timeframe.M1, start=pd.Timestamp("2024-01-01", tz=UTC), end=pd.Timestamp("2024-01-02", tz=UTC)
        )
        with pytest.raises(ManifestError):
            loader.load(request)


class TestLoadForEngine:
    def test_projects_to_ohlcv_columns(self, loaded_dataset) -> None:
        loader, df = loaded_dataset
        request = LoadRequest(symbol="XAUUSD", timeframe=Timeframe.M1, start=df["open_time"].iloc[10], end=df["open_time"].iloc[50])
        engine_df = loader.load_for_engine(request)
        assert list(engine_df.columns) == list(OHLCV_COLUMNS)

    def test_volume_defaults_to_tick_volume(self, loaded_dataset) -> None:
        loader, df = loaded_dataset
        request = LoadRequest(symbol="XAUUSD", timeframe=Timeframe.M1, start=df["open_time"].iloc[10], end=df["open_time"].iloc[50])
        full = loader.load(request)
        engine_df = loader.load_for_engine(request)
        assert (engine_df["volume"] == full["tick_volume"]).all()

    def test_volume_source_is_explicit_and_selectable(self, loaded_dataset) -> None:
        loader, df = loaded_dataset
        request = LoadRequest(symbol="XAUUSD", timeframe=Timeframe.M1, start=df["open_time"].iloc[10], end=df["open_time"].iloc[50])
        full = loader.load(request)
        engine_df = loader.load_for_engine(request, volume_source="real_volume")
        assert (engine_df["volume"] == full["real_volume"]).all()

    def test_engine_frame_is_tz_aware_and_ordered(self, loaded_dataset) -> None:
        loader, df = loaded_dataset
        request = LoadRequest(symbol="XAUUSD", timeframe=Timeframe.M1, start=df["open_time"].iloc[10], end=df["open_time"].iloc[50])
        engine_df = loader.load_for_engine(request)
        assert engine_df["open_time"].dt.tz is not None
        assert engine_df["open_time"].is_monotonic_increasing
