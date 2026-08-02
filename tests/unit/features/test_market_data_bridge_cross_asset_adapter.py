"""`features.market_data_bridge.cross_asset_adapter`: pin verification,
conflicting-coordinate rejection, and the availability-shift adapter
(spec Section 7)."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest
from _market_data_bridge_test_helpers import BASE_TIME, make_cross_asset_fixture

from quant_platform.core.exceptions import SourceVerificationError
from quant_platform.core.types import OHLCV_COLUMNS, Timeframe
from quant_platform.features.alignment import align_higher_timeframe
from quant_platform.features.market_data_bridge.cross_asset_adapter import (
    resolve_cross_asset_dataframe,
    verify_cross_asset_binding,
)


class TestVerifyCrossAssetBinding:
    def test_resolves_expected_bar_count(self, tmp_path) -> None:
        fixture = make_cross_asset_fixture(tmp_path, days=10)
        bars = verify_cross_asset_binding(fixture.bar_store, fixture.manifest_store, fixture.binding)
        assert len(bars) == 10

    def test_stale_pin_fails_closed(self, tmp_path) -> None:
        from decimal import Decimal

        from quant_platform.market_data.collectors.cross_asset.instrument_form import InstrumentForm
        from quant_platform.market_data.collectors.cross_asset.market_record import create_market_driver_bar

        fixture = make_cross_asset_fixture(tmp_path, days=5)
        extra = create_market_driver_bar(
            canonical_driver_id="us_dollar_strength", provider="alpha_vantage", provider_symbol="UUP", instrument_form=InstrumentForm.ETF,
            open_time=BASE_TIME + timedelta(days=100), timeframe=Timeframe.D1, open=Decimal("29"), high=Decimal("29.5"), low=Decimal("28.5"),
            close=Decimal("29.2"), volume=Decimal("1"), volume_unit="shares", availability_time=BASE_TIME + timedelta(days=102),
            availability_policy_id="close_plus_conservative_delay", session_policy_id="nyse", adjustment_policy_id="raw_unadjusted",
            request_manifest_id="r" * 10, response_manifest_id="p" * 10, source_manifest_id="s" * 10, source_row_index=999,
        )
        fixture.bar_store.append_many_and_read_all("alpha_vantage", "us_dollar_strength", InstrumentForm.ETF, [extra])
        with pytest.raises(SourceVerificationError):
            verify_cross_asset_binding(fixture.bar_store, fixture.manifest_store, fixture.binding)

    def test_no_component_manifest_fails_closed(self, tmp_path) -> None:
        fixture = make_cross_asset_fixture(tmp_path, days=1)
        other_binding = replace(fixture.binding, mapping_id="does_not_exist", binding_id="")
        with pytest.raises(SourceVerificationError):
            verify_cross_asset_binding(fixture.bar_store, fixture.manifest_store, other_binding)


class TestResolveCrossAssetDataframe:
    def test_shape_and_columns(self, tmp_path) -> None:
        fixture = make_cross_asset_fixture(tmp_path, days=10)
        df = resolve_cross_asset_dataframe(fixture.bar_store, fixture.manifest_store, fixture.binding)
        assert list(df.columns) == list(OHLCV_COLUMNS)
        assert len(df) == 10
        assert df["open_time"].is_monotonic_increasing
        assert not df["open_time"].duplicated().any()

    def test_synthetic_open_time_matches_availability_minus_duration(self, tmp_path) -> None:
        fixture = make_cross_asset_fixture(tmp_path, days=3, extra_delay=timedelta(hours=2))
        df = resolve_cross_asset_dataframe(fixture.bar_store, fixture.manifest_store, fixture.binding)
        for row, bar in zip(df.itertuples(), sorted(fixture.bars, key=lambda b: b.open_time), strict=True):
            expected = bar.availability_time - Timeframe.D1.duration
            assert row.open_time.to_pydatetime() == expected

    def test_bar_not_revealed_before_true_availability_time(self, tmp_path) -> None:
        """The end-to-end PIT proof: feeding the resolved frame into
        `align_higher_timeframe` must reveal each bar exactly at its true
        `availability_time`, not at the naive `open_time + duration`
        close (spec Section 7's core requirement)."""
        import pandas as pd

        fixture = make_cross_asset_fixture(tmp_path, days=1, extra_delay=timedelta(hours=3))
        df = resolve_cross_asset_dataframe(fixture.bar_store, fixture.manifest_store, fixture.binding)
        true_availability = fixture.bars[0].availability_time
        just_before = pd.Series([pd.Timestamp(true_availability) - pd.Timedelta(seconds=1)])
        at_availability = pd.Series([pd.Timestamp(true_availability)])
        aligned_before = align_higher_timeframe(just_before, df, Timeframe.D1)
        aligned_at = align_higher_timeframe(at_availability, df, Timeframe.D1)
        assert aligned_before["htf_D1_bar_index"].iloc[0] == -1
        assert aligned_at["htf_D1_bar_index"].iloc[0] == 0
