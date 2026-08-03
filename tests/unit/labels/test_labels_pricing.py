from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelRequestError
from quant_platform.labels.pricing import PriceBasis, compute_forward_return, resolve_entry_exit_series


class TestResolveEntryExitSeries:
    def test_close_to_close(self, ohlcv_source_data: pd.DataFrame) -> None:
        entry, exit_ = resolve_entry_exit_series(ohlcv_source_data, PriceBasis.CLOSE_TO_CLOSE)
        pd.testing.assert_series_equal(entry, ohlcv_source_data["close"], check_names=False)
        pd.testing.assert_series_equal(exit_, ohlcv_source_data["close"], check_names=False)

    def test_open_to_close(self, ohlcv_source_data: pd.DataFrame) -> None:
        entry, exit_ = resolve_entry_exit_series(ohlcv_source_data, PriceBasis.OPEN_TO_CLOSE)
        pd.testing.assert_series_equal(entry, ohlcv_source_data["open"], check_names=False)
        pd.testing.assert_series_equal(exit_, ohlcv_source_data["close"], check_names=False)

    def test_close_to_open(self, ohlcv_source_data: pd.DataFrame) -> None:
        entry, exit_ = resolve_entry_exit_series(ohlcv_source_data, PriceBasis.CLOSE_TO_OPEN)
        pd.testing.assert_series_equal(entry, ohlcv_source_data["close"], check_names=False)
        pd.testing.assert_series_equal(exit_, ohlcv_source_data["open"], check_names=False)

    def test_mid_to_mid(self, ohlcv_source_data: pd.DataFrame) -> None:
        entry, exit_ = resolve_entry_exit_series(ohlcv_source_data, PriceBasis.MID_TO_MID)
        expected = (ohlcv_source_data["high"] + ohlcv_source_data["low"]) / 2.0
        pd.testing.assert_series_equal(entry, expected, check_names=False)
        pd.testing.assert_series_equal(exit_, expected, check_names=False)

    def test_missing_open_column_raises_for_open_basis(self, source_data: pd.DataFrame) -> None:
        with pytest.raises(LabelRequestError):
            resolve_entry_exit_series(source_data, PriceBasis.OPEN_TO_CLOSE)

    def test_close_to_close_does_not_require_open(self, source_data: pd.DataFrame) -> None:
        entry, _exit = resolve_entry_exit_series(source_data, PriceBasis.CLOSE_TO_CLOSE)
        assert len(entry) == len(source_data)


class TestComputeForwardReturn:
    def test_trailing_nan_tail(self, ohlcv_source_data: pd.DataFrame) -> None:
        result = compute_forward_return(ohlcv_source_data, PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5)
        assert result.iloc[:-5].notna().all()
        assert result.iloc[-5:].isna().all()

    def test_never_reads_beyond_configured_horizon(self, ohlcv_source_data: pd.DataFrame) -> None:
        close = ohlcv_source_data["close"]
        result = compute_forward_return(ohlcv_source_data, PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5)
        expected_row_10 = close.iloc[15] / close.iloc[10] - 1.0
        assert result.iloc[10] == pytest.approx(expected_row_10)

    def test_non_positive_horizon_rejected(self, ohlcv_source_data: pd.DataFrame) -> None:
        with pytest.raises(LabelRequestError):
            compute_forward_return(ohlcv_source_data, PriceBasis.CLOSE_TO_CLOSE, horizon_bars=0)

    def test_different_price_basis_gives_different_values(self, ohlcv_source_data: pd.DataFrame) -> None:
        close_to_close = compute_forward_return(ohlcv_source_data, PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5)
        open_to_close = compute_forward_return(ohlcv_source_data, PriceBasis.OPEN_TO_CLOSE, horizon_bars=5)
        assert not close_to_close.equals(open_to_close)
