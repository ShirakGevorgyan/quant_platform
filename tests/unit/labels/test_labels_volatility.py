from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelRequestError
from quant_platform.labels.volatility import (
    REALIZED_PARKINSON_ESTIMATOR_NAME,
    REALIZED_STDDEV_ESTIMATOR_NAME,
    realized_parkinson_estimator,
    realized_stddev_estimator,
    resolve_estimator_by_name,
)


class TestRealizedStddevEstimator:
    def test_trailing_only_warmup_is_nan(self, ohlcv_source_data: pd.DataFrame) -> None:
        # returns = close.pct_change() has 1 leading NaN of its own, so a
        # 20-period rolling std needs 21 closes (rows 0..20) before its
        # first valid value at row 20, not row 19.
        result = realized_stddev_estimator(ohlcv_source_data, window_bars=20)
        assert result.iloc[:20].isna().all()
        assert result.iloc[20:].notna().all()

    def test_non_negative(self, ohlcv_source_data: pd.DataFrame) -> None:
        result = realized_stddev_estimator(ohlcv_source_data, window_bars=20)
        assert (result.dropna() >= 0).all()

    def test_non_positive_window_rejected(self, ohlcv_source_data: pd.DataFrame) -> None:
        with pytest.raises(LabelRequestError):
            realized_stddev_estimator(ohlcv_source_data, window_bars=0)

    def test_missing_close_column_rejected(self) -> None:
        with pytest.raises(LabelRequestError):
            realized_stddev_estimator(pd.DataFrame({"open_time": pd.date_range("2024-01-01", periods=5, tz="UTC")}), window_bars=3)


class TestRealizedParkinsonEstimator:
    def test_trailing_only_warmup_is_nan(self, ohlcv_source_data: pd.DataFrame) -> None:
        result = realized_parkinson_estimator(ohlcv_source_data, window_bars=20)
        assert result.iloc[:19].isna().all()
        assert result.iloc[19:].notna().all()

    def test_non_negative(self, ohlcv_source_data: pd.DataFrame) -> None:
        result = realized_parkinson_estimator(ohlcv_source_data, window_bars=20)
        assert (result.dropna() >= 0).all()

    def test_missing_high_low_columns_rejected(self, source_data: pd.DataFrame) -> None:
        stripped = source_data.drop(columns=["high", "low"])
        with pytest.raises(LabelRequestError):
            realized_parkinson_estimator(stripped, window_bars=5)

    def test_genuinely_different_from_stddev_estimator(self, ohlcv_source_data: pd.DataFrame) -> None:
        stddev = realized_stddev_estimator(ohlcv_source_data, window_bars=20)
        parkinson = realized_parkinson_estimator(ohlcv_source_data, window_bars=20)
        assert not stddev.dropna().equals(parkinson.dropna())


class TestResolveEstimatorByName:
    def test_resolves_both_shipped_estimators(self) -> None:
        assert resolve_estimator_by_name(REALIZED_STDDEV_ESTIMATOR_NAME) is realized_stddev_estimator
        assert resolve_estimator_by_name(REALIZED_PARKINSON_ESTIMATOR_NAME) is realized_parkinson_estimator

    def test_unknown_name_rejected(self) -> None:
        with pytest.raises(LabelRequestError):
            resolve_estimator_by_name("not_a_real_estimator")
