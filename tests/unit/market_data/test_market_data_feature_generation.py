"""Unit tests for `market_data.feature_generation`: correctness of each
pure indicator function at its boundary window, determinism, and the
`generate_candle_features` driver's store-writing/skip-`None` behavior."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from quant_platform.core.exceptions import FeatureGenerationError
from quant_platform.core.types import Timeframe
from quant_platform.market_data.candles import create_candle
from quant_platform.market_data.feature_generation import (
    atr,
    ema,
    generate_candle_features,
    log_returns,
    price_delta,
    returns,
    rolling_mean,
    rolling_std,
    rsi,
    volume_delta,
    vwap,
    wick_ratios,
)
from quant_platform.market_data.feature_store import FeatureStore

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)


def _candle(hour: int, *, open_: str, high: str, low: str, close: str, volume: str | None = "10") -> object:
    return create_candle(
        instrument_id="mt5__XAUUSD", provider="mt5", symbol="XAUUSD", event_time=_T0 + timedelta(hours=hour), timeframe=Timeframe.H1,
        sequence=hour, open=Decimal(open_), high=Decimal(high), low=Decimal(low), close=Decimal(close),
        volume=(None if volume is None else Decimal(volume)),
    )


def _rising_candles(count: int) -> list[object]:
    candles = []
    price = Decimal("2000")
    for h in range(count):
        candles.append(_candle(h, open_=str(price), high=str(price + 5), low=str(price - 5), close=str(price + 1)))
        price += 1
    return candles


class TestReturnsAndLogReturns:
    def test_first_point_is_none(self) -> None:
        assert returns([Decimal("100"), Decimal("110")])[0] is None
        assert log_returns([Decimal("100"), Decimal("110")])[0] is None

    def test_simple_return_matches_hand_computed_value(self) -> None:
        result = returns([Decimal("100"), Decimal("110")])
        assert result[1] == Decimal("110") / Decimal("100") - 1

    def test_log_return_matches_hand_computed_value(self) -> None:
        result = log_returns([Decimal("100"), Decimal("110")])
        assert result[1] == (Decimal("110") / Decimal("100")).ln()


class TestPriceAndVolumeDelta:
    def test_price_delta(self) -> None:
        assert price_delta([Decimal("100"), Decimal("105")]) == [None, Decimal("5")]

    def test_volume_delta_none_safe(self) -> None:
        assert volume_delta([Decimal("10"), None, Decimal("20")]) == [None, None, None]


class TestRollingMeanAndStd:
    def test_none_before_window_is_full(self) -> None:
        result = rolling_mean([Decimal("1"), Decimal("2"), Decimal("3")], window=3)
        assert result[0] is None
        assert result[1] is None
        assert result[2] == Decimal(2)

    def test_exact_boundary_at_window_minus_one_is_still_none(self) -> None:
        values = [Decimal(str(i)) for i in range(5)]
        result = rolling_mean(values, window=5)
        assert result[3] is None
        assert result[4] == sum(values) / 5

    def test_rolling_std_matches_hand_computed_sample_std(self) -> None:
        values = [Decimal("2"), Decimal("4"), Decimal("4"), Decimal("4"), Decimal("5"), Decimal("5"), Decimal("7"), Decimal("9")]
        result = rolling_std(values, window=8)
        mean = sum(values) / 8
        variance = sum((v - mean) ** 2 for v in values) / 7
        assert result[7] == variance.sqrt()

    def test_rolling_std_requires_window_at_least_two(self) -> None:
        with pytest.raises(FeatureGenerationError):
            rolling_std([Decimal("1")], window=1)

    def test_non_positive_window_is_rejected(self) -> None:
        with pytest.raises(FeatureGenerationError):
            rolling_mean([Decimal("1")], window=0)


class TestEMA:
    def test_seeded_by_sma_at_the_boundary(self) -> None:
        values = [Decimal("1"), Decimal("2"), Decimal("3")]
        result = ema(values, window=3)
        assert result[2] == sum(values) / 3

    def test_none_before_window(self) -> None:
        assert ema([Decimal("1"), Decimal("2")], window=3) == [None, None]


class TestATR:
    def test_first_bar_true_range_is_its_own_high_low_range(self) -> None:
        candles = [_candle(0, open_="100", high="105", low="95", close="102")]
        result = atr(candles, window=1)
        assert result[0] == Decimal(10)

    def test_none_before_window_is_full(self) -> None:
        candles = _rising_candles(3)
        result = atr(candles, window=3)
        assert result[0] is None
        assert result[1] is None
        assert result[2] is not None


class TestRSI:
    def test_all_gains_yields_100(self) -> None:
        candles = _rising_candles(15)
        closes = [c.close for c in candles]
        result = rsi(closes, window=14)
        assert result[14] == Decimal(100)

    def test_flat_series_yields_neutral_50(self) -> None:
        closes = [Decimal("100")] * 15
        result = rsi(closes, window=14)
        assert result[14] == Decimal(50)


class TestVWAP:
    def test_matches_hand_computed_value_for_two_bars(self) -> None:
        candles = [
            _candle(0, open_="100", high="102", low="98", close="100", volume="10"),
            _candle(1, open_="100", high="104", low="96", close="102", volume="20"),
        ]
        result = vwap(candles)
        typical_0 = (Decimal("102") + Decimal("98") + Decimal("100")) / 3
        typical_1 = (Decimal("104") + Decimal("96") + Decimal("102")) / 3
        expected = (typical_0 * 10 + typical_1 * 20) / 30
        assert result[1] == expected

    def test_missing_volume_breaks_the_cumulative_series_from_that_point_on(self) -> None:
        candles = [
            _candle(0, open_="100", high="102", low="98", close="100", volume="10"),
            _candle(1, open_="100", high="104", low="96", close="102", volume=None),
            _candle(2, open_="100", high="104", low="96", close="102", volume="5"),
        ]
        result = vwap(candles)
        assert result[0] is not None
        assert result[1] is None
        assert result[2] is None


class TestWickRatios:
    def test_zero_range_candle_yields_zero_zero(self) -> None:
        candle = _candle(0, open_="100", high="100", low="100", close="100")
        assert wick_ratios([candle]) == [(Decimal(0), Decimal(0))]

    def test_ratios_sum_to_at_most_one(self) -> None:
        candle = _candle(0, open_="100", high="110", low="90", close="105")
        upper, lower = wick_ratios([candle])[0]
        assert upper + lower <= 1


class TestDeterminism:
    def test_calling_twice_with_identical_inputs_yields_identical_results(self) -> None:
        candles = _rising_candles(30)
        closes = [c.close for c in candles]
        assert rolling_mean(closes, window=10) == rolling_mean(closes, window=10)
        assert rsi(closes, window=14) == rsi(closes, window=14)
        assert atr(candles, window=14) == atr(candles, window=14)


class TestGenerateCandleFeaturesDriver:
    def test_generates_and_stores_every_default_feature(self) -> None:
        candles = _rising_candles(30)
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            records = generate_candle_features(candles, feature_version=1, store=store)
            assert records
            names = {r.feature_name for r in records}
            assert "return" in names
            assert "sma_20" in names
            assert "rsi_14" in names

    def test_warm_up_none_points_are_never_written(self) -> None:
        candles = _rising_candles(5)  # shorter than the default sma window (20)
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            records = generate_candle_features(candles, feature_version=1, store=store, feature_names=("sma",), windows={"sma": 20})
            assert records == []

    def test_re_running_over_the_same_candles_is_idempotent(self) -> None:
        candles = _rising_candles(30)
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            first = generate_candle_features(candles, feature_version=1, store=store)
            second = generate_candle_features(candles, feature_version=1, store=store)
            assert {r.feature_id for r in first} == {r.feature_id for r in second}

    def test_duplicate_event_time_is_rejected(self) -> None:
        candles = _rising_candles(3)
        duplicate = _candle(1, open_="2000", high="2005", low="1995", close="2001")
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            with pytest.raises(FeatureGenerationError):
                generate_candle_features([*candles, duplicate], feature_version=1, store=store)

    def test_mixed_instruments_are_rejected(self) -> None:
        candles = _rising_candles(3)
        other = create_candle(
            instrument_id="mt5__EURUSD", provider="mt5", symbol="EURUSD", event_time=_T0 + timedelta(hours=5), timeframe=Timeframe.H1,
            sequence=5, open=Decimal("1"), high=Decimal("2"), low=Decimal("1"), close=Decimal("1.5"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            with pytest.raises(FeatureGenerationError):
                generate_candle_features([*candles, other], feature_version=1, store=store)

    def test_unknown_feature_name_is_rejected(self) -> None:
        candles = _rising_candles(3)
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            with pytest.raises(FeatureGenerationError):
                generate_candle_features(candles, feature_version=1, store=store, feature_names=("not_a_real_feature",))

    def test_empty_candle_list_returns_no_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(Path(tmp))
            assert generate_candle_features([], feature_version=1, store=store) == []

    def test_out_of_order_input_is_sorted_before_generation(self) -> None:
        candles = _rising_candles(10)
        shuffled = [candles[3], candles[0], candles[7], candles[1], candles[2], candles[4], candles[5], candles[6], candles[8], candles[9]]
        with tempfile.TemporaryDirectory() as tmp:
            store_a = FeatureStore(Path(tmp) / "a")
            store_b = FeatureStore(Path(tmp) / "b")
            records_sorted = generate_candle_features(candles, feature_version=1, store=store_a, feature_names=("return",))
            records_shuffled = generate_candle_features(shuffled, feature_version=1, store=store_b, feature_names=("return",))
            assert {r.feature_id for r in records_sorted} == {r.feature_id for r in records_shuffled}
