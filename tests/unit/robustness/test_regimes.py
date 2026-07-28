"""Milestone 6, Section 16: leakage-safe regime classification and
insufficient-sample handling. Every classifier is checked against a
synthetic series with a KNOWN trend/session/quantile shape, and the
trailing-window classifiers are checked for LEAKAGE PREVENTION: a bar
with insufficient trailing history must be entirely absent from the
returned mapping, never coerced to a default label."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.robustness.models import RegimeDimensionKind
from quant_platform.robustness.regimes import (
    _BarRecord,
    _classify_calendar,
    _classify_price_regime,
    _classify_quantile_dimension,
    _classify_trend_direction,
    _compute_bucket,
    _session_label,
)

N = 80
_START = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")  # a Monday


def _synthetic_bars() -> pd.DataFrame:
    open_times = [_START + pd.Timedelta(hours=i) for i in range(N)]
    closes = []
    price = 100.0
    for i in range(N):
        price *= 1.002 if i < N // 2 else 0.998
        closes.append(price)
    volumes = [1000.0 + (i % 5) * 100.0 for i in range(N)]
    return pd.DataFrame({"open_time": open_times, "open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes})


class TestSessionBoundaries:
    @pytest.mark.parametrize(
        ("hour", "expected"),
        [(0, "asia"), (6, "asia"), (7, "london"), (12, "london"), (13, "london_ny_overlap"), (15, "london_ny_overlap"), (16, "new_york"), (20, "new_york"), (21, "other"), (23, "other")],
    )
    def test_hour_maps_to_expected_session(self, hour: int, expected: str) -> None:
        assert _session_label(hour) == expected


class TestCalendarClassificationLeakageSafety:
    def test_day_of_week_and_hour_use_only_the_bars_own_timestamp(self) -> None:
        bars = _synthetic_bars()
        positions = list(range(N))
        dow = _classify_calendar(RegimeDimensionKind.DAY_OF_WEEK, positions, bars=bars)
        hour = _classify_calendar(RegimeDimensionKind.HOUR_OF_DAY, positions, bars=bars)
        assert dow[0] == "Monday"
        assert hour[5] == "05"
        assert len(dow) == N and len(hour) == N  # every bar classified -- no trailing history needed


class TestTrendAndPriceRegimeLeakagePrevention:
    def test_bars_before_the_trailing_window_are_absent_not_defaulted(self) -> None:
        """`window=10` -> positions 0..9 have insufficient trailing
        history and must be ENTIRELY ABSENT from the returned mapping,
        never silently classified with a placeholder label."""
        bars = _synthetic_bars()
        labels = _classify_trend_direction(list(range(N)), bars=bars, window=10)
        assert all(pos not in labels for pos in range(10))
        assert 10 in labels  # exactly at the window boundary IS classified

    def test_uptrend_then_downtrend_correctly_classified(self) -> None:
        bars = _synthetic_bars()
        trend = _classify_trend_direction(list(range(N)), bars=bars, window=10)
        assert trend[39] == "up"
        assert trend[79] == "down"
        price_regime = _classify_price_regime(list(range(N)), bars=bars, window=20)
        assert price_regime[39] == "bull"
        assert price_regime[79] == "bear"


class TestQuantileClassificationLeakageSafety:
    def test_expanding_rank_only_uses_values_observed_at_or_before(self) -> None:
        """Monotonically increasing series: every new value IS the
        maximum of everything seen so far by construction -- both the
        first (trivially) and last (globally maximal) points must land
        in the top bucket, a mathematically forced consequence of an
        EXPANDING (never a full-series) ranking window."""
        metric = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        labels = _classify_quantile_dimension(list(range(6)), metric=metric, n_quantiles=2)
        assert labels[0] == "q2_of_2"
        assert labels[5] == "q2_of_2"

    def test_discriminates_a_later_small_value_into_a_low_bucket(self) -> None:
        """[10, 20, 30, 1, 40], n_quantiles=4: at position 3, value=1.0 is
        the smallest of {10,20,30,1} seen so far -> rank=1/4=0.25 ->
        bucket_index=min(3, int(0.25*4))=min(3,1)=1 -> 'q2_of_4', NOT the
        top bucket -- proves the ranking genuinely discriminates rather
        than always reporting the top bucket."""
        metric = pd.Series([10.0, 20.0, 30.0, 1.0, 40.0])
        labels = _classify_quantile_dimension(list(range(5)), metric=metric, n_quantiles=4)
        assert labels[3] == "q2_of_4"
        assert labels[4] == "q4_of_4"  # globally maximal so far -> top bucket


def _flat_then_jump_bars(*, flat_bars: int = 50, jump_bars: int = 30) -> pd.DataFrame:
    """Constant price for `flat_bars`, then a sharp, sustained jump for
    `jump_bars` more -- a leakage probe: any classifier that peeks past
    the bar being classified will report the POST-jump regime for a bar
    that occurred strictly BEFORE the jump; a correctly trailing-only
    classifier cannot, because nothing has moved yet as of that bar."""
    open_times = [_START + pd.Timedelta(hours=i) for i in range(flat_bars + jump_bars)]
    closes = [100.0] * flat_bars + [100.0 * (1.05**i) for i in range(1, jump_bars + 1)]
    volumes = [1000.0] * (flat_bars + jump_bars)
    return pd.DataFrame({"open_time": open_times, "open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes})


class TestRegimeLeakageMutationDetection:
    """Release-audit Section 21: explicit mutation tests reintroducing
    each specific known leakage bug class and confirming the CURRENT
    (correct) classifiers behave differently from -- i.e. detectably do
    NOT reproduce -- the leaky variant on a series constructed so the
    leak would change the answer."""

    def test_centered_window_would_leak_but_trailing_window_does_not(self) -> None:
        """A CENTERED window (`[pos - half, pos + half]`) at a bar just
        before a future price jump would incorrectly see the jump; the
        real, backward-only `_classify_trend_direction` must not."""
        bars = _flat_then_jump_bars()
        window = 10
        probe_pos = 49  # last flat bar; window [39, 49] is entirely flat
        correct = _classify_trend_direction([probe_pos], bars=bars, window=window)
        assert correct[probe_pos] == "flat", "trailing-only classification of an entirely-flat window must report 'flat'"

        def leaky_centered_trend(bar_positions: list[int], *, bars: pd.DataFrame, window: int) -> dict[int, str]:
            close = bars["close"]
            half = window // 2
            labels: dict[int, str] = {}
            for pos in bar_positions:
                lo, hi = pos - half, pos + half
                if lo < 0 or hi >= len(close):
                    continue
                trailing_return = float(close.iloc[hi]) / float(close.iloc[lo]) - 1.0
                labels[pos] = "up" if trailing_return > 0.0 else ("down" if trailing_return < 0.0 else "flat")
            return labels

        leaky = leaky_centered_trend([probe_pos], bars=bars, window=window)
        assert leaky[probe_pos] == "up", "sanity: the leaky centered classifier DOES see the future jump from this probe position"
        assert correct[probe_pos] != leaky[probe_pos], "the correct trailing classifier must diverge from the leaky centered one exactly where the leak would matter"

    def test_next_bar_label_shift_would_leak_but_current_bar_does_not(self) -> None:
        """An off-by-one variant that reads `close.iloc[pos+1]` instead of
        `close.iloc[pos]` as the window's endpoint leaks exactly one bar
        of future information -- detectable at the precise boundary bar
        where the jump begins."""
        bars = _flat_then_jump_bars()
        window = 10
        probe_pos = 49
        correct = _classify_price_regime([probe_pos], bars=bars, window=window)
        assert correct[probe_pos] == "sideways"

        def leaky_next_bar_price_regime(bar_positions: list[int], *, bars: pd.DataFrame, window: int) -> dict[int, str]:
            close = bars["close"]
            labels: dict[int, str] = {}
            for pos in bar_positions:
                shifted = pos + 1  # BUG: reads one bar past the classified bar
                if shifted - window < 0 or shifted >= len(close):
                    continue
                trailing_return = float(close.iloc[shifted]) / float(close.iloc[shifted - window]) - 1.0
                if trailing_return > 0.005:
                    labels[pos] = "bull"
                elif trailing_return < -0.005:
                    labels[pos] = "bear"
                else:
                    labels[pos] = "sideways"
            return labels

        leaky = leaky_next_bar_price_regime([probe_pos], bars=bars, window=window)
        assert leaky[probe_pos] == "bull", "sanity: the leaky next-bar classifier DOES see one bar of future jump from this probe position"
        assert correct[probe_pos] != leaky[probe_pos]

    def test_future_inclusive_quantile_would_leak_but_expanding_rank_does_not(self) -> None:
        """A FULL-SAMPLE (rather than expanding, at-or-before) quantile
        rank at an EARLY position would already reflect values that have
        not been observed yet -- detectable because the correct expanding
        rank of the first ascending value is trivially always the top
        bucket (nothing to compare against yet), while a full-sample rank
        of that same early, small value against the WHOLE (larger, later)
        series lands in a low bucket."""
        metric = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
        correct = _classify_quantile_dimension([0], metric=metric, n_quantiles=4)
        assert correct[0] == "q4_of_4", "expanding rank of the FIRST observation is trivially the top bucket -- nothing precedes it"

        def leaky_full_sample_quantile(bar_positions: list[int], *, metric: pd.Series, n_quantiles: int) -> dict[int, str]:
            all_values = sorted(float(v) for v in metric if pd.notna(v))
            labels: dict[int, str] = {}
            for pos in bar_positions:
                value = float(metric.iloc[pos])
                rank = sum(1 for v in all_values if v <= value) / len(all_values)  # BUG: ranks against the FULL series, including future bars
                labels[pos] = f"q{min(n_quantiles - 1, int(rank * n_quantiles)) + 1}_of_{n_quantiles}"
            return labels

        leaky = leaky_full_sample_quantile([0], metric=metric, n_quantiles=4)
        assert leaky[0] == "q1_of_4", "sanity: the leaky full-sample classifier ranks the smallest-ever value into the bottom bucket using future context"
        assert correct[0] != leaky[0]


class TestBucketMetricsHandComputed:
    def test_total_return_and_drawdown_match_hand_computed_synthetic_equity_curve(self) -> None:
        """net returns [0.02, -0.01, 0.03, -0.02] -> equity path 1.00 ->
        1.02 -> 1.0098 -> 1.040394 -> 1.01958612; max drawdown is the
        larger of the two intra-path peak-to-trough drops, computed here
        independently via the identical elementary compounding formula."""
        bar_records = {
            0: _BarRecord(net_return=0.02, gross_return=0.02, transaction_costs=0.0, total_absolute_exposure=1.0, benchmark_bar_return=0.0),
            1: _BarRecord(net_return=-0.01, gross_return=-0.01, transaction_costs=0.0, total_absolute_exposure=1.0, benchmark_bar_return=0.0),
            2: _BarRecord(net_return=0.03, gross_return=0.03, transaction_costs=0.0, total_absolute_exposure=1.0, benchmark_bar_return=0.0),
            3: _BarRecord(net_return=-0.02, gross_return=-0.02, transaction_costs=0.0, total_absolute_exposure=1.0, benchmark_bar_return=0.0),
        }
        equity, peak, expected_max_dd = 1.0, 1.0, 0.0
        for r in (0.02, -0.01, 0.03, -0.02):
            equity *= 1.0 + r
            peak = max(peak, equity)
            expected_max_dd = max(expected_max_dd, (peak - equity) / peak)

        result = _compute_bucket("test_label", [0, 1, 2, 3], bar_records=bar_records, trades_by_entry_position={}, periods_per_year=252.0, minimum_regime_samples=2)
        assert result.skipped is False
        assert result.total_net_return == pytest.approx(0.02 - 0.01 + 0.03 - 0.02, abs=1e-12)
        assert result.maximum_drawdown == pytest.approx(expected_max_dd, abs=1e-12)
        assert result.trade_count == 0 and result.hit_rate is None

    def test_insufficient_samples_are_explicitly_skipped_not_dropped(self) -> None:
        bar_records = {0: _BarRecord(net_return=0.01, gross_return=0.01, transaction_costs=0.0, total_absolute_exposure=1.0, benchmark_bar_return=0.0)}
        result = _compute_bucket("rare_label", [0], bar_records=bar_records, trades_by_entry_position={}, periods_per_year=252.0, minimum_regime_samples=5)
        assert result.skipped is True
        assert result.total_net_return is None
        assert "insufficient samples" in (result.skip_reason or "")
        assert result.observation_count == 1  # the bucket itself is still reported, never dropped
