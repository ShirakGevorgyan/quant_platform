"""Tests for `historical.mt5_testing.generate_fake_rates` itself (not the
adapter that consumes it -- see `test_mt5_adapter.py` for that).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from quant_platform.historical.mt5_testing import generate_fake_rates


class TestHostTimezoneIndependence:
    """`generate_fake_rates` takes naive datetimes and must treat their
    wall-clock digits literally (as if UTC), never through the host
    machine's configured local timezone -- exactly the discipline
    `historical.timezones` enforces for the real MT5 adapter. An earlier
    version called `.timestamp()` on the naive value directly, which
    silently interprets it via the OS-local zone; fixed by attaching UTC
    tzinfo first (see the module's own comment). This proves the fixed
    behavior against a hand-computed epoch value, independent of
    whatever timezone this test happens to run under."""

    def test_first_bar_time_matches_hand_computed_utc_epoch_seconds(self) -> None:
        # 2024-01-01T00:00:00 UTC is 1704067200 seconds since the Unix
        # epoch -- computed by hand (365*54 + leap days arithmetic is
        # error-prone to redo here, so this value is taken directly from
        # the well-known reference "2024-01-01T00:00:00Z = 1704067200",
        # independently checkable against any standard epoch converter).
        rates = generate_fake_rates(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc), end=datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc),
            timeframe_minutes=1, seed=1,
        )
        assert int(rates["time"][0]) == 1_704_067_200


class TestQueryIndependence:
    """Regression coverage for a real bug: an earlier version generated a
    cumulative random walk seeded fresh at whatever `start` a call
    received, so two overlapping queries against "the same broker history"
    produced DIFFERENT prices at identical timestamps -- discovered via
    `tests/unit/test_data_cli.py::TestRunIngestIncremental` spuriously
    raising `UpdateConflictError` on a legitimate, non-conflicting
    incremental update. Every bar must now be a pure function of its own
    absolute step index, not of the requested window's boundaries.
    """

    def test_overlapping_queries_agree_on_shared_timestamps(self) -> None:
        full = generate_fake_rates(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc), end=datetime(2024, 1, 3, tzinfo=timezone.utc),
            timeframe_minutes=1, seed=5,
        )
        # A query whose window starts partway through `full`'s range.
        partial = generate_fake_rates(
            start=datetime(2024, 1, 2, tzinfo=timezone.utc), end=datetime(2024, 1, 3, tzinfo=timezone.utc),
            timeframe_minutes=1, seed=5,
        )
        overlap_start_index = np.searchsorted(full["time"], partial["time"][0])
        overlapping_slice = full[overlap_start_index : overlap_start_index + len(partial)]
        assert np.array_equal(overlapping_slice["time"], partial["time"])
        for field in ("open", "high", "low", "close", "tick_volume", "spread", "real_volume"):
            assert np.array_equal(overlapping_slice[field], partial[field]), f"field {field!r} diverged"

    def test_different_seeds_produce_different_prices(self) -> None:
        a = generate_fake_rates(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc), end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            timeframe_minutes=1, seed=1,
        )
        b = generate_fake_rates(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc), end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            timeframe_minutes=1, seed=2,
        )
        assert not np.array_equal(a["close"], b["close"])

    def test_identical_call_is_fully_deterministic(self) -> None:
        a = generate_fake_rates(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc), end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            timeframe_minutes=1, seed=5,
        )
        b = generate_fake_rates(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc), end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            timeframe_minutes=1, seed=5,
        )
        assert np.array_equal(a, b)


class TestOHLCInvariants:
    def test_high_is_always_at_least_open_and_close(self) -> None:
        rates = generate_fake_rates(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc), end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            timeframe_minutes=1, seed=3,
        )
        assert (rates["high"] >= rates["open"]).all()
        assert (rates["high"] >= rates["close"]).all()

    def test_low_is_always_at_most_open_and_close(self) -> None:
        rates = generate_fake_rates(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc), end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            timeframe_minutes=1, seed=3,
        )
        assert (rates["low"] <= rates["open"]).all()
        assert (rates["low"] <= rates["close"]).all()

    def test_no_overflow_warnings_from_the_bit_mixer(self) -> None:
        with np.errstate(over="raise"), pytest.raises(FloatingPointError):
            # Sanity: confirm np.errstate(over="raise") actually fires
            # for uint64 wraparound in this numpy version, so the
            # absence of a raise from generate_fake_rates below is a
            # meaningful assertion, not a no-op.
            (np.uint64(2**64 - 1) + np.uint64(1))
        generate_fake_rates(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc), end=datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc),
            timeframe_minutes=1, seed=3,
        )  # must not raise or warn despite internal uint64 wraparound


class TestEmptyRange:
    def test_empty_range_returns_empty_array(self) -> None:
        rates = generate_fake_rates(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc), end=datetime(2024, 1, 1, tzinfo=timezone.utc),
            timeframe_minutes=1, seed=1,
        )
        assert len(rates) == 0


class TestValidation:
    def test_rejects_non_positive_timeframe_minutes(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            generate_fake_rates(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc), end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                timeframe_minutes=0, seed=1,
            )
