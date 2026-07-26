"""Tests for `historical.source`: the broker-agnostic request/batch types
and the default `fetch_all` pagination helper."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.exceptions import SourceError
from quant_platform.core.types import Timeframe
from quant_platform.historical.source import HistoricalSource, SourceBatch, SourceMetadata, SourceRequest

UTC = "UTC"


def _metadata() -> SourceMetadata:
    return SourceMetadata(
        source_name="fake", source_version="1", broker="TestBroker",
        source_symbol="XAUUSDm", extracted_at=pd.Timestamp.now(tz=UTC),
    )


def _bars(times: list[str]) -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime(times, utc=True),
            "open": np.full(n, 2000.0),
            "high": np.full(n, 2001.0),
            "low": np.full(n, 1999.0),
            "close": np.full(n, 2000.5),
            "tick_volume": np.full(n, 100, dtype=np.int64),
            "real_volume": np.zeros(n, dtype=np.int64),
            "spread": np.full(n, 15, dtype=np.int64),
        }
    )


class _PagedFakeSource(HistoricalSource):
    """Splits whatever range it's asked for into fixed-size pages of M1
    bars, marking every page but the last as partial -- a minimal,
    hand-controlled `HistoricalSource` used only to prove `fetch_all`'s
    pagination/dedup/forward-progress contract independent of the MT5
    adapter."""

    def __init__(self, *, page_size: int = 5, extra_rows: int = 0, force_empty: bool = False) -> None:
        self.page_size = page_size
        self.extra_rows = extra_rows  # simulate a misbehaving source re-returning a seam row
        self.force_empty = force_empty
        self.calls: list[SourceRequest] = []

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    def fetch(self, request: SourceRequest) -> SourceBatch:
        self.calls.append(request)
        if self.force_empty:
            return SourceBatch(
                data=_bars([]), metadata=_metadata(), requested_start=request.start,
                requested_end=request.end, is_partial=False,
            )
        all_open_times = pd.date_range(request.start, request.end, freq="1min", tz="UTC", inclusive="left")
        page_open_times = all_open_times[: self.page_size + self.extra_rows]
        if self.extra_rows and self.calls[0] is not self.calls[-1]:
            # On every call after the first, re-prepend the seam bar to
            # simulate a source that (incorrectly) re-returns the boundary.
            page_open_times = pd.DatetimeIndex(
                [request.start - pd.Timedelta(minutes=1)]
            ).append(page_open_times[: self.page_size])
        data = _bars([str(t) for t in page_open_times])
        is_partial = len(all_open_times) > len(page_open_times)
        return SourceBatch(
            data=data, metadata=_metadata(), requested_start=request.start, requested_end=request.end,
            is_partial=is_partial,
        )


class TestSourceRequestValidation:
    def _req(self, **overrides):
        base = {
            "symbol": "XAUUSDm", "timeframe": Timeframe.M1,
            "start": pd.Timestamp("2024-01-01", tz=UTC), "end": pd.Timestamp("2024-01-02", tz=UTC),
        }
        base.update(overrides)
        return SourceRequest(**base)

    def test_rejects_empty_symbol(self) -> None:
        with pytest.raises(ValueError, match="symbol must not be empty"):
            self._req(symbol="")

    def test_rejects_naive_start(self) -> None:
        with pytest.raises(SourceError, match="tz-aware UTC"):
            self._req(start=pd.Timestamp("2024-01-01"))

    def test_rejects_naive_end(self) -> None:
        with pytest.raises(SourceError, match="tz-aware UTC"):
            self._req(end=pd.Timestamp("2024-01-02"))

    def test_rejects_end_not_after_start(self) -> None:
        with pytest.raises(ValueError, match="must be strictly after"):
            self._req(start=pd.Timestamp("2024-01-02", tz=UTC), end=pd.Timestamp("2024-01-02", tz=UTC))

    def test_rejects_non_positive_max_batch_size(self) -> None:
        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            self._req(max_batch_size=0)


class TestFetchAllPagination:
    def _request(self, **overrides) -> SourceRequest:
        base = {
            "symbol": "XAUUSDm", "timeframe": Timeframe.M1,
            "start": pd.Timestamp("2024-01-01T00:00:00", tz=UTC),
            "end": pd.Timestamp("2024-01-01T00:20:00", tz=UTC),
        }
        base.update(overrides)
        return SourceRequest(**base)

    def test_pages_through_entire_range_without_duplicates_or_gaps(self) -> None:
        source = _PagedFakeSource(page_size=5)
        batches = list(source.fetch_all(self._request()))
        assert len(batches) == 4  # 20 minutes / 5-minute pages
        all_times = pd.concat([b.data["open_time"] for b in batches]).reset_index(drop=True)
        assert len(all_times) == 20
        assert all_times.is_monotonic_increasing
        assert not all_times.duplicated().any()
        assert all_times.iloc[0] == self._request().start
        assert all_times.iloc[-1] == self._request().end - timedelta(minutes=1)

    def test_last_page_is_not_marked_partial(self) -> None:
        source = _PagedFakeSource(page_size=5)
        batches = list(source.fetch_all(self._request()))
        assert [b.is_partial for b in batches] == [True, True, True, False]

    def test_empty_result_stops_pagination(self) -> None:
        source = _PagedFakeSource(page_size=5, force_empty=True)
        batches = list(source.fetch_all(self._request()))
        assert batches == []

    def test_out_of_order_batch_is_rejected(self) -> None:
        """Adversarial-audit regression: a source whose own returned batch
        is internally out of order must not be trusted to derive the next
        page boundary from -- `fetch_all` used to take `iloc[-1]` as the
        batch's maximum timestamp unconditionally, which an out-of-order
        batch would silently violate, corrupting pagination rather than
        raising. Found by adversarial review, not a failing test."""

        class _ScrambledSource(HistoricalSource):
            def connect(self) -> None:
                pass

            def disconnect(self) -> None:
                pass

            def is_connected(self) -> bool:
                return True

            def fetch(self, request: SourceRequest) -> SourceBatch:
                scrambled_times = [
                    str(request.start + timedelta(minutes=2)),
                    str(request.start),
                    str(request.start + timedelta(minutes=1)),
                ]
                return SourceBatch(
                    data=_bars(scrambled_times), metadata=_metadata(),
                    requested_start=request.start, requested_end=request.end, is_partial=False,
                )

        with pytest.raises(SourceError, match="out-of-order"):
            list(_ScrambledSource().fetch_all(self._request()))

    def test_seam_duplicate_is_trimmed_not_double_counted(self) -> None:
        # A source that mis-returns the boundary bar on every call after
        # the first must still yield a monotonic, duplicate-free series.
        source_with_seam = _PagedFakeSource(page_size=5)
        source_with_seam.extra_rows = 1
        batches = list(source_with_seam.fetch_all(self._request()))
        all_times = pd.concat([b.data["open_time"] for b in batches]).reset_index(drop=True)
        assert not all_times.duplicated().any()
        assert all_times.is_monotonic_increasing

    def test_source_stuck_re_returning_only_the_seam_bar_raises_rather_than_looping_forever(self) -> None:
        class _StuckSource(HistoricalSource):
            """Always answers with the single bar at the ORIGINAL request
            start, regardless of the (advancing) `request.start` it is
            actually asked for, while claiming more data remains. This is
            the only way to make zero forward progress under `fetch_all`'s
            contract: every row at or after the requested `cursor_start` is
            trusted verbatim (only rows strictly BEFORE `cursor_start` are
            ever trimmed as seam duplicates), so a source that keeps
            re-answering with a row that falls further and further behind
            the advancing cursor eventually has nothing left after
            trimming -- exactly the "no new rows, but is_partial=True"
            case `fetch_all` must refuse to spin on forever.
            """

            def __init__(self) -> None:
                self.fixed_bar_time = pd.Timestamp("2024-01-01T00:00:00", tz=UTC)

            def connect(self) -> None:
                pass

            def disconnect(self) -> None:
                pass

            def is_connected(self) -> bool:
                return True

            def fetch(self, request: SourceRequest) -> SourceBatch:
                return SourceBatch(
                    data=_bars([str(self.fixed_bar_time)]), metadata=_metadata(),
                    requested_start=request.start, requested_end=request.end, is_partial=True,
                )

        with pytest.raises(SourceError, match="cannot make forward progress"):
            list(_StuckSource().fetch_all(self._request()))
