"""Broker-agnostic historical market-data source protocol.

Every concrete source (the real MT5 adapter, a fake MT5 double for tests,
and any future venue) implements `HistoricalSource`, so nothing above this
layer -- normalization, validation, storage, resampling, the loader -- ever
depends on a specific vendor's SDK types, quirks, or connection model. This
mirrors the existing `data.interfaces.DataSource` split for canonical-file
reading (Open/Closed: new venues are added by implementing this interface,
never by modifying the pipeline), but is a distinct interface from it: a
`DataSource` reads already-canonical bars back out of local CSV/Parquet for
backtest consumption; a `HistoricalSource` performs the original extraction
of raw historical data from a live broker/vendor connection, with the
connection lifecycle, pagination, and provenance concerns a file read
simply doesn't have.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, replace

import pandas as pd

from quant_platform.core.exceptions import SourceError
from quant_platform.core.types import Timeframe
from quant_platform.historical.timezones import require_utc

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceRequest:
    """A single historical-data request. `symbol` is the SOURCE-side
    symbol (the broker's own alias, e.g. "XAUUSDm") -- mapping from the
    platform's canonical symbol to this is the caller's/adapter config's
    responsibility, not this request's; keeping that mapping out of the
    request type is what stops broker-specific symbol quirks leaking into
    the rest of the pipeline.

    `start` is inclusive, `end` is exclusive, both required to already be
    tz-aware UTC -- this request type makes no timezone decision itself;
    by the time a `SourceRequest` exists, the caller has already resolved
    what UTC range it wants.
    """

    symbol: str
    timeframe: Timeframe
    start: pd.Timestamp
    end: pd.Timestamp
    max_batch_size: int = 5_000

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise SourceError(
                "SourceRequest.start/end must be tz-aware UTC timestamps; naive "
                "timestamps are ambiguous about which real-world instant is being "
                "requested.",
                context={"start": str(self.start), "end": str(self.end)},
            )
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be strictly after start ({self.start})")
        if self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {self.max_batch_size}")


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Provenance for one extraction call, independent of which concrete
    `HistoricalSource` produced it."""

    source_name: str
    source_version: str
    broker: str
    source_symbol: str
    extracted_at: pd.Timestamp

    def __post_init__(self) -> None:
        require_utc(pd.Series([self.extracted_at]), context="SourceMetadata.extracted_at")


@dataclass(frozen=True, slots=True)
class SourceBatch:
    """One chunk of historical bars, in `historical.models.RAW_HISTORICAL_COLUMNS`
    schema, together with its provenance and the request that produced it."""

    data: pd.DataFrame
    metadata: SourceMetadata
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    is_partial: bool = False


class HistoricalSource(ABC):
    """Broker-agnostic historical-data extraction interface.

    Lifecycle: `connect()` must be called before `fetch()`/`fetch_all()`,
    and `disconnect()` should be called when done. Implementations that
    wrap a shared/externally-owned connection (e.g. a single MT5 terminal
    session another part of the process also uses) must document their own
    ownership semantics rather than unconditionally tearing down a
    connection they did not establish -- see `historical.mt5_adapter` for
    the concrete policy this platform uses.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish the underlying connection. Raise `SourceError` (or a
        `MissingDependencyError` if an optional package is absent) on
        failure -- never return successfully without a usable connection."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_connected(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def fetch(self, request: SourceRequest) -> SourceBatch:
        """Fetch a single batch of at most `request.max_batch_size` rows
        starting at `request.start`. Must set `SourceBatch.is_partial=True`
        whenever `[request.start, request.end)` contains more rows than
        were returned, so `fetch_all` (or a caller's own loop) knows to
        continue paginating. Must return an empty (zero-row) `SourceBatch`
        -- not raise -- when the source genuinely has no data in range;
        raising is reserved for actual failures (disconnection, invalid
        symbol, unavailable timeframe, malformed response)."""
        raise NotImplementedError

    def fetch_all(self, request: SourceRequest) -> Iterator[SourceBatch]:
        """Page through `[request.start, request.end)` by repeatedly
        calling `fetch`, advancing the requested start boundary to exactly
        one bar past the last row actually returned each time. This is the
        source-agnostic pagination/dedup mechanism every concrete source
        gets for free; a source with a more efficient native pagination
        primitive may override this, but must preserve the same
        no-duplicate, no-infinite-loop, forward-progress guarantees this
        implementation proves (see
        `tests/unit/historical/test_source.py::TestFetchAllPagination`).
        """
        cursor_start = request.start
        while cursor_start < request.end:
            sub_request = replace(request, start=cursor_start)
            batch = self.fetch(sub_request)

            if len(batch.data) == 0:
                return

            # A batch's own rows must be internally ordered before this
            # function trusts `iloc[-1]` as its maximum timestamp (used
            # below to compute the next page's start) -- an out-of-order
            # batch from a misbehaving source would otherwise silently
            # corrupt the pagination cursor (e.g. computing `next_start`
            # from a row that isn't actually the latest one, which could
            # cause rows to be skipped or re-fetched forever without ever
            # tripping the forward-progress guard). Found via adversarial
            # review, not by a failing test -- see
            # `TestFetchAllPagination::test_out_of_order_batch_is_rejected`.
            if not batch.data["open_time"].is_monotonic_increasing:
                raise SourceError(
                    "fetch_all: source returned an internally out-of-order batch "
                    "(open_time is not monotonically increasing); cannot safely "
                    "determine the next page boundary from it",
                    context={"symbol": request.symbol, "cursor_start": str(cursor_start)},
                )

            # Defensive trim: a misbehaving source that (despite our
            # start-exclusive-of-the-prior-batch's-last-bar advance below)
            # re-returns the seam bar would otherwise duplicate one row at
            # every page boundary. This is the one exact-duplicate case
            # provably safe to drop unconditionally (identical timestamp,
            # identical bar, caused purely by our own re-request boundary)
            # -- see `historical.repair` for the broader policy governing
            # every other kind of duplicate.
            seam_mask = batch.data["open_time"] < cursor_start
            if seam_mask.any():
                logger.warning(
                    "fetch_all: dropping %d row(s) at pagination seam before %s (source_symbol=%s)",
                    int(seam_mask.sum()), cursor_start, request.symbol,
                )
                trimmed = batch.data.loc[~seam_mask].reset_index(drop=True)
                batch = replace(batch, data=trimmed)
                if len(batch.data) == 0:
                    if not batch.is_partial:
                        return
                    raise SourceError(
                        "fetch_all: source returned only a seam duplicate with no new "
                        "rows but reported is_partial=True; cannot make forward progress",
                        context={"symbol": request.symbol, "cursor_start": str(cursor_start)},
                    )

            yield batch

            last_open_time = pd.Timestamp(batch.data["open_time"].iloc[-1])
            next_start = last_open_time + request.timeframe.duration
            if next_start <= cursor_start:
                raise SourceError(
                    "fetch_all: pagination did not advance past the previous cursor "
                    "position; refusing to loop forever",
                    context={"symbol": request.symbol, "cursor_start": str(cursor_start)},
                )
            cursor_start = next_start

            if not batch.is_partial:
                return


__all__ = ["HistoricalSource", "SourceBatch", "SourceMetadata", "SourceRequest"]
