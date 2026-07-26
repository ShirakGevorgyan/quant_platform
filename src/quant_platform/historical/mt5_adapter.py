"""MetaTrader5 historical-data source adapter.

The `MetaTrader5` package is lazily imported -- only inside `connect()`,
and only when no test-double `client` was injected -- so this module,
`MT5HistoricalSource`, and everything built on top of it (the ingestion
pipeline, the CLI) remain importable, constructible, and unit-testable on a
machine (or in a CI runner) that does not have the `MetaTrader5` package
installed at all, which is the common case: it is a Windows-only package
that talks to a locally running MT5 terminal, and this platform's own test
suite must not require one. Actually invoking MT5 functionality without the
package installed raises `MissingDependencyError` with an actionable
message -- it never silently no-ops, and it never fakes a successful
connection.

Two hard-won operational lessons from a prior, separate MT5 integration are
encoded here directly:

1. `copy_rates_range`'s `time` field is Unix-epoch-shaped but represents
   the broker/trade-server's own wall clock, not necessarily true UTC.
   Treating it as UTC without checking is exactly the mistake
   `historical.timezones` exists to make structurally difficult: this
   adapter always parses it as a NAIVE timestamp first and then localizes
   it via `historical.timezones.localize_broker_timestamps` using the
   explicitly configured `server_timezone` -- never `utc=True` directly.
2. `copy_rates_range` has an undocumented per-call ceiling on the width of
   the requested date range (empirically, a ~30-calendar-day M1 request
   failed with "Invalid params" where a ~25-day request succeeded). This
   adapter cannot discover or enforce that ceiling itself -- it varies and
   is not published -- so responsibility for keeping any single `fetch()`
   call's date width conservative belongs to the ingestion pipeline
   orchestrator (`historical.pipeline`), which chunks a large requested
   range into bounded sub-ranges *before* constructing each `SourceRequest`
   (see its `extraction_chunk_size` configuration). This adapter still
   performs its own row-count-based pagination via the inherited
   `fetch_all` for defense in depth, but that alone would not have caught
   the date-width failure mode above, which is why both layers exist.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import MissingDependencyError, SourceError
from quant_platform.core.types import Timeframe
from quant_platform.historical.models import RAW_HISTORICAL_COLUMNS, coerce_historical_dtypes
from quant_platform.historical.source import HistoricalSource, SourceBatch, SourceMetadata, SourceRequest
from quant_platform.historical.timezones import SourceTimezone, localize_broker_timestamps

logger = logging.getLogger(__name__)

_MT5_TIMEFRAME_ATTR: dict[Timeframe, str] = {
    Timeframe.M1: "TIMEFRAME_M1",
    Timeframe.M5: "TIMEFRAME_M5",
    Timeframe.M15: "TIMEFRAME_M15",
    Timeframe.M30: "TIMEFRAME_M30",
    Timeframe.H1: "TIMEFRAME_H1",
    Timeframe.H4: "TIMEFRAME_H4",
    Timeframe.H12: "TIMEFRAME_H12",
    Timeframe.D1: "TIMEFRAME_D1",
}

_MT5_RATES_FIELDS = ("time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume")
_MT5_OK_ERROR_CODE = 1  # MetaTrader5.RES_S_OK, duplicated here to avoid importing the package


class Mt5ClientProtocol(Protocol):
    """The minimal subset of the `MetaTrader5` package's module-level
    surface this adapter depends on. The real package (a plain module, not
    a class) satisfies this structurally; `historical.mt5_testing.FakeMt5Client`
    implements the identical surface for tests, so this adapter's
    connection lifecycle, symbol handling, and pagination logic are fully
    exercised without a live MT5 terminal."""

    def initialize(
        self,
        *,
        path: str | None = ...,
        login: int | None = ...,
        password: str | None = ...,
        server: str | None = ...,
    ) -> bool: ...

    def shutdown(self) -> None: ...

    def last_error(self) -> tuple[int, str]: ...

    def symbol_select(self, symbol: str, enable: bool = ...) -> bool: ...

    def copy_rates_range(self, symbol: str, timeframe: int, date_from: Any, date_to: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class MT5AdapterConfig:
    """Everything the adapter needs, with nothing guessed. `broker` and
    `server_timezone` in particular are REQUIRED -- there is no default
    server timezone, because guessing one is precisely the bug class this
    pipeline exists to prevent (see `historical.timezones`)."""

    broker: str
    source_symbol: str
    server_timezone: SourceTimezone
    terminal_path: str | None = None
    login: int | None = None
    password: str | None = field(default=None, repr=False)
    server: str | None = None

    def __post_init__(self) -> None:
        if not self.broker:
            raise ValueError("broker identity must be explicitly configured (never inferred)")
        if not self.source_symbol:
            raise ValueError("source_symbol (the broker's own alias for this instrument) is required")


class MT5HistoricalSource(HistoricalSource):
    """Broker-agnostic `HistoricalSource` backed by the MetaTrader5
    terminal API.

    `client`, if provided, is used as-is and the real `MetaTrader5` package
    is never imported -- this is how tests exercise this adapter's full
    lifecycle/pagination/error-handling logic without a live terminal (see
    `historical.mt5_testing.FakeMt5Client`). If omitted (the production
    path), the package is lazily imported on `connect()`.

    Ownership: if the caller supplies their own `client` (an externally
    managed MT5 session another part of the process may also be using),
    `disconnect()` does NOT call `shutdown()` on it unless `owns_client=True`
    is passed explicitly -- tearing down a connection this adapter did not
    establish would break whatever else is using it. When `client` is
    omitted and this adapter lazily imports and initializes MT5 itself, it
    always owns that session regardless of the `owns_client` argument.
    """

    def __init__(
        self,
        config: MT5AdapterConfig,
        *,
        client: Mt5ClientProtocol | None = None,
        owns_client: bool = True,
    ) -> None:
        self._config = config
        self._client: Mt5ClientProtocol | None = client
        self._owns_client = True if client is None else owns_client
        self._connected = False

    def connect(self) -> None:
        if self._client is None:
            try:
                import MetaTrader5 as mt5_module  # type: ignore[import-not-found]  # noqa: N813
            except ImportError as exc:
                raise MissingDependencyError(
                    "MT5HistoricalSource requires the optional 'MetaTrader5' package, which "
                    "is not installed in this environment. Install it with `pip install "
                    "MetaTrader5` (Windows only, requires a local MT5 terminal), or inject a "
                    "test double via the `client` constructor parameter for environments "
                    "(CI, non-Windows, no terminal) that cannot install it.",
                    context={"broker": self._config.broker},
                ) from exc
            self._client = mt5_module

        ok = self._client.initialize(
            path=self._config.terminal_path,
            login=self._config.login,
            password=self._config.password,
            server=self._config.server,
        )
        if not ok:
            code, message = self._client.last_error()
            raise SourceError(
                f"MT5 initialize() failed: {message} (code={code})",
                context={"broker": self._config.broker},
            )
        self._connected = True
        logger.info("MT5 source connected: broker=%s symbol=%s", self._config.broker, self._config.source_symbol)

    def disconnect(self) -> None:
        if self._connected and self._owns_client:
            assert self._client is not None
            self._client.shutdown()
            logger.info("MT5 source disconnected: broker=%s", self._config.broker)
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def fetch(self, request: SourceRequest) -> SourceBatch:
        started_at = time.perf_counter()
        if not self._connected or self._client is None:
            raise SourceError(
                "fetch() called before a successful connect()", context={"broker": self._config.broker}
            )

        if not self._client.symbol_select(request.symbol, True):
            raise SourceError(
                f"MT5 symbol_select failed for {request.symbol!r}: symbol is invalid or "
                "unavailable on this broker/account",
                context={"broker": self._config.broker, "symbol": request.symbol},
            )

        mt5_timeframe = getattr(self._client, _MT5_TIMEFRAME_ATTR[request.timeframe])
        extracted_at = pd.Timestamp.now(tz="UTC")

        # MT5's `copy_rates_range` takes `date_from`/`date_to` on the same
        # broker-server wall-clock axis as the `time` field it returns (see
        # module docstring) -- NOT true UTC despite the naive-datetime shape
        # both directions share. Converting our UTC-aware request boundaries
        # to naive broker-local wall-clock time here, symmetric with how
        # `_parse_rates` converts the naive broker-local response back to
        # UTC, is the only self-consistent reading of that contract: a
        # request for [start, end) in UTC must ask MT5 for the *server-local*
        # wall-clock instants that correspond to those same two real-world
        # moments, not for the numerically identical wall-clock values
        # interpreted as if they were already server-local (which would
        # silently shift the requested window by the server's UTC offset).
        # `SourceRequest.end` is EXCLUSIVE (this platform's convention
        # throughout, matching `data.interfaces.DataSource.load`), but MT5's
        # `copy_rates_range` is INCLUSIVE of `date_to` on the wire. Passing
        # `request.end` straight through would silently pull in one extra
        # bar exactly AT the exclusive boundary whenever the server happens
        # to have one there -- and that bar would then be fetched AGAIN as
        # the first row of the next paginated request, a genuine duplicate
        # masquerading as a boundary artifact. Every supported `Timeframe`
        # has a duration of at least one minute, so subtracting one second
        # from the exclusive boundary before handing it to MT5's inclusive
        # `date_to` always lands strictly before the next aligned bar open
        # time without ever excluding a bar this request should include.
        one_second = pd.Timedelta(seconds=1)
        tzinfo_obj = self._config.server_timezone.to_tzinfo()
        date_from_local = request.start.tz_convert(tzinfo_obj).tz_localize(None).to_pydatetime()
        date_to_local = (request.end - one_second).tz_convert(tzinfo_obj).tz_localize(None).to_pydatetime()

        try:
            rates = self._client.copy_rates_range(request.symbol, mt5_timeframe, date_from_local, date_to_local)
        except Exception as exc:
            raise SourceError(
                f"MT5 copy_rates_range raised for {request.symbol!r}: {exc}",
                context={"broker": self._config.broker, "symbol": request.symbol},
            ) from exc

        empty_result = self._empty_batch(request, extracted_at)

        if rates is None:
            code, message = self._client.last_error()
            if code != _MT5_OK_ERROR_CODE:
                raise SourceError(
                    f"MT5 copy_rates_range failed for {request.symbol!r}: {message} (code={code})",
                    context={"broker": self._config.broker, "symbol": request.symbol},
                )
            # `last_error` reports no error, so `None` here means "genuinely
            # no bars in range" (e.g. the range fell entirely inside a
            # weekend/holiday closure), not a fetch failure.
            return empty_result

        if len(rates) == 0:
            return empty_result

        try:
            df = self._parse_rates(rates)
        except (KeyError, ValueError) as exc:
            raise SourceError(
                f"MT5 copy_rates_range returned a malformed response for {request.symbol!r}: {exc}",
                context={"broker": self._config.broker, "symbol": request.symbol},
            ) from exc

        is_partial = len(df) > request.max_batch_size
        if is_partial:
            df = df.iloc[: request.max_batch_size].reset_index(drop=True)

        metadata = SourceMetadata(
            source_name="mt5",
            source_version=self._client_version(),
            broker=self._config.broker,
            source_symbol=request.symbol,
            extracted_at=extracted_at,
        )
        logger.info(
            "MT5 fetch complete: broker=%s symbol=%s timeframe=%s range=[%s, %s) rows=%d partial=%s "
            "duration_s=%.3f",
            self._config.broker, request.symbol, request.timeframe.value, request.start, request.end,
            len(df), is_partial, time.perf_counter() - started_at,
        )
        return SourceBatch(
            data=df, metadata=metadata,
            requested_start=request.start, requested_end=request.end, is_partial=is_partial,
        )

    def _empty_batch(self, request: SourceRequest, extracted_at: pd.Timestamp) -> SourceBatch:
        empty_df = pd.DataFrame(
            {
                "open_time": pd.Series([], dtype="datetime64[ns, UTC]"),
                "open": pd.Series([], dtype=np.float64),
                "high": pd.Series([], dtype=np.float64),
                "low": pd.Series([], dtype=np.float64),
                "close": pd.Series([], dtype=np.float64),
                "tick_volume": pd.Series([], dtype=np.int64),
                "real_volume": pd.Series([], dtype=np.int64),
                "spread": pd.Series([], dtype=np.int64),
            }
        )[list(RAW_HISTORICAL_COLUMNS)]
        metadata = SourceMetadata(
            source_name="mt5",
            source_version=self._client_version(),
            broker=self._config.broker,
            source_symbol=request.symbol,
            extracted_at=extracted_at,
        )
        return SourceBatch(
            data=empty_df, metadata=metadata,
            requested_start=request.start, requested_end=request.end, is_partial=False,
        )

    def _parse_rates(self, rates: Any) -> pd.DataFrame:
        raw = pd.DataFrame(rates)
        missing = set(_MT5_RATES_FIELDS) - set(raw.columns)
        if missing:
            raise KeyError(f"missing expected field(s) in MT5 response: {sorted(missing)}")

        naive_open_time = pd.to_datetime(raw["time"], unit="s")
        utc_open_time = localize_broker_timestamps(naive_open_time, self._config.server_timezone)

        df = pd.DataFrame(
            {
                "open_time": utc_open_time,
                "open": raw["open"],
                "high": raw["high"],
                "low": raw["low"],
                "close": raw["close"],
                "tick_volume": raw["tick_volume"],
                "real_volume": raw["real_volume"],
                "spread": raw["spread"],
            }
        )
        return coerce_historical_dtypes(df)

    def _client_version(self) -> str:
        assert self._client is not None
        version_fn = getattr(self._client, "version", None)
        if callable(version_fn):
            try:
                return str(version_fn())
            except Exception:  # pragma: no cover - defensive, version() is informational only
                return "unknown"
        return "unknown"


__all__ = ["MT5AdapterConfig", "MT5HistoricalSource", "Mt5ClientProtocol"]
