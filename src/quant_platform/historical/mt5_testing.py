"""A fake `MetaTrader5`-shaped client for tests.

Implements `historical.mt5_adapter.Mt5ClientProtocol`'s surface without
importing (or requiring) the real `MetaTrader5` package, so
`MT5HistoricalSource`'s connection lifecycle, symbol validation, pagination,
timezone handling, and error paths are all exercised by the test suite on
any platform, with no live MT5 terminal, and with no credentials of any
kind. This is the test double the Milestone 2 spec requires the adapter be
built against; it deliberately reproduces the real package's rough data
shape (naive epoch-seconds `time` field, unsigned volume/spread fields,
`initialize`/`shutdown`/`last_error`/`symbol_select`/`copy_rates_range`
surface, `RES_S_OK`-shaped error codes) closely enough that adapter code
exercised against it is exercised against the same contract the real
package presents -- not a simplified stand-in that only accidentally
happens to satisfy today's adapter implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

_RATES_DTYPE = np.dtype(
    [
        ("time", "i8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("tick_volume", "u8"),
        ("spread", "i4"),
        ("real_volume", "u8"),
    ]
)

RES_S_OK = 1


def _deterministic_unit_noise(step_index: np.ndarray, seed: int) -> np.ndarray:
    """A fast, vectorized, deterministic pseudo-random function of
    `(step_index, seed)` in the range [-0.5, 0.5) -- a SplitMix64-style bit
    mixer, not a stateful RNG. The key property this exists for: calling it
    twice with an overlapping subset of the same `step_index` values (from
    two different, differently-bounded arrays) always returns identical
    noise for the shared indices, since the result depends only on each
    individual index's own value, never on its position within whatever
    array it happens to be computed alongside.

    Unsigned 64-bit wraparound on every multiply/add below is the intended
    mixing behavior (this is exactly what a SplitMix64-style mixer is), not
    an error -- suppressed explicitly rather than left to numpy's default
    overflow warning, which would otherwise fire on every single call."""
    with np.errstate(over="ignore"):
        x = (step_index.astype(np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15))
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
        uniform_0_1 = (x >> np.uint64(11)).astype(np.float64) / np.float64(1 << 53)
    return uniform_0_1 - 0.5


def generate_fake_rates(
    *, start: datetime, end: datetime, timeframe_minutes: int, seed: int = 7, base_price: float = 2000.0
) -> np.ndarray:
    """Deterministically generate MT5-shaped rate rows (naive epoch-second
    `time`, matching the real package's server-local-wall-clock semantics)
    covering `[start, end)` at `timeframe_minutes` spacing.

    Every bar's price is a pure, closed-form function of its OWN absolute
    step index (epoch seconds // step width) and `seed` -- deliberately
    NOT a cumulative random walk seeded once per call. A real broker's
    already-recorded history does not depend on which window a client
    happens to query; an earlier cumulative-random-walk version of this
    generator restarted from `base_price` at whatever `start` a call was
    given, so two overlapping queries against "the same history" (e.g. an
    incremental update's overlap window vs. the original ingest) produced
    DIFFERENT prices at the SAME timestamps -- discovered via
    `tests/unit/test_data_cli.py::TestRunIngestIncremental` spuriously
    raising `UpdateConflictError` on a second, legitimately-overlapping
    ingest of data that should have reconciled as unchanged. Keying every
    bar strictly off its absolute step index fixes that at the root: `fetch`
    the same absolute time range twice, from any two enclosing queries, and
    every bar in the overlap is bit-for-bit identical.
    """
    if timeframe_minutes <= 0:
        raise ValueError(f"timeframe_minutes must be positive, got {timeframe_minutes}")
    step = timeframe_minutes * 60
    # `start`/`end` are naive datetimes representing literal broker-local
    # wall-clock values (mirroring what the real MT5 API is given/returns).
    # `datetime.timestamp()` on a NAIVE value would instead interpret it
    # through the *host OS's* local timezone, silently making this
    # generator's output depend on the machine it runs on -- exactly the
    # class of bug this whole module exists to prevent. Attaching UTC
    # tzinfo before converting treats the wall-clock digits literally,
    # independent of the host's configured timezone.
    start_epoch = int(start.replace(tzinfo=timezone.utc).timestamp())
    end_epoch = int(end.replace(tzinfo=timezone.utc).timestamp())
    times = np.arange(start_epoch, end_epoch, step, dtype=np.int64)
    n = len(times)
    if n == 0:
        return np.empty(0, dtype=_RATES_DTYPE)

    step_index = times // step  # absolute, query-independent bar index
    noise = _deterministic_unit_noise(step_index, seed)  # in [-0.5, 0.5), per absolute step_index
    drift = 0.02 * np.sin(step_index / 240.0) + 0.01 * np.sin(step_index / 17.0)
    closes = base_price * np.exp(drift + noise * 0.0024)
    # `opens` = the previous absolute step's close, also computed in
    # closed form (not from array position), so the very first row of any
    # query has a well-defined, query-independent open just like every
    # other row.
    prev_noise = _deterministic_unit_noise(step_index - 1, seed)
    prev_drift = 0.02 * np.sin((step_index - 1) / 240.0) + 0.01 * np.sin((step_index - 1) / 17.0)
    opens = base_price * np.exp(prev_drift + prev_noise * 0.0024)

    spread_noise = np.abs(_deterministic_unit_noise(step_index, seed + 1)) * 0.3
    highs = np.maximum(opens, closes) + spread_noise
    lows = np.minimum(opens, closes) - spread_noise
    tick_volumes = (50 + _deterministic_unit_noise(step_index, seed + 2) * 900 + 450).astype(np.int64)
    spreads = (10 + _deterministic_unit_noise(step_index, seed + 3) * 20 + 10).astype(np.int64)

    rates = np.empty(n, dtype=_RATES_DTYPE)
    rates["time"] = times
    rates["open"] = opens
    rates["high"] = highs
    rates["low"] = lows
    rates["close"] = closes
    rates["tick_volume"] = tick_volumes
    rates["spread"] = spreads
    rates["real_volume"] = 0  # spot/CFD XAUUSD: real exchange volume is not reported
    return rates


@dataclass
class FakeMt5Client:
    """Configurable fake. Every simulated-failure knob defaults to "off" so
    a plain `FakeMt5Client()` behaves like a healthy terminal with one
    valid symbol (`XAUUSDm`) and deterministically generated history."""

    valid_symbols: frozenset[str] = frozenset({"XAUUSDm"})
    fail_initialize: bool = False
    fail_symbol_select: bool = False
    return_none_with_error: bool = False
    return_malformed: bool = False
    seed: int = 7
    base_price: float = 2000.0
    max_rows_per_call: int | None = None
    """Simulates a source-side row ceiling per `copy_rates_range` call
    (distinct from `SourceRequest.max_batch_size`, which the adapter also
    enforces) -- set to reproduce "the source itself truncates/rejects
    overly wide ranges" without the adapter necessarily setting a small
    `max_batch_size`."""

    TIMEFRAME_M1: int = field(default=1, init=False)
    TIMEFRAME_M5: int = field(default=5, init=False)
    TIMEFRAME_M15: int = field(default=15, init=False)
    TIMEFRAME_M30: int = field(default=30, init=False)
    TIMEFRAME_H1: int = field(default=60, init=False)
    TIMEFRAME_H4: int = field(default=240, init=False)
    TIMEFRAME_H12: int = field(default=720, init=False)
    TIMEFRAME_D1: int = field(default=1_440, init=False)

    _initialized: bool = field(default=False, init=False)
    _last_error: tuple[int, str] = field(default=(RES_S_OK, "Success"), init=False)
    connect_call_count: int = field(default=0, init=False)
    shutdown_call_count: int = field(default=0, init=False)
    fetch_call_count: int = field(default=0, init=False)
    fetch_calls: list[tuple[str, int, datetime, datetime]] = field(default_factory=list, init=False)

    def initialize(
        self,
        *,
        path: str | None = None,  # noqa: ARG002 -- kept to match Mt5ClientProtocol's signature
        login: int | None = None,  # noqa: ARG002
        password: str | None = None,  # noqa: ARG002 -- the fake never checks credentials by design
        server: str | None = None,  # noqa: ARG002
    ) -> bool:
        self.connect_call_count += 1
        if self.fail_initialize:
            self._last_error = (-1, "Simulated initialize() failure")
            return False
        self._initialized = True
        self._last_error = (RES_S_OK, "Success")
        return True

    def shutdown(self) -> None:
        self.shutdown_call_count += 1
        self._initialized = False

    def last_error(self) -> tuple[int, str]:
        return self._last_error

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:  # noqa: ARG002 -- matches Mt5ClientProtocol
        if self.fail_symbol_select:
            return False
        return symbol in self.valid_symbols

    def version(self) -> tuple[int, int, str]:
        return (500, 5735, "fake")

    def copy_rates_range(self, symbol: str, timeframe: int, date_from: datetime, date_to: datetime) -> Any:
        self.fetch_call_count += 1
        self.fetch_calls.append((symbol, timeframe, date_from, date_to))

        if self.return_none_with_error:
            self._last_error = (-2, "Simulated copy_rates_range failure")
            return None

        timeframe_minutes = self._minutes_for(timeframe)
        rates = generate_fake_rates(
            start=date_from, end=date_to, timeframe_minutes=timeframe_minutes,
            seed=self.seed, base_price=self.base_price,
        )
        if self.max_rows_per_call is not None:
            rates = rates[: self.max_rows_per_call]

        self._last_error = (RES_S_OK, "Success")
        if len(rates) == 0:
            return None  # matches the real package: None for "no data", not an empty array

        if self.return_malformed:
            # Drop a required field to simulate a malformed/corrupted response.
            return rates[["time", "open", "high", "low", "close"]]

        return rates

    def _minutes_for(self, timeframe: int) -> int:
        # Every TIMEFRAME_* constant above was deliberately chosen to equal
        # its own duration in minutes, so no separate lookup table is
        # needed -- just validate it's one of the recognized constants.
        known = (
            self.TIMEFRAME_M1, self.TIMEFRAME_M5, self.TIMEFRAME_M15, self.TIMEFRAME_M30,
            self.TIMEFRAME_H1, self.TIMEFRAME_H4, self.TIMEFRAME_H12, self.TIMEFRAME_D1,
        )
        if timeframe not in known:
            raise ValueError(f"unrecognized fake timeframe constant: {timeframe}")
        return timeframe


__all__ = ["RES_S_OK", "FakeMt5Client", "generate_fake_rates"]
