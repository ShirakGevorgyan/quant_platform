"""Deterministic feature generation from candle series (Milestone 10,
Phase 1).

Every `_compute_*` function below is PURE: given the same `Decimal`
inputs it always returns the same `Decimal` outputs, with no I/O, no
randomness, and no reliance on wall-clock time. A point whose window has
not yet accumulated enough history returns `None` at that index rather
than a placeholder value -- `generate_candle_features` skips `None`
points entirely rather than writing them to the store, so a warm-up
period never produces a spurious, later-to-be-corrected record (the
feature store's own "no overwriting" guarantee would otherwise make a
warm-up placeholder impossible to later replace with a real value).

DISCLOSED SIMPLIFICATIONS (deliberate, not oversights -- Phase 1 favors
one well-defined, fully deterministic variant of each indicator over
configurable alternatives that would multiply the surface this phase
would need to test):
- `atr`/`rsi` use a plain rolling (simple) average, not Wilder's
  recursive smoothing. Wilder smoothing is ALSO deterministic, but is a
  materially different formula most platforms distinguish explicitly
  (often as "RSI (Wilder)" vs "RSI (simple)"); this phase implements the
  simple-average variant only.
- `vwap` is CUMULATIVE over the whole supplied series, not reset at a
  session boundary -- session-based VWAP requires wiring `calendar.py`'s
  session boundaries into this module, which is a reasonable Phase 2
  extension, not a Phase 1 requirement.
- `rolling_std` is the SAMPLE standard deviation (`ddof=1`).

Every windowed feature's stored `feature_name` embeds its window (e.g.
`"sma_20"`, `"atr_14"`) -- two different windows of the same indicator
are different named feature series, exactly like `"sma_20"` and
`"sma_50"` are treated as distinct indicators in common practice, not one
indicator with a hidden extra dimension the `FeatureRecord` schema would
otherwise need a new field for."""

from __future__ import annotations

import itertools
from datetime import datetime
from decimal import Decimal

from quant_platform.core.exceptions import FeatureGenerationError
from quant_platform.core.types import Timeframe
from quant_platform.market_data.candles import (
    Candle,
    candle_body_size,
    candle_lower_wick,
    candle_range,
    candle_upper_wick,
)
from quant_platform.market_data.feature_store import FeatureRecord, FeatureStore, create_feature_record

__all__ = [
    "DEFAULT_WINDOWS",
    "WINDOWED_FEATURE_BASE_NAMES",
    "WINDOWLESS_FEATURE_NAMES",
    "atr",
    "body_size_series",
    "ema",
    "generate_candle_features",
    "high_low_range_series",
    "log_returns",
    "price_delta",
    "returns",
    "rolling_mean",
    "rolling_std",
    "rsi",
    "sma",
    "volume_delta",
    "vwap",
    "wick_ratios",
]

DEFAULT_WINDOWS: dict[str, int] = {"rolling_mean": 20, "rolling_std": 20, "atr": 14, "rsi": 14, "ema": 12, "sma": 20}
WINDOWED_FEATURE_BASE_NAMES: tuple[str, ...] = ("rolling_mean", "rolling_std", "atr", "rsi", "ema", "sma")
WINDOWLESS_FEATURE_NAMES: tuple[str, ...] = (
    "return", "log_return", "vwap", "price_delta", "volume_delta", "high_low_range", "body_size", "upper_wick_ratio", "lower_wick_ratio",
)


def _require_positive_window(window: int) -> None:
    if window < 1:
        raise FeatureGenerationError(f"window must be >= 1, got {window}")


# --------------------------------------------------------------------------
# Windowless, pointwise/pairwise features.
# --------------------------------------------------------------------------
def returns(closes: list[Decimal]) -> list[Decimal | None]:
    result: list[Decimal | None] = [None]
    for previous, current in itertools.pairwise(closes):
        result.append(None if previous == 0 else (current / previous) - 1)
    return result


def log_returns(closes: list[Decimal]) -> list[Decimal | None]:
    result: list[Decimal | None] = [None]
    for previous, current in itertools.pairwise(closes):
        result.append(None if previous <= 0 or current <= 0 else (current / previous).ln())
    return result


def price_delta(closes: list[Decimal]) -> list[Decimal | None]:
    result: list[Decimal | None] = [None]
    for previous, current in itertools.pairwise(closes):
        result.append(current - previous)
    return result


def volume_delta(volumes: list[Decimal | None]) -> list[Decimal | None]:
    result: list[Decimal | None] = [None]
    for previous, current in itertools.pairwise(volumes):
        result.append(None if previous is None or current is None else current - previous)
    return result


def high_low_range_series(candles: list[Candle]) -> list[Decimal]:
    return [candle_range(c) for c in candles]


def body_size_series(candles: list[Candle]) -> list[Decimal]:
    return [candle_body_size(c) for c in candles]


def wick_ratios(candles: list[Candle]) -> list[tuple[Decimal, Decimal]]:
    """`(upper_wick / range, lower_wick / range)` per candle -- `(0, 0)`
    for a zero-range (doji, or a single-tick) candle rather than dividing
    by zero, since a zero range genuinely has no wick proportion to
    report."""
    result: list[tuple[Decimal, Decimal]] = []
    for candle in candles:
        total_range = candle_range(candle)
        if total_range == 0:
            result.append((Decimal(0), Decimal(0)))
            continue
        result.append((candle_upper_wick(candle) / total_range, candle_lower_wick(candle) / total_range))
    return result


def vwap(candles: list[Candle]) -> list[Decimal | None]:
    """Cumulative volume-weighted average price over the supplied series
    (see module docstring's disclosed-simplifications note). Once a
    candle with a missing `volume` is encountered, that point and every
    point after it is `None` -- a cumulative statistic cannot skip a gap
    and remain correct, so it fails closed rather than silently treating
    a missing volume as zero."""
    result: list[Decimal | None] = []
    cumulative_pv = Decimal(0)
    cumulative_volume = Decimal(0)
    broken = False
    for candle in candles:
        if broken or candle.volume is None:
            broken = True
            result.append(None)
            continue
        typical_price = (candle.high + candle.low + candle.close) / 3
        cumulative_pv += typical_price * candle.volume
        cumulative_volume += candle.volume
        result.append(None if cumulative_volume == 0 else cumulative_pv / cumulative_volume)
    return result


# --------------------------------------------------------------------------
# Windowed features.
# --------------------------------------------------------------------------
def rolling_mean(values: list[Decimal], window: int) -> list[Decimal | None]:
    _require_positive_window(window)
    result: list[Decimal | None] = []
    for index in range(len(values)):
        if index + 1 < window:
            result.append(None)
            continue
        window_slice = values[index + 1 - window : index + 1]
        result.append(sum(window_slice, start=Decimal(0)) / window)
    return result


def sma(values: list[Decimal], window: int) -> list[Decimal | None]:
    return rolling_mean(values, window)


def rolling_std(values: list[Decimal], window: int) -> list[Decimal | None]:
    """Sample standard deviation (`ddof=1`) -- requires `window >= 2`."""
    if window < 2:
        raise FeatureGenerationError(f"rolling_std requires window >= 2, got {window}")
    result: list[Decimal | None] = []
    for index in range(len(values)):
        if index + 1 < window:
            result.append(None)
            continue
        window_slice = values[index + 1 - window : index + 1]
        mean = sum(window_slice, start=Decimal(0)) / window
        variance = sum(((v - mean) ** 2 for v in window_slice), start=Decimal(0)) / (window - 1)
        result.append(variance.sqrt())
    return result


def ema(values: list[Decimal], window: int) -> list[Decimal | None]:
    _require_positive_window(window)
    result: list[Decimal | None] = []
    multiplier = Decimal(2) / (window + 1)
    previous_ema: Decimal | None = None
    for index in range(len(values)):
        if index + 1 < window:
            result.append(None)
            continue
        if previous_ema is None:
            window_slice = values[index + 1 - window : index + 1]
            previous_ema = sum(window_slice, start=Decimal(0)) / window
        else:
            previous_ema = (values[index] - previous_ema) * multiplier + previous_ema
        result.append(previous_ema)
    return result


def atr(candles: list[Candle], window: int) -> list[Decimal | None]:
    _require_positive_window(window)
    true_ranges: list[Decimal] = []
    for index, candle in enumerate(candles):
        if index == 0:
            true_ranges.append(candle_range(candle))
            continue
        previous_close = candles[index - 1].close
        true_ranges.append(max(candle.high - candle.low, abs(candle.high - previous_close), abs(candle.low - previous_close)))
    return rolling_mean(true_ranges, window)


def rsi(closes: list[Decimal], window: int) -> list[Decimal | None]:
    """Simple-average RSI (see module docstring). Neutral `50` when there
    has been neither a gain nor a loss anywhere in the window (a frozen
    price series); `100` when there have been only gains (no loss to
    divide by)."""
    _require_positive_window(window)
    gains: list[Decimal] = [Decimal(0)]
    losses: list[Decimal] = [Decimal(0)]
    for previous, current in itertools.pairwise(closes):
        delta = current - previous
        gains.append(max(delta, Decimal(0)))
        losses.append(max(-delta, Decimal(0)))
    avg_gains = rolling_mean(gains, window)
    avg_losses = rolling_mean(losses, window)
    result: list[Decimal | None] = []
    for avg_gain, avg_loss in zip(avg_gains, avg_losses, strict=True):
        if avg_gain is None or avg_loss is None:
            result.append(None)
            continue
        if avg_gain == 0 and avg_loss == 0:
            result.append(Decimal(50))
        elif avg_loss == 0:
            result.append(Decimal(100))
        else:
            rs = avg_gain / avg_loss
            result.append(Decimal(100) - (Decimal(100) / (1 + rs)))
    return result


# --------------------------------------------------------------------------
# Driver: computes the requested features over a chronologically sorted
# candle series and appends every non-`None` point to `store`.
# --------------------------------------------------------------------------
def generate_candle_features(
    candles: list[Candle], *, feature_version: int, store: FeatureStore, feature_names: tuple[str, ...] | None = None,
    windows: dict[str, int] | None = None,
) -> list[FeatureRecord]:
    if not candles:
        return []
    instrument_ids = {c.instrument_id for c in candles}
    if len(instrument_ids) != 1:
        raise FeatureGenerationError(f"generate_candle_features requires a single instrument, got {sorted(instrument_ids)}")
    timeframes = {c.timeframe for c in candles}
    if len(timeframes) != 1:
        raise FeatureGenerationError(f"generate_candle_features requires a single timeframe, got {sorted(t.value for t in timeframes)}")
    instrument_id = next(iter(instrument_ids))
    timeframe: Timeframe = next(iter(timeframes))

    sorted_candles = sorted(candles, key=lambda c: c.event_time)
    timestamps: list[datetime] = [c.event_time for c in sorted_candles]
    if len(set(timestamps)) != len(timestamps):
        raise FeatureGenerationError("generate_candle_features requires unique event_time values -- run quality.py first")

    resolved_windows = dict(DEFAULT_WINDOWS)
    if windows:
        resolved_windows.update(windows)
    all_names = WINDOWLESS_FEATURE_NAMES + WINDOWED_FEATURE_BASE_NAMES
    requested = all_names if feature_names is None else feature_names
    unknown = set(requested) - set(all_names)
    if unknown:
        raise FeatureGenerationError(f"Unknown feature name(s): {sorted(unknown)}")

    closes = [c.close for c in sorted_candles]
    volumes = [c.volume for c in sorted_candles]

    series_by_stored_name: dict[str, list[Decimal | None]] = {}
    for name in requested:
        if name == "return":
            series_by_stored_name["return"] = returns(closes)
        elif name == "log_return":
            series_by_stored_name["log_return"] = log_returns(closes)
        elif name == "price_delta":
            series_by_stored_name["price_delta"] = price_delta(closes)
        elif name == "volume_delta":
            series_by_stored_name["volume_delta"] = volume_delta(volumes)
        elif name == "high_low_range":
            series_by_stored_name["high_low_range"] = list(high_low_range_series(sorted_candles))
        elif name == "body_size":
            series_by_stored_name["body_size"] = list(body_size_series(sorted_candles))
        elif name == "vwap":
            series_by_stored_name["vwap"] = vwap(sorted_candles)
        elif name == "upper_wick_ratio" or name == "lower_wick_ratio":
            ratios = wick_ratios(sorted_candles)
            series_by_stored_name["upper_wick_ratio"] = [r[0] for r in ratios]
            series_by_stored_name["lower_wick_ratio"] = [r[1] for r in ratios]
        elif name in WINDOWED_FEATURE_BASE_NAMES:
            window = resolved_windows[name]
            stored_name = f"{name}_{window}"
            if name == "rolling_mean":
                series_by_stored_name[stored_name] = rolling_mean(closes, window)
            elif name == "sma":
                series_by_stored_name[stored_name] = sma(closes, window)
            elif name == "rolling_std":
                series_by_stored_name[stored_name] = rolling_std(closes, window)
            elif name == "ema":
                series_by_stored_name[stored_name] = ema(closes, window)
            elif name == "atr":
                series_by_stored_name[stored_name] = atr(sorted_candles, window)
            elif name == "rsi":
                series_by_stored_name[stored_name] = rsi(closes, window)

    records: list[FeatureRecord] = []
    for stored_name, series in series_by_stored_name.items():
        for timestamp, value in zip(timestamps, series, strict=True):
            if value is None:
                continue
            record = create_feature_record(
                feature_name=stored_name, feature_version=feature_version, instrument_id=instrument_id, timestamp=timestamp,
                timeframe=timeframe, value=value, metadata={},
            )
            records.append(store.append(record))
    return records
