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
from dataclasses import dataclass
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
from quant_platform.market_data.checkpoints import (
    CheckpointStore,
    FeatureGenerationCheckpoint,
    create_feature_generation_checkpoint,
)
from quant_platform.market_data.feature_store import FeatureRecord, FeatureStore, create_feature_record
from quant_platform.market_data.identity import compute_content_id
from quant_platform.market_data.ingestion import rebuild_touched_partitions
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    DatasetManifest,
    PartitioningSpec,
    create_dataset_manifest,
)
from quant_platform.market_data.partitions import partition_key_for
from quant_platform.market_data.repository import MarketDataRepository

__all__ = [
    "DEFAULT_WINDOWS",
    "WINDOWED_FEATURE_BASE_NAMES",
    "WINDOWLESS_FEATURE_NAMES",
    "FeatureIncrementalResult",
    "atr",
    "body_size_series",
    "carry_window_size_for",
    "ema",
    "generate_candle_features",
    "generate_feature_dataset_incremental",
    "high_low_range_series",
    "log_returns",
    "price_delta",
    "rebuild_feature_dataset_manifest",
    "returns",
    "rolling_mean",
    "rolling_std",
    "rsi",
    "sma",
    "stored_feature_name",
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
    windows: dict[str, int] | None = None, only_persist_timestamps: frozenset[datetime] | None = None,
) -> list[FeatureRecord]:
    """`only_persist_timestamps`, when given, restricts which computed
    points are actually written to `store` -- every point is still
    COMPUTED over the full `candles` series (needed for correct rolling
    context), but only those whose timestamp is in the set are persisted.
    Milestone 10 Phase 2's `generate_feature_dataset_incremental` uses
    this to recompute a bounded LEADING "carry" window purely for
    context, without re-attempting to persist those carry positions --
    `atr`/`rsi` both use an artificial "insufficient prior data" fallback
    at index 0 of WHATEVER series they are given (correct when that
    series is genuine full history, wrong when it is a truncated carry
    window), so re-persisting a recomputed carry-window value would
    otherwise spuriously conflict with the CORRECT value already stored
    for that same timestamp from the original, non-truncated computation."""
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
            if only_persist_timestamps is not None and timestamp not in only_persist_timestamps:
                continue
            record = create_feature_record(
                feature_name=stored_name, feature_version=feature_version, instrument_id=instrument_id, timestamp=timestamp,
                timeframe=timeframe, value=value, metadata={},
            )
            records.append(store.append(record))
    return records


# --------------------------------------------------------------------------
# Milestone 10, Phase 2: incremental generation over a durable repository.
#
# One `DatasetKey` (`DatasetKind.DERIVED_FEATURES`) is exactly one STORED
# feature name (e.g. `"sma_20"`, matching Phase 1's own `FeatureStore`
# partitioning by `(feature_name, feature_version, instrument_id)`) -- a
# caller wanting several named features incrementally calls
# `generate_feature_dataset_incremental` once per feature, each with its
# own manifest/partition/checkpoint lineage. This mirrors the milestone's
# own "changed feature spec/version creates a new feature dataset
# lineage" requirement exactly: two different indicators (or the same
# indicator at two different windows) are two different lineages, never
# one dataset with a hidden extra dimension.
# --------------------------------------------------------------------------
_UNBOUNDED_CARRY_FEATURES = frozenset({"vwap", "ema"})
"""`vwap` (as implemented in Phase 1 -- cumulative since the start of the
series, see module docstring) has no bounded trailing-window context that
suffices to continue it correctly; a bounded carry window would silently
compute a WRONG cumulative average from the wrong starting point.

`ema` is unbounded for a DIFFERENT, equally fundamental reason, found
during this phase's own adversarial testing (a parametrized "incremental
equals full recomputation" test caught it -- see the delivery report's
"Defects found and fixed" section): EMA is RECURSIVE, seeded by a plain
SMA of the first `window` values of WHATEVER series it is given, then
each subsequent value depends on the PREVIOUS one. A bounded carry window
re-seeds EMA from a DIFFERENT starting point than the original full
computation used -- and because the recursion never "forgets" its seed,
every value computed from a re-seeded carry window is silently WRONG,
not merely at one boundary point but for the entire carry-forward chain.
Unlike `atr`/`rsi` (whose own boundary approximation is confined to
their first `window` positions and therefore never corrupts a genuinely
NEW point once `only_persist_timestamps` stops re-persisting the carry
region -- see `generate_candle_features`'s own docstring), EMA has no
such confinement: there is no bounded carry window that is ever
sufficient. Phase 2 keeps both `vwap` and `ema` correct by always
re-reading the FULL raw history for them -- a documented, deliberate
PERFORMANCE (never correctness) limitation."""

_PAIRWISE_CARRY_FEATURES = frozenset({"return", "log_return", "price_delta", "volume_delta"})


def stored_feature_name(base_name: str, *, window: int | None = None) -> str:
    if base_name in WINDOWED_FEATURE_BASE_NAMES:
        resolved_window = DEFAULT_WINDOWS[base_name] if window is None else window
        return f"{base_name}_{resolved_window}"
    if window is not None:
        raise FeatureGenerationError(f"{base_name!r} is not a windowed feature; window must be None")
    return base_name


def carry_window_size_for(base_name: str, *, window: int | None = None) -> int | None:
    """The number of trailing raw candles, immediately BEFORE the first
    genuinely new one, that must be re-read to give `base_name` correct
    rolling context. `None` means "the entire history" (`vwap` only)."""
    if base_name in _UNBOUNDED_CARRY_FEATURES:
        return None
    if base_name in WINDOWED_FEATURE_BASE_NAMES:
        return DEFAULT_WINDOWS[base_name] if window is None else window
    if base_name in _PAIRWISE_CARRY_FEATURES:
        return 1
    return 0  # high_low_range / body_size / upper_wick_ratio / lower_wick_ratio: pointwise, no context needed


def _semantic_digest_for_features(records: list[FeatureRecord]) -> str:
    ordered = sorted(records, key=lambda r: (r.timestamp, r.feature_id))
    canonical = [{k: v for k, v in r.to_json_dict().items() if k != "feature_id"} for r in ordered]
    return compute_content_id("feature_dataset_semantic_digest", {"records": canonical})


def rebuild_feature_dataset_manifest(
    *, repository: MarketDataRepository, dataset_key: DatasetKey, partitioning: PartitioningSpec, raw_source_dataset_id: str,
    creation_time: datetime,
) -> DatasetManifest:
    """The `DERIVED_FEATURES` analogue of `ingestion.
    rebuild_dataset_manifest_from_events` -- recomputes FRESH from
    `repository.feature_store`'s and `repository.partition_store`'s
    current durable state, never from a diff."""
    if dataset_key.dataset_kind is not DatasetKind.DERIVED_FEATURES:
        raise FeatureGenerationError("rebuild_feature_dataset_manifest requires a DERIVED_FEATURES dataset_key")
    assert dataset_key.feature_name is not None and dataset_key.feature_version is not None
    all_records = repository.feature_store.read_records(dataset_key.feature_name, dataset_key.feature_version, dataset_key.instrument_id)
    if not all_records:
        manifest = create_dataset_manifest(
            dataset_key=dataset_key, schema_version=1, timeframe=None, partitioning=partitioning, first_event_time=None,
            last_event_time=None, event_count=0, ordered_partition_ids=(), raw_source_dataset_id=raw_source_dataset_id,
            semantic_digest=compute_content_id("feature_dataset_semantic_digest", {"records": []}),
            physical_digest=compute_content_id("dataset_physical_digest", {"member_ids": []}), creation_time=creation_time,
        )
        return repository.manifest_store.append(dataset_key, manifest)

    first_event_time = min(r.timestamp for r in all_records)
    last_event_time = max(r.timestamp for r in all_records)
    timeframe = all_records[0].timeframe
    partition_keys = repository.partition_store.list_partition_keys(dataset_key)
    ordered_partitions = []
    for partition_key in partition_keys:
        partition = repository.partition_store.read(dataset_key, partition_key)
        if partition is None:
            raise FeatureGenerationError(f"listed partition_key {partition_key!r} has no readable partition file")
        ordered_partitions.append(partition)
    ordered_partition_ids = tuple(p.partition_id for p in ordered_partitions)
    semantic_digest = _semantic_digest_for_features(all_records)
    physical_digest = compute_content_id("dataset_physical_digest", {"member_ids": sorted(r.feature_id for r in all_records)})
    manifest = create_dataset_manifest(
        dataset_key=dataset_key, schema_version=1, timeframe=timeframe, partitioning=partitioning, first_event_time=first_event_time,
        last_event_time=last_event_time, event_count=len(all_records), ordered_partition_ids=ordered_partition_ids,
        raw_source_dataset_id=raw_source_dataset_id, semantic_digest=semantic_digest, physical_digest=physical_digest,
        creation_time=creation_time,
    )
    return repository.manifest_store.append(dataset_key, manifest)


@dataclass(frozen=True, slots=True)
class FeatureIncrementalResult:
    feature_dataset_key: DatasetKey
    resulting_feature_dataset_id: str | None
    new_record_count: int
    rebuilt_partition_keys: tuple[str, ...]
    was_no_op: bool


def generate_feature_dataset_incremental(
    *, repository: MarketDataRepository, raw_dataset_key: DatasetKey, feature_base_name: str, feature_version: int,
    partitioning: PartitioningSpec, checkpoint_time: datetime, window: int | None = None,
) -> FeatureIncrementalResult:
    """Incrementally (re)generates ONE named feature series for
    `raw_dataset_key`'s instrument, from whatever new raw candles have
    been committed since the last checkpoint. A no-op (returns
    `was_no_op=True`, touches nothing) if the raw dataset's current
    manifest version is UNCHANGED since the last checkpoint -- there is
    nothing new to process. Correctness invariant: the result is always
    IDENTICAL to calling `generate_candle_features` fresh over the raw
    dataset's ENTIRE history and restricting to this one feature -- see
    `tests/unit/market_data/test_market_data_incremental_features.py`'s
    `TestIncrementalEqualsFullRecomputation`."""
    if raw_dataset_key.dataset_kind is not DatasetKind.RAW_MARKET_EVENTS:
        raise FeatureGenerationError("generate_feature_dataset_incremental requires a RAW_MARKET_EVENTS raw_dataset_key")
    stored_name = stored_feature_name(feature_base_name, window=window)
    feature_dataset_key = DatasetKey(
        dataset_kind=DatasetKind.DERIVED_FEATURES, instrument_id=raw_dataset_key.instrument_id, feature_name=stored_name, feature_version=feature_version,
    )
    raw_manifest = repository.manifest_store.read_current(raw_dataset_key)
    if raw_manifest is None or raw_manifest.event_count == 0:
        return FeatureIncrementalResult(feature_dataset_key=feature_dataset_key, resulting_feature_dataset_id=None, new_record_count=0, rebuilt_partition_keys=(), was_no_op=True)

    checkpoint_store = CheckpointStore(repository.root)
    checkpoint = checkpoint_store.read_current(feature_dataset_key)
    previous_feature_checkpoint = checkpoint if isinstance(checkpoint, FeatureGenerationCheckpoint) else None
    if previous_feature_checkpoint is not None and previous_feature_checkpoint.raw_dataset_id == raw_manifest.dataset_id:
        return FeatureIncrementalResult(
            feature_dataset_key=feature_dataset_key, resulting_feature_dataset_id=previous_feature_checkpoint.resulting_feature_dataset_id,
            new_record_count=0, rebuilt_partition_keys=(), was_no_op=True,
        )

    assert raw_dataset_key.provider is not None
    all_candles = sorted(
        (e for e in repository.event_store.read_events(raw_dataset_key.provider, raw_dataset_key.instrument_id) if isinstance(e, Candle)),
        key=lambda c: c.event_time,
    )
    carry_window_size = carry_window_size_for(feature_base_name, window=window)
    cutoff = previous_feature_checkpoint.last_processed_raw_event_time if previous_feature_checkpoint is not None else None
    if cutoff is None:
        working_set = all_candles
        new_candles = all_candles
    elif carry_window_size is None:
        working_set = all_candles
        new_candles = [c for c in all_candles if c.event_time > cutoff]
    else:
        new_candles = [c for c in all_candles if c.event_time > cutoff]
        cutoff_index = len(all_candles) - len(new_candles)
        context_start = max(0, cutoff_index - carry_window_size)
        working_set = all_candles[context_start:]

    if not new_candles:
        return FeatureIncrementalResult(feature_dataset_key=feature_dataset_key, resulting_feature_dataset_id=None, new_record_count=0, rebuilt_partition_keys=(), was_no_op=True)

    resolved_windows = {feature_base_name: window} if (feature_base_name in WINDOWED_FEATURE_BASE_NAMES and window is not None) else None
    only_persist = frozenset(c.event_time for c in new_candles)
    new_records = generate_candle_features(
        working_set, feature_version=feature_version, store=repository.feature_store, feature_names=(feature_base_name,), windows=resolved_windows,
        only_persist_timestamps=only_persist,
    )

    touched_partition_keys = {partition_key_for(c.event_time, partitioning) for c in new_candles}
    all_feature_records = repository.feature_store.read_records(stored_name, feature_version, raw_dataset_key.instrument_id)
    all_members = [(r.feature_id, r.timestamp) for r in all_feature_records]
    rebuild_touched_partitions(
        repository=repository, dataset_key=feature_dataset_key, partitioning=partitioning, all_members=all_members, touched_partition_keys=touched_partition_keys,
    )
    manifest = rebuild_feature_dataset_manifest(
        repository=repository, dataset_key=feature_dataset_key, partitioning=partitioning, raw_source_dataset_id=raw_manifest.dataset_id,
        creation_time=checkpoint_time,
    )

    new_checkpoint = create_feature_generation_checkpoint(
        raw_dataset_key=raw_dataset_key, raw_dataset_id=raw_manifest.dataset_id, feature_dataset_key=feature_dataset_key,
        last_processed_raw_event_time=all_candles[-1].event_time, carry_window_size=carry_window_size,
        resulting_feature_dataset_id=manifest.dataset_id, semantic_digest=manifest.semantic_digest, checkpoint_time=checkpoint_time,
    )
    checkpoint_store.append(feature_dataset_key, new_checkpoint)

    return FeatureIncrementalResult(
        feature_dataset_key=feature_dataset_key, resulting_feature_dataset_id=manifest.dataset_id, new_record_count=len(new_records),
        rebuilt_partition_keys=tuple(sorted(touched_partition_keys)), was_no_op=False,
    )
