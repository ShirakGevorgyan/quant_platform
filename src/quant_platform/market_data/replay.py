"""Deterministic replay for `quant_platform.market_data` (Milestone 10,
Phase 1). `replay_candle_features_from_events` is the literal "rebuild
the feature store only from raw events" capability the milestone
requires -- it reads every `Candle` event durably recorded for one
`(provider, instrument_id)` partition of a `MarketEventStore` and feeds
it straight through `feature_generation.generate_candle_features`, never
touching any other source of truth. `compute_feature_semantic_digest`/
`assert_replay_deterministic` are the comparison primitives tests use to
prove two independent replays (a fresh temp directory, a different
process, a different `PYTHONHASHSEED`) produce byte-identical results --
mirroring `portfolio_risk.replay`'s identical role."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.core.exceptions import MarketDataReplayError
from quant_platform.market_data.candles import Candle
from quant_platform.market_data.events import MarketEventStore
from quant_platform.market_data.feature_generation import generate_candle_features
from quant_platform.market_data.feature_store import FeatureRecord, FeatureStore
from quant_platform.market_data.identity import compute_content_id

__all__ = [
    "FEATURE_SEMANTIC_DIGEST_KIND",
    "MarketDataReplayResult",
    "assert_replay_deterministic",
    "compute_feature_semantic_digest",
    "compute_replay_result",
    "replay_candle_features_from_events",
]

FEATURE_SEMANTIC_DIGEST_KIND = "market_data_feature_semantic_digest"


def replay_candle_features_from_events(
    *, event_store: MarketEventStore, provider: str, instrument_id: str, feature_store: FeatureStore, feature_version: int,
    feature_names: tuple[str, ...] | None = None, windows: dict[str, int] | None = None,
) -> tuple[FeatureRecord, ...]:
    """Rebuilds `feature_store`'s feature series for `instrument_id`
    purely from the raw `Candle` events already durably recorded in
    `event_store` -- no other input. Non-candle events in the same
    partition (ticks/quotes/trades) are read but ignored: Phase 1's
    feature catalog is candle-derived only (see `feature_generation.py`'s
    module docstring)."""
    raw_events = event_store.read_events(provider, instrument_id)
    candles = [e for e in raw_events if isinstance(e, Candle)]
    return tuple(generate_candle_features(candles, feature_version=feature_version, store=feature_store, feature_names=feature_names, windows=windows))


def compute_feature_semantic_digest(records: tuple[FeatureRecord, ...]) -> str:
    """A digest of the ECONOMIC content of `records` alone -- sorted by
    `(feature_name, timestamp)` so digest equality never depends on the
    order features happened to be generated/appended in."""
    canonical = [
        {"feature_name": r.feature_name, "feature_version": r.feature_version, "instrument_id": r.instrument_id,
         "timestamp": r.to_json_dict()["timestamp"], "timeframe": r.to_json_dict()["timeframe"], "value": r.to_json_dict()["value"]}
        for r in sorted(records, key=lambda r: (r.feature_name, r.timestamp))
    ]
    return compute_content_id(FEATURE_SEMANTIC_DIGEST_KIND, {"records": canonical})


@dataclass(frozen=True, slots=True)
class MarketDataReplayResult:
    instrument_id: str
    feature_semantic_digest: str
    feature_ids: tuple[str, ...]
    record_count: int


def compute_replay_result(records: tuple[FeatureRecord, ...], *, instrument_id: str) -> MarketDataReplayResult:
    return MarketDataReplayResult(
        instrument_id=instrument_id, feature_semantic_digest=compute_feature_semantic_digest(records),
        feature_ids=tuple(sorted(r.feature_id for r in records)), record_count=len(records),
    )


def assert_replay_deterministic(a: MarketDataReplayResult, b: MarketDataReplayResult) -> None:
    if a.feature_semantic_digest != b.feature_semantic_digest:
        raise MarketDataReplayError(f"Replay divergence: semantic digests differ ({a.feature_semantic_digest!r} != {b.feature_semantic_digest!r})")
    if a.feature_ids != b.feature_ids:
        raise MarketDataReplayError("Replay divergence: feature id sequences differ")
    if a.record_count != b.record_count:
        raise MarketDataReplayError(f"Replay divergence: record counts differ ({a.record_count} != {b.record_count})")
