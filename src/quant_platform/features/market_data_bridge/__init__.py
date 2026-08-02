"""Milestone 10, Phase 4D: the point-in-time multi-source alignment
bridge between `quant_platform.market_data` (durable, versioned,
Decimal-typed market/macro/cross-asset repositories) and the existing
Milestone 3 feature-engineering / research-dataset pipeline
(`quant_platform.features`).

This package computes NO feature values itself -- every feature family
(technical, multi-timeframe, cross-asset, macro, temporal) still lives
in, and is computed by, the unmodified `quant_platform.features` modules
via `features.engine.FeatureEngine`/`features.registry.FeatureRegistry`.
This package's only job is translating already-durable `market_data`
evidence into the exact input shapes those modules already accept
(`pandas.DataFrame`s conforming to `core.types.OHLCV_COLUMNS`, or the
`value`/`release_time` shape `features.macro.macro_features` expects),
while preserving point-in-time correctness, dataset lineage, and every
Milestone 3 leakage guarantee.

Dependency direction: `market_data` -> this package -> `features` ->
research dataset artifacts. `market_data` never imports this package or
any other part of `features` -- every import here runs the other
direction only.
"""

from __future__ import annotations
