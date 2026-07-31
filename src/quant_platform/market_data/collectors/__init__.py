"""Secure external historical collector infrastructure (Milestone 10,
Phase 4A) -- a strictly isolated boundary within `market_data`.

This is the ONLY part of `quant_platform.market_data` that opens a
network connection, and it does so under tight, explicit constraints:
HTTPS only, an explicit per-collector host allowlist, historical data
only (a caller-supplied `observation_start`/`observation_end`, never a
live/streaming subscription), no broker/MT5/FxPro code, no order
execution.

Required flow (never short-circuited): remote request -> immutable
request manifest -> raw response bytes (persisted before parsing) ->
immutable response manifest -> Phase 3-compatible source manifest ->
strict adapter -> normalization/validation/quarantine -> durable
market-data repository. The collector layer itself never constructs a
`MarketDataEvent`/`MacroEvent` and appends it to a store directly --
`orchestration.py` (this subpackage's own, collector-specific stage
machine) owns that, mirroring `market_data.orchestration`'s own
discipline for the purely offline Phase 3 pipeline."""

from __future__ import annotations
