"""Deterministic paper trading and shadow execution engine (Milestone 7).

Consumes ONLY strategies independently verified `ELIGIBLE_FOR_PAPER_
TRADING` by `quant_platform.robustness` (Milestone 6) and deterministically
simulates the complete trading lifecycle -- decisions, orders, fills,
positions, accounting -- without ever transmitting an order to a broker,
exchange, MetaTrader terminal, or live-trading API. There is no
`ELIGIBLE_FOR_LIVE_TRADING`, no `LIVE` session mode, and no network
trading client anywhere in this package.

Broker-neutral and instrument-neutral: `InstrumentSpec` (Section 14) is
capable of representing a future XAUUSD/MT5 contract, but no MT5-specific
value is hardcoded, and no MT5 adapter is implemented in this milestone."""

from __future__ import annotations
