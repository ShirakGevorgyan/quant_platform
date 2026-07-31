"""`quant_platform.execution_gateway` -- Milestone 8: a broker-neutral,
event-sourced, deterministic TEST-ONLY execution gateway between the
Milestone 7 paper-trading system and a future broker adapter.

THIS IS NOT LIVE TRADING. THIS IS NOT MT5 INTEGRATION. This package
never opens a network connection, never imports a broker SDK, never
defines a credential field, and defines exactly one adapter kind
(`DETERMINISTIC_DUMMY`) and exactly one execution mode (`TEST_ONLY`) --
see `models.ExecutionMode`/`models.AdapterKind`, each a single-member
enum, so no LIVE-like value can ever be constructed, let alone reached.
Every fill and every broker response in this package is produced by
`dummy_broker.DeterministicDummyBrokerAdapter`, an in-process, seeded,
deterministic simulator -- never a real exchange or broker.

Passing every test in this package does not prove profitability, broker
compatibility, broker readiness, or operational live-trading readiness,
and does not authorize real-money execution. See
`docs/execution_gateway_architecture.md` for the full architecture and
`docs/milestone8_delivery_report.md` for the delivery report.

Dependency direction is strictly one-way: this package depends on
`quant_platform.paper_trading` (and, transitively, everything paper_trading
depends on), never the reverse."""

from __future__ import annotations
