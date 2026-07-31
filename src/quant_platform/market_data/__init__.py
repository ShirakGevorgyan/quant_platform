"""Deterministic market data platform and feature store (Milestone 10).

The single authoritative source for market, macro, calendar, and derived
feature data consumed by research, ML, backtesting, portfolio risk,
execution, and replay. Every event and feature is immutable and
content-addressed; the same input data always produces identical output.

This package never imports a broker SDK and never streams live data. The
sole, explicitly isolated exception to "never opens a network connection"
is the `collectors` subpackage (Milestone 10, Phase 4A): HTTPS-only,
historical-data-only requests to an explicit host allowlist (FRED), never
invoked implicitly by anything elsewhere in this package -- see
`docs/market_data_architecture.md` for the full design and
`docs/milestone10_phase1_delivery_report.md` for Phase 1's delivered
scope."""

from __future__ import annotations
