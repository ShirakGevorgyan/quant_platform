"""Provider-neutral cross-asset historical market collectors and curated
XAUUSD market-driver universe (Milestone 10, Phase 4C).

Builds on Phase 4A's secure transport/retry/rate-limit/cache/manifest
infrastructure and mirrors Phase 4B's curated-universe architecture
(registry -> policies -> orchestration -> component/combined datasets ->
reconciliation -> verification) for a materially different domain: OHLCV
market bars for cross-asset drivers of XAUUSD (dollar strength, WTI,
Brent, silver, the gold reference market itself, and several optional
regime-context concepts) rather than single-value macro observations.

PROVIDER-NEUTRAL BY DESIGN: `protocols.HistoricalMarketCollector` is a
structural `Protocol` every concrete provider adapter implements;
`providers/alpha_vantage.py` is the ONE concrete adapter this phase
ships (see `providers/alpha_vantage.py`'s own module docstring for the
bounded provider-selection decision and exactly which endpoint was
independently, live-verified before being implemented). Nothing outside
`collectors/` imports this subpackage; nothing in it opens a live/
streaming connection, imports a broker SDK, or executes an order."""
