# Quant Platform

A broker-agnostic quantitative research platform. **Milestone 1** is a
leak-free, event-driven backtesting engine with realistic transaction
costs, pluggable position sizing, and purged walk-forward validation
infrastructure. **Milestone 2** (this release) adds the production-grade
historical data foundation everything else in this platform ultimately
reads from: broker-agnostic ingestion (MetaTrader5 as the first concrete
adapter), immutable raw snapshots, typed data-quality validation, an
auditable repair/quarantine policy, versioned canonical Parquet storage,
leak-free multi-timeframe resampling, and a reproducible dataset loader
that feeds directly into Milestone 1's `TimeframeCursor`/`BacktestEngine`.
See [`docs/historical_data_pipeline.md`](docs/historical_data_pipeline.md)
for the full technical reference on that layer.

## Why this exists

A prior MT5-specific backtest (a separate, earlier project) shipped with a
look-ahead bug: its multi-timeframe alignment logic compared a higher
timeframe bar's *open* time against the base clock instead of its *close*
time, silently revealing bars up to `duration - 1 base bar` early. It was
caught by manual inspection of a suspiciously-fast stop-loss hit, not by
any automated safeguard. This platform exists so that class of bug is
structurally impossible and continuously *proven* impossible, not just
avoided by convention.

## Architecture

```
quant_platform/
├── core/            Domain types (Bar, Order, Fill, Position, Trade, ...),
│                     exceptions, pure time/timeframe arithmetic. Zero
│                     dependencies on any other subpackage.
├── multiframe/       TimeframeCursor — the core no-look-ahead primitive.
│                     Reveals higher-timeframe bars strictly by close time.
├── data/             DataSource interface + CSV/Parquet implementations,
│                     a seeded synthetic OHLCV generator, and data-quality
│                     validation (OHLC invariants, gaps, monotonicity).
├── costs/            Transaction cost models: fixed spread/commission/
│                     slippage and volatility-scaled slippage.
├── risk/             Position sizing: fixed-fractional, volatility-target,
│                     half-Kelly.
├── strategy/         Strategy interface (point-in-time context in, Signal
│                     out) + a reference SMA-crossover implementation.
├── engine/           Portfolio state, BrokerSimulator (order fills +
│                     intrabar SL/TP/time-stop resolution), and the
│                     BacktestEngine orchestrator tying everything together
│                     via TimeframeCursor.
├── validation/       Purged walk-forward cross-validation splitter (with
│                     embargo), for leakage-safe ML model evaluation later.
├── analytics/        Performance metrics: Sharpe/Sortino/Calmar, max
│                     drawdown, CAGR, profit factor, VaR/CVaR.
├── config/           Pydantic configuration schemas (backtest run config,
│                     plus the Milestone 2 historical-ingestion config).
├── historical/        Milestone 2: broker-agnostic historical data source
│                     protocol + MT5 adapter, immutable raw snapshots,
│                     typed quality validation, repair/quarantine policy,
│                     versioned canonical Parquet storage, leak-free
│                     resampling, dataset manifests, and the reproducible
│                     loader feeding TimeframeCursor/BacktestEngine. See
│                     docs/historical_data_pipeline.md for the full
│                     lifecycle and every contract this subpackage makes.
└── data_cli.py        `python -m quant_platform.data_cli` — ingest /
                      validate / resample / inspect-manifest commands.
```

### The no-look-ahead guarantee

`TimeframeCursor` (see `multiframe/cursor.py`) is the only component
allowed to decide whether a higher-timeframe bar is visible at a given
instant. It reveals a bar if and only if `open_time + timeframe.duration
<= as_of`. This is:

1. **Enforced structurally** — the engine never lets a strategy see raw
   higher-timeframe DataFrames; it only ever hands out `cursor.window(n)`,
   which is bounded by the cursor's current position.
2. **Proven by test** — `tests/unit/test_cursor_no_lookahead.py` includes
   both exact regression tests for the originally-shipped bug and a
   Hypothesis property-based test asserting the invariant holds for
   hundreds of randomized series lengths, start times, and irregular
   clock-stepping patterns.
3. **Guarded at runtime** — `TimeframeCursor.advance_to` raises
   `LookaheadViolationError` if its own post-condition is ever violated,
   so a future refactor that breaks the invariant fails loudly in CI and
   in production, not silently in a performance report.

## Design principles

- **Point-in-time data access only.** Strategies receive a `StrategyContext`
  built fresh each step from cursor windows — there is no code path through
  which a strategy can reach a DataFrame that extends past "now".
- **Costs and sizing are decoupled from signal generation.** A `Strategy`
  emits a directional `Signal`; a `PositionSizer` turns it into a sized
  `Order`; a `CostModel` determines the realistic fill price; the engine
  never asks a strategy to reason about spread, slippage, or account size.
- **Two distinct leakage concerns, two distinct mechanisms.** Temporal
  look-ahead within a single backtest run is `TimeframeCursor`'s job.
  Train/test leakage across an ML walk-forward evaluation (overlapping
  labels, serial correlation) is `validation.walk_forward`'s job (purging +
  embargo). Conflating them was a common source of the "backtest looks
  great, live account doesn't" gap this platform is designed to avoid.
- **Deterministic by default.** The synthetic data generator, and every
  stochastic test, take an explicit seed.

## Historical data pipeline quickstart

```bash
# See examples/ingestion_config.example.json for a complete, safe
# (no real credentials) config; MT5 credentials are read from
# MT5_LOGIN / MT5_PASSWORD / MT5_SERVER environment variables, never
# stored in the config file.
python -m quant_platform.data_cli ingest --config config.json --start 2024-01-01T00:00:00Z --end 2024-02-01T00:00:00Z
python -m quant_platform.data_cli validate --config config.json --symbol XAUUSD --timeframe M1 --start ... --end ...
python -m quant_platform.data_cli resample --config config.json --symbol XAUUSD --source-timeframe M1 --target-timeframe H1 --start ... --end ...
python -m quant_platform.data_cli inspect-manifest --config config.json --symbol XAUUSD --timeframe H1
```

The `MetaTrader5` package is not required to develop against or test this
pipeline: `historical.mt5_testing.FakeMt5Client` implements the same
protocol surface, deterministically, so every code path (including the
adapter itself) is exercised without a live terminal. See
[`docs/historical_data_pipeline.md`](docs/historical_data_pipeline.md) for
the full lifecycle, the timezone/session/validation/repair contracts, and
known limitations.

## Development

```bash
python -m venv .venv
.venv/Scripts/activate          # or `source .venv/bin/activate` on POSIX
pip install -e ".[dev]"

pytest                          # unit + integration + property-based tests
mypy src                        # strict static typing
ruff check src tests            # lint
```

## Roadmap (explicitly out of scope for these milestones)

- Live/paper broker execution adapters (order placement/management --
  Milestone 2 added historical *data* ingestion only, deliberately not
  live trading, per its own scope).
- Feature engineering and ML model training (the walk-forward splitter is
  built now because it is validation infrastructure the engine itself
  needs to expose correctly; the models that consume it are a later
  milestone).
- Live/near-real-time data ingestion (Milestone 2 is batch/historical;
  `DerivedBarPolicy.RETAIN_INCOMPLETE` exists for a still-forming current
  bar, but no live polling loop does).
- Multi-symbol batch ingestion orchestration (Milestone 2 is
  single-symbol-at-a-time by design).
- Experiment tracking / model registry integration.
- Distributed/parallel execution for multi-year, multi-symbol universes.
- A general exchange-calendar dependency (Milestone 2's session/calendar
  model is intentionally scoped to spot/CFD XAUUSD's actual closure
  pattern, not exchange-traded instruments with rolls/half-days).

See `docs/historical_data_pipeline.md`'s "Recommended Milestone 3" section
for the prioritized next-steps list.
