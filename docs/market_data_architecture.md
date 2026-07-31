# Deterministic Market Data Platform and Feature Store (Milestone 10) -- Architecture

## Status: Phase 1 (immutable market events, calendar, macro, quality, normalization, deterministic feature generation and storage, replay, verification, reports) delivered

## Primary goal

`quant_platform.market_data` is the single authoritative source for
market, macro, calendar, and derived feature data used by research, ML,
backtesting, portfolio risk, execution, and replay. The governing
guarantee: **the same input data always produces identical feature
datasets** -- the same raw events, fed through the same generator at the
same version, always produce byte-identical `FeatureRecord`s, in any
process, on any machine, in any call order.

This package never opens a network connection, never imports a broker
SDK, and never streams live data. It is a storage and computation layer
over data a caller already has (from a file, a prior ingestion run, or a
test fixture) -- not an ingestion pipeline for a live feed.

## Strict scope (Phase 1)

Allowed: the market-data domain, the feature-store domain, immutable
market events, deterministic feature generation, append-only feature
storage, replay, verification, reconciliation-adjacent quality checks,
data quality validation, documentation, tests.

Forbidden (and not present anywhere in this package): new ML models,
strategy optimization, a live broker, MT5, execution changes,
portfolio-risk changes, prediction logic, cloud services, network
streaming, websocket clients, credentials, live trading.

## Package architecture and dependency direction

```
market_data/
    identity.py          content id, Decimal<->JSON, shared field validators
    candles.py            Candle (OHLCV) event + pure structural math
    ticks.py               Tick event
    events.py              Quote/Trade events, MarketDataEvent union, MarketEventStore
    calendar.py            reuse of historical.calendar.TradingCalendar + expected-open-time enumeration
    macro.py                MacroEvent + MacroEventStore
    normalization.py       raw-row -> trusted domain object ingestion boundary
    quality.py              structured quality reports over raw rows, fail-closed gate
    feature_store.py        FeatureRecord + FeatureStore (append-only)
    feature_generation.py   pure deterministic indicators + store-writing driver
    replay.py                rebuild feature store from raw events; determinism comparison
    verification.py         independent re-verification of event/feature stores
    reports.py               fresh-recompute session reporting
```

Dependency direction is strictly one-way: `identity.py` depends on
nothing else in this package. `candles.py`/`ticks.py` depend only on
`identity.py` (never on `events.py`), specifically to avoid a circular
import -- `events.py` imports `Candle` from `candles.py` and `Tick` from
`ticks.py` to build the `MarketDataEvent` union and its dispatch table,
so the dependency could not run the other way. `calendar.py`/`macro.py`
depend only on `identity.py`. `normalization.py` depends on
`candles.py`/`ticks.py`/`events.py`. `quality.py` depends on
`normalization.py`/`calendar.py`. `feature_store.py` depends only on
`identity.py`. `feature_generation.py` depends on `candles.py`/
`feature_store.py`. `replay.py` depends on `events.py`/
`feature_generation.py`/`feature_store.py`. `verification.py` depends on
`events.py`/`feature_store.py`. `reports.py` depends on `events.py`/
`feature_store.py`/`verification.py`.

This package does not modify, and is not imported by, `historical`,
`features`, `data`, `execution_gateway`, or `portfolio_risk` -- it is a
genuinely new, self-contained domain that happens to reuse a few already-
proven low-level building blocks from elsewhere in the repository (see
"Reuse decisions" below). Nothing in those other packages changed.

## Reuse decisions

- **Content identity** (`compute_content_id`): reused directly from
  `paper_trading.identity`, exactly like `portfolio_risk.identity` and
  `execution_gateway.identity` already do. There is exactly one
  implementation of "namespaced sha256 of canonical JSON" in the whole
  codebase.
- **Trading calendar** (`calendar.py`): reused directly from
  `historical.calendar.TradingCalendar` (weekly sessions, daily
  maintenance breaks, holiday closures, explicit local-timezone-to-UTC
  conversion) rather than re-implementing the same DST-sensitive
  arithmetic a second time. `market_data.calendar` re-exports it and adds
  the one capability `historical.calendar` does not provide:
  `enumerate_expected_open_times`, a deterministic bar-by-bar walk used
  by `quality.py`'s missing-candle/timeframe-gap detection.
- **Locking, JSON, timestamps**: `ml.concurrency.experiment_lock`,
  `core.json.canonical_json_bytes`, and `ml.persistence.
  format_utc_timestamp`/`parse_utc_timestamp` are reused unchanged --
  the same shared infrastructure `portfolio_risk`/`execution_gateway`
  already build on.
- **Validation reporting** (`ValidationIssue`/`ValidationReport`/
  `ValidationSeverity`): reused from `ml.models`, exactly like
  `portfolio_risk.verification`/`execution_gateway.verification`.
  `quality.py` and `verification.py` both produce this same report
  shape rather than inventing a new one.

## Event model

Every event kind (`Tick`, `Quote`, `Trade`, `Candle`) shares one envelope
schema:

```
event_id, instrument_id, provider, symbol, event_time, arrival_time,
timeframe, sequence, source_event_id, payload
```

`timeframe` is always `None` for `Tick`/`Quote`/`Trade` (present as a
field for schema uniformity, not applicable) and always a real
`Timeframe` for `Candle`. `arrival_time` defaults to `event_time` (a
batch/backfill ingest has no separately observable arrival latency) and
must never be earlier than `event_time`.

**Design decision -- typed payload, not top-level fields.** The
milestone's literal field list for "every event" is exactly the envelope
above, ending in `payload`. Rather than storing kind-specific numeric
values (`open`/`high`/`low`/`close`/`volume`; `bid`/`ask`; `price`/`size`)
as separate top-level dataclass fields, they live inside `payload` (a
JSON-safe dict of `Decimal`-as-string values), with the envelope schema
therefore byte-for-byte identical across all four kinds. Construction-
time validation and ergonomic typed access are preserved via typed
`@property` accessors (`candle.open`, `quote.bid`, `trade.price`, ...)
that parse the relevant `payload` key on every access -- callers never
touch the raw dict.

For a `Candle`, `event_time` is the bar's OPEN time (the conventional
index key for an OHLCV series, matching `core.types.OHLCV_COLUMNS`); the
close time is not a separately stored field (which could drift out of
sync with `timeframe`) but a `close_time` property computed via
`core.time_utils.compute_close_time`, the same invariant the rest of the
platform already relies on.

Every value is `Decimal`, never `float` -- computed via `identity.
parse_decimal`/`decimal_to_json`, which reject `NaN`/`Infinity` and
serialize via `str()`, never `Decimal(float)` (which would reproduce
binary floating-point imprecision).

## `MarketEventStore` -- raw event storage

Storage layout: `{root}/market_events/{provider}/{instrument_id}/
events.jsonl`. All four event kinds interleave in one partition,
sequence-ordered, because `replay.py` needs the full raw stream for one
instrument to rebuild derived features deterministically.

Unlike `portfolio_risk.ledger.PortfolioRiskLedgerStore`, there is no
separate hash-chain field (`previous_entry_hash`): every event already
carries its own content-addressed `event_id`, so physical append-order
integrity reduces to "sequence numbers are gapless, and no id repeats
with different content" -- exactly what `verification.
verify_market_event_store` checks. A separate chain field would only
re-prove what the per-event id already proves.

`append` is idempotent for an identical re-append at an already-occupied
sequence (returns the existing event unchanged) and rejects a
conflicting append (a different event at the same sequence) or a
sequence gap.

## `FeatureStore` -- feature storage

Storage layout: `{root}/features/{feature_name}/v{feature_version}/
{instrument_id}/records.jsonl`.

**Design decision -- coordinate-keyed, not position-keyed conflict
detection.** `MarketEventStore` checks conflicts by physical sequence
position (a raw feed genuinely has one, and must be gapless).
`FeatureStore` instead checks conflicts by the record's own economic
coordinate -- `(feature_name, feature_version, instrument_id, timeframe,
timestamp)` -- because feature generation may legitimately be re-run
over an overlapping window (e.g. after a gap-fill), and as long as it
reproduces the SAME value at each coordinate every time (this package's
core determinism guarantee), that re-run must be an idempotent no-op
rather than a spurious "gap" error. A genuinely DIFFERENT value at an
already-recorded coordinate is never silently accepted -- since
generation is deterministic, that can only happen if something upstream
changed without a new `feature_version`, which is exactly the kind of
silent corruption `FeatureStoreError` exists to surface. No overwriting,
no mutable history, ever.

`read_records` always returns records sorted by `(timestamp,
feature_id)`, independent of physical append order -- so two callers who
generated the same feature set in a different order (or ran a partial
backfill) still observe an identical, deterministically ordered result.
This is what makes replay's "identical ordering" requirement hold
regardless of how many times, or in what order, generation was invoked.

## Feature generation catalog (Phase 1)

Pure functions, each taking `Decimal` inputs and returning `Decimal`
(or `None` during a window's warm-up period, never a placeholder):
`returns`, `log_returns`, `rolling_mean`/`sma`, `rolling_std`, `atr`,
`rsi`, `ema`, `vwap`, `price_delta`, `volume_delta`,
`high_low_range_series`, `body_size_series`, `wick_ratios`
(upper/lower, stored as two separate named features).

`generate_candle_features` is the driver: given a candle series (sorted
internally by `event_time`, rejecting duplicate timestamps), it computes
every requested feature and appends every non-`None` point to a
`FeatureStore`. A warm-up `None` is never written -- the feature store's
own "no overwriting" guarantee would otherwise make a warm-up placeholder
impossible to later replace with a real value once enough history
accumulates.

Every windowed feature's stored `feature_name` embeds its window (e.g.
`"sma_20"`, `"atr_14"`) -- two different windows of the same indicator
are different named feature series, matching how `"sma_20"` and
`"sma_50"` are already treated as distinct indicators in common practice,
rather than one indicator with a hidden extra dimension the
`FeatureRecord` schema would otherwise need a new field for.

**Disclosed simplifications** (deliberate, not oversights -- Phase 1
favors one well-defined, fully deterministic variant of each indicator
over configurable alternatives that would multiply the surface this
phase would need to test):

- `atr`/`rsi` use a plain rolling (simple) average, not Wilder's
  recursive smoothing. Both are legitimate, but materially different,
  formulas most platforms distinguish explicitly (often "RSI (Wilder)"
  vs "RSI (simple)"); Phase 1 implements the simple-average variant only.
- `vwap` is cumulative over the whole supplied series, not reset at a
  session boundary. Session-based VWAP requires wiring `calendar.py`'s
  session boundaries into `feature_generation.py`, a reasonable Phase 2
  extension, not a Phase 1 requirement.
- `rolling_std` is the sample standard deviation (`ddof=1`).

## Data quality (`quality.py`)

`run_candle_quality_checks` operates on RAW rows (plain mappings), not
already-constructed `Candle` objects, because `Candle.__post_init__`
already rejects an invalid OHLC relationship/negative volume/non-finite
value at construction time -- correct for a single already-trusted
candle, but wrong for quality reporting, whose entire point is to
examine a batch of untrusted rows and report EVERY problem found, never
stopping at the first bad one (mirrors `historical.quality`'s identical
"every check is independent and additive" philosophy).

Detected: missing candles (via `calendar.enumerate_expected_open_times`,
when a calendar is supplied), duplicate candles (same `open_time`
appearing more than once, regardless of content), timestamp disorder (a
non-chronological input sequence), future timestamps (`open_time >
as_of`, `as_of` always caller-supplied, never an internal wall-clock
read), negative volume, invalid OHLC, NaN, Infinity, duplicate ids
(explained below), and timeframe gaps (the same calendar-driven check as
missing candles).

**Design note -- "duplicate ids" at the quality-scan stage.**
`Candle.event_id` bakes in `sequence` (part of its content identity), and
`run_candle_quality_checks` assigns each row its own unique
`sequence=row_index` before constructing a candidate `Candle` -- so
comparing `event_id` values could never detect a byte-identical repeated
row at this pre-sequencing stage (an earlier draft that did exactly this
was caught during Phase 1's own testing and fixed -- see the delivery
report's "Defects found and fixed" section). "duplicate ids" instead
means: two rows whose full `(open_time, open, high, low, close, volume)`
content is identical -- the row-level analogue of an id collision,
detectable before any real sequence exists. It is reported at WARNING
severity (a harmless, idempotent repeat), distinct from `duplicate_candle`
(CRITICAL -- two DIFFERENT values claiming the same timestamp, a genuine
conflict). True event-id collision detection on an already-sequenced,
already-stored stream is `verification.verify_market_event_store`'s job.

`assert_quality_gate` is the explicit, opt-in fail-closed step: it
raises `MarketDataQualityError` if the report contains any CRITICAL
issue. Report generation itself never raises on data-quality grounds.

## Replay (`replay.py`)

`replay_candle_features_from_events` is the literal "rebuild the feature
store only from raw events" capability: it reads every `Candle` event
durably recorded in a `MarketEventStore` partition and feeds it straight
through `feature_generation.generate_candle_features`, touching no other
source of truth. Non-candle events in the same partition (ticks, quotes,
trades) are read but ignored -- Phase 1's feature catalog is candle-
derived only.

`compute_feature_semantic_digest`/`assert_replay_deterministic` are the
comparison primitives used to prove two independent replays (a fresh
temp directory, a different OS process, a different `PYTHONHASHSEED`)
produce byte-identical results -- see
`tests/unit/market_data/test_market_data_replay.py`'s
`TestCrossProcessReproducibility` for the actual subprocess-based proof.

## Independent verification (`verification.py`)

Mirrors `portfolio_risk.verification`'s explicit two-tier honesty
classification:

- **Structurally independent** (this module's entire scope): event/
  feature append-only sequence integrity, forged-identity detection
  (recomputing each object's own content id from its stored payload and
  comparing), cross-event/cross-feature ordering, and duplicate
  detection. None of these trust a cached report, an in-memory set, or a
  caller assertion -- every one is a pure recomputation from the store's
  own raw bytes (proven in tests by hand-corrupting a stored `.jsonl`
  line and confirming the corruption is caught).
- **Not independently re-verified** (an honest, explicit limitation, not
  an oversight): this module does not re-run `feature_generation.py`'s
  own arithmetic against the store's raw candle events to confirm a
  stored feature VALUE is the economically correct one -- that is
  `replay.py`'s job, invoked separately, not folded into this module's
  checks. `verify_feature_store` verifies a `FeatureRecord`'s own
  internal identity coherence and its store-level positioning, never
  whether its `value` is the correct output of the generator that
  produced it.

## Concurrency

Both `MarketEventStore` and `FeatureStore` guard their append path with
`market_data_lock`/`feature_store_lock`, thin wrappers around `ml.
concurrency.experiment_lock` that translate `ExperimentLockError`/a
Windows release-side sharing-violation `OSError` into `MarketDataLockError`
-- mirroring `portfolio_risk.ledger.portfolio_risk_lock`'s identical,
already-documented translation exactly, including its own noted
pre-existing, shared-infrastructure Windows stale-lock-reclaim
limitation (`historical.locking.DatasetLock`'s own documented race,
entirely out of this package's scope to fix). Phase 1's own tests do not
add dedicated high-concurrency race tests beyond this reused,
already-proven locking primitive -- the milestone's own required test
list ("normal generation, boundary windows, missing data, duplicates,
bad timestamps, replay, verification, quality reports, determinism,
cross-process reproducibility") does not ask for one, unlike Milestone 9
Phase 4's explicit adversarial-concurrency requirement.

## Exceptions

`MarketDataError` (base) with a granular subclass per concern:
`MarketDataEventError`, `MarketDataOrderError`, `MarketDataIdentityError`,
`MarketDataPersistenceError`, `MarketDataLockError`, `MarketCalendarError`,
`MacroDataError`, `MarketDataQualityError`, `FeatureStoreError`,
`FeatureIdentityError`, `FeatureGenerationError`, `MarketDataReplayError`,
`MarketDataVerificationError` -- see `core/exceptions.py`'s "Deterministic
market data platform and feature store (Milestone 10)" section for each
one's exact raise condition. Shared low-level field validators in
`identity.py` (`require_tz_aware`, `require_non_empty`, ...) raise the
base `MarketDataError` directly (they are reused across modules with
different, more specific exception types -- e.g. `feature_store.py`'s
`FeatureStoreError`, `macro.py`'s `MacroDataError` -- so they cannot
hardcode any one of them); each module's own `__post_init__`/business
logic raises its own specific subclass for domain-level violations.

## Known limitations (Phase 1, honestly disclosed)

- No instrument registry/master-data service: `instrument_id` defaults
  to `f"{provider}__{symbol}"` (derived in `normalization.
  derive_instrument_id`) when not explicitly supplied. `__`, not `:` or
  `/`, joins the two -- `:` is a reserved drive-separator character on
  Windows and broke `MarketEventStore`/`FeatureStore`'s own path-based
  partitioning during this phase's own smoke testing (see the delivery
  report's "Defects found and fixed" section).
- No tick-to-candle resampling/aggregation: out of Phase 1's literal
  scope (not in the milestone's feature-generation list), and
  `historical.resampling` already exists for the historical pipeline's
  own leak-free resampling concern.
- `vwap` is cumulative, not session-reset (see "Disclosed
  simplifications" above).
- `atr`/`rsi` use simple, not Wilder, smoothing (see above).
- Macro events (`macro.py`) are modeled and point-in-time-safe
  (`is_macro_event_available_at`), but Phase 1 does not wire them into
  `feature_generation.py` -- no macro-derived feature exists yet. This is
  an honest scope boundary: the milestone's feature-generation list is
  entirely candle-derived ("returns... wick ratios"); macro-derived
  features are a reasonable future phase.
- No CLI surface: this package is a library only in Phase 1, matching
  the milestone's own module list (no CLI expansion was requested or
  added).

## Future phases (not started)

Session-reset VWAP; Wilder-smoothed ATR/RSI variants; macro-derived
features with point-in-time enforcement wired into generation; tick-to-
candle resampling; a real instrument registry; integration with
`portfolio_risk`/`execution_gateway` as their own authoritative market-
data source (this milestone explicitly forbade touching either package
in Phase 1).
