# Deterministic Market Data Platform and Feature Store (Milestone 10) -- Architecture

## Status: Phase 1 (immutable market events, calendar, macro, quality, normalization, deterministic feature generation and storage, replay, verification, reports) + Phase 2 (durable repository, dataset versioning, incremental ingestion, partitioning, checkpoints, recovery, reconciliation, compaction, export) + Phase 3 (historical ingestion orchestration and offline source adapters) + Phase 4A (secure external historical collector infrastructure and FRED integration) + Phase 4B (curated FRED macro universe and verified historical backfill workflow for XAUUSD research) + Phase 4C (provider-neutral cross-asset historical market collectors and curated XAUUSD market-driver universe) delivered

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

## Durable repository, dataset versioning, and incremental ingestion (Phase 2)

Phase 2 adds a REPOSITORY layer on top of Phase 1's already-durable
`MarketEventStore`/`FeatureStore` (both were already real, file-backed,
append-only, lock-protected stores -- Phase 2 does not make them durable
for the first time; it adds dataset MANIFESTS, PARTITIONS, INCREMENTAL
ingestion/generation, CHECKPOINTS, RECOVERY, RECONCILIATION, COMPACTION,
and EXPORT on top of them).

### Two identity concepts: `DatasetKey` vs `dataset_id`

Exactly analogous to a git branch vs. a commit:

- **`DatasetKey`** (`manifests.py`) is the STABLE routing/lineage
  identity of "which dataset" -- e.g. "raw XAUUSD from mt5", or "sma_20
  v1 for XAUUSD". It never changes as more data is committed. It is a
  small, caller-declared composite, not content-addressed: `(dataset_kind,
  instrument_id, provider)` for `RAW_MARKET_EVENTS`; `(dataset_kind,
  instrument_id, feature_name, feature_version)` for `DERIVED_FEATURES`
  (mutually exclusive field requirements enforced in
  `DatasetKey.__post_init__`).
- **`dataset_id`** (on `DatasetManifest`) is a per-VERSION,
  content-addressed identity that changes every time new data is
  committed. `DatasetManifest.to_identity_payload()` excludes
  `dataset_id` itself, `physical_digest` (a physical-layout signal --
  see Compaction below), `creation_time` (a caller-supplied operational
  label, exactly like `portfolio_risk.ledger.RiskLedgerEntry.
  recorded_time`), and `completion_status` (always `"complete"` -- a
  manifest object is only ever constructed once a commit is fully
  complete, see Atomicity below). `DatasetManifestStore` retains the
  FULL version history per `DatasetKey` (append-only, never overwritten)
  -- "dataset version identities" means every commit, not only the
  latest.

### Partitioning

`partitions.py` supports DAILY and MONTHLY calendar-based granularities
only -- the "minimum useful set" the specification explicitly permits
choosing; fixed-event-count partitioning is a documented, out-of-scope
extension (every required correctness property -- boundary-timestamp
handling, deterministic membership, checkpoint carry windows -- is fully
exercised by time-based partitioning alone). A `Partition` binds
`dataset_key` + `partition_key` (e.g. `"2026-01-05"`) + an ORDERED
member-id list + `schema_version` + a content digest. Member ordering
within a partition is `(member_time, member_id)` -- NEVER raw
arrival/ingestion order -- so partition (and therefore dataset) identity
depends only on economic content and event-time ordering, never on which
order a caller happened to submit ingestion batches in.
`PartitionStore` keeps only the CURRENT version of each partition file
(atomically replaced via `core.json.write_json_atomic`'s temp-then-rename)
-- a partition is a derived index over already-durable primary facts
(raw events or feature records), always fully reconstructible from them,
unlike the primary event/feature records themselves, which are
append-only and never overwritten.

### Incremental ingestion (`ingestion.py`)

Three orderings, never assumed identical (per the specification's own
explicit requirement):

1. **Provider/source sequence** -- whatever a raw provider itself
   claims; carried, if present, in `source_event_id`; never interpreted.
2. **Repository append sequence** -- `MarketDataEvent.sequence`,
   assigned by the CALLER (via `ingestion.next_sequence_for`) before
   construction, since `sequence` participates in an event's own content
   identity. Events are appended in EXACTLY the order the caller submits
   them -- arrival order, not event-time order.
3. **Event-time ordering** -- partition membership (and therefore
   dataset identity) is always ordered by this.

**Late-arriving historical events -- the chosen, explicit model**: a
late event is appended to the arrival-ordered raw store like any other
event, and the ONE partition its `event_time` belongs to is REBUILT from
its current complete membership, producing a new `partition_id` and
therefore a NEW manifest VERSION. The event is never rejected, and no
existing manifest version is ever mutated. This is "append to an
arrival-ordered raw store while changing canonical event-time views" --
chosen because it neither loses information (rejection) nor requires an
expensive full-dataset rebuild for one late event (a brand-new version
covering the whole history).

**Batch idempotency**: `IngestionBatchStore` is an append-only ledger
(`{batch_id, content_digest, status}`) per `DatasetKey`. A `batch_id`,
once used, is permanently bound to one content digest -- an exact retry
(same `batch_id`, same digest) is idempotently absorbed; a conflicting
retry (same `batch_id`, different digest) fails closed with
`IngestionConflictError`. This is what `MarketEventStore.append`'s own
per-event idempotency does NOT give for free: the underlying store has
no concept of "batch," only individual events.

### Checkpoints (`checkpoints.py`)

A checkpoint is NEVER the primary source of truth -- always
independently re-derivable and re-verified from the underlying manifest/
store state (`verify_raw_ingestion_checkpoint`/
`verify_feature_generation_checkpoint` recompute a fresh checkpoint and
compare field-by-field; a stale checkpoint, behind OR ahead of durable
data, raises `StaleCheckpointError`; a hand-edited one raises
`CheckpointError`).

**Carry state, by design choice**: `FeatureGenerationCheckpoint` stores a
`carry_window_size: int | None` (how many trailing raw candles
incremental generation must re-read before the new batch), NOT cached
partial sums/EMA state. The raw event store is itself durable and
replayable; re-deriving needed context from it via a bounded backward
read is strictly safer than trusting a cached accumulator that could
silently drift with no way to detect it.

### Incremental feature generation -- the core correctness property

`feature_generation.generate_feature_dataset_incremental` must always
produce output IDENTICAL to a full fresh recomputation over the entire
raw history, restricted to one feature. One `DatasetKey`
(`DERIVED_FEATURES`) is exactly one STORED feature name (e.g.
`"sma_20"`), matching `FeatureStore`'s own Phase 1 partitioning -- a
caller wanting several named features calls this function once per
feature, each with its own manifest/partition/checkpoint lineage.

**Carry window per feature** (`carry_window_size_for`):
- Windowed indicators (`rolling_mean`/`rolling_std`/`atr`/`rsi`/`sma`):
  carry = `window` trailing candles.
- Pairwise indicators (`return`/`log_return`/`price_delta`/
  `volume_delta`): carry = 1.
- Pointwise indicators (`high_low_range`/`body_size`/wick ratios):
  carry = 0 (no context needed).
- `vwap` AND `ema`: carry = `None` ("unbounded" -- the ENTIRE raw
  history is always re-read). `vwap` is cumulative since the start of
  the series by definition (Phase 1), so no bounded window suffices.
  `ema` is unbounded for a different, equally fundamental reason (found
  during this phase's own adversarial testing -- see the delivery
  report's "Defects found and fixed"): it is RECURSIVE, seeded by an SMA
  of the first `window` values of whatever series it is given, then each
  value depends on the previous one. A bounded carry window re-seeds EMA
  from a different starting point than a full computation would use --
  and because the recursion never forgets its seed, every subsequent
  value is silently wrong, not confined to one boundary point the way
  `atr`/`rsi`'s own boundary approximation is. Both are a documented,
  deliberate PERFORMANCE (never correctness) limitation.

**`only_persist_timestamps`**: `generate_candle_features` (Phase 1)
gained an optional parameter restricting which computed points are
actually WRITTEN to the store, while every point is still COMPUTED over
the full input series (needed for correct rolling context). Incremental
generation recomputes a bounded leading "carry" window purely for
context and must NEVER re-attempt to persist those carry positions --
`atr`/`rsi` both use an artificial "insufficient prior data" fallback at
position 0 of whatever series they are given (correct for genuine full
history, wrong for a truncated carry window), so re-persisting a
recomputed carry-window value would otherwise spuriously conflict with
the correct value already stored for that timestamp. This parameter
defaults to `None` (persist everything, Phase 1's original behavior) --
zero behavior change for any existing caller.

### Recovery (`recovery.py`)

Recovery NEVER GUESSES: it reconstructs partitions/manifests EXCLUSIVELY
from whatever `MarketEventStore`/`FeatureStore` durably, verifiably hold
RIGHT NOW -- the same "always recompute fresh from current store state"
functions ingestion itself uses, never a diff against a prior manifest,
and never a replay of an ingestion batch's original (not durably
retained) event list. A batch left `RESERVED` without a matching
`COMMITTED` entry is reported PENDING, never fabricated -- the caller
must resubmit that exact batch (idempotently convergent either way).

**Truncated trailing record** -- the one form of "repair" this module
performs: a process killed mid-write of the LAST line of a `.jsonl` file
leaves a partial, unparseable fragment as the final line, with every
earlier line still complete. `read_jsonl_tolerating_truncated_tail`
parses every line strictly and, ONLY if the final line fails, discards
it and reports it; a failure on any non-final line is genuine corruption
and raises `RepositoryCorruptionError`. When a truncated tail is
discarded, `recovery.py` PHYSICALLY rewrites the file (atomically, via
temp-then-rename) to remove the corrupted bytes -- otherwise every
subsequent STRICT read (`MarketEventStore.read_events` etc.) would keep
failing on the same corrupted line.

### Atomicity -- the honest transaction boundary

A completed manifest never points to a missing/partially-written
partition or a digest mismatch, because manifests are always the LAST
thing written in any flow (ingestion, recovery, compaction), always
derived FRESH from already-durably-written partitions/events at the
moment of manifest construction. Multi-file atomicity across
(event-append, partition-write, manifest-append) is NOT attempted as one
indivisible transaction -- instead, a deterministic STAGED protocol with
recoverable states: each individual write (`MarketEventStore.append`,
`FeatureStore.append`, `PartitionStore.write` via `write_json_atomic`,
`DatasetManifestStore.append`) is independently atomic and idempotent,
and `recovery.py`'s "rebuild fresh from current store state" functions
are what make an interruption BETWEEN those individual writes safe --
never a lost or duplicated economic fact, at worst a manifest that has
not yet caught up (which recovery/reconciliation both detect and fix).

### Reconciliation vs. verification -- the honesty split

`reconciliation.py` produces STRUCTURED ISSUES (missing/orphan
partition, wrong digest/count, duplicate coordinate, broken lineage,
stale checkpoint, semantic mismatch) -- ordinary, expected, non-raising
findings, reusing `ml.models.ValidationReport` exactly like
`quality.py`/`verification.py` already do. `verification.py`'s Phase 2
additions (`verify_raw_dataset`/`verify_feature_dataset`) go one step
further: they recompute each PARTITION's and MANIFEST's own content id
from its own recorded fields and compare (forged-identity detection,
distinct from "does the content match the raw store," which
reconciliation already checks), and REUSE Phase 1's
`verify_market_event_store`/`verify_feature_store` directly for
event/feature-level identity (documented reuse, not re-derivation).
`verify_feature_dataset`'s optional
`cross_check_against_fresh_recomputation=True` regenerates every
requested feature fresh, into a throwaway scratch store, straight from
the raw candles the dataset's own manifest claims as lineage, and
compares every resulting value against what is actually stored --
catching COHERENT tampering (a value changed AND consistently
re-hashed) that pure identity/digest recomputation cannot, since a
consistently-forged id is, by construction, invisible to any check that
only asks "does this object reproduce its own id." This is proven, not
merely claimed: `tests/unit/market_data/
test_market_data_repository_verification.py`'s
`TestFreshRecomputeCatchesCoherentTampering` hand-tampers a stored value
AND correctly re-computes its own `feature_id` to match, confirms the
basic (identity-only) check reports zero criticals, then confirms the
cross-check catches it.

### Compaction (`compaction.py`) -- narrowly scoped, by design

The specification marks compaction OPTIONAL ("only if it can be done
safely within scope"). This phase implements the safe subset: rebuilding
every partition and the manifest for a dataset fresh from its own
current durable data -- normalizing physical storage back to exactly
what `build_partition`/the manifest-rebuild functions would produce from
scratch (the identical operation `recovery.py` performs after an
interruption, here invoked as routine maintenance). Deliberately NOT
implemented: combining small partitions into larger ones by CHANGING
granularity (e.g. daily -> monthly) -- doing that safely would require
partition physical paths to be scoped by granularity so old and new
partition files never collide at the same path, a storage-layout change
touching every module that writes a `Partition`. Verified invariants:
event/feature identities, `semantic_digest`, logical ordering, and
replay result never change; `dataset_id` may change if a partition was
physically hand-edited/drifted (compaction corrects it back to canonical
form) -- a new, still fully valid manifest version, never a mutation of
a prior one.

### Deterministic export (`export.py`)

JSON Lines only, not CSV -- every value already round-trips exactly
through this package's own `to_json_dict()` convention (`Decimal` as
exact strings, ISO-8601 UTC timestamps), so JSONL reuses already-tested
serialization rather than a second CSV-specific quoting scheme, and adds
no pandas dependency purely for export. Row order is always
`(timestamp, id)` -- canonical logical order, independent of physical
append order. `export_semantic_digest` is a digest of the exported ROW
SET alone (sorted before hashing, order-independent) -- proven identical
across independent filesystem roots and nested vs. shallow paths in
`tests/unit/market_data/test_market_data_export.py`.

### Concurrency

Every Phase 2 store reuses the SAME `ml.concurrency.experiment_lock`
primitive Phase 1/`portfolio_risk`/`execution_gateway` already build on
-- confirmed, via direct testing, to be FAIL-FAST (a second concurrent
caller for the same lock path is rejected immediately with
`MarketDataLockError`, never made to wait). This means two genuinely
concurrent writers targeting different, pre-assigned repository-append
sequences for the same `DatasetKey` are not both guaranteed to land on
their first attempt -- `ingestion.py`'s own module docstring states that
sequence assignment is the CALLER's responsibility, requiring external
coordination for true multi-writer concurrency. The guarantee this
package actually provides, and the one its own concurrency tests verify:
no corruption ever occurs under a race, any failure is a typed,
retryable `MarketDataLockError` (or an ordinary sequence-gap
`MarketDataPersistenceError` from a caller-side coordination gap, never
silently misclassified as a business/data-quality denial), and a lock
loser always converges to the correct final state on retry. The same
pre-existing, shared `historical.locking.DatasetLock` stale-lock-reclaim
race already documented during Milestone 9 Phase 3/4's own concurrency
testing was independently reproduced again here (confirmed via its own
diagnostic log line, `"Reclaiming unreadable/corrupted dataset lock
file"`); out of this phase's scope to fix, mitigated the same way M9
did -- keeping concurrency-test thread/iteration counts modest.

## Historical ingestion orchestration and offline source adapters (Phase 3)

Phase 3 answers "how does externally acquired historical data actually
get into the Phase 2 repository, with full provenance, deterministically,
and safely resumable after a crash" -- entirely OFFLINE. No network
access of any kind: adapters read local files or in-memory fixtures
only. This layer never connects to Yahoo Finance, FRED, MT5, a broker,
or any live feed.

**Source-neutral adapter contract** (`adapters.py`). `HistoricalSourceAdapter`
is a `Protocol` (structural typing) exposing `source_kind()`,
`source_schema_version()`, `record_kind()`, `content_digest()`,
`byte_size()`, `describe()`, `iter_records()` -- deterministic iteration
of `RawSourceRecord` (untyped TEXT fields exactly as read, never a
prematurely-trusted typed event). An adapter never normalizes, validates,
or decides final repository identity; that is entirely `orchestration.py`'s
job. Three adapters are provided: `csv_adapter.py` (configurable column
mapping, strict/lenient extra-column handling, BOM/quoted-field handling),
`jsonl_adapter.py` (strict per-`RecordKind` schema, JSON numbers rejected
for financial fields to avoid a float intermediate, reuses `core.json.
parse_json_strict` for NaN/Infinity/duplicate-key rejection), and
`InMemorySourceAdapter` (deterministic, for tests).

**Source manifests** (`source_manifests.py`). `SourceManifest` binds a
CONTENT DIGEST -- never a filesystem path -- plus references (not
embedded copies) to the instrument mapping, timeframe mapping, and
timezone policy that apply to it; changing any of those changes
`source_manifest_id` transitively. `creation_time`/`row_count` are
excluded from identity (operational/derived, exactly like `creation_time`
elsewhere in this package).

**Mapping specs** (`mappings.py`). `InstrumentMappingSpec`/
`TimeframeMappingSpec` are immutable, versioned, content-addressed --
NOT a global registry (a caller constructs whatever mapping their own
operation needs). Provider-specific entries take precedence over a
provider-wildcard (`provider=None`) entry; an unmapped symbol/timeframe
fails closed.

**Normalization** (`source_normalization.py`). `parse_source_timestamp`
requires an explicit `TimestampParsingPolicy` (format list; locale-
dependent auto-parsing is forbidden) and reuses `historical.timezones.
localize_broker_timestamps` for DST-ambiguous/nonexistent rejection --
never re-derived. `parse_source_decimal` never routes through `float`,
rejects NaN/Infinity/commas, and normalizes signed zero
(`Decimal("-0")` -> `Decimal("0")`, since the two compare equal but
serialize differently, which would otherwise silently break content-hash
determinism).

**Provenance** (`provenance.py`). `ProvenanceRecord` binds a source row
coordinate (`source_manifest_id` + `row_index`) to the event it produced,
plus every identity involved (mapping/normalization/batch ids) --
answering "which source row produced this event" and the reverse,
independently of any log. `ProvenanceStore.append` is idempotent for an
EXACT retry and fails closed (`ProvenanceError`) for a conflicting one.
`dataset_id` is deliberately excluded from identity (a repository-state
snapshot that can legitimately drift for reasons unrelated to this row;
see the delivery report's defect list) -- `event_id` is what actually
matters for conflict detection.

**Quarantine** (`quarantine.py`). `QuarantineRecord` stores the SAFE,
already-text-only `raw_fields` (never a blob) plus stable, machine-
readable issue codes (18 total as of Phase 4A -- `missing_observation_value`
added for FRED's `"."` missing-value convention, see Phase 4A section
below) and a `RetryEligibility` classification
(RETRYABLE if corrected source content alone could fix it; PERMANENT if
it needs a config/mapping change). Keyed by the PHYSICAL
`(source_manifest_id, row_index)` coordinate; `ingestion_batch_id` is
deliberately excluded from identity so two independent operations
rediscovering the identical bad row converge instead of conflicting (see
delivery report).

**Backfill planning** (`backfill.py`). `create_backfill_plan` is a PURE
function (no I/O) that classifies a requested interval into
partition-aligned missing/overlapping/gap intervals against
caller-supplied `existing_covered_partition_keys`, under an
`OverlapPolicy` (`EXACT_DUPLICATES_ONLY`/`REJECT_ANY_OVERLAP`/
`ALLOW_LATE_ARRIVAL_NEW_VERSION`) and `GapPolicy`
(`ALLOW_AND_REPORT`/`REJECT`/`REQUIRE_EXPECTED_MARKET_CALENDAR`, the last
reusing `calendar.TradingCalendar` directly). Same inputs always produce
the same `backfill_plan_id` and ordered `batches`.

**Orchestration** (`orchestration.py`). The smallest honest stage
machine: `SOURCE_VERIFIED -> PLAN_CREATED -> BATCH_RESERVED ->
ROWS_PARSED -> ROWS_VALIDATED -> EVENTS_NORMALIZED ->
REPOSITORY_COMMITTED -> PROVENANCE_COMMITTED -> CHECKPOINT_COMMITTED ->
VERIFIED -> COMPLETED`, with durable per-stage evidence via a single
mechanism, `OperationStore.advance` (idempotent for an exact retry --
including a full replay of an already-`COMPLETED` operation from stage
one -- fails closed for a conflicting one, rejects illegal jumps/
regressions). The one piece of state that cannot be recomputed fresh on
every call -- the repository append sequence this operation's events
must use -- is durably pinned the first time `BATCH_RESERVED` is
recorded and always reused, so a resumed operation reproduces the exact
same content-addressed `event_id`s as the original attempt. `dry_run=True`
performs the identical parse/validate/normalize computation but writes
nothing; its `resulting_dataset_id` preview reuses the same pure
digest/manifest functions Phase 2's own commit path uses, so the preview
always matches a real commit exactly. `replay_ingestion_operation` is a
thin, explicitly-named alias proving the same call against a fresh
repository reproduces identical results.

**Reconciliation and verification extensions.** `verification.py` gained
`verify_source_manifest`/`verify_backfill_plan`/`verify_provenance_store`/
`verify_quarantine_store`/`verify_historical_ingestion_checkpoint` --
each recomputes an object's own content id from its own recorded fields
(forged-identity detection) and, where relevant, cross-checks against the
repository; none re-reads the original source bytes (the caller's own
`SOURCE_VERIFIED` stage already does that, the strongest available
check). `reconciliation.py` gained `reconcile_historical_ingestion_operation`,
spanning the operation ledger, quarantine store, provenance store,
checkpoint store, and manifest history for one operation.

## Secure external historical collector infrastructure and FRED integration (Phase 4A)

Phase 4A answers "how does the first REAL external network source get
in, without weakening anything Phases 1-3 already guarantee." It adds a
strictly isolated, network-CAPABLE subpackage, `market_data.collectors`,
that never bypasses Phase 3's own ingestion pipeline: every byte a
collector downloads is persisted, verified, and normalized through the
identical source-manifest / provenance / quarantine machinery Phase 3
already shipped, entered through one new bridge (`DatasetKind.
MACRO_OBSERVATIONS`) rather than a parallel path. No other package in
this repository imports `collectors`; nothing outside it ever opens a
socket.

**Required flow** (never short-circuited): remote request -> immutable
request manifest -> raw response bytes -> immutable response manifest
-> Phase 3-compatible source manifest -> strict FRED adapter ->
normalization/validation/quarantine -> durable repository (`macro.
MacroEventStore`). The collector layer never writes a `MarketEvent`/
`MacroEvent` directly; `orchestration.run_fred_macro_ingestion_operation`
is the only path.

**Transport protocol** (`protocols.py`). `HistoricalHttpTransport` is a
`Protocol` (structural typing, mirroring `adapters.
HistoricalSourceAdapter`'s own convention) -- every collector depends on
this shape, never a concrete HTTP library, so collector-level logic is
tested with a deterministic `FakeTransport` double, never a real
connection. `TransportRequest` mandates a non-empty `allowed_hosts`,
positive timeouts, and a non-negative `max_redirects`; `request_time` is
caller-supplied only -- the transport itself never reads the wall clock.

**`StdlibHttpsTransport`** (`transport.py`). No third-party HTTP library
is an existing repository dependency (`pyproject.toml` was checked;
there is none), so this uses only `http.client`/`ssl`/`socket`/
`ipaddress` from the standard library. TWO-PHASE SSRF defense:
1. `_validate_url_static` -- a pure, pre-DNS check: scheme must be
   `https`, no userinfo, host present, host on the caller's EXPLICIT
   allowlist (case-insensitive), host is NOT an IP literal (rejected
   outright, regardless of allowlist -- blocks `127.0.0.1`, private
   ranges, and metadata-service addresses like `169.254.169.254` even if
   a caller ever mistakenly allowlisted one).
2. `_resolve_and_validate_address` -- resolves the (already-allowlisted)
   HOSTNAME via `socket.getaddrinfo` and validates the ACTUAL RESOLVED
   IP is globally routable (`not (is_private or is_loopback or
   is_link_local or is_multicast or is_reserved or is_unspecified)`),
   defending against DNS rebinding (a hostname legitimately public at
   allowlist-check time resolving to a private address by connect time).
   Resolves ONCE, validates every candidate, connects DIRECTLY to a
   validated IP (never re-resolving the hostname internally) -- closing
   the classic time-of-check/time-of-use gap. `_PinnedHTTPSConnection`
   still sends the ORIGINAL hostname via SNI/`Host` for correct TLS
   certificate validation against the pinned IP.

   RESIDUAL, HONESTLY-DISCLOSED LIMITATION: a compromised, ALREADY-
   VALIDATED IP's own routing at the OS/network level is out of this
   layer's control -- no application-layer SSRF defense can fully close
   that gap; `test_collectors_adversarial.py` asserts this limitation
   stays documented, not silently assumed away.

Redirects are NEVER followed unless `allow_redirects=True`; every
redirect hop re-runs the FULL static+DNS validation against its own
target (never trusts the original URL's validation), a scheme downgrade
(https->http) fails closed, and `max_redirects` is enforced. Responses
are read incrementally and bounded by `max_response_bytes` (never
buffered unbounded first); a non-identity `Content-Encoding` is rejected
outright (a decompression-bomb defense -- the decompressed size cannot
be bounded, so compressed responses are refused rather than decoded).
`ForbiddenTransport` raises immediately on `.get()`, proving a code path
(offline replay) makes zero network calls, structurally rather than by
inspection.

**Credential handling.** An API key is supplied by the CALLER, at call
time, as a plain function parameter (`execute_fred_request(..., api_key:
str | None, ...)`) -- never a stored field on any dataclass anywhere in
`collectors/` (enforced two ways: `request_manifest.py`'s structural
`_reject_secret_shaped_keys` blocklist on canonical query params/headers,
raising `SecretExposureError`; and an AST-based safety-scan check that
walks every `@dataclass` in the subpackage and confirms none declares a
credential-shaped field -- see "Safety scan" below). A manifest records
only `credential_mode: "anonymous" | "api_key"`, never a secret value or
a secret-derived digest. The real key reaches exactly one place: the
in-flight `TransportRequest.url` built by `execute_request.
build_transport_request` (Phase 4C: extracted from `fred.py`, where it
originally lived as a private helper -- see that phase's own section),
never logged, printed, or persisted as a whole object.

**Request/response manifests** (`request_manifest.py`/
`response_manifest.py`). Both immutable and content-addressed, mirroring
`source_manifests.py`'s own `to_identity_payload()` pattern exactly.
`CollectorRequestManifest` excludes `request_time` from identity (the
same semantic request resubmitted later is still "the same request");
`CollectorResponseManifest` excludes `received_time`/
`transport_attempt_count` (operational). Response headers are
CANONICALIZED VIA AN ALLOWLIST (`content-type`, `content-length`,
`date`, `last-modified`, `etag`) rather than a blocklist -- deliberately
safer, since an unanticipated future header (a vendor session token, a
new auth scheme) is excluded by default rather than requiring the
blocklist to have already known about it; a second, independent
blocklist check (`authorization`, `set-cookie`, `x-api-key`, ...) still
defends the allowlist itself.

**Raw response cache** (`cache.py`). Content-addressed
(`{response_manifest_id}/manifest.json` + `body.bin`), atomic write, and
idempotent for an exact retry (byte-identical body already on disk);
raises `CacheCorruptionError` for a conflicting write under the same
identity, never silently overwrites. Re-hashes on every read by default
(`verify=True` is the NORMAL path, not an opt-in). A response/request
manifest id is validated as exactly 64 lowercase hex characters before
touching the filesystem -- path traversal is structurally unreachable, not
merely discouraged (reuses `MarketDataPathSecurityError`). A separate,
append-only per-request index (`request_index/{request_manifest_id}/
responses.jsonl`) explicitly models the spec's own required distinction:
`response_manifest_id` is the ACTUAL response's identity (changes if the
bytes differ); `request_manifest_id` is the SEMANTIC request's identity
(stable even when the server legitimately returns new content at a later
`request_time`, e.g. FRED publishing a fresh observation).

**Retry policy** (`retry.py`). `classify_failure`/`plan_next_wait_seconds`/
`parse_retry_after` are PURE (no sleep, no wall-clock read) -- every
retry-DECISION test runs in microseconds. A permanent client error
(400/401/403) is structurally never retryable (`RetryPolicy.__post_init__`
refuses to construct a policy that tries to make one retryable, so this
cannot even be misconfigured). The actual attempt LOOP -- which does call
a transport and does, in real use, sleep between attempts -- lives in
`fred.execute_fred_request`, the one place transport, retry, and
rate-limiting genuinely need to coordinate together; `sleep_fn` is fully
injectable (default `time.sleep`), so tests assert the exact deterministic
attempt sequence without ever waiting.

**Rate limiting** (`rate_limit.py`). A pure, immutable token-bucket model
-- every function returns a NEW `TokenBucketState`, never mutates in
place; the caller supplies `now` explicitly. No global mutable singleton;
rate-limit state never participates in any semantic identity (a
`TokenBucketState` has no `to_identity_payload` at all).

**FRED collector** (`fred.py`, `fred_schemas.py`, `macro_normalization.py`).
`FRED_ALLOWED_HOSTS = frozenset({"api.stlouisfed.org"})` -- the ONLY host
this collector will ever contact. `FRED_EXAMPLE_SERIES` (`DFII10`,
`DGS10`, `CPIAUCSL`, `DFF`) is explicitly documented as examples, never
enforced as an allowlist; `build_fred_request_manifest` accepts any
`series_id`. `parse_fred_json_response`/`parse_fred_csv_response` are
STRICT and TEXT-ONLY: a JSON `value`/`date` must be a JSON string (a JSON
NUMBER is rejected outright -- the single most important rule, since a
number would silently round-trip through float on the way in), required
keys are enforced, undeclared keys are rejected under strict mode.
`is_missing_value` checks FRED's own `"."` convention explicitly --
`macro_normalization.normalize_macro_row` never silently coerces a
missing observation to `0`; it quarantines under the new
`MISSING_OBSERVATION_VALUE` issue code (see below), keeping it distinct
from `INVALID_DECIMAL` (a genuinely malformed, PRESENT value) -- two
different failure modes that call for different remediation.

POINT-IN-TIME SAFETY, the single most important design decision in this
phase: `MacroEvent.event_time` is derived from FRED's `realtime_start`
field (the vintage/publication-date proxy FRED's own schema provides),
NEVER from FRED's `date` field (the OBSERVATION PERIOD a value
describes -- e.g. `"2024-01-01"` for January CPI, published only in
mid-February). Using `date` as `event_time` would be exactly the
look-ahead bias `core.exceptions.PointInTimeViolationError` exists to
catch. Both are parsed as UTC MIDNIGHT (`observation_date_to_event_time`
-- a bare calendar date carries no intraday precision and needs no DST
disambiguation, so this bypasses `source_normalization.
parse_source_timestamp`'s DST machinery entirely). The true observation
period is never lost -- it is preserved in `source_event_id`
(`f"fred:{series_id}:date={date}"`), so a monthly CPI observation's own
monthly meaning survives even though `event_time` reflects the
(potentially much later) vintage date. A response format lacking
per-row vintage information (this module's own CSV parsing path) cannot
honestly derive `event_time` this way; `normalize_macro_row` accepts an
explicit `default_realtime_start` for exactly that case and quarantines
(`EMPTY_TIMESTAMP`) a row with neither.

`FredSourceAdapter` implements Phase 3's `HistoricalSourceAdapter`
Protocol structurally and performs ZERO network I/O itself (mirrors
`CsvCandleAdapter` reading a local file); `load_fred_adapter_from_cache`
is the ONLY construction path, reading exclusively from `RawResponseCache`
-- "persist raw downloaded bytes before parsing" is enforced
structurally, not by convention, since there is no constructor that
accepts raw bytes directly from a live transport call.

**Unit mapping** (`macro_normalization.py`). `UnitMappingSpec` mirrors
`mappings.py`'s own versioned, content-addressed pattern exactly.
`scale_factor` defaults to `1` (store exactly as FRED reports it, only
labeled with the correct `MacroUnit`); a genuine conversion (e.g. percent
-> basis points) is an explicit, non-default `scale_factor`, applied via
Decimal multiplication, never float. An unmapped series fails closed
(`UNKNOWN_SYMBOL`).

**Narrow, additive extensions to already-shipped Phase 1-3 infrastructure**
(the same "reuse where reuse is cheap, extend narrowly where it is not"
discipline Phase 3 itself established): `DatasetKind.MACRO_OBSERVATIONS`
(scoping-only identity for `ProvenanceStore`/`QuarantineStore`/the new
`CollectorOperationStore` storage paths -- does NOT participate in
`DatasetManifest`/`PartitionStore` versioning, which stays
candle/tick/quote/trade-shaped only; the durable economic record store
for macro data remains Phase 1's own `macro.MacroEventStore`, constructed
directly since `MarketDataRepository` has no `macro_event_store` field);
`RecordKind.MACRO_OBSERVATION` and `SourceKind.FRED_API`
(`source_manifests.py`); `MISSING_OBSERVATION_VALUE`, the 18th quarantine
issue code, classified `PERMANENT` (a missing observation is not fixed by
resubmitting the identical request).

**Orchestration** (`collectors/orchestration.py`). A SEPARATE,
self-contained `CollectorOperationStage`/`CollectorOperationStore` --
deliberately NOT a reuse of Phase 3's own `OperationStore`/
`IngestionStage` (which are typed specifically to committing
`MarketDataEvent`s via `ingest_raw_events`, a materially different commit
target than a `MacroEvent`; genericizing them would have touched
`reconciliation.py`'s multiple `is IngestionStage.X` comparisons and
existing Phase 3 tests -- too wide a blast radius for this phase). The
11-stage machine duplicates the SAME proven idempotent/conflict/
monotonic-progression algorithm `OperationStore.advance` already
established, applied to a new but structurally identical problem:
`REQUEST_PLANNED -> REQUEST_MANIFEST_COMMITTED -> RESPONSE_DOWNLOADED ->
RAW_RESPONSE_COMMITTED -> RESPONSE_VERIFIED -> SOURCE_MANIFEST_CREATED ->
SOURCE_PARSED -> NORMALIZED_RECORDS_PRODUCED ->
REPOSITORY_INGESTION_COMMITTED -> PROVENANCE_COMMITTED ->
VERIFICATION_COMPLETED`.

`FetchMode.FRESH` calls `execute_fred_request` for real (via an injected
transport); `FetchMode.CACHED_REPLAY` reads an already-cached response by
`request_manifest_id` (or a caller-pinned `reference_response_manifest_id`)
and NEVER touches a transport -- a caller wanting a structural guarantee
of this passes `transport=None` (the default; untouched in
`CACHED_REPLAY` mode) or a `ForbiddenTransport`. `fetch_mode` and
`reference_response_manifest_id` are deliberately EXCLUDED from the
operation's own content-digest identity: they describe HOW an operation
is executed THIS particular time, not WHAT it semantically is -- this is
what lets an exact retry of the same `operation_id` switch fetch modes
and still be recognized as the same operation, while a FRESH retry that
happens to pull genuinely DIFFERENT response bytes is still caught, one
level down, as a per-stage evidence conflict at `RESPONSE_DOWNLOADED`
(modeling the spec's own "a fresh-network policy may produce a new
response version" case explicitly, without conflating it with an exact
retry).

The raw-response CACHE WRITE itself is unconditional, even under
`dry_run=True` -- "persist raw response bytes before parsing" is a
structural invariant Stage 7 depends on (`load_fred_adapter_from_cache`
reads exclusively from the cache), not a business-record commit; writing
to a content-addressed, idempotent store carries no semantic weight of
"this operation happened" the way appending to the operation ledger/
quarantine/provenance/macro-event stores does. `dry_run=True` therefore
still exercises the full parse/normalize computation (an honest preview)
while writing NOTHING to any of those four business-record stores.

A PRE-FLIGHT provenance-conflict check runs before Stage 9's repository
write (mirroring the exact defect-preventing pattern Phase 3's own
`orchestration.py` already established): if the row this operation would
produce resolves to a DIFFERENT `event_id` than an already-bound
`(source_manifest_id, row_index)` coordinate, the operation aborts before
touching `macro.MacroEventStore` at all -- preventing an orphaned event
with no matching provenance record. `ProvenanceStore.append` itself
independently refuses a conflicting write too (defense in depth). One
consequence, deliberately not softened: a genuinely NEW `operation_id`
over ALREADY-INGESTED source data is correctly REJECTED, not silently
re-ingested under a new sequence number -- exact retry of already-
committed rows must reuse the SAME `operation_id`, identical to Phase 3's
own documented limitation for historical ingestion.

**Verification** (`collectors/verification.py`). Reuses `market_data.
verification`'s own `ValidationIssue`/`ValidationReport`/
`ValidationSeverity` vocabulary directly. Unlike reconciliation (below),
`verify_fred_macro_operation` REDERIVES every artifact fresh from
caller-declared collector construction parameters -- rebuilding the
request manifest via `fred.build_fred_request_manifest`, re-reading raw
bytes via the cache, reparsing them via `fred_schemas.parse_fred_*_response`
directly (never `load_fred_adapter_from_cache`, which exists to serve
orchestration, not verification), and re-normalizing every row via
`macro_normalization.normalize_macro_row` -- the same PURE functions
orchestration used, invoked completely independently against durable
artifacts only. Never trusts a cached parsed result.

**Reconciliation** (`collectors/reconciliation.py`). Complementary to
verification: needs no original construction parameters, only a
`provider`/`series_id`, scanning ALREADY-STORED evidence purely against
ITSELF -- the shape of check an ops/recovery tool runs without knowing
how an operation was originally built. Detects: missing raw response,
digest mismatch, truncated payload, unexpected content type, wrong
request/response linkage, missing/duplicate observation, a repository
record with no matching provenance, provenance for an event absent from
the repository, a series ingested under two DIFFERENT unit mappings, and
conflicting vintages (the same observation date and the same vintage
recording two DIFFERENT values). "STALE CHECKPOINT," SCOPE NOTE: Phase
4A deliberately did not add a separate `CollectorCheckpoint`/
`CollectorCheckpointStore` -- Phase 3's own `checkpoints.py` is shaped for
partitioned, multi-batch raw-ingestion backfills, a materially different
resumability concern than a single per-`operation_id` FRED fetch, which
`CollectorOperationStore` already makes fully resumable on its own; the
honest analogue reported here is an operation ledger entry that never
reached `VERIFICATION_COMPLETED` (`stalled_operation`).

**Reports** (`collectors/reports.py`). Mirrors `market_data.reports`'s
own convention: every function wraps an object this package already
produces into a stable, deterministic dict, never re-deriving new facts.
`generate_quarantine_summary_report`/`generate_provenance_summary_report`/
`generate_reconciliation_report`/`generate_verification_report` are
RE-EXPORTED UNCHANGED from `market_data.reports` (all four are already
`DatasetKey`-generic, so Phase 3's own implementations serve macro
datasets exactly as they are). Every report here is secret-free BY
CONSTRUCTION -- assembled purely from objects that are themselves
structurally secret-free; no function in this module ever even accepts a
raw credential as an argument.

**Safety scan** (`test_market_data_safety_scan.py`). `_all_source_files()`
now uses `rglob`, reaching `collectors/*.py`; two of the pre-existing
checks (`_FORBIDDEN_NETWORK_IMPORTS`, `_CREDENTIAL_FIELD`) are scoped
away from `collectors/` (which legitimately needs `socket` in
`transport.py` and has function parameters named `api_key`) and replaced
by narrower, more precise `TestCollectorSpecificSafety` checks: an
AST-based scan confirming no `@dataclass` field (as opposed to a function
parameter -- structurally impossible to confuse, since a parameter lives
in `FunctionDef.args`, never a class body) is ever credential-shaped;
confirmation that `socket` is imported nowhere in `collectors/` except
`transport.py`, and that no third-party HTTP/WebSocket library is
imported even there; HTTPS-only-scheme and single-exact-hostname
allowlist assertions; direct SSRF non-vacuity proofs (HTTP URL rejected,
localhost rejected); and a scan for an accidentally-committed long
opaque literal following `api_key=`, applied to BOTH the `collectors/`
source and this test suite's own `test_collectors_*.py` fixtures.

## Curated FRED macro universe and verified historical backfill workflow (Phase 4B)

Phase 4B answers "how does XAUUSD research get a CURATED, versioned,
point-in-time-safe macro-driver universe on top of Phase 4A's generic
FRED collector infrastructure, without weakening any guarantee Phases
1-4A already established." It adds `market_data.collectors.curated`, a
new subpackage that REUSES Phase 4A's transport/retry/rate-limit/cache/
request-and-response-manifest machinery unchanged, and REUSES Phase 3's
`ProvenanceStore`/`QuarantineStore` (already record-kind-agnostic) via
the same `DatasetKind.MACRO_OBSERVATIONS` scoping Phase 4A established
-- while introducing its own new record types, stores, and a THIRD
independent stage-machine implementation where the shape of the problem
(many series per operation, not one) is genuinely different from either
existing one.

**Curated series registry** (`registry.py`). `CuratedFredSeriesSpec` is
an immutable value object (series_id, canonical_series_name,
registry_version, tier, economic_category, expected native
frequency/units/seasonal-adjustment, target_macro_instrument_id,
normalization_kind, unit_conversion, missing_value_policy, revision/
availability policy id references, default_observation_start, request
overrides, enabled, notes); `notes` is the ONE field excluded from
identity (documentation only, never computation-affecting).
`CuratedFredRegistry` is a KEYED SET, not an ordered list:
`create_curated_registry` always SORTS specs by `series_id` before
computing `registry_id`, so identity AND iteration order are BOTH
independent of declaration order (`create_curated_registry(specs=(a,
b))` and `(b, a)` produce the identical `registry_id`). Rejects
duplicate series ids, duplicate canonical names, an enabled series with
no target instrument id, and an unsupported frequency/unit/normalization
combination. `default_core_series_specs` returns exactly the 4 MANDATORY
series (`DFII10`->`us_10y_real_yield`, `DGS10`->`us_10y_nominal_yield`,
`CPIAUCSL`->`us_cpi_all_urban`, `DFF`->`effective_federal_funds_rate`),
all `SeriesTier.CORE_XAUUSD_DRIVER`, all `enabled=True`.
`default_extended_series_specs` returns the 14 reviewed extended
candidates (`T10YIE`, `T5YIE`, `DGS2`, `DGS5`, `DGS30`, `DTWEXBGS`,
`UNRATE`, `PAYEMS`, `PCEPI`, `PCEPILFE`, `INDPRO`, `VIXCLS`, `WALCL`,
`M2SL`) -- ALL constructed `enabled=False` by deliberate design: only
the 4 core series are individually fixture-verified this phase; adding
one to a live universe requires an explicit opt-in flip, disclosed as a
scope decision, not an oversight.

**Official metadata contract** (`fred_series_metadata.py`,
`metadata.py`). `FRED_SERIES_ENDPOINT_PATH = "/fred/series"` (metadata)
is a SEPARATE endpoint from Phase 4A's `/fred/series/observations`;
`execute_fred_series_metadata_request` is a THIN ALIAS for `fred.
execute_fred_request` (zero duplication -- `execute_request.
build_transport_request` already builds URLs generically from
`manifest.endpoint_host/endpoint_path/canonical_query_params`, so the
same attempt loop serves
both endpoints). `parse_fred_series_metadata_response` is PARSE-ONLY
(mirrors `fred_schemas.py`'s own layering) and never compares the
returned `series_id` against a "requested" one -- that drift decision
belongs entirely to `metadata.verify_series_metadata`, which NEVER
trusts a manually-written curated label over the official response.
DRIFT POLICY, exactly as specified: unexpected series id, incompatible
frequency, incompatible units, or a changed seasonal-adjustment code
(only where the spec declared an expectation) each FAIL CLOSED; a
changed title is informational only (no curated expectation exists to
compare it against); a changed supported observation range is REPORTED,
and the requested backfill interval is permitted to proceed only if it
still falls within the metadata's own currently-reported range
(`last_updated` is captured for provenance only, never compared).

**Revision policy** (`revision_policy.py`). Four kinds --
`LATEST_AVAILABLE` (no override; NOT automatically point-in-time-safe on
its own), `FIRST_RELEASE_ONLY` (FRED `output_type=4`), `AS_OF_REALTIME_
DATE` (requires an explicit `as_of_realtime_date`; resolves `realtime_
start=realtime_end=` that date), `VINTAGE_SERIES` (FRED `output_type=2`,
retaining distinct revisions) -- with `resolve_fred_request_overrides`
as the SINGLE place FRED-specific parameter names (`output_type`,
`realtime_start`, `realtime_end`) are ever produced from a named
`RevisionPolicyKind`; every other module reasons purely in terms of the
4 kinds, never raw FRED parameter names. `create_revision_policy`
rejects `as_of_realtime_date` supplied on any kind other than `AS_OF_
REALTIME_DATE` (an invalid combination, caught at construction, not at
request time).

**Point-in-time availability** (`availability.py`) -- THE SINGLE MOST
IMPORTANT DESIGN DECISION IN THIS PHASE, extending Phase 4A's own
`realtime_start`-not-`date` discipline with an explicit, versioned,
identity-relevant POLICY rather than a single hardcoded rule. Four
distinct times are never conflated: `observation_date` (the economic
period FRED reports, e.g. `"2024-01-01"` for January CPI), `realtime_
start`/`realtime_end` (FRED/ALFRED's own real-time validity window),
`availability_time` (the earliest time this platform permits
POINT-IN-TIME use of the value -- what every downstream PIT join must
actually filter on), `ingestion_time` (purely operational). Six
`AvailabilityPolicyKind`s: `OBSERVATION_DATE_END_OF_DAY` (daily market
rates -- available end of the observation day itself, in an explicit
timezone), `NEXT_BUSINESS_DAY_CONSERVATIVE` (simple Mon-Fri arithmetic,
NO public-holiday awareness -- a disclosed limitation, not silently
assumed away), `EXPLICIT_RELEASE_TIMESTAMP` (a caller-supplied exact
datetime; forbids also supplying time-of-day fields, since they would be
redundant/conflicting), `RELEASE_CALENDAR_REFERENCE` (deliberately
UNIMPLEMENTED this phase -- `resolve_availability_time` raises
`AvailabilityPolicyError`, an honest "not yet supported" rather than a
silent fallback), `REALTIME_START_DATE_CONSERVATIVE` (monthly releases
like CPI -- availability is the series' own `realtime_start`, at an
explicit time of day; REQUIRES a `realtime_start_text`, fails closed
-- `AvailabilityUnresolvedError` -- if one is not available),
`MANUAL_CURATED_RELEASE_RULE` (the only kind permitting an explicit
`delay_days` override). `resolve_availability_time` is STRUCTURALLY
fail-closed: no code path returns "immediately available" as a silent
default -- every branch either resolves a concrete, timezone-aware
datetime or raises. FRED reports only DATES, never exact publication
times; this phase does not fabricate false timestamp precision -- an
`availability_hour`/`availability_minute` is an explicit, configurable,
DISCLOSED-AS-APPROXIMATE convention (not a claim of a real published
release time), and the policy object itself (including its timezone and
hour/minute) is fully identity-relevant, so a policy CHANGE is a
detectable, auditable event, never silently reinterpreted in place.

**Curated macro observation** (`macro_observation.py`).
`CuratedMacroObservation` is a materially richer record than Phase 1's
`MacroEvent` -- native AND normalized unit, native frequency, the full
vintage/realtime lineage, the resolved `availability_time` AND the
policy id that produced it, and direct request/response/source manifest
references -- so it lives in its OWN new store rather than being forced
into `macro.MacroEventStore`. IDENTITY DISTINGUISHES ECONOMICALLY
DISTINCT REVISIONS BY CONSTRUCTION: `observation_id` is content-addressed
over EVERY field, including `realtime_start`/`value`/`availability_
policy_id` -- two vintages of the same `observation_date` with a
different value, or the same value resolved under a different policy,
always produce DIFFERENT ids; nothing ever collapses them. Deliberately
NO `provenance_id` field (would be circular -- a `ProvenanceRecord` is
built FROM this observation's own id); Phase 3's `ProvenanceStore` is
reused unchanged for that binding. `CuratedObservationStore` is PURELY
content-addressed append-only (no sequence numbers needed, unlike
`MacroEventStore`) -- "append" is naturally idempotent by simple
id-membership check under its own per-series lock. `append_many_and_
read_all` performs an append-then-read as ONE atomic, lock-held
operation (rather than a separate locked `append()` loop followed by a
separate UNLOCKED `read_observations()` call) -- closing a real race
window a concurrent caller could otherwise hit between a write
completing and a later read observing it; discovered and hardened
during this phase's own concurrency testing (see "Known limitations"
for the fuller story of that investigation).

**Missing-value policy.** FRED's `"."` NEVER becomes zero, matching
Phase 4A's own discipline. Per-series `MissingValuePolicy`:
`QUARANTINE` (default -- the safest, most conservative choice),
`STORE_AS_MISSING_FACT` (durably records `is_missing=True`, `value=
None`), `SKIP_AND_REPORT` (excluded WITHOUT quarantining, but always
counted -- never silently dropped). No forward-fill anywhere in this
package.

**Curated backfill spec** (`backfill.py`). Immutable, content-addressed
multi-series plan: `curated_registry_id`, `selected_series_ids`
(ALWAYS sorted by `create_curated_backfill_spec` -- the SAME mechanism
satisfies both "identity independent of declaration order" and
orchestration's own "stable series processing order" requirement),
observation window, optional realtime window, `revision_policy_id`,
output type, page size, `CachePolicy` (`PREFER_CACHE`/`FORCE_FRESH`),
optional registry-wide availability/normalization overrides,
`target_dataset_namespace`, `fail_fast`, and three explicit BOUNDS
(`max_series_count`, `max_observations_per_series`,
`max_total_raw_bytes`) -- an unbounded request is structurally
unconstructable. `create_curated_backfill_spec` validates every
selected series against the supplied registry BEFORE a spec can exist
at all: unknown or disabled series is rejected at construction, not at
run time.

**Multi-series orchestration** (`orchestration.py`, ~600 lines, the
largest module in this phase). `CuratedOperationStage`/
`CuratedOperationStore` are a THIRD, independent, small stage-machine
implementation (after Phase 3's `OperationStore` and Phase 4A's
`CollectorOperationStore`), duplicating the SAME proven idempotent/
conflict/monotonic-progression `advance()` algorithm a third time --
scoped to `target_dataset_namespace` (not `operation_id` + `DatasetKey`)
because ONE operation here spans MANY series at once, a materially
different shape than either existing single-series-scoped machine
commits to. The 12-stage machine: `REGISTRY_VERIFIED -> PLAN_CREATED ->
SERIES_METADATA_VERIFIED -> REQUESTS_COMMITTED -> RESPONSES_COMMITTED ->
OBSERVATIONS_PARSED -> AVAILABILITY_RESOLVED -> SERIES_DATASETS_
COMMITTED -> COMBINED_MANIFEST_COMMITTED -> RECONCILED -> VERIFIED ->
COMPLETED`.

Per series, in the SORTED (deterministic) order `CuratedBackfillSpec.
selected_series_ids` already guarantees: fetch-or-replay metadata,
verify it against the curated spec (fail-closed drift aborts that
series), fetch-or-replay observations, re-hash-verify the raw bytes,
parse strictly, resolve availability/normalize/quarantine each row via
`_normalize_curated_row` (a PURE row processor, reused UNCHANGED by
`verification.py` for independent rederivation), run the SAME pre-flight
provenance-conflict check Phase 3/4A established, then commit the
series' own component dataset. `backfill_spec.fail_fast=True` (the
default) re-raises immediately the moment ANY series fails ANY stage --
nothing is committed for ANY series, `COMPLETED` is never reached.
`fail_fast=False` records each series' own `SeriesOutcome` independently
and continues; the combined manifest's `completeness_status` becomes
`PARTIAL` the moment even one series failed, `COMPLETE` only if every
selected series succeeded; zero successes still raises (nothing to
commit). Both `CachePolicy.PREFER_CACHE`/`FORCE_FRESH` and cache-vs-
transport fetch mode are decided INDEPENDENTLY per series inside the
loop, never once for the whole operation. The raw-response cache write
is unconditional even under `dry_run=True` (identical rationale to
Phase 4A); only the business-record stores (observation/component/
combined-manifest/provenance/quarantine/operation-ledger) are `dry_run`
-gated.

**Dataset layout** (`datasets.py`). One immutable `ComponentDatasetManifest`
PER SERIES plus one `CombinedUniverseManifest` binding the exact
component versions that make up one curated-universe backfill --
DIFFERENT native frequencies (`DGS10` daily, `CPIAUCSL` monthly) are
NEVER forced into one physically regular time series; no implicit
resampling, forward-fill, or alignment happens anywhere in this package.
VERSIONING IS A STORE-LEVEL CONCEPT, not baked into either manifest's
own content hash: `component_manifest_id`/`combined_manifest_id` are
each content-addressed purely from observed facts (which observations
are included, coverage, missing/revision counts, the exact component
ids bound); "version N" is simply "the Nth distinct manifest ever
durably recorded" (`len(history)`), read from the append-only store.
Both stores' `append` is IDEMPOTENT no-op (returns the existing version
number unchanged) when the incoming content id already matches the
CURRENT one -- this idempotency IS the mechanism that satisfies "an
exact no-op update must mint no new dataset version," not a separate
special case anywhere else.

**Incremental update planning** (`update_plan.py`). PURE and
deterministic: NEVER reads the wall clock ("today"), never touches the
network. Compares each selected series' CURRENT `ComponentDatasetManifest`
coverage against a caller-supplied `desired_observation_end` and
`RevisionPolicy`, producing one of `NO_UPDATE_NEEDED`, `APPEND_
OBSERVATIONS` (from the day after the current coverage end, or the
series' own `default_observation_start` if it has no history yet), or
`REVISION_REFRESH` (the revision policy in effect CHANGED since the
existing combined manifest was built -- already-covered dates may now
resolve to different vintages, so a full refresh is required).
`planning_time` is excluded from the plan's own identity (two plans
computed minutes apart from identical inputs are the SAME plan); this
module's job is only to REPORT the no-op case correctly -- the actual
"no new version" guarantee lives in `datasets.py`'s own idempotent
`append`, as above.

**Reconciliation** (`reconciliation.py`) and **verification**
(`verification.py`) mirror Phase 4A's own honesty split exactly.
Reconciliation scans ALREADY-STORED evidence purely against itself (no
original construction parameters needed): registry-vs-combined-manifest
linkage, component manifest version linkage, coverage/observation-count
recomputation from the observation store, conflicting-vintage detection
(same `observation_date` + `realtime_start` recording two DIFFERENT
values), and provenance completeness (both directions). Verification
takes the ORIGINAL construction parameters plus a `CuratedIngestionReport`'s
own `SeriesOutcome`s and REDERIVES everything fresh, independently, using
the exact same pure functions orchestration used: registry self-check
identity, per-series response-manifest re-hash, a STRICT reparse of the
cached raw bytes, recomputed availability/normalized values via
`_normalize_curated_row` (imported directly from `orchestration.py`),
recomputed component/combined manifest self-check identities and
recomputed `completeness_status` -- never trusting a cached parsed
observation, a recorded count, or any final "is_verified" flag anywhere.

**PIT consumer contract** (documented, not yet a joined feature --
matching Phase 4A's own "modeled but not wired into feature generation"
discipline): a future consumer joining curated macro data into market
bars MUST filter on `availability_time`, never `observation_date` alone
-- `test_collectors_curated_pit_concurrency_adversarial.py` proves this
concretely (an earlier-`observation_date`-but-later-`availability_time`
value stays invisible before its availability time) and
`test_market_data_safety_scan.py`'s new `TestCuratedSpecificSafety`
class makes any FUTURE reintroduction of the anti-pattern (comparing
`observation_date` directly against a time value) structurally loud,
not merely documented in prose.

**Real-FRED acceptance workflow** (`acceptance.py`). `acceptance.py` is
the ONE place anywhere in `collectors/` (curated included) that reads an
environment variable (`FRED_API_KEY_ENV_VAR = "FRED_API_KEY"`) --
explicitly the sanctioned exception to "no arbitrary environment reads
in pure domain code." `resolve_fred_api_key_from_environment` never
raises and never falls back to a placeholder; `run_real_fred_acceptance_
workflow` REQUIRES an explicit, non-empty `api_key` argument (disabled
by default), runs the full pipeline over a caller-BOUNDED interval with
a small page size, then a SECOND `ForbiddenTransport`-backed pass over
the SAME `operation_id` to prove offline replay makes zero network
calls, comparing the two runs' semantic results. `RedactedAcceptanceReport`
is structurally incapable of carrying a secret -- its own field set has
no place to put one. The corresponding pytest test resolves the key via
the same function and calls `pytest.skip(...)` with a precise reason
when absent (the expected state for ordinary CI and this offline
development environment); a missing credential is never treated as an
application failure, and the ordinary full suite never requires a key or
network access.

**Mandatory fixture-based acceptance** (`test_collectors_curated_
fixture_acceptance.py`). The ALWAYS-RUN, no-network counterpart:
realistic, hand-built FRED fixtures covering all 4 core series across
two native frequencies, one missing observation (a weekend "."), and one
revision (`DGS10`, same `observation_date`, two vintages with different
values and `realtime_start`s) -- exercising the complete pipeline
(orchestration -> component/combined datasets -> reconciliation ->
verification -> offline replay) fully deterministically. Also proves two
properties beyond the ordinary happy path: running the identical
semantic workflow from two INDEPENDENT temp repositories converges on
byte-identical `combined_manifest_id`/component/observation ids (proving
identity is purely content-addressed, never influenced by filesystem
path), and running it in two child interpreters under DIFFERENT explicit
`PYTHONHASHSEED` values produces the identical `combined_manifest_id`
(proving identity never depends on Python's per-process-randomized
`hash()`/dict-iteration order, only on `compute_content_id`'s canonical,
sorted-key JSON + sha256).

## Provider-neutral cross-asset historical market collectors and curated XAUUSD market-driver universe (Phase 4C)

**Primary goal.** Deliver a provider-neutral, deterministic,
point-in-time-aware historical market-data collection layer for the
CROSS-ASSET variables that influence XAUUSD (US dollar strength, WTI and
Brent crude, silver, and the gold reference market itself, plus five
strong-optional regime/secondary concepts) -- distinct from Phase 4B's
FRED MACRO universe: this phase collects tradable-instrument market bars
(OHLCV), not economic-release observations, and the provider surface is
architected to support MULTIPLE providers/instrument forms per concept
from day one, unlike Phase 4B's single-source FRED design.

**Package boundary.** Entirely new subpackage,
`collectors/cross_asset/` (13 core modules + `providers/alpha_vantage.py`
+ package `__init__.py` = 22 files, ~2,300 lines): `instrument_form.py`,
`adjustment.py`, `sessions.py`, `futures.py`, `availability.py`,
`registry.py`, `symbol_mapping.py`, `market_record.py`, `protocols.py`,
`market_normalization.py`, `market_backfill.py`, `datasets.py`,
`gap_policy.py`, `market_orchestration.py`, `update_plan.py`,
`market_reconciliation.py`, `market_verification.py`,
`market_reports.py`, `acceptance.py`. One shared, provider-neutral
extraction lives one level up: `collectors/execute_request.py`
(`CollectorRequestExecution`, `build_transport_request`,
`execute_collector_request`) -- Phase 4A's `fred.py` attempt loop was
ALREADY 100% provider-neutral in its actual implementation (no
FRED-specific logic anywhere in it), so this phase promotes it to a
shared home rather than duplicating it a second time; `fred.
execute_fred_request` is now a THIN ALIAS of `execute_collector_request`
(confirmed zero behavioral change via a full Phase 1-4B regression
re-run before and after the extraction). No file outside
`collectors/cross_asset/` (and this one shared extraction) was modified
except two stale-docstring-reference fixes and two narrow, additive
`DatasetKind`/`SourceKind`/`RecordKind`/exception additions (below).
`ml`/`execution_gateway`/`portfolio_risk`/`paper_trading`/`backtesting`/
`optimization`/`robustness` were not touched.

**Provider selection** (bounded assessment against OFFICIAL
documentation only, never remembered/guessed behavior). Three
candidates assessed: **Stooq** disqualified outright -- no official,
documented API exists, only reverse-engineered CSV endpoints (exactly
the "undocumented endpoint guessing" this phase's scope forbids).
**Alpha Vantage** and the **EIA Open Data API** are both officially
documented; Alpha Vantage's `TIME_SERIES_DAILY` endpoint was
additionally LIVE-VERIFIED via a real HTTPS `GET` against the
provider's own public `demo` key (confirmed the exact JSON envelope
shape this phase's adapter parses, and confirmed the `demo` key is
restricted to a small fixed symbol set, `IBM` only -- not a general
free-tier credential). The EIA route structure was confirmed live via a
real `API_KEY_MISSING` 403 JSON error, but no actual data response was
obtainable without registering a real account, which this phase
declines to do autonomously (a consequential external-identity action
requiring the user's own involvement). Alpha Vantage's dedicated
commodity endpoints (`WTI`/`BRENT`/`GOLD_SILVER_SPOT`) exist in official
documentation but could not be live-verified the same way (the `demo`
key does not cover them) -- **this phase implements ONLY the ONE
endpoint that was genuinely, live-verified: `TIME_SERIES_DAILY`**, used
exclusively for ETF-form PROXY instruments. Free tier confirmed via the
provider's own pricing page: 25 requests/day. This is a deliberately
conservative, self-imposed narrowing beyond what the spec strictly
requires, justified by "do not fabricate an integration."

**Instrument-form and proxy semantics** (`instrument_form.py`) -- the
central discipline this phase's package exists to enforce: an economic
CONCEPT (e.g. "WTI crude oil") is never the same object as a specific
tradable INSTRUMENT FORM that approximates it. `InstrumentForm` (8
values: `SPOT`, `CASH_INDEX`, `EXCHANGE_FUTURES_CONTRACT`,
`PROVIDER_CONTINUOUS_FUTURES`, `ETF`, `EQUITY`, `SYNTHETIC_INDEX`,
`ECONOMIC_PROXY`) names the SHAPE of the tradable object; `ProxyPolicy`
(`is_proxy`, `proxy_for`, `proxy_quality: HIGH|MODERATE|LOW`, plus six
free-text risk-disclosure fields for basis/roll/tracking-error/currency/
session/adjustment differences) names how faithfully it approximates the
concept. `ProxyPolicy.__post_init__` structurally enforces `proxy_for`/
`proxy_quality` required iff `is_proxy=True`, forbidden otherwise --
`create_proxy_policy` does NOT duplicate this check itself, so every
violation surfaces through exactly one exception type
(`InstrumentFormError`) regardless of construction path (a real
inconsistency was found and fixed here during this phase's own test
authoring: the factory originally raised a different, generic exception
than the dataclass's own guard for the identical violation).

**Curated cross-asset driver registry** (`registry.py`). Mirrors Phase
4B's `curated.registry` keyed-set pattern exactly: `CuratedMarketDriverSpec`
is a per-ECONOMIC-CONCEPT value object (`canonical_driver_id`,
`canonical_name`, `registry_version`, `tier: DriverTier`, `economic_role`,
`is_required`, `asset_class`, `preferred_instrument_form`,
`allowed_instrument_forms`, `canonical_currency`, `canonical_quote_unit`,
`expected_frequency`, `session_policy_id`, `adjustment_policy_id`,
`availability_policy_id`, `continuation_policy_id` (futures forms only),
`provider_mapping_ids`, `enabled`, `notes`); `CuratedMarketDriverRegistry`
ALWAYS sorts by `canonical_driver_id` before computing `registry_id`, so
identity AND iteration order are both independent of declaration order.
`DriverTier` has four values: `CORE_XAUUSD_MARKET_DRIVER`,
`SECONDARY_MARKET_DRIVER`, `REGIME_CONTEXT`, `EXPERIMENTAL`. Construction
rejects: duplicate ids/names, empty/duplicate `allowed_instrument_forms`,
`preferred_instrument_form` not in `allowed_instrument_forms`, a futures
form present without `continuation_policy_id` (and vice versa), `enabled`
without a non-empty `provider_mapping_ids`, and -- via `create_curated_
market_driver_spec`'s own semantic check (it accepts the RESOLVED
`AdjustmentPolicy` object, not merely an id, specifically to enforce
this) -- an ETF/equity-allowing spec declaring a non-equity-like
adjustment kind.

**The 10 curated concepts** (`default_core_market_driver_specs`/
`default_optional_market_driver_specs`, mirroring Phase 4B's
`default_core_series_specs`/`default_extended_series_specs` shape --
pure factories taking policy ids/objects and a `provider_mapping_ids_
by_driver` dict, never constructing a module-level singleton): 5
MANDATORY core drivers (`us_dollar_strength`, `wti_crude`,
`brent_crude`, `silver`, `gold_reference`; `is_required=True` on every
one, `tier=CORE_XAUUSD_MARKET_DRIVER`) and 5 strong-optional drivers
(`us_equity_market_stress`, `treasury_volatility`,
`broad_commodity_index`, `copper_industrial_growth`, `gold_miner_equity`;
`is_required=False`). `enabled` is derived from whether the caller
actually supplied a provider mapping for that driver id -- a required
concept EXISTS in the registry even when this phase's own provider
cannot supply it. Of the 10, **9 are mapped to real Alpha Vantage ETF
proxies this phase** (`UUP`, `USO`, `BNO`, `SLV`, `GLD`, `VIXY`, `DBC`,
`CPER`, `GDX`); **`treasury_volatility` ships UNSUPPORTED AND
FAIL-CLOSED** -- no single-ticker ETF with a defensible, disclosable
tracking relationship to Treasury-market implied volatility (the MOVE
index has no directly investable ETF) was identified through this
phase's shipped provider; the concept is documented so it exists for a
future phase, with `enabled=False`/`provider_mapping_ids=()`, and no
mapping is fabricated to fill the gap.

**Provider symbol mapping** (`symbol_mapping.py`). `ProviderSymbolMapping`
binds provider + provider_symbol + canonical_driver_id + instrument_form
+ exchange_or_venue + currency + adjustment_policy_kind +
continuation_policy_id + mapping_version + proxy_policy into one
content-addressed identity. Structural guards: a futures-form mapping
REQUIRES `continuation_policy_id` (and a non-futures form forbids it);
**an ETF-form mapping structurally REQUIRES `proxy_policy.is_proxy=True`**
-- an ETF can never be labeled the literal underlying it tracks (spec
Section 5's "no code may label a proxy instrument as the underlying it
approximates", also exercised as a dedicated safety-scan behavioral
check, see below). `SymbolMappingSet` validates the cross-mapping
invariant no single mapping's own `__post_init__` can check alone: one
`(provider, provider_symbol, mapping_version)` cannot resolve to two
DIFFERENT `canonical_driver_id`s -- an alias change is expressed as a
NEW mapping version, never an in-place edit.

**Adjustment policy** (`adjustment.py`). Five kinds:
`RAW_UNADJUSTED` (this phase's shipped adapter's only produced kind --
Alpha Vantage's `TIME_SERIES_DAILY` is documented as raw/as-traded),
`SPLIT_ADJUSTED`, `TOTAL_RETURN_ADJUSTED`,
`PROVIDER_ADJUSTED_UNVERIFIED`, `NOT_APPLICABLE` (spot/index/futures --
no corporate action to adjust for). No corporate-action arithmetic is
ever performed; a policy CHANGE changes dataset identity.

**Timezone and session policy** (`sessions.py`). `TimezoneSessionPolicy`
carries `timezone_key` (validated against a small allowlist), session
open/close times or `is_24_hour_session`, `CandleTimestampConvention`
(`OPEN_LABELED`/`CLOSE_LABELED` -- whether a provider's own daily-bar
date labels the session OPEN or CLOSE), a trading-week note, an optional
holiday-calendar reference, and a free-text `provider_session_note`
disclosing the PROVIDER's own documented session semantics -- never an
invented centralized-exchange truth. `market_normalization.
resolve_bar_open_time` is the pure function that interprets a provider's
raw date text into a genuine UTC open timestamp, honoring both
`is_24_hour_session` and the timestamp convention; two curated concepts
under different `TimezoneSessionPolicy`s (e.g. an NYSE ETF vs. an
Asia/Tokyo-session fixture) resolve materially different UTC open times
for the identical calendar date -- confirmed directly in tests and
exercised end-to-end in the mandatory fixture-acceptance universe.
**Known constraint**: `TimezoneSessionPolicy` is a per-DRIVER-SPEC
field, not per-mapping -- a driver with multiple mappings from different
providers currently shares one session policy across all of them; a
genuinely different per-mapping session would need a registry_version
split or a future schema extension.

**Futures contract and continuation policy** (`futures.py`).
`FuturesContractMetadata` models ONE specific, individually identified
contract (root/full symbol, exchange, expiry, optional first-notice/
last-trade dates, contract month/year, multiplier, quote unit, currency,
tick size, session timezone) -- rejected outright if result-critical
fields are missing; a provider that cannot supply this must be
classified `PROVIDER_CONTINUOUS_FUTURES` instead of a policy asserting
individual-contract knowledge it does not have. `ContinuationPolicyKind`
has 6 values (`PROVIDER_NATIVE_CONTINUOUS`,
`FRONT_MONTH_NO_BACK_ADJUSTMENT`, `ROLL_ON_FIXED_DAYS_BEFORE_EXPIRY`,
`ROLL_ON_VOLUME_CROSSOVER`, `BACK_ADJUSTED_DIFFERENCE`,
`RATIO_ADJUSTED`); `require_adjustment_evidence` structurally guards
that a `BACK_ADJUSTED_DIFFERENCE`/`RATIO_ADJUSTED` continuation never
produces a bar without its own `RollProvenance.adjustment_amount`/
`adjustment_ratio` evidence. `RollProvenance` (active/prior/next
contract symbols, roll timestamp text, adjustment amount/ratio,
continuation_policy_id) is attached to every bar of a continuous series
-- never optional for a `PROVIDER_CONTINUOUS_FUTURES`-form bar (a
structural `MarketRecordError` guard on `MarketDriverBar` itself, also
exercised as a dedicated safety-scan behavioral check). This phase's
shipped provider maps no futures instrument -- the futures/continuation
code paths are exercised end-to-end through the mandatory fixture
universe's synthetic `FakeMarketCollector` (below), never against a real
provider this phase.

**Point-in-time availability** (`availability.py`) -- mirrors Phase 4B's
`curated.availability`'s own discipline exactly for market bars.
`resolve_bar_availability_time` is STRUCTURALLY fail-closed: every
branch resolves a concrete, timezone-aware datetime `>= bar_close_time`
or raises `MarketAvailabilityUnresolvedError`; no branch can silently
default to "available at candle open." Three kinds:
`CLOSE_PLUS_CONSERVATIVE_DELAY` (this phase's shipped adapter's own
policy -- a delay of `0` still requires close to have actually passed,
never open), `NEXT_SESSION_OPEN_CONSERVATIVE`, `EXPLICIT_PUBLICATION_
DELAY_MINUTES`. The caller must derive `bar_close_time` itself (via
`core.time_utils.compute_close_time`) before calling -- the function
never derives a close time on its own, only the availability delay atop
one.

**Raw and canonical market-bar records** (`market_record.py`).
`RawMarketRecord` holds provider text UNPARSED (every financial field
stays TEXT until normalization parses it directly to `Decimal`, never
through `float`). `MarketDriverBar` is a deliberately NEW, materially
richer record type -- NOT Phase 1's sequence-based `candles.Candle` --
carrying canonical-driver/instrument-form/proxy/adjustment/futures-
contract/continuation/availability semantics `Candle`'s envelope was
never shaped to hold (the same "build a new record type rather than
force-fit" precedent Phase 4B already established for
`CuratedMacroObservation` vs. `macro.MacroEvent`). Validates: `high >=
low`, `open`/`close` within `[low, high]`, `low > 0`, `volume >= 0` or
`None` (never coerced to zero), `availability_time >= close_time`,
futures-form requires `contract_metadata_id`, provider-continuous
requires `roll_provenance`. `bar_id` identity includes `request_
manifest_id`/`response_manifest_id`/`source_manifest_id`/`source_row_
index` -- the SAME provenance-in-identity precedent Phase 4B's
`CuratedMacroObservation.observation_id` already established (a bar "as
observed in this specific response" is the identity unit, not merely
its OHLCV values; two independent fetches of the economically-identical
bar via different responses legitimately mint different `bar_id`s --
see the Gap policy subsection below for why this matters and how it is
handled). `MarketDriverBarStore` is purely content-addressed
append-only, mirroring `curated.macro_observation.CuratedObservationStore`
exactly, including its `append_many_and_read_all` atomic-lock-held
append-then-read hardening (applied proactively here from the start,
not discovered via a bug this time).

**Provider-neutral collector contract** (`protocols.py`).
`HistoricalMarketCollector` is a structural `Protocol` (mirrors
`collectors.protocols.HistoricalHttpTransport`'s own convention):
`provider_metadata()`, `supported_capabilities()`, `build_metadata_
request(...)`, `build_history_request(...)`, `parse_metadata_
response(...)`, `parse_history_response(...)`. `MarketCollectorCapabilities`
declares candles/quotes/trades/adjusted/unadjusted/corporate-actions/
futures-contracts/continuous-futures/pagination/anonymous-access support,
whether a runtime credential is required, max interval/rows-per-page,
supported granularities, and supported instrument forms.
`require_within_capabilities` is the orchestrator's own fail-closed gate,
called BEFORE any request is built -- REJECTS a request exceeding
declared capabilities, never silently downgrading interval, adjustment
mode, or instrument semantics.

**Alpha Vantage adapter** (`providers/alpha_vantage.py`). The ONE
concrete provider adapter this phase ships. `build_metadata_request`
and `build_history_request` return the IDENTICAL request manifest --
Alpha Vantage has no separate metadata endpoint for `TIME_SERIES_DAILY`
(metadata and data arrive in ONE response), documented as an honest
provider limitation; orchestration detects this (`request_manifest_id`
equality) and fetches once, reusing the same response for both.
`_parse_daily_envelope` fails closed on the provider's own `Information`/
`Error Message`/`Note` top-level error/rate-limit keys rather than
treating them as an empty-but-valid response.
`parse_alpha_vantage_daily_records` sorts dates ASCENDING for
deterministic `source_sequence` assignment (the provider's own response
order is descending). Capabilities: `supported_instrument_forms=(ETF,
EQUITY)`, `unadjusted_data_supported=True`/`adjusted_data_supported=
False`, `futures_contracts_supported=False`/`continuous_futures_
supported=False`, `runtime_credential_required=True`,
`max_interval_days_per_request=None`/`max_rows_per_page=None` (one call
returns the provider's ENTIRE available history when `outputsize=full`
is requested; there is no server-side date-range filter to request a
narrower window -- a caller-side `max_records_per_mapping` bound in
`MarketBackfillSpec` is what actually caps ingestion size).

**Raw-to-canonical normalization** (`market_normalization.py`). Pure,
mirrors `curated.orchestration._normalize_curated_row`'s own contract
exactly: `normalize_raw_market_record` never raises for an ordinary
malformed row (quarantines instead via `INVALID_MARKET_RECORD`/
`MISSING_MARKET_VOLUME` issue codes), reused UNCHANGED by both
orchestration and independent verification.

**Curated market backfill spec** (`market_backfill.py`).
`MarketBackfillSpec` binds `selected_driver_ids` (the ECONOMIC CONCEPTS
being backfilled) SEPARATELY from `selected_mapping_ids` (the EXACT
provider surfaces supplying them) -- deliberate: more than one mapping
per driver is permitted (the cross-provider conflict model depends on
this; fetching the same concept from two providers simultaneously is a
caller's explicit choice, never an automatic behind-the-scenes
arbitration). `create_market_backfill_spec` validates every selected
mapping against a `CuratedMarketDriverRegistry` AND a `SymbolMappingSet`
(unknown driver, disabled driver, unknown mapping, disabled mapping, or
a mapping belonging to a driver outside `selected_driver_ids` are all
rejected) before a spec can even exist. `MarketCachePolicy` has two
values, `PREFER_CACHE`/`FORCE_FRESH`, mirroring Phase 4B's own
`CachePolicy`.

**Multi-mapping orchestration** (`market_orchestration.py`, ~600
lines). The 12-stage state machine spec Section 18 requires, exactly:
`REGISTRY_VERIFIED -> PLAN_CREATED -> PROVIDER_METADATA_VERIFIED ->
REQUESTS_COMMITTED -> RESPONSES_COMMITTED -> RAW_RECORDS_PARSED ->
RECORDS_NORMALIZED -> COMPONENT_DATASETS_COMMITTED -> COMBINED_
MANIFEST_COMMITTED -> RECONCILED -> VERIFIED -> COMPLETED`.
`CrossAssetOperationStage`/`CrossAssetOperationStore` are a
self-contained THIRD stage-machine implementation (after Phase 3's
`OperationStore` and Phase 4A's `CollectorOperationStore`, following
Phase 4B's own `CuratedOperationStore` precedent), duplicating the SAME
proven idempotent/conflict/monotonic-progression algorithm, scoped to
`target_dataset_namespace`. PROVIDER-NEUTRAL BY CONSTRUCTION: depends
only on `HistoricalMarketCollector`'s structural shape --
`collectors_by_provider`/`allowed_hosts_by_provider` let one operation
span mappings served by DIFFERENT providers simultaneously (exercised
directly in tests: a real `AlphaVantageCollector` and a synthetic
fixture provider committing to the SAME operation). Per mapping: capability
check -> provider metadata fetch+verify (symbol/driver/instrument-form/
currency/exchange/granularity fail-closed comparisons against the
mapping's own declared fields; a field the provider leaves undisclosed,
e.g. Alpha Vantage's `currency=None`, is skipped, never treated as an
automatic pass) -> history fetch (reusing the metadata response when the
request manifests are identical) -> raw parse -> per-row normalize/
quarantine -> a candidate-batch conflicting-duplicate-coordinate
pre-commit check (fail-closed via `_MappingFailureError`, respects
`fail_fast`) -> commit to `MarketDriverBarStore` -> a FULL, post-commit
conflicting-duplicate-coordinate re-check across the ENTIRE now-durable
bar set (raises `MarketProviderResponseError` directly, UNCONDITIONALLY,
regardless of `fail_fast` -- once a bar is durably appended, the
append-only store cannot roll it back, so a corruption signal here must
escalate loudly no matter what) -> component manifest -> provenance.
`backfill_spec.fail_fast=True` raises immediately on the first mapping
failure, committing nothing for ANY mapping; `fail_fast=False` records
each mapping's own outcome independently, and the combined manifest's
`completeness_status` reflects the registry's OWN required-driver
tracking (never merely "every selected mapping succeeded" -- a universe
that only ever selects a SUBSET of the registry's required drivers
correctly stays `PARTIAL`, confirmed directly in tests). Known,
disclosed simplification: `contract_metadata_id_by_mapping`/`roll_
provenance_by_mapping` (when supplied) apply the SAME futures identity
to every bar produced for that mapping in ONE call -- adequate for this
phase's fixture coverage, not a general per-row roll resolver.

**Gap and conflict analysis** (`gap_policy.py`). `analyze_bar_gaps` is
PURE, scoped to one `(canonical_driver_id, provider, provider_symbol)`
coordinate space. Missing-bar detection uses a Mon-Fri BUSINESS-DAY
heuristic (`calendar_assurance="limited"`, always, honestly disclosed --
no holiday calendar is loaded this phase; a legitimate weekday holiday
will be flagged as a candidate missing bar, a KNOWN limitation, never
silently presented as verified). A genuine weekend closure is never
reported missing. 24-hour sessions report zero missing-bar candidates
(no weekday-closure assumption applies) rather than inventing an
unverified continuous-session expectation. **Conflicting-duplicate-
coordinate detection compares ECONOMIC CONTENT** (open/high/low/close/
volume/volume_unit/adjustment_policy_id/availability_time/availability_
policy_id/session_policy_id/contract_metadata_id/roll_provenance), NOT
`bar_id` -- this was a genuine defect found and fixed during this
phase's own test authoring: since `bar_id` embeds provenance
(response_manifest_id etc.), the SAME economically-identical bar
re-fetched under a legitimately different response (e.g. a `FORCE_
FRESH` refetch whose envelope happens to include one more trailing row)
mints a different `bar_id`, and a naive `bar_id`-based conflict check
would have misclassified that harmless re-fetch as a data-integrity
violation. `GapPolicy` (4 values: `ALLOW_AND_REPORT`,
`QUARANTINE_COMPONENT`, `FAIL_REQUIRED_DRIVER`, `FAIL_UNIVERSE`) governs
only MISSING bars -- conflicting duplicates are never a policy choice,
always a hard integrity failure (see Orchestration above).

**Dataset layout** (`datasets.py`). Mirrors `curated.datasets` exactly,
one level more granular: one immutable `ComponentMarketDatasetManifest`
per `ProviderSymbolMapping` (a mapping already binds provider +
provider_symbol + canonical_driver_id + instrument_form + currency +
adjustment + continuation + version -- exactly the granularity the spec
requires components to never be merged across), `conflicting_coordinate_
count` structurally REQUIRED to be `0` at construction (a component with
unresolved conflicts can never be committed). `CombinedCrossAssetManifest`
binds `component_manifest_ids` keyed by `mapping_id`, a
`driver_id_by_mapping` back-reference, `required_driver_ids`/`missing_
required_driver_ids` (recomputed as `required - satisfied`, where
`satisfied` is the set of driver ids with at least one successful
component THIS combined manifest binds), and enforces at construction
that `missing_required_driver_ids` non-empty implies
`completeness_status=PARTIAL` -- a universe missing a required driver
can structurally never claim `COMPLETE`. Both stores are versioned
purely by append-position (`len(history)`), with idempotent no-op
appends on an identical id, matching `curated.datasets`'s own precedent.

**Cross-provider conflict model** (folded into `market_reconciliation.
py`, spec Section 20). When more than one mapping in a combined manifest
serves the SAME `canonical_driver_id`, overlapping bars (by `open_time`)
are compared deterministically: exact equality (identical `close`),
tolerance-level difference (`|close_a - close_b| / max(|close_a|,
|close_b|, 1) <= PRICE_TOLERANCE_RATIO`, a deliberately conservative
0.5% constant), or material conflict (anything else) -- reported as
`WARNING`-severity issues, never silently averaged or auto-resolved;
every component dataset stays independently readable.

**Incremental update planning** (`update_plan.py`). PURE and
deterministic, mirrors `curated.update_plan` exactly: NEVER reads the
wall clock, NEVER touches the network. `MappingUpdateAction` has three
values: `NO_UPDATE_NEEDED`, `APPEND_BARS`, `POLICY_REFRESH` (triggered
when the mapping's OWN `adjustment_policy_kind`/`continuation_policy_id`
no longer matches what the CURRENT component manifest was built under --
Phase 4B's `REVISION_REFRESH` analog, since cross-asset has no single
global "revision policy" the way FRED does). Each entry additionally
carries `needs_futures_roll_refresh: bool` (spec Section 22's own
"futures-roll refresh needs") -- an honest flag for any futures-form
mapping, never a computed roll decision this module does not have
evidence for. The exact-no-op guarantee is the SAME two-layer proof
Phase 4B established: this module's own job is only to REPORT
`NO_UPDATE_NEEDED` correctly; `ComponentMarketDatasetManifestStore.
append`'s own idempotent no-op-on-identical-id behavior independently
guarantees no new version is minted even if a caller runs the backfill
anyway.

**Reconciliation** (`market_reconciliation.py`) and **verification**
(`market_verification.py`). Reconciliation re-derives coverage/bar
counts from the actual `MarketDriverBarStore`, cross-checks provenance
completeness (bar without provenance / provenance without bar / duplicate
coordinate), and runs the cross-provider conflict model above.
Verification mirrors `curated.verification`'s own discipline: EVERY
check REDERIVES an artifact fresh from durable state (re-read raw
bytes, re-hash, re-parse via the SAME collector, re-normalize, re-derive
component/combined manifest ids) using the exact same PURE functions
orchestration used, never trusting a cached parse or a stored count.
INDEPENDENCE CLASSIFICATION (documented in the module's own docstring,
per spec Section 27): the reparse/renormalize checks are structurally
independent of orchestration's OWN RUN but NOT independent of the
provider adapter's OWN parsing implementation -- a bug inside `providers/
alpha_vantage.py` itself would not be caught here (that risk is instead
covered by the dedicated adversarial tamper tests); the remaining
registry/mapping/manifest self-identity and completeness checks are
fully structurally independent, requiring no provider-specific code at
all.

**Reports** (`market_reports.py`). Mirrors `collectors.reports`'s own
convention exactly: every function wraps an already-produced object
into a stable, deterministic dict, never re-deriving new facts.
`generate_quarantine_summary_report`/`generate_provenance_summary_report`
are re-exported UNCHANGED from `collectors.reports`. No report function
anywhere in this module accepts a raw credential as an argument, and
every dict is assembled purely from objects that are themselves
structurally secret-free.

**Offline replay.** No separate `replay.py` module exists -- exactly
matching Phase 4B's own precedent: replay is simply re-running
`run_cross_asset_backfill_operation` with `transport=None`, which
STRUCTURALLY guarantees zero network calls (the fetch path only reaches
the transport branch on a cache miss, and `transport=None` there raises
before any call is attempted) -- a STRONGER guarantee than "transport
fails when invoked," since it can never be invoked at all. Proven
directly in tests via both the `transport=None` path and an explicit
`_ForbiddenTransport` double that raises `AssertionError` on any
`.get()` call.

**Opt-in real-Alpha-Vantage acceptance workflow** (`acceptance.py`).
The ONE sanctioned place in all of `collectors/cross_asset/` that reads
an environment variable (`ALPHA_VANTAGE_API_KEY`) -- every other module
requires a credential to be passed explicitly by the caller.
`resolve_alpha_vantage_api_key_from_environment` returns `None` (never
raises) when absent or blank; `run_real_alpha_vantage_acceptance_workflow`
requires a non-empty `api_key` argument, with no implicit fallback.
BOUNDED SUBSET (spec Section 24): exercises the ONE highest-value core
driver this platform can genuinely verify against a live provider,
`gold_reference` via `GLD` -- never claiming to validate the full
10-concept universe against a real provider (fixture-based acceptance,
below, is what covers the full conceptual universe). Runs the full
pipeline, then a SECOND cached-replay pass with a `_ForbiddenTransport`
to prove offline replay reproduces an identical semantic result with
zero network calls. `RedactedCrossAssetAcceptanceReport`'s own field set
structurally cannot carry a secret (no field name resembles a
credential); the corresponding test resolves the key via the same
function and calls `pytest.skip(...)` with a precise reason when absent
-- the expected state for ordinary CI and this offline development
environment.

**Mandatory fixture-based acceptance**
(`test_collectors_cross_asset_fixture_acceptance.py`). The ALWAYS-RUN,
no-network counterpart, assembling ONE curated universe covering: all 5
core concepts (dollar/WTI/Brent/silver/gold, via Alpha-Vantage-shaped
ETF fixtures), a genuinely separate `copper_industrial_growth` driver on
an Asia/Tokyo session (multiple timezones/session cutoffs, WITHIN one
universe -- `TimezoneSessionPolicy` being per-driver-spec, not
per-mapping, means this required a second driver rather than a second
mapping on `wti_crude`, see the Sessions subsection's "known
constraint"), a `wti_crude` `PROVIDER_CONTINUOUS_FUTURES` mapping with
full roll provenance via a synthetic `FakeMarketCollector` (mixed
instrument forms, cross-provider: `wti_crude` has BOTH an Alpha Vantage
ETF mapping and this fixture futures mapping), one deliberately missing
business day, an exact-duplicate-bar idempotency proof, a conflicting-
duplicate-bar hard-fail proof (raises `MarketProviderResponseError`
unconditionally, per the Orchestration subsection above), raw-response
caching, provider metadata verification, normalization, component/
combined datasets, gap analysis, reconciliation, verification, offline
replay (byte-identical `combined_manifest_id` on a second, `transport=
None` pass), a deterministic credential-free report export, and
point-in-time visibility after close only. `FakeMarketCollector`
(`_cross_asset_test_helpers.py`) is a fully synthetic
`HistoricalMarketCollector` double -- NOT shaped like Alpha Vantage's
real schema -- that builds GENUINELY SEPARATE metadata/history request
manifests (unlike Alpha Vantage's single-endpoint reuse), exercising
orchestration's two-fetch code path the real adapter never triggers.

**Safety scan extension**
(`test_market_data_safety_scan.py::TestCrossAssetSpecificSafety`).
`_collector_source_files()`'s existing recursive `rglob` already reaches
`collectors/cross_asset/`, so every general check (no network/broker
imports, no credential-shaped dataclass fields, no float-typed financial
fields, no wall-clock reads, no bare/swallowed exceptions, no pickle/
cloud-SDK/eval-exec/shell-execution, no committed long-literal API key)
already applies with zero changes needed. This phase adds Phase-
4C-specific structural checks mirroring Phase 4B's own `curated/`-
specific section exactly: `open_time` (a bar's raw OPEN) must never be
compared directly against a wall-clock-shaped value as if it proved
availability (the `curated/` `observation_date`-as-availability-proof
check, replayed for market bars -- `resolve_bar_availability_time`
exists precisely to prevent this bypass), `ALPHA_VANTAGE_API_KEY` must
appear nowhere outside `acceptance.py`/`alpha_vantage.py`, and two
BEHAVIORAL (not merely static) confirmations: an ETF-form mapping
cannot be constructed with `is_proxy=False`, and a `PROVIDER_
CONTINUOUS_FUTURES` bar cannot be constructed without `roll_provenance`.

**Adversarial and concurrency coverage**
(`test_collectors_cross_asset_pit_concurrency_adversarial.py`).
Concurrent identical backfills (4 threads racing the SAME operation)
tolerate `MarketDataLockError` from the ledger's deliberately FAIL-FAST
lock (mirrors `ml.concurrency.experiment_lock`'s own documented
contract -- never block-waits) but must NEVER produce duplicate or
corrupt bars from whichever threads win; concurrent cache readers get
byte-identical bytes. Forged-component-manifest and combined-manifest-
component-swap detection: a forged manifest's SELF-CLAIMED id stays
unchanged (`dataclasses.replace` does not recompute a content-addressed
id), but independently RECOMPUTING the id from the manifest's own
recorded fields no longer matches -- exactly how verification catches a
forgery. `RetryExhaustedError` (and its full exception chain) never
contains the real `api_key`, confirmed with a realistic secret literal.
Registry identity is confirmed independent of `PYTHONHASHSEED` via two
child-process subprocess runs under different explicit seed values.

**New shared `DatasetKind`/`SourceKind`/`RecordKind` values** (narrow,
additive, mirroring Phase 4B's own `MACRO_OBSERVATIONS`/`FRED_API`
precedent exactly): `DatasetKind.CROSS_ASSET_MARKET_BARS` (same
scoping-only pattern, `instrument_id` repurposed to mean
`f"{canonical_driver_id}__{instrument_form}"` since one provider may map
a driver through more than one instrument form, each needing its own
provenance/quarantine scope), `SourceKind.MARKET_DATA_PROVIDER_API`
(deliberately provider-NEUTRAL, unlike Phase 4A's `FRED_API` which names
one specific source), `RecordKind.MARKET_DRIVER_BAR`. ~16 new exceptions
in `core/exceptions.py` (`MarketDriverRegistryError`, `ProviderCapabilityError`,
`InstrumentFormError`, `SymbolMappingError`, `AdjustmentPolicyError`,
`SessionPolicyError`, `FuturesContractError`, `ContinuationPolicyError`,
`MarketAvailabilityPolicyError`/`MarketAvailabilityUnresolvedError`,
`MarketRecordError`, `MarketProviderResponseError`, `MarketBackfillSpecError`,
`MarketCombinedManifestError`, `MarketUpdatePlanError`, `GapPolicyError`)
-- `CollectorOrchestrationStateError`/`CollectorOrchestrationConflictError`/
`CollectorReconciliationError`/`CollectorVerificationError` (Phase 4A)
are REUSED directly for this phase's own stage machine/reconciliation/
verification, matching Phase 4B's precedent.

**Example config**
(`examples/xauusd_cross_asset_config.example.json`). Mirrors Phase 4B's
own `xauusd_macro_fred_config.example.json` shape: the full 10-concept
curated registry (honestly marking `treasury_volatility` disabled), the
9 Alpha Vantage ETF provider mappings, a bounded backfill spec, session/
availability/adjustment policy blocks, and a `credentials` block that
references only the `ALPHA_VANTAGE_API_KEY` environment-variable NAME --
never a literal value. A dedicated test
(`test_cross_asset_example_config.py`) constructs REAL registry/mapping/
backfill objects directly from the file and confirms zero secret-shaped
literals anywhere in it.

## Known limitations (honestly disclosed)

Phase 1:
- No instrument registry/master-data service: `instrument_id` defaults
  to `f"{provider}__{symbol}"` (derived in `normalization.
  derive_instrument_id`) when not explicitly supplied. `__`, not `:` or
  `/`, joins the two -- `:` is a reserved drive-separator character on
  Windows and broke `MarketEventStore`/`FeatureStore`'s own path-based
  partitioning during Phase 1's own smoke testing.
- No tick-to-candle resampling/aggregation.
- `vwap` is cumulative, not session-reset; `atr`/`rsi` use simple, not
  Wilder, smoothing (deliberate, disclosed formula choices).
- Macro events (`macro.py`) are modeled and point-in-time-safe but not
  wired into feature generation -- no macro-derived feature exists yet.
- No CLI surface: library only.

Phase 2:
- Partitioning supports DAILY/MONTHLY only, not fixed-event-count.
- Compaction does not combine partitions across a granularity change
  (see "Compaction" above for the exact reason).
- Partition-rebuild and manifest-rebuild both read the FULL underlying
  event/feature history and filter/aggregate in memory -- correct and
  deterministic, but not O(partition size); a genuinely large-scale
  deployment would need windowed store reads, out of Phase 1/2's scope.
- `ema`/`vwap` incremental generation always re-reads the full raw
  history (a documented performance, not correctness, limitation -- see
  above).
- Recovery's file-rewrite (removing a discarded truncated trailing
  record) assumes no concurrent writer targets the same path during the
  recovery call -- a reasonable operational assumption for crash
  recovery (run before ordinary traffic resumes), not enforced by an
  additional lock.
- No CLI surface: library only, matching Phase 1.

Phase 3:
- CSV candle adapter has no per-row timeframe column (a CSV file always
  shares one timeframe, declared on `SourceManifest.expected_timeframe`);
  only JSONL rows carry a per-line `timeframe` field.
- Reprocessing the exact same source rows under a NEW `operation_id`
  against an already-populated dataset correctly fails closed
  (`ProvenanceError`, checked in a pre-flight pass before any repository
  write) rather than silently duplicating the economic event -- a
  genuine retry of already-committed rows MUST reuse the SAME
  `operation_id`; this is a deliberate consequence of sequence numbers
  being pinned per-operation, not a bug, but it does mean two
  independently-scheduled operations must coordinate on `existing_
  covered_partition_keys` (read fresh from the repository) rather than
  blindly resubmitting the same interval.
- Gap-policy calendar awareness (`REQUIRE_EXPECTED_MARKET_CALENDAR`)
  inherits every limitation `calendar.py`/`historical.calendar.
  TradingCalendar` already discloses for OTC/provider-specific sessions.
- `DUPLICATE_SOURCE_ROW_COORDINATE` (a repeated `row_index` from a single
  adapter) is structurally unreachable through any of the three shipped
  adapters (each assigns `row_index` via `enumerate()`); the check exists
  defensively and is only reachable via a hand-constructed, adversarial
  record list in tests.
- No CLI surface: library only, matching Phases 1 and 2.

Phase 4A:
- `StdlibHttpsTransport`'s own wire-level pinned-IP connection and HTTP
  response construction (including the `Content-Encoding` rejection
  path) is exercised indirectly, not by a dedicated offline unit-test
  harness -- that would require deep mocking of stdlib `http.client`'s
  internals. The collector layer above it is fully protocol-abstracted
  (`HistoricalHttpTransport`) and thoroughly tested via `FakeTransport`;
  `StdlibHttpsTransport`'s own SECURITY-CRITICAL logic (URL/host/IP
  validation, redirect rejection, size-bounded reading) IS directly
  unit-tested (`test_collectors_transport_security.py`) without this gap.
- CSV support in `fred_schemas.parse_fred_csv_response` targets FRED's
  documented two-column `DATE,<SERIES_ID>` downloadable-series export
  shape, an explicit, disclosed assumption -- this phase performs zero
  live FRED requests, so it is never verified against a live response.
- No separate `CollectorCheckpoint`/`CollectorCheckpointStore` (see the
  Reconciliation subsection above for the reasoning); a single
  per-`operation_id` fetch is fully resumable through
  `CollectorOperationStore` alone, which is this phase's actual scope.
- No pagination support: FRED's `/fred/series/observations` endpoint is
  called with an explicit `observation_start`/`observation_end` window
  per request; a caller needing more than one page issues more than one
  operation. The request/response manifest schemas both carry pagination
  fields (`pagination_page_index`/`pagination_next_token`) for a future
  phase to use without a breaking schema change, but nothing in this
  phase populates or consumes them.
- No CLI surface: library only, matching Phases 1-3.

Phase 4B:
- `NEXT_BUSINESS_DAY_CONSERVATIVE` uses simple Mon-Fri weekday
  arithmetic with NO public-holiday calendar awareness -- a disclosed
  limitation carried over from the same honest gap Phase 3's
  `REQUIRE_EXPECTED_MARKET_CALENDAR` gap policy already discloses for
  OTC/provider-specific sessions.
- `AvailabilityPolicyKind.RELEASE_CALENDAR_REFERENCE` is deliberately
  UNIMPLEMENTED this phase -- `resolve_availability_time` raises
  `AvailabilityPolicyError` rather than silently falling back to a
  different policy kind; a future phase wiring an actual BLS/Treasury
  release calendar can implement this branch without a schema change.
- FRED reports only DATES for `realtime_start`, never exact intraday
  publication times; every `AvailabilityPolicy`'s `availability_hour`/
  `availability_minute` is a disclosed, configurable APPROXIMATION of
  when a value becomes usable, not a claim of verified real release
  timing -- the policy itself is fully identity-relevant, so this
  approximation is auditable, never silently baked in.
- Only the 4 mandatory core series are individually fixture-verified
  this phase; the 14 extended-universe candidates are registered
  (`default_extended_series_specs`) but all constructed `enabled=False`
  -- enabling one for a live backfill requires an explicit opt-in flip
  by a future caller, not a code change.
- No curated-data-to-market-bar join exists yet (matching Phase 4A's own
  "modeled but not wired into feature generation" limitation for
  Phase 1 macro events) -- the PIT consumer contract is documented and
  enforced by a dedicated safety-scan check, not yet exercised by an
  actual feature-generation join.
- Concurrency testing under AGGRESSIVE same-process thread contention
  (multiple threads simultaneously retrying the identical operation in
  a tight loop) occasionally surfaces `experiment_lock`'s own stale-
  lock-reclaim heuristic (`ml.concurrency`, shared infrastructure
  predating this phase) needing more attempts than a single bare call
  provides before any one thread completes; every individual failure
  observed under this stress remains a clean, recognized
  `MarketDataLockError` (never corruption -- confirmed via extensive
  repeated testing, see `test_collectors_curated_pit_concurrency_
  adversarial.py::TestConcurrency`), and a bounded caller-level retry
  with a small jittered backoff (the pattern a real caller would use)
  reliably converges. A deeper fix to `experiment_lock`'s own contention
  behavior under heavy same-process thread load is out of this phase's
  scope (shared infrastructure used by every prior phase) and is noted
  here as a candidate for dedicated follow-up work, not fixed as a
  side effect of this phase's own deliverable.
- No pagination support, matching Phase 4A: `page_size` bounds a single
  request; `CuratedBackfillSpec.max_series_count`/`max_observations_
  per_series`/`max_total_raw_bytes` bound a whole operation, but a
  caller needing more than one FRED page per series issues more than
  one operation, exactly as Phase 4A's own collector does.
- No CLI surface: library only, matching Phases 1-4A.

Phase 4C:
- Only ONE provider (Alpha Vantage, `TIME_SERIES_DAILY` only) is
  genuinely wired to a live universe this phase; every mapped concept is
  an ETF-form PROXY, never a literal spot/futures instrument -- honestly
  disclosed via `is_proxy=True`/`proxy_quality` on every mapping, never
  silently upgraded.
- `treasury_volatility` ships UNSUPPORTED AND FAIL-CLOSED: no ETF with a
  defensible tracking relationship to Treasury-market implied volatility
  was identified through the shipped provider; the concept exists in the
  registry (`enabled=False`) for a future phase to fill.
- No real provider this phase maps a futures instrument form -- the
  futures-contract/continuous-series/roll-provenance code paths are
  fully implemented and exercised end-to-end, but only through the
  mandatory fixture universe's synthetic `FakeMarketCollector`, never
  against a live provider.
- Gap-policy missing-bar detection uses a Mon-Fri business-day heuristic
  with NO public-holiday calendar awareness (`calendar_assurance=
  "limited"`, always) -- the same disclosed gap Phase 3's
  `REQUIRE_EXPECTED_MARKET_CALENDAR` and Phase 4B's `NEXT_BUSINESS_DAY_
  CONSERVATIVE` already carry for OTC/provider-specific sessions.
- `TimezoneSessionPolicy`/adjustment/availability policies are per-
  DRIVER-SPEC fields, not per-mapping -- a driver with multiple mappings
  from different providers currently shares one session/adjustment/
  availability policy set across all of them.
- `contract_metadata_id_by_mapping`/`roll_provenance_by_mapping` (when
  supplied to orchestration) apply the SAME futures identity to every
  bar produced for a mapping in ONE call -- adequate for this phase's
  fixture coverage of the code paths, not a general per-row roll
  resolver a real multi-roll futures provider would eventually need.
- No cross-asset-to-XAUUSD-feature join exists yet (matching Phase 4B's
  own "modeled but not wired into feature generation" limitation) --
  the PIT consumer contract is documented and enforced by the safety
  scan, not yet exercised by an actual feature-generation join.
- `MarketBackfillSpec.start_time`/`end_time` are NOT enforced as a
  server-side date-range filter against Alpha Vantage's own response
  (the provider has none to request -- `outputsize=full` always returns
  the provider's entire available history); `max_records_per_mapping`
  is the actual, caller-controlled bound on how much of that response is
  accepted downstream.
- No pagination support, matching Phase 4A/4B: a caller needing more
  than one page from a future paginating provider issues more than one
  operation.
- No CLI surface: library only, matching Phases 1-4B.

## Future phases (not started)

Session-reset VWAP; Wilder-smoothed ATR/RSI variants; macro-derived
features with point-in-time enforcement wired into generation
(including an actual curated-macro-to-market-bar join using
`availability_time`); tick-to-candle resampling; a real instrument
registry; fixed-event-count and granularity-changing
(daily-to-monthly) compaction; windowed/bounded store reads for
large-scale partition rebuild; integration with `portfolio_risk`/
`execution_gateway` as their own authoritative market-data source (this
milestone explicitly forbade touching either package); FRED pagination
and additional collector endpoints beyond `/fred/series/observations`
and `/fred/series`; a real BLS/Treasury release-calendar implementation
for `AvailabilityPolicyKind.RELEASE_CALENDAR_REFERENCE`; public-holiday
awareness for `NEXT_BUSINESS_DAY_CONSERVATIVE`; enabling and
fixture-verifying the extended macro-series universe beyond the 4 core
drivers; a real, live-verified futures provider (individual-contract or
provider-continuous) to exercise `futures.py`/`ContinuationPolicy`
outside the fixture universe; a viable `treasury_volatility` provider
mapping; per-mapping (rather than per-driver-spec) session/adjustment/
availability policies; a cross-asset-to-XAUUSD-feature join with
point-in-time enforcement wired into generation; a real EIA integration
(route structure was live-verified, but no data was fetched without
registering a real account); further external collectors (Yahoo
Finance, MT5, broker/news APIs -- Phase 4A/4B/4C delivered FRED and
Alpha Vantage only), live streaming, and CLI expansion (explicitly out
of scope through Phase 4C, reserved for a future milestone).
