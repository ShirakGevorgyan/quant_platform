# Deterministic Market Data Platform and Feature Store (Milestone 10) -- Architecture

## Status: Phase 1 (immutable market events, calendar, macro, quality, normalization, deterministic feature generation and storage, replay, verification, reports) + Phase 2 (durable repository, dataset versioning, incremental ingestion, partitioning, checkpoints, recovery, reconciliation, compaction, export) delivered

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

## Future phases (not started)

Session-reset VWAP; Wilder-smoothed ATR/RSI variants; macro-derived
features with point-in-time enforcement wired into generation; tick-to-
candle resampling; a real instrument registry; fixed-event-count and
granularity-changing (daily-to-monthly) compaction; windowed/bounded
store reads for large-scale partition rebuild; integration with
`portfolio_risk`/`execution_gateway` as their own authoritative
market-data source (this milestone explicitly forbade touching either
package).
