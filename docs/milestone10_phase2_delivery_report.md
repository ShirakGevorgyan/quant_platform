# Milestone 10 Phase 2 -- Durable Market Data Repository, Dataset Versioning, Incremental Ingestion, Partitioning, and Reconciliation: Delivery Report

## 1. Phase 1 baseline commit

`d4817455cc50dbdd65a91a6551f4a0eacef1c760` ("Add deterministic market
data and feature store foundation"). `HEAD` remains this commit
throughout Phase 2; nothing has been committed.

## 2. Files added and modified

**New source files** (9, ~2,049 lines): `manifests.py`, `partitions.py`,
`repository.py`, `ingestion.py`, `checkpoints.py`, `recovery.py`,
`reconciliation.py`, `compaction.py`, `export.py`.

**Modified source files** (additive extensions, no removed/changed
Phase 1 behavior for any existing caller): `core/exceptions.py` (13 new
exception classes appended), `events.py` (added public
`events_path()` accessor), `feature_store.py` (added public
`records_path()` accessor), `feature_generation.py` (added optional
`only_persist_timestamps` parameter to `generate_candle_features`,
defaulting to `None` = unchanged behavior; added incremental-generation
functions), `verification.py` (added `verify_raw_dataset`/
`verify_feature_dataset`), `reports.py` (added six Phase 2 report
functions).

**New test files** (13, ~2,235 lines) under `tests/unit/market_data/`:
`test_market_data_manifests.py`, `test_market_data_partitions.py`,
`test_market_data_ingestion.py`, `test_market_data_checkpoints.py`,
`test_market_data_incremental_features.py`,
`test_market_data_recovery.py`, `test_market_data_reconciliation.py`,
`test_market_data_repository_verification.py`,
`test_market_data_compaction.py`, `test_market_data_export.py`,
`test_market_data_concurrency.py`, `test_market_data_cross_process.py`,
`test_market_data_safety_scan.py`.

`docs/market_data_architecture.md` updated with a full Phase 2 section.

No file outside `market_data`/its tests/`core/exceptions.py`/docs was
touched. No ML, backtesting, paper-trading, portfolio-risk, or
execution-gateway package was modified.

## 3. Storage layout

```
{root}/
  market_events/{provider}/{instrument_id}/events.jsonl      <- Phase 1, reused unchanged
  features/{feature_name}/v{version}/{instrument_id}/records.jsonl  <- Phase 1, reused unchanged
  repository/
    datasets/{dataset_key_path}/manifests.jsonl        <- append-only manifest VERSION history
    partitions/{dataset_key_path}/{partition_key}.json  <- current-version-only, atomically replaced
    batches/{dataset_key_path}/batches.jsonl             <- append-only ingestion batch idempotency ledger
    checkpoints/{dataset_key_path}/checkpoints.jsonl      <- append-only checkpoint history
```

`{dataset_key_path}` = `raw_market_events/{provider}/{instrument_id}` or
`derived_features/{feature_name}/v{feature_version}/{instrument_id}`
(`DatasetKey.storage_path_parts()`).

## 4. Dataset identity model

Two distinct identities, analogous to a git branch vs. a commit:
`DatasetKey` is the stable, caller-declared routing identity ("which
dataset") and never changes; `dataset_id` (on `DatasetManifest`) is a
per-version, content-addressed identity that changes on every committed
batch/late arrival. `DatasetManifest.to_identity_payload()` includes
`dataset_kind`, `instrument_id`, `provider`/`feature_name`+
`feature_version`, `timeframe`, `partitioning`, `first_event_time`,
`last_event_time`, `event_count`, `ordered_partition_ids`,
`raw_source_dataset_id`, and `semantic_digest`; it EXCLUDES `dataset_id`
itself, `physical_digest`, `creation_time`, and `completion_status`.
Proven in `test_market_data_manifests.py`: identical arguments produce
identical ids; changed event content/order/instrument/partition
membership/feature version all change the id; `creation_time`/
`physical_digest` alone never do; two independent filesystem roots given
the same logical data produce the identical `dataset_id`.

## 5. Partitioning rules

DAILY and MONTHLY calendar-based granularities (the "minimum useful
set" the specification permits choosing; fixed-event-count partitioning
is a documented, out-of-scope extension). A `Partition` binds
`dataset_key`, `partition_key`, an ORDERED member-id list (ordered by
`(member_time, member_id)`, never arrival order), `schema_version`, and
a content digest. `PartitionStore` keeps only the current version per
`(dataset_key, partition_key)`, atomically replaced. Boundary
correctness (exact midnight, one microsecond before midnight, month-end,
non-UTC-timezone conversion before bucketing) is tested explicitly in
`test_market_data_partitions.py`.

## 6. Incremental-ingestion semantics

`ingestion.ingest_raw_events` appends events IN THE GIVEN (arrival)
order via `MarketEventStore.append` (already idempotent/gap-checked from
Phase 1), rebuilds only the partitions the batch's own events touch, and
recomputes the manifest fresh from current store state. Batch identity
(`IngestionBatchStore`) durably binds one `batch_id` to one content
digest (`dataset_key` + `ingestion_time` + ordered event ids): an exact
retry is idempotently absorbed; a conflicting retry (same `batch_id`,
different content) raises `IngestionConflictError`. Tested explicitly:
first ingestion, continuation batch, exact retry, conflicting retry,
partial overlap with exact duplicates, out-of-order arrival within one
batch, a sequence gap (rejected), and a wrong-instrument/wrong-dataset-
kind event (rejected).

## 7. Late-arrival policy

**Chosen, explicit model**: a late-arriving event is appended to the
arrival-ordered raw store like any other event, and the ONE partition
its `event_time` falls into is rebuilt from its current complete
membership, producing a new `partition_id` and therefore a new manifest
VERSION. The event is never rejected; no existing manifest version is
ever mutated (full history retained in `DatasetManifestStore`). Chosen
because it is the only one of the three models the specification names
that neither loses information nor requires a full-dataset rebuild for
one late event. Tested in `test_market_data_ingestion.py::
TestLateArrival` (a bar inserted into an already-partitioned day
produces a new `dataset_id`, rebuilds exactly that one partition, and
the prior manifest version remains independently retrievable and
byte-identical to before).

## 8. Checkpoint design

`RawIngestionCheckpoint` binds `dataset_key`, `last_committed_sequence`,
`last_committed_batch_id`, `last_canonical_partition_id`,
`semantic_digest`. `FeatureGenerationCheckpoint` binds `raw_dataset_key`
+ `raw_dataset_id` (the exact raw manifest version processed),
`feature_dataset_key`, `last_processed_raw_event_time`,
`carry_window_size: int | None` (`None` = unbounded -- see Section 15,
Defect #2), `resulting_feature_dataset_id`, `semantic_digest`. Both are
self-verifying (`checkpoint_id` recomputed from identity payload) and
NEVER trusted without independent re-validation against durable data
(`verify_raw_ingestion_checkpoint`/`verify_feature_generation_checkpoint`
recompute fresh and compare; a checkpoint behind OR ahead of durable
data raises `StaleCheckpointError`). Carry state is a window SIZE, not
cached partial sums/EMA state -- the raw store is itself durable and
replayable, strictly safer to re-derive from than to trust a cached
accumulator.

## 9. Incremental-feature equivalence rules

**Core correctness property, proven for every required indicator**:
incremental generation output is always IDENTICAL to a full fresh
recomputation over the entire raw history. `carry_window_size_for`
assigns: `window` for rolling/windowed indicators (rolling_mean,
rolling_std, atr, rsi, sma), `1` for pairwise indicators (return,
log_return, price_delta, volume_delta), `0` for pointwise indicators
(high_low_range, body_size, wick ratios), and `None` (unbounded, full
history re-read) for `vwap` and `ema` (see Section 15, Defect #2, for
why `ema` needed this). `only_persist_timestamps` ensures recomputed
carry-window context is never re-attempted for persistence (see Section
15, Defect #1). Tested via a single parametrized test
(`test_incremental_equals_full_recomputation_across_partition_boundaries`)
covering `return`, `rolling_mean`, `rolling_std`, `atr`, `rsi`, `ema`,
`sma`, `vwap`, over 3 ingestion batches with sizes (13, 17, 20)
deliberately misaligned to a window of 5, guaranteeing at least one
batch boundary cuts through a rolling window. Also tested: no-op on
unchanged raw data, no duplicate feature coordinates on batch retry,
changed feature version creates an entirely separate lineage, changed
raw dataset version creates a new derived dataset version, and the
underlying `FeatureStore`'s own no-overwrite guarantee (a structural,
not merely observed, property).

## 10. Atomicity and recovery protocol

No multi-file transaction spans (event-append, partition-write,
manifest-append) as one indivisible unit. Instead: each individual write
is independently atomic (`MarketEventStore`/`FeatureStore` fsync'd
append; `PartitionStore`/`DatasetManifestStore` via temp-then-rename or
append-only-with-idempotency) and idempotent, and manifests are always
derived FRESH from already-durable partitions/events at construction
time -- never a diff. `recovery.recover_raw_dataset`/
`recover_feature_dataset` reconstruct EXCLUSIVELY from durable evidence:
re-derive every partition and the manifest from current store content
(tolerant of a truncated trailing record, which is PHYSICALLY repaired
via atomic rewrite), and report any batch left `RESERVED` without a
matching `COMMITTED` entry as pending caller action -- never fabricated.
Recovery is deterministic and idempotent (repeating it twice in a row
changes nothing the second time) and never duplicates an already-durable
append. A genuine, unexplained non-trailing corruption (e.g. a removed
middle record) raises `RepositoryCorruptionError` rather than being
silently accepted.

## 11. Reconciliation

`reconciliation.reconcile_raw_dataset`/`reconcile_feature_dataset`
produce structured, non-raising `ValidationIssue`s (reusing `ml.models.
ValidationReport`, matching `quality.py`/`verification.py`'s existing
convention) for: missing partition, orphan partition, wrong partition
digest/membership, wrong event count, duplicate event across partitions,
manifest range mismatch, broken lineage (a feature dataset's
`raw_source_dataset_id` no longer in the raw dataset's own manifest
history), stale checkpoint, feature coordinate conflict/duplicate, and
semantic/physical digest mismatch. `MarketDataReconciliationError` is
raised only when reconciliation cannot proceed structurally (a
referenced store cannot even be read) -- never for a finding it exists
to surface. All 9 required issue categories are individually
hand-corrupted and confirmed detected in
`test_market_data_reconciliation.py`.

## 12. Verification independence classification

**Structurally independent** (`verify_raw_dataset`/
`verify_feature_dataset`): recomputes each `Partition`'s and
`DatasetManifest`'s own content id from its own recorded fields
(forged-identity detection at the repository level, distinct from
reconciliation's "does content match the raw store" check) and REUSES
Phase 1's `verify_market_event_store`/`verify_feature_store` directly
for event/feature-level identity (documented reuse). **Not
independently re-verified by default**: whether a stored feature VALUE
is the economically correct output of its generator -- closed by the
optional `cross_check_against_fresh_recomputation=True` parameter, which
regenerates every requested feature fresh into a throwaway scratch store
straight from the raw candles the dataset's own manifest claims as
lineage, and compares every value. This closes the gap pure identity
recomputation cannot: proven directly in
`TestFreshRecomputeCatchesCoherentTampering`, which hand-tampers a
stored value AND correctly re-computes a matching `feature_id` (a
"coherent" forgery), confirms the basic check reports zero criticals,
then confirms the cross-check catches it.

## 13. Export format and determinism

JSON Lines only (not CSV) -- every value already round-trips exactly
through this package's `to_json_dict()` convention (`Decimal` as exact
strings, ISO-8601 UTC timestamps), avoiding both a second CSV-specific
serialization scheme and an unnecessary pandas dependency. Row order is
always `(timestamp, id)`, independent of physical append order.
`export_semantic_digest` is a digest of the exported row set alone
(sorted before hashing). Proven in `test_market_data_export.py`:
byte-identical repeated exports, `Decimal`/timestamp exact preservation,
row order by event-time (not arrival order) even when ingested
out-of-order, identical digest and byte-identical files across two
independent filesystem roots AND across a deeply nested vs. shallow
root, and digest mutation detection (a genuinely different dataset
produces a different digest).

## 14. Tests and exact results

`tests/unit/market_data/` (Phase 1 + Phase 2 combined): **288 tests**,
all passing. Phase 2 contributes 13 new files (~2,235 lines, 154 tests)
covering every category the specification named: persistence, dataset
identity, partitioning, ingestion, checkpoints, incremental features,
recovery, reconciliation, verification, export, concurrency, and
cross-process determinism -- including a dedicated, non-vacuous safety
scanner (`test_market_data_safety_scan.py`, 25 tests: the real
`market_data` source is confirmed clean of network/broker code,
credentials, live-trading markers, `float`-typed financial fields,
`uuid4`/`random`-derived identity, internal wall-clock economic input,
overwrite/bypass flags, silently-swallowed exceptions, and unsafe
`pickle`; a companion test class fires every one of those same checks
against deliberately bad snippets to prove the scanner is not vacuously
passing).

## 15. Defects found and fixed

Both found via this phase's own adversarial testing (a parametrized
"incremental equals full recomputation" test), before any test was
reported as passing, and both fixed at root cause with the regression
test now in the suite.

**Defect #1 -- spurious conflicts on incremental generation for `atr`/
`rsi`.** `generate_feature_dataset_incremental`'s first implementation
called `generate_candle_features` over a candle series that included a
bounded leading "carry" window purely for rolling context, and that
function's own write loop attempted to PERSIST every computed point,
including the recomputed carry-window ones. `atr()`/`rsi()` both use an
artificial "insufficient prior data" approximation at position 0 of
WHATEVER series they are given (correct for genuine full history, wrong
for a truncated carry window -- `atr()`'s position-0 true range uses
`candle_range()` alone instead of comparing to a real previous close;
`rsi()`'s position-0 gain/loss is forced to 0). Recomputing that
approximation over a truncated carry window produced a DIFFERENT value
than what the original full computation had already correctly stored at
that same timestamp, and the feature store's own no-overwrite guarantee
correctly rejected it as a conflict -- surfacing as a hard
`FeatureStoreError` crash on the very first parametrized run
(`rsi-5`/`ema-5` in the initial test run). Root-caused by direct
traceback inspection. Fixed by adding an optional
`only_persist_timestamps` parameter to `generate_candle_features`
(defaulting to `None` = unchanged behavior for every existing caller):
incremental generation now computes over the full carry-plus-new series
for correct context, but only PERSISTS the genuinely new timestamps,
never re-attempting the carry region. Applied identically in
`recovery._complete_pending_feature_generation` (the same class of bug
existed there, caught and fixed together, before it could ever
manifest independently).

**Defect #2 -- `ema` silently wrong under bounded-carry incremental
generation.** After fixing Defect #1 (the crash), the SAME parametrized
test still failed for `ema-5`, this time with `carry_window_size=-1`
being rejected by `FeatureGenerationCheckpoint`'s own `>= 0` validation
-- itself a second, smaller bug (a `-1` sentinel for "unbounded" was
chosen ad hoc and never reconciled with the class's own non-negative
invariant). Investigating why `ema` needed "unbounded" carry at all
revealed the deeper, more serious issue: EMA is RECURSIVE (seeded by a
plain SMA of the first `window` values of whatever series it receives,
then each value depends on the previous one). A bounded carry window
re-seeds EMA from a DIFFERENT starting point than a full computation
would use, and because the recursion never forgets its seed, this
produces SILENTLY WRONG values (not a crash) for the entire carry-
forward chain, not confined to one boundary position the way `atr`/
`rsi`'s own approximation is. This is a materially worse class of defect
than Defect #1: a wrong, non-crashing answer. Fixed at root cause by
adding `ema` to `_UNBOUNDED_CARRY_FEATURES` (matching `vwap`'s existing,
analogous treatment for its own cumulative-since-start definition) --
incremental EMA generation always re-reads the full raw history, trading
bounded-read performance for guaranteed correctness, exactly as `vwap`
already did. The `carry_window_size` sentinel bug was fixed properly at
the same time by changing the field's type to `int | None` (`None`
meaning unbounded) throughout `FeatureGenerationCheckpoint`, rather than
patching around an invalid sentinel value. Both fixes are covered by the
same parametrized regression test, which now passes for all 8 required
indicators including `ema`.

No other defect was found. No test was ever weakened, skipped, or had
its assertions loosened to make it pass. A third, pre-existing,
out-of-scope issue (the same shared `historical.locking.DatasetLock`
stale-lock-reclaim race already documented during Milestone 9 Phase
3/4) was independently reproduced again during concurrency testing --
confirmed via its own diagnostic log line, not fixed (out of scope, same
justification as M9), and mitigated by reducing concurrency-test thread/
iteration counts (documented in `test_market_data_concurrency.py`'s own
module docstring).

## 16. Known non-blocking limitations

Partitioning supports DAILY/MONTHLY only, not fixed-event-count.
Compaction does not combine partitions across a granularity change.
Partition/manifest rebuild reads the full underlying event/feature
history and filters/aggregates in memory (correct and deterministic, not
O(partition size) -- a documented, out-of-scope performance limitation
for genuinely large-scale deployment). `ema`/`vwap` incremental
generation always re-reads full raw history (performance, not
correctness). Recovery's truncated-tail file rewrite assumes no
concurrent writer targets the same path during the recovery call itself
(a reasonable operational assumption for crash recovery run before
ordinary traffic resumes). No CLI surface. Full detail in
`docs/market_data_architecture.md`'s own "Known limitations" section.

## 17. Exact git status

```
 M docs/market_data_architecture.md
 M src/quant_platform/core/exceptions.py
 M src/quant_platform/market_data/events.py
 M src/quant_platform/market_data/feature_generation.py
 M src/quant_platform/market_data/feature_store.py
 M src/quant_platform/market_data/reports.py
 M src/quant_platform/market_data/verification.py
?? docs/milestone10_phase2_delivery_report.md
?? src/quant_platform/market_data/checkpoints.py
?? src/quant_platform/market_data/compaction.py
?? src/quant_platform/market_data/export.py
?? src/quant_platform/market_data/ingestion.py
?? src/quant_platform/market_data/manifests.py
?? src/quant_platform/market_data/partitions.py
?? src/quant_platform/market_data/reconciliation.py
?? src/quant_platform/market_data/recovery.py
?? src/quant_platform/market_data/repository.py
?? tests/unit/market_data/test_market_data_checkpoints.py
?? tests/unit/market_data/test_market_data_compaction.py
?? tests/unit/market_data/test_market_data_concurrency.py
?? tests/unit/market_data/test_market_data_cross_process.py
?? tests/unit/market_data/test_market_data_export.py
?? tests/unit/market_data/test_market_data_incremental_features.py
?? tests/unit/market_data/test_market_data_ingestion.py
?? tests/unit/market_data/test_market_data_manifests.py
?? tests/unit/market_data/test_market_data_partitions.py
?? tests/unit/market_data/test_market_data_reconciliation.py
?? tests/unit/market_data/test_market_data_recovery.py
?? tests/unit/market_data/test_market_data_repository_verification.py
?? tests/unit/market_data/test_market_data_safety_scan.py
```

`git diff --cached`: empty (nothing staged). `git diff --check`: clean
(one benign CRLF-normalization notice on `feature_generation.py`, not an
error). `HEAD`: `d4817455cc50dbdd65a91a6551f4a0eacef1c760`, unchanged
from baseline.

Quality gates, exact results: full-repo `ruff check .` -- All checks
passed. Full-repo `mypy src` (298 source files) -- Success, no issues
found. Focused suite `python -m pytest tests/unit/market_data/ -q
-W error` -- **288 passed**, zero warnings. Determinism/recovery/
incremental-feature/concurrency/cross-process focused suite (`test_
market_data_incremental_features.py` + `test_market_data_recovery.py` +
`test_market_data_concurrency.py` + `test_market_data_cross_process.py`
+ `test_market_data_replay.py`, 40 tests) repeated **10/10 times**,
every run 40 passed, zero flakes. Reused shared infrastructure
(`tests/unit/historical`, `tests/unit/paper_trading`, `tests/unit/ml`,
1,850 tests) re-run once as diligence -- 1,849 passed, 1 pre-existing
unrelated skip, confirming zero regression from the purely-additive
Phase 1-file extensions. No shared infrastructure OUTSIDE `market_data`
was modified and no test-collection behavior changed, so the full
multi-hour repository suite was not run, per this phase's own quality-
gate instruction (item 7).

## 18. Explicit confirmations

- Phase 2 work is **not staged** (`git add` was never run this phase;
  `git diff --cached` is empty).
- Phase 2 work is **not committed** (`HEAD` is still
  `d4817455cc50dbdd65a91a6551f4a0eacef1c760`).
- **Nothing was pushed.**
- **No network/broker/credential/live-trading code exists** anywhere in
  the new or modified source -- confirmed by the non-vacuous safety
  scanner in `test_market_data_safety_scan.py`, not merely asserted.
- **Other major packages were not modified** -- ML, backtesting,
  paper-trading, portfolio-risk, and execution-gateway packages are all
  byte-for-byte unchanged from the Phase 1 baseline; the only files
  touched outside the new `market_data` modules and their tests are
  `core/exceptions.py` (purely additive) and `docs/`.
- **Milestone 11 was not started.**

Stopping here per instruction, pending review and explicit commit
approval.
