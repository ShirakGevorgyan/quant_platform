# Milestone 10 Phase 3 -- Historical Ingestion Orchestration and Offline Source Adapters: Delivery Report

## 1. Baseline commit

Work began at commit `0a555db81d1d071f6002c69b36fffcffe878ed06` ("Add
durable versioned market data repository", the Milestone 10 Phase 2
commit). Confirmed before starting: `HEAD` matched this commit and
`git status --short` was clean. This report is written with `HEAD`
still at this same commit -- nothing has been committed during Phase 3.

## 2. Files added and modified

**New files** (`src/quant_platform/market_data/`):

| File | Lines | Purpose |
|---|---|---|
| `adapters.py` | 149 | Source-neutral adapter `Protocol`, `RawSourceRecord`, `SourceRowCoordinate`, `InMemorySourceAdapter` |
| `mappings.py` | 221 | `InstrumentMappingSpec`/`TimeframeMappingSpec`, versioned and content-addressed |
| `source_normalization.py` | 110 | Timestamp/Decimal parsing from raw source text |
| `source_manifests.py` | 198 | `SourceManifest` identity |
| `csv_adapter.py` | 273 | CSV candle adapter with configurable column mapping |
| `jsonl_adapter.py` | 191 | JSON Lines market-event adapter, strict per-`RecordKind` schema |
| `provenance.py` | 297 | `ProvenanceRecord`/`ProvenanceStore`, bidirectional row<->event linkage |
| `quarantine.py` | 319 | `QuarantineRecord`/`QuarantineStore`, 17 stable issue codes |
| `backfill.py` | 378 | Pure `BackfillPlan` planner, overlap/gap policy |
| `orchestration.py` | 1011 | 11-stage ingestion machine, dry-run, replay |

**Modified files** (all narrow, additive):

| File | Change |
|---|---|
| `src/quant_platform/core/exceptions.py` | +122 lines: 15 new exceptions (`HistoricalIngestionError` and subclasses) |
| `src/quant_platform/market_data/checkpoints.py` | +104/-2: added `HistoricalIngestionCheckpoint`, joined the existing `Checkpoint` union |
| `src/quant_platform/market_data/verification.py` | +169: `verify_source_manifest`, `verify_backfill_plan`, `verify_provenance_store`, `verify_quarantine_store`, `verify_historical_ingestion_checkpoint` |
| `src/quant_platform/market_data/reconciliation.py` | +112/-1: `reconcile_historical_ingestion_operation` |
| `src/quant_platform/market_data/reports.py` | +103: 7 new report functions |
| `tests/unit/market_data/test_market_data_safety_scan.py` | +80/-6: cloud-SDK/eval-exec/shell-execution/float-call checks, extended to cover every Phase 3 file automatically (the scanner globs the whole `market_data/` directory) |

**New test files** (`tests/unit/market_data/`, 11 files, 207 tests total):
`test_market_data_source_manifests.py` (24), `test_market_data_source_normalization.py` (34),
`test_market_data_csv_adapter.py` (16), `test_market_data_jsonl_adapter.py` (18),
`test_market_data_provenance.py` (16), `test_market_data_quarantine.py` (17),
`test_market_data_backfill.py` (21), `test_market_data_orchestration.py` (26),
`test_market_data_phase3_verification.py` (10), `test_market_data_phase3_reconciliation.py` (8),
`test_market_data_phase3_reports.py` (8), plus 9 new safety-scan tests.

No file outside `src/quant_platform/core/exceptions.py` and
`src/quant_platform/market_data/` was touched. `ml`, `historical`,
`backtesting`, `paper_trading`, `portfolio_risk`, and `execution_gateway`
are all unmodified.

## 3. Source adapter contract

`HistoricalSourceAdapter` (`adapters.py`) is a `Protocol` (structural
typing, no shared base class) exposing `source_kind()`,
`source_schema_version()`, `record_kind()`, `content_digest()`,
`byte_size()`, `describe()`, `iter_records()`. `iter_records()` yields
`RawSourceRecord` -- `row_index`, `raw_fields: dict[str, str]` (untyped
text exactly as read), `raw_text` (used for `record_digest()`). An
adapter never normalizes a value, never resolves a mapping, never
validates beyond what "reading it at all" requires (a CSV row of the
wrong physical width, or a JSONL line that is not valid JSON matching
its declared schema, are structural read failures --
`SourceAdapterError` -- everything else is `orchestration.py`'s job).

`RawSourceRecord` deliberately does NOT carry `source_manifest_id`: a
manifest's own `source_manifest_id` depends on `content_digest`, which
the adapter must expose BEFORE the manifest (and therefore its id) can
exist. `SourceRowCoordinate` (constructed later, by provenance/quarantine
code) pairs a `row_index` with the eventually-computed
`source_manifest_id`.

Three adapters ship: `CsvCandleAdapter` (`csv_adapter.py`,
`CsvColumnMapping` with its own content-addressed `mapping_id`, strict/
lenient extra-column handling, BOM stripping, `csv` module quoted-field
handling, ragged-row rejection), `JsonlMarketEventAdapter`
(`jsonl_adapter.py`, one strict schema per `RecordKind`, financial fields
must be JSON strings never JSON numbers, reuses `core.json.
parse_json_strict` for NaN/Infinity/duplicate-key rejection), and
`InMemorySourceAdapter` (`adapters.py`, for tests -- deliberately
performs NO schema enforcement of its own, which is exactly what lets
`orchestration.py`'s own row-shape re-validation be exercised rather
than dead code).

## 4. Source manifest identity

`SourceManifest.source_manifest_id` is `compute_content_id` over every
field except `source_manifest_id` itself, `creation_time` (operational),
and `row_count` (fully derived from `content_digest`, adds no
information). Confirmed by test: identical content + different
`creation_time` -> identical id; changed `content_digest`,
`instrument_mapping_id`, or `timezone_policy_id` -> changed id;
`source_label` (a caller-declared semantic assertion) participates in
identity, an actual filesystem path never does (it is never stored on
the object at all -- `SourceManifest` binds `content_digest`, and a
caller's own `describe()`/label choice, never a mutable path).

## 5. Timestamp and Decimal normalization rules

`TimestampParsingPolicy` requires an explicit, non-empty format list
(locale-dependent auto-parsing is forbidden by construction --
`__post_init__` rejects an empty list). A format carrying an explicit UTC
offset converts directly; a naive parse requires an explicit
`source_timezone`, else `TimezoneError`; DST-ambiguous/nonexistent local
times fail closed via `historical.timezones.localize_broker_timestamps`
(reused directly, confirmed via a real 2026-10-25 Europe/Berlin
fall-back-transition reproduction, both in the original architecture
work and in this phase's own test suite).

`parse_source_decimal` never routes through `float`; rejects NaN,
Infinity, and comma thousands-separators; normalizes signed zero
(`Decimal("-0")` -> `Decimal("0")`, since the two compare equal but
serialize to different strings, which would otherwise silently break
content-hash determinism for a source reporting a flat delta as
`"-0.00"`). **Defect found and fixed** (see Section 18): both functions
originally leaked `identity.py`'s more generic `MarketDataError` for
some failure modes instead of consistently raising this module's own
`HistoricalIngestionError`.

## 6. Instrument and timeframe mapping model

`InstrumentMappingSpec`/`TimeframeMappingSpec` (`mappings.py`) are
immutable, versioned, content-addressed -- NOT a global registry (per
the specification's explicit instruction); a caller constructs whatever
spec their own operation needs. `mapping_id` is a pure function of
sorted entries (order-independent). Resolution: an exact
`(source_symbol, provider)` match takes precedence over a
provider-wildcard (`provider=None`) entry; an unmapped symbol/timeframe
fails closed (`InstrumentMappingError`/`TimeframeMappingError`);
duplicate `(source_symbol, provider)` or duplicate `source_label` keys
are rejected at construction. `default_timeframe_mapping_spec()`
provides one reasonable, explicit starting table (`1m`/`M1` through
`1d`/`D1`) -- not the only legal spec.

## 7. Provenance model

`ProvenanceRecord` (`provenance.py`) binds `source_manifest_id` +
`source_row_index` to `event_id`, plus `source_record_digest`,
`original_timestamp_text`, `normalized_event_time`,
`instrument_mapping_id`, `resolved_instrument_id`,
`timeframe_mapping_id`, `timezone_policy_id`, `ingestion_batch_id`, and
`dataset_id`. `ProvenanceStore` is append-only, keyed by source
coordinate: an exact retry (identical `provenance_id`) is idempotently
absorbed; a conflicting one (same coordinate, different `provenance_id`)
raises `ProvenanceError`. `find_provenance_conflicts` is a pure
cross-record audit distinguishing `coordinate_bound_to_multiple_events`
(always a genuine defect -- the store itself refuses to durably record
this) from `event_bound_to_multiple_source_rows` (reported, not raised
-- legitimate exact-duplicate absorption is expected to look like this).
**`dataset_id` is excluded from `to_identity_payload()`** -- see Section
18, defect 3.

## 8. Row validation and quarantine behavior

17 stable, machine-readable issue codes (`quarantine.py`):
`missing_required_column`, `extra_forbidden_column`, `empty_timestamp`,
`malformed_timestamp`, `naive_timestamp_without_policy`,
`ambiguous_or_nonexistent_local_time`, `invalid_decimal`,
`non_finite_decimal`, `negative_volume`, `invalid_ohlc`,
`unknown_symbol`, `unknown_timeframe`, `duplicate_source_row_coordinate`,
`duplicate_source_record_digest`, `conflicting_source_sequence`,
`future_timestamp`, `timestamp_outside_declared_range`. Row validation
lives in `orchestration._process_row` (never the adapter): every issue
found on a row is collected (not just the first), so a pathological row
can carry multiple codes at once. `NEGATIVE_VOLUME`/`INVALID_OHLC` are
deliberately reused for TICK/QUOTE/TRADE's own positivity/ordering rules
(price > 0, bid <= ask, etc.) -- the 17-code vocabulary has no dedicated
code for these, and both are the closest existing semantic match; this
also prevents an unhandled `MarketDataEventError` from the Phase 1
event constructors, which would otherwise crash the whole operation on
a single bad row.

`QuarantineRecord` stores the already-text-only `raw_fields` (never a
blob -- every adapter already restricts a raw record to string fields) a
`RetryEligibility` classification (RETRYABLE if corrected source content
alone could fix it; PERMANENT if it needs a mapping/timezone-policy
config change), keyed by the physical `(source_manifest_id, row_index)`
coordinate. **`ingestion_batch_id` is excluded from
`to_identity_payload()`** -- see Section 18, defect 1.

## 9. Backfill planning

`create_backfill_plan` (`backfill.py`) is a pure function: no filesystem
or repository I/O. Inputs include `existing_covered_partition_keys:
frozenset[str]` (a plain value the caller reads from the repository
first) rather than a repository handle. Every reported interval
(`missing_intervals`, `overlapping_intervals`, `gap_intervals`, each
`BackfillBatch`) is aligned to a whole PARTITION, matching Phase 2's own
whole-partition rebuild model. `backfill_plan_id` and ordered `batches`
are a pure function of inputs (`to_identity_payload` excludes only the id
itself and `creation_time`); confirmed deterministic across repeated
calls and independent of `creation_time`.

## 10. Overlap and gap policy

`OverlapPolicy`: `REJECT_ANY_OVERLAP` is the only variant fully
decidable at plan time (blocks immediately if any touched partition is
already covered). `EXACT_DUPLICATES_ONLY`/`ALLOW_LATE_ARRIVAL_NEW_VERSION`
both keep the plan admissible but differ in the RUNTIME CONTRACT
orchestration must honor for an overlapping partition's rows
(`EXACT_DUPLICATES_ONLY` carries a warning noting this deferred
contract).

`GapPolicy`: a "gap" is the portion of missing partitions that falls
entirely outside the source manifest's own declared
`[expected_start, expected_end)` range (computed purely from
`SourceManifest` fields, no row data needed). `allow_and_report` never
blocks; `reject` blocks on any gap; `require_expected_market_calendar`
re-checks each gap against `calendar.TradingCalendar` (reused directly)
via `enumerate_expected_open_times`, using `SourceManifest.
expected_timeframe` -- only a gap during expected-open market time
blocks the plan. Confirmed via direct reproduction: a weekend gap under
the default XAUUSD calendar does NOT block; a weekday gap does
(`GAP_CALENDAR_OPEN` present in `blocking_issue_codes`).

## 11. Orchestration state machine

`IngestionStage`: `SOURCE_VERIFIED -> PLAN_CREATED -> BATCH_RESERVED ->
ROWS_PARSED -> ROWS_VALIDATED -> EVENTS_NORMALIZED ->
REPOSITORY_COMMITTED -> PROVENANCE_COMMITTED -> CHECKPOINT_COMMITTED ->
VERIFIED -> COMPLETED`. `OperationStore.advance` is the single mutator:
for a NEW `operation_id`, the first recorded stage must be
`SOURCE_VERIFIED`; for an EXISTING one, the submitted `content_digest`
(computed from source manifest, backfill plan, mapping/normalization
spec ids, row-failure policy, and target dataset) must match what is
already bound, else `OrchestrationConflictError`. A retry of ANY
previously-recorded stage (not only the latest -- a full re-play of an
already-`COMPLETED` operation re-submits every stage from
`SOURCE_VERIFIED` again) is idempotently absorbed if its evidence
matches the historical record at that exact stage, and raises
`OrchestrationConflictError` if it does not; advancing past latest+1
raises `OrchestrationStateError`.

`VERIFIED` explicitly re-reads both the repository and the provenance
store and cross-checks them (`_verify_repository_provenance_agreement`)
before `COMPLETED` is ever recorded -- satisfying "no completed status
before repository and provenance agree" structurally, not by convention.

## 12. Resume and recovery semantics

Every stage's evidence is a pure function of (adapter content + specs),
recomputed fresh on every call -- "no in-memory-only correctness state."
The one genuine exception is the repository append SEQUENCE this
operation's events must use: durably pinned the first time
`BATCH_RESERVED` is recorded (`_resolve_sequence_start` walks the
operation's own history first; only computes fresh via
`next_sequence_for` if no prior `BATCH_RESERVED` record exists) and
always reused on every subsequent call -- so a resumed operation
reproduces the exact same content-addressed `event_id`s as the original
attempt, even if the repository's own "next sequence" has moved on for
unrelated reasons in between. A changed source file (different
`adapter.content_digest()`) fails closed immediately, before any stage
advances.

## 13. Dry-run guarantees

`dry_run=True` performs the identical parse/validate/normalize
computation as a real run (so its counts/digests are genuinely accurate)
but calls `OperationStore.advance`, `QuarantineStore.append`,
`ProvenanceStore.append`, `ingest_raw_events`, and `CheckpointStore.append`
NOWHERE. Its `resulting_dataset_id` preview reuses
`ingestion.semantic_digest_for_raw_events`/`physical_digest_for_ids`
(pure) and `partitions.build_partition` (pure) against the union of
currently-durable events (a READ) and this call's newly-normalized
events -- confirmed by direct test to produce the EXACT same
`resulting_dataset_id` a subsequent real commit produces. Filesystem
state (`OperationStore`/`QuarantineStore`/`ProvenanceStore`/
`CheckpointStore`/`MarketEventStore`/`DatasetManifestStore`) is asserted
empty before and after every dry-run test.

## 14. Reconciliation

`reconcile_historical_ingestion_operation` (`reconciliation.py`) spans
four stores plus manifest history for one operation: quarantined and
provenance-covered coordinates must be disjoint; `valid_row_count +
quarantined_row_count == parsed_row_count`; every provenance record's
count matches the operation's own `valid_row_count`; quarantine
COMPLETENESS is checked by COORDINATE (via the durably-recorded
`quarantined_row_indices` evidence list), never by re-filtering the
quarantine store on this operation's own `ingestion_batch_id` -- see
Section 18, defect 1, for why the naive count-filter approach was wrong;
`resulting_dataset_id` must appear in the dataset's own manifest
history; a `HistoricalIngestionCheckpoint` must exist for a `COMPLETED`
operation and itself verify clean. Ordinary inconsistencies become
structured `ValidationIssue`s; `reconcile_historical_ingestion_operation`
never raises for an unknown `operation_id`, it reports
`operation_not_found`.

## 15. Verification independence classification

**Structurally independent** (recomputes an object's own content id
from its own recorded fields, or cross-checks durable evidence across
stores -- never trusts a cached report or in-memory set):
`verify_source_manifest`, `verify_backfill_plan`, `verify_provenance_store`
(plus `find_provenance_conflicts`), `verify_quarantine_store`,
`verify_historical_ingestion_checkpoint` (recomputes its own id, and the
referenced `RawIngestionCheckpoint`'s own id, from recorded fields).

**NOT independently re-verified by default** (an honest, disclosed
limitation): none of the above re-reads the ORIGINAL source bytes/file
-- the strongest possible check, "does a fresh reparse from source bytes
reproduce this exact manifest/plan/provenance." That check is what
`orchestration.run_ingestion_operation`'s own `SOURCE_VERIFIED` stage
already performs (`adapter.content_digest() != source_manifest.
content_digest` fails closed), and what `replay_ingestion_operation`
performs end-to-end against a fresh repository; a caller wanting an
independent, standalone version of exactly that check re-reads the
source file/adapter and compares digests directly.

`verify_historical_ingestion_checkpoint` deliberately does NOT call
`checkpoints.verify_raw_ingestion_checkpoint` against CURRENT live
repository state (see Section 18, defect 4) -- it only confirms the
referenced `RawIngestionCheckpoint` reproduces ITS OWN id from its own
recorded fields, proving "this repository state genuinely existed at
commit time," never "the repository has not changed since."

## 16. Replay determinism

`replay_ingestion_operation` is a thin, explicitly-named alias for
`run_ingestion_operation` -- there is no separate replay mechanism to
write, because the function is already pure-deterministic given the same
adapter/manifest/plan/specs. Confirmed by direct test across TWO
independent, separately-created temporary filesystem roots: identical
`resulting_dataset_id`, identical `normalized_events_digest`, identical
`quarantine_issue_counts`, and identical committed `event_id` sets.
`generate_replay_comparison_report` deliberately excludes `operation_id`/
`dataset_key` from its comparison (a replay legitimately targets a
different `operation_id` and/or repository namespace).

## 17. Tests and exact results

11 new test files, 207 new tests, plus 9 new safety-scan tests (34 total
in that file, up from 25). Full breakdown: source manifests/mappings 24,
normalization/adapters-contract 34, CSV adapter 16, JSONL adapter 18,
provenance 16, quarantine 17, backfill planning 21, orchestration
(stage machine, fail-fast/quarantine, dry-run, replay, concurrency,
adversarial) 26, Phase 3 verification 10, Phase 3 reconciliation 8,
Phase 3 reports/replay-comparison 8.

Exact results, all commands run from the repository root:

- `git diff --check`: clean (one pre-existing CRLF-normalization notice
  on `reconciliation.py`, not a `--check` violation; exit code 0).
- `ruff check .` (full repo): **All checks passed!**
- `mypy src` (full repo, strict): **Success: no issues found in 308
  source files.**
- `pytest tests/unit/market_data -q -W error`: **495 passed** (288
  pre-existing Phase 1/2 + 207 new Phase 3).
- `pytest tests/unit -q` (full repo suite, run as an extra sanity gate
  even though no shared infrastructure OUTSIDE `market_data` was
  touched): **5700 passed, 1 skipped** (the skip is `ml/test_artifacts.py`'s
  pre-existing, unrelated "symlink creation requires elevated privileges
  on Windows" skip).
- x10 repeat of `test_market_data_orchestration.py`,
  `test_market_data_backfill.py`, `test_market_data_phase3_reports.py`
  (replay-comparison coverage), `test_market_data_recovery.py`,
  `test_market_data_replay.py`, and `test_market_data_concurrency.py`
  together (76 tests per run): **10/10 runs, 76 passed each, zero
  flakes.**

No test was weakened, skipped, or had its assertions loosened to pass.

## 18. Genuine defects found and fixed

All four were found by this phase's own test-writing process (not
present in any spec bullet as a known issue) and each received a
regression test.

**Defect 1 -- `QuarantineRecord` identity included `ingestion_batch_id`.**
Two independent operations (different `operation_id`s) re-reading the
SAME unmodified source and rediscovering the SAME physically-bad row
produced two records with the same evidence but a different
`ingestion_batch_id`, hence a different `quarantine_record_id`, hence a
spurious `SourceQuarantineError` on the second operation -- a plausible,
ordinary scenario (e.g. an automation minting a fresh `operation_id` per
scheduled run against a file that has not changed) crashing instead of
converging. **Fix**: excluded `ingestion_batch_id` from
`QuarantineRecord.to_identity_payload()` (kept in `to_json_dict()` for
observability; first writer's batch id is retained on idempotent
absorption, exactly like `creation_time` elsewhere in this package).
This in turn required reconciliation's row-count check to stop
filtering the quarantine store by `ingestion_batch_id` (which would now
undercount a row absorbed under an earlier operation's id) and instead
check coordinate-level completeness against a newly-added
`quarantined_row_indices` list in the `ROWS_VALIDATED` stage evidence.
Regression tests: `test_market_data_quarantine.py::TestQuarantineStore::
test_two_independent_operations_rediscovering_the_same_bad_row_converge`,
plus `test_market_data_orchestration.py`'s
`test_many_threads_quarantining_the_same_row_converge`.

**Defect 2 -- inconsistent exception types in `source_normalization.py`.**
`parse_source_timestamp`'s empty-string check and `parse_source_decimal`'s
delegation to `identity.parse_decimal` both leaked the more generic
`MarketDataError` instead of this module's own `HistoricalIngestionError`
for some failure modes, while other failure modes in the SAME two
functions correctly raised `HistoricalIngestionError` -- an inconsistent
contract for any direct caller (or future code) catching this module's
own exception type. **Fix**: the empty-string check now raises
`HistoricalIngestionError` directly; `parse_source_decimal` now catches
`MarketDataError` from `identity.parse_decimal` and re-raises as
`HistoricalIngestionError`. Regression tests:
`test_market_data_source_normalization.py::TestParseSourceTimestamp::
test_empty_timestamp_rejected` and `TestParseSourceDecimal::
test_rejects_nan`/`test_rejects_infinity`/`test_rejects_malformed_text`.

**Defect 3 -- `ProvenanceRecord` identity included `dataset_id`, and a
provenance conflict could leave an ORPHAN event.** Two problems compounded:
(a) `dataset_id` (the resulting dataset VERSION) is a repository-state
snapshot that can legitimately drift for reasons unrelated to a specific
row (unrelated concurrent ingestion into the same dataset), so including
it in provenance identity made an otherwise-exact retry spuriously
conflict; (b) far more seriously, `run_ingestion_operation` committed to
the repository (`ingest_raw_events`) BEFORE writing provenance records --
if the provenance step then failed (a genuine, permanent conflict:
row already claimed by a different operation's event_id), the just-
committed event was left durably in the repository with NO provenance
and NO way to ever acquire one, discovered via a reproduction where the
event count read 2 after a failed operation instead of the expected 1.
**Fix**: excluded `dataset_id` from `ProvenanceRecord.to_identity_payload()`
(kept in `to_json_dict()`), AND added a pre-flight conflict check --
before `ingest_raw_events` is ever called, every valid row's coordinate
is checked against already-durable provenance; any event_id conflict
raises `ProvenanceError` before any repository write happens at all.
Regression tests: `test_market_data_provenance.py::
test_dataset_id_excluded_from_identity`,
`test_market_data_orchestration.py::TestAdversarial::
test_provenance_conflict_leaves_no_orphan_event`,
`test_market_data_phase3_reconciliation.py::
test_reprocessing_the_same_rows_under_a_new_operation_id_fails_closed`.

**Defect 4 -- `verify_historical_ingestion_checkpoint` treated expected
staleness as a defect.** It called `checkpoints.verify_raw_ingestion_checkpoint`
against the referenced `RawIngestionCheckpoint`, which checks that
checkpoint against CURRENT live repository state -- but a
`HistoricalIngestionCheckpoint` embeds a SPECIFIC, point-in-time
reference; once ANY later activity (this operation's own later stages,
or an unrelated later operation) advances the repository further, that
reference is legitimately "stale" relative to current state, which is
normal, not tampering. Confirmed via reproduction: a second, genuinely
independent, non-overlapping operation against the same `dataset_key`
caused the first operation's own already-COMPLETED checkpoint to report
`stale_repository_checkpoint`. **Fix**: replaced the live-state check
with a pure self-consistency recomputation of the referenced
`RawIngestionCheckpoint`'s own id from its own recorded fields (forged-
identity detection only, no dependency on current repository state).
Regression tests: `test_market_data_phase3_verification.py::
TestVerifyHistoricalIngestionCheckpoint::
test_checkpoint_still_valid_after_unrelated_later_activity`.

## 19. Known non-blocking limitations

See `docs/market_data_architecture.md`'s "Known limitations" section,
Phase 3 subsection, for the authoritative, maintained list. Summary:
CSV candles have no per-row timeframe column (declared once on the
manifest); reprocessing the exact same rows under a NEW `operation_id`
against an already-populated dataset correctly fails closed rather than
silently duplicating (a deliberate consequence of per-operation sequence
pinning, not a bug -- genuine retries must reuse the same
`operation_id`); calendar-aware gap policy inherits every disclosed
`TradingCalendar` limitation for OTC/provider-specific sessions;
`DUPLICATE_SOURCE_ROW_COORDINATE` is structurally unreachable through any
shipped adapter (defensive only, exercised via an adversarial hand-built
record list in tests, not through real adapter code).

## 20. Exact git status

```
 M docs/market_data_architecture.md
 M src/quant_platform/core/exceptions.py
 M src/quant_platform/market_data/checkpoints.py
 M src/quant_platform/market_data/reconciliation.py
 M src/quant_platform/market_data/reports.py
 M src/quant_platform/market_data/verification.py
 M tests/unit/market_data/test_market_data_safety_scan.py
?? docs/milestone10_phase3_delivery_report.md
?? src/quant_platform/market_data/adapters.py
?? src/quant_platform/market_data/backfill.py
?? src/quant_platform/market_data/csv_adapter.py
?? src/quant_platform/market_data/jsonl_adapter.py
?? src/quant_platform/market_data/mappings.py
?? src/quant_platform/market_data/orchestration.py
?? src/quant_platform/market_data/provenance.py
?? src/quant_platform/market_data/quarantine.py
?? src/quant_platform/market_data/source_manifests.py
?? src/quant_platform/market_data/source_normalization.py
?? tests/unit/market_data/test_market_data_backfill.py
?? tests/unit/market_data/test_market_data_csv_adapter.py
?? tests/unit/market_data/test_market_data_jsonl_adapter.py
?? tests/unit/market_data/test_market_data_orchestration.py
?? tests/unit/market_data/test_market_data_phase3_reconciliation.py
?? tests/unit/market_data/test_market_data_phase3_reports.py
?? tests/unit/market_data/test_market_data_phase3_verification.py
?? tests/unit/market_data/test_market_data_provenance.py
?? tests/unit/market_data/test_market_data_quarantine.py
?? tests/unit/market_data/test_market_data_source_manifests.py
?? tests/unit/market_data/test_market_data_source_normalization.py
```

`git diff --cached` is empty; nothing has been staged.

## 21. Explicit confirmations

- **Nothing staged**: `git diff --cached --stat` is empty.
- **Nothing committed**: `HEAD` is still `0a555db81d1d071f6002c69b36fffcffe878ed06`,
  identical to the confirmed Phase 2 baseline; `git log --oneline -3`
  shows no new commit.
- **Nothing pushed**: no `git push` was run at any point in this phase.
- **No network, broker, credential, or live-trading code**: confirmed by
  the extended `test_market_data_safety_scan.py` (34 tests, each proven
  non-vacuous against a deliberately bad snippet) -- no `socket`/
  `requests`/`httpx`/`aiohttp`/`urllib.request`/`websocket`/`ftplib`/
  `telnetlib` imports, no `MetaTrader5`/`mt5`/`fxpro` imports, no
  `boto3`/`azure`/`google.cloud`/`gcloud` imports, no credential-shaped
  field declarations, no `LIVE_TRADING`/`place_live_order`/
  `submit_to_broker` markers, no `eval`/`exec`, no `subprocess`/
  `os.system`/`os.popen`, no unsafe `pickle`, no `float`-typed financial
  dataclass fields or `float(...)` calls outside prose, no `uuid4`/
  `random`-derived economic identity, no internal wall-clock read, no
  overwrite/bypass flag, no bare/silently-swallowed exception handling.
- **Other packages unmodified**: `ml`, `historical`, `backtesting`,
  `paper_trading`, `portfolio_risk`, `execution_gateway` all show zero
  changes in `git status --short`.
- **Milestone 11 not started**: no code, module, or reference to a
  future milestone exists anywhere in this diff.
- **No real external collectors, live streaming, or CLI expansion**:
  confirmed -- every adapter is offline-only (local file or in-memory
  fixture); `quant_platform.cli` (if it exists) was not touched.

Stopping here per instruction. Awaiting explicit commit approval before
staging or committing any of the above.
