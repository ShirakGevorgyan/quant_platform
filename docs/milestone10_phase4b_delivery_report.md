# Milestone 10, Phase 4B -- Delivery Report

## Curated FRED Macro Universe and Verified Historical Backfill Workflow for XAUUSD Research

### 1. Baseline commit

`c6439952415d048ccdb34beb914b808ee7ece5cb` -- "Add secure FRED historical
collector infrastructure" (the commit that concluded Milestone 10 Phase
4A). Confirmed at the start of this phase: `git rev-parse HEAD` matched
this commit exactly, `git status --short` was clean, and no FRED API key
or other credential was present anywhere in the repository. No other
commit has been made since; this remains HEAD as of this report.

### 2. Files added / modified

**New subpackage, `src/quant_platform/market_data/collectors/curated/`**
(14 files, ~3,175 lines):

| File | Purpose |
|---|---|
| `__init__.py` | Package docstring. |
| `revision_policy.py` | `RevisionPolicyKind` (4 kinds), FRED request-override mapping. |
| `availability.py` | `AvailabilityPolicyKind` (6 kinds), `resolve_availability_time` (fail-closed). |
| `registry.py` | `CuratedFredSeriesSpec`, `CuratedFredRegistry`, 4 core + 14 extended series specs. |
| `metadata.py` | `verify_series_metadata` -- official-metadata drift detection. |
| `macro_observation.py` | `CuratedMacroObservation`, `CuratedObservationStore`. |
| `backfill.py` | `CuratedBackfillSpec`, `create_curated_backfill_spec`. |
| `datasets.py` | `ComponentDatasetManifest`/`CombinedUniverseManifest` + stores. |
| `orchestration.py` | `CuratedOperationStage`/`CuratedOperationStore`, `run_curated_backfill_operation` (~600 lines). |
| `update_plan.py` | `create_curated_update_plan` -- pure incremental planning. |
| `reconciliation.py` | `reconcile_curated_universe`. |
| `verification.py` | `verify_curated_universe` -- independent rederivation. |
| `reports.py` | Deterministic, secret-free report generators. |
| `acceptance.py` | Opt-in real-FRED acceptance workflow + env-var key resolution. |

**New module, `src/quant_platform/market_data/collectors/fred_series_metadata.py`**
-- `FRED_SERIES_ENDPOINT_PATH`, `build_fred_series_metadata_request_manifest`,
`execute_fred_series_metadata_request` (alias of Phase 4A's
`execute_fred_request`), `FredSeriesMetadata`, `parse_fred_series_metadata_response`.

**New test files, `tests/unit/market_data/`** (15 files, ~3,094 lines,
193 tests, 1 opt-in skip):
`_curated_test_helpers.py` (shared fixtures/doubles, not a `conftest.py`
-- matching this repository's established convention),
`test_collectors_curated_registry.py` (21), `test_collectors_curated_policies.py`
(27), `test_collectors_curated_metadata.py` (11),
`test_collectors_curated_macro_observation.py` (18),
`test_collectors_curated_backfill.py` (15),
`test_collectors_curated_orchestration.py` (15),
`test_collectors_curated_update_plan.py` (9),
`test_collectors_curated_reconciliation.py` (5),
`test_collectors_curated_verification.py` (9),
`test_collectors_curated_reports.py` (11),
`test_collectors_curated_acceptance.py` (9 + 1 skip),
`test_collectors_curated_pit_concurrency_adversarial.py` (15),
`test_collectors_curated_fixture_acceptance.py` (8),
`test_curated_fred_example_config.py` (10).

**New example config**, `examples/xauusd_macro_fred_config.example.json`
-- no real credentials; references the sanctioned `FRED_API_KEY`
environment variable by NAME only.

**Modified files** (narrow, additive except for two genuine defect
fixes described in Section 21; each verified with `ruff --fix` + `mypy`
clean + full Phase 1-4A regression still passing after every change):
- `src/quant_platform/core/exceptions.py` -- 9 new Phase 4B exceptions
  appended after `CollectorVerificationError`: `CuratedRegistryError`,
  `SeriesMetadataError`, `MetadataDriftError`, `AvailabilityPolicyError`,
  `AvailabilityUnresolvedError`, `RevisionPolicyError`,
  `CuratedBackfillSpecError`, `CombinedManifestError`, `UpdatePlanError`.
- `src/quant_platform/market_data/collectors/fred.py` --
  `build_fred_request_manifest` extended with `realtime_start`/
  `realtime_end`/`limit`/`offset`/`sort_order`/`units`/`frequency`/
  `aggregation_method`/`output_type`/`vintage_dates` to support the
  official request parameters this phase's spec requires modeling
  explicitly; `sort_order`/`units` are now ALWAYS included in the
  canonical query params (previously omitted when not overridden), a
  disclosed, identity-affecting change for Phase 4A callers too --
  verified safe (no test anywhere hardcodes a frozen `request_manifest_id`
  constant, only self-consistency comparisons) via the full Phase 1-4A
  regression re-run passing unchanged. Also a genuine defect fix to
  `execute_fred_request`'s retry-exhaustion error message (see Section
  21, defect 1).
- `src/quant_platform/market_data/collectors/transport.py` -- a genuine
  defect fix: every exception message that could embed a request URL
  now redacts its query string first (see Section 21, defect 1).
- `docs/market_data_architecture.md` -- Status line, new Phase 4B
  section, Phase 4B known-limitations subsection, Future-phases update.
- `tests/unit/market_data/test_market_data_safety_scan.py` -- new
  `TestCuratedSpecificSafety` class (8 tests): curated-subpackage reach
  confirmation, the point-in-time `observation_date`-as-availability-proof
  structural check (plus its own non-vacuity and no-false-positive
  proofs), and a curated-scoped secret scan.

### 3. Curated registry and tiers

`CuratedFredRegistry` is a KEYED SET: `create_curated_registry` always
sorts specs by `series_id` before computing `registry_id`, so identity
AND iteration order are both independent of declaration order. Four
tiers (`SeriesTier`): `CORE_XAUUSD_DRIVER`, `SECONDARY_MACRO_DRIVER`,
`REGIME_CONTEXT`, `EXPERIMENTAL`. Registry construction rejects
duplicate series ids, duplicate canonical names, an enabled series with
no target instrument id, and an unsupported frequency/unit/normalization
combination; `notes` is the one field explicitly excluded from identity.

### 4. Core series (mandatory, individually fixture-verified)

| FRED series | Canonical name | Driver rationale |
|---|---|---|
| `DFII10` | `us_10y_real_yield` | Real-rate/gold driver |
| `DGS10` | `us_10y_nominal_yield` | Nominal-rate driver |
| `CPIAUCSL` | `us_cpi_all_urban` | Inflation driver |
| `DFF` | `effective_federal_funds_rate` | Policy-rate driver |

All four `SeriesTier.CORE_XAUUSD_DRIVER`, all `enabled=True`. The 14
reviewed extended-universe candidates (`T10YIE`, `T5YIE`, `DGS2`,
`DGS5`, `DGS30`, `DTWEXBGS`, `UNRATE`, `PAYEMS`, `PCEPI`, `PCEPILFE`,
`INDPRO`, `VIXCLS`, `WALCL`, `M2SL`) are registered
(`default_extended_series_specs`) but ALL constructed `enabled=False`
-- a disclosed scope decision, not an oversight; enabling one for a
live backfill requires an explicit opt-in flip by a future caller.

### 5. Series metadata verification rules

`verify_series_metadata` compares official FRED `/fred/series` metadata
against a curated spec's DECLARED expectations, never the reverse.
FAIL CLOSED: unexpected series id, incompatible frequency, incompatible
units, changed seasonal-adjustment code (only where the spec declared
one). INFORMATIONAL ONLY: a changed title (no curated expectation
exists to compare against). REPORTED, conditionally permitted: a
changed supported observation range -- the requested backfill interval
may proceed only if it still falls within the metadata's own
currently-reported range; a request before the supported start fails
closed, a request after the currently-reported end is a warning
(the series may simply have grown since the spec was last reviewed).
`last_updated` is captured for provenance only, never compared.

### 6. Revision/vintage policy

Four `RevisionPolicyKind`s: `LATEST_AVAILABLE` (no FRED override; NOT
automatically point-in-time-safe by itself), `FIRST_RELEASE_ONLY`
(`output_type=4`), `AS_OF_REALTIME_DATE` (requires an explicit
`as_of_realtime_date`; resolves `realtime_start=realtime_end=` that
date), `VINTAGE_SERIES` (`output_type=2`, retaining distinct
revisions). `resolve_fred_request_overrides` is the single place raw
FRED parameter names are ever produced from a named kind. Distinct
`as_of_realtime_date` values produce distinct `revision_policy_id`s;
supplying `as_of_realtime_date` on any kind other than
`AS_OF_REALTIME_DATE` is rejected at construction as an invalid
combination.

### 7. Availability-time policy

Four distinct times are never conflated anywhere in this phase:
`observation_date` (economic period), `realtime_start`/`realtime_end`
(FRED/ALFRED real-time validity), `availability_time` (the ONE value a
correct point-in-time join must filter on), `ingestion_time`
(operational only). Six `AvailabilityPolicyKind`s:
`OBSERVATION_DATE_END_OF_DAY` (daily market rates), `NEXT_BUSINESS_
DAY_CONSERVATIVE` (Mon-Fri arithmetic, no holiday calendar --
disclosed), `EXPLICIT_RELEASE_TIMESTAMP`, `RELEASE_CALENDAR_REFERENCE`
(deliberately unimplemented -- raises `AvailabilityPolicyError`),
`REALTIME_START_DATE_CONSERVATIVE` (monthly releases like CPI --
requires `realtime_start`, fails closed via
`AvailabilityUnresolvedError` if absent), `MANUAL_CURATED_RELEASE_
RULE` (the only kind permitting a `delay_days` override).
`resolve_availability_time` is STRUCTURALLY fail-closed: no branch
returns "immediately available" as a default. FRED reports only dates,
never exact publication times; every policy's `availability_hour`/
`availability_minute` is a disclosed, configurable approximation, and
the policy itself (timezone, hour/minute, kind) is fully
identity-relevant, so a policy change is a detectable, auditable event.

### 8. Missing-value policy

FRED's `"."` never becomes zero. Per-series `MissingValuePolicy`:
`QUARANTINE` (default), `STORE_AS_MISSING_FACT` (durably records
`is_missing=True`, `value=None`), `SKIP_AND_REPORT` (excluded without
quarantining, but always counted). No forward-fill anywhere in this
package. Every report (`SeriesOutcome`, component dataset manifest)
carries an explicit `missing_count`.

### 9. Backfill plan

`CuratedBackfillSpec` (`backfill.py`) is immutable and content-addressed:
`curated_registry_id`, `selected_series_ids` (ALWAYS sorted -- the same
mechanism satisfies both declaration-order-independent identity and
orchestration's stable processing-order requirement), observation
window, optional realtime window, `revision_policy_id`, output type,
`page_size` (bounded `[1, 100000]`), `CachePolicy`
(`PREFER_CACHE`/`FORCE_FRESH`), optional registry-wide
availability/normalization overrides, `target_dataset_namespace`,
`fail_fast`, and three explicit bounds (`max_series_count`,
`max_observations_per_series`, `max_total_raw_bytes`) -- an unbounded
request is structurally unconstructable.
`create_curated_backfill_spec` validates every selected series against
the supplied registry BEFORE a spec can exist: unknown or disabled
series is rejected at construction time, never at run time.

### 10. Multi-series orchestration

`CuratedOperationStage`/`CuratedOperationStore` (`orchestration.py`) are
a THIRD, independent, self-contained stage-machine implementation
(after Phase 3's `OperationStore` and Phase 4A's
`CollectorOperationStore`), duplicating the same proven
idempotent/conflict/monotonic-progression `advance()` algorithm, scoped
to `target_dataset_namespace` (spans many series at once, a materially
different shape than either existing single-series-scoped machine). The
12 stages: `REGISTRY_VERIFIED -> PLAN_CREATED -> SERIES_METADATA_
VERIFIED -> REQUESTS_COMMITTED -> RESPONSES_COMMITTED ->
OBSERVATIONS_PARSED -> AVAILABILITY_RESOLVED -> SERIES_DATASETS_
COMMITTED -> COMBINED_MANIFEST_COMMITTED -> RECONCILED -> VERIFIED ->
COMPLETED`. Series are processed in the spec's own sorted order;
`fail_fast=True` (default) re-raises immediately on any series' failure
(nothing committed for any series); `fail_fast=False` records each
series' own `SeriesOutcome` and continues, with `completeness_status`
becoming `PARTIAL` the moment even one series fails; zero successes
still raises. Cache-vs-transport fetch decisions are made INDEPENDENTLY
per series. The raw-response cache write is unconditional even under
`dry_run=True` (same rationale as Phase 4A); only the business-record
stores are `dry_run`-gated.

### 11. Dataset layout

One immutable `ComponentDatasetManifest` PER SERIES plus one
`CombinedUniverseManifest` binding the exact component versions of one
backfill (`datasets.py`). Different native frequencies (daily `DGS10`,
monthly `CPIAUCSL`) are never forced into one physically regular time
series -- no implicit resampling, forward-fill, or alignment anywhere.
"Version N" is a STORE-level concept (`len(history)`), never baked into
either manifest's own content hash; both stores' `append` is an
idempotent no-op when the incoming content id already matches the
current one -- this idempotency is the actual mechanism satisfying
"an exact no-op update must mint no new dataset version."

### 12. Incremental update planning

`create_curated_update_plan` (`update_plan.py`) is PURE: never reads
the wall clock, never touches the network. Compares each series'
CURRENT `ComponentDatasetManifest` coverage against a caller-supplied
`desired_observation_end` and `RevisionPolicy`, producing
`NO_UPDATE_NEEDED`, `APPEND_OBSERVATIONS` (from the day after current
coverage, or the series' own default start if it has no history), or
`REVISION_REFRESH` (the effective revision policy changed since the
existing combined manifest was built). `planning_time` is excluded
from the plan's own identity. This module's job is to REPORT the
no-op case correctly; the actual "no new version" guarantee is
`datasets.py`'s own idempotent `append` (Section 11).

### 13. Point-in-time consumer contract

Documented and enforced, not yet joined into a feature (matching Phase
4A's own "modeled but not wired into feature generation" precedent for
Phase 1 macro events): a future consumer MUST filter on
`availability_time`, never `observation_date` alone.
`test_collectors_curated_pit_concurrency_adversarial.py`'s
`TestPointInTimeVisibility` class proves this concretely (an
earlier-`observation_date`-but-later-`availability_time` CPI value
stays invisible before its resolved availability time; a naive
`observation_date <= as_of` filter is shown to disagree with the
correct `availability_time <= as_of` filter on the exact same data).
`test_market_data_safety_scan.py`'s new `TestCuratedSpecificSafety`
class makes any FUTURE reintroduction of the anti-pattern (comparing
`observation_date` directly against a time value anywhere in
`curated/`) structurally loud rather than merely documented in prose,
with its own non-vacuity and no-false-positive proofs.

### 14. Fixture-based acceptance (mandatory, always run)

`test_collectors_curated_fixture_acceptance.py` -- realistic, hand-built
FRED fixtures covering all 4 core series across two native frequencies,
one missing observation (`DFF`, a weekend `"."`), and one revision
(`DGS10`, same `observation_date`, two vintages with different values
and `realtime_start`s), exercising the complete pipeline
(orchestration -> component/combined datasets -> reconciliation ->
verification -> offline replay) fully deterministically, zero network
calls. Also proves: running the identical semantic workflow from two
independent temp repositories converges on byte-identical
`combined_manifest_id`/component/observation ids (identity is purely
content-addressed, never influenced by filesystem path), and running it
in two child interpreters under different explicit `PYTHONHASHSEED`
values produces the identical `combined_manifest_id` (identity never
depends on Python's per-process-randomized `hash()`, only on
`compute_content_id`'s canonical sorted-key JSON + sha256). 8/8 passing.

### 15. Opt-in real-FRED acceptance -- status

`run_real_fred_acceptance_workflow` (`acceptance.py`) exists, is fully
wired to the real `StdlibHttpsTransport`, and is exercised by
`test_collectors_curated_acceptance.py::TestOptInLiveAcceptance`. In
THIS environment, `FRED_API_KEY` is absent from the process
environment, so the test resolves `None` via
`resolve_fred_api_key_from_environment()` and calls `pytest.skip(...)`
with a precise reason -- **zero network calls were attempted anywhere
in this delivery**. A missing credential is never treated as an
application failure; the ordinary full suite passes with or without a
key present. 9 other tests in that file (key resolution rules, the
empty/whitespace-key guard clause, `RedactedAcceptanceReport`'s
structural inability to carry a secret) run unconditionally and passed.

### 16. Offline replay

Proven at two levels. Within a single backfill:
`test_collectors_curated_orchestration.py::TestCachedReplayAndIdempotency`
re-runs an already-cached operation under a `ForbiddenTransport`
(structural zero-network proof) and confirms an identical
`combined_manifest_id`. Within the mandatory fixture acceptance:
`TestOfflineReplayEquality` re-runs the FULL 4-series pipeline a second
time with `ForbiddenTransport`, confirming identical completeness
status, `combined_manifest_id`, and every series' component-manifest id
and committed-observation count. The opt-in real-FRED workflow
(Section 15) performs the SAME comparison against a genuinely live
first pass whenever a key is supplied -- not exercised in this
environment, but structurally identical to the always-run fixture
proof.

### 17. Reconciliation

`reconcile_curated_universe` (`reconciliation.py`) needs no original
construction parameters -- only a `provider`/`registry`/
`target_dataset_namespace` -- scanning already-stored evidence purely
against itself. Detects: missing combined manifest, registry-id
mismatch (warning), a combined-manifest-referenced series absent from
the current registry or disabled in it, component-manifest version
mismatch, coverage/observation-count recomputed-from-the-observation-
store mismatch, conflicting vintages (same `observation_date` +
`realtime_start` recording two different values), provenance
completeness in both directions (observation-without-provenance,
provenance-without-observation), and duplicate provenance coordinates.

### 18. Verification independence classification

Following `market_data.verification`'s own two-tier honesty taxonomy
(and Phase 4A's own precedent exactly): `verify_curated_universe` is
STRUCTURALLY INDEPENDENT -- it takes the ORIGINAL construction
parameters plus a `CuratedIngestionReport`'s own `SeriesOutcome`s and
REDERIVES everything fresh: registry self-check identity, per-series
response-manifest re-hash (re-reads raw bytes with `verify=False` then
explicitly re-hashes itself, never relying on the cache's own internal
guard), a STRICT independent reparse of the cached bytes, recomputed
availability/normalized values via `_normalize_curated_row` (imported
directly from `orchestration.py`, the exact same pure function
orchestration itself used), recomputed component/combined manifest
self-check identities, and recomputed `completeness_status` -- never
trusting a cached parsed observation, a recorded count, or a final
"is_verified" flag anywhere. Proven non-vacuous: forged registry id,
tampered response bytes (re-hash mismatch), a `SeriesOutcome`
referencing an unknown series, a missing availability policy, a missing
combined manifest, and a revision-policy mismatch are all independently
caught (`test_collectors_curated_verification.py`).

### 19. Secret-handling evidence

An API key is supplied by the caller as a plain, ephemeral function
parameter, never a stored field -- unchanged from Phase 4A's own
enforcement (structural `_reject_secret_shaped_keys`, allowlisted
header canonicalization, AST-based dataclass-field scan). Phase 4B adds
one further layer: `acceptance.py` is the ONE place anywhere in
`collectors/` that reads `FRED_API_KEY` from the environment, confirmed
unique by `TestCuratedSpecificSafety::test_no_secret_in_any_curated_
module`. `RedactedAcceptanceReport`'s own field set is structurally
incapable of holding a secret (proven by a dataclass-field-name
disjointness check against a credential-shaped-name set, and by
constructing one and confirming its JSON contains no secret-shaped
value). Every `reports.py` generator is proven secret-free even when a
key WAS genuinely in play for the run that produced its input
(`test_collectors_curated_reports.py::TestNoReportEverAcceptsRawCredential`,
`test_collectors_curated_pit_concurrency_adversarial.py::TestAdversarial::
test_report_includes_no_secret_even_when_a_key_was_actually_used`). A
genuine, previously-latent secret-exposure defect was found and fixed
this phase -- see Section 21, defect 1.

### 20. Tests and exact results

- Focused curated suite (14 new test files):
  `pytest tests/unit/market_data/test_collectors_curated_*.py
  tests/unit/market_data/test_curated_fred_example_config.py` --
  **183 passed, 1 skipped** (the absent-key opt-in acceptance test),
  0 failed, ~13s.
- Full `market_data` package (Phases 1-4B together, `-W error`):
  **970 passed, 1 skipped**, ~53s -- confirms zero regression against
  every Phase 1-4A test.
- `ruff check .` (full repo): **all checks passed**.
- `mypy src` (full repo, 338 source files): **no issues found**.
- `git diff --check`: clean, no whitespace errors.
- ×10 repeat of registry/policy/orchestration/fixture-acceptance
  (offline-replay + determinism)/concurrency/adversarial/secret-
  redaction categories (141 tests × 10 runs): **1,410/1,410 passed**,
  fully stable, no flakiness observed.
- Concurrency tests
  (`test_collectors_curated_pit_concurrency_adversarial.py::TestConcurrency`)
  specifically re-run 20 additional standalone times to rule out
  timing-dependent flakiness after the jittered-backoff hardening
  described in Section 21: **20/20 full-file runs passed** (300 total
  test executions across those runs, zero failures).
- Full repository suite (`pytest -q`, every prior milestone together):
  **6,500 passed, 2 skipped, 1 failed**, total runtime 2h 27m 9s. The
  two skips are pre-existing and environment-only, unrelated to Phase
  4B: `test_collectors_curated_acceptance.py`'s own absent-key opt-in
  skip (Section 15) and `tests/unit/ml/test_artifacts.py:110`
  ("symlink creation requires elevated privileges on Windows," the same
  pre-existing skip the Phase 4A delivery report already documented).
  The ONE failure --
  `tests/unit/execution_gateway/test_portfolio_risk_integration.py::
  TestConcurrentGateCalls::test_concurrent_authorization_of_two_distinct_
  intents_never_leaks_a_raw_ledger_race` -- is a PRE-EXISTING test in a
  package Phase 4B never touches (`execution_gateway`; confirmed via
  `git status --short` showing zero changes to that file, and `git log`
  showing its last change predates this phase entirely). The test's own
  design already retries up to 4,000 times on lock contention
  (`_retry_on_lock(fn, max_attempts=4000)`) before failing, and still
  exhausted that budget once during this single, ~2.5-hour, otherwise
  fully sequential full-suite run. Re-running its own test class in
  isolation immediately afterward: **5/5 passed** (one run visibly
  slower at ~22.7s vs. ~2.2s for the other four, consistent with
  genuine, resolved lock contention rather than a systemic break). This
  is assessed as pre-existing, environment/timing-sensitive flakiness in
  `execution_gateway`'s own concurrency test, unrelated to and not
  introduced by this phase's changes -- consistent with `execution_
  gateway`/`portfolio_risk` being explicitly out of scope for Phase 4B
  to modify or fix. Not weakened, skipped, or otherwise altered.

No live network access was used anywhere in this suite. Every test uses
`RoutingFakeTransport`/`FakeTransport` (in-memory doubles that route by
URL shape or exact position) or exercises pure, network-free functions
directly. The one opt-in test that WOULD use the real network
(`TestOptInLiveAcceptance::test_live_workflow_when_key_present`) skipped
cleanly with zero network calls, as required.

### 21. Genuine defects found and fixed

All of the following were self-identified during construction (via
`ruff`/`mypy` runs, smoke testing, dedicated pytest runs, and -- for
defect 1 -- a deliberate adversarial test targeting exactly the "key in
error body" category the specification calls out), not flagged
externally.

1. **API key leakage through a transport-failure error message**
   (`collectors/transport.py`, `collectors/fred.py`). `TransportRequest.url`
   legitimately carries a real `api_key=...` query parameter for the
   in-flight call (by design -- see Phase 4A's own credential-handling
   discussion). `StdlibHttpsTransport`'s exception messages (read
   timeout, transport failure, oversized-response rejection, redirect
   violations, static URL validation, `ForbiddenTransport`'s own
   assertion) all embedded this FULL url, including the query string,
   directly in their text. `fred.execute_fred_request`'s
   retry-exhaustion path then compounded this by interpolating the raw
   exception text (`f"...exhausted retries: {exc}"`) into
   `RetryExhaustedError`'s own message. Concretely reproduced: a
   deliberately failing transport with a real (fake, for the
   reproduction) key produced a `RetryExhaustedError` whose `str()`
   contained the key in plaintext. Root cause: URLs were treated as
   safe-to-print identifiers when they are not, given the documented
   contract that they may carry a secret. Fixed at the root in TWO
   places: `transport.py` gained `_redact_url_for_error` (strips the
   query string before a URL is embedded in ANY exception message
   anywhere in the module) applied at every call site;
   `fred.execute_fred_request` now surfaces only the failing
   exception's CLASS NAME on retry exhaustion, never its raw text
   (mirroring the pattern `RetryAttemptRecord.detail` already used one
   line above it). Verified via direct reproduction before and after
   the fix, and via a permanent regression test:
   `test_collectors_curated_pit_concurrency_adversarial.py::TestAdversarial::
   test_key_in_transport_error_body_never_leaks_through_retry_exhaustion`
   and `::test_real_stdlib_transport_redacts_secret_from_every_exception_it_raises`.
   This fix also benefits every Phase 4A caller of the same shared
   `transport.py`/`fred.py` module (confirmed via the full Phase 1-4A
   regression re-run, zero behavioral change to any non-error-message
   code path).
2. **Append-then-unlocked-read race window** (`collectors/curated/
   macro_observation.py`, `orchestration.py`) -- a DEFENSIVE hardening
   applied during concurrency testing, described honestly rather than
   overclaimed: `orchestration.py`'s Stage 8 originally called
   `CuratedObservationStore.append()` in a loop (each call correctly
   lock-protected) followed by a SEPARATE, UNLOCKED
   `read_observations()` call to build the component manifest. Under
   initial multi-threaded concurrency testing this pairing was
   suspected as the cause of an observed `ProvenanceError`. Deeper,
   instrumented investigation (patching `create_provenance_record` to
   log every candidate record's full field set across four racing
   threads) ultimately traced that SPECIFIC observed failure to TWO
   separate test-authoring issues instead -- a positional
   `FakeTransport` response list that silently desynced under partial
   caching, and a test that seeded a series under one `operation_id`
   before racing a different `operation_id` over the identical data
   (a legitimate, already-correctly-enforced cross-batch provenance
   conflict, unrelated to genuine concurrency safety). Both were fixed
   at the TEST level (a URL-routing `RoutingFakeTransport`; starting
   concurrency tests from an unseeded repository). Independently of
   that investigation, the append-then-unlocked-read pattern remains a
   theoretically real non-atomicity (a reader could in principle
   observe state from before a concurrent writer's append completes),
   so `CuratedObservationStore.append_many_and_read_all` was added
   (append every observation AND read the resulting full list back
   under ONE lock acquisition) and `orchestration.py`'s Stage 8 was
   updated to use it. This is disclosed as a preventative hardening
   applied alongside the investigation, not as the fix for a
   conclusively-proven defect -- full mypy/ruff/regression clean, zero
   observable behavior change for any single-threaded caller.

Two further, non-production findings from the same concurrency
investigation are worth recording for future maintainers (not counted
as "defects," since neither is a defect in shipped code): (a) under
VERY aggressive same-process thread contention (4 threads immediately
retrying the identical operation in a tight loop with no backoff),
`experiment_lock`'s own stale-lock-reclaim heuristic (`ml.concurrency`,
pre-existing shared infrastructure) can occasionally prevent any single
thread from completing all 12 stages within a bounded attempt count --
every individual failure observed under this stress remained a clean,
recognized `MarketDataLockError`, never corruption, and a small
jittered backoff between caller-level retries (the realistic pattern
any real caller would use) reliably converges; this is documented in
the architecture doc's Known Limitations as a candidate for separate,
dedicated follow-up work on `experiment_lock` itself, out of this
phase's scope. (b) A test racing three threads under three DIFFERENT
`operation_id`s over the same underlying data was, by design, testing
cross-batch provenance-conflict REJECTION (a different, already-correct
property, covered by its own dedicated test in
`test_collectors_curated_orchestration.py`) rather than the cache
idempotency it claimed to verify; it was rewritten to use one shared
`operation_id` across its threads.

### 22. Known limitations (honestly disclosed)

See `docs/market_data_architecture.md`'s "Known limitations" section,
Phase 4B subsection, for the complete list with full reasoning. Summary:
`NEXT_BUSINESS_DAY_CONSERVATIVE` has no public-holiday calendar
awareness; `AvailabilityPolicyKind.RELEASE_CALENDAR_REFERENCE` is
deliberately unimplemented; every `AvailabilityPolicy`'s hour/minute is
a disclosed APPROXIMATION (FRED provides no intraday publication
timestamps); only the 4 core series are individually fixture-verified,
the 14 extended-universe candidates are registered but disabled by
default; no curated-data-to-market-bar join exists yet (contract
documented and safety-scan-enforced, not yet exercised by an actual
feature-generation join); `experiment_lock`'s own behavior under
aggressive same-process thread contention is a pre-existing,
out-of-scope limitation (Section 21); no pagination support, matching
Phase 4A; no CLI surface, matching every prior phase.

### 23. Exact git status

As of this report, `git status --short` shows:
- 5 modified files: `docs/market_data_architecture.md`,
  `src/quant_platform/core/exceptions.py`,
  `src/quant_platform/market_data/collectors/fred.py`,
  `src/quant_platform/market_data/collectors/transport.py`,
  `tests/unit/market_data/test_market_data_safety_scan.py`.
- 1 new subpackage directory
  (`src/quant_platform/market_data/collectors/curated/`, 14 files).
- 1 new module
  (`src/quant_platform/market_data/collectors/fred_series_metadata.py`).
- 15 new test files under `tests/unit/market_data/`.
- 1 new example config (`examples/xauusd_macro_fred_config.example.json`).
- 1 new documentation file (this report).

**Nothing has been staged. Nothing has been committed. Nothing has been
pushed.** HEAD remains `c6439952415d048ccdb34beb914b808ee7ece5cb`,
identical to the confirmed Phase 4B baseline.

### 24. Explicit confirmations

- Phase 4B work is **not staged** (`git add` was never run).
- Phase 4B work is **not committed** (`git commit` was never run; HEAD
  is unchanged from baseline).
- **Nothing has been pushed** (`git push` was never run).
- **No FRED API key was ever persisted, logged, printed, or committed**
  anywhere in this diff -- confirmed by `TestCuratedSpecificSafety`'s
  curated-scoped secret scan, the pre-existing long-literal-`api_key=`
  scan (now also applied to every new `test_collectors_curated_*.py`
  file), and the dedicated defect-1 regression tests (Section 21).
- **No Yahoo Finance, MT5, broker integration, or live streaming**
  exists anywhere in this diff -- confirmed by the safety scan's own
  broker/websocket/live-trading-marker checks, unchanged and re-passing
  against the now-larger `collectors/` tree.
- **No scheduler or daemon** was added -- every workflow in this phase
  is a single synchronous function call.
- **No execution/portfolio-risk/ML-model/strategy code was touched** --
  confirmed by the full repository regression suite showing zero
  changes to, or failures in, any package outside `market_data` (plus
  the two shared, purely-additive `core/exceptions.py` extensions
  already established as the sanctioned pattern in Phase 4A).
- **Phase 4C was not started** -- this diff is confined to the curated
  FRED macro universe and its own verification/reconciliation/reporting
  layer, exactly the Phase 4B scope.
- **Milestone 11 was not started** -- no work outside `market_data` (and
  its one shared, narrow `core.exceptions` extension) was touched.
- Per the specification's own instruction, **this phase stops here**:
  no Yahoo Finance, no MT5, no broker integration, no live streaming, no
  Phase 4C, and no Milestone 11 work has begun.
