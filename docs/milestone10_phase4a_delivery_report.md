# Milestone 10, Phase 4A -- Delivery Report

## Secure External Historical Collector Infrastructure and FRED Integration

### 1. Baseline commit

`0e50b02c3c57f22e3c7943ca63e6ef2c839782f9` -- "Add deterministic offline
historical ingestion pipeline" (the commit that concluded Milestone 10
Phases 1-3). Confirmed at the start of this phase: `git rev-parse HEAD`
matched this commit exactly, and `git status --short` was clean (no
staged, unstaged, or untracked changes). No other commit has been made
since; this remains HEAD as of this report.

### 2. Files added / modified

**New package, `src/quant_platform/market_data/collectors/`** (15
files, ~3,015 lines):

| File | Purpose |
|---|---|
| `__init__.py` | Package docstring: the isolated, network-capable boundary and required flow. |
| `protocols.py` | `HistoricalHttpTransport` Protocol, `TransportRequest`/`TransportResponse`. |
| `transport.py` | `StdlibHttpsTransport` (SSRF-hardened, stdlib-only), `ForbiddenTransport`. |
| `retry.py` | `RetryPolicy`, `classify_failure`, `plan_next_wait_seconds`, `parse_retry_after`. |
| `rate_limit.py` | Pure immutable token-bucket rate limiting. |
| `request_manifest.py` | `CollectorRequestManifest`, structural secret-shaped-key rejection. |
| `response_manifest.py` | `CollectorResponseManifest`, allowlisted header canonicalization. |
| `cache.py` | `RawResponseCache` -- content-addressed raw response storage. |
| `fred_schemas.py` | `FredObservation`, strict JSON/CSV parsing. |
| `macro_normalization.py` | `UnitMappingSpec`, `normalize_macro_row`, point-in-time `event_time` derivation. |
| `fred.py` | `execute_fred_request` (the attempt loop), `FredSourceAdapter`. |
| `orchestration.py` | `CollectorOperationStage`/`CollectorOperationStore`, `run_fred_macro_ingestion_operation`. |
| `verification.py` | `verify_fred_macro_operation`, `verify_secret_absence`. |
| `reconciliation.py` | `reconcile_fred_macro_dataset`. |
| `reports.py` | Deterministic, secret-free report generators. |

**New test files, `tests/unit/market_data/`** (17 files, ~2,933 lines,
271 tests):
`_collectors_test_helpers.py` (shared `FakeTransport` double, not a
`conftest.py` -- this repository has none anywhere under `tests/`),
`test_collectors_transport_security.py`, `test_collectors_secrets.py`,
`test_collectors_identity.py`, `test_collectors_cache.py`,
`test_collectors_retry.py`, `test_collectors_rate_limit.py`,
`test_collectors_fred_parsing.py`, `test_collectors_macro_normalization.py`,
`test_collectors_fred.py`, `test_collectors_orchestration.py`,
`test_collectors_replay.py`, `test_collectors_reconciliation.py`,
`test_collectors_verification.py`, `test_collectors_reports.py`,
`test_collectors_concurrency.py`, `test_collectors_adversarial.py`.

**Modified files** (narrow, additive; each verified with `ruff --fix` +
`mypy` clean + Phase 1-3 regression tests still passing after every
change):
- `src/quant_platform/core/exceptions.py` -- 19 new `CollectorError`-family
  exceptions appended.
- `src/quant_platform/market_data/manifests.py` -- `DatasetKind.
  MACRO_OBSERVATIONS` added; `DatasetKey.__post_init__`/
  `storage_path_parts()` extended.
- `src/quant_platform/market_data/source_manifests.py` -- `RecordKind.
  MACRO_OBSERVATION`, `SourceKind.FRED_API` added.
- `src/quant_platform/market_data/__init__.py` -- module docstring
  updated to disclose the one, explicitly isolated `collectors`
  exception to "never opens a network connection."
- `src/quant_platform/market_data/quarantine.py` -- `MISSING_OBSERVATION_VALUE`
  added (18th issue code), classified `PERMANENT`.
- `src/quant_platform/market_data/orchestration.py` -- one stale-comment
  fix (hardcoded "17-code vocabulary" -> "Phase 3 code vocabulary").
- `docs/market_data_architecture.md` -- Status line, new Phase 4A
  section, Phase 4A known-limitations subsection, Future-phases update.
- `tests/unit/market_data/test_market_data_quarantine.py` -- issue-code
  count assertion updated 17 -> 18.
- `tests/unit/market_data/test_market_data_safety_scan.py` -- recursive
  glob, two narrowed checks, new `TestCollectorSpecificSafety` class
  (AST-based credential-field scan, third-party-import ban, SSRF
  non-vacuity, long-literal-API-key scan).

### 3. Collector boundary

`src/quant_platform/market_data/collectors/` is the ONLY subpackage in
this repository that opens a network connection. Nothing outside it
imports it implicitly; every other `market_data` module remains exactly
as network-free as Phases 1-3 left it (confirmed by the safety scan's
own recursive, whole-tree checks continuing to pass for every
non-collector file). The collector layer never writes a `MarketEvent`/
`MacroEvent` directly into the repository -- the required flow (remote
request -> request manifest -> raw bytes -> response manifest -> source
manifest -> strict adapter -> normalize/validate/quarantine -> durable
repository) is enforced structurally: `FredSourceAdapter` can only be
constructed via `load_fred_adapter_from_cache` (reads exclusively from
`RawResponseCache`), and `orchestration.run_fred_macro_ingestion_operation`
is the only function that ever calls `macro.MacroEventStore.append`.

### 4. Transport protocol

`protocols.HistoricalHttpTransport` is a structural `Protocol` (`.get(
request) -> response`); every collector-level function depends on this
shape, never a concrete library. `TransportRequest` mandates HTTPS GET
only, an explicit non-empty `allowed_hosts`, positive connect/read
timeouts, a non-negative `max_redirects`, and caller-supplied
`request_time` (the transport itself never reads the wall clock). One
concrete implementation, `StdlibHttpsTransport`, uses only
`http.client`/`ssl`/`socket`/`ipaddress` from the standard library --
confirmed no third-party HTTP/WebSocket library is an existing
repository dependency (`pyproject.toml` inspected directly), so none was
added. `ForbiddenTransport` raises `AssertionError` immediately on
`.get()`, structurally proving a code path makes zero network calls.

### 5. SSRF and URL security model

Two-phase defense, detailed in `docs/market_data_architecture.md`'s
Phase 4A section:
1. `_validate_url_static` -- pure, pre-DNS: HTTPS-only scheme, no
   userinfo, host present, host on an EXPLICIT caller-supplied
   allowlist (case-insensitive), host not an IP literal (rejected
   outright regardless of allowlist).
2. `_resolve_and_validate_address` -- resolves the allowlisted hostname
   via `socket.getaddrinfo`, validates every candidate's ACTUAL resolved
   IP is globally routable, connects DIRECTLY to the validated IP
   (never re-resolving internally) -- closing the DNS-rebinding
   time-of-check/time-of-use gap.

Redirects are never followed unless explicitly enabled; every hop
re-validates in full against its own target; a scheme downgrade fails
closed; `max_redirects` is enforced. Responses are read incrementally,
bounded by `max_response_bytes`; a non-identity `Content-Encoding` is
rejected outright (decompression-bomb defense). `file://`/`ftp://`/
`data://`/custom schemes are all rejected by the single HTTPS-only
scheme check. `FRED_ALLOWED_HOSTS = frozenset({"api.stlouisfed.org"})`
is the sole permitted destination this milestone ever configures.
Residual, honestly-disclosed limitation: a compromised, already-validated
IP's own OS/network-level routing is outside this layer's control (see
Known Non-Blocking Limitations, and `test_collectors_adversarial.
py::TestDnsRebindingAssumptionsAreDocumented`, which asserts this stays
documented).

### 6. Secret-handling guarantees

An API key is supplied by the caller as a plain, ephemeral function
parameter (`execute_fred_request(..., api_key: str | None, ...)`) --
never a stored field. Enforced three independent ways: (a)
`request_manifest.py`'s structural `_reject_secret_shaped_keys`
blocklist on canonical query params/headers, raising
`SecretExposureError`; (b) `response_manifest.py`'s ALLOWLIST-based
header canonicalization (only `content-type`/`content-length`/`date`/
`last-modified`/`etag` ever survive), plus an independent blocklist as a
second layer; (c) an AST-based safety-scan check confirming no
`@dataclass` field anywhere in `collectors/` is ever credential-shaped
(structurally distinct from a function parameter, which lives in
`FunctionDef.args`, never a class body). A manifest records only
`credential_mode: "anonymous" | "api_key"` -- never a secret value or a
secret-derived digest. Proven, not merely asserted: `test_collectors_
secrets.py`, `test_collectors_adversarial.py` (secret reflected in an
error body / a rejected redirect's Location never reaches an exception
message), and the safety scan's own long-literal-`api_key=` scan
(applied to both `collectors/` source and the test suite's own
fixtures).

### 7. Request identity

`CollectorRequestManifest.request_manifest_id` is content-addressed
(`compute_content_id` over `to_identity_payload()`), excluding only
`request_time` (operational). Same semantic request -> same id
regardless of query-param insertion order, filesystem root, or dict
iteration order (`test_collectors_identity.py`, `test_collectors_
replay.py::TestIdentityIsInsensitiveToDictInsertionOrder`). Changing the
interval, retry-policy id, rate-limit-policy id, timeout-policy id,
credential mode, series, or response format all change the id; the
secret value itself never influences it (the manifest never sees it).

### 8. Response identity

`CollectorResponseManifest.response_manifest_id` excludes only
`received_time`/`transport_attempt_count`. Same bytes -> same id;
changed bytes, changed request linkage, or changed HTTP status all
change it; the storage path never influences it. A "forged digest"
cannot be constructed through the public API at all --
`create_response_manifest` always recomputes the digest from
`raw_bytes` itself; there is no parameter through which a caller could
supply a mismatched one (`test_collectors_identity.py`). Truncated
content produces both a different digest and a different `byte_length`
than the full response.

### 9. Raw cache model

`RawResponseCache` is content-addressed
(`{response_manifest_id}/manifest.json` + `body.bin`), atomic write
(`_atomic_write_bytes`), idempotent for an exact retry, and raises
`CacheCorruptionError` for a conflicting write under the same identity
-- never silently overwritten. Every response/request manifest id used
as a path component is validated as exactly 64 lowercase hex characters
before touching the filesystem (`MarketDataPathSecurityError` on
anything else) -- path traversal is structurally unreachable. Re-hashes
on every read by default. A separate, append-only per-request index
explicitly distinguishes semantic request identity (stable) from actual
response identity (changes with content), modeling "the same semantic
request may legitimately receive different response content at a later
`request_time`" directly (`test_collectors_cache.py::TestRequestIndexAndOfflineReplay`).

### 10. Retry policy

`classify_failure`/`plan_next_wait_seconds`/`parse_retry_after` are pure
-- no sleep, no wall-clock read anywhere in `retry.py`. Connection/read
timeouts, HTTP 408/429, and configured 5xx statuses are retryable;
400/401/403 are structurally never retryable (`RetryPolicy.__post_init__`
refuses to even construct a policy that tries to make one retryable).
`Retry-After` is parsed strictly (digit-seconds or RFC 7231 HTTP-date;
anything else fails closed to `None`) and takes precedence over the
configured backoff schedule when present and respected. The attempt
LOOP lives in `fred.execute_fred_request` with a fully injectable
`sleep_fn` (default `time.sleep`) -- every retry-sequence test in this
suite runs in well under a second with zero real sleeping
(`test_collectors_fred.py::TestExecuteFredRequestRetry::test_no_real_sleep_ever_happens_in_this_test`).

### 11. Rate limiting

Pure, immutable token-bucket model (`rate_limit.py`) -- every function
returns a NEW `TokenBucketState`; the caller supplies `now` explicitly;
no global mutable singleton. `TokenBucketState` has no
`to_identity_payload` at all -- rate-limit state structurally cannot
influence any semantic dataset/request/response identity. Unit-testable
without sleeping (`test_collectors_rate_limit.py`); safe under
concurrent read access by construction (immutability).

### 12. FRED request semantics

`build_fred_request_manifest` accepts any `series_id` --
`FRED_EXAMPLE_SERIES` (`DFII10`, `DGS10`, `CPIAUCSL`, `DFF`) is
documented as examples only, never enforced as an allowlist.
Deterministic query generation (`series_id`, `file_type`,
`observation_start`/`observation_end` when supplied); official FRED
`/fred/series/observations` semantics only, no HTML scraping. No
pagination is implemented this phase (see Known Non-Blocking
Limitations) -- the request/response manifest schemas both reserve
pagination fields for a future phase without a breaking change.

### 13. FRED raw data model

`FredObservation` (`series_id`, `row_index`, `observation_date`,
`value_text`, `realtime_start`, `realtime_end`) is entirely TEXT --
`parse_fred_json_response` rejects a JSON NUMBER for `date`/`value`
outright (never a float intermediate); `is_missing_value` checks FRED's
own `"."` convention explicitly. Strict mode: required keys enforced,
undeclared keys rejected. CSV parsing targets FRED's documented
two-column `DATE,<SERIES>` export shape (a disclosed assumption, never
exercised against a live response this phase).

### 14. Macro normalization

`UnitMappingSpec` mirrors `mappings.py`'s versioned, content-addressed
pattern. `normalize_macro_row` is pure, Decimal-only (never float),
operates on generic `raw_fields: dict[str, str]` (adapter-agnostic).
Missing values (`"."`) are quarantined under the new
`MISSING_OBSERVATION_VALUE` code, distinct from `INVALID_DECIMAL`
(genuinely malformed, present text) -- never silently coerced to zero.
Point-in-time safety: `event_time` is derived from FRED's own
`realtime_start` (the vintage/publication proxy), never `date` (the
observation period) -- using `date` would be exactly the look-ahead
bias `PointInTimeViolationError` exists to catch. Both parsed as UTC
midnight. The observation period itself is preserved in
`source_event_id`, so a monthly CPI observation's monthly meaning
survives. A daily series is never falsely given an intraday timestamp
(`test_collectors_macro_normalization.py::test_daily_series_never_falsely_given_intraday_timestamp`).
Revised values are represented through source/vintage identity (a
different `realtime_start` -> a different `event_time` -> a distinct,
non-conflicting `MacroEvent`).

### 15. Backfill orchestration

`CollectorOperationStage`/`CollectorOperationStore` (`orchestration.py`)
implement the exact required 11 stages: `REQUEST_PLANNED ->
REQUEST_MANIFEST_COMMITTED -> RESPONSE_DOWNLOADED ->
RAW_RESPONSE_COMMITTED -> RESPONSE_VERIFIED -> SOURCE_MANIFEST_CREATED
-> SOURCE_PARSED -> NORMALIZED_RECORDS_PRODUCED ->
REPOSITORY_INGESTION_COMMITTED -> PROVENANCE_COMMITTED ->
VERIFICATION_COMPLETED`. Deliberately a SEPARATE, self-contained
implementation from Phase 3's own `OperationStore`/`IngestionStage`
(see Section 21 and the architecture doc for the reasoning), duplicating
the same proven idempotent/conflict/monotonic-progression algorithm
rather than widening Phase 3's own shipped code's blast radius.
`FetchMode.FRESH`/`FetchMode.CACHED_REPLAY` model "exact retry reuses
verified cached response" vs. "fresh-network policy may produce a new
response version" explicitly -- `fetch_mode` is excluded from the
operation's own content-digest identity (it describes HOW, not WHAT),
so a retry can freely switch modes and stay recognized as the same
operation, while genuinely different response bytes are still caught
one level down as a per-stage evidence conflict. A pre-flight
provenance-conflict check runs before any repository write, preventing
an orphaned `MacroEvent` (mirroring Phase 3's own established
defect-prevention pattern).

### 16. Offline replay

`FetchMode.CACHED_REPLAY` performs zero network calls, proven
structurally (`ForbiddenTransport` injected and never touched -- 
`test_collectors_orchestration.py`, `test_collectors_replay.py`).
Given only the persisted request manifest, response manifest, raw
bytes, and mappings, replay reproduces identical request/response/
source-manifest ids, identical normalized-events digest, identical
`MacroEvent`/provenance record ids -- confirmed across two ENTIRELY
SEPARATE temp roots (`test_collectors_replay.py::TestIdenticalIdsAndRecordsAcrossDifferentTempRoots`)
and across different dict-insertion orderings, the practical in-process
analogue of "different PYTHONHASHSEED" (identity always goes through
sorted-key canonical JSON, so it cannot depend on hash-seed-driven
iteration order by construction).

### 17. Recovery

The same `CollectorOperationStore.advance` idempotency that makes exact
retry safe also makes recovery-from-interruption safe: re-running an
operation that crashed after any stage recognizes its own partial
history (matching `content_digest` and matching per-stage evidence) and
continues forward rather than restarting or double-recording
(`test_collectors_orchestration.py::TestInterruptionAndRecovery`).
Recovery from the TERMINAL stage itself (a fully-completed operation,
re-run again) is exercised as the strongest form of this property. No
stage is ever marked complete before the one before it (`advance`
structurally rejects out-of-order/skipped stages).

### 18. Reconciliation

`reconcile_fred_macro_dataset` needs no original construction
parameters -- only `provider`/`series_id` -- scanning already-stored
evidence purely against itself. Detects: missing raw response, digest
mismatch, truncated payload, unexpected content type, wrong
request/response linkage, missing/duplicate observation coordinate, a
repository record with no matching provenance, provenance for an event
absent from the repository, a series ingested under two different unit
mappings, conflicting vintages, and stalled operations (this phase's
honest analogue of "stale checkpoint" -- see Section 22).

### 19. Verification independence classification

Following `market_data.verification`'s own established two-tier
honesty taxonomy: `verify_fred_macro_operation` is STRUCTURALLY
INDEPENDENT in the strongest sense this phase offers -- it does not
merely recompute an id from a stored payload (which a coherently-forged
tamper could survive), it REDERIVES the request manifest, source
manifest, and every normalized observation FRESH from caller-declared
collector parameters and the raw cached bytes, using the exact same
pure functions orchestration used, but invoked completely
independently. It re-hashes raw bytes explicitly (not relying on the
cache's own internal `verify=True` guard) and strictly reparses via
`fred_schemas.parse_fred_*_response` directly -- never `load_fred_
adapter_from_cache` (which exists to serve orchestration, not
verification) and never a cached `FredSourceAdapter`. `verify_secret_
absence` is proven non-vacuous against both a deliberately leaking fake
artifact and confirmed clean against real manifests.

### 20. Tests and exact results

- Focused collector suite: `pytest tests/unit/market_data/test_collectors_*.py`
  -- **271 passed**, 0 failed, 0 skipped, 0 warnings, ~3.7s.
- Full `market_data` package (Phases 1-4A together, `-W error`):
  **782 passed**, 0 failed, ~40s -- confirms zero regression against
  every Phase 1-3 test.
- Full repository suite (`pytest -q -m "not performance"`, the entire
  repository across every prior milestone): **6256 passed, 1 skipped,
  57 deselected, 0 failed**, total runtime 2h 26m 30s. The single skip
  (`tests/unit/ml/test_artifacts.py:110`, "symlink creation requires
  elevated privileges on Windows") is a pre-existing, environment-only
  skip unrelated to Phase 4A; the 57 deselected tests are the
  `performance` marker's own explicit exclusion. Zero failures anywhere
  in the repository -- confirms Phase 4A introduced no regression in
  any other package (`portfolio_risk`, `execution_gateway`, `ml`,
  `backtesting`, etc.), consistent with every Phase 4A change being
  additive/narrow and confined to `market_data` plus one shared,
  purely-additive extension of `core/exceptions.py`.
- `ruff check .` (full repo): **all checks passed**.
- `mypy src` (full repo, 323 source files): **no issues found**.
- `git diff --check`: clean, no whitespace errors.
- ×10 repeat of transport-security/retry/replay/orchestration/concurrency
  categories (99 tests × 10 runs): **990/990 passed**, fully stable, no
  flakiness observed.
- Concurrency tests (`test_collectors_concurrency.py`) specifically
  re-run 5 additional times standalone to rule out timing-dependent
  flakiness: **15/15 passed** across those runs.

No live network access was used anywhere in this suite. Every test uses
`FakeTransport` (an in-memory double) or exercises pure, network-free
functions directly (`_validate_url_static`, `_is_globally_routable`,
etc.). The one exception -- `TestConnectTimeoutMapping` -- connects to
an intentionally-unused LOOPBACK port to get a fast, real
`ConnectionRefusedError`, never leaving the local machine.

### 21. Genuine defects found and fixed

All of the following were self-identified during construction (via
`ruff`/`mypy` runs, smoke testing, and dedicated pytest runs), not
flagged externally -- consistent with this repository's established
"test everything, catch real defects during construction" discipline.

1. **Operation-identity `fetch_mode` collision** (`orchestration.py`):
   originally included `fetch_mode` in the operation's content-digest
   identity payload, which meant an exact retry of the same
   `operation_id` that switched from `FetchMode.FRESH` to `FetchMode.
   CACHED_REPLAY` was wrongly treated as a DIFFERENT operation
   (`CollectorOrchestrationConflictError`), breaking the spec's own
   "exact retry must reuse verified cached response when explicitly
   requested" requirement. Root cause: conflating HOW an operation
   executes with WHAT it semantically is. Fixed by excluding
   `fetch_mode`/`reference_response_manifest_id` from the digest, relying
   instead on the (already-correct) per-stage evidence comparison to
   catch a GENUINELY different response. Regression test:
   `test_collectors_orchestration.py::TestExactRetryIdempotency`.
2. **Dry-run blocked the persist-before-parse invariant**
   (`orchestration.py`): `cache.store()` was originally gated behind
   `if not dry_run`, but Stage 7 (`SOURCE_PARSED`) unconditionally reads
   from the cache via `load_fred_adapter_from_cache` -- so a dry run
   crashed with `MalformedFredResponseError` (no cached manifest) rather
   than producing an honest preview. Root cause: conflating "write
   nothing to durable BUSINESS records" with "never touch the
   content-addressed cache," which are different concerns. Fixed by
   making the cache write unconditional (harmless -- content-addressed,
   idempotent, no semantic weight) while keeping the operation
   ledger/quarantine/provenance/macro-event writes correctly gated.
   Regression test: `test_collectors_orchestration.py::TestDryRun`.
3. **Missing-value/malformed-value conflation** (`macro_normalization.py`):
   the Decimal-parsing branch originally caught any parse failure and
   classified it under `MISSING_OBSERVATION_VALUE`, even for a
   genuinely malformed (but present) value like `"not_a_number"` --
   losing the real distinction between "explicit absence" (never fixed
   by resubmitting) and "malformed presence" (a genuine parse defect).
   Fixed to classify present-but-invalid text as `INVALID_DECIMAL`,
   reserving `MISSING_OBSERVATION_VALUE` exclusively for FRED's `"."`
   convention. Regression test: `test_collectors_macro_normalization.
   py::TestNormalizeMacroRowFailurePaths::test_genuinely_malformed_value_is_invalid_decimal_not_missing`.
4. **Safety-scan false positives from the `collectors/` carve-out**:
   extending the safety scan's recursive glob to reach `collectors/*.py`
   for the first time surfaced FOUR expected-but-unhandled collisions
   against pre-existing whole-tree regexes: (a) `_FORBIDDEN_NETWORK_
   IMPORTS` flagging `transport.py`'s legitimate `socket` import; (b)
   `_CREDENTIAL_FIELD` flagging `api_key` FUNCTION PARAMETERS (not
   fields) in `fred.py`/`orchestration.py`; (c) `_FLOAT_FIELD`/
   `_float_call_lines_outside_prose` flagging legitimate HTTP-timeout
   and retry-backoff DURATION fields/calls (`connect_timeout`,
   `wait_seconds_before_next`, `parse_retry_after`'s `float(stripped)`)
   as if they were financial fields; (d) a bare-word scan for "MT5"
   incorrectly flagging `collectors/__init__.py`'s own docstring PROSE
   ("no broker/MT5/FxPro code"). None were real violations -- each was
   fixed by narrowing the relevant check's SCOPE (excluding `collectors/`
   from the two checks with a genuine, documented carve-out) and adding
   a more PRECISE, collectors-specific replacement (an AST-based
   dataclass-field scan that cannot confuse a parameter with a field;
   an explicit duration-shaped-name allowlist; an import-statement-shaped
   check instead of a bare-word scan). See Section 6 above and the
   architecture doc's "Safety scan" subsection for the full reasoning.

### 22. Known non-blocking limitations

- `StdlibHttpsTransport`'s own wire-level pinned-IP connection and HTTP
  response construction (including the `Content-Encoding` rejection
  path) is not exercised by a dedicated offline unit-test harness --
  doing so would require deep mocking of stdlib `http.client`'s
  internals for limited additional assurance, since the collector layer
  above it is fully protocol-abstracted and thoroughly tested via
  `FakeTransport`, and `StdlibHttpsTransport`'s own SECURITY-CRITICAL
  logic (URL/host/IP validation, redirect rejection, size-bounded
  reading) IS directly unit-tested without this gap.
- No separate `CollectorCheckpoint`/`CollectorCheckpointStore` --
  Phase 3's own `checkpoints.py` is shaped for partitioned, multi-batch
  raw-ingestion backfills, a materially different resumability concern
  than a single per-`operation_id` FRED fetch, which
  `CollectorOperationStore` already makes fully resumable on its own.
  Reconciliation's "stale checkpoint" detection is therefore an
  `stalled_operation` finding (an operation ledger entry that never
  reached `VERIFICATION_COMPLETED`), the honest analogue given this
  scope decision.
- No pagination: the request/response manifest schemas both reserve
  pagination fields for a future phase; nothing in this phase populates
  or consumes them, since `/fred/series/observations` was called with
  an explicit interval per request.
- FRED CSV parsing targets a documented, disclosed assumption about
  FRED's two-column export shape, never verified against a live
  response (this phase performs zero live FRED requests by design).
- No CLI surface: library only, matching every prior phase.

### 23. Exact git status

As of this report, `git status --short` shows:
- 8 modified files (all narrow, additive extensions to Phase 1-3 code --
  see Section 2).
- 1 new package directory (`src/quant_platform/market_data/collectors/`,
  15 files).
- 17 new test files under `tests/unit/market_data/`.
- 2 new documentation files (`docs/market_data_architecture.md` modified
  in place; this report is new).

**Nothing has been staged. Nothing has been committed. Nothing has been
pushed.** HEAD remains `0e50b02c3c57f22e3c7943ca63e6ef2c839782f9`,
identical to the confirmed Phase 4A baseline.

### 24. Explicit confirmations

- Phase 4A work is **not staged** (`git add` was never run).
- Phase 4A work is **not committed** (`git commit` was never run; HEAD
  is unchanged from baseline).
- **Nothing has been pushed** (`git push` was never run).
- **No live-streaming, broker, or execution code** exists anywhere in
  this diff -- confirmed by the safety scan's own broker/websocket/
  live-trading-marker checks, now also applied to `collectors/`.
- **No secrets are stored** anywhere in this diff -- confirmed by
  `SecretExposureError`'s structural enforcement, the AST-based
  dataclass-field scan, and the long-literal-`api_key=` scan applied to
  both source and tests.
- **Yahoo Finance was not implemented** in this subphase -- `collectors/`
  contains exactly one concrete collector, FRED.
- **Milestone 11 was not started** -- no work outside `market_data`
  (and its two required, narrow `core.exceptions` extensions) was
  touched.
- Per the specification's own instruction, **this phase stops here**:
  no Yahoo Finance, no MT5, no broker integration, no live streaming, no
  Phase 4B, and no Milestone 11 work has begun.
