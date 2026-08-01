# Milestone 10, Phase 4C -- Delivery Report

## Provider-Neutral Cross-Asset Historical Market Collectors and Curated XAUUSD Market-Driver Universe

### 1. Baseline commit

`c42df1bcb1b9b87ca7556a9a3e0c29b8e23dfbc8` -- "Add curated point-in-time
FRED macro universe" (the commit that concluded Milestone 10 Phase 4B).
Confirmed at the start of this phase: `git rev-parse HEAD` matched this
commit exactly, `git status --short` was clean, and no downloaded
dataset, API key, credential, secret, cache, temp file, or export was
visible as untracked repository content. No other commit has been made
since; this remains HEAD as of this report.

### 2. Files added / modified

**New subpackage, `src/quant_platform/market_data/collectors/cross_asset/`**
(22 files, ~4,407 lines):

| File | Purpose |
|---|---|
| `__init__.py` | Package docstring. |
| `instrument_form.py` | `InstrumentForm` (8 forms), `ProxyQuality`, `ProxyPolicy`. |
| `adjustment.py` | `AdjustmentPolicyKind` (5 kinds), `AdjustmentPolicy`, `require_equity_like_adjustment`. |
| `sessions.py` | `TimezoneSessionPolicy`, `CandleTimestampConvention`. |
| `futures.py` | `FuturesContractMetadata`, `ContinuationPolicyKind` (6 kinds), `ContinuationPolicy`, `RollProvenance`. |
| `availability.py` | `BarAvailabilityPolicyKind` (3 kinds), `resolve_bar_availability_time` (fail-closed). |
| `registry.py` | `CuratedMarketDriverSpec`, `CuratedMarketDriverRegistry`, 5 core + 5 optional driver specs. |
| `symbol_mapping.py` | `ProviderSymbolMapping`, `SymbolMappingSet`. |
| `market_record.py` | `RawMarketRecord`, `MarketDriverBar`, `MarketDriverBarStore`. |
| `protocols.py` | `HistoricalMarketCollector` (structural `Protocol`), `MarketCollectorCapabilities`, `require_within_capabilities`. |
| `market_normalization.py` | `resolve_bar_open_time`, `normalize_raw_market_record`. |
| `market_backfill.py` | `MarketBackfillSpec`, `create_market_backfill_spec`. |
| `datasets.py` | `ComponentMarketDatasetManifest`/`CombinedCrossAssetManifest` + stores. |
| `gap_policy.py` | `analyze_bar_gaps` -- missing-bar + conflicting-duplicate-coordinate detection. |
| `market_orchestration.py` | `CrossAssetOperationStage`/`CrossAssetOperationStore`, `run_cross_asset_backfill_operation` (~600 lines). |
| `update_plan.py` | `create_cross_asset_update_plan` -- pure incremental planning. |
| `market_reconciliation.py` | `reconcile_cross_asset_universe` + cross-provider conflict model. |
| `market_verification.py` | `verify_cross_asset_universe` -- independent rederivation. |
| `market_reports.py` | Deterministic, secret-free report generators. |
| `acceptance.py` | Opt-in real-Alpha-Vantage acceptance workflow + env-var key resolution. |
| `providers/__init__.py` | Package docstring. |
| `providers/alpha_vantage.py` | `AlphaVantageCollector` -- the ONE concrete provider adapter this phase ships. |

**New shared module,
`src/quant_platform/market_data/collectors/execute_request.py`** --
`CollectorRequestExecution`, `build_transport_request`,
`execute_collector_request`: the fully generic transport/retry/rate-limit
attempt loop, extracted UNCHANGED from Phase 4A's `fred.py` (confirmed
already 100% provider-neutral in implementation before this phase, no
FRED-specific logic anywhere in it). `fred.execute_fred_request` is now
a thin alias.

**New test files, `tests/unit/market_data/`** (12 files, ~3,446 lines,
197 tests, 1 opt-in skip):
`_cross_asset_test_helpers.py` (shared fixtures/doubles, including the
synthetic `FakeMarketCollector`, not a `conftest.py` -- matching this
repository's established convention),
`test_collectors_cross_asset_registry.py` (38),
`test_collectors_cross_asset_policies.py` (27),
`test_collectors_cross_asset_normalization.py` (36),
`test_collectors_cross_asset_alpha_vantage.py` (15),
`test_collectors_cross_asset_orchestration.py` (11),
`test_collectors_cross_asset_datasets_update_plan.py` (19),
`test_collectors_cross_asset_reconciliation_verification_reports.py` (15),
`test_collectors_cross_asset_fixture_acceptance.py` (5),
`test_collectors_cross_asset_acceptance.py` (9 + 1 skip),
`test_collectors_cross_asset_pit_concurrency_adversarial.py` (7),
`test_cross_asset_example_config.py` (14).

**New example config**, `examples/xauusd_cross_asset_config.example.json`
-- no real credentials; references the sanctioned `ALPHA_VANTAGE_API_KEY`
environment variable by NAME only.

**Modified files** (narrow, additive except for one genuine defect fix
described in Section 25; each verified with `ruff --fix` + `mypy` clean
+ full Phase 1-4B regression still passing after every change):
- `src/quant_platform/core/exceptions.py` -- 16 new Phase 4C exceptions
  appended after `UpdatePlanError`: `MarketDriverRegistryError`,
  `ProviderCapabilityError`, `InstrumentFormError`, `SymbolMappingError`,
  `AdjustmentPolicyError`, `SessionPolicyError`, `FuturesContractError`,
  `ContinuationPolicyError`, `MarketAvailabilityPolicyError`,
  `MarketAvailabilityUnresolvedError`, `MarketRecordError`,
  `MarketProviderResponseError`, `MarketBackfillSpecError`,
  `MarketCombinedManifestError`, `MarketUpdatePlanError`,
  `GapPolicyError`. `CollectorOrchestrationStateError`/
  `CollectorOrchestrationConflictError`/`CollectorReconciliationError`/
  `CollectorVerificationError` (Phase 4A) are REUSED directly, matching
  Phase 4B's own precedent.
- `src/quant_platform/market_data/manifests.py` -- new
  `DatasetKind.CROSS_ASSET_MARKET_BARS` (same scoping-only pattern
  `MACRO_OBSERVATIONS` established), extending `DatasetKey.__post_init__`'s
  provider-required tuple and `storage_path_parts()`'s dispatch.
- `src/quant_platform/market_data/source_manifests.py` -- new
  `SourceKind.MARKET_DATA_PROVIDER_API` (deliberately provider-neutral,
  unlike Phase 4A's `FRED_API`) and `RecordKind.MARKET_DRIVER_BAR`.
- `src/quant_platform/market_data/collectors/fred.py` -- major refactor:
  the full attempt-loop implementation (`_build_transport_request`,
  `FredRequestExecution`, `execute_fred_request`) moved to the new
  shared `execute_request.py` (see above); `fred.py` now imports and
  thinly aliases it (`FredRequestExecution = CollectorRequestExecution`;
  `execute_fred_request` calls `execute_collector_request(...,
  allowed_hosts=FRED_ALLOWED_HOSTS, ...)`) -- verified zero behavioral
  change via a full Phase 1-4B regression re-run before and after.
- `src/quant_platform/market_data/collectors/fred_series_metadata.py` --
  one stale docstring reference updated (`fred._build_transport_request`
  -> `execute_request.build_transport_request`).
- `docs/market_data_architecture.md` -- Status line, new Phase 4C
  section, Phase 4C known-limitations subsection, Future-phases update,
  plus two stale docstring-reference prose fixes (Phase 4A/4B sections)
  matching the `fred.py` refactor above.
- `tests/unit/market_data/test_market_data_safety_scan.py` -- new
  `TestCrossAssetSpecificSafety` class (7 tests): cross_asset-subpackage
  reach confirmation, the point-in-time `open_time`-as-availability-proof
  structural check (plus its own non-vacuity and no-false-positive
  proofs), a cross_asset-scoped secret scan, and two behavioral
  (not merely static) structural-guard confirmations (ETF-form-requires-
  proxy, continuous-futures-requires-roll-provenance).

### 3. Provider assessment and selection

Bounded assessment against OFFICIAL documentation or explicitly
documented public interfaces only -- never remembered or guessed
behavior. Three candidates:

- **Stooq**: DISQUALIFIED outright. No official, documented API exists;
  only reverse-engineered CSV endpoints, exactly the "undocumented
  endpoint guessing" this phase's scope forbids.
- **EIA Open Data API** (`https://api.eia.gov/v2/`): officially
  documented. Route structure confirmed LIVE via a real HTTPS request
  that returned a genuine `API_KEY_MISSING` 403 JSON error (proving the
  endpoint shape is real and documented), but no actual data response
  was obtainable without registering a real account -- which this
  phase declines to do autonomously (registering a third-party account
  is a consequential external-identity action requiring the user's own
  involvement, outside what an assistant should do unprompted). Not
  implemented this phase; a candidate for a future phase once a
  credential is available.
- **Alpha Vantage** (`https://www.alphavantage.co/documentation/`):
  officially documented AND live-verified. A real HTTPS `GET` against
  the provider's own public `demo` key against `TIME_SERIES_DAILY`
  (symbol `IBM`) returned the exact JSON envelope this phase's adapter
  parses (`Meta Data` + `Time Series (Daily)` keys, confirmed via
  direct `curl`). The `demo` key was confirmed restricted to a small,
  fixed demo symbol set (`IBM` only, NOT a general free-tier
  credential) -- so per-symbol data for the actual target ETF proxies
  (`GLD`, `UUP`, etc.) was never itself fetched with real data this
  phase, only the endpoint MECHANISM and schema. Free-tier rate limit
  confirmed via the provider's own pricing page: 25 requests/day.
  Alpha Vantage's dedicated commodity endpoints (`WTI`/`BRENT`/
  `GOLD_SILVER_SPOT`) exist in official documentation but could not be
  live-verified the same way (the `demo` key does not cover them).

**Selection: Alpha Vantage, narrowly scoped to ONLY `TIME_SERIES_DAILY`**
-- the one endpoint that was genuinely, live-verified end-to-end. This
is a deliberately conservative, self-imposed narrowing beyond what the
spec strictly requires (which would have permitted implementing the
commodity endpoints too, on documentation alone), justified by "do not
fabricate an integration." Every driver this adapter maps is honestly
classified `is_proxy=True` (ETF proxies only); the adapter never claims
direct spot/futures/index coverage it cannot back with verified
behavior.

### 4. Provider-neutral collector contract

`protocols.HistoricalMarketCollector` is a structural `Protocol`
(mirrors `collectors.protocols.HistoricalHttpTransport`'s own
convention exactly): `provider_metadata()`, `supported_capabilities()`,
`build_metadata_request(...)`, `build_history_request(...)`,
`parse_metadata_response(...)`, `parse_history_response(...)`. No
orchestration-level function anywhere in this phase depends on a
concrete provider client -- confirmed directly by exercising the SAME
`run_cross_asset_backfill_operation` against both the real
`AlphaVantageCollector` and a fully synthetic `FakeMarketCollector` in
the same test suite, including within ONE operation spanning both
providers simultaneously.
`MarketCollectorCapabilities` declares candles/quotes/trades/adjusted/
unadjusted/corporate-actions/futures-contracts/continuous-futures/
pagination/anonymous-access support, credential requirement, max
interval/rows-per-page, supported granularities, and supported
instrument forms. `require_within_capabilities` is the orchestrator's
own fail-closed gate, called BEFORE any request is built -- REJECTS a
request exceeding declared capabilities, never silently downgrading
interval, adjustment mode, or instrument semantics.

### 5. Curated cross-asset driver universe

`registry.py`, mirroring Phase 4B's `curated.registry` keyed-set
pattern exactly. `CuratedMarketDriverRegistry` ALWAYS sorts specs by
`canonical_driver_id` before computing `registry_id` -- identity AND
iteration order are both independent of declaration order (confirmed
directly in tests: building the same 10-spec set in reverse declaration
order produces the identical `registry_id`). Construction rejects:
duplicate ids/names, empty/duplicate `allowed_instrument_forms`,
`preferred_instrument_form` outside `allowed_instrument_forms`, a
futures form present without `continuation_policy_id` (and vice versa),
`enabled` without a non-empty `provider_mapping_ids`, and an ETF/equity-
allowing spec declaring a non-equity-like adjustment kind
(`create_curated_market_driver_spec` accepts the RESOLVED
`AdjustmentPolicy` OBJECT, not merely an id, specifically to enforce
this real semantic check).

### 6. The 10 core concepts

5 MANDATORY core drivers (`us_dollar_strength`, `wti_crude`,
`brent_crude`, `silver`, `gold_reference`; `is_required=True`,
`tier=CORE_XAUUSD_MARKET_DRIVER`) and 5 strong-optional drivers
(`us_equity_market_stress`, `treasury_volatility`,
`broad_commodity_index`, `copper_industrial_growth`, `gold_miner_equity`;
`is_required=False`). All 10 exist in the registry unconditionally
(`default_core_market_driver_specs`/`default_optional_market_driver_specs`,
pure factories mirroring Phase 4B's own `default_core_series_specs`/
`default_extended_series_specs` shape). Of the 10, **9 are mapped to
real Alpha Vantage ETF proxies this phase**: `UUP` (dollar), `USO`
(WTI), `BNO` (Brent), `SLV` (silver), `GLD` (gold), `VIXY` (equity
stress), `DBC` (broad commodity), `CPER` (copper), `GDX` (gold miner).
**`treasury_volatility` ships UNSUPPORTED AND FAIL-CLOSED**: no
single-ticker ETF with a defensible, disclosable tracking relationship
to Treasury-market implied volatility (the MOVE index has no directly
investable ETF) was identified through this phase's shipped provider --
`enabled=False`, `provider_mapping_ids=()`, no mapping fabricated to
fill the gap.

### 7. Instrument-form and proxy policy

`instrument_form.py`. `InstrumentForm` (8 values: `SPOT`,
`CASH_INDEX`, `EXCHANGE_FUTURES_CONTRACT`,
`PROVIDER_CONTINUOUS_FUTURES`, `ETF`, `EQUITY`, `SYNTHETIC_INDEX`,
`ECONOMIC_PROXY`) names the SHAPE of the tradable object; `ProxyPolicy`
(`is_proxy`, `proxy_for`, `proxy_quality: HIGH|MODERATE|LOW`, plus six
free-text risk-disclosure fields for basis/roll/tracking-error/
currency/session/adjustment differences) names how faithfully it
approximates the economic concept. **An ETF-form mapping structurally
REQUIRES `proxy_policy.is_proxy=True`** (enforced in
`ProviderSymbolMapping.__post_init__`) -- no code anywhere in this
subpackage can label an ETF proxy as the underlying it tracks; behavior
confirmed directly (`test_etf_form_cannot_be_constructed_without_
proxy_true`, also exercised as a safety-scan behavioral check). Every
one of the 9 real Alpha Vantage mappings carries an honest
`proxy_quality`: `HIGH` for the physically-backed bullion ETFs (`GLD`,
`SLV`), `MODERATE` for futures-based commodity/currency funds (`UUP`,
`USO`, `BNO`, `DBC`, `CPER`, `GDX`), `LOW` for the VIX-futures-based
equity-stress proxy (`VIXY`, whose well-documented long-run decay
disqualifies it as anything beyond regime context).

### 8. Provider-symbol mappings

`symbol_mapping.py`. `ProviderSymbolMapping` binds provider +
provider_symbol + canonical_driver_id + instrument_form +
exchange_or_venue + currency + adjustment_policy_kind +
continuation_policy_id + mapping_version + proxy_policy into one
content-addressed identity. `SymbolMappingSet` validates the
cross-mapping invariant no single mapping's own `__post_init__` can
check alone: one `(provider, provider_symbol, mapping_version)` cannot
resolve to two DIFFERENT `canonical_driver_id`s within the same version
-- an alias change is expressed as a NEW mapping version, never an
in-place edit (confirmed directly: two mappings sharing a symbol but
different `mapping_version`s are permitted; the same version is
rejected).

### 9. Adjustment policy

`adjustment.py`. Five kinds: `RAW_UNADJUSTED` (this phase's shipped
adapter's only produced kind -- Alpha Vantage's `TIME_SERIES_DAILY` is
documented as raw/as-traded, confirmed via the live-verified response
schema), `SPLIT_ADJUSTED`, `TOTAL_RETURN_ADJUSTED`,
`PROVIDER_ADJUSTED_UNVERIFIED`, `NOT_APPLICABLE` (spot/index/futures --
no corporate action to adjust for). No corporate-action arithmetic is
ever performed anywhere in this phase; a policy CHANGE changes dataset
identity (the policy id feeds `MarketDriverBar.adjustment_policy_id`,
which is part of the bar's own content-addressed identity).

### 10. Futures contract and continuation policy

`futures.py`. `FuturesContractMetadata` models ONE specific,
individually identified contract -- rejected outright if
result-critical fields (root/full symbol, exchange, expiry, month/year,
multiplier, quote unit, currency, tick size, session timezone) are
missing; a provider unable to supply this must be classified
`PROVIDER_CONTINUOUS_FUTURES` instead. `ContinuationPolicyKind` has 6
values; `require_adjustment_evidence` structurally guards that
`BACK_ADJUSTED_DIFFERENCE`/`RATIO_ADJUSTED` never produce a bar without
`RollProvenance.adjustment_amount`/`adjustment_ratio` evidence.
**`MarketDriverBar` structurally REQUIRES `roll_provenance` for any
`PROVIDER_CONTINUOUS_FUTURES`-form bar** (a `MarketRecordError` guard,
confirmed directly and via a dedicated safety-scan behavioral check).
No real provider this phase maps a futures instrument -- the full
futures/continuation/roll-provenance code path is exercised end-to-end
through the mandatory fixture universe's synthetic `FakeMarketCollector`
(`wti_crude` gets a SECOND mapping, `PROVIDER_CONTINUOUS_FUTURES` via a
fixture provider, alongside its real Alpha Vantage ETF mapping --
exercising both mixed instrument forms AND the cross-provider model in
one universe).

### 11. Timezone and session policy

`sessions.py`. `TimezoneSessionPolicy` carries `timezone_key`
(validated against a small allowlist), open/close times or
`is_24_hour_session`, `CandleTimestampConvention`
(`OPEN_LABELED`/`CLOSE_LABELED`), a trading-week note, an optional
holiday-calendar reference, and a free-text `provider_session_note`
disclosing the PROVIDER's own documented session semantics -- never an
invented centralized-exchange truth. `market_normalization.
resolve_bar_open_time` interprets a provider's raw date text into a
genuine UTC open timestamp, honoring both the 24-hour flag and the
timestamp convention; confirmed directly that an identical calendar
date under an NYSE `TimezoneSessionPolicy` vs. an Asia/Tokyo one
resolves to materially DIFFERENT UTC open times, and exercised
end-to-end in the mandatory fixture universe (a genuinely separate
`copper_industrial_growth` driver on Asia/Tokyo, since
`TimezoneSessionPolicy` is a per-driver-spec field, not per-mapping --
see Section 26's known limitations).

### 12. Point-in-time availability policy

`availability.py`. `resolve_bar_availability_time` is STRUCTURALLY
fail-closed: every branch resolves a concrete, tz-aware datetime `>=
bar_close_time`, or raises `MarketAvailabilityUnresolvedError`; no
branch can default to "available at candle open" (confirmed by a
dedicated PIT test asserting `availability_time > open_time` for every
constructed bar, and by the safety scan's own `open_time`-as-
availability-proof structural check). Three kinds:
`CLOSE_PLUS_CONSERVATIVE_DELAY` (this phase's shipped adapter's policy,
`delay_minutes=1440` in the example config -- Alpha Vantage documents
no exact intraday publication SLA for daily bars, so a full
next-trading-day conservative buffer is used rather than claiming
real-time availability the platform cannot back), `NEXT_SESSION_OPEN_
CONSERVATIVE`, `EXPLICIT_PUBLICATION_DELAY_MINUTES`.

### 13. Multi-mapping backfill orchestration

`market_backfill.py` + `market_orchestration.py` (~600 lines).
`MarketBackfillSpec` separates `selected_driver_ids` (economic
concepts) from `selected_mapping_ids` (exact provider surfaces) --
deliberate, enabling more than one provider per concept in one
operation. `run_cross_asset_backfill_operation` implements the exact
12-stage state machine spec Section 18 requires: `REGISTRY_VERIFIED ->
PLAN_CREATED -> PROVIDER_METADATA_VERIFIED -> REQUESTS_COMMITTED ->
RESPONSES_COMMITTED -> RAW_RECORDS_PARSED -> RECORDS_NORMALIZED ->
COMPONENT_DATASETS_COMMITTED -> COMBINED_MANIFEST_COMMITTED ->
RECONCILED -> VERIFIED -> COMPLETED`, via a self-contained
`CrossAssetOperationStage`/`CrossAssetOperationStore` (a THIRD stage
machine implementation after Phase 3/4A/4B's own, duplicating the same
proven idempotent/conflict/monotonic-progression algorithm). Per
mapping: capability check -> provider metadata fetch+verify
(symbol/driver/instrument-form/currency/exchange/granularity
fail-closed comparisons, skipping fields the provider leaves
undisclosed rather than treating that as an automatic pass) -> history
fetch (reusing the metadata response when request manifests are
identical, Alpha Vantage's own case) -> raw parse -> normalize/
quarantine -> candidate-batch conflict pre-check (respects
`fail_fast`) -> commit -> FULL post-commit conflict re-check (raises
UNCONDITIONALLY, regardless of `fail_fast`, since an append-only store
cannot roll back an already-durable write) -> component manifest ->
provenance. `fail_fast=True` commits nothing for ANY mapping on the
first failure; `fail_fast=False` records each mapping's own outcome
independently, and completeness reflects the registry's OWN
required-driver tracking, never merely "every selected mapping
succeeded."

### 14. Component and combined dataset layout

`datasets.py`, one level more granular than Phase 4B's own
`curated.datasets`: one immutable `ComponentMarketDatasetManifest` per
`ProviderSymbolMapping` (the natural component key, since a mapping
already binds provider+symbol+driver+form+currency+adjustment+
continuation+version). `conflicting_coordinate_count` is structurally
REQUIRED to be `0` at construction -- a component with unresolved
conflicts can never be committed. `CombinedCrossAssetManifest` binds
`component_manifest_ids` keyed by `mapping_id`, tracks `required_
driver_ids`/`missing_required_driver_ids`, and enforces at construction
that a non-empty `missing_required_driver_ids` implies
`completeness_status=PARTIAL` -- a universe missing a required driver
can structurally never claim `COMPLETE` (confirmed directly, and via a
component-swap identity test: swapping bound component ids changes the
combined manifest's own recomputed identity).

### 15. Cross-provider conflict model

Folded into `market_reconciliation.py` (spec Section 20). When more
than one mapping in a combined manifest serves the SAME
`canonical_driver_id`, overlapping bars (by `open_time`) are compared
deterministically: exact equality, tolerance-level difference
(`PRICE_TOLERANCE_RATIO = 0.5%`, deliberately conservative), or
material conflict -- reported as `WARNING`-severity reconciliation
issues, never silently averaged or auto-resolved. Every component
dataset stays independently readable; no automatic "preferred provider"
selection exists this phase.

### 16. Gap and missing-bar policy

`gap_policy.py`. `analyze_bar_gaps` is PURE, scoped to one
`(canonical_driver_id, provider, provider_symbol)` coordinate space.
Missing-bar detection uses a Mon-Fri business-day heuristic
(`calendar_assurance="limited"`, always, honestly disclosed -- no
holiday calendar loaded this phase); a genuine weekend closure is NEVER
reported missing (confirmed directly). 24-hour sessions report zero
missing-bar candidates rather than inventing an unverified continuous-
session expectation. **Conflicting-duplicate-coordinate detection
compares ECONOMIC CONTENT, not `bar_id`** -- see Section 25, defect 1,
for the real bug this design choice fixes. `GapPolicy` (4 values)
governs only MISSING bars; conflicting duplicates are never a policy
choice, always a hard integrity failure.

### 17. Incremental update planning

`update_plan.py`. PURE and deterministic, mirrors `curated.update_plan`
exactly: NEVER reads the wall clock, NEVER touches the network.
`MappingUpdateAction` has three values: `NO_UPDATE_NEEDED`,
`APPEND_BARS`, `POLICY_REFRESH` (triggered when a mapping's own
declared adjustment/continuation policy no longer matches what the
CURRENT component manifest was built under). Each entry additionally
carries `needs_futures_roll_refresh: bool` for any futures-form mapping
-- an honest flag, never a computed roll decision. The exact-no-op
guarantee is proven at two independent layers: this module correctly
reports `NO_UPDATE_NEEDED` (confirmed directly), and
`ComponentMarketDatasetManifestStore.append`'s own idempotent
no-op-on-identical-id behavior independently guarantees no new version
mints even if a caller ignores the plan and runs the backfill anyway
(confirmed directly).

### 18. Mandatory fixture-based acceptance

`test_collectors_cross_asset_fixture_acceptance.py` (5 tests, ALWAYS
run, zero network). Assembles ONE curated universe covering: all 5 core
concepts via Alpha-Vantage-shaped ETF fixtures; a genuinely separate
`copper_industrial_growth` driver on an Asia/Tokyo session (multiple
timezones/session cutoffs within one universe); a `wti_crude`
`PROVIDER_CONTINUOUS_FUTURES` mapping with full roll provenance via a
synthetic `FakeMarketCollector` (mixed instrument forms + cross-
provider, since `wti_crude` simultaneously has an Alpha Vantage ETF
mapping too); one deliberately missing business day; an exact-
duplicate-bar idempotency proof; a conflicting-duplicate-bar hard-fail
proof; raw-response caching; provider metadata verification;
normalization; component/combined datasets; gap analysis;
reconciliation; verification; offline replay (byte-identical
`combined_manifest_id` on a second, `transport=None` pass); a
deterministic credential-free report export; and point-in-time
visibility after close only.

### 19. Opt-in real-provider acceptance -- status

`acceptance.py` + `test_collectors_cross_asset_acceptance.py` (9 tests
+ 1 opt-in skip). `resolve_alpha_vantage_api_key_from_environment` is
the ONE sanctioned place in `collectors/cross_asset/` that reads
`ALPHA_VANTAGE_API_KEY` from the environment; returns `None` (never
raises) when absent. `run_real_alpha_vantage_acceptance_workflow`
requires a non-empty `api_key` argument, exercises the ONE bounded,
real-verifiable mapping (`gold_reference` via `GLD` -- spec Section
24's own "supported subset" allowance, never claiming to validate the
full 10-concept universe against a live provider), then a second
cached-replay pass with a `_ForbiddenTransport` to prove zero network
calls. **Status in THIS run: SKIPPED cleanly** --
`ALPHA_VANTAGE_API_KEY` is not set in this environment, confirmed via
`pytest.skip("...zero network calls attempted")`; zero network calls
were made by the ordinary test suite. `RedactedCrossAssetAcceptanceReport`'s
own field set structurally cannot carry a secret (confirmed: no field
name resembles a credential).

### 20. Offline replay

No separate `replay.py` module -- exactly matching Phase 4B's own
precedent: replay is re-running `run_cross_asset_backfill_operation`
with `transport=None`, which STRUCTURALLY guarantees zero network calls
(the fetch path only reaches the transport branch on a cache miss, and
`transport=None` there raises `_MappingFailureError` before any call is
attempted -- a STRONGER guarantee than "transport fails when invoked,"
since it can never be invoked at all). Proven via both the
`transport=None` path (multiple tests) and an explicit
`_ForbiddenTransport` double that raises `AssertionError` on any
`.get()` call (`test_collectors_cross_asset_pit_concurrency_adversarial.py::
TestOfflineReplayNeverCallsTransport`).

### 21. Reconciliation

`market_reconciliation.py`. Re-derives coverage/bar counts from the
actual `MarketDriverBarStore`, cross-checks provenance completeness
(bar without provenance / provenance without bar / duplicate
coordinate), verifies component/combined manifest linkage against the
supplied registry and mapping set, and runs the cross-provider conflict
model (Section 15). Confirmed clean (`0` criticals) against the
fixture-acceptance universe's own successful backfill.

### 22. Verification independence classification

`market_verification.py`. Mirrors `curated.verification`'s own
discipline: EVERY check REDERIVES an artifact fresh from durable state
(re-read raw bytes, re-hash, re-parse via the SAME collector,
re-normalize, re-derive component/combined manifest ids), never
trusting a cached parse or a stored count. Per the module's own
docstring (spec Section 27's own "document which checks reuse provider
parsing logic vs. structurally independent"): checks 5/6/9 (re-hash,
reparse, recompute events) call `collector.parse_history_response`/
`market_normalization.normalize_raw_market_record` -- the SAME pure
functions orchestration used, invoked freshly against independently
re-read/re-hashed raw bytes, structurally independent of
orchestration's OWN RUN but NOT independent of the provider adapter's
OWN parsing implementation (a bug inside `providers/alpha_vantage.py`
itself would not be caught here; that risk is covered instead by the
dedicated adversarial tamper tests, Section 25). Checks 1-4, 7-8, 10-17
(registry/mapping/manifest self-identity, component/combined digests,
completeness) are fully structurally independent, requiring no
provider-specific code at all.

### 23. Secret/security evidence

- Every credential is a runtime PARAMETER (`api_key: str | None`),
  never a stored dataclass field -- confirmed by the safety scan's own
  AST-based `@dataclass`-field check (unchanged, now also covering
  `cross_asset/`).
- `ALPHA_VANTAGE_API_KEY` appears nowhere outside `acceptance.py`/
  `alpha_vantage.py` (the latter only DECLARES the constant NAME, never
  reads the environment) -- confirmed by a dedicated safety-scan check
  that also confirms no OTHER `cross_asset/` module reads `os.environ`
  directly.
- No long, opaque literal assigned to an `api_key=`-shaped field
  anywhere in `collectors/` or any `test_collectors_*.py` fixture --
  confirmed by the pre-existing `_LONG_LITERAL_API_KEY` scan, now
  reapplied against the larger tree.
- `RetryExhaustedError` (and its full `__cause__` chain) never contains
  a real API key -- confirmed directly with a realistic secret literal
  (`test_collectors_cross_asset_pit_concurrency_adversarial.py::
  TestNoSecretLeakage`), exercising the SAME `_redact_url_for_error`/
  class-name-only-on-retry-exhaustion fix Phase 4B already applied to
  the shared `transport.py`/`execute_request.py` attempt loop.
- `RedactedCrossAssetAcceptanceReport`'s own field set structurally
  cannot carry a secret (no field name resembles a credential).
- No Yahoo Finance, MT5, broker integration, websocket, or live
  streaming exists anywhere in this diff -- confirmed by the safety
  scan's own broker/websocket/live-trading-marker checks, unchanged and
  re-passing against the now-larger `collectors/` tree.
- Only HTTPS, allowlisted-host traffic is possible -- `Alpha
  VantageCollector` never constructs a `TransportRequest` itself;
  `execute_collector_request`'s existing, UNCHANGED URL/scheme/host/
  redirect validation (Phase 4A) applies unmodified.

### 24. Tests and exact results

- Focused cross_asset suite (11 new test files + example config test):
  `pytest tests/unit/market_data/test_collectors_cross_asset_*.py
  tests/unit/market_data/test_cross_asset_example_config.py` --
  **197 passed, 1 skipped** (the absent-key opt-in acceptance test), 0
  failed, ~2.1s.
- Full `market_data` package (Phases 1-4C together, `-W error`
  equivalent): **1,173 passed, 2 skipped**, ~61s -- confirms zero
  regression against every Phase 1-4B test (baseline was 970 passed, 1
  skipped; delta of 204 new tests: 197 cross_asset + 7 new
  `TestCrossAssetSpecificSafety` safety-scan additions).
- `ruff check .` (full repo): **all checks passed**.
- `mypy src` (full repo, 361 source files): **no issues found**.
- `git diff --check`: clean, no whitespace errors.
- ×10 repeat of the full cross_asset + example-config + safety-scan
  suite (258 tests × 10 runs): **2,580/2,580 passed**, fully stable, no
  flakiness observed, ~8.7s per run.
- ×4 repeat under DIFFERENT explicit `PYTHONHASHSEED` values (0, 1, 42,
  12345): **182/182 passed identically** in every run (the cross_asset
  test files alone, excluding the safety scan/example config, which
  have no hash-seed sensitivity of their own).
- A DEDICATED subprocess-based determinism test
  (`TestPythonHashSeedIndependence::test_registry_identity_independent_
  of_hash_seed`) additionally confirms the exact same populated
  registry's `registry_id` is byte-identical across two independent
  child interpreters started with `PYTHONHASHSEED=1` and
  `PYTHONHASHSEED=42` respectively.
- Full repository suite (`pytest -q`, every prior milestone together):
  **6,704 passed, 3 skipped, 0 failed**, total runtime 2h 24m 6s. All
  three skips are pre-existing/expected, unrelated to any single test
  run's environment quirk: this phase's own absent-key opt-in skip
  (Section 19), Phase 4B's own absent-key opt-in skip
  (`test_collectors_curated_acceptance.py`), and
  `tests/unit/ml/test_artifacts.py:110` ("symlink creation requires
  elevated privileges on Windows," the same pre-existing skip the Phase
  4A/4B delivery reports already documented). **Zero failures** -- a
  cleaner full-suite result than Phase 4B's own report (which recorded
  one pre-existing, timing-sensitive `execution_gateway` flake
  unrelated to that phase's changes); no such flake reproduced in this
  run.

No live network access was used anywhere in the ordinary suite. Every
test uses `FakeTransport`/`FakeMarketCollector` (in-memory doubles) or
exercises pure, network-free functions directly. The one opt-in test
that WOULD use the real network
(`TestOptInLiveAcceptance::test_live_workflow_when_key_present`) skipped
cleanly with zero network calls, as required.

### 25. Genuine defects found and fixed

Both of the following were self-identified during construction (via
dedicated pytest runs and direct debugging of an unexpected test
failure), not flagged externally.

1. **Conflicting-duplicate-coordinate false positive from `bar_id`-based
   comparison** (`collectors/cross_asset/gap_policy.py`). `analyze_bar_
   gaps`'s original implementation detected a "conflicting duplicate
   coordinate" by grouping bars by `open_time` and checking whether
   more than one DISTINCT `bar_id` existed at that coordinate. Because
   `MarketDriverBar.bar_id` embeds PROVENANCE fields
   (`request_manifest_id`/`response_manifest_id`/`source_manifest_id`/
   `source_row_index`) as part of its content-addressed identity -- the
   same precedent Phase 4B's own `CuratedMacroObservation.observation_id`
   already established -- the SAME economically-identical bar, fetched
   via two legitimately DIFFERENT responses (e.g. a `FORCE_FRESH`
   re-fetch whose envelope happens to include one more trailing row
   than the first fetch), mints a DIFFERENT `bar_id` even though its
   OHLCV/volume/classification are byte-identical. Concretely
   reproduced: a `FORCE_FRESH` re-fetch of the same 3-day GLD history
   plus one new day incorrectly raised `MarketProviderResponseError`
   ("3 conflicting coordinate(s)") for the 3 OVERLAPPING, economically
   identical days, purely because their `bar_id`s legitimately differed
   across the two fetches' responses. Root cause: the conflict check
   compared IDENTITY (`bar_id`) when it should have compared ECONOMIC
   CONTENT. Fixed at the root: the comparison now groups by `open_time`
   and compares a content tuple (open/high/low/close/volume/volume_unit/
   adjustment_policy_id/availability_time/availability_policy_id/
   session_policy_id/contract_metadata_id/roll_provenance) -- two bars
   with IDENTICAL economic content but different `bar_id` (differing
   only in provenance) are now correctly treated as a benign duplicate,
   never a conflict; two bars with genuinely DIFFERENT economic content
   at the same coordinate are still correctly flagged. Verified via a
   permanent regression test
   (`test_collectors_cross_asset_orchestration.py::TestSingleMappingBackfill::
   test_force_fresh_with_new_data_produces_new_component_version`, which
   failed before the fix and passes after) plus the pre-existing
   `test_collectors_cross_asset_normalization.py::TestGapAnalysis::
   test_conflicting_duplicate_bar_detected`, confirming the fix does
   not weaken genuine conflict detection.
2. **Inconsistent exception type from a redundant factory-level guard**
   (`collectors/cross_asset/instrument_form.py`). `create_proxy_policy`'s
   own early `require_non_empty(proxy_for or "", ...)` check (for
   `is_proxy=True` with a missing `proxy_for`) raised the generic
   `MarketDataError`, while `ProxyPolicy.__post_init__`'s own,
   functionally IDENTICAL check raises the more specific
   `InstrumentFormError` for the exact same violation -- meaning the
   exception type a caller observed depended on which construction path
   (factory vs. direct dataclass construction) triggered the violation,
   an inconsistency a dedicated registry test caught directly
   (`test_etf_form... test_is_proxy_requires_proxy_for` expected
   `InstrumentFormError`, observed `MarketDataError`). Root cause: the
   factory duplicated a check the dataclass's own `__post_init__`
   already performs correctly. Fixed by removing the redundant early
   check from `create_proxy_policy` entirely -- every violation now
   surfaces through exactly ONE exception type
   (`InstrumentFormError`) regardless of construction path. Verified via
   the full registry/policy test suite passing unchanged afterward (the
   removed check was genuinely dead code, not a behavior change for any
   valid input).

### 26. Known non-blocking limitations

See `docs/market_data_architecture.md`'s "Known limitations" section,
Phase 4C subsection, for the complete list with full reasoning.
Summary: only ONE provider (Alpha Vantage, `TIME_SERIES_DAILY` only) is
genuinely wired to a live universe, every mapped concept is an ETF-form
PROXY; `treasury_volatility` ships unsupported and fail-closed; no real
provider maps a futures instrument (the futures/continuation/roll code
path is fully implemented and exercised only via the fixture universe's
synthetic collector); gap-policy missing-bar detection has no
public-holiday calendar awareness (`calendar_assurance="limited"`,
always); `TimezoneSessionPolicy`/adjustment/availability policies are
per-driver-spec fields, not per-mapping; `contract_metadata_id_by_
mapping`/`roll_provenance_by_mapping` apply one futures identity per
mapping per call, not a general per-row roll resolver; no cross-asset-
to-XAUUSD-feature join exists yet (contract documented and safety-scan-
enforced, not yet exercised by an actual feature-generation join); no
pagination support, matching Phase 4A/4B; no CLI surface, matching
every prior phase.

### 27. Exact git status

As of this report, `git status --short` shows:
- 7 modified files: `docs/market_data_architecture.md`,
  `src/quant_platform/core/exceptions.py`,
  `src/quant_platform/market_data/collectors/fred.py`,
  `src/quant_platform/market_data/collectors/fred_series_metadata.py`,
  `src/quant_platform/market_data/manifests.py`,
  `src/quant_platform/market_data/source_manifests.py`,
  `tests/unit/market_data/test_market_data_safety_scan.py`.
- 1 new subpackage directory
  (`src/quant_platform/market_data/collectors/cross_asset/`, 22 files).
- 1 new shared module
  (`src/quant_platform/market_data/collectors/execute_request.py`).
- 12 new test files under `tests/unit/market_data/`.
- 1 new example config (`examples/xauusd_cross_asset_config.example.json`).
- 1 new documentation file (this report).

**Nothing has been staged. Nothing has been committed. Nothing has been
pushed.** HEAD remains `c42df1bcb1b9b87ca7556a9a3e0c29b8e23dfbc8`,
identical to the confirmed Phase 4C baseline.

### 28. Explicit confirmations

- Phase 4C work is **not staged** (`git add` was never run).
- Phase 4C work is **not committed** (`git commit` was never run; HEAD
  is unchanged from baseline).
- **Nothing has been pushed** (`git push` was never run).
- **No Alpha Vantage API key (or any other credential) was ever
  persisted, logged, printed, or committed** anywhere in this diff --
  confirmed by `TestCrossAssetSpecificSafety`'s cross_asset-scoped
  secret scan, the pre-existing long-literal-`api_key=` scan (now also
  applied to every new `test_collectors_cross_asset_*.py` file), and
  the dedicated no-secret-leakage regression test (Section 24).
- **No Yahoo Finance, MT5, broker integration, or live streaming**
  exists anywhere in this diff -- confirmed by the safety scan's own
  broker/websocket/live-trading-marker checks, unchanged and re-passing
  against the now-larger `collectors/` tree.
- **No scheduler or daemon** was added -- every workflow in this phase
  is a single synchronous function call.
- **No execution/portfolio-risk/ML-model/strategy code was touched** --
  confirmed by the full repository regression suite showing zero
  changes to, or failures caused in, any package outside `market_data`
  (plus the shared, purely-additive `core/exceptions.py` extension
  already established as the sanctioned pattern in Phase 4A/4B).
- **Phase 4D was not started** -- this diff is confined to the
  provider-neutral cross-asset collector layer and its own curated
  registry/orchestration/verification/reconciliation/reporting, exactly
  the Phase 4C scope.
- **Milestone 11 was not started** -- no work outside `market_data`
  (and its one shared, narrow `core.exceptions` extension) was touched.
- **No MT5, no broker integration, no live streaming, no feature join,
  no model training, no production scheduling** was started -- per the
  specification's own instruction, this phase stops here.
