# Milestone 10, Phase 4D — Delivery Report

**Point-in-time multi-source alignment bridge into the existing research
dataset pipeline.**

This report follows the exact 28-section structure the governing Phase
4D specification requires.

---

## 1. Baseline commit

Work began at commit `5d4352ba772969d056a9336fa80f7566b339b822` ("Add
provider-neutral cross-asset historical collectors" — Phase 4C). Before
any Phase 4D work began: `HEAD` was confirmed to match this hash exactly,
`git status --short` was confirmed clean, and (after a read-only `git
fetch`) `git status -sb` showed master fully synchronized with
`origin/master` (no ahead/behind). Nothing was pushed at any point during
this phase. `HEAD` remains `5d4352ba772969d056a9336fa80f7566b339b822`
throughout this entire phase — no commit was made (per explicit
instruction; see Section 27's confirmations).

## 2. Files added / modified

**Modified (4 source files, all narrow and additive — see Sections 3–4
and 16 for exactly what changed and why each change is safe):**

- `src/quant_platform/core/exceptions.py` — added one new Milestone 10
  Phase 4D exception block (`MarketDataBridgeError` and 8 subclasses),
  appended after the existing Phase 4C block. No existing exception
  class touched.
- `src/quant_platform/historical/loader.py` — added
  `HistoricalDatasetLoaderProtocol`/`HistoricalManifestLike` (two new
  `Protocol` classes). No existing class or function body touched.
- `src/quant_platform/features/dataset_builder.py` — narrowed
  `ResearchDatasetBuilder.__init__`'s `historical_loader` type hint from
  the concrete `DatasetLoader` to `HistoricalDatasetLoaderProtocol`
  (type-hint-only, zero runtime behavior change); added one new optional
  field, `ResearchDatasetBuildRequest.market_data_lineage: dict[str,
  object] | None = None`, threaded through into the one
  `ResearchDatasetManifest(...)` construction call inside `build()`.
- `src/quant_platform/features/manifests.py` — added one new optional
  field, `ResearchDatasetManifest.market_data_lineage: dict[str, object]
  | None = None`, included in `to_json_dict`/`from_json_dict` (absent
  key or explicit `null` both load as `None`).

**New (2 documentation files + 1 example config, modified 2 existing
docs — see Sections 3, 28):**

- `docs/milestone10_phase4d_delivery_report.md` (this file).
- `docs/market_data_architecture.md` — modified: updated Status line;
  updated the package-architecture dependency-direction paragraph;
  inserted a new "Point-in-time multi-source alignment bridge into the
  research-dataset pipeline (Phase 4D)" section; added a Phase 4D bullet
  list to "Known limitations"; updated "Future phases" to mark the
  macro/cross-asset-to-feature join as resolved.
- `docs/feature_engineering.md` — modified: inserted a new "Milestone 10
  Phase 4D: point-in-time multi-source alignment bridge" section (design,
  every module, identity mechanism, known limitations) before the
  existing "Known limitations" section.
- `examples/xauusd_point_in_time_research_dataset.example.json` — new,
  no credentials, every pinned id an explicitly marked, clearly-labeled
  placeholder (see Section 21/26).

**New (13 source files, one new package, 2,311 lines total):**

```
src/quant_platform/features/market_data_bridge/
    __init__.py              24 lines
    bindings.py              395 lines
    base_asset_adapter.py    258 lines
    macro_adapter.py         167 lines
    cross_asset_adapter.py   160 lines
    coverage.py              270 lines
    staleness.py              89 lines
    lineage.py                 89 lines
    request.py                162 lines
    rebuild_planner.py       210 lines
    reconciliation.py        230 lines
    verification.py          159 lines
    reports.py                 98 lines
```

**New (19 test files, one shared, non-collected helper module, 2,672
lines total, 196 new tests — verified via `pytest --collect-only`,
per-file counts below sum exactly to the 196 confirmed by running all 19
files together):**

```
tests/unit/features/
    _market_data_bridge_test_helpers.py                      (shared fixture builders, not collected)
    test_market_data_bridge_bindings.py                       26 tests
    test_market_data_bridge_base_asset_adapter.py              13 tests
    test_market_data_bridge_macro_adapter.py                   11 tests
    test_market_data_bridge_cross_asset_adapter.py               6 tests
    test_market_data_bridge_coverage.py                          8 tests
    test_market_data_bridge_staleness.py                         6 tests
    test_market_data_bridge_lineage.py                           4 tests
    test_market_data_bridge_rebuild_planner.py                   8 tests
    test_market_data_bridge_reconciliation.py                    6 tests
    test_market_data_bridge_verification.py                      4 tests
    test_market_data_bridge_reports.py                           1 test
    test_market_data_bridge_request.py                            3 tests
    test_market_data_bridge_identity_compatibility.py             4 tests
    test_market_data_bridge_safety_scan.py                       30 tests
    test_market_data_bridge_fixture_acceptance.py                13 tests
    test_market_data_bridge_adversarial.py                       28 tests
    test_market_data_bridge_backward_compatibility.py              7 tests
    test_market_data_bridge_performance.py                        5 tests
    test_market_data_bridge_example_config.py                    13 tests
```

`tests/unit/features/` totals 459 tests after this phase: 263
pre-existing M3 tests (completely unmodified by this phase — every one
of these files was left untouched) + 196 new Phase 4D tests.
`tests/unit/historical/` (289 tests) was likewise left completely
unmodified; `289 + 263 = 552` was confirmed passing immediately after
the `HistoricalDatasetLoaderProtocol` change, before any Phase 4D test
file was written, isolating that one change's own safety from
everything added afterward.

No file outside this list was created, modified, or deleted. No cache,
virtual environment, log, build artifact, dataset, credential, or secret
was staged or is present in the working tree (confirmed via `git status
--short` — see Section 27).

## 3. Existing M3 architecture audit

Conducted before any implementation, per Section 2's mandatory gate.
Read directly (not merely summarized): `historical/loader.py` (full,
199 lines), `features/dataset_builder.py` (full, 306 lines),
`features/manifests.py` (full, 518 lines), `features/engine.py` (full),
`features/interfaces.py` (full), `features/alignment.py` (full),
`features/models.py` (full), `features/multi_timeframe.py` (full),
`features/macro/macro_features.py` (full), `features/cross_asset/
cross_asset.py` (full), `features/validation.py` (full),
`features/registry.py` (full), `features/lineage.py` (full),
`market_data/candles.py`, `market_data/events.py`, `market_data/
repository.py`, `market_data/manifests.py`, `market_data/partitions.py`,
`market_data/ingestion.py`, `market_data/identity.py`,
`collectors/curated/revision_policy.py`, `collectors/curated/
macro_observation.py`, `collectors/curated/datasets.py`,
`collectors/cross_asset/market_record.py`, `collectors/cross_asset/
datasets.py`, `collectors/cross_asset/instrument_form.py`. A
general-purpose research agent additionally cross-checked 14 targeted
questions about exact data shapes before this direct reading began, to
scope the reading efficiently; every finding it reported was
subsequently verified against the actual source directly by this
session, not taken on trust.

**Key findings that shaped the design:**
- `FeatureEngine.compute` derives each row's availability instant
  internally as `open_time + timeframe.duration` — it never reads a
  stored close/availability timestamp from `base_df`. This is the exact
  rule the base-asset adapter's own availability policy replicates (see
  Section 8).
- `features/alignment.py`'s two pure functions
  (`align_higher_timeframe`/`as_of_join_external`) are the ONLY
  sanctioned point-in-time join primitives in the existing pipeline —
  the bridge reuses both unchanged, never reimplementing either (see
  Sections 9–10).
- `ResearchDatasetBuilder`'s ONLY base-asset coupling point is a
  concrete `historical.loader.DatasetLoader` instance, called via
  exactly 2 methods (`resolve_manifest`, `load_for_engine`); `higher_
  timeframe_data`/`cross_asset_data`/`macro_data` are pure
  caller-supplied `Mapping[..., pd.DataFrame]` dicts with no loader
  abstraction — the natural, lowest-friction integration surface (see
  Section 4).
- `historical_manifest` (from `resolve_manifest`) is used inside
  `build()` for exactly 3 attribute reads: `.dataset_id`, `.version`,
  `.content_checksum` — meaning a market_data-backed replacement loader
  needs only expose those 3 fields, never construct a full real
  `historical.manifest.DatasetManifest` (see Section 4).
- `compute_dataset_id` does NOT hash source-data content — it hashes
  only `{symbol, base_timeframe, feature_registry_fingerprint, label_
  definition, split_definition, preprocessing_definition}`, a
  deliberately stable "recipe id" across data revisions. Source-content
  changes flow into the manifest's separately-versioned `content_id`/
  `version` via the pre-existing `input_content_hashes`/`aux_input_
  content_hashes` free-form dict — already the sanctioned extension
  point (`historical_dataset_content_checksum` already uses it exactly
  this way). This is the mechanism Section 16/17 use, unmodified (see
  Section 16).
- `ResearchDatasetManifest._identity_fields()` is `to_json_dict()` minus
  `version`/`created_at`/`environment` — it INCLUDES `input_content_
  hashes`, which is what makes `ResearchManifestStore.save`'s own
  content-duplicate-vs-new-version comparison sensitive to a lineage
  change once `market_data_lineage` is added as a manifest field (see
  Section 16).
- No pre-existing import coupling exists between `features/` and
  `market_data/` in either direction; `market_data/` already imports
  `historical/` in exactly 3 places (timezone/calendar helpers only) —
  fully compatible with the required dependency direction.
- `market_data/`'s durable stores (Phase 1's `MarketEventStore`, Phase
  4B's `CuratedObservationStore`, Phase 4C's `MarketDriverBarStore`)
  each return their FULL current append-only content on read, with no
  date-range filter and no selective-by-manifest-version read parameter.
  Phase 2's `PartitionStore` is explicitly current-version-only storage.
  This single fact governs the whole verification design in Sections
  8–10 and is documented as a known limitation in Section 26.

## 4. Chosen dependency direction

`historical/market_data → features.market_data_bridge → features
(unmodified) → research dataset artifacts → ml/execution`, exactly as
required. `market_data` was not modified in this phase and does not
import `features` or any part of it (verified structurally by
`test_market_data_bridge_safety_scan.py::TestMarketDataNeverImportsFeatures`,
which scans every file under `market_data/` for a `features` import and
is proven non-vacuous against a deliberately bad snippet). The bridge
package lives inside `features/` (not `market_data/`) specifically so it
is the one and only side of the boundary that imports both packages —
`market_data` remains reusable by any future consumer with zero
knowledge of `features`' existence.

## 5. Bridge design

`features/market_data_bridge/` — see Section 2 of `docs/
feature_engineering.md`'s own new Phase 4D section for the full package
layout diagram and per-module description; not repeated here to avoid
drift between two copies of the same information. Summary of the design
principles that shaped every module:
- **Verify, then resolve.** Every source adapter exposes a fail-closed
  `verify_*_binding` function (independent re-verification against live
  `market_data` state) and a separate `resolve_*_dataframe`/
  `load_for_engine` function (the actual Decimal → float64 projection).
  Never conflated into one function that trusts its own input.
- **Reuse, never reimplement, the existing PIT join primitives.**
  `align_higher_timeframe`/`as_of_join_external` (both from `features/
  alignment.py`, unmodified) are the ONLY alignment logic anywhere in
  this bridge. Where their exact contract needed an adapter (the
  cross-asset availability-shift, Section 9), the adapter reshapes the
  INPUT to fit the existing function, never adds a parallel alignment
  path.
- **Pure functions where the spec calls for purity.** `rebuild_planner.
  plan_rebuild` takes no `market_data`/`features` I/O of its own — every
  input is a value the caller already resolved.
- **Additive, not replacing.** Every change to a pre-existing M3 file
  (Section 2) is a new optional field or a new Protocol; no existing
  class, function signature, or behavior was altered.

## 6. Source binding models

`bindings.py` — `BaseAssetDatasetBinding`/`MacroDatasetBinding`/
`CrossAssetDatasetBinding`, exactly the fields spec Section 4 named for
each (verified field-by-field against the spec during design). Every
binding is immutable (`frozen=True` dataclass), content-addressed (its
own `binding_id` is `compute_content_id` over every other field, via a
`create_*` factory mirroring `market_data`'s own established
provisional-id-then-real-id pattern), and structurally rejects a fixed
set of mutable-alias tokens (`latest`/`current`/`newest`/`active`/
`default`/`provider_default`/`head`) in every identity-bearing string
field at construction time (`SourceBindingError`). `CrossAssetDataset
Binding` additionally rejects `instrument_form=ETF` paired with
`proxy_policy.is_proxy=False` at construction (reuses `market_data`'s
own `ProxyPolicy`/`ProxyQuality` types directly — no duplicate proxy
vocabulary). 26 dedicated tests
(`test_market_data_bridge_bindings.py`) cover deterministic construction,
per-field identity sensitivity, mutable-alias rejection (parametrized
over 7 alias spellings including case/whitespace variants), the ETF/
proxy guard, and full JSON round-tripping for all three binding types.

## 7. Base-asset data loading

`base_asset_adapter.py`. `verify_base_asset_binding` re-derives the
current `market_data.manifests.DatasetManifest` for `(RAW_MARKET_EVENTS,
provider, canonical_instrument_id)`, requires the binding's pinned
`dataset_id` to equal it exactly (fail-closed `SourceVerificationError`
otherwise — see Section 26 for why an older, superseded `dataset_id`
cannot be safely re-read), re-reads the full live event stream, and
independently recomputes `market_data.ingestion.semantic_digest_for_raw_
events` over it, requiring an exact match against the manifest's own
recorded digest. Candles are filtered to the exact `(instrument_id,
timeframe)` pair (a `MarketEventStore` partition interleaves every
timeframe for one instrument), deterministically ordered by `(event_time,
event_id)`, and same-`open_time`-different-content conflicts are
detected and rejected rather than silently resolved. `resolve_base_asset_
dataframe` is the one, documented Decimal → float64 boundary crossing,
producing a strictly-ascending, duplicate-free `core.types.OHLCV_COLUMNS`
frame. `MarketDataBaseAssetLoader` implements `HistoricalDatasetLoaderProtocol`
structurally (verified via `issubclass` in the backward-compatibility
tests) and is a drop-in `historical_loader=` for the real, unmodified
`ResearchDatasetBuilder`. 13 tests
(`test_market_data_bridge_base_asset_adapter.py`) cover normal
resolution, stale-pin fail-closed rejection, missing-manifest fail-closed
rejection, zero-matching-timeframe rejection, conflicting-candle
rejection, range filtering, naive-timestamp rejection, and full Protocol
conformance (symbol/timeframe/dataset_version mismatch rejection,
stable manifest resolution, correct OHLCV shape).

## 8. Macro availability alignment

`macro_adapter.py`. Every macro value enters the join keyed on
`CuratedMacroObservation.availability_time` (Phase 4B's own release/
availability proof), mapped 1:1 to `features.alignment.
as_of_join_external`'s `release_time` parameter — never `observation_
date`. `verify_macro_binding` re-derives `collectors.curated.datasets.
ComponentDatasetManifest`'s own `semantic_digest` formula
(`compute_content_id("curated_component_semantic_digest", {"observation_
ids": ...})`, the exact same kind string and payload shape Phase 4B
itself uses to mint it) from a live `CuratedObservationStore` read and
requires an exact match against the current manifest — fail-closed
otherwise. `select_observations_for_policy` is a pure selection over the
already-verified, already-multi-vintage set; `resolve_macro_dataframe`
shapes the selected result into the exact `value`/`release_time` frame
`features.macro.macro_features.register_macro_features` already expects.
11 tests (`test_market_data_bridge_macro_adapter.py`) cover resolution,
stale-pin rejection, missing-manifest rejection, and — importantly — a
regression test proving verification is insensitive to a
`canonical_series_name` typo on the binding (a false-negative bug caught
and fixed during development; see Section 25).

## 9. Revision/vintage handling

`RevisionPolicyKind.VINTAGE_SERIES` passes every distinct vintage
through unchanged — because each vintage carries its own accurate
`availability_time`, `as_of_join_external`'s existing backward-looking
merge_asof over the full multi-vintage stream already reproduces true
point-in-time-correct "most recently released value as of T" with zero
new alignment logic; this is the general-purpose default.
`FIRST_RELEASE_ONLY` keeps only the earliest-released vintage per
`observation_date` (sorted deterministically by `(availability_time,
observation_date, realtime_start, observation_id)` before selection,
never relying on input order). `LATEST_AVAILABLE`/`AS_OF_REALTIME_DATE`
are refused outright (`AlignmentPolicyError`) for research/training
dataset construction — both are explicitly non-point-in-time-safe by
their own `market_data.collectors.curated.revision_policy` docstrings.
A genuinely missing observation (`is_missing=True`) is included with
`value=NaN` at its own real `availability_time`, never dropped — proven
by `test_missing_observation_becomes_nan_not_dropped`. 8 tests exercise
`select_observations_for_policy` directly (vintage retention, first-
release deduplication, both non-PIT-safe kinds parametrized) plus
end-to-end verification via `resolve_macro_dataframe`.

## 10. Cross-asset closed-bar alignment

`cross_asset_adapter.py`. `verify_cross_asset_binding` mirrors the
macro adapter's verification design exactly (recomputes `collectors.
cross_asset.datasets.ComponentMarketDatasetManifest`'s own
`"cross_asset_component_semantic_digest"` formula from a live
`MarketDriverBarStore` read, requires an exact match), plus rejects
same-`open_time`-different-content conflicting bars. **The one genuinely
new alignment idea in this phase**: `align_higher_timeframe` derives its
reveal instant internally as `open_time + timeframe.duration` with no
parameter for an externally-supplied availability time, but a
`MarketDriverBar` carries its OWN resolved `availability_time` (which
can be materially later than a naive close-time guess — e.g. Alpha
Vantage daily ETF bars under `CLOSE_PLUS_CONSERVATIVE_DELAY`, next-day
availability). `resolve_cross_asset_dataframe` therefore emits a
SYNTHETIC `open_time` per row (`availability_time - timeframe.duration`)
so `align_higher_timeframe`'s own rule lands exactly on the bar's true
availability instant, without any change to that shared primitive. This
was proven end-to-end (not merely unit-tested in isolation): a base row
1 second before a bar's true `availability_time` sees `bar_index=-1`
(not yet revealed); the same row at exactly `availability_time` sees
`bar_index=0` — `test_bar_not_revealed_before_true_availability_time`.
6 tests total, plus reuse in the adversarial and acceptance suites.

## 11. Session/timezone handling

Cross-asset session/timezone semantics are NOT re-derived by this
bridge — `CrossAssetDatasetBinding.session_policy_id` is a bound,
opaque reference to the exact Phase 4C `TimezoneSessionPolicy` used at
backfill time, propagated into `market_data_lineage` for audit/lineage
purposes. The bridge's own contribution is exclusively the availability-
time-based reveal instant (Section 10), which already correctly reflects
whatever session convention was baked into the bar's `availability_time`
at collection time — a daily bar with an incompatible session cutoff is
distinguished by its differing `availability_time`, never treated as
identical to a same-calendar-day bar from a different session (proven by
adversarial item 7, `Test07IncompatibleSessionCutoffsNotTreatedAsIdentical`,
which constructs two otherwise-identical cross-asset fixtures differing
only in their availability delay and confirms they resolve to different
synthetic `open_time` sequences).

## 12. Proxy and instrument-form lineage

`CrossAssetDatasetBinding.instrument_form`/`proxy_policy` reuse
`market_data`'s own `InstrumentForm`/`ProxyPolicy` types directly (never
a duplicate vocabulary) and flow, via `lineage.build_market_data_
lineage`, into `ResearchDatasetManifest.market_data_lineage` — so a
completed research dataset's own manifest can always answer whether
dollar strength was a cash index or an ETF proxy, what proxy quality was
declared, and what basis/roll/tracking-error/currency/session/
adjustment risk notes accompanied that mapping, without embedding this
descriptive metadata in every row. `adjustment_policy_id`/`continuation_
policy_id` are likewise bound, opaque policy references, also
lineage-visible. See Section 26 for the one disclosed gap here
(per-bar `roll_provenance`/`contract_metadata_id` are not currently
propagated into the aligned OHLCV frame — unexercised by this phase's
ETF-only fixtures).

## 13. Feature-engine integration

Zero feature-computation logic exists anywhere in `features.market_data_
bridge` — every feature family (`features.macro.macro_features`,
`features.cross_asset.cross_asset`, `features.multi_timeframe`,
`features.technical`, `features.temporal`) is the REAL, unmodified M3
module, invoked exactly as it always was, fed `macro_data`/`cross_asset_
data`/`higher_timeframe_data` dicts the bridge populates in the exact
pre-existing shape those modules already expect. `request.
build_research_dataset_from_market_data` is the single orchestration
entry point: it resolves every binding, evaluates coverage, assembles
lineage, and hands off to the REAL `ResearchDatasetBuilder.build()` —
never a second builder, never a parallel feature-computation path.
Proven directly: `test_market_data_bridge_request.py` and `test_market_
data_bridge_fixture_acceptance.py` both construct a REAL `FeatureRegistry`,
register the REAL `register_macro_features`/`register_cross_asset_
features`, and confirm the resulting manifest's `row_counts`/`feature_
names` reflect genuine computed output, not a stub.

## 14. Label/split/preprocessing isolation

Untouched, by construction: `request.build_research_dataset_from_market_
data`'s own module docstring states the fixed orchestration order
explicitly — coverage evaluation and lineage assembly run entirely on
source `open_time`/`release_time` coverage, BEFORE `ResearchDatasetBuilder.
build()` ever computes a label. `coverage.evaluate_source_coverage`'s
signature has no `labels`/`label_definition` parameter at all
(confirmed by `test_market_data_bridge_adversarial.py::
Test20LabelsNeverInfluenceSourceTrimmingOrAlignment`, which also greps
every bridge module and confirms none imports `features.labels`). Every
other M3 leakage guarantee (trailing-only features, train-only
preprocessing, per-fold-group fitting, purged walk-forward compatibility)
is inherited unchanged because `ResearchDatasetBuilder.build()`'s own
body was not modified beyond threading through one new optional field.

## 15. Research manifest evolution

`ResearchDatasetManifest` gained exactly one new field,
`market_data_lineage: dict[str, object] | None = None`, added AFTER the
existing `content_id` field (dataclass field-ordering requirement — no
existing field moved). Included in `to_json_dict`/`from_json_dict`;
`from_json_dict` treats an absent key exactly the same as an explicit
`null` (`None`) — an old manifest written before Phase 4D existed loads
with `market_data_lineage=None`, never a fabricated or reinterpreted
value (`test_market_data_bridge_backward_compatibility.py::
test_json_missing_the_key_entirely_still_loads`). No parallel manifest
class was created — this is the additive extension of the EXISTING
class the spec required, not a second implementation.
`lineage.build_market_data_lineage` assembles the payload (schema-
versioned via `MARKET_DATA_LINEAGE_SCHEMA_VERSION = 1`, stable-sorted so
two calls with the same bindings in different dict order produce
byte-identical output — `test_market_data_bridge_lineage.py::
test_deterministic_regardless_of_dict_order`), binding directly or
transitively every field Section 16 named: base dataset id (via the
base binding), macro/cross-asset universe ids + exact component ids
(via each binding), provider mappings/proxy/instrument forms/adjustment/
continuation/session/availability policy ids (all binding fields),
alignment policy ids (the same), and source coverage result (the
embedded `coverage_decision`). Feature registry fingerprint/order/label/
split/preprocessing already live on the pre-existing manifest fields
this is additive to and are deliberately NOT repeated inside `market_
data_lineage`.

## 16. Dataset identity changes

`compute_dataset_id` (features/manifests.py) was NOT modified — its
signature still accepts exactly the 6 original parameters
(`test_market_data_bridge_identity_compatibility.py::
test_compute_dataset_id_hash_inputs_are_untouched_by_bridge` asserts
this directly via `inspect.signature`). Instead, because `market_data_
lineage` is NOT excluded from `ResearchDatasetManifest._identity_
fields()`, `ResearchManifestStore.save`'s own pre-existing content-
duplicate-vs-new-version comparison already becomes sensitive to any
lineage change with zero changes to that store's own logic.

**The precise mechanism, stated exactly (it is easy to get subtly
wrong, and was — see Section 25):** `version` is
`f"{sequence:06d}-{content_id_prefix}"`; `content_id` hashes only the
WRITTEN feature/label/split Parquet bytes, so two builds with different
lineage but coincidentally identical output bytes CAN share the same
`content_id_prefix` — but `sequence` always advances whenever
`_identity_fields()` differs from the latest existing version in the
SAME manifest history (`ResearchManifestStore.save` loads the latest
version and compares), so the FULL version string still changes,
provided the rebuild targets the same `ResearchManifestStore` root (the
realistic "rebuild after a source revision" scenario). Two isolated,
freshly-created stores would each independently mint `sequence=1` and
could coincidentally produce an identical version string despite
genuinely different lineage — this is why every identity-compatibility
test deliberately reuses one store across both builds being compared.

Golden identity tests (`test_market_data_bridge_identity_compatibility.py`,
4 tests): a different macro binding (one with an added revision)
produces the same `dataset_id` but a different, sequence-advanced
`version`, a different `market_data_lineage`, and a different `market_
data_lineage_content_id`; a different cross-asset proxy quality
produces the same `dataset_id` but a different version; rebuilding with
IDENTICAL bindings against the same store is confirmed to be a TRUE
no-op (`version` unchanged, still `000001-...`) — the necessary control
proving the mechanism is sensitive to genuine changes only, never merely
to being invoked twice. No existing M3 golden identity test
(`test_manifests.py`/`test_dataset_builder.py`) was modified, and all
continue to pass unmodified (see Section 22).

## 17. Coverage/staleness policy

`coverage.py` implements all 4 named policy kinds. `FAIL_REQUIRED_
SOURCE` (default-safe) raises `SourceCoverageError` immediately when any
required source's coverage fraction of the requested range falls below
`minimum_observation_coverage_fraction`. `ALLOW_OPTIONAL_MISSING_AND_
REPORT` records the identical shortfall for an optional source as a
`CoverageFinding` without raising. `TRIM_TO_COMMON_SAFE_RANGE` computes
the intersection of every required source's own coverage with the
requested range, returning it as `CoverageReport.safe_start`/`safe_end`
with an explicit, human-readable `trim_reason` naming exactly which
source forced the trim and by how much — never a silent trim (8 tests,
`test_market_data_bridge_coverage.py`, including the exact no-overlap
failure case). `QUARANTINE_INTERVAL` is evaluated separately, post-hoc,
by `evaluate_missing_runs` over an already-computed missing indicator,
identifying every maximal run of consecutive missing rows exceeding a
configured tolerance and returning explicit `(start_index, end_index)`
intervals — reported, never auto-dropped by this function itself.
`staleness.py` reuses `as_of_join_external`/`align_higher_timeframe`
directly (no new alignment logic) to produce a per-source `Staleness
Finding`, distinguishing "unavailable" (no qualifying release/close yet)
from "stale" (a qualifying release/close older than a source-specific
threshold) — never one global threshold across unlike frequencies (6
tests, `test_market_data_bridge_staleness.py`).

## 18. Incremental rebuild planning

`rebuild_planner.plan_rebuild` is a pure function — no `market_data`/
`features` I/O of its own; every input (existing lineage, proposed
bindings, optional `SourceChangeEvidence`) is a value the caller already
resolved separately. Returns one of `NO_OP`/`APPEND_ONLY_SAFE_EXTENSION`/
`PARTIAL_RECOMPUTATION_REQUIRED`/`FULL_REBUILD_REQUIRED` with reason
codes, affected source names, a `required_warmup_from` timestamp (for
`PARTIAL_RECOMPUTATION_REQUIRED`), and a deterministic `plan_id`
(content hash of kind/reasons/affected sources). A missing/unrecognized
existing lineage or a changed feature/label/split/preprocessing recipe
both force `FULL_REBUILD_REQUIRED`. Distinguishing a safe append from a
content-only revision fundamentally cannot be done from a bare changed-
component-id signal alone (a content hash carries no diff), so the
planner requires the CALLER to separately supply `SourceChangeEvidence`
(old/new covered-time range and observation count); absent that evidence,
any changed binding conservatively forces `FULL_REBUILD_REQUIRED` rather
than guessing. `SourceChangeEvidence.is_append_only` requires the
covered range's start to be unchanged AND strict growth in at least one
of covered-end-time/count — NOT merely `>=` in both — because identical
range-and-count evidence alongside a changed component id is exactly the
signature of a same-range, same-count content revision, which an
earlier, more permissive `>=`-based version of this check incorrectly
classified as safe to append-skip (a real bug caught by this phase's own
adversarial test 22 and fixed with a regression test — see Section 25).
8 tests directly (`test_market_data_bridge_rebuild_planner.py`) plus the
mandatory acceptance workflow's own Step 12.

## 19. Reconciliation

`reconciliation.py` is the non-raising, structured-issue-reporting
counterpart to the adapters' own fail-closed `verify_*` functions
(`reconcile_binding_source`, wrapping all three, converting a raised
`SourceVerificationError` into a classified `ReconciliationIssue` instead
of propagating it), plus `reconcile_manifest_lineage` (recomputes
lineage from the same bindings a caller presents and requires an exact
match against a manifest's own recorded lineage), `reconcile_output_
range` (confirms a manifest's own `utc_start`/`utc_end` never extends
beyond a coverage report's safe range), and two standalone PIT
re-checks — `reconcile_no_pre_availability_macro_leakage`/`reconcile_
no_pre_close_cross_asset_leakage` — that independently re-derive the
as-of/close-time join's own output and assert no selected value's
timestamp exceeds the row's own availability instant. `ReconciliationIssueCode`
enumerates 8 of the spec's named issue codes with real, checkable logic
behind each; the remaining named codes (wrong vintage selected, session-
policy mismatch, duplicate aligned coordinate as a STANDALONE finding
distinct from the adapters' own construction-time rejection) are
detectable today only via the adapters' existing fail-closed checks
(Sections 7–10), not as a separately-invocable non-raising reconciliation
function — disclosed in Section 26, not silently omitted. 6 tests
(`test_market_data_bridge_reconciliation.py`).

## 20. Verification independence classification

`verification.py`'s `INDEPENDENCE_CLASSIFICATION` constant classifies
every check HONESTLY into exactly 3 kinds, stated plainly rather than
overclaimed: `"independent_re_read"` (a genuinely separate code path
re-reads durable evidence — source binding identity, dataset existence,
source record re-reads, structural validation, truncation invariance);
`"same_formula_re_derivation"` (reproduces an already-published content
digest using the exact same hash formula `market_data` itself used to
mint it — semantic digest recomputation, manifest lineage verification —
proves store/manifest CONSISTENCY, explicitly NOT an algorithmically
independent re-derivation of a differently-implemented hash);
`"reused_shared_primitive"` (reuses the SAME shared M3 alignment
primitive the production feature-computation path itself uses — aligned
value/missing/stale indicators, the no-pre-availability-leakage checks —
proves internal self-consistency, explicitly NOT independence from a bug
that might exist in that shared primitive itself, which M3's own
`tests/unit/features/test_alignment_boundaries.py` covers separately).
`verify_truncation_invariance_macro`/`_cross_asset` are the two
genuinely NEW pieces (not a re-run of an adapter's own verification):
each builds alignment against the FULL source, then against a version
truncated after a cutoff, and requires every base row eligible under
that cutoff to produce an IDENTICAL result in both — directly exercising
spec's own "truncating all records after T must not change any aligned
row at/before T" requirement. This caught a real bug during development
(Section 25). 4 dedicated tests plus reuse across the acceptance and
adversarial suites (10+ additional invocations).

## 21. Fixture acceptance

`test_market_data_bridge_fixture_acceptance.py`, 13 tests structured as
the mandatory 12-step sequential workflow (a `scope="module"` fixture
builds durable state once; later test classes depend on it). Fixtures:
base XAUUSD H1 candles (240 bars); all 4 required macro series (DFII10,
DGS10, CPIAUCSL, DFF) — with one genuinely missing observation
(CPIAUCSL) and one genuine revision (DFF, a second vintage for an
already-covered date, released 30+ days later); all 5 required
cross-asset drivers (a dollar-strength proxy/UUP, WTI proxy/USO, Brent
proxy/BNO, silver proxy/SLV, gold reference/GLD), each with explicit
ETF-proxy metadata (`instrument_form=ETF`, `proxy_policy.is_proxy=True`,
basis/roll/tracking-error risk notes) and differing availability delays
across drivers (differing session cutoffs). Steps 1–2 (create + verify
every durable source), 3 (bindings reject mutable aliases), 4–5 (align +
build via the REAL `ResearchDatasetBuilder`, never a test replacement —
confirmed directly by asserting `sum(manifest.row_counts.values()) > 0`
against genuinely computed splits), 7 (manifest lineage reconciles), 8–9
(no pre-availability macro leakage / no incomplete cross-asset candle
leakage, PLUS truncation invariance for every one of the 4 macro series
and 5 cross-asset drivers individually), 10–11 (replay into a fresh
research-store root, compare `dataset_id`/`content_id`/`output_content_
hashes`/`market_data_lineage` for byte-for-byte equality), 12 (update
one macro source's component id with realistic append-only evidence,
produce a deterministic `APPEND_ONLY_SAFE_EXTENSION` plan, confirm
identical inputs reproduce the identical `plan_id`).

## 22. Backward compatibility

`test_market_data_bridge_backward_compatibility.py`, 7 tests: the exact
pre-Phase-4D `ResearchDatasetBuildRequest` construction (no `market_
data_lineage` kwarg at all) still works, defaulting the new field to
`None`; a legacy-shaped `ResearchDatasetManifest` (constructed the
pre-Phase-4D way) round-trips through JSON with `market_data_lineage`
staying `None`, including the specific case where the JSON key is
ABSENT entirely (simulating a manifest file literally written before
this phase existed) rather than merely `null`; the real, unmodified
`historical.loader.DatasetLoader` class is confirmed via genuine runtime
`issubclass` (not merely a static/mypy-only check — `Historical
DatasetLoaderProtocol` is `@runtime_checkable` with methods only) to
still structurally satisfy the new Protocol. A separate research pass
(delegated to a background agent, results independently reviewed)
confirmed no consumer in `ml/`, `execution/`, or `feature_cli.py` does
exhaustive field enumeration over `ResearchDatasetManifest` that a new
optional field could break, and that neither existing example config
(`xauusd_research_dataset.example.json`/`xauusd_ml_experiment.example.json`)
references any manifest field the new field could affect — both are
CLI-config shapes, not serialized manifests. Every pre-existing M3 test
(`tests/unit/features/test_*.py` minus the new `test_market_data_bridge_
*.py` files, and all of `tests/unit/historical/`) continues to pass
completely unmodified — 552 tests before this phase's own additions
(confirmed directly, not merely inferred, immediately after making the
`HistoricalDatasetLoaderProtocol` change).

## 23. Performance

`test_market_data_bridge_performance.py`, 5 tests, generous (60s),
non-flaky wall-clock sanity bounds — never a tight microbenchmark floor,
per spec's own "no unrealistic microbenchmark floors" instruction.
Measured on reference hardware (informational; expect run-to-run
variance): base-asset verify + resolve over 800 H1 bars; macro/cross-
asset truncation-invariance checks over an 800-row base timeline against
a 60-day source; 5-source coverage evaluation; 8-source lineage assembly
+ fingerprint. All 5 completed in 38.17s combined (well under the 60s
per-test bound). Complexity is documented, not re-proven: `align_higher_
timeframe`/`as_of_join_external` are vectorized O(n log m) operations
already benchmarked by M3's own `tests/performance/
test_feature_throughput.py`; this phase's own tests confirm the
bridge's ADDITIVE work (verification's fresh read + digest
recomputation, coverage evaluation, lineage assembly) does not introduce
an unexpected extra full-data pass per source added.

## 24. Tests and exact results

Focused bridge suite (all 19 `tests/unit/features/test_market_data_bridge_*.py`
files, run together): **196 passed**, `-W error`, 57.75s.

Full `tests/unit/features/` (263 pre-existing M3 tests + 196 Phase 4D
tests): **459 passed**, run with `-W error` (0 warnings tolerated), in
64.09s.

`tests/unit/historical/` (289 tests) + `tests/unit/features/` (263
pre-existing tests, before any Phase 4D test file existed) combined —
confirmed immediately after the `HistoricalDatasetLoaderProtocol`/
`ResearchDatasetBuilder` type-hint change, before further Phase 4D work
proceeded, to isolate that one change's own safety: **552 passed**,
`-W error`, 18.31s.

Full repository quality gates:
- `git diff --check`: clean (no whitespace errors).
- `ruff check .` (full repo): **all checks passed**.
- `mypy src` (full repo, 374 source files): **no issues found**.
- Full repository `pytest` suite: **6887 passed, 3 skipped, 0 failed**,
  in 2:25:24. The 3 skips are the same pre-existing, deliberate ones
  Phase 4C's own delivery report recorded (`ALPHA_VANTAGE_API_KEY`-gated
  and `FRED_API_KEY`-gated opt-in real-provider acceptance workflows,
  and one Windows-symlink-privilege-gated ML artifact test) — zero new
  skips, zero failures, zero errors introduced anywhere in the
  repository by this phase.

## 25. Genuine defects found and fixed

Two real, non-cosmetic bugs were found and fixed during this phase's own
adversarial testing, each with a regression test:

1. **`SourceChangeEvidence.is_append_only` false positive.** The
   original implementation used `>=` (not strict `>`) for BOTH
   `last_covered_time` and `observation_count` growth checks, so a
   binding change with IDENTICAL covered-range and IDENTICAL count
   (the exact signature of a same-range, same-count content-only
   revision — e.g. a macro value correction for an already-covered
   date) was incorrectly classified `APPEND_ONLY_SAFE_EXTENSION`,
   which could cause a rebuild planner consumer to skip recomputing a
   row range that actually needs it. Found by adversarial test 22
   (`Test22IncrementalPlannerDoesNotMissAHistoricalRevisionImpact`).
   Fixed by requiring STRICT growth in at least one of the two
   dimensions whenever the component id itself changed; the earlier,
   permissive test expectation was corrected alongside the fix (not
   loosened to match the bug).
2. **`verify_truncation_invariance_macro`'s `release_diff` computation
   used a plain `!=` comparison on `release_time` values, which is
   `NaT != NaT` — True in pandas (NaN-style unordered comparison
   semantics) — so every row with no qualifying release yet in EITHER
   the full or the truncated join (a normal, frequent, genuinely
   non-differing case) was misreported as "differing," producing a
   false NON-invariance finding.** Found by adversarial test 2
   (`Test02RevisedCpiNotInjectedIntoEarlierRows`), which failed with a
   real assertion mismatch rather than a construction error. Fixed by
   adding the same explicit both-NaT exemption `value_diff` already had
   (mirroring the existing pattern in the same function rather than
   inventing a new one).

Both fixes are in the shipped source (`rebuild_planner.py`,
`verification.py`) with the corresponding test now passing and serving
as the permanent regression guard. No other genuine correctness,
identity, leakage, or compatibility defect was found during this phase's
development, adversarial testing, or fixture acceptance workflow.

## 26. Known non-blocking limitations

- **No selective historical read for a superseded `market_data`
  manifest version, for any of the 3 source kinds.** `MarketEventStore.
  read_events`/`CuratedObservationStore.read_observations`/
  `MarketDriverBarStore.read_bars` always return the full CURRENT
  content; `partitions.PartitionStore` is explicitly current-version-
  only storage. A binding's pin can only be VERIFIED against the
  CURRENT repository state, never re-read byte-for-byte once the
  repository has advanced past it — the bridge fails closed
  (`SourceVerificationError`) rather than silently substituting current
  state, but cannot recover the old content either. Base-asset bindings
  get a strictly stronger guarantee here than macro/cross-asset ones:
  Phase 2's `DatasetManifest.ordered_partition_ids` + current
  `PartitionStore` content lets `verify_base_asset_binding` reconstruct
  the exact CURRENT member-event-id set deterministically; Phase 4B/4C's
  manifests retain no member-id list at all (only a digest + counts), so
  macro/cross-asset verification is limited to digest reproduction. This
  is an existing `market_data` API characteristic, not something Phase
  4D could or should have changed (out of this phase's scope, and
  changing Phase 2/4B/4C's own storage contract was never authorized).
- **Cross-asset `roll_provenance`/`contract_metadata_id` (per-bar
  futures/continuation detail) are not propagated into the aligned
  OHLCV frame** by `cross_asset_adapter.resolve_cross_asset_dataframe`
  — only binding-level `instrument_form`/`adjustment_policy_id`/
  `continuation_policy_id` (policy identifiers, not per-bar roll detail)
  reach the research dataset's lineage. Entirely unexercised by this
  phase's own fixtures (ETF proxies only, matching Phase 4C's own
  shipped provider coverage) — a real gap only if a future phase wires
  in a genuine futures/continuous-series cross-asset driver.
- **No genuine `market_data`-backed higher-timeframe derivation for the
  SAME base instrument.** `MarketDataResearchDatasetRequest.higher_
  timeframe_data` accepts a caller-resolved frame unchanged (via a
  second `base_asset_adapter.resolve_base_asset_dataframe` call at a
  coarser timeframe) rather than deriving it automatically — spec
  Section 8 explicitly scoped this as "reuse M3's own higher-timeframe
  support unchanged," which this satisfies; automatic derivation was
  never in scope.
- **`SourceChangeEvidence` must be supplied by the caller from a
  separate `market_data` read** — `rebuild_planner.plan_rebuild` does
  not auto-compute it from two binding sets alone, since a bare
  component/dataset id carries no diff (see Section 18). This is a
  deliberate purity boundary (spec's own "pure planner" requirement),
  not an oversight.
- **`reconciliation.py`'s `ReconciliationIssueCode` enumerates several
  spec-named codes without a standalone, separately-invocable
  non-raising check behind every one of them** — wrong vintage selected,
  session-policy mismatch, and duplicate-aligned-coordinate-as-a-
  standalone-finding are detectable today only via the adapters' own
  fail-closed construction-time checks (Sections 7–10), not as an
  independent reconciliation pass a caller could run without also
  triggering the adapter's own raise. Disclosed here rather than
  silently omitted from the enum.
- **No CLI surface for the bridge** — `feature_cli.py` was not modified;
  a caller invokes `request.build_research_dataset_from_market_data`
  directly. Matches every `market_data` phase's own disclosed
  library-only limitation and was never in this phase's scope.
- **Performance figures (Section 23) are single-machine, moderate-scale
  measurements** (hundreds of bars/observations per source) — nothing
  here claims or has been measured at large-scale/many-symbol scale.

## 27. Exact git status and explicit confirmations

`HEAD` at the time of writing this report:
`5d4352ba772969d056a9336fa80f7566b339b822` (unchanged from Section 1's
baseline — no commit was made at any point in this phase).

`git status --short` at the time of writing this report shows exactly:
4 modified files (`src/quant_platform/core/exceptions.py`, `src/
quant_platform/features/dataset_builder.py`, `src/quant_platform/
features/manifests.py`, `src/quant_platform/historical/loader.py`), 2
modified docs (`docs/feature_engineering.md`, `docs/market_data_
architecture.md`), and untracked: the new `src/quant_platform/features/
market_data_bridge/` package (13 files), the new example config
(`examples/xauusd_point_in_time_research_dataset.example.json`), this
delivery report, and 21 new test files under `tests/unit/features/`
(including the shared, non-collected `_market_data_bridge_test_helpers.py`).
No other file is modified, added, or deleted anywhere in the working
tree. `git status -sb` shows `master...origin/master` with no ahead/
behind divergence markers beyond the pre-existing state.

**Explicit confirmations:**
- Phase 4D work is **not staged** (`git diff --cached` is empty; nothing
  was ever `git add`ed during this phase).
- Phase 4D work is **not committed** — `HEAD` is unchanged from the
  Section 1 baseline.
- **Nothing was pushed** at any point.
- **No second `FeatureRegistry`/`FeatureEngine`/`ResearchDatasetBuilder`**
  was created — every feature-computation and dataset-building code path
  in this phase terminates in a call to the real, pre-existing,
  unmodified classes (Section 13).
- **No network call, credential field, or live-trading/broker/MT5 code**
  was introduced anywhere in this phase — confirmed structurally by
  `test_market_data_bridge_safety_scan.py` (30 tests, network/broker/
  credential/live-trading/wall-clock/uuid4/pickle/eval/exec/shell-
  execution scans, each proven non-vacuous against a deliberately bad
  snippet) in addition to manual review of every new source file.
- **Milestone 11 was not started** — no file outside the scope listed in
  Section 2 was touched, and no work toward feature discovery, model
  training, deep learning, live prediction, MT5, broker integration, or
  production scheduling was performed.

## 28. Explicit stop confirmation

Per the governing specification's own final instruction: **this phase
stops here.** No feature discovery, model training, deep learning, live
prediction, MT5 integration, broker integration, production scheduling,
or Milestone 11 work has begun or will begin without further explicit
instruction. This delivery report, together with the updated
`docs/market_data_architecture.md`/`docs/feature_engineering.md` and the
new example config, is the complete Phase 4D deliverable awaiting
review and explicit commit approval — exactly mirroring the Phase 4A/4B/
4C precedent this session's own prior work established.
