# Label Infrastructure (Milestone 11, Phase 3)

This is the authoritative technical reference for `quant_platform.labels`
-- the deterministic, versioned, content-addressed, replayable,
auditable, point-in-time-safe framework every label family in this
platform is generated through.

**A label is not a model. A label is not a prediction. A label is
immutable scientific evidence derived from historical observations.**
This package never generates a predictive model, never evaluates model
quality, never performs feature selection, and never computes a
statistic that requires a prediction target (Information Coefficient,
Rank IC, Mutual Information, SHAP, Permutation Importance, Boruta,
Recursive Feature Elimination, or any correlation to a label). Those
belong to a later phase, once a model exists to evaluate.

Delivered in two parts:

- **Part A** -- infrastructure only. `LabelSpecification`, `LabelIdentity`,
  `LabelRegistry`, `LabelBuilder`/`LabelBundle` (a generic harness around
  a caller-supplied, pluggable generator), `LabelManifest`,
  `LabelDiagnostics`, `LabelVerifier`, `LabelReplay`, `LabelRecovery`,
  `LabelReconciliation`, and 7 reports. Treats 6 named label families
  (Next Return, Multi Horizon Return, Direction, Triple Barrier, Forward
  Volatility, Future Extension Placeholder) as first-class identity/
  versioning/reporting citizens, but ships zero concrete generation
  logic for any of them.
- **Part B** (this report's newest addition) -- the first 5 concrete
  label families' real generation logic, built entirely on top of Part
  A's infrastructure via the SAME pluggable-generator contract Part A
  established (`builder.LabelGeneratorFn`) -- never a parallel pipeline.
  Adds per-row `LabelRecord`/`LabelRecordLedger` (append-only recovery),
  and `CompositeLabelBundle` (deterministic multi-family groupings).
  `Future Extension Placeholder` still ships no generation logic --
  reserved, as its name says, for a future family.

## Relationship to `features.labels` (Milestone 3)

`quant_platform.features.labels` already exists and already computes
real label values (`FUTURE_RETURN`, `FUTURE_LOG_RETURN`,
`BINARY_DIRECTION`, `VOL_ADJUSTED_RETURN`, `TRIPLE_BARRIER`) embedded
directly into a `ResearchDatasetManifest`'s split columns at
dataset-build time -- it is the pre-existing, simple, single-shot label
system `ResearchDatasetBuilder` has always used. `quant_platform.labels`
is a deliberately SEPARATE, later, richer layer: it never imports
`features.labels` and never wraps it. Where `features.labels` bakes
exactly one label definition into a dataset at build time with no
identity/versioning/replay/audit trail of its own, `quant_platform.labels`
lets a researcher generate MULTIPLE independent, versioned label
families against the SAME already-built, already-qualified,
already-feature-discovered dataset, post-hoc -- each with its own
content-addressed identity, manifest, and full audit trail. The two
systems coexist; this phase does not modify, deprecate, or migrate
`features.labels`.

## Architecture

```
features.manifests.ResearchDatasetManifest (Milestone 3, unmodified, referenced only by identity string)
   |
   |  created_from_dataset / created_from_manifest -- plain strings, never a live object reference
   v
labels.models.LabelSpecification          -- content-addressed recipe metadata (no generation logic)
   |
labels.registry.LabelRegistry              -- append-only: register/lookup/freeze/version/compare/verify
   |
labels.builder.LabelBuilder                 -- generic harness: calls a CALLER-SUPPLIED generator, wraps
   |                                            the result in identity/bookkeeping (ships no generator itself)
   v
labels.builder.LabelBundle  +  labels.manifest.LabelManifest
   |
   +-- labels.diagnostics.compute_label_diagnostics   -- 7 structural dimensions, no predictive statistic
   +-- labels.verification.LabelVerifier               -- self-consistency + fresh re-derivation & diff
   +-- labels.replay.LabelReplay                         -- regenerate from scratch, prove byte-identical
   +-- labels.recovery.LabelRecovery                      -- replay-based recovery, fails closed, never guesses
   +-- labels.reconciliation.LabelReconciliation           -- diff two bundles/manifests for the same spec
   +-- labels.reports (render_*)                            -- 7 deterministic plain-text reports
```

Package layout:

```
src/quant_platform/labels/
    __init__.py          package docstring, mission, "Do NOT" list
    evidence.py            LabelEvidence, LabelDimensionKind (7), LabelEvidenceCode -- the atomic finding record
    models.py                LabelFamily (6), LabelSpecification, build_label_specification, compute_parameter_hash
    identity.py                 LabelIdentity, compute_label_identity -- content-addressed identity of a GENERATED bundle
    versioning.py                 LabelVersion, LabelVersionHistory -- append-only per-family history
    registry.py                     LabelRegistry -- register/lookup/freeze/versions/compare/verify/build_manifest/for_dataset
    builder.py                        LabelDefinition, LabelBuilder, LabelBundle -- the generic generation harness
    manifest.py                         LabelManifest, build_label_manifest -- self-contained lineage summary
    diagnostics.py                        LabelDiagnostics, LabelDimensionResult -- 7-dimension structural evaluation
    reconciliation.py                       LabelReconciliation -- diff two bundles/manifests for the same spec
    verification.py                           LabelVerifier -- self-consistency + full re-derivation
    replay.py                                   LabelReplay -- regenerate and prove byte-identical reproduction
    recovery.py                                   LabelRecovery -- replay-based recovery, fails closed
    reports.py                                      7 render_* functions: deterministic plain-text reports

    pricing.py            (Part B) PriceBasis (4), compute_forward_return -- shared point-in-time-safe return math
    volatility.py            (Part B) VolatilityEstimatorFn contract + 2 shipped, non-privileged estimators
    next_return.py              (Part B) Next Return: generate_next_return_labels, build_next_return_specification
    multi_horizon_return.py       (Part B) Multi Horizon Return: one independently-identified spec per horizon
    direction.py                    (Part B) Direction: UP/DOWN/NEUTRAL, configurable neutral_threshold
    triple_barrier.py                 (Part B) Triple Barrier: upper/lower/time barrier, volatility-scaled width
    forward_volatility.py               (Part B) Forward Volatility: pluggable-estimator realized volatility
    records.py                            (Part B) LabelRecord, LabelRecordLedger -- per-row identity + append-only recovery
    composite.py                            (Part B) CompositeLabelBundle -- deterministic multi-family "Label Bundles"
```

## Dependency isolation

`labels/` imports nothing from `features`, `qualification`,
`feature_discovery`, `ml`'s domain packages, or `paper_trading`/
`execution_gateway`/`portfolio_risk` -- only `core` exceptions/types,
`historical.quality.Severity` (an established, shared severity
vocabulary every evidence-model package in this platform already
reuses), `ml.persistence`'s dependency-neutral JSON helpers (the same
`as_json_dict`/`as_json_list`/`require_schema_version`/
`format_utc_timestamp`/`utc_now` `qualification` and `feature_discovery`
already import directly), and `pandas`/`numpy`. `created_from_dataset`/
`created_from_manifest`/`feature_identity`/`qualification_identity` are
plain caller-supplied identity strings, never live object references.

This is what makes the preferred dependency graph (Market Data ->
Features -> Qualification -> Feature Discovery -> Labels -> Machine
Learning) a WORKFLOW ordering rather than a Python import requirement --
a circular dependency between `labels` and anything downstream or
upstream of it is structurally impossible, since `labels` never imports
any of those packages in the first place.

## Label Specification (14 fields)

Every field the governing specification names, plus one implementation-
necessary supporting field (`parameters`):

| Field | Meaning |
|---|---|
| `label_specification_id` | Content-addressed sha256 over every other field -- the recipe's identity |
| `label_family` | One of the 6 named `LabelFamily` values |
| `schema_version` | JSON-shape version (currently `1`) |
| `generation_version` | Version of the (not-yet-implemented) generation ALGORITHM for this family |
| `parameter_hash` | sha256 over `parameters`' canonical JSON |
| `price_basis` | Which price series a real generator would read (descriptive) |
| `prediction_horizon` | Descriptive horizon text (e.g. "5 bars") |
| `availability_rule` | When the label itself becomes point-in-time-known (descriptive) |
| `reference_price` | Which specific observation anchors the label (descriptive) |
| `event_time_rule` | How the label's own "event time" is defined (descriptive) |
| `generation_rule` | Free-text description of the (not-yet-implemented) algorithm |
| `identity_algorithm` | Names the hashing scheme this id was computed with (currently `sha256-v1`) |
| `created_from_dataset` | The source `ResearchDatasetManifest.dataset_id` (plain string) |
| `created_from_manifest` | The source `ResearchDatasetManifest.version` (plain string) |
| `parameters` | The raw generation-parameter dict `parameter_hash` is computed over |

`build_label_specification(...)` is the ONE sanctioned constructor --
it computes `parameter_hash`/`label_specification_id` FROM the supplied
fields, so a spec built through it is self-consistent by construction.
A hand-constructed or deserialized instance is verified via
`LabelSpecification.verify_self_consistency()`, which independently
recomputes both hashes and compares.

**Versioning is append-only.** Changing prediction horizon, barrier,
price basis, neutral threshold, or volatility estimator always changes
`parameters` and therefore `parameter_hash` and therefore
`label_specification_id` -- a genuinely new specification, never a
mutation of an existing one (`LabelSpecification` is a frozen
dataclass; `LabelRegistry.register` refuses a duplicate id outright via
`DuplicateLabelSpecificationError`).

## Label Identity

`compute_label_identity(label_specification_id, values, *,
source_content_id)` produces a `LabelIdentity` whose `content_id` is a
sha256 over `(label_specification_id, source_content_id, row_count, the
values themselves IN ROW ORDER)` -- row order matters (labels are a time
series, never a set); `NaN` is a legitimate, expected value (an
unresolved row near the embargo edge) and encodes as JSON `null` rather
than being rejected or dropped. Depends only on its own arguments --
never the filesystem, wall clock, process id, `PYTHONHASHSEED`,
`random`, a temp directory, the machine, or the operating system,
exactly as the governing specification requires; proven directly by
`test_labels_determinism.py`'s subprocess `PYTHONHASHSEED` proof.

## Label Registry

Append-only, keyed by `label_specification_id`. Responsibilities named
by the governing specification, each a real method: `register` (refuses
tampered specs via `LabelIdentityError`, refuses duplicates via
`DuplicateLabelSpecificationError`), `lookup` (raises
`UnknownLabelSpecificationError`), `freeze` (marks a lifecycle state --
registration alone already makes a spec immutable; freezing is an
explicit "no longer provisional" signal for downstream reports/lineage
to check), `versions` (returns a `LabelVersionHistory` for one family,
sorted, never registration-order-dependent), `compare` (a field-by-field
diff of two registered specs), `verify` (self-consistency), plus
manifest integration (`build_manifest`) and dataset integration
(`for_dataset`).

## Label Builder -- the generic harness, never a generation algorithm

`LabelBuilder.build(definition, source_data, *, source_content_id)`
calls `definition.generate` (the caller-supplied, pluggable
`LabelGeneratorFn`) and validates its OUTPUT structurally: it must be a
`pd.Series`, the same length as `source_data`, numeric dtype, and must
not share underlying memory with any column of `source_data`
(`numpy.shares_memory`, checked per column -- never Python `is`
identity, which pandas' column-access machinery does not reliably
preserve across repeated `df[col]` calls). A violation of any of these
raises `LabelGenerationContractError`/`LabelMutableAliasError`
immediately, fail-closed. **It never evaluates whether the VALUES are
scientifically correct for the declared family** -- Part A has no
family-specific logic to check them against; that arrives with Part 2's
real generators.

The mutable-alias guard exists because a generator that mistakenly
returns `source_data["close"]` unchanged would silently corrupt an
already-built `LabelBundle` the moment `source_data` is later mutated,
violating this package's immutability guarantee.

## Label Manifest

Self-contained: a consumer never needs to cross-reference
`LabelRegistry` or re-read the specification. `feature_identity`/
`qualification_identity` are plain, optional, caller-supplied identity
strings (e.g. a `ResearchDatasetManifest.feature_registry_fingerprint`,
or a hash of a `DatasetQualificationReport`) -- this module never
computes them itself, for the dependency-direction reason given above.
`dependency_chain` renders the Market Data -> Features -> Qualification
-> Feature Discovery -> Labels lineage as an ordered tuple of identity
references -- a linear chain, never a branching graph, since exactly one
specification produces exactly one manifest with exactly one upstream
lineage. `manifest_checksum` is a sha256 over every other field,
deliberately EXCLUDING `generation_timestamp` (wall-clock, non-identity)
so two manifests built at different times for the identical
specification/lineage always checksum identically.

## Label Diagnostics -- 7 structural dimensions

`compute_label_diagnostics(bundle, manifest)` evaluates, in fixed order
(`LABEL_DIMENSION_ORDER`): **Identity** (fresh recomputation matches),
**Versioning** (specification self-consistency), **Availability** (the
5 point-in-time rules this package can independently verify -- see
below), **Manifest Integrity** (fresh checksum recomputation matches),
**Determinism** (two independent identity recomputations over the same
values agree), **Reproducibility** (a recognized `identity_algorithm`),
**Lineage** (required lineage fields present). No dimension evaluates
the scientific quality, predictive value, or correctness of a label's
VALUES.

### Point-in-time rule mapping (disclosed scope boundary)

The governing specification names 7 point-in-time rules. Two of them
("no future macro release", "no future cross asset") are NOT
independently verifiable from a bundle alone -- doing so would require
re-reading raw macro/cross-asset source data, which is out of this
package's scope (it never imports `market_data`/`features`).
**Availability** reports this honestly as an INFO-severity, non-blocking
finding rather than fabricating a check it cannot actually perform --
the same disclosed-scope-boundary discipline `qualification`'s own
"Macro/cross-asset scope" section established. "No revised data" is out
of scope for the identical reason. The other 4 rules ARE checked for
real: "no mutable aliases" (enforced by `LabelBuilder` at construction
time, confirmed here), "no wall clock semantics" (folded into
**Determinism** -- identity excludes `generated_at` by construction),
"no future visibility"/"no unavailable observation" (the trailing-NaN-
tail shape heuristic: a forward-looking label's only legitimate NaN
source in this infrastructure-only phase is "not enough future data
yet," which always produces a single TRAILING run of NaN, never a NaN
"hole" followed by more valid data -- a documented heuristic, not a
proof about any specific family's semantics).

## Verification, Replay, Recovery

All three mirror `qualification`/`feature_discovery`'s established
"verify, never trust" pattern:

- **`LabelVerifier.verify`** -- `verify_bundle_self_consistency` (pure,
  no I/O: recomputes identity/specification/manifest self-consistency)
  plus a full fresh `LabelBuilder.build()` re-derivation, diffed via
  `LabelReconciliation`. A mismatch is a normal, non-raising
  `verified=False`; `LabelVerificationError` is reserved for genuinely
  being unable to attempt re-derivation at all.
- **`LabelReplay.replay`** -- the INVARIANTS section ("changing nothing
  -> same labels, same hashes, same manifests, same reports") promoted
  to a first-class module: regenerates from scratch and proves
  byte-identical reproduction against an `original` bundle.
- **`LabelRecovery.recover`** -- recovers a lost/corrupted bundle by
  REPLAYING from the original specification, generator, and source data
  -- never by guessing. If the evidence needed to replay is unavailable,
  or a regenerated bundle does not match a supplied `expected_identity`,
  recovery FAILS CLOSED (`recoverable=False`, `recovered_bundle=None`)
  rather than returning a best-effort result. Mirrors
  `PortfolioRiskRecoveryError`'s own "surfaced rather than guessed"
  discipline.

## Reconciliation

`LabelReconciliation.reconcile(baseline, candidate, *,
baseline_manifest, candidate_manifest)` detects 4 drift kinds for two
bundles sharing the SAME `label_specification_id`: `specification_drift`
(the `LabelSpecification` itself differs -- a tamper signal, since a
matching id should imply matching content), `identity_drift` (`content_id`
differs -- the values changed), `manifest_drift` (`manifest_checksum`
differs), `lineage_drift` (dataset/manifest/feature/qualification
identity differs). Reconciling two bundles for DIFFERENT specifications
is a structural precondition violation (there is nothing to reconcile)
and raises `LabelReconciliationError`; every other disagreement is a
normal, non-raising `LabelReconciliationIssue`.

## Reports

7 deterministic, sorted, plain-text renderers (`reports.py`), mirroring
`qualification.reports`/`feature_discovery.reports`'s established
convention: Specification Report, Manifest Report, Bundle Report,
Diagnostics Report, Verification Report, Reconciliation Report, Version
History Report. Every renderer performs no discovery/verification/
reconciliation logic of its own -- it renders already-computed objects
only.

## Part B: the 5 concrete label families

Every family below is implemented INDEPENDENTLY -- none reads another
family's generated output, even where two families share a pure
computational primitive (`pricing.compute_forward_return`). Every
family ships a `generate_*` function matching Part A's
`builder.LabelGeneratorFn` contract exactly (`(source_data,
specification) -> pd.Series`) and a `build_*_specification` function
that assembles a well-formed `LabelSpecification` via
`models.build_label_specification` -- neither module invents a second
identity scheme.

### Shared primitives (`pricing.py`, `volatility.py`)

`pricing.PriceBasis` (`CLOSE_TO_CLOSE`, `OPEN_TO_CLOSE`,
`CLOSE_TO_OPEN`, `MID_TO_MID`) and `pricing.compute_forward_return`
resolve the entry/exit price pair and compute `exit.shift(-horizon_bars)
/ entry - 1.0` -- point-in-time safe by construction (row `t` reads
EXACTLY row `t + horizon_bars`, never beyond it). Every family that
needs a forward return (Next Return, Multi Horizon Return, Direction)
calls this SAME function; reusing it is not "one family depending on
another's output" -- no family ever reads another's generated VALUES,
they simply share pure math.

`volatility.VolatilityEstimatorFn` is a plain structural contract
(`(source_data, window_bars) -> rolling volatility series`); no
estimator is privileged. Two shipped, genuinely different
implementations prove real pluggability: `realized_stddev_estimator`
(trailing stddev of close-to-close returns) and
`realized_parkinson_estimator` (the Parkinson 1980 high/low-range
estimator) -- different inputs, different formulas, interchangeable
through the same contract. `triple_barrier.py` and
`forward_volatility.py` both accept an estimator BY NAME
(`resolve_estimator_by_name`), so `LabelSpecification.parameters`
stays JSON-safe while still recording exactly which estimator produced
a given bundle.

### 1. Next Return (`next_return.py`)

`generate_next_return_labels` is `pricing.compute_forward_return`
applied with an explicit, REQUIRED `price_basis` (no default -- a
caller must always name Close->Close, Open->Close, Close->Open, or
Mid->Mid) and `horizon_bars`.

### 2. Multi Horizon Return (`multi_horizon_return.py`)

The SAME return computation as Next Return, independently parameterized
per horizon. `build_multi_horizon_return_specifications(horizons=...)`
returns one `LabelSpecification` PER horizon -- "horizons belong to
`LabelSpecification`" means each horizon is its own, independently
identified specification, never one label smeared across many
horizons. `MULTI_HORIZON_RETURN_MINIMUM_HORIZONS = (1, 5, 10, 20, 50,
100)` names the minimum required set; the function accepts ANY
`horizons` tuple ("no hardcoded assumptions").

### 3. Direction (`direction.py`)

UP (`1.0`)/DOWN (`-1.0`)/NEUTRAL (`0.0`), thresholded off the same
forward-return primitive. `neutral_threshold` is a REQUIRED, explicit
argument (no default) and always lands in `parameters`, so it
participates in `parameter_hash` -> `label_specification_id` by
construction -- any threshold change is automatically a new,
independently identified specification.

### 4. Triple Barrier (`triple_barrier.py`)

+1 if the upper barrier is touched before the lower one within
`max_holding_bars`; -1 if the lower touches first (or both touch within
the same forward bar -- OHLC alone cannot say which happened first
within one bar, so this resolves the tie toward the stop-like outcome);
the SIGN of the terminal return (the "time barrier") if neither is
touched. Barrier width is `close * (1 +/- multiplier *
trailing_volatility)` -- `profit_multiplier`/`loss_multiplier` scale a
PAST-only trailing volatility estimate (via the same pluggable
`volatility.VolatilityEstimatorFn` contract, applied directly/
un-shifted so row `t`'s barrier depends only on rows `<= t`) named by
`volatility_estimator_reference`. `NaN` where trailing volatility is
unavailable (insufficient warmup) or the full holding horizon's data
does not exist. Deliberately REIMPLEMENTED natively rather than
importing `features.labels.build_triple_barrier_labels` -- see
"Relationship to `features.labels`" above; the two systems never share
code, only a similar shape.

### 5. Forward Volatility (`forward_volatility.py`)

The realized volatility of the NEXT `horizon_bars` bars: the chosen
estimator's own trailing rolling statistic, shifted `-horizon_bars` so
row `t`'s value covers exactly rows `[t+1, t+horizon_bars]` -- never
reaching beyond the configured horizon, never including row `t` itself.
No estimator is privileged; both shipped estimators are equally usable.

## Label Records (`records.py`) -- per-row identity and Recovery

The governing specification requires every GENERATED label carry
`label_id`, `label_specification_id`, `dataset_id`, `row_identity`,
`event_time`, `availability_time`, `generation_version`, `content_hash`
-- `records.materialize_label_records(bundle, source_data, *,
dataset_id, timeframe, horizon_bars)` produces one `LabelRecord` per row
FROM an already-built `builder.LabelBundle`, never a second generation
pass.

- **`row_identity`** is the row's `event_time` rendered as an ISO-8601
  UTC string -- content-based and portable, never a positional
  DataFrame index (which says nothing about WHICH row it is once rows
  are re-ordered, filtered, or reloaded). By construction `row_identity
  == event_time` always; `LabelRecord.verify_self_consistency` checks
  this, catching a "future timestamp" tamper that bumps one field
  without the other.
- **`event_time = open_time + timeframe.duration`** (the bar's CLOSE
  time -- mirrors `features.interfaces.FeatureContext`'s identical
  convention). **`availability_time = event_time + timeframe.duration *
  horizon_bars`** -- the WORST-CASE instant this label is guaranteed
  knowable; a label that resolves earlier (an early triple-barrier
  touch) is still conservatively marked available only this late.
  Neither field ever reads the wall clock -- both are pure functions of
  `source_data["open_time"]`.
- **`label_id = sha256(label_specification_id | row_identity)`**
  identifies WHICH label slot this is, independent of its value.
  **`content_hash = sha256(label_id | value)`** additionally covers the
  value, so a value change alone (identity unchanged) is still
  detectable.
- `materialize_label_records` cross-checks its own `dataset_id`
  argument against `bundle.specification.created_from_dataset`, raising
  `LabelRequestError` on a mismatch -- a real "dataset corruption" guard
  added specifically because nothing else in the call chain would have
  caught a caller passing a record set at the wrong dataset.

**`LabelRecordLedger`** is the append-only commit tracker "Recovery"
requires: `commit(records)` refuses (raises
`LabelRecordConflictError`, ALL-or-nothing -- a batch containing even
one already-committed `row_identity` is refused in full, nothing
partially applied) to overwrite an already-committed `row_identity` for
a given specification; `recover(candidates)` returns only the subset
NOT yet committed, so resuming an interrupted generation run re-derives
candidates for the whole requested range but only acts on the genuinely
missing ones -- "never regenerate partially committed labels." In-memory
only, matching this package's own "no persistence store yet" scope (see
Known Limitations) -- the APPEND-ONLY CONTRACT is what Recovery
requires, not durable storage.

## Label Bundles (`composite.py`)

`CompositeLabelBundle` groups several independently-generated,
independently-identified `builder.LabelBundle`s (Return + Direction,
Return + Volatility, Direction + Triple Barrier, Return + Direction +
Volatility, or any other combination) under one content-addressed
`composite_id` (`sha256(dataset_id, sorted member content_ids)`).
`build_composite_from_definitions` builds every member via Part A's own
`LabelBuilder` (one call per definition) and groups the results --
never a parallel generation path.

`verify_composite`/`replay_composite`/`reconcile_composite` are thin
aggregations over Part A's own single-bundle `LabelVerifier`/
`LabelReplay`/`LabelReconciliation` -- one call per member, results
collected, never a parallel reimplementation of any of the three.
`reconcile_composite` additionally detects `member_set_drift` (members
added/removed) and `bundle_drift` (`composite_id` changed) at the
composite level, on top of each member's own drift kinds.

## A real defect found and fixed during testing

`LabelBundle` (a frozen dataclass) originally carried its default,
dataclass-generated `__eq__`, which compares every field with bare
`==` -- including `values: pd.Series`. Comparing two Series with `==`
returns an element-wise boolean Series, not a bool; the moment that
ambiguous Series is coerced to a bool (exactly what happens when ANOTHER
dataclass containing a `LabelBundle | None` field, such as
`LabelRecoveryResult`, uses ITS OWN generated `__eq__`), pandas raises
`ValueError: The truth value of a Series is ambiguous`. Found by
`test_labels_recovery.py::TestLabelRecovery::test_json_round_trip`.
Fixed by giving `LabelBundle` `eq=False` and a hand-written `__eq__`
that compares every field normally except `values`, which it compares
via `Series.equals` (the correct, NaN-aware, position-aware
comparison) -- and `__hash__ = None`, making `LabelBundle` explicitly
unhashable rather than silently inheriting an identity-based hash
inconsistent with the new `__eq__`. This is the only dataclass in this
package carrying a `pd.Series`/`pd.DataFrame` field; every other type is
JSON-primitive-only and needed no such treatment (confirmed by grep).

## Real defects found and fixed during Part B's development

**`materialize_label_records` did not cross-check its `dataset_id`
argument against the bundle's own specification.** Discovered while
designing the "dataset corruption" adversarial scenario: nothing in the
call chain would have caught a caller passing a `LabelRecord` batch
tagged with the WRONG dataset id (`created_from_dataset` on the
specification and `dataset_id` on the materialized records could
silently disagree). Fixed by raising `LabelRequestError` the moment
`materialize_label_records` is called with a `dataset_id` that does not
match `bundle.specification.created_from_dataset`
(`test_labels_phase3b_adversarial.py::Test11DatasetCorruption`).

**`LabelRecord.verify_self_consistency` did not check `row_identity`
against `event_time`**, even though the two are ALWAYS identical by
construction (`row_identity = event_time.isoformat()`). A hand-tampered
record bumping one field without the other (a "future timestamp"
attack) went undetected. Fixed by adding the cross-check
(`test_labels_phase3b_adversarial.py::Test01FutureTimestamp`).

## Known limitations

- **`Future Extension Placeholder` still ships no generation logic** --
  by design; the family exists purely as a registration/identity
  placeholder for whichever family is added next.
- **Two of the 7 point-in-time rules are disclosed as out-of-scope** --
  see "Point-in-time rule mapping" above. Re-verifying them requires
  real macro/cross-asset source data this package deliberately never
  reads.
- **The trailing-NaN-tail shape check is a heuristic, not a proof** --
  documented in its own function docstring (`diagnostics.
  _trailing_nan_tail_is_well_formed`), not silently presented as
  definitive.
- **Triple Barrier's tie-break (both barriers touched within the same
  forward bar) resolves toward the stop-like (-1) outcome** -- OHLC
  data alone cannot say which barrier was touched first intrabar; this
  is a documented, conservative convention, not a claim of intrabar
  ordering knowledge this package does not have.
- **`LabelRecordLedger` is in-memory only** -- matches this package's
  own "no persistence store yet" scope; the append-only CONTRACT
  ("Recovery") is implemented, durable storage is not.
- **No CLI surface** -- library only, matching every prior milestone's
  own disclosed pattern before a CLI phase followed.
- **Not yet wired into a persistence store** -- this package produces
  in-memory dataclasses with `to_json_dict`/`from_json_dict`; nothing
  here writes to `ml.persistence`'s artifact store, matching
  `qualification`'s and `feature_discovery`'s own disclosed limitation.
- **No statistic in this package depends on a prediction target or a
  label's own VALUE quality** -- by design; Information Coefficient,
  Rank IC, Mutual Information, Feature/Permutation Importance, SHAP,
  Boruta, RFE, and correlation-to-labels are explicitly out of scope.
