# Feature Engineering & Research Dataset Platform (Milestone 3)

This is the authoritative technical reference for `quant_platform.features`
and `quant_platform.feature_cli` -- the point-in-time-correct feature
engineering and ML-ready research dataset platform built on top of
Milestone 1's backtesting engine and Milestone 2's historical data
pipeline. The README gives the one-paragraph overview; this document gives
the contract each stage makes and the reasoning behind it.

**This layer produces research datasets. It does not produce a trading
edge or a predictive model.** Nothing here trains, fits, or evaluates a
model; it stops at "leak-free features and labels, correctly split,
train-only-fit preprocessing applied, written as an immutable, versioned,
fully-lineaged artifact." Whether a model trained on that artifact is any
good is out of scope, same as Milestone 2's "a leak-free dataset is
necessary, not sufficient" framing.

## Architecture

```
historical.loader.DatasetLoader (Milestone 2, exact version reconstruction)
   v
features.registry.FeatureRegistry        -- feature definitions, dependency graph
   |
features.engine.FeatureEngine            -- point-in-time feature computation
   |    uses: features.alignment (higher-timeframe / as-of external joins)
   |          features.missing (structural null policies)
   v
features.labels.build_label              -- SEPARATE, future-aware label construction
   v
features.splitting                       -- chronological / walk-forward splits
   v
features.missing + features.normalization -- train-only imputation/scaling, per fold-group
   v
features.validation.validate_research_dataset -- structural leakage/quality gate
   v
features.manifests.ResearchDatasetStore/ResearchManifestStore -- immutable, content-addressed artifacts + manifest
```

`features.dataset_builder.ResearchDatasetBuilder` is the single
orchestrator that runs this whole pipeline in this fixed order; see its
module docstring for the exact ten-step sequence and why the order itself
is the leakage guarantee.

Package layout:

```
src/quant_platform/features/
    interfaces.py       FeatureContext, Feature protocol, FeatureDefinition
    models.py           FeatureSpec, FeatureCategory, MissingPolicySpec/Kind
    registry.py         FeatureRegistry: register/get/dependency-order/fingerprint
    engine.py           FeatureEngine: DAG execution, leakage guards
    lineage.py          FeatureLineage, human-readable lineage report
    alignment.py        align_higher_timeframe, as_of_join_external
    missing.py          apply_missing_policy, compute_training_statistic
    normalization.py    fit_transform/apply_transform, TransformPipeline
    labels.py           ISOLATED label construction (future_return, triple_barrier, ...)
    splitting.py        chronological / expanding / rolling walk-forward splits
    validation.py       validate_research_dataset -- structural leakage/quality report
    drift.py            descriptive train/val/test drift diagnostics (PSI, correlation)
    manifests.py        content-addressed ResearchDatasetStore + ResearchManifestStore
    dataset_builder.py  ResearchDatasetBuilder -- the ten-step orchestrator
    technical/price.py       returns, ATR, z-score, MA distance, candle ratios, ...
    temporal/calendar_features.py  hour/day/month, session flags, cyclical encodings
    multi_timeframe.py        higher-timeframe OHLCV/return/volatility/trend-distance
    cross_asset/cross_asset.py  another instrument's return/momentum/vol/correlation
    macro/macro_features.py     release-time-gated macro level/change/age/staleness
```

## Point-in-time semantics (the highest-priority guarantee)

A feature computed for base-timeframe row `i` may use:

- `ctx.base_df` rows `0..i` only (trailing windows via `technical.price.
  trailing_rolling`, which pins `center=False` and `min_periods=window`
  explicitly -- never pandas' rolling defaults).
- `ctx.higher_timeframe_data`/`ctx.cross_asset_data` bars whose own CLOSE
  time is `<= ` row `i`'s availability instant (`open_time + timeframe.
  duration`), via `alignment.align_higher_timeframe` -- never a raw join
  on `open_time`.
- `ctx.macro_data` observations whose RELEASE time is `<=` row `i`'s
  availability instant, via `alignment.as_of_join_external` -- never the
  observation's own reference period.
- `ctx.resolved_features`, other already-computed features' output, in
  dependency-resolved order (`registry.FeatureRegistry.
  resolve_dependency_order`), so composition never breaks the guarantee.

**The general-purpose proof used throughout the test suite**:
`tests/unit/features/test_technical_price.py::TestTruncationInvariance`
recomputes an entire feature set against a truncated PREFIX of the base
data and asserts every row up to the truncation point is byte-identical to
the full computation. Any feature that peeked at a future row fails this
test immediately -- it is a single, general leak detector covering the
whole technical family at once, not a per-feature spot check.

**What is explicitly protected against** (Milestone 3 Section 2's list,
and where each is enforced):

| Threat | Enforced by |
|---|---|
| Future candles | `engine.FeatureEngine` truncation-invariant by construction (trailing windows only); proven by `test_technical_price.py` |
| Incomplete higher-timeframe bars | `alignment.align_higher_timeframe` (close-time gated); cross-validated against `multiframe.cursor.TimeframeCursor` |
| Future macro releases | `alignment.as_of_join_external` (release-time gated, backward-only `merge_asof`) |
| Revised macro values | as-of join naturally vintage-aware: a later release simply supersedes an earlier one at its OWN release instant, never retroactively |
| Centered rolling windows | `technical.price.trailing_rolling` pins `center=False` explicitly; no function in this package calls `.shift()` with a negative period |
| Backward-filling from future observations | `missing.py` has no backward-fill code path at all -- structurally absent, not merely discouraged |
| Forward-looking resampling | `alignment.py` never uses `pandas.DataFrame.resample`; bucket/join logic is explicit close-time arithmetic |
| Timezone conversion leakage | all internal comparisons are tz-aware UTC; `historical.timezones`'s DST-rejection policy is inherited unchanged |
| Publication-time leakage | `as_of_join_external` keys strictly on `release_time`, never `observation_time` |
| Target/label leakage | `engine.FeatureEngine.compute` refuses any input DataFrame containing a `label_`/`target_`-prefixed column (`LabelLeakageError`); `features.labels` is never imported by any feature module (statically verified in `test_labels.py`) |

`core.exceptions.PointInTimeViolationError` is raised for structural
violations (unsorted/duplicate base data, unsorted higher-timeframe data);
`LabelLeakageError` for the label-isolation guard specifically.

**Honest limitation**: nothing stops a badly-written CUSTOM feature from
calling `ctx.base_df["close"].shift(-1)` directly -- `FeatureContext`
hands a feature its inputs, and Python cannot forbid misuse of an array a
function already holds. The guarantee is that the STANDARD library never
does this (proven by truncation invariance) and that labels specifically
cannot reach a feature (structural, not just conventional). A "cheating"
feature is directly demonstrated in
`tests/unit/features/test_dataset_builder.py::TestValidationGate` --
precisely so `validate_research_dataset`'s target-leakage-correlation
check has a real, working example to catch, and so this limitation is
demonstrated rather than merely asserted.

## Feature interface, registry, and lineage

`interfaces.FeatureContext` bundles everything a feature's `compute`
callable receives: `base_df`, `symbol`, `timeframe`, precomputed
`availability_times`, and the auxiliary `higher_timeframe_data`/
`cross_asset_data`/`macro_data` mappings plus `resolved_features` (filled
in as the engine proceeds).

`models.FeatureSpec` is pure, JSON-serializable metadata (name, version,
description, category, required raw input columns, source symbols/
timeframe, output dtype, lookback/warmup bars, availability delay, null
policy, deterministic params, feature-on-feature dependencies) -- see its
own docstring for why `(name, version)` bumping, not in-place mutation, is
the only sanctioned way to change a feature's behavior (this is what makes
"changing feature configuration changes the dataset ID" true).
`FeatureSpec.fingerprint()` is a deterministic sha256 of the sorted JSON
representation.

`registry.FeatureRegistry`:

- `register(definition)` -- append-only; raises `DuplicateFeatureError`
  on a repeated `(name, version)`; raises `UnknownFeatureError` if a
  declared `feature_dependencies` entry isn't registered yet (dependencies
  must be registered before dependents).
- `get(name, version=None)` -- `None` resolves to the latest-registered
  version for that name.
- `resolve_dependency_order(names)` -- DFS topological sort over the
  closure of `names` and their transitive dependencies; raises
  `CyclicFeatureDependencyError` on a cycle. A cycle is reachable even
  though registration requires pre-existing dependencies, because
  dependencies resolve by NAME (latest version) at resolution time, not a
  pinned version -- registering a NEWER version of an earlier feature that
  now depends on something depending on it retroactively creates one; see
  `test_registry.py::test_cycle_created_via_version_bump_is_detected` for
  the concrete adversarial scenario.
- `fingerprint(names=None)` -- deterministic sha256 over the sorted,
  JSON-serialized selected `FeatureSpec`s; independent of registration
  order (proven directly in `test_registry.py`).

`lineage.build_lineage(spec, source_dataset_manifest_id, transformation)`
produces a `FeatureLineage` (source dataset, symbol/timeframe, required
inputs, transformation description, parameters, dependencies, lookback/
availability contract, spec fingerprint) for every computed feature;
`render_lineage_report` gives a sorted, human-readable text report (also
exposed via `feature_cli inspect-lineage`).

## Core feature library (Section 4)

A deliberately small, curated set -- not hundreds of arbitrary indicators.

- **Technical/price** (`technical/price.py`): simple/log returns,
  momentum, rolling volatility, true range/ATR, rolling high/low distance,
  candle body/upper-wick/lower-wick ratios, rolling z-score, moving-average
  distance, rolling volume mean/std, optional rolling spread mean (when a
  `spread` column is present). All parameterized via `TechnicalWindows`.
- **Temporal** (`temporal/calendar_features.py`): hour of day, day of
  week, month, sin/cos cyclical encodings (so hour 23 and hour 0 are
  numerically adjacent), and -- only if a `historical.calendar.
  TradingCalendar` is supplied -- session-open flag, minutes-since-session-
  open, and market-open proximity. The latter two require the calendar to
  define exactly ONE weekly session (a documented, deliberate scope
  limit); minute-offset arithmetic is DST-safe because it converts through
  `calendar.local_tz` per row rather than assuming a fixed UTC delta (see
  `test_temporal_features.py::TestDSTHandling` for both the 2024
  spring-forward and fall-back transitions exercised directly).
- **Multi-timeframe** (`multi_timeframe.py`): last completed higher-
  timeframe OHLCV, return/volatility/trend-distance computed AT THE
  HIGHER TIMEFRAME'S OWN CADENCE first and only then aligned into the base
  timeframe (see "why ordering matters" below), plus elapsed seconds since
  that bar's close.
- **Cross-asset** (`cross_asset/cross_asset.py`): another instrument's
  aligned return/momentum, a base-vs-cross volatility ratio, and rolling
  correlation between base and cross-asset 1-bar returns. Requires the
  cross-asset DataFrame to have a `close` column (raises
  `FeatureComputationError` with an actionable message otherwise).
- **Macro** (`macro/macro_features.py`): release-time-gated level,
  period-over-period change, release-age (in days), and an explicit
  `is_stale` flag when no qualifying release exists (or none within a
  configured `tolerance`).

### Why higher-timeframe/cross-asset return ordering matters

Aligning a raw higher-timeframe `close` into the base timeframe first
necessarily REPEATS the same value across every base bar until the next
higher-timeframe bar closes. Taking a rolling return/diff on that
already-upsampled series measures change over N BASE bars of mostly
repeated values, not N higher-timeframe bars -- a different (and wrong)
quantity. `multi_timeframe.py` and `cross_asset.py` both compute
return/volatility/trend-distance on the source bar sequence FIRST, at its
own native granularity, and only then align the already-computed columns
into the base timeframe (`alignment.align_higher_timeframe` treats them as
just more columns to carry through unchanged). See
`test_multi_timeframe.py::TestReturnComputedAtNativeCadence` for a
hand-crafted numeric proof of the specific bug this avoids.

## Multi-timeframe alignment (Section 5)

`alignment.align_higher_timeframe` is the vectorized (`numpy.searchsorted`)
batch equivalent of `multiframe.cursor.TimeframeCursor.advance_to`: for
every base row, it finds the LAST higher-timeframe bar whose own close
time is `<=` that row's availability instant. It is cross-validated
bar-by-bar against `TimeframeCursor` itself in
`test_alignment.py::TestCrossValidationAgainstTimeframeCursor` (five
random seeds at M1->H1 and one at M5->D1), and covers the explicit
boundary cases Section 5 calls out directly:

- 59 minutes into an H1 candle -> the PREVIOUS H1 bar is visible, not the
  still-forming one (`test_59_minutes_into_h1_candle_not_yet_visible`).
- Exact H1 close -> the bar becomes visible at that exact instant
  (`test_exact_h1_close_reveals_the_bar`).
- One second before close -> still not visible
  (`test_one_second_before_close_does_not_reveal_the_bar`).
- A D1 bar crossing a UTC day boundary -> not revealed a moment before
  midnight UTC (`test_daily_close_crossing_utc_boundary`).
- Sparse/missing higher-timeframe bars -> the LAST bar before a gap
  carries forward; nothing is fabricated for the missing bar
  (`test_sparse_missing_higher_timeframe_bar_carries_forward_not_fabricated`).
- Empty higher-timeframe data -> every row gets the warm-up sentinel
  (`bar_index=-1`, `NaT` open/close time), never a crash or a silent zero.

`align_higher_timeframe` requires both merge keys normalized to the same
datetime64 storage resolution before comparison -- a real bug found and
fixed during development: `pandas.date_range`/Parquet round-trips can
produce `datetime64[ns]` vs `datetime64[us]` for what is semantically the
identical UTC instant, and comparing raw `.astype("int64")` epoch values
across differing resolutions silently compares wrongly-scaled integers
(not a crash -- a wrong answer). Both `align_higher_timeframe` and
`as_of_join_external` now explicitly normalize to `datetime64[ns, UTC]`
first.

## External and macro data alignment (Section 6)

`alignment.as_of_join_external` is a backward-looking (`direction=
"backward"`) `pandas.merge_asof`, keyed on an explicit `release_time_column`
(never the observation's own reference period), with an optional
`tolerance` for staleness. Duplicate release timestamps (e.g. a same-
instant correction) resolve deterministically: the input is stable-sorted
by release time first, so among exactly-tied releases the one appearing
LATER in the input order wins for any query at or after that instant.

**The Section 6 required proof**: a value observed for January but first
released in February must not appear in a January feature row. Exercised
directly in `test_alignment.py::TestAsOfJoinExternal::
test_revision_only_visible_after_its_own_release` (a preliminary February
release followed by a March revision -- the January-onward feature row
before ANY release is null; the row after the February release shows the
preliminary value; the row after the March revision shows the revised
value, never retroactively) and again through the full registered-feature
path in `test_macro_features.py::
test_january_value_not_visible_before_february_release`.

## Missing-value policies (Section 7)

`models.MissingPolicyKind`: `PRESERVE_NULL`, `FORWARD_FILL_MAX_AGE`,
`CONSTANT_FILL`, `TRAINING_STATISTIC_FILL`, `DROP_ROW`. **There is no
backward-fill option anywhere in this enum or in `missing.
apply_missing_policy`'s implementation** -- structurally absent, not a
restricted/discouraged path (`test_missing.py::
test_no_backward_fill_option_exists` asserts the enum's exact member set).

`FORWARD_FILL_MAX_AGE` nulls back out any fill more than `max_age_bars`
positions stale relative to the last real observation -- a leading run of
nulls (before ANY valid value) stays null regardless of `max_age_bars`,
since there is nothing to forward-fill FROM (never backward).

`TRAINING_STATISTIC_FILL` never computes its own statistic:
`apply_missing_policy` REQUIRES an externally-supplied `fitted_statistic`
for this policy, and `missing.compute_training_statistic` is the only
function that produces one -- callers (`dataset_builder`) are responsible
for calling it ONLY against a training-partition slice.
`engine.FeatureEngine.compute` deliberately does NOT apply this policy
(features are computed across the full requested range, before any split
exists yet); `dataset_builder.ResearchDatasetBuilder` applies it
AFTER splitting, per fold-group (see below).

## Train-only preprocessing (Section 8)

`normalization.py` implements `STANDARD_SCALE`, `ROBUST_SCALE`,
`WINSORIZE`, and `SIGNED_LOG1P` as lightweight internal transforms (no new
dependency -- `pyproject.toml` is unchanged). `fit_transform` computes
frozen parameters from whatever series it is given; `apply_transform`
is a pure function of already-frozen parameters and never recomputes
anything. `TransformPipeline.fit()` raises `PreprocessingLeakageError` if
called a second time without an explicit `allow_refit=True` -- turning
"the validation split accidentally got fit on" into a loud, immediate
failure. `TransformPipeline.fingerprint()` is deterministic and sensitive
to both the fitted data and the chosen transform kind.

**The central adversarial proof** (`test_normalization.py::
test_fitted_parameters_never_reflect_validation_data`): fit on a
constant-mean-1.0 train set, apply to a validation set with mean ~20000 --
the scaled validation output must still reflect the TRAIN mean/std, never
the validation set's own statistics.

### Per-fold-group fitting (walk-forward's real leakage trap)

For a `chronological` split there is exactly one train split, so fitting
is a single pass. For expanding/rolling WALK-FORWARD plans, each fold's
train range is progressively larger and, critically, OVERLAPS earlier
folds' test range (fold `j`'s test data legitimately becomes part of fold
`j+1`'s training data once "time has passed" -- the defining behavior of
expanding-window CV). **Fitting one GLOBAL pipeline across the union of
every fold's train rows would fit on data that includes an earlier fold's
own test rows, leaking into that fold's evaluation.** This was found and
fixed during development (not merely anticipated): `dataset_builder.
_fold_groups` partitions the split plan into independent fit/apply groups
(one "global" group for a chronological plan, one `fold_k` group per
walk-forward fold), and preprocessing/training-statistic-fill is fit ONCE
PER GROUP, using only that group's own train split.
`test_dataset_builder.py::TestWalkForwardPerFoldIsolation::
test_fold_preprocessing_never_reflects_a_later_folds_train_data` is the
regression test: a monotonically-increasing synthetic feature makes each
fold's train-only mean trivially predictable, and the test asserts
`fold_0`'s fitted mean < `fold_1`'s < `fold_2`'s -- which would be equal
under the (fixed) global-fit bug.

## Target and label construction (Section 9)

`features.labels` is a SEPARATE module: `future_return`,
`future_log_return`, `binary_direction`, `vol_adjusted_return` (scaled by
PAST-only realized volatility, never a window spanning the same future
period the return itself covers), and a real (not stubbed) triple-barrier
implementation (`build_triple_barrier_labels`) -- vectorized across
`horizon_bars` forward offsets, resolving same-bar upper/lower ties toward
the conservative (stop-like) outcome, falling back to the sign of the
actual terminal return as the "time barrier" only where the full horizon's
data genuinely exists, and `NaN` wherever neither a barrier touch nor a
resolvable time-barrier outcome exists.

**Isolation is structural, not conventional**: `features.labels` is never
imported by `engine.py` or any feature-group module (statically verified
via AST parsing in `test_labels.py::
test_labels_module_never_imported_by_feature_group_modules` -- deliberately
NOT a naive substring check, since this docstring itself mentions the
module name in prose). `engine.FeatureEngine.compute` refuses any input
DataFrame carrying a `label_`/`target_`-prefixed column
(`LabelLeakageError`), so even a caller mistake (joining labels onto the
features input before computing features) is caught at the boundary.
Rows without sufficient future horizon get `NaN` + `is_valid=False` --
never a fabricated value; `dataset_builder`'s `drop_unlabeled_rows` flag
makes "drop them" vs. "keep them with an explicit `label_valid` marker
column" an explicit per-build choice, never an implicit default silently
picked by this module.

## Dataset splitting (Section 10)

`splitting.build_chronological_split` produces a single train/validation/
test split with explicit `purge_bars` (trimmed from the tail of the
PRECEDING split) and `embargo_bars`/`gap_bars` (skipped at the head of the
FOLLOWING split) at each of the two boundaries.
`splitting.build_walk_forward_splits` wraps the ALREADY-TESTED
`validation.walk_forward.PurgedWalkForwardSplitter` (Milestone 1) rather
than reimplementing purge/embargo arithmetic a second time --
`max_train_size=None` gives expanding-window folds, a set value gives
rolling-window folds. **No code path in this module ever shuffles rows.**

`SplitPlan.to_json_dict()` records the strategy, purge/embargo/gap
settings, and per-split name/row-count/start/end timestamps -- not raw row
indices (too large for a manifest; the split is reproducible by rerunning
the same builder function against the same data, which is the actual
point).

## Research dataset validation (Section 13)

`validation.validate_research_dataset` mirrors `historical.quality.
run_quality_checks`'s never-raises-itself philosophy and reuses its
`Severity` enum. Checks: duplicate/non-monotonic timestamps, excessive
missingness, constant/near-constant features, infinite values, extreme
(z-score) outliers, stale external (`*_is_stale`) features, missing
lineage metadata, target-leakage-suspected (feature-label correlation
above a configurable threshold), and split overlap.

### The split-overlap check's real subtlety

A naive "no two splits may share a row" check is WRONG for walk-forward
plans: `fold_0`'s test rows legitimately become part of `fold_1`'s (larger)
train range, and different folds' train ranges legitimately overlap each
other -- neither is a leak. This was found and fixed during development:
`_check_split_overlaps` only flags overlap between two "eval"-role splits
(validation/test/`fold_k_test`, any groups -- always a real bug if they
overlap) or between a "train" and "eval" split WITHIN THE SAME fold/plan
(defense-in-depth; should already be impossible given `splitting.py`'s own
purge/embargo construction). Train-vs-train across folds, and an earlier
fold's eval vs. a later fold's train, are explicitly NOT flagged. See
`test_validation.py`'s four dedicated boundary tests for the exact line
this draws.

`ResearchDatasetBuilder.build` refuses to write artifacts if validation
finds any CRITICAL issue, unless the caller passes
`allow_critical_validation_issues=True` (not recommended) --
`test_dataset_builder.py::TestValidationGate` proves this against a real
"cheating" feature that peeks at the exact future value its label is
derived from.

## Feature stability and drift (Section 14)

`drift.compare_splits` reports, per shared numeric column: summary
statistics (`summarize_column`), a Population Stability Index
(`population_stability_index`, quantile-binned against the reference
distribution, with the OUTERMOST bin edges extended to +/-inf so a
comparison distribution that has drifted entirely outside the reference's
observed range still lands in the extreme bins instead of silently
producing `NaN` -- a real bug found and fixed during development), and
mean/std shift. Separately: highly-correlated-pair and constant/near-
constant feature detection across the reference split. **These are
descriptive diagnostics only** -- nothing in this module drops, reweights,
or selects a feature based on its own findings.

## Configuration (Section 15)

`config.feature_schemas.ResearchDatasetConfig` (pydantic, frozen,
`extra="forbid"`, same conventions as `config.historical_schemas`) is the
single validated object `feature_cli` needs: symbol/base_timeframe/date
range, historical + research storage roots, `TechnicalFeatureConfig`,
`TemporalFeatureConfig`, optional `MultiTimeframeFeatureConfig`, a list of
`CrossAssetInstrumentConfig`, a list of `MacroSourceFeatureConfig` (each
pointing at a `observation_time,release_time,value` CSV file -- macro data
is not part of the Milestone 2 OHLCV pipeline, so it is supplied
directly), `LabelConfig`, `SplitConfig` (validates required fields per
strategy at construction time), `PreprocessingConfig`, and
`ValidationConfig`. See `examples/xauusd_research_dataset.example.json`
for a complete, worked example (below).

## Dataset manifests and reproducibility (Section 11)

`manifests.py` mirrors `historical.canonical_store`/`historical.manifest`'s
proven content-addressed pattern exactly, for the identical reason: exact,
byte-for-byte reconstruction of a dataset build months after the
underlying historical data has since been revised.

```
<research_storage_root>/research_datasets/dataset_id=<id>/
    content/<sha256>/
        <split_name>.parquet   -- one file per named split
        preprocessing.json     -- {fold_group: fitted TransformPipeline state}
        metadata.json          -- row counts, per-file checksums
        _SUCCESS
    manifests/<version>.json
    CURRENT                    -- one line: current content id
    _LATEST                    -- one line: current manifest version
```

`dataset_id` is `sha256(symbol | base_timeframe | feature_registry_
fingerprint | label_definition | split_definition | preprocessing_
definition)` -- unlike `historical.manifest`'s `dataset_id` (stable across
revisions of the SAME raw data), a research dataset's identity is a
function of its FULL configuration: changing ANY feature/label/split/
preprocessing setting produces a genuinely DIFFERENT dataset id, not a new
version of the old one. `version` still follows the Milestone 2 convention
(`{monotonic sequence}-{content checksum prefix}`, never a date, immutable
once saved, exact content-duplicate save is a no-op).

**The reproducibility proof**
(`tests/integration/test_research_dataset_reproducibility.py`): build
historical version V1, build a research dataset pinned to V1, revise one
historical bar producing V2, rebuild pinned to V1 again -- the manifest
`dataset_id`, `version`, and `content_id` are all identical, and every
split's Parquet content is byte-identical (`pandas.testing.
assert_frame_equal`), even though the underlying historical dataset has
since been revised. Building pinned to V2 produces a genuinely different
manifest VERSION (same `dataset_id`) whose feature values reflect the
revision, and both versions remain independently loadable afterward.

## Feature/label input hashes and code revision

`ResearchDatasetManifest.input_content_hashes` records the historical
dataset's content checksum plus (via `aux_input_content_hashes`) any
higher-timeframe/cross-asset/macro source hashes the caller supplies.
`code_revision` reuses `historical.code_revision.capture_code_revision`
unchanged (git commit hash if available, else a deterministic content hash
of the pipeline's own source -- directly relevant since this repository's
own git history is whatever it is at build time, not assumed to exist).
`environment` records Python version, platform, and pinned
pandas/numpy/pyarrow/pydantic versions (`manifests.
capture_environment_metadata`) -- informational only; this platform does
not refuse to load a manifest built under a different environment.

## CLI usage (Section 16)

```bash
python -m quant_platform.feature_cli list-features --config config.json
python -m quant_platform.feature_cli describe-feature --config config.json --name return_simple_1
python -m quant_platform.feature_cli build-research-dataset --config config.json
python -m quant_platform.feature_cli validate-research-dataset --config config.json --dataset-id ID [--version V]
python -m quant_platform.feature_cli inspect-lineage --config config.json --dataset-id ID [--version V]
python -m quant_platform.feature_cli inspect-dataset-manifest --config config.json --dataset-id ID [--version V]
python -m quant_platform.feature_cli compare-feature-drift --config config.json --dataset-id ID --reference train --comparison test
```

Same operability conventions as `data_cli`: every command returns 0 on
success, a non-zero code on failure, and prints an actionable stderr
message -- never a raw traceback.

## Worked XAUUSD example

`examples/xauusd_research_dataset.example.json` builds a complete
research dataset for XAUUSD at M1, with H1/H4 multi-timeframe features,
DXY and XAGUSD cross-asset features, a Fed Funds rate macro source
(`examples/fed_funds_rate.example.csv`), a 15-bar future-return label, a
chronological 70/15/15 split with a 15-bar purge and embargo at each
boundary, and standard-scale/robust-scale preprocessing on a handful of
technical features:

```bash
python -m quant_platform.feature_cli build-research-dataset \
    --config examples/xauusd_research_dataset.example.json
python -m quant_platform.feature_cli inspect-dataset-manifest \
    --config examples/xauusd_research_dataset.example.json --dataset-id <printed id>
python -m quant_platform.feature_cli compare-feature-drift \
    --config examples/xauusd_research_dataset.example.json --dataset-id <printed id> \
    --reference train --comparison test
```

(Requires an already-ingested XAUUSD/DXY/XAGUSD historical dataset at
`historical_storage_root` via `quant_platform.data_cli ingest` first --
this platform builds research datasets FROM canonical historical data, it
does not ingest raw broker data itself.)

## Performance (Section 18)

Measured on reference hardware, one real run of `tests/performance/
test_feature_throughput.py` (informational; expect +/-30% run-to-run
variance, same conservative-floor philosophy as Milestone 2's benchmarks):

- Raw technical feature computation (19 features, 500,000 rows): 0.254s,
  ~1,971,000 rows/sec.
- `align_higher_timeframe` (500,000 base rows vs. 2,000 H1 bars): 0.441s,
  ~1,134,000 rows/sec.
- Full `FeatureEngine.compute` (35 features incl. multi-timeframe,
  200,000 rows): 1.652s, ~121,000 rows/sec -- markedly slower than
  technical-only because every multi-timeframe feature independently
  recomputes its own alignment pass (a documented simplicity-over-micro-
  optimization tradeoff, see `multi_timeframe.py`'s module docstring).
- Research dataset artifact write (3 splits, 200,000 rows total, zstd):
  0.148s, ~1,349,000 rows/sec.
- Manifest save + load + artifact read round trip (200,000-row dataset):
  34.3ms.

## Milestone 10 Phase 4D: point-in-time multi-source alignment bridge

`quant_platform.features.market_data_bridge` translates Milestone 10's
durable, versioned, content-addressed `market_data` repositories (Phase
2's raw candle store, Phase 4B's curated FRED macro layer, Phase 4C's
cross-asset market-driver layer) into the exact input shapes THIS
package already accepts, so a research dataset can be built from
`market_data`-backed sources through the REAL, unmodified
`FeatureEngine`/`FeatureRegistry`/`ResearchDatasetBuilder` above -- never
a second feature-computation or dataset-building path. Everything in
this section is additive to the architecture described above; nothing
above this section changed in behavior (the two exceptions --
`ResearchDatasetBuildRequest.market_data_lineage`/`ResearchDatasetManifest.
market_data_lineage`, both new optional fields defaulting to `None` --
are noted inline in their own sections).

### Dependency direction and package layout

`historical`/`market_data` -> `features.market_data_bridge` -> `features`
(unmodified) -> research dataset artifacts. `market_data` never imports
this bridge or any other part of `features` (verified structurally by
`test_market_data_bridge_safety_scan.py::TestMarketDataNeverImportsFeatures`,
which scans every file under `market_data/` for a `features` import).

```
features/market_data_bridge/
    bindings.py             immutable BaseAssetDatasetBinding/MacroDatasetBinding/CrossAssetDatasetBinding
    base_asset_adapter.py    resolves+verifies a base-asset binding -> HistoricalDatasetLoaderProtocol
    macro_adapter.py         resolves+verifies a macro binding -> value/release_time DataFrame
    cross_asset_adapter.py   resolves+verifies a cross-asset binding -> OHLCV DataFrame (availability-shifted)
    coverage.py               source-coverage policy (FAIL_REQUIRED_SOURCE/ALLOW_OPTIONAL_MISSING_AND_REPORT/
                               TRIM_TO_COMMON_SAFE_RANGE/QUARANTINE_INTERVAL)
    staleness.py              per-source-type staleness reporting (reuses as_of_join_external/align_higher_timeframe)
    lineage.py                assembles market_data_lineage + its content fingerprint
    request.py                the orchestration entry point: build_research_dataset_from_market_data
    rebuild_planner.py        pure incremental rebuild planner (no I/O)
    reconciliation.py         non-raising structured-issue-reporting counterpart to the adapters' verify_* functions
    verification.py           independent verification incl. the truncation-invariance proofs
    reports.py                deterministic plain-text reports
```

Two narrow, additive changes were made OUTSIDE this new package, both
backward compatible:
- `historical/loader.py` gained `HistoricalDatasetLoaderProtocol`
  (structural interface: `resolve_manifest`/`load_for_engine`) and
  `HistoricalManifestLike` (structural interface:
  `.dataset_id`/`.version`/`.content_checksum`, declared via `@property`
  so a frozen dataclass can satisfy it). `ResearchDatasetBuilder.__init__`'s
  `historical_loader` parameter is now typed against the Protocol instead
  of the concrete `DatasetLoader` class -- a type-hint-only change; the
  real `DatasetLoader` already satisfies it structurally, confirmed by
  `test_market_data_bridge_backward_compatibility.py::
  TestHistoricalDatasetLoaderProtocolBackwardCompatibility`.
- `ResearchDatasetBuildRequest`/`ResearchDatasetManifest` each gained one
  new field, `market_data_lineage: dict[str, object] | None = None`
  (default `None`, so every pre-existing call site is unaffected).

### Source bindings (`bindings.py`)

`BaseAssetDatasetBinding`/`MacroDatasetBinding`/`CrossAssetDatasetBinding`
are immutable, content-addressed (their own `binding_id` is a
deterministic hash of every other field, computed by a `create_*`
factory mirroring `market_data`'s own `create_*`-then-`compute_content_id`
convention). Every identity-bearing string field is validated against a
small set of forbidden mutable-alias tokens (`latest`/`current`/`newest`/
`active`/`default`/...) -- `SourceBindingError` on construction, never a
downstream surprise. `CrossAssetDatasetBinding` structurally rejects
`instrument_form=ETF` paired with `proxy_policy.is_proxy=False` at
construction time (reuses `market_data`'s own `ProxyPolicy` type
directly rather than re-declaring proxy vocabulary).

### The three source adapters: verify, then resolve

Each adapter (`base_asset_adapter.py`/`macro_adapter.py`/
`cross_asset_adapter.py`) exposes a `verify_*_binding` function (re-reads
the live `market_data` store, requires the binding's pinned id to equal
the store's CURRENT manifest, and requires a freshly recomputed digest
to match that manifest's own recorded digest -- `SourceVerificationError`
fail-closed on any mismatch) and a `resolve_*_dataframe`/`load_for_engine`
function (the single, documented Decimal -> float64 boundary crossing,
producing exactly the shape this package's existing feature families
expect). Base-asset conflicting-candle and cross-asset conflicting-bar
coordinates (two different records claiming the same timestamp) are
detected and rejected, never silently resolved by picking one.

**The cross-asset availability-shift adapter** is the one genuinely new
alignment idea here (spec's own "if conventions differ, implement an
explicit adapter and document the conversion"): `align_higher_timeframe`
derives its reveal instant internally as `open_time + timeframe.duration`
and has no parameter for an externally-supplied availability time, but a
`MarketDriverBar` carries its OWN resolved `availability_time` (which can
be materially later than a naive close-time guess, e.g. next-day
availability for daily ETF proxy bars). `cross_asset_adapter.
resolve_cross_asset_dataframe` therefore emits a SYNTHETIC `open_time`
per row (`availability_time - timeframe.duration`) so that
`align_higher_timeframe`'s own close-time-based reveal rule lands exactly
on the bar's true availability instant, without touching that shared,
already-tested primitive. Proven end-to-end by
`test_market_data_bridge_cross_asset_adapter.py::
TestResolveCrossAssetDataframe::test_bar_not_revealed_before_true_availability_time`.

Base-asset data uses a different, simpler, equally explicit policy: a
`market_data.candles.Candle` carries no `availability_time` field at all
(unlike the richer Phase 4B/4C records), so the bridge's own honest
policy is that a candle becomes visible exactly at its `close_time`
(`open_time + timeframe.duration`) -- which is EXACTLY `FeatureEngine.
compute`'s own availability-instant derivation for the base timeframe;
the adapter changes nothing about how that's computed, it only supplies
the values.

**Macro revision-policy selection** (`macro_adapter.
select_observations_for_policy`) is a pure selection over an already-
verified, already-multi-vintage observation set -- it decides WHICH
observations enter the join, never how the join itself works.
`RevisionPolicyKind.VINTAGE_SERIES` passes every distinct vintage
through unchanged (the general-purpose default: because each vintage
carries its own accurate `availability_time`, `as_of_join_external`'s
existing backward-looking as-of join over the full multi-vintage stream
already reproduces true point-in-time-correct behavior with no new
alignment logic). `FIRST_RELEASE_ONLY` keeps only the earliest-released
vintage per `observation_date`. `LATEST_AVAILABLE`/`AS_OF_REALTIME_DATE`
are REFUSED outright (`AlignmentPolicyError`) for research/training
dataset construction -- both are explicitly non-point-in-time-safe by
their own `market_data` docstrings (a single global "whatever is true
right now" or "whatever was true on one fixed historical date" snapshot,
never a per-row as-of view). A genuinely missing observation
(`is_missing=True`) is included with `value=NaN` at its own real
`availability_time` rather than dropped, so the series correctly shows
as unavailable starting exactly when the official gap was published,
never silently masked by the prior value's carry-forward.

### Coverage, staleness, lineage, and manifest identity

`coverage.py` implements the four named policy kinds (spec's own
vocabulary): `FAIL_REQUIRED_SOURCE` (default-safe, raises
`SourceCoverageError`), `ALLOW_OPTIONAL_MISSING_AND_REPORT`,
`TRIM_TO_COMMON_SAFE_RANGE` (computes and reports the exact trimmed
range and why -- never silent), and `QUARANTINE_INTERVAL` (a separate,
post-hoc function, `evaluate_missing_runs`, over an already-computed
missing indicator, identifying consecutive-missing runs beyond a
configured tolerance -- reported, never auto-dropped). `staleness.py`
reuses the existing `as_of_join_external`/`align_higher_timeframe`
primitives directly (rather than reimplementing staleness) to produce a
per-source `StalenessFinding`, applying a source-specific age threshold
independent of which features a caller actually registered.

`lineage.build_market_data_lineage` assembles a deterministic, stable-
sorted JSON payload from every resolved binding plus the coverage
decision; `lineage_content_id` gives it one content-addressed
fingerprint. `request.build_research_dataset_from_market_data` (the
orchestration entry point) threads BOTH into the `ResearchDatasetBuildRequest`
it hands to the REAL `ResearchDatasetBuilder.build()` --
`market_data_lineage` directly, and the fingerprint via
`aux_input_content_hashes["market_data_lineage_content_id"]` (the SAME
free-form extension point `historical_dataset_content_checksum` already
used).

**How this satisfies "changing a bound source version changes dataset
identity" without touching `compute_dataset_id`.** `compute_dataset_id`
(unmodified) stays a stable, source-content-independent "recipe id"
exactly as originally designed. `ResearchDatasetManifest.
market_data_lineage` is NOT excluded from `_identity_fields()`, so
`ResearchManifestStore.save`'s own existing content-duplicate-vs-new-
version comparison already picks up any lineage change. The precise
mechanism (worth stating exactly, since it is easy to get subtly wrong
-- see `test_market_data_bridge_identity_compatibility.py`'s own module
docstring for the full worked explanation): `version` is
`f"{sequence:06d}-{content_id_prefix}"`, where `content_id` hashes only
the WRITTEN feature/label/split bytes (so two builds with different
lineage but coincidentally identical output bytes can share the same
`content_id_prefix`), but `sequence` always advances whenever
`_identity_fields()` differs from the latest existing version in the
SAME manifest history -- so the full version string still changes,
provided the rebuild targets the same `ResearchManifestStore` root
(the realistic "rebuild after a source revision" scenario).

### Incremental rebuild planning, reconciliation, and verification

`rebuild_planner.plan_rebuild` is a PURE function (no `market_data`/
`features` I/O of its own) comparing an existing manifest's recorded
lineage against newly proposed bindings, returning one of `NO_OP`/
`APPEND_ONLY_SAFE_EXTENSION`/`PARTIAL_RECOMPUTATION_REQUIRED`/
`FULL_REBUILD_REQUIRED` with reason codes. Distinguishing a safe append
from a content revision requires more than a bare changed-id signal (a
content hash carries no diff), so the planner accepts OPTIONAL
`SourceChangeEvidence` (old/new covered-time-range and count, obtained
by the caller separately) and only classifies `APPEND_ONLY_SAFE_EXTENSION`
when BOTH the covered range's start is unchanged AND there is STRICT
growth in at least one of covered-end-time/count -- identical
range-and-count evidence alongside a changed component id (the
signature of a same-range, same-count content-only revision) is never
classified append-only, even though it satisfies a naive `>=`-based
check (a real bug caught and fixed by this phase's own adversarial
tests: see `test_market_data_bridge_adversarial.py::
Test22IncrementalPlannerDoesNotMissAHistoricalRevisionImpact`).

`reconciliation.py` is the non-raising, structured-issue-reporting
counterpart to the adapters' own fail-closed `verify_*` functions, plus
two standalone PIT re-checks (`reconcile_no_pre_availability_macro_
leakage`/`reconcile_no_pre_close_cross_asset_leakage`) that independently
re-derive the as-of/close-time join's own output and assert no selected
value's timestamp exceeds the row's own availability instant.

`verification.py` assembles the independent-verification checklist,
classifying each check's actual independence HONESTLY
(`INDEPENDENCE_CLASSIFICATION`): re-reading durable evidence via a
genuinely separate code path (`independent_re_read`) vs. reproducing an
already-published digest via the exact same hash formula
(`same_formula_re_derivation` -- proves store/manifest consistency, not
algorithmic independence of the hash itself) vs. reusing the SAME shared
M3 alignment primitive the production path itself uses
(`reused_shared_primitive` -- proves internal self-consistency, not
independence from a bug in that shared primitive, which M3's own
`test_alignment_boundaries.py` covers separately). The truncation-
invariance proofs (`verify_truncation_invariance_macro`/`_cross_asset`)
directly exercise spec's own "truncating all records after T must not
change any aligned row at/before T" requirement by comparing alignment
against the full source vs. a version truncated after a cutoff -- this
caught a real bug during development (a `NaT != NaT` pandas comparison
pitfall that misreported "no release yet in either join" as "differing";
fixed, with a regression test).

### Tests

`tests/unit/features/test_market_data_bridge_*.py` (bindings, all three
adapters, coverage, staleness, lineage, rebuild planner, reconciliation,
verification, reports, the orchestration entry point, golden identity
compatibility, backward compatibility, the mandatory 12-step
fixture-based acceptance workflow, the 22-item adversarial suite, a
safety scan, and performance sanity bounds) plus a shared fixture-helper
module (`_market_data_bridge_test_helpers.py`, not itself collected).
Every test builds real `market_data` repositories through the REAL
ingestion/collector-store APIs and feeds the REAL `ResearchDatasetBuilder`
-- never a mock or a test-only replacement.

### Known limitations (Phase 4D)

- No selective historical read for a superseded `market_data` manifest
  version across any of the three source kinds (see `docs/
  market_data_architecture.md`'s own Phase 4D section for the full
  explanation and the base-asset-vs-macro/cross-asset asymmetry).
- Cross-asset `roll_provenance`/`contract_metadata_id` (per-bar futures
  detail) are not propagated into the aligned OHLCV frame -- unexercised
  by this phase's ETF-only fixtures, a real gap for a future
  futures/continuous-series cross-asset driver.
- No genuine `market_data`-backed higher-timeframe derivation for the
  SAME base instrument -- a caller resolves it via a second
  `base_asset_adapter.resolve_base_asset_dataframe` call themselves.
- `SourceChangeEvidence` (the rebuild planner's append-only-detection
  input) must be supplied by the caller from a SEPARATE `market_data`
  read; this phase does not auto-compute it from two binding sets alone
  (a bare component/dataset id has no diff).
- No CLI surface for the bridge itself (`feature_cli.py` unchanged) --
  library-only, matching every `market_data` phase's own disclosed
  limitation.

## Known limitations (honest, as measured)

- **A custom feature can still misuse raw future data directly** (see
  above) -- the guarantee is about the standard library and about label
  isolation, not about forbidding arbitrary Python inside a `compute`
  callable.
- **Session-relative temporal features require exactly one weekly
  session** in the supplied `TradingCalendar` -- a deliberate, documented
  scope limit (raises `ConfigurationError` otherwise), not a general
  multi-session scheduler.
- **Cross-asset rolling correlation is computed at the BASE timeframe's
  cadence**, using the cross-asset return aligned (and necessarily
  repeated between updates) into that cadence -- if the cross-asset
  series' native cadence is coarser than the base timeframe, this
  mechanically inflates the apparent correlation magnitude relative to a
  "true" same-cadence comparison. Still point-in-time correct; the caveat
  is about magnitude interpretation, not leakage.
- **Multi-timeframe/cross-asset features each independently recompute
  their own alignment pass** rather than sharing a cached result across
  the ~4-8 features in each family -- a deliberate simplicity-over-micro-
  optimization tradeoff (see Performance above for the measured cost).
- **Preprocessing/imputation is fit per fold-group, not per-feature-pair
  interaction** -- e.g. no attempt at joint/multivariate normalization.
- **`feature_registry_fingerprint`/`dataset_id` hash DECLARED `FeatureSpec`
  metadata, not a feature's `compute` closure bytecode.** Two different
  closures registered under the exact same `(name, version)` and otherwise
  identical spec produce the identical declared identity even though their
  actual output differs -- an inherent limitation of any metadata-based
  fingerprint (there is no portable, stable way to hash an arbitrary
  Python closure across interpreter versions). This does NOT silently
  corrupt data: `ResearchManifestStore.save`'s content-duplicate check
  compares the manifest's FULL JSON representation (which includes
  `content_id`, itself derived from actual written bytes), so differing
  output under identical declared identity still mints a genuinely new
  manifest VERSION rather than being treated as a no-op -- proven directly
  in `test_dataset_builder.py::TestSpecIdentityVsActualClosureContent`.
  The practical mitigation is discipline, not tooling: always bump
  `version` when a feature's actual computation changes (see
  `models.FeatureSpec`'s own docstring).
- **Manifest `environment` metadata is informational only** -- loading a
  manifest built under a different pandas/numpy/pyarrow version is not
  refused or warned about beyond what's visible in the recorded field.
- **The macro CSV format is fixed** (`observation_time,release_time,value`)
  -- no support yet for a source with more than one value column per row,
  or for a revision/vintage identifier column beyond release-time-based
  ordering.
- **Performance figures are single-machine, single-symbol measurements**
  (low hundreds-of-thousands of rows); nothing here claims or has been
  measured at distributed/multi-symbol/billion-row scale.
