# Label Validation (Milestone 11, Phase 3, Part C)

This is the authoritative technical reference for
`quant_platform.label_validation` -- the package that establishes
whether GENERATED labels (Milestone 11 Phase 3 Parts A/B) are
scientifically suitable for machine-learning research.

**The output of this package is NOT labels. The output is LABEL
QUALITY.** It never creates a new label family, never trains a model,
never evaluates a model, and never computes a statistic that requires a
prediction target -- no Information Coefficient, no Rank IC, no Mutual
Information, no SHAP, no Permutation Importance, no feature selection
of any kind. It reads an already-generated `labels.builder.LabelBundle`
and its `labels.manifest.LabelManifest` ONLY; it never generates a
label itself and never modifies one.

## Position in the pipeline

```
Market Data -> Features -> Qualification -> Feature Discovery -> Labels -> Label Validation -> Machine Learning
```

A new, independent package -- never merged into `labels/`,
`qualification/`, or `feature_discovery/`. Its only in-platform
dependency is `labels/`, its direct upstream predecessor, plus
`core.exceptions`, `core.types`, `historical.quality.Severity`, and
`ml.persistence` (the shared JSON-codec helpers every Milestone 11
package uses). It imports NOTHING from `features`, `market_data`,
`qualification`, or `feature_discovery` -- the identical
dependency-isolation discipline `labels/` itself already established.
Two structural consequences of this boundary:

- **Regime assignment is always caller-supplied.** `stability.
  compute_label_stability(..., regime_assignment: pd.Series | None)`
  never computes bull/bear/sideways/volatility/macro regimes itself --
  doing so would require reading raw price or macro data, which is out
  of this package's scope. The caller passes a `pd.Series` aligned 1:1
  with the bundle's own rows.
- **Future macro / future cross-asset leakage checks are disclosed as
  out of scope**, not silently skipped or fabricated. `leakage.
  validate_leakage` always emits one INFO-severity, non-blocking
  `LabelEvidence` record stating plainly that this package never reads
  raw macro/cross-asset source data, so those two of the governing
  specification's six named leakage checks cannot be independently
  verified from a bundle/manifest/records alone.

## Architecture

```
labels.builder.LabelBundle  +  labels.manifest.LabelManifest  (+ optional labels.records.LabelRecord tuple)
   |
   v
label_validation.statistics.compute_label_statistics        -- foundational descriptive stats (mean/std/percentiles)
label_validation.distribution.compute_label_distribution     -- entropy, cardinality, class ratios, rare labels
label_validation.balance.compute_label_balance                -- binary/multiclass/neutral/imbalance -- evidence only, no recommendations
label_validation.degeneracy.compute_label_degeneracy            -- constant/near-constant/empty/all-neutral/impossible values
label_validation.coverage.compute_label_coverage                  -- valid/missing, warmup vs. trailing vs. interior holes
label_validation.stability.compute_label_stability                  -- temporal (rolling) + regime (caller-supplied) stability
label_validation.drift.compute_label_drift                           -- PSI / KL / JS divergence between two bundles
label_validation.leakage.validate_leakage                              -- point-in-time + tamper self-consistency checks
   |
   v
label_validation.diagnostics.compute_label_diagnostics   -- aggregates the 8 dimensions above into one scored report
   |
   v
label_validation.engine.LabelQualificationEngine.qualify   -- APPROVED / CONDITIONALLY_APPROVED / REJECTED
   |
   +-- label_validation.verification.LabelValidationVerifier      -- independent re-derivation, never trusts a cached report
   +-- label_validation.replay.LabelValidationReplay                -- destroy, regenerate from source_data, re-qualify
   +-- label_validation.reconciliation.LabelValidationReconciliation -- diff two qualification reports
   +-- label_validation.horizon.compare_horizons                       -- descriptive-only, cross-horizon comparison
   +-- label_validation.overlap.detect_overlap                          -- duplicate / redundant / horizon / barrier overlap
   +-- label_validation.reports (render_*)                                -- 7 deterministic plain-text reports
```

Package layout (`src/quant_platform/label_validation/`, 18 files, 2,455
lines):

```
__init__.py            32 lines    Package docstring, dependency-isolation statement
evidence.py            127 lines   LabelEvidence, LabelValidationDimensionKind (8), BlockingFindingCode (7)
statistics.py           79 lines   LabelStatistics
distribution.py        163 lines   LabelDistribution, shared bucket_values()/is_discrete_family()
balance.py              136 lines   LabelBalance
degeneracy.py           193 lines   LabelDegeneracy, detect_duplicate_labels
coverage.py             112 lines   LabelCoverage
stability.py            268 lines   LabelStability (TemporalStabilityResult, RegimeStabilityResult)
drift.py                161 lines   LabelDrift (PSI / KL / JS)
leakage.py              192 lines   LeakageValidationResult, validate_leakage
diagnostics.py          218 lines   LabelDiagnostics, LabelValidationDimensionResult, compute_label_diagnostics
engine.py                91 lines   LabelQualificationDecision, LabelQualificationReport, LabelQualificationEngine
verification.py         123 lines   LabelValidationVerifier, verify_report_self_consistency
reconciliation.py       130 lines   LabelValidationReconciliation (5 drift kinds)
replay.py                90 lines   LabelValidationReplay
horizon.py               112 lines   HorizonComparisonReport, compare_horizons
overlap.py               123 lines   OverlapReport, detect_overlap
reports.py                105 lines   7 render_* functions
```

## The evidence model

Every finding in this package is a `LabelEvidence` record -- never a
bare verdict. It carries exactly the 7 fields the governing
specification names: `finding` (str), `evidence` (tuple of facts),
`severity` (`historical.quality.Severity`: INFO/WARNING/CRITICAL),
`affected_labels` (a tuple, since drift/overlap/reconciliation findings
inherently concern a pair or set of labels), `statistics` (a
`dict[str, float]`), `recommendation` (`str | None`), and `blocking`
(`bool`), plus one addition beyond the named 7: `blocking_code` (a
`BlockingFindingCode` enum member, required exactly when `blocking` is
`True` -- enforced in `__post_init__`, never left implicit).

`LabelValidationDimensionKind` names the 8 fixed dimensions this
package evaluates, and `LABEL_VALIDATION_DIMENSION_ORDER` fixes their
canonical iteration order everywhere in the package (`LabelDiagnostics.
dimension_results` is validated in `__post_init__` to cover exactly
this order, in exactly this order -- never dict/set iteration order, so
two qualification runs over the same bundle always produce
byte-identical dimension ordering):

```
DISTRIBUTION -> BALANCE -> DEGENERACY -> COVERAGE -> TEMPORAL_STABILITY -> REGIME_STABILITY -> DRIFT -> LEAKAGE
```

`BlockingFindingCode` names the 7 conditions that can force a
REJECTED decision: `EMPTY_LABELS`, `CONSTANT_LABELS`,
`IDENTITY_MISMATCH`, `MANIFEST_MISMATCH`, `BARRIER_VIOLATION`,
`AVAILABILITY_VIOLATION`, `REPLAY_DIVERGENCE`.

## Discrete vs. continuous bucketing

`distribution.bucket_values(values, *, label_family, bucket_count=10)`
is the ONE shared bucketing helper every other dimension module
(`balance.py`, `degeneracy.py`, `stability.py`, `horizon.py`) reuses
rather than reimplementing:

- **Discrete families** (`DIRECTION`, `TRIPLE_BARRIER` --
  `DISCRETE_LABEL_FAMILIES`) are grouped by their exact value
  (`format_discrete_value`, `f"{value:g}"`).
- **Continuous families** (`NEXT_RETURN`, `MULTI_HORIZON_RETURN`,
  `FORWARD_VOLATILITY`) are grouped into quantile buckets via `pandas.
  qcut(values, q=bucket_count, duplicates="drop")`, gracefully
  collapsing to fewer buckets (down to a single bucket) for
  low-cardinality or near-constant continuous series rather than
  raising.

## Label balance -- evidence only, no recommendations

`balance.py`'s `_balance_evidence` helper always constructs its
`LabelEvidence` with `recommendation=None, blocking=False,
blocking_code=None`, enforced by construction: an imbalanced label is
information a researcher needs, not, by itself, grounds to reject a
label (unlike degeneracy), and a balance FACT ("73% of rows are
NEUTRAL") is not a prescription ("rebalance via class weighting"),
which would presume a specific downstream modeling technique this
package has no opinion on. `EXTREME_IMBALANCE_RATIO_THRESHOLD = 20.0`
(majority/minority class fraction) is a documented, disclosed
threshold, not a claim of statistical significance.

## Label degeneracy -- the one dimension that can block

Unlike balance, degeneracy findings CAN block: a constant or empty
label carries zero information no model could ever learn from.
`compute_label_degeneracy` checks, per bundle: `is_empty` (zero valid
values, blocking `EMPTY_LABELS`), `is_constant` (single distinct
value, blocking `CONSTANT_LABELS`), `is_near_constant`
(`NEAR_CONSTANT_FRACTION_THRESHOLD = 0.99` majority-value fraction,
WARNING, non-blocking), `is_single_class` (the BUCKETED representation
collapses to one class -- distinct from `is_constant`, which checks
raw values), `is_all_neutral` (`DIRECTION` only, blocking
`CONSTANT_LABELS`), `is_zero_variance`, and `has_impossible_labels`
(values outside the family's valid domain -- `{-1, 0, 1}` for
`DIRECTION`/`TRIPLE_BARRIER`, `>= 0` for `FORWARD_VOLATILITY`, `> -1.0`
for `NEXT_RETURN`/`MULTI_HORIZON_RETURN` -- blocking
`BARRIER_VIOLATION`).

`detect_duplicate_labels(records: tuple[LabelRecord, ...])` lives here
too (a records-based, not bundle-based check): a `label_id` collision
between two different records is a genuine hash-collision or
identity-computation defect, checked via a plain `collections.Counter`.

## Temporal and regime stability

`stability.py` measures only stability -- never predictive power.
Nothing here computes a correlation, an information coefficient, or any
statistic relating a label to a future outcome; every number is a pure
property of the label's own value sequence over time or across regimes.

**Temporal** (`TemporalStabilityResult`): rolling-window majority-class
fraction std (`rolling_balance_std`), rolling-window entropy std
(`rolling_entropy_std`), rolling-window value std ("vol of vol",
`rolling_variance_std`), a `window_stability_score` (`1 - std(per-window
means) / (overall_std + eps)`, clipped to `[0, 1]`), an
`expanding_stability_score` (how much the expanding mean has converged
by the end of the series), and `availability_stability` (std of the
rolling coverage fraction). A genuine production bug was found and
fixed while building this module: `pandas.Series.rolling().apply()`
cannot operate directly on a `Series` of STRING bucket labels
(`TypeError: cannot handle this type -> str`). Fixed via `_encode_buckets`
(`pd.factorize(bucketed, use_na_sentinel=True)`, preserving `NaN`),
with both rolling functions rewritten to operate on `np.ndarray` via
`raw=True`.

**Regime** (`RegimeStabilityResult`): per-regime mean, per-regime valid
count, and `cross_regime_mean_spread` (max minus min per-regime mean),
computed only when the caller supplies a `regime_assignment: pd.Series`
aligned 1:1 with the bundle's rows (raises `LabelValidationRequestError`
on a length mismatch) -- this package never computes the regime
segmentation itself.

## Drift

`drift.compute_label_drift(baseline, candidate, *, bucket_count=10,
rolling_window_bars=20)` compares two already-generated bundles of the
SAME label family (raises `LabelValidationRequestError` otherwise) via:

- **PSI** -- `sum((q - p) * ln(q / p))`
- **KL divergence** -- `sum(p * ln(p / q))` (KL(P‖Q))
- **JS divergence** -- `0.5 * KL(P‖M) + 0.5 * KL(Q‖M)`, `M = 0.5*(P+Q)`

with `_EPSILON = 1e-6` smoothing applied to both distributions to avoid
`log(0)`/division-by-zero. `SIGNIFICANT_PSI_THRESHOLD = 0.2` is the
commonly-used PSI convention (0.1 = moderate, 0.2 = significant) --
disclosed as a documented convention, never a formal significance
claim. `rolling_drift` is the PSI of each successive
`rolling_window_bars`-sized window of the candidate's values against
the FIXED baseline distribution, and `class_drift` is the per-class
`candidate_fraction - baseline_fraction`.

## Leakage validation and tamper detection

`leakage.validate_leakage(bundle, manifest, *, records=None)` is an
INDEPENDENT re-verification -- it deliberately does not call `labels.
diagnostics`'s own point-in-time checks, re-deriving its own
conclusions from the bundle/manifest/records alone. In order, it
checks:

1. `manifest.label_specification_id` matches the bundle being
   validated (blocking `MANIFEST_MISMATCH` otherwise).
2. `manifest.verify_self_consistency()` -- the manifest's own checksum
   matches a fresh recomputation (blocking `MANIFEST_MISMATCH`).
3. A fresh `labels.identity.compute_label_identity` recomputation over
   `bundle.values` matches `bundle.identity.content_id` (blocking
   `IDENTITY_MISMATCH`) -- an independent tamper check that never
   trusts the bundle's own claimed identity.
4. If `records` are supplied, each record's own
   `verify_self_consistency()` (blocking `IDENTITY_MISMATCH` if any
   record fails).
5. The trailing-`NaN`-tail shape is well-formed (an independently
   reimplemented version of the same check `labels.diagnostics`
   already performs -- re-derived, never imported, so a shared bug
   between the two would not go undetected; WARNING, non-blocking).
6. If `records` are supplied, `availability_time >= event_time` for
   every record (blocking `AVAILABILITY_VIOLATION`).
7. If the family is `TRIPLE_BARRIER`, every value falls in `{-1, 0,
   1}` (blocking `BARRIER_VIOLATION`).
8. A disclosed INFO-severity, non-blocking finding stating that future
   macro / future cross-asset checks are out of this package's scope.

Checks 2-4 were added mid-development, after the adversarial audit
(below) exposed that nothing in this package independently re-verified
a tampered manifest checksum, a tampered bundle identity, or a
tampered record -- `labels/`'s own `LabelRecord.verify_self_consistency`/
`LabelManifest.verify_self_consistency` existed but were never called
from `label_validation`.

## Label qualification -- three tiers, explicit blocking reasons

`engine.LabelQualificationEngine.qualify(bundle, manifest, *,
drift_baseline=None, regime_assignment=None, records=None)` runs
`diagnostics.compute_label_diagnostics` and derives exactly one
`LabelQualificationDecision`:

- **REJECTED** -- any blocking evidence anywhere.
- **CONDITIONALLY_APPROVED** -- no blocking evidence, but at least one
  WARNING or CRITICAL (non-blocking) finding.
- **APPROVED** -- no blocking evidence, no WARNING/CRITICAL finding at
  all.

`blocking_reasons` cites every blocking evidence record's own
`blocking_code` and `finding` text verbatim (`f"[{code}] {finding}"`)
-- never a generic message.

**Dimension scoring** (`diagnostics._score_from_evidence`): each
dimension's score starts at `1.0`; any blocking evidence anywhere in
that dimension forces the score to `0.0`; otherwise `-0.4` per CRITICAL
finding and `-0.15` per WARNING finding, clipped to `[0, 1]`.
`overall_score` is the mean of the 8 dimension scores.

## Independent verification, replay, reconciliation

Three mechanisms, mirroring the identical pattern already established
in `qualification/`, `feature_discovery/`, and `labels/` itself:

- **Verification** (`verification.py`) -- `verify_report_self_consistency`
  is a pure, no-I/O check that independently recomputes the decision
  and `overall_score` from a report's own `diagnostics.
  dimension_results`, using `_independent_decide` -- a DELIBERATELY
  separate reimplementation of `engine._decide`'s rule, never importing
  the private function, so a shared bug between the two would not go
  undetected. `LabelValidationVerifier.verify` additionally runs a
  fresh `LabelQualificationEngine().qualify()` against the live
  bundle/manifest and reconciles it against the supplied report.
- **Replay** (`replay.py`) -- "destroy labels, replay labels,
  qualification identical." `LabelValidationReplay.
  replay_and_requalify` never re-references the original bundle
  object, only its `LabelDefinition` and the immutable `source_data` it
  came from; it regenerates a fresh bundle via `labels.builder.
  LabelBuilder`, re-qualifies it, and compares the resulting DECISION
  and OVERALL SCORE (not byte-identical values) to the supplied
  `original_report`. This is a deliberate, disclosed design choice: the
  qualification verdict is a SHAPE-level judgement (distribution,
  balance, degeneracy), not a byte-level comparison -- see "Known
  limitations" below.
- **Reconciliation** (`reconciliation.py`) -- `LabelValidationReconciliation.
  reconcile(baseline, candidate)` compares two `LabelQualificationReport`s
  for the SAME `label_specification_id` (raises
  `LabelValidationReconciliationError` for different specifications --
  a structural precondition, not a normal finding) and detects 5
  named drift kinds: `decision_drift`, `score_drift` (beyond
  `score_tolerance=0.01`), `evidence_drift` (full `(dimension, finding,
  severity)` set diff), `warning_drift` (the WARNING-severity subset of
  that diff), and `distribution_drift` (`class_ratios` diff).

## Horizon analysis and overlap detection

`horizon.compare_horizons(bundles)` compares coverage, balance,
degeneracy, and stability across a set of same-family,
different-horizon bundles -- **purely descriptive**: `HorizonComparisonReport`
never ranks or scores which horizon is "best," it lists each horizon's
own facts side by side (sorted ascending by `horizon_bars`), reusing
`coverage.py`/`degeneracy.py`/`balance.py`/`stability.py`'s own compute
functions directly rather than duplicating any of their logic.

`overlap.detect_overlap(bundles)` performs a pairwise O(n²) comparison
across a set of bundles, detecting 4 kinds of finding:

- `horizon_overlap` -- same family, same `horizon_bars` parameter.
- `barrier_overlap` -- both `TRIPLE_BARRIER`, identical
  `profit_multiplier`/`loss_multiplier`/`max_holding_bars`/
  `volatility_estimator_reference`.
- `duplicate_target` -- an EXACT check: `identity.content_id` equality
  (the same content-addressed identity Part A already computes, never
  recomputed a second way here).
- `redundant_target` -- a disclosed statistical-similarity HEURISTIC:
  Pearson correlation `>= REDUNDANCY_CORRELATION_THRESHOLD` (`0.999`)
  between two DIFFERENTLY-identified bundles' values -- catching, e.g.,
  a Next Return horizon=5 bundle and a Multi Horizon Return horizon=5
  bundle: same math, different family, so a different `content_id`, but
  perfectly correlated values. `duplicate_target` and `redundant_target`
  are two genuinely different, complementary checks (verified directly
  by a smoke test proving each catches what the other structurally
  cannot); a `duplicate_target` finding skips the correlation check for
  that pair (`continue`) rather than double-reporting.

## Quality gates

- `git diff --check`: clean.
- `ruff check .`: all checks passed.
- `mypy src`: no issues found (442 source files).
- `pytest tests/unit/label_validation/`: 133 tests passed (109 per-module
  + 17 adversarial + 7 repetition/determinism).
- Combined Milestone 11 suite (`feature_discovery` + `labels` +
  `label_validation` + `qualification`): 687 tests passed.
- Full repository suite: see `docs/milestone11_phase3c_delivery_report.md`
  for the exact count from this task's own run.

## Known limitations

- **Replay proves verdict reproducibility, not byte-identical value
  reproducibility.** `LabelValidationReplay.replay_and_requalify`
  compares the regenerated bundle's qualification DECISION and OVERALL
  SCORE to the original report's, not the underlying label values
  themselves (that proof belongs to `labels.replay.LabelReplay`, one
  layer down, which does compare values). A source-data corruption that
  preserves the label's distributional SHAPE (e.g. an affine rescale of
  price before computing a return) can legitimately reproduce an
  identical qualification verdict even though the underlying values
  differ substantially -- this is a deliberate design choice (label
  QUALITY, not label VALUE, is this package's subject), not an
  oversight, but it means replay divergence detection is only as
  sensitive as the qualification dimensions themselves.
- **Regime segmentation is always caller-supplied.** This package has
  no opinion on how bull/bear/sideways/high-volatility/low-volatility/
  macro-tightening/macro-easing regimes are defined; it only measures
  stability GIVEN a caller's own assignment.
- **Future macro release and future cross-asset leakage checks are
  disclosed as out of scope**, not fabricated -- see "Position in the
  pipeline" above.
- **Overlap detection is O(n²) in the number of bundles supplied** --
  acceptable for the handful of label families/horizons a research
  workflow typically compares at once, not designed for large-scale
  bundle catalogs.
