# Dataset Qualification Engine (Milestone 11, Phase 1)

This is the authoritative technical reference for `quant_platform.qualification`
-- the deterministic engine that decides whether an ALREADY-BUILT research
dataset (a `features.manifests.ResearchDatasetManifest` produced by the
real, unmodified Milestone 3 `features.dataset_builder.ResearchDatasetBuilder`)
is scientifically suitable for ML.

**This layer never trains a model, never computes feature importance, and
never performs feature selection.** It sits ABOVE the Milestone 3 research
dataset pipeline, one full milestone later, evaluating what M3 already
built rather than extending M3 internally -- the same relationship
Milestone 6's `robustness` package has to Milestone 5's backtesting
engine. Whether a dataset the engine approves eventually produces a good
model is out of scope; "is this dataset trustworthy enough to attempt
that with" is the entire question this package answers.

Delivered in two parts, matching the two halves of the governing
specification:

- **Part 1** -- the 7 named deliverables with real, working logic across
  all 8 dimensions and all 6 blocking-failure codes: `DatasetQualificationEngine`,
  `DatasetQualificationReport`, `QualificationDecision`, `QualificationVerifier`,
  `QualificationDiagnostics`, `QualificationReconciliation`, `QualificationReports`.
- **Part 2** -- deep diagnostics with an evidence model, independent
  verification, truncation/replay/determinism proofs, a 26-item adversarial
  audit, reconciliation extended to warning/recommendation/lineage drift,
  4 additional report types, and the quality-gate/documentation ceremony.
  Part 2 explicitly reuses every Part 1 type and function; nothing in
  Part 1's architecture was redesigned.

## Architecture

```
features.manifests.ResearchDatasetManifest (Milestone 3, unmodified)
features.manifests.ResearchDatasetStore     (Milestone 3, unmodified)
   |
qualification.verifier.QualificationVerifier    -- raw facts: identity, artifacts, lineage, leakage
   |
qualification.dimensions (8 pure evaluators)    -- score + findings/warnings/blocking per dimension
   |
qualification.engine.DatasetQualificationEngine -- orchestrates all 8, derives ONE QualificationDecision
   |
   +-- qualification.diagnostics.compute_diagnostics       -- WHY: split summaries + 6 evidence sections
   +-- qualification.verification.QualificationIndependentVerifier -- re-qualifies from scratch, diffs
   +-- qualification.reconciliation.QualificationReconciliation    -- diffs two reports
   +-- qualification.reports (render_*)                    -- deterministic plain-text output
```

Package layout:

```
src/quant_platform/qualification/
    models.py          enums, BlockingFailure, DimensionResult, QualificationDecision,
                        DatasetQualificationReport -- shared vocabulary every other module imports
    verifier.py         QualificationVerifier: identity/artifact/lineage/leakage FACTS (Part 1)
    dimensions.py        the 8 pure evaluate_* functions (Part 1)
    engine.py            DatasetQualificationEngine: orchestration + the 2-tier decision rule (Part 1)
    diagnostics.py        QualificationDiagnostics: split summaries (Part 1) + 6 Evidence sections (Part 2)
    evidence.py            Evidence: the atomic finding/evidence/severity/recommendation record (Part 2)
    verification.py         QualificationIndependentVerifier: self-consistency + re-qualify-and-diff (Part 2)
    reconciliation.py        QualificationReconciliation: diff two reports (Part 1 + Part 2 drift kinds)
    reports.py                 7 render_* functions: deterministic plain-text reports (Part 1 + Part 2)
```

`DatasetQualificationEngine.qualify(manifest, research_store, *,
required_feature_names=frozenset()) -> DatasetQualificationReport` is the
single entry point. Everything else in this package either feeds it
(`verifier.py`, `dimensions.py`) or consumes its output
(`diagnostics.py`, `verification.py`, `reconciliation.py`, `reports.py`).

## The 8 qualification dimensions

`QUALIFICATION_DIMENSION_ORDER` (`models.py`) fixes the canonical
iteration order used everywhere in this package -- two qualification
runs over the same dataset always produce byte-identical
`dimension_results` ordering, never dict/set iteration order:

1. **Structural Integrity** -- split presence, declared-feature-column
   presence/dtype, content-store metadata cross-check, required-feature
   presence.
2. **Temporal Integrity** -- `open_time` monotonicity/duplicates per
   split, recognized train/eval and fold-group ordering, future leakage.
3. **Statistical Integrity** -- reuses `features.validation.
   validate_research_dataset` over the train split (every issue type
   EXCEPT `TARGET_LEAKAGE_SUSPECTED`, which Temporal Integrity owns).
   Never blocking.
4. **Coverage** -- requested-vs-observed date-range coverage fraction,
   per-feature null fractions. Never blocking.
5. **Stability** -- reuses `features.drift.compare_splits` for
   train-vs-validation/test PSI, constant/near-constant features. Never
   blocking.
6. **Determinism** -- cross-checks `ResearchDatasetManifest.
   output_content_hashes`/`row_counts` against the content store's own,
   SEPARATELY-persisted `metadata.json`.
7. **Reproducibility** -- re-derives `dataset_id` via
   `features.manifests.compute_dataset_id` (the SAME function
   `ResearchDatasetBuilder.build` itself uses); checks required
   provenance fields are present.
8. **Safety** -- reserved `label_`/`target_`-prefixed feature columns,
   non-finite label values. Never blocking.

## Blocking-failure-to-dimension mapping

The spec names exactly 6 blocking-failure codes; every other finding,
however severe, is a `finding`/`warning` only:

| Code | Owning dimension |
|---|---|
| `FUTURE_LEAKAGE` | Temporal Integrity |
| `MANIFEST_CORRUPTION` | Structural Integrity |
| `REPLAY_MISMATCH` | Determinism |
| `IDENTITY_MISMATCH` | Reproducibility |
| `MISSING_LINEAGE` | Reproducibility |
| `REQUIRED_FEATURE_MISSING` | Structural Integrity |

## Decision rule

Deliberately the simplest rule that satisfies the spec, and simpler than
`robustness.promotion`'s 4-tier mandatory/advisory precedence (which this
package's spec has no equivalent of): any blocking failure, from any
dimension, forces `REJECTED_FOR_RESEARCH`; zero blocking failures means
`APPROVED_FOR_RESEARCH`. A WARNING-level finding never by itself blocks
approval. `overall_score` is the unweighted mean of the 8 dimension
scores -- deterministic, no hidden per-dimension weighting.

## The evidence model (Part 2)

`evidence.Evidence` is the atomic unit every deep-diagnostic check
emits: `finding`, `evidence` (a tuple of strings, always referencing an
IMMUTABLE identity -- `dataset_id`/`content_id`/`version`/split name,
never a filesystem path), `severity` (reused `historical.quality.
Severity`), `dimension`, `recommendation`, `affected_artifacts`, and a
`blocking` flag. `evidence.affected_split(dataset_id, content_id,
split_name)` is the one canonical way every check spells a split's
identity, so two records about the same split are always textually
identical.

This is additive to, not a replacement for, `DimensionResult`'s Part 1
`findings`/`warnings`/`recommendations` plain strings -- those are
unchanged. `Evidence` exists specifically for `QualificationDiagnostics`'
6 new sections (`diagnostics.py`).

## Deep diagnostics (Part 2)

`compute_diagnostics(manifest, report, research_store, *,
required_feature_names=frozenset())` re-verifies the manifest fresh (via
`QualificationVerifier`, never trusting a cached reference) and produces,
in addition to Part 1's per-split summary and flat dimension-score map,
6 sections of `Evidence`:

- **Structural** -- schema, manifests, lineage, identity, replay
  evidence (one record each, plus one per split).
- **Temporal** -- macro/cross-asset availability, session alignment,
  stale-macro/stale-cross-asset (see "Macro/cross-asset scope" below),
  future visibility.
- **Statistical** -- NaN, Infinity, duplicate rows, duplicate
  timestamps, zero variance, near-zero variance (all four of the last
  via a SINGLE `features.drift.compare_splits(df, df, ...)`
  self-comparison per split -- constant/near-constant/highly-correlated
  detection is a side effect of that one call, never reimplemented),
  abnormal distributions (population skew/kurtosis via a dedicated
  numpy helper, finite-values-only -- see "Real defects found" below).
- **Coverage** -- feature coverage, source coverage, macro/cross-asset
  coverage, warmup (leading run of any-null rows), missing intervals
  (`open_time` gaps larger than `base_timeframe.duration`).
- **Stability** -- rolling variance/missingness (pandas `.rolling()`
  over the train split), distribution drift/PSI (itemized per-feature,
  reusing the same `compare_splits` call Part 1's `evaluate_stability`
  makes), regime drift (train's own first-half-vs-second-half PSI via
  the same primitive -- a legitimate, real proxy given no model-training
  or regime-labeling machinery is in scope).
- **Safety** -- leakage evidence, mutable aliases (the SAME
  `compare_splits(df, df, ...)` self-comparison Statistical's
  zero-variance check uses -- `highly_correlated_pairs` is the mutable-
  alias signal), label contamination (direct feature/label correlation
  check, distinct from the reserved-prefix check), preprocessing
  contamination (`preprocessing_definition` declared but
  `fitted_preprocessing_fingerprint` absent).

**Deliberate, documented overlap** (not hidden): `temporal_evidence`'s
"future visibility" and `safety_evidence`'s "leakage evidence" both
surface the SAME `VerificationFacts.leakage_messages` -- two different
questions asked of one fact ("is temporal ordering violated" vs. "is
this safe to train on"), not two competing detectors.

### Macro/cross-asset scope

Availability/coverage evidence for macro and cross-asset sources is read
EXCLUSIVELY from `ResearchDatasetManifest.market_data_lineage` (the
payload Milestone 10 Phase 4D's `features.market_data_bridge.lineage.
build_market_data_lineage` already persisted at build time). This module
never re-reads raw macro/cross-asset source data and never recomputes a
fresh `features.market_data_bridge.staleness.StalenessFinding` -- that
would require re-joining raw external series, crossing the "no second
FeatureEngine/builder" boundary this package has held since Part 1.
Per-row staleness re-verification is therefore explicitly OUT OF SCOPE
post-hoc; every stale-macro/stale-cross-asset `Evidence` record says so
directly, and `market_data_lineage.coverage_decision`'s own `status`/
`coverage_fraction` per source (frozen at build time) is the closest
available signal. For the (default, most common) case where a manifest
was built the ORIGINAL way -- a real `historical.loader.DatasetLoader`,
no `market_data_bridge` involvement -- `market_data_lineage` is `None`
and the section reports that plainly rather than fabricating findings.

## Independent verification (Part 2)

`verification.QualificationIndependentVerifier` answers a different
question than `QualificationVerifier`: given an ALREADY-PRODUCED
`DatasetQualificationReport` (e.g. one loaded back from persisted JSON),
can it be trusted? Two checks, neither trusting the report's own claims:

1. **Self-consistency** (`verify_report_self_consistency`, pure, no
   I/O) -- recomputes `overall_score`/`blocking_failure_count`/
   `decision` from the report's own `dimension_results`, using an
   INDEPENDENTLY reimplemented copy of `engine.py`'s tiny decision rule
   (never importing the private `engine._decide`, so a shared bug
   between the two would not go undetected).
2. **Re-qualification** -- runs a FRESH `DatasetQualificationEngine.
   qualify()` against the live manifest/store and diffs the result
   against the supplied report using `QualificationReconciliation` at
   zero score tolerance.

A mismatch in either check is a normal, non-raising outcome
(`IndependentVerificationResult.verified=False`) -- exactly like a
`ReconciliationIssue`, never an exception. A report whose `dataset_id`
doesn't even match the manifest being verified against is likewise a
graceful `verified=False` (a `dataset_id_mismatch` reconciliation
issue), not a crash. `QualificationVerificationError` is reserved for
genuinely being unable to attempt the checks at all (e.g. live artifacts
unreadable).

## Reconciliation (Part 1 + Part 2)

`QualificationReconciliation.reconcile(baseline, candidate, *,
score_tolerance=0.01)` compares two reports for the SAME `dataset_id`.
Part 1 detected `decision_mismatch`, per-dimension `dimension_score_drift`,
and `blocking_failure_set_changed`. Part 2 adds per-dimension
`warning_drift`/`recommendation_drift` (set diffs over `DimensionResult.
warnings`/`.recommendations`), and `finding_drift` for every dimension
EXCEPT `REPRODUCIBILITY`, whose own finding drift is reported as
`lineage_drift` instead -- the spec names "lineage drift" as its own
category, and `REPRODUCIBILITY` is the dimension that owns lineage
(`MISSING_LINEAGE -> Reproducibility`). Reconciling reports for two
different `dataset_id`s remains a structural precondition violation
(`QualificationReconciliationError`) -- there is nothing to reconcile.

## Reports

`reports.py`, 7 deterministic, sorted, diff-friendly plain-text
renderers (mirrors `features.market_data_bridge.reports`'s established
convention): `render_dataset_qualification_report` (Qualification
Report), `render_qualification_diagnostics` (Diagnostics Report, now
including all 6 evidence sections), `render_qualification_reconciliation`
(Reconciliation Report), `render_independent_verification_report`
(Verification Report), `render_evidence_report` (every `Evidence` record
across all 6 sections, grouped by dimension), `render_blocking_failure_report`,
`render_recommendation_report` (both filtered views over a single
report's `dimension_results`).

## Invariance and determinism proofs

Proven directly against the real pipeline (`tests/unit/qualification/
test_qualification_invariance.py`), never asserted by inspection alone:

- **Truncation invariance** -- (1) row-level, order-sensitive checks
  (`verify_no_future_leakage`, monotonicity) produce identical findings
  for a fixed row window regardless of whether rows after that window
  exist in the frame passed in; (2) a REAL rebuild with a shorter
  requested date range reproduces byte-identical `open_time`/feature
  values for every row the two builds share.
- **Replay invariance** -- build, qualify, `shutil.rmtree` the entire
  content directory, rebuild from the same source/recipe, re-qualify:
  `dataset_id`/`content_id`/decision/every dimension score/every
  blocking failure are identical.
- **Determinism** -- the qualification report is run as a subprocess
  with `PYTHONHASHSEED` set to `0`, `1`, and `random`, each against its
  own fresh filesystem root, and the resulting reports (`generated_at`
  fields stripped) are asserted byte-identical.

## Adversarial audit

26 tests (`test_qualification_adversarial.py`), one per attack the spec
names, each run against the real pipeline. See the delivery report for
the full list and results; one real defect was found and fixed during
this audit (documented there in full).

## Known limitations

- **Macro/cross-asset staleness is a build-time snapshot, not a live
  re-verification** -- see "Macro/cross-asset scope" above. This is a
  deliberate architectural boundary, not an oversight.
- **`Evidence.affected_artifacts` row-level citations are positional,
  not UUID-based** -- individual rows carry no identity of their own in
  this system; a citation is only as stable as `(dataset_id, content_id,
  split, row position)`, the closest available proxy.
- **Abnormal-distribution detection (skew/kurtosis) is a heuristic, not
  a formal statistical test** -- documented in the `Evidence.recommendation`
  text itself, not silently presented as definitive.
- **`features.drift.compare_splits` (Milestone 3, unmodified, reused
  here) emits benign `RuntimeWarning`s from pandas/numpy internals when
  a column contains injected `+/-inf` values** (computing `.std()`/
  `.corr()` over non-finite data). This is pre-existing M3 behavior,
  out of this package's scope to modify; it does not corrupt results
  (verified by `TestInfinityInjection`'s regression test) and the
  Infinity check itself still flags the column at CRITICAL severity
  through a separate, unaffected code path.
- **`QualificationDiagnostics`/`Evidence`/`IndependentVerificationResult`
  are not yet wired into a persistence store** -- this package produces
  in-memory dataclasses with `to_json_dict`/`from_json_dict`; nothing
  here writes to `ml.persistence`'s artifact store. Out of scope for
  Phase 1; a natural Phase 2+/Milestone 12 extension point.
- **No CLI surface** -- this package is a library only, matching every
  prior milestone's own disclosed pattern before a CLI phase followed.
