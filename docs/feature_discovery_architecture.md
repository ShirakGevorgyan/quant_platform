# Feature Discovery (Milestone 11, Phase 2)

This is the authoritative technical reference for
`quant_platform.feature_discovery` -- the deterministic engine that
scientifically evaluates every feature in an ALREADY-BUILT research
dataset (a `features.manifests.ResearchDatasetManifest` produced by the
real, unmodified `features.dataset_builder.ResearchDatasetBuilder`, and
typically already `APPROVED_FOR_RESEARCH` by `quant_platform.
qualification`).

**The output is NOT a model, and it is not a prediction target signal.**
This package never trains a model, never computes SHAP, permutation
importance, Boruta, recursive feature elimination, mutual information,
Information Coefficient, Rank IC, feature importance, or any statistic
that requires a prediction target -- those belong after the Label
Framework exists (a later milestone). It never builds a second
`FeatureEngine`/`FeatureRegistry`/`ResearchDatasetBuilder`/
`DatasetQualificationEngine` -- it reads an already-built manifest, its
durable artifacts, and the caller's own live `FeatureRegistry` only, and
reuses `qualification.verifier`'s pure identity/artifact-integrity/
leakage helpers and `features.drift`/`features.validation`'s pure
statistical functions directly rather than reimplementing them.

Delivered in two parts, matching the two halves of the governing
specification:

- **Part 1 -- Feature Discovery & Signal Diagnostics.** 8 named
  deliverables (`FeatureDiscoveryEngine`, `FeatureStatistics`,
  `FeatureSignalDiagnostics`, `FeatureDiscoveryReport`,
  `FeatureDiscoveryVerifier`, `FeatureDiscoveryReconciliation`,
  `FeatureDiscoveryEvidence`, `FeatureDiscoveryReports`) covering 10
  scientific-quality dimensions per feature: Information Content,
  Temporal Stability, Regime Stability, Drift Behaviour, Redundancy,
  Coverage, Availability, Leakage Safety, Determinism, Reproducibility.
- **Part 2 -- Feature Discovery Infrastructure Completion** (this
  report's primary subject) -- feature identity, provenance, lineage,
  dependency graph, catalog/inventory/manifest, health/usage/group
  reports, independent verification, reconciliation, and 7 report
  types. Part 2 explicitly reuses every Part 1 type and function
  (`SharedDiscoveryFacts`, `compute_feature_signal_diagnostics`) and
  reuses `features.models.FeatureSpec`/`features.registry.
  FeatureRegistry`/`features.lineage.build_lineage` directly rather
  than duplicating any of them; nothing in Part 1's architecture was
  redesigned.

## Architecture

```
features.manifests.ResearchDatasetManifest (Milestone 3, unmodified)
features.manifests.ResearchDatasetStore     (Milestone 3, unmodified)
features.registry.FeatureRegistry           (Milestone 1/2, unmodified, caller-supplied)
   |
qualification.verifier (reused)   -- identity/artifact/lineage/leakage FACTS
   |
feature_discovery.diagnostics.compute_shared_discovery_facts  -- SharedDiscoveryFacts (Part 1)
   |
   +-- feature_discovery.statistics (10 pure per-feature statistics, Part 1)
   +-- feature_discovery.diagnostics.compute_feature_signal_diagnostics  -- 10-dimension scoring (Part 1)
   +-- feature_discovery.engine.FeatureDiscoveryEngine  -- orchestrates all features (Part 1)
   +-- feature_discovery.verification / .reconciliation / .reports  -- Part 1 verify/diff/render
   |
   +-- feature_discovery.registry_snapshot.capture_feature_registry_snapshot  -- FeatureRegistrySnapshot (Part 2)
          |
          +-- feature_discovery.graph.build_feature_dependency_graph        -- FeatureDependencyGraph (Part 2)
          +-- feature_discovery.catalog.build_feature_catalog/_inventory/_manifest  (Part 2)
          +-- feature_discovery.catalog.build_feature_infrastructure_bundle  -- FeatureInfrastructureBundle (Part 2)
                 |
                 +-- feature_discovery.health   -- FeatureHealthReport / Usage / Group (Part 2)
                 +-- feature_discovery.infra_verification / .infra_reconciliation / .infra_reports  (Part 2)
```

Package layout:

```
src/quant_platform/feature_discovery/
    __init__.py             package docstring, mission, "Do NOT" list
    models.py                 Part 1: shared vocabulary -- FeatureStatistics, FeatureSignalDiagnostics,
                               FeatureDiscoveryReport, dimension enums
    evidence.py                 Part 1: FeatureDiscoveryEvidence -- the atomic finding/evidence record
    statistics.py                 Part 1: the 10 pure per-feature statistic computations
    diagnostics.py                  Part 1: SharedDiscoveryFacts + compute_feature_signal_diagnostics
                                     (10-dimension evaluators)
    engine.py                        Part 1: FeatureDiscoveryEngine orchestration
    verification.py                    Part 1: FeatureDiscoveryVerifier
    reconciliation.py                    Part 1: FeatureDiscoveryReconciliation
    reports.py                             Part 1: 7 render_* functions

    metadata.py                Part 2: FeatureMetadata, FeatureProvenance, FeatureVersionHistory
    registry_snapshot.py         Part 2: FeatureRegistrySnapshot (captures metadata+lineage+provenance)
    graph.py                       Part 2: FeatureDependencyGraph (nodes, edges, cycles, missing
                                    parents, duplicate derivations, orphans)
    catalog.py                       Part 2: FeatureCatalog, FeatureInventory, FeatureManifest,
                                      FeatureInfrastructureBundle
    health.py                          Part 2: FeatureHealthReport, FeatureUsageReport,
                                        FeatureGroupReport
    infra_reconciliation.py              Part 2: FeatureInfrastructureReconciliation
    infra_verification.py                  Part 2: FeatureInfrastructureVerifier
    infra_reports.py                         Part 2: 7 render_* infrastructure report functions
```

## Part 1 -- the 10 feature discovery dimensions

`FeatureDiscoveryEngine.discover(manifest, research_store)` runs every
feature in `manifest.feature_names` through `compute_feature_signal_
diagnostics`, which evaluates 10 dimensions purely from `features.
statistics`' pre-computed per-feature statistics and `SharedDiscoveryFacts`
(itself built once per dataset via `qualification.verifier`, `features.
drift`, and `features.validation` -- never recomputed per feature):
Information Content, Temporal Stability, Regime Stability, Drift
Behaviour, Redundancy, Coverage, Availability, Leakage Safety,
Determinism, Reproducibility. Each dimension produces `FeatureDiscoveryEvidence`
records (`finding`, `evidence`, `severity`, `dimension`, `blocking_code`);
a feature with any blocking evidence is excluded from the discovery
report's "usable" set, mirroring `qualification`'s own blocking-failure
convention without duplicating its decision-rule code.

Full detail on the 10 dimensions' individual logic, the Part 1
adversarial audit (16 items), and Part 1's determinism proofs lives in
the module docstrings of `statistics.py`/`diagnostics.py` and
`test_feature_discovery_adversarial.py`/`test_feature_discovery_
determinism.py` -- not duplicated here, since Part 2 makes no change to
any of it.

## Part 2 -- infrastructure

Part 2 answers a different question than Part 1: not "is this feature's
signal any good" (Part 1), but "what IS this feature, where did it come
from, what does it depend on, and can two snapshots of that be proven
identical or shown to have drifted." No statistic in Part 2 depends on
a prediction target, a label, or a model.

### Feature Metadata, Provenance, Version History (`metadata.py`)

Every field is read directly from the REAL `features.models.FeatureSpec`
(via the caller's own live `FeatureRegistry.get(name, version).spec` --
exactly the calling convention `feature_cli.py`'s own `report-lineage`
command already uses) and `ResearchDatasetManifest` -- never fabricated,
never recomputed a second way:

- **`FeatureMetadata`** -- the 11 fields the spec names: `schema_version,
  feature_id, feature_name, feature_group, origin_dataset,
  origin_manifest, creation_stage, availability_rule,
  warmup_requirement, dependencies, deterministic_identity`.
  - `feature_id = spec.qualified_name` (`f"{name}@{version}"`) --
    dataset-independent, matching `FeatureSpec`'s own `(name, version)`
    identity.
  - `feature_group = spec.category.value`.
  - `creation_stage` -- `"higher_order_feature"` if `spec.
    feature_dependencies` is non-empty; `"derived_feature"` if only
    `spec.required_inputs` is non-empty; `"raw_source"` only for a
    feature needing no inputs of any kind.
  - `availability_rule` -- a descriptive string derived from `spec.
    source_timeframe`/`spec.availability_delay`, e.g. `"available at
    open_time + M1 bar duration + 30s release delay"`.
  - `dependencies = spec.feature_dependencies` -- other FEATURE names,
    distinct from `spec.required_inputs` (raw OHLCV/aux column names).
  - `deterministic_identity = spec.fingerprint()` -- unchanged iff
    every field of the spec is unchanged; no new hash invented.
- **`FeatureProvenance`** -- the chain-of-custody a bare `FeatureMetadata`
  doesn't carry: `source_historical_dataset_id`,
  `source_historical_manifest_version`, `source_symbols`,
  `source_timeframe`, `code_revision`, `has_market_data_lineage`.
- **`FeatureVersionHistory`** -- every version of a feature name
  registered in the live `FeatureRegistry` (via `registry.
  list_features()`, filtered by name, sorted by version) -- not merely
  the one version the current dataset happens to use.

### Feature Registry Snapshot (`registry_snapshot.py`)

`capture_feature_registry_snapshot(registry, manifest) ->
FeatureRegistrySnapshot` is the ONE place `metadata.py`'s two `compute_*`
functions and the real `features.lineage.build_lineage` are called, for
every feature in `sorted(manifest.feature_names)`. Everything else in
Part 2 (the graph, catalog, inventory, health reports) is built FROM the
resulting snapshot, never by re-reading the registry a second time --
this decouples every downstream check from requiring a live registry at
all once a snapshot has been captured, which matters directly for
`infra_verification.py`'s re-derivation and `infra_reconciliation.py`'s
two-snapshot comparison. A manifest declaring a feature the supplied
registry cannot resolve raises `FeatureDiscoveryVerificationError`
(never silently drops the feature).

### Feature Dependency Graph (`graph.py`)

`build_feature_dependency_graph(snapshot, declared_feature_names=None)
-> FeatureDependencyGraph` is built PURELY from an already-captured
snapshot's own `metadata` (for `feature_dependencies`) and `lineages`
(for `required_inputs`) -- never re-reading a live registry. Models the
spec's DAG directly:

```
raw source / market data  ->  derived feature  ->  higher-order feature
```

- **Node kinds** -- `RAW_SOURCE`/`MARKET_DATA` (one input node per
  distinct `required_inputs` name; classified `MARKET_DATA` if the
  referencing feature's `feature_group` is in `{"macro",
  "cross_asset"}`, else `RAW_SOURCE` -- a documented heuristic since raw
  input column names alone don't indicate source kind) and
  `DERIVED_FEATURE`/`HIGHER_ORDER_FEATURE` (feature nodes, `node_id =
  feature_id`, classified by whether `dependencies` is non-empty).
- **Cycle detection** -- `_detect_cycles`, an INDEPENDENT DFS
  reimplementation, deliberately NOT calling `FeatureRegistry.
  resolve_dependency_order()` -- a live registry may not be available
  when reconciling two PERSISTED snapshots, and independently
  re-deriving is the entire point of verification.
- **Missing parents** -- a `dependencies` name with no corresponding
  feature node in the graph; returned as `(feature_name,
  missing_dependency_name)` pairs.
- **Duplicate derivations** -- a narrow, explicitly documented heuristic
  (`_recipe_signature(feature_group, source_timeframe, required_inputs,
  dependencies)`); two features sharing this signature are flagged.
  Does NOT inspect `deterministic_params` (not carried by
  `FeatureMetadata`/`FeatureLineage`) -- a disclosed false-positive-
  acceptance tradeoff, not silently overclaimed precision.
- **Orphan features** -- a metadata entry OUTSIDE the caller-supplied
  `declared_feature_names` set (defaults to the snapshot's own metadata
  names, i.e. zero orphans by construction in the non-tampered case)
  that nothing else depends on. This definition was chosen specifically
  to avoid false positives on legitimate leaf/base features (e.g. a
  feature with zero `required_inputs` and zero `feature_dependencies`
  is completely normal and must not be flagged) -- it only fires
  against a hand-modified/tampered snapshot, which is its intended use.
- **`is_valid`** -- `not (cycles or missing_parents)`. Duplicate
  derivations and orphans are reported but do not, by themselves, make
  a graph invalid (they are quality signals, not structural breaks).

### Feature Catalog, Inventory, Manifest, Bundle (`catalog.py`)

- **`FeatureCatalog`** -- the flat, complete listing of every
  `FeatureMetadata` in a snapshot, sorted by `feature_name`.
- **`FeatureInventory`** -- the 5 named catalog views over one
  `FeatureCatalog`: `complete_catalog`, `grouped_catalog` (by
  `feature_group`), `dataset_catalog` (by `origin_dataset`),
  `origin_catalog` (by `creation_stage`), `availability_catalog` (by
  `availability_rule`) -- each a `dict[str, tuple[str, ...]]`.
- **`FeatureManifest`** -- this package's OWN deterministic,
  content-addressed identity for a snapshot's exact feature set:
  `manifest_id = sha256(f"{dataset_id}|{','.join(sorted(feature_ids))}")
  [:16]`. Distinct from, and never a replacement for, `features.
  manifests.ResearchDatasetManifest` (the dataset's own manifest).
- **`FeatureInfrastructureBundle`** -- bundles `snapshot, graph,
  catalog, inventory, manifest` together; the single object
  `infra_verification.py`/`infra_reconciliation.py` operate on, built
  in one call via `build_feature_infrastructure_bundle(registry,
  manifest)`.

### Feature Health, Usage, Group Reports (`health.py`)

No predictive metric anywhere in this module.

- **`FeatureHealthReport`** -- REUSES Part 1's `FeatureSignalDiagnostics`
  directly (`diagnostics.compute_feature_signal_diagnostics`) rather
  than reimplementing constant/near-constant/missing/warmup/coverage/
  availability/determinism/reproducibility(identity) checks a second
  time -- Part 1's own 10 dimensions already cover 9 of the 10 named
  health checks. The 10th, **lineage**, was a real, pre-existing gap:
  `SharedDiscoveryFacts.lineage_present`/`.missing_lineage_fields` were
  already computed by Part 1 (via `qualification.verifier.
  verify_lineage`) but never referenced by any of Part 1's 10 dimension
  evaluators. `FeatureHealthReport` is where this fact is finally
  surfaced: `is_healthy = not diagnostics.is_blocking and not
  has_warning_or_worse and facts.lineage_present`.
- **`FeatureUsageReport`** -- built from `FeatureDependencyGraph`:
  `depends_on` (forward edges), `depended_on_by` (reverse edges),
  `required_input_count`, `is_root` (no `depends_on`), `is_leaf` (no
  `depended_on_by`).
- **`FeatureGroupReport`** -- aggregates `FeatureCatalog` entries by
  `feature_group`, cross-referencing `FeatureHealthReport`s for
  `healthy_count`/`unhealthy_count`.

### Verification (`infra_verification.py`)

Never trusts a cached `FeatureInfrastructureBundle`. Two independent
checks, exactly mirroring the pattern `qualification.verification`/
`feature_discovery.verification` (Part 1) already established:

1. **`verify_bundle_self_consistency`** -- pure, no I/O -- independently
   recomputes `FeatureManifest.manifest_id` and `feature_ids` from
   `snapshot.metadata` (a small, DELIBERATELY separate reimplementation
   of `catalog.build_feature_manifest`'s own hash formula, so a bug
   shared between the two would not go undetected), and, wherever a
   live `registry` is supplied, re-verifies every catalog entry's
   `deterministic_identity` against a FRESH `FeatureSpec.fingerprint()`
   recomputation (identity verification). Catches `UnknownFeatureError`
   narrowly, not a bare `except Exception`.
2. **Full re-capture** -- a fresh `build_feature_infrastructure_bundle()`
   run against the live `registry`/`manifest`, diffed against the
   supplied bundle via `FeatureInfrastructureReconciliation`.

A mismatch in either check is a normal, non-raising outcome
(`FeatureInfrastructureVerificationResult.verified=False`), never an
exception. A `bundle.snapshot.dataset_id` that doesn't match the
supplied `manifest.dataset_id` is likewise a graceful `verified=False`
with a `dataset_id_mismatch` reconciliation issue, not a crash.

### Reconciliation (`infra_reconciliation.py`)

`FeatureInfrastructureReconciliation.reconcile(baseline, candidate)`
compares two bundles (two inventories, two manifests, two catalogs --
all bundled together since they are built from the same snapshot) for
the SAME `dataset_id`, detecting the 5 named drift kinds:

- **`feature_drift`** -- feature name sets differ.
- **`metadata_drift`** -- per-feature, via `FeatureMetadata` dataclass
  `!=` (catches any tampered field: `feature_group`, `availability_rule`,
  `warmup_requirement`, etc.).
- **`lineage_drift`** -- per-feature, via `FeatureLineage` dataclass
  `!=`.
- **`dependency_drift`** -- graph `edges`/`cycles`/`missing_parents`/
  `orphan_features` tuple comparison.
- **`manifest_drift`** -- `manifest_id` differs.

Reconciling two bundles for different `dataset_id`s is a structural
precondition violation (there is nothing to reconcile) and raises
`FeatureDiscoveryReconciliationError`; every other disagreement is a
normal, non-raising `FeatureInfrastructureReconciliationIssue`.

### Reports (`infra_reports.py`)

7 deterministic, sorted, plain-text renderers, mirroring `qualification.
reports`/`feature_discovery.reports`'s (Part 1) established convention:
`render_feature_catalog_report`, `render_feature_inventory_report`,
`render_dependency_report`, `render_health_report`,
`render_metadata_report`, `render_infrastructure_verification_report`,
`render_infrastructure_reconciliation_report`. All render
already-computed objects only -- no discovery/verification/
reconciliation logic lives in this module.

## Determinism

Every Part 2 type is a frozen dataclass with `to_json_dict`/
`from_json_dict` (schema-versioned via `require_schema_version`), and
every builder function is a pure function of its inputs -- no
randomization, no wall-clock dependency except the deliberately-excluded
`captured_at`/`generated_at` timestamp fields. Proven directly
(`test_feature_discovery_infra_determinism.py`, 5 tests): repeat
verification x10, repeat reconciliation x10, repeat catalog-report-render
x10, repeat verification-report-render x10, and repeat bundle-capture
x10 (the underlying operation every other repeat test builds on) are
each byte-identical to their own first run once volatile timestamp
fields are stripped.

## Adversarial audit (Part 2)

15 tests (`test_feature_discovery_infra_adversarial.py`), one class per
each of the 13 named attacks plus one extra structural guard, all run
against the real infrastructure pipeline -- never a mock: feature id
tampering, metadata tampering, dependency corruption, lineage
corruption, manifest corruption, cycle injection, orphan feature,
duplicate feature (refused by the real `FeatureRegistry.register` via
`DuplicateFeatureError`), missing parent, schema mismatch, availability
mismatch, warmup corruption, identity corruption, plus a cross-dataset
reconciliation guard (reconciling two genuinely unrelated bundles must
never be silently accepted). Full results and the two test-authoring
bugs found (and fixed) during this audit are in the delivery report.

## Known limitations

- **Duplicate-derivation detection is a heuristic, not a proof** -- see
  `_recipe_signature`'s own docstring in `graph.py`: it does not inspect
  `deterministic_params`, so two features that share a recipe signature
  but differ only in a parameter value are a disclosed false positive.
- **`MARKET_DATA` vs. `RAW_SOURCE` node classification is a heuristic**
  -- based on the referencing feature's `feature_group`, since raw input
  column names alone carry no explicit source-kind marker.
- **No CLI surface** -- this package is a library only, matching every
  prior milestone's own disclosed pattern before a CLI phase followed.
- **Not yet wired into a persistence store** -- Part 2 produces
  in-memory dataclasses with `to_json_dict`/`from_json_dict`; nothing
  here writes to `ml.persistence`'s artifact store, matching Part 1's
  own disclosed limitation.
- **No statistic in this package depends on a prediction target or a
  label** -- by design; Information Coefficient, Rank IC, Mutual
  Information, Feature/Permutation Importance, SHAP, Boruta, RFE, and
  correlation-to-labels are explicitly out of scope until the Label
  Framework exists.
