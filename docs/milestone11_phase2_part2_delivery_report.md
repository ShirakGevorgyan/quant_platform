# Milestone 11, Phase 2, Part 2 — Delivery Report

**Feature Discovery Infrastructure Completion: feature identity,
provenance, lineage, dependency graph, catalog/inventory/manifest,
health/usage/group reports, independent verification, reconciliation,
and reports — no statistic anywhere that requires a prediction target.**

Part 2 of Phase 2 completes the Feature Discovery infrastructure begun
in Part 1 (Feature Discovery & Signal Diagnostics). Part 1 answered
"how good is this feature's signal"; Part 2 answers "what is this
feature, where did it come from, what does it depend on, and can two
snapshots of that be proven identical or shown to have drifted." Every
type in Part 2 reuses Part 1's `SharedDiscoveryFacts`/
`compute_feature_signal_diagnostics` and the real, unmodified
`features.models.FeatureSpec`/`features.registry.FeatureRegistry`/
`features.lineage.build_lineage` directly — nothing in Part 1's
architecture, or in any earlier milestone, was redesigned.

---

## 1. Baseline

Work began at `HEAD` = `33b4f2218e09fafdf2f95927f193aa1dbdfe70a6`
("Add deterministic research dataset qualification engine" — Milestone
11, Phase 1, committed and reviewed). Milestone 11, Phase 2, Part 1
(Feature Discovery & Signal Diagnostics) was implemented on top of this
baseline with nothing committed: 8 source modules, 102 tests, all
passing, `ruff`/`mypy` clean. Part 2 began by confirming: Part 1's 8
source modules exist and its 102 tests pass; `git status --short`
reflects exactly the expected uncommitted Phase 2 work (one modified
file from Part 1 plus new Part 1 package/test directories); nothing
staged, committed, or pushed. `HEAD` remains
`33b4f2218e09fafdf2f95927f193aa1dbdfe70a6` throughout — no commit was
made at any point in Phase 2 (Parts 1 or 2).

## 2. Files added / modified

**Modified (4 files, all in `tests/unit/qualification/`, fixing a
pre-existing test-infrastructure defect — see Section 9):**

```
tests/unit/qualification/conftest.py                        +27 lines
tests/unit/qualification/test_qualification_adversarial.py   +4/-3 lines
tests/unit/qualification/test_qualification_invariance.py   +14/-9 lines
tests/unit/qualification/test_qualification_repetition_gates.py +5/-3 lines
```

`src/quant_platform/core/exceptions.py` (Part 1's own addition of
`FeatureDiscoveryError` and its subclasses) was **not** touched again
in Part 2 — Part 2 needed no new exception types, reusing Part 1's
hierarchy (`FeatureDiscoveryError`, `FeatureDiscoveryRequestError`,
`FeatureDiscoveryVerificationError`, `FeatureDiscoveryReconciliationError`)
as-is.

**New (8 source files, 1,355 lines, added to the existing
`src/quant_platform/feature_discovery/` package):**

```
src/quant_platform/feature_discovery/
    metadata.py             211 lines   FeatureMetadata, FeatureProvenance, FeatureVersionHistory
    registry_snapshot.py     105 lines   FeatureRegistrySnapshot + capture_feature_registry_snapshot
    graph.py                  228 lines   FeatureDependencyGraph + build_feature_dependency_graph
    catalog.py                  221 lines   FeatureCatalog, FeatureInventory, FeatureManifest,
                                            FeatureInfrastructureBundle
    health.py                     174 lines   FeatureHealthReport, FeatureUsageReport, FeatureGroupReport
    infra_reconciliation.py         141 lines   FeatureInfrastructureReconciliation
    infra_verification.py             144 lines   FeatureInfrastructureVerifier
    infra_reports.py                    131 lines   7 render_* infrastructure report functions
```

(Package total after Part 2, including Part 1's 9 source files: 17
files, 3,247 lines.)

**New (1 documentation file):**

- `docs/feature_discovery_architecture.md` (377 lines) — full
  architecture reference covering both Part 1 and Part 2 (Part 1 had no
  standalone architecture doc; this is the first, written fresh to
  cover the whole package rather than duplicated per-part).

**New (1 documentation file, this report):**

- `docs/milestone11_phase2_part2_delivery_report.md`.

**New (9 test files, 828 lines, 75 tests, added to the existing
`tests/unit/feature_discovery/` directory; `conftest.py` there was
extended, not replaced):**

```
tests/unit/feature_discovery/
    test_feature_discovery_metadata.py               87 lines    10 tests
    test_feature_discovery_graph.py                 104 lines     7 tests
    test_feature_discovery_catalog.py                 82 lines     9 tests
    test_feature_discovery_health.py                  75 lines     7 tests
    test_feature_discovery_infra_reconciliation.py    67 lines     6 tests
    test_feature_discovery_infra_verification.py      73 lines     8 tests
    test_feature_discovery_infra_reports.py            77 lines     8 tests
    test_feature_discovery_infra_adversarial.py       185 lines    15 tests
    test_feature_discovery_infra_determinism.py        78 lines     5 tests
```

(Directory total after Part 2, including Part 1's 9 test files plus
`conftest.py`: 18 files, 2,089 lines, 177 tests.)

Every new test file is prefixed `test_feature_discovery_infra_` or
`test_feature_discovery_` (matching Part 1's own established
convention), avoiding the basename-collision class of defect Phase 1
discovered and fixed for `tests/unit/qualification/`.

No file outside this list was created, modified, or deleted. No cache,
virtual environment, log, build artifact, dataset, credential, or secret
is present in the working tree (confirmed via `git status --short` —
Section 12).

## 3. Feature Metadata, Provenance, Version History

Every `FeatureMetadata`'s 11 named fields (`feature_id, feature_name,
feature_group, origin_dataset, origin_manifest, creation_stage,
availability_rule, warmup_requirement, dependencies,
deterministic_identity`, plus `schema_version`) are assembled directly
from the real `features.models.FeatureSpec` — read via the caller's own
live `FeatureRegistry`, exactly the calling convention `feature_cli.py`'s
own `report-lineage` command already uses (`registry.get(name,
manifest.feature_versions.get(name)).spec`). `feature_id` is `FeatureSpec.
qualified_name`; `deterministic_identity` is `FeatureSpec.fingerprint()`
directly — no new hash invented anywhere in this module.
`FeatureProvenance` captures the chain-of-custody a bare metadata record
doesn't (source historical dataset/manifest, code revision, whether a
`market_data_lineage` exists). `FeatureVersionHistory` enumerates every
version of a feature registered in the live registry (via `registry.
list_features()`), not merely the one version the current dataset uses.
Full detail in `docs/feature_discovery_architecture.md`.

## 4. Feature Registry Snapshot

`capture_feature_registry_snapshot(registry, manifest) ->
FeatureRegistrySnapshot` is the single place metadata, lineage (via the
real, unmodified `features.lineage.build_lineage`), and provenance are
captured for every feature in a dataset. Everything downstream (graph,
catalog, inventory, health) is built from the resulting snapshot, never
by re-reading the registry a second time. A manifest declaring a
feature the supplied registry cannot resolve raises
`FeatureDiscoveryVerificationError` rather than silently dropping it.

## 5. Feature Dependency Graph

`build_feature_dependency_graph` models `raw source / market data ->
derived feature -> higher-order feature` purely from an
already-captured snapshot (never re-reading a live registry), and
detects all 4 named defect classes:

- **Cycles** — an independent DFS reimplementation, deliberately not
  calling `FeatureRegistry.resolve_dependency_order()`.
- **Missing parents** — a declared dependency with no corresponding
  feature node.
- **Duplicate derivations** — a documented, narrow recipe-signature
  heuristic (feature group, source timeframe, required inputs,
  dependencies); does not inspect `deterministic_params`, a disclosed
  false-positive-acceptance tradeoff.
- **Orphan features** — a metadata entry outside the caller-supplied
  `declared_feature_names` set that nothing else depends on. This
  definition was chosen after explicitly rejecting several alternatives
  that would have falsely flagged legitimate leaf/base features (e.g. a
  feature with zero inputs and zero dependencies, like the `"trend"`
  test fixture, is normal and must not be flagged) — it only fires
  against a hand-tampered snapshot.

`is_valid = not (cycles or missing_parents)`; duplicate derivations and
orphans are reported as quality signals without making the graph
invalid by themselves.

## 6. Feature Catalog, Inventory, Manifest

`FeatureCatalog` is the flat, complete listing. `FeatureInventory`
bundles the 5 named views: complete, grouped (by `feature_group`),
dataset (by `origin_dataset`), origin (by `creation_stage`),
availability (by `availability_rule`). `FeatureManifest` is this
package's own deterministic, content-addressed identity for a
snapshot's exact feature set (`sha256(dataset_id + sorted feature_ids)`,
truncated to 16 hex chars) — distinct from, and never a replacement
for, `features.manifests.ResearchDatasetManifest`. `FeatureInfrastructure
Bundle` bundles `snapshot, graph, catalog, inventory, manifest`
together, built in one call via `build_feature_infrastructure_bundle
(registry, manifest)` — the single object verification and
reconciliation operate on.

## 7. Feature Health, Usage, Group Reports

`FeatureHealthReport` reuses Part 1's `FeatureSignalDiagnostics`
directly rather than reimplementing constant/near-constant/missing/
warmup/coverage/availability/determinism/reproducibility checks a
second time — Part 1's 10 dimensions already cover 9 of the 10 named
health checks. The 10th, **lineage**, was a real, pre-existing gap
discovered during this task: `SharedDiscoveryFacts.lineage_present`/
`.missing_lineage_fields` were already computed by Part 1 (via
`qualification.verifier.verify_lineage`) but never referenced by any of
Part 1's 10 dimension evaluators. `FeatureHealthReport` is where this
fact is finally surfaced (`is_healthy` requires lineage present, in
addition to no blocking/warning-or-worse signal diagnostics evidence).
`FeatureUsageReport` (forward/reverse dependency edges, root/leaf
classification) and `FeatureGroupReport` (per-`feature_group`
healthy/unhealthy counts) are pure aggregations over the dependency
graph and catalog — no new statistics.

## 8. Verification

`FeatureInfrastructureVerifier.verify(bundle, registry, manifest)`
never trusts a cached bundle. Two independent checks, mirroring the
pattern `qualification.verification`/`feature_discovery.verification`
(Part 1) already established:

1. `verify_bundle_self_consistency` — pure, no I/O — independently
   recomputes `FeatureManifest.manifest_id`/`feature_ids` from
   `snapshot.metadata` (a deliberately separate reimplementation of
   `catalog.build_feature_manifest`'s own hash formula, so a bug shared
   between the two would not go undetected), and, wherever a live
   `registry` is supplied, re-verifies every catalog entry's
   `deterministic_identity` against a fresh `FeatureSpec.fingerprint()`
   recomputation.
2. Full re-capture — a fresh `build_feature_infrastructure_bundle()`
   run against the live registry/manifest, diffed via
   `FeatureInfrastructureReconciliation`.

A mismatch in either is a normal, non-raising `verified=False` outcome,
never an exception. A `dataset_id` mismatch between the bundle and the
supplied manifest is likewise a graceful `verified=False` with a
`dataset_id_mismatch` reconciliation issue rather than a crash.

## 9. Real defect found and fixed (test infrastructure)

One real, pre-existing defect was found and fixed during this task's
baseline verification — not in Part 2's own new production code, but in
the test infrastructure shared with Milestone 11 Phase 1's
`qualification` package:

**`sys.modules["conftest"]` collision between
`tests/unit/qualification/` and `tests/unit/feature_discovery/`.** Both
directories have a bare-named `conftest.py`; three qualification test
files (`test_qualification_repetition_gates.py`,
`test_qualification_adversarial.py`, `test_qualification_invariance.py`)
used `from conftest import build_request, trend_registry` — a bare,
non-fixture import of helper functions. Under pytest's default
`prepend` import mode, this resolves `conftest` to whichever
directory's `conftest.py` Python imported first into `sys.modules` in
the session; running `tests/unit/qualification/` and `tests/unit/
feature_discovery/` together (as the "full repository suite" quality
gate does) broke whichever directory's bare import lost that race, with
a confusing `ImportError` for a name that plainly exists in its own
`conftest.py`. This would have made the "full repository suite" quality
gate fail non-deterministically depending on collection order — a real
gap this task's own "repeat verification" and "repeat reconciliation"
requirements would otherwise not have caught.

**Fix**: added `build_request_factory`, `trend_registry_factory`, and
`two_feature_registry_factory` pytest fixtures to `tests/unit/
qualification/conftest.py` (each simply returning the existing bare
function), and converted all 3 affected test files to receive these as
injected fixture parameters instead of importing them via `from
conftest import ...` — pytest's fixture resolution is directory-scoped
and immune to the `sys.modules` flat-cache collision, unlike a bare
Python import. `feature_discovery`'s own test files were confirmed
(via grep) to never use `from conftest import ...`, so no change was
needed there. Verified by running both directories together: 302 tests,
zero import errors, zero failures (Section 11).

No production-code defect was found anywhere in Part 2's own 8 new
modules — this task's own 15-item adversarial audit (Section 10) passed
on its first full run, with two test-authoring bugs (below) needing
correction, not production-code changes.

**Test-authoring bugs found and fixed while writing the adversarial
audit (not production defects):**

- Hand-tampering a `FeatureInfrastructureBundle`'s `snapshot` field via
  `dataclasses.replace` without also rebuilding the derived `catalog`/
  `inventory`/`manifest` fields left those fields stale, causing a
  `metadata_drift` detection test to see no drift (comparing two
  already-identical stale catalogs). Fixed by adding a
  `_rebuild_bundle`/`_rebuild_bundle_from_tampered_snapshot` test helper
  that properly rebuilds every derived field from the tampered
  snapshot before comparing.
- `Test10SchemaMismatch` initially expected
  `(FeatureDiscoveryError, KeyError)`, but the actual (correct)
  exception `require_schema_version` raises is `SchemaVersionError` (a
  sibling under `MLError`, not under `FeatureDiscoveryError`). Fixed by
  importing and expecting `SchemaVersionError` directly — the
  production behavior (schema mismatch correctly rejected) was always
  correct; only the test's expected-exception-type was wrong.

## 10. Adversarial audit

15 tests (`test_feature_discovery_infra_adversarial.py`), one class per
each of the 13 named attacks plus one extra structural guard, all run
against the real infrastructure pipeline — never a mock:

Feature id tampering (caught by `verify_bundle_self_consistency`'s
identity check), metadata tampering (`metadata_drift`), dependency
corruption (a corrupted `dependencies` entry produces a `missing_parent`
and `is_valid=False`), lineage corruption (`lineage_drift`), manifest
corruption (a tampered `manifest_id` fails self-consistency), cycle
injection (`graph.cycles != ()`, `is_valid=False`), orphan feature (a
disconnected extra feature outside `declared_feature_names` is flagged
in `orphan_features`), duplicate feature (refused by the real
`FeatureRegistry.register` via `DuplicateFeatureError` — the registry's
own pre-existing guarantee, not reimplemented), missing parent (a
dependency referencing an unregistered feature), schema mismatch (wrong
`schema_version` rejected by `require_schema_version` for both
`FeatureCatalog` and `FeatureManifest`), availability mismatch
(`metadata_drift`), warmup corruption (`metadata_drift`), identity
corruption (a tampered `deterministic_identity` fails full
verification, both `verified=False` and `self_consistent=False`), plus
a cross-dataset reconciliation guard (reconciling two bundles for
different `dataset_id`s always raises
`FeatureDiscoveryReconciliationError`, never silently accepted).

All 13 named attacks, plus the extra structural guard, are caught or
correctly, honestly handled. See Section 9 for the two test-authoring
bugs found while writing this audit (both fixed; neither indicated a
production defect).

## 11. Reconciliation

`FeatureInfrastructureReconciliation.reconcile(baseline, candidate)`
detects the 5 named drift kinds for two bundles sharing the same
`dataset_id`: `feature_drift` (name sets differ), `metadata_drift`
(per-feature `FeatureMetadata` inequality), `lineage_drift` (per-feature
`FeatureLineage` inequality), `dependency_drift` (graph edges/cycles/
missing-parents/orphans tuple inequality), `manifest_drift`
(`manifest_id` differs). Reconciling two different `dataset_id`s raises
`FeatureDiscoveryReconciliationError` (a structural precondition
violation); every other disagreement is a normal, non-raising
`FeatureInfrastructureReconciliationIssue`. Verified directly
(`test_feature_discovery_infra_reconciliation.py`, 6 tests): clean
self-reconciliation, cross-dataset raise, metadata/manifest/feature
drift detection by construction, full JSON round-trip.

## 12. Reports

7 deterministic, sorted, plain-text renderers
(`infra_reports.py`), mirroring `qualification.reports`/
`feature_discovery.reports`'s (Part 1) established convention: Feature
Catalog Report, Feature Inventory Report, Dependency Report, Health
Report, Metadata Report, Verification Report, Reconciliation Report.
Every renderer performs no discovery/verification/reconciliation logic
of its own — it renders already-computed objects only. Verified against
real, built infrastructure bundles, never a hand-constructed fixture
standing in for genuine output (`test_feature_discovery_infra_reports.py`,
8 tests).

## 13. Determinism

Every Part 2 type is a frozen dataclass with schema-versioned
`to_json_dict`/`from_json_dict`, and every builder function is a pure
function of its inputs — no randomization, no wall-clock dependency
except the deliberately-excluded `captured_at`/`generated_at`
timestamp fields. Proven directly against the real pipeline
(`test_feature_discovery_infra_determinism.py`, 5 tests, each repeating
its operation 10 times and asserting byte-identical results once
volatile timestamp fields are stripped): repeat verification x10,
repeat reconciliation x10, repeat catalog-report-render x10, repeat
verification-report-render x10, and repeat bundle-capture x10 (the
underlying capture operation every other repeat test builds on, proving
`build_feature_infrastructure_bundle` itself is deterministic, not
merely that rendering/verifying an already-fixed bundle is).

## 14. Quality gates

- `git diff --check`: clean (no whitespace errors).
- `ruff check .` (full repository): **all checks passed**.
- `mypy src` (full repository, 401 source files): **no issues found**.
- `pytest tests/unit/feature_discovery/ tests/unit/qualification/`
  (302 tests — 177 feature_discovery [Part 1: 102, Part 2: 75] + 125
  qualification): **all passed**, 92.3s. 8 benign `RuntimeWarning`s,
  all from the pre-existing, documented `features.drift.compare_splits`
  infinity-arithmetic characteristic (Milestone 11 Phase 1's own
  disclosed, unmodified limitation) — zero new warnings introduced by
  Part 2.
- Repeat verification x10, repeat reconciliation x10, repeat
  catalog-report-render x10, repeat verification-report-render x10,
  repeat bundle-capture x10 (`test_feature_discovery_infra_determinism.py`,
  5 tests, Section 13): **all passed** — every one of the 50 repetitions
  byte-identical to its own first run.
- Full repository `pytest` suite: **1 failed, 7201 passed, 3 skipped**,
  in 2:25:52. The 3 skips are the same pre-existing, deliberate ones
  prior phases' own delivery reports recorded
  (`ALPHA_VANTAGE_API_KEY`-gated and `FRED_API_KEY`-gated opt-in
  real-provider acceptance workflows, and one Windows-symlink-privilege-
  gated ML artifact test). The 1 failure —
  `tests/unit/portfolio_risk/test_portfolio_risk_ledger_concurrency.py::
  TestConflictingSecondConsumptionRace::
  test_two_threads_consuming_with_different_identities_exactly_one_wins_and_the_loser_is_audited`
  — is **not caused by this task's work**: the file was last modified in
  commit `7ef860cb964ffd752f58c8c68cc81add08ccb16f` ("Add durable
  portfolio risk authorization lifecycle"), an unrelated, much earlier
  commit this task never touched, and neither Phase 2 Part 2 nor Part 1
  touches `portfolio_risk` anywhere. The test spawns two real threads
  racing to consume the same risk-ledger authorization and asserts a
  specific interleaving outcome (exactly one "ok", exactly one audited
  rejection); re-run in isolation 5 consecutive times immediately after
  the full-suite failure, it **passed every time** (0.41-0.42s each) —
  consistent with a pre-existing, load-sensitive flaky concurrency test
  (thread-scheduling timing shifted by the other ~7,200 tests' CPU/memory
  pressure ahead of it in the same 2h26m process, not a logic defect this
  task introduced or is in scope to fix). No `feature_discovery`,
  `qualification`, or `portfolio_risk` test failed in the dedicated
  302-test run (previous bullet) or in isolation. This is reported here
  in full rather than silently omitted or worked around; fixing an
  unrelated, pre-existing flaky test in `portfolio_risk` is out of this
  task's scope and was not attempted.

No test was weakened, skipped, or loosened to make it pass; every
assertion added or modified during Part 2 is a genuine strengthening of
coverage, and the one pre-existing defect found (Section 9) was fixed
at its root cause, not worked around.

## 15. Known limitations

See `docs/feature_discovery_architecture.md`'s own "Known limitations"
section for the complete, current list (duplicate-derivation detection
is a documented heuristic that does not inspect `deterministic_params`;
`MARKET_DATA`/`RAW_SOURCE` node classification is a `feature_group`
heuristic; no CLI surface; not yet wired into a persistence store; no
statistic anywhere depends on a prediction target or label) — not
duplicated here to avoid drift between two copies of the same
information.

## 16. Exact git status and explicit confirmations

`HEAD` at the time of writing this report:
`33b4f2218e09fafdf2f95927f193aa1dbdfe70a6` (unchanged from Section 1's
baseline — no commit was made at any point in Phase 2, Parts 1 or 2).

`git status --short` at the time of writing this report shows exactly:

```
 M src/quant_platform/core/exceptions.py
 M tests/unit/qualification/conftest.py
 M tests/unit/qualification/test_qualification_adversarial.py
 M tests/unit/qualification/test_qualification_invariance.py
 M tests/unit/qualification/test_qualification_repetition_gates.py
?? docs/feature_discovery_architecture.md
?? docs/milestone11_phase2_part2_delivery_report.md
?? src/quant_platform/feature_discovery/
?? tests/unit/feature_discovery/
```

No other file is modified, added, or deleted anywhere in the working
tree.

**Explicit confirmations:**

- Phase 2 work (Parts 1 and 2) is **not staged** — `git diff --cached`
  is empty; nothing was ever `git add`ed.
- Phase 2 work is **not committed** — `HEAD` is unchanged from the
  Section 1 baseline.
- **Nothing was pushed** at any point.
- **No second `FeatureEngine`/`FeatureRegistry`/`ResearchDatasetBuilder`/
  `DatasetQualificationEngine`** was created — every feature-computation
  and dataset-building code path this package touches terminates in a
  call to the real, pre-existing, unmodified classes; `feature_discovery`
  only ever READS an already-built manifest, its durable artifacts, and
  the caller's own live `FeatureRegistry`.
- **No statistic requiring a prediction target was computed anywhere**
  — no Information Coefficient, Rank IC, Mutual Information, Feature
  Importance, Permutation Importance, SHAP, Boruta, Recursive Feature
  Elimination, or correlation-to-labels. This package has no notion of
  a label or prediction target at all.
- **No model training, fitting, or feature selection** was performed
  anywhere in this package.
- **No Part 1 architecture was redesigned** — every Part 2 addition is
  a new module reusing Part 1's existing types/functions directly
  (`SharedDiscoveryFacts`, `compute_feature_signal_diagnostics`); every
  Part 1 test still passes unmodified.
- **Milestone 12 was not started** — no model training, deep learning,
  live prediction, MT5, broker integration, or production scheduling
  work was performed or will begin without further explicit
  instruction.

## 17. Explicit stop confirmation

Per the governing specification's own final instruction: **this phase
stops here.** This delivery report, together with `docs/
feature_discovery_architecture.md`, is the complete Milestone 11 Phase
2 Part 2 deliverable awaiting review and explicit commit approval. The
full repository suite's one failure (Section 14) is a pre-existing,
unrelated, load-sensitive flaky concurrency test in `portfolio_risk` —
confirmed by isolation re-run and by the file's git history predating
this task — and is flagged for the reviewer's attention rather than
silently worked around or fixed outside this task's scope.
