"""Milestone 11, Phase 2, Part 1: deterministic Feature Discovery &
Signal Diagnostics.

`quant_platform.feature_discovery` scientifically evaluates every
feature in an ALREADY-BUILT research dataset
(`features.manifests.ResearchDatasetManifest`, produced by the real,
unmodified `features.dataset_builder.ResearchDatasetBuilder`, and
typically already `APPROVED_FOR_RESEARCH` by `quant_platform.
qualification`) across 10 dimensions -- Information Content, Temporal
Stability, Regime Stability, Drift Behaviour, Redundancy, Coverage,
Availability, Leakage Safety, Determinism, Reproducibility.

**The output is NOT a model. The output is scientific evidence.** This
package never trains a model, never computes SHAP or permutation
importance, never runs Boruta or recursive feature elimination, never
performs feature selection, never fits anything, and never builds a
second `FeatureEngine`/`FeatureRegistry`/`ResearchDatasetBuilder`/
`DatasetQualificationEngine` -- it reads an already-built manifest and
its durable artifacts (via the existing `features.manifests.
ResearchDatasetStore`) only, and reuses `qualification.verifier`'s pure
identity/artifact-integrity/leakage helper functions and `features.
drift`/`features.validation`'s pure statistical functions directly
rather than reimplementing any of them.

**Never classifies a feature as predictive. Never estimates alpha.
Never recommends buying or selling. Never infers model quality.** This
package measures only scientific feature quality -- whether a feature
is informative, stable, non-redundant, drift-free, leak-free,
deterministic, and reproducible; never whether it would make a good
trading signal.

8 named deliverables: `FeatureDiscoveryEngine`, `FeatureStatistics`,
`FeatureSignalDiagnostics`, `FeatureDiscoveryReport`,
`FeatureDiscoveryVerifier`, `FeatureDiscoveryReconciliation`,
`FeatureDiscoveryEvidence`, `FeatureDiscoveryReports`.
"""

from __future__ import annotations
