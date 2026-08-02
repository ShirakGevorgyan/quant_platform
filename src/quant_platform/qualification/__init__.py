"""Milestone 11, Phase 1: the deterministic Dataset Qualification Engine.

`quant_platform.qualification` decides whether an ALREADY-BUILT research
dataset (`features.manifests.ResearchDatasetManifest`, produced by the
real, unmodified `features.dataset_builder.ResearchDatasetBuilder`) is
scientifically suitable for ML across 8 dimensions (Structural Integrity,
Temporal Integrity, Statistical Integrity, Coverage, Stability,
Determinism, Reproducibility, Safety), producing exactly one of two
decisions: `APPROVED_FOR_RESEARCH` or `REJECTED_FOR_RESEARCH`.

This package never trains a model, never computes feature importance,
never performs feature selection, and never builds a second
`FeatureEngine`/`FeatureRegistry`/`ResearchDatasetBuilder` -- it reads an
already-built manifest and its durable artifacts (via the existing
`features.manifests.ResearchDatasetStore`) only.

Part 1 covered the 7 named deliverables: `DatasetQualificationEngine`,
`DatasetQualificationReport`, `QualificationDecision`,
`QualificationVerifier`, `QualificationDiagnostics`,
`QualificationReconciliation`, `QualificationReports`. Part 2 (this
delivery) added, purely additively -- no Part 1 architecture was
redesigned -- an `Evidence` model and 6 deep-diagnostic sections
(`diagnostics.py`), independent verification (`verification.py`),
truncation/replay/determinism proofs, a 26-item adversarial audit,
warning/recommendation/lineage-drift reconciliation, 4 more report
types, and the quality-gate/documentation ceremony. See
`docs/qualification_architecture.md` for the full reference and
`docs/milestone11_phase1_delivery_report.md` for the delivery record.
"""

from __future__ import annotations
