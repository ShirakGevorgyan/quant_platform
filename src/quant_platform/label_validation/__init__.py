"""Milestone 11, Phase 3, Part C: Label Diagnostics, Qualification &
Validation.

`quant_platform.label_validation` establishes whether GENERATED labels
(Milestone 11 Phase 3 Parts A/B) are scientifically suitable for ML
research. **The output is NOT labels. The output is LABEL QUALITY.**

This package never creates new label families, never trains a model,
and never computes feature importance, Information Coefficient, Mutual
Information, SHAP, or any statistic requiring a prediction target. It
reads an already-generated `labels.builder.LabelBundle` and its
`labels.manifest.LabelManifest` ONLY -- it never generates a label
itself.

A new, independent package: never merged into `labels/`,
`qualification/`, or `feature_discovery/`. Deliberately standalone
beyond its one necessary upstream dependency (`labels/`, its direct
predecessor in the pipeline: Market Data -> Features -> Qualification
-> Feature Discovery -> Labels -> Label Validation -> Machine
Learning) -- it imports nothing from `features`, `market_data`,
`qualification`, or `feature_discovery`, for the identical dependency-
isolation reason `labels/` itself never imports them.

13 named primary modules: `LabelQualificationEngine`, `LabelDiagnostics`,
`LabelStatistics`, `LabelDistribution`, `LabelBalance`, `LabelDegeneracy`,
`LabelDrift`, `LabelStability`, `LabelCoverage`, `LabelVerification`
(`LabelValidationVerifier`), `LabelReconciliation`
(`LabelValidationReconciliation`), `LabelEvidence`, `LabelReports`
(`reports.py`).
"""

from __future__ import annotations
