"""Milestone 11, Phase 3, Part A: deterministic Label Infrastructure.

`quant_platform.labels` establishes the complete, versioned, content-
addressed, replayable, auditable, point-in-time-safe framework every
future label family will be generated through.

**A label is not a model. A label is not a prediction. A label is
immutable scientific evidence derived from historical observations.**
This package never generates predictive models, never evaluates model
quality, never performs feature selection, and never computes a
statistic that requires a prediction target (Information Coefficient,
Rank IC, Mutual Information, SHAP, Permutation Importance, Boruta,
Recursive Feature Elimination, or any correlation to a label). Those
belong to a later phase, once a model exists to evaluate.

**Part A ships infrastructure only.** It supports 6 named label
families (Next Return, Multi Horizon Return, Direction, Triple Barrier,
Forward Volatility, Future Extension Placeholder) as first-class
identity/versioning/reporting citizens, but implements the actual
generation LOGIC for none of them -- every generated label's values come
from a caller-supplied, pluggable `builder.LabelGeneratorFn`; this phase
ships zero concrete implementations of one. Those belong to Part 2.

Deliberately standalone: this package imports nothing from `features`,
`qualification`, `feature_discovery`, `ml`'s domain packages, or
`paper_trading`/`execution_gateway`/`portfolio_risk` -- only `core`
exceptions/types, `historical.quality.Severity` (an established, shared
severity vocabulary), `ml.persistence`'s dependency-neutral JSON
helpers, and `pandas`/`numpy`. This is what makes the preferred
dependency graph (Market Data -> Features -> Qualification -> Feature
Discovery -> Labels -> Machine Learning) a workflow ordering rather than
a Python import requirement, and a circular dependency structurally
impossible.

15 named deliverables: `LabelRegistry`, `LabelSpecification`,
`LabelDefinition`, `LabelManifest`, `LabelIdentity`, `LabelVersion`,
`LabelBuilder`, `LabelBundle`, `LabelVerification` (`LabelVerifier`),
`LabelReplay`, `LabelRecovery`, `LabelReconciliation`, `LabelReports`
(`reports.py`), `LabelDiagnostics`, `LabelEvidence`.
"""

from __future__ import annotations
