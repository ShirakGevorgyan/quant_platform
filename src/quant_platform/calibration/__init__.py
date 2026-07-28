"""Milestone 4E: leakage-safe prediction calibration, decision
thresholding, confidence, and uncertainty framework.

Depends on `quant_platform.ml`, `quant_platform.execution`,
`quant_platform.features`, and `quant_platform.optimization`. None of
those packages import anything from this one -- the dependency is
one-way, and this package sits ONE LAYER ABOVE `optimization` (it
post-processes an already-selected, already-refit base model's raw
outputs; it never selects features, hyperparameters, or models itself).

THE CENTRAL RULE
--------------------------------------------------------------------------
    OUTER TRAIN
        |-- inner time-safe walk-forward splits (reusing
        |   optimization.inner_splits directly)
                |-- inner train  -> fresh base model per inner fold
                |-- inner validation -> inner out-of-fold (OOF) predictions

    Calibrator candidates, the decision threshold, confidence, and
    uncertainty policies are ALL fit/selected/frozen from inner OOF
    predictions alone.

    OUTER TEST
        never used during: calibrator fitting or selection, threshold
        optimization, abstention threshold selection, uncertainty
        boundary definition, confidence bucket definition, fallback
        decisions, model ranking, feature selection, early stopping, or
        reporting configuration.

After the post-processing policy is frozen using only inner OOF data:
(1) refit the base model on the complete outer-train partition; (2)
predict once on the untouched outer-test partition; (3) apply the
FROZEN calibrator/threshold/confidence/uncertainty/abstention policy;
(4) read outer-test labels for the first time, to compute final
evaluation metrics only. `calibration.runner.run_outer_fold_calibration`
is the ONLY function in this package that ever reads an outer fold's
test partition -- see that function's own module docstring for the
structural (not conventional) isolation argument.

MODULE MAP
--------------------------------------------------------------------------
`models` (raw prediction contract, calibration stage state machine,
shared enums) -> `specs` (`CalibrationSpec` identity, `ThresholdSpec`/
`ConfidenceSpec`/`UncertaintySpec`/`AbstentionSpec`, seed derivation) ->
`methods` (Identity/Platt/Isotonic/Beta calibrators, safe serialization,
no pickle) -> `metrics` (log loss, Brier, ECE, MCE, calibration slope/
intercept) -> `diagnostics` (reliability bins) -> `thresholds` (the one
authoritative `probability >= threshold` boundary, 8 threshold policies,
stability) -> `confidence`/`uncertainty` (transparent, documented
component proxies, never silently zero-filled) -> `abstention`
(selective prediction, coverage/risk always reported together) ->
`fitting` (inner OOF generation + calibrator/threshold selection --
the most leakage-critical module) -> `manifests` (`CalibrationManifest`
+ append-only event log, `CalibrationStage` state machine) -> `reporting`
(JSON/Markdown reports, standard unsupported-claims disclaimers) ->
`runner` (`CalibrationRunner`, the top-level orchestrator + outer-fold
execution) -> `resume` (verified-artifact resume planning) ->
`verification` (`verify_calibration`, including the recomputation check
that proves persisted calibrated probabilities/decisions are actually
reproducible from persisted parameters, not just hash-consistent).

EXPLICITLY OUT OF SCOPE (see the delivery report for the complete list):
backtesting, PnL simulation, transaction costs, position sizing,
portfolio construction, live trading, market data collection, new
feature engineering, ensembles, deep learning, online model updates.
"""

from __future__ import annotations

from quant_platform.calibration.abstention import (
    SELECTIVE_EVALUATION_SCHEMA_VERSION,
    SelectivePredictionEvaluation,
    decide,
    evaluate_selective_prediction,
)
from quant_platform.calibration.confidence import (
    CONFIDENCE_RESULT_SCHEMA_VERSION,
    ConfidenceResult,
    compute_confidence,
    distance_from_threshold_component,
    probability_extremity_component,
)
from quant_platform.calibration.diagnostics import (
    RELIABILITY_REPORT_SCHEMA_VERSION,
    ReliabilityBin,
    ReliabilityReport,
    compute_reliability_bins,
)
from quant_platform.calibration.fitting import (
    CALIBRATOR_SELECTION_REPORT_SCHEMA_VERSION,
    INNER_OOF_PREDICTION_SET_SCHEMA_VERSION,
    CalibratorCandidateResult,
    CalibratorSelectionReport,
    FrozenDecisionPolicy,
    InnerOofPredictionSet,
    fit_decision_policy,
    generate_inner_oof_predictions,
    select_calibrator,
)
from quant_platform.calibration.manifests import (
    CALIBRATION_MANIFEST_SCHEMA_VERSION,
    CalibrationEventRecord,
    CalibrationEventStore,
    CalibrationEventType,
    CalibrationManifest,
    CalibrationManifestStore,
)
from quant_platform.calibration.methods import (
    METHOD_SCHEMA_VERSION,
    BetaCalibrator,
    FittedBetaCalibrator,
    FittedCalibrationMethod,
    FittedIdentityCalibrator,
    FittedIsotonicCalibrator,
    FittedMethodUnion,
    FittedPlattCalibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    build_unfit_method,
    fitted_method_from_json_dict,
    method_complexity_rank,
)
from quant_platform.calibration.metrics import CALIBRATION_METRIC_NAMES, compute_calibration_metrics
from quant_platform.calibration.models import (
    RAW_PREDICTION_SET_SCHEMA_VERSION,
    TERMINAL_CALIBRATION_STAGES,
    AbstentionPolicyKind,
    AbstentionReasonCode,
    BinningStrategy,
    CalibrationMethodKind,
    CalibrationStage,
    CalibrationTieBreakPolicy,
    ConfidenceCategory,
    Decision,
    DeterminismPolicy,
    FailedCandidateReason,
    ModelIdentity,
    ProbabilityRepresentation,
    RawPredictionSet,
    ScoreProvenance,
    SelectionMetric,
    ThresholdPolicyKind,
    is_legal_calibration_transition,
    is_terminal_calibration_stage,
)
from quant_platform.calibration.reporting import (
    build_calibration_report_json,
    render_calibration_report_markdown,
)
from quant_platform.calibration.resume import (
    can_resume,
    require_calibration_resumable,
    resolve_resume_start_stage,
    verify_completed_calibration_outer_folds,
)
from quant_platform.calibration.runner import (
    OUTER_FOLD_CALIBRATION_RESULT_SCHEMA_VERSION,
    CalibrationOutcome,
    CalibrationRunner,
    OuterFoldCalibrationResult,
    ResolvedCalibrationInputs,
    resolve_calibration_inputs,
    run_outer_fold_calibration,
)
from quant_platform.calibration.specs import (
    CALIBRATION_SPEC_SCHEMA_VERSION,
    AbstentionSpec,
    CalibrationIdentity,
    CalibrationSpec,
    ConfidenceSpec,
    CostMatrix,
    ProbabilityClippingPolicy,
    ReliabilityBinningSpec,
    ThresholdSpec,
    UncertaintySpec,
    compute_calibration_identity,
)
from quant_platform.calibration.thresholds import (
    THRESHOLD_REPORT_SCHEMA_VERSION,
    THRESHOLD_STABILITY_SCHEMA_VERSION,
    ThresholdCandidateResult,
    ThresholdReport,
    ThresholdStabilityReport,
    apply_threshold,
    compute_threshold_stability,
    evaluate_threshold_candidates,
)
from quant_platform.calibration.uncertainty import (
    UNCERTAINTY_RESULT_SCHEMA_VERSION,
    UncertaintyResult,
    bin_support_uncertainty_component,
    calibrator_disagreement_component,
    compute_uncertainty,
    entropy_component,
    margin_component,
    model_disagreement_component,
)
from quant_platform.calibration.verification import verify_calibration

CALIBRATION_FRAMEWORK_VERSION = "1.0.0"
"""Version of this package's own calibration-lifecycle/identity/manifest
semantics -- recorded informationally, independent of `quant_platform.
__version__` and of earlier milestones' own version constants. Bump on
any change to those semantics."""

__all__ = [
    "CALIBRATION_FRAMEWORK_VERSION",
    "CALIBRATION_MANIFEST_SCHEMA_VERSION",
    "CALIBRATION_METRIC_NAMES",
    "CALIBRATION_SPEC_SCHEMA_VERSION",
    "CALIBRATOR_SELECTION_REPORT_SCHEMA_VERSION",
    "CONFIDENCE_RESULT_SCHEMA_VERSION",
    "INNER_OOF_PREDICTION_SET_SCHEMA_VERSION",
    "METHOD_SCHEMA_VERSION",
    "OUTER_FOLD_CALIBRATION_RESULT_SCHEMA_VERSION",
    "RAW_PREDICTION_SET_SCHEMA_VERSION",
    "RELIABILITY_REPORT_SCHEMA_VERSION",
    "SELECTIVE_EVALUATION_SCHEMA_VERSION",
    "TERMINAL_CALIBRATION_STAGES",
    "THRESHOLD_REPORT_SCHEMA_VERSION",
    "THRESHOLD_STABILITY_SCHEMA_VERSION",
    "UNCERTAINTY_RESULT_SCHEMA_VERSION",
    "AbstentionPolicyKind",
    "AbstentionReasonCode",
    "AbstentionSpec",
    "BetaCalibrator",
    "BinningStrategy",
    "CalibrationEventRecord",
    "CalibrationEventStore",
    "CalibrationEventType",
    "CalibrationIdentity",
    "CalibrationManifest",
    "CalibrationManifestStore",
    "CalibrationMethodKind",
    "CalibrationOutcome",
    "CalibrationRunner",
    "CalibrationSpec",
    "CalibrationStage",
    "CalibrationTieBreakPolicy",
    "CalibratorCandidateResult",
    "CalibratorSelectionReport",
    "ConfidenceCategory",
    "ConfidenceResult",
    "ConfidenceSpec",
    "CostMatrix",
    "Decision",
    "DeterminismPolicy",
    "FailedCandidateReason",
    "FittedBetaCalibrator",
    "FittedCalibrationMethod",
    "FittedIdentityCalibrator",
    "FittedIsotonicCalibrator",
    "FittedMethodUnion",
    "FittedPlattCalibrator",
    "FrozenDecisionPolicy",
    "IdentityCalibrator",
    "InnerOofPredictionSet",
    "IsotonicCalibrator",
    "ModelIdentity",
    "OuterFoldCalibrationResult",
    "PlattCalibrator",
    "ProbabilityClippingPolicy",
    "ProbabilityRepresentation",
    "RawPredictionSet",
    "ReliabilityBin",
    "ReliabilityBinningSpec",
    "ReliabilityReport",
    "ResolvedCalibrationInputs",
    "ScoreProvenance",
    "SelectionMetric",
    "SelectivePredictionEvaluation",
    "ThresholdCandidateResult",
    "ThresholdPolicyKind",
    "ThresholdReport",
    "ThresholdSpec",
    "ThresholdStabilityReport",
    "UncertaintyResult",
    "UncertaintySpec",
    "apply_threshold",
    "bin_support_uncertainty_component",
    "build_calibration_report_json",
    "build_unfit_method",
    "calibrator_disagreement_component",
    "can_resume",
    "compute_calibration_identity",
    "compute_calibration_metrics",
    "compute_confidence",
    "compute_reliability_bins",
    "compute_threshold_stability",
    "compute_uncertainty",
    "decide",
    "distance_from_threshold_component",
    "entropy_component",
    "evaluate_selective_prediction",
    "evaluate_threshold_candidates",
    "fit_decision_policy",
    "fitted_method_from_json_dict",
    "generate_inner_oof_predictions",
    "is_legal_calibration_transition",
    "is_terminal_calibration_stage",
    "margin_component",
    "method_complexity_rank",
    "model_disagreement_component",
    "probability_extremity_component",
    "render_calibration_report_markdown",
    "require_calibration_resumable",
    "resolve_calibration_inputs",
    "resolve_resume_start_stage",
    "run_outer_fold_calibration",
    "select_calibrator",
    "verify_calibration",
    "verify_completed_calibration_outer_folds",
]
