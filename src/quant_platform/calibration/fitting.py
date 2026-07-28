"""Inner out-of-fold prediction generation and training-side calibrator/
threshold/decision-policy fitting (Milestone 4E, Sections 6/8/13).

THIS MODULE NEVER SEES OUTER-TEST DATA -- STRUCTURALLY, NOT BY CONVENTION
--------------------------------------------------------------------------
Every function here accepts an `execution.splitters.Fold` (for its
`train_indices` ONLY -- see `generate_inner_oof_predictions`, which reads
`outer_fold.train_indices` exclusively, exactly mirroring `optimization.
inner_splits.build_inner_fold_plan`'s own identical, already-audited
isolation argument) or an already-built `RawPredictionSet`/
`InnerOofPredictionSet` -- there is no parameter anywhere in this module
through which `outer_fold.test_indices` (or its data) could even be
passed. `calibration.runner` is the ONLY module in this package that
ever reads `outer_fold.test_indices`, exactly like `optimization.
outer_fold.finalize_outer_fold` is the only function in `optimization`
that does.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.calibration.diagnostics import ReliabilityReport, compute_reliability_bins
from quant_platform.calibration.methods import (
    FittedMethodUnion,
    build_unfit_method,
    method_complexity_rank,
)
from quant_platform.calibration.metrics import compute_calibration_metrics
from quant_platform.calibration.models import (
    CalibrationMethodKind,
    CalibrationTieBreakPolicy,
    FailedCandidateReason,
    RawPredictionSet,
    SelectionMetric,
)
from quant_platform.calibration.specs import CalibrationSpec, calibration_inner_fold_seed
from quant_platform.calibration.thresholds import (
    ThresholdReport,
    ThresholdStabilityReport,
    compute_threshold_stability,
    evaluate_threshold_candidates,
)
from quant_platform.core.exceptions import (
    CalibrationDataError,
    CalibrationFitError,
    CalibrationSelectionError,
)
from quant_platform.execution.splitters import Fold
from quant_platform.ml.interfaces import FeatureSchema, ModelFactory, ProbabilisticPredictor
from quant_platform.ml.models import ModelHyperparameters, ObjectiveType
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.optimization.inner_splits import (
    InnerFoldPlan,
    build_inner_fold_plan,
    validate_nested_plan,
)

INNER_OOF_PREDICTION_SET_SCHEMA_VERSION = 1
CALIBRATOR_SELECTION_REPORT_SCHEMA_VERSION = 1
_MINIMUM_METRIC_BINS = 10
"""A conservative, fixed bin count used ONLY for the metrics computed
DURING calibrator selection (never the caller-declared `reliability_
binning_specs`, which govern the PERSISTED diagnostic report) -- keeps
candidate comparison stable even when the spec requests a very fine
binning for final reporting."""


def _as_float_label(value: object) -> float:
    """`ProbabilisticPredictor.class_labels` is typed `tuple[object, ...]`
    (see `ml.interfaces`) because a model layer conceivably could use
    non-numeric class identities; THIS platform's fixed 0.0/1.0
    positive-class convention (`calibration.models._POSITIVE_CLASS_
    LABEL`) requires numeric labels, so an unexpected type is a genuine
    contract violation to surface, never a value to silently coerce via
    `str()`-round-tripping."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationFitError(f"generate_inner_oof_predictions: model class label {value!r} is not a numeric (non-bool) type")
    return float(value)


def _concat_optional_floats(sequences: Sequence[tuple[float, ...] | None]) -> tuple[float, ...] | None:
    """`None` if ANY input sequence is `None` (all-or-nothing, matching
    `RawPredictionSet`'s own "a field is either populated for every row
    or entirely absent" convention); otherwise the flattened
    concatenation in the given order."""
    if any(s is None for s in sequences):
        return None
    flattened: list[float] = []
    for s in sequences:
        assert s is not None
        flattened.extend(s)
    return tuple(flattened)


# --------------------------------------------------------------------------
# Inner out-of-fold prediction generation (Section 6)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class InnerOofPredictionSet:
    """One outer fold's complete inner-fold OOF prediction record --
    per-inner-fold granularity is preserved (never flattened away) so
    `calibration.verification` can independently re-check, for EVERY
    inner fold separately, that its predictions came from a model that
    never trained on those exact rows (`RawPredictionSet.fitted_on_rows`
    disjointness, already enforced by `RawPredictionSet.__post_init__`
    itself, at construction time, not just at verification time)."""

    schema_version: int
    outer_fold_index: int
    inner_fold_plan: InnerFoldPlan
    per_inner_fold: tuple[RawPredictionSet, ...]

    def __post_init__(self) -> None:
        if len(self.per_inner_fold) != len(self.inner_fold_plan.inner_folds):
            raise CalibrationDataError(
                f"InnerOofPredictionSet: per_inner_fold has {len(self.per_inner_fold)} entries, expected "
                f"{len(self.inner_fold_plan.inner_folds)} (one per InnerFoldPlan.inner_folds entry)"
            )
        for i, (predictions, inner_fold) in enumerate(zip(self.per_inner_fold, self.inner_fold_plan.inner_folds, strict=True)):
            if predictions.inner_fold_index != inner_fold.inner_fold_index:
                raise CalibrationDataError(
                    f"InnerOofPredictionSet.per_inner_fold[{i}].inner_fold_index ({predictions.inner_fold_index}) "
                    f"does not match inner_fold_plan.inner_folds[{i}].inner_fold_index ({inner_fold.inner_fold_index})"
                )
            if predictions.outer_fold_index != self.outer_fold_index:
                raise CalibrationDataError(
                    f"InnerOofPredictionSet.per_inner_fold[{i}].outer_fold_index ({predictions.outer_fold_index}) "
                    f"does not match outer_fold_index ({self.outer_fold_index})"
                )
            if predictions.fitted_on_rows is None:
                raise CalibrationDataError(
                    f"InnerOofPredictionSet.per_inner_fold[{i}]: fitted_on_rows must be recorded for every inner "
                    "OOF prediction set (provenance is mandatory here, unlike the final outer-test set)"
                )
            expected_positions = tuple(int(v) for v in inner_fold.validation_indices.tolist())
            if predictions.sample_positions != expected_positions:
                raise CalibrationDataError(
                    f"InnerOofPredictionSet.per_inner_fold[{i}].sample_positions does not match "
                    f"inner_fold_plan.inner_folds[{i}].validation_indices"
                )

    def concatenated(self) -> RawPredictionSet:
        """Merges every inner fold's predictions into ONE chronologically
        -ordered `RawPredictionSet` for calibrator/threshold fitting.
        `fitted_on_rows=None` at this merged level (mixed provenance
        across inner folds -- the per-fold disjointness is what matters
        and is already independently guaranteed for each constituent
        `RawPredictionSet`, not re-asserted here as one set)."""
        ordered = sorted(self.per_inner_fold, key=lambda p: p.sample_positions[0])
        first = ordered[0]
        return RawPredictionSet(
            schema_version=first.schema_version, outer_fold_index=self.outer_fold_index, inner_fold_index=None,
            sample_positions=tuple(pos for p in ordered for pos in p.sample_positions),
            timestamps=tuple(ts for p in ordered for ts in p.timestamps),
            raw_scores=_concat_optional_floats([p.raw_scores for p in ordered]),
            raw_probabilities=_concat_optional_floats([p.raw_probabilities for p in ordered]),
            class_labels=first.class_labels, positive_class_index=first.positive_class_index,
            source_model_identity=first.source_model_identity, source_experiment_id=first.source_experiment_id,
            true_labels=_concat_optional_floats([p.true_labels for p in ordered]),
            fitted_on_rows=None,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "outer_fold_index": self.outer_fold_index,
            "inner_fold_plan": self.inner_fold_plan.to_json_dict(), "per_inner_fold": [p.to_json_dict() for p in self.per_inner_fold],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> InnerOofPredictionSet:
        require_schema_version(raw, supported=INNER_OOF_PREDICTION_SET_SCHEMA_VERSION, context="InnerOofPredictionSet")
        return cls(
            schema_version=INNER_OOF_PREDICTION_SET_SCHEMA_VERSION, outer_fold_index=int(str(raw["outer_fold_index"])),
            inner_fold_plan=InnerFoldPlan.from_json_dict(as_json_dict(raw["inner_fold_plan"], field_name="inner_fold_plan")),
            per_inner_fold=tuple(
                RawPredictionSet.from_json_dict(as_json_dict(p, field_name="per_inner_fold[]"))
                for p in as_json_list(raw["per_inner_fold"], field_name="per_inner_fold")
            ),
        )


def generate_inner_oof_predictions(
    *, outer_fold: Fold, timeline: pd.DataFrame, feature_names: Sequence[str], label_column: str,
    label_horizon_bars: int, model_factory: ModelFactory, hyperparameters: ModelHyperparameters, objective: ObjectiveType,
    seed_configuration: SeedConfiguration, spec: CalibrationSpec, source_model_identity: str, source_experiment_id: str,
) -> InnerOofPredictionSet:
    """Section 6: for each inner fold, train a FRESH base model on that
    inner fold's `train_indices` ONLY, predict ONLY on its
    `validation_indices`. `outer_fold.test_indices`/`.validation_indices`
    are never read here -- `build_inner_fold_plan` itself has no code
    path that touches them (see that function's own docstring).

    `label_horizon_bars` must be resolved by the caller (`calibration.
    runner`) from the source-bound `ExperimentSpec.label_binding.
    horizon_bars` -- this module deliberately has no artifact-storage
    access of its own, so it cannot resolve that identity itself, and
    does not guess a default."""
    inner_plan = build_inner_fold_plan(
        outer_fold, config=spec.inner_oof_policy, label_horizon_bars=label_horizon_bars, timeline=timeline,
    )
    report = validate_nested_plan(outer_fold, inner_plan)
    if not report.is_ready:
        raise CalibrationDataError(
            f"generate_inner_oof_predictions: inner fold plan for outer fold {outer_fold.fold_index} failed "
            f"leakage validation: {[i.message for i in report.criticals]}"
        )

    feature_schema = FeatureSchema(feature_names=tuple(feature_names))
    per_inner_fold: list[RawPredictionSet] = []
    for inner_fold in inner_plan.inner_folds:
        train_df = timeline.iloc[inner_fold.train_indices]
        validation_df = timeline.iloc[inner_fold.validation_indices]
        seed = calibration_inner_fold_seed(seed_configuration, outer_fold_index=outer_fold.fold_index, inner_fold_index=inner_fold.inner_fold_index)
        model = model_factory.create(hyperparameters=hyperparameters, feature_schema=feature_schema, objective=objective)
        fitted = model.fit(train_df[list(feature_names)], train_df[label_column], seeds=SeedConfiguration(master_seed=seed))

        validation_features = validation_df[list(feature_names)]
        raw_probabilities = None
        class_labels: tuple[float, ...] = (0.0, 1.0)
        positive_index = 1
        if fitted.metadata.capabilities.supports_predict_proba and isinstance(fitted, ProbabilisticPredictor):
            proba = fitted.predict_proba(validation_features)
            labels_out = list(fitted.class_labels)
            positive_index = labels_out.index(1)
            class_labels = tuple(_as_float_label(c) for c in labels_out)
            raw_probabilities = tuple(float(v) for v in proba[:, positive_index])
        else:
            raise CalibrationFitError(
                f"generate_inner_oof_predictions: model {source_model_identity!r} does not support predict_proba "
                "-- the calibration framework requires a probabilistic base model"
            )

        timestamps = tuple(pd.Timestamp(t).isoformat() for t in validation_df["open_time"])
        per_inner_fold.append(RawPredictionSet(
            schema_version=1, outer_fold_index=outer_fold.fold_index, inner_fold_index=inner_fold.inner_fold_index,
            sample_positions=tuple(int(v) for v in inner_fold.validation_indices.tolist()), timestamps=timestamps,
            raw_scores=None, raw_probabilities=raw_probabilities, class_labels=class_labels, positive_class_index=positive_index,
            source_model_identity=source_model_identity, source_experiment_id=source_experiment_id,
            true_labels=tuple(float(v) for v in validation_df[label_column]),
            fitted_on_rows=tuple(int(v) for v in inner_fold.train_indices.tolist()),
        ))

    return InnerOofPredictionSet(
        schema_version=INNER_OOF_PREDICTION_SET_SCHEMA_VERSION, outer_fold_index=outer_fold.fold_index,
        inner_fold_plan=inner_plan, per_inner_fold=tuple(per_inner_fold),
    )


# --------------------------------------------------------------------------
# Calibrator selection (Section 8)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CalibratorCandidateResult:
    kind: str
    succeeded: bool
    fitted: FittedMethodUnion | None
    metrics: dict[str, float]
    selection_metric_value: float | None
    failure: FailedCandidateReason | None

    def __post_init__(self) -> None:
        for name, value in self.metrics.items():
            if not math.isfinite(value):
                raise CalibrationDataError(f"CalibratorCandidateResult.metrics[{name!r}] must be finite, got {value!r}")
        if self.selection_metric_value is not None and not math.isfinite(self.selection_metric_value):
            raise CalibrationDataError(f"CalibratorCandidateResult.selection_metric_value must be finite if set, got {self.selection_metric_value!r}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind, "succeeded": self.succeeded, "fitted": (None if self.fitted is None else self.fitted.to_json_dict()),
            "metrics": dict(sorted(self.metrics.items())), "selection_metric_value": self.selection_metric_value,
            "failure": (None if self.failure is None else self.failure.to_json_dict()),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CalibratorCandidateResult:
        from quant_platform.calibration.methods import fitted_method_from_json_dict

        fitted_raw = raw.get("fitted")
        failure_raw = raw.get("failure")
        return cls(
            kind=str(raw["kind"]), succeeded=bool(raw["succeeded"]),
            fitted=(None if fitted_raw is None else fitted_method_from_json_dict(as_json_dict(fitted_raw, field_name="fitted"))),
            metrics={str(k): float(v) for k, v in as_json_dict(raw.get("metrics") or {}, field_name="metrics").items()},
            selection_metric_value=(None if raw.get("selection_metric_value") is None else float(str(raw["selection_metric_value"]))),
            failure=(None if failure_raw is None else FailedCandidateReason.from_json_dict(as_json_dict(failure_raw, field_name="failure"))),
        )


@dataclass(frozen=True, slots=True)
class CalibratorSelectionReport:
    schema_version: int
    outer_fold_index: int
    selection_metric: SelectionMetric
    tie_break_policy: CalibrationTieBreakPolicy
    candidates: tuple[CalibratorCandidateResult, ...]
    selected_kind: str
    selected_reason: str
    tie_break_note: str | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "outer_fold_index": self.outer_fold_index,
            "selection_metric": self.selection_metric.value, "tie_break_policy": self.tie_break_policy.value,
            "candidates": [c.to_json_dict() for c in self.candidates], "selected_kind": self.selected_kind,
            "selected_reason": self.selected_reason, "tie_break_note": self.tie_break_note,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CalibratorSelectionReport:
        require_schema_version(raw, supported=CALIBRATOR_SELECTION_REPORT_SCHEMA_VERSION, context="CalibratorSelectionReport")
        return cls(
            schema_version=CALIBRATOR_SELECTION_REPORT_SCHEMA_VERSION, outer_fold_index=int(str(raw["outer_fold_index"])),
            selection_metric=SelectionMetric(raw["selection_metric"]), tie_break_policy=CalibrationTieBreakPolicy(raw["tie_break_policy"]),
            candidates=tuple(
                CalibratorCandidateResult.from_json_dict(c) for c in as_json_list(raw["candidates"], field_name="candidates")
            ),
            selected_kind=str(raw["selected_kind"]), selected_reason=str(raw["selected_reason"]),
            tie_break_note=(None if raw.get("tie_break_note") is None else str(raw["tie_break_note"])),
        )

    def selected_candidate(self) -> CalibratorCandidateResult:
        for c in self.candidates:
            if c.kind == self.selected_kind:
                return c
        raise CalibrationSelectionError(f"CalibratorSelectionReport: selected_kind {self.selected_kind!r} not found among candidates")  # pragma: no cover - defensive


_LOWER_IS_BETTER_METRICS = frozenset({SelectionMetric.LOG_LOSS, SelectionMetric.BRIER_SCORE, SelectionMetric.EXPECTED_CALIBRATION_ERROR, SelectionMetric.MAXIMUM_CALIBRATION_ERROR})
_SECONDARY_METRIC_FOR: dict[SelectionMetric, str] = {
    SelectionMetric.LOG_LOSS: "brier_score", SelectionMetric.BRIER_SCORE: "log_loss",
    SelectionMetric.EXPECTED_CALIBRATION_ERROR: "log_loss", SelectionMetric.MAXIMUM_CALIBRATION_ERROR: "log_loss",
}


def select_calibrator(oof: RawPredictionSet, *, spec: CalibrationSpec) -> CalibratorSelectionReport:
    """Section 8: fits every declared candidate method on the POOLED
    inner-OOF `(probabilities, labels)`, computes calibration metrics for
    each, and selects one via the fixed, deterministic tie-break chain
    (Section 8: primary metric -> simpler method -> secondary metric ->
    lexical identifier). Never touches outer-test data (`oof` is
    constructed exclusively from `InnerOofPredictionSet.concatenated()`
    by the caller)."""
    if oof.raw_probabilities is None:
        raise CalibrationDataError("select_calibrator requires oof.raw_probabilities to be populated")
    if oof.true_labels is None:
        raise CalibrationDataError("select_calibrator requires oof.true_labels to be populated (training-side labels)")
    n = oof.n_samples
    n_positive = sum(1 for lab in oof.true_labels if lab == 1.0)
    n_negative = n - n_positive
    binning_spec = spec.reliability_binning_specs[0]

    candidates: list[CalibratorCandidateResult] = []
    for kind in spec.calibration_method_candidates:
        if n < spec.minimum_calibration_sample_count:
            candidates.append(CalibratorCandidateResult(
                kind.value, False, None, {}, None,
                FailedCandidateReason(kind.value, f"only {n} calibration sample(s) available, below minimum_calibration_sample_count={spec.minimum_calibration_sample_count}"),
            ))
            continue
        if min(n_positive, n_negative) < spec.minimum_samples_per_class:
            candidates.append(CalibratorCandidateResult(
                kind.value, False, None, {}, None,
                FailedCandidateReason(kind.value, f"minority class has {min(n_positive, n_negative)} sample(s), below minimum_samples_per_class={spec.minimum_samples_per_class}"),
            ))
            continue
        try:
            method = build_unfit_method(kind)
            fitted = method.fit(np.asarray(oof.raw_probabilities), np.asarray(oof.true_labels))
            transformed = fitted.transform(np.asarray(oof.raw_probabilities))
            if not np.all(np.isfinite(transformed)) or np.any((transformed < 0.0) | (transformed > 1.0)):
                raise CalibrationFitError(f"{kind.value} candidate emitted invalid probabilities (non-finite or outside [0, 1])")
            metric_report = compute_calibration_metrics(transformed, np.asarray(oof.true_labels), binning_spec=binning_spec)
        except (CalibrationFitError, CalibrationDataError) as exc:
            candidates.append(CalibratorCandidateResult(kind.value, False, None, {}, None, FailedCandidateReason(kind.value, str(exc))))
            continue
        selection_value = metric_report.values.get(spec.calibration_selection_metric.value)
        candidates.append(CalibratorCandidateResult(kind.value, True, fitted, dict(metric_report.values), selection_value, None))

    succeeded = [c for c in candidates if c.succeeded and c.selection_metric_value is not None]
    if not succeeded:
        raise CalibrationSelectionError(
            f"select_calibrator: no candidate (including IDENTITY) produced a usable "
            f"{spec.calibration_selection_metric.value!r} value for outer fold {oof.outer_fold_index}"
        )

    direction = -1.0 if spec.calibration_selection_metric in _LOWER_IS_BETTER_METRICS else 1.0
    secondary_name = _SECONDARY_METRIC_FOR[spec.calibration_selection_metric]

    def tie_break_key(c: CalibratorCandidateResult) -> tuple[float, int, float, str]:
        assert c.selection_metric_value is not None
        primary = -direction * c.selection_metric_value
        complexity = method_complexity_rank(_kind_from_value(c.kind))
        secondary = c.metrics.get(secondary_name, 0.0)
        return (primary, complexity, secondary, c.kind)

    best = min(succeeded, key=tie_break_key)
    tie_count = sum(1 for c in succeeded if abs(tie_break_key(c)[0] - tie_break_key(best)[0]) < 1e-12)
    note = f"{tie_count} candidate(s) tied on {spec.calibration_selection_metric.value!r}; resolved via simpler-method preference, then {secondary_name!r}, then lexical identifier" if tie_count > 1 else None

    return CalibratorSelectionReport(
        schema_version=CALIBRATOR_SELECTION_REPORT_SCHEMA_VERSION, outer_fold_index=oof.outer_fold_index,
        selection_metric=spec.calibration_selection_metric, tie_break_policy=spec.calibration_tie_break_policy,
        candidates=tuple(candidates), selected_kind=best.kind,
        selected_reason=f"best {spec.calibration_selection_metric.value}={best.selection_metric_value:.6g} among {len(succeeded)} successful candidate(s)",
        tie_break_note=note,
    )


def _kind_from_value(value: str) -> CalibrationMethodKind:
    return CalibrationMethodKind(value)


# --------------------------------------------------------------------------
# Threshold selection + stability (Sections 12/13)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FrozenDecisionPolicy:
    """Everything frozen BEFORE the final base model is ever refit on
    complete outer-train data (Section 18, steps 3-7): the selected
    calibrator, the operational (pooled-OOF) threshold, per-inner-fold
    threshold stability, and the reliability report(s) requested by the
    spec -- all computed from inner OOF data alone."""

    schema_version: int
    outer_fold_index: int
    calibrator_selection: CalibratorSelectionReport
    threshold_report: ThresholdReport
    threshold_stability: ThresholdStabilityReport
    reliability_reports: tuple[ReliabilityReport, ...]

    def selected_calibrator(self) -> FittedMethodUnion:
        fitted = self.calibrator_selection.selected_candidate().fitted
        if fitted is None:
            raise CalibrationSelectionError("FrozenDecisionPolicy: selected candidate has no fitted method")  # pragma: no cover - defensive
        return fitted

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "outer_fold_index": self.outer_fold_index,
            "calibrator_selection": self.calibrator_selection.to_json_dict(), "threshold_report": self.threshold_report.to_json_dict(),
            "threshold_stability": self.threshold_stability.to_json_dict(),
            "reliability_reports": [r.to_json_dict() for r in self.reliability_reports],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FrozenDecisionPolicy:
        require_schema_version(raw, supported=1, context="FrozenDecisionPolicy")
        return cls(
            schema_version=1, outer_fold_index=int(str(raw["outer_fold_index"])),
            calibrator_selection=CalibratorSelectionReport.from_json_dict(as_json_dict(raw["calibrator_selection"], field_name="calibrator_selection")),
            threshold_report=ThresholdReport.from_json_dict(as_json_dict(raw["threshold_report"], field_name="threshold_report")),
            threshold_stability=ThresholdStabilityReport.from_json_dict(as_json_dict(raw["threshold_stability"], field_name="threshold_stability")),
            reliability_reports=tuple(
                ReliabilityReport.from_json_dict(r) for r in as_json_list(raw["reliability_reports"], field_name="reliability_reports")
            ),
        )


def fit_decision_policy(oof_predictions: InnerOofPredictionSet, *, spec: CalibrationSpec) -> FrozenDecisionPolicy:
    """Orchestrates Section 18 steps 3-6: calibrator selection, threshold
    selection (pooled), and per-inner-fold threshold stability -- all
    from `oof_predictions` (inner OOF, training-side) alone."""
    pooled = oof_predictions.concatenated()
    selection = select_calibrator(pooled, spec=spec)
    selected = selection.selected_candidate()
    assert selected.fitted is not None

    assert pooled.raw_probabilities is not None and pooled.true_labels is not None
    pooled_calibrated = selected.fitted.transform(np.asarray(pooled.raw_probabilities))
    pooled_labels = np.asarray(pooled.true_labels)

    threshold_report = evaluate_threshold_candidates(pooled_calibrated, pooled_labels, spec=spec.threshold_spec)

    per_fold_threshold_reports: list[ThresholdReport] = []
    for inner in oof_predictions.per_inner_fold:
        assert inner.raw_probabilities is not None and inner.true_labels is not None
        fold_calibrated = selected.fitted.transform(np.asarray(inner.raw_probabilities))
        fold_labels = np.asarray(inner.true_labels)
        if len(set(fold_labels.tolist())) < 2:
            continue
        per_fold_threshold_reports.append(evaluate_threshold_candidates(fold_calibrated, fold_labels, spec=spec.threshold_spec))
    if not per_fold_threshold_reports:
        per_fold_threshold_reports = [threshold_report]
    stability = compute_threshold_stability(per_fold_threshold_reports)

    reliability_reports = tuple(
        compute_reliability_bins(pooled_calibrated, pooled_labels, spec=binning_spec) for binning_spec in spec.reliability_binning_specs
    )

    return FrozenDecisionPolicy(
        schema_version=1, outer_fold_index=oof_predictions.outer_fold_index, calibrator_selection=selection,
        threshold_report=threshold_report, threshold_stability=stability, reliability_reports=reliability_reports,
    )


_ = time  # imported for future duration instrumentation call sites in calibration.runner; referenced here to keep the import intentional


__all__ = [
    "CALIBRATOR_SELECTION_REPORT_SCHEMA_VERSION",
    "INNER_OOF_PREDICTION_SET_SCHEMA_VERSION",
    "CalibratorCandidateResult",
    "CalibratorSelectionReport",
    "FrozenDecisionPolicy",
    "InnerOofPredictionSet",
    "fit_decision_policy",
    "generate_inner_oof_predictions",
    "select_calibrator",
]
