"""Adversarial leakage tests (Milestone 4E, Section 28) -- fail-LOUD
sentinels that prove outer-test isolation structurally, not just by
comparing a final metric before/after a change.

THE PRECISE BOUNDARY THESE TESTS TARGET
--------------------------------------------------------------------------
An earlier draft of this suite poisoned `Fold.test_indices`/
`.validation_indices` themselves (the raw integer POSITION arrays) to
raise on any access, expecting zero touches. That test failed against
correct code: `optimization.inner_splits.validate_nested_plan` (an
already-audited Milestone 4D leakage guard) legitimately reads those
position integers to verify no inner row overlaps them -- a defense-in-
depth SAFETY CHECK, not a leakage vector, since position integers reveal
nothing about the outer-test DATA. The real boundary is narrower and
more important: training-side code must never read the timeline ROWS
(feature/label VALUES) at those positions. `TestLandmineDataProvesOuter
TestRowsAreNeverRead` targets exactly that, leaving `test_indices`/
`.validation_indices` as ordinary arrays so the legitimate check still
runs.

Every test here either (a) replaces outer-test row VALUES with a
landmine object that raises on any numeric use and proves leakage-
critical functions complete without ever touching it, (b) mutates real
outer-test data and proves training-side artifacts are BYTE-IDENTICAL
before and after, or (c) constructs deliberately-leaky data and proves
the platform's own structural guards reject it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.calibration.fitting import (
    InnerOofPredictionSet,
    fit_decision_policy,
    generate_inner_oof_predictions,
    select_calibrator,
)
from quant_platform.calibration.models import (
    AbstentionPolicyKind,
    BinningStrategy,
    CalibrationMethodKind,
    CalibrationTieBreakPolicy,
    DeterminismPolicy,
    RawPredictionSet,
    SelectionMetric,
    ThresholdPolicyKind,
)
from quant_platform.calibration.specs import (
    AbstentionSpec,
    CalibrationSpec,
    ConfidenceSpec,
    ProbabilityClippingPolicy,
    ReliabilityBinningSpec,
    ThresholdSpec,
    UncertaintySpec,
)
from quant_platform.core.exceptions import CalibrationDataError
from quant_platform.execution.splitters import Fold
from quant_platform.ml.models import ModelHyperparameters, ObjectiveType
from quant_platform.ml.seeds import SeedConfiguration
from quant_platform.ml.testing import ConstantTestModelFactory
from quant_platform.optimization.inner_splits import InnerSplitConfig


def _make_timeline(n: int = 400, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    feature_a = rng.normal(size=n)
    labels = (feature_a + rng.normal(scale=0.5, size=n) > 0).astype(float)
    return pd.DataFrame({"open_time": timestamps, "feature_a": feature_a, "label": labels})


def _make_spec(*, seed: int = 42) -> CalibrationSpec:
    return CalibrationSpec(
        schema_version=1, task=ObjectiveType.BINARY_CLASSIFICATION, positive_class_label=1.0,
        source_experiment_id="a" * 64, base_model_definition_identity="constant_test_model:1",
        dataset_content_id="b" * 64, split_plan_fingerprint="c" * 64,
        calibration_method_candidates=(
            CalibrationMethodKind.IDENTITY, CalibrationMethodKind.PLATT, CalibrationMethodKind.ISOTONIC, CalibrationMethodKind.BETA,
        ),
        calibration_selection_metric=SelectionMetric.LOG_LOSS, calibration_tie_break_policy=CalibrationTieBreakPolicy.CANONICAL,
        minimum_calibration_sample_count=10, minimum_samples_per_class=2,
        inner_oof_policy=InnerSplitConfig(strategy="expanding_walk_forward", n_splits=3, test_size_fraction=0.15, embargo_bars=1),
        threshold_spec=ThresholdSpec(policy=ThresholdPolicyKind.F1, candidate_grid_size=51),
        abstention_spec=AbstentionSpec(policy=AbstentionPolicyKind.NONE),
        confidence_spec=ConfidenceSpec(very_low_max=0.2, low_max=0.4, medium_max=0.6, high_max=0.8),
        uncertainty_spec=UncertaintySpec(components=("entropy",), aggregation="mean"),
        probability_clipping=ProbabilityClippingPolicy(enabled=True, epsilon=1e-6),
        reliability_binning_specs=(ReliabilityBinningSpec(strategy=BinningStrategy.EQUAL_WIDTH, n_bins=10),),
        seed=seed, determinism_policy=DeterminismPolicy.STRICT,
    )


def _generate_oof(outer_fold: Fold, timeline: pd.DataFrame, spec: CalibrationSpec) -> InnerOofPredictionSet:
    return generate_inner_oof_predictions(
        outer_fold=outer_fold, timeline=timeline, feature_names=["feature_a"], label_column="label",
        label_horizon_bars=1, model_factory=ConstantTestModelFactory(), hyperparameters=ModelHyperparameters(values={}),
        objective=ObjectiveType.BINARY_CLASSIFICATION, seed_configuration=SeedConfiguration(master_seed=spec.seed), spec=spec,
        source_model_identity="constant_test_model:1", source_experiment_id=spec.source_experiment_id,
    )


class _Landmine:
    """A value that raises `AssertionError` on almost every operation a
    numeric feature/label column value could be subjected to (equality,
    float conversion, arithmetic, hashing). Placed at every outer-test
    row's `feature_a`/`label` value -- as opposed to poisoning `Fold.
    test_indices` itself (see the note below), this targets the ACTUAL
    leakage boundary: `optimization.inner_splits.validate_nested_plan`'s
    own leakage-verification legitimately reads `test_indices`' raw
    integer POSITIONS (a defense-in-depth overlap check, not a leakage
    vector -- position integers reveal nothing about the outer-test
    DATA); what must never happen is training-side code reading the
    timeline DATA rows at those positions."""

    def __repr__(self) -> str:
        return "<LANDMINE: outer-test row data was touched>"

    def __eq__(self, other: object) -> bool:
        raise AssertionError("LEAKAGE: an outer-test row value was compared")

    def __hash__(self) -> int:
        raise AssertionError("LEAKAGE: an outer-test row value was hashed")

    def __float__(self) -> float:
        raise AssertionError("LEAKAGE: an outer-test row value was converted to float")

    def __add__(self, other: object) -> object:
        raise AssertionError("LEAKAGE: an outer-test row value was used in arithmetic")

    __radd__ = __add__
    __sub__ = __add__
    __rsub__ = __add__
    __mul__ = __add__
    __rmul__ = __add__


class TestLandmineDataProvesOuterTestRowsAreNeverRead:
    """`generate_inner_oof_predictions` (and everything `fit_decision_
    policy` does downstream of it) must complete successfully even when
    every `feature_a`/`label` VALUE at an outer-test row position is a
    landmine that raises on any numeric use -- while `Fold.test_indices`/
    `.validation_indices` themselves remain ordinary, real arrays (so
    `validate_nested_plan`'s own legitimate position-only overlap check
    still runs normally). A passing test proves outer-test row DATA is
    structurally out of scope, not merely unused by convention."""

    def test_generate_inner_oof_predictions_never_reads_outer_test_row_data(self) -> None:
        timeline = _make_timeline()
        spec = _make_spec()
        fold = Fold(
            fold_index=0, train_indices=np.arange(0, 300), test_indices=np.arange(310, 400),
            train_start=timeline["open_time"].iloc[0], train_end=timeline["open_time"].iloc[299],
            test_start=timeline["open_time"].iloc[310], test_end=timeline["open_time"].iloc[399],
        )
        poisoned = timeline.astype({"feature_a": object, "label": object})
        landmine = _Landmine()
        # pandas-stubs' .loc overloads don't model arbitrary object-dtype
        # scalars -- this is a deliberate, valid runtime assignment.
        poisoned.loc[poisoned.index[310:], "feature_a"] = landmine  # type: ignore[call-overload]
        poisoned.loc[poisoned.index[310:], "label"] = landmine  # type: ignore[call-overload]

        # If ANY code path reads a poisoned row's feature/label value
        # (equality check, arithmetic, float conversion, hashing), this
        # raises AssertionError immediately -- a passing test proves the
        # negative structurally, at the exact call site.
        oof = _generate_oof(fold, poisoned, spec)
        assert len(oof.per_inner_fold) == 3
        policy = fit_decision_policy(oof, spec=spec)
        assert policy.threshold_report.selected_threshold is not None

    def test_fit_decision_policy_never_touches_poisoned_test_indices_indirectly(self) -> None:
        """Builds the OOF from an UNPOISONED fold (so real data exists to
        fit against), then proves `fit_decision_policy` -- which never
        even receives a `Fold` argument -- has no way to reach the outer
        fold's test partition at all (a structural, not run-time, proof:
        see `calibration.fitting.fit_decision_policy`'s signature)."""
        timeline = _make_timeline()
        spec = _make_spec()
        real_fold = Fold(
            fold_index=0, train_indices=np.arange(0, 300), test_indices=np.arange(310, 400),
            train_start=timeline["open_time"].iloc[0], train_end=timeline["open_time"].iloc[299],
            test_start=timeline["open_time"].iloc[310], test_end=timeline["open_time"].iloc[399],
        )
        oof = _generate_oof(real_fold, timeline, spec)
        import inspect

        signature = inspect.signature(fit_decision_policy)
        assert "outer_fold" not in signature.parameters
        assert "timeline" not in signature.parameters
        policy = fit_decision_policy(oof, spec=spec)
        assert policy.threshold_report.selected_threshold is not None


class TestOuterTestMutationNeverChangesTrainingSideArtifacts:
    """Mutating outer-test labels/features (to any value, including
    physically impossible ones) must produce a BYTE-IDENTICAL
    `InnerOofPredictionSet`/`FrozenDecisionPolicy` -- the strongest
    possible proof that training-side computation has no data-flow path
    from outer-test bytes at all."""

    def test_mutating_outer_test_labels_and_features_does_not_change_inner_oof(self) -> None:
        timeline = _make_timeline()
        spec = _make_spec()
        fold = Fold(
            fold_index=0, train_indices=np.arange(0, 300), test_indices=np.arange(310, 400),
            train_start=timeline["open_time"].iloc[0], train_end=timeline["open_time"].iloc[299],
            test_start=timeline["open_time"].iloc[310], test_end=timeline["open_time"].iloc[399],
        )
        baseline_oof = _generate_oof(fold, timeline, spec)
        baseline_policy = fit_decision_policy(baseline_oof, spec=spec)

        mutated = timeline.copy()
        mutated.loc[mutated.index[310:], "label"] = 1.0 - mutated.loc[mutated.index[310:], "label"]
        mutated.loc[mutated.index[310:], "feature_a"] = 999_999.0

        mutated_oof = _generate_oof(fold, mutated, spec)
        mutated_policy = fit_decision_policy(mutated_oof, spec=spec)

        assert mutated_oof.to_json_dict() == baseline_oof.to_json_dict()
        assert mutated_policy.to_json_dict() == baseline_policy.to_json_dict()

    def test_setting_outer_test_labels_to_an_invalid_domain_value_does_not_raise(self) -> None:
        """A stronger sentinel than mutation-and-compare: outer-test
        labels are set to a value (`-999.0`) that `RawPredictionSet.
        __post_init__`'s `true_labels` domain check would immediately
        REJECT if it ever reached a constructed `RawPredictionSet`.
        Success (no exception) proves those bytes were never read into
        one."""
        timeline = _make_timeline()
        spec = _make_spec()
        fold = Fold(
            fold_index=0, train_indices=np.arange(0, 300), test_indices=np.arange(310, 400),
            train_start=timeline["open_time"].iloc[0], train_end=timeline["open_time"].iloc[299],
            test_start=timeline["open_time"].iloc[310], test_end=timeline["open_time"].iloc[399],
        )
        poisoned = timeline.copy()
        poisoned.loc[poisoned.index[310:], "label"] = -999.0
        oof = _generate_oof(fold, poisoned, spec)  # must not raise
        fit_decision_policy(oof, spec=spec)  # must not raise


class TestNoOuterTestReferencePersistedInTrainingArtifacts:
    """Every `sample_positions` entry recorded in a training-stage
    artifact must be drawn exclusively from the outer fold's TRAIN
    partition -- never from its test partition, checked by direct set
    membership against the fold's own declared row positions."""

    def test_inner_oof_sample_positions_are_subset_of_outer_train(self) -> None:
        timeline = _make_timeline()
        spec = _make_spec()
        fold = Fold(
            fold_index=0, train_indices=np.arange(0, 300), test_indices=np.arange(310, 400),
            train_start=timeline["open_time"].iloc[0], train_end=timeline["open_time"].iloc[299],
            test_start=timeline["open_time"].iloc[310], test_end=timeline["open_time"].iloc[399],
        )
        oof = _generate_oof(fold, timeline, spec)
        train_positions = set(fold.train_indices.tolist())
        test_positions = set(fold.test_indices.tolist())
        for inner in oof.per_inner_fold:
            assert set(inner.sample_positions) <= train_positions
            assert set(inner.sample_positions).isdisjoint(test_positions)
            assert inner.fitted_on_rows is not None
            assert set(inner.fitted_on_rows) <= train_positions
            assert set(inner.fitted_on_rows).isdisjoint(test_positions)


class TestStructuralGuardsRejectDeliberatelyLeakyData:
    """Constructs deliberately-invalid `RawPredictionSet`s and confirms
    the platform's own structural guards reject them -- proving the
    guard exists and actually fires, not merely that it is documented."""

    def test_fitted_on_rows_overlapping_sample_positions_is_rejected(self) -> None:
        with pytest.raises(CalibrationDataError, match="leakage"):
            RawPredictionSet(
                schema_version=1, outer_fold_index=0, inner_fold_index=0,
                sample_positions=(10, 11, 12), timestamps=("2020-01-01T00:00:00+00:00",) * 3,
                raw_scores=None, raw_probabilities=(0.5, 0.5, 0.5), class_labels=(0.0, 1.0), positive_class_index=1,
                source_model_identity="m", source_experiment_id="e",
                true_labels=(0.0, 1.0, 0.0),
                fitted_on_rows=(5, 6, 11),  # 11 overlaps a sample_position -- the model trained on a row it also predicts
            )

    def test_inner_oof_prediction_set_rejects_outer_fold_index_mismatch(self) -> None:
        timeline = _make_timeline()
        spec = _make_spec()
        fold = Fold(
            fold_index=0, train_indices=np.arange(0, 300), test_indices=np.arange(310, 400),
            train_start=timeline["open_time"].iloc[0], train_end=timeline["open_time"].iloc[299],
            test_start=timeline["open_time"].iloc[310], test_end=timeline["open_time"].iloc[399],
        )
        oof = _generate_oof(fold, timeline, spec)
        raw = oof.to_json_dict()
        raw["outer_fold_index"] = 99  # claim a different outer fold than the inner folds actually belong to

        from quant_platform.calibration.fitting import InnerOofPredictionSet

        with pytest.raises(CalibrationDataError):
            InnerOofPredictionSet.from_json_dict(raw)

    def test_select_calibrator_never_receives_true_labels_from_outside_pooled_oof(self) -> None:
        """`select_calibrator`'s only label input is `RawPredictionSet.
        true_labels` on the POOLED (inner-OOF-only) object it is handed
        -- there is no second, separate label parameter through which an
        outer-test label could be smuggled in."""
        import inspect

        signature = inspect.signature(select_calibrator)
        label_like_params = [p for p in signature.parameters if "label" in p.lower() or "test" in p.lower()]
        assert label_like_params == [], f"select_calibrator has an unexpected label/test parameter: {label_like_params}"


class TestReleaseAuditCallGraphSentinels:
    """Release audit Section 4: an explicit, per-route structural proof
    for every stage `run_outer_fold_calibration` freezes BEFORE it ever
    reads outer-test labels -- threshold selection, confidence/uncertainty
    policy construction, and report rendering. Each check is signature-
    based (mirrors `test_select_calibrator_never_receives_true_labels_
    from_outside_pooled_oof` immediately above): no `Fold`, `timeline`, or
    label-shaped parameter exists anywhere in these functions' call
    surface, so there is no Python expression through which an outer-test
    label could reach them, regardless of what the caller does."""

    _DANGEROUS_EXACT_NAMES = frozenset({"fold", "outer_fold", "inner_fold", "test_fold", "timeline"})
    """Parameter names that would signal a raw `Fold`/timeline object in
    scope -- checked by EXACT match, not substring, so a legitimate
    post-evaluation parameter like `outer_fold_results`/`outer_fold_index`
    (a list of already-evaluated results / a plain int tag, never a raw
    `Fold`) is never a false positive."""

    @classmethod
    def _assert_no_fold_timeline_or_label_parameter(cls, fn: object, *, allow_result: bool = False) -> None:
        import inspect

        signature = inspect.signature(fn)  # type: ignore[arg-type]
        for name in signature.parameters:
            lowered = name.lower()
            assert lowered not in cls._DANGEROUS_EXACT_NAMES, f"{fn!r} has an unexpected fold/timeline parameter: {name!r}"
            if not allow_result:
                assert "label" not in lowered, f"{fn!r} has an unexpected label-shaped parameter: {name!r}"

    def test_threshold_selection_has_no_fold_or_timeline_parameter(self) -> None:
        """`evaluate_threshold_candidates` takes plain `(probabilities,
        labels)` arrays plus `spec` -- `labels` here means the CALLER-
        supplied training-side array (`fit_decision_policy` only ever
        passes pooled/per-inner-fold labels, never outer-test ones; see
        `test_fit_decision_policy_never_touches_poisoned_test_indices_
        indirectly` above for the proof that `fit_decision_policy` itself
        has no route to outer-test data at all). This test only confirms
        the narrower, structural half: no `Fold`/`timeline` parameter
        exists through which outer-test row identity could arrive."""
        import inspect

        from quant_platform.calibration.thresholds import evaluate_threshold_candidates

        signature = inspect.signature(evaluate_threshold_candidates)
        for name in signature.parameters:
            assert "fold" not in name.lower(), f"evaluate_threshold_candidates has an unexpected fold parameter: {name!r}"
            assert "timeline" not in name.lower(), f"evaluate_threshold_candidates has an unexpected timeline parameter: {name!r}"

    def test_confidence_policy_construction_has_no_fold_timeline_or_label_parameter(self) -> None:
        from quant_platform.calibration.confidence import compute_confidence

        self._assert_no_fold_timeline_or_label_parameter(compute_confidence)

    def test_uncertainty_policy_construction_has_no_fold_timeline_or_label_parameter(self) -> None:
        from quant_platform.calibration.uncertainty import compute_uncertainty

        self._assert_no_fold_timeline_or_label_parameter(compute_uncertainty)

    def test_reporting_functions_only_accept_already_evaluated_results_no_fold_or_timeline(self) -> None:
        """`build_calibration_report_json`/`render_calibration_report_
        markdown` take `outer_fold_results: Sequence[OuterFoldCalibrationResult]`
        -- by construction, an `OuterFoldCalibrationResult` object cannot
        exist without `run_outer_fold_calibration` having ALREADY read
        outer-test labels to populate its `classification_metrics`/
        `calibration_metrics_on_outer_test` fields (see that function's
        docstring: labels are read strictly after every calibration/
        threshold/confidence/uncertainty/abstention decision is final).
        Reporting therefore cannot inspect outer labels BEFORE evaluation
        -- there is no code path into these functions except through an
        already-evaluated result, and no `Fold`/`timeline` parameter
        through which raw outer-test data could arrive some other way."""
        from quant_platform.calibration.reporting import (
            build_calibration_report_json,
            render_calibration_report_markdown,
        )

        self._assert_no_fold_timeline_or_label_parameter(build_calibration_report_json, allow_result=True)
        self._assert_no_fold_timeline_or_label_parameter(render_calibration_report_markdown, allow_result=True)

    def test_fit_decision_policy_signature_has_no_fold_or_timeline_parameter(self) -> None:
        """Restates `test_fit_decision_policy_never_touches_poisoned_
        test_indices_indirectly`'s signature check as its own named,
        directly-discoverable proof for the release audit's call-graph
        table (Section 4): `fit_decision_policy` -- which orchestrates
        calibrator selection, threshold selection, and reliability
        binning together -- has no `Fold`/`timeline` parameter at all."""
        self._assert_no_fold_timeline_or_label_parameter(fit_decision_policy)
