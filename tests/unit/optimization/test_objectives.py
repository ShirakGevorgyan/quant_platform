"""Milestone 4D: primary-metric/objective compatibility validation and the
undefined-primary-metric-on-a-trial aggregation policy (never silently
converted to zero)."""

from __future__ import annotations

import pytest

from quant_platform.ml.models import ObjectiveType
from quant_platform.optimization.objectives import (
    aggregate_primary_metric,
    metric_direction_multiplier,
    validate_primary_metric,
)


class TestValidatePrimaryMetric:
    def test_unknown_metric_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="not declared"):
            validate_primary_metric(ObjectiveType.BINARY_CLASSIFICATION, "not_a_real_metric")

    def test_regression_metric_rejected_for_classification_objective(self) -> None:
        with pytest.raises(ValueError, match="not compatible"):
            validate_primary_metric(ObjectiveType.BINARY_CLASSIFICATION, "rmse")

    def test_classification_metric_rejected_for_regression_objective(self) -> None:
        with pytest.raises(ValueError, match="not compatible"):
            validate_primary_metric(ObjectiveType.REGRESSION, "accuracy")

    def test_compatible_pairs_accepted(self) -> None:
        validate_primary_metric(ObjectiveType.BINARY_CLASSIFICATION, "accuracy")
        validate_primary_metric(ObjectiveType.REGRESSION, "rmse")


class TestMetricDirectionMultiplier:
    def test_higher_is_better_metric_gets_positive_multiplier(self) -> None:
        assert metric_direction_multiplier("accuracy") == 1

    def test_lower_is_better_metric_gets_negative_multiplier(self) -> None:
        assert metric_direction_multiplier("rmse") == -1


class TestAggregatePrimaryMetric:
    def test_none_values_never_silently_become_zero(self) -> None:
        outcome = aggregate_primary_metric([1.0, None, 2.0], min_successful_inner_folds=1)
        assert outcome.is_valid
        assert outcome.aggregate_value == pytest.approx(1.5)  # mean of [1.0, 2.0] only, never [1.0, 0.0, 2.0]
        assert outcome.successful_inner_folds == 2
        assert outcome.total_inner_folds == 3

    def test_too_few_successful_folds_is_invalid(self) -> None:
        outcome = aggregate_primary_metric([None, None, 1.0], min_successful_inner_folds=2)
        assert not outcome.is_valid
        assert outcome.aggregate_value is None
        assert outcome.reason is not None

    def test_all_none_is_invalid_with_zero_successful(self) -> None:
        outcome = aggregate_primary_metric([None, None], min_successful_inner_folds=1)
        assert not outcome.is_valid
        assert outcome.successful_inner_folds == 0

    def test_exactly_meeting_the_minimum_is_valid(self) -> None:
        outcome = aggregate_primary_metric([1.0, None], min_successful_inner_folds=1)
        assert outcome.is_valid

    def test_empty_sequence_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            aggregate_primary_metric([], min_successful_inner_folds=1)
