from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from quant_platform.label_validation.degeneracy import (
    LabelDegeneracy,
    compute_label_degeneracy,
    detect_duplicate_labels,
)
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.forward_volatility import build_forward_volatility_specification
from quant_platform.labels.records import LabelRecord
from quant_platform.labels.volatility import REALIZED_STDDEV_ESTIMATOR_NAME


class TestComputeLabelDegeneracyHealthyBundle:
    def test_healthy_bundle_is_not_degenerate(self, next_return_bundle: LabelBundle) -> None:
        degeneracy = compute_label_degeneracy(next_return_bundle)
        assert degeneracy.is_empty is False
        assert degeneracy.is_constant is False
        assert degeneracy.has_impossible_labels is False
        assert degeneracy.is_blocking is False

    def test_json_round_trip(self, next_return_bundle: LabelBundle) -> None:
        degeneracy = compute_label_degeneracy(next_return_bundle)
        restored = LabelDegeneracy.from_json_dict(degeneracy.to_json_dict())
        assert restored == degeneracy


class TestComputeLabelDegeneracyConstant:
    def test_constant_values_flagged_and_blocking(self, next_return_bundle: LabelBundle) -> None:
        constant_values = pd.Series(np.full(next_return_bundle.row_count, 0.5))
        constant_bundle = replace(next_return_bundle, values=constant_values)
        degeneracy = compute_label_degeneracy(constant_bundle)
        assert degeneracy.is_constant is True
        assert degeneracy.is_blocking is True


class TestComputeLabelDegeneracyEmpty:
    def test_all_nan_values_flagged_and_blocking(self, next_return_bundle: LabelBundle) -> None:
        empty_values = pd.Series(np.full(next_return_bundle.row_count, np.nan))
        empty_bundle = replace(next_return_bundle, values=empty_values)
        degeneracy = compute_label_degeneracy(empty_bundle)
        assert degeneracy.is_empty is True
        assert degeneracy.is_blocking is True


class TestComputeLabelDegeneracyAllNeutral:
    def test_all_neutral_direction_flagged_and_blocking(self, direction_bundle: LabelBundle) -> None:
        all_neutral_values = direction_bundle.values.where(direction_bundle.values.isna(), 0.0)
        all_neutral_bundle = replace(direction_bundle, values=all_neutral_values)
        degeneracy = compute_label_degeneracy(all_neutral_bundle)
        assert degeneracy.is_all_neutral is True
        assert degeneracy.is_blocking is True


class TestComputeLabelDegeneracyImpossibleLabels:
    def test_out_of_domain_direction_values_flagged(self, direction_bundle: LabelBundle) -> None:
        tampered_values = direction_bundle.values.copy()
        first_valid_index = tampered_values.first_valid_index()
        tampered_values.loc[first_valid_index] = 42.0
        tampered_bundle = replace(direction_bundle, values=tampered_values)
        degeneracy = compute_label_degeneracy(tampered_bundle)
        assert degeneracy.has_impossible_labels is True
        assert degeneracy.impossible_value_count >= 1
        assert degeneracy.is_blocking is True

    def test_negative_forward_volatility_flagged(self, next_return_bundle: LabelBundle) -> None:
        spec = build_forward_volatility_specification(
            horizon_bars=10, volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        fake_values = pd.Series(np.full(20, -1.0))
        fake_bundle = replace(next_return_bundle, specification=spec, values=fake_values, row_count=20, valid_count=20)
        degeneracy = compute_label_degeneracy(fake_bundle)
        assert degeneracy.has_impossible_labels is True


class TestDetectDuplicateLabels:
    def test_no_duplicates_in_a_healthy_record_set(self, next_return_bundle: LabelBundle, next_return_records: tuple[LabelRecord, ...]) -> None:
        assert detect_duplicate_labels(next_return_records) == ()

    def test_detects_a_genuine_label_id_collision(self, next_return_records: tuple[LabelRecord, ...]) -> None:
        colliding = replace(next_return_records[1], label_id=next_return_records[0].label_id)
        records_with_collision = (next_return_records[0], colliding, *next_return_records[2:])
        duplicates = detect_duplicate_labels(records_with_collision)
        assert duplicates == (next_return_records[0].label_id,)
