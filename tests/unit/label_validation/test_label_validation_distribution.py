from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from quant_platform.label_validation.distribution import (
    LabelDistribution,
    bucket_values,
    compute_label_distribution,
    format_discrete_value,
    is_discrete_family,
)
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.models import LabelFamily


class TestIsDiscreteFamily:
    def test_direction_and_triple_barrier_are_discrete(self) -> None:
        assert is_discrete_family(LabelFamily.DIRECTION) is True
        assert is_discrete_family(LabelFamily.TRIPLE_BARRIER) is True

    def test_return_and_volatility_families_are_continuous(self) -> None:
        assert is_discrete_family(LabelFamily.NEXT_RETURN) is False
        assert is_discrete_family(LabelFamily.FORWARD_VOLATILITY) is False


class TestBucketValues:
    def test_discrete_family_buckets_by_exact_value(self) -> None:
        values = pd.Series([-1.0, 0.0, 1.0, 0.0])
        buckets = bucket_values(values, label_family=LabelFamily.DIRECTION)
        assert list(buckets) == [format_discrete_value(-1.0), format_discrete_value(0.0), format_discrete_value(1.0), format_discrete_value(0.0)]

    def test_continuous_family_buckets_into_deciles(self, next_return_bundle: LabelBundle) -> None:
        buckets = bucket_values(next_return_bundle.values, label_family=LabelFamily.NEXT_RETURN, bucket_count=10)
        assert buckets.dropna().nunique() <= 10

    def test_nan_stays_missing(self) -> None:
        # pandas may represent the missing marker as `None`, `nan`, or
        # `pd.NA` depending on the dtype `.map()` infers -- `pd.isna`,
        # never an identity check, is the correct way to test this
        # (production code uses `.dropna()`/`pd.isna()` throughout, never
        # `is None`).
        values = pd.Series([1.0, float("nan"), 2.0])
        buckets = bucket_values(values, label_family=LabelFamily.NEXT_RETURN)
        assert pd.isna(buckets.iloc[1])
        assert not pd.isna(buckets.iloc[0])
        assert not pd.isna(buckets.iloc[2])

    def test_constant_continuous_series_does_not_raise(self) -> None:
        values = pd.Series([1.0, 1.0, 1.0, 1.0])
        buckets = bucket_values(values, label_family=LabelFamily.NEXT_RETURN)
        assert buckets.nunique() == 1


class TestComputeLabelDistribution:
    def test_basic_fields(self, next_return_bundle: LabelBundle) -> None:
        distribution = compute_label_distribution(next_return_bundle)
        assert distribution.cardinality > 0
        assert distribution.entropy >= 0.0
        assert distribution.effective_cardinality >= 1.0
        assert abs(sum(distribution.class_ratios.values()) - 1.0) < 1e-9

    def test_direction_family_has_low_cardinality(self, direction_bundle: LabelBundle) -> None:
        distribution = compute_label_distribution(direction_bundle)
        assert distribution.cardinality <= 3

    def test_sparsity_matches_missing_fraction(self, next_return_bundle: LabelBundle) -> None:
        distribution = compute_label_distribution(next_return_bundle)
        expected = (next_return_bundle.row_count - next_return_bundle.valid_count) / next_return_bundle.row_count
        assert abs(distribution.sparsity - expected) < 1e-9

    def test_json_round_trip(self, next_return_bundle: LabelBundle) -> None:
        distribution = compute_label_distribution(next_return_bundle)
        restored = LabelDistribution.from_json_dict(distribution.to_json_dict())
        assert restored == distribution

    def test_high_sparsity_produces_warning_evidence(self, next_return_bundle: LabelBundle) -> None:
        mostly_nan = next_return_bundle.values.copy()
        mostly_nan.iloc[: int(len(mostly_nan) * 0.9)] = np.nan
        sparse_bundle = replace(next_return_bundle, values=mostly_nan)
        distribution = compute_label_distribution(sparse_bundle)
        assert any("sparsity" in e.finding for e in distribution.evidence)
