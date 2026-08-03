from __future__ import annotations

from dataclasses import replace

import numpy as np

from quant_platform.label_validation.balance import (
    EXTREME_IMBALANCE_RATIO_THRESHOLD,
    LabelBalance,
    compute_label_balance,
)
from quant_platform.labels.builder import LabelBundle


class TestComputeLabelBalance:
    def test_basic_fields(self, next_return_bundle: LabelBundle) -> None:
        balance = compute_label_balance(next_return_bundle)
        assert abs(sum(balance.class_fractions.values()) - 1.0) < 1e-9
        assert balance.imbalance_ratio is not None
        assert balance.imbalance_ratio >= 1.0

    def test_direction_family_reports_neutral_fraction(self, direction_bundle: LabelBundle) -> None:
        balance = compute_label_balance(direction_bundle)
        assert balance.neutral_fraction is not None

    def test_non_direction_family_has_no_neutral_fraction(self, next_return_bundle: LabelBundle) -> None:
        balance = compute_label_balance(next_return_bundle)
        assert balance.neutral_fraction is None

    def test_no_evidence_carries_a_recommendation(self, next_return_bundle: LabelBundle) -> None:
        balance = compute_label_balance(next_return_bundle)
        assert all(e.recommendation is None for e in balance.evidence)

    def test_no_evidence_is_blocking(self, next_return_bundle: LabelBundle) -> None:
        balance = compute_label_balance(next_return_bundle)
        assert all(e.blocking is False for e in balance.evidence)

    def test_extreme_imbalance_detected(self, direction_bundle: LabelBundle) -> None:
        # A discrete family (exact-value bucketing, never qcut-collapsed):
        # almost all UP, a small number DOWN -- a reliable 2-class split
        # with a large, exact imbalance ratio.
        n = direction_bundle.row_count
        skewed = np.full(n, 1.0)
        skewed[:5] = -1.0
        skewed_bundle = replace(direction_bundle, values=direction_bundle.values.__class__(skewed, index=direction_bundle.values.index))
        balance = compute_label_balance(skewed_bundle)
        assert balance.imbalance_ratio is not None
        assert balance.imbalance_ratio > EXTREME_IMBALANCE_RATIO_THRESHOLD
        assert balance.extreme_imbalance is True
        assert any("extreme" in e.finding for e in balance.evidence)

    def test_json_round_trip(self, next_return_bundle: LabelBundle) -> None:
        balance = compute_label_balance(next_return_bundle)
        restored = LabelBalance.from_json_dict(balance.to_json_dict())
        assert restored == balance
