from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelValidationRequestError
from quant_platform.label_validation.stability import LabelStability, compute_label_stability
from quant_platform.labels.builder import LabelBundle


class TestComputeLabelStabilityTemporal:
    def test_basic_fields_present(self, next_return_bundle: LabelBundle) -> None:
        stability = compute_label_stability(next_return_bundle, window_bars=20)
        assert stability.temporal.rolling_entropy_std is not None
        assert stability.temporal.rolling_variance_std is not None
        assert stability.temporal.window_stability_score is not None
        assert 0.0 <= stability.temporal.window_stability_score <= 1.0

    def test_too_few_rows_returns_none_fields(self, next_return_bundle: LabelBundle) -> None:
        stability = compute_label_stability(next_return_bundle, window_bars=10_000)
        assert stability.temporal.rolling_entropy_std is None
        assert stability.temporal.window_stability_score is None

    def test_no_regime_by_default(self, next_return_bundle: LabelBundle) -> None:
        stability = compute_label_stability(next_return_bundle)
        assert stability.regime is None

    def test_json_round_trip(self, next_return_bundle: LabelBundle) -> None:
        stability = compute_label_stability(next_return_bundle)
        restored = LabelStability.from_json_dict(stability.to_json_dict())
        assert restored.temporal == stability.temporal
        assert restored.label_specification_id == stability.label_specification_id


class TestComputeLabelStabilityRegime:
    def test_regime_grouping(self, next_return_bundle: LabelBundle) -> None:
        n = next_return_bundle.row_count
        regimes = pd.Series(["bull"] * (n // 2) + ["bear"] * (n - n // 2))
        stability = compute_label_stability(next_return_bundle, regime_assignment=regimes)
        assert stability.regime is not None
        assert set(stability.regime.per_regime_mean) <= {"bull", "bear"}
        assert stability.regime.cross_regime_mean_spread is not None

    def test_mismatched_length_raises(self, next_return_bundle: LabelBundle) -> None:
        regimes = pd.Series(["bull"] * 3)
        with pytest.raises(LabelValidationRequestError):
            compute_label_stability(next_return_bundle, regime_assignment=regimes)

    def test_single_regime_has_no_spread(self, next_return_bundle: LabelBundle) -> None:
        regimes = pd.Series(["bull"] * next_return_bundle.row_count)
        stability = compute_label_stability(next_return_bundle, regime_assignment=regimes)
        assert stability.regime is not None
        assert stability.regime.cross_regime_mean_spread is None


class TestWindowStabilityDetectsInstability:
    def test_a_regime_shift_lowers_window_stability(self, next_return_bundle: LabelBundle) -> None:
        n = next_return_bundle.row_count
        shifted = next_return_bundle.values.to_numpy().copy()
        shifted[: n // 2] = np.nan_to_num(shifted[: n // 2], nan=0.0) + 0.0
        shifted[n // 2 :] = np.nan_to_num(shifted[n // 2 :], nan=0.0) + 10.0  # a huge level shift halfway through
        shifted_bundle = replace(next_return_bundle, values=pd.Series(shifted, index=next_return_bundle.values.index))
        stability = compute_label_stability(shifted_bundle, window_bars=20)
        assert stability.temporal.window_stability_score is not None
        assert stability.temporal.window_stability_score < 0.5
