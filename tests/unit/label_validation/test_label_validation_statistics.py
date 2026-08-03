from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from quant_platform.label_validation.statistics import LabelStatistics, compute_label_statistics
from quant_platform.labels.builder import LabelBundle


class TestComputeLabelStatistics:
    def test_basic_fields(self, next_return_bundle: LabelBundle) -> None:
        stats = compute_label_statistics(next_return_bundle)
        assert stats.row_count == next_return_bundle.row_count
        assert stats.valid_count == next_return_bundle.valid_count
        assert stats.missing_count == stats.row_count - stats.valid_count
        assert stats.minimum is not None and stats.maximum is not None
        assert stats.minimum <= stats.median <= stats.maximum  # type: ignore[operator]

    def test_percentiles_are_ordered(self, next_return_bundle: LabelBundle) -> None:
        stats = compute_label_statistics(next_return_bundle)
        assert stats.p05 is not None and stats.p95 is not None
        assert stats.p05 <= stats.p95

    def test_empty_bundle_returns_none_stats(self, next_return_bundle: LabelBundle) -> None:
        all_nan_values = pd.Series(np.full(next_return_bundle.row_count, np.nan))
        empty_bundle = replace(next_return_bundle, values=all_nan_values, valid_count=0)
        stats = compute_label_statistics(empty_bundle)
        assert stats.valid_count == 0
        assert stats.missing_count == stats.row_count
        assert stats.mean is None
        assert stats.std is None
        assert stats.minimum is None

    def test_json_round_trip(self, next_return_bundle: LabelBundle) -> None:
        stats = compute_label_statistics(next_return_bundle)
        restored = LabelStatistics.from_json_dict(stats.to_json_dict())
        assert restored == stats
