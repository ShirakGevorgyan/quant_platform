"""Tests for the purged walk-forward cross-validation splitter."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from quant_platform.core.exceptions import ValidationSplitError
from quant_platform.validation.walk_forward import PurgedWalkForwardSplitter

UTC = timezone.utc


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range(start=datetime(2024, 1, 1, tzinfo=UTC), periods=n, freq="1h")


class TestConstructionValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_splits": 0, "test_size": 10},
            {"n_splits": -1, "test_size": 10},
            {"n_splits": 4, "test_size": 0},
            {"n_splits": 4, "test_size": -5},
            {"n_splits": 4, "test_size": 10, "label_horizon": -1},
            {"n_splits": 4, "test_size": 10, "embargo": -1},
            {"n_splits": 4, "test_size": 10, "max_train_size": 0},
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            PurgedWalkForwardSplitter(**kwargs)


class TestBasicSplitting:
    def test_produces_correct_number_of_folds(self) -> None:
        splitter = PurgedWalkForwardSplitter(n_splits=4, test_size=10)
        splits = list(splitter.split(_index(100)))
        assert len(splits) == 4

    def test_test_blocks_are_contiguous_and_non_overlapping(self) -> None:
        splitter = PurgedWalkForwardSplitter(n_splits=4, test_size=10)
        splits = list(splitter.split(_index(100)))
        expected_test_starts = [60, 70, 80, 90]
        for split, expected_start in zip(splits, expected_test_starts, strict=True):
            assert split.test_indices[0] == expected_start
            assert split.test_indices[-1] == expected_start + 9
            assert len(split.test_indices) == 10

    def test_train_never_overlaps_test_with_zero_gap(self) -> None:
        splitter = PurgedWalkForwardSplitter(n_splits=4, test_size=10)
        for split in splitter.split(_index(100)):
            assert split.train_indices[-1] < split.test_indices[0]
            assert set(split.train_indices).isdisjoint(set(split.test_indices))

    def test_expanding_window_train_always_starts_at_zero(self) -> None:
        splitter = PurgedWalkForwardSplitter(n_splits=4, test_size=10)
        for split in splitter.split(_index(100)):
            assert split.train_indices[0] == 0

    def test_timestamps_match_index_positions(self) -> None:
        index = _index(100)
        splitter = PurgedWalkForwardSplitter(n_splits=4, test_size=10)
        first_split = next(splitter.split(index))
        assert first_split.train_start == index[0]
        assert first_split.train_end == index[first_split.train_indices[-1]]
        assert first_split.test_start == index[60]
        assert first_split.test_end == index[69]

    def test_fold_number_is_recorded(self) -> None:
        splitter = PurgedWalkForwardSplitter(n_splits=3, test_size=5)
        folds = [split.fold for split in splitter.split(_index(50))]
        assert folds == [0, 1, 2]


class TestPurgingAndEmbargo:
    def test_label_horizon_shrinks_training_end_boundary(self) -> None:
        splitter = PurgedWalkForwardSplitter(n_splits=1, test_size=10, label_horizon=5)
        split = next(splitter.split(_index(100)))
        # test_start = 90; train_end should stop 5 bars earlier (at 85)
        assert split.train_indices[-1] == 84  # last index BEFORE the gap
        assert split.test_indices[0] == 90

    def test_embargo_shrinks_training_end_boundary(self) -> None:
        splitter = PurgedWalkForwardSplitter(n_splits=1, test_size=10, embargo=3)
        split = next(splitter.split(_index(100)))
        assert split.train_indices[-1] == 86  # gap of 3 before test_start=90

    def test_label_horizon_and_embargo_combine(self) -> None:
        splitter = PurgedWalkForwardSplitter(n_splits=1, test_size=10, label_horizon=5, embargo=3)
        assert splitter.purge_gap == 8
        split = next(splitter.split(_index(100)))
        assert split.train_indices[-1] == 81  # 90 - 8 - 1

    def test_gap_consuming_all_training_data_raises(self) -> None:
        # test_start for the only fold with n_splits=1, test_size=10, n=15 -> test_start=5.
        # A gap of 10 would push train_end to 0 (no bars left before the boundary,
        # since train_start also = 0) -> no training samples remain.
        splitter = PurgedWalkForwardSplitter(n_splits=1, test_size=10, label_horizon=10)
        with pytest.raises(ValidationSplitError, match="no training samples remain"):
            list(splitter.split(_index(15)))


class TestRollingWindow:
    def test_max_train_size_caps_training_window(self) -> None:
        splitter = PurgedWalkForwardSplitter(n_splits=4, test_size=10, max_train_size=20)
        splits = list(splitter.split(_index(100)))
        last_fold = splits[-1]  # test_start=90, train_end=90 -> train_start=max(0,70)=70
        assert len(last_fold.train_indices) == 20
        assert last_fold.train_indices[0] == 70
        assert last_fold.train_indices[-1] == 89

    def test_max_train_size_clips_to_zero_for_early_folds(self) -> None:
        # First fold: test_start=60, train_end=60, max_train_size=1000 -> would
        # want train_start=-940, clipped to 0.
        splitter = PurgedWalkForwardSplitter(n_splits=4, test_size=10, max_train_size=1000)
        first_split = next(splitter.split(_index(100)))
        assert first_split.train_indices[0] == 0


class TestErrorHandling:
    def test_insufficient_samples_raises(self) -> None:
        splitter = PurgedWalkForwardSplitter(n_splits=5, test_size=10)
        with pytest.raises(ValidationSplitError, match="Not enough samples"):
            list(splitter.split(_index(30)))  # need >= 50

    def test_non_monotonic_timestamps_raises(self) -> None:
        index = _index(50)
        shuffled = index[::-1]
        splitter = PurgedWalkForwardSplitter(n_splits=2, test_size=5)
        with pytest.raises(ValidationSplitError, match="monotonically increasing"):
            list(splitter.split(shuffled))

    def test_accepts_plain_series_of_timestamps(self) -> None:
        series = pd.Series(_index(50))
        splitter = PurgedWalkForwardSplitter(n_splits=2, test_size=5)
        splits = list(splitter.split(series))
        assert len(splits) == 2
