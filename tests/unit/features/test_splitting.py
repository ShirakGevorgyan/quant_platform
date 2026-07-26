from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.exceptions import ValidationSplitError
from quant_platform.features.splitting import build_chronological_split, build_walk_forward_splits


def _timestamps(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"))


class TestChronologicalSplit:
    def test_basic_proportions_and_ordering(self) -> None:
        timestamps = _timestamps(1000)
        plan = build_chronological_split(timestamps, train_fraction=0.7, validation_fraction=0.15)
        train, validation, test = plan.get("train"), plan.get("validation"), plan.get("test")
        assert train.indices.max() < validation.indices.min()
        assert validation.indices.max() < test.indices.min()

    def test_no_overlap_between_any_two_splits(self) -> None:
        timestamps = _timestamps(1000)
        plan = build_chronological_split(
            timestamps, train_fraction=0.6, validation_fraction=0.2, purge_bars=10, embargo_bars=10
        )
        train_set = set(plan.get("train").indices.tolist())
        val_set = set(plan.get("validation").indices.tolist())
        test_set = set(plan.get("test").indices.tolist())
        assert train_set.isdisjoint(val_set)
        assert val_set.isdisjoint(test_set)
        assert train_set.isdisjoint(test_set)

    def test_purge_and_embargo_actually_remove_boundary_rows(self) -> None:
        timestamps = _timestamps(100)
        no_gap_plan = build_chronological_split(timestamps, train_fraction=0.7, validation_fraction=0.15)
        gapped_plan = build_chronological_split(
            timestamps, train_fraction=0.7, validation_fraction=0.15, purge_bars=5, embargo_bars=5
        )
        assert len(gapped_plan.get("train").indices) < len(no_gap_plan.get("train").indices)
        assert gapped_plan.get("validation").indices.min() > no_gap_plan.get("validation").indices.min()

    def test_rejects_invalid_fractions(self) -> None:
        timestamps = _timestamps(100)
        with pytest.raises(ValidationSplitError):
            build_chronological_split(timestamps, train_fraction=0.7, validation_fraction=0.4)
        with pytest.raises(ValidationSplitError):
            build_chronological_split(timestamps, train_fraction=0.0, validation_fraction=0.5)

    def test_rejects_unsorted_timestamps(self) -> None:
        timestamps = _timestamps(10).sample(frac=1.0, random_state=0).reset_index(drop=True)
        with pytest.raises(ValidationSplitError):
            build_chronological_split(timestamps, train_fraction=0.5, validation_fraction=0.3)

    def test_too_much_purge_leaves_empty_split_raises(self) -> None:
        timestamps = _timestamps(20)
        with pytest.raises(ValidationSplitError):
            build_chronological_split(timestamps, train_fraction=0.1, validation_fraction=0.1, purge_bars=100)

    def test_to_json_dict_contains_expected_metadata(self) -> None:
        timestamps = _timestamps(100)
        plan = build_chronological_split(timestamps, train_fraction=0.7, validation_fraction=0.15, purge_bars=2, embargo_bars=3)
        payload = plan.to_json_dict()
        assert payload["strategy"] == "chronological"
        assert payload["purge_bars"] == 2
        assert payload["embargo_bars"] == 3
        assert {s["name"] for s in payload["splits"]} == {"train", "validation", "test"}


class TestWalkForwardSplit:
    def test_expanding_window_train_grows_across_folds(self) -> None:
        timestamps = _timestamps(500)
        plan = build_walk_forward_splits(timestamps, n_splits=3, test_size=50, label_horizon=2, embargo=2)
        train_sizes = [len(plan.get(f"fold_{k}_train").indices) for k in range(3)]
        assert train_sizes[0] < train_sizes[1] < train_sizes[2]

    def test_rolling_window_train_bounded_by_max_train_size(self) -> None:
        timestamps = _timestamps(500)
        plan = build_walk_forward_splits(timestamps, n_splits=3, test_size=50, max_train_size=80)
        for k in range(3):
            assert len(plan.get(f"fold_{k}_train").indices) <= 80

    def test_purge_and_embargo_gap_present_between_train_and_test(self) -> None:
        timestamps = _timestamps(500)
        plan = build_walk_forward_splits(timestamps, n_splits=2, test_size=50, label_horizon=5, embargo=5)
        for k in range(2):
            train = plan.get(f"fold_{k}_train")
            test = plan.get(f"fold_{k}_test")
            gap = test.indices.min() - train.indices.max()
            assert gap >= 10  # label_horizon + embargo

    def test_test_folds_do_not_overlap_each_other(self) -> None:
        timestamps = _timestamps(500)
        plan = build_walk_forward_splits(timestamps, n_splits=3, test_size=50)
        test_sets = [set(plan.get(f"fold_{k}_test").indices.tolist()) for k in range(3)]
        assert test_sets[0].isdisjoint(test_sets[1])
        assert test_sets[1].isdisjoint(test_sets[2])

    def test_strategy_label_reflects_expanding_vs_rolling(self) -> None:
        timestamps = _timestamps(300)
        expanding = build_walk_forward_splits(timestamps, n_splits=2, test_size=50)
        rolling = build_walk_forward_splits(timestamps, n_splits=2, test_size=50, max_train_size=40)
        assert expanding.strategy == "expanding_walk_forward"
        assert rolling.strategy == "rolling_walk_forward"
