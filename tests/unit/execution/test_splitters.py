from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.unit.execution.conftest import make_timeline, write_synthetic_research_dataset

from quant_platform.core.exceptions import FoldValidationError
from quant_platform.core.types import Timeframe
from quant_platform.execution.splitters import (
    EmbargoSpec,
    Fold,
    FoldPlan,
    PurgeSpec,
    build_folds_from_split_binding,
    fold_row_counts,
    generate_blocked_time_folds,
    generate_expanding_folds,
    generate_grouped_walk_forward_folds,
    generate_rolling_folds,
    iter_fold_bounds,
    reconstruct_dataset_timeline,
    required_label_purge_bars_for,
)
from quant_platform.ml.models import SplitBinding


def _timestamps(n: int = 1000) -> pd.Series:
    return pd.Series(pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"))


class TestPurgeSpec:
    def test_bars_used_directly(self) -> None:
        assert PurgeSpec(bars=7).resolve_bars(Timeframe.M1) == 7

    def test_timedelta_converted_and_rounded_up(self) -> None:
        assert PurgeSpec(timedelta=pd.Timedelta(minutes=10)).resolve_bars(Timeframe.M1) == 10
        # 90 seconds over M1 bars rounds UP to 2 bars, never under-purges.
        assert PurgeSpec(timedelta=pd.Timedelta(seconds=90)).resolve_bars(Timeframe.M1) == 2

    def test_calendar_days_converted(self) -> None:
        assert EmbargoSpec(calendar_days=1).resolve_bars(Timeframe.H1) == 24

    def test_zero_timedelta_and_zero_calendar_days_resolve_to_zero(self) -> None:
        assert PurgeSpec(timedelta=pd.Timedelta(0)).resolve_bars(Timeframe.M1) == 0
        assert EmbargoSpec(calendar_days=0).resolve_bars(Timeframe.M1) == 0

    def test_requires_exactly_one_field(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            PurgeSpec().resolve_bars(Timeframe.M1)
        with pytest.raises(ValueError, match="exactly one"):
            PurgeSpec(bars=1, timedelta=pd.Timedelta(minutes=1)).resolve_bars(Timeframe.M1)

    def test_negative_values_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            PurgeSpec(bars=-1).resolve_bars(Timeframe.M1)
        with pytest.raises(ValueError, match=">= 0"):
            EmbargoSpec(timedelta=pd.Timedelta(minutes=-1)).resolve_bars(Timeframe.M1)
        with pytest.raises(ValueError, match=">= 0"):
            PurgeSpec(calendar_days=-1).resolve_bars(Timeframe.M1)


class TestExpandingFolds:
    def test_produces_requested_fold_count(self) -> None:
        plan = generate_expanding_folds(
            _timestamps(), n_splits=5, test_size=100, purge_bars=5, embargo_bars=2, label_horizon_bars=5,
        )
        assert len(plan.folds) == 5
        assert plan.strategy == "expanding_walk_forward"

    def test_train_always_starts_at_row_zero(self) -> None:
        plan = generate_expanding_folds(_timestamps(), n_splits=4, test_size=50, label_horizon_bars=0)
        for fold in plan.folds:
            assert int(fold.train_indices[0]) == 0

    def test_train_size_grows_across_folds(self) -> None:
        plan = generate_expanding_folds(_timestamps(), n_splits=4, test_size=50, label_horizon_bars=0)
        sizes = [len(f.train_indices) for f in plan.folds]
        assert sizes == sorted(sizes)
        assert len(set(sizes)) == len(sizes)  # strictly increasing

    def test_purge_and_embargo_produce_correct_gap(self) -> None:
        plan = generate_expanding_folds(
            _timestamps(), n_splits=3, test_size=100, purge_bars=5, embargo_bars=3, label_horizon_bars=5,
        )
        for fold in plan.folds:
            gap = int(fold.test_indices.min()) - int(fold.train_indices.max()) - 1
            assert gap == 8

    def test_validation_carve_out(self) -> None:
        plan = generate_expanding_folds(
            _timestamps(), n_splits=3, test_size=100, purge_bars=5, validation_fraction=0.2, label_horizon_bars=5,
        )
        for fold in plan.folds:
            assert len(fold.validation_indices) > 0
            assert fold.validation_start is not None and fold.validation_end is not None
            train_val_gap = int(fold.validation_indices.min()) - int(fold.train_indices.max()) - 1
            assert train_val_gap == 5
            assert int(fold.validation_indices.max()) < int(fold.test_indices.min())

    def test_validation_fraction_zero_means_no_validation(self) -> None:
        plan = generate_expanding_folds(
            _timestamps(), n_splits=3, test_size=100, validation_fraction=0.0, label_horizon_bars=0,
        )
        for fold in plan.folds:
            assert len(fold.validation_indices) == 0
            assert fold.validation_start is None and fold.validation_end is None

    def test_too_few_rows_raises(self) -> None:
        from quant_platform.core.exceptions import ValidationSplitError

        with pytest.raises(ValidationSplitError):
            generate_expanding_folds(_timestamps(10), n_splits=5, test_size=100, label_horizon_bars=0)

    def test_label_horizon_bars_is_required_not_defaulted(self) -> None:
        """There is no default for `label_horizon_bars` -- a caller cannot
        accidentally get a silently-disabled (0) label-information purge
        check merely by omitting it; this must fail loudly at the call
        site, not at runtime deep inside validation."""
        with pytest.raises(TypeError, match="label_horizon_bars"):
            generate_expanding_folds(_timestamps(), n_splits=5, test_size=100)  # type: ignore[call-arg]


class TestRollingFolds:
    def test_train_size_capped(self) -> None:
        plan = generate_rolling_folds(_timestamps(), n_splits=4, test_size=50, max_train_size=200, label_horizon_bars=0)
        assert plan.strategy == "rolling_walk_forward"
        for fold in plan.folds:
            assert len(fold.train_indices) <= 200

    def test_train_window_slides_forward(self) -> None:
        plan = generate_rolling_folds(_timestamps(), n_splits=4, test_size=50, max_train_size=200, label_horizon_bars=0)
        starts = [int(f.train_indices[0]) for f in plan.folds]
        assert starts == sorted(starts)
        assert len(set(starts)) > 1


class TestBlockedTimeFolds:
    def test_produces_n_blocks_minus_one_folds(self) -> None:
        plan = generate_blocked_time_folds(_timestamps(1000), n_blocks=5, label_horizon_bars=0)
        assert len(plan.folds) == 4
        assert plan.strategy == "blocked_time_split"

    def test_requires_at_least_two_blocks(self) -> None:
        with pytest.raises(ValueError, match="n_blocks"):
            generate_blocked_time_folds(_timestamps(100), n_blocks=1, label_horizon_bars=0)

    def test_too_many_blocks_for_row_count_raises(self) -> None:
        with pytest.raises(ValueError, match="too large"):
            generate_blocked_time_folds(_timestamps(3), n_blocks=10, label_horizon_bars=0)


class TestGroupedWalkForwardFolds:
    def test_generates_folds_per_group(self) -> None:
        ts = _timestamps(600)
        df = pd.DataFrame({"open_time": ts, "sym": ["A"] * 300 + ["B"] * 300})
        plan = generate_grouped_walk_forward_folds(
            df, timestamp_column="open_time", group_column="sym", n_splits=2, test_size=50, purge_bars=1,
            embargo_bars=1, label_horizon_bars=1,
        )
        assert plan.strategy == "grouped_walk_forward"
        groups = {f.group for f in plan.folds}
        assert groups == {"A", "B"}
        assert len(plan.folds) == 4  # 2 splits x 2 groups

    def test_fold_indices_are_sequential_regardless_of_group(self) -> None:
        ts = _timestamps(600)
        df = pd.DataFrame({"open_time": ts, "sym": ["A"] * 300 + ["B"] * 300})
        plan = generate_grouped_walk_forward_folds(
            df, timestamp_column="open_time", group_column="sym", n_splits=2, test_size=50, label_horizon_bars=0,
        )
        assert [f.fold_index for f in plan.folds] == list(range(len(plan.folds)))

    def test_groups_never_cross_contaminate_row_positions(self) -> None:
        """Each group's folds must only ever reference row positions that
        actually belong to THAT group -- proving the local-to-global
        position mapping is correct even when groups interleave."""
        ts = _timestamps(400)
        # Interleaved groups (not contiguous blocks) -- a stronger proof
        # than two contiguous halves that position mapping is correct.
        df = pd.DataFrame({"open_time": ts, "sym": (["A", "B"] * 200)})
        plan = generate_grouped_walk_forward_folds(
            df, timestamp_column="open_time", group_column="sym", n_splits=2, test_size=20, label_horizon_bars=0,
        )
        a_positions = set(df.index[df["sym"] == "A"])
        b_positions = set(df.index[df["sym"] == "B"])
        for fold in plan.folds:
            all_positions = set(fold.train_indices.tolist()) | set(fold.test_indices.tolist())
            if fold.group == "A":
                assert all_positions <= a_positions
            else:
                assert all_positions <= b_positions

    def test_missing_columns_raise(self) -> None:
        df = pd.DataFrame({"open_time": _timestamps(10), "sym": ["A"] * 10})
        with pytest.raises(ValueError, match="group_column"):
            generate_grouped_walk_forward_folds(
                df, timestamp_column="open_time", group_column="missing", n_splits=1, test_size=2, label_horizon_bars=0,
            )
        with pytest.raises(ValueError, match="timestamp_column"):
            generate_grouped_walk_forward_folds(
                df, timestamp_column="missing", group_column="sym", n_splits=1, test_size=2, label_horizon_bars=0,
            )


class TestFoldConstruction:
    def test_fold_index_must_be_non_negative(self) -> None:
        with pytest.raises(FoldValidationError):
            Fold(
                fold_index=-1, train_indices=np.arange(5), test_indices=np.arange(5, 10),
                train_start=pd.Timestamp("2024-01-01", tz="UTC"), train_end=pd.Timestamp("2024-01-02", tz="UTC"),
                test_start=pd.Timestamp("2024-01-03", tz="UTC"), test_end=pd.Timestamp("2024-01-04", tz="UTC"),
            )

    def test_empty_train_or_test_rejected(self) -> None:
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        with pytest.raises(FoldValidationError):
            Fold(fold_index=0, train_indices=np.array([], dtype=np.int64), test_indices=np.arange(5), train_start=ts, train_end=ts, test_start=ts, test_end=ts)
        with pytest.raises(FoldValidationError):
            Fold(fold_index=0, train_indices=np.arange(5), test_indices=np.array([], dtype=np.int64), train_start=ts, train_end=ts, test_start=ts, test_end=ts)


class TestFoldPlanConstruction:
    def test_empty_folds_rejected(self) -> None:
        with pytest.raises(FoldValidationError, match="at least one"):
            FoldPlan(strategy="x", folds=(), purge_bars=0, embargo_bars=0, total_rows=0, label_horizon_bars=0, required_label_purge_bars=0)

    def test_fold_indices_must_be_0_to_n_minus_1_in_order(self) -> None:
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        f0 = Fold(fold_index=0, train_indices=np.arange(5), test_indices=np.arange(5, 10), train_start=ts, train_end=ts, test_start=ts, test_end=ts)
        f2 = Fold(fold_index=2, train_indices=np.arange(5), test_indices=np.arange(5, 10), train_start=ts, train_end=ts, test_start=ts, test_end=ts)
        with pytest.raises(FoldValidationError, match=r"0\.\.N-1"):
            FoldPlan(strategy="x", folds=(f0, f2), purge_bars=0, embargo_bars=0, total_rows=10, label_horizon_bars=0, required_label_purge_bars=0)

    def test_negative_purge_or_embargo_rejected(self) -> None:
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        f0 = Fold(fold_index=0, train_indices=np.arange(5), test_indices=np.arange(5, 10), train_start=ts, train_end=ts, test_start=ts, test_end=ts)
        with pytest.raises(FoldValidationError, match=">= 0"):
            FoldPlan(strategy="x", folds=(f0,), purge_bars=-1, embargo_bars=0, total_rows=10, label_horizon_bars=0, required_label_purge_bars=0)

    def test_negative_label_horizon_bars_rejected(self) -> None:
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        f0 = Fold(fold_index=0, train_indices=np.arange(5), test_indices=np.arange(5, 10), train_start=ts, train_end=ts, test_start=ts, test_end=ts)
        with pytest.raises(FoldValidationError, match=">= 0"):
            FoldPlan(strategy="x", folds=(f0,), purge_bars=0, embargo_bars=0, total_rows=10, label_horizon_bars=-1, required_label_purge_bars=0)

    def test_required_label_purge_bars_inconsistent_with_horizon_rejected(self) -> None:
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        f0 = Fold(fold_index=0, train_indices=np.arange(5), test_indices=np.arange(5, 10), train_start=ts, train_end=ts, test_start=ts, test_end=ts)
        with pytest.raises(FoldValidationError, match="inconsistent"):
            FoldPlan(strategy="x", folds=(f0,), purge_bars=0, embargo_bars=0, total_rows=10, label_horizon_bars=5, required_label_purge_bars=999)


class TestRequiredLabelPurgeBarsFor:
    """Pins the off-by-one proof `required_label_purge_bars_for` documents:
    the minimum purge equals `label_horizon_bars` exactly, for every
    horizon this codebase's label kinds can produce (see
    `features.labels`, whose `close.shift(-horizon_bars)`/
    `range(1, horizon_bars + 1)` forward windows are all `[i+1,
    i+horizon_bars]`, never beyond)."""

    def test_identity_for_positive_horizons(self) -> None:
        for horizon in (1, 2, 5, 12, 100):
            assert required_label_purge_bars_for(horizon) == horizon

    def test_zero_horizon_requires_zero_purge(self) -> None:
        assert required_label_purge_bars_for(0) == 0

    def test_negative_horizon_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            required_label_purge_bars_for(-1)


class TestBuildFoldsFromSplitBinding:
    def test_expanding_dispatch(self) -> None:
        binding = SplitBinding(strategy="expanding_walk_forward", params={"n_splits": 3, "test_size": 100})
        plan = build_folds_from_split_binding(binding, _timestamps(), label_horizon_bars=0)
        assert plan.strategy == "expanding_walk_forward"
        assert len(plan.folds) == 3

    def test_rolling_dispatch(self) -> None:
        binding = SplitBinding(strategy="rolling_walk_forward", params={"n_splits": 3, "test_size": 100, "max_train_size": 200})
        plan = build_folds_from_split_binding(binding, _timestamps(), label_horizon_bars=0)
        assert plan.strategy == "rolling_walk_forward"

    def test_blocked_dispatch(self) -> None:
        binding = SplitBinding(strategy="blocked_time_split", params={"n_blocks": 5})
        plan = build_folds_from_split_binding(binding, _timestamps(), label_horizon_bars=0)
        assert plan.strategy == "blocked_time_split"

    def test_purge_embargo_validation_fraction_params_applied(self) -> None:
        binding = SplitBinding(
            strategy="expanding_walk_forward",
            params={"n_splits": 2, "test_size": 100, "purge_bars": 5, "embargo_bars": 3, "validation_fraction": 0.1},
        )
        plan = build_folds_from_split_binding(binding, _timestamps(), label_horizon_bars=5)
        assert plan.purge_bars == 5
        assert plan.embargo_bars == 3
        assert any(len(f.validation_indices) > 0 for f in plan.folds)

    def test_unsupported_strategy_raises(self) -> None:
        binding = SplitBinding(strategy="chronological", params={})
        with pytest.raises(ValueError, match="Unsupported"):
            build_folds_from_split_binding(binding, _timestamps(), label_horizon_bars=0)

    def test_missing_required_param_raises(self) -> None:
        binding = SplitBinding(strategy="expanding_walk_forward", params={"n_splits": 3})
        with pytest.raises(ValueError, match="test_size"):
            build_folds_from_split_binding(binding, _timestamps(), label_horizon_bars=0)

    def test_label_horizon_bars_param_in_split_binding_is_ignored(self) -> None:
        """`label_horizon_bars` is deliberately NOT a `SplitBinding.params`
        key -- proves a caller cannot smuggle a different label horizon
        in through `params` to override the explicit, manifest-derived
        `label_horizon_bars` argument. `SplitBinding.params` accepts
        arbitrary JSON-primitive extra keys without rejecting them, so
        this key is simply inert, never read by the dispatcher."""
        binding = SplitBinding(
            strategy="expanding_walk_forward",
            params={"n_splits": 2, "test_size": 100, "purge_bars": 5, "label_horizon_bars": 999},
        )
        plan = build_folds_from_split_binding(binding, _timestamps(), label_horizon_bars=5)
        assert plan.label_horizon_bars == 5
        assert plan.required_label_purge_bars == 5


class TestReconstructDatasetTimeline:
    def test_reconstructs_single_split(self, tmp_path) -> None:
        timeline = make_timeline(200)
        manifest, research_store, _ = write_synthetic_research_dataset(tmp_path, timeline=timeline)
        reconstructed = reconstruct_dataset_timeline(
            research_store, dataset_id=manifest.dataset_id, content_id=manifest.content_id,
        )
        assert len(reconstructed) == 200
        assert list(reconstructed["open_time"]) == list(timeline["open_time"])

    def test_reconstructs_multiple_splits_sorted(self, tmp_path) -> None:
        from quant_platform.features.manifests import ResearchDatasetStore

        full = make_timeline(300)
        store = ResearchDatasetStore(tmp_path / "research")
        # Deliberately write "test" (later rows) before "train" (earlier
        # rows) to prove reconstruction sorts by timestamp, not by
        # whatever order splits happen to be stored/iterated in.
        content_id, _ = store.write_artifacts(
            "multi_split", splits={"test": full.iloc[200:].reset_index(drop=True), "train": full.iloc[:200].reset_index(drop=True)},
            preprocessing_json={},
        )
        reconstructed = reconstruct_dataset_timeline(store, dataset_id="multi_split", content_id=content_id)
        assert list(reconstructed["open_time"]) == list(full["open_time"])

    def test_missing_content_raises(self, tmp_path) -> None:
        from quant_platform.features.manifests import ResearchDatasetStore

        store = ResearchDatasetStore(tmp_path / "research")
        with pytest.raises(FoldValidationError, match="No stored content"):
            reconstruct_dataset_timeline(store, dataset_id="nope", content_id="a" * 64)

    def test_duplicate_timestamps_across_splits_raise(self, tmp_path) -> None:
        from quant_platform.features.manifests import ResearchDatasetStore

        full = make_timeline(100)
        store = ResearchDatasetStore(tmp_path / "research")
        content_id, _ = store.write_artifacts(
            "dup_split", splits={"a": full, "b": full}, preprocessing_json={},  # identical rows, genuine duplicates
        )
        with pytest.raises(FoldValidationError, match="duplicate"):
            reconstruct_dataset_timeline(store, dataset_id="dup_split", content_id=content_id)

    def test_null_timestamp_raises(self, tmp_path) -> None:
        from quant_platform.features.manifests import ResearchDatasetStore

        broken = make_timeline(20)
        broken.loc[5, "open_time"] = pd.NaT
        store = ResearchDatasetStore(tmp_path / "research")
        content_id, _ = store.write_artifacts("broken", splits={"train": broken}, preprocessing_json={})
        with pytest.raises(FoldValidationError, match="null"):
            reconstruct_dataset_timeline(store, dataset_id="broken", content_id=content_id)


class TestSmallHelpers:
    def test_fold_row_counts(self) -> None:
        plan = generate_expanding_folds(_timestamps(), n_splits=2, test_size=100, validation_fraction=0.1, label_horizon_bars=0)
        for fold in plan.folds:
            train, validation, test = fold_row_counts(fold)
            assert train == len(fold.train_indices)
            assert validation == len(fold.validation_indices)
            assert test == len(fold.test_indices)

    def test_iter_fold_bounds(self) -> None:
        plan = generate_expanding_folds(_timestamps(), n_splits=3, test_size=100, label_horizon_bars=0)
        bounds = iter_fold_bounds(plan)
        assert len(bounds) == 3
        for (idx, train_start, train_end, test_start, test_end), fold in zip(bounds, plan.folds, strict=True):
            assert idx == fold.fold_index
            assert train_start == fold.train_start
            assert train_end == fold.train_end
            assert test_start == fold.test_start
            assert test_end == fold.test_end
