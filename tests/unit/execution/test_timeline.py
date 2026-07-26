from __future__ import annotations

from tests.unit.execution.conftest import make_timeline

from quant_platform.execution.results import FoldResult, FoldStatus
from quant_platform.execution.splitters import generate_expanding_folds
from quant_platform.execution.timeline import Timeline, render_timeline_markdown
from quant_platform.ml.persistence import format_utc_timestamp, utc_now

_NOW = format_utc_timestamp(utc_now())
EID = "a" * 64


class TestTimelineFromFoldPlan:
    def test_one_entry_per_fold_in_order(self) -> None:
        timeline_df = make_timeline()
        plan = generate_expanding_folds(timeline_df["open_time"], n_splits=3, test_size=100, label_horizon_bars=0)
        timeline = Timeline.from_fold_plan(EID, plan)
        assert [e.fold_index for e in timeline.entries] == [0, 1, 2]
        assert all(e.status is None for e in timeline.entries)

    def test_round_trip(self) -> None:
        timeline_df = make_timeline()
        plan = generate_expanding_folds(timeline_df["open_time"], n_splits=2, test_size=100, label_horizon_bars=0)
        timeline = Timeline.from_fold_plan(EID, plan)
        assert Timeline.from_json_dict(timeline.to_json_dict()) == timeline


class TestTimelineFromFoldResults:
    def _result(self, fold_index: int, status: FoldStatus, **overrides: object) -> FoldResult:
        base: dict[str, object] = {
            "schema_version": 1, "fold_index": fold_index, "train_start": _NOW, "train_end": _NOW,
            "test_start": _NOW, "test_end": _NOW, "train_size": 10, "test_size": 5,
            "status": status, "duration_seconds": 0.1,
        }
        if status is FoldStatus.FAILED:
            base["failure_reason"] = "bad"
        base.update(overrides)
        return FoldResult(**base)  # type: ignore[arg-type]

    def test_entries_sorted_by_fold_index_regardless_of_input_order(self) -> None:
        results = [self._result(2, FoldStatus.COMPLETED), self._result(0, FoldStatus.COMPLETED), self._result(1, FoldStatus.FAILED)]
        timeline = Timeline.from_fold_results(EID, results)
        assert [e.fold_index for e in timeline.entries] == [0, 1, 2]
        assert timeline.entries[1].status == "failed"
        assert timeline.entries[0].status == "completed"

    def test_round_trip(self) -> None:
        results = [self._result(0, FoldStatus.COMPLETED)]
        timeline = Timeline.from_fold_results(EID, results)
        assert Timeline.from_json_dict(timeline.to_json_dict()) == timeline


class TestRenderTimelineMarkdown:
    def test_contains_experiment_id_and_table_header(self) -> None:
        timeline_df = make_timeline()
        plan = generate_expanding_folds(timeline_df["open_time"], n_splits=2, test_size=100, label_horizon_bars=0)
        timeline = Timeline.from_fold_plan(EID, plan)
        markdown = render_timeline_markdown(timeline)
        assert EID in markdown
        assert "| Fold | Train | Validation | Test | Status |" in markdown

    def test_validation_column_renders_dash_when_absent(self) -> None:
        timeline_df = make_timeline()
        plan = generate_expanding_folds(timeline_df["open_time"], n_splits=1, test_size=100, label_horizon_bars=0)
        timeline = Timeline.from_fold_plan(EID, plan)
        markdown = render_timeline_markdown(timeline)
        assert "| -" in markdown or " - " in markdown
