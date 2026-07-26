from __future__ import annotations

from pathlib import Path

from tests.unit.execution.conftest import (
    build_registry,
    make_experiment_spec_kwargs,
    write_synthetic_research_dataset,
)

from quant_platform.execution.reporting import build_execution_report_json, render_execution_report_markdown
from quant_platform.execution.results import AggregatedExecutionResult
from quant_platform.execution.runner import ExecutionRunner
from quant_platform.execution.timeline import Timeline
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.models import ArtifactCategory


def _completed_execution(tmp_path: Path):
    dataset_manifest, research_store, research_manifest_store = write_synthetic_research_dataset(tmp_path)
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
    )
    spec = ExperimentSpec(**make_experiment_spec_kwargs(dataset_manifest=dataset_manifest))
    manifest = preparer.prepare(spec)
    runner = ExecutionRunner(
        ml_artifacts_root=tmp_path / "ml", model_registry=build_registry(), research_manifest_store=research_manifest_store,
        research_dataset_store=research_store,
    )
    runner.run(manifest.identity.experiment_id)
    exec_manifest = runner.execution_manifest_store.load(manifest.identity.experiment_id)
    return runner, exec_manifest


def _load_aggregate_and_timeline(runner: ExecutionRunner, exec_manifest):
    import json

    aggregate = None
    timeline = None
    for ref in exec_manifest.artifact_references:
        raw = runner.artifact_store.read_artifact(ref.content_hash)
        if ref.category is ArtifactCategory.EXECUTION_SUMMARY:
            aggregate = AggregatedExecutionResult.from_json_dict(json.loads(raw.decode("utf-8")))
        elif ref.category is ArtifactCategory.TIMELINE:
            timeline = Timeline.from_json_dict(json.loads(raw.decode("utf-8")))
    return aggregate, timeline


class TestBuildExecutionReportJson:
    def test_contains_expected_top_level_fields(self, tmp_path: Path) -> None:
        _runner, exec_manifest = _completed_execution(tmp_path)
        report = build_execution_report_json(exec_manifest)
        for key in (
            "schema_version", "experiment_id", "stage", "fold_plan_strategy", "total_folds",
            "completed_fold_indices", "failed_fold_indices", "resume_count", "created_at", "updated_at",
            "completed_at", "failure_summary", "artifact_references", "aggregate", "timeline", "limitations",
        ):
            assert key in report

    def test_without_aggregate_or_timeline_fields_are_none(self, tmp_path: Path) -> None:
        _runner, exec_manifest = _completed_execution(tmp_path)
        report = build_execution_report_json(exec_manifest)
        assert report["aggregate"] is None
        assert report["timeline"] is None

    def test_with_aggregate_and_timeline_expands_details(self, tmp_path: Path) -> None:
        runner, exec_manifest = _completed_execution(tmp_path)
        aggregate, timeline = _load_aggregate_and_timeline(runner, exec_manifest)
        report = build_execution_report_json(exec_manifest, aggregate=aggregate, timeline=timeline)
        assert report["aggregate"]["overall_status"] == "completed"
        assert len(report["timeline"]) == 3

    def test_deterministic_across_calls(self, tmp_path: Path) -> None:
        from quant_platform.ml.persistence import canonical_json_bytes

        _runner, exec_manifest = _completed_execution(tmp_path)
        b1 = canonical_json_bytes(build_execution_report_json(exec_manifest))
        b2 = canonical_json_bytes(build_execution_report_json(exec_manifest))
        assert b1 == b2


class TestRenderExecutionReportMarkdown:
    def test_contains_key_sections(self, tmp_path: Path) -> None:
        _runner, exec_manifest = _completed_execution(tmp_path)
        markdown = render_execution_report_markdown(exec_manifest)
        for heading in ("# Execution Report", "## Status History", "## Resume History", "## Duration", "## Timeline", "## Limitations"):
            assert heading in markdown

    def test_experiment_id_present(self, tmp_path: Path) -> None:
        _runner, exec_manifest = _completed_execution(tmp_path)
        markdown = render_execution_report_markdown(exec_manifest)
        assert exec_manifest.experiment_id in markdown

    def test_without_data_loaded_notes_not_expanded(self, tmp_path: Path) -> None:
        _runner, exec_manifest = _completed_execution(tmp_path)
        markdown = render_execution_report_markdown(exec_manifest)
        assert "not loaded" in markdown

    def test_with_timeline_shows_fold_table(self, tmp_path: Path) -> None:
        runner, exec_manifest = _completed_execution(tmp_path)
        aggregate, timeline = _load_aggregate_and_timeline(runner, exec_manifest)
        markdown = render_execution_report_markdown(exec_manifest, aggregate=aggregate, timeline=timeline)
        assert "| Fold | Train | Test | Status |" in markdown
        assert "completed" in markdown
