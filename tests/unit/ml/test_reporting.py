from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.ml.conftest import build_registry, make_dataset_manifest, make_experiment_spec_kwargs

from quant_platform.features.manifests import ResearchManifestStore
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.manifests import ExperimentManifest
from quant_platform.ml.models import ValidationReport
from quant_platform.ml.persistence import canonical_json_bytes, parse_json_strict
from quant_platform.ml.reporting import build_report_json, render_report_markdown


@pytest.fixture
def prepared_manifest(tmp_path: Path) -> tuple[ExperimentManifest, ExperimentPreparer]:
    research_store = ResearchManifestStore(tmp_path / "research")
    research_store.save(make_dataset_manifest())
    preparer = ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=build_registry(),
        research_manifest_store=research_store,
    )
    spec = ExperimentSpec(**make_experiment_spec_kwargs())
    manifest = preparer.prepare(spec)
    return manifest, preparer


def _load_validation_report(manifest: ExperimentManifest, preparer: ExperimentPreparer) -> ValidationReport:
    assert manifest.validation_report_reference is not None
    raw = preparer.artifact_store.read_artifact(manifest.validation_report_reference.content_hash)
    return ValidationReport.from_json_dict(parse_json_strict(raw.decode("utf-8")))


class TestBuildReportJson:
    def test_contains_expected_top_level_fields(self, prepared_manifest: tuple[ExperimentManifest, ExperimentPreparer]) -> None:
        manifest, _ = prepared_manifest
        report = build_report_json(manifest)
        for key in (
            "schema_version", "experiment_id", "status", "dataset", "model", "objective", "feature_count",
            "label", "split", "preprocessing", "seed_fingerprint", "code_revision", "environment_summary",
            "validation", "created_at", "completed_at", "failure_summary", "artifact_references", "limitations",
        ):
            assert key in report

    def test_experiment_id_matches_manifest(self, prepared_manifest: tuple[ExperimentManifest, ExperimentPreparer]) -> None:
        manifest, _ = prepared_manifest
        report = build_report_json(manifest)
        assert report["experiment_id"] == manifest.identity.experiment_id

    def test_without_validation_report_fields_are_none(self, prepared_manifest: tuple[ExperimentManifest, ExperimentPreparer]) -> None:
        manifest, _ = prepared_manifest
        report = build_report_json(manifest)
        assert report["validation"]["is_ready"] is None  # type: ignore[index]
        assert report["validation"]["performed"] is True  # type: ignore[index]

    def test_with_validation_report_expands_details(self, prepared_manifest: tuple[ExperimentManifest, ExperimentPreparer]) -> None:
        manifest, preparer = prepared_manifest
        validation_report = _load_validation_report(manifest, preparer)
        report = build_report_json(manifest, validation_report=validation_report)
        assert report["validation"]["is_ready"] is True  # type: ignore[index]
        assert report["validation"]["critical_count"] == 0  # type: ignore[index]

    def test_deterministic_across_calls(self, prepared_manifest: tuple[ExperimentManifest, ExperimentPreparer]) -> None:
        manifest, _ = prepared_manifest
        b1 = canonical_json_bytes(build_report_json(manifest))
        b2 = canonical_json_bytes(build_report_json(manifest))
        assert b1 == b2

    def test_always_includes_standard_limitation_notice(self, prepared_manifest: tuple[ExperimentManifest, ExperimentPreparer]) -> None:
        manifest, _ = prepared_manifest
        report = build_report_json(manifest)
        assert any("No model was fitted" in limitation for limitation in report["limitations"])  # type: ignore[union-attr]


class TestRenderReportMarkdown:
    def test_contains_key_sections(self, prepared_manifest: tuple[ExperimentManifest, ExperimentPreparer]) -> None:
        manifest, _ = prepared_manifest
        markdown = render_report_markdown(manifest)
        for heading in ("# Experiment Preparation Report", "## Dataset", "## Model", "## Reproducibility", "## Validation", "## Limitations"):
            assert heading in markdown

    def test_experiment_id_present(self, prepared_manifest: tuple[ExperimentManifest, ExperimentPreparer]) -> None:
        manifest, _ = prepared_manifest
        markdown = render_report_markdown(manifest)
        assert manifest.identity.experiment_id in markdown

    def test_without_validation_report_notes_not_expanded(self, prepared_manifest: tuple[ExperimentManifest, ExperimentPreparer]) -> None:
        manifest, _ = prepared_manifest
        markdown = render_report_markdown(manifest)
        assert "not loaded" in markdown

    def test_with_validation_report_shows_ready_status(self, prepared_manifest: tuple[ExperimentManifest, ExperimentPreparer]) -> None:
        manifest, preparer = prepared_manifest
        validation_report = _load_validation_report(manifest, preparer)
        markdown = render_report_markdown(manifest, validation_report=validation_report)
        assert "is_ready: True" in markdown

    def test_failed_manifest_shows_completed_at_and_failure_summary(self, tmp_path: Path) -> None:
        from tests.unit.ml.conftest import FEATURE_REGISTRY_FINGERPRINT

        from quant_platform.ml.models import ExperimentStatus, FeatureBinding

        research_store = ResearchManifestStore(tmp_path / "research")
        research_store.save(make_dataset_manifest())
        preparer = ExperimentPreparer(
            ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=build_registry(),
            research_manifest_store=research_store,
        )
        bad_spec = ExperimentSpec(**make_experiment_spec_kwargs(feature_binding=FeatureBinding(
            feature_names=("rsi_14", "atr_14"), feature_versions={"atr_14": "1", "rsi_14": "1"},
            feature_registry_fingerprint=FEATURE_REGISTRY_FINGERPRINT,
        )))
        manifest = preparer.prepare(bad_spec)
        assert manifest.status is ExperimentStatus.FAILED

        markdown = render_report_markdown(manifest)
        assert "**Completed at:**" in markdown
        assert "## Failure Summary" in markdown

    def test_validation_report_with_warnings_lists_them(self, prepared_manifest: tuple[ExperimentManifest, ExperimentPreparer]) -> None:
        from quant_platform.ml.models import ValidationIssue, ValidationSeverity

        manifest, _ = prepared_manifest
        report_with_warning = ValidationReport(
            schema_version=1,
            issues=(ValidationIssue(severity=ValidationSeverity.WARNING, code="test_warning", message="a warning message"),),
            generated_at="2024-01-01T00:00:00+00:00",
        )
        markdown = render_report_markdown(manifest, validation_report=report_with_warning)
        assert "warning: [test_warning] a warning message" in markdown
