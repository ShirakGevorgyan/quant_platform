"""Milestone 11 Phase 1, Part 2 quality gate: repeat qualification x10,
repeat replay x10, repeat verification x10, repeat diagnostics x10 --
each operation run 10 times against the real pipeline, asserting every
repetition (after stripping the legitimately wall-clock-dependent
`generated_at` fields) is byte-identical to the first."""

from __future__ import annotations

from conftest import build_request, trend_registry

from quant_platform.features.dataset_builder import ResearchDatasetBuilder
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.qualification.diagnostics import compute_diagnostics
from quant_platform.qualification.engine import DatasetQualificationEngine
from quant_platform.qualification.verification import QualificationIndependentVerifier

_REPEAT_COUNT = 10


def _strip_generated_at(raw: dict) -> dict:
    raw = dict(raw)
    raw.pop("generated_at", None)
    if isinstance(raw.get("decision"), dict):
        raw["decision"] = {k: v for k, v in raw["decision"].items() if k != "generated_at"}
    return raw


class TestRepeatQualification:
    def test_ten_repeated_qualify_calls_are_identical(self, qualified_manifest, research_store) -> None:
        engine = DatasetQualificationEngine()
        reports = [
            _strip_generated_at(engine.qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"})).to_json_dict())
            for _ in range(_REPEAT_COUNT)
        ]
        assert all(report == reports[0] for report in reports)


class TestRepeatReplay:
    def test_ten_repeated_rebuilds_are_identical(self, tmp_path, seeded_loader) -> None:
        research_store = ResearchDatasetStore(tmp_path / "research")
        manifest_store = ResearchManifestStore(tmp_path / "research")
        builder = ResearchDatasetBuilder(historical_loader=seeded_loader, registry=trend_registry(), research_store=research_store, manifest_store=manifest_store)

        manifests = [builder.build(build_request()) for _ in range(_REPEAT_COUNT)]
        assert all(m.dataset_id == manifests[0].dataset_id for m in manifests)
        assert all(m.content_id == manifests[0].content_id for m in manifests)
        assert all(m.version == manifests[0].version for m in manifests)

        engine = DatasetQualificationEngine()
        reports = [
            _strip_generated_at(engine.qualify(m, research_store, required_feature_names=frozenset({"trend"})).to_json_dict())
            for m in manifests
        ]
        assert all(report == reports[0] for report in reports)


class TestRepeatVerification:
    def test_ten_repeated_independent_verifications_are_identical(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        verifier = QualificationIndependentVerifier()
        results = [
            _strip_generated_at(verifier.verify(report, qualified_manifest, research_store, required_feature_names=frozenset({"trend"})).to_json_dict())
            for _ in range(_REPEAT_COUNT)
        ]
        for result in results:
            result["reconciliation"] = {k: v for k, v in result["reconciliation"].items() if k != "generated_at"}
        assert all(result == results[0] for result in results)
        assert results[0]["verified"] is True


class TestRepeatDiagnostics:
    def test_ten_repeated_diagnostics_computations_are_identical(self, qualified_manifest, research_store) -> None:
        report = DatasetQualificationEngine().qualify(qualified_manifest, research_store, required_feature_names=frozenset({"trend"}))
        diagnostics_runs = [
            _strip_generated_at(compute_diagnostics(qualified_manifest, report, research_store, required_feature_names=frozenset({"trend"})).to_json_dict())
            for _ in range(_REPEAT_COUNT)
        ]
        assert all(d == diagnostics_runs[0] for d in diagnostics_runs)
