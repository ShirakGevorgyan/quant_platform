from __future__ import annotations

import pytest

from quant_platform.core.exceptions import FeatureDiscoveryRequestError
from quant_platform.feature_discovery.engine import FeatureDiscoveryEngine
from quant_platform.feature_discovery.evidence import FEATURE_DISCOVERY_DIMENSION_ORDER
from quant_platform.feature_discovery.models import FeatureDiscoveryReport


class TestFeatureDiscoveryEngine:
    def test_discovers_every_declared_feature_by_default(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        assert report.feature_count == 3
        assert {d.feature_name for d in report.per_feature_diagnostics} == {"trend", "const", "trend_copy"}

    def test_dimension_scores_cover_every_dimension(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        assert set(report.dimension_scores) == {d.value for d in FEATURE_DISCOVERY_DIMENSION_ORDER}

    def test_redundancy_warning_surfaces_at_dataset_level(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        assert any("redundan" in w.lower() for w in report.warnings)

    def test_recommendations_are_deduplicated_and_non_empty(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        assert len(report.recommendations) == len(set(report.recommendations))
        assert len(report.recommendations) > 0

    def test_subset_request_evaluates_only_requested_features(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store, feature_names=frozenset({"trend"}))
        assert report.feature_count == 1
        assert report.per_feature_diagnostics[0].feature_name == "trend"

    def test_unknown_feature_name_raises(self, discovered_manifest, research_store) -> None:
        with pytest.raises(FeatureDiscoveryRequestError):
            FeatureDiscoveryEngine().discover(discovered_manifest, research_store, feature_names=frozenset({"does_not_exist"}))

    def test_no_blocking_findings_for_a_clean_dataset(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        assert report.blocking_findings == ()
        assert report.summary.blocked_count == 0

    def test_json_round_trip(self, discovered_manifest, research_store) -> None:
        report = FeatureDiscoveryEngine().discover(discovered_manifest, research_store)
        assert FeatureDiscoveryReport.from_json_dict(report.to_json_dict()) == report

    def test_repeated_calls_are_deterministic(self, discovered_manifest, research_store) -> None:
        engine = FeatureDiscoveryEngine()
        report1 = engine.discover(discovered_manifest, research_store)
        report2 = engine.discover(discovered_manifest, research_store)
        raw1, raw2 = report1.to_json_dict(), report2.to_json_dict()
        raw1.pop("evaluation_time"), raw2.pop("evaluation_time")
        assert raw1 == raw2
