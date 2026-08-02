"""`FeatureHealthReport`, `FeatureUsageReport`, `FeatureGroupReport`
(Milestone 11, Phase 2, Part 2) -- no predictive metric anywhere.

`FeatureHealthReport` REUSES Part 1's `FeatureSignalDiagnostics`
(`diagnostics.compute_feature_signal_diagnostics`) directly rather than
re-implementing constant/near-constant/missing/warmup/coverage/
availability/determinism/reproducibility(identity) checks a second
time -- Part 1's own 10 dimensions already cover 9 of this task's 10
named health checks. The 10th, "lineage", was a real gap: `diagnostics.
SharedDiscoveryFacts.lineage_present`/`.missing_lineage_fields` were
already computed by Part 1 (via `qualification.verifier.verify_lineage`)
but never surfaced anywhere in any of the 10 dimension evaluators --
this module is where that fact is finally exposed, not a second
lineage check invented from scratch.

`FeatureUsageReport` and `FeatureGroupReport` are built from
`graph.py`'s `FeatureDependencyGraph` and `catalog.py`'s
`FeatureCatalog` respectively -- pure aggregation, no new statistics."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.feature_discovery.catalog import FeatureCatalog
from quant_platform.feature_discovery.diagnostics import (
    SharedDiscoveryFacts,
    compute_feature_signal_diagnostics,
)
from quant_platform.feature_discovery.graph import FeatureDependencyGraph
from quant_platform.feature_discovery.models import FeatureSignalDiagnostics
from quant_platform.historical.quality import Severity
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = [
    "FEATURE_GROUP_REPORT_SCHEMA_VERSION",
    "FEATURE_HEALTH_REPORT_SCHEMA_VERSION",
    "FEATURE_USAGE_REPORT_SCHEMA_VERSION",
    "FeatureGroupReport",
    "FeatureHealthReport",
    "FeatureUsageReport",
    "compute_feature_group_reports",
    "compute_feature_health",
    "compute_feature_usage",
]

FEATURE_HEALTH_REPORT_SCHEMA_VERSION = 1
FEATURE_USAGE_REPORT_SCHEMA_VERSION = 1
FEATURE_GROUP_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FeatureHealthReport:
    schema_version: int
    feature_name: str
    is_healthy: bool
    """`True` iff `signal_diagnostics` has no blocking evidence, no WARNING/CRITICAL evidence, AND
    lineage is present."""
    signal_diagnostics: FeatureSignalDiagnostics
    lineage_present: bool
    missing_lineage_fields: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "feature_name": self.feature_name, "is_healthy": self.is_healthy,
            "signal_diagnostics": self.signal_diagnostics.to_json_dict(), "lineage_present": self.lineage_present,
            "missing_lineage_fields": list(self.missing_lineage_fields),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureHealthReport:
        require_schema_version(raw, supported=FEATURE_HEALTH_REPORT_SCHEMA_VERSION, context="FeatureHealthReport")
        return cls(
            schema_version=FEATURE_HEALTH_REPORT_SCHEMA_VERSION, feature_name=str(raw["feature_name"]), is_healthy=bool(raw["is_healthy"]),
            signal_diagnostics=FeatureSignalDiagnostics.from_json_dict(as_json_dict(raw["signal_diagnostics"], field_name="signal_diagnostics")),
            lineage_present=bool(raw["lineage_present"]),
            missing_lineage_fields=tuple(str(s) for s in as_json_list(raw.get("missing_lineage_fields") or [], field_name="missing_lineage_fields")),
        )


def compute_feature_health(feature_name: str, facts: SharedDiscoveryFacts) -> FeatureHealthReport:
    diagnostics = compute_feature_signal_diagnostics(feature_name, facts)
    has_warning_or_worse = any(e.severity in (Severity.WARNING, Severity.CRITICAL) for e in diagnostics.all_evidence)
    is_healthy = not diagnostics.is_blocking and not has_warning_or_worse and facts.lineage_present
    return FeatureHealthReport(
        schema_version=FEATURE_HEALTH_REPORT_SCHEMA_VERSION, feature_name=feature_name, is_healthy=is_healthy, signal_diagnostics=diagnostics,
        lineage_present=facts.lineage_present, missing_lineage_fields=facts.missing_lineage_fields,
    )


@dataclass(frozen=True, slots=True)
class FeatureUsageReport:
    schema_version: int
    dataset_id: str
    feature_name: str
    depends_on: tuple[str, ...]
    """Other FEATURE names this feature declares as `feature_dependencies`."""
    depended_on_by: tuple[str, ...]
    """Other FEATURE names that declare THIS feature as a dependency -- the reverse edge."""
    required_input_count: int
    is_root: bool
    """No feature dependencies -- computed directly from raw/market-data inputs."""
    is_leaf: bool
    """Nothing else in this dataset's graph depends on it."""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "feature_name": self.feature_name,
            "depends_on": list(self.depends_on), "depended_on_by": list(self.depended_on_by),
            "required_input_count": self.required_input_count, "is_root": self.is_root, "is_leaf": self.is_leaf,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureUsageReport:
        require_schema_version(raw, supported=FEATURE_USAGE_REPORT_SCHEMA_VERSION, context="FeatureUsageReport")
        return cls(
            schema_version=FEATURE_USAGE_REPORT_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]), feature_name=str(raw["feature_name"]),
            depends_on=tuple(str(s) for s in as_json_list(raw.get("depends_on") or [], field_name="depends_on")),
            depended_on_by=tuple(str(s) for s in as_json_list(raw.get("depended_on_by") or [], field_name="depended_on_by")),
            required_input_count=int(str(raw["required_input_count"])), is_root=bool(raw["is_root"]), is_leaf=bool(raw["is_leaf"]),
        )


def compute_feature_usage(feature_name: str, graph: FeatureDependencyGraph) -> FeatureUsageReport:
    node = next((n for n in graph.nodes if n.label == feature_name), None)
    node_id = node.node_id if node is not None else feature_name
    depends_on = tuple(sorted(e.source.split("@")[0] for e in graph.edges if e.target == node_id and not e.source.startswith("input:")))
    depended_on_by = tuple(sorted(e.target.split("@")[0] for e in graph.edges if e.source == node_id))
    required_input_count = sum(1 for e in graph.edges if e.target == node_id and e.source.startswith("input:"))
    return FeatureUsageReport(
        schema_version=FEATURE_USAGE_REPORT_SCHEMA_VERSION, dataset_id=graph.dataset_id, feature_name=feature_name, depends_on=depends_on,
        depended_on_by=depended_on_by, required_input_count=required_input_count, is_root=not depends_on, is_leaf=not depended_on_by,
    )


@dataclass(frozen=True, slots=True)
class FeatureGroupReport:
    schema_version: int
    dataset_id: str
    feature_group: str
    feature_names: tuple[str, ...]
    healthy_count: int
    unhealthy_count: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "feature_group": self.feature_group,
            "feature_names": list(self.feature_names), "healthy_count": self.healthy_count, "unhealthy_count": self.unhealthy_count,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureGroupReport:
        require_schema_version(raw, supported=FEATURE_GROUP_REPORT_SCHEMA_VERSION, context="FeatureGroupReport")
        return cls(
            schema_version=FEATURE_GROUP_REPORT_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]), feature_group=str(raw["feature_group"]),
            feature_names=tuple(str(s) for s in as_json_list(raw.get("feature_names") or [], field_name="feature_names")),
            healthy_count=int(str(raw["healthy_count"])), unhealthy_count=int(str(raw["unhealthy_count"])),
        )


def compute_feature_group_reports(catalog: FeatureCatalog, health_reports: tuple[FeatureHealthReport, ...]) -> tuple[FeatureGroupReport, ...]:
    health_by_name = {h.feature_name: h for h in health_reports}
    groups: dict[str, list[str]] = {}
    for entry in catalog.entries:
        groups.setdefault(entry.feature_group, []).append(entry.feature_name)

    reports = []
    for group, names in sorted(groups.items()):
        sorted_names = tuple(sorted(names))
        healthy = sum(1 for n in sorted_names if health_by_name.get(n) is not None and health_by_name[n].is_healthy)
        reports.append(FeatureGroupReport(
            schema_version=FEATURE_GROUP_REPORT_SCHEMA_VERSION, dataset_id=catalog.dataset_id, feature_group=group, feature_names=sorted_names,
            healthy_count=healthy, unhealthy_count=len(sorted_names) - healthy,
        ))
    return tuple(reports)
