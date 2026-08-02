"""`FeatureDependencyGraph` (Milestone 11, Phase 2, Part 2): raw source /
market data -> derived feature -> higher-order feature, built PURELY
from an already-captured `FeatureRegistrySnapshot`'s own `metadata`
(for `feature_dependencies`) and `lineages` (for `required_inputs`) --
never re-reading a live `FeatureRegistry`. This decouples graph
verification from requiring a live registry object at all: a
persisted, previously-captured snapshot is everything this module
needs, which is exactly what `infra_verification.py`'s independent
re-derivation and `infra_reconciliation.py`'s two-graph comparison
require.

Detects cycles (an independent DFS reimplementation of `features.
registry.FeatureRegistry.resolve_dependency_order`'s own algorithm --
deliberately NOT calling that method, since a live registry may not be
available when reconciling two PERSISTED snapshots, and because
independently re-deriving is the whole point of verification), missing
parents (a declared `feature_dependencies` name with no corresponding
feature node), duplicate derivations (a documented, narrow heuristic --
see `_recipe_signature`), and orphan features (a metadata entry outside
the caller-supplied `declared_feature_names` set that nothing else
depends on -- by construction, `registry_snapshot.py`'s own capture
never produces one; this only fires against a hand-modified/tampered
snapshot)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from quant_platform.feature_discovery.registry_snapshot import FeatureRegistrySnapshot
from quant_platform.ml.persistence import as_json_dict, as_json_list, require_schema_version

__all__ = [
    "FEATURE_DEPENDENCY_GRAPH_SCHEMA_VERSION",
    "FeatureDependencyGraph",
    "FeatureGraphEdge",
    "FeatureGraphNode",
    "FeatureGraphNodeKind",
    "build_feature_dependency_graph",
]

FEATURE_DEPENDENCY_GRAPH_SCHEMA_VERSION = 1


class FeatureGraphNodeKind(Enum):
    RAW_SOURCE = "raw_source"
    MARKET_DATA = "market_data"
    DERIVED_FEATURE = "derived_feature"
    HIGHER_ORDER_FEATURE = "higher_order_feature"


_MARKET_DATA_GROUPS = frozenset({"macro", "cross_asset"})


@dataclass(frozen=True, slots=True)
class FeatureGraphNode:
    node_id: str
    kind: FeatureGraphNodeKind
    label: str

    def to_json_dict(self) -> dict[str, object]:
        return {"node_id": self.node_id, "kind": self.kind.value, "label": self.label}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureGraphNode:
        return cls(node_id=str(raw["node_id"]), kind=FeatureGraphNodeKind(raw["kind"]), label=str(raw["label"]))


@dataclass(frozen=True, slots=True)
class FeatureGraphEdge:
    source: str
    """The node_id that must be available BEFORE `target` -- the dependency."""
    target: str
    """The node_id that depends on `source` -- the dependent."""

    def to_json_dict(self) -> dict[str, object]:
        return {"source": self.source, "target": self.target}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureGraphEdge:
        return cls(source=str(raw["source"]), target=str(raw["target"]))


def _recipe_signature(feature_group: str, source_timeframe: str, required_inputs: tuple[str, ...], dependencies: tuple[str, ...]) -> tuple[object, ...]:
    """A narrow, documented duplicate-derivation heuristic: two features
    sharing this signature have the same feature group, source
    timeframe, raw inputs, and feature dependencies -- everything this
    module's own inputs (`FeatureMetadata`/`FeatureLineage`) expose that
    is recipe-relevant. This does NOT inspect `deterministic_params`
    (not carried by either of those two types) -- two features sharing
    a signature but genuinely differing only in a parameter value are a
    false positive this heuristic accepts, disclosed here rather than
    silently overclaimed precision."""
    return (feature_group, source_timeframe, tuple(sorted(required_inputs)), tuple(sorted(dependencies)))


@dataclass(frozen=True, slots=True)
class FeatureDependencyGraph:
    schema_version: int
    dataset_id: str
    nodes: tuple[FeatureGraphNode, ...]
    edges: tuple[FeatureGraphEdge, ...]
    cycles: tuple[tuple[str, ...], ...]
    missing_parents: tuple[tuple[str, str], ...]
    """`(feature_name, missing_dependency_name)` pairs."""
    duplicate_derivations: tuple[tuple[str, str], ...]
    """`(feature_name_a, feature_name_b)` pairs sharing a `_recipe_signature`."""
    orphan_features: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not (self.cycles or self.missing_parents)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "dataset_id": self.dataset_id, "nodes": [n.to_json_dict() for n in self.nodes],
            "edges": [e.to_json_dict() for e in self.edges], "cycles": [list(c) for c in self.cycles],
            "missing_parents": [list(p) for p in self.missing_parents], "duplicate_derivations": [list(p) for p in self.duplicate_derivations],
            "orphan_features": list(self.orphan_features),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> FeatureDependencyGraph:
        require_schema_version(raw, supported=FEATURE_DEPENDENCY_GRAPH_SCHEMA_VERSION, context="FeatureDependencyGraph")
        return cls(
            schema_version=FEATURE_DEPENDENCY_GRAPH_SCHEMA_VERSION, dataset_id=str(raw["dataset_id"]),
            nodes=tuple(
                FeatureGraphNode.from_json_dict(as_json_dict(n, field_name="nodes[]"))
                for n in as_json_list(raw.get("nodes") or [], field_name="nodes")
            ),
            edges=tuple(
                FeatureGraphEdge.from_json_dict(as_json_dict(e, field_name="edges[]"))
                for e in as_json_list(raw.get("edges") or [], field_name="edges")
            ),
            cycles=tuple(tuple(str(n) for n in as_json_list(c, field_name="cycles[]")) for c in as_json_list(raw.get("cycles") or [], field_name="cycles")),
            missing_parents=tuple(
                (str(p[0]), str(p[1])) for p in as_json_list(raw.get("missing_parents") or [], field_name="missing_parents")
            ),
            duplicate_derivations=tuple(
                (str(p[0]), str(p[1])) for p in as_json_list(raw.get("duplicate_derivations") or [], field_name="duplicate_derivations")
            ),
            orphan_features=tuple(str(s) for s in as_json_list(raw.get("orphan_features") or [], field_name="orphan_features")),
        )


def _detect_cycles(dependencies_by_name: dict[str, tuple[str, ...]]) -> tuple[tuple[str, ...], ...]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycles: list[tuple[str, ...]] = []

    def visit(name: str, chain: tuple[str, ...]) -> None:
        if name in visited or name not in dependencies_by_name:
            return
        if name in visiting:
            cycle_start = chain.index(name)
            cycles.append((*chain[cycle_start:], name))
            return
        visiting.add(name)
        for dep in dependencies_by_name[name]:
            visit(dep, (*chain, name))
        visiting.discard(name)
        visited.add(name)

    for feature_name in sorted(dependencies_by_name):
        visit(feature_name, ())
    return tuple(cycles)


def build_feature_dependency_graph(snapshot: FeatureRegistrySnapshot, *, declared_feature_names: frozenset[str] | None = None) -> FeatureDependencyGraph:
    declared = declared_feature_names if declared_feature_names is not None else frozenset(m.feature_name for m in snapshot.metadata)
    lineage_by_name = {ln.feature_name: ln for ln in snapshot.lineages}

    nodes: list[FeatureGraphNode] = []
    edges: list[FeatureGraphEdge] = []
    dependencies_by_name: dict[str, tuple[str, ...]] = {}
    input_node_kind: dict[str, FeatureGraphNodeKind] = {}
    referenced_by_something: set[str] = set()

    for entry in sorted(snapshot.metadata, key=lambda m: m.feature_name):
        feature_kind = FeatureGraphNodeKind.HIGHER_ORDER_FEATURE if entry.dependencies else FeatureGraphNodeKind.DERIVED_FEATURE
        nodes.append(FeatureGraphNode(node_id=entry.feature_id, kind=feature_kind, label=entry.feature_name))
        dependencies_by_name[entry.feature_name] = entry.dependencies

        lineage = lineage_by_name.get(entry.feature_name)
        required_inputs = lineage.required_inputs if lineage is not None else ()
        input_kind = FeatureGraphNodeKind.MARKET_DATA if entry.feature_group in _MARKET_DATA_GROUPS else FeatureGraphNodeKind.RAW_SOURCE
        for input_name in required_inputs:
            input_node_kind[input_name] = input_kind
            edges.append(FeatureGraphEdge(source=f"input:{input_name}", target=entry.feature_id))

        for dep_name in entry.dependencies:
            referenced_by_something.add(dep_name)

    node_id_by_name = {n.label: n.node_id for n in nodes if n.kind in (FeatureGraphNodeKind.DERIVED_FEATURE, FeatureGraphNodeKind.HIGHER_ORDER_FEATURE)}
    for input_name, kind in sorted(input_node_kind.items()):
        nodes.insert(0, FeatureGraphNode(node_id=f"input:{input_name}", kind=kind, label=input_name))

    missing_parents: list[tuple[str, str]] = []
    for entry in snapshot.metadata:
        for dep_name in entry.dependencies:
            if dep_name not in node_id_by_name:
                missing_parents.append((entry.feature_name, dep_name))
            else:
                edges.append(FeatureGraphEdge(source=node_id_by_name[dep_name], target=entry.feature_id))

    cycles = _detect_cycles(dependencies_by_name)

    orphan_features = tuple(
        sorted(m.feature_name for m in snapshot.metadata if m.feature_name not in declared and m.feature_name not in referenced_by_something)
    )

    signatures: dict[tuple[object, ...], list[str]] = {}
    for entry in snapshot.metadata:
        lineage = lineage_by_name.get(entry.feature_name)
        required_inputs = lineage.required_inputs if lineage is not None else ()
        signature = _recipe_signature(entry.feature_group, lineage.source_timeframe if lineage else "", required_inputs, entry.dependencies)
        signatures.setdefault(signature, []).append(entry.feature_name)
    duplicate_derivations: list[tuple[str, str]] = []
    for names in signatures.values():
        ordered = sorted(names)
        for i, name_a in enumerate(ordered):
            for name_b in ordered[i + 1 :]:
                duplicate_derivations.append((name_a, name_b))

    return FeatureDependencyGraph(
        schema_version=FEATURE_DEPENDENCY_GRAPH_SCHEMA_VERSION, dataset_id=snapshot.dataset_id, nodes=tuple(nodes), edges=tuple(edges),
        cycles=cycles, missing_parents=tuple(missing_parents), duplicate_derivations=tuple(duplicate_derivations), orphan_features=orphan_features,
    )
