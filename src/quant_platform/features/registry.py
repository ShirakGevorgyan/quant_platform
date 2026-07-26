"""The feature registry: where feature definitions live, how their
dependency order is resolved, and how a set of registered features is
deterministically fingerprinted for a research dataset manifest.

Registration is append-only. `(name, version)` is a feature's identity --
re-registering the same pair is refused (`DuplicateFeatureError`); changing
a feature's behavior always means registering a NEW version, never
mutating an existing one in place. This is what makes "modifying feature
configuration changes the dataset ID" true: `fingerprint()` is derived
purely from the exact, immutable `FeatureSpec` objects selected, so any
real behavioral change is only reachable through a version bump that also
changes the fingerprint.

Dependency resolution: a feature may declare `feature_dependencies` (other
feature NAMES, resolved to whichever version is latest-registered for that
name at resolution time -- see `get`). `resolve_dependency_order` performs
a DFS-based topological sort over the closure of the requested names,
raising `CyclicFeatureDependencyError` if a cycle is found. A cycle is
reachable even though registration itself requires dependencies to already
exist, because dependencies are resolved by NAME, not by pinned version:
registering a newer version of an earlier feature can retroactively create
a cycle through a feature that depended on its older version (see
`tests/unit/features/test_registry.py` for the concrete adversarial
scenario this guards against).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from quant_platform.core.exceptions import (
    CyclicFeatureDependencyError,
    DuplicateFeatureError,
    UnknownFeatureError,
)
from quant_platform.features.interfaces import FeatureDefinition
from quant_platform.features.models import FeatureCategory, FeatureSpec


class FeatureRegistry:
    def __init__(self) -> None:
        self._definitions: dict[tuple[str, str], FeatureDefinition] = {}
        self._latest_version: dict[str, str] = {}

    def register(self, definition: FeatureDefinition) -> None:
        key = (definition.name, definition.version)
        if key in self._definitions:
            raise DuplicateFeatureError(
                f"Feature {definition.name!r} version {definition.version!r} is already registered",
                context={"name": definition.name, "version": definition.version},
            )
        for dep_name in definition.spec.feature_dependencies:
            if dep_name not in self._latest_version:
                raise UnknownFeatureError(
                    f"Feature {definition.name!r} declares a dependency on {dep_name!r}, "
                    "which is not registered yet -- register dependencies before dependents.",
                    context={"name": definition.name, "missing_dependency": dep_name},
                )
        self._definitions[key] = definition
        self._latest_version[definition.name] = definition.version

    def get(self, name: str, version: str | None = None) -> FeatureDefinition:
        resolved_version = version if version is not None else self._latest_version.get(name)
        if resolved_version is None:
            raise UnknownFeatureError(f"No feature named {name!r} is registered", context={"name": name})
        key = (name, resolved_version)
        if key not in self._definitions:
            raise UnknownFeatureError(
                f"Feature {name!r} version {resolved_version!r} is not registered",
                context={"name": name, "version": resolved_version},
            )
        return self._definitions[key]

    def list_features(self, category: FeatureCategory | None = None) -> list[FeatureSpec]:
        specs = [
            d.spec for d in self._definitions.values() if category is None or d.spec.category is category
        ]
        return sorted(specs, key=lambda s: (s.name, s.version))

    def resolve_dependency_order(self, names: Sequence[str]) -> list[str]:
        """Topological order (dependencies before dependents) of the full
        closure of `names` and their transitive `feature_dependencies`.
        Raises `CyclicFeatureDependencyError` on a cycle, `UnknownFeatureError`
        if any name (direct or transitive) is not registered."""
        visiting: set[str] = set()
        visited: set[str] = set()
        order: list[str] = []

        def visit(name: str, chain: tuple[str, ...]) -> None:
            if name in visited:
                return
            if name in visiting:
                raise CyclicFeatureDependencyError(
                    f"Cyclic feature dependency detected: {' -> '.join((*chain, name))}",
                    context={"cycle": [*chain, name]},
                )
            visiting.add(name)
            definition = self.get(name)
            for dep in definition.spec.feature_dependencies:
                visit(dep, (*chain, name))
            visiting.discard(name)
            visited.add(name)
            order.append(name)

        for requested in names:
            visit(requested, ())
        return order

    def fingerprint(self, names: Sequence[str] | None = None) -> str:
        """Deterministic sha256 over the sorted, JSON-serialized
        `FeatureSpec`s selected by `names` (or every registered feature, if
        `None`) -- independent of registration order or Python object
        identity. Two registries (even in different processes) with the
        same selected feature specs always produce the same fingerprint."""
        if names is None:
            keys = sorted(self._definitions.keys())
        else:
            resolved_names = self.resolve_dependency_order(names)
            keys = sorted((self.get(n).name, self.get(n).version) for n in resolved_names)
        payload = [self._definitions[key].spec.to_json_dict() for key in keys]
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self._definitions)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, tuple) and len(key) == 2:
            return key in self._definitions
        return isinstance(key, str) and key in self._latest_version


__all__ = ["FeatureRegistry"]
