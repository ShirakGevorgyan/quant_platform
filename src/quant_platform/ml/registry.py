"""Deterministic model-definition registry -- stores model DEFINITIONS and
FACTORIES, never fitted model instances (Milestone 4A trains nothing).
Mirrors `features.registry.FeatureRegistry`'s exact conventions: append-only
registration, duplicate `(name, version)` rejection, deterministic listing,
and a registry fingerprint that is independent of registration order,
dictionary insertion order, process ID, memory addresses, or wall-clock
time -- it is a pure function of each registered `ModelDefinition`'s own
DECLARED (JSON-serializable) metadata, never the factory object's identity
or `repr()`.

No dependency-cycle detection here: unlike `FeatureRegistry` (where one
feature's `compute` can legitimately read another feature's output),
nothing in this milestone lets one `ModelDefinition` depend on another --
adding cycle-detection machinery for a graph that cannot exist would be
unjustified complexity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from quant_platform.core.exceptions import DuplicateModelDefinitionError, UnknownModelDefinitionError
from quant_platform.ml.interfaces import ModelFactory
from quant_platform.ml.models import ModelCapabilities, ObjectiveType
from quant_platform.ml.persistence import canonical_json_bytes, sha256_hex_bytes


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    name: str
    version: str
    description: str
    capabilities: ModelCapabilities
    factory: ModelFactory
    """Excluded from `to_json_dict`/`fingerprint` -- a factory is a Python
    object with no stable, portable content representation; only its
    DECLARED metadata (name/version/capabilities/serializer_id) is
    identity-relevant, exactly like `features.registry.FeatureDefinition`
    excluding its `compute` closure from `FeatureSpec.fingerprint()`."""
    serializer_id: str
    """Identifies which `ModelSerializer`/`ModelDeserializer` pair this
    model's fitted instances are compatible with (e.g.
    `"constant_test_model_json_v1"`) -- resolved by a caller-provided
    lookup, never instantiated by the registry itself."""
    deprecated: bool = False
    deprecation_note: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ModelDefinition.name must not be empty")
        if not self.version:
            raise ValueError("ModelDefinition.version must not be empty")
        if not self.serializer_id:
            raise ValueError("ModelDefinition.serializer_id must not be empty")
        if self.deprecated and not self.deprecation_note:
            raise ValueError("ModelDefinition.deprecation_note is required when deprecated=True")
        if not self.deprecated and self.deprecation_note is not None:
            raise ValueError("ModelDefinition.deprecation_note must be None when deprecated=False")

    @property
    def qualified_name(self) -> str:
        return f"{self.name}@{self.version}"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "version": self.version, "description": self.description,
            "capabilities": self.capabilities.to_json_dict(), "serializer_id": self.serializer_id,
            "deprecated": self.deprecated, "deprecation_note": self.deprecation_note,
        }

    def fingerprint(self) -> str:
        return sha256_hex_bytes(canonical_json_bytes(self.to_json_dict()))


class ModelRegistry:
    def __init__(self) -> None:
        self._definitions: dict[tuple[str, str], ModelDefinition] = {}
        self._latest_version: dict[str, str] = {}

    def register(self, definition: ModelDefinition) -> None:
        key = (definition.name, definition.version)
        if key in self._definitions:
            raise DuplicateModelDefinitionError(
                f"Model {definition.name!r} version {definition.version!r} is already registered",
                context={"name": definition.name, "version": definition.version},
            )
        self._definitions[key] = definition
        self._latest_version[definition.name] = definition.version

    def get(self, name: str, version: str | None = None) -> ModelDefinition:
        resolved_version = version if version is not None else self._latest_version.get(name)
        if resolved_version is None:
            raise UnknownModelDefinitionError(f"No model named {name!r} is registered", context={"name": name})
        key = (name, resolved_version)
        if key not in self._definitions:
            raise UnknownModelDefinitionError(
                f"Model {name!r} version {resolved_version!r} is not registered",
                context={"name": name, "version": resolved_version},
            )
        return self._definitions[key]

    def list_definitions(self, *, include_deprecated: bool = True) -> list[ModelDefinition]:
        definitions = list(self._definitions.values())
        if not include_deprecated:
            definitions = [d for d in definitions if not d.deprecated]
        return sorted(definitions, key=lambda d: (d.name, d.version))

    def supported_objectives(self, name: str, version: str | None = None) -> tuple[ObjectiveType, ...]:
        return self.get(name, version).capabilities.supported_objectives

    def fingerprint(self, selection: Sequence[tuple[str, str]] | None = None) -> str:
        """Deterministic sha256 over the sorted, JSON-serialized selected
        `ModelDefinition`s (or every registered definition, if `selection`
        is `None`). Changes if and only if a materially relevant field of
        a selected definition changes -- never due to registration order,
        dict insertion order, process ID, memory addresses, or wall-clock
        time (none of those ever appear in `to_json_dict`)."""
        if selection is None:
            keys = sorted(self._definitions.keys())
        else:
            keys = sorted(selection)
            for key in keys:
                if key not in self._definitions:
                    raise UnknownModelDefinitionError(
                        f"Model {key[0]!r} version {key[1]!r} is not registered", context={"name": key[0], "version": key[1]}
                    )
        payload = {"models": [self._definitions[key].to_json_dict() for key in keys]}
        return sha256_hex_bytes(canonical_json_bytes(payload))

    def __len__(self) -> int:
        return len(self._definitions)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, tuple) and len(key) == 2:
            return key in self._definitions
        return isinstance(key, str) and key in self._latest_version


__all__ = ["ModelDefinition", "ModelRegistry"]
