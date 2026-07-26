from __future__ import annotations

import pytest

from quant_platform.core.exceptions import DuplicateModelDefinitionError, UnknownModelDefinitionError
from quant_platform.ml.models import ModelCapabilities, ObjectiveType
from quant_platform.ml.registry import ModelDefinition, ModelRegistry
from quant_platform.ml.testing import ConstantTestModelFactory


def _definition(*, name: str = "m", version: str = "1", deprecated: bool = False, deprecation_note: str | None = None) -> ModelDefinition:
    return ModelDefinition(
        name=name, version=version, description="test model",
        capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION,)),
        factory=ConstantTestModelFactory(), serializer_id="sid_v1",
        deprecated=deprecated, deprecation_note=deprecation_note,
    )


class TestModelDefinition:
    def test_qualified_name(self) -> None:
        assert _definition(name="m", version="2").qualified_name == "m@2"

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _definition(name="")

    def test_empty_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _definition(version="")

    def test_deprecated_requires_note(self) -> None:
        with pytest.raises(ValueError, match="deprecation_note is required"):
            _definition(deprecated=True)

    def test_non_deprecated_forbids_note(self) -> None:
        with pytest.raises(ValueError, match="must be None"):
            _definition(deprecated=False, deprecation_note="why")

    def test_deprecated_with_note_ok(self) -> None:
        d = _definition(deprecated=True, deprecation_note="superseded")
        assert d.deprecated
        assert d.deprecation_note == "superseded"

    def test_to_json_dict_excludes_factory(self) -> None:
        d = _definition()
        assert "factory" not in d.to_json_dict()

    def test_fingerprint_deterministic_and_independent_of_factory_identity(self) -> None:
        d1 = _definition()
        d2 = _definition()  # different factory instance, same declared metadata
        assert d1.fingerprint() == d2.fingerprint()

    def test_fingerprint_changes_with_capabilities(self) -> None:
        d1 = _definition()
        d2 = ModelDefinition(
            name="m", version="1", description="test model",
            capabilities=ModelCapabilities(supported_objectives=(ObjectiveType.REGRESSION, ObjectiveType.BINARY_CLASSIFICATION)),
            factory=ConstantTestModelFactory(), serializer_id="sid_v1",
        )
        assert d1.fingerprint() != d2.fingerprint()


class TestModelRegistry:
    def test_register_and_get(self) -> None:
        registry = ModelRegistry()
        registry.register(_definition())
        assert registry.get("m", "1").name == "m"

    def test_get_without_version_returns_latest_registered(self) -> None:
        registry = ModelRegistry()
        registry.register(_definition(version="1"))
        registry.register(_definition(version="2"))
        assert registry.get("m").version == "2"

    def test_duplicate_registration_rejected(self) -> None:
        registry = ModelRegistry()
        registry.register(_definition())
        with pytest.raises(DuplicateModelDefinitionError):
            registry.register(_definition())

    def test_unknown_model_raises(self) -> None:
        registry = ModelRegistry()
        with pytest.raises(UnknownModelDefinitionError):
            registry.get("does_not_exist")

    def test_unknown_version_raises(self) -> None:
        registry = ModelRegistry()
        registry.register(_definition(version="1"))
        with pytest.raises(UnknownModelDefinitionError):
            registry.get("m", "2")

    def test_list_definitions_sorted_deterministically(self) -> None:
        registry = ModelRegistry()
        registry.register(_definition(name="zeta"))
        registry.register(_definition(name="alpha"))
        names = [d.name for d in registry.list_definitions()]
        assert names == ["alpha", "zeta"]

    def test_list_definitions_excludes_deprecated_when_requested(self) -> None:
        registry = ModelRegistry()
        registry.register(_definition(name="active"))
        registry.register(_definition(name="old", deprecated=True, deprecation_note="gone"))
        names = [d.name for d in registry.list_definitions(include_deprecated=False)]
        assert names == ["active"]

    def test_supported_objectives(self) -> None:
        registry = ModelRegistry()
        registry.register(_definition())
        assert registry.supported_objectives("m", "1") == (ObjectiveType.REGRESSION,)

    def test_fingerprint_deterministic_across_independent_registries(self) -> None:
        r1, r2 = ModelRegistry(), ModelRegistry()
        r1.register(_definition())
        r2.register(_definition())
        assert r1.fingerprint() == r2.fingerprint()

    def test_fingerprint_independent_of_registration_order(self) -> None:
        r1, r2 = ModelRegistry(), ModelRegistry()
        r1.register(_definition(name="a"))
        r1.register(_definition(name="b"))
        r2.register(_definition(name="b"))
        r2.register(_definition(name="a"))
        assert r1.fingerprint() == r2.fingerprint()

    def test_fingerprint_changes_when_definition_changes(self) -> None:
        r1, r2 = ModelRegistry(), ModelRegistry()
        r1.register(_definition())
        r2.register(_definition(deprecated=True, deprecation_note="x"))
        assert r1.fingerprint() != r2.fingerprint()

    def test_fingerprint_selection_subset(self) -> None:
        registry = ModelRegistry()
        registry.register(_definition(name="a"))
        registry.register(_definition(name="b"))
        fp_all = registry.fingerprint()
        fp_a_only = registry.fingerprint(selection=[("a", "1")])
        assert fp_all != fp_a_only

    def test_fingerprint_unknown_selection_raises(self) -> None:
        registry = ModelRegistry()
        registry.register(_definition(name="a"))
        with pytest.raises(UnknownModelDefinitionError):
            registry.fingerprint(selection=[("nonexistent", "1")])

    def test_len_and_contains(self) -> None:
        registry = ModelRegistry()
        assert len(registry) == 0
        registry.register(_definition())
        assert len(registry) == 1
        assert ("m", "1") in registry
        assert "m" in registry
        assert "nonexistent" not in registry
