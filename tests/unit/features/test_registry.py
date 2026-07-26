from __future__ import annotations

import pytest

from quant_platform.core.exceptions import (
    CyclicFeatureDependencyError,
    DuplicateFeatureError,
    UnknownFeatureError,
)
from quant_platform.core.types import Timeframe
from quant_platform.features.interfaces import FeatureDefinition
from quant_platform.features.models import FeatureCategory, FeatureSpec
from quant_platform.features.registry import FeatureRegistry


def _definition(name: str, *, version: str = "1", deps: tuple[str, ...] = ()) -> FeatureDefinition:
    spec = FeatureSpec(
        name=name, version=version, description="d", category=FeatureCategory.PRICE, required_inputs=(),
        source_symbols=(), source_timeframe=Timeframe.M1, output_dtype="float64", lookback_bars=0, warmup_bars=0,
        feature_dependencies=deps,
    )
    return FeatureDefinition(spec=spec, compute=lambda ctx: ctx.base_df["close"])


class TestRegistration:
    def test_register_and_get(self) -> None:
        registry = FeatureRegistry()
        registry.register(_definition("a"))
        assert registry.get("a").name == "a"

    def test_duplicate_registration_rejected(self) -> None:
        registry = FeatureRegistry()
        registry.register(_definition("a"))
        with pytest.raises(DuplicateFeatureError):
            registry.register(_definition("a"))

    def test_same_name_new_version_allowed(self) -> None:
        registry = FeatureRegistry()
        registry.register(_definition("a", version="1"))
        registry.register(_definition("a", version="2"))
        assert registry.get("a").version == "2"
        assert registry.get("a", version="1").version == "1"

    def test_get_unknown_feature_raises(self) -> None:
        registry = FeatureRegistry()
        with pytest.raises(UnknownFeatureError):
            registry.get("missing")

    def test_get_unknown_version_raises(self) -> None:
        registry = FeatureRegistry()
        registry.register(_definition("a", version="1"))
        with pytest.raises(UnknownFeatureError):
            registry.get("a", version="99")

    def test_registering_dependency_on_unregistered_feature_rejected(self) -> None:
        registry = FeatureRegistry()
        with pytest.raises(UnknownFeatureError):
            registry.register(_definition("b", deps=("a",)))

    def test_list_features_filters_by_category(self) -> None:
        registry = FeatureRegistry()
        registry.register(_definition("a"))
        specs = registry.list_features(category=FeatureCategory.PRICE)
        assert len(specs) == 1
        assert registry.list_features(category=FeatureCategory.MACRO) == []

    def test_len_and_contains(self) -> None:
        registry = FeatureRegistry()
        registry.register(_definition("a"))
        assert len(registry) == 1
        assert "a" in registry
        assert "b" not in registry


class TestDependencyResolution:
    def test_simple_chain_order(self) -> None:
        registry = FeatureRegistry()
        registry.register(_definition("a"))
        registry.register(_definition("b", deps=("a",)))
        registry.register(_definition("c", deps=("b",)))
        order = registry.resolve_dependency_order(["c"])
        assert order == ["a", "b", "c"]

    def test_diamond_dependency_each_computed_once(self) -> None:
        registry = FeatureRegistry()
        registry.register(_definition("a"))
        registry.register(_definition("b", deps=("a",)))
        registry.register(_definition("c", deps=("a",)))
        registry.register(_definition("d", deps=("b", "c")))
        order = registry.resolve_dependency_order(["d"])
        assert order.count("a") == 1
        assert order.index("a") < order.index("b") < order.index("d")
        assert order.index("a") < order.index("c") < order.index("d")

    def test_cycle_created_via_version_bump_is_detected(self) -> None:
        """Registration itself requires dependencies to pre-exist, but a
        cycle is still reachable: register A (no deps), B depends on A,
        then register a NEW version of A that depends on B. Since
        dependencies resolve by NAME to whatever is latest, this creates
        A(v2) -> B -> A(v2) -- straight back to itself."""
        registry = FeatureRegistry()
        registry.register(_definition("a", version="1"))
        registry.register(_definition("b", version="1", deps=("a",)))
        registry.register(_definition("a", version="2", deps=("b",)))
        with pytest.raises(CyclicFeatureDependencyError):
            registry.resolve_dependency_order(["a"])

    def test_resolve_unknown_feature_raises(self) -> None:
        registry = FeatureRegistry()
        with pytest.raises(UnknownFeatureError):
            registry.resolve_dependency_order(["nope"])


class TestFingerprint:
    def test_deterministic_across_registries(self) -> None:
        r1 = FeatureRegistry()
        r1.register(_definition("a"))
        r1.register(_definition("b", deps=("a",)))

        r2 = FeatureRegistry()
        r2.register(_definition("a"))
        r2.register(_definition("b", deps=("a",)))

        assert r1.fingerprint() == r2.fingerprint()

    def test_fingerprint_changes_when_a_feature_changes(self) -> None:
        r1 = FeatureRegistry()
        r1.register(_definition("a"))
        fp1 = r1.fingerprint()

        r2 = FeatureRegistry()
        r2.register(_definition("a", version="2"))
        fp2 = r2.fingerprint()
        assert fp1 != fp2

    def test_fingerprint_scoped_to_requested_names_and_their_dependencies(self) -> None:
        registry = FeatureRegistry()
        registry.register(_definition("a"))
        registry.register(_definition("b"))
        fp_a_only = registry.fingerprint(["a"])
        fp_both = registry.fingerprint(["a", "b"])
        assert fp_a_only != fp_both

    def test_fingerprint_independent_of_registration_order_for_same_selection(self) -> None:
        r1 = FeatureRegistry()
        r1.register(_definition("a"))
        r1.register(_definition("b"))

        r2 = FeatureRegistry()
        r2.register(_definition("b"))
        r2.register(_definition("a"))

        assert r1.fingerprint(["a", "b"]) == r2.fingerprint(["a", "b"])
