"""Regression test for `quant_platform.ml`'s package-level import surface
-- this package went a while without an `__init__.py` at all (relying on
implicit namespace-package behavior); this test exists so a future
refactor that breaks the explicit public re-export surface fails loudly
here rather than silently reverting to namespace-package behavior."""

from __future__ import annotations

import quant_platform.ml as ml


def test_package_exposes_version_constant() -> None:
    assert isinstance(ml.ML_INFRASTRUCTURE_VERSION, str)
    assert ml.ML_INFRASTRUCTURE_VERSION


def test_package_all_symbols_are_importable_and_match_all() -> None:
    for name in ml.__all__:
        assert hasattr(ml, name), f"{name!r} listed in __all__ but not actually an attribute of quant_platform.ml"


def test_testing_module_is_not_re_exported_at_package_level() -> None:
    """`ml.testing.ConstantTestModel` must stay scoped to `ml.testing` --
    never promoted into the package's main public surface, where it
    could be mistaken for a real predictive model."""
    assert "ConstantTestModel" not in ml.__all__
    assert not hasattr(ml, "ConstantTestModel")


def test_key_orchestrator_and_store_classes_importable_from_package_root() -> None:
    from quant_platform.ml import (
        ExperimentManifestStore,
        ExperimentPreparer,
        ExperimentSpec,
        MLArtifactStore,
        ModelRegistry,
    )

    assert ExperimentPreparer is not None
    assert MLArtifactStore is not None
    assert ExperimentManifestStore is not None
    assert ExperimentSpec is not None
    assert ModelRegistry is not None
