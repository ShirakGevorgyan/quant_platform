"""Regression test for `quant_platform.execution`'s package-level import
surface -- mirrors `tests/unit/ml/test_ml_package_init.py` exactly, for
the identical reason: this package went through several iterations of
new modules during development, and a future refactor that breaks the
explicit public re-export surface should fail loudly here."""

from __future__ import annotations

import quant_platform.execution as execution


def test_package_exposes_version_constant() -> None:
    assert isinstance(execution.EXECUTION_ENGINE_VERSION, str)
    assert execution.EXECUTION_ENGINE_VERSION


def test_package_all_symbols_are_importable_and_match_all() -> None:
    for name in execution.__all__:
        assert hasattr(execution, name), f"{name!r} listed in __all__ but not actually an attribute of quant_platform.execution"


def test_key_orchestrator_and_store_classes_importable_from_package_root() -> None:
    from quant_platform.execution import (
        DeterministicFoldExecutor,
        ExecutionManifestStore,
        ExecutionRunner,
        FoldExecutor,
    )

    assert ExecutionRunner is not None
    assert ExecutionManifestStore is not None
    assert FoldExecutor is not None
    assert DeterministicFoldExecutor is not None
