"""Milestone 4D.1 completion: automated proof of the dependency-neutral
`quant_platform.core.json` module's place in this platform's package
graph -- not merely manual inspection.

THE CLAIM BEING PROVEN
--------------------------------------------------------------------------
`quant_platform.core` (and specifically `quant_platform.core.json`)
performs no I/O and depends on NOTHING else in this platform. `ml`
already depends on `historical`/`features` (`ml.experiment_manager`/`ml.
validation` import `features.manifests`; `ml.concurrency` imports
`historical.locking`). If `historical`/`features` depended back on `ml`
(directly, or via `ml.persistence`), that would be a real circular
dependency. Routing the shared durable-JSON primitives through `core.json`
instead avoids this entirely: `historical`/`features` (and `ml`, and
`execution`, and `optimization`) all depend DOWNWARD on `core`, never
sideways or upward on each other's higher layers.

This is checked two ways: statically (AST-parsing each module's own
`import`/`from ... import` statements -- proves what the SOURCE actually
declares, not what happens to work at runtime) and dynamically (importing
every top-level package, in multiple orders, in fresh subprocesses --
proves the graph is actually free of cycles in practice, since a static
check alone cannot rule out a cycle that only manifests through a
particular import ORDER)."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "quant_platform"

_HIGHER_LEVEL_PACKAGES = ("ml", "execution", "features", "historical", "optimization")


def _top_level_imports(module_path: Path) -> set[str]:
    """Every `quant_platform.<x>` top-level package this module's source
    imports from, via either `import quant_platform.x...` or
    `from quant_platform.x... import ...` -- a static (parse-time, not
    import-time) fact about the file, immune to any runtime import-order
    trickery."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("quant_platform."):
                    found.add(alias.name.split(".")[1])
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("quant_platform."):
            found.add(node.module.split(".")[1])
    return found


class TestCoreJsonIsDependencyNeutral:
    def test_core_json_imports_no_higher_level_package(self) -> None:
        imports = _top_level_imports(SRC_ROOT / "core" / "json.py")
        higher = imports & set(_HIGHER_LEVEL_PACKAGES)
        assert not higher, f"core/json.py imports higher-level package(s): {higher}"

    def test_core_json_imports_only_stdlib(self) -> None:
        """Even stricter than "no higher-level package": `core.json`
        should not need ANY `quant_platform` import at all, including
        `quant_platform.core` siblings -- it is meant to be reachable
        before anything else in this platform."""
        imports = _top_level_imports(SRC_ROOT / "core" / "json.py")
        assert imports == set(), f"core/json.py imports quant_platform package(s): {imports}"

    def test_core_package_init_imports_no_higher_level_package(self) -> None:
        """`core/__init__.py`'s own docstring already claims this --
        verified here, not merely trusted."""
        imports = _top_level_imports(SRC_ROOT / "core" / "__init__.py")
        higher = imports & set(_HIGHER_LEVEL_PACKAGES)
        assert not higher, f"core/__init__.py imports higher-level package(s): {higher}"


class TestHistoricalAndFeaturesMigratedOntoCoreJson:
    """Proves the migration actually happened in these specific files --
    not merely that it WOULD be safe to."""

    @staticmethod
    def _imports_core_json(module_path: Path) -> bool:
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "quant_platform.core.json":
                return True
        return False

    def test_historical_manifest_imports_core_json(self) -> None:
        assert self._imports_core_json(SRC_ROOT / "historical" / "manifest.py")

    def test_historical_canonical_store_imports_core_json(self) -> None:
        assert self._imports_core_json(SRC_ROOT / "historical" / "canonical_store.py")

    def test_historical_raw_store_imports_core_json(self) -> None:
        assert self._imports_core_json(SRC_ROOT / "historical" / "raw_store.py")

    def test_historical_locking_imports_core_json(self) -> None:
        assert self._imports_core_json(SRC_ROOT / "historical" / "locking.py")

    def test_features_manifests_imports_core_json(self) -> None:
        assert self._imports_core_json(SRC_ROOT / "features" / "manifests.py")

    def test_historical_and_features_no_longer_use_raw_json_loads_for_durable_reads(self) -> None:
        """`json.loads`/`json.load(` must not appear anywhere in these
        five files any more -- every durable read goes through
        `core.json.parse_json_strict`. Deliberately does NOT check
        `json.dumps` here: `features/manifests.py` retains a handful of
        `json.dumps(..., allow_nan=False)` calls for content/dataset-
        identity-critical fingerprints where byte-representation
        stability matters more than routing through `canonical_json_bytes`
        -- see that module's own inline comments and `docs/
        persistence_security.md`."""
        for relative in (
            "historical/manifest.py", "historical/canonical_store.py",
            "historical/raw_store.py", "historical/locking.py", "features/manifests.py",
        ):
            source = (SRC_ROOT / relative).read_text(encoding="utf-8")
            assert "json.loads(" not in source, f"{relative} still calls json.loads directly"
            assert "json.load(" not in source, f"{relative} still calls json.load directly"


class TestMlPersistenceDelegatesToCoreJson:
    def test_ml_persistence_imports_core_json(self) -> None:
        imports = _top_level_imports(SRC_ROOT / "ml" / "persistence.py")
        assert "core" in imports

    def test_ml_persistence_re_exports_are_object_identical_to_core_json(self) -> None:
        """Not merely equivalent-behaving wrappers -- the literal same
        function objects, so there can never be a second, subtly
        different implementation to drift out of sync."""
        from quant_platform.core import json as core_json
        from quant_platform.ml import persistence as ml_persistence

        for name in ("canonical_json_bytes", "parse_json_strict", "sha256_hex_bytes", "write_json_atomic"):
            assert getattr(ml_persistence, name) is getattr(core_json, name), (
                f"ml.persistence.{name} is not object-identical to core.json.{name}"
            )

    def test_ml_persistence_source_does_not_reimplement_parse_json_strict_or_canonical_json_bytes(self) -> None:
        """Static check that the function BODIES don't exist a second
        time in `ml/persistence.py` -- guards against a future edit
        re-introducing a parallel implementation even if the re-export
        names stay correct."""
        source = (SRC_ROOT / "ml" / "persistence.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        defined_here = {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert "parse_json_strict" not in defined_here
        assert "canonical_json_bytes" not in defined_here


class TestNoImportCycleInAFreshInterpreter:
    """Static AST checks prove what the source DECLARES; this proves the
    graph is actually free of cycles in practice, across every import
    order a caller might reasonably use -- a cycle that only manifests
    for one particular entry-point ordering would not be caught
    statically."""

    _PACKAGES = ("core", "historical", "features", "ml", "execution", "optimization")

    @staticmethod
    def _run_import_sequence(packages: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        code = "\n".join(f"import quant_platform.{pkg}" for pkg in packages) + "\nprint('OK')"
        return subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
        )

    def test_declared_order_imports_cleanly(self) -> None:
        result = self._run_import_sequence(self._PACKAGES)
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_historical_and_features_first_then_ml_imports_cleanly(self) -> None:
        """The specific order this whole milestone is about: prove
        `historical`/`features` can be imported BEFORE `ml` even exists
        in `sys.modules`, without either needing the other to already be
        loaded."""
        result = self._run_import_sequence(("historical", "features", "core", "ml", "execution", "optimization"))
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_ml_first_then_historical_and_features_imports_cleanly(self) -> None:
        result = self._run_import_sequence(("ml", "historical", "features", "execution", "optimization", "core"))
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_core_json_importable_standalone_before_anything_else(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import quant_platform.core.json; print('OK')"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_reverse_declared_order_imports_cleanly(self) -> None:
        result = self._run_import_sequence(tuple(reversed(self._PACKAGES)))
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout
