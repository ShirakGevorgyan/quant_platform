"""Deterministic environment and code-revision capture for experiment
reproducibility.

TWO SEPARATE THINGS, ONE FILE
--------------------------------------------------------------------------
1. `capture_environment_snapshot()` -> `models.EnvironmentSnapshot`.
   INFORMATIONAL ONLY -- Python/OS/architecture/package versions, and an
   optional CPU core count (no external dependency is added purely to
   capture memory info; that is recorded as `None` rather than pulled in
   via a new third-party package). Never touches experiment identity (see
   `EnvironmentSnapshot`'s own docstring). Machine-specific data here must
   NOT cause identical scientific experiments on different machines to
   receive different IDs -- the only way environment can affect identity
   is `ExperimentSpec.environment_requirements`, an explicit DECLARATION a
   human writes into the spec, never a detected fact.

2. `capture_code_revision_binding()` -> `models.CodeRevisionBinding`.
   IDENTITY-RELEVANT -- reuses `historical.code_revision`'s git-or-content
   fallback philosophy (a repository with zero commits must still produce
   a usable, deterministic revision marker), but broadened to hash the
   entire `quant_platform` source tree (not just one subpackage) for the
   content fallback, since an ML experiment's behavior can depend on any
   part of the codebase. Also captures the git working-tree dirty flag
   (`git status --porcelain` non-empty), which the Milestone 2 helper does
   not expose.

Neither function ever executes an unsafe shell string -- `git` is invoked
via `subprocess.run` with an explicit argument list (never `shell=True`),
a short timeout, and `check=False` (failure is treated as "no git
available", not raised).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

from quant_platform.ml.models import CodeRevisionBinding, EnvironmentSnapshot
from quant_platform.ml.persistence import format_utc_timestamp, utc_now

_SCHEMA_VERSION = 1
_TRACKED_PACKAGES = (
    "numpy", "pandas", "pydantic", "pyarrow", "quant-platform",
    # Milestone 4C: the real predictive-model libraries -- tracked here
    # (informationally, never identity-relevant -- see module docstring)
    # so `ml.training_metadata.TrainingMetadata.library_version` can be
    # looked up from this SAME already-captured snapshot rather than a
    # second, parallel introspection mechanism.
    "lightgbm", "xgboost", "catboost", "scikit-learn", "scipy",
)
_GIT_TIMEOUT_SECONDS = 5


def capture_environment_snapshot() -> EnvironmentSnapshot:
    package_versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None

    return EnvironmentSnapshot(
        schema_version=_SCHEMA_VERSION,
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_release=platform.release(),
        architecture=platform.machine(),
        package_versions=package_versions,
        cpu_count=os.cpu_count(),
        captured_at=format_utc_timestamp(utc_now()),
    )


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=_GIT_TIMEOUT_SECONDS, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git_head_and_dirty(repo_dir: Path) -> tuple[str | None, bool | None]:
    head_result = _run_git(["rev-parse", "HEAD"], cwd=repo_dir)
    if head_result is None or head_result.returncode != 0:
        return None, None
    head = head_result.stdout.strip() or None
    if head is None:
        return None, None

    status_result = _run_git(["status", "--porcelain"], cwd=repo_dir)
    if status_result is None or status_result.returncode != 0:
        return head, None
    return head, bool(status_result.stdout.strip())


def _package_content_revision() -> str:
    """Deterministic content hash over every `.py` file in the
    `quant_platform` source tree, sorted by relative path -- broader than
    `historical.code_revision`'s own single-subpackage fallback, since an
    ML experiment can depend on any part of the codebase."""
    package_root = Path(__file__).resolve().parents[1]  # .../src/quant_platform
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root)
        digest.update(str(relative).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def capture_code_revision_binding(*, repo_dir: Path | str | None = None) -> CodeRevisionBinding:
    search_dir = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parents[3]
    git_head, is_dirty = _git_head_and_dirty(search_dir)
    if git_head is not None:
        return CodeRevisionBinding(revision=f"git:{git_head}", source="git", is_dirty=is_dirty)
    return CodeRevisionBinding(revision=f"content:{_package_content_revision()}", source="content", is_dirty=None)


__all__ = ["capture_code_revision_binding", "capture_environment_snapshot"]
