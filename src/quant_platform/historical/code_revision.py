"""Capturing "what code produced this dataset" without requiring a Git
commit to exist.

`DatasetManifest.code_revision` is meant to answer "which version of the
pipeline's logic built this?" -- but this repository (like any freshly
started one) may have zero commits, and requiring one would make the
manifest's most basic provenance field silently absent for exactly the
common case of active local development. `capture_code_revision` always
returns *something* usable:

1. If a Git commit exists (`git rev-parse HEAD` succeeds), use it --
   prefixed `git:` so a reader knows how to interpret it (e.g. `git log
   <hash>`).
2. Otherwise, fall back to a deterministic content hash of every `.py`
   file under `quant_platform.historical` (sorted by path, so the result
   does not depend on filesystem iteration order) -- prefixed `content:`.
   This changes exactly when the pipeline's own logic changes, which is
   the property that actually matters for provenance; it does not require
   version control at all.

Neither path ever shells out further than a single, argument-free,
read-only `git rev-parse HEAD` -- no repository state is read or modified
(this module never runs `git add`/`git commit`/anything else).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def _git_head_revision(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    head = result.stdout.strip()
    return head or None


def _content_revision() -> str:
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def capture_code_revision(*, repo_dir: Path | str | None = None) -> str:
    """Best-effort, always-available "what code produced this" marker.
    `repo_dir` defaults to this file's own repository checkout; pass an
    explicit path when calling from an installed package outside a Git
    checkout entirely (in which case the Git lookup will simply fail and
    the content-hash fallback is used, exactly as when there are zero
    commits)."""
    search_dir = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parent
    git_revision = _git_head_revision(search_dir)
    if git_revision is not None:
        return f"git:{git_revision}"
    return f"content:{_content_revision()}"


__all__ = ["capture_code_revision"]
