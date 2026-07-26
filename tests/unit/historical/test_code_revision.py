"""Tests for `historical.code_revision.capture_code_revision` -- recording
"what code produced this dataset" without requiring a Git commit to exist
(this repository itself has none, which is exactly the scenario this
module exists to handle)."""

from __future__ import annotations

import subprocess

from quant_platform.historical.code_revision import capture_code_revision


class TestCaptureCodeRevision:
    def test_returns_a_non_empty_string(self) -> None:
        revision = capture_code_revision()
        assert isinstance(revision, str)
        assert revision

    def test_is_deterministic_across_calls(self) -> None:
        assert capture_code_revision() == capture_code_revision()

    def test_uses_git_prefix_or_content_prefix(self) -> None:
        revision = capture_code_revision()
        assert revision.startswith("git:") or revision.startswith("content:")

    def test_falls_back_to_content_hash_when_no_git_repo_present(self, tmp_path) -> None:
        # `tmp_path` is guaranteed not to be inside a Git checkout with
        # commits (a fresh pytest temp dir), so this exercises exactly the
        # "no commits yet" path this repository itself is currently in.
        revision = capture_code_revision(repo_dir=tmp_path)
        assert revision.startswith("content:")

    def test_content_hash_fallback_does_not_depend_on_repo_dir_argument(self, tmp_path) -> None:
        # The content-hash fallback is computed from the package's own
        # source files, not from `repo_dir` (which is only used for the
        # Git lookup) -- so it must be identical regardless of which
        # (git-less) directory was passed.
        revision_a = capture_code_revision(repo_dir=tmp_path)
        another_dir = tmp_path / "nested"
        another_dir.mkdir()
        revision_b = capture_code_revision(repo_dir=another_dir)
        assert revision_a == revision_b

    def test_uses_git_head_when_a_real_commit_exists(self, tmp_path) -> None:
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True, check=True)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "test commit"], cwd=tmp_path, capture_output=True, check=True)

        revision = capture_code_revision(repo_dir=tmp_path)
        assert revision.startswith("git:")

        expected_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
        ).stdout.strip()
        assert revision == f"git:{expected_head}"
