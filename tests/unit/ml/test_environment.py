from __future__ import annotations

import importlib.metadata
import subprocess
from pathlib import Path
from unittest.mock import patch

from quant_platform.ml.environment import (
    _git_head_and_dirty,
    _run_git,
    capture_code_revision_binding,
    capture_environment_snapshot,
)
from quant_platform.ml.models import EnvironmentSnapshot


class TestCaptureEnvironmentSnapshot:
    def test_returns_valid_snapshot(self) -> None:
        snapshot = capture_environment_snapshot()
        assert isinstance(snapshot, EnvironmentSnapshot)
        assert snapshot.python_version
        assert snapshot.platform_system
        assert snapshot.schema_version == 1

    def test_round_trip(self) -> None:
        snapshot = capture_environment_snapshot()
        assert EnvironmentSnapshot.from_json_dict(snapshot.to_json_dict()) == snapshot

    def test_deterministic_ordering_of_package_versions_keys(self) -> None:
        snapshot = capture_environment_snapshot()
        # to_json_dict always sorts keys -- verify no KeyError/ordering issue
        keys = list(snapshot.to_json_dict()["package_versions"].keys())  # type: ignore[union-attr]
        assert keys == sorted(keys)

    def test_captured_at_parses_as_utc(self) -> None:
        from quant_platform.ml.persistence import parse_utc_timestamp

        snapshot = capture_environment_snapshot()
        parse_utc_timestamp(snapshot.captured_at)  # must not raise

    def test_missing_tracked_package_recorded_as_none(self) -> None:
        def fake_version(name: str) -> str:
            if name == "numpy":
                raise importlib.metadata.PackageNotFoundError(name)
            return "1.0.0"

        with patch("quant_platform.ml.environment.importlib.metadata.version", side_effect=fake_version):
            snapshot = capture_environment_snapshot()
        assert snapshot.package_versions["numpy"] is None


class TestCaptureCodeRevisionBinding:
    def test_returns_git_or_content_source(self) -> None:
        binding = capture_code_revision_binding()
        assert binding.source in ("git", "content")
        assert binding.revision

    def test_deterministic_for_same_repo_state(self) -> None:
        b1 = capture_code_revision_binding()
        b2 = capture_code_revision_binding()
        assert b1.revision == b2.revision
        assert b1.source == b2.source

    def test_outside_git_repo_falls_back_to_content(self, tmp_path: Path) -> None:
        binding = capture_code_revision_binding(repo_dir=tmp_path)
        assert binding.source == "content"
        assert binding.revision.startswith("content:")
        assert binding.is_dirty is None

    def test_content_fallback_is_deterministic(self, tmp_path: Path) -> None:
        b1 = capture_code_revision_binding(repo_dir=tmp_path)
        b2 = capture_code_revision_binding(repo_dir=tmp_path)
        assert b1.revision == b2.revision

    def test_git_source_never_leaks_absolute_path_in_revision(self) -> None:
        binding = capture_code_revision_binding()
        if binding.source == "git":
            assert "\\" not in binding.revision
            assert "/" not in binding.revision.removeprefix("git:")


class TestRunGitFailureModes:
    def test_run_git_returns_none_on_oserror(self, tmp_path: Path) -> None:
        with patch("quant_platform.ml.environment.subprocess.run", side_effect=OSError("no such executable")):
            assert _run_git(["status"], cwd=tmp_path) is None

    def test_run_git_returns_none_on_subprocess_error(self, tmp_path: Path) -> None:
        with patch("quant_platform.ml.environment.subprocess.run", side_effect=subprocess.SubprocessError("boom")):
            assert _run_git(["status"], cwd=tmp_path) is None

    def test_git_head_and_dirty_returns_none_when_rev_parse_fails(self, tmp_path: Path) -> None:
        with patch("quant_platform.ml.environment._run_git", return_value=None):
            head, dirty = _git_head_and_dirty(tmp_path)
        assert head is None
        assert dirty is None

    def test_git_head_and_dirty_returns_none_when_rev_parse_nonzero(self, tmp_path: Path) -> None:
        failed = subprocess.CompletedProcess(args=["git"], returncode=128, stdout="", stderr="fatal: not a git repository")
        with patch("quant_platform.ml.environment._run_git", return_value=failed):
            head, dirty = _git_head_and_dirty(tmp_path)
        assert head is None
        assert dirty is None

    def test_git_head_and_dirty_head_is_none_when_stdout_empty(self, tmp_path: Path) -> None:
        empty = subprocess.CompletedProcess(args=["git"], returncode=0, stdout="   \n", stderr="")
        with patch("quant_platform.ml.environment._run_git", return_value=empty):
            head, dirty = _git_head_and_dirty(tmp_path)
        assert head is None
        assert dirty is None

    def test_git_head_and_dirty_status_command_fails_head_still_returned(self, tmp_path: Path) -> None:
        rev_parse_ok = subprocess.CompletedProcess(args=["git"], returncode=0, stdout="abc123\n", stderr="")

        def fake_run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str] | None:
            if args[0] == "rev-parse":
                return rev_parse_ok
            return None

        with patch("quant_platform.ml.environment._run_git", side_effect=fake_run_git):
            head, dirty = _git_head_and_dirty(tmp_path)
        assert head == "abc123"
        assert dirty is None
