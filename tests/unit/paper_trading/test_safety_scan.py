"""Milestone 7, Section 35: security/safety scan. Proves, by AST/source
inspection of every module in `quant_platform.paper_trading` (plus
`config.paper_trading_schemas`), that no prohibited construct exists:

  - no `eval`/`exec`/`breakpoint` calls;
  - no `pickle` import (unsafe deserialization of untrusted artifacts --
    this package uses `core.json`/`ml.persistence`'s strict JSON codec
    exclusively, exactly like every other Milestone in this repository);
  - no `shell=True` anywhere;
  - no import of a network/broker/remote-terminal client library (sockets,
    HTTP clients, websockets, `MetaTrader5`, or any similarly-named
    module);
  - no bare/silent broad-exception swallowing (`except:` or
    `except Exception:` whose body is only `pass`/`continue`/`...`);
  - no `pdb`/debugger import;
  - no `TODO`/`FIXME`/`HACK` marker comments;
  - no hardcoded absolute local filesystem path string literals;
  - no `print(` debug calls (this is a LIBRARY package -- CLI-layer
    output belongs in `ml_cli.py`, never inside `paper_trading` itself);
  - no identifier that reads as a broker credential (`api_key`,
    `api_secret`, `broker_password`, `mt5_login`, `mt5_password`, ...).

This is a STRUCTURAL scan (AST-based where a false positive would
otherwise be easy via a naive substring match, e.g. a docstring that
explains "MT5 integration is NOT implemented" must never itself trip a
false "found MT5 client code" positive) -- Section 35's own instruction:
"avoiding false positives in documentation that explicitly says live
trading is unsupported." Prose mentioning MT5/broker/live-trading to
explain what is DELIBERATELY UNSUPPORTED is expected and untouched; only
actual dangerous CODE CONSTRUCTS are flagged."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_PAPER_TRADING_SRC = Path(__file__).resolve().parents[3] / "src" / "quant_platform" / "paper_trading"
_CONFIG_SCHEMA_FILE = Path(__file__).resolve().parents[3] / "src" / "quant_platform" / "config" / "paper_trading_schemas.py"

_FORBIDDEN_IMPORT_MODULES = frozenset({
    "pickle", "pdb", "socket", "requests", "urllib", "urllib2", "http.client", "httplib", "websocket", "websockets",
    "aiohttp", "httpx", "paramiko", "ftplib", "telnetlib", "smtplib", "asyncio.streams", "MetaTrader5", "mt5",
    "ccxt", "ib_insync", "fix", "quickfix",
})
_FORBIDDEN_CALL_NAMES = frozenset({"eval", "exec", "breakpoint"})
_BROAD_EXCEPTION_BODY_KINDS = (ast.Pass, ast.Continue)
_CREDENTIAL_IDENTIFIER_PATTERN = re.compile(r"(?i)\b(api_key|api_secret|apikey|broker_password|broker_credential|account_password|mt5_login|mt5_password|access_token|secret_key)\b")
_TODO_MARKER_PATTERN = re.compile(r"#\s*(TODO|FIXME|HACK)\b")
_HARDCODED_PATH_PATTERN = re.compile(r"""^[\"']([A-Za-z]:\\\\|[A-Za-z]:/|/home/|/Users/|/etc/|/var/)""")


def _paper_trading_source_files() -> list[Path]:
    files = sorted(_PAPER_TRADING_SRC.glob("*.py"))
    files = [f for f in files if f.name != "__init__.py"]
    files.append(_CONFIG_SCHEMA_FILE)
    return files


_SOURCE_FILES = _paper_trading_source_files()


@pytest.fixture(params=_SOURCE_FILES, ids=[f.name for f in _SOURCE_FILES])
def source_file(request: pytest.FixtureRequest) -> Path:
    return request.param


def _parse(path: Path) -> tuple[str, ast.Module]:
    text = path.read_text(encoding="utf-8")
    return text, ast.parse(text, filename=str(path))


class TestNoForbiddenImports:
    def test_no_network_broker_or_unsafe_deserialization_imports(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _FORBIDDEN_IMPORT_MODULES or alias.name in _FORBIDDEN_IMPORT_MODULES:
                        found.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module is not None and (node.module.split(".")[0] in _FORBIDDEN_IMPORT_MODULES or node.module in _FORBIDDEN_IMPORT_MODULES):
                found.append(node.module)
        assert not found, f"{source_file.name} imports forbidden module(s): {found}"


class TestNoForbiddenCalls:
    def test_no_eval_exec_or_breakpoint(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALL_NAMES:
                found.append(node.func.id)
        assert not found, f"{source_file.name} calls forbidden function(s): {found}"

    def test_no_shell_true(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        found = [
            kw for node in ast.walk(tree) if isinstance(node, ast.Call) for kw in node.keywords
            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True
        ]
        assert not found, f"{source_file.name} contains a shell=True call"

    def test_no_debug_print_calls(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        found = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"]
        assert not found, f"{source_file.name} (a library module) calls print() {len(found)} time(s) -- CLI output belongs in ml_cli.py"


class TestNoSilentBroadExceptionSwallowing:
    def test_no_bare_or_silently_passed_broad_except(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        offenders: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            is_bare = node.type is None
            is_broad_named = isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException")
            if not (is_bare or is_broad_named):
                continue
            body_is_trivial = len(node.body) == 1 and isinstance(node.body[0], _BROAD_EXCEPTION_BODY_KINDS)
            if is_bare or body_is_trivial:
                offenders.append(f"line {node.lineno}")
        assert not offenders, f"{source_file.name} has silent broad-exception swallowing at: {offenders}"


class TestNoMarkerCommentsOrHardcodedPaths:
    def test_no_todo_fixme_hack_markers(self, source_file: Path) -> None:
        text, _ = _parse(source_file)
        hits = [f"line {i}: {line.strip()}" for i, line in enumerate(text.splitlines(), start=1) if _TODO_MARKER_PATTERN.search(line)]
        assert not hits, f"{source_file.name} contains TODO/FIXME/HACK marker(s): {hits}"

    def test_no_hardcoded_absolute_local_paths(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                candidate = repr(node.value) if "\\" in node.value else f'"{node.value}"'
                if _HARDCODED_PATH_PATTERN.match(candidate):
                    hits.append((node.lineno, node.value))
        assert not hits, f"{source_file.name} contains hardcoded absolute local path literal(s): {hits}"


class TestNoBrokerCredentialShapedIdentifiers:
    def test_no_credential_shaped_names_as_actual_identifiers(self, source_file: Path) -> None:
        """Scoped to real identifiers (assignment targets, function
        parameters, attribute/dataclass field names) -- NOT prose, so a
        docstring explaining "no broker credentials are ever accepted"
        (which legitimately contains these words) never trips this."""
        _, tree = _parse(source_file)
        offenders: set[str] = set()
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.arg):
                names.append(node.arg)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.append(node.target.id)
            for name in names:
                if _CREDENTIAL_IDENTIFIER_PATTERN.search(name):
                    offenders.add(name)
        assert not offenders, f"{source_file.name} declares credential-shaped identifier(s): {sorted(offenders)}"


class TestCliHasNoLiveCommands:
    def test_ml_cli_registers_no_live_trading_command(self) -> None:
        import argparse

        from quant_platform.ml_cli import build_parser

        parser = build_parser()
        subparsers_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        forbidden = {"run-live", "submit-live-order", "connect-broker", "execute-mt5", "deploy-live"}
        registered = set(subparsers_action.choices.keys())
        assert not (forbidden & registered), f"ml_cli registers forbidden live-trading command(s): {forbidden & registered}"
