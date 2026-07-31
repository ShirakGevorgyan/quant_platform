"""Milestone 8, Section 35: security/safety scan. Proves, by AST/source
inspection of every module in `quant_platform.execution_gateway` (plus
`config.execution_gateway_schemas`), that no prohibited construct
exists -- mirrors `tests/unit/paper_trading/test_safety_scan.py`'s
identical structural (not naive-substring) approach exactly, plus the
Milestone-8-specific checks Section 35 adds: MT5/FxPro/FIX client
imports, live adapter classes, live execution-mode values, and forbidden
CLI command names.

NON-VACUOUS BY CONSTRUCTION: `TestScannerIsNonVacuous` runs every
detector against DELIBERATELY BAD source snippets (never against the
real package) and asserts each one actually fires -- proving this file
would catch a real violation, not silently pass everything."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_EXECUTION_GATEWAY_SRC = Path(__file__).resolve().parents[3] / "src" / "quant_platform" / "execution_gateway"
_CONFIG_SCHEMA_FILE = Path(__file__).resolve().parents[3] / "src" / "quant_platform" / "config" / "execution_gateway_schemas.py"

_FORBIDDEN_IMPORT_MODULES = frozenset({
    "pickle", "pdb", "socket", "requests", "urllib", "urllib2", "http.client", "httplib", "websocket", "websockets",
    "aiohttp", "httpx", "paramiko", "ftplib", "telnetlib", "smtplib", "asyncio.streams", "MetaTrader5", "mt5", "metatrader5",
    "ccxt", "ib_insync", "fix", "quickfix", "simplefix",
})
_FORBIDDEN_CALL_NAMES = frozenset({"eval", "exec", "breakpoint"})
_BROAD_EXCEPTION_BODY_KINDS = (ast.Pass, ast.Continue)
_CREDENTIAL_IDENTIFIER_PATTERN = re.compile(r"(?i)\b(api_key|api_secret|apikey|broker_password|broker_credential|account_password|mt5_login|mt5_password|access_token|secret_key)\b")
_TODO_MARKER_PATTERN = re.compile(r"#\s*(TODO|FIXME|HACK)\b")
_HARDCODED_PATH_PATTERN = re.compile(r"""^[\"']([A-Za-z]:\\\\|[A-Za-z]:/|/home/|/Users/|/etc/|/var/)""")
_FORBIDDEN_CLI_COMMAND_NAMES = frozenset({"run-live", "submit-live-order", "connect-mt5", "connect-broker", "deploy-live", "trade-real-money", "execute-mt5"})
_LIVE_MODE_VALUE_PATTERN = re.compile(r"(?i)\b(live|real_broker|production_trading|broker_live)\b")


def _execution_gateway_source_files() -> list[Path]:
    files = sorted(_EXECUTION_GATEWAY_SRC.glob("*.py"))
    files = [f for f in files if f.name != "__init__.py"]
    files.append(_CONFIG_SCHEMA_FILE)
    return files


_SOURCE_FILES = _execution_gateway_source_files()


@pytest.fixture(params=_SOURCE_FILES, ids=[f.name for f in _SOURCE_FILES])
def source_file(request: pytest.FixtureRequest) -> Path:
    return request.param


def _parse(path: Path) -> tuple[str, ast.Module]:
    text = path.read_text(encoding="utf-8")
    return text, ast.parse(text, filename=str(path))


def _parse_source(text: str) -> ast.Module:
    return ast.parse(text)


# --------------------------------------------------------------------------
# Detector functions -- each is exercised BOTH against the real package
# (below) AND against deliberately bad snippets (TestScannerIsNonVacuous),
# so the same function is what proves both "the real code is clean" and
# "the check itself actually works".
# --------------------------------------------------------------------------
def _find_forbidden_imports(tree: ast.Module) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0].lower() in _FORBIDDEN_IMPORT_MODULES or alias.name.lower() in _FORBIDDEN_IMPORT_MODULES:
                    found.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and (node.module.split(".")[0].lower() in _FORBIDDEN_IMPORT_MODULES or node.module.lower() in _FORBIDDEN_IMPORT_MODULES):
            found.append(node.module)
    return found


def _find_forbidden_calls(tree: ast.Module) -> list[str]:
    return [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALL_NAMES]


def _find_shell_true(tree: ast.Module) -> list[int]:
    return [kw.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) for kw in node.keywords if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True]


def _find_debug_print(tree: ast.Module) -> list[int]:
    return [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"]


def _find_silent_broad_except(tree: ast.Module) -> list[int]:
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        is_bare = node.type is None
        is_broad_named = isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException")
        if not (is_bare or is_broad_named):
            continue
        body_is_trivial = len(node.body) == 1 and isinstance(node.body[0], _BROAD_EXCEPTION_BODY_KINDS)
        if is_bare or body_is_trivial:
            offenders.append(node.lineno)
    return offenders


def _find_todo_markers(text: str) -> list[str]:
    return [f"line {i}: {line.strip()}" for i, line in enumerate(text.splitlines(), start=1) if _TODO_MARKER_PATTERN.search(line)]


def _find_hardcoded_paths(tree: ast.Module) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            candidate = repr(node.value) if "\\" in node.value else f'"{node.value}"'
            if _HARDCODED_PATH_PATTERN.match(candidate):
                hits.append((node.lineno, node.value))
    return hits


def _find_credential_identifiers(tree: ast.Module) -> set[str]:
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
    return offenders


def _find_live_mode_string_constants(tree: ast.Module) -> list[tuple[int, str]]:
    """Flags a string literal ENUM-VALUE-shaped constant (assigned to a
    Name in an Enum class body) that looks live-trading-shaped -- NOT
    prose. `execution_gateway.models.ExecutionMode`/`AdapterKind` are
    single-member enums (`test_only`/`deterministic_dummy`); this proves
    no sibling live-like member was ever added."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(isinstance(base, ast.Name) and base.id == "Enum" for base in node.bases):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str) and _LIVE_MODE_VALUE_PATTERN.search(stmt.value.value):
                    hits.append((stmt.lineno, stmt.value.value))
    return hits


class TestNoForbiddenImports:
    def test_no_network_broker_or_unsafe_deserialization_imports(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        found = _find_forbidden_imports(tree)
        assert not found, f"{source_file.name} imports forbidden module(s): {found}"


class TestNoForbiddenCalls:
    def test_no_eval_exec_or_breakpoint(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        found = _find_forbidden_calls(tree)
        assert not found, f"{source_file.name} calls forbidden function(s): {found}"

    def test_no_shell_true(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        found = _find_shell_true(tree)
        assert not found, f"{source_file.name} contains a shell=True call"

    def test_no_debug_print_calls(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        found = _find_debug_print(tree)
        assert not found, f"{source_file.name} (a library module) calls print() {len(found)} time(s) -- CLI output belongs in ml_cli.py"


class TestNoSilentBroadExceptionSwallowing:
    def test_no_bare_or_silently_passed_broad_except(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        offenders = _find_silent_broad_except(tree)
        assert not offenders, f"{source_file.name} has silent broad-exception swallowing at line(s): {offenders}"


class TestNoMarkerCommentsOrHardcodedPaths:
    def test_no_todo_fixme_hack_markers(self, source_file: Path) -> None:
        text, _ = _parse(source_file)
        hits = _find_todo_markers(text)
        assert not hits, f"{source_file.name} contains TODO/FIXME/HACK marker(s): {hits}"

    def test_no_hardcoded_absolute_local_paths(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        hits = _find_hardcoded_paths(tree)
        assert not hits, f"{source_file.name} contains hardcoded absolute local path literal(s): {hits}"


class TestNoBrokerCredentialShapedIdentifiers:
    def test_no_credential_shaped_names_as_actual_identifiers(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        offenders = _find_credential_identifiers(tree)
        assert not offenders, f"{source_file.name} declares credential-shaped identifier(s): {sorted(offenders)}"


class TestNoLiveModeEnumValues:
    def test_no_live_shaped_enum_member_values(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        hits = _find_live_mode_string_constants(tree)
        assert not hits, f"{source_file.name} defines a live-trading-shaped enum value: {hits}"


class TestCliHasNoLiveCommands:
    def test_ml_cli_registers_no_live_trading_command(self) -> None:
        import argparse

        from quant_platform.ml_cli import build_parser

        parser = build_parser()
        subparsers_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        registered = set(subparsers_action.choices.keys())
        assert not (_FORBIDDEN_CLI_COMMAND_NAMES & registered), f"ml_cli registers forbidden live-trading command(s): {_FORBIDDEN_CLI_COMMAND_NAMES & registered}"

    def test_ml_cli_registers_every_required_milestone_8_command(self) -> None:
        import argparse

        from quant_platform.ml_cli import build_parser

        parser = build_parser()
        subparsers_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        registered = set(subparsers_action.choices.keys())
        required = {
            "create-execution-gateway-spec", "run-dummy-execution-session", "pause-execution-session", "resume-execution-session",
            "inspect-execution-session", "inspect-execution-intents", "inspect-execution-commands", "inspect-execution-orders",
            "inspect-execution-fills", "inspect-broker-events", "inspect-execution-health", "inspect-execution-reconciliation",
            "report-execution-session", "verify-execution-session", "replay-execution-session", "compare-execution-to-paper",
        }
        assert required <= registered, f"ml_cli is missing required Milestone 8 command(s): {required - registered}"


class TestOnlyOneAdapterKindAndExecutionModeExist:
    def test_execution_mode_is_single_member(self) -> None:
        from quant_platform.execution_gateway.models import ExecutionMode

        assert [m.value for m in ExecutionMode] == ["test_only"]

    def test_adapter_kind_is_single_member(self) -> None:
        from quant_platform.execution_gateway.models import AdapterKind

        assert [a.value for a in AdapterKind] == ["deterministic_dummy"]

    def test_only_one_adapter_class_exists_in_the_package(self) -> None:
        """Structural proof there is exactly one concrete adapter
        implementation in the whole package -- a future MT5 adapter
        milestone would ADD a second, never silently coexist unnoticed."""
        adapter_class_names = []
        for path in _SOURCE_FILES:
            if path.name in ("adapter.py", "execution_gateway_schemas.py"):
                continue
            _, tree = _parse(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.endswith("BrokerAdapter"):
                    adapter_class_names.append(f"{path.name}:{node.name}")
        assert adapter_class_names == ["dummy_broker.py:DeterministicDummyBrokerAdapter"], adapter_class_names


class TestScannerIsNonVacuous:
    """Runs every detector above against DELIBERATELY BAD source, never
    the real package, proving each one actually fires."""

    def test_forbidden_import_detector_fires(self) -> None:
        assert _find_forbidden_imports(_parse_source("import socket\n"))
        assert _find_forbidden_imports(_parse_source("import MetaTrader5\n"))
        assert _find_forbidden_imports(_parse_source("from quickfix import Session\n"))

    def test_forbidden_call_detector_fires(self) -> None:
        assert _find_forbidden_calls(_parse_source("eval('1+1')\n"))
        assert _find_forbidden_calls(_parse_source("exec('x=1')\n"))

    def test_shell_true_detector_fires(self) -> None:
        assert _find_shell_true(_parse_source("subprocess.run(['ls'], shell=True)\n"))

    def test_debug_print_detector_fires(self) -> None:
        assert _find_debug_print(_parse_source("print('debug')\n"))

    def test_silent_broad_except_detector_fires(self) -> None:
        assert _find_silent_broad_except(_parse_source("try:\n    x = 1\nexcept Exception:\n    pass\n"))
        assert _find_silent_broad_except(_parse_source("try:\n    x = 1\nexcept:\n    pass\n"))

    def test_todo_marker_detector_fires(self) -> None:
        assert _find_todo_markers("x = 1  # TODO: fix this later\n")

    def test_hardcoded_path_detector_fires(self) -> None:
        assert _find_hardcoded_paths(_parse_source('x = "C:\\\\Users\\\\me\\\\secret"\n'))
        assert _find_hardcoded_paths(_parse_source('x = "/home/me/secret"\n'))

    def test_credential_identifier_detector_fires(self) -> None:
        assert _find_credential_identifiers(_parse_source("api_key = 'x'\n"))
        assert _find_credential_identifiers(_parse_source("def f(mt5_password): pass\n"))

    def test_live_mode_enum_detector_fires(self) -> None:
        bad_enum = "from enum import Enum\nclass Mode(Enum):\n    LIVE = 'live'\n"
        assert _find_live_mode_string_constants(_parse_source(bad_enum))
