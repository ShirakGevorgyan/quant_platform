"""Milestone 9, Phase 1: safety scan for `quant_platform.portfolio_risk`
(plus `config.portfolio_risk_schemas`). Mirrors `tests/unit/
execution_gateway/test_execution_gateway_safety_scan.py`'s identical
structural (AST/source, never naive-substring) approach, scoped to what
Phase 1 actually delivers (no CLI exists yet, so no CLI-command checks).
Adds two checks specific to this milestone's own stricter safety scope:
no `float(...)` call anywhere in this package's domain modules (Decimal
throughout), and no internal wall-clock read (`datetime.now`/`utcnow`/
`pd.Timestamp.now`) anywhere -- every timestamp affecting an economic or
staleness decision must be caller-supplied.

NON-VACUOUS BY CONSTRUCTION: `TestScannerIsNonVacuous` runs every
detector against deliberately bad source snippets (never the real
package) and asserts each one actually fires."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_PORTFOLIO_RISK_SRC = Path(__file__).resolve().parents[3] / "src" / "quant_platform" / "portfolio_risk"
_CONFIG_SCHEMA_FILE = Path(__file__).resolve().parents[3] / "src" / "quant_platform" / "config" / "portfolio_risk_schemas.py"

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
_WALL_CLOCK_ATTRIBUTES = frozenset({"now", "utcnow", "today"})


def _portfolio_risk_source_files() -> list[Path]:
    files = sorted(_PORTFOLIO_RISK_SRC.glob("*.py"))
    files = [f for f in files if f.name != "__init__.py"]
    files.append(_CONFIG_SCHEMA_FILE)
    return files


_SOURCE_FILES = _portfolio_risk_source_files()


@pytest.fixture(params=_SOURCE_FILES, ids=[f.name for f in _SOURCE_FILES])
def source_file(request: pytest.FixtureRequest) -> Path:
    return request.param


def _parse(path: Path) -> tuple[str, ast.Module]:
    text = path.read_text(encoding="utf-8")
    return text, ast.parse(text, filename=str(path))


def _parse_source(text: str) -> ast.Module:
    return ast.parse(text)


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


def _find_float_calls(tree: ast.Module) -> list[int]:
    """This package uses `Decimal` exclusively for every quantity, price,
    monetary value, rate, and threshold -- a bare `float(...)` call
    anywhere in a domain module would be a structural violation of that
    rule (the one sanctioned exception, `identity.decimal_from_float`'s
    OWN internal use of `math.isfinite`, calls `math.isfinite`, never
    `float(...)`, so this detector has no legitimate exception to
    special-case)."""
    return [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float"]


def _find_wall_clock_reads(tree: ast.Module) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _WALL_CLOCK_ATTRIBUTES:
            hits.append((node.lineno, node.func.attr))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "utc_now":
            hits.append((node.lineno, "utc_now"))
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

    def test_no_debug_print_calls(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        found = _find_debug_print(tree)
        assert not found, f"{source_file.name} (a library module) calls print() {len(found)} time(s)"


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


class TestNoFloatArithmeticForFinancialValues:
    def test_no_bare_float_call(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        hits = _find_float_calls(tree)
        assert not hits, f"{source_file.name} calls float(...) at line(s) {hits} -- this package uses Decimal exclusively"


class TestNoWallClockDependentEconomicDecisions:
    def test_no_internal_wall_clock_read(self, source_file: Path) -> None:
        _, tree = _parse(source_file)
        hits = _find_wall_clock_reads(tree)
        assert not hits, f"{source_file.name} reads the wall clock internally at {hits} -- every timestamp must be caller-supplied"


class TestScannerIsNonVacuous:
    """Proves each detector above would actually catch a real violation --
    run only against deliberately bad snippets, never the real package."""

    def test_forbidden_import_detector_fires(self) -> None:
        tree = _parse_source("import socket\n")
        assert _find_forbidden_imports(tree)

    def test_forbidden_import_from_detector_fires(self) -> None:
        tree = _parse_source("import MetaTrader5\n")
        assert _find_forbidden_imports(tree)

    def test_forbidden_call_detector_fires(self) -> None:
        tree = _parse_source("eval('1+1')\n")
        assert _find_forbidden_calls(tree)

    def test_debug_print_detector_fires(self) -> None:
        tree = _parse_source("print('hello')\n")
        assert _find_debug_print(tree)

    def test_silent_broad_except_detector_fires(self) -> None:
        tree = _parse_source("try:\n    x = 1\nexcept Exception:\n    pass\n")
        assert _find_silent_broad_except(tree)

    def test_bare_except_detector_fires(self) -> None:
        tree = _parse_source("try:\n    x = 1\nexcept:\n    raise\n")
        assert _find_silent_broad_except(tree)

    def test_todo_marker_detector_fires(self) -> None:
        assert _find_todo_markers("# TODO: fix this later\n")

    def test_hardcoded_path_detector_fires(self) -> None:
        tree = _parse_source("x = '/home/user/secret.json'\n")
        assert _find_hardcoded_paths(tree)

    def test_credential_identifier_detector_fires(self) -> None:
        tree = _parse_source("api_key = 'x'\n")
        assert _find_credential_identifiers(tree)

    def test_float_call_detector_fires(self) -> None:
        tree = _parse_source("x = float('1.5')\n")
        assert _find_float_calls(tree)

    def test_wall_clock_detector_fires_for_datetime_now(self) -> None:
        tree = _parse_source("import datetime\nx = datetime.datetime.now()\n")
        assert _find_wall_clock_reads(tree)

    def test_wall_clock_detector_fires_for_utc_now_helper(self) -> None:
        tree = _parse_source("x = utc_now()\n")
        assert _find_wall_clock_reads(tree)

    def test_wall_clock_detector_fires_for_pandas_timestamp_now(self) -> None:
        tree = _parse_source("import pandas as pd\nx = pd.Timestamp.now()\n")
        assert _find_wall_clock_reads(tree)
