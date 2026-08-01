"""Safety scan (Milestone 10, Phases 2, 3, and 4A): confirms the shipped
`market_data` source (EVERY `*.py` file in the package, RECURSIVELY --
`_all_source_files()` uses `rglob`, so this reaches the `collectors/`
subpackage too, and nothing needs re-pointing when a new module is
added anywhere in the tree) contains no network clients, no broker
code, no credentials, no live-trading code, no `float`-typed financial
dataclass fields, no `uuid4`/`random`-derived economic identity, no
internal wall-clock economic input, no overwrite/bypass flags, no
silent broad exception handling, no unsafe `pickle` usage, no
cloud-service SDK imports, no `eval`/`exec`, and no shell command
execution -- plus dedicated path-traversal-rejection tests.

PHASE 4A CARVE-OUT, EXPLAINED: `collectors/` is the one, deliberately
isolated, network-CAPABLE subpackage (Milestone 10, Phase 4A) -- it
legitimately needs `socket` (in `transport.py` ONLY, for pinned-IP
SSRF-safe connections) and legitimately has function PARAMETERS named
`api_key` (never a stored field -- see `request_manifest.py`'s own
structural `SecretExposureError` guard). Two checks below --
`_FORBIDDEN_NETWORK_IMPORTS` and `_CREDENTIAL_FIELD` -- are therefore
scanned over `_non_collector_source_files()` (everything this repo has
always run them over) rather than the full tree, and
`TestCollectorSpecificSafety` below applies NARROWER, MORE PRECISE
replacements scoped to `collectors/` specifically: an AST-based check
that no `@dataclass` field (as opposed to a function parameter) is ever
credential-shaped, confirmation that `socket` is imported nowhere in
`collectors/` EXCEPT `transport.py`, and that no third-party HTTP/
WebSocket library is imported even there.

THE SCANNER IS PROVEN NON-VACUOUS: `TestScannerCatchesDeliberatelyBadCode`
(plus `TestCollectorSpecificSafety`'s own non-vacuity checks) runs every
check against a small, deliberately BAD snippet engineered to violate
exactly that one rule, asserting the scanner DOES flag it -- a scanner
that always reports "clean" would pass the real-source checks for a
trivial (wrong) reason; this class rules that out."""

from __future__ import annotations

import ast
import re
import tempfile
from pathlib import Path

import pytest

from quant_platform.core.exceptions import MarketDataPathSecurityError
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
)
from quant_platform.market_data.partitions import PartitionStore

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "quant_platform" / "market_data"
_COLLECTORS_ROOT = _SRC_ROOT / "collectors"

_FORBIDDEN_NETWORK_IMPORTS = re.compile(r"^\s*(import|from)\s+(socket|requests|httpx|aiohttp|urllib\.request|websocket|websockets|ftplib|telnetlib)\b", re.MULTILINE)
_FORBIDDEN_THIRD_PARTY_NETWORK_IMPORTS = re.compile(r"^\s*(import|from)\s+(requests|httpx|aiohttp|urllib\.request|websocket|websockets|ftplib|telnetlib)\b", re.MULTILINE)
_RAW_SOCKET_IMPORT = re.compile(r"^\s*(import|from)\s+socket\b", re.MULTILINE)
_FORBIDDEN_BROKER_IMPORTS = re.compile(r"^\s*(import|from)\s+(MetaTrader5|mt5(?!__)|fxpro)\b", re.MULTILINE)
_CREDENTIAL_FIELD = re.compile(r"^\s*(password|api_key|api_secret|secret_key|access_token|client_secret)\s*[:=]", re.MULTILINE | re.IGNORECASE)
_LIVE_TRADING = re.compile(r"\bLIVE_TRADING\b|\bplace_live_order\b|\bsubmit_to_broker\b")
_FLOAT_FIELD = re.compile(r"^\s+\w+\s*:\s*float\b", re.MULTILINE)
_UUID4_OR_RANDOM = re.compile(r"\buuid\.uuid4\(\)|\brandom\.\w+\(")
_WALL_CLOCK = re.compile(r"\bdatetime\.now\(|\.utcnow\(|\btime\.time\(\)")
_OVERWRITE_BYPASS_FLAG = re.compile(r"\b(force|overwrite|skip_verification|bypass)\s*[:=]\s*(bool\b|True\b|False\b)")
_BARE_EXCEPT = re.compile(r"^\s*except\s*:\s*$", re.MULTILINE)
_SWALLOWED_EXCEPTION = re.compile(r"except\s+Exception\s*:\s*\n\s*pass\b")
_UNSAFE_PICKLE = re.compile(r"^\s*import\s+pickle\b|\bpickle\.loads?\(", re.MULTILINE)
_CLOUD_SDK_IMPORTS = re.compile(r"^\s*(import|from)\s+(boto3|botocore|azure|google\.cloud|gcloud)\b", re.MULTILINE)
_EVAL_OR_EXEC = re.compile(r"\beval\(|\bexec\(")
_SHELL_EXECUTION = re.compile(r"\bsubprocess\.\w+\(|\bos\.system\(|\bos\.popen\(")
_FLOAT_CALL = re.compile(r"\bfloat\(")
_CREDENTIAL_SHAPED_NAMES = frozenset({"password", "api_key", "apikey", "api_secret", "secret_key", "access_token", "client_secret", "token", "authorization", "secret"})
_LONG_LITERAL_API_KEY = re.compile(r"api_key\s*=\s*[\"'][A-Za-z0-9_\-]{20,}[\"']", re.IGNORECASE)


def _all_source_files() -> list[Path]:
    return sorted(_SRC_ROOT.rglob("*.py"))


def _non_collector_source_files() -> list[Path]:
    return sorted(p for p in _all_source_files() if _COLLECTORS_ROOT not in p.parents)


def _collector_source_files() -> list[Path]:
    return sorted(_COLLECTORS_ROOT.rglob("*.py"))


def _combined_source() -> str:
    """The full tree (`rglob`), used by every check EXCEPT the two with a
    deliberate, documented `collectors/` carve-out (see module
    docstring)."""
    return "\n".join(p.read_text(encoding="utf-8") for p in _all_source_files())


def _combined_source_non_collector() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in _non_collector_source_files())


def _float_call_lines_outside_prose(text: str) -> list[str]:
    """`float(` appears legitimately in this package's own PROSE (module
    docstrings use backtick-quoted inline code, e.g. `` `float(Decimal
    ("0.1"))` `` in `identity.py`'s own "never float" explanation) but
    never in real Python syntax with a backtick on the same line -- this
    filters those prose mentions out while still catching a genuine
    `float(...)` call site, which would mean a financial value was
    parsed through binary floating point somewhere it should not be."""
    flagged = []
    for line in text.splitlines():
        if _FLOAT_CALL.search(line) and "`" not in line:
            flagged.append(line)
    return flagged


def _uuid4_lines_outside_temp_naming(text: str) -> list[str]:
    """`uuid.uuid4()` is legitimately used for TEMP FILE NAMING in the
    atomic-write helpers (`core.json.write_json_atomic`,
    `recovery._atomically_rewrite_jsonl`) -- never for economic identity.
    Every content id in this package goes through `compute_content_id`
    (a deterministic sha256), never a random/uuid value. This helper
    flags any `uuid4`/`random` usage NOT on a line naming a temp path."""
    flagged = []
    for line in text.splitlines():
        if _UUID4_OR_RANDOM.search(line) and "tmp_path" not in line and "tmp" not in line.lower():
            flagged.append(line)
    return flagged


class TestNoNetworkOrBrokerCode:
    def test_no_network_client_imports(self) -> None:
        """Scoped to `_combined_source_non_collector()`: `collectors/`
        is the one deliberately network-capable subpackage (Milestone
        10, Phase 4A); `TestCollectorSpecificSafety` applies a narrower,
        more precise replacement to it (below)."""
        source = _combined_source_non_collector()
        assert not _FORBIDDEN_NETWORK_IMPORTS.search(source)

    def test_no_broker_sdk_imports(self) -> None:
        source = _combined_source()
        assert not _FORBIDDEN_BROKER_IMPORTS.search(source)


class TestNoCredentials:
    def test_no_credential_field_declarations(self) -> None:
        """Scoped to `_combined_source_non_collector()`: `collectors/`
        legitimately has function PARAMETERS named `api_key` (FRED
        credential support, supplied by the caller at call time, never
        stored -- see `request_manifest.py`'s own structural
        `SecretExposureError` guard). `TestCollectorSpecificSafety`
        applies a narrower, AST-based replacement scoped to actual
        dataclass FIELDS (below), which correctly does not confuse a
        parameter with a stored field."""
        source = _combined_source_non_collector()
        assert not _CREDENTIAL_FIELD.search(source)


class TestNoLiveTrading:
    def test_no_live_trading_markers(self) -> None:
        source = _combined_source()
        assert not _LIVE_TRADING.search(source)


_DURATION_SHAPED_FLOAT_FIELD_NAMES = frozenset({"connect_timeout", "read_timeout", "wait_seconds_before_next", "retry_after_seconds"})


def _is_duration_shaped_float_field_line(line: str) -> bool:
    """A `float`-typed field/parameter is legitimate ONLY for HTTP
    transport timeouts and retry/backoff DURATIONS in seconds (never an
    economic value) -- `collectors/` has several of these
    (`connect_timeout`, `read_timeout`, `wait_seconds_before_next`,
    `retry_after_seconds`). This narrowly exempts exactly those
    duration-shaped names (by exact name, or a `*_timeout`/`*_seconds`
    suffix) while still catching any OTHER float-typed field, including
    a genuine financial one that might later appear in `collectors/`."""
    name = line.strip().split(":", 1)[0].strip()
    return name in _DURATION_SHAPED_FLOAT_FIELD_NAMES or name.endswith("_timeout") or name.endswith("_seconds")


class TestNoFloatFinancialFields:
    def test_no_dataclass_field_annotated_float(self) -> None:
        """Scoped to `_non_collector_source_files()`; `collectors/`'s own
        (narrowly duration-shaped) float fields are separately verified
        by `TestCollectorSpecificSafety.test_float_fields_are_duration_
        shaped_only` below."""
        for path in _non_collector_source_files():
            text = path.read_text(encoding="utf-8")
            matches = _FLOAT_FIELD.findall(text)
            assert matches == [], f"{path.name} declares a float-typed field: {matches}"


class TestNoRandomOrUuidEconomicIdentity:
    def test_no_uuid4_or_random_outside_temp_file_naming(self) -> None:
        for path in _all_source_files():
            flagged = _uuid4_lines_outside_temp_naming(path.read_text(encoding="utf-8"))
            assert flagged == [], f"{path.name} uses uuid4/random outside temp-file naming: {flagged}"


class TestNoInternalWallClockEconomicInput:
    def test_no_wall_clock_reads_in_source(self) -> None:
        source = _combined_source()
        assert not _WALL_CLOCK.search(source)


class TestNoOverwriteOrBypassFlags:
    def test_no_force_overwrite_or_skip_verification_parameters(self) -> None:
        source = _combined_source()
        assert not _OVERWRITE_BYPASS_FLAG.search(source)


class TestNoSilentBroadException:
    def test_no_bare_except(self) -> None:
        source = _combined_source()
        assert not _BARE_EXCEPT.search(source)

    def test_no_swallowed_exception_exception_pass(self) -> None:
        source = _combined_source()
        assert not _SWALLOWED_EXCEPTION.search(source)


class TestNoUnsafePickle:
    def test_no_pickle_usage(self) -> None:
        source = _combined_source()
        assert not _UNSAFE_PICKLE.search(source)


class TestNoCloudServiceSdkImports:
    def test_no_cloud_sdk_imports(self) -> None:
        source = _combined_source()
        assert not _CLOUD_SDK_IMPORTS.search(source)


class TestNoEvalOrExec:
    def test_no_eval_or_exec(self) -> None:
        source = _combined_source()
        assert not _EVAL_OR_EXEC.search(source)


class TestNoShellCommandExecution:
    def test_no_subprocess_or_os_system(self) -> None:
        source = _combined_source()
        assert not _SHELL_EXECUTION.search(source)


class TestNoFloatFinancialParsing:
    """Beyond `TestNoFloatFinancialFields`'s dataclass-field check: no
    `float(...)` CALL anywhere in the package -- every Decimal parsing
    path (`identity.parse_decimal`, `source_normalization.
    parse_source_decimal`) goes through `Decimal(str(raw))` directly,
    never a float intermediate."""

    def test_no_float_call_outside_prose(self) -> None:
        """Scoped to `_non_collector_source_files()`; `collectors/retry.py`'s
        own (narrowly duration-parsing-only) `float(...)` calls are
        separately verified by `TestCollectorSpecificSafety.
        test_float_calls_in_collectors_are_duration_parsing_only` below."""
        for path in _non_collector_source_files():
            flagged = _float_call_lines_outside_prose(path.read_text(encoding="utf-8"))
            assert flagged == [], f"{path.name} calls float(...) outside prose: {flagged}"


class TestPathTraversalRejection:
    def test_partition_key_path_traversal_is_rejected(self) -> None:
        key = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="mt5__XAUUSD", provider="mt5")
        with tempfile.TemporaryDirectory() as tmp:
            store = PartitionStore(Path(tmp))
            for malicious_key in ("../../../etc/passwd", "..\\..\\windows\\system32", "/etc/passwd", "2026-01-05/../../secret"):
                with pytest.raises(MarketDataPathSecurityError):
                    store.read(key, malicious_key)


class TestScannerCatchesDeliberatelyBadCode:
    """Proves every regex above is non-vacuous -- each is fired against a
    tiny snippet engineered to violate exactly that rule."""

    def test_network_import_is_caught(self) -> None:
        assert _FORBIDDEN_NETWORK_IMPORTS.search("import requests\n")
        assert _FORBIDDEN_NETWORK_IMPORTS.search("from urllib.request import urlopen\n")

    def test_broker_import_is_caught(self) -> None:
        assert _FORBIDDEN_BROKER_IMPORTS.search("import MetaTrader5 as mt5\n")

    def test_credential_field_is_caught(self) -> None:
        assert _CREDENTIAL_FIELD.search("    password: str\n")
        assert _CREDENTIAL_FIELD.search("api_key = 'abc123'\n")

    def test_live_trading_marker_is_caught(self) -> None:
        assert _LIVE_TRADING.search("mode = LIVE_TRADING\n")

    def test_float_field_is_caught(self) -> None:
        assert _FLOAT_FIELD.search("class Foo:\n    price: float\n")

    def test_uuid4_outside_temp_naming_is_caught(self) -> None:
        flagged = _uuid4_lines_outside_temp_naming("event_id = uuid.uuid4().hex\n")
        assert flagged

    def test_uuid4_for_temp_naming_is_not_flagged(self) -> None:
        flagged = _uuid4_lines_outside_temp_naming("tmp_path = path.parent / f'.{path.name}.{uuid.uuid4().hex}.tmp'\n")
        assert flagged == []

    def test_wall_clock_read_is_caught(self) -> None:
        assert _WALL_CLOCK.search("now = datetime.now()\n")
        assert _WALL_CLOCK.search("now = datetime.utcnow()\n")

    def test_overwrite_flag_is_caught(self) -> None:
        assert _OVERWRITE_BYPASS_FLAG.search("def f(force: bool = False):\n")
        assert _OVERWRITE_BYPASS_FLAG.search("def f(overwrite=True):\n")

    def test_bare_except_is_caught(self) -> None:
        assert _BARE_EXCEPT.search("try:\n    pass\nexcept:\n    pass\n")

    def test_swallowed_exception_is_caught(self) -> None:
        assert _SWALLOWED_EXCEPTION.search("try:\n    pass\nexcept Exception:\n    pass\n")

    def test_pickle_usage_is_caught(self) -> None:
        assert _UNSAFE_PICKLE.search("import pickle\n")
        assert _UNSAFE_PICKLE.search("pickle.loads(data)\n")

    def test_cloud_sdk_import_is_caught(self) -> None:
        assert _CLOUD_SDK_IMPORTS.search("import boto3\n")
        assert _CLOUD_SDK_IMPORTS.search("from google.cloud import storage\n")

    def test_eval_or_exec_is_caught(self) -> None:
        assert _EVAL_OR_EXEC.search("eval('1 + 1')\n")
        assert _EVAL_OR_EXEC.search("exec(user_code)\n")

    def test_shell_execution_is_caught(self) -> None:
        assert _SHELL_EXECUTION.search("subprocess.run(['ls'])\n")
        assert _SHELL_EXECUTION.search("os.system('rm -rf /')\n")

    def test_float_call_outside_prose_is_caught(self) -> None:
        flagged = _float_call_lines_outside_prose("value = float(raw_text)\n")
        assert flagged

    def test_float_call_inside_backtick_prose_is_not_flagged(self) -> None:
        flagged = _float_call_lines_outside_prose('"""`float(Decimal("0.1"))` is not exactly 0.1"""\n')
        assert flagged == []


class TestSafetyScanIsActuallyExercisedAgainstRealSource:
    def test_scanner_actually_reads_a_nonempty_source_tree(self) -> None:
        # Guards against the scanner silently checking zero files (which
        # would make every "clean" assertion above vacuously true).
        files = _all_source_files()
        assert len(files) >= 13
        combined = _combined_source()
        assert len(combined) > 10_000

    def test_scanner_reaches_the_collectors_subpackage(self) -> None:
        # Guards against `_all_source_files()` silently reverting to a
        # non-recursive glob (which would make every check above pass
        # for `collectors/*.py` vacuously, by never reading it at all).
        collector_files = _collector_source_files()
        assert len(collector_files) >= 10
        assert any(p.name == "transport.py" for p in collector_files)
        assert any(p in _all_source_files() for p in collector_files)


# --------------------------------------------------------------------------
# Milestone 10, Phase 4A: checks specific to the `collectors/` subpackage
# -- the one deliberately network-capable part of `market_data` (see
# module docstring's PHASE 4A CARVE-OUT discussion for why the two
# checks above are scoped away from it, and what replaces them here).
# --------------------------------------------------------------------------
def _collector_combined_source() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in _collector_source_files())


def _dataclass_fields_with_credential_shaped_names(tree: ast.Module, *, filename: str) -> list[str]:
    """AST-based (not regex): walks every `@dataclass`-decorated class,
    inspects only its own direct body statements (`AnnAssign`/`Assign`
    class-level field declarations) -- structurally CANNOT confuse this
    with a function parameter (which lives in `FunctionDef.args`, a
    completely different AST location), unlike the whole-tree regex
    `_CREDENTIAL_FIELD` this replaces for `collectors/`."""
    flagged: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        is_dataclass = any(
            (isinstance(dec, ast.Name) and dec.id == "dataclass")
            or (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "dataclass")
            for dec in node.decorator_list
        )
        if not is_dataclass:
            continue
        for stmt in node.body:
            name = None
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                name = stmt.targets[0].id
            if name is not None and name.lower() in _CREDENTIAL_SHAPED_NAMES:
                flagged.append(f"{filename}:{node.name}.{name} (line {stmt.lineno})")
    return flagged


class TestCollectorSpecificSafety:
    def test_no_credential_shaped_dataclass_fields(self) -> None:
        """Replaces `_CREDENTIAL_FIELD` for `collectors/`: no `@dataclass`
        anywhere in the subpackage stores a raw secret as a FIELD --
        `CollectorRequestManifest`/`CollectorResponseManifest`/etc. only
        ever carry `credential_mode: CredentialMode` (a bare label), per
        `request_manifest.py`'s own structural guard."""
        for path in _collector_source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            flagged = _dataclass_fields_with_credential_shaped_names(tree, filename=path.name)
            assert flagged == [], f"credential-shaped dataclass field(s) found: {flagged}"

    def test_ast_check_is_non_vacuous(self) -> None:
        bad_source = "from dataclasses import dataclass\n\n@dataclass\nclass Bad:\n    api_key: str\n"
        tree = ast.parse(bad_source, filename="<bad>")
        flagged = _dataclass_fields_with_credential_shaped_names(tree, filename="<bad>")
        assert flagged, "AST-based credential-field check failed to catch a deliberately bad dataclass field"

    def test_ast_check_does_not_flag_a_function_parameter(self) -> None:
        # The exact shape `fred.py`'s `execute_fred_request` legitimately
        # uses -- must NOT be flagged, since it is a parameter, not a
        # stored field.
        ok_source = "def f(*, api_key: str | None) -> None:\n    pass\n"
        tree = ast.parse(ok_source, filename="<ok>")
        flagged = _dataclass_fields_with_credential_shaped_names(tree, filename="<ok>")
        assert flagged == []

    def test_no_third_party_network_library_even_in_transport(self) -> None:
        """`collectors/transport.py` may use the stdlib `socket`/`ssl`/
        `http.client` trio (it is the ONE sanctioned transport
        implementation) but must never import `requests`/`httpx`/
        `aiohttp`/a websocket library/`ftplib`/`telnetlib` -- this
        applies to EVERY file in `collectors/`, transport.py included."""
        source = _collector_combined_source()
        assert not _FORBIDDEN_THIRD_PARTY_NETWORK_IMPORTS.search(source)

    def test_raw_socket_import_appears_only_in_transport_py(self) -> None:
        offenders = [p.name for p in _collector_source_files() if p.name != "transport.py" and _RAW_SOCKET_IMPORT.search(p.read_text(encoding="utf-8"))]
        assert offenders == [], f"raw `socket` import found outside transport.py: {offenders}"
        transport_source = (_COLLECTORS_ROOT / "transport.py").read_text(encoding="utf-8")
        assert _RAW_SOCKET_IMPORT.search(transport_source), "transport.py is expected to import socket (sanity check that the exemption is even exercised)"

    def test_no_websocket_broker_mt5_fxpro_or_live_streaming_code(self) -> None:
        """Checks actual CODE markers (import statements, the existing
        `_FORBIDDEN_BROKER_IMPORTS`/`_FORBIDDEN_THIRD_PARTY_NETWORK_
        IMPORTS` patterns, plus a live-trading-marker regex applied
        per-file) rather than a bare-word scan -- `collectors/__init__.py`'s
        own docstring legitimately SAYS "no broker/MT5/FxPro code" in
        prose while containing none of it."""
        for path in _collector_source_files():
            text = path.read_text(encoding="utf-8")
            assert not _FORBIDDEN_BROKER_IMPORTS.search(text), f"{path.name} imports a broker SDK"
            assert not re.search(r"^\s*(import|from)\s+websockets?\b", text, re.MULTILINE), f"{path.name} imports a websocket library"
            assert not _LIVE_TRADING.search(text), f"{path.name} contains a live-trading marker"

    def test_https_only_scheme_enforced(self) -> None:
        transport_source = (_COLLECTORS_ROOT / "transport.py").read_text(encoding="utf-8")
        assert '"https"' in transport_source or "'https'" in transport_source

    def test_fred_host_allowlist_is_a_single_exact_hostname(self) -> None:
        from quant_platform.market_data.collectors.fred import FRED_ALLOWED_HOSTS, FRED_ENDPOINT_HOST

        assert frozenset({"api.stlouisfed.org"}) == FRED_ALLOWED_HOSTS
        assert FRED_ENDPOINT_HOST == "api.stlouisfed.org"

    def test_transport_request_rejects_empty_allowed_hosts(self) -> None:
        from datetime import datetime, timezone

        from quant_platform.core.exceptions import CollectorError
        from quant_platform.market_data.collectors.protocols import TransportRequest

        with pytest.raises(CollectorError):
            TransportRequest(url="https://api.stlouisfed.org/x", headers={}, allowed_hosts=frozenset(), request_time=datetime(2024, 1, 1, tzinfo=timezone.utc))

    def test_http_url_is_rejected(self) -> None:
        from quant_platform.core.exceptions import DisallowedUrlError
        from quant_platform.market_data.collectors.transport import _validate_url_static

        with pytest.raises(DisallowedUrlError):
            _validate_url_static("http://api.stlouisfed.org/fred/series/observations", allowed_hosts=frozenset({"api.stlouisfed.org"}))

    def test_localhost_url_is_rejected(self) -> None:
        """Uses the REAL FRED allowlist (never a self-referential one) --
        the realistic threat model is an attacker-controlled or
        misconfigured URL pointing at localhost/loopback while the
        collector's only legitimate intent is ever `api.stlouisfed.org`."""
        from quant_platform.core.exceptions import DisallowedUrlError
        from quant_platform.market_data.collectors.fred import FRED_ALLOWED_HOSTS
        from quant_platform.market_data.collectors.transport import _validate_url_static

        for host in ("localhost", "127.0.0.1", "[::1]"):
            with pytest.raises(DisallowedUrlError):
                _validate_url_static(f"https://{host}/x", allowed_hosts=FRED_ALLOWED_HOSTS)

    def test_no_committed_long_literal_api_key(self) -> None:
        """Distinguishes a short test placeholder (`"SECRET_KEY_VALUE"`,
        `"test"`, ...) from something that LOOKS like a real, long,
        opaque committed credential. Scans both the `collectors/` source
        AND this test file's own sibling `test_collectors_*.py` fixtures
        (which legitimately construct manifests with `api_key=...`
        arguments), catching the literal mistake this repo must never
        make in either place."""
        source = _collector_combined_source()
        assert not _LONG_LITERAL_API_KEY.search(source)
        tests_dir = Path(__file__).resolve().parent
        for path in sorted(tests_dir.glob("test_collectors_*.py")):
            assert not _LONG_LITERAL_API_KEY.search(path.read_text(encoding="utf-8")), f"{path.name} contains a long literal api_key= value"

    def test_float_fields_are_duration_shaped_only(self) -> None:
        """Replaces `_FLOAT_FIELD` for `collectors/`: every float-typed
        field/parameter here is an HTTP timeout or retry/backoff
        DURATION in seconds -- never an economic value (see
        `_is_duration_shaped_float_field_line`'s own docstring)."""
        for path in _collector_source_files():
            for line in path.read_text(encoding="utf-8").splitlines():
                if _FLOAT_FIELD.match(line) and not _is_duration_shaped_float_field_line(line):
                    raise AssertionError(f"{path.name} declares a non-duration-shaped float field: {line.strip()!r}")

    def test_duration_shaped_float_field_check_is_non_vacuous(self) -> None:
        assert not _is_duration_shaped_float_field_line("    price: float")
        assert _is_duration_shaped_float_field_line("    connect_timeout: float")
        assert _is_duration_shaped_float_field_line("    wait_seconds_before_next: float")

    def test_float_calls_in_collectors_are_duration_parsing_only(self) -> None:
        """Replaces `_float_call_lines_outside_prose` for `collectors/`:
        the only `float(...)` call sites in the whole subpackage live in
        `retry.py`, parsing a configured backoff DURATION or an
        HTTP `Retry-After` header's seconds value -- never a financial
        quantity (which, throughout `collectors/`, is always parsed via
        `source_normalization.parse_source_decimal`, never `float`)."""
        for path in _collector_source_files():
            flagged = _float_call_lines_outside_prose(path.read_text(encoding="utf-8"))
            if not flagged:
                continue
            assert path.name == "retry.py", f"{path.name} calls float(...) outside retry.py's own duration parsing: {flagged}"
            for line in flagged:
                assert "seconds" in line or "stripped" in line, f"unexpected float(...) call shape in retry.py: {line!r}"


# --------------------------------------------------------------------------
# Milestone 10, Phase 4B: checks specific to `collectors/curated/` -- the
# curated multi-series FRED universe layer. `_collector_source_files()`
# already reaches this subpackage (it is a RECURSIVE `rglob` over all of
# `collectors/`), so every check above already covers it; this section adds
# ONE further, Phase-4B-specific structural check the earlier phases had no
# occasion to need: a curated observation's `observation_date` (the
# ECONOMIC PERIOD text FRED reports) must never be compared directly
# against a wall-clock-shaped time value as if it were proof of
# availability -- that is exactly the point-in-time leakage
# `availability_time`/`availability.py` exists to prevent (see
# `macro_observation.py`'s own module docstring: "never the availability
# proof"). A future change that filtered "is this observation visible
# yet" by checking `observation_date <= as_of` directly (skipping
# `resolve_availability_time` entirely) would reintroduce look-ahead bias
# silently; this check makes that reintroduction structurally loud.
# --------------------------------------------------------------------------
_CURATED_ROOT = _COLLECTORS_ROOT / "curated"
_OBSERVATION_DATE_AS_AVAILABILITY = re.compile(
    r"\bobservation_date\w*\s*(<=|>=|==|<|>)\s*(as_of|planning_time|operation_time|desired_observation_end)\b"
    r"|\b(as_of|planning_time|operation_time|desired_observation_end)\s*(<=|>=|==|<|>)\s*observation_date\w*\b"
)


def _curated_source_files() -> list[Path]:
    return sorted(_CURATED_ROOT.rglob("*.py"))


class TestCuratedSpecificSafety:
    def test_curated_subpackage_is_reached_by_the_scanner(self) -> None:
        # Guards against a silent path-computation mistake that would make
        # every check in this class vacuously true by scanning zero files.
        files = _curated_source_files()
        assert len(files) >= 10
        assert any(p.name == "availability.py" for p in files)
        assert all(p in _collector_source_files() for p in files)

    def test_no_direct_observation_date_used_as_availability_proof(self) -> None:
        """`observation_date` may be PASSED to `resolve_availability_time`
        (a named keyword argument, `availability.py`'s own legitimate
        entry point) but must never be COMPARED directly against a
        wall-clock-shaped time value anywhere in `curated/` -- doing so
        would bypass `AvailabilityPolicy` resolution entirely and treat
        the economic period as if it were proof of release timing."""
        for path in _curated_source_files():
            text = path.read_text(encoding="utf-8")
            match = _OBSERVATION_DATE_AS_AVAILABILITY.search(text)
            assert match is None, f"{path.name} compares observation_date directly against a time value: {match.group(0) if match else ''!r}"

    def test_observation_date_as_availability_check_is_non_vacuous(self) -> None:
        bad_snippets = (
            "if observation.observation_date <= as_of:\n",
            "visible = observation_date_text >= planning_time\n",
            "if desired_observation_end < observation_date:\n",
        )
        for snippet in bad_snippets:
            assert _OBSERVATION_DATE_AS_AVAILABILITY.search(snippet), f"scanner failed to catch: {snippet!r}"

    def test_legitimate_availability_time_comparison_is_not_flagged(self) -> None:
        """The CORRECT pattern (comparing the RESOLVED `availability_time`,
        never the raw `observation_date`, against `as_of`) must not be
        flagged -- proves the check is precise, not merely banning the
        word "observation_date" from ever appearing near a comparison."""
        good_snippet = "visible = observation.availability_time <= as_of\n"
        assert not _OBSERVATION_DATE_AS_AVAILABILITY.search(good_snippet)

    def test_no_secret_in_any_curated_module(self) -> None:
        """`acceptance.py` is the ONE sanctioned place in all of
        `collectors/` (curated included) that reads `FRED_API_KEY` from
        the environment -- confirmed here to be the ONLY curated module
        naming that env var, and that no curated module hardcodes a
        credential-shaped literal."""
        offenders = [p.name for p in _curated_source_files() if p.name != "acceptance.py" and "FRED_API_KEY" in p.read_text(encoding="utf-8")]
        assert offenders == [], f"unexpected FRED_API_KEY reference outside acceptance.py: {offenders}"
        source = _collector_combined_source()
        assert not _LONG_LITERAL_API_KEY.search(source)
