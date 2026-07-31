"""Safety scan (Milestone 10, Phases 2 and 3): confirms the shipped
`market_data` source (EVERY `*.py` file in the package, including every
Phase 3 historical-ingestion module -- `_all_source_files()` globs the
whole directory, so nothing needs re-pointing when a new module is
added) contains no network clients, no broker code, no credentials, no
live-trading code, no `float`-typed financial dataclass fields, no
`uuid4`/`random`-derived economic identity, no internal wall-clock
economic input, no overwrite/bypass flags, no silent broad exception
handling, no unsafe `pickle` usage, no cloud-service SDK imports, no
`eval`/`exec`, and no shell command execution -- plus dedicated
path-traversal-rejection tests.

THE SCANNER IS PROVEN NON-VACUOUS: `TestScannerCatchesDeliberatelyBadCode`
runs every check against a small, deliberately BAD snippet engineered to
violate exactly that one rule, asserting the scanner DOES flag it -- a
scanner that always reports "clean" would pass the real-source checks
for a trivial (wrong) reason; this class rules that out."""

from __future__ import annotations

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

_FORBIDDEN_NETWORK_IMPORTS = re.compile(r"^\s*(import|from)\s+(socket|requests|httpx|aiohttp|urllib\.request|websocket|websockets|ftplib|telnetlib)\b", re.MULTILINE)
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


def _all_source_files() -> list[Path]:
    return sorted(_SRC_ROOT.glob("*.py"))


def _combined_source() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in _all_source_files())


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
        source = _combined_source()
        assert not _FORBIDDEN_NETWORK_IMPORTS.search(source)

    def test_no_broker_sdk_imports(self) -> None:
        source = _combined_source()
        assert not _FORBIDDEN_BROKER_IMPORTS.search(source)


class TestNoCredentials:
    def test_no_credential_field_declarations(self) -> None:
        source = _combined_source()
        assert not _CREDENTIAL_FIELD.search(source)


class TestNoLiveTrading:
    def test_no_live_trading_markers(self) -> None:
        source = _combined_source()
        assert not _LIVE_TRADING.search(source)


class TestNoFloatFinancialFields:
    def test_no_dataclass_field_annotated_float(self) -> None:
        for path in _all_source_files():
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
        for path in _all_source_files():
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
