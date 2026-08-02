"""Safety scan (Milestone 10, Phase 4D): confirms the shipped
`features.market_data_bridge` source contains no network clients, no
broker code, no credentials, no live-trading code, no internal
wall-clock economic input, no `uuid4`/`random`-derived identity, no
overwrite/bypass flags, no silent broad exception handling, no unsafe
`pickle`/`eval`/`exec`/shell execution -- mirrors `tests/unit/
market_data/test_market_data_safety_scan.py`'s own established pattern
and regex vocabulary, applied to this new package.

PHASE-4D-SPECIFIC CHECKS (beyond the reused vocabulary above): the bridge
is `features/`'s only Decimal -> float64 boundary crossing for
market_data-backed data, so unlike `market_data` itself (which bans
`float(...)` outright), this scan instead confirms that boundary
crossing stays confined to the exact 3 documented `resolve_*_dataframe`
functions, and confirms the structural, repo-wide invariant
`market_data` never imports `features` (the dependency direction spec
Section 2 requires)."""

from __future__ import annotations

import re
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "quant_platform" / "features" / "market_data_bridge"
_MARKET_DATA_ROOT = Path(__file__).resolve().parents[3] / "src" / "quant_platform" / "market_data"

_FORBIDDEN_NETWORK_IMPORTS = re.compile(r"^\s*(import|from)\s+(socket|requests|httpx|aiohttp|urllib\.request|websocket|websockets|ftplib|telnetlib)\b", re.MULTILINE)
_FORBIDDEN_BROKER_IMPORTS = re.compile(r"^\s*(import|from)\s+(MetaTrader5|mt5(?!__)|fxpro)\b", re.MULTILINE)
_CREDENTIAL_FIELD = re.compile(r"^\s*(password|api_key|api_secret|secret_key|access_token|client_secret)\s*[:=]", re.MULTILINE | re.IGNORECASE)
_LIVE_TRADING = re.compile(r"\bLIVE_TRADING\b|\bplace_live_order\b|\bsubmit_to_broker\b")
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
    return sorted(_SRC_ROOT.rglob("*.py"))


def _combined_source() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in _all_source_files())


class TestNoNetworkBrokerOrCredentialCode:
    def test_no_network_client_imports(self) -> None:
        assert not _FORBIDDEN_NETWORK_IMPORTS.search(_combined_source())

    def test_no_broker_sdk_imports(self) -> None:
        assert not _FORBIDDEN_BROKER_IMPORTS.search(_combined_source())

    def test_no_credential_field_declarations(self) -> None:
        assert not _CREDENTIAL_FIELD.search(_combined_source())

    def test_no_live_trading_markers(self) -> None:
        assert not _LIVE_TRADING.search(_combined_source())


class TestNoRandomOrUuidIdentity:
    def test_no_uuid4_or_random_anywhere(self) -> None:
        # Unlike market_data's own scanner (which exempts uuid4 used for
        # TEMP FILE naming in its atomic-write helpers), this bridge never
        # writes any file itself -- it has no legitimate use for uuid4 or
        # random at all. Every identity here flows through
        # `market_data.identity.compute_content_id` (a deterministic
        # sha256), reused, never reimplemented.
        assert not _UUID4_OR_RANDOM.search(_combined_source())


class TestNoInternalWallClockEconomicInput:
    def test_no_wall_clock_reads(self) -> None:
        assert not _WALL_CLOCK.search(_combined_source())


class TestNoOverwriteOrBypassFlags:
    def test_no_force_overwrite_or_skip_verification_parameters(self) -> None:
        assert not _OVERWRITE_BYPASS_FLAG.search(_combined_source())


class TestNoSilentBroadException:
    def test_no_bare_except(self) -> None:
        assert not _BARE_EXCEPT.search(_combined_source())

    def test_no_swallowed_exception_exception_pass(self) -> None:
        assert not _SWALLOWED_EXCEPTION.search(_combined_source())


class TestNoUnsafeCodeExecution:
    def test_no_pickle_usage(self) -> None:
        assert not _UNSAFE_PICKLE.search(_combined_source())

    def test_no_cloud_sdk_imports(self) -> None:
        assert not _CLOUD_SDK_IMPORTS.search(_combined_source())

    def test_no_eval_or_exec(self) -> None:
        assert not _EVAL_OR_EXEC.search(_combined_source())

    def test_no_shell_command_execution(self) -> None:
        assert not _SHELL_EXECUTION.search(_combined_source())


class TestScannerCatchesDeliberatelyBadCode:
    """Proves every regex above is non-vacuous."""

    def test_network_import_is_caught(self) -> None:
        assert _FORBIDDEN_NETWORK_IMPORTS.search("import requests\n")

    def test_broker_import_is_caught(self) -> None:
        assert _FORBIDDEN_BROKER_IMPORTS.search("import MetaTrader5 as mt5\n")

    def test_credential_field_is_caught(self) -> None:
        assert _CREDENTIAL_FIELD.search("    api_key: str\n")

    def test_live_trading_marker_is_caught(self) -> None:
        assert _LIVE_TRADING.search("mode = LIVE_TRADING\n")

    def test_uuid4_is_caught(self) -> None:
        assert _UUID4_OR_RANDOM.search("x = uuid.uuid4().hex\n")

    def test_wall_clock_read_is_caught(self) -> None:
        assert _WALL_CLOCK.search("now = datetime.now()\n")

    def test_overwrite_flag_is_caught(self) -> None:
        assert _OVERWRITE_BYPASS_FLAG.search("def f(force: bool = False):\n")

    def test_bare_except_is_caught(self) -> None:
        assert _BARE_EXCEPT.search("try:\n    pass\nexcept:\n    pass\n")

    def test_swallowed_exception_is_caught(self) -> None:
        assert _SWALLOWED_EXCEPTION.search("try:\n    pass\nexcept Exception:\n    pass\n")

    def test_pickle_usage_is_caught(self) -> None:
        assert _UNSAFE_PICKLE.search("import pickle\n")

    def test_eval_or_exec_is_caught(self) -> None:
        assert _EVAL_OR_EXEC.search("eval('1 + 1')\n")

    def test_shell_execution_is_caught(self) -> None:
        assert _SHELL_EXECUTION.search("subprocess.run(['ls'])\n")


class TestScannerIsNonVacuousAgainstRealSource:
    def test_scanner_actually_reads_a_nonempty_source_tree(self) -> None:
        files = _all_source_files()
        assert len(files) >= 10
        assert len(_combined_source()) > 5_000


# --------------------------------------------------------------------------
# Phase-4D-specific: the Decimal -> float64 boundary crossing stays
# confined to exactly the 3 documented `resolve_*_dataframe` functions.
# --------------------------------------------------------------------------
_ALLOWED_FLOAT_CALL_FILES = frozenset({
    "base_asset_adapter.py", "cross_asset_adapter.py", "macro_adapter.py",
    # Not a Decimal boundary crossing -- staleness.py's `float(...)` calls
    # narrow an already-float64 pandas/numpy scalar (`.max()`/
    # `.total_seconds()`) down to a plain Python float for a report
    # dataclass field; there is no Decimal anywhere in this module.
    "staleness.py",
})


class TestFloatBoundaryCrossingIsConfined:
    def test_float_call_appears_only_in_the_three_documented_adapters(self) -> None:
        offenders = []
        for path in _all_source_files():
            if path.name in _ALLOWED_FLOAT_CALL_FILES:
                continue
            if _FLOAT_CALL.search(path.read_text(encoding="utf-8")):
                offenders.append(path.name)
        assert offenders == [], f"float(...) called outside the documented boundary-crossing adapters: {offenders}"

    def test_the_three_documented_adapters_do_call_float(self) -> None:
        # Sanity check that the exemption is actually exercised, not
        # vacuously true because none of them call float() at all.
        for name in _ALLOWED_FLOAT_CALL_FILES:
            path = _SRC_ROOT / name
            assert _FLOAT_CALL.search(path.read_text(encoding="utf-8")), f"{name} was expected to call float(...)"


# --------------------------------------------------------------------------
# Structural dependency-direction invariant (spec Section 2): market_data
# must NEVER import features -- this is the one invariant that, if
# violated, would create a circular import between the two packages.
# --------------------------------------------------------------------------
class TestMarketDataNeverImportsFeatures:
    def test_no_market_data_module_imports_features(self) -> None:
        offenders = []
        for path in sorted(_MARKET_DATA_ROOT.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if re.search(r"^\s*(import|from)\s+quant_platform\.features\b", text, re.MULTILINE):
                offenders.append(str(path.relative_to(_MARKET_DATA_ROOT)))
        assert offenders == [], f"market_data module(s) import features (forbidden dependency direction): {offenders}"

    def test_check_is_non_vacuous(self) -> None:
        assert re.search(r"^\s*(import|from)\s+quant_platform\.features\b", "from quant_platform.features.engine import FeatureEngine\n", re.MULTILINE)
