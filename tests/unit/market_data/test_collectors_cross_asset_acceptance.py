"""Opt-in real-Alpha-Vantage acceptance workflow tests (Milestone 10,
Phase 4C).

The ONE ordinary test in this file that would touch the real network
(`TestOptInLiveAcceptance.test_live_workflow_when_key_present`) resolves
its key via `resolve_alpha_vantage_api_key_from_environment` and calls
`pytest.skip(...)` with a precise reason when absent -- ZERO network
calls are attempted in that case, and a missing credential is never
treated as a test failure. Every other test here exercises pure,
no-network logic (key resolution rules, the empty-key guard clause,
`RedactedCrossAssetAcceptanceReport`'s structural inability to carry a
secret)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from quant_platform.market_data.collectors.cross_asset.acceptance import (
    ALPHA_VANTAGE_ACCEPTANCE_DRIVER_ID,
    ALPHA_VANTAGE_ACCEPTANCE_SYMBOL,
    RedactedCrossAssetAcceptanceReport,
    resolve_alpha_vantage_api_key_from_environment,
    run_real_alpha_vantage_acceptance_workflow,
)
from quant_platform.market_data.collectors.cross_asset.providers.alpha_vantage import (
    ALPHA_VANTAGE_API_KEY_ENV_VAR,
)


class TestResolveApiKeyFromEnvironment:
    def test_absent_key_returns_none(self) -> None:
        assert resolve_alpha_vantage_api_key_from_environment({}) is None

    def test_blank_key_returns_none(self) -> None:
        assert resolve_alpha_vantage_api_key_from_environment({ALPHA_VANTAGE_API_KEY_ENV_VAR: "   "}) is None

    def test_present_key_is_returned_stripped(self) -> None:
        assert resolve_alpha_vantage_api_key_from_environment({ALPHA_VANTAGE_API_KEY_ENV_VAR: "  abc123  "}) == "abc123"

    def test_never_raises_on_missing_mapping_entry(self) -> None:
        assert resolve_alpha_vantage_api_key_from_environment({"UNRELATED_VAR": "x"}) is None

    def test_reads_real_os_environ_by_default_when_no_mapping_supplied(self) -> None:
        """This is the ONE sanctioned place in `collectors/cross_asset/`
        that reads an environment variable -- confirm the default
        `env=None` path actually consults `os.environ` rather than
        silently returning `None` unconditionally."""
        import os

        marker = "unit-test-marker-value-12345"
        os.environ[ALPHA_VANTAGE_API_KEY_ENV_VAR] = marker
        try:
            assert resolve_alpha_vantage_api_key_from_environment() == marker
        finally:
            del os.environ[ALPHA_VANTAGE_API_KEY_ENV_VAR]


class TestEmptyKeyGuardClauseNeedsNoNetwork:
    def test_empty_string_api_key_raises_before_any_network_use(self) -> None:
        with pytest.raises(ValueError, match="non-empty api_key"):
            run_real_alpha_vantage_acceptance_workflow(
                api_key="", repository=None, cache=None, transport=None, retry_policy=None,  # type: ignore[arg-type]
                rate_limit_policy=None, rate_limit_state=None, operation_time=None,  # type: ignore[arg-type]
            )

    def test_whitespace_only_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty api_key"):
            run_real_alpha_vantage_acceptance_workflow(
                api_key="   ", repository=None, cache=None, transport=None, retry_policy=None,  # type: ignore[arg-type]
                rate_limit_policy=None, rate_limit_state=None, operation_time=None,  # type: ignore[arg-type]
            )


class TestRedactedAcceptanceReportStructurallyCannotCarryASecret:
    def test_no_field_name_resembles_a_credential(self) -> None:
        suspicious = {"key", "api_key", "secret", "token", "credential", "password", "url", "headers"}
        field_names = {f.name for f in dataclasses.fields(RedactedCrossAssetAcceptanceReport)}
        assert field_names.isdisjoint(suspicious)

    def test_to_json_dict_of_a_constructed_report_contains_no_secret_shaped_value(self) -> None:
        fake_secret = "sk-should-never-appear-anywhere-1234567890"
        report = RedactedCrossAssetAcceptanceReport(
            ran=True, mappings_checked=("m" * 64,), backfill_stage="completed", backfill_completeness_status="complete",
            committed_bar_counts={"m" * 64: 5}, reconciliation_critical_count=0, verification_critical_count=0, replay_identical=True, notes=(),
        )
        text = json.dumps(report.to_json_dict())
        assert fake_secret not in text
        assert "api_key" not in text.lower()


class TestOptInLiveAcceptance:
    def test_live_workflow_when_key_present(self) -> None:
        """The mandated opt-in acceptance path itself. Skips cleanly with
        a precise reason and makes ZERO network calls whenever
        `ALPHA_VANTAGE_API_KEY` is absent from the real environment --
        the expected state for this offline development environment and
        for the ordinary CI run of the full suite. When a key IS
        present, this genuinely exercises the real HTTPS transport
        against the official Alpha Vantage API over the ONE bounded,
        real-verifiable mapping (spec Section 24's own "supported
        subset" allowance -- `gold_reference` via `GLD` only)."""
        api_key = resolve_alpha_vantage_api_key_from_environment()
        if api_key is None:
            pytest.skip(
                f"{ALPHA_VANTAGE_API_KEY_ENV_VAR} is not set in the environment -- opt-in real-Alpha-Vantage acceptance workflow skipped, zero network calls attempted"
            )

        import tempfile
        from datetime import datetime, timezone
        from decimal import Decimal
        from pathlib import Path

        from quant_platform.market_data.collectors.cache import RawResponseCache
        from quant_platform.market_data.collectors.rate_limit import (
            create_rate_limit_policy,
            initial_bucket_state,
        )
        from quant_platform.market_data.collectors.retry import create_retry_policy
        from quant_platform.market_data.collectors.transport import StdlibHttpsTransport
        from quant_platform.market_data.repository import MarketDataRepository

        root = Path(tempfile.mkdtemp())
        repository = MarketDataRepository.open(root)
        cache = RawResponseCache(root)
        operation_time = datetime.now(tz=timezone.utc)
        retry_policy = create_retry_policy(max_attempts=3, backoff_schedule_seconds=(1.0, 2.0))
        rate_limit_policy = create_rate_limit_policy(max_tokens=Decimal(5), refill_rate_per_second=Decimal(1))

        result = run_real_alpha_vantage_acceptance_workflow(
            api_key=api_key, repository=repository, cache=cache, transport=StdlibHttpsTransport(), retry_policy=retry_policy,
            rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=operation_time),
            operation_time=operation_time,
        )
        assert result.ran is True
        assert result.mappings_checked
        text = json.dumps(result.to_json_dict())
        assert api_key not in text
        assert ALPHA_VANTAGE_ACCEPTANCE_DRIVER_ID
        assert ALPHA_VANTAGE_ACCEPTANCE_SYMBOL == "GLD"
