"""Opt-in real-FRED acceptance workflow tests (Milestone 10, Phase 4B).

The ONE ordinary test in this file that would touch the real network
(`TestOptInLiveAcceptance.test_live_workflow_when_key_present`) resolves
its key via `resolve_fred_api_key_from_environment` and calls
`pytest.skip(...)` with a precise reason when absent -- ZERO network
calls are attempted in that case, and a missing credential is never
treated as a test failure. Every other test here exercises pure,
no-network logic (key resolution rules, the empty-key guard clause,
`RedactedAcceptanceReport`'s structural inability to carry a secret)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from quant_platform.market_data.collectors.curated.acceptance import (
    FRED_API_KEY_ENV_VAR,
    RedactedAcceptanceReport,
    resolve_fred_api_key_from_environment,
    run_real_fred_acceptance_workflow,
)


class TestResolveApiKeyFromEnvironment:
    def test_absent_key_returns_none(self) -> None:
        assert resolve_fred_api_key_from_environment({}) is None

    def test_blank_key_returns_none(self) -> None:
        assert resolve_fred_api_key_from_environment({FRED_API_KEY_ENV_VAR: "   "}) is None

    def test_present_key_is_returned_stripped(self) -> None:
        assert resolve_fred_api_key_from_environment({FRED_API_KEY_ENV_VAR: "  abc123  "}) == "abc123"

    def test_never_raises_on_missing_mapping_entry(self) -> None:
        assert resolve_fred_api_key_from_environment({"UNRELATED_VAR": "x"}) is None

    def test_reads_real_os_environ_by_default_when_no_mapping_supplied(self) -> None:
        """This is the ONE sanctioned place in `collectors/` that reads
        an environment variable -- confirm the default `env=None` path
        actually consults `os.environ` rather than silently returning
        `None` unconditionally (which would make every downstream
        "key present" test path unreachable in a real environment)."""
        import os

        marker = "unit-test-marker-value-12345"
        os.environ[FRED_API_KEY_ENV_VAR] = marker
        try:
            assert resolve_fred_api_key_from_environment() == marker
        finally:
            del os.environ[FRED_API_KEY_ENV_VAR]


class TestEmptyKeyGuardClauseNeedsNoNetwork:
    def test_empty_string_api_key_raises_before_any_network_use(self) -> None:
        with pytest.raises(ValueError, match="non-empty api_key"):
            run_real_fred_acceptance_workflow(
                api_key="", repository=None, cache=None, transport=None, retry_policy=None,  # type: ignore[arg-type]
                rate_limit_policy=None, rate_limit_state=None, revision_policy=None, availability_policies=None,  # type: ignore[arg-type]
                observation_start=None, observation_end=None, operation_time=None,  # type: ignore[arg-type]
            )

    def test_whitespace_only_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty api_key"):
            run_real_fred_acceptance_workflow(
                api_key="   ", repository=None, cache=None, transport=None, retry_policy=None,  # type: ignore[arg-type]
                rate_limit_policy=None, rate_limit_state=None, revision_policy=None, availability_policies=None,  # type: ignore[arg-type]
                observation_start=None, observation_end=None, operation_time=None,  # type: ignore[arg-type]
            )


class TestRedactedAcceptanceReportStructurallyCannotCarryASecret:
    def test_no_field_name_resembles_a_credential(self) -> None:
        suspicious = {"key", "api_key", "secret", "token", "credential", "password", "url", "headers"}
        field_names = {f.name for f in dataclasses.fields(RedactedAcceptanceReport)}
        assert field_names.isdisjoint(suspicious)

    def test_to_json_dict_of_a_constructed_report_contains_no_secret_shaped_value(self) -> None:
        fake_secret = "sk-should-never-appear-anywhere-1234567890"
        report = RedactedAcceptanceReport(
            ran=True, series_checked=("DGS10", "DFF", "DFII10", "CPIAUCSL"), backfill_stage="completed", backfill_completeness_status="complete",
            committed_observation_counts={"DGS10": 1, "DFF": 1, "DFII10": 1, "CPIAUCSL": 1}, reconciliation_critical_count=0,
            verification_critical_count=0, replay_identical=True, notes=(),
        )
        text = json.dumps(report.to_json_dict())
        assert fake_secret not in text
        assert "api_key" not in text.lower()


class TestOptInLiveAcceptance:
    def test_live_workflow_when_key_present(self) -> None:
        """The mandated opt-in acceptance path itself. Skips cleanly
        with a precise reason and makes ZERO network calls whenever
        `FRED_API_KEY` is absent from the real environment -- which is
        the expected state for this offline development environment and
        for the ordinary CI run of the full suite. When a key IS
        present, this genuinely exercises the real HTTPS transport
        against the official FRED API over a small, bounded interval."""
        api_key = resolve_fred_api_key_from_environment()
        if api_key is None:
            pytest.skip(f"{FRED_API_KEY_ENV_VAR} is not set in the environment -- opt-in real-FRED acceptance workflow skipped, zero network calls attempted")

        import tempfile
        from datetime import datetime, timezone
        from decimal import Decimal
        from pathlib import Path

        from quant_platform.market_data.collectors.cache import RawResponseCache
        from quant_platform.market_data.collectors.curated.availability import (
            AvailabilityPolicyKind,
            create_availability_policy,
        )
        from quant_platform.market_data.collectors.curated.revision_policy import (
            RevisionPolicyKind,
            create_revision_policy,
        )
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
        daily_policy = create_availability_policy(kind=AvailabilityPolicyKind.OBSERVATION_DATE_END_OF_DAY, timezone_key="America/New_York", availability_hour=17, availability_minute=0)
        monthly_policy = create_availability_policy(kind=AvailabilityPolicyKind.REALTIME_START_DATE_CONSERVATIVE, timezone_key="America/New_York", availability_hour=8, availability_minute=30)
        availability_policies = {"DFII10": daily_policy, "DGS10": daily_policy, "DFF": daily_policy, "CPIAUCSL": monthly_policy}
        revision_policy = create_revision_policy(kind=RevisionPolicyKind.LATEST_AVAILABLE)
        retry_policy = create_retry_policy(max_attempts=3, backoff_schedule_seconds=(1.0, 2.0))
        rate_limit_policy = create_rate_limit_policy(max_tokens=Decimal(10), refill_rate_per_second=Decimal(2))

        result = run_real_fred_acceptance_workflow(
            api_key=api_key, repository=repository, cache=cache, transport=StdlibHttpsTransport(), retry_policy=retry_policy,
            rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=operation_time),
            revision_policy=revision_policy, availability_policies=availability_policies,
            observation_start=datetime(2024, 1, 1, tzinfo=timezone.utc), observation_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            operation_time=operation_time, page_size=50,
        )
        assert result.ran is True
        text = json.dumps(result.to_json_dict())
        assert api_key not in text
