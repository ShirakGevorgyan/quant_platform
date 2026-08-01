"""Tests for the `AlphaVantageCollector` adapter (Milestone 10, Phase
4C, spec Section 30 "Provider selection")."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _cross_asset_test_helpers import alpha_vantage_daily_body, alpha_vantage_error_body

from quant_platform.core.exceptions import MarketProviderResponseError
from quant_platform.market_data.collectors.cross_asset.instrument_form import InstrumentForm
from quant_platform.market_data.collectors.cross_asset.providers.alpha_vantage import (
    ALPHA_VANTAGE_ALLOWED_HOSTS,
    ALPHA_VANTAGE_COLLECTOR_NAME,
    ALPHA_VANTAGE_ENDPOINT_HOST,
    AlphaVantageCollector,
    parse_alpha_vantage_daily_metadata,
    parse_alpha_vantage_daily_records,
)
from quant_platform.market_data.collectors.request_manifest import CredentialMode

T0 = datetime(2024, 1, 5, tzinfo=timezone.utc)


class TestAlphaVantageParsing:
    def test_valid_response_parses_records_ascending(self) -> None:
        body = alpha_vantage_daily_body(symbol="GLD", rows={
            "2024-01-05": {"1. open": "191.90", "2. high": "193.00", "3. low": "191.50", "4. close": "192.50", "5. volume": "1100000"},
            "2024-01-03": {"1. open": "190.00", "2. high": "191.50", "3. low": "189.80", "4. close": "191.00", "5. volume": "1000000"},
            "2024-01-04": {"1. open": "191.10", "2. high": "192.00", "3. low": "190.50", "4. close": "191.80", "5. volume": "900000"},
        })
        records = parse_alpha_vantage_daily_records(body, provider_symbol="GLD")
        assert [r.provider_timestamp_text for r in records] == ["2024-01-03", "2024-01-04", "2024-01-05"]
        assert [r.source_sequence for r in records] == [0, 1, 2]

    def test_metadata_parses(self) -> None:
        body = alpha_vantage_daily_body(symbol="GLD", rows={"2024-01-05": {"1. open": "1", "2. high": "1", "3. low": "1", "4. close": "1", "5. volume": "1"}})
        metadata = parse_alpha_vantage_daily_metadata(body, canonical_driver_id="gold_reference", instrument_form=InstrumentForm.ETF)
        assert metadata.provider_symbol == "GLD"
        assert metadata.provider == ALPHA_VANTAGE_COLLECTOR_NAME
        assert metadata.canonical_driver_id == "gold_reference"
        assert metadata.adjustment_supported is False

    def test_note_error_envelope_fails_closed(self) -> None:
        body = alpha_vantage_error_body(key="Note", message="rate limit exceeded")
        with pytest.raises(MarketProviderResponseError):
            parse_alpha_vantage_daily_records(body, provider_symbol="GLD")

    def test_error_message_envelope_fails_closed(self) -> None:
        body = alpha_vantage_error_body(key="Error Message", message="Invalid API call.")
        with pytest.raises(MarketProviderResponseError):
            parse_alpha_vantage_daily_records(body, provider_symbol="GLD")

    def test_information_envelope_fails_closed(self) -> None:
        body = alpha_vantage_error_body(key="Information", message="demo key restricted to IBM")
        with pytest.raises(MarketProviderResponseError):
            parse_alpha_vantage_daily_records(body, provider_symbol="GLD")

    def test_missing_time_series_key_fails_closed(self) -> None:
        import json

        body = json.dumps({"Meta Data": {"1. Information": "x", "2. Symbol": "GLD", "3. Last Refreshed": "x", "4. Output Size": "x", "5. Time Zone": "x"}}).encode()
        with pytest.raises(MarketProviderResponseError):
            parse_alpha_vantage_daily_records(body, provider_symbol="GLD")

    def test_missing_meta_data_key_fails_closed(self) -> None:
        import json

        body = json.dumps({"Time Series (Daily)": {}}).encode()
        with pytest.raises(MarketProviderResponseError):
            parse_alpha_vantage_daily_metadata(body, canonical_driver_id="gold_reference", instrument_form=InstrumentForm.ETF)

    def test_non_json_body_fails_closed(self) -> None:
        with pytest.raises(MarketProviderResponseError):
            parse_alpha_vantage_daily_records(b"not json at all", provider_symbol="GLD")

    def test_missing_row_key_fails_closed(self) -> None:
        import json

        body = json.dumps({
            "Meta Data": {"1. Information": "x", "2. Symbol": "GLD", "3. Last Refreshed": "x", "4. Output Size": "x", "5. Time Zone": "x"},
            "Time Series (Daily)": {"2024-01-05": {"1. open": "1", "2. high": "1"}},
        }).encode()
        with pytest.raises(MarketProviderResponseError):
            parse_alpha_vantage_daily_records(body, provider_symbol="GLD")

    def test_never_produces_float(self) -> None:
        body = alpha_vantage_daily_body(symbol="GLD", rows={"2024-01-05": {"1. open": "191.123456789", "2. high": "193.00", "3. low": "189.80", "4. close": "192.50", "5. volume": "1"}})
        records = parse_alpha_vantage_daily_records(body, provider_symbol="GLD")
        assert records[0].open_text == "191.123456789"
        assert isinstance(records[0].open_text, str)


class TestAlphaVantageCollector:
    def test_capabilities_declare_etf_and_equity_only(self) -> None:
        collector = AlphaVantageCollector()
        capabilities = collector.supported_capabilities()
        assert InstrumentForm.ETF in capabilities.supported_instrument_forms
        assert InstrumentForm.EXCHANGE_FUTURES_CONTRACT not in capabilities.supported_instrument_forms
        assert capabilities.futures_contracts_supported is False
        assert capabilities.continuous_futures_supported is False
        assert capabilities.adjusted_data_supported is False
        assert capabilities.unadjusted_data_supported is True

    def test_metadata_and_history_requests_are_identical(self) -> None:
        """Documented honest provider limitation -- one endpoint serves
        both metadata and history."""
        collector = AlphaVantageCollector()
        meta_req = collector.build_metadata_request(provider_symbol="GLD", request_time=T0, credential_mode=CredentialMode.API_KEY)
        hist_req = collector.build_history_request(provider_symbol="GLD", granularity="1d", request_time=T0, credential_mode=CredentialMode.API_KEY)
        assert meta_req.request_manifest_id == hist_req.request_manifest_id

    def test_unsupported_granularity_rejected(self) -> None:
        collector = AlphaVantageCollector()
        with pytest.raises(MarketProviderResponseError):
            collector.build_history_request(provider_symbol="GLD", granularity="1h", request_time=T0, credential_mode=CredentialMode.API_KEY)

    def test_request_manifest_never_contains_api_key(self) -> None:
        collector = AlphaVantageCollector()
        request = collector.build_metadata_request(provider_symbol="GLD", request_time=T0, credential_mode=CredentialMode.API_KEY)
        assert "api_key" not in request.canonical_query_params
        assert ALPHA_VANTAGE_ENDPOINT_HOST in ALPHA_VANTAGE_ALLOWED_HOSTS

    def test_full_parse_round_trip_through_collector(self) -> None:
        collector = AlphaVantageCollector()
        body = alpha_vantage_daily_body(symbol="GLD", rows={"2024-01-05": {"1. open": "191.00", "2. high": "193.00", "3. low": "189.80", "4. close": "192.50", "5. volume": "1000000"}})
        metadata = collector.parse_metadata_response(body, provider_symbol="GLD", canonical_driver_id="gold_reference", instrument_form=InstrumentForm.ETF)
        assert metadata.provider_symbol == "GLD"
        response_manifest_stub = type("Stub", (), {"response_manifest_id": "x"})()
        records = collector.parse_history_response(body, provider_symbol="GLD", response_manifest=response_manifest_stub)  # type: ignore[arg-type]
        assert len(records) == 1
