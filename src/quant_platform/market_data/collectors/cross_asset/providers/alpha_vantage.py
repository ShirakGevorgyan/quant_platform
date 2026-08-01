"""Alpha Vantage `HistoricalMarketCollector` adapter (Milestone 10,
Phase 4C) -- the ONE concrete provider adapter this phase ships.

BOUNDED PROVIDER-SELECTION DECISION (spec Section 2), summarized here
and recorded in full in `docs/milestone10_phase4c_delivery_report.md`:
three candidates were assessed against their OFFICIAL documentation
(`https://www.alphavantage.co/documentation/`, `https://api.eia.gov/v2/`,
and Stooq's own community pages). Stooq was DISQUALIFIED outright --
independently confirmed to have no official, documented API, only
reverse-engineered CSV endpoints (exactly the "undocumented endpoint
guessing" this phase's spec forbids). Both Alpha Vantage and the EIA
Open Data API are officially documented; Alpha Vantage's `TIME_SERIES_
DAILY` endpoint was additionally LIVE-VERIFIED (a real HTTP GET against
the provider's own public `demo` key returned the exact JSON shape this
adapter parses below) -- the strongest form of "verified documented
behavior" available without registering a real account on the user's
behalf, which this assistant will not do autonomously. Alpha Vantage's
dedicated commodities endpoints (`WTI`/`BRENT`/`GOLD_SILVER_SPOT`) and
the EIA petroleum-price route exist in official documentation but could
NOT be live-verified without a registered API key (the `demo` key is
restricted to a small, fixed set of demo symbols), so THIS ADAPTER
IMPLEMENTS ONLY THE ONE ENDPOINT THAT WAS GENUINELY, LIVE-VERIFIED --
`TIME_SERIES_DAILY` -- used exclusively for ETF-form PROXY instruments
mapping to the curated cross-asset concepts (see `registry.py`'s own
default mappings). Every driver mapped through this adapter is honestly
classified `is_proxy=True`; this adapter never claims direct spot/
futures/index coverage it cannot back with verified behavior.

EXACT VERIFIED SCHEMA (Alpha Vantage's own JSON, confirmed via a real
HTTP call, 2026): `{"Meta Data": {"1. Information": ..., "2. Symbol":
..., "3. Last Refreshed": ..., "4. Output Size": ..., "5. Time Zone":
...}, "Time Series (Daily)": {"<YYYY-MM-DD>": {"1. open": ..., "2.
high": ..., "3. low": ..., "4. close": ..., "5. volume": ...}, ...}}`.
Documented as RAW (as-traded) daily OHLCV -- never adjusted; this
adapter's own capabilities declare `adjusted_data_supported=False`.
Free-tier rate limit, confirmed via the provider's own pricing page:
25 requests/day -- `MarketCollectorCapabilities.max_rows_per_page`/
`max_interval_days_per_request` are both `None` (one call returns the
provider's ENTIRE available history when `outputsize=full` is
requested; there is no server-side date-range filter to request a
narrower window)."""

from __future__ import annotations

from datetime import datetime

from quant_platform.core.exceptions import MarketProviderResponseError
from quant_platform.core.json import parse_json_strict
from quant_platform.market_data.collectors.cross_asset.instrument_form import InstrumentForm
from quant_platform.market_data.collectors.cross_asset.market_record import RawMarketRecord
from quant_platform.market_data.collectors.cross_asset.protocols import (
    MarketCollectorCapabilities,
    ProviderMetadataRecord,
)
from quant_platform.market_data.collectors.request_manifest import (
    CollectorRequestManifest,
    CredentialMode,
    create_request_manifest,
)
from quant_platform.market_data.collectors.response_manifest import (
    CollectorResponseManifest,
    compute_raw_content_digest,
)
from quant_platform.market_data.identity import require_non_empty

__all__ = [
    "ALPHA_VANTAGE_ALLOWED_HOSTS",
    "ALPHA_VANTAGE_API_KEY_ENV_VAR",
    "ALPHA_VANTAGE_COLLECTOR_NAME",
    "ALPHA_VANTAGE_COLLECTOR_VERSION",
    "ALPHA_VANTAGE_ENDPOINT_HOST",
    "ALPHA_VANTAGE_ENDPOINT_PATH",
    "AlphaVantageCollector",
    "build_alpha_vantage_daily_request_manifest",
    "parse_alpha_vantage_daily_metadata",
    "parse_alpha_vantage_daily_records",
]

ALPHA_VANTAGE_COLLECTOR_NAME = "alpha_vantage"
ALPHA_VANTAGE_COLLECTOR_VERSION = "1.0.0"
ALPHA_VANTAGE_ENDPOINT_HOST = "www.alphavantage.co"
ALPHA_VANTAGE_ENDPOINT_PATH = "/query"
ALPHA_VANTAGE_ALLOWED_HOSTS = frozenset({ALPHA_VANTAGE_ENDPOINT_HOST})
ALPHA_VANTAGE_API_KEY_ENV_VAR = "ALPHA_VANTAGE_API_KEY"

_TIME_SERIES_DAILY = "TIME_SERIES_DAILY"


def build_alpha_vantage_daily_request_manifest(
    *, provider_symbol: str, timeout_policy_id: str, retry_policy_id: str, rate_limit_policy_id: str, request_time: datetime,
    collector_version: str = ALPHA_VANTAGE_COLLECTOR_VERSION,
) -> CollectorRequestManifest:
    """The ONE request this adapter ever builds -- `outputsize=full`
    (the provider's entire available history; there is no server-side
    date-range filter to request a narrower window) and
    `datatype=json` are ALWAYS included explicitly, never left to an
    undocumented default."""
    query_params = {"function": _TIME_SERIES_DAILY, "symbol": provider_symbol, "outputsize": "full", "datatype": "json"}
    return create_request_manifest(
        collector_name=ALPHA_VANTAGE_COLLECTOR_NAME, collector_version=collector_version, endpoint_host=ALPHA_VANTAGE_ENDPOINT_HOST,
        endpoint_path=ALPHA_VANTAGE_ENDPOINT_PATH, canonical_query_params=query_params, canonical_headers={},
        requested_series_or_dataset=provider_symbol, response_format="json", timeout_policy_id=timeout_policy_id, retry_policy_id=retry_policy_id,
        rate_limit_policy_id=rate_limit_policy_id, credential_mode=CredentialMode.API_KEY, request_time=request_time,
    )


def _parse_daily_envelope(raw_bytes: bytes, *, encoding: str = "utf-8") -> dict[str, object]:
    try:
        text = raw_bytes.decode(encoding)
    except UnicodeDecodeError as exc:
        raise MarketProviderResponseError(f"Alpha Vantage TIME_SERIES_DAILY response is not valid {encoding}: {exc}") from exc
    try:
        parsed = parse_json_strict(text)
    except ValueError as exc:
        raise MarketProviderResponseError(f"Alpha Vantage TIME_SERIES_DAILY response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise MarketProviderResponseError(f"Alpha Vantage TIME_SERIES_DAILY response must decode to a JSON object, got {type(parsed).__name__}")
    # Alpha Vantage's OWN documented error/rate-limit/demo-restriction shape is a bare
    # `{"Information": "..."}`/`{"Error Message": "..."}`/`{"Note": "..."}` object in place
    # of the expected data envelope -- fail closed rather than silently treating it as an
    # empty (zero-row) but otherwise valid response.
    for error_key in ("Information", "Error Message", "Note"):
        if error_key in parsed and "Time Series (Daily)" not in parsed:
            raise MarketProviderResponseError(f"Alpha Vantage returned a {error_key!r} envelope instead of TIME_SERIES_DAILY data: {parsed[error_key]!r}")
    if "Meta Data" not in parsed or "Time Series (Daily)" not in parsed:
        raise MarketProviderResponseError("Alpha Vantage TIME_SERIES_DAILY response is missing 'Meta Data'/'Time Series (Daily)' -- not the expected schema")
    return parsed


def parse_alpha_vantage_daily_metadata(raw_bytes: bytes, *, canonical_driver_id: str, instrument_form: InstrumentForm) -> ProviderMetadataRecord:
    """Alpha Vantage exposes no SEPARATE metadata endpoint for
    `TIME_SERIES_DAILY` -- metadata and data arrive in ONE response (see
    module docstring); this parses the SAME raw bytes `parse_alpha_
    vantage_daily_records` parses, from the `Meta Data` envelope only."""
    parsed = _parse_daily_envelope(raw_bytes)
    meta_raw = parsed["Meta Data"]
    if not isinstance(meta_raw, dict):
        raise MarketProviderResponseError(f"Alpha Vantage 'Meta Data' must be a JSON object, got {type(meta_raw).__name__}")
    required = ("1. Information", "2. Symbol", "3. Last Refreshed", "4. Output Size", "5. Time Zone")
    missing = [k for k in required if k not in meta_raw]
    if missing:
        raise MarketProviderResponseError(f"Alpha Vantage 'Meta Data' is missing required key(s): {missing}")
    symbol = str(meta_raw["2. Symbol"])
    timezone_text = str(meta_raw["5. Time Zone"])
    digest = compute_raw_content_digest(raw_bytes)
    return ProviderMetadataRecord(
        provider=ALPHA_VANTAGE_COLLECTOR_NAME, provider_symbol=symbol, canonical_driver_id=canonical_driver_id,
        provider_instrument_name=None, asset_class=None, instrument_form=instrument_form, exchange_or_venue=None, currency=None,
        quote_unit=None, timezone_key=timezone_text, supported_intervals=("1d",), first_available_timestamp=None, last_available_timestamp=None,
        adjustment_supported=False, provider_metadata_digest=digest,
    )


def parse_alpha_vantage_daily_records(raw_bytes: bytes, *, provider_symbol: str) -> tuple[RawMarketRecord, ...]:
    parsed = _parse_daily_envelope(raw_bytes)
    series_raw = parsed["Time Series (Daily)"]
    if not isinstance(series_raw, dict):
        raise MarketProviderResponseError(f"Alpha Vantage 'Time Series (Daily)' must be a JSON object, got {type(series_raw).__name__}")
    required_row_keys = ("1. open", "2. high", "3. low", "4. close", "5. volume")
    records: list[RawMarketRecord] = []
    # Sorted ascending by date -- Alpha Vantage's own response order is descending
    # (most recent first); a stable, deterministic ascending order is this adapter's own
    # canonical `source_sequence` assignment, never left to depend on response ordering.
    for sequence, date_text in enumerate(sorted(series_raw.keys())):
        row_raw = series_raw[date_text]
        if not isinstance(row_raw, dict):
            raise MarketProviderResponseError(f"Alpha Vantage 'Time Series (Daily)'[{date_text!r}] must be a JSON object, got {type(row_raw).__name__}")
        missing = [k for k in required_row_keys if k not in row_raw]
        if missing:
            raise MarketProviderResponseError(f"Alpha Vantage 'Time Series (Daily)'[{date_text!r}] is missing required key(s): {missing}")
        records.append(RawMarketRecord(
            provider=ALPHA_VANTAGE_COLLECTOR_NAME, provider_symbol=provider_symbol, provider_timestamp_text=date_text, interval="1d",
            open_text=str(row_raw["1. open"]), high_text=str(row_raw["2. high"]), low_text=str(row_raw["3. low"]), close_text=str(row_raw["4. close"]),
            volume_text=str(row_raw["5. volume"]), adjusted_close_text=None, trade_count_text=None, source_sequence=sequence, contract_symbol=None,
        ))
    return tuple(records)


class AlphaVantageCollector:
    """Implements `protocols.HistoricalMarketCollector` structurally
    (no explicit base class needed -- see that module's own `Protocol`
    docstring)."""

    def __init__(self, *, timeout_policy_id: str = "0" * 64, retry_policy_id: str = "0" * 64, rate_limit_policy_id: str = "0" * 64) -> None:
        require_non_empty(timeout_policy_id, field_name="timeout_policy_id")
        self._timeout_policy_id = timeout_policy_id
        self._retry_policy_id = retry_policy_id
        self._rate_limit_policy_id = rate_limit_policy_id

    def provider_metadata(self) -> MarketCollectorCapabilities:
        return self.supported_capabilities()

    def supported_capabilities(self) -> MarketCollectorCapabilities:
        return MarketCollectorCapabilities(
            provider=ALPHA_VANTAGE_COLLECTOR_NAME, candles_supported=True, quotes_supported=False, trades_supported=False,
            adjusted_data_supported=False, unadjusted_data_supported=True, corporate_actions_supported=False, futures_contracts_supported=False,
            continuous_futures_supported=False, pagination_supported=False, anonymous_access_supported=False, runtime_credential_required=True,
            max_interval_days_per_request=None, max_rows_per_page=None, supported_granularities=("1d",),
            supported_instrument_forms=(InstrumentForm.ETF, InstrumentForm.EQUITY),
        )

    def build_metadata_request(self, *, provider_symbol: str, request_time: datetime, credential_mode: CredentialMode) -> CollectorRequestManifest:
        """Alpha Vantage has no separate metadata endpoint for daily
        series -- this returns the IDENTICAL manifest `build_history_
        request` does, so orchestration fetches ONCE and derives both
        metadata and records from the same cached response (see module
        docstring)."""
        assert credential_mode is CredentialMode.API_KEY
        return build_alpha_vantage_daily_request_manifest(
            provider_symbol=provider_symbol, timeout_policy_id=self._timeout_policy_id, retry_policy_id=self._retry_policy_id,
            rate_limit_policy_id=self._rate_limit_policy_id, request_time=request_time,
        )

    def build_history_request(self, *, provider_symbol: str, granularity: str, request_time: datetime, credential_mode: CredentialMode) -> CollectorRequestManifest:
        if granularity != "1d":
            raise MarketProviderResponseError(f"AlphaVantageCollector only supports granularity='1d', got {granularity!r}")
        return self.build_metadata_request(provider_symbol=provider_symbol, request_time=request_time, credential_mode=credential_mode)

    def parse_metadata_response(
        self, raw_bytes: bytes, *, provider_symbol: str, canonical_driver_id: str, instrument_form: InstrumentForm,  # noqa: ARG002 -- `provider_symbol` is part of the `HistoricalMarketCollector` Protocol shape; this adapter derives the symbol from the response body itself instead (see `parse_alpha_vantage_daily_metadata`'s own docstring)
    ) -> ProviderMetadataRecord:
        return parse_alpha_vantage_daily_metadata(raw_bytes, canonical_driver_id=canonical_driver_id, instrument_form=instrument_form)

    def parse_history_response(
        self, raw_bytes: bytes, *, provider_symbol: str, response_manifest: CollectorResponseManifest,  # noqa: ARG002 -- `response_manifest` is part of the Protocol shape; this adapter needs only the raw bytes to parse
    ) -> tuple[RawMarketRecord, ...]:
        return parse_alpha_vantage_daily_records(raw_bytes, provider_symbol=provider_symbol)
