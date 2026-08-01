"""Provider-neutral historical market collector contract (Milestone 10,
Phase 4C) -- mirrors `collectors.protocols.HistoricalHttpTransport`'s own
structural-`Protocol` convention exactly: every orchestration-level
function depends on THIS shape, never a concrete provider client
directly, so orchestration logic is tested with a deterministic fake
adapter, and a real provider's actual request/response mechanics stay
fully isolated inside `providers/<name>.py`.

`MarketCollectorCapabilities` is the honesty contract: the orchestrator
REJECTS a request exceeding what a provider adapter actually declared
(`ProviderCapabilityError`) rather than silently downgrading interval,
adjustment mode, or instrument semantics -- see `market_orchestration.py`'s
own capability-check stage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from quant_platform.core.exceptions import ProviderCapabilityError
from quant_platform.market_data.collectors.cross_asset.instrument_form import InstrumentForm
from quant_platform.market_data.collectors.cross_asset.market_record import RawMarketRecord
from quant_platform.market_data.collectors.request_manifest import CollectorRequestManifest, CredentialMode
from quant_platform.market_data.collectors.response_manifest import CollectorResponseManifest
from quant_platform.market_data.identity import require_non_empty

__all__ = [
    "HistoricalMarketCollector",
    "MarketCollectorCapabilities",
    "ProviderMetadataRecord",
    "require_within_capabilities",
]


@dataclass(frozen=True, slots=True)
class MarketCollectorCapabilities:
    provider: str
    candles_supported: bool
    quotes_supported: bool
    trades_supported: bool
    adjusted_data_supported: bool
    unadjusted_data_supported: bool
    corporate_actions_supported: bool
    futures_contracts_supported: bool
    continuous_futures_supported: bool
    pagination_supported: bool
    anonymous_access_supported: bool
    runtime_credential_required: bool
    max_interval_days_per_request: int | None
    """`None` means unbounded (the provider returns full history in one
    call, as `providers/alpha_vantage.py`'s own verified endpoint does --
    NOT the same as "no limit was checked"; see that module's own
    `max_records_per_request` bound, enforced post-hoc client-side)."""
    max_rows_per_page: int | None
    supported_granularities: tuple[str, ...]
    supported_instrument_forms: tuple[InstrumentForm, ...]

    def __post_init__(self) -> None:
        require_non_empty(self.provider, field_name="MarketCollectorCapabilities.provider")
        if not self.supported_granularities:
            raise ProviderCapabilityError("MarketCollectorCapabilities.supported_granularities must not be empty")
        if not self.supported_instrument_forms:
            raise ProviderCapabilityError("MarketCollectorCapabilities.supported_instrument_forms must not be empty")


@dataclass(frozen=True, slots=True)
class ProviderMetadataRecord:
    """Independently verifiable provider metadata for one symbol -- see
    spec Section 8. `providers/alpha_vantage.py`'s own verified endpoint
    does not expose a dedicated metadata endpoint (only the DATA
    response's own `Meta Data` envelope, which this record is built
    from); fields this phase's provider genuinely cannot supply are
    `None`, never fabricated."""

    provider: str
    provider_symbol: str
    canonical_driver_id: str
    provider_instrument_name: str | None
    asset_class: str | None
    instrument_form: InstrumentForm
    exchange_or_venue: str | None
    currency: str | None
    quote_unit: str | None
    timezone_key: str | None
    supported_intervals: tuple[str, ...]
    first_available_timestamp: datetime | None
    last_available_timestamp: datetime | None
    adjustment_supported: bool
    provider_metadata_digest: str
    """Content digest of the raw metadata evidence this record was
    built from -- lets verification re-derive and compare, never
    trusting this record's own claims in isolation."""

    def to_json_dict(self) -> dict[str, object]:
        from quant_platform.market_data.identity import serialize_timestamp

        return {
            "provider": self.provider, "provider_symbol": self.provider_symbol, "canonical_driver_id": self.canonical_driver_id,
            "provider_instrument_name": self.provider_instrument_name, "asset_class": self.asset_class, "instrument_form": self.instrument_form.value,
            "exchange_or_venue": self.exchange_or_venue, "currency": self.currency, "quote_unit": self.quote_unit, "timezone_key": self.timezone_key,
            "supported_intervals": list(self.supported_intervals),
            "first_available_timestamp": (None if self.first_available_timestamp is None else serialize_timestamp(self.first_available_timestamp, field_name="first_available_timestamp")),
            "last_available_timestamp": (None if self.last_available_timestamp is None else serialize_timestamp(self.last_available_timestamp, field_name="last_available_timestamp")),
            "adjustment_supported": self.adjustment_supported, "provider_metadata_digest": self.provider_metadata_digest,
        }


class HistoricalMarketCollector(Protocol):
    """Structural protocol every concrete provider adapter implements.
    Deliberately narrow: request/response construction and RAW parsing
    only -- canonical normalization (`market_normalization.py`) and
    orchestration (`market_orchestration.py`) are provider-neutral
    layers ABOVE this, never duplicated per adapter."""

    def provider_metadata(self) -> MarketCollectorCapabilities: ...

    def supported_capabilities(self) -> MarketCollectorCapabilities: ...

    def build_metadata_request(self, *, provider_symbol: str, request_time: datetime, credential_mode: CredentialMode) -> CollectorRequestManifest: ...

    def build_history_request(
        self, *, provider_symbol: str, granularity: str, request_time: datetime, credential_mode: CredentialMode,
    ) -> CollectorRequestManifest: ...

    def parse_metadata_response(
        self, raw_bytes: bytes, *, provider_symbol: str, canonical_driver_id: str, instrument_form: InstrumentForm,
    ) -> ProviderMetadataRecord: ...

    def parse_history_response(self, raw_bytes: bytes, *, provider_symbol: str, response_manifest: CollectorResponseManifest) -> tuple[RawMarketRecord, ...]: ...


def require_within_capabilities(
    capabilities: MarketCollectorCapabilities, *, instrument_form: InstrumentForm, granularity: str, requires_adjusted: bool, requires_credential: bool,
) -> None:
    """The orchestrator's own fail-closed capability gate -- called
    BEFORE any request is built. Never silently downgrades; rejects
    outright."""
    if instrument_form not in capabilities.supported_instrument_forms:
        raise ProviderCapabilityError(f"provider {capabilities.provider!r} does not support instrument_form={instrument_form.value!r}")
    if granularity not in capabilities.supported_granularities:
        raise ProviderCapabilityError(f"provider {capabilities.provider!r} does not support granularity={granularity!r}")
    if requires_adjusted and not capabilities.adjusted_data_supported:
        raise ProviderCapabilityError(f"provider {capabilities.provider!r} does not support adjusted data")
    if not requires_adjusted and not capabilities.unadjusted_data_supported:
        raise ProviderCapabilityError(f"provider {capabilities.provider!r} does not support unadjusted data")
    if requires_credential and not capabilities.runtime_credential_required and not capabilities.anonymous_access_supported:
        raise ProviderCapabilityError(f"provider {capabilities.provider!r} capabilities are internally inconsistent: neither anonymous access nor a credential path is available")
