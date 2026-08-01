"""Immutable, provider-specific symbol mappings (Milestone 10, Phase 4C).

A `ProviderSymbolMapping` binds ONE provider's ONE symbol to ONE
canonical driver under ONE specific instrument form -- carrying the
proxy classification (`instrument_form.ProxyPolicy`) wherever the
mapping is not the literal underlying. `SymbolMappingSet` is the keyed
collection every curated registry construction validates against: one
provider symbol can never resolve to two DIFFERENT active canonical
instruments within the same mapping VERSION (an alias change is
expressed as a NEW mapping version, never an in-place edit)."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.core.exceptions import SymbolMappingError
from quant_platform.market_data.collectors.cross_asset.adjustment import AdjustmentPolicyKind
from quant_platform.market_data.collectors.cross_asset.instrument_form import InstrumentForm, ProxyPolicy
from quant_platform.market_data.identity import compute_content_id, require_non_empty

__all__ = [
    "PROVIDER_SYMBOL_MAPPING_KIND",
    "ProviderSymbolMapping",
    "SymbolMappingSet",
    "create_provider_symbol_mapping",
    "create_symbol_mapping_set",
]

PROVIDER_SYMBOL_MAPPING_KIND = "cross_asset_provider_symbol_mapping"


@dataclass(frozen=True, slots=True)
class ProviderSymbolMapping:
    mapping_id: str
    provider: str
    provider_symbol: str
    canonical_driver_id: str
    instrument_form: InstrumentForm
    exchange_or_venue: str | None
    currency: str
    adjustment_policy_kind: AdjustmentPolicyKind
    continuation_policy_id: str | None
    mapping_version: int
    proxy_policy: ProxyPolicy
    enabled: bool
    notes: str
    """Documentation only, excluded from identity."""

    def __post_init__(self) -> None:
        require_non_empty(self.provider, field_name="ProviderSymbolMapping.provider")
        require_non_empty(self.provider_symbol, field_name="ProviderSymbolMapping.provider_symbol")
        require_non_empty(self.canonical_driver_id, field_name="ProviderSymbolMapping.canonical_driver_id")
        require_non_empty(self.currency, field_name="ProviderSymbolMapping.currency")
        if self.mapping_version < 1:
            raise SymbolMappingError(f"ProviderSymbolMapping.mapping_version must be >= 1, got {self.mapping_version}")
        if self.instrument_form in (InstrumentForm.EXCHANGE_FUTURES_CONTRACT, InstrumentForm.PROVIDER_CONTINUOUS_FUTURES):
            if self.continuation_policy_id is None:
                raise SymbolMappingError(f"ProviderSymbolMapping {self.provider}:{self.provider_symbol} is a futures form but declares no continuation_policy_id")
        elif self.continuation_policy_id is not None:
            raise SymbolMappingError(f"ProviderSymbolMapping {self.provider}:{self.provider_symbol} declares continuation_policy_id but is not a futures form")
        # ETF is (per Section 5's own examples: DXY cash index vs. dollar ETF,
        # spot silver vs. SI futures vs. a silver ETF) ALWAYS a proxy -- never
        # the literal underlying it tracks.
        if self.instrument_form is InstrumentForm.ETF and not self.proxy_policy.is_proxy:
            raise SymbolMappingError(f"ProviderSymbolMapping {self.provider}:{self.provider_symbol} is instrument_form=ETF but proxy_policy.is_proxy=False")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": PROVIDER_SYMBOL_MAPPING_KIND, "mapping_id": self.mapping_id, "provider": self.provider, "provider_symbol": self.provider_symbol,
            "canonical_driver_id": self.canonical_driver_id, "instrument_form": self.instrument_form.value, "exchange_or_venue": self.exchange_or_venue,
            "currency": self.currency, "adjustment_policy_kind": self.adjustment_policy_kind.value, "continuation_policy_id": self.continuation_policy_id,
            "mapping_version": self.mapping_version, "proxy_policy": self.proxy_policy.to_json_dict(), "enabled": self.enabled, "notes": self.notes,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["mapping_id"]
        del payload["notes"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ProviderSymbolMapping:
        from quant_platform.ml.persistence import as_json_dict

        return cls(
            mapping_id=str(raw["mapping_id"]), provider=str(raw["provider"]), provider_symbol=str(raw["provider_symbol"]),
            canonical_driver_id=str(raw["canonical_driver_id"]), instrument_form=InstrumentForm(raw["instrument_form"]),
            exchange_or_venue=(None if raw.get("exchange_or_venue") is None else str(raw["exchange_or_venue"])), currency=str(raw["currency"]),
            adjustment_policy_kind=AdjustmentPolicyKind(raw["adjustment_policy_kind"]),
            continuation_policy_id=(None if raw.get("continuation_policy_id") is None else str(raw["continuation_policy_id"])),
            mapping_version=int(str(raw["mapping_version"])), proxy_policy=ProxyPolicy.from_json_dict(as_json_dict(raw["proxy_policy"], field_name="proxy_policy")),
            enabled=bool(raw["enabled"]), notes=str(raw.get("notes", "")),
        )


def create_provider_symbol_mapping(
    *, provider: str, provider_symbol: str, canonical_driver_id: str, instrument_form: InstrumentForm, currency: str,
    adjustment_policy_kind: AdjustmentPolicyKind, proxy_policy: ProxyPolicy, exchange_or_venue: str | None = None,
    continuation_policy_id: str | None = None, mapping_version: int = 1, enabled: bool = True, notes: str = "",
) -> ProviderSymbolMapping:
    provisional = ProviderSymbolMapping(
        mapping_id="0" * 64, provider=provider, provider_symbol=provider_symbol, canonical_driver_id=canonical_driver_id,
        instrument_form=instrument_form, exchange_or_venue=exchange_or_venue, currency=currency, adjustment_policy_kind=adjustment_policy_kind,
        continuation_policy_id=continuation_policy_id, mapping_version=mapping_version, proxy_policy=proxy_policy, enabled=enabled, notes=notes,
    )
    mapping_id = compute_content_id(PROVIDER_SYMBOL_MAPPING_KIND, provisional.to_identity_payload())
    return ProviderSymbolMapping(
        mapping_id=mapping_id, provider=provider, provider_symbol=provider_symbol, canonical_driver_id=canonical_driver_id,
        instrument_form=instrument_form, exchange_or_venue=exchange_or_venue, currency=currency, adjustment_policy_kind=adjustment_policy_kind,
        continuation_policy_id=continuation_policy_id, mapping_version=mapping_version, proxy_policy=proxy_policy, enabled=enabled, notes=notes,
    )


@dataclass(frozen=True, slots=True)
class SymbolMappingSet:
    """A keyed collection of `ProviderSymbolMapping`s -- validates the
    cross-mapping invariant no single mapping's own `__post_init__` can
    check alone: one `(provider, provider_symbol, mapping_version)`
    cannot resolve to two DIFFERENT `canonical_driver_id`s."""

    mappings: tuple[ProviderSymbolMapping, ...]

    def get(self, mapping_id: str) -> ProviderSymbolMapping | None:
        for mapping in self.mappings:
            if mapping.mapping_id == mapping_id:
                return mapping
        return None

    def for_driver(self, canonical_driver_id: str) -> tuple[ProviderSymbolMapping, ...]:
        return tuple(m for m in self.mappings if m.canonical_driver_id == canonical_driver_id)


def create_symbol_mapping_set(mappings: tuple[ProviderSymbolMapping, ...]) -> SymbolMappingSet:
    seen: dict[tuple[str, str, int], str] = {}
    for mapping in mappings:
        if not mapping.enabled:
            continue
        key = (mapping.provider, mapping.provider_symbol, mapping.mapping_version)
        if key in seen and seen[key] != mapping.canonical_driver_id:
            raise SymbolMappingError(
                f"ambiguous mapping: provider={mapping.provider!r} symbol={mapping.provider_symbol!r} version={mapping.mapping_version} "
                f"resolves to both {seen[key]!r} and {mapping.canonical_driver_id!r}"
            )
        seen[key] = mapping.canonical_driver_id
    ids = [m.mapping_id for m in mappings]
    if len(set(ids)) != len(ids):
        raise SymbolMappingError("SymbolMappingSet contains a duplicate mapping_id")
    return SymbolMappingSet(mappings=tuple(sorted(mappings, key=lambda m: m.mapping_id)))
