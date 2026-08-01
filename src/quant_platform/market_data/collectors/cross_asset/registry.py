"""Curated cross-asset market-driver registry (Milestone 10, Phase 4C) --
mirrors Phase 4B's `curated.registry` pattern exactly: an immutable,
versioned, content-addressed KEYED SET of `CuratedMarketDriverSpec`
objects, one per ECONOMIC CONCEPT (never a specific provider symbol --
see `instrument_form.py`'s own discussion of why a concept and a
tradable instrument form must never be conflated).

`create_curated_market_driver_registry` ALWAYS sorts specs by
`canonical_driver_id` before computing `registry_id`, so identity AND
iteration order are both independent of declaration order."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from quant_platform.core.exceptions import MarketDriverRegistryError
from quant_platform.market_data.collectors.cross_asset.adjustment import (
    AdjustmentPolicy,
    require_equity_like_adjustment,
)
from quant_platform.market_data.collectors.cross_asset.instrument_form import InstrumentForm
from quant_platform.market_data.identity import compute_content_id, require_non_empty

__all__ = [
    "CURATED_MARKET_DRIVER_REGISTRY_KIND",
    "CURATED_MARKET_DRIVER_SPEC_KIND",
    "CuratedMarketDriverRegistry",
    "CuratedMarketDriverSpec",
    "DriverTier",
    "create_curated_market_driver_registry",
    "create_curated_market_driver_spec",
    "default_core_driver_ids",
    "default_core_market_driver_specs",
    "default_optional_driver_ids",
    "default_optional_market_driver_specs",
]

CURATED_MARKET_DRIVER_SPEC_KIND = "cross_asset_curated_market_driver_spec"
CURATED_MARKET_DRIVER_REGISTRY_KIND = "cross_asset_curated_market_driver_registry"

_FUTURES_FORMS = frozenset({InstrumentForm.EXCHANGE_FUTURES_CONTRACT, InstrumentForm.PROVIDER_CONTINUOUS_FUTURES})
_ADJUSTMENT_RELEVANT_FORMS = frozenset({InstrumentForm.ETF, InstrumentForm.EQUITY})


class DriverTier(Enum):
    CORE_XAUUSD_MARKET_DRIVER = "core_xauusd_market_driver"
    SECONDARY_MARKET_DRIVER = "secondary_market_driver"
    REGIME_CONTEXT = "regime_context"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True, slots=True)
class CuratedMarketDriverSpec:
    canonical_driver_id: str
    canonical_name: str
    registry_version: int
    tier: DriverTier
    economic_role: str
    is_required: bool
    asset_class: str
    preferred_instrument_form: InstrumentForm
    allowed_instrument_forms: tuple[InstrumentForm, ...]
    canonical_currency: str
    canonical_quote_unit: str
    expected_frequency: str
    session_policy_id: str
    adjustment_policy_id: str
    availability_policy_id: str
    continuation_policy_id: str | None
    provider_mapping_ids: tuple[str, ...]
    enabled: bool
    notes: str
    """Documentation only, excluded from identity unless computation-
    affecting (it never is)."""

    def __post_init__(self) -> None:
        require_non_empty(self.canonical_driver_id, field_name="CuratedMarketDriverSpec.canonical_driver_id")
        require_non_empty(self.canonical_name, field_name="CuratedMarketDriverSpec.canonical_name")
        if self.registry_version < 1:
            raise MarketDriverRegistryError(f"CuratedMarketDriverSpec.registry_version must be >= 1, got {self.registry_version}")
        require_non_empty(self.economic_role, field_name="CuratedMarketDriverSpec.economic_role")
        require_non_empty(self.asset_class, field_name="CuratedMarketDriverSpec.asset_class")
        if not self.allowed_instrument_forms:
            raise MarketDriverRegistryError("CuratedMarketDriverSpec.allowed_instrument_forms must not be empty")
        if len(set(self.allowed_instrument_forms)) != len(self.allowed_instrument_forms):
            raise MarketDriverRegistryError("CuratedMarketDriverSpec.allowed_instrument_forms must not repeat a form")
        if self.preferred_instrument_form not in self.allowed_instrument_forms:
            raise MarketDriverRegistryError(
                f"CuratedMarketDriverSpec.preferred_instrument_form ({self.preferred_instrument_form.value!r}) must be one of allowed_instrument_forms"
            )
        require_non_empty(self.canonical_currency, field_name="CuratedMarketDriverSpec.canonical_currency")
        require_non_empty(self.canonical_quote_unit, field_name="CuratedMarketDriverSpec.canonical_quote_unit")
        require_non_empty(self.expected_frequency, field_name="CuratedMarketDriverSpec.expected_frequency")
        require_non_empty(self.session_policy_id, field_name="CuratedMarketDriverSpec.session_policy_id")
        require_non_empty(self.adjustment_policy_id, field_name="CuratedMarketDriverSpec.adjustment_policy_id")
        require_non_empty(self.availability_policy_id, field_name="CuratedMarketDriverSpec.availability_policy_id")
        if _FUTURES_FORMS & set(self.allowed_instrument_forms) and self.continuation_policy_id is None:
            raise MarketDriverRegistryError(
                f"CuratedMarketDriverSpec {self.canonical_driver_id!r} allows a futures instrument form but declares no continuation_policy_id"
            )
        if not (_FUTURES_FORMS & set(self.allowed_instrument_forms)) and self.continuation_policy_id is not None:
            raise MarketDriverRegistryError(
                f"CuratedMarketDriverSpec {self.canonical_driver_id!r} declares continuation_policy_id but allows no futures instrument form"
            )
        if self.enabled and not self.provider_mapping_ids:
            raise MarketDriverRegistryError(f"CuratedMarketDriverSpec {self.canonical_driver_id!r} is enabled but has no provider_mapping_ids")
        if len(set(self.provider_mapping_ids)) != len(self.provider_mapping_ids):
            raise MarketDriverRegistryError(f"CuratedMarketDriverSpec {self.canonical_driver_id!r} lists a duplicate provider_mapping_id")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": CURATED_MARKET_DRIVER_SPEC_KIND, "canonical_driver_id": self.canonical_driver_id, "canonical_name": self.canonical_name,
            "registry_version": self.registry_version, "tier": self.tier.value, "economic_role": self.economic_role,
            "is_required": self.is_required, "asset_class": self.asset_class, "preferred_instrument_form": self.preferred_instrument_form.value,
            "allowed_instrument_forms": [f.value for f in self.allowed_instrument_forms], "canonical_currency": self.canonical_currency,
            "canonical_quote_unit": self.canonical_quote_unit, "expected_frequency": self.expected_frequency,
            "session_policy_id": self.session_policy_id, "adjustment_policy_id": self.adjustment_policy_id,
            "availability_policy_id": self.availability_policy_id, "continuation_policy_id": self.continuation_policy_id,
            "provider_mapping_ids": list(self.provider_mapping_ids), "enabled": self.enabled, "notes": self.notes,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["notes"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CuratedMarketDriverSpec:
        from quant_platform.ml.persistence import as_json_list

        return cls(
            canonical_driver_id=str(raw["canonical_driver_id"]), canonical_name=str(raw["canonical_name"]),
            registry_version=int(str(raw["registry_version"])), tier=DriverTier(raw["tier"]), economic_role=str(raw["economic_role"]),
            is_required=bool(raw["is_required"]), asset_class=str(raw["asset_class"]),
            preferred_instrument_form=InstrumentForm(raw["preferred_instrument_form"]),
            allowed_instrument_forms=tuple(InstrumentForm(f) for f in as_json_list(raw["allowed_instrument_forms"], field_name="allowed_instrument_forms")),
            canonical_currency=str(raw["canonical_currency"]), canonical_quote_unit=str(raw["canonical_quote_unit"]),
            expected_frequency=str(raw["expected_frequency"]), session_policy_id=str(raw["session_policy_id"]),
            adjustment_policy_id=str(raw["adjustment_policy_id"]), availability_policy_id=str(raw["availability_policy_id"]),
            continuation_policy_id=(None if raw.get("continuation_policy_id") is None else str(raw["continuation_policy_id"])),
            provider_mapping_ids=tuple(str(m) for m in as_json_list(raw.get("provider_mapping_ids") or [], field_name="provider_mapping_ids")),
            enabled=bool(raw["enabled"]), notes=str(raw.get("notes", "")),
        )


def create_curated_market_driver_spec(
    *, canonical_driver_id: str, canonical_name: str, registry_version: int, tier: DriverTier, economic_role: str, is_required: bool,
    asset_class: str, preferred_instrument_form: InstrumentForm, allowed_instrument_forms: tuple[InstrumentForm, ...], canonical_currency: str,
    canonical_quote_unit: str, expected_frequency: str, session_policy_id: str, adjustment_policy: AdjustmentPolicy, availability_policy_id: str,
    continuation_policy_id: str | None = None, provider_mapping_ids: tuple[str, ...] = (), enabled: bool = False, notes: str = "",
) -> CuratedMarketDriverSpec:
    """Accepts the resolved `AdjustmentPolicy` OBJECT (not merely its id)
    so this factory can enforce "equity/ETF mapping without adjustment
    policy" (spec Section 6) as a real semantic check -- an ETF/equity
    form declared alongside a `NOT_APPLICABLE` adjustment kind is
    rejected here, not merely a missing id (which is structurally
    impossible; `adjustment_policy_id` is always required)."""
    if _ADJUSTMENT_RELEVANT_FORMS & set(allowed_instrument_forms):
        require_equity_like_adjustment(adjustment_policy.kind, instrument_form=",".join(f.value for f in allowed_instrument_forms))
    return CuratedMarketDriverSpec(
        canonical_driver_id=canonical_driver_id, canonical_name=canonical_name, registry_version=registry_version, tier=tier,
        economic_role=economic_role, is_required=is_required, asset_class=asset_class, preferred_instrument_form=preferred_instrument_form,
        allowed_instrument_forms=allowed_instrument_forms, canonical_currency=canonical_currency, canonical_quote_unit=canonical_quote_unit,
        expected_frequency=expected_frequency, session_policy_id=session_policy_id, adjustment_policy_id=adjustment_policy.adjustment_policy_id,
        availability_policy_id=availability_policy_id, continuation_policy_id=continuation_policy_id, provider_mapping_ids=provider_mapping_ids,
        enabled=enabled, notes=notes,
    )


@dataclass(frozen=True, slots=True)
class CuratedMarketDriverRegistry:
    registry_id: str
    registry_version: int
    specs: tuple[CuratedMarketDriverSpec, ...]
    """ALWAYS sorted by `canonical_driver_id` -- see module docstring."""

    def get(self, canonical_driver_id: str) -> CuratedMarketDriverSpec | None:
        for spec in self.specs:
            if spec.canonical_driver_id == canonical_driver_id:
                return spec
        return None

    def enabled_driver_ids(self) -> tuple[str, ...]:
        return tuple(s.canonical_driver_id for s in self.specs if s.enabled)

    def required_driver_ids(self) -> tuple[str, ...]:
        return tuple(s.canonical_driver_id for s in self.specs if s.is_required)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": CURATED_MARKET_DRIVER_REGISTRY_KIND, "registry_id": self.registry_id, "registry_version": self.registry_version,
            "specs": [s.to_json_dict() for s in self.specs],
        }

    def to_identity_payload(self) -> dict[str, object]:
        return {
            "kind": CURATED_MARKET_DRIVER_REGISTRY_KIND, "registry_version": self.registry_version,
            "specs": [s.to_identity_payload() for s in self.specs],
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CuratedMarketDriverRegistry:
        from quant_platform.ml.persistence import as_json_dict, as_json_list

        specs = tuple(CuratedMarketDriverSpec.from_json_dict(as_json_dict(s, field_name="specs[]")) for s in as_json_list(raw["specs"], field_name="specs"))
        return cls(registry_id=str(raw["registry_id"]), registry_version=int(str(raw["registry_version"])), specs=specs)


def create_curated_market_driver_registry(*, registry_version: int, specs: tuple[CuratedMarketDriverSpec, ...]) -> CuratedMarketDriverRegistry:
    ids = [s.canonical_driver_id for s in specs]
    if len(set(ids)) != len(ids):
        raise MarketDriverRegistryError(f"duplicate canonical_driver_id in registry specs: {sorted({i for i in ids if ids.count(i) > 1})!r}")
    names = [s.canonical_name for s in specs]
    if len(set(names)) != len(names):
        raise MarketDriverRegistryError(f"duplicate canonical_name in registry specs: {sorted({n for n in names if names.count(n) > 1})!r}")
    for spec in specs:
        if spec.registry_version != registry_version:
            raise MarketDriverRegistryError(
                f"CuratedMarketDriverSpec {spec.canonical_driver_id!r} registry_version={spec.registry_version} does not match registry_version={registry_version}"
            )
    ordered_specs = tuple(sorted(specs, key=lambda s: s.canonical_driver_id))
    provisional = CuratedMarketDriverRegistry(registry_id="0" * 64, registry_version=registry_version, specs=ordered_specs)
    registry_id = compute_content_id(CURATED_MARKET_DRIVER_REGISTRY_KIND, provisional.to_identity_payload())
    return CuratedMarketDriverRegistry(registry_id=registry_id, registry_version=registry_version, specs=ordered_specs)


# --------------------------------------------------------------------------
# The 5 mandatory core economic concepts (spec Section 4, item 1-5) and the
# 5 strong-optional concepts (items 6-10). Canonical ids only -- see
# `providers/alpha_vantage.py`'s own module for the concrete
# `ProviderSymbolMapping`s that supply these this phase.
# --------------------------------------------------------------------------
def default_core_driver_ids() -> tuple[str, ...]:
    return ("us_dollar_strength", "wti_crude", "brent_crude", "silver", "gold_reference")


def default_optional_driver_ids() -> tuple[str, ...]:
    return (
        "us_equity_market_stress", "treasury_volatility", "broad_commodity_index", "copper_industrial_growth", "gold_miner_equity",
    )


def default_core_market_driver_specs(
    *, registry_version: int, adjustment_policy: AdjustmentPolicy, session_policy_id: str, availability_policy_id: str,
    provider_mapping_ids_by_driver: dict[str, tuple[str, ...]],
) -> tuple[CuratedMarketDriverSpec, ...]:
    """The 5 MANDATORY core economic concepts (spec Section 4, items
    1-5) -- `is_required=True` on every one; `enabled` is derived from
    whether the caller actually supplied a provider mapping for that
    driver id (a required concept EXISTS in the registry even if this
    phase's own provider cannot supply it -- see module docstring's own
    "mandatory core concepts must exist even if provider can't supply
    all" discipline). `preferred_instrument_form` names the IDEAL,
    literal form for the concept; `allowed_instrument_forms` additionally
    includes `ETF` because this phase's only wired provider
    (`providers/alpha_vantage.py`) can supply nothing but ETF-form
    proxies -- never conflate the two (see `instrument_form.py`)."""

    def _mapping_ids(driver_id: str) -> tuple[str, ...]:
        return provider_mapping_ids_by_driver.get(driver_id, ())

    return (
        create_curated_market_driver_spec(
            canonical_driver_id="us_dollar_strength", canonical_name="US Dollar Strength", registry_version=registry_version,
            tier=DriverTier.CORE_XAUUSD_MARKET_DRIVER, economic_role="Broad US dollar strength -- inversely correlated with USD-denominated gold.",
            is_required=True, asset_class="fx_index", preferred_instrument_form=InstrumentForm.CASH_INDEX,
            allowed_instrument_forms=(InstrumentForm.CASH_INDEX, InstrumentForm.ETF), canonical_currency="USD", canonical_quote_unit="index_points",
            expected_frequency="daily", session_policy_id=session_policy_id, adjustment_policy=adjustment_policy, availability_policy_id=availability_policy_id,
            provider_mapping_ids=_mapping_ids("us_dollar_strength"), enabled=bool(_mapping_ids("us_dollar_strength")),
            notes="Ideal form is a dollar cash index (e.g. DXY-style); this phase's only mapped instrument is a USD-index-tracking ETF proxy.",
        ),
        create_curated_market_driver_spec(
            canonical_driver_id="wti_crude", canonical_name="WTI Crude Oil", registry_version=registry_version,
            tier=DriverTier.CORE_XAUUSD_MARKET_DRIVER,
            economic_role="US benchmark crude oil price -- a real-asset/inflation and risk-appetite driver relevant to gold.",
            is_required=True, asset_class="energy_commodity", preferred_instrument_form=InstrumentForm.SPOT,
            allowed_instrument_forms=(InstrumentForm.SPOT, InstrumentForm.ETF), canonical_currency="USD", canonical_quote_unit="usd_per_barrel",
            expected_frequency="daily", session_policy_id=session_policy_id, adjustment_policy=adjustment_policy, availability_policy_id=availability_policy_id,
            provider_mapping_ids=_mapping_ids("wti_crude"), enabled=bool(_mapping_ids("wti_crude")),
            notes="Ideal form is a WTI spot assessment; this phase's only mapped instrument is a futures-based oil ETF proxy with material roll/contango risk.",
        ),
        create_curated_market_driver_spec(
            canonical_driver_id="brent_crude", canonical_name="Brent Crude Oil", registry_version=registry_version,
            tier=DriverTier.CORE_XAUUSD_MARKET_DRIVER,
            economic_role="Global benchmark crude oil price -- a real-asset/inflation and risk-appetite driver relevant to gold.",
            is_required=True, asset_class="energy_commodity", preferred_instrument_form=InstrumentForm.SPOT,
            allowed_instrument_forms=(InstrumentForm.SPOT, InstrumentForm.ETF), canonical_currency="USD", canonical_quote_unit="usd_per_barrel",
            expected_frequency="daily", session_policy_id=session_policy_id, adjustment_policy=adjustment_policy, availability_policy_id=availability_policy_id,
            provider_mapping_ids=_mapping_ids("brent_crude"), enabled=bool(_mapping_ids("brent_crude")),
            notes="Ideal form is a Brent spot assessment; this phase's only mapped instrument is a futures-based oil ETF proxy with material roll/contango risk.",
        ),
        create_curated_market_driver_spec(
            canonical_driver_id="silver", canonical_name="Silver", registry_version=registry_version, tier=DriverTier.CORE_XAUUSD_MARKET_DRIVER,
            economic_role="Precious-metal sibling to gold, high co-movement, own industrial-demand component.", is_required=True,
            asset_class="precious_metal", preferred_instrument_form=InstrumentForm.SPOT,
            allowed_instrument_forms=(InstrumentForm.SPOT, InstrumentForm.ETF), canonical_currency="USD", canonical_quote_unit="usd_per_troy_ounce",
            expected_frequency="daily", session_policy_id=session_policy_id, adjustment_policy=adjustment_policy, availability_policy_id=availability_policy_id,
            provider_mapping_ids=_mapping_ids("silver"), enabled=bool(_mapping_ids("silver")),
            notes="Ideal form is spot silver; this phase's only mapped instrument is a physically-backed silver-bullion ETF proxy.",
        ),
        create_curated_market_driver_spec(
            canonical_driver_id="gold_reference", canonical_name="Gold (XAUUSD Reference Market)", registry_version=registry_version,
            tier=DriverTier.CORE_XAUUSD_MARKET_DRIVER,
            economic_role="The XAUUSD market itself -- included as a cross-asset driver concept for reconciliation against the platform's own primary gold price source.",
            is_required=True, asset_class="precious_metal", preferred_instrument_form=InstrumentForm.SPOT,
            allowed_instrument_forms=(InstrumentForm.SPOT, InstrumentForm.ETF), canonical_currency="USD", canonical_quote_unit="usd_per_troy_ounce",
            expected_frequency="daily", session_policy_id=session_policy_id, adjustment_policy=adjustment_policy, availability_policy_id=availability_policy_id,
            provider_mapping_ids=_mapping_ids("gold_reference"), enabled=bool(_mapping_ids("gold_reference")),
            notes=(
                "Ideal form is spot XAUUSD; this phase's only mapped instrument is a physically-backed gold-bullion ETF proxy, NEVER to be "
                "confused with or substituted for this platform's own primary XAUUSD spot feed."
            ),
        ),
    )


def default_optional_market_driver_specs(
    *, registry_version: int, adjustment_policy: AdjustmentPolicy, session_policy_id: str, availability_policy_id: str,
    provider_mapping_ids_by_driver: dict[str, tuple[str, ...]],
) -> tuple[CuratedMarketDriverSpec, ...]:
    """The 5 STRONG-OPTIONAL concepts (spec Section 4, items 6-10) --
    `is_required=False` on every one. `treasury_volatility` ships with
    NO viable ETF proxy this phase (no single-ticker instrument was
    identified with a defensible, disclosable tracking relationship to
    Treasury-market implied volatility); its spec exists so the concept
    is documented, with `enabled=False` and `provider_mapping_ids=()` --
    UNSUPPORTED AND FAIL-CLOSED per spec Section 2's own provider-gap
    fallback discipline, applied here at single-driver granularity."""

    def _mapping_ids(driver_id: str) -> tuple[str, ...]:
        return provider_mapping_ids_by_driver.get(driver_id, ())

    return (
        create_curated_market_driver_spec(
            canonical_driver_id="us_equity_market_stress", canonical_name="US Equity Market Stress", registry_version=registry_version,
            tier=DriverTier.REGIME_CONTEXT,
            economic_role="Equity-market implied-volatility regime context -- risk-off equity stress often coincides with gold safe-haven demand.",
            is_required=False, asset_class="volatility_index", preferred_instrument_form=InstrumentForm.CASH_INDEX,
            allowed_instrument_forms=(InstrumentForm.CASH_INDEX, InstrumentForm.ETF), canonical_currency="USD", canonical_quote_unit="index_points",
            expected_frequency="daily", session_policy_id=session_policy_id, adjustment_policy=adjustment_policy, availability_policy_id=availability_policy_id,
            provider_mapping_ids=_mapping_ids("us_equity_market_stress"), enabled=bool(_mapping_ids("us_equity_market_stress")),
            notes=(
                "Ideal form is a VIX-style cash index; this phase's only mapped instrument is a short-term VIX-futures-based ETF proxy with "
                "severe, well-documented long-run decay/roll cost -- LOW proxy quality, regime-context use only, never a primary driver signal."
            ),
        ),
        create_curated_market_driver_spec(
            canonical_driver_id="treasury_volatility", canonical_name="Treasury Market Volatility", registry_version=registry_version,
            tier=DriverTier.REGIME_CONTEXT,
            economic_role="Rates-market implied-volatility regime context -- elevated Treasury volatility often coincides with gold-relevant policy uncertainty.",
            is_required=False, asset_class="volatility_index", preferred_instrument_form=InstrumentForm.CASH_INDEX,
            allowed_instrument_forms=(InstrumentForm.CASH_INDEX, InstrumentForm.ETF), canonical_currency="USD", canonical_quote_unit="index_points",
            expected_frequency="daily", session_policy_id=session_policy_id, adjustment_policy=adjustment_policy, availability_policy_id=availability_policy_id,
            provider_mapping_ids=(), enabled=False,
            notes=(
                "UNSUPPORTED AND FAIL-CLOSED this phase: no single-ticker ETF with a defensible, disclosable tracking relationship to "
                "Treasury-market implied volatility (the MOVE index has no directly investable ETF) was identified through this phase's "
                "shipped provider. The concept is documented so it exists for a future phase; no mapping is fabricated to fill the gap."
            ),
        ),
        create_curated_market_driver_spec(
            canonical_driver_id="broad_commodity_index", canonical_name="Broad Commodity Index", registry_version=registry_version,
            tier=DriverTier.SECONDARY_MARKET_DRIVER,
            economic_role="Broad real-asset/commodity-complex regime context beyond energy and precious metals alone.", is_required=False,
            asset_class="commodity_index", preferred_instrument_form=InstrumentForm.SYNTHETIC_INDEX,
            allowed_instrument_forms=(InstrumentForm.SYNTHETIC_INDEX, InstrumentForm.ETF), canonical_currency="USD", canonical_quote_unit="index_points",
            expected_frequency="daily", session_policy_id=session_policy_id, adjustment_policy=adjustment_policy, availability_policy_id=availability_policy_id,
            provider_mapping_ids=_mapping_ids("broad_commodity_index"), enabled=bool(_mapping_ids("broad_commodity_index")),
            notes=(
                "Ideal form is a broad commodity benchmark index; this phase's only mapped instrument is a futures-based diversified-commodity "
                "ETF proxy with material roll risk across its underlying basket."
            ),
        ),
        create_curated_market_driver_spec(
            canonical_driver_id="copper_industrial_growth", canonical_name="Copper (Industrial Growth Proxy)", registry_version=registry_version,
            tier=DriverTier.SECONDARY_MARKET_DRIVER,
            economic_role="Industrial-demand/global-growth regime context -- a classic 'growth vs. safe-haven' counterpoint to gold.",
            is_required=False, asset_class="industrial_commodity", preferred_instrument_form=InstrumentForm.SPOT,
            allowed_instrument_forms=(InstrumentForm.SPOT, InstrumentForm.ETF), canonical_currency="USD", canonical_quote_unit="usd_per_pound",
            expected_frequency="daily", session_policy_id=session_policy_id, adjustment_policy=adjustment_policy, availability_policy_id=availability_policy_id,
            provider_mapping_ids=_mapping_ids("copper_industrial_growth"), enabled=bool(_mapping_ids("copper_industrial_growth")),
            notes="Ideal form is spot copper; this phase's only mapped instrument is a futures-based copper ETF proxy with material roll risk.",
        ),
        create_curated_market_driver_spec(
            canonical_driver_id="gold_miner_equity", canonical_name="Gold Miner Equity Basket", registry_version=registry_version,
            tier=DriverTier.SECONDARY_MARKET_DRIVER,
            economic_role="Gold-mining-sector equity sentiment -- a levered, imperfect echo of gold price expectations plus company/equity-market-beta risk.",
            is_required=False, asset_class="equity_basket", preferred_instrument_form=InstrumentForm.ETF,
            allowed_instrument_forms=(InstrumentForm.ETF,), canonical_currency="USD", canonical_quote_unit="usd_per_share",
            expected_frequency="daily", session_policy_id=session_policy_id, adjustment_policy=adjustment_policy, availability_policy_id=availability_policy_id,
            provider_mapping_ids=_mapping_ids("gold_miner_equity"), enabled=bool(_mapping_ids("gold_miner_equity")),
            notes=(
                "An ETF basket of gold-mining equities IS the natural, most direct form for this concept (there is no single 'spot "
                "gold-miner' asset) -- still classified as a proxy (never gold itself) per instrument_form.py's ETF-is-always-a-proxy rule."
            ),
        ),
    )
