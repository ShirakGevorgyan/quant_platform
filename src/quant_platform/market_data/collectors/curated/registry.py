"""Curated FRED series registry (Milestone 10, Phase 4B) -- an
immutable, versioned, content-addressed universe of hand-reviewed FRED
series intended for XAUUSD macro research. Mirrors `macro_normalization.
UnitMappingSpec`'s own established pattern (a keyed set, sorted before
hashing so identity is independent of construction order) and reuses
its `MacroUnit` enum directly for `normalization_kind` (a curated
series' declared unit classification IS exactly a `MacroUnit`, not a
parallel concept).

EXTENDED-UNIVERSE DISCLOSURE: only the four REQUIRED core series
(`DFII10`/`DGS10`/`CPIAUCSL`/`DFF`) are individually fixture-verified
end-to-end this phase (see `test_collectors_curated_fixture_acceptance.
py`). The 14 extended candidates below use their real, well-known FRED
series identifiers (high confidence -- these are among FRED's most
widely referenced series) and a curator's best-effort documented
metadata expectation, but are shipped with `enabled=False` -- an
explicit opt-in flip is required before any of them is used in a real
backfill, exactly so this registry never silently implies "these are
individually verified" when only the four core series are. See the
Phase 4B delivery report's Known Non-Blocking Limitations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from quant_platform.core.exceptions import CuratedRegistryError
from quant_platform.market_data.collectors.macro_normalization import MacroUnit
from quant_platform.market_data.identity import compute_content_id, require_non_empty, require_tz_aware

__all__ = [
    "CURATED_REGISTRY_KIND",
    "CURATED_SERIES_SPEC_KIND",
    "CuratedFredRegistry",
    "CuratedFredSeriesSpec",
    "MissingValuePolicy",
    "SeriesTier",
    "create_curated_registry",
    "create_curated_series_spec",
    "default_core_series_specs",
    "default_extended_series_specs",
]

CURATED_SERIES_SPEC_KIND = "curated_fred_series_spec"
CURATED_REGISTRY_KIND = "curated_fred_registry"

_VALID_FREQUENCY_SHORT_CODES = frozenset({"D", "W", "BW", "M", "Q", "SA", "A"})


class SeriesTier(Enum):
    CORE_XAUUSD_DRIVER = "core_xauusd_driver"
    SECONDARY_MACRO_DRIVER = "secondary_macro_driver"
    REGIME_CONTEXT = "regime_context"
    EXPERIMENTAL = "experimental"


class MissingValuePolicy(Enum):
    QUARANTINE = "quarantine"
    """The safest, default-recommended policy: a missing `"."` row is
    quarantined (see Phase 4A's own `MISSING_OBSERVATION_VALUE` issue
    code) -- never silently dropped, never coerced to zero."""

    STORE_AS_MISSING_FACT = "store_as_missing_fact"
    """The missing observation IS durably recorded, as an explicit
    missing FACT (`CuratedMacroObservation.is_missing=True`, no
    `value`) -- for a downstream consumer that needs to know a gap
    exists at a specific coordinate, not just that it was dropped."""

    SKIP_AND_REPORT = "skip_and_report"
    """Neither quarantined nor stored -- simply excluded from the
    component dataset, but counted and reported (missing count/
    coordinates) so the omission is never silent."""


@dataclass(frozen=True, slots=True)
class CuratedFredSeriesSpec:
    series_id: str
    canonical_series_name: str
    registry_version: int
    tier: SeriesTier
    economic_category: str
    expected_native_frequency: str
    """FRED's own `frequency_short` code (`"D"`, `"W"`, `"M"`, ...)."""
    expected_units: tuple[str, ...]
    """One or more ACCEPTABLE `units_short` codes (usually a single
    element) -- a tuple so a series legitimately reported under more
    than one acceptable unit label does not require a registry change."""
    expected_seasonal_adjustment: str | None
    target_macro_instrument_id: str
    normalization_kind: MacroUnit
    unit_conversion: Decimal
    missing_value_policy: MissingValuePolicy
    revision_policy_id: str
    release_availability_policy_id: str
    default_observation_start: datetime
    request_output_type: int | None
    request_units: str
    request_frequency: str | None
    aggregation_method: str | None
    enabled: bool
    notes: str = ""
    """Excluded from identity -- prose only, never affects computation
    (see module docstring's own "notes excluded... unless they affect
    computation" requirement: nothing here ever branches on `notes`)."""

    def __post_init__(self) -> None:
        require_non_empty(self.series_id, field_name="CuratedFredSeriesSpec.series_id")
        require_non_empty(self.canonical_series_name, field_name="CuratedFredSeriesSpec.canonical_series_name")
        require_non_empty(self.economic_category, field_name="CuratedFredSeriesSpec.economic_category")
        if self.registry_version < 1:
            raise CuratedRegistryError(f"CuratedFredSeriesSpec.registry_version must be >= 1, got {self.registry_version}")
        if self.expected_native_frequency not in _VALID_FREQUENCY_SHORT_CODES:
            raise CuratedRegistryError(f"CuratedFredSeriesSpec.expected_native_frequency {self.expected_native_frequency!r} is not a recognized FRED frequency_short code")
        if not self.expected_units:
            raise CuratedRegistryError("CuratedFredSeriesSpec.expected_units must be non-empty")
        require_non_empty(self.revision_policy_id, field_name="CuratedFredSeriesSpec.revision_policy_id")
        require_non_empty(self.release_availability_policy_id, field_name="CuratedFredSeriesSpec.release_availability_policy_id")
        require_tz_aware(self.default_observation_start, field_name="CuratedFredSeriesSpec.default_observation_start")
        if self.unit_conversion <= 0:
            raise CuratedRegistryError(f"CuratedFredSeriesSpec.unit_conversion must be > 0, got {self.unit_conversion}")
        if self.aggregation_method is not None and self.request_frequency is None:
            raise CuratedRegistryError("CuratedFredSeriesSpec.aggregation_method requires request_frequency to also be set")
        if self.enabled:
            require_non_empty(self.target_macro_instrument_id, field_name="CuratedFredSeriesSpec.target_macro_instrument_id (required for an enabled series)")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": CURATED_SERIES_SPEC_KIND, "series_id": self.series_id, "canonical_series_name": self.canonical_series_name,
            "registry_version": self.registry_version, "tier": self.tier.value, "economic_category": self.economic_category,
            "expected_native_frequency": self.expected_native_frequency, "expected_units": list(self.expected_units),
            "expected_seasonal_adjustment": self.expected_seasonal_adjustment, "target_macro_instrument_id": self.target_macro_instrument_id,
            "normalization_kind": self.normalization_kind.value, "unit_conversion": str(self.unit_conversion),
            "missing_value_policy": self.missing_value_policy.value, "revision_policy_id": self.revision_policy_id,
            "release_availability_policy_id": self.release_availability_policy_id,
            "default_observation_start": self.default_observation_start.isoformat(), "request_output_type": self.request_output_type,
            "request_units": self.request_units, "request_frequency": self.request_frequency, "aggregation_method": self.aggregation_method,
            "enabled": self.enabled, "notes": self.notes,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["notes"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CuratedFredSeriesSpec:
        from quant_platform.ml.persistence import as_json_list

        return cls(
            series_id=str(raw["series_id"]), canonical_series_name=str(raw["canonical_series_name"]),
            registry_version=int(str(raw["registry_version"])), tier=SeriesTier(raw["tier"]), economic_category=str(raw["economic_category"]),
            expected_native_frequency=str(raw["expected_native_frequency"]),
            expected_units=tuple(str(u) for u in as_json_list(raw["expected_units"], field_name="expected_units")),
            expected_seasonal_adjustment=(None if raw.get("expected_seasonal_adjustment") is None else str(raw["expected_seasonal_adjustment"])),
            target_macro_instrument_id=str(raw["target_macro_instrument_id"]), normalization_kind=MacroUnit(raw["normalization_kind"]),
            unit_conversion=Decimal(str(raw["unit_conversion"])), missing_value_policy=MissingValuePolicy(raw["missing_value_policy"]),
            revision_policy_id=str(raw["revision_policy_id"]), release_availability_policy_id=str(raw["release_availability_policy_id"]),
            default_observation_start=datetime.fromisoformat(str(raw["default_observation_start"])),
            request_output_type=(None if raw.get("request_output_type") is None else int(str(raw["request_output_type"]))),
            request_units=str(raw["request_units"]), request_frequency=(None if raw.get("request_frequency") is None else str(raw["request_frequency"])),
            aggregation_method=(None if raw.get("aggregation_method") is None else str(raw["aggregation_method"])),
            enabled=bool(raw["enabled"]), notes=str(raw.get("notes") or ""),
        )


def create_curated_series_spec(
    *, series_id: str, canonical_series_name: str, registry_version: int, tier: SeriesTier, economic_category: str,
    expected_native_frequency: str, expected_units: tuple[str, ...], target_macro_instrument_id: str, normalization_kind: MacroUnit,
    revision_policy_id: str, release_availability_policy_id: str, default_observation_start: datetime, missing_value_policy: MissingValuePolicy = MissingValuePolicy.QUARANTINE,
    unit_conversion: Decimal = Decimal(1), expected_seasonal_adjustment: str | None = None, request_output_type: int | None = None,
    request_units: str = "lin", request_frequency: str | None = None, aggregation_method: str | None = None, enabled: bool = True, notes: str = "",
) -> CuratedFredSeriesSpec:
    return CuratedFredSeriesSpec(
        series_id=series_id, canonical_series_name=canonical_series_name, registry_version=registry_version, tier=tier,
        economic_category=economic_category, expected_native_frequency=expected_native_frequency, expected_units=expected_units,
        expected_seasonal_adjustment=expected_seasonal_adjustment, target_macro_instrument_id=target_macro_instrument_id,
        normalization_kind=normalization_kind, unit_conversion=unit_conversion, missing_value_policy=missing_value_policy,
        revision_policy_id=revision_policy_id, release_availability_policy_id=release_availability_policy_id,
        default_observation_start=default_observation_start, request_output_type=request_output_type, request_units=request_units,
        request_frequency=request_frequency, aggregation_method=aggregation_method, enabled=enabled, notes=notes,
    )


@dataclass(frozen=True, slots=True)
class CuratedFredRegistry:
    registry_id: str
    registry_version: int
    specs: tuple[CuratedFredSeriesSpec, ...]
    """Always stored sorted by `series_id` -- see module docstring:
    identity (and iteration order) is independent of construction
    order, since this is semantically a KEYED SET."""

    def __post_init__(self) -> None:
        if self.registry_version < 1:
            raise CuratedRegistryError(f"CuratedFredRegistry.registry_version must be >= 1, got {self.registry_version}")
        series_ids: set[str] = set()
        canonical_names: set[str] = set()
        for spec in self.specs:
            if spec.registry_version != self.registry_version:
                raise CuratedRegistryError(f"CuratedFredSeriesSpec {spec.series_id!r} has registry_version={spec.registry_version}, does not match registry's own {self.registry_version}")
            if spec.series_id in series_ids:
                raise CuratedRegistryError(f"duplicate FRED series id in registry: {spec.series_id!r}")
            series_ids.add(spec.series_id)
            if spec.canonical_series_name in canonical_names:
                raise CuratedRegistryError(f"duplicate canonical_series_name in registry: {spec.canonical_series_name!r}")
            canonical_names.add(spec.canonical_series_name)
        ordered = tuple(sorted(self.specs, key=lambda s: s.series_id))
        if tuple(s.series_id for s in self.specs) != tuple(s.series_id for s in ordered):
            raise CuratedRegistryError("CuratedFredRegistry.specs must be constructed via create_curated_registry (which sorts by series_id) -- do not construct out of order directly")

    def get(self, series_id: str) -> CuratedFredSeriesSpec | None:
        for spec in self.specs:
            if spec.series_id == series_id:
                return spec
        return None

    def enabled_series_ids(self) -> tuple[str, ...]:
        return tuple(s.series_id for s in self.specs if s.enabled)

    def to_json_dict(self) -> dict[str, object]:
        return {"kind": CURATED_REGISTRY_KIND, "registry_id": self.registry_id, "registry_version": self.registry_version, "specs": [s.to_json_dict() for s in self.specs]}

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["registry_id"]
        payload["specs"] = [s.to_identity_payload() for s in self.specs]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CuratedFredRegistry:
        from quant_platform.ml.persistence import as_json_dict, as_json_list

        specs = tuple(CuratedFredSeriesSpec.from_json_dict(as_json_dict(s, field_name="spec")) for s in as_json_list(raw["specs"], field_name="specs"))
        return cls(registry_id=str(raw["registry_id"]), registry_version=int(str(raw["registry_version"])), specs=specs)


def create_curated_registry(*, registry_version: int, specs: tuple[CuratedFredSeriesSpec, ...]) -> CuratedFredRegistry:
    ordered = tuple(sorted(specs, key=lambda s: s.series_id))
    provisional = CuratedFredRegistry(registry_id="0" * 64, registry_version=registry_version, specs=ordered)
    registry_id = compute_content_id(CURATED_REGISTRY_KIND, provisional.to_identity_payload())
    return CuratedFredRegistry(registry_id=registry_id, registry_version=registry_version, specs=ordered)


def default_core_series_specs(
    *, registry_version: int, revision_policy_id: str, release_availability_policy_id_daily: str, release_availability_policy_id_monthly: str,
    default_observation_start: datetime,
) -> tuple[CuratedFredSeriesSpec, ...]:
    """The four REQUIRED core series -- see module docstring. Daily
    market-rate series use `release_availability_policy_id_daily`
    (an `OBSERVATION_DATE_END_OF_DAY`-shaped policy is the natural fit,
    per the spec's own "define daily-rate availability explicitly
    rather than reusing monthly macro-release behavior"); the monthly
    CPI release uses `release_availability_policy_id_monthly` (a
    `REALTIME_START_DATE_CONSERVATIVE`- or `NEXT_BUSINESS_DAY_
    CONSERVATIVE`-shaped policy is the natural fit)."""
    return (
        create_curated_series_spec(
            series_id="DFII10", canonical_series_name="us_10y_real_yield", registry_version=registry_version, tier=SeriesTier.CORE_XAUUSD_DRIVER,
            economic_category="real_rates", expected_native_frequency="D", expected_units=("%",), expected_seasonal_adjustment="NSA",
            target_macro_instrument_id="us_10y_real_yield", normalization_kind=MacroUnit.PERCENT, revision_policy_id=revision_policy_id,
            release_availability_policy_id=release_availability_policy_id_daily, default_observation_start=default_observation_start,
            notes="10-Year Treasury Inflation-Indexed Security, real-rate opportunity cost / major gold driver.",
        ),
        create_curated_series_spec(
            series_id="DGS10", canonical_series_name="us_10y_nominal_yield", registry_version=registry_version, tier=SeriesTier.CORE_XAUUSD_DRIVER,
            economic_category="nominal_rates", expected_native_frequency="D", expected_units=("%",), expected_seasonal_adjustment="NSA",
            target_macro_instrument_id="us_10y_nominal_yield", normalization_kind=MacroUnit.PERCENT, revision_policy_id=revision_policy_id,
            release_availability_policy_id=release_availability_policy_id_daily, default_observation_start=default_observation_start,
            notes="10-Year Treasury Constant Maturity Rate.",
        ),
        create_curated_series_spec(
            series_id="CPIAUCSL", canonical_series_name="us_cpi_all_urban", registry_version=registry_version, tier=SeriesTier.CORE_XAUUSD_DRIVER,
            economic_category="inflation", expected_native_frequency="M", expected_units=("Index 1982-1984=100",), expected_seasonal_adjustment="SA",
            target_macro_instrument_id="us_cpi_all_urban", normalization_kind=MacroUnit.INDEX, revision_policy_id=revision_policy_id,
            release_availability_policy_id=release_availability_policy_id_monthly, default_observation_start=default_observation_start,
            notes="Consumer Price Index for All Urban Consumers: All Items.",
        ),
        create_curated_series_spec(
            series_id="DFF", canonical_series_name="effective_federal_funds_rate", registry_version=registry_version, tier=SeriesTier.CORE_XAUUSD_DRIVER,
            economic_category="policy_rate", expected_native_frequency="D", expected_units=("%",), expected_seasonal_adjustment="NSA",
            target_macro_instrument_id="effective_federal_funds_rate", normalization_kind=MacroUnit.RATE, revision_policy_id=revision_policy_id,
            release_availability_policy_id=release_availability_policy_id_daily, default_observation_start=default_observation_start,
            notes="Effective Federal Funds Rate.",
        ),
    )


def _daily_percent_spec(
    *, series_id: str, canonical_series_name: str, registry_version: int, tier: SeriesTier, economic_category: str, target_macro_instrument_id: str,
    revision_policy_id: str, release_availability_policy_id: str, default_observation_start: datetime, notes: str,
) -> CuratedFredSeriesSpec:
    """Shared shape for the several extended-universe series that are
    all daily, percent-denominated, not-seasonally-adjusted rates --
    avoids an untyped `**kwargs` dict splat (mypy cannot verify a
    heterogeneous `dict[str, object]` splat against a typed signature)."""
    return create_curated_series_spec(
        series_id=series_id, canonical_series_name=canonical_series_name, registry_version=registry_version, tier=tier,
        economic_category=economic_category, expected_native_frequency="D", expected_units=("%",), expected_seasonal_adjustment="NSA",
        normalization_kind=MacroUnit.PERCENT, target_macro_instrument_id=target_macro_instrument_id, revision_policy_id=revision_policy_id,
        release_availability_policy_id=release_availability_policy_id, default_observation_start=default_observation_start, enabled=False, notes=notes,
    )


def default_extended_series_specs(
    *, registry_version: int, revision_policy_id: str, release_availability_policy_id_daily: str, release_availability_policy_id_monthly: str,
    release_availability_policy_id_weekly: str, default_observation_start: datetime,
) -> tuple[CuratedFredSeriesSpec, ...]:
    """Carefully reviewed, but NOT individually fixture-verified this
    phase -- see module docstring. All `enabled=False` by default."""
    return (
        _daily_percent_spec(series_id="T10YIE", canonical_series_name="us_10y_breakeven_inflation", registry_version=registry_version, tier=SeriesTier.SECONDARY_MACRO_DRIVER, economic_category="inflation_expectations", target_macro_instrument_id="us_10y_breakeven_inflation", revision_policy_id=revision_policy_id, release_availability_policy_id=release_availability_policy_id_daily, default_observation_start=default_observation_start, notes="10-Year Breakeven Inflation Rate."),
        _daily_percent_spec(series_id="T5YIE", canonical_series_name="us_5y_breakeven_inflation", registry_version=registry_version, tier=SeriesTier.SECONDARY_MACRO_DRIVER, economic_category="inflation_expectations", target_macro_instrument_id="us_5y_breakeven_inflation", revision_policy_id=revision_policy_id, release_availability_policy_id=release_availability_policy_id_daily, default_observation_start=default_observation_start, notes="5-Year Breakeven Inflation Rate."),
        _daily_percent_spec(series_id="DGS2", canonical_series_name="us_2y_nominal_yield", registry_version=registry_version, tier=SeriesTier.SECONDARY_MACRO_DRIVER, economic_category="nominal_rates", target_macro_instrument_id="us_2y_nominal_yield", revision_policy_id=revision_policy_id, release_availability_policy_id=release_availability_policy_id_daily, default_observation_start=default_observation_start, notes="2-Year Treasury Constant Maturity Rate."),
        _daily_percent_spec(series_id="DGS5", canonical_series_name="us_5y_nominal_yield", registry_version=registry_version, tier=SeriesTier.SECONDARY_MACRO_DRIVER, economic_category="nominal_rates", target_macro_instrument_id="us_5y_nominal_yield", revision_policy_id=revision_policy_id, release_availability_policy_id=release_availability_policy_id_daily, default_observation_start=default_observation_start, notes="5-Year Treasury Constant Maturity Rate."),
        _daily_percent_spec(series_id="DGS30", canonical_series_name="us_30y_nominal_yield", registry_version=registry_version, tier=SeriesTier.SECONDARY_MACRO_DRIVER, economic_category="nominal_rates", target_macro_instrument_id="us_30y_nominal_yield", revision_policy_id=revision_policy_id, release_availability_policy_id=release_availability_policy_id_daily, default_observation_start=default_observation_start, notes="30-Year Treasury Constant Maturity Rate."),
        create_curated_series_spec(series_id="DTWEXBGS", canonical_series_name="us_dollar_broad_index", registry_version=registry_version, tier=SeriesTier.SECONDARY_MACRO_DRIVER, economic_category="fx", expected_native_frequency="D", expected_units=("Index Jan 2006=100",), expected_seasonal_adjustment="NSA", normalization_kind=MacroUnit.INDEX, target_macro_instrument_id="us_dollar_broad_index", revision_policy_id=revision_policy_id, release_availability_policy_id=release_availability_policy_id_daily, default_observation_start=default_observation_start, enabled=False, notes="Nominal Broad U.S. Dollar Index (Goods and Services)."),
        create_curated_series_spec(series_id="UNRATE", canonical_series_name="us_unemployment_rate", registry_version=registry_version, tier=SeriesTier.REGIME_CONTEXT, economic_category="labor_market", expected_native_frequency="M", expected_units=("%",), expected_seasonal_adjustment="SA", normalization_kind=MacroUnit.PERCENT, target_macro_instrument_id="us_unemployment_rate", revision_policy_id=revision_policy_id, release_availability_policy_id=release_availability_policy_id_monthly, default_observation_start=default_observation_start, enabled=False, notes="Civilian Unemployment Rate."),
        create_curated_series_spec(series_id="PAYEMS", canonical_series_name="us_nonfarm_payrolls", registry_version=registry_version, tier=SeriesTier.REGIME_CONTEXT, economic_category="labor_market", expected_native_frequency="M", expected_units=("Thous. of Persons",), expected_seasonal_adjustment="SA", normalization_kind=MacroUnit.INDEX, target_macro_instrument_id="us_nonfarm_payrolls", revision_policy_id=revision_policy_id, release_availability_policy_id=release_availability_policy_id_monthly, default_observation_start=default_observation_start, enabled=False, notes="All Employees, Total Nonfarm."),
        create_curated_series_spec(series_id="PCEPI", canonical_series_name="us_pce_price_index", registry_version=registry_version, tier=SeriesTier.REGIME_CONTEXT, economic_category="inflation", expected_native_frequency="M", expected_units=("Index 2017=100",), expected_seasonal_adjustment="SA", normalization_kind=MacroUnit.INDEX, target_macro_instrument_id="us_pce_price_index", revision_policy_id=revision_policy_id, release_availability_policy_id=release_availability_policy_id_monthly, default_observation_start=default_observation_start, enabled=False, notes="Personal Consumption Expenditures: Chain-type Price Index."),
        create_curated_series_spec(series_id="PCEPILFE", canonical_series_name="us_core_pce_price_index", registry_version=registry_version, tier=SeriesTier.REGIME_CONTEXT, economic_category="inflation", expected_native_frequency="M", expected_units=("Index 2017=100",), expected_seasonal_adjustment="SA", normalization_kind=MacroUnit.INDEX, target_macro_instrument_id="us_core_pce_price_index", revision_policy_id=revision_policy_id, release_availability_policy_id=release_availability_policy_id_monthly, default_observation_start=default_observation_start, enabled=False, notes="PCE Excluding Food and Energy (core PCE)."),
        create_curated_series_spec(series_id="INDPRO", canonical_series_name="us_industrial_production", registry_version=registry_version, tier=SeriesTier.REGIME_CONTEXT, economic_category="output", expected_native_frequency="M", expected_units=("Index 2017=100",), expected_seasonal_adjustment="SA", normalization_kind=MacroUnit.INDEX, target_macro_instrument_id="us_industrial_production", revision_policy_id=revision_policy_id, release_availability_policy_id=release_availability_policy_id_monthly, default_observation_start=default_observation_start, enabled=False, notes="Industrial Production: Total Index."),
        create_curated_series_spec(series_id="VIXCLS", canonical_series_name="cboe_vix_close", registry_version=registry_version, tier=SeriesTier.REGIME_CONTEXT, economic_category="volatility", expected_native_frequency="D", expected_units=("Index",), expected_seasonal_adjustment="NSA", normalization_kind=MacroUnit.INDEX, target_macro_instrument_id="cboe_vix_close", revision_policy_id=revision_policy_id, release_availability_policy_id=release_availability_policy_id_daily, default_observation_start=default_observation_start, enabled=False, notes="CBOE Volatility Index: VIX."),
        create_curated_series_spec(series_id="WALCL", canonical_series_name="fed_total_assets", registry_version=registry_version, tier=SeriesTier.REGIME_CONTEXT, economic_category="liquidity", expected_native_frequency="W", expected_units=("Mil. of $",), expected_seasonal_adjustment="NSA", normalization_kind=MacroUnit.INDEX, target_macro_instrument_id="fed_total_assets", revision_policy_id=revision_policy_id, release_availability_policy_id=release_availability_policy_id_weekly, default_observation_start=default_observation_start, enabled=False, notes="Federal Reserve total assets (H.4.1)."),
        create_curated_series_spec(series_id="M2SL", canonical_series_name="us_m2_money_stock", registry_version=registry_version, tier=SeriesTier.REGIME_CONTEXT, economic_category="liquidity", expected_native_frequency="M", expected_units=("Bil. of $",), expected_seasonal_adjustment="SA", normalization_kind=MacroUnit.INDEX, target_macro_instrument_id="us_m2_money_stock", revision_policy_id=revision_policy_id, release_availability_policy_id=release_availability_policy_id_monthly, default_observation_start=default_observation_start, enabled=False, notes="M2 Money Stock."),
    )
