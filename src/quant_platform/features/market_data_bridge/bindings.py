"""Immutable source bindings (Milestone 10, Phase 4D): the exact, pinned
identity of one `market_data`-backed input this bridge is permitted to
read -- the base-asset candle timeline, one macro series, or one
cross-asset driver mapping.

EVERY BINDING PINS AN EXACT, IMMUTABLE VERSION. No field here ever
accepts a mutable alias (`"latest"`/`"current"`/`"newest"`/`"active"`/a
provider default) -- `_reject_mutable_alias` (used by every binding's
`__post_init__`) refuses one outright (`SourceBindingError`). This is
what makes "changing a bound source version changes the resulting
research dataset identity" true by construction: a binding's own
`binding_id` is a deterministic content hash of every other field
(mirroring the `create_*`-then-`compute_content_id` pattern every other
identity object in `market_data` already uses -- see e.g.
`collectors.curated.revision_policy.create_revision_policy`), so pinning
a different exact source version always produces a different binding,
which flows into `lineage.py`'s `market_data_lineage` payload and
therefore into the research dataset manifest's own version identity
(see that module's docstring for the exact mechanism).

Each binding's `component_manifest_id`/`pinned_dataset_id`/
`combined_manifest_id`/`combined_universe_manifest_id` fields are
resolved-and-VERIFIED (never merely trusted) against the live
`market_data` repository by the corresponding `*_adapter.py` module --
see `base_asset_adapter.py`/`macro_adapter.py`/`cross_asset_adapter.py`
and `SourceVerificationError`'s docstring in `core.exceptions` for
exactly what "verified" means for each source kind (base-asset bindings
are verified against an exact, content-addressed partition-member
reconstruction; macro/cross-asset bindings are verified by reproducing
the pinned manifest's own `semantic_digest` from a live store read,
fail-closed on any mismatch)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from quant_platform.core.exceptions import SourceBindingError
from quant_platform.core.types import Timeframe
from quant_platform.market_data.collectors.cross_asset.instrument_form import InstrumentForm, ProxyPolicy
from quant_platform.market_data.collectors.curated.revision_policy import RevisionPolicyKind
from quant_platform.market_data.identity import compute_content_id, require_non_empty

__all__ = [
    "BASE_ASSET_BINDING_KIND",
    "CROSS_ASSET_BINDING_KIND",
    "MACRO_BINDING_KIND",
    "BaseAssetDatasetBinding",
    "CrossAssetDatasetBinding",
    "MacroDatasetBinding",
    "create_base_asset_binding",
    "create_cross_asset_dataset_binding",
    "create_macro_dataset_binding",
]

BASE_ASSET_BINDING_KIND = "market_data_bridge_base_asset_binding"
MACRO_BINDING_KIND = "market_data_bridge_macro_binding"
CROSS_ASSET_BINDING_KIND = "market_data_bridge_cross_asset_binding"

def _as_dict(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SourceBindingError(f"{field_name} must be a JSON object, got {type(value).__name__}")
    return value


_MUTABLE_ALIAS_TOKENS = frozenset(
    {"latest", "current", "newest", "active", "default", "provider_default", "head", "head~0"}
)


def _reject_mutable_alias(value: str, *, field_name: str) -> None:
    require_non_empty(value, field_name=field_name)
    if value.strip().lower() in _MUTABLE_ALIAS_TOKENS:
        raise SourceBindingError(
            f"{field_name}={value!r} looks like a mutable alias, not a pinned immutable identity -- every "
            "market-data source binding must pin an exact, content-addressed dataset/manifest/component id "
            "(e.g. a sha256 hex digest), never a mutable label that could silently resolve to different "
            "content on a later read.",
            context={"field": field_name, "value": value},
        )


def _require_tz_aware_range(start: pd.Timestamp | None, end: pd.Timestamp | None, *, field_prefix: str) -> None:
    if start is None and end is None:
        return
    if start is None or end is None:
        raise SourceBindingError(f"{field_prefix}_start and {field_prefix}_end must both be set, or both be None")
    if start.tzinfo is None or end.tzinfo is None:
        raise SourceBindingError(f"{field_prefix}_start/{field_prefix}_end must be timezone-aware")
    if end <= start:
        raise SourceBindingError(f"{field_prefix}_end ({end}) must be after {field_prefix}_start ({start})")


@dataclass(frozen=True, slots=True)
class BaseAssetDatasetBinding:
    """Pins the exact XAUUSD (or any other) base-asset candle timeline
    this bridge reads from `market_data`'s Phase 2 durable repository
    (`events.MarketEventStore` + `manifests.DatasetManifestStore`/
    `partitions.PartitionStore`) -- resolved into an actual
    `pandas.DataFrame` by `base_asset_adapter.py`."""

    canonical_instrument_id: str
    provider: str
    pinned_dataset_id: str
    """The exact `market_data.manifests.DatasetManifest.dataset_id` this
    binding reads. `base_asset_adapter.verify_base_asset_binding` requires
    this to equal the CURRENT manifest for `(RAW_MARKET_EVENTS, provider,
    canonical_instrument_id)` -- `partitions.PartitionStore` is
    current-version-only storage (see its own module docstring), so an
    OLDER, superseded `dataset_id` can no longer be reconstructed byte-
    for-byte once its partitions have been rebuilt by a later commit;
    that case fails closed (`SourceVerificationError`) rather than
    silently substituting the current version."""
    timeframe: Timeframe
    expected_event_kind: str = "candle"
    session_policy_id: str = "continuous_utc"
    availability_policy_id: str = "close_time_as_availability"
    """`market_data.candles.Candle` carries no `availability_time` field
    (unlike Phase 4B/4C's richer records) -- this bridge's own honest,
    documented policy is that a candle becomes visible exactly at its own
    `close_time` (`open_time + timeframe.duration`), matching
    `features.engine.FeatureEngine.compute`'s own availability-instant
    derivation for the base timeframe exactly. This id exists so a future,
    different base-asset availability policy is representable (and
    identity-visible) without a binding schema change."""
    required_coverage_start: pd.Timestamp | None = None
    required_coverage_end: pd.Timestamp | None = None
    binding_id: str = ""

    def __post_init__(self) -> None:
        _reject_mutable_alias(self.canonical_instrument_id, field_name="canonical_instrument_id")
        _reject_mutable_alias(self.provider, field_name="provider")
        _reject_mutable_alias(self.pinned_dataset_id, field_name="pinned_dataset_id")
        require_non_empty(self.expected_event_kind, field_name="expected_event_kind")
        require_non_empty(self.session_policy_id, field_name="session_policy_id")
        require_non_empty(self.availability_policy_id, field_name="availability_policy_id")
        _require_tz_aware_range(self.required_coverage_start, self.required_coverage_end, field_prefix="required_coverage")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": BASE_ASSET_BINDING_KIND,
            "canonical_instrument_id": self.canonical_instrument_id,
            "provider": self.provider,
            "pinned_dataset_id": self.pinned_dataset_id,
            "timeframe": self.timeframe.value,
            "expected_event_kind": self.expected_event_kind,
            "session_policy_id": self.session_policy_id,
            "availability_policy_id": self.availability_policy_id,
            "required_coverage_start": (None if self.required_coverage_start is None else self.required_coverage_start.isoformat()),
            "required_coverage_end": (None if self.required_coverage_end is None else self.required_coverage_end.isoformat()),
            "binding_id": self.binding_id,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["binding_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> BaseAssetDatasetBinding:
        return cls(
            canonical_instrument_id=str(raw["canonical_instrument_id"]), provider=str(raw["provider"]),
            pinned_dataset_id=str(raw["pinned_dataset_id"]), timeframe=Timeframe(raw["timeframe"]),
            expected_event_kind=str(raw.get("expected_event_kind", "candle")),
            session_policy_id=str(raw.get("session_policy_id", "continuous_utc")),
            availability_policy_id=str(raw.get("availability_policy_id", "close_time_as_availability")),
            required_coverage_start=(None if raw.get("required_coverage_start") is None else pd.Timestamp(str(raw["required_coverage_start"]))),
            required_coverage_end=(None if raw.get("required_coverage_end") is None else pd.Timestamp(str(raw["required_coverage_end"]))),
            binding_id=str(raw.get("binding_id", "")),
        )


def create_base_asset_binding(
    *, canonical_instrument_id: str, provider: str, pinned_dataset_id: str, timeframe: Timeframe,
    expected_event_kind: str = "candle", session_policy_id: str = "continuous_utc",
    availability_policy_id: str = "close_time_as_availability",
    required_coverage_start: pd.Timestamp | None = None, required_coverage_end: pd.Timestamp | None = None,
) -> BaseAssetDatasetBinding:
    provisional = BaseAssetDatasetBinding(
        canonical_instrument_id=canonical_instrument_id, provider=provider, pinned_dataset_id=pinned_dataset_id,
        timeframe=timeframe, expected_event_kind=expected_event_kind, session_policy_id=session_policy_id,
        availability_policy_id=availability_policy_id, required_coverage_start=required_coverage_start,
        required_coverage_end=required_coverage_end, binding_id="0" * 64,
    )
    return replace(provisional, binding_id=compute_content_id(BASE_ASSET_BINDING_KIND, provisional.to_identity_payload()))


@dataclass(frozen=True, slots=True)
class MacroDatasetBinding:
    """Pins one exact macro series version this bridge reads from
    `market_data`'s Phase 4B curated FRED layer (`collectors.curated.
    macro_observation.CuratedObservationStore` + `collectors.curated.
    datasets.ComponentDatasetManifestStore`) -- resolved into the
    `value`/`release_time` shape `features.macro.macro_features` expects
    by `macro_adapter.py`."""

    curated_registry_id: str
    combined_universe_manifest_id: str
    series_id: str
    canonical_series_name: str
    provider: str
    component_manifest_id: str
    """The exact `collectors.curated.datasets.ComponentDatasetManifest.
    component_manifest_id` this binding reads -- verified by
    `macro_adapter.verify_macro_binding` via recomputing that manifest's
    own `semantic_digest` from a live `CuratedObservationStore.
    read_observations` call and requiring an exact match (fail closed on
    any mismatch, e.g. new observations appended since this binding was
    pinned -- see `SourceVerificationError`)."""
    revision_policy_id: str
    revision_policy_kind: RevisionPolicyKind
    availability_policy_id: str
    native_frequency: str
    normalized_unit: str
    required: bool = True
    binding_id: str = ""

    def __post_init__(self) -> None:
        _reject_mutable_alias(self.curated_registry_id, field_name="curated_registry_id")
        _reject_mutable_alias(self.combined_universe_manifest_id, field_name="combined_universe_manifest_id")
        _reject_mutable_alias(self.component_manifest_id, field_name="component_manifest_id")
        require_non_empty(self.series_id, field_name="series_id")
        require_non_empty(self.canonical_series_name, field_name="canonical_series_name")
        require_non_empty(self.provider, field_name="provider")
        require_non_empty(self.revision_policy_id, field_name="revision_policy_id")
        require_non_empty(self.availability_policy_id, field_name="availability_policy_id")
        require_non_empty(self.native_frequency, field_name="native_frequency")
        require_non_empty(self.normalized_unit, field_name="normalized_unit")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": MACRO_BINDING_KIND,
            "curated_registry_id": self.curated_registry_id,
            "combined_universe_manifest_id": self.combined_universe_manifest_id,
            "series_id": self.series_id,
            "canonical_series_name": self.canonical_series_name,
            "provider": self.provider,
            "component_manifest_id": self.component_manifest_id,
            "revision_policy_id": self.revision_policy_id,
            "revision_policy_kind": self.revision_policy_kind.value,
            "availability_policy_id": self.availability_policy_id,
            "native_frequency": self.native_frequency,
            "normalized_unit": self.normalized_unit,
            "required": self.required,
            "binding_id": self.binding_id,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["binding_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> MacroDatasetBinding:
        return cls(
            curated_registry_id=str(raw["curated_registry_id"]),
            combined_universe_manifest_id=str(raw["combined_universe_manifest_id"]),
            series_id=str(raw["series_id"]), canonical_series_name=str(raw["canonical_series_name"]),
            provider=str(raw["provider"]), component_manifest_id=str(raw["component_manifest_id"]),
            revision_policy_id=str(raw["revision_policy_id"]),
            revision_policy_kind=RevisionPolicyKind(raw["revision_policy_kind"]),
            availability_policy_id=str(raw["availability_policy_id"]), native_frequency=str(raw["native_frequency"]),
            normalized_unit=str(raw["normalized_unit"]), required=bool(raw.get("required", True)),
            binding_id=str(raw.get("binding_id", "")),
        )


def create_macro_dataset_binding(
    *, curated_registry_id: str, combined_universe_manifest_id: str, series_id: str, canonical_series_name: str,
    provider: str, component_manifest_id: str, revision_policy_id: str, revision_policy_kind: RevisionPolicyKind,
    availability_policy_id: str, native_frequency: str, normalized_unit: str, required: bool = True,
) -> MacroDatasetBinding:
    provisional = MacroDatasetBinding(
        curated_registry_id=curated_registry_id, combined_universe_manifest_id=combined_universe_manifest_id,
        series_id=series_id, canonical_series_name=canonical_series_name, provider=provider,
        component_manifest_id=component_manifest_id, revision_policy_id=revision_policy_id,
        revision_policy_kind=revision_policy_kind, availability_policy_id=availability_policy_id,
        native_frequency=native_frequency, normalized_unit=normalized_unit, required=required, binding_id="0" * 64,
    )
    return replace(provisional, binding_id=compute_content_id(MACRO_BINDING_KIND, provisional.to_identity_payload()))


@dataclass(frozen=True, slots=True)
class CrossAssetDatasetBinding:
    """Pins one exact cross-asset driver mapping version this bridge
    reads from `market_data`'s Phase 4C cross-asset layer (`collectors.
    cross_asset.market_record.MarketDriverBarStore` + `collectors.
    cross_asset.datasets.ComponentMarketDatasetManifestStore`) --
    resolved into an `open_time`-indexed OHLCV `pandas.DataFrame` by
    `cross_asset_adapter.py`."""

    curated_registry_id: str
    combined_manifest_id: str
    canonical_driver_id: str
    mapping_id: str
    provider: str
    provider_symbol: str
    component_manifest_id: str
    """The exact `collectors.cross_asset.datasets.
    ComponentMarketDatasetManifest.component_manifest_id` this binding
    reads -- verified identically to `MacroDatasetBinding.
    component_manifest_id` (see `cross_asset_adapter.
    verify_cross_asset_binding`)."""
    instrument_form: InstrumentForm
    proxy_policy: ProxyPolicy
    """Reused directly from `market_data.collectors.cross_asset.
    instrument_form` -- carries the full basis/roll/tracking-error/
    currency/session/adjustment difference notes spec Section 12 requires
    a research dataset to be able to answer (e.g. "was dollar strength a
    cash index or an ETF proxy"), without duplicating that vocabulary."""
    adjustment_policy_id: str
    continuation_policy_id: str | None
    session_policy_id: str
    availability_policy_id: str
    timeframe: Timeframe
    required: bool = False
    binding_id: str = ""

    def __post_init__(self) -> None:
        _reject_mutable_alias(self.curated_registry_id, field_name="curated_registry_id")
        _reject_mutable_alias(self.combined_manifest_id, field_name="combined_manifest_id")
        _reject_mutable_alias(self.component_manifest_id, field_name="component_manifest_id")
        require_non_empty(self.canonical_driver_id, field_name="canonical_driver_id")
        require_non_empty(self.mapping_id, field_name="mapping_id")
        require_non_empty(self.provider, field_name="provider")
        require_non_empty(self.provider_symbol, field_name="provider_symbol")
        require_non_empty(self.adjustment_policy_id, field_name="adjustment_policy_id")
        require_non_empty(self.session_policy_id, field_name="session_policy_id")
        require_non_empty(self.availability_policy_id, field_name="availability_policy_id")
        if self.instrument_form is InstrumentForm.ETF and not self.proxy_policy.is_proxy:
            raise SourceBindingError(
                f"CrossAssetDatasetBinding for driver {self.canonical_driver_id!r}: instrument_form=ETF must be "
                "classified is_proxy=True -- see market_data.collectors.cross_asset.instrument_form's own "
                "ETF-is-always-a-proxy structural rule. No mapping may silently label an ETF as its underlying.",
                context={"canonical_driver_id": self.canonical_driver_id},
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": CROSS_ASSET_BINDING_KIND,
            "curated_registry_id": self.curated_registry_id,
            "combined_manifest_id": self.combined_manifest_id,
            "canonical_driver_id": self.canonical_driver_id,
            "mapping_id": self.mapping_id,
            "provider": self.provider,
            "provider_symbol": self.provider_symbol,
            "component_manifest_id": self.component_manifest_id,
            "instrument_form": self.instrument_form.value,
            "proxy_policy": self.proxy_policy.to_json_dict(),
            "adjustment_policy_id": self.adjustment_policy_id,
            "continuation_policy_id": self.continuation_policy_id,
            "session_policy_id": self.session_policy_id,
            "availability_policy_id": self.availability_policy_id,
            "timeframe": self.timeframe.value,
            "required": self.required,
            "binding_id": self.binding_id,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["binding_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CrossAssetDatasetBinding:
        return cls(
            curated_registry_id=str(raw["curated_registry_id"]), combined_manifest_id=str(raw["combined_manifest_id"]),
            canonical_driver_id=str(raw["canonical_driver_id"]), mapping_id=str(raw["mapping_id"]),
            provider=str(raw["provider"]), provider_symbol=str(raw["provider_symbol"]),
            component_manifest_id=str(raw["component_manifest_id"]), instrument_form=InstrumentForm(raw["instrument_form"]),
            proxy_policy=ProxyPolicy.from_json_dict(_as_dict(raw["proxy_policy"], field_name="proxy_policy")),
            adjustment_policy_id=str(raw["adjustment_policy_id"]),
            continuation_policy_id=(None if raw.get("continuation_policy_id") is None else str(raw["continuation_policy_id"])),
            session_policy_id=str(raw["session_policy_id"]), availability_policy_id=str(raw["availability_policy_id"]),
            timeframe=Timeframe(raw["timeframe"]), required=bool(raw.get("required", False)),
            binding_id=str(raw.get("binding_id", "")),
        )


def create_cross_asset_dataset_binding(
    *, curated_registry_id: str, combined_manifest_id: str, canonical_driver_id: str, mapping_id: str, provider: str,
    provider_symbol: str, component_manifest_id: str, instrument_form: InstrumentForm, proxy_policy: ProxyPolicy,
    adjustment_policy_id: str, continuation_policy_id: str | None, session_policy_id: str, availability_policy_id: str,
    timeframe: Timeframe, required: bool = False,
) -> CrossAssetDatasetBinding:
    provisional = CrossAssetDatasetBinding(
        curated_registry_id=curated_registry_id, combined_manifest_id=combined_manifest_id,
        canonical_driver_id=canonical_driver_id, mapping_id=mapping_id, provider=provider,
        provider_symbol=provider_symbol, component_manifest_id=component_manifest_id, instrument_form=instrument_form,
        proxy_policy=proxy_policy, adjustment_policy_id=adjustment_policy_id,
        continuation_policy_id=continuation_policy_id, session_policy_id=session_policy_id,
        availability_policy_id=availability_policy_id, timeframe=timeframe, required=required, binding_id="0" * 64,
    )
    return replace(provisional, binding_id=compute_content_id(CROSS_ASSET_BINDING_KIND, provisional.to_identity_payload()))
