"""Immutable, versioned, content-addressed instrument/timeframe mapping
specs (Milestone 10, Phase 3) -- external source symbols and timeframe
labels never resolve to a canonical `instrument_id`/`Timeframe` through
ad hoc, unversioned logic. A mapping spec's own `mapping_id` is a pure
function of its entries; two ingestion operations using the SAME mapping
content always resolve identically, and any change to that content
(adding/removing/re-pointing an alias) produces a NEW `mapping_id` --
which, threaded into `SourceManifest`/`ProvenanceRecord`/
`BackfillPlan` identity, is exactly what makes "mapping changes produce
new dataset lineage/version" hold structurally rather than by
convention.

NOT A GLOBAL REGISTRY: this module defines no built-in table of every
future symbol (XAUUSD, DXY, WTI, ...) as platform-wide truth -- per the
specification's own explicit instruction. A caller constructs whatever
`InstrumentMappingSpec` their own ingestion operation needs; two
different operations may use two different, independently versioned
specs without conflict."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.core.exceptions import InstrumentMappingError, TimeframeMappingError
from quant_platform.core.types import Timeframe
from quant_platform.market_data.identity import compute_content_id, require_non_empty

__all__ = [
    "INSTRUMENT_MAPPING_KIND",
    "TIMEFRAME_MAPPING_KIND",
    "InstrumentMappingEntry",
    "InstrumentMappingSpec",
    "TimeframeMappingEntry",
    "TimeframeMappingSpec",
    "create_instrument_mapping_spec",
    "create_timeframe_mapping_spec",
    "default_timeframe_mapping_spec",
    "resolve_instrument_id",
    "resolve_timeframe",
]

INSTRUMENT_MAPPING_KIND = "instrument_mapping_spec"
TIMEFRAME_MAPPING_KIND = "timeframe_mapping_spec"


# --------------------------------------------------------------------------
# Instrument/symbol mapping.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class InstrumentMappingEntry:
    source_symbol: str
    instrument_id: str
    provider: str | None = None
    """`None` means this entry applies across every provider (a
    wildcard); a specific value SCOPES the entry to one provider only,
    letting the SAME `source_symbol` (e.g. `"GC"`, a common futures
    ticker root) resolve to different instruments for different
    providers without ambiguity -- resolution always prefers an exact
    `(source_symbol, provider)` match over a wildcard one (see
    `resolve_instrument_id`)."""

    def __post_init__(self) -> None:
        require_non_empty(self.source_symbol, field_name="InstrumentMappingEntry.source_symbol")
        require_non_empty(self.instrument_id, field_name="InstrumentMappingEntry.instrument_id")

    def to_json_dict(self) -> dict[str, object]:
        return {"source_symbol": self.source_symbol, "instrument_id": self.instrument_id, "provider": self.provider}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> InstrumentMappingEntry:
        return cls(
            source_symbol=str(raw["source_symbol"]), instrument_id=str(raw["instrument_id"]),
            provider=(None if raw.get("provider") is None else str(raw["provider"])),
        )


@dataclass(frozen=True, slots=True)
class InstrumentMappingSpec:
    mapping_id: str
    mapping_version: int
    entries: tuple[InstrumentMappingEntry, ...]

    def __post_init__(self) -> None:
        if self.mapping_version < 1:
            raise InstrumentMappingError(f"InstrumentMappingSpec.mapping_version must be >= 1, got {self.mapping_version}")
        seen_keys: set[tuple[str, str | None]] = set()
        for entry in self.entries:
            key = (entry.source_symbol, entry.provider)
            if key in seen_keys:
                raise InstrumentMappingError(f"InstrumentMappingSpec has a duplicate entry for (source_symbol={entry.source_symbol!r}, provider={entry.provider!r})")
            seen_keys.add(key)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": INSTRUMENT_MAPPING_KIND, "mapping_id": self.mapping_id, "mapping_version": self.mapping_version,
            "entries": [e.to_json_dict() for e in self.entries],
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["mapping_id"]
        entries = [e.to_json_dict() for e in self.entries]
        payload["entries"] = sorted(entries, key=lambda e: (str(e["source_symbol"]), str(e["provider"])))
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> InstrumentMappingSpec:
        from quant_platform.ml.persistence import as_json_dict, as_json_list

        entries = tuple(InstrumentMappingEntry.from_json_dict(as_json_dict(e, field_name="entry")) for e in as_json_list(raw["entries"], field_name="entries"))
        return cls(mapping_id=str(raw["mapping_id"]), mapping_version=int(str(raw["mapping_version"])), entries=entries)


def create_instrument_mapping_spec(*, mapping_version: int, entries: tuple[InstrumentMappingEntry, ...]) -> InstrumentMappingSpec:
    provisional = InstrumentMappingSpec(mapping_id="0" * 64, mapping_version=mapping_version, entries=entries)
    mapping_id = compute_content_id(INSTRUMENT_MAPPING_KIND, provisional.to_identity_payload())
    return InstrumentMappingSpec(mapping_id=mapping_id, mapping_version=mapping_version, entries=entries)


def resolve_instrument_id(spec: InstrumentMappingSpec, *, source_symbol: str, provider: str) -> str:
    """Fails closed (`InstrumentMappingError`) rather than guessing: an
    unmapped symbol never silently passes through as its own
    `instrument_id`."""
    by_key = {(e.source_symbol, e.provider): e.instrument_id for e in spec.entries}
    exact = by_key.get((source_symbol, provider))
    if exact is not None:
        return exact
    wildcard = by_key.get((source_symbol, None))
    if wildcard is not None:
        return wildcard
    raise InstrumentMappingError(f"No instrument mapping for source_symbol={source_symbol!r} provider={provider!r} in mapping {spec.mapping_id!r}")


# --------------------------------------------------------------------------
# Timeframe mapping.
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TimeframeMappingEntry:
    source_label: str
    timeframe: Timeframe

    def __post_init__(self) -> None:
        require_non_empty(self.source_label, field_name="TimeframeMappingEntry.source_label")

    def to_json_dict(self) -> dict[str, object]:
        return {"source_label": self.source_label, "timeframe": self.timeframe.value}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> TimeframeMappingEntry:
        return cls(source_label=str(raw["source_label"]), timeframe=Timeframe(raw["timeframe"]))


@dataclass(frozen=True, slots=True)
class TimeframeMappingSpec:
    mapping_id: str
    mapping_version: int
    entries: tuple[TimeframeMappingEntry, ...]

    def __post_init__(self) -> None:
        if self.mapping_version < 1:
            raise TimeframeMappingError(f"TimeframeMappingSpec.mapping_version must be >= 1, got {self.mapping_version}")
        seen_labels: set[str] = set()
        for entry in self.entries:
            if entry.source_label in seen_labels:
                raise TimeframeMappingError(f"TimeframeMappingSpec has a duplicate entry for source_label={entry.source_label!r}")
            seen_labels.add(entry.source_label)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": TIMEFRAME_MAPPING_KIND, "mapping_id": self.mapping_id, "mapping_version": self.mapping_version,
            "entries": [e.to_json_dict() for e in self.entries],
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["mapping_id"]
        entries = [e.to_json_dict() for e in self.entries]
        payload["entries"] = sorted(entries, key=lambda e: str(e["source_label"]))
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> TimeframeMappingSpec:
        from quant_platform.ml.persistence import as_json_dict, as_json_list

        entries = tuple(TimeframeMappingEntry.from_json_dict(as_json_dict(e, field_name="entry")) for e in as_json_list(raw["entries"], field_name="entries"))
        return cls(mapping_id=str(raw["mapping_id"]), mapping_version=int(str(raw["mapping_version"])), entries=entries)


def create_timeframe_mapping_spec(*, mapping_version: int, entries: tuple[TimeframeMappingEntry, ...]) -> TimeframeMappingSpec:
    provisional = TimeframeMappingSpec(mapping_id="0" * 64, mapping_version=mapping_version, entries=entries)
    mapping_id = compute_content_id(TIMEFRAME_MAPPING_KIND, provisional.to_identity_payload())
    return TimeframeMappingSpec(mapping_id=mapping_id, mapping_version=mapping_version, entries=entries)


def resolve_timeframe(spec: TimeframeMappingSpec, *, source_label: str) -> Timeframe:
    for entry in spec.entries:
        if entry.source_label == source_label:
            return entry.timeframe
    raise TimeframeMappingError(f"No timeframe mapping for source_label={source_label!r} in mapping {spec.mapping_id!r}")


def default_timeframe_mapping_spec() -> TimeframeMappingSpec:
    """A reasonable, explicit starting point covering the aliases the
    specification names (`"1m"`/`"M1"`, `"5m"`/`"M5"`, ..., `"1d"`/
    `"D1"`) -- NOT the only legal spec; any caller may construct their
    own via `create_timeframe_mapping_spec` instead. Provided because
    every test/example in this phase needs SOME concrete spec, and
    hand-writing this exact table repeatedly would itself be a form of
    undocumented duplication."""
    pairs = (
        ("1m", Timeframe.M1), ("M1", Timeframe.M1),
        ("5m", Timeframe.M5), ("M5", Timeframe.M5),
        ("15m", Timeframe.M15), ("M15", Timeframe.M15),
        ("30m", Timeframe.M30), ("M30", Timeframe.M30),
        ("1h", Timeframe.H1), ("H1", Timeframe.H1),
        ("4h", Timeframe.H4), ("H4", Timeframe.H4),
        ("12h", Timeframe.H12), ("H12", Timeframe.H12),
        ("1d", Timeframe.D1), ("D1", Timeframe.D1),
    )
    entries = tuple(TimeframeMappingEntry(source_label=label, timeframe=tf) for label, tf in pairs)
    return create_timeframe_mapping_spec(mapping_version=1, entries=entries)
