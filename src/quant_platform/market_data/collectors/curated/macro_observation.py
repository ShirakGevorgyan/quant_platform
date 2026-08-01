"""Canonical curated macro observation (Milestone 10, Phase 4B) -- the
rich, content-addressed record type this whole layer exists to
produce. Deliberately a NEW type, not an overload of Phase 1's
`macro.MacroEvent` (which models a single normalized value + sequence
number for the ALREADY-TRUSTED durable repository) -- this record
additionally preserves native vs. normalized unit, native frequency,
the full vintage/realtime lineage, the resolved `availability_time` AND
the policy that produced it, and direct source/request/response
manifest references, none of which `MacroEvent` carries.

IDENTITY DISTINGUISHES ECONOMICALLY DISTINCT REVISIONS BY CONSTRUCTION:
`observation_id` is content-addressed over EVERY field, including
`realtime_start`/`value`/`availability_policy_id` -- two vintages of the
same `observation_date` with different values (or the same value but
resolved under a different policy) always produce DIFFERENT ids.
Nothing here ever collapses them.

NO PROVENANCE FIELD, DELIBERATELY: attaching a `provenance_id` at
construction time would be circular (a `ProvenanceRecord` is built FROM
this observation's own `observation_id`, which must exist first).
Phase 3's `provenance.ProvenanceStore` is reused UNCHANGED for that
binding, scoped via the very same `DatasetKind.MACRO_OBSERVATIONS`
Phase 4A already added -- a consumer wanting "the provenance reference"
looks it up via `(source_manifest_id, source_row_index)`, exactly like
every other Phase 3/4A source-derived record."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from quant_platform.core.exceptions import (
    CollectorError,
    ExperimentLockError,
    MarketDataLockError,
    MarketDataPersistenceError,
)
from quant_platform.core.json import canonical_json_bytes, parse_json_strict
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)
from quant_platform.ml.concurrency import experiment_lock

__all__ = ["CURATED_MACRO_OBSERVATION_KIND", "CuratedMacroObservation", "CuratedObservationStore", "create_curated_macro_observation"]

CURATED_MACRO_OBSERVATION_KIND = "curated_macro_observation"


@dataclass(frozen=True, slots=True)
class CuratedMacroObservation:
    observation_id: str
    series_id: str
    canonical_series_name: str
    target_macro_instrument_id: str
    observation_date: str
    """FRED's own observation PERIOD text (`"YYYY-MM-DD"`) -- never the
    availability proof; see module docstring / `availability.py`."""
    value: Decimal | None
    is_missing: bool
    normalized_unit: str
    native_unit: str
    native_frequency: str
    realtime_start: str | None
    realtime_end: str | None
    vintage_identity: str
    availability_time: datetime
    availability_policy_id: str
    request_manifest_id: str
    response_manifest_id: str
    source_manifest_id: str
    source_row_index: int

    def __post_init__(self) -> None:
        require_non_empty(self.series_id, field_name="CuratedMacroObservation.series_id")
        require_non_empty(self.canonical_series_name, field_name="CuratedMacroObservation.canonical_series_name")
        require_non_empty(self.target_macro_instrument_id, field_name="CuratedMacroObservation.target_macro_instrument_id")
        require_non_empty(self.observation_date, field_name="CuratedMacroObservation.observation_date")
        require_non_empty(self.normalized_unit, field_name="CuratedMacroObservation.normalized_unit")
        require_non_empty(self.native_unit, field_name="CuratedMacroObservation.native_unit")
        require_non_empty(self.native_frequency, field_name="CuratedMacroObservation.native_frequency")
        require_non_empty(self.vintage_identity, field_name="CuratedMacroObservation.vintage_identity")
        require_non_empty(self.availability_policy_id, field_name="CuratedMacroObservation.availability_policy_id")
        require_non_empty(self.request_manifest_id, field_name="CuratedMacroObservation.request_manifest_id")
        require_non_empty(self.response_manifest_id, field_name="CuratedMacroObservation.response_manifest_id")
        require_non_empty(self.source_manifest_id, field_name="CuratedMacroObservation.source_manifest_id")
        require_tz_aware(self.availability_time, field_name="CuratedMacroObservation.availability_time")
        if self.source_row_index < 0:
            raise CollectorError(f"CuratedMacroObservation.source_row_index must be >= 0, got {self.source_row_index}")
        if self.is_missing and self.value is not None:
            raise CollectorError("CuratedMacroObservation.value must be None when is_missing=True")
        if not self.is_missing and self.value is None:
            raise CollectorError("CuratedMacroObservation.value must not be None when is_missing=False")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": CURATED_MACRO_OBSERVATION_KIND, "observation_id": self.observation_id, "series_id": self.series_id,
            "canonical_series_name": self.canonical_series_name, "target_macro_instrument_id": self.target_macro_instrument_id,
            "observation_date": self.observation_date, "value": (None if self.value is None else str(self.value)), "is_missing": self.is_missing,
            "normalized_unit": self.normalized_unit, "native_unit": self.native_unit, "native_frequency": self.native_frequency,
            "realtime_start": self.realtime_start, "realtime_end": self.realtime_end, "vintage_identity": self.vintage_identity,
            "availability_time": serialize_timestamp(self.availability_time, field_name="availability_time"),
            "availability_policy_id": self.availability_policy_id, "request_manifest_id": self.request_manifest_id,
            "response_manifest_id": self.response_manifest_id, "source_manifest_id": self.source_manifest_id, "source_row_index": self.source_row_index,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["observation_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CuratedMacroObservation:
        raw_value = raw.get("value")
        return cls(
            observation_id=str(raw["observation_id"]), series_id=str(raw["series_id"]), canonical_series_name=str(raw["canonical_series_name"]),
            target_macro_instrument_id=str(raw["target_macro_instrument_id"]), observation_date=str(raw["observation_date"]),
            value=(None if raw_value is None else Decimal(str(raw_value))), is_missing=bool(raw["is_missing"]),
            normalized_unit=str(raw["normalized_unit"]), native_unit=str(raw["native_unit"]), native_frequency=str(raw["native_frequency"]),
            realtime_start=(None if raw.get("realtime_start") is None else str(raw["realtime_start"])),
            realtime_end=(None if raw.get("realtime_end") is None else str(raw["realtime_end"])), vintage_identity=str(raw["vintage_identity"]),
            availability_time=deserialize_timestamp(raw["availability_time"], field_name="availability_time"),
            availability_policy_id=str(raw["availability_policy_id"]), request_manifest_id=str(raw["request_manifest_id"]),
            response_manifest_id=str(raw["response_manifest_id"]), source_manifest_id=str(raw["source_manifest_id"]),
            source_row_index=int(str(raw["source_row_index"])),
        )


def create_curated_macro_observation(
    *, series_id: str, canonical_series_name: str, target_macro_instrument_id: str, observation_date: str, value: Decimal | None,
    is_missing: bool, normalized_unit: str, native_unit: str, native_frequency: str, realtime_start: str | None, realtime_end: str | None,
    availability_time: datetime, availability_policy_id: str, request_manifest_id: str, response_manifest_id: str, source_manifest_id: str,
    source_row_index: int,
) -> CuratedMacroObservation:
    vintage_identity = f"{series_id}:{observation_date}:{realtime_start or 'na'}"
    provisional = CuratedMacroObservation(
        observation_id="0" * 64, series_id=series_id, canonical_series_name=canonical_series_name, target_macro_instrument_id=target_macro_instrument_id,
        observation_date=observation_date, value=value, is_missing=is_missing, normalized_unit=normalized_unit, native_unit=native_unit,
        native_frequency=native_frequency, realtime_start=realtime_start, realtime_end=realtime_end, vintage_identity=vintage_identity,
        availability_time=availability_time, availability_policy_id=availability_policy_id, request_manifest_id=request_manifest_id,
        response_manifest_id=response_manifest_id, source_manifest_id=source_manifest_id, source_row_index=source_row_index,
    )
    observation_id = compute_content_id(CURATED_MACRO_OBSERVATION_KIND, provisional.to_identity_payload())
    return CuratedMacroObservation(
        observation_id=observation_id, series_id=series_id, canonical_series_name=canonical_series_name, target_macro_instrument_id=target_macro_instrument_id,
        observation_date=observation_date, value=value, is_missing=is_missing, normalized_unit=normalized_unit, native_unit=native_unit,
        native_frequency=native_frequency, realtime_start=realtime_start, realtime_end=realtime_end, vintage_identity=vintage_identity,
        availability_time=availability_time, availability_policy_id=availability_policy_id, request_manifest_id=request_manifest_id,
        response_manifest_id=response_manifest_id, source_manifest_id=source_manifest_id, source_row_index=source_row_index,
    )


@contextmanager
def _observation_lock(lock_path: Path) -> Iterator[None]:
    try:
        with experiment_lock(lock_path):
            yield
    except ExperimentLockError as exc:
        raise MarketDataLockError(f"Could not acquire curated observation store lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
    except OSError as exc:
        raise MarketDataLockError(f"Curated observation store lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc


class CuratedObservationStore:
    """Storage layout: `{storage_root}/collectors/curated/observations/
    {provider}/{series_id}/observations.jsonl` -- append-only, PURELY
    content-addressed (no sequence numbers): since `observation_id` is
    a hash of every field, two calls that would produce identical
    content always produce the identical id, so "append" is naturally
    idempotent by simple id-membership, never a sequence-gap concern
    the way `macro.MacroEventStore` (built for a single canonical
    per-series ordering) has to guard against."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _dir(self, provider: str, series_id: str) -> Path:
        return self._root / "collectors" / "curated" / "observations" / provider / series_id

    def _path(self, provider: str, series_id: str) -> Path:
        return self._dir(provider, series_id) / "observations.jsonl"

    def _lock_path(self, provider: str, series_id: str) -> Path:
        return self._dir(provider, series_id) / ".observations.lock"

    def read_observations(self, provider: str, series_id: str) -> list[CuratedMacroObservation]:
        path = self._path(provider, series_id)
        if not path.is_file():
            return []
        observations: list[CuratedMacroObservation] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted curated observation ledger line for {provider}/{series_id}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted curated observation ledger line for {provider}/{series_id}: expected a JSON object")
            observations.append(CuratedMacroObservation.from_json_dict(raw))
        return observations

    def append(self, provider: str, observation: CuratedMacroObservation) -> CuratedMacroObservation:
        self._dir(provider, observation.series_id).mkdir(parents=True, exist_ok=True)
        with _observation_lock(self._lock_path(provider, observation.series_id)):
            existing_ids = {o.observation_id for o in self.read_observations(provider, observation.series_id)}
            if observation.observation_id in existing_ids:
                return observation
            path = self._path(provider, observation.series_id)
            with path.open("ab") as handle:
                handle.write(canonical_json_bytes(observation.to_json_dict()))
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        return observation

    def append_many_and_read_all(
        self, provider: str, series_id: str, observations: Iterable[CuratedMacroObservation],
    ) -> list[CuratedMacroObservation]:
        """Appends every observation (idempotent per id, exactly like
        `append`) AND returns the resulting up-to-date full list for the
        series, all under ONE lock acquisition. A caller that needs a
        CONSISTENT snapshot immediately after writing (e.g. to build a
        `ComponentDatasetManifest`) must use this rather than a separate
        `append()` loop followed by a separate, UNLOCKED
        `read_observations()` call: under concurrent writers to the same
        series, that two-call pattern has a real race window between the
        write completing and the read observing it -- confirmed via
        Milestone 10 Phase 4B concurrency testing, where two threads
        racing the same operation occasionally computed a
        `component_manifest_id` from a stale/incomplete read, producing
        DIFFERENT `dataset_id`s (hence different `provenance_id`s) for
        the identical underlying observation."""
        self._dir(provider, series_id).mkdir(parents=True, exist_ok=True)
        with _observation_lock(self._lock_path(provider, series_id)):
            existing_ids = {o.observation_id for o in self.read_observations(provider, series_id)}
            path = self._path(provider, series_id)
            for observation in observations:
                if observation.observation_id in existing_ids:
                    continue
                with path.open("ab") as handle:
                    handle.write(canonical_json_bytes(observation.to_json_dict()))
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                existing_ids.add(observation.observation_id)
            return self.read_observations(provider, series_id)
