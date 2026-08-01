"""Raw and canonical market-bar record models (Milestone 10, Phase 4C).

`RawMarketRecord` represents ONE bar exactly as a provider reported it,
before canonical normalization -- every financial field stays TEXT
until `market_normalization.py` parses it directly to `Decimal` (never
through `float`). `MarketDriverBar` is the canonical, immutable,
content-addressed event a verified raw record becomes -- deliberately a
NEW, materially richer record type, NOT Phase 1's sequence-based
`candles.Candle`: a market-driver bar carries canonical-driver/
instrument-form/proxy/adjustment/futures-contract/continuation/
availability semantics `Candle`'s envelope was never shaped to hold (the
exact same "build a new record type rather than force-fit" precedent
Phase 4B already established for `CuratedMacroObservation` vs. `macro.
MacroEvent`). `MarketDriverBarStore` is PURELY content-addressed
append-only, mirroring `curated.macro_observation.CuratedObservationStore`
exactly, including its `append_many_and_read_all` atomic-read-after-
write hardening."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from quant_platform.core.exceptions import (
    ExperimentLockError,
    MarketDataLockError,
    MarketDataPersistenceError,
    MarketRecordError,
)
from quant_platform.core.json import canonical_json_bytes, parse_json_strict
from quant_platform.core.time_utils import compute_close_time
from quant_platform.core.types import Timeframe
from quant_platform.market_data.collectors.cross_asset.futures import RollProvenance
from quant_platform.market_data.collectors.cross_asset.instrument_form import InstrumentForm
from quant_platform.market_data.identity import (
    compute_content_id,
    deserialize_timestamp,
    parse_decimal,
    require_non_empty,
    require_tz_aware,
    serialize_timestamp,
)
from quant_platform.ml.concurrency import experiment_lock

__all__ = [
    "MARKET_DRIVER_BAR_KIND",
    "MarketDriverBar",
    "MarketDriverBarStore",
    "RawMarketRecord",
    "create_market_driver_bar",
]

MARKET_DRIVER_BAR_KIND = "cross_asset_market_driver_bar"


@dataclass(frozen=True, slots=True)
class RawMarketRecord:
    """Provider text, unparsed -- see module docstring. Not persisted on
    its own; a durable record is always the ALREADY-CACHED raw response
    bytes (`collectors.cache.RawResponseCache`) plus this in-memory
    parse of one row from them."""

    provider: str
    provider_symbol: str
    provider_timestamp_text: str
    interval: str
    open_text: str
    high_text: str
    low_text: str
    close_text: str
    volume_text: str | None
    adjusted_close_text: str | None
    trade_count_text: str | None
    source_sequence: int
    contract_symbol: str | None

    def __post_init__(self) -> None:
        require_non_empty(self.provider, field_name="RawMarketRecord.provider")
        require_non_empty(self.provider_symbol, field_name="RawMarketRecord.provider_symbol")
        require_non_empty(self.provider_timestamp_text, field_name="RawMarketRecord.provider_timestamp_text")
        require_non_empty(self.interval, field_name="RawMarketRecord.interval")
        if self.source_sequence < 0:
            raise MarketRecordError(f"RawMarketRecord.source_sequence must be >= 0, got {self.source_sequence}")


@dataclass(frozen=True, slots=True)
class MarketDriverBar:
    bar_id: str
    canonical_driver_id: str
    provider: str
    provider_symbol: str
    instrument_form: InstrumentForm
    open_time: datetime
    timeframe: Timeframe
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal | None
    volume_unit: str
    availability_time: datetime
    availability_policy_id: str
    session_policy_id: str
    adjustment_policy_id: str
    contract_metadata_id: str | None
    roll_provenance: RollProvenance | None
    request_manifest_id: str
    response_manifest_id: str
    source_manifest_id: str
    source_row_index: int

    def __post_init__(self) -> None:
        require_non_empty(self.canonical_driver_id, field_name="MarketDriverBar.canonical_driver_id")
        require_non_empty(self.provider, field_name="MarketDriverBar.provider")
        require_non_empty(self.provider_symbol, field_name="MarketDriverBar.provider_symbol")
        require_non_empty(self.volume_unit, field_name="MarketDriverBar.volume_unit")
        require_non_empty(self.availability_policy_id, field_name="MarketDriverBar.availability_policy_id")
        require_non_empty(self.session_policy_id, field_name="MarketDriverBar.session_policy_id")
        require_non_empty(self.adjustment_policy_id, field_name="MarketDriverBar.adjustment_policy_id")
        require_non_empty(self.request_manifest_id, field_name="MarketDriverBar.request_manifest_id")
        require_non_empty(self.response_manifest_id, field_name="MarketDriverBar.response_manifest_id")
        require_non_empty(self.source_manifest_id, field_name="MarketDriverBar.source_manifest_id")
        require_tz_aware(self.open_time, field_name="MarketDriverBar.open_time")
        require_tz_aware(self.availability_time, field_name="MarketDriverBar.availability_time")
        if self.source_row_index < 0:
            raise MarketRecordError(f"MarketDriverBar.source_row_index must be >= 0, got {self.source_row_index}")
        if self.high < self.low:
            raise MarketRecordError(f"MarketDriverBar.high ({self.high}) must be >= low ({self.low})")
        if not (self.low <= self.open <= self.high):
            raise MarketRecordError(f"MarketDriverBar.open ({self.open}) must be within [low, high] ([{self.low}, {self.high}])")
        if not (self.low <= self.close <= self.high):
            raise MarketRecordError(f"MarketDriverBar.close ({self.close}) must be within [low, high] ([{self.low}, {self.high}])")
        if self.low <= 0:
            raise MarketRecordError(f"MarketDriverBar.low must be > 0, got {self.low}")
        if self.volume is not None and self.volume < 0:
            raise MarketRecordError(f"MarketDriverBar.volume must be >= 0, got {self.volume}")
        close_time = self.close_time
        if self.availability_time < close_time:
            raise MarketRecordError(f"MarketDriverBar.availability_time ({self.availability_time}) must be >= close_time ({close_time})")
        if self.instrument_form is InstrumentForm.EXCHANGE_FUTURES_CONTRACT and self.contract_metadata_id is None:
            raise MarketRecordError("MarketDriverBar with instrument_form=EXCHANGE_FUTURES_CONTRACT requires contract_metadata_id")
        if self.instrument_form is InstrumentForm.PROVIDER_CONTINUOUS_FUTURES and self.roll_provenance is None:
            raise MarketRecordError("MarketDriverBar with instrument_form=PROVIDER_CONTINUOUS_FUTURES requires roll_provenance")

    @property
    def close_time(self) -> datetime:
        return compute_close_time(self.open_time, self.timeframe)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": MARKET_DRIVER_BAR_KIND, "bar_id": self.bar_id, "canonical_driver_id": self.canonical_driver_id, "provider": self.provider,
            "provider_symbol": self.provider_symbol, "instrument_form": self.instrument_form.value,
            "open_time": serialize_timestamp(self.open_time, field_name="open_time"), "timeframe": self.timeframe.value,
            "open": str(self.open), "high": str(self.high), "low": str(self.low), "close": str(self.close),
            "volume": (None if self.volume is None else str(self.volume)), "volume_unit": self.volume_unit,
            "availability_time": serialize_timestamp(self.availability_time, field_name="availability_time"),
            "availability_policy_id": self.availability_policy_id, "session_policy_id": self.session_policy_id,
            "adjustment_policy_id": self.adjustment_policy_id, "contract_metadata_id": self.contract_metadata_id,
            "roll_provenance": (None if self.roll_provenance is None else self.roll_provenance.to_json_dict()),
            "request_manifest_id": self.request_manifest_id, "response_manifest_id": self.response_manifest_id,
            "source_manifest_id": self.source_manifest_id, "source_row_index": self.source_row_index,
        }

    def to_identity_payload(self) -> dict[str, object]:
        payload = dict(self.to_json_dict())
        del payload["bar_id"]
        return payload

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> MarketDriverBar:
        raw_volume = raw.get("volume")
        raw_roll = raw.get("roll_provenance")
        return cls(
            bar_id=str(raw["bar_id"]), canonical_driver_id=str(raw["canonical_driver_id"]), provider=str(raw["provider"]),
            provider_symbol=str(raw["provider_symbol"]), instrument_form=InstrumentForm(raw["instrument_form"]),
            open_time=deserialize_timestamp(raw["open_time"], field_name="open_time"), timeframe=Timeframe(raw["timeframe"]),
            open=parse_decimal(raw["open"], field_name="open"), high=parse_decimal(raw["high"], field_name="high"),
            low=parse_decimal(raw["low"], field_name="low"), close=parse_decimal(raw["close"], field_name="close"),
            volume=(None if raw_volume is None else parse_decimal(raw_volume, field_name="volume")), volume_unit=str(raw["volume_unit"]),
            availability_time=deserialize_timestamp(raw["availability_time"], field_name="availability_time"),
            availability_policy_id=str(raw["availability_policy_id"]), session_policy_id=str(raw["session_policy_id"]),
            adjustment_policy_id=str(raw["adjustment_policy_id"]),
            contract_metadata_id=(None if raw.get("contract_metadata_id") is None else str(raw["contract_metadata_id"])),
            roll_provenance=(None if raw_roll is None else RollProvenance.from_json_dict(raw_roll)),  # type: ignore[arg-type]
            request_manifest_id=str(raw["request_manifest_id"]), response_manifest_id=str(raw["response_manifest_id"]),
            source_manifest_id=str(raw["source_manifest_id"]), source_row_index=int(str(raw["source_row_index"])),
        )


def create_market_driver_bar(
    *, canonical_driver_id: str, provider: str, provider_symbol: str, instrument_form: InstrumentForm, open_time: datetime, timeframe: Timeframe,
    open: Decimal, high: Decimal, low: Decimal, close: Decimal, volume: Decimal | None, volume_unit: str, availability_time: datetime,
    availability_policy_id: str, session_policy_id: str, adjustment_policy_id: str, request_manifest_id: str, response_manifest_id: str,
    source_manifest_id: str, source_row_index: int, contract_metadata_id: str | None = None, roll_provenance: RollProvenance | None = None,
) -> MarketDriverBar:
    provisional = MarketDriverBar(
        bar_id="0" * 64, canonical_driver_id=canonical_driver_id, provider=provider, provider_symbol=provider_symbol,
        instrument_form=instrument_form, open_time=open_time, timeframe=timeframe, open=open, high=high, low=low, close=close, volume=volume,
        volume_unit=volume_unit, availability_time=availability_time, availability_policy_id=availability_policy_id,
        session_policy_id=session_policy_id, adjustment_policy_id=adjustment_policy_id, contract_metadata_id=contract_metadata_id,
        roll_provenance=roll_provenance, request_manifest_id=request_manifest_id, response_manifest_id=response_manifest_id,
        source_manifest_id=source_manifest_id, source_row_index=source_row_index,
    )
    bar_id = compute_content_id(MARKET_DRIVER_BAR_KIND, provisional.to_identity_payload())
    return MarketDriverBar(
        bar_id=bar_id, canonical_driver_id=canonical_driver_id, provider=provider, provider_symbol=provider_symbol,
        instrument_form=instrument_form, open_time=open_time, timeframe=timeframe, open=open, high=high, low=low, close=close, volume=volume,
        volume_unit=volume_unit, availability_time=availability_time, availability_policy_id=availability_policy_id,
        session_policy_id=session_policy_id, adjustment_policy_id=adjustment_policy_id, contract_metadata_id=contract_metadata_id,
        roll_provenance=roll_provenance, request_manifest_id=request_manifest_id, response_manifest_id=response_manifest_id,
        source_manifest_id=source_manifest_id, source_row_index=source_row_index,
    )


@contextmanager
def _bar_store_lock(lock_path: Path) -> Iterator[None]:
    try:
        with experiment_lock(lock_path):
            yield
    except ExperimentLockError as exc:
        raise MarketDataLockError(f"Could not acquire market-driver-bar store lock at {lock_path}: {exc}", context={"lock_path": str(lock_path)}) from exc
    except OSError as exc:
        raise MarketDataLockError(f"Market-driver-bar store lock at {lock_path} hit a filesystem race: {exc}", context={"lock_path": str(lock_path)}) from exc


class MarketDriverBarStore:
    """Storage layout: `{storage_root}/collectors/cross_asset/bars/
    {provider}/{canonical_driver_id}__{instrument_form}/bars.jsonl` --
    append-only, purely content-addressed, mirroring `curated.
    macro_observation.CuratedObservationStore` exactly."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _scope(self, canonical_driver_id: str, instrument_form: InstrumentForm) -> str:
        return f"{canonical_driver_id}__{instrument_form.value}"

    def _dir(self, provider: str, canonical_driver_id: str, instrument_form: InstrumentForm) -> Path:
        return self._root / "collectors" / "cross_asset" / "bars" / provider / self._scope(canonical_driver_id, instrument_form)

    def _path(self, provider: str, canonical_driver_id: str, instrument_form: InstrumentForm) -> Path:
        return self._dir(provider, canonical_driver_id, instrument_form) / "bars.jsonl"

    def _lock_path(self, provider: str, canonical_driver_id: str, instrument_form: InstrumentForm) -> Path:
        return self._dir(provider, canonical_driver_id, instrument_form) / ".bars.lock"

    def read_bars(self, provider: str, canonical_driver_id: str, instrument_form: InstrumentForm) -> list[MarketDriverBar]:
        path = self._path(provider, canonical_driver_id, instrument_form)
        if not path.is_file():
            return []
        bars: list[MarketDriverBar] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = parse_json_strict(line)
            except ValueError as exc:
                raise MarketDataPersistenceError(f"Corrupted market-driver-bar ledger line for {provider}/{canonical_driver_id}/{instrument_form.value}: {exc}") from exc
            if not isinstance(raw, dict):
                raise MarketDataPersistenceError(f"Corrupted market-driver-bar ledger line for {provider}/{canonical_driver_id}/{instrument_form.value}: expected a JSON object")
            bars.append(MarketDriverBar.from_json_dict(raw))
        return bars

    def append_many_and_read_all(
        self, provider: str, canonical_driver_id: str, instrument_form: InstrumentForm, bars: Iterable[MarketDriverBar],
    ) -> list[MarketDriverBar]:
        """Append-then-read as ONE atomic, lock-held operation -- see
        `curated.macro_observation.CuratedObservationStore.
        append_many_and_read_all`'s own docstring for why a separate
        append-loop followed by a separate unlocked read is unsafe under
        concurrent writers."""
        self._dir(provider, canonical_driver_id, instrument_form).mkdir(parents=True, exist_ok=True)
        with _bar_store_lock(self._lock_path(provider, canonical_driver_id, instrument_form)):
            existing_ids = {b.bar_id for b in self.read_bars(provider, canonical_driver_id, instrument_form)}
            path = self._path(provider, canonical_driver_id, instrument_form)
            for bar in bars:
                if bar.bar_id in existing_ids:
                    continue
                with path.open("ab") as handle:
                    handle.write(canonical_json_bytes(bar.to_json_dict()))
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                existing_ids.add(bar.bar_id)
            return self.read_bars(provider, canonical_driver_id, instrument_form)
