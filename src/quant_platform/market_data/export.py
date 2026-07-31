"""Deterministic JSON Lines export (Milestone 10, Phase 2). JSONL only --
not CSV: every value this package exports (`Decimal` prices/values,
ISO-8601 UTC timestamps, enum-valued fields) already round-trips exactly
through this repository's own `to_json_dict()` convention, so JSONL reuses
existing, already-tested serialization rather than inventing a second,
CSV-specific quoting/escaping scheme -- and avoids adding a pandas
dependency purely for export, per the specification's own guidance.

Row order is always `(timestamp, id)` -- the same canonical logical
ordering `FeatureStore.read_records`/replay already use -- independent of
physical append order, so export is deterministic regardless of how many
ingestion batches (in what order) produced the underlying data. Each
row's OWN field order is whatever `to_json_dict()` declares (a fixed,
already-deterministic Python dict literal, unchanged across calls);
export writes it WITHOUT re-sorting keys (unlike `canonical_json_bytes`,
which sorts for content-hashing purposes) so the exported file's column
order is stable and human-inspectable, while `allow_nan=False` still
rejects any non-finite float from ever reaching the file.

`export_semantic_digest` is a digest of the EXPORTED ROW SET alone
(order-independent -- sorted before hashing) -- a caller can confirm two
exports (different filesystem roots, different processes) produced
identical economic content without diffing raw bytes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from quant_platform.core.exceptions import ExportError
from quant_platform.market_data.identity import compute_content_id
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.repository import MarketDataRepository

__all__ = ["ExportResult", "export_feature_dataset_jsonl", "export_raw_dataset_jsonl"]


def _write_jsonl_line(handle: TextIO, row: dict[str, object]) -> None:
    try:
        text = json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except ValueError as exc:
        raise ExportError(f"row contains a non-finite value and cannot be exported: {exc}") from exc
    handle.write(text)
    handle.write("\n")


@dataclass(frozen=True, slots=True)
class ExportResult:
    dataset_key: DatasetKey
    destination: Path
    row_count: int
    export_semantic_digest: str


def export_raw_dataset_jsonl(*, repository: MarketDataRepository, dataset_key: DatasetKey, destination: Path) -> ExportResult:
    if dataset_key.dataset_kind is not DatasetKind.RAW_MARKET_EVENTS:
        raise ExportError("export_raw_dataset_jsonl requires a RAW_MARKET_EVENTS dataset_key")
    assert dataset_key.provider is not None
    events = repository.event_store.read_events(dataset_key.provider, dataset_key.instrument_id)
    rows = [e.to_json_dict() for e in sorted(events, key=lambda e: (e.to_json_dict()["event_time"], e.to_json_dict()["event_id"]))]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            _write_jsonl_line(handle, row)

    digest = compute_content_id("raw_dataset_export_digest", {"rows": sorted(rows, key=lambda r: (str(r["event_time"]), str(r["event_id"])))})
    return ExportResult(dataset_key=dataset_key, destination=destination, row_count=len(rows), export_semantic_digest=digest)


def export_feature_dataset_jsonl(*, repository: MarketDataRepository, dataset_key: DatasetKey, destination: Path) -> ExportResult:
    if dataset_key.dataset_kind is not DatasetKind.DERIVED_FEATURES:
        raise ExportError("export_feature_dataset_jsonl requires a DERIVED_FEATURES dataset_key")
    assert dataset_key.feature_name is not None and dataset_key.feature_version is not None
    records = repository.feature_store.read_records(dataset_key.feature_name, dataset_key.feature_version, dataset_key.instrument_id)
    rows = [r.to_json_dict() for r in records]  # already sorted by (timestamp, feature_id) per FeatureStore.read_records

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            _write_jsonl_line(handle, row)

    digest = compute_content_id("feature_dataset_export_digest", {"rows": sorted(rows, key=lambda r: (str(r["timestamp"]), str(r["feature_id"])))})
    return ExportResult(dataset_key=dataset_key, destination=destination, row_count=len(rows), export_semantic_digest=digest)
