"""Dataset manifests: the single machine-readable AND human-inspectable
record of exactly what a canonical (or derived) dataset is, how it was
built, and how to verify it hasn't drifted from what it claims to be.

A manifest's `version` is NEVER just a date -- Section L of the Milestone 2
spec explicitly calls this out, because a date alone cannot distinguish two
different builds produced on the same day, and encourages exactly the kind
of "just check when it was made" reasoning that hides silent content
drift. Versions here are `{monotonic sequence}-{content checksum prefix}`:
monotonic so "newer than" is a trivial string/integer comparison, content-
derived so two versions can never collide unless they are byte-for-byte
the same dataset.

Manifests are immutable once saved (same philosophy as
`historical.raw_store`'s snapshots): `ManifestStore.save` never overwrites
an existing version file. Saving an EXACT content-duplicate of the current
latest manifest is a no-op that returns the existing version rather than
minting a redundant new one -- this is what makes an incremental update
that legitimately changed nothing produce no new manifest version (see
`historical.update_pipeline`), a concrete form of the idempotency Section K
requires.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from quant_platform.core.exceptions import ManifestError
from quant_platform.core.types import Timeframe
from quant_platform.data.interfaces import DataSource

logger = logging.getLogger(__name__)

_LATEST_POINTER_FILE = "_LATEST"


def _as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ManifestError(f"Expected a JSON array, got {type(value).__name__}")
    return value


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ManifestError(f"Expected a JSON object, got {type(value).__name__}")
    return value


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    version: str
    parent_snapshot_ids: tuple[str, ...]
    symbol: str
    source_name: str
    broker: str
    timeframe: Timeframe
    utc_start: pd.Timestamp
    utc_end: pd.Timestamp
    row_count: int
    schema_fingerprint: str
    content_checksum: str
    created_at: pd.Timestamp
    pipeline_version: str
    normalization_settings: dict[str, object]
    validation_policy: str
    quality_summary: dict[str, object]
    repair_summary: dict[str, object]
    calendar_version: str | None = None
    resampling_config: dict[str, object] | None = None
    code_revision: str | None = None
    reproducibility_seed: int | None = None
    partition_content_ids: dict[int, str] = field(default_factory=dict)
    """Exactly which immutable, content-addressed
    `CanonicalStore` partition blob (see `content_dir`/`content_id`) was
    current for each UTC year at the moment this manifest version was
    saved. This is what makes a dataset version exactly reconstructable
    later, even after `CURRENT` has since moved on to a newer revision --
    see `historical.canonical_store`'s module docstring and
    `DatasetLoader.load` for how this is used. Never inferred or
    recomputed after the fact; always the literal content ids the update
    that produced this manifest actually wrote/observed."""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "parent_snapshot_ids": list(self.parent_snapshot_ids),
            "symbol": self.symbol,
            "source_name": self.source_name,
            "broker": self.broker,
            "timeframe": self.timeframe.value,
            "utc_start": self.utc_start.isoformat(),
            "utc_end": self.utc_end.isoformat(),
            "row_count": self.row_count,
            "schema_fingerprint": self.schema_fingerprint,
            "content_checksum": self.content_checksum,
            "created_at": self.created_at.isoformat(),
            "pipeline_version": self.pipeline_version,
            "normalization_settings": self.normalization_settings,
            "validation_policy": self.validation_policy,
            "quality_summary": self.quality_summary,
            "repair_summary": self.repair_summary,
            "calendar_version": self.calendar_version,
            "resampling_config": self.resampling_config,
            "code_revision": self.code_revision,
            "reproducibility_seed": self.reproducibility_seed,
            "partition_content_ids": {str(year): content_id for year, content_id in self.partition_content_ids.items()},
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> DatasetManifest:
        return cls(
            dataset_id=str(raw["dataset_id"]),
            version=str(raw["version"]),
            parent_snapshot_ids=tuple(str(s) for s in _as_list(raw["parent_snapshot_ids"])),
            symbol=str(raw["symbol"]),
            source_name=str(raw["source_name"]),
            broker=str(raw["broker"]),
            timeframe=Timeframe(raw["timeframe"]),
            utc_start=pd.Timestamp(str(raw["utc_start"])),
            utc_end=pd.Timestamp(str(raw["utc_end"])),
            row_count=int(str(raw["row_count"])),
            schema_fingerprint=str(raw["schema_fingerprint"]),
            content_checksum=str(raw["content_checksum"]),
            created_at=pd.Timestamp(str(raw["created_at"])),
            pipeline_version=str(raw["pipeline_version"]),
            normalization_settings=_as_dict(raw["normalization_settings"]),
            validation_policy=str(raw["validation_policy"]),
            quality_summary=_as_dict(raw["quality_summary"]),
            repair_summary=_as_dict(raw["repair_summary"]),
            calendar_version=(None if raw.get("calendar_version") is None else str(raw["calendar_version"])),
            resampling_config=(
                None if raw.get("resampling_config") is None else _as_dict(raw["resampling_config"])
            ),
            code_revision=(None if raw.get("code_revision") is None else str(raw["code_revision"])),
            reproducibility_seed=(
                None if raw.get("reproducibility_seed") is None else int(str(raw["reproducibility_seed"]))
            ),
            partition_content_ids={
                int(year): str(content_id)
                for year, content_id in _as_dict(raw.get("partition_content_ids") or {}).items()
            },
        )

    def _identity_fields(self) -> dict[str, object]:
        """Everything except `version`/`created_at` -- used to detect a
        content-duplicate save (see `ManifestStore.save`)."""
        payload = self.to_json_dict()
        del payload["version"]
        del payload["created_at"]
        return payload


class ManifestStore:
    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def compute_dataset_id(self, *, symbol: str, timeframe: Timeframe, source_name: str, broker: str) -> str:
        """A dataset's identity is stable across every version -- derived
        from the fields that define WHICH dataset this is (symbol, source,
        broker, timeframe), never from its content, so version N and
        version N+1 of the same dataset share this id."""
        signature = f"{symbol}|{timeframe.value}|{source_name}|{broker}"
        return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]

    def _dataset_dir(self, *, symbol: str, timeframe: Timeframe) -> Path:
        safe_symbol = DataSource.sanitize_identifier(symbol, field_name="symbol")
        return self._root / "manifests" / f"symbol={safe_symbol}" / f"timeframe={timeframe.value}"

    def list_versions(self, *, symbol: str, timeframe: Timeframe) -> list[str]:
        dataset_dir = self._dataset_dir(symbol=symbol, timeframe=timeframe)
        if not dataset_dir.is_dir():
            return []
        versions = [p.stem for p in dataset_dir.glob("*.json") if p.stem != _LATEST_POINTER_FILE]
        return sorted(versions)

    def save(self, manifest: DatasetManifest) -> str:
        """Persist `manifest` under a freshly computed version and update
        the `latest` pointer, UNLESS it is an exact content-duplicate of
        the current latest version, in which case nothing is written and
        the existing version string is returned."""
        dataset_dir = self._dataset_dir(symbol=manifest.symbol, timeframe=manifest.timeframe)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        existing_versions = self.list_versions(symbol=manifest.symbol, timeframe=manifest.timeframe)
        if existing_versions:
            latest_version = existing_versions[-1]
            latest_manifest = self._load_version(
                symbol=manifest.symbol, timeframe=manifest.timeframe, version=latest_version
            )
            if latest_manifest._identity_fields() == manifest._identity_fields():
                logger.info(
                    "ManifestStore.save: identical content to latest version %s; no new version written",
                    latest_version,
                )
                return latest_version
            next_seq = int(latest_version.split("-", 1)[0]) + 1
        else:
            next_seq = 1

        version = f"{next_seq:06d}-{manifest.content_checksum[:12]}"
        versioned_manifest = replace(manifest, version=version)
        self._write_version_file(dataset_dir, version, versioned_manifest)
        self._write_latest_pointer(dataset_dir, version)
        logger.info(
            "Manifest saved: dataset_id=%s symbol=%s timeframe=%s version=%s rows=%d",
            manifest.dataset_id, manifest.symbol, manifest.timeframe.value, version, manifest.row_count,
        )
        return version

    def _write_version_file(self, dataset_dir: Path, version: str, manifest: DatasetManifest) -> None:
        path = dataset_dir / f"{version}.json"
        if path.exists():
            raise ManifestError(
                f"Manifest version {version!r} already exists -- manifests are immutable once saved",
                context={"path": str(path)},
            )
        tmp_path = dataset_dir / f".{version}.json.tmp"
        tmp_path.write_text(json.dumps(manifest.to_json_dict(), indent=2))
        tmp_path.replace(path)

    def _write_latest_pointer(self, dataset_dir: Path, version: str) -> None:
        tmp_path = dataset_dir / f".{_LATEST_POINTER_FILE}.tmp"
        tmp_path.write_text(version)
        tmp_path.replace(dataset_dir / _LATEST_POINTER_FILE)

    def _load_version(self, *, symbol: str, timeframe: Timeframe, version: str) -> DatasetManifest:
        dataset_dir = self._dataset_dir(symbol=symbol, timeframe=timeframe)
        path = dataset_dir / f"{version}.json"
        if not path.is_file():
            raise ManifestError(f"Manifest version {version!r} not found", context={"path": str(path)})
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ManifestError(f"Manifest {version!r} is corrupted: {exc}", context={"path": str(path)}) from exc
        return DatasetManifest.from_json_dict(raw)

    def load(self, *, symbol: str, timeframe: Timeframe, version: str | None = None) -> DatasetManifest:
        """Load a specific `version`, or the latest if `version` is None."""
        dataset_dir = self._dataset_dir(symbol=symbol, timeframe=timeframe)
        if version is None:
            pointer_path = dataset_dir / _LATEST_POINTER_FILE
            if not pointer_path.is_file():
                raise ManifestError(
                    f"No manifest versions exist yet for symbol={symbol} timeframe={timeframe.value}",
                    context={"symbol": symbol, "timeframe": timeframe.value},
                )
            version = pointer_path.read_text().strip()
        return self._load_version(symbol=symbol, timeframe=timeframe, version=version)


__all__ = ["DatasetManifest", "ManifestStore"]
