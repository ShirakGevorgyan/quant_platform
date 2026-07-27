"""Content-addressed, immutable research dataset storage and manifests --
the Milestone 3 analogue of `historical.canonical_store`/`historical.manifest`,
reusing the identical architectural pattern (proven in the Milestone 2
release-readiness audit) for the identical reason: exact, byte-for-byte
reconstruction of a dataset build months after the underlying historical
data has since been revised.

Layout:

    <root>/research_datasets/dataset_id=<id>/
        content/<sha256>/
            <split_name>.parquet   -- one file per named split (train/
                                       validation/test, or fold_k_train/
                                       fold_k_test/...), feature columns
                                       plus `open_time` and the label column
            preprocessing.json     -- fitted TransformPipeline state
            metadata.json          -- row counts, per-file checksums
            _SUCCESS
        manifests/<version>.json
        CURRENT                   -- one line: current content id
        _LATEST                   -- one line: current manifest version

`dataset_id` is deliberately NOT stable across a feature/label/split/
preprocessing configuration change (unlike `historical.manifest`'s
`dataset_id`, which IS stable across revisions of the same raw data) --
here it is `sha256(symbol|base_timeframe|feature_registry_fingerprint|
label_definition|split_definition|preprocessing_definition)`, so two
different research configurations are different datasets outright, never
different "versions" of the same one. `version` (within one `dataset_id`)
still follows the M2 convention: `{monotonic sequence}-{content checksum
prefix}`, never just a date, with an exact content-duplicate save being a
no-op (see `ResearchManifestStore.save`) -- this is what makes "rebuild the
same manifest, get the same outputs" a structurally-enforced no-new-version
operation rather than an accident of timing.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from quant_platform.core.exceptions import PathSecurityError, ResearchDatasetError
from quant_platform.core.json import canonical_json_bytes, parse_json_strict
from quant_platform.core.types import Timeframe
from quant_platform.data.interfaces import DataSource
from quant_platform.historical.models import sha256_file


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ResearchDatasetError(f"Expected a JSON object, got {type(value).__name__}")
    return value


def _as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ResearchDatasetError(f"Expected a JSON array, got {type(value).__name__}")
    return value

logger = logging.getLogger(__name__)

_SUCCESS_MARKER = "_SUCCESS"
_METADATA_FILE = "metadata.json"
_PREPROCESSING_FILE = "preprocessing.json"
_CONTENT_DIR_NAME = "content"
_CURRENT_POINTER_FILE = "CURRENT"
_LATEST_POINTER_FILE = "_LATEST"


def capture_environment_metadata() -> dict[str, object]:
    """Environment/package metadata recorded in every manifest -- enough to
    tell, months later, whether a reproduction attempt is running under a
    materially different Python/pandas/numpy/pyarrow than the original
    build (informational only; this platform does not refuse to load a
    manifest built under a different environment)."""
    packages: dict[str, str | None] = {}
    for name in ("pandas", "numpy", "pyarrow", "pydantic"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - defensive
            packages[name] = None
    return {"python_version": sys.version, "platform": platform.platform(), "packages": packages}


@dataclass(frozen=True, slots=True)
class ResearchDatasetManifest:
    dataset_id: str
    version: str
    source_historical_dataset_id: str
    source_historical_manifest_version: str
    symbol: str
    base_timeframe: Timeframe
    utc_start: pd.Timestamp
    utc_end: pd.Timestamp
    feature_names: tuple[str, ...]
    feature_versions: dict[str, str]
    feature_registry_fingerprint: str
    label_definition: dict[str, object]
    split_definition: dict[str, object]
    preprocessing_definition: dict[str, str]
    fitted_preprocessing_fingerprint: str | None
    code_revision: str
    input_content_hashes: dict[str, str]
    output_content_hashes: dict[str, str]
    row_counts: dict[str, int]
    missing_data_summary: dict[str, object]
    leakage_validation_result: dict[str, object]
    created_at: pd.Timestamp
    environment: dict[str, object] = field(default_factory=capture_environment_metadata)
    content_id: str = ""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "source_historical_dataset_id": self.source_historical_dataset_id,
            "source_historical_manifest_version": self.source_historical_manifest_version,
            "symbol": self.symbol,
            "base_timeframe": self.base_timeframe.value,
            "utc_start": self.utc_start.isoformat(),
            "utc_end": self.utc_end.isoformat(),
            "feature_names": list(self.feature_names),
            "feature_versions": self.feature_versions,
            "feature_registry_fingerprint": self.feature_registry_fingerprint,
            "label_definition": self.label_definition,
            "split_definition": self.split_definition,
            "preprocessing_definition": self.preprocessing_definition,
            "fitted_preprocessing_fingerprint": self.fitted_preprocessing_fingerprint,
            "code_revision": self.code_revision,
            "input_content_hashes": self.input_content_hashes,
            "output_content_hashes": self.output_content_hashes,
            "row_counts": self.row_counts,
            "missing_data_summary": self.missing_data_summary,
            "leakage_validation_result": self.leakage_validation_result,
            "created_at": self.created_at.isoformat(),
            "environment": self.environment,
            "content_id": self.content_id,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ResearchDatasetManifest:
        return cls(
            dataset_id=str(raw["dataset_id"]), version=str(raw["version"]),
            source_historical_dataset_id=str(raw["source_historical_dataset_id"]),
            source_historical_manifest_version=str(raw["source_historical_manifest_version"]),
            symbol=str(raw["symbol"]), base_timeframe=Timeframe(raw["base_timeframe"]),
            utc_start=pd.Timestamp(str(raw["utc_start"])), utc_end=pd.Timestamp(str(raw["utc_end"])),
            feature_names=tuple(str(s) for s in _as_list(raw["feature_names"])),
            feature_versions={str(k): str(v) for k, v in _as_dict(raw["feature_versions"]).items()},
            feature_registry_fingerprint=str(raw["feature_registry_fingerprint"]),
            label_definition=_as_dict(raw["label_definition"]),
            split_definition=_as_dict(raw["split_definition"]),
            preprocessing_definition={str(k): str(v) for k, v in _as_dict(raw["preprocessing_definition"]).items()},
            fitted_preprocessing_fingerprint=(
                None if raw.get("fitted_preprocessing_fingerprint") is None
                else str(raw["fitted_preprocessing_fingerprint"])
            ),
            code_revision=str(raw["code_revision"]),
            input_content_hashes={str(k): str(v) for k, v in _as_dict(raw["input_content_hashes"]).items()},
            output_content_hashes={str(k): str(v) for k, v in _as_dict(raw["output_content_hashes"]).items()},
            row_counts={str(k): int(str(v)) for k, v in _as_dict(raw["row_counts"]).items()},
            missing_data_summary=_as_dict(raw["missing_data_summary"]),
            leakage_validation_result=_as_dict(raw["leakage_validation_result"]),
            created_at=pd.Timestamp(str(raw["created_at"])),
            environment=_as_dict(raw.get("environment") or {}),
            content_id=str(raw.get("content_id", "")),
        )

    def _identity_fields(self) -> dict[str, object]:
        payload = self.to_json_dict()
        del payload["version"]
        del payload["created_at"]
        del payload["environment"]
        return payload


class ResearchDatasetStore:
    """Content-addressed storage for a research dataset's split artifacts
    (Parquet files) plus fitted preprocessing state -- mirrors
    `historical.canonical_store.CanonicalStore`'s write-once, never-
    overwritten content directories."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _dataset_dir(self, dataset_id: str) -> Path:
        safe_id = DataSource.sanitize_identifier(dataset_id, field_name="dataset_id")
        path = self._root / "research_datasets" / f"dataset_id={safe_id}"
        self._assert_within_root(path)
        return path

    def content_dir(self, dataset_id: str, content_id: str) -> Path:
        safe_content_id = DataSource.sanitize_identifier(content_id, field_name="content_id")
        path = self._dataset_dir(dataset_id) / _CONTENT_DIR_NAME / safe_content_id
        self._assert_within_root(path)
        return path

    def _current_pointer_path(self, dataset_id: str) -> Path:
        return self._dataset_dir(dataset_id) / _CURRENT_POINTER_FILE

    def _assert_within_root(self, path: Path) -> None:
        resolved = path.resolve()
        if self._root not in resolved.parents and resolved != self._root:
            raise PathSecurityError(
                f"Resolved path {resolved} escapes storage root {self._root}", context={"path": str(path)}
            )

    def write_artifacts(
        self,
        dataset_id: str,
        *,
        splits: dict[str, pd.DataFrame],
        preprocessing_json: Mapping[str, object],
        extra_metadata: dict[str, object] | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Write every named split DataFrame as its own Parquet file, plus
        fitted preprocessing state, into one new content-addressed
        directory. Returns `(content_id, per_split_checksums)`. If the
        EXACT same content already exists (byte-identical rebuild), nothing
        new is written (content-addressed dedup, per `CanonicalStore.
        write_partition`)."""
        dataset_dir = self._dataset_dir(dataset_id)
        tmp_parent = dataset_dir / ".tmp"
        tmp_parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_parent / uuid.uuid4().hex
        tmp_dir.mkdir(parents=False, exist_ok=False)
        try:
            per_split_checksums: dict[str, str] = {}
            for split_name in sorted(splits):
                split_path = tmp_dir / f"{split_name}.parquet"
                splits[split_name].to_parquet(split_path, index=False, compression="zstd")
                per_split_checksums[split_name] = sha256_file(split_path)

            # NOT migrated to `canonical_json_bytes`: `preprocessing_checksum`
            # below hashes this file's own serialized TEXT (not a re-
            # derivation of the logical payload), and directly feeds
            # `_combined_checksum` -> `content_id` -- this dataset's
            # permanent, content-addressed identity. Changing the byte
            # representation here (compact vs. indented separators) would
            # change `content_id` for every future write of otherwise-
            # identical content, breaking "content IDs remain unchanged"
            # for no security benefit (`allow_nan=False` alone closes the
            # actual NaN/Infinity gap without touching a single byte of
            # any payload that was already finite).
            preprocessing_path = tmp_dir / _PREPROCESSING_FILE
            preprocessing_path.write_text(json.dumps(preprocessing_json, indent=2, sort_keys=True, allow_nan=False))
            preprocessing_checksum = _sha256_text(preprocessing_path.read_text())

            metadata = {
                "split_names": sorted(splits),
                "per_split_checksums": per_split_checksums,
                "preprocessing_checksum": preprocessing_checksum,
                "row_counts": {name: len(df) for name, df in splits.items()},
                **(extra_metadata or {}),
            }
            # SAFE to migrate: unlike preprocessing.json above, nothing
            # hashes metadata.json's own bytes -- content_id is already
            # fixed by the time this is written.
            (tmp_dir / _METADATA_FILE).write_bytes(canonical_json_bytes(metadata))

            content_id = _combined_checksum(per_split_checksums, preprocessing_checksum)
            (tmp_dir / _SUCCESS_MARKER).write_text("")
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        final_content_dir = self.content_dir(dataset_id, content_id)
        if (final_content_dir / _SUCCESS_MARKER).is_file():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("Research dataset content already stored (dedup): dataset_id=%s content_id=%s", dataset_id, content_id[:12])
        else:
            final_content_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                Path(tmp_dir).replace(final_content_dir)
            except OSError:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                if not (final_content_dir / _SUCCESS_MARKER).is_file():
                    raise

        self._set_current(dataset_id, content_id)
        return content_id, per_split_checksums

    def _set_current(self, dataset_id: str, content_id: str) -> None:
        pointer_path = self._current_pointer_path(dataset_id)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = pointer_path.parent / f".{_CURRENT_POINTER_FILE}.{uuid.uuid4().hex}.tmp"
        tmp_path.write_text(content_id)
        tmp_path.replace(pointer_path)

    def current_content_id(self, dataset_id: str) -> str | None:
        pointer_path = self._current_pointer_path(dataset_id)
        if not pointer_path.is_file():
            return None
        return pointer_path.read_text().strip()

    def read_artifacts(self, dataset_id: str, content_id: str) -> dict[str, pd.DataFrame] | None:
        """Read every split Parquet file from ONE specific, immutable
        content directory -- bypassing `CURRENT` entirely, exactly like
        `CanonicalStore.read_partition_by_content_id`. Returns `None` if
        that content directory does not exist."""
        content_dir = self.content_dir(dataset_id, content_id)
        if not content_dir.is_dir():
            return None
        if not (content_dir / _SUCCESS_MARKER).is_file():
            raise ResearchDatasetError(
                "Research dataset content is incomplete or corrupted: missing _SUCCESS marker",
                context={"path": str(content_dir)},
            )
        try:
            metadata = parse_json_strict((content_dir / _METADATA_FILE).read_text(encoding="utf-8"))
            split_names = metadata["split_names"]
            per_split_checksums = metadata["per_split_checksums"]
        except (UnicodeDecodeError, KeyError, ValueError, TypeError) as exc:
            raise ResearchDatasetError(
                f"Research dataset metadata.json is corrupted: {exc}", context={"path": str(content_dir)}
            ) from exc
        splits: dict[str, pd.DataFrame] = {}
        for split_name in split_names:
            split_path = content_dir / f"{split_name}.parquet"
            actual_checksum = sha256_file(split_path)
            try:
                expected_checksum = per_split_checksums[split_name]
            except (KeyError, TypeError) as exc:
                raise ResearchDatasetError(
                    f"Research dataset metadata.json has no per_split_checksums entry for split {split_name!r}",
                    context={"path": str(content_dir)},
                ) from exc
            if actual_checksum != expected_checksum:
                raise ResearchDatasetError(
                    f"Research dataset split {split_name!r} checksum mismatch: data is corrupted",
                    context={"path": str(split_path), "expected": expected_checksum, "actual": actual_checksum},
                )
            splits[split_name] = pd.read_parquet(split_path)
        return splits

    def read_preprocessing(self, dataset_id: str, content_id: str) -> dict[str, object]:
        content_dir = self.content_dir(dataset_id, content_id)
        preprocessing_path = content_dir / _PREPROCESSING_FILE
        if not preprocessing_path.is_file():
            raise ResearchDatasetError(
                "Research dataset preprocessing.json is missing", context={"path": str(content_dir)}
            )
        try:
            decoded = parse_json_strict(preprocessing_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ResearchDatasetError(
                f"Research dataset preprocessing.json is corrupted: {exc}", context={"path": str(content_dir)}
            ) from exc
        if not isinstance(decoded, dict):
            raise ResearchDatasetError(
                f"Research dataset preprocessing.json must decode to a JSON object, got {type(decoded).__name__}",
                context={"path": str(content_dir)},
            )
        return decoded


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _combined_checksum(per_split_checksums: dict[str, str], preprocessing_checksum: str) -> str:
    """NOT migrated to `canonical_json_bytes`: this directly computes
    `content_id`, this dataset's permanent, content-addressed identity --
    see `write_artifacts`' identical note on `preprocessing.json`.
    `allow_nan=False` costs nothing here (every value passed in is
    already a hex checksum string, never a float) but keeps the write
    path uniformly NaN-safe regardless of future callers."""
    payload = json.dumps(
        {"splits": per_split_checksums, "preprocessing": preprocessing_checksum}, sort_keys=True, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResearchManifestStore:
    """Manifest persistence for research datasets -- mirrors
    `historical.manifest.ManifestStore` exactly (monotonic-sequence +
    content-checksum-prefix versioning, immutable once saved, content-
    duplicate save is a no-op)."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    def _manifest_dir(self, dataset_id: str) -> Path:
        safe_id = DataSource.sanitize_identifier(dataset_id, field_name="dataset_id")
        return self._root / "research_datasets" / f"dataset_id={safe_id}" / "manifests"

    def list_versions(self, dataset_id: str) -> list[str]:
        manifest_dir = self._manifest_dir(dataset_id)
        if not manifest_dir.is_dir():
            return []
        return sorted(p.stem for p in manifest_dir.glob("*.json") if p.stem != _LATEST_POINTER_FILE)

    def save(self, manifest: ResearchDatasetManifest) -> str:
        manifest_dir = self._manifest_dir(manifest.dataset_id)
        manifest_dir.mkdir(parents=True, exist_ok=True)

        existing_versions = self.list_versions(manifest.dataset_id)
        if existing_versions:
            latest_version = existing_versions[-1]
            latest_manifest = self._load_version(manifest.dataset_id, latest_version)
            if latest_manifest._identity_fields() == manifest._identity_fields():
                logger.info(
                    "ResearchManifestStore.save: identical content to latest version %s; no new version written",
                    latest_version,
                )
                return latest_version
            next_seq = int(latest_version.split("-", 1)[0]) + 1
        else:
            next_seq = 1

        content_prefix = manifest.content_id[:12] if manifest.content_id else "nocontent"
        version = f"{next_seq:06d}-{content_prefix}"
        versioned_manifest = replace(manifest, version=version)
        self._write_version_file(manifest_dir, version, versioned_manifest)
        self._write_latest_pointer(manifest_dir, version)
        logger.info(
            "Research dataset manifest saved: dataset_id=%s version=%s symbol=%s",
            manifest.dataset_id, version, manifest.symbol,
        )
        return version

    def _write_version_file(self, manifest_dir: Path, version: str, manifest: ResearchDatasetManifest) -> None:
        path = manifest_dir / f"{version}.json"
        if path.exists():
            raise ResearchDatasetError(
                f"Research dataset manifest version {version!r} already exists -- manifests are "
                "immutable once saved", context={"path": str(path)},
            )
        tmp_path = manifest_dir / f".{version}.json.tmp"
        tmp_path.write_bytes(canonical_json_bytes(manifest.to_json_dict()))
        tmp_path.replace(path)

    def _write_latest_pointer(self, manifest_dir: Path, version: str) -> None:
        tmp_path = manifest_dir / f".{_LATEST_POINTER_FILE}.tmp"
        tmp_path.write_text(version)
        tmp_path.replace(manifest_dir / _LATEST_POINTER_FILE)

    def _load_version(self, dataset_id: str, version: str) -> ResearchDatasetManifest:
        manifest_dir = self._manifest_dir(dataset_id)
        path = manifest_dir / f"{version}.json"
        if not path.is_file():
            raise ResearchDatasetError(f"Research dataset manifest version {version!r} not found", context={"path": str(path)})
        try:
            raw = parse_json_strict(path.read_text(encoding="utf-8"))
            manifest = ResearchDatasetManifest.from_json_dict(raw)
        except (UnicodeDecodeError, KeyError, ValueError, TypeError) as exc:
            raise ResearchDatasetError(f"Research dataset manifest {version!r} is corrupted: {exc}") from exc
        return manifest

    def load(self, dataset_id: str, version: str | None = None) -> ResearchDatasetManifest:
        manifest_dir = self._manifest_dir(dataset_id)
        if version is None:
            pointer_path = manifest_dir / _LATEST_POINTER_FILE
            if not pointer_path.is_file():
                raise ResearchDatasetError(f"No research dataset manifest versions exist yet for dataset_id={dataset_id!r}")
            version = pointer_path.read_text().strip()
        return self._load_version(dataset_id, version)


def compute_dataset_id(
    *,
    symbol: str,
    base_timeframe: Timeframe,
    feature_registry_fingerprint: str,
    label_definition: dict[str, object],
    split_definition: dict[str, object],
    preprocessing_definition: dict[str, str],
) -> str:
    """A research dataset's identity is a deterministic function of its
    FULL configuration -- symbol, timeframe, exact feature set (via the
    registry fingerprint), label definition, split definition, and
    preprocessing definition. Changing ANY of these changes the dataset
    ID outright (a different dataset, not a new version of the old one);
    only a revision of the underlying historical data, with configuration
    held fixed, produces a new VERSION of the same dataset ID.

    NOT migrated to `canonical_json_bytes`: this directly computes
    `dataset_id`, a permanent identity used as a storage path component
    everywhere in this package -- see `write_artifacts`' identical note
    on `preprocessing.json`. `allow_nan=False` added below (closes the
    actual NaN/Infinity gap) without touching `default=str`/key ordering/
    separators, so bytes for any already-finite payload -- every
    legitimate caller today -- are completely unchanged."""
    payload = json.dumps(
        {
            "symbol": symbol, "base_timeframe": base_timeframe.value,
            "feature_registry_fingerprint": feature_registry_fingerprint,
            "label_definition": label_definition, "split_definition": split_definition,
            "preprocessing_definition": preprocessing_definition,
        },
        sort_keys=True, default=str, allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "ResearchDatasetManifest",
    "ResearchDatasetStore",
    "ResearchManifestStore",
    "capture_environment_metadata",
    "compute_dataset_id",
]
