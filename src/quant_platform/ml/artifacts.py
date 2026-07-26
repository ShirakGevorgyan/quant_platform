"""Immutable, content-addressed generic artifact storage for the ML
subsystem -- mirrors `historical.canonical_store.CanonicalStore` and
`features.manifests.ResearchDatasetStore`'s proven write-once,
content-addressed, atomic-rename pattern, applied to arbitrary ML
artifact bytes (fitted model blobs, prediction/probability arrays,
metrics, feature importance, calibration data, reports, logs) rather
than domain-specific Parquet partitions.

Layout (a deliberate, documented variant of the milestone's suggested
layout -- see "WHY NO `CURRENT` POINTER" below):

    <ml_artifacts_root>/
        content/<hash[:2]>/<sha256>     -- raw artifact bytes, write-once
        metadata/<sha256>.json          -- {schema_version, content_hash,
                                             category, size_bytes, created_at}

`experiments/<experiment_id>/manifest.json` (also rooted at the same
`ml_artifacts_root`) is handled by `ml/manifests.py`'s
`ExperimentManifestStore`, a sibling store over the same root -- exactly
how `CanonicalStore`/`historical.manifest.ManifestStore` and
`ResearchDatasetStore`/`ResearchManifestStore` split "content" and
"manifest" concerns across two focused classes sharing one root in
Milestones 2 and 3.

WHY NO `CURRENT` POINTER
--------------------------------------------------------------------------
`CanonicalStore` and `ResearchDatasetStore` each manage ONE evolving
named entity (a `(symbol, timeframe)` partition; a `dataset_id`) that has
a meaningful "latest version" -- `CURRENT` is a convenience pointer to
that. This store holds many unrelated, independently content-addressed
blobs with no such entity to have a "latest" version of: two different
metrics artifacts from two different experiments share this store but
have no ordering relationship. Every caller that needs to find a
specific artifact again already holds its `ArtifactReference` (recorded
explicitly in an `ExperimentManifest`) -- a `CURRENT` pointer here would
have no referent and would invite exactly the "convenience pointer
mistaken for identity" confusion Section 9 warns against. This
deviation from the suggested layout is deliberate and is called out
again in the delivery report.

TRUST BOUNDARY: NO CODE EXECUTION ON READ
--------------------------------------------------------------------------
`read_artifact` returns raw `bytes` and nothing else -- there is no
pickle, no `eval`, no dynamic import anywhere in this module. A caller
that needs to reconstruct a fitted model from those bytes must go
through an explicitly-trusted `ModelDeserializer` (`ml.interfaces`),
which is this milestone's documented, isolated exception to "never
deserialize untrusted bytes into executable objects" -- never this
generic layer. JSON-structured artifacts (metrics, feature importance)
are encoded/decoded by the CALLER via `ml.persistence.canonical_json_bytes`/
`parse_json_strict`, never by this store, which only ever sees `bytes`.

CONFLICT DETECTION, NOT SILENT OVERWRITE
--------------------------------------------------------------------------
Content addressing means the SAME hash always means the SAME bytes --
but the small sidecar metadata (`category`) is supplied by the caller,
not derived from the bytes. If the same content hash is ever written
again under a DIFFERENT declared `category` than its first write, that
is treated as store corruption (`ArtifactCorruptionError`), never
silently accepted -- "no silent overwrite" applies to metadata, not
just content.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from quant_platform.core.exceptions import ArtifactCorruptionError, ArtifactNotFoundError
from quant_platform.ml.models import ArtifactCategory, ArtifactReference, validate_artifact_hash
from quant_platform.ml.persistence import (
    as_json_dict,
    assert_within_root,
    canonical_json_bytes,
    format_utc_timestamp,
    read_json_file,
    require_schema_version,
    sha256_hex_bytes,
    utc_now,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_CONTENT_DIR_NAME = "content"
_METADATA_DIR_NAME = "metadata"


@dataclass(frozen=True, slots=True)
class _ContentSidecar:
    """Internal, storage-layer-only record -- not a public domain model.
    Deliberately narrow: everything here is either derivable from the
    content bytes themselves (`content_hash`, `size_bytes`) or is the
    caller's own declaration (`category`, `created_at` of first write)."""

    schema_version: int
    content_hash: str
    category: ArtifactCategory
    size_bytes: int
    created_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "category": self.category.value,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> _ContentSidecar:
        require_schema_version(raw, supported=_SCHEMA_VERSION, context="MLArtifactStore content sidecar")
        return cls(
            schema_version=_SCHEMA_VERSION,
            content_hash=str(raw["content_hash"]),
            category=ArtifactCategory(raw["category"]),
            size_bytes=int(str(raw["size_bytes"])),
            created_at=str(raw["created_at"]),
        )


class MLArtifactStore:
    """Content-addressed store for generic ML artifact bytes. Every write
    is immutable and idempotent: writing byte-identical content (and the
    same declared `category`) twice is a safe no-op; writing the same
    content hash under a different `category` is treated as corruption."""

    def __init__(self, storage_root: Path | str) -> None:
        self._root = Path(storage_root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _content_path(self, content_hash: str) -> Path:
        path = self._root / _CONTENT_DIR_NAME / content_hash[:2] / content_hash
        self._assert_within_root(path)
        return path

    def _metadata_path(self, content_hash: str) -> Path:
        path = self._root / _METADATA_DIR_NAME / f"{content_hash}.json"
        self._assert_within_root(path)
        return path

    def _assert_within_root(self, path: Path) -> None:
        assert_within_root(path, root=self._root)

    def write_artifact(self, data: bytes, *, category: ArtifactCategory) -> ArtifactReference:
        content_hash = sha256_hex_bytes(data)
        content_path = self._content_path(content_hash)
        metadata_path = self._metadata_path(content_hash)

        if content_path.is_file():
            existing = self._read_sidecar_if_present(metadata_path, content_hash=content_hash)
            if existing is not None:
                if existing.category is not category:
                    raise ArtifactCorruptionError(
                        f"Content hash {content_hash!r} was already stored under category "
                        f"{existing.category.value!r}; refusing to silently accept a different "
                        f"declared category {category.value!r} for the same bytes",
                        context={"content_hash": content_hash, "existing_category": existing.category.value, "requested_category": category.value},
                    )
                logger.info("ML artifact already stored (dedup): content_hash=%s category=%s", content_hash[:12], category.value)
                return ArtifactReference(
                    category=category, content_hash=content_hash, size_bytes=existing.size_bytes, created_at=existing.created_at,
                )
            # Content present but sidecar missing: an interrupted prior
            # write (crash between content rename and metadata rename).
            # Content is immutable and already verified complete by virtue
            # of `content_path.is_file()` at its content-addressed path;
            # regenerating the sidecar now is safe and idempotent.
            logger.warning(
                "ML artifact content present without metadata sidecar (recovering from interrupted write): content_hash=%s",
                content_hash[:12],
            )

        content_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_content_path = content_path.parent / f".{content_path.name}.{uuid.uuid4().hex}.tmp"
        try:
            tmp_content_path.write_bytes(data)
        except BaseException:
            tmp_content_path.unlink(missing_ok=True)
            raise
        try:
            tmp_content_path.replace(content_path)
        except OSError:
            # Lost a race with a concurrent writer submitting the exact
            # same content (same hash => byte-identical by definition).
            # On POSIX this branch is unreachable (rename onto an
            # existing path is atomic and simply "wins" or "loses"
            # silently); on Windows, `Path.replace` can raise
            # `PermissionError` when another thread/process is
            # renaming onto the same destination concurrently -- mirrors
            # `historical.canonical_store.write_partition`'s identical
            # race-tolerant rename.
            tmp_content_path.unlink(missing_ok=True)
            if not content_path.is_file():
                raise

        created_at = format_utc_timestamp(utc_now())
        sidecar = _ContentSidecar(
            schema_version=_SCHEMA_VERSION, content_hash=content_hash, category=category,
            size_bytes=len(data), created_at=created_at,
        )
        self._write_sidecar_tolerating_race(metadata_path, sidecar)

        logger.info("ML artifact written: content_hash=%s category=%s size_bytes=%d", content_hash[:12], category.value, len(data))
        return ArtifactReference(category=category, content_hash=content_hash, size_bytes=len(data), created_at=created_at)

    def _write_sidecar_tolerating_race(self, metadata_path: Path, sidecar: _ContentSidecar) -> None:
        """Same race-tolerant atomic-rename pattern as the content write
        above: two concurrent writers producing the SAME sidecar content
        (same content hash => same category/size, by construction) must
        not fail each other. Never overwrites a sidecar that already
        exists for a genuinely different reason -- `write_artifact`
        already checked category consistency before reaching here."""
        tmp_path = metadata_path.parent / f".{metadata_path.name}.{uuid.uuid4().hex}.tmp"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp_path.write_bytes(canonical_json_bytes(sidecar.to_json_dict()))
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        try:
            tmp_path.replace(metadata_path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            if not metadata_path.is_file():
                raise

    def _read_sidecar_if_present(self, metadata_path: Path, *, content_hash: str) -> _ContentSidecar | None:
        if not metadata_path.is_file():
            return None
        raw = read_json_file(metadata_path)
        sidecar = _ContentSidecar.from_json_dict(as_json_dict(raw, field_name="content sidecar"))
        if sidecar.content_hash != content_hash:
            raise ArtifactCorruptionError(
                f"Metadata sidecar at {metadata_path} records content_hash {sidecar.content_hash!r}, "
                f"which does not match its own filename-derived hash {content_hash!r}",
                context={"path": str(metadata_path), "recorded": sidecar.content_hash, "expected": content_hash},
            )
        return sidecar

    def artifact_exists(self, content_hash: str) -> bool:
        validate_artifact_hash(content_hash)
        return self._content_path(content_hash).is_file()

    def read_artifact(self, content_hash: str) -> bytes:
        """Read and verify one artifact's raw bytes. Raises
        `ArtifactNotFoundError` if no content exists at that hash;
        `ArtifactCorruptionError` if the bytes on disk do not hash to the
        requested content id, or the metadata sidecar is missing/corrupted/
        inconsistent -- corruption is always detected before use, never
        silently returned."""
        validate_artifact_hash(content_hash)
        content_path = self._content_path(content_hash)
        if not content_path.is_file():
            raise ArtifactNotFoundError(
                f"No ML artifact stored for content_hash={content_hash!r}", context={"content_hash": content_hash}
            )
        data = content_path.read_bytes()
        actual_hash = sha256_hex_bytes(data)
        if actual_hash != content_hash:
            raise ArtifactCorruptionError(
                f"ML artifact at {content_path} does not match its content-addressed hash "
                f"(expected {content_hash!r}, actual {actual_hash!r}) -- data is corrupted",
                context={"path": str(content_path), "expected": content_hash, "actual": actual_hash},
            )

        metadata_path = self._metadata_path(content_hash)
        if not metadata_path.is_file():
            raise ArtifactCorruptionError(
                f"ML artifact content exists for {content_hash!r} but its metadata sidecar is missing",
                context={"content_hash": content_hash, "path": str(metadata_path)},
            )
        sidecar = self._read_sidecar_if_present(metadata_path, content_hash=content_hash)
        assert sidecar is not None  # metadata_path.is_file() was just checked above
        if sidecar.size_bytes != len(data):
            raise ArtifactCorruptionError(
                f"ML artifact {content_hash!r} metadata records size_bytes={sidecar.size_bytes}, "
                f"but actual content is {len(data)} bytes",
                context={"content_hash": content_hash, "expected_size": sidecar.size_bytes, "actual_size": len(data)},
            )
        return data

    def artifact_reference(self, content_hash: str) -> ArtifactReference:
        """The `ArtifactReference` for an already-written artifact, read
        from its metadata sidecar (verifying content in the process)."""
        self.read_artifact(content_hash)
        sidecar = self._read_sidecar_if_present(self._metadata_path(content_hash), content_hash=content_hash)
        assert sidecar is not None
        return ArtifactReference(
            category=sidecar.category, content_hash=sidecar.content_hash,
            size_bytes=sidecar.size_bytes, created_at=sidecar.created_at,
        )


__all__ = ["MLArtifactStore"]
