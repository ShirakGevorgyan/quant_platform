"""Safe, explicit persistence boundary for every durable ML JSON structure
(specs, identities, manifests, validation reports, artifact metadata,
environment snapshots, seed configurations, tracking events).

THE THREE GUARANTEES THIS MODULE EXISTS TO ENFORCE
--------------------------------------------------------------------------
1. **Determinism.** `canonical_json_bytes` sorts dictionary keys and uses
   compact, fixed separators -- the same logical payload always produces
   the same bytes, independent of insertion order, process, or machine.
   List/tuple ORDER is deliberately NOT touched (callers decide whether
   order is semantically meaningful, e.g. an ordered feature list).
2. **No non-finite floats.** Both the write path (`allow_nan=False`) and
   the read path (`parse_constant` rejecting the `NaN`/`Infinity`/
   `-Infinity` tokens Python's `json` module otherwise accepts as a
   non-standard extension) refuse `NaN`/`Infinity` -- a durable experiment
   record silently containing `NaN` is exactly the kind of corruption this
   platform's "never silently repair" philosophy forbids.
3. **No arbitrary Python object serialization.** Every function in this
   module operates on plain JSON-native structures (`dict`/`list`/`str`/
   `int`/`float`/`bool`/`None`) -- there is no code path here that can
   serialize or deserialize an arbitrary Python object, let alone execute
   one. Model artifacts that genuinely need a richer (and unsafe) format
   are an explicit, separately-trusted concern for a later milestone's
   model serializer -- never this generic layer (see `ml/artifacts.py`'s
   module docstring).

All durable JSON structures carry an explicit integer `schema_version`
field; `require_schema_version` rejects anything this code version does
not know how to read rather than guessing.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    MLError,
    PathSecurityError,
    SchemaVersionError,
)


def _strip_windows_extended_prefix(path: Path) -> Path:
    """Strip the `\\\\?\\` extended-length-path prefix Windows sometimes
    attaches to a resolved path (via `GetFinalPathNameByHandle`) -- a
    no-op on POSIX. Observed in practice under concurrent access to
    nearby paths (multiple threads creating sibling `content/<prefix>/`
    directories at once): one path resolves with the prefix and a
    sibling resolves without it, even though both are the same short
    length and neither involves a real reparse point -- comparing them
    directly then makes a same-root path look like it "escapes" its own
    root. Stripping the prefix before comparison fixes the false
    positive without weakening the actual containment check."""
    text = str(path)
    if text.startswith("\\\\?\\"):
        text = text[4:]
    return Path(text)


def assert_within_root(path: Path, *, root: Path) -> None:
    """Shared path-containment guard for every content-addressed ML
    store (`artifacts.py`, `manifests.py`, `tracking.py`) -- raises
    `PathSecurityError` if `path`'s resolved (symlink-following)
    location is not `root` or a descendant of it."""
    resolved = _strip_windows_extended_prefix(path.resolve())
    resolved_root = _strip_windows_extended_prefix(root.resolve())
    if resolved_root not in resolved.parents and resolved != resolved_root:
        raise PathSecurityError(
            f"Resolved path {resolved} escapes storage root {resolved_root}",
            context={"path": str(path), "root": str(root)},
        )


def sha256_hex_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Deterministic JSON encoding used for BOTH content fingerprinting and
    durable storage. Raises `ValueError` if `payload` contains a NaN/
    Infinity float anywhere (rather than silently encoding a non-standard
    token another reader might not accept, or worse, mis-fingerprinting)."""
    try:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)
    except ValueError as exc:
        raise ValueError(
            f"Payload contains a non-finite float (NaN/Infinity), which is forbidden in "
            f"durable ML JSON: {exc}"
        ) from exc
    return text.encode("utf-8")


def _reject_non_finite_constant(token: str) -> float:
    raise ValueError(f"Non-finite JSON token {token!r} is forbidden in durable ML JSON")


def parse_json_strict(text: str) -> Any:
    """`json.loads` with `NaN`/`Infinity`/`-Infinity` tokens rejected --
    Python's `json` module accepts them by default as a non-standard
    extension, which would otherwise let a corrupted or malicious file
    smuggle a non-finite value straight past `canonical_json_bytes`'s
    write-side guard."""
    try:
        return json.loads(text, parse_constant=_reject_non_finite_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON: {exc}") from exc


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def format_utc_timestamp(ts: pd.Timestamp) -> str:
    """Explicit UTC ISO-8601 rendering. Raises `ValueError` on a naive
    timestamp -- every timestamp entering durable ML JSON must be
    unambiguous, exactly like the historical pipeline's timezone policy."""
    if ts.tzinfo is None:
        raise ValueError(f"Timestamp {ts} is not timezone-aware; ML durable JSON requires explicit UTC timestamps")
    return ts.tz_convert("UTC").isoformat()


def parse_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError(f"Timestamp {value!r} is not timezone-aware UTC")
    return ts.tz_convert("UTC")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Temp-file-then-rename atomic write, consistent with every other
    storage layer in this platform (`historical.canonical_store`,
    `features.manifests`, ...) -- a reader never observes a partially
    written file under the final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp_path.write_bytes(canonical_json_bytes(payload))
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def read_json_file(path: Path) -> Any:
    if not path.is_file():
        raise ArtifactCorruptionError(
            f"Expected JSON file does not exist: {path}", context={"path": str(path)}
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactCorruptionError(
            f"Failed to read JSON file {path}: {exc}", context={"path": str(path)}
        ) from exc
    try:
        return parse_json_strict(text)
    except ValueError as exc:
        raise ArtifactCorruptionError(f"Malformed JSON in {path}: {exc}", context={"path": str(path)}) from exc


def as_json_dict(value: object, *, field_name: str) -> dict[str, Any]:
    """Narrow a raw JSON `object` field to `dict[str, Any]` with an
    actionable error -- used throughout every `from_json_dict` in this
    package instead of a `# type: ignore` comment, so a malformed durable
    record fails with a clear message rather than a confusing
    `AttributeError` deep inside a comprehension."""
    if not isinstance(value, dict):
        raise MLError(f"{field_name}: expected a JSON object, got {type(value).__name__}", context={"field": field_name})
    return value


def as_json_list(value: object, *, field_name: str) -> list[Any]:
    """Narrow a raw JSON `object` field to `list[Any]` -- see `as_json_dict`."""
    if not isinstance(value, list):
        raise MLError(f"{field_name}: expected a JSON array, got {type(value).__name__}", context={"field": field_name})
    return value


def require_schema_version(raw: dict[str, Any], *, supported: int, context: str, field: str = "schema_version") -> int:
    """Reject anything other than EXACTLY `supported` -- this platform has
    no schema-migration logic yet, so "unknown" and "older/newer than
    supported" are both refused rather than guessed at."""
    value = raw.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SchemaVersionError(
            f"{context}: missing or non-integer {field!r} field", context={"raw_value": value}
        )
    if value != supported:
        raise SchemaVersionError(
            f"{context}: unsupported {field}={value} (this code only supports {supported})",
            context={"found": value, "supported": supported},
        )
    return value


__all__ = [
    "as_json_dict",
    "as_json_list",
    "assert_within_root",
    "canonical_json_bytes",
    "format_utc_timestamp",
    "parse_json_strict",
    "parse_utc_timestamp",
    "read_json_file",
    "require_schema_version",
    "sha256_hex_bytes",
    "utc_now",
    "write_json_atomic",
]
