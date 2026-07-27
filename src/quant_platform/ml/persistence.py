"""Safe, explicit persistence boundary for every durable ML JSON structure
(specs, identities, manifests, validation reports, artifact metadata,
environment snapshots, seed configurations, tracking events).

`canonical_json_bytes`, `parse_json_strict`, `sha256_hex_bytes`, and
`write_json_atomic` are RE-EXPORTED, byte-for-byte unchanged, from
`quant_platform.core.json` (Milestone 4D.1's dependency-neutral home for
these primitives -- see that module's docstring for the full rationale:
`historical`/`features` need them too, and importing `quant_platform.ml`
from either would invert this platform's dependency graph). This module
adds ML-specific concerns on top: path-containment (`assert_within_root`),
schema-version enforcement (`require_schema_version`), JSON-primitive type
narrowing (`as_json_dict`/`as_json_list`), UTC timestamp formatting, and
`read_json_file` -- a convenience wrapper that additionally translates a
missing/unreadable/malformed file into `ArtifactCorruptionError`, the
ML-domain exception `core.json` itself deliberately has no opinion on.

There is exactly ONE implementation of `canonical_json_bytes`/
`parse_json_strict`/`sha256_hex_bytes`/`write_json_atomic` in this
codebase, in `quant_platform.core.json`; nothing here reimplements or
overrides their behavior -- `quant_platform.ml.persistence.
parse_json_strict is quant_platform.core.json.parse_json_strict` (and
likewise for the other three) holds by construction, not by convention.

All durable JSON structures carry an explicit integer `schema_version`
field; `require_schema_version` rejects anything this code version does
not know how to read rather than guessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.core.exceptions import (
    ArtifactCorruptionError,
    MLError,
    PathSecurityError,
    SchemaVersionError,
)
from quant_platform.core.json import (
    canonical_json_bytes,
    parse_json_strict,
    sha256_hex_bytes,
    write_json_atomic,
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


def read_json_file(path: Path) -> Any:
    if not path.is_file():
        raise ArtifactCorruptionError(
            f"Expected JSON file does not exist: {path}", context={"path": str(path)}
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
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
