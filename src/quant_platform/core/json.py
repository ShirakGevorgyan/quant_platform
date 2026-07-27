"""Dependency-neutral durable-JSON primitives (Milestone 4D.1 completion).

THE authoritative implementation of this platform's durable-JSON policy --
canonical (deterministic, finite-only, duplicate-key-free) encoding and
strict (finite-only, duplicate-key-rejecting) decoding. Lives in
`quant_platform.core` -- which performs no I/O and depends on nothing
else in this platform (see `core/__init__.py`'s own docstring) -- so
every subpackage (`historical`, `features`, `ml`, `execution`,
`optimization`) can depend on it directly without inverting the
platform's dependency graph. `ml` already depends on `historical`/
`features` (`ml.experiment_manager`/`ml.validation` import `features.
manifests`; `ml.concurrency` imports `historical.locking`); those
packages must never depend back on `ml.persistence`, which is why this
module does not live there. `quant_platform.ml.persistence` re-exports
every name below unchanged, purely for backward compatibility with
existing `from quant_platform.ml.persistence import ...` call sites --
there is exactly ONE implementation of each of these functions in the
whole codebase.

THE THREE GUARANTEES THIS MODULE EXISTS TO ENFORCE
--------------------------------------------------------------------------
1. **Determinism.** `canonical_json_bytes` sorts dictionary keys and uses
   compact, fixed separators -- the same logical payload always produces
   the same bytes, independent of insertion order, process, or machine.
   List/tuple ORDER is deliberately NOT touched (callers decide whether
   order is semantically meaningful, e.g. an ordered feature list).
2. **No non-finite floats, on read or write.** `canonical_json_bytes`
   (`allow_nan=False`) and `parse_json_strict` (`parse_constant` rejecting
   the `NaN`/`Infinity`/`-Infinity` tokens Python's `json` module
   otherwise accepts as a non-standard extension) both refuse them -- a
   durable platform record silently containing `NaN` is exactly the kind
   of corruption this platform's "never silently repair" philosophy
   forbids.
3. **No duplicate keys.** `canonical_json_bytes` can structurally never
   produce one (it serializes from a `dict`, which cannot hold one);
   `parse_json_strict` rejects one on read via `object_pairs_hook` --
   Python's standard library otherwise silently keeps the LAST occurrence
   and drops every earlier one, which would let a hand-edited or
   corrupted file shadow a value without any signal. Because guarantee 1
   already makes a duplicate key structurally impossible in anything this
   platform's own writer ever produces, rejecting one is free: it can
   only ever reject a file that was never legitimately written by this
   codebase in the first place.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
--------------------------------------------------------------------------
It raises nothing more specific than the standard library's own
`ValueError`/`OSError` (`write_json_atomic`'s temp-file-then-rename is a
pure filesystem operation with no domain knowledge). Translating a
failure here into a domain-specific exception (`ArtifactCorruptionError`,
`ManifestError`, `SnapshotError`, `ResearchDatasetError`, `DatasetLockError`,
...) is every CALLER's job -- this module has no opinion on which
subpackage, or which artifact category, is reading its output. It also
performs no path-containment checking, no schema-version enforcement, and
no JSON-primitive type narrowing: those remain `ml.persistence`-specific
concerns (`assert_within_root`, `require_schema_version`, `as_json_dict`/
`as_json_list`) since they are not needed by every dependent package and
moving them would not by itself resolve any circular-import risk.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any


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
            f"durable platform JSON: {exc}"
        ) from exc
    return text.encode("utf-8")


def _reject_non_finite_constant(token: str) -> float:
    raise ValueError(f"Non-finite JSON token {token!r} is forbidden in durable platform JSON")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """`object_pairs_hook` for `json.loads` -- called once per JSON object
    encountered (including every nested object), so this single hook
    enforces the policy at every nesting depth without extra plumbing."""
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"Duplicate key {key!r} is forbidden in durable platform JSON")
        seen[key] = value
    return seen


def parse_json_strict(text: str) -> Any:
    """`json.loads` with `NaN`/`Infinity`/`-Infinity` tokens AND duplicate
    object keys rejected -- Python's `json` module otherwise accepts the
    former as a non-standard extension and silently resolves the latter
    to its last occurrence. Both would otherwise let a corrupted or
    malicious file smuggle a value straight past `canonical_json_bytes`'s
    write-side guarantees. Raises `ValueError` for malformed JSON, a
    non-finite token, or a duplicate key."""
    try:
        return json.loads(text, parse_constant=_reject_non_finite_constant, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON: {exc}") from exc


def sha256_hex_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Temp-file-then-rename atomic write -- a reader never observes a
    partially written file under the final path. Pure filesystem
    operation: raises `ValueError` (from `canonical_json_bytes`) or
    `OSError` only, never a domain-specific exception."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp_path.write_bytes(canonical_json_bytes(payload))
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


__all__ = [
    "canonical_json_bytes",
    "parse_json_strict",
    "sha256_hex_bytes",
    "write_json_atomic",
]
