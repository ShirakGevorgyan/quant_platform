"""Domain-level content fingerprinting built on `persistence.
canonical_json_bytes`. Every fingerprint in `quant_platform.ml` -- model
registry, experiment identity, fitted-preprocessing binding, seed
configuration -- ultimately goes through `fingerprint_json`, so they all
share the same determinism guarantees (sorted dict keys, no non-finite
floats, no object identity/repr/address ever involved).
"""

from __future__ import annotations

import hashlib
from typing import Any

from quant_platform.ml.persistence import canonical_json_bytes, sha256_hex_bytes

_VALID_HEX_CHARS = set("0123456789abcdef")


def is_valid_sha256_hex(value: str) -> bool:
    """The one shared definition of "looks like a sha256 hex digest" --
    every module that validates a content hash, experiment id, or
    fingerprint string (`models.py`, `experiment_identity.py`,
    `manifests.py`, ...) delegates to this rather than re-deriving its
    own character-class check."""
    return len(value) == 64 and all(c in _VALID_HEX_CHARS for c in value.lower())


def fingerprint_json(payload: dict[str, Any]) -> str:
    """Sha256 hex digest of `payload`'s canonical JSON encoding."""
    return sha256_hex_bytes(canonical_json_bytes(payload))


def combine_fingerprints(*fingerprints: str) -> str:
    """Deterministically combine already-computed fingerprints into one.
    ORDER-SENSITIVE by design (callers who want an order-independent
    combination must sort their input first, e.g. `ModelRegistry.
    fingerprint()` sorts by `(name, version)` before combining)."""
    digest = hashlib.sha256()
    for fingerprint in fingerprints:
        digest.update(fingerprint.encode("utf-8"))
        digest.update(b"|")
    return digest.hexdigest()


__all__ = ["combine_fingerprints", "fingerprint_json", "is_valid_sha256_hex"]
