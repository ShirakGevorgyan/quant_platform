"""Centralized, deterministic seed management.

A single `SeedConfiguration.master_seed` deterministically derives a
distinct child seed per named domain (`SeedDomain`) via `derive_seed`.
Derivation is a pure SHA-256-based function of `(master_seed, domain
name)` -- deliberately NOT Python's builtin `hash()`, which is randomized
per-process (via `PYTHONHASHSEED`) unless explicitly disabled, and would
therefore make "repeated derivation produces identical values across
processes" false. SHA-256 has no such randomization: the same
`(master_seed, domain)` pair always produces the same derived seed, in any
process, on any machine, forever.

Domains are plain strings (`SeedDomain` is a convenience enum over the
domains this milestone anticipates) -- adding a new domain later needs no
change to the derivation function itself, satisfying "the seed model
should be extensible" without speculative unused fields today.

**Explicit limitation, not a promise this module cannot keep**: seeding
Python's `random` and NumPy's `Generator` deterministically does NOT
guarantee bit-for-bit determinism from every future third-party algorithm
(e.g. a native/GPU-accelerated library with its own non-deterministic
reduction order). This module seeds what it controls; it records, rather
than hides, that boundary.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import Enum

import numpy as np

from quant_platform.core.exceptions import InvalidSeedError
from quant_platform.ml.fingerprints import fingerprint_json
from quant_platform.ml.persistence import require_schema_version

MAX_SEED = 2**32 - 1
_SCHEMA_VERSION = 1


class SeedDomain(Enum):
    GLOBAL = "global"
    MODEL_INIT = "model_init"
    DATA_OPERATIONS = "data_operations"
    CROSS_VALIDATION = "cross_validation"
    HYPERPARAMETER_SEARCH = "hyperparameter_search"
    FEATURE_SELECTION = "feature_selection"
    CALIBRATION = "calibration"
    ENSEMBLE = "ensemble"


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise InvalidSeedError(f"seed must be a plain int, got {type(seed).__name__}", context={"seed": seed})
    if not (0 <= seed <= MAX_SEED):
        raise InvalidSeedError(
            f"seed must be in [0, {MAX_SEED}], got {seed}", context={"seed": seed, "max_seed": MAX_SEED}
        )


def derive_seed(master_seed: int, domain: SeedDomain | str) -> int:
    """Deterministic child seed for `domain`, derived from `master_seed` via
    SHA-256 -- never Python's randomized `hash()`. Always in `[0, MAX_SEED]`,
    so the result is valid for both `random.Random` and
    `numpy.random.default_rng`."""
    _validate_seed(master_seed)
    domain_name = domain.value if isinstance(domain, SeedDomain) else str(domain)
    if not domain_name:
        raise InvalidSeedError("domain name must not be empty")
    digest = hashlib.sha256(f"{master_seed}:{domain_name}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big")


@dataclass(frozen=True, slots=True)
class SeedConfiguration:
    """Identity-relevant seed state -- deliberately holds ONLY
    `master_seed` (plus a schema version). There is no descriptive/notes
    field here at all, which is what makes "changing an unused descriptive
    seed field must not be possible" true by construction: no such field
    exists to change."""

    master_seed: int
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_seed(self.master_seed)

    def derive(self, domain: SeedDomain | str) -> int:
        return derive_seed(self.master_seed, domain)

    def random_for(self, domain: SeedDomain | str) -> random.Random:
        return random.Random(self.derive(domain))

    def numpy_generator_for(self, domain: SeedDomain | str) -> np.random.Generator:
        return np.random.default_rng(self.derive(domain))

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "master_seed": self.master_seed}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SeedConfiguration:
        require_schema_version(raw, supported=_SCHEMA_VERSION, context="SeedConfiguration")
        return cls(master_seed=int(str(raw["master_seed"])), schema_version=_SCHEMA_VERSION)

    def fingerprint(self) -> str:
        return fingerprint_json(self.to_json_dict())


__all__ = ["MAX_SEED", "SeedConfiguration", "SeedDomain", "derive_seed"]
