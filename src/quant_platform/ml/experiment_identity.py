"""Deterministic experiment identity: the same scientifically-relevant
`ExperimentSpec` always produces the same `experiment_id`; a spec that
differs in any scientifically-relevant way always produces a different one.

WHAT PARTICIPATES IN IDENTITY (and nothing else)
--------------------------------------------------------------------------
Exactly the fields returned by `ExperimentSpec.to_identity_payload()`:
dataset binding, feature binding (ordered feature names + versions +
registry fingerprint), label binding, split binding, preprocessing binding
(including the fitted preprocessing fingerprint), model name + version,
hyperparameters, objective, seed configuration (master seed only),
code revision binding, and `environment_requirements` (empty unless a
human explicitly declares one). See `experiment_spec.py`'s module
docstring for the full identity-vs-descriptive rationale.

WHAT NEVER PARTICIPATES
--------------------------------------------------------------------------
`primary_metric`, `tags`, `notes` (descriptive/reporting fields on
`ExperimentSpec`); anything from `EnvironmentSnapshot` (informational
only, see `models.py`); wall-clock timestamps; process IDs; memory
addresses; local/temp file paths; Python's randomized `hash()`; JSON key
order or dict insertion order (`canonical_json_bytes` sorts keys and this
module never hashes a raw `repr()`).

WHY A SEPARATE `IDENTITY_SCHEMA_VERSION`
--------------------------------------------------------------------------
`ExperimentSpec.schema_version` versions the SPEC's own structure.
`IDENTITY_SCHEMA_VERSION` versions the HASHING SCHEME itself (e.g. which
fields are folded into the payload, in case a future milestone needs to
change what counts as identity-relevant without silently colliding old
and new IDs) -- it is folded into the hashed payload so a scheme change
always changes every `experiment_id`, even for byte-identical specs under
the old scheme.
"""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.core.exceptions import ExperimentIdentityError
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.fingerprints import fingerprint_json, is_valid_sha256_hex
from quant_platform.ml.persistence import require_schema_version

IDENTITY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExperimentIdentity:
    """The durable, immutable result of hashing an `ExperimentSpec`'s
    identity payload. `experiment_id` is a 64-character lowercase hex
    SHA-256 digest -- never a raw `repr()`, never machine- or
    time-dependent."""

    schema_version: int
    experiment_id: str

    def __post_init__(self) -> None:
        if not is_valid_sha256_hex(self.experiment_id):
            raise ExperimentIdentityError(
                f"ExperimentIdentity.experiment_id must be a 64-character lowercase hex SHA-256 digest, "
                f"got {self.experiment_id!r}",
                context={"experiment_id": self.experiment_id},
            )

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "experiment_id": self.experiment_id}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> ExperimentIdentity:
        require_schema_version(raw, supported=IDENTITY_SCHEMA_VERSION, context="ExperimentIdentity")
        return cls(schema_version=IDENTITY_SCHEMA_VERSION, experiment_id=str(raw["experiment_id"]))


def compute_experiment_identity(spec: ExperimentSpec) -> ExperimentIdentity:
    """Pure function: `ExperimentSpec` -> `ExperimentIdentity`. Calling this
    twice on two independently constructed `ExperimentSpec` instances that
    are scientifically identical (same identity payload) always returns
    the same `experiment_id`, regardless of process, machine, dict
    insertion order, or wall-clock time."""
    payload = dict(spec.to_identity_payload())
    payload["identity_schema_version"] = IDENTITY_SCHEMA_VERSION
    experiment_id = fingerprint_json(payload)
    return ExperimentIdentity(schema_version=IDENTITY_SCHEMA_VERSION, experiment_id=experiment_id)


def verify_experiment_identity(spec: ExperimentSpec, identity: ExperimentIdentity) -> bool:
    """True iff `identity` is exactly what `compute_experiment_identity(spec)`
    would produce right now -- used by validation/reconstruction to detect
    a corrupted or tampered manifest (an `experiment_id` that does not
    match its own recorded spec)."""
    return compute_experiment_identity(spec) == identity


__all__ = [
    "IDENTITY_SCHEMA_VERSION",
    "ExperimentIdentity",
    "compute_experiment_identity",
    "verify_experiment_identity",
]
