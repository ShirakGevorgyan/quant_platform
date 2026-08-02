"""`LabelFamily` and `LabelSpecification` (Milestone 11, Phase 3, Part
A) -- the immutable, content-addressed description of WHAT a label
would be, completely separate from any callable that actually computes
one (see `builder.LabelDefinition`, mirroring `features.models.FeatureSpec`
vs. `features.interfaces.FeatureDefinition`'s own split exactly).

`LabelSpecification` carries every field the governing specification
names (`label_specification_id`, `label_family`, `schema_version`,
`generation_version`, `parameter_hash`, `price_basis`,
`prediction_horizon`, `availability_rule`, `reference_price`,
`event_time_rule`, `generation_rule`, `identity_algorithm`,
`created_from_dataset`, `created_from_manifest`) plus one implementation-
necessary supporting field, `parameters` -- the raw generation-parameter
dict `parameter_hash` is a deterministic digest OF, so a spec's own
self-consistency (parameter tampering) can be independently re-verified
rather than trusting an opaque hash nobody can recompute.

Deliberately standalone: this module imports nothing from `features`,
`qualification`, or `feature_discovery` -- `created_from_dataset`/
`created_from_manifest` are plain caller-supplied identity strings (a
`ResearchDatasetManifest`'s own `dataset_id`/`version`, in practice),
never a live object reference. This is what keeps the preferred
dependency graph (Market Data -> Features -> Qualification -> Feature
Discovery -> Labels -> Machine Learning) a workflow ordering rather than
a Python import requirement, and makes a circular dependency structurally
impossible: nothing downstream of this package needs to be imported for
this package to do its job."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

from quant_platform.core.exceptions import LabelIdentityError, LabelRequestError
from quant_platform.ml.persistence import as_json_dict, require_schema_version

__all__ = [
    "LABEL_IDENTITY_ALGORITHM",
    "LABEL_SPECIFICATION_SCHEMA_VERSION",
    "LabelFamily",
    "LabelSpecification",
    "build_label_specification",
    "compute_label_specification_id",
    "compute_parameter_hash",
]

LABEL_SPECIFICATION_SCHEMA_VERSION = 1

LABEL_IDENTITY_ALGORITHM = "sha256-v1"
"""The identity-hashing scheme this code version produces for a NEW
specification. A specification replayed from an older run is free to
carry a different value (recorded, never silently rewritten) --
`identity_algorithm` is itself part of a spec's content and therefore
part of its `label_specification_id`, so a future algorithm change is
automatically a new identity, never a silent reinterpretation of an old
one."""


class LabelFamily(Enum):
    """The 6 named families this infrastructure must support. No family
    is privileged -- nothing in this package ever branches on which
    value is set for behavior, only for grouping/reporting. Part A ships
    no generation logic for any of them; see `builder.py`'s module
    docstring."""

    NEXT_RETURN = "next_return"
    MULTI_HORIZON_RETURN = "multi_horizon_return"
    DIRECTION = "direction"
    TRIPLE_BARRIER = "triple_barrier"
    FORWARD_VOLATILITY = "forward_volatility"
    FUTURE_EXTENSION_PLACEHOLDER = "future_extension_placeholder"


def compute_parameter_hash(parameters: dict[str, object]) -> str:
    """Deterministic sha256 over `parameters`' canonical JSON (sorted
    keys, `allow_nan=False` -- a label parameter is never a legitimate
    NaN/Infinity)."""
    payload = json.dumps(parameters, sort_keys=True, default=str, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_label_specification_id(
    *, schema_version: int, label_family: LabelFamily, generation_version: str, parameter_hash: str, price_basis: str,
    prediction_horizon: str, availability_rule: str, reference_price: str, event_time_rule: str, generation_rule: str,
    identity_algorithm: str, created_from_dataset: str, created_from_manifest: str,
) -> str:
    """Deterministic sha256 over every OTHER field of a `LabelSpecification`
    -- content-addressed, immutable, portable: independent of filesystem,
    clock, process id, Python hash seed, `random`, temp directory,
    machine, or operating system, exactly as the governing specification
    requires. `default=str`/`allow_nan=False`/sorted keys mirror
    `features.models.FeatureSpec.fingerprint`'s own established
    convention -- never migrated to a different serializer, so the exact
    byte representation this hash is computed over is always inspectable
    from the payload dict alone."""
    payload = {
        "schema_version": schema_version, "label_family": label_family.value, "generation_version": generation_version,
        "parameter_hash": parameter_hash, "price_basis": price_basis, "prediction_horizon": prediction_horizon,
        "availability_rule": availability_rule, "reference_price": reference_price, "event_time_rule": event_time_rule,
        "generation_rule": generation_rule, "identity_algorithm": identity_algorithm, "created_from_dataset": created_from_dataset,
        "created_from_manifest": created_from_manifest,
    }
    blob = json.dumps(payload, sort_keys=True, default=str, allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LabelSpecification:
    """Pure metadata for one label recipe. Immutable and fully
    JSON-serializable. `(label_family, generation_version, parameter_hash)`
    is the recipe's logical identity; `label_specification_id` is its
    content-addressed digest -- changing ANY field (a new horizon, a new
    barrier, a new price basis, a new neutral threshold, a new volatility
    estimator) always produces a new id. Never mutate a registered
    specification in place; register a new one instead (append-only, per
    `registry.LabelRegistry`)."""

    schema_version: int
    label_specification_id: str
    label_family: LabelFamily
    generation_version: str
    parameter_hash: str
    price_basis: str
    prediction_horizon: str
    availability_rule: str
    reference_price: str
    event_time_rule: str
    generation_rule: str
    identity_algorithm: str
    created_from_dataset: str
    created_from_manifest: str
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_from_dataset:
            raise LabelRequestError("LabelSpecification.created_from_dataset must not be empty")
        if not self.created_from_manifest:
            raise LabelRequestError("LabelSpecification.created_from_manifest must not be empty")
        if not self.generation_version:
            raise LabelRequestError("LabelSpecification.generation_version must not be empty")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id,
            "label_family": self.label_family.value, "generation_version": self.generation_version, "parameter_hash": self.parameter_hash,
            "price_basis": self.price_basis, "prediction_horizon": self.prediction_horizon, "availability_rule": self.availability_rule,
            "reference_price": self.reference_price, "event_time_rule": self.event_time_rule, "generation_rule": self.generation_rule,
            "identity_algorithm": self.identity_algorithm, "created_from_dataset": self.created_from_dataset,
            "created_from_manifest": self.created_from_manifest, "parameters": self.parameters,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelSpecification:
        require_schema_version(raw, supported=LABEL_SPECIFICATION_SCHEMA_VERSION, context="LabelSpecification")
        return cls(
            schema_version=LABEL_SPECIFICATION_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            label_family=LabelFamily(raw["label_family"]), generation_version=str(raw["generation_version"]),
            parameter_hash=str(raw["parameter_hash"]), price_basis=str(raw["price_basis"]), prediction_horizon=str(raw["prediction_horizon"]),
            availability_rule=str(raw["availability_rule"]), reference_price=str(raw["reference_price"]),
            event_time_rule=str(raw["event_time_rule"]), generation_rule=str(raw["generation_rule"]),
            identity_algorithm=str(raw["identity_algorithm"]), created_from_dataset=str(raw["created_from_dataset"]),
            created_from_manifest=str(raw["created_from_manifest"]),
            parameters=as_json_dict(raw.get("parameters") or {}, field_name="parameters"),
        )

    def verify_self_consistency(self) -> tuple[bool, tuple[str, ...]]:
        """Independently recomputes `parameter_hash` (from `parameters`)
        and `label_specification_id` (from every other field) and
        compares against what this instance itself claims -- the
        tampering detector every registration/verification path in this
        package calls, never trusting a hand-constructed or
        deserialized instance at face value."""
        issues: list[str] = []
        recomputed_parameter_hash = compute_parameter_hash(self.parameters)
        if recomputed_parameter_hash != self.parameter_hash:
            issues.append(f"parameter_hash: claims {self.parameter_hash!r}, recomputed from parameters is {recomputed_parameter_hash!r}")

        recomputed_id = compute_label_specification_id(
            schema_version=self.schema_version, label_family=self.label_family, generation_version=self.generation_version,
            parameter_hash=self.parameter_hash, price_basis=self.price_basis, prediction_horizon=self.prediction_horizon,
            availability_rule=self.availability_rule, reference_price=self.reference_price, event_time_rule=self.event_time_rule,
            generation_rule=self.generation_rule, identity_algorithm=self.identity_algorithm, created_from_dataset=self.created_from_dataset,
            created_from_manifest=self.created_from_manifest,
        )
        if recomputed_id != self.label_specification_id:
            issues.append(f"label_specification_id: claims {self.label_specification_id!r}, recomputed is {recomputed_id!r}")
        return (not issues, tuple(issues))


def build_label_specification(
    *, label_family: LabelFamily, generation_version: str, price_basis: str, prediction_horizon: str, availability_rule: str,
    reference_price: str, event_time_rule: str, generation_rule: str, created_from_dataset: str, created_from_manifest: str,
    parameters: dict[str, object] | None = None, identity_algorithm: str = LABEL_IDENTITY_ALGORITHM,
) -> LabelSpecification:
    """The ONE sanctioned way to construct a new `LabelSpecification` --
    computes `parameter_hash` and `label_specification_id` FROM the
    supplied fields rather than accepting them as caller-supplied
    values, so a spec built through this function is self-consistent by
    construction. A hand-constructed or deserialized instance is instead
    verified via `LabelSpecification.verify_self_consistency`."""
    resolved_parameters = parameters or {}
    parameter_hash = compute_parameter_hash(resolved_parameters)
    label_specification_id = compute_label_specification_id(
        schema_version=LABEL_SPECIFICATION_SCHEMA_VERSION, label_family=label_family, generation_version=generation_version,
        parameter_hash=parameter_hash, price_basis=price_basis, prediction_horizon=prediction_horizon, availability_rule=availability_rule,
        reference_price=reference_price, event_time_rule=event_time_rule, generation_rule=generation_rule, identity_algorithm=identity_algorithm,
        created_from_dataset=created_from_dataset, created_from_manifest=created_from_manifest,
    )
    specification = LabelSpecification(
        schema_version=LABEL_SPECIFICATION_SCHEMA_VERSION, label_specification_id=label_specification_id, label_family=label_family,
        generation_version=generation_version, parameter_hash=parameter_hash, price_basis=price_basis, prediction_horizon=prediction_horizon,
        availability_rule=availability_rule, reference_price=reference_price, event_time_rule=event_time_rule, generation_rule=generation_rule,
        identity_algorithm=identity_algorithm, created_from_dataset=created_from_dataset, created_from_manifest=created_from_manifest,
        parameters=resolved_parameters,
    )
    consistent, issues = specification.verify_self_consistency()
    if not consistent:  # pragma: no cover - unreachable unless this function's own hashing is internally inconsistent
        raise LabelIdentityError(f"build_label_specification produced a self-inconsistent specification: {issues}", context={"issues": list(issues)})
    return specification
