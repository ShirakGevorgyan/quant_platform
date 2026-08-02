"""`LabelRegistry` (Milestone 11, Phase 3, Part A): where label
specifications live, append-only, keyed by their own content-addressed
`label_specification_id` -- mirrors `features.registry.FeatureRegistry`'s
append-only discipline exactly, one layer up. Registering the same id
twice is refused (`DuplicateLabelSpecificationError`); a genuine
parameter/recipe change always produces a NEW id (via `models.
build_label_specification`), never a mutation of an already-registered
one.

Responsibilities named by the governing specification: register, lookup,
freeze, version, compare, verify a specification; manifest integration;
dataset integration."""

from __future__ import annotations

from quant_platform.core.exceptions import (
    DuplicateLabelSpecificationError,
    LabelIdentityError,
    UnknownLabelSpecificationError,
)
from quant_platform.labels.manifest import LabelManifest, build_label_manifest
from quant_platform.labels.models import LabelFamily, LabelSpecification
from quant_platform.labels.versioning import LabelVersion, LabelVersionHistory
from quant_platform.ml.persistence import format_utc_timestamp, utc_now

__all__ = ["LabelRegistry"]


class LabelRegistry:
    def __init__(self) -> None:
        self._specifications: dict[str, LabelSpecification] = {}
        self._registered_at: dict[str, str] = {}
        self._frozen: set[str] = set()

    def register(self, specification: LabelSpecification, *, registered_at: str | None = None) -> None:
        consistent, issues = specification.verify_self_consistency()
        if not consistent:
            raise LabelIdentityError(
                f"Refusing to register a self-inconsistent LabelSpecification: {issues}",
                context={"label_specification_id": specification.label_specification_id, "issues": list(issues)},
            )
        if specification.label_specification_id in self._specifications:
            raise DuplicateLabelSpecificationError(
                f"LabelSpecification {specification.label_specification_id!r} is already registered",
                context={"label_specification_id": specification.label_specification_id},
            )
        self._specifications[specification.label_specification_id] = specification
        self._registered_at[specification.label_specification_id] = registered_at or format_utc_timestamp(utc_now())

    def lookup(self, label_specification_id: str) -> LabelSpecification:
        try:
            return self._specifications[label_specification_id]
        except KeyError:
            raise UnknownLabelSpecificationError(
                f"No LabelSpecification registered with id {label_specification_id!r}",
                context={"label_specification_id": label_specification_id},
            ) from None

    def freeze(self, label_specification_id: str) -> None:
        self.lookup(label_specification_id)  # raises UnknownLabelSpecificationError if absent
        self._frozen.add(label_specification_id)

    def is_frozen(self, label_specification_id: str) -> bool:
        return label_specification_id in self._frozen

    def list_specifications(self, label_family: LabelFamily | None = None) -> tuple[LabelSpecification, ...]:
        specs = [s for s in self._specifications.values() if label_family is None or s.label_family is label_family]
        return tuple(sorted(specs, key=lambda s: (s.label_family.value, s.generation_version, s.parameter_hash)))

    def versions(self, label_family: LabelFamily) -> LabelVersionHistory:
        matching = self.list_specifications(label_family)
        versions = tuple(
            LabelVersion(
                generation_version=s.generation_version, label_specification_id=s.label_specification_id, parameter_hash=s.parameter_hash,
                registered_at=self._registered_at[s.label_specification_id],
            )
            for s in matching
        )
        return LabelVersionHistory(schema_version=1, label_family=label_family.value, versions=versions)

    def compare(self, label_specification_id_a: str, label_specification_id_b: str) -> tuple[bool, tuple[str, ...]]:
        """A field-by-field diff of two registered specifications --
        "compare specification" per the governing spec. Bundle-level
        (identity/manifest/lineage) drift is `reconciliation.
        LabelReconciliation`'s job; this is purely a spec-vs-spec
        metadata diff."""
        spec_a = self.lookup(label_specification_id_a)
        spec_b = self.lookup(label_specification_id_b)
        if spec_a == spec_b:
            return True, ()
        differences: list[str] = []
        for field_name in (
            "label_family", "generation_version", "parameter_hash", "price_basis", "prediction_horizon", "availability_rule",
            "reference_price", "event_time_rule", "generation_rule", "identity_algorithm", "created_from_dataset",
            "created_from_manifest", "parameters",
        ):
            value_a, value_b = getattr(spec_a, field_name), getattr(spec_b, field_name)
            if value_a != value_b:
                differences.append(f"{field_name}: {value_a!r} != {value_b!r}")
        return (not differences, tuple(differences))

    def verify(self, label_specification_id: str) -> tuple[bool, tuple[str, ...]]:
        return self.lookup(label_specification_id).verify_self_consistency()

    def build_manifest(
        self, label_specification_id: str, *, feature_identity: str | None = None, qualification_identity: str | None = None,
        generation_timestamp: str | None = None,
    ) -> LabelManifest:
        specification = self.lookup(label_specification_id)
        return build_label_manifest(
            specification, generation_timestamp=(generation_timestamp or format_utc_timestamp(utc_now())),
            feature_identity=feature_identity, qualification_identity=qualification_identity,
        )

    def for_dataset(self, dataset_id: str) -> tuple[LabelSpecification, ...]:
        matching = [s for s in self._specifications.values() if s.created_from_dataset == dataset_id]
        return tuple(sorted(matching, key=lambda s: (s.label_family.value, s.generation_version, s.parameter_hash)))

    def __len__(self) -> int:
        return len(self._specifications)

    def __contains__(self, label_specification_id: object) -> bool:
        return isinstance(label_specification_id, str) and label_specification_id in self._specifications
