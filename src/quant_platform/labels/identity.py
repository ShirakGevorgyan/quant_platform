"""`LabelIdentity` (Milestone 11, Phase 3, Part A): the content-addressed
identity of a GENERATED `builder.LabelBundle` -- distinct from
`models.LabelSpecification.label_specification_id`, which identifies the
RECIPE, not the OUTPUT. Two bundles built from the identical
specification and identical source data always produce an identical
`LabelIdentity`; changing either the recipe or the underlying data always
changes it.

Immutable, content-addressed, deterministic, portable: `compute_label_identity`
depends only on its own arguments -- never the filesystem, the wall
clock, a process id, `PYTHONHASHSEED`, `random`, a temp directory, the
machine, or the operating system, exactly as the governing specification
requires. NaN is a legitimate, expected label value (an unresolved
row near the embargo edge) -- encoded as JSON `null` rather than
rejected outright, unlike `models.compute_label_specification_id`'s
metadata fields, which are never legitimately non-finite."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from quant_platform.ml.persistence import require_schema_version

__all__ = ["LABEL_IDENTITY_SCHEMA_VERSION", "LabelIdentity", "compute_label_identity"]

LABEL_IDENTITY_SCHEMA_VERSION = 1


def _canonical_values_json(values: pd.Series) -> str:
    encoded = [None if pd.isna(v) else float(v) for v in values.to_numpy()]
    return json.dumps(encoded, allow_nan=False)


@dataclass(frozen=True, slots=True)
class LabelIdentity:
    schema_version: int
    label_specification_id: str
    content_id: str
    source_content_id: str
    row_count: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id,
            "content_id": self.content_id, "source_content_id": self.source_content_id, "row_count": self.row_count,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelIdentity:
        require_schema_version(raw, supported=LABEL_IDENTITY_SCHEMA_VERSION, context="LabelIdentity")
        return cls(
            schema_version=LABEL_IDENTITY_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            content_id=str(raw["content_id"]), source_content_id=str(raw["source_content_id"]), row_count=int(str(raw["row_count"])),
        )


def compute_label_identity(label_specification_id: str, values: pd.Series, *, source_content_id: str) -> LabelIdentity:
    """`content_id` is a sha256 over `(label_specification_id,
    source_content_id, row_count, the values themselves in row order)` --
    row ORDER matters (this is a time series, never a set), NaN encodes
    as JSON `null` rather than being dropped or rejected, and nothing
    about the computation reads the clock, a path, or process state."""
    row_count = len(values)
    payload = json.dumps(
        {"label_specification_id": label_specification_id, "source_content_id": source_content_id, "row_count": row_count},
        sort_keys=True,
    )
    combined = payload + "|" + _canonical_values_json(values)
    content_id = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return LabelIdentity(
        schema_version=LABEL_IDENTITY_SCHEMA_VERSION, label_specification_id=label_specification_id, content_id=content_id,
        source_content_id=source_content_id, row_count=row_count,
    )
