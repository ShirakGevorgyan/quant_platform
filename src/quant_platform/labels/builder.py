"""`LabelDefinition`, `LabelBuilder`, `LabelBundle` (Milestone 11, Phase
3, Part A) -- the generic generation HARNESS, never a generation
ALGORITHM. Mirrors `features.interfaces.FeatureDefinition`/
`features.engine.FeatureEngine`'s own split exactly, one layer up:
`LabelDefinition` pairs an immutable `models.LabelSpecification` with a
pluggable `generate` callable; `LabelBuilder` calls it and wraps the
result in identity/bookkeeping. This module ships NO concrete `generate`
implementation for ANY of the 6 named families (Next Return, Multi
Horizon Return, Direction, Triple Barrier, Forward Volatility, Future
Extension Placeholder) -- the governing specification is explicit that
"actual labels" belong to a later phase. Every check `LabelBuilder`
performs on a generator's OUTPUT is purely structural (shape, dtype,
memory aliasing) -- it never evaluates whether the VALUES are
scientifically correct for the declared family, because Part A has no
family-specific logic to check them against.

THE MUTABLE-ALIAS GUARD
--------------------------------------------------------------------------
A generator that returns `source_data["close"]` unchanged (instead of a
freshly computed `pd.Series`) would silently corrupt an already-built
`LabelBundle` the moment `source_data` is later mutated -- violating this
package's "immutable scientific evidence" guarantee. Detected via
`numpy.shares_memory` against every column of `source_data` (never
Python `is` identity, which pandas' column-access machinery does not
reliably preserve across repeated `df[col]` calls)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.core.exceptions import LabelGenerationContractError, LabelMutableAliasError
from quant_platform.labels.identity import LabelIdentity, compute_label_identity
from quant_platform.labels.models import LabelSpecification
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)

__all__ = ["LABEL_BUNDLE_SCHEMA_VERSION", "LabelBuilder", "LabelBundle", "LabelDefinition", "LabelGeneratorFn"]

LABEL_BUNDLE_SCHEMA_VERSION = 1

LabelGeneratorFn = Callable[[pd.DataFrame, LabelSpecification], pd.Series]
"""The pluggable hook. A REAL generator (Part 2+) would read whichever
`source_data` columns its specification's `price_basis`/`reference_price`
describe and compute forward-looking values from them; Part A ships none.
Test-only generators used in this package's own test suite are
deliberately trivial/synthetic (e.g. a fixed marker value), never a
recognizable financial return/barrier/volatility formula, so nothing in
this phase could be mistaken for a real label family's implementation."""


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    specification: LabelSpecification
    generate: LabelGeneratorFn

    @property
    def label_specification_id(self) -> str:
        return self.specification.label_specification_id


@dataclass(frozen=True, slots=True, eq=False)
class LabelBundle:
    """`eq=False`: the dataclass-generated `__eq__` would compare
    `values` (a `pd.Series`) with bare `==`, which returns an
    element-wise Series rather than a bool and raises `ValueError` the
    moment it is coerced to one (e.g. inside another dataclass's own
    generated `__eq__`, as `recovery.LabelRecoveryResult`'s
    `recovered_bundle: LabelBundle | None` field does). `__eq__` below
    uses `Series.equals` instead, which is exactly the "same values,
    same positions, NaN-aware" comparison this package actually means.
    Explicitly unhashable (`__hash__ = None`) rather than silently
    inheriting an identity-based hash inconsistent with this `__eq__`."""

    schema_version: int
    specification: LabelSpecification
    identity: LabelIdentity
    values: pd.Series
    row_count: int
    valid_count: int
    generated_at: str

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LabelBundle):
            return NotImplemented
        return (
            self.schema_version == other.schema_version and self.specification == other.specification and self.identity == other.identity
            and self.row_count == other.row_count and self.valid_count == other.valid_count and self.generated_at == other.generated_at
            and self.values.equals(other.values)
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "specification": self.specification.to_json_dict(),
            "identity": self.identity.to_json_dict(), "values": [None if pd.isna(v) else float(v) for v in self.values.to_numpy()],
            "row_count": self.row_count, "valid_count": self.valid_count, "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelBundle:
        require_schema_version(raw, supported=LABEL_BUNDLE_SCHEMA_VERSION, context="LabelBundle")
        values_raw = as_json_list(raw.get("values") or [], field_name="values")
        values = pd.Series([np.nan if v is None else float(str(v)) for v in values_raw], dtype="float64")
        return cls(
            schema_version=LABEL_BUNDLE_SCHEMA_VERSION, specification=LabelSpecification.from_json_dict(as_json_dict(raw["specification"], field_name="specification")),
            identity=LabelIdentity.from_json_dict(as_json_dict(raw["identity"], field_name="identity")), values=values,
            row_count=int(str(raw["row_count"])), valid_count=int(str(raw["valid_count"])), generated_at=str(raw["generated_at"]),
        )


def _assert_no_mutable_alias(values: pd.Series, source_data: pd.DataFrame, *, label_specification_id: str) -> None:
    values_array = values.to_numpy(copy=False)
    for column in source_data.columns:
        column_array = source_data[column].to_numpy(copy=False)
        if np.shares_memory(values_array, column_array):
            raise LabelMutableAliasError(
                f"Generator for label_specification_id={label_specification_id!r} returned a Series that shares "
                f"underlying memory with source_data column {column!r} -- return a freshly computed Series, never a "
                "view/reference into the source data.",
                context={"label_specification_id": label_specification_id, "aliased_column": column},
            )


class LabelBuilder:
    def build(self, definition: LabelDefinition, source_data: pd.DataFrame, *, source_content_id: str) -> LabelBundle:
        values = definition.generate(source_data, definition.specification)

        if not isinstance(values, pd.Series):
            raise LabelGenerationContractError(
                f"Generator for label_specification_id={definition.label_specification_id!r} must return a pandas Series, "
                f"got {type(values).__name__}",
                context={"label_specification_id": definition.label_specification_id},
            )
        if len(values) != len(source_data):
            raise LabelGenerationContractError(
                f"Generator for label_specification_id={definition.label_specification_id!r} returned {len(values)} "
                f"values for {len(source_data)} source rows -- lengths must match exactly.",
                context={"label_specification_id": definition.label_specification_id, "expected": len(source_data), "actual": len(values)},
            )
        if not pd.api.types.is_numeric_dtype(values):
            raise LabelGenerationContractError(
                f"Generator for label_specification_id={definition.label_specification_id!r} returned dtype "
                f"{values.dtype}, which is not numeric.",
                context={"label_specification_id": definition.label_specification_id, "dtype": str(values.dtype)},
            )
        _assert_no_mutable_alias(values, source_data, label_specification_id=definition.label_specification_id)

        identity = compute_label_identity(definition.label_specification_id, values, source_content_id=source_content_id)
        row_count = len(values)
        valid_count = int(values.notna().sum())
        return LabelBundle(
            schema_version=LABEL_BUNDLE_SCHEMA_VERSION, specification=definition.specification, identity=identity, values=values,
            row_count=row_count, valid_count=valid_count, generated_at=format_utc_timestamp(utc_now()),
        )
