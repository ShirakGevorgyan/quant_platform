"""`LabelRecovery` (Milestone 11, Phase 3, Part A): recovers a lost or
corrupted `builder.LabelBundle` by replaying it from the ORIGINAL
`models.LabelSpecification`, generator, and source data (via `replay.
LabelReplay`) -- never by guessing at a plausible value. Mirrors
`PortfolioRiskRecoveryError`'s own "surfaced rather than guessed"
discipline: if the evidence needed to replay is unavailable, or a
regenerated bundle does not match a supplied `expected_identity`,
recovery FAILS CLOSED (`recoverable=False`, `recovered_bundle=None`)
rather than returning a best-effort result."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.core.exceptions import LabelError, LabelRecoveryError
from quant_platform.labels.builder import LabelBuilder, LabelBundle, LabelDefinition
from quant_platform.labels.identity import LabelIdentity
from quant_platform.labels.models import LabelSpecification
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)

__all__ = ["LABEL_RECOVERY_SCHEMA_VERSION", "LabelRecovery", "LabelRecoveryResult"]

LABEL_RECOVERY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LabelRecoveryResult:
    schema_version: int
    label_specification_id: str
    recoverable: bool
    recovered_bundle: LabelBundle | None
    issues: tuple[str, ...]
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id, "recoverable": self.recoverable,
            "recovered_bundle": (None if self.recovered_bundle is None else self.recovered_bundle.to_json_dict()),
            "issues": list(self.issues), "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelRecoveryResult:
        require_schema_version(raw, supported=LABEL_RECOVERY_SCHEMA_VERSION, context="LabelRecoveryResult")
        recovered_raw = raw.get("recovered_bundle")
        return cls(
            schema_version=LABEL_RECOVERY_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            recoverable=bool(raw["recoverable"]),
            recovered_bundle=(None if recovered_raw is None else LabelBundle.from_json_dict(as_json_dict(recovered_raw, field_name="recovered_bundle"))),
            issues=tuple(str(s) for s in as_json_list(raw.get("issues") or [], field_name="issues")), generated_at=str(raw["generated_at"]),
        )


class LabelRecovery:
    def recover(
        self, specification: LabelSpecification, definition: LabelDefinition | None, source_data: pd.DataFrame | None, *,
        source_content_id: str | None, expected_identity: LabelIdentity | None = None,
    ) -> LabelRecoveryResult:
        if definition is None or source_data is None or source_content_id is None:
            return LabelRecoveryResult(
                schema_version=LABEL_RECOVERY_SCHEMA_VERSION, label_specification_id=specification.label_specification_id, recoverable=False,
                recovered_bundle=None,
                issues=("no generator/source data/source_content_id supplied -- cannot recover without replaying from the original recipe",),
                generated_at=format_utc_timestamp(utc_now()),
            )

        if definition.label_specification_id != specification.label_specification_id:
            raise LabelRecoveryError(
                f"Cannot attempt recovery: definition's label_specification_id={definition.label_specification_id!r} does not match "
                f"specification.label_specification_id={specification.label_specification_id!r}",
                context={"label_specification_id": specification.label_specification_id},
            )

        try:
            candidate = LabelBuilder().build(definition, source_data, source_content_id=source_content_id)
        except LabelError as exc:
            raise LabelRecoveryError(
                f"Could not attempt recovery for label_specification_id={specification.label_specification_id!r}: {exc}",
                context={"label_specification_id": specification.label_specification_id},
            ) from exc

        if expected_identity is not None and candidate.identity.content_id != expected_identity.content_id:
            return LabelRecoveryResult(
                schema_version=LABEL_RECOVERY_SCHEMA_VERSION, label_specification_id=specification.label_specification_id, recoverable=False,
                recovered_bundle=None,
                issues=(f"replayed content_id={candidate.identity.content_id!r} does not match expected_identity.content_id={expected_identity.content_id!r}",),
                generated_at=format_utc_timestamp(utc_now()),
            )

        return LabelRecoveryResult(
            schema_version=LABEL_RECOVERY_SCHEMA_VERSION, label_specification_id=specification.label_specification_id, recoverable=True,
            recovered_bundle=candidate, issues=(), generated_at=format_utc_timestamp(utc_now()),
        )
