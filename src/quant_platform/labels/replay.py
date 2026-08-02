"""`LabelReplay` (Milestone 11, Phase 3, Part A): the INVARIANTS section
of the governing specification, promoted to a first-class module --
"changing nothing -> same labels, same hashes, same manifests, same
reports." Regenerates a bundle from scratch via a fresh `builder.
LabelBuilder` call and proves byte-identical reproduction against an
already-built `original` -- the label-package analogue of `qualification`'s
replay-invariance proof."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.core.exceptions import LabelError, LabelReplayError
from quant_platform.labels.builder import LabelBuilder, LabelBundle, LabelDefinition
from quant_platform.ml.persistence import as_json_list, format_utc_timestamp, require_schema_version, utc_now

__all__ = ["LABEL_REPLAY_SCHEMA_VERSION", "LabelReplay", "LabelReplayResult"]

LABEL_REPLAY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LabelReplayResult:
    schema_version: int
    label_specification_id: str
    replayed: bool
    issues: tuple[str, ...]
    original_content_id: str
    replayed_content_id: str
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id, "replayed": self.replayed,
            "issues": list(self.issues), "original_content_id": self.original_content_id, "replayed_content_id": self.replayed_content_id,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelReplayResult:
        require_schema_version(raw, supported=LABEL_REPLAY_SCHEMA_VERSION, context="LabelReplayResult")
        return cls(
            schema_version=LABEL_REPLAY_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]), replayed=bool(raw["replayed"]),
            issues=tuple(str(s) for s in as_json_list(raw.get("issues") or [], field_name="issues")), original_content_id=str(raw["original_content_id"]),
            replayed_content_id=str(raw["replayed_content_id"]), generated_at=str(raw["generated_at"]),
        )


class LabelReplay:
    def replay(self, definition: LabelDefinition, source_data: pd.DataFrame, *, source_content_id: str, original: LabelBundle) -> LabelReplayResult:
        if definition.label_specification_id != original.specification.label_specification_id:
            raise LabelReplayError(
                f"Cannot replay: definition's label_specification_id={definition.label_specification_id!r} does not match "
                f"original.specification.label_specification_id={original.specification.label_specification_id!r}",
                context={"label_specification_id": definition.label_specification_id},
            )

        try:
            fresh = LabelBuilder().build(definition, source_data, source_content_id=source_content_id)
        except LabelError as exc:
            raise LabelReplayError(
                f"Could not attempt replay for label_specification_id={definition.label_specification_id!r}: {exc}",
                context={"label_specification_id": definition.label_specification_id},
            ) from exc

        issues: list[str] = []
        if fresh.identity.content_id != original.identity.content_id:
            issues.append(f"content_id: original={original.identity.content_id!r} replayed={fresh.identity.content_id!r}")
        if fresh.row_count != original.row_count:
            issues.append(f"row_count: original={original.row_count} replayed={fresh.row_count}")
        if fresh.valid_count != original.valid_count:
            issues.append(f"valid_count: original={original.valid_count} replayed={fresh.valid_count}")

        return LabelReplayResult(
            schema_version=LABEL_REPLAY_SCHEMA_VERSION, label_specification_id=definition.label_specification_id, replayed=not issues,
            issues=tuple(issues), original_content_id=original.identity.content_id, replayed_content_id=fresh.identity.content_id,
            generated_at=format_utc_timestamp(utc_now()),
        )
