"""Point-in-time replay proof (Milestone 11, Phase 3, Part C): "destroy
labels, replay labels, qualification identical." Regenerates a bundle
from scratch via `labels.builder.LabelBuilder` (never reusing the
original, "destroyed" bundle object -- only its `LabelDefinition` and
the immutable `source_data` it came from), re-qualifies the freshly
regenerated bundle, and proves the QUALIFICATION DECISION and score are
byte-identical to the original report's -- the label-validation-level
analogue of `labels.replay.LabelReplay`'s own bundle-level proof, one
layer up."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.core.exceptions import LabelError, LabelValidationReplayError
from quant_platform.label_validation.engine import LabelQualificationEngine, LabelQualificationReport
from quant_platform.labels.builder import LabelBuilder, LabelBundle, LabelDefinition
from quant_platform.labels.manifest import LabelManifest
from quant_platform.ml.persistence import as_json_list, format_utc_timestamp, require_schema_version, utc_now

__all__ = ["LABEL_VALIDATION_REPLAY_SCHEMA_VERSION", "LabelValidationReplay", "LabelValidationReplayResult"]

LABEL_VALIDATION_REPLAY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LabelValidationReplayResult:
    schema_version: int
    label_specification_id: str
    qualification_identical: bool
    issues: tuple[str, ...]
    original_decision: str
    replayed_decision: str
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id,
            "qualification_identical": self.qualification_identical, "issues": list(self.issues),
            "original_decision": self.original_decision, "replayed_decision": self.replayed_decision, "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelValidationReplayResult:
        require_schema_version(raw, supported=LABEL_VALIDATION_REPLAY_SCHEMA_VERSION, context="LabelValidationReplayResult")
        return cls(
            schema_version=LABEL_VALIDATION_REPLAY_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            qualification_identical=bool(raw["qualification_identical"]),
            issues=tuple(str(s) for s in as_json_list(raw.get("issues") or [], field_name="issues")),
            original_decision=str(raw["original_decision"]), replayed_decision=str(raw["replayed_decision"]), generated_at=str(raw["generated_at"]),
        )


class LabelValidationReplay:
    def replay_and_requalify(
        self, definition: LabelDefinition, source_data: pd.DataFrame, *, source_content_id: str, manifest: LabelManifest,
        original_report: LabelQualificationReport, drift_baseline: LabelBundle | None = None, regime_assignment: pd.Series | None = None,
    ) -> LabelValidationReplayResult:
        if definition.label_specification_id != original_report.label_specification_id:
            raise LabelValidationReplayError(
                f"Cannot replay: definition's label_specification_id={definition.label_specification_id!r} does not match "
                f"original_report.label_specification_id={original_report.label_specification_id!r}",
                context={"label_specification_id": definition.label_specification_id},
            )

        try:
            fresh_bundle = LabelBuilder().build(definition, source_data, source_content_id=source_content_id)
            fresh_report = LabelQualificationEngine().qualify(
                fresh_bundle, manifest, drift_baseline=drift_baseline, regime_assignment=regime_assignment,
            )
        except LabelError as exc:
            raise LabelValidationReplayError(
                f"Could not attempt replay-and-requalify for label_specification_id={definition.label_specification_id!r}: {exc}",
                context={"label_specification_id": definition.label_specification_id},
            ) from exc

        issues: list[str] = []
        if fresh_report.decision != original_report.decision:
            issues.append(f"decision: original={original_report.decision.value!r} replayed={fresh_report.decision.value!r}")
        score_delta = abs(fresh_report.diagnostics.overall_score - original_report.diagnostics.overall_score)
        if score_delta > 1e-9:
            issues.append(f"overall_score: original={original_report.diagnostics.overall_score!r} replayed={fresh_report.diagnostics.overall_score!r}")

        return LabelValidationReplayResult(
            schema_version=LABEL_VALIDATION_REPLAY_SCHEMA_VERSION, label_specification_id=definition.label_specification_id,
            qualification_identical=not issues, issues=tuple(issues), original_decision=original_report.decision.value,
            replayed_decision=fresh_report.decision.value, generated_at=format_utc_timestamp(utc_now()),
        )
