"""Consolidated robustness report (Milestone 6, Sections 14/15/21). The
`ROBUSTNESS_REPORT` artifact persisted at `RobustnessStage.COMPLETED` --
a lean INDEX bundling every stage's own artifact reference plus the
headline `PromotionDecisionKind`, never a re-statement of their contents.
Every underlying number lives in its own already-persisted report (source
verification, return series, bootstrap, downside, fold stability,
sensitivity, stress, regime, selection, promotion); the CLI's `report-
robustness`/`inspect-robustness` commands (Section 15) dereference these
references through `MLArtifactStore` for human-readable output rather
than this module duplicating their formatting."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.ml.models import ArtifactReference
from quant_platform.ml.persistence import as_json_dict, require_schema_version
from quant_platform.robustness.models import PromotionDecisionKind

ROBUSTNESS_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RobustnessReport:
    schema_version: int
    robustness_id: str
    source_backtest_id: str
    source_verification_reference: ArtifactReference
    return_series_reference: ArtifactReference
    bootstrap_report_reference: ArtifactReference
    downside_analysis_reference: ArtifactReference
    fold_stability_reference: ArtifactReference
    sensitivity_reference: ArtifactReference | None
    stress_reference: ArtifactReference
    regime_reference: ArtifactReference | None
    selection_reference: ArtifactReference
    promotion_decision_reference: ArtifactReference
    promotion_decision: PromotionDecisionKind
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        def _opt_ref(ref: ArtifactReference | None) -> object:
            return None if ref is None else ref.to_json_dict()

        return {
            "schema_version": self.schema_version, "robustness_id": self.robustness_id, "source_backtest_id": self.source_backtest_id,
            "source_verification_reference": self.source_verification_reference.to_json_dict(),
            "return_series_reference": self.return_series_reference.to_json_dict(),
            "bootstrap_report_reference": self.bootstrap_report_reference.to_json_dict(),
            "downside_analysis_reference": self.downside_analysis_reference.to_json_dict(),
            "fold_stability_reference": self.fold_stability_reference.to_json_dict(),
            "sensitivity_reference": _opt_ref(self.sensitivity_reference), "stress_reference": self.stress_reference.to_json_dict(),
            "regime_reference": _opt_ref(self.regime_reference), "selection_reference": self.selection_reference.to_json_dict(),
            "promotion_decision_reference": self.promotion_decision_reference.to_json_dict(), "promotion_decision": self.promotion_decision.value,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RobustnessReport:
        require_schema_version(raw, supported=ROBUSTNESS_REPORT_SCHEMA_VERSION, context="RobustnessReport")

        def _ref(key: str) -> ArtifactReference:
            return ArtifactReference.from_json_dict(as_json_dict(raw[key], field_name=key))

        def _opt_ref(key: str) -> ArtifactReference | None:
            value = raw.get(key)
            return None if value is None else ArtifactReference.from_json_dict(as_json_dict(value, field_name=key))

        return cls(
            schema_version=ROBUSTNESS_REPORT_SCHEMA_VERSION, robustness_id=str(raw["robustness_id"]), source_backtest_id=str(raw["source_backtest_id"]),
            source_verification_reference=_ref("source_verification_reference"), return_series_reference=_ref("return_series_reference"),
            bootstrap_report_reference=_ref("bootstrap_report_reference"), downside_analysis_reference=_ref("downside_analysis_reference"),
            fold_stability_reference=_ref("fold_stability_reference"), sensitivity_reference=_opt_ref("sensitivity_reference"),
            stress_reference=_ref("stress_reference"), regime_reference=_opt_ref("regime_reference"), selection_reference=_ref("selection_reference"),
            promotion_decision_reference=_ref("promotion_decision_reference"), promotion_decision=PromotionDecisionKind(raw["promotion_decision"]),
            generated_at=str(raw["generated_at"]),
        )


__all__ = ["ROBUSTNESS_REPORT_SCHEMA_VERSION", "RobustnessReport"]
