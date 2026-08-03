"""`LabelQualificationEngine` (Milestone 11, Phase 3, Part C): the
single entry point -- runs `diagnostics.compute_label_diagnostics` and
derives ONE `LabelQualificationDecision`. **The output is label
QUALITY, never a label value, never a model.**

Three-tier decision, deliberately the simplest rule that satisfies the
governing specification: any BLOCKING evidence anywhere -> `REJECTED`;
no blocking evidence but at least one WARNING/CRITICAL (non-blocking)
finding -> `CONDITIONALLY_APPROVED`; no blocking, no warning/critical
finding at all -> `APPROVED`. Blocking reasons are always explicit --
`blocking_reasons` cites every blocking evidence record's own
`blocking_code` and `finding` text verbatim, never a generic message."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from quant_platform.historical.quality import Severity
from quant_platform.label_validation.diagnostics import LabelDiagnostics, compute_label_diagnostics
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.manifest import LabelManifest
from quant_platform.labels.records import LabelRecord
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)

__all__ = ["LABEL_QUALIFICATION_REPORT_SCHEMA_VERSION", "LabelQualificationDecision", "LabelQualificationEngine", "LabelQualificationReport"]

LABEL_QUALIFICATION_REPORT_SCHEMA_VERSION = 1


class LabelQualificationDecision(Enum):
    APPROVED = "APPROVED"
    CONDITIONALLY_APPROVED = "CONDITIONALLY_APPROVED"
    REJECTED = "REJECTED"


def _decide(diagnostics: LabelDiagnostics) -> LabelQualificationDecision:
    if diagnostics.is_blocking:
        return LabelQualificationDecision.REJECTED
    if any(e.severity in (Severity.WARNING, Severity.CRITICAL) for e in diagnostics.all_evidence):
        return LabelQualificationDecision.CONDITIONALLY_APPROVED
    return LabelQualificationDecision.APPROVED


@dataclass(frozen=True, slots=True)
class LabelQualificationReport:
    schema_version: int
    label_specification_id: str
    decision: LabelQualificationDecision
    diagnostics: LabelDiagnostics
    blocking_reasons: tuple[str, ...]
    qualified_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "label_specification_id": self.label_specification_id, "decision": self.decision.value,
            "diagnostics": self.diagnostics.to_json_dict(), "blocking_reasons": list(self.blocking_reasons), "qualified_at": self.qualified_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> LabelQualificationReport:
        require_schema_version(raw, supported=LABEL_QUALIFICATION_REPORT_SCHEMA_VERSION, context="LabelQualificationReport")
        return cls(
            schema_version=LABEL_QUALIFICATION_REPORT_SCHEMA_VERSION, label_specification_id=str(raw["label_specification_id"]),
            decision=LabelQualificationDecision(raw["decision"]), diagnostics=LabelDiagnostics.from_json_dict(as_json_dict(raw["diagnostics"], field_name="diagnostics")),
            blocking_reasons=tuple(str(s) for s in as_json_list(raw.get("blocking_reasons") or [], field_name="blocking_reasons")), qualified_at=str(raw["qualified_at"]),
        )


class LabelQualificationEngine:
    def qualify(
        self, bundle: LabelBundle, manifest: LabelManifest, *, drift_baseline: LabelBundle | None = None,
        regime_assignment: pd.Series | None = None, records: tuple[LabelRecord, ...] | None = None,
    ) -> LabelQualificationReport:
        diagnostics = compute_label_diagnostics(bundle, manifest, drift_baseline=drift_baseline, regime_assignment=regime_assignment, records=records)
        decision = _decide(diagnostics)
        blocking_reasons = tuple(
            f"[{e.blocking_code.value}] {e.finding}" for e in diagnostics.blocking_evidence if e.blocking_code is not None
        )
        return LabelQualificationReport(
            schema_version=LABEL_QUALIFICATION_REPORT_SCHEMA_VERSION, label_specification_id=bundle.specification.label_specification_id,
            decision=decision, diagnostics=diagnostics, blocking_reasons=blocking_reasons, qualified_at=format_utc_timestamp(utc_now()),
        )
