from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.label_validation.engine import (
    LabelQualificationDecision,
    LabelQualificationEngine,
    LabelQualificationReport,
)
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.manifest import LabelManifest


class TestLabelQualificationEngine:
    def test_healthy_bundle_is_never_rejected(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        # A genuinely untampered, freshly generated bundle over a modest
        # (300-row) synthetic sample can legitimately show WARNING-level
        # noise in one dimension (e.g. temporal stability) without being
        # scientifically unsound -- CONDITIONALLY_APPROVED is a normal,
        # honest outcome here, never REJECTED, and never a blocking reason.
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        assert report.decision in (LabelQualificationDecision.APPROVED, LabelQualificationDecision.CONDITIONALLY_APPROVED)
        assert report.blocking_reasons == ()

    def test_constant_bundle_is_rejected_with_explicit_blocking_reasons(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest, rebuild_bundle_with_values_fn) -> None:
        constant_bundle = rebuild_bundle_with_values_fn(next_return_bundle, pd.Series(np.full(next_return_bundle.row_count, 0.5)))
        report = LabelQualificationEngine().qualify(constant_bundle, next_return_manifest)
        assert report.decision is LabelQualificationDecision.REJECTED
        assert len(report.blocking_reasons) == 1
        assert report.blocking_reasons[0].startswith("[constant_labels]")

    def test_extreme_imbalance_produces_conditionally_approved(self, direction_bundle: LabelBundle, direction_manifest: LabelManifest, rebuild_bundle_with_values_fn) -> None:
        n = direction_bundle.row_count
        skewed = np.full(n, 1.0)
        skewed[:5] = -1.0
        skewed_bundle = rebuild_bundle_with_values_fn(direction_bundle, pd.Series(skewed, index=direction_bundle.values.index))
        report = LabelQualificationEngine().qualify(skewed_bundle, direction_manifest)
        assert report.decision is LabelQualificationDecision.CONDITIONALLY_APPROVED
        assert report.blocking_reasons == ()

    def test_json_round_trip(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        restored = LabelQualificationReport.from_json_dict(report.to_json_dict())
        assert restored == report
