from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelValidationError
from quant_platform.label_validation.diagnostics import LabelDiagnostics, compute_label_diagnostics
from quant_platform.label_validation.evidence import (
    LABEL_VALIDATION_DIMENSION_ORDER,
    LabelValidationDimensionKind,
)
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.manifest import LabelManifest


class TestComputeLabelDiagnostics:
    def test_covers_all_eight_dimensions_in_fixed_order(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        diagnostics = compute_label_diagnostics(next_return_bundle, next_return_manifest)
        assert tuple(r.dimension for r in diagnostics.dimension_results) == LABEL_VALIDATION_DIMENSION_ORDER

    def test_healthy_bundle_scores_high(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        diagnostics = compute_label_diagnostics(next_return_bundle, next_return_manifest)
        assert diagnostics.overall_score > 0.8
        assert diagnostics.is_blocking is False

    def test_drift_dimension_none_without_baseline(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        diagnostics = compute_label_diagnostics(next_return_bundle, next_return_manifest)
        assert diagnostics.drift is None
        assert diagnostics.dimension_result(LabelValidationDimensionKind.DRIFT).score == 1.0

    def test_drift_dimension_populated_with_baseline(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        diagnostics = compute_label_diagnostics(next_return_bundle, next_return_manifest, drift_baseline=next_return_bundle)
        assert diagnostics.drift is not None

    def test_constant_bundle_forces_zero_degeneracy_score_and_blocking(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        constant_bundle = replace(next_return_bundle, values=pd.Series(np.full(next_return_bundle.row_count, 0.5)))
        diagnostics = compute_label_diagnostics(constant_bundle, next_return_manifest)
        assert diagnostics.dimension_result(LabelValidationDimensionKind.DEGENERACY).score == 0.0
        assert diagnostics.is_blocking is True

    def test_incomplete_dimension_set_rejected(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        diagnostics = compute_label_diagnostics(next_return_bundle, next_return_manifest)
        with pytest.raises(LabelValidationError):
            LabelDiagnostics(
                schema_version=diagnostics.schema_version, label_specification_id=diagnostics.label_specification_id,
                dimension_results=diagnostics.dimension_results[:-1], overall_score=diagnostics.overall_score, statistics=diagnostics.statistics,
                distribution=diagnostics.distribution, balance=diagnostics.balance, degeneracy=diagnostics.degeneracy,
                coverage=diagnostics.coverage, stability=diagnostics.stability, drift=diagnostics.drift, leakage=diagnostics.leakage,
            )

    def test_json_round_trip(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        diagnostics = compute_label_diagnostics(next_return_bundle, next_return_manifest, drift_baseline=next_return_bundle)
        restored = LabelDiagnostics.from_json_dict(diagnostics.to_json_dict())
        assert restored == diagnostics
