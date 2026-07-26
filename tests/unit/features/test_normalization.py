from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.exceptions import PreprocessingLeakageError
from quant_platform.features.normalization import (
    FittedTransform,
    TransformKind,
    TransformPipeline,
    apply_transform,
    fit_transform,
)


class TestStandardScale:
    def test_fit_apply_roundtrip_gives_zero_mean_unit_std(self) -> None:
        series = pd.Series(np.arange(100.0))
        fitted = fit_transform(series, kind=TransformKind.STANDARD_SCALE, feature_name="x")
        scaled = apply_transform(series, fitted)
        assert scaled.mean() == pytest.approx(0.0, abs=1e-9)
        assert scaled.std() == pytest.approx(1.0, rel=1e-6)

    def test_zero_variance_column_does_not_divide_by_zero(self) -> None:
        series = pd.Series([5.0, 5.0, 5.0])
        fitted = fit_transform(series, kind=TransformKind.STANDARD_SCALE, feature_name="x")
        scaled = apply_transform(series, fitted)
        assert not scaled.isna().any()
        assert not np.isinf(scaled.to_numpy()).any()


class TestRobustScale:
    def test_fit_apply(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])  # one extreme outlier
        fitted = fit_transform(series, kind=TransformKind.ROBUST_SCALE, feature_name="x")
        scaled = apply_transform(series, fitted)
        # median-centered: the median row should map close to 0
        median_row = series.sort_values().index[len(series) // 2]
        assert abs(scaled.iloc[median_row]) < 1.0


class TestWinsorize:
    def test_clips_extreme_values(self) -> None:
        series = pd.Series(list(range(100)))
        fitted = fit_transform(series, kind=TransformKind.WINSORIZE, feature_name="x", clip_quantiles=(0.05, 0.95))
        scaled = apply_transform(series, fitted)
        assert scaled.max() <= fitted.params["upper"]
        assert scaled.min() >= fitted.params["lower"]


class TestSignedLog1p:
    def test_handles_negative_values(self) -> None:
        series = pd.Series([-100.0, -1.0, 0.0, 1.0, 100.0])
        fitted = fit_transform(series, kind=TransformKind.SIGNED_LOG1P, feature_name="x")
        scaled = apply_transform(series, fitted)
        assert scaled.iloc[0] < 0
        assert scaled.iloc[4] > 0
        assert not scaled.isna().any()


class TestFittedTransformSerialization:
    def test_json_round_trip(self) -> None:
        fitted = fit_transform(pd.Series([1.0, 2.0, 3.0]), kind=TransformKind.STANDARD_SCALE, feature_name="x")
        restored = FittedTransform.from_json_dict(fitted.to_json_dict())
        assert restored == fitted


class TestTransformPipeline:
    def test_fit_then_apply(self) -> None:
        train_df = pd.DataFrame({"x": np.arange(100.0)})
        pipeline = TransformPipeline()
        pipeline.fit(train_df, specs={"x": TransformKind.STANDARD_SCALE})
        applied = pipeline.apply(train_df)
        assert applied["x"].mean() == pytest.approx(0.0, abs=1e-9)

    def test_apply_before_fit_raises(self) -> None:
        pipeline = TransformPipeline()
        with pytest.raises(PreprocessingLeakageError):
            pipeline.apply(pd.DataFrame({"x": [1.0]}))

    def test_second_fit_without_allow_refit_raises(self) -> None:
        pipeline = TransformPipeline()
        pipeline.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), specs={"x": TransformKind.STANDARD_SCALE})
        with pytest.raises(PreprocessingLeakageError):
            pipeline.fit(pd.DataFrame({"x": [4.0, 5.0, 6.0]}), specs={"x": TransformKind.STANDARD_SCALE})

    def test_second_fit_with_allow_refit_succeeds(self) -> None:
        pipeline = TransformPipeline()
        pipeline.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), specs={"x": TransformKind.STANDARD_SCALE})
        pipeline.fit(pd.DataFrame({"x": [4.0, 5.0, 6.0]}), specs={"x": TransformKind.STANDARD_SCALE}, allow_refit=True)
        assert pipeline.is_fitted

    def test_fitted_parameters_never_reflect_validation_data(self) -> None:
        """The central adversarial proof for Section 17 item 4: fit on
        train (mean=1), apply to validation data with a WILDLY different
        distribution (mean=10000) -- the validation data's own statistics
        must never appear in the transform output; the frozen train-derived
        mean/std are used as-is."""
        train_df = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0]})
        validation_df = pd.DataFrame({"x": [10000.0, 20000.0, 30000.0]})
        pipeline = TransformPipeline()
        pipeline.fit(train_df, specs={"x": TransformKind.STANDARD_SCALE})
        applied_validation = pipeline.apply(validation_df)
        # If validation stats had leaked in, scaled values would center near 0;
        # instead they must be huge, since they are scaled by TRAIN's tiny std.
        assert applied_validation["x"].abs().min() > 1000

    def test_fingerprint_deterministic_and_sensitive_to_data(self) -> None:
        pipeline_a = TransformPipeline()
        pipeline_a.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), specs={"x": TransformKind.STANDARD_SCALE})
        pipeline_b = TransformPipeline()
        pipeline_b.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), specs={"x": TransformKind.STANDARD_SCALE})
        pipeline_c = TransformPipeline()
        pipeline_c.fit(pd.DataFrame({"x": [10.0, 20.0, 30.0]}), specs={"x": TransformKind.STANDARD_SCALE})
        assert pipeline_a.fingerprint() == pipeline_b.fingerprint()
        assert pipeline_a.fingerprint() != pipeline_c.fingerprint()

    def test_json_round_trip(self) -> None:
        pipeline = TransformPipeline()
        pipeline.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), specs={"x": TransformKind.STANDARD_SCALE})
        restored = TransformPipeline.from_json_dict(pipeline.to_json_dict())
        assert restored.fingerprint() == pipeline.fingerprint()
        assert restored.is_fitted
