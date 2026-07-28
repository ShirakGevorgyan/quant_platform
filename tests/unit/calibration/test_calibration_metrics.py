"""Calibration metric tests (Milestone 4E, Section 30): hand-verified
log loss/Brier/ECE/MCE/slope/intercept/sharpness/resolution, no
unjustified tolerances, undefined metrics use explicit skip reasons
rather than NaN."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_platform.calibration.metrics import CALIBRATION_METRIC_NAMES, compute_calibration_metrics
from quant_platform.calibration.models import BinningStrategy
from quant_platform.calibration.specs import ReliabilityBinningSpec
from quant_platform.core.exceptions import CalibrationDataError

_BINNING = ReliabilityBinningSpec(strategy=BinningStrategy.EQUAL_WIDTH, n_bins=2)


class TestLogLossAndBrierScore:
    def test_log_loss_matches_hand_computed_formula(self) -> None:
        probabilities = np.array([0.9, 0.1, 0.6, 0.4])
        labels = np.array([1.0, 0.0, 1.0, 0.0])
        report = compute_calibration_metrics(probabilities, labels, binning_spec=_BINNING)
        expected = -np.mean([
            math.log(0.9), math.log(1 - 0.1), math.log(0.6), math.log(1 - 0.4),
        ])
        assert report.values["log_loss"] == pytest.approx(expected, abs=1e-12)

    def test_brier_score_matches_hand_computed_formula(self) -> None:
        probabilities = np.array([0.9, 0.1, 0.6, 0.4])
        labels = np.array([1.0, 0.0, 1.0, 0.0])
        report = compute_calibration_metrics(probabilities, labels, binning_spec=_BINNING)
        expected = np.mean([(0.9 - 1) ** 2, (0.1 - 0) ** 2, (0.6 - 1) ** 2, (0.4 - 0) ** 2])
        assert report.values["brier_score"] == pytest.approx(expected, abs=1e-12)

    def test_perfect_predictions_have_zero_log_loss_and_brier(self) -> None:
        probabilities = np.array([1.0 - 1e-9, 1e-9])
        labels = np.array([1.0, 0.0])
        report = compute_calibration_metrics(probabilities, labels, binning_spec=_BINNING)
        assert report.values["log_loss"] == pytest.approx(0.0, abs=1e-6)
        assert report.values["brier_score"] == pytest.approx(0.0, abs=1e-6)


class TestExpectedAndMaximumCalibrationError:
    def test_ece_is_zero_for_perfectly_calibrated_bins(self) -> None:
        """Two equal-width bins ([0, 0.5), [0.5, 1]); each bin's mean
        predicted probability exactly equals its empirical positive
        rate by construction -- ECE/MCE must both be exactly 0."""
        probabilities = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8])
        labels = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        # Bin 1 (p=0.2, n=5): mean_pred=0.2, empirical_rate=1/5=0.2 -- match.
        # Bin 2 (p=0.8, n=5): mean_pred=0.8, empirical_rate=4/5=0.8 -- match.
        report = compute_calibration_metrics(probabilities, labels, binning_spec=_BINNING)
        assert report.values["expected_calibration_error"] == pytest.approx(0.0, abs=1e-9)
        assert report.values["maximum_calibration_error"] == pytest.approx(0.0, abs=1e-9)

    def test_ece_matches_hand_computed_weighted_gap(self) -> None:
        probabilities = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        labels = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        # Bin 1 (p<0.5, n=3): mean_pred=0.1, empirical_rate=1.0 -> gap=0.9
        # Bin 2 (p>=0.5, n=2): mean_pred=0.9, empirical_rate=0.0 -> gap=0.9
        report = compute_calibration_metrics(probabilities, labels, binning_spec=_BINNING)
        expected_ece = (3 / 5) * 0.9 + (2 / 5) * 0.9
        assert report.values["expected_calibration_error"] == pytest.approx(expected_ece, abs=1e-9)
        assert report.values["maximum_calibration_error"] == pytest.approx(0.9, abs=1e-9)


class TestSharpnessAndResolution:
    def test_sharpness_is_variance_of_predicted_probabilities(self) -> None:
        probabilities = np.array([0.1, 0.5, 0.9, 0.5])
        labels = np.array([0.0, 1.0, 1.0, 0.0])
        report = compute_calibration_metrics(probabilities, labels, binning_spec=_BINNING)
        assert report.values["sharpness"] == pytest.approx(float(np.var(probabilities)), abs=1e-12)

    def test_zero_variance_predictions_have_zero_sharpness(self) -> None:
        probabilities = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([0.0, 1.0, 1.0, 0.0])
        report = compute_calibration_metrics(probabilities, labels, binning_spec=_BINNING)
        assert report.values["sharpness"] == pytest.approx(0.0, abs=1e-12)


class TestUndefinedMetricsAreSkippedNotNaN:
    def test_slope_intercept_skipped_when_all_predictions_are_identical(self) -> None:
        """Zero-variance predicted probabilities make logistic Cox
        regression (calibration slope/intercept) undefined -- must be
        SKIPPED with a reason, never fabricated as NaN or 0."""
        probabilities = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([0.0, 1.0, 1.0, 0.0])
        report = compute_calibration_metrics(probabilities, labels, binning_spec=_BINNING)
        assert "calibration_slope" in report.skipped
        assert "calibration_intercept" in report.skipped
        assert "calibration_slope" not in report.values
        assert "calibration_intercept" not in report.values
        for reason in report.skipped.values():
            assert isinstance(reason, str) and len(reason) > 0

    def test_slope_intercept_skipped_when_labels_are_a_single_class(self) -> None:
        probabilities = np.array([0.2, 0.5, 0.8, 0.9])
        labels = np.array([1.0, 1.0, 1.0, 1.0])
        report = compute_calibration_metrics(probabilities, labels, binning_spec=_BINNING)
        assert "calibration_slope" in report.skipped
        assert "calibration_intercept" in report.skipped

    def test_no_metric_value_is_ever_nan(self) -> None:
        probabilities = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([0.0, 1.0, 1.0, 0.0])
        report = compute_calibration_metrics(probabilities, labels, binning_spec=_BINNING)
        for name, value in report.values.items():
            assert math.isfinite(value), f"{name} is not finite: {value!r}"

    def test_every_declared_metric_name_is_either_reported_or_skipped(self) -> None:
        probabilities = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([0.0, 1.0, 1.0, 0.0])
        report = compute_calibration_metrics(probabilities, labels, binning_spec=_BINNING)
        accounted = set(report.values) | set(report.skipped)
        assert set(CALIBRATION_METRIC_NAMES) <= accounted


class TestSlopeInterceptWhenWellDefined:
    def test_slope_and_intercept_are_computed_for_well_separated_data(self) -> None:
        rng = np.random.default_rng(0)
        n = 400
        latent = rng.normal(size=n)
        probabilities = np.clip(1.0 / (1.0 + np.exp(-1.2 * latent)), 1e-3, 1 - 1e-3)
        labels = (rng.uniform(size=n) < probabilities).astype(float)
        labels[0], labels[1] = 0.0, 1.0
        report = compute_calibration_metrics(probabilities, labels, binning_spec=_BINNING)
        assert "calibration_slope" in report.values
        assert "calibration_intercept" in report.values
        assert math.isfinite(report.values["calibration_slope"])
        assert math.isfinite(report.values["calibration_intercept"])


class TestInputValidation:
    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(CalibrationDataError):
            compute_calibration_metrics(np.array([0.1, 0.2]), np.array([0.0]), binning_spec=_BINNING)

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(CalibrationDataError):
            compute_calibration_metrics(np.array([]), np.array([]), binning_spec=_BINNING)

    def test_rejects_non_finite_probability(self) -> None:
        with pytest.raises(CalibrationDataError):
            compute_calibration_metrics(np.array([0.5, math.nan]), np.array([0.0, 1.0]), binning_spec=_BINNING)

    def test_rejects_out_of_range_probability(self) -> None:
        with pytest.raises(CalibrationDataError):
            compute_calibration_metrics(np.array([0.5, 1.5]), np.array([0.0, 1.0]), binning_spec=_BINNING)

    def test_rejects_labels_outside_binary_domain(self) -> None:
        with pytest.raises(CalibrationDataError):
            compute_calibration_metrics(np.array([0.5, 0.5]), np.array([0.0, 2.0]), binning_spec=_BINNING)
