"""Calibration method tests (Milestone 4E, Section 29): fit/transform
correctness (hand-verified against closed-form formulas, not just
"runs without error"), serialization round-trips, minimum-sample/class
requirements, malformed-input rejection, determinism, and tampering
rejection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_platform.calibration.methods import (
    BetaCalibrator,
    FittedBetaCalibrator,
    FittedIdentityCalibrator,
    FittedIsotonicCalibrator,
    FittedPlattCalibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    build_unfit_method,
    fitted_method_from_json_dict,
    method_complexity_rank,
)
from quant_platform.calibration.models import CalibrationMethodKind
from quant_platform.core.exceptions import CalibrationFitError, CalibrationValidationError


def _correlated_probabilities_and_labels(n: int = 200, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    probabilities = 1.0 / (1.0 + np.exp(-1.5 * latent))
    labels = (rng.uniform(size=n) < probabilities).astype(float)
    # Guarantee both classes present.
    labels[0], labels[1] = 0.0, 1.0
    return probabilities, labels


class TestIdentityCalibrator:
    def test_transform_is_the_identity_function(self) -> None:
        probabilities, labels = _correlated_probabilities_and_labels()
        fitted = IdentityCalibrator().fit(probabilities, labels)
        assert isinstance(fitted, FittedIdentityCalibrator)
        out = fitted.transform(probabilities)
        assert np.array_equal(out, probabilities)

    def test_identity_never_fails_regardless_of_input(self) -> None:
        # No minimum-sample/class requirement -- always the baseline.
        fitted = IdentityCalibrator().fit(np.array([0.5]), np.array([1.0]))
        assert fitted.transform(np.array([0.1, 0.9]))[0] == pytest.approx(0.1)

    def test_round_trip_serialization(self) -> None:
        probabilities, labels = _correlated_probabilities_and_labels()
        fitted = IdentityCalibrator().fit(probabilities, labels)
        restored = fitted_method_from_json_dict(fitted.to_json_dict())
        assert isinstance(restored, FittedIdentityCalibrator)
        assert np.array_equal(restored.transform(probabilities), fitted.transform(probabilities))


class TestPlattCalibrator:
    def test_transform_matches_hand_computed_sigmoid(self) -> None:
        probabilities, labels = _correlated_probabilities_and_labels()
        fitted = PlattCalibrator().fit(probabilities, labels)
        assert isinstance(fitted, FittedPlattCalibrator)
        out = fitted.transform(probabilities)

        logit = np.log(probabilities / (1.0 - probabilities))
        expected = 1.0 / (1.0 + np.exp(-(fitted.coefficient * logit + fitted.intercept)))
        np.testing.assert_allclose(out, expected, atol=1e-9)

    def test_output_is_monotonic_in_input_probability(self) -> None:
        """Platt scaling is a monotonic sigmoid transform of the logit --
        higher input probability must never produce a lower output."""
        probabilities, labels = _correlated_probabilities_and_labels()
        fitted = PlattCalibrator().fit(probabilities, labels)
        grid = np.linspace(0.01, 0.99, 50)
        out = fitted.transform(grid)
        assert np.all(np.diff(out) >= -1e-12)

    def test_round_trip_serialization_reproduces_identical_transform(self) -> None:
        probabilities, labels = _correlated_probabilities_and_labels()
        fitted = PlattCalibrator().fit(probabilities, labels)
        restored = fitted_method_from_json_dict(fitted.to_json_dict())
        assert isinstance(restored, FittedPlattCalibrator)
        np.testing.assert_array_equal(restored.transform(probabilities), fitted.transform(probabilities))

    def test_tampered_non_finite_coefficient_is_rejected(self) -> None:
        raw = {
            "schema_version": 1, "kind": "platt", "coefficient": float("nan"), "intercept": 0.0,
            "input_representation": "predict_proba",
        }
        with pytest.raises(CalibrationValidationError):
            fitted_method_from_json_dict(raw)

    def test_fit_requires_both_classes(self) -> None:
        with pytest.raises(CalibrationFitError):
            PlattCalibrator().fit(np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.0, 0.0]))


class TestIsotonicCalibrator:
    def test_fitted_thresholds_are_monotonically_non_decreasing(self) -> None:
        probabilities, labels = _correlated_probabilities_and_labels(n=300)
        fitted = IsotonicCalibrator().fit(probabilities, labels)
        assert isinstance(fitted, FittedIsotonicCalibrator)
        assert list(fitted.y_thresholds) == sorted(fitted.y_thresholds)

    def test_transform_reproduces_sklearn_isotonic_regression_directly(self) -> None:
        from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]

        probabilities, labels = _correlated_probabilities_and_labels(n=300)
        sk_model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        sk_model.fit(probabilities, labels)
        expected = sk_model.predict(probabilities)

        fitted = IsotonicCalibrator().fit(probabilities, labels)
        out = fitted.transform(probabilities)
        np.testing.assert_allclose(out, expected, atol=1e-9)

    def test_round_trip_serialization_reproduces_identical_transform(self) -> None:
        probabilities, labels = _correlated_probabilities_and_labels(n=300)
        fitted = IsotonicCalibrator().fit(probabilities, labels)
        restored = fitted_method_from_json_dict(fitted.to_json_dict())
        assert isinstance(restored, FittedIsotonicCalibrator)
        np.testing.assert_allclose(restored.transform(probabilities), fitted.transform(probabilities), atol=1e-12)

    def test_tampered_non_monotone_thresholds_are_rejected(self) -> None:
        raw = {
            "schema_version": 1, "kind": "isotonic",
            "x_thresholds": [0.1, 0.5, 0.9], "y_thresholds": [0.2, 0.1, 0.8],  # 0.1 < 0.2 -- not monotone
            "input_representation": "predict_proba",
        }
        with pytest.raises(CalibrationValidationError):
            fitted_method_from_json_dict(raw)

    def test_transform_output_stays_within_unit_interval(self) -> None:
        probabilities, labels = _correlated_probabilities_and_labels(n=300)
        fitted = IsotonicCalibrator().fit(probabilities, labels)
        out = fitted.transform(np.array([0.0, 1.0, 0.5]))
        assert np.all((out >= 0.0) & (out <= 1.0))


class TestBetaCalibrator:
    def test_transform_matches_hand_computed_beta_formula(self) -> None:
        """logit(calibrated) = a*log(p) + b*log(1-p) + c (Kull, Silva
        Filho & Flach 2017) -- reproduced directly from the fitted
        (a, b, c) parameters, independent of the fitting code path."""
        probabilities, labels = _correlated_probabilities_and_labels(n=300)
        fitted = BetaCalibrator().fit(probabilities, labels)
        assert isinstance(fitted, FittedBetaCalibrator)
        out = fitted.transform(probabilities)

        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        logit = fitted.log_p_coefficient * np.log(clipped) + fitted.log_one_minus_p_coefficient * np.log(1 - clipped) + fitted.intercept
        expected = 1.0 / (1.0 + np.exp(-logit))
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_round_trip_serialization_reproduces_identical_transform(self) -> None:
        probabilities, labels = _correlated_probabilities_and_labels(n=300)
        fitted = BetaCalibrator().fit(probabilities, labels)
        restored = fitted_method_from_json_dict(fitted.to_json_dict())
        assert isinstance(restored, FittedBetaCalibrator)
        np.testing.assert_allclose(restored.transform(probabilities), fitted.transform(probabilities), atol=1e-12)

    def test_fit_emits_no_warnings(self, recwarn: pytest.WarningsRecorder) -> None:
        """Regression guard for the sklearn `penalty=None` FutureWarning
        discovered during development (fixed via `C=np.inf`)."""
        probabilities, labels = _correlated_probabilities_and_labels(n=300)
        BetaCalibrator().fit(probabilities, labels)
        assert len(recwarn) == 0


class TestMinimumSampleAndClassRequirements:
    @pytest.mark.parametrize("kind", [CalibrationMethodKind.PLATT, CalibrationMethodKind.ISOTONIC, CalibrationMethodKind.BETA])
    def test_fit_with_only_one_class_raises(self, kind: CalibrationMethodKind) -> None:
        method = build_unfit_method(kind)
        with pytest.raises(CalibrationFitError):
            method.fit(np.array([0.2, 0.3, 0.4, 0.5]), np.array([0.0, 0.0, 0.0, 0.0]))

    @pytest.mark.parametrize("kind", [CalibrationMethodKind.IDENTITY, CalibrationMethodKind.PLATT, CalibrationMethodKind.ISOTONIC, CalibrationMethodKind.BETA])
    def test_transform_rejects_out_of_range_probabilities(self, kind: CalibrationMethodKind) -> None:
        probabilities, labels = _correlated_probabilities_and_labels()
        fitted = build_unfit_method(kind).fit(probabilities, labels)
        with pytest.raises(CalibrationValidationError):
            fitted.transform(np.array([1.5]))

    @pytest.mark.parametrize("kind", [CalibrationMethodKind.IDENTITY, CalibrationMethodKind.PLATT, CalibrationMethodKind.ISOTONIC, CalibrationMethodKind.BETA])
    def test_transform_rejects_non_finite_input(self, kind: CalibrationMethodKind) -> None:
        probabilities, labels = _correlated_probabilities_and_labels()
        fitted = build_unfit_method(kind).fit(probabilities, labels)
        with pytest.raises(CalibrationValidationError):
            fitted.transform(np.array([math.nan]))


class TestDeterminism:
    @pytest.mark.parametrize("kind", [CalibrationMethodKind.PLATT, CalibrationMethodKind.ISOTONIC, CalibrationMethodKind.BETA])
    def test_fitting_twice_on_identical_data_produces_identical_parameters(self, kind: CalibrationMethodKind) -> None:
        probabilities, labels = _correlated_probabilities_and_labels(n=300)
        first = build_unfit_method(kind).fit(probabilities, labels)
        second = build_unfit_method(kind).fit(probabilities, labels)
        assert first.to_json_dict() == second.to_json_dict()


class TestMethodComplexityRank:
    def test_identity_is_simplest_and_isotonic_is_most_complex(self) -> None:
        ranks = {kind: method_complexity_rank(kind) for kind in CalibrationMethodKind}
        assert ranks[CalibrationMethodKind.IDENTITY] < ranks[CalibrationMethodKind.PLATT]
        assert ranks[CalibrationMethodKind.PLATT] < ranks[CalibrationMethodKind.BETA]
        assert ranks[CalibrationMethodKind.BETA] < ranks[CalibrationMethodKind.ISOTONIC]

    def test_every_method_kind_has_a_unique_rank(self) -> None:
        ranks = [method_complexity_rank(kind) for kind in CalibrationMethodKind]
        assert len(set(ranks)) == len(ranks)


class TestSummary:
    @pytest.mark.parametrize("kind", [CalibrationMethodKind.IDENTITY, CalibrationMethodKind.PLATT, CalibrationMethodKind.ISOTONIC, CalibrationMethodKind.BETA])
    def test_summary_is_a_non_empty_human_readable_string(self, kind: CalibrationMethodKind) -> None:
        probabilities, labels = _correlated_probabilities_and_labels(n=300)
        fitted = build_unfit_method(kind).fit(probabilities, labels)
        summary = fitted.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestSerializationRoundTripDestroysInMemoryObject:
    """Release audit Section 7: fit; serialize; DESTROY the in-memory
    object (`del fitted`, `gc.collect()`); reload from the durable JSON
    dict alone; transform the same bounded inputs; compare outputs. This
    is a stronger proof than a plain round-trip test that merely keeps
    both objects alive side by side -- it proves `to_json_dict()` alone
    (not any residual Python object state) is sufficient to reconstruct
    identical behavior."""

    _BOUNDED_INPUTS = np.asarray([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])

    @pytest.mark.parametrize("kind", [CalibrationMethodKind.IDENTITY, CalibrationMethodKind.PLATT, CalibrationMethodKind.ISOTONIC, CalibrationMethodKind.BETA])
    def test_reload_after_destroying_original_reproduces_identical_transform(self, kind: CalibrationMethodKind) -> None:
        import gc

        probabilities, labels = _correlated_probabilities_and_labels(n=300)
        fitted = build_unfit_method(kind).fit(probabilities, labels)
        persisted = fitted.to_json_dict()
        original_transform = fitted.transform(self._BOUNDED_INPUTS)

        del fitted
        gc.collect()

        restored = fitted_method_from_json_dict(persisted)
        restored_transform = restored.transform(self._BOUNDED_INPUTS)
        np.testing.assert_allclose(restored_transform, original_transform, atol=1e-12)


class TestSerializationTamperMatrix:
    """Release audit Section 7: independently tamper each of the 9 named
    dimensions and confirm EITHER (a) `from_json_dict`'s own structural
    validation rejects it outright, or (b) where the tampered value is
    still structurally legal (e.g. a different but still-finite
    coefficient), the resulting `.transform()` output measurably diverges
    from the untampered original -- proving the tamper has a real,
    detectable effect rather than being silently absorbed. Content-hash-
    level tampering (write a new artifact under a freshly, validly
    computed hash, then prove `verify_calibration`'s recomputation check
    still fails) is exercised separately at the full artifact-store level
    in `tests/integration/test_calibration_engine.py::TestCorruptionAndTampering`
    -- this class isolates just the calibrator method's own decode/
    transform contract."""

    def _platt_json(self, **overrides: object) -> dict[str, object]:
        base = {"schema_version": 1, "kind": "platt", "coefficient": 2.0, "intercept": -0.5, "input_representation": "predict_proba"}
        return {**base, **overrides}

    def _isotonic_json(self, **overrides: object) -> dict[str, object]:
        base = {
            "schema_version": 1, "kind": "isotonic", "x_thresholds": [0.1, 0.4, 0.7, 0.9], "y_thresholds": [0.05, 0.35, 0.65, 0.95],
            "interpolation": "linear", "out_of_bounds": "clip", "input_representation": "predict_proba",
        }
        return {**base, **overrides}

    def _beta_json(self, **overrides: object) -> dict[str, object]:
        base = {
            "schema_version": 1, "kind": "beta", "log_p_coefficient": 1.2, "log_one_minus_p_coefficient": -0.8,
            "intercept": 0.1, "input_representation": "predict_proba",
        }
        return {**base, **overrides}

    # 1. coefficient (Platt)
    def test_tampered_coefficient_changes_transform_output(self) -> None:
        original = fitted_method_from_json_dict(self._platt_json())
        tampered = fitted_method_from_json_dict(self._platt_json(coefficient=-9.0))
        probes = np.asarray([0.1, 0.5, 0.9])
        assert not np.allclose(original.transform(probes), tampered.transform(probes))

    # 2. intercept (Platt)
    def test_tampered_intercept_changes_transform_output(self) -> None:
        original = fitted_method_from_json_dict(self._platt_json())
        tampered = fitted_method_from_json_dict(self._platt_json(intercept=7.0))
        probes = np.asarray([0.1, 0.5, 0.9])
        assert not np.allclose(original.transform(probes), tampered.transform(probes))

    # 3. class order -- lives on RawPredictionSet, not a fitted-calibrator
    # parameter: calibrators only ever see already-extracted POSITIVE-class
    # probabilities, never the raw multi-column class-indexed array. The
    # structural guard is `RawPredictionSet.__post_init__` requiring
    # `class_labels[positive_class_index] == 1.0` (this platform's fixed
    # convention) -- a self-inconsistent tamper is rejected outright.
    def test_tampered_positive_class_index_pointing_at_wrong_label_is_rejected(self) -> None:
        from quant_platform.calibration.models import RawPredictionSet
        from quant_platform.core.exceptions import CalibrationDataError

        with pytest.raises(CalibrationDataError, match="positive_class_index"):
            RawPredictionSet(
                schema_version=1, outer_fold_index=0, inner_fold_index=0, sample_positions=(0, 1, 2),
                timestamps=("2020-01-01T00:00:00+00:00",) * 3, raw_scores=None, raw_probabilities=(0.2, 0.5, 0.8),
                class_labels=(0.0, 1.0), positive_class_index=0,  # index 0 -> label 0.0, not the fixed 1.0 convention
                source_model_identity="m", source_experiment_id="e", true_labels=(0.0, 1.0, 0.0),
            )

    # 4. input_representation (Platt)
    def test_tampered_input_representation_silently_reinterprets_the_same_raw_number(self) -> None:
        """Flipping `predict_proba` <-> `decision_function` for the IDENTICAL
        persisted coefficient/intercept must not be a no-op: the same raw
        input is fed through a structurally different feature transform
        (logit-of-probability vs. the raw value itself), so the two
        interpretations diverge for any non-trivial input."""
        predict_proba_version = fitted_method_from_json_dict(self._platt_json(input_representation="predict_proba"))
        decision_function_version = fitted_method_from_json_dict(self._platt_json(input_representation="decision_function"))
        probe = np.asarray([0.8])
        assert not np.allclose(predict_proba_version.transform(probe), decision_function_version.transform(probe))

    # 5. isotonic x ordering
    def test_tampered_non_monotone_x_thresholds_are_rejected(self) -> None:
        with pytest.raises(CalibrationValidationError, match="non-decreasing"):
            fitted_method_from_json_dict(self._isotonic_json(x_thresholds=[0.5, 0.1, 0.9, 0.95]))

    # 6. isotonic y bounds
    def test_tampered_out_of_bounds_y_threshold_is_rejected(self) -> None:
        with pytest.raises(CalibrationValidationError, match=r"\[0, 1\]"):
            fitted_method_from_json_dict(self._isotonic_json(y_thresholds=[0.05, 0.35, 0.65, 1.5]))

    def test_tampered_negative_y_threshold_is_rejected(self) -> None:
        with pytest.raises(CalibrationValidationError, match=r"\[0, 1\]"):
            fitted_method_from_json_dict(self._isotonic_json(y_thresholds=[-0.2, 0.35, 0.65, 0.95]))

    # 7. beta parameters
    def test_tampered_beta_log_p_coefficient_changes_transform_output(self) -> None:
        original = fitted_method_from_json_dict(self._beta_json())
        tampered = fitted_method_from_json_dict(self._beta_json(log_p_coefficient=-5.0))
        probes = np.asarray([0.1, 0.5, 0.9])
        assert not np.allclose(original.transform(probes), tampered.transform(probes))

    def test_tampered_beta_intercept_changes_transform_output(self) -> None:
        original = fitted_method_from_json_dict(self._beta_json())
        tampered = fitted_method_from_json_dict(self._beta_json(intercept=10.0))
        probes = np.asarray([0.1, 0.5, 0.9])
        assert not np.allclose(original.transform(probes), tampered.transform(probes))

    # 8. schema_version
    @pytest.mark.parametrize("json_builder_name", ["_platt_json", "_isotonic_json", "_beta_json"])
    def test_tampered_schema_version_is_rejected(self, json_builder_name: str) -> None:
        from quant_platform.core.exceptions import SchemaVersionError

        builder = getattr(self, json_builder_name)
        with pytest.raises(SchemaVersionError):
            fitted_method_from_json_dict(builder(schema_version=999))

    # 9. calibration_id -- not a per-method concept (a `FittedXCalibrator`
    # carries no calibration_id at all); the corresponding guard lives at
    # `OuterFoldCalibrationResult`/manifest cross-reference level and is
    # exercised at the full artifact-store level in
    # `test_calibration_engine.py` (`outer_fold_result_key_mismatch`,
    # triggered whenever `decoded.calibration_id != manifest.calibration_id`).
    def test_kind_mismatch_is_rejected_as_the_analogous_identity_tamper(self) -> None:
        """The nearest per-method analogue of a "wrong identity" tamper:
        a `kind` field that does not match the dict shape it is embedded
        in (e.g. claiming `"kind": "beta"` while supplying Platt's
        `coefficient`/`intercept` fields) must not be silently accepted
        as if it were the OTHER method."""
        mismatched = {**self._platt_json(), "kind": "beta"}
        with pytest.raises(CalibrationValidationError, match="kind mismatch"):
            FittedPlattCalibrator.from_json_dict(mismatched)
