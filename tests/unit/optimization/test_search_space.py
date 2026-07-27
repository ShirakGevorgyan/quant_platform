"""Milestone 4D: typed search-space parameter validation, canonicalization/
fingerprint determinism, model-specific default spaces, and sampled-value
validation."""

from __future__ import annotations

import pytest

from quant_platform.optimization.search_space import (
    BooleanParameter,
    CategoricalParameter,
    FixedParameter,
    FloatParameter,
    IntegerParameter,
    ParameterKind,
    SearchSpace,
    baseline_fixed_search_space,
    build_search_space,
    catboost_default_search_space,
    default_search_space_for_model,
    lightgbm_default_search_space,
    validate_sampled_values,
    xgboost_default_search_space,
)


class TestIntegerParameterValidation:
    def test_low_greater_than_high_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="low"):
            IntegerParameter(name="x", low=10, high=5)

    def test_log_scale_requires_positive_lower_bound(self) -> None:
        with pytest.raises(ValueError, match="log=True"):
            IntegerParameter(name="x", low=0, high=10, log=True)

    def test_log_scale_requires_step_1(self) -> None:
        with pytest.raises(ValueError, match="log=True"):
            IntegerParameter(name="x", low=1, high=10, step=2, log=True)

    def test_step_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="step"):
            IntegerParameter(name="x", low=1, high=10, step=0)

    def test_round_trip(self) -> None:
        p = IntegerParameter(name="x", low=1, high=10, step=2, log=False)
        assert IntegerParameter.from_json_dict(p.to_json_dict()) == p


class TestFloatParameterValidation:
    def test_low_greater_than_high_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="low"):
            FloatParameter(name="x", low=1.0, high=0.5)

    def test_log_scale_requires_positive_lower_bound(self) -> None:
        with pytest.raises(ValueError, match="log=True"):
            FloatParameter(name="x", low=0.0, high=1.0, log=True)

    def test_log_scale_cannot_combine_with_step(self) -> None:
        with pytest.raises(ValueError, match="log=True"):
            FloatParameter(name="x", low=0.1, high=1.0, step=0.1, log=True)

    def test_non_finite_bounds_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            FloatParameter(name="x", low=float("nan"), high=1.0)

    def test_round_trip(self) -> None:
        p = FloatParameter(name="x", low=0.01, high=1.0, log=True)
        assert FloatParameter.from_json_dict(p.to_json_dict()) == p


class TestCategoricalParameterValidation:
    def test_empty_choices_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            CategoricalParameter(name="x", choices=())

    def test_duplicate_choices_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            CategoricalParameter(name="x", choices=("a", "a"))

    def test_non_primitive_choice_rejected_with_a_clean_error_not_a_raw_typeerror(self) -> None:
        """Regression test: an unhashable choice (a nested dict) used to
        crash `set(self.choices)` with a raw, unhandled `TypeError` before
        the type-validation loop ever ran -- type/finiteness is now
        checked BEFORE the duplicate check, so this always fails with the
        module's own clear ValueError instead."""
        with pytest.raises(ValueError, match="JSON primitive"):
            CategoricalParameter(name="x", choices=({"nested": "dict"},))  # type: ignore[arg-type]

    def test_mixed_primitive_choices_allowed(self) -> None:
        p = CategoricalParameter(name="x", choices=("a", 1, 2.5, None))
        assert p.choices == ("a", 1, 2.5, None)

    def test_int_and_bool_choices_collide_under_python_equality(self) -> None:
        """Documented limitation: `CategoricalParameter` choice
        distinctness uses plain Python equality/hashing, under which
        `1 == True` and `0 == False` -- so `1` and `True` cannot both be
        declared as separate choices of the same parameter (this mirrors
        JSON Schema's own enum-equality convention). Use string choices
        (e.g. `"1"`/`"true"`) if genuinely distinguishable values are needed."""
        with pytest.raises(ValueError, match="duplicates"):
            CategoricalParameter(name="x", choices=(1, True))

    def test_round_trip(self) -> None:
        p = CategoricalParameter(name="x", choices=("a", "b", "c"))
        assert CategoricalParameter.from_json_dict(p.to_json_dict()) == p


class TestFixedAndBooleanParameters:
    def test_fixed_parameter_round_trip(self) -> None:
        p = FixedParameter(name="x", value=42)
        assert FixedParameter.from_json_dict(p.to_json_dict()) == p

    def test_fixed_parameter_rejects_non_finite_float(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            FixedParameter(name="x", value=float("inf"))

    def test_boolean_parameter_round_trip(self) -> None:
        p = BooleanParameter(name="flag")
        assert BooleanParameter.from_json_dict(p.to_json_dict()) == p


class TestSearchSpaceCanonicalizationAndFingerprint:
    def test_empty_search_space_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_search_space([])

    def test_duplicate_parameter_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            build_search_space([IntegerParameter(name="x", low=1, high=2), FloatParameter(name="x", low=0.0, high=1.0)])

    def test_fingerprint_is_deterministic(self) -> None:
        space_a = build_search_space([IntegerParameter(name="x", low=1, high=10)])
        space_b = build_search_space([IntegerParameter(name="x", low=1, high=10)])
        assert space_a.fingerprint() == space_b.fingerprint()

    def test_fingerprint_changes_with_bounds(self) -> None:
        space_a = build_search_space([IntegerParameter(name="x", low=1, high=10)])
        space_b = build_search_space([IntegerParameter(name="x", low=1, high=20)])
        assert space_a.fingerprint() != space_b.fingerprint()

    def test_fingerprint_is_order_sensitive(self) -> None:
        space_a = build_search_space([IntegerParameter(name="x", low=1, high=10), FloatParameter(name="y", low=0.0, high=1.0)])
        space_b = build_search_space([FloatParameter(name="y", low=0.0, high=1.0), IntegerParameter(name="x", low=1, high=10)])
        assert space_a.fingerprint() != space_b.fingerprint()

    def test_round_trip_preserves_all_parameter_kinds(self) -> None:
        space = build_search_space([
            IntegerParameter(name="i", low=1, high=10),
            FloatParameter(name="f", low=0.01, high=1.0, log=True),
            CategoricalParameter(name="c", choices=("a", "b")),
            BooleanParameter(name="b"),
            FixedParameter(name="fx", value=1.5),
        ])
        decoded = SearchSpace.from_json_dict(space.to_json_dict())
        assert decoded == space

    def test_unknown_kind_rejected_on_decode(self) -> None:
        raw = {"schema_version": 1, "parameters": [{"kind": "not_a_real_kind", "name": "x"}]}
        with pytest.raises(ValueError, match="Unknown search-space parameter kind"):
            SearchSpace.from_json_dict(raw)


class TestSampledValueValidation:
    def test_missing_parameter_rejected(self) -> None:
        space = build_search_space([IntegerParameter(name="x", low=1, high=10)])
        with pytest.raises(ValueError, match="missing"):
            validate_sampled_values(space, {})

    def test_undeclared_parameter_rejected(self) -> None:
        space = build_search_space([IntegerParameter(name="x", low=1, high=10)])
        with pytest.raises(ValueError, match="undeclared"):
            validate_sampled_values(space, {"x": 5, "y": 1})

    def test_out_of_bounds_integer_rejected(self) -> None:
        space = build_search_space([IntegerParameter(name="x", low=1, high=10)])
        with pytest.raises(ValueError, match="outside"):
            validate_sampled_values(space, {"x": 99})

    def test_integer_not_reachable_with_step_rejected(self) -> None:
        space = build_search_space([IntegerParameter(name="x", low=0, high=10, step=2)])
        with pytest.raises(ValueError, match="not reachable"):
            validate_sampled_values(space, {"x": 3})

    def test_categorical_choice_outside_declared_set_rejected(self) -> None:
        space = build_search_space([CategoricalParameter(name="x", choices=("a", "b"))])
        with pytest.raises(ValueError, match="not one of"):
            validate_sampled_values(space, {"x": "z"})

    def test_fixed_parameter_value_mismatch_rejected(self) -> None:
        space = build_search_space([FixedParameter(name="x", value=1)])
        with pytest.raises(ValueError, match="fixed"):
            validate_sampled_values(space, {"x": 2})

    def test_valid_values_pass(self) -> None:
        space = build_search_space([
            IntegerParameter(name="i", low=1, high=10), FloatParameter(name="f", low=0.0, high=1.0),
            CategoricalParameter(name="c", choices=("a", "b")), BooleanParameter(name="b"), FixedParameter(name="fx", value=1),
        ])
        validate_sampled_values(space, {"i": 5, "f": 0.5, "c": "a", "b": True, "fx": 1})


class TestSampledValueBoundaryInclusivity:
    """Adversarial audit, Section 7: 'sample boundary values.' Both ends
    of a declared range are legal sampled values (`low <= value <= high`,
    inclusive on both sides, never an off-by-one exclusive bound), and the
    immediately-adjacent out-of-range values are rejected."""

    def test_integer_exact_low_and_high_are_both_accepted(self) -> None:
        space = build_search_space([IntegerParameter(name="x", low=5, high=20)])
        validate_sampled_values(space, {"x": 5})
        validate_sampled_values(space, {"x": 20})

    def test_integer_one_below_low_and_one_above_high_are_both_rejected(self) -> None:
        space = build_search_space([IntegerParameter(name="x", low=5, high=20)])
        with pytest.raises(ValueError, match="outside"):
            validate_sampled_values(space, {"x": 4})
        with pytest.raises(ValueError, match="outside"):
            validate_sampled_values(space, {"x": 21})

    def test_float_exact_low_and_high_are_both_accepted(self) -> None:
        space = build_search_space([FloatParameter(name="x", low=0.1, high=0.9)])
        validate_sampled_values(space, {"x": 0.1})
        validate_sampled_values(space, {"x": 0.9})

    def test_float_just_outside_either_bound_is_rejected(self) -> None:
        space = build_search_space([FloatParameter(name="x", low=0.1, high=0.9)])
        with pytest.raises(ValueError, match="outside"):
            validate_sampled_values(space, {"x": 0.099999})
        with pytest.raises(ValueError, match="outside"):
            validate_sampled_values(space, {"x": 0.900001})

    def test_integer_with_step_reaches_exactly_the_declared_high_when_evenly_divisible(self) -> None:
        space = build_search_space([IntegerParameter(name="x", low=0, high=10, step=5)])
        validate_sampled_values(space, {"x": 0})
        validate_sampled_values(space, {"x": 5})
        validate_sampled_values(space, {"x": 10})

    def test_integer_with_step_rejects_high_when_not_evenly_divisible_from_low(self) -> None:
        """`high` is the declared INCLUSIVE ceiling, not a guarantee the
        step lattice actually lands on it -- proven here as a real,
        non-obvious edge: high=10 is itself UNREACHABLE with step=3 from
        low=0 (0, 3, 6, 9 are the only reachable values <= 10)."""
        space = build_search_space([IntegerParameter(name="x", low=0, high=10, step=3)])
        validate_sampled_values(space, {"x": 9})
        with pytest.raises(ValueError, match="not reachable"):
            validate_sampled_values(space, {"x": 10})


class TestModelSpecificDefaultSpaces:
    @pytest.mark.parametrize("factory", [lightgbm_default_search_space, xgboost_default_search_space, catboost_default_search_space])
    def test_default_space_is_internally_valid_and_has_a_rounds_key(self, factory) -> None:
        space = factory()
        names = space.parameter_names
        assert "learning_rate" in names
        assert ("num_boost_round" in names) or ("iterations" in names)

    def test_default_search_space_for_model_dispatches_correctly(self) -> None:
        assert default_search_space_for_model("lightgbm").fingerprint() == lightgbm_default_search_space().fingerprint()

    def test_default_search_space_for_model_raises_for_unknown_model(self) -> None:
        with pytest.raises(ValueError, match="No default search space"):
            default_search_space_for_model("logistic_regression")

    def test_baseline_fixed_search_space_is_all_fixed_parameters(self) -> None:
        space = baseline_fixed_search_space({"constant": 0.5})
        assert all(p.__class__.__name__ == "FixedParameter" for p in space.parameters)

    def test_baseline_fixed_search_space_requires_at_least_one_value(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            baseline_fixed_search_space({})

    def test_parameter_kind_values_are_stable_strings(self) -> None:
        assert ParameterKind.INTEGER.value == "integer"
        assert ParameterKind.FLOAT.value == "float"
        assert ParameterKind.CATEGORICAL.value == "categorical"
        assert ParameterKind.BOOLEAN.value == "boolean"
        assert ParameterKind.FIXED.value == "fixed"
