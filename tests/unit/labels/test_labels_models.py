from __future__ import annotations

import pytest

from quant_platform.core.exceptions import LabelRequestError
from quant_platform.labels.models import (
    LABEL_IDENTITY_ALGORITHM,
    LabelFamily,
    LabelSpecification,
    build_label_specification,
    compute_label_specification_id,
    compute_parameter_hash,
)


class TestLabelFamily:
    def test_six_named_families_exist(self) -> None:
        assert {f.value for f in LabelFamily} == {
            "next_return", "multi_horizon_return", "direction", "triple_barrier", "forward_volatility", "future_extension_placeholder",
        }


class TestComputeParameterHash:
    def test_deterministic(self) -> None:
        assert compute_parameter_hash({"a": 1, "b": 2}) == compute_parameter_hash({"b": 2, "a": 1})

    def test_different_parameters_different_hash(self) -> None:
        assert compute_parameter_hash({"horizon_bars": 5}) != compute_parameter_hash({"horizon_bars": 10})

    def test_empty_parameters(self) -> None:
        assert compute_parameter_hash({}) == compute_parameter_hash({})


class TestBuildLabelSpecification:
    def test_self_consistent_by_construction(self, specification: LabelSpecification) -> None:
        consistent, issues = specification.verify_self_consistency()
        assert consistent is True
        assert issues == ()

    def test_identity_algorithm_defaults(self, specification: LabelSpecification) -> None:
        assert specification.identity_algorithm == LABEL_IDENTITY_ALGORITHM

    def test_empty_created_from_dataset_rejected(self) -> None:
        with pytest.raises(LabelRequestError):
            build_label_specification(
                label_family=LabelFamily.NEXT_RETURN, generation_version="v1", price_basis="close", prediction_horizon="5 bars",
                availability_rule="rule", reference_price="close", event_time_rule="close time", generation_rule="rule",
                created_from_dataset="", created_from_manifest="manifest-0001",
            )

    def test_empty_generation_version_rejected(self) -> None:
        with pytest.raises(LabelRequestError):
            build_label_specification(
                label_family=LabelFamily.NEXT_RETURN, generation_version="", price_basis="close", prediction_horizon="5 bars",
                availability_rule="rule", reference_price="close", event_time_rule="close time", generation_rule="rule",
                created_from_dataset="dataset-0001", created_from_manifest="manifest-0001",
            )


class TestVersioningChangesIdentity:
    def test_changing_prediction_horizon_produces_a_new_id(self, specification: LabelSpecification) -> None:
        other = build_label_specification(
            label_family=specification.label_family, generation_version=specification.generation_version, price_basis=specification.price_basis,
            prediction_horizon="20 bars", availability_rule=specification.availability_rule, reference_price=specification.reference_price,
            event_time_rule=specification.event_time_rule, generation_rule=specification.generation_rule,
            created_from_dataset=specification.created_from_dataset, created_from_manifest=specification.created_from_manifest,
            parameters={"horizon_bars": 20},
        )
        assert other.label_specification_id != specification.label_specification_id

    def test_changing_price_basis_produces_a_new_id(self, specification: LabelSpecification) -> None:
        other = build_label_specification(
            label_family=specification.label_family, generation_version=specification.generation_version, price_basis="mid",
            prediction_horizon=specification.prediction_horizon, availability_rule=specification.availability_rule,
            reference_price=specification.reference_price, event_time_rule=specification.event_time_rule,
            generation_rule=specification.generation_rule, created_from_dataset=specification.created_from_dataset,
            created_from_manifest=specification.created_from_manifest, parameters=specification.parameters,
        )
        assert other.label_specification_id != specification.label_specification_id

    def test_identical_inputs_produce_identical_id(self, specification: LabelSpecification) -> None:
        rebuilt = build_label_specification(
            label_family=specification.label_family, generation_version=specification.generation_version, price_basis=specification.price_basis,
            prediction_horizon=specification.prediction_horizon, availability_rule=specification.availability_rule,
            reference_price=specification.reference_price, event_time_rule=specification.event_time_rule,
            generation_rule=specification.generation_rule, created_from_dataset=specification.created_from_dataset,
            created_from_manifest=specification.created_from_manifest, parameters=specification.parameters,
        )
        assert rebuilt.label_specification_id == specification.label_specification_id

    def test_never_mutates_a_frozen_dataclass(self, specification: LabelSpecification) -> None:
        with pytest.raises(AttributeError):
            specification.prediction_horizon = "999 bars"  # type: ignore[misc]


class TestJsonRoundTrip:
    def test_round_trip(self, specification: LabelSpecification) -> None:
        restored = LabelSpecification.from_json_dict(specification.to_json_dict())
        assert restored == specification
        assert restored.verify_self_consistency() == (True, ())


class TestComputeLabelSpecificationIdIndependence:
    def test_matches_specification_own_id(self, specification: LabelSpecification) -> None:
        recomputed = compute_label_specification_id(
            schema_version=specification.schema_version, label_family=specification.label_family, generation_version=specification.generation_version,
            parameter_hash=specification.parameter_hash, price_basis=specification.price_basis, prediction_horizon=specification.prediction_horizon,
            availability_rule=specification.availability_rule, reference_price=specification.reference_price, event_time_rule=specification.event_time_rule,
            generation_rule=specification.generation_rule, identity_algorithm=specification.identity_algorithm,
            created_from_dataset=specification.created_from_dataset, created_from_manifest=specification.created_from_manifest,
        )
        assert recomputed == specification.label_specification_id
