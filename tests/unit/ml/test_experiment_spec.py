from __future__ import annotations

import pytest
from tests.unit.ml.conftest import make_experiment_spec_kwargs, make_label_binding

from quant_platform.core.exceptions import UnsupportedObjectiveError
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.models import LabelType, ObjectiveType


class TestExperimentSpecConstruction:
    def test_valid_spec_builds(self) -> None:
        ExperimentSpec(**make_experiment_spec_kwargs())

    def test_empty_model_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="model_name"):
            ExperimentSpec(**make_experiment_spec_kwargs(model_name=""))

    def test_empty_model_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="model_version"):
            ExperimentSpec(**make_experiment_spec_kwargs(model_version=""))

    def test_empty_primary_metric_rejected(self) -> None:
        with pytest.raises(ValueError, match="primary_metric"):
            ExperimentSpec(**make_experiment_spec_kwargs(primary_metric=""))

    def test_duplicate_tags_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            ExperimentSpec(**make_experiment_spec_kwargs(tags=("a", "a")))

    def test_invalid_environment_requirements_value_rejected(self) -> None:
        with pytest.raises(ValueError):
            ExperimentSpec(**make_experiment_spec_kwargs(environment_requirements={"": "x"}))

    def test_incompatible_label_objective_rejected_at_construction(self) -> None:
        bad_label = make_label_binding(label_type=LabelType.BINARY)
        with pytest.raises(UnsupportedObjectiveError):
            ExperimentSpec(**make_experiment_spec_kwargs(label_binding=bad_label, objective=ObjectiveType.REGRESSION))


class TestIdentityVsDescriptivePayload:
    def test_identity_payload_excludes_descriptive_fields(self) -> None:
        spec = ExperimentSpec(**make_experiment_spec_kwargs(notes="n", tags=("t",), primary_metric="rmse"))
        payload = spec.to_identity_payload()
        assert "notes" not in payload
        assert "tags" not in payload
        assert "primary_metric" not in payload

    def test_json_dict_includes_descriptive_fields(self) -> None:
        spec = ExperimentSpec(**make_experiment_spec_kwargs(notes="n", tags=("t",), primary_metric="rmse"))
        full = spec.to_json_dict()
        assert full["notes"] == "n"
        assert full["tags"] == ["t"]
        assert full["primary_metric"] == "rmse"

    def test_changing_notes_tags_metric_does_not_change_identity_payload(self) -> None:
        spec1 = ExperimentSpec(**make_experiment_spec_kwargs(notes="first", tags=(), primary_metric="rmse"))
        spec2 = ExperimentSpec(**make_experiment_spec_kwargs(notes="second, totally different", tags=("a", "b"), primary_metric="mae"))
        assert spec1.to_identity_payload() == spec2.to_identity_payload()

    def test_from_json_dict_round_trip(self) -> None:
        spec = ExperimentSpec(**make_experiment_spec_kwargs(notes="n", tags=("a",)))
        restored = ExperimentSpec.from_json_dict(spec.to_json_dict())
        assert restored == spec
        assert restored.to_json_dict() == spec.to_json_dict()

    def test_environment_requirements_sorted_in_identity_payload(self) -> None:
        spec = ExperimentSpec(**make_experiment_spec_kwargs(environment_requirements={"z": "1", "a": "2"}))
        payload = spec.to_identity_payload()
        assert list(payload["environment_requirements"].keys()) == ["a", "z"]  # type: ignore[union-attr]
