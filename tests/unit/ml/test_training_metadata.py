from __future__ import annotations

import math

import pytest

from quant_platform.ml.persistence import format_utc_timestamp, utc_now
from quant_platform.ml.training_metadata import TrainingMetadata


def _make(**overrides: object) -> TrainingMetadata:
    base: dict[str, object] = {
        "schema_version": 1, "experiment_id": "e" * 64, "fold_index": 0, "model_name": "lightgbm",
        "model_version": "1", "library_name": "lightgbm", "library_version": "4.7.0", "seed": 42,
        "training_duration_seconds": 1.23, "feature_schema_fingerprint": "a" * 64, "dataset_content_id": "b" * 64,
        "fitted_at": format_utc_timestamp(utc_now()), "hyperparameters": {"num_leaves": 31},
    }
    base.update(overrides)
    return TrainingMetadata(**base)  # type: ignore[arg-type]


class TestConstructionValidation:
    def test_valid_construction(self) -> None:
        tm = _make()
        assert tm.model_name == "lightgbm"

    @pytest.mark.parametrize(
        "field_name,bad_value,match",
        [
            ("fold_index", -1, "fold_index"),
            ("model_name", "", "model_name"),
            ("model_version", "", "model_version"),
            ("library_name", "", "library_name"),
            ("library_version", "", "library_version"),
            ("seed", -1, "seed"),
            ("training_duration_seconds", -0.5, "training_duration_seconds"),
            ("training_duration_seconds", math.nan, "training_duration_seconds"),
            ("training_duration_seconds", math.inf, "training_duration_seconds"),
            ("feature_schema_fingerprint", "", "feature_schema_fingerprint"),
            ("dataset_content_id", "", "dataset_content_id"),
        ],
    )
    def test_invalid_fields_rejected(self, field_name: str, bad_value: object, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _make(**{field_name: bad_value})

    def test_invalid_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError):
            _make(fitted_at="not-a-timestamp")

    def test_hyperparameters_must_be_json_primitive(self) -> None:
        with pytest.raises(ValueError):
            _make(hyperparameters={"bad": object()})


class TestRoundTrip:
    def test_to_json_dict_from_json_dict(self) -> None:
        tm = _make()
        restored = TrainingMetadata.from_json_dict(tm.to_json_dict())
        assert restored == tm

    def test_hyperparameters_sorted_in_json_dict(self) -> None:
        tm = _make(hyperparameters={"z": 1, "a": 2})
        raw = tm.to_json_dict()
        assert list(raw["hyperparameters"].keys()) == ["a", "z"]  # type: ignore[union-attr]

    def test_wrong_schema_version_rejected(self) -> None:
        raw = _make().to_json_dict()
        raw["schema_version"] = 999
        with pytest.raises(Exception):  # noqa: B017 - SchemaVersionError, exact type is ml.persistence's concern
            TrainingMetadata.from_json_dict(raw)

    def test_empty_hyperparameters_round_trips(self) -> None:
        tm = _make(hyperparameters={})
        restored = TrainingMetadata.from_json_dict(tm.to_json_dict())
        assert restored.hyperparameters == {}

    def test_string_coerced_nan_duration_rejected_on_from_json_dict(self) -> None:
        raw = _make().to_json_dict()
        raw["training_duration_seconds"] = "nan"
        with pytest.raises(ValueError, match="finite"):
            TrainingMetadata.from_json_dict(raw)
