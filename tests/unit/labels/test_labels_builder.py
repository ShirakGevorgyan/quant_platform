from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelGenerationContractError, LabelMutableAliasError
from quant_platform.labels.builder import LabelBuilder, LabelBundle, LabelDefinition
from quant_platform.labels.models import LabelSpecification


class TestLabelBuilderBuild:
    def test_happy_path(self, definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str) -> None:
        bundle = LabelBuilder().build(definition, source_data, source_content_id=source_content_id)
        assert bundle.row_count == len(source_data)
        assert bundle.valid_count < bundle.row_count  # marker_generator leaves a trailing NaN tail
        assert bundle.identity.source_content_id == source_content_id
        assert bundle.specification == definition.specification

    def test_wrong_length_generator_rejected(self, specification: LabelSpecification, source_data: pd.DataFrame, source_content_id: str, wrong_length_generator_fn) -> None:
        definition = LabelDefinition(specification=specification, generate=wrong_length_generator_fn)
        with pytest.raises(LabelGenerationContractError):
            LabelBuilder().build(definition, source_data, source_content_id=source_content_id)

    def test_non_series_generator_rejected(self, specification: LabelSpecification, source_data: pd.DataFrame, source_content_id: str, non_series_generator_fn) -> None:
        definition = LabelDefinition(specification=specification, generate=non_series_generator_fn)  # type: ignore[arg-type]
        with pytest.raises(LabelGenerationContractError):
            LabelBuilder().build(definition, source_data, source_content_id=source_content_id)

    def test_non_numeric_generator_rejected(self, specification: LabelSpecification, source_data: pd.DataFrame, source_content_id: str, non_numeric_generator_fn) -> None:
        definition = LabelDefinition(specification=specification, generate=non_numeric_generator_fn)
        with pytest.raises(LabelGenerationContractError):
            LabelBuilder().build(definition, source_data, source_content_id=source_content_id)

    def test_aliasing_generator_rejected(self, specification: LabelSpecification, source_data: pd.DataFrame, source_content_id: str, aliasing_generator_fn) -> None:
        definition = LabelDefinition(specification=specification, generate=aliasing_generator_fn)
        with pytest.raises(LabelMutableAliasError):
            LabelBuilder().build(definition, source_data, source_content_id=source_content_id)

    def test_two_builds_from_identical_inputs_are_identical(self, definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str) -> None:
        a = LabelBuilder().build(definition, source_data, source_content_id=source_content_id)
        b = LabelBuilder().build(definition, source_data, source_content_id=source_content_id)
        assert a.identity.content_id == b.identity.content_id


class TestLabelBundleJsonRoundTrip:
    def test_round_trip(self, bundle: LabelBundle) -> None:
        restored = LabelBundle.from_json_dict(bundle.to_json_dict())
        assert restored.specification == bundle.specification
        assert restored.identity == bundle.identity
        assert restored.row_count == bundle.row_count
        assert restored.valid_count == bundle.valid_count
        pd.testing.assert_series_equal(restored.values, bundle.values, check_names=False)
