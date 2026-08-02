from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelReplayError
from quant_platform.labels.builder import LabelBundle, LabelDefinition
from quant_platform.labels.models import LabelSpecification
from quant_platform.labels.replay import LabelReplay, LabelReplayResult


class TestLabelReplay:
    def test_clean_replay_matches(self, definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str, bundle: LabelBundle) -> None:
        result = LabelReplay().replay(definition, source_data, source_content_id=source_content_id, original=bundle)
        assert result.replayed is True
        assert result.issues == ()
        assert result.original_content_id == result.replayed_content_id

    def test_mismatched_specification_raises(
        self, other_family_specification: LabelSpecification, source_data: pd.DataFrame, source_content_id: str, bundle: LabelBundle, marker_generator_fn,
    ) -> None:
        wrong_definition = LabelDefinition(specification=other_family_specification, generate=marker_generator_fn)
        with pytest.raises(LabelReplayError):
            LabelReplay().replay(wrong_definition, source_data, source_content_id=source_content_id, original=bundle)

    def test_divergent_source_data_detected(
        self, definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str, bundle: LabelBundle,
    ) -> None:
        shorter_source = source_data.iloc[:-1].reset_index(drop=True)
        result = LabelReplay().replay(definition, shorter_source, source_content_id=source_content_id, original=bundle)
        assert result.replayed is False
        assert any("row_count" in issue for issue in result.issues)

    def test_json_round_trip(self, definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str, bundle: LabelBundle) -> None:
        result = LabelReplay().replay(definition, source_data, source_content_id=source_content_id, original=bundle)
        assert LabelReplayResult.from_json_dict(result.to_json_dict()) == result
