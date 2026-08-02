from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelRecoveryError
from quant_platform.labels.builder import LabelBundle, LabelDefinition
from quant_platform.labels.identity import compute_label_identity
from quant_platform.labels.models import LabelSpecification
from quant_platform.labels.recovery import LabelRecovery, LabelRecoveryResult


class TestLabelRecovery:
    def test_clean_recovery(
        self, specification: LabelSpecification, definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str, bundle: LabelBundle,
    ) -> None:
        result = LabelRecovery().recover(specification, definition, source_data, source_content_id=source_content_id, expected_identity=bundle.identity)
        assert result.recoverable is True
        assert result.recovered_bundle is not None
        assert result.recovered_bundle.identity.content_id == bundle.identity.content_id

    def test_no_evidence_supplied_fails_closed(self, specification: LabelSpecification) -> None:
        result = LabelRecovery().recover(specification, None, None, source_content_id=None)
        assert result.recoverable is False
        assert result.recovered_bundle is None
        assert result.issues != ()

    def test_mismatched_expected_identity_fails_closed_never_returns_wrong_bundle(
        self, specification: LabelSpecification, definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str,
    ) -> None:
        bogus_identity = compute_label_identity(specification.label_specification_id, pd.Series([0.0, 0.0]), source_content_id="different-source")
        result = LabelRecovery().recover(specification, definition, source_data, source_content_id=source_content_id, expected_identity=bogus_identity)
        assert result.recoverable is False
        assert result.recovered_bundle is None

    def test_mismatched_definition_specification_raises(
        self, specification: LabelSpecification, other_family_specification: LabelSpecification, source_data: pd.DataFrame, source_content_id: str, marker_generator_fn,
    ) -> None:
        wrong_definition = LabelDefinition(specification=other_family_specification, generate=marker_generator_fn)
        with pytest.raises(LabelRecoveryError):
            LabelRecovery().recover(specification, wrong_definition, source_data, source_content_id=source_content_id)

    def test_no_expected_identity_still_recovers(
        self, specification: LabelSpecification, definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str,
    ) -> None:
        result = LabelRecovery().recover(specification, definition, source_data, source_content_id=source_content_id)
        assert result.recoverable is True
        assert result.recovered_bundle is not None

    def test_json_round_trip(
        self, specification: LabelSpecification, definition: LabelDefinition, source_data: pd.DataFrame, source_content_id: str,
    ) -> None:
        result = LabelRecovery().recover(specification, definition, source_data, source_content_id=source_content_id)
        assert LabelRecoveryResult.from_json_dict(result.to_json_dict()) == result
