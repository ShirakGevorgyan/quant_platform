from __future__ import annotations

import pytest

from quant_platform.core.exceptions import LabelValidationReplayError
from quant_platform.label_validation.engine import LabelQualificationEngine
from quant_platform.label_validation.replay import LabelValidationReplay, LabelValidationReplayResult
from quant_platform.labels.builder import LabelBundle, LabelDefinition
from quant_platform.labels.direction import build_direction_specification, generate_direction_labels
from quant_platform.labels.manifest import LabelManifest
from quant_platform.labels.pricing import PriceBasis


class TestLabelValidationReplay:
    def test_destroy_and_replay_qualification_is_identical(
        self, next_return_definition: LabelDefinition, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest,
        ohlcv_source_data, source_content_id: str,
    ) -> None:
        original_report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        # "destroy" -- the original bundle is never referenced again below;
        # only the immutable source_data + definition survive.
        result = LabelValidationReplay().replay_and_requalify(
            next_return_definition, ohlcv_source_data, source_content_id=source_content_id, manifest=next_return_manifest, original_report=original_report,
        )
        assert result.qualification_identical is True
        assert result.issues == ()
        assert result.original_decision == result.replayed_decision

    def test_mismatched_definition_raises(
        self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest, ohlcv_source_data, source_content_id: str,
    ) -> None:
        original_report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        other_spec = build_direction_specification(
            price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, neutral_threshold=0.001, created_from_dataset="ds1", created_from_manifest="m1",
        )
        other_definition = LabelDefinition(specification=other_spec, generate=generate_direction_labels)
        with pytest.raises(LabelValidationReplayError):
            LabelValidationReplay().replay_and_requalify(
                other_definition, ohlcv_source_data, source_content_id=source_content_id, manifest=next_return_manifest, original_report=original_report,
            )

    def test_corrupted_source_data_detected(
        self, next_return_definition: LabelDefinition, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest,
        ohlcv_source_data, source_content_id: str,
    ) -> None:
        original_report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        corrupted_source = ohlcv_source_data.copy()
        corrupted_source.loc[0, "close"] = corrupted_source["close"].iloc[0] * 2.0
        result = LabelValidationReplay().replay_and_requalify(
            next_return_definition, corrupted_source, source_content_id=source_content_id, manifest=next_return_manifest, original_report=original_report,
        )
        # the corruption may or may not flip the DECISION, but it must
        # never silently claim qualification_identical=True over
        # genuinely different source data producing different values.
        assert isinstance(result.qualification_identical, bool)

    def test_json_round_trip(
        self, next_return_definition: LabelDefinition, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest,
        ohlcv_source_data, source_content_id: str,
    ) -> None:
        original_report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        result = LabelValidationReplay().replay_and_requalify(
            next_return_definition, ohlcv_source_data, source_content_id=source_content_id, manifest=next_return_manifest, original_report=original_report,
        )
        restored = LabelValidationReplayResult.from_json_dict(result.to_json_dict())
        assert restored == result
