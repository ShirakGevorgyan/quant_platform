from __future__ import annotations

from quant_platform.labels.versioning import LabelVersion, LabelVersionHistory


class TestLabelVersion:
    def test_json_round_trip(self) -> None:
        version = LabelVersion(generation_version="v1", label_specification_id="spec-1", parameter_hash="hash-1", registered_at="2024-01-01T00:00:00+00:00")
        assert LabelVersion.from_json_dict(version.to_json_dict()) == version


class TestLabelVersionHistory:
    def test_json_round_trip(self) -> None:
        history = LabelVersionHistory(
            schema_version=1, label_family="next_return",
            versions=(LabelVersion(generation_version="v1", label_specification_id="spec-1", parameter_hash="hash-1", registered_at="t"),),
        )
        assert LabelVersionHistory.from_json_dict(history.to_json_dict()) == history

    def test_empty_history_round_trips(self) -> None:
        history = LabelVersionHistory(schema_version=1, label_family="direction", versions=())
        assert LabelVersionHistory.from_json_dict(history.to_json_dict()) == history
