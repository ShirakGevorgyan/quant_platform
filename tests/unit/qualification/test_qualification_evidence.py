from __future__ import annotations

from quant_platform.historical.quality import Severity
from quant_platform.qualification.evidence import Evidence, affected_split, make_evidence
from quant_platform.qualification.models import QualificationDimensionKind


class TestAffectedSplit:
    def test_is_stable_and_textually_canonical(self) -> None:
        a = affected_split("dataset123", "content456", "train")
        b = affected_split("dataset123", "content456", "train")
        assert a == b
        assert a == "dataset_id=dataset123 content_id=content456 split=train"


class TestEvidence:
    def test_json_round_trip(self) -> None:
        evidence = make_evidence(
            finding="NaN found", evidence=("nan_counts={'trend': 3}",), severity=Severity.WARNING,
            dimension=QualificationDimensionKind.STATISTICAL_INTEGRITY, affected_artifacts=("dataset_id=abc split=train",),
            recommendation="Investigate", blocking=False,
        )
        restored = Evidence.from_json_dict(evidence.to_json_dict())
        assert restored == evidence

    def test_blocking_defaults_to_false(self) -> None:
        evidence = make_evidence(
            finding="ok", evidence=(), severity=Severity.INFO, dimension=QualificationDimensionKind.SAFETY,
            affected_artifacts=(),
        )
        assert evidence.blocking is False
        assert evidence.recommendation is None
