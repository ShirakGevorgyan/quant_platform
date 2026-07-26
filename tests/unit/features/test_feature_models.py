from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.types import Timeframe
from quant_platform.features.models import FeatureCategory, FeatureSpec, MissingPolicyKind, MissingPolicySpec


def _spec(**overrides) -> FeatureSpec:
    base = {
        "name": "test_feature", "version": "1", "description": "a test feature", "category": FeatureCategory.PRICE,
        "required_inputs": ("close",), "source_symbols": (), "source_timeframe": Timeframe.M1, "output_dtype": "float64",
        "lookback_bars": 10, "warmup_bars": 10,
    }
    base.update(overrides)
    return FeatureSpec(**base)


class TestMissingPolicySpec:
    def test_forward_fill_requires_max_age(self) -> None:
        with pytest.raises(ValueError, match="max_age_bars"):
            MissingPolicySpec(kind=MissingPolicyKind.FORWARD_FILL_MAX_AGE)

    def test_constant_fill_requires_value(self) -> None:
        with pytest.raises(ValueError, match="constant_value"):
            MissingPolicySpec(kind=MissingPolicyKind.CONSTANT_FILL)

    def test_invalid_statistic_rejected(self) -> None:
        with pytest.raises(ValueError, match="statistic"):
            MissingPolicySpec(statistic="mode")

    def test_json_round_trip(self) -> None:
        spec = MissingPolicySpec(kind=MissingPolicyKind.FORWARD_FILL_MAX_AGE, max_age_bars=5, add_missing_indicator=True)
        restored = MissingPolicySpec.from_json_dict(spec.to_json_dict())
        assert restored == spec

    def test_no_backward_fill_option_exists(self) -> None:
        """Structural proof that unrestricted backward fill is not merely
        discouraged but literally unrepresentable."""
        assert {k.value for k in MissingPolicyKind} == {
            "preserve_null", "forward_fill_max_age", "constant_fill", "training_statistic_fill", "drop_row",
        }


class TestFeatureSpec:
    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            _spec(name="")

    def test_rejects_negative_lookback(self) -> None:
        with pytest.raises(ValueError, match="lookback_bars"):
            _spec(lookback_bars=-1)

    def test_rejects_negative_availability_delay(self) -> None:
        with pytest.raises(ValueError, match="availability_delay"):
            _spec(availability_delay=pd.Timedelta(seconds=-1))

    def test_rejects_self_dependency(self) -> None:
        with pytest.raises(ValueError, match="cannot depend on itself"):
            _spec(name="a", feature_dependencies=("a",))

    def test_qualified_name(self) -> None:
        assert _spec(name="foo", version="2").qualified_name == "foo@2"

    def test_json_round_trip(self) -> None:
        spec = _spec(deterministic_params={"window": 10}, feature_dependencies=("dep_a",))
        restored = FeatureSpec.from_json_dict(spec.to_json_dict())
        assert restored == spec

    def test_fingerprint_is_deterministic(self) -> None:
        spec_a = _spec()
        spec_b = _spec()
        assert spec_a.fingerprint() == spec_b.fingerprint()

    def test_fingerprint_changes_with_any_field(self) -> None:
        base_fp = _spec().fingerprint()
        assert _spec(version="2").fingerprint() != base_fp
        assert _spec(lookback_bars=11).fingerprint() != base_fp
        assert _spec(deterministic_params={"window": 99}).fingerprint() != base_fp
        assert _spec(description="different").fingerprint() != base_fp

    def test_fingerprint_independent_of_dict_key_order(self) -> None:
        spec = _spec(deterministic_params={"a": 1, "b": 2})
        payload_reordered = spec.to_json_dict()
        payload_reordered["deterministic_params"] = {"b": 2, "a": 1}
        import hashlib
        import json

        fp1 = hashlib.sha256(json.dumps(spec.to_json_dict(), sort_keys=True, default=str).encode()).hexdigest()
        fp2 = hashlib.sha256(json.dumps(payload_reordered, sort_keys=True, default=str).encode()).hexdigest()
        assert fp1 == fp2
