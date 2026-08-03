from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelReconciliationError, LabelRequestError
from quant_platform.labels.builder import LabelBuilder, LabelDefinition
from quant_platform.labels.composite import (
    build_composite_from_definitions,
    build_composite_label_bundle,
    reconcile_composite,
    replay_composite,
    verify_composite,
)
from quant_platform.labels.direction import build_direction_specification, generate_direction_labels
from quant_platform.labels.forward_volatility import (
    build_forward_volatility_specification,
    generate_forward_volatility_labels,
)
from quant_platform.labels.manifest import build_label_manifest
from quant_platform.labels.next_return import build_next_return_specification, generate_next_return_labels
from quant_platform.labels.pricing import PriceBasis
from quant_platform.labels.triple_barrier import (
    build_triple_barrier_specification,
    generate_triple_barrier_labels,
)
from quant_platform.labels.volatility import REALIZED_STDDEV_ESTIMATOR_NAME
from quant_platform.ml.persistence import format_utc_timestamp, utc_now


def _return_definition() -> LabelDefinition:
    spec = build_next_return_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1")
    return LabelDefinition(specification=spec, generate=generate_next_return_labels)


def _direction_definition() -> LabelDefinition:
    spec = build_direction_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, neutral_threshold=0.001, created_from_dataset="ds1", created_from_manifest="m1")
    return LabelDefinition(specification=spec, generate=generate_direction_labels)


def _volatility_definition() -> LabelDefinition:
    spec = build_forward_volatility_specification(horizon_bars=10, volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1")
    return LabelDefinition(specification=spec, generate=generate_forward_volatility_labels)


def _triple_barrier_definition() -> LabelDefinition:
    spec = build_triple_barrier_specification(
        profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
        volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
    )
    return LabelDefinition(specification=spec, generate=generate_triple_barrier_labels)


class TestBuildCompositeFromDefinitions:
    @pytest.mark.parametrize(
        "definitions", [
            (_return_definition(), _direction_definition()),
            (_return_definition(), _volatility_definition()),
            (_direction_definition(), _triple_barrier_definition()),
            (_return_definition(), _direction_definition(), _volatility_definition()),
        ],
        ids=["return+direction", "return+volatility", "direction+triple_barrier", "return+direction+volatility"],
    )
    def test_named_example_combinations_build_successfully(self, definitions, ohlcv_source_data: pd.DataFrame) -> None:
        composite = build_composite_from_definitions(definitions, ohlcv_source_data, dataset_id="ds1", source_content_id="src1")
        assert len(composite.members) == len(definitions)
        consistent, issues = composite.verify_self_consistency()
        assert consistent is True
        assert issues == ()

    def test_members_sorted_deterministically(self, ohlcv_source_data: pd.DataFrame) -> None:
        definitions = (_direction_definition(), _return_definition())
        composite = build_composite_from_definitions(definitions, ohlcv_source_data, dataset_id="ds1", source_content_id="src1")
        ids = [m.specification.label_specification_id for m in composite.members]
        assert ids == sorted(ids)

    def test_empty_members_rejected(self) -> None:
        with pytest.raises(LabelRequestError):
            build_composite_label_bundle("ds1", ())

    def test_duplicate_specification_rejected(self, ohlcv_source_data: pd.DataFrame) -> None:
        definition = _return_definition()
        bundle = LabelBuilder().build(definition, ohlcv_source_data, source_content_id="src1")
        with pytest.raises(LabelRequestError):
            build_composite_label_bundle("ds1", (bundle, bundle))

    def test_two_builds_from_identical_definitions_have_identical_composite_id(self, ohlcv_source_data: pd.DataFrame) -> None:
        definitions = (_return_definition(), _direction_definition())
        a = build_composite_from_definitions(definitions, ohlcv_source_data, dataset_id="ds1", source_content_id="src1")
        b = build_composite_from_definitions(definitions, ohlcv_source_data, dataset_id="ds1", source_content_id="src1")
        assert a.composite_id == b.composite_id


class TestVerifyReplayReconcileComposite:
    def test_verify_composite_clean(self, ohlcv_source_data: pd.DataFrame) -> None:
        definitions = (_return_definition(), _direction_definition())
        composite = build_composite_from_definitions(definitions, ohlcv_source_data, dataset_id="ds1", source_content_id="src1")
        manifests = tuple(
            build_label_manifest(d.specification, generation_timestamp=format_utc_timestamp(utc_now())) for d in definitions
        )
        result = verify_composite(composite, manifests, definitions, ohlcv_source_data, source_content_id="src1")
        assert result.verified is True
        assert len(result.member_results) == 2

    def test_replay_composite_clean(self, ohlcv_source_data: pd.DataFrame) -> None:
        definitions = (_return_definition(), _direction_definition())
        composite = build_composite_from_definitions(definitions, ohlcv_source_data, dataset_id="ds1", source_content_id="src1")
        result = replay_composite(composite, definitions, ohlcv_source_data, source_content_id="src1")
        assert result.replayed is True

    def test_reconcile_composite_self_is_clean(self, ohlcv_source_data: pd.DataFrame) -> None:
        definitions = (_return_definition(), _direction_definition())
        composite = build_composite_from_definitions(definitions, ohlcv_source_data, dataset_id="ds1", source_content_id="src1")
        manifests = tuple(
            build_label_manifest(d.specification, generation_timestamp=format_utc_timestamp(utc_now())) for d in definitions
        )
        result = reconcile_composite(composite, composite, baseline_manifests=manifests, candidate_manifests=manifests)
        assert result.reconciled is True
        assert result.issues == ()

    def test_reconcile_composite_different_dataset_raises(self, ohlcv_source_data: pd.DataFrame) -> None:
        definitions = (_return_definition(),)
        composite = build_composite_from_definitions(definitions, ohlcv_source_data, dataset_id="ds1", source_content_id="src1")
        other = replace(composite, dataset_id="ds2")
        manifests = (build_label_manifest(definitions[0].specification, generation_timestamp=format_utc_timestamp(utc_now())),)
        with pytest.raises(LabelReconciliationError):
            reconcile_composite(composite, other, baseline_manifests=manifests, candidate_manifests=manifests)

    def test_reconcile_composite_member_set_drift_detected(self, ohlcv_source_data: pd.DataFrame) -> None:
        baseline_definitions = (_return_definition(), _direction_definition())
        baseline = build_composite_from_definitions(baseline_definitions, ohlcv_source_data, dataset_id="ds1", source_content_id="src1")
        candidate_definitions = (_return_definition(),)
        candidate = build_composite_from_definitions(candidate_definitions, ohlcv_source_data, dataset_id="ds1", source_content_id="src1")
        baseline_manifests = tuple(build_label_manifest(d.specification, generation_timestamp=format_utc_timestamp(utc_now())) for d in baseline_definitions)
        candidate_manifests = tuple(build_label_manifest(d.specification, generation_timestamp=format_utc_timestamp(utc_now())) for d in candidate_definitions)
        result = reconcile_composite(baseline, candidate, baseline_manifests=baseline_manifests, candidate_manifests=candidate_manifests)
        assert result.reconciled is False
        assert any(i.kind == "member_set_drift" for i in result.issues)


class TestCompositeJsonRoundTrip:
    def test_round_trip(self, ohlcv_source_data: pd.DataFrame) -> None:
        from quant_platform.labels.composite import CompositeLabelBundle

        definitions = (_return_definition(), _direction_definition())
        composite = build_composite_from_definitions(definitions, ohlcv_source_data, dataset_id="ds1", source_content_id="src1")
        restored = CompositeLabelBundle.from_json_dict(composite.to_json_dict())
        assert restored.composite_id == composite.composite_id
        assert restored.dataset_id == composite.dataset_id
        assert len(restored.members) == len(composite.members)
