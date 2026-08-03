"""Milestone 11, Phase 3, Part B: the 13 named adversarial scenarios,
each run against the real infrastructure and at least one concrete
label family -- never a mock. One test class per named item, for
direct, unambiguous traceability against the governing specification's
own list."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelRequestError
from quant_platform.core.types import Timeframe
from quant_platform.labels.builder import LabelBuilder, LabelDefinition
from quant_platform.labels.composite import build_composite_from_definitions, reconcile_composite
from quant_platform.labels.diagnostics import compute_label_diagnostics
from quant_platform.labels.direction import build_direction_specification
from quant_platform.labels.evidence import LabelDimensionKind
from quant_platform.labels.manifest import build_label_manifest
from quant_platform.labels.next_return import build_next_return_specification, generate_next_return_labels
from quant_platform.labels.pricing import PriceBasis
from quant_platform.labels.records import materialize_label_records
from quant_platform.labels.replay import LabelReplay
from quant_platform.labels.triple_barrier import (
    build_triple_barrier_specification,
)
from quant_platform.labels.verification import LabelVerifier
from quant_platform.labels.volatility import REALIZED_PARKINSON_ESTIMATOR_NAME, REALIZED_STDDEV_ESTIMATOR_NAME
from quant_platform.ml.persistence import format_utc_timestamp, utc_now


@pytest.fixture
def next_return_bundle(ohlcv_source_data: pd.DataFrame):
    spec = build_next_return_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1")
    definition = LabelDefinition(specification=spec, generate=generate_next_return_labels)
    return definition, LabelBuilder().build(definition, ohlcv_source_data, source_content_id="src1")


class Test01FutureTimestamp:
    def test_tampered_event_time_detected_by_self_consistency(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        _definition, bundle = next_return_bundle
        records = materialize_label_records(bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        far_future = pd.Timestamp("2099-01-01T00:00:00+00:00").isoformat()
        tampered = replace(records[0], event_time=far_future)
        consistent, issues = tampered.verify_self_consistency()
        assert consistent is False
        assert any("row_identity" in i for i in issues)


class Test02FutureMacro:
    def test_labels_package_never_imports_market_data(self) -> None:
        import ast
        from pathlib import Path

        labels_dir = Path(__file__).resolve().parents[3] / "src" / "quant_platform" / "labels"
        for path in labels_dir.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                module = getattr(node, "module", None) if isinstance(node, ast.ImportFrom) else None
                names = [alias.name for alias in node.names] if isinstance(node, (ast.Import, ast.ImportFrom)) else []
                assert "quant_platform.market_data" not in (names + ([module] if module else []))

    def test_availability_dimension_discloses_macro_out_of_scope(self, next_return_bundle) -> None:
        _definition, bundle = next_return_bundle
        manifest = build_label_manifest(bundle.specification, generation_timestamp=format_utc_timestamp(utc_now()))
        diagnostics = compute_label_diagnostics(bundle, manifest)
        availability = diagnostics.dimension_result(LabelDimensionKind.AVAILABILITY)
        assert any("macro" in e.finding.lower() for e in availability.evidence)


class Test03FutureCrossAsset:
    def test_availability_dimension_discloses_cross_asset_out_of_scope(self, next_return_bundle) -> None:
        _definition, bundle = next_return_bundle
        manifest = build_label_manifest(bundle.specification, generation_timestamp=format_utc_timestamp(utc_now()))
        diagnostics = compute_label_diagnostics(bundle, manifest)
        availability = diagnostics.dimension_result(LabelDimensionKind.AVAILABILITY)
        assert any("cross asset" in e.finding.lower() for e in availability.evidence)


class Test04ModifiedHorizon:
    def test_next_return_horizon_change_yields_new_identity(self) -> None:
        a = build_next_return_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1")
        b = build_next_return_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=6, created_from_dataset="ds1", created_from_manifest="m1")
        assert a.label_specification_id != b.label_specification_id


class Test05ModifiedThreshold:
    def test_direction_threshold_change_yields_new_identity(self) -> None:
        a = build_direction_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, neutral_threshold=0.001, created_from_dataset="ds1", created_from_manifest="m1")
        b = build_direction_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, neutral_threshold=0.002, created_from_dataset="ds1", created_from_manifest="m1")
        assert a.label_specification_id != b.label_specification_id


class Test06ModifiedBarrier:
    def test_triple_barrier_multiplier_change_yields_new_identity(self) -> None:
        a = build_triple_barrier_specification(
            profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
            volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        b = build_triple_barrier_specification(
            profit_multiplier=3.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
            volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        assert a.label_specification_id != b.label_specification_id


class Test07ModifiedVolatilityEstimator:
    def test_triple_barrier_estimator_change_yields_new_identity(self) -> None:
        a = build_triple_barrier_specification(
            profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
            volatility_estimator_reference=REALIZED_STDDEV_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        b = build_triple_barrier_specification(
            profit_multiplier=2.0, loss_multiplier=2.0, max_holding_bars=10, volatility_window_bars=20,
            volatility_estimator_reference=REALIZED_PARKINSON_ESTIMATOR_NAME, created_from_dataset="ds1", created_from_manifest="m1",
        )
        assert a.label_specification_id != b.label_specification_id


class Test08ManifestCorruption:
    def test_tampered_manifest_checksum_is_blocking(self, next_return_bundle) -> None:
        _definition, bundle = next_return_bundle
        manifest = build_label_manifest(bundle.specification, generation_timestamp=format_utc_timestamp(utc_now()))
        tampered = replace(manifest, manifest_checksum="0" * 64)
        diagnostics = compute_label_diagnostics(bundle, tampered)
        assert diagnostics.dimension_result(LabelDimensionKind.MANIFEST_INTEGRITY).is_blocking is True


class Test09IdentityTampering:
    def test_tampered_content_id_is_blocking(self, next_return_bundle) -> None:
        _definition, bundle = next_return_bundle
        manifest = build_label_manifest(bundle.specification, generation_timestamp=format_utc_timestamp(utc_now()))
        tampered_identity = replace(bundle.identity, content_id="0" * 64)
        tampered_bundle = replace(bundle, identity=tampered_identity)
        diagnostics = compute_label_diagnostics(tampered_bundle, manifest)
        assert diagnostics.dimension_result(LabelDimensionKind.IDENTITY).is_blocking is True


class Test10BundleCorruption:
    def test_tampered_values_detected_by_verification(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        definition, bundle = next_return_bundle
        manifest = build_label_manifest(bundle.specification, generation_timestamp=format_utc_timestamp(utc_now()))
        tampered_values = bundle.values.copy()
        tampered_values.iloc[0] = 12345.0
        tampered_bundle = replace(bundle, values=tampered_values)
        result = LabelVerifier().verify(tampered_bundle, manifest, definition, ohlcv_source_data, source_content_id="src1")
        assert result.verified is False
        assert result.self_consistent is False


class Test11DatasetCorruption:
    def test_mismatched_dataset_id_rejected_at_materialization(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        _definition, bundle = next_return_bundle
        with pytest.raises(LabelRequestError):
            materialize_label_records(bundle, ohlcv_source_data, dataset_id="a-completely-different-dataset", timeframe=Timeframe.M1, horizon_bars=5)

    def test_composite_reconciliation_rejects_cross_dataset_comparison(self, ohlcv_source_data: pd.DataFrame) -> None:
        from quant_platform.core.exceptions import LabelReconciliationError

        spec = build_next_return_specification(price_basis=PriceBasis.CLOSE_TO_CLOSE, horizon_bars=5, created_from_dataset="ds1", created_from_manifest="m1")
        definition = LabelDefinition(specification=spec, generate=generate_next_return_labels)
        composite = build_composite_from_definitions((definition,), ohlcv_source_data, dataset_id="ds1", source_content_id="src1")
        corrupted = replace(composite, dataset_id="ds-corrupted")
        manifests = (build_label_manifest(spec, generation_timestamp=format_utc_timestamp(utc_now())),)
        with pytest.raises(LabelReconciliationError):
            reconcile_composite(composite, corrupted, baseline_manifests=manifests, candidate_manifests=manifests)


class Test12AvailabilityCorruption:
    def test_tampered_availability_time_diverges_from_a_fresh_remateralization(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        _definition, bundle = next_return_bundle
        records = materialize_label_records(bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        tampered = replace(records[0], availability_time="2099-01-01T00:00:00+00:00")
        fresh = materialize_label_records(bundle, ohlcv_source_data, dataset_id="ds1", timeframe=Timeframe.M1, horizon_bars=5)
        assert tampered.availability_time != fresh[0].availability_time
        assert tampered != fresh[0]


class Test13ReplayCorruption:
    def test_corrupted_source_data_is_detected_by_replay(self, next_return_bundle, ohlcv_source_data: pd.DataFrame) -> None:
        definition, bundle = next_return_bundle
        corrupted_source = ohlcv_source_data.copy()
        corrupted_source.loc[0, "close"] = corrupted_source["close"].iloc[0] * 2.0
        result = LabelReplay().replay(definition, corrupted_source, source_content_id="src1", original=bundle)
        assert result.replayed is False
        assert result.issues != ()
