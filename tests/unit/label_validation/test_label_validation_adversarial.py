"""Milestone 11, Phase 3, Part C: the 14 named adversarial scenarios,
each run against the real infrastructure -- never a mock. One test
class per named item, for direct, unambiguous traceability against the
governing specification's own list."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.exceptions import LabelValidationRequestError
from quant_platform.label_validation.degeneracy import compute_label_degeneracy, detect_duplicate_labels
from quant_platform.label_validation.engine import LabelQualificationDecision, LabelQualificationEngine
from quant_platform.label_validation.horizon import compare_horizons
from quant_platform.label_validation.leakage import validate_leakage
from quant_platform.label_validation.replay import LabelValidationReplay
from quant_platform.labels.builder import LabelBundle, LabelDefinition
from quant_platform.labels.identity import compute_label_identity
from quant_platform.labels.manifest import LabelManifest
from quant_platform.labels.records import LabelRecord


class Test01ConstantLabels:
    def test_constant_bundle_is_rejected(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest, rebuild_bundle_with_values_fn) -> None:
        constant_bundle = rebuild_bundle_with_values_fn(next_return_bundle, pd.Series(np.full(next_return_bundle.row_count, 0.5)))
        report = LabelQualificationEngine().qualify(constant_bundle, next_return_manifest)
        assert report.decision is LabelQualificationDecision.REJECTED
        assert any("constant_labels" in reason for reason in report.blocking_reasons)


class Test02AllNeutral:
    def test_all_neutral_direction_bundle_is_rejected(self, direction_bundle: LabelBundle, direction_manifest: LabelManifest, rebuild_bundle_with_values_fn) -> None:
        all_neutral_values = direction_bundle.values.where(direction_bundle.values.isna(), 0.0)
        all_neutral_bundle = rebuild_bundle_with_values_fn(direction_bundle, all_neutral_values)
        report = LabelQualificationEngine().qualify(all_neutral_bundle, direction_manifest)
        assert report.decision is LabelQualificationDecision.REJECTED
        degeneracy = compute_label_degeneracy(all_neutral_bundle)
        assert degeneracy.is_all_neutral is True


class Test03EmptyLabels:
    def test_empty_bundle_is_rejected(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest, rebuild_bundle_with_values_fn) -> None:
        empty_bundle = rebuild_bundle_with_values_fn(next_return_bundle, pd.Series(np.full(next_return_bundle.row_count, np.nan)))
        report = LabelQualificationEngine().qualify(empty_bundle, next_return_manifest)
        assert report.decision is LabelQualificationDecision.REJECTED
        assert any("empty_labels" in reason for reason in report.blocking_reasons)


class Test04DuplicateLabels:
    def test_label_id_collision_detected(self, next_return_records: tuple[LabelRecord, ...]) -> None:
        colliding = replace(next_return_records[1], label_id=next_return_records[0].label_id)
        records_with_collision = (next_return_records[0], colliding, *next_return_records[2:])
        duplicates = detect_duplicate_labels(records_with_collision)
        assert duplicates == (next_return_records[0].label_id,)

    def test_healthy_records_have_no_duplicates(self, next_return_records: tuple[LabelRecord, ...]) -> None:
        assert detect_duplicate_labels(next_return_records) == ()


class Test05FutureTimestamps:
    def test_tampered_record_self_consistency_is_blocking(
        self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest, next_return_records: tuple[LabelRecord, ...],
    ) -> None:
        far_future = pd.Timestamp("2099-01-01T00:00:00+00:00").isoformat()
        tampered = replace(next_return_records[0], event_time=far_future)
        tampered_records = (tampered, *next_return_records[1:])
        result = validate_leakage(next_return_bundle, next_return_manifest, records=tampered_records)
        assert result.records_self_consistent is False
        assert result.is_blocking is True


class Test06FutureMacro:
    def test_disclosed_out_of_scope(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        result = validate_leakage(next_return_bundle, next_return_manifest)
        assert any("macro" in e.finding.lower() for e in result.evidence)


class Test07FutureCrossAsset:
    def test_disclosed_out_of_scope(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        result = validate_leakage(next_return_bundle, next_return_manifest)
        assert any("cross-asset" in e.finding.lower() for e in result.evidence)


class Test08TamperedManifests:
    def test_tampered_checksum_is_blocking(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        tampered_manifest = replace(next_return_manifest, manifest_checksum="0" * 64)
        result = validate_leakage(next_return_bundle, tampered_manifest)
        assert result.manifest_self_consistent is False
        assert result.is_blocking is True

    def test_mismatched_specification_id_is_blocking(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        wrong_manifest = replace(next_return_manifest, label_specification_id="a-different-specification")
        result = validate_leakage(next_return_bundle, wrong_manifest)
        assert result.is_blocking is True


class Test09TamperedIdentities:
    def test_tampered_content_id_is_blocking(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        tampered_identity = replace(next_return_bundle.identity, content_id="0" * 64)
        tampered_bundle = replace(next_return_bundle, identity=tampered_identity)
        result = validate_leakage(tampered_bundle, next_return_manifest)
        assert result.identity_consistent is False
        assert result.is_blocking is True

    def test_untampered_identity_matches_a_fresh_recomputation(self, next_return_bundle: LabelBundle) -> None:
        recomputed = compute_label_identity(
            next_return_bundle.specification.label_specification_id, next_return_bundle.values, source_content_id=next_return_bundle.identity.source_content_id,
        )
        assert recomputed.content_id == next_return_bundle.identity.content_id


class Test10DistributionCorruption:
    def test_corrupted_values_are_reflected_in_a_fresh_distribution(self, next_return_bundle: LabelBundle, rebuild_bundle_with_values_fn) -> None:
        from quant_platform.label_validation.distribution import compute_label_distribution

        baseline = compute_label_distribution(next_return_bundle)
        corrupted_values = next_return_bundle.values.copy()
        corrupted_values.iloc[: len(corrupted_values) // 2] = 999.0
        corrupted_bundle = rebuild_bundle_with_values_fn(next_return_bundle, corrupted_values)
        corrupted = compute_label_distribution(corrupted_bundle)
        assert corrupted.class_ratios != baseline.class_ratios


class Test11BarrierCorruption:
    def test_out_of_domain_triple_barrier_value_is_blocking(
        self, triple_barrier_bundle: LabelBundle, triple_barrier_manifest: LabelManifest, rebuild_bundle_with_values_fn,
    ) -> None:
        tampered_values = triple_barrier_bundle.values.copy()
        first_valid = tampered_values.first_valid_index()
        tampered_values.loc[first_valid] = 42.0
        tampered_bundle = rebuild_bundle_with_values_fn(triple_barrier_bundle, tampered_values)
        result = validate_leakage(tampered_bundle, triple_barrier_manifest)
        assert result.barrier_domain_valid is False
        assert result.is_blocking is True


class Test12AvailabilityCorruption:
    def test_availability_before_event_time_is_blocking(
        self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest, next_return_records: tuple[LabelRecord, ...],
    ) -> None:
        tampered = replace(next_return_records[0], availability_time="2020-01-01T00:00:00+00:00")
        tampered_records = (tampered, *next_return_records[1:])
        result = validate_leakage(next_return_bundle, next_return_manifest, records=tampered_records)
        assert result.availability_time_consistent is False
        assert result.is_blocking is True


class Test13ReplayCorruption:
    def test_corrupted_source_data_diverges_from_the_original_qualification(
        self, next_return_definition: LabelDefinition, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest,
        ohlcv_source_data: pd.DataFrame, source_content_id: str,
    ) -> None:
        # A rescale (e.g. close * 5 + 1000) leaves the label DISTRIBUTION
        # SHAPE -- and therefore the quality verdict -- unchanged, since
        # qualification is a shape-level judgement, not a byte-level
        # comparison; that is not a meaningful "corruption" for this
        # check. Flattening price to a constant, by contrast, collapses
        # next-return labels toward zero and genuinely changes label
        # QUALITY (degeneracy/balance), which replay must catch.
        original_report = LabelQualificationEngine().qualify(next_return_bundle, next_return_manifest)
        corrupted_source = ohlcv_source_data.copy()
        corrupted_source["close"] = corrupted_source["close"].iloc[0]
        result = LabelValidationReplay().replay_and_requalify(
            next_return_definition, corrupted_source, source_content_id=source_content_id, manifest=next_return_manifest, original_report=original_report,
        )
        assert result.qualification_identical is False
        assert result.issues != ()
        assert result.replayed_decision != result.original_decision


class Test14HorizonCorruption:
    def test_specification_missing_a_horizon_parameter_fails_closed(self, next_return_bundle: LabelBundle) -> None:
        stripped_parameters = {k: v for k, v in next_return_bundle.specification.parameters.items() if k != "horizon_bars"}
        stripped_spec = replace(next_return_bundle.specification, parameters=stripped_parameters)
        stripped_bundle = replace(next_return_bundle, specification=stripped_spec)
        with pytest.raises(LabelValidationRequestError):
            compare_horizons((stripped_bundle,))
