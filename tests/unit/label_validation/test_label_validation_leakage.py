from __future__ import annotations

from dataclasses import replace

from quant_platform.label_validation.leakage import LeakageValidationResult, validate_leakage
from quant_platform.labels.builder import LabelBundle
from quant_platform.labels.manifest import LabelManifest
from quant_platform.labels.records import LabelRecord


class TestValidateLeakageHealthyBundle:
    def test_trailing_tail_is_well_formed(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        result = validate_leakage(next_return_bundle, next_return_manifest)
        assert result.trailing_nan_tail_well_formed is True
        assert result.is_blocking is False

    def test_with_records_availability_consistent(
        self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest, next_return_records: tuple[LabelRecord, ...],
    ) -> None:
        result = validate_leakage(next_return_bundle, next_return_manifest, records=next_return_records)
        assert result.availability_time_consistent is True

    def test_no_records_leaves_availability_none(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        result = validate_leakage(next_return_bundle, next_return_manifest)
        assert result.availability_time_consistent is None

    def test_discloses_macro_cross_asset_out_of_scope(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        result = validate_leakage(next_return_bundle, next_return_manifest)
        assert any("macro" in e.finding.lower() and "cross-asset" in e.finding.lower() for e in result.evidence)

    def test_json_round_trip(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        result = validate_leakage(next_return_bundle, next_return_manifest)
        restored = LeakageValidationResult.from_json_dict(result.to_json_dict())
        assert restored == result


class TestValidateLeakageManifestMismatch:
    def test_mismatched_manifest_is_blocking(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        wrong_manifest = replace(next_return_manifest, label_specification_id="some-other-spec")
        result = validate_leakage(next_return_bundle, wrong_manifest)
        assert result.is_blocking is True


class TestValidateLeakageAvailabilityViolation:
    def test_availability_before_event_time_is_blocking(
        self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest, next_return_records: tuple[LabelRecord, ...],
    ) -> None:
        tampered = replace(next_return_records[0], availability_time="2020-01-01T00:00:00+00:00")
        tampered_records = (tampered, *next_return_records[1:])
        result = validate_leakage(next_return_bundle, next_return_manifest, records=tampered_records)
        assert result.availability_time_consistent is False
        assert result.is_blocking is True


class TestValidateLeakageBarrierViolation:
    def test_out_of_domain_triple_barrier_value_is_blocking(self, triple_barrier_bundle: LabelBundle, triple_barrier_manifest: LabelManifest) -> None:
        tampered_values = triple_barrier_bundle.values.copy()
        first_valid = tampered_values.first_valid_index()
        tampered_values.loc[first_valid] = 99.0
        tampered_bundle = replace(triple_barrier_bundle, values=tampered_values)
        result = validate_leakage(tampered_bundle, triple_barrier_manifest)
        assert result.barrier_domain_valid is False
        assert result.is_blocking is True

    def test_healthy_triple_barrier_is_valid(self, triple_barrier_bundle: LabelBundle, triple_barrier_manifest: LabelManifest) -> None:
        result = validate_leakage(triple_barrier_bundle, triple_barrier_manifest)
        assert result.barrier_domain_valid is True

    def test_non_triple_barrier_family_skips_barrier_check(self, next_return_bundle: LabelBundle, next_return_manifest: LabelManifest) -> None:
        result = validate_leakage(next_return_bundle, next_return_manifest)
        assert result.barrier_domain_valid is True
