"""Milestone 11, Phase 2, Part 2 quality gate: repeat verification x10,
repeat reconciliation x10, repeat reports x10 -- each operation run 10
times against the real infrastructure pipeline, asserting every
repetition (after stripping the legitimately wall-clock-dependent
`generated_at`/`captured_at` fields) is byte-identical to the first."""

from __future__ import annotations

from quant_platform.feature_discovery.catalog import build_feature_infrastructure_bundle
from quant_platform.feature_discovery.infra_reconciliation import FeatureInfrastructureReconciliation
from quant_platform.feature_discovery.infra_reports import (
    render_feature_catalog_report,
    render_infrastructure_verification_report,
)
from quant_platform.feature_discovery.infra_verification import FeatureInfrastructureVerifier

_REPEAT_COUNT = 10


def _strip_volatile_fields(raw: dict) -> dict:
    raw = dict(raw)
    raw.pop("generated_at", None)
    raw.pop("captured_at", None)
    if isinstance(raw.get("snapshot"), dict):
        raw["snapshot"] = {k: v for k, v in raw["snapshot"].items() if k != "captured_at"}
    if isinstance(raw.get("manifest"), dict):
        raw["manifest"] = {k: v for k, v in raw["manifest"].items() if k != "generated_at"}
    if isinstance(raw.get("reconciliation"), dict):
        raw["reconciliation"] = {k: v for k, v in raw["reconciliation"].items() if k != "generated_at"}
    return raw


class TestRepeatVerification:
    def test_ten_repeated_verify_calls_are_identical(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        verifier = FeatureInfrastructureVerifier()
        results = [
            _strip_volatile_fields(verifier.verify(bundle, discovered_registry, discovered_manifest).to_json_dict())
            for _ in range(_REPEAT_COUNT)
        ]
        assert all(r == results[0] for r in results)
        assert results[0]["verified"] is True


class TestRepeatReconciliation:
    def test_ten_repeated_reconcile_calls_are_identical(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        reconciler = FeatureInfrastructureReconciliation()
        results = [_strip_volatile_fields(reconciler.reconcile(bundle, bundle).to_json_dict()) for _ in range(_REPEAT_COUNT)]
        assert all(r == results[0] for r in results)
        assert results[0]["reconciled"] is True


class TestRepeatReports:
    def test_ten_repeated_catalog_report_renders_are_identical(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        renders = [render_feature_catalog_report(bundle.catalog) for _ in range(_REPEAT_COUNT)]
        assert all(r == renders[0] for r in renders)

    def test_ten_repeated_verification_report_renders_are_identical_modulo_timestamp(self, discovered_registry, discovered_manifest) -> None:
        bundle = build_feature_infrastructure_bundle(discovered_registry, discovered_manifest)
        result = FeatureInfrastructureVerifier().verify(bundle, discovered_registry, discovered_manifest)
        renders = [render_infrastructure_verification_report(result) for _ in range(_REPEAT_COUNT)]
        assert all(r == renders[0] for r in renders)


class TestRepeatBundleCapture:
    """Repeat capture x10 (the underlying operation every other repeat
    test in this file builds on) -- confirms `build_feature_
    infrastructure_bundle` itself is deterministic, not merely that
    rendering/verifying an already-fixed bundle is."""

    def test_ten_repeated_bundle_captures_are_identical(self, discovered_registry, discovered_manifest) -> None:
        bundles_raw = [
            _strip_volatile_fields(build_feature_infrastructure_bundle(discovered_registry, discovered_manifest).to_json_dict())
            for _ in range(_REPEAT_COUNT)
        ]
        assert all(b == bundles_raw[0] for b in bundles_raw)
