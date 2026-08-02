from __future__ import annotations

from dataclasses import replace

import pytest

from quant_platform.core.exceptions import (
    DuplicateLabelSpecificationError,
    LabelIdentityError,
    UnknownLabelSpecificationError,
)
from quant_platform.labels.models import LabelFamily, LabelSpecification
from quant_platform.labels.registry import LabelRegistry


class TestRegister:
    def test_register_then_lookup(self, specification: LabelSpecification) -> None:
        registry = LabelRegistry()
        registry.register(specification)
        assert registry.lookup(specification.label_specification_id) == specification
        assert len(registry) == 1
        assert specification.label_specification_id in registry

    def test_duplicate_registration_refused(self, specification: LabelSpecification) -> None:
        registry = LabelRegistry()
        registry.register(specification)
        with pytest.raises(DuplicateLabelSpecificationError):
            registry.register(specification)

    def test_tampered_specification_refused_at_registration(self, specification: LabelSpecification) -> None:
        tampered = replace(specification, label_specification_id="tampered-id")
        registry = LabelRegistry()
        with pytest.raises(LabelIdentityError):
            registry.register(tampered)

    def test_unknown_lookup_raises(self) -> None:
        registry = LabelRegistry()
        with pytest.raises(UnknownLabelSpecificationError):
            registry.lookup("never-registered")


class TestFreeze:
    def test_freeze_then_is_frozen(self, specification: LabelSpecification) -> None:
        registry = LabelRegistry()
        registry.register(specification)
        assert registry.is_frozen(specification.label_specification_id) is False
        registry.freeze(specification.label_specification_id)
        assert registry.is_frozen(specification.label_specification_id) is True

    def test_freeze_unknown_raises(self) -> None:
        registry = LabelRegistry()
        with pytest.raises(UnknownLabelSpecificationError):
            registry.freeze("never-registered")


class TestVersions:
    def test_versions_filtered_by_family(self, specification: LabelSpecification, other_family_specification: LabelSpecification) -> None:
        registry = LabelRegistry()
        registry.register(specification)
        registry.register(other_family_specification)
        history = registry.versions(LabelFamily.NEXT_RETURN)
        assert history.label_family == "next_return"
        assert len(history.versions) == 1
        assert history.versions[0].label_specification_id == specification.label_specification_id

    def test_versions_empty_for_unregistered_family(self) -> None:
        registry = LabelRegistry()
        history = registry.versions(LabelFamily.TRIPLE_BARRIER)
        assert history.versions == ()


class TestCompare:
    def test_identical_specifications_compare_equal(self, specification: LabelSpecification) -> None:
        registry = LabelRegistry()
        registry.register(specification)
        equal, differences = registry.compare(specification.label_specification_id, specification.label_specification_id)
        assert equal is True
        assert differences == ()

    def test_different_specifications_report_differences(self, specification: LabelSpecification, other_family_specification: LabelSpecification) -> None:
        registry = LabelRegistry()
        registry.register(specification)
        registry.register(other_family_specification)
        equal, differences = registry.compare(specification.label_specification_id, other_family_specification.label_specification_id)
        assert equal is False
        assert any("label_family" in d for d in differences)


class TestVerify:
    def test_verify_clean_specification(self, specification: LabelSpecification) -> None:
        registry = LabelRegistry()
        registry.register(specification)
        consistent, issues = registry.verify(specification.label_specification_id)
        assert consistent is True
        assert issues == ()


class TestBuildManifest:
    def test_build_manifest_from_registered_specification(self, specification: LabelSpecification) -> None:
        registry = LabelRegistry()
        registry.register(specification)
        manifest = registry.build_manifest(specification.label_specification_id, feature_identity="feat-1", qualification_identity="qual-1")
        assert manifest.label_specification_id == specification.label_specification_id
        assert manifest.feature_identity == "feat-1"
        consistent, issues = manifest.verify_self_consistency()
        assert consistent is True
        assert issues == ()


class TestForDataset:
    def test_for_dataset_filters_by_created_from_dataset(self, specification: LabelSpecification) -> None:
        registry = LabelRegistry()
        registry.register(specification)
        matches = registry.for_dataset(specification.created_from_dataset)
        assert matches == (specification,)
        assert registry.for_dataset("some-other-dataset") == ()
