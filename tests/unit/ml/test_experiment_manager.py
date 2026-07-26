from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from tests.unit.ml.conftest import (
    FEATURE_REGISTRY_FINGERPRINT,
    build_registry,
    make_dataset_manifest,
    make_experiment_spec_kwargs,
)

from quant_platform.core.exceptions import UnknownModelDefinitionError
from quant_platform.features.manifests import ResearchManifestStore
from quant_platform.ml.experiment_manager import ExperimentPreparer
from quant_platform.ml.experiment_spec import ExperimentSpec
from quant_platform.ml.models import ExperimentStatus, FeatureBinding
from quant_platform.ml.tracking import EventType


@pytest.fixture
def research_store(tmp_path: Path) -> ResearchManifestStore:
    store = ResearchManifestStore(tmp_path / "research")
    store.save(make_dataset_manifest())
    return store


@pytest.fixture
def preparer(tmp_path: Path, research_store: ResearchManifestStore) -> ExperimentPreparer:
    return ExperimentPreparer(
        ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=build_registry(),
        research_manifest_store=research_store,
    )


def _spec(**overrides: object) -> ExperimentSpec:
    return ExperimentSpec(**make_experiment_spec_kwargs(**overrides))


class TestPrepareHappyPath:
    def test_prepare_reaches_ready(self, preparer: ExperimentPreparer) -> None:
        manifest = preparer.prepare(_spec())
        assert manifest.status is ExperimentStatus.READY
        assert manifest.validation_report_reference is not None
        assert len(manifest.artifact_references) == 1

    def test_prepare_writes_expected_events(self, preparer: ExperimentPreparer) -> None:
        manifest = preparer.prepare(_spec())
        events = preparer.event_store.read_events(manifest.identity.experiment_id)
        assert [e.event_type for e in events] == [
            EventType.EXPERIMENT_CREATED, EventType.VALIDATION_STARTED, EventType.VALIDATION_PASSED,
        ]

    def test_validation_report_artifact_is_readable(self, preparer: ExperimentPreparer) -> None:
        import json

        manifest = preparer.prepare(_spec())
        assert manifest.validation_report_reference is not None
        raw = preparer.artifact_store.read_artifact(manifest.validation_report_reference.content_hash)
        report = json.loads(raw)
        assert report["schema_version"] == 1
        assert any(i["severity"] in ("error", "critical") for i in report["issues"]) is False


class TestIdempotency:
    def test_repeated_prepare_returns_same_manifest(self, preparer: ExperimentPreparer) -> None:
        m1 = preparer.prepare(_spec())
        m2 = preparer.prepare(_spec())
        assert m1 == m2

    def test_repeated_prepare_with_different_notes_keeps_original(self, preparer: ExperimentPreparer) -> None:
        m1 = preparer.prepare(_spec(notes="first"))
        m2 = preparer.prepare(_spec(notes="second, completely different"))
        assert m1.identity.experiment_id == m2.identity.experiment_id
        assert m2.spec.notes == "first"

    def test_repeated_prepare_does_not_duplicate_events(self, preparer: ExperimentPreparer) -> None:
        manifest = preparer.prepare(_spec())
        preparer.prepare(_spec())
        events = preparer.event_store.read_events(manifest.identity.experiment_id)
        assert len(events) == 3  # not 6 -- second call is a pure no-op read

    def test_different_specs_produce_different_experiments(self, preparer: ExperimentPreparer) -> None:
        m1 = preparer.prepare(_spec())
        m2 = preparer.prepare(_spec(hyperparameters=replace(_spec().hyperparameters, values={"alpha": 0.99})))
        assert m1.identity.experiment_id != m2.identity.experiment_id


class TestHardFailures:
    def test_unknown_model_raises_before_any_manifest_created(self, preparer: ExperimentPreparer) -> None:
        spec = _spec(model_name="does_not_exist")
        with pytest.raises(UnknownModelDefinitionError):
            preparer.prepare(spec)
        from quant_platform.ml.experiment_identity import compute_experiment_identity

        assert not preparer.manifest_store.exists(compute_experiment_identity(spec).experiment_id)

    def test_unknown_dataset_raises(self, tmp_path: Path) -> None:
        empty_research_store = ResearchManifestStore(tmp_path / "empty_research")
        preparer = ExperimentPreparer(
            ml_artifacts_root=tmp_path / "ml_artifacts", model_registry=build_registry(),
            research_manifest_store=empty_research_store,
        )
        with pytest.raises(Exception, match="Research dataset manifest version"):
            preparer.prepare(_spec())


class TestValidationFailurePath:
    def test_validation_failure_transitions_to_failed_with_summary(self, preparer: ExperimentPreparer) -> None:
        bad_spec = _spec(feature_binding=FeatureBinding(
            feature_names=("rsi_14", "atr_14"), feature_versions={"atr_14": "1", "rsi_14": "1"},
            feature_registry_fingerprint=FEATURE_REGISTRY_FINGERPRINT,
        ))
        manifest = preparer.prepare(bad_spec)
        assert manifest.status is ExperimentStatus.FAILED
        assert manifest.failure_summary
        assert manifest.completed_at is not None

    def test_validation_failure_writes_expected_events(self, preparer: ExperimentPreparer) -> None:
        bad_spec = _spec(feature_binding=FeatureBinding(
            feature_names=("rsi_14", "atr_14"), feature_versions={"atr_14": "1", "rsi_14": "1"},
            feature_registry_fingerprint=FEATURE_REGISTRY_FINGERPRINT,
        ))
        manifest = preparer.prepare(bad_spec)
        events = preparer.event_store.read_events(manifest.identity.experiment_id)
        assert [e.event_type for e in events] == [
            EventType.EXPERIMENT_CREATED, EventType.VALIDATION_STARTED, EventType.VALIDATION_FAILED,
        ]

    def test_failed_experiment_cannot_be_re_prepared_into_ready(self, preparer: ExperimentPreparer) -> None:
        bad_spec = _spec(feature_binding=FeatureBinding(
            feature_names=("rsi_14", "atr_14"), feature_versions={"atr_14": "1", "rsi_14": "1"},
            feature_registry_fingerprint=FEATURE_REGISTRY_FINGERPRINT,
        ))
        m1 = preparer.prepare(bad_spec)
        m2 = preparer.prepare(bad_spec)  # idempotent -- returns the same FAILED manifest, no re-attempt
        assert m1 == m2
        assert m2.status is ExperimentStatus.FAILED

    def test_label_binding_mismatch_transitions_to_failed(self, preparer: ExperimentPreparer) -> None:
        from quant_platform.ml.models import LabelBinding, LabelType

        bad_spec = _spec(label_binding=LabelBinding(name="wrong", kind="forward_return", horizon_bars=10, label_type=LabelType.CONTINUOUS))
        manifest = preparer.prepare(bad_spec)
        assert manifest.status is ExperimentStatus.FAILED
        assert manifest.failure_summary is not None
        assert "label_name_mismatch" in manifest.failure_summary

    def test_label_binding_mismatch_cannot_be_re_prepared_into_ready(self, preparer: ExperimentPreparer) -> None:
        from quant_platform.ml.models import LabelBinding, LabelType

        bad_spec = _spec(label_binding=LabelBinding(name="wrong", kind="forward_return", horizon_bars=10, label_type=LabelType.CONTINUOUS))
        m1 = preparer.prepare(bad_spec)
        m2 = preparer.prepare(bad_spec)
        assert m1 == m2
        assert m2.status is ExperimentStatus.FAILED


class TestManifestEventOrderingInvariant:
    """Proves the ordering invariant `experiment_manager.py`'s module
    docstring documents: every manifest write/transition happens-before
    the event-store append describing it, never the reverse -- so the
    event log can never claim a transition the manifest hasn't already
    durably recorded. This wraps the REAL store methods (via `wraps=`)
    so the pipeline still runs for real; it only additionally records
    call order. Since this is single-threaded, synchronous code, call
    order recorded this way exactly reflects source-code call order."""

    def test_ready_transition_precedes_its_validation_passed_event(self, preparer: ExperimentPreparer) -> None:
        from unittest.mock import patch

        from quant_platform.ml.manifests import ExperimentManifestStore
        from quant_platform.ml.tracking import ExperimentEventStore

        call_order: list[str] = []
        real_transition = ExperimentManifestStore.transition
        real_append = ExperimentEventStore.append

        def recording_transition(self: ExperimentManifestStore, experiment_id: str, *, new_status: ExperimentStatus, **kwargs: object) -> object:
            result = real_transition(self, experiment_id, new_status=new_status, **kwargs)  # type: ignore[arg-type]
            call_order.append(f"transition:{new_status.value}")
            return result

        def recording_append(self: ExperimentEventStore, experiment_id: str, event_type: EventType, **kwargs: object) -> object:
            result = real_append(self, experiment_id, event_type, **kwargs)  # type: ignore[arg-type]
            call_order.append(f"event:{event_type.value}")
            return result

        with patch.object(ExperimentManifestStore, "transition", recording_transition), \
                patch.object(ExperimentEventStore, "append", recording_append):
            manifest = preparer.prepare(_spec())

        assert manifest.status is ExperimentStatus.READY
        assert call_order.index("transition:ready") < call_order.index("event:validation_passed")

    def test_failed_transition_precedes_its_validation_failed_event(self, preparer: ExperimentPreparer) -> None:
        from unittest.mock import patch

        from quant_platform.ml.manifests import ExperimentManifestStore
        from quant_platform.ml.tracking import ExperimentEventStore

        call_order: list[str] = []
        real_transition = ExperimentManifestStore.transition
        real_append = ExperimentEventStore.append

        def recording_transition(self: ExperimentManifestStore, experiment_id: str, *, new_status: ExperimentStatus, **kwargs: object) -> object:
            result = real_transition(self, experiment_id, new_status=new_status, **kwargs)  # type: ignore[arg-type]
            call_order.append(f"transition:{new_status.value}")
            return result

        def recording_append(self: ExperimentEventStore, experiment_id: str, event_type: EventType, **kwargs: object) -> object:
            result = real_append(self, experiment_id, event_type, **kwargs)  # type: ignore[arg-type]
            call_order.append(f"event:{event_type.value}")
            return result

        bad_spec = _spec(feature_binding=FeatureBinding(
            feature_names=("rsi_14", "atr_14"), feature_versions={"atr_14": "1", "rsi_14": "1"},
            feature_registry_fingerprint=FEATURE_REGISTRY_FINGERPRINT,
        ))
        with patch.object(ExperimentManifestStore, "transition", recording_transition), \
                patch.object(ExperimentEventStore, "append", recording_append):
            manifest = preparer.prepare(bad_spec)

        assert manifest.status is ExperimentStatus.FAILED
        assert call_order.index("transition:failed") < call_order.index("event:validation_failed")
