"""Milestone 4B leakage-audit fix: dedicated tests for the two runner-level
building blocks the PRIMARY BLOCKER fix is built on --
`extract_label_horizon_bars` (fail-closed extraction of the REAL label
horizon from a `features.manifests.ResearchDatasetManifest`, via the
typed `features.labels.LabelDefinition` interface, never a private JSON
key read) and `assert_preprocessing_is_safe_for_execution` (fail-closed
preprocessing-leakage gate) -- plus the experiment-identity-implications
proof that a label-definition change needs no new identity field because
it already produces a different `dataset_id`/`ExperimentSpec` identity.

End-to-end (runner-level) rejection/acceptance behavior lives in
`test_runner.py::TestLabelHorizonPurgeEnforcement`; the off-by-one
arithmetic proof for `required_label_purge_bars_for` itself lives in
`test_splitters.py::TestRequiredLabelPurgeBarsFor`; the `FoldPlan`-level
policy gate lives in `test_execution_validation.py::
TestLabelHorizonPurgeCheck`. This file uses the REAL `LabelDefinition`/
`ResearchDatasetManifest` models throughout -- nothing here mocks away
the scientific boundary.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from tests.unit.execution.conftest import write_synthetic_research_dataset

from quant_platform.core.exceptions import FoldValidationError
from quant_platform.core.types import Timeframe
from quant_platform.execution.runner import (
    assert_preprocessing_is_safe_for_execution,
    extract_label_horizon_bars,
)
from quant_platform.features.labels import LabelDefinition, LabelKind
from quant_platform.features.manifests import compute_dataset_id


class TestExtractLabelHorizonBars:
    def test_extracts_real_horizon_from_dataset_manifest(self, tmp_path: Path) -> None:
        manifest, _, _ = write_synthetic_research_dataset(tmp_path)
        assert manifest.label_definition["horizon_bars"] == 5
        assert extract_label_horizon_bars(manifest) == 5

    def test_missing_horizon_bars_key_fails_closed(self, tmp_path: Path) -> None:
        manifest, _, _ = write_synthetic_research_dataset(tmp_path)
        broken = dataclasses.replace(
            manifest, label_definition={"name": "fut", "kind": "future_return"},  # no horizon_bars at all
        )
        with pytest.raises(FoldValidationError, match="could not be parsed"):
            extract_label_horizon_bars(broken)

    def test_missing_kind_key_fails_closed(self, tmp_path: Path) -> None:
        manifest, _, _ = write_synthetic_research_dataset(tmp_path)
        broken = dataclasses.replace(manifest, label_definition={"name": "fut", "horizon_bars": 5})
        with pytest.raises(FoldValidationError, match="could not be parsed"):
            extract_label_horizon_bars(broken)

    def test_unknown_label_kind_fails_closed(self, tmp_path: Path) -> None:
        manifest, _, _ = write_synthetic_research_dataset(tmp_path)
        broken = dataclasses.replace(
            manifest, label_definition={"name": "fut", "kind": "not_a_real_kind", "horizon_bars": 5},
        )
        with pytest.raises(FoldValidationError, match="could not be parsed"):
            extract_label_horizon_bars(broken)

    def test_non_numeric_horizon_bars_fails_closed(self, tmp_path: Path) -> None:
        manifest, _, _ = write_synthetic_research_dataset(tmp_path)
        broken = dataclasses.replace(
            manifest, label_definition={"name": "fut", "kind": "future_return", "horizon_bars": "not-a-number"},
        )
        with pytest.raises(FoldValidationError, match="could not be parsed"):
            extract_label_horizon_bars(broken)

    def test_non_positive_horizon_bars_fails_closed(self, tmp_path: Path) -> None:
        """`LabelDefinition.__post_init__` itself requires `horizon_bars >
        0` -- a stored manifest claiming 0 or negative is just as
        malformed as a missing key, and must fail the same way."""
        manifest, _, _ = write_synthetic_research_dataset(tmp_path)
        broken = dataclasses.replace(
            manifest, label_definition={"name": "fut", "kind": "future_return", "horizon_bars": 0},
        )
        with pytest.raises(FoldValidationError, match="could not be parsed"):
            extract_label_horizon_bars(broken)

    def test_malformed_params_fails_closed_not_raw_feature_error(self, tmp_path: Path) -> None:
        """`LabelDefinition.from_json_dict` internally raises the
        (non-builtin) `FeatureError` for a non-dict `params` value --
        must still be converted to `FoldValidationError`, never left as
        a raw `FeatureError` escaping this function."""
        manifest, _, _ = write_synthetic_research_dataset(tmp_path)
        broken = dataclasses.replace(
            manifest, label_definition={"name": "fut", "kind": "future_return", "horizon_bars": 5, "params": "not-a-dict"},
        )
        with pytest.raises(FoldValidationError, match="could not be parsed"):
            extract_label_horizon_bars(broken)


class TestPreprocessingSafetyCheck:
    def test_no_preprocessing_and_no_fingerprint_is_safe(self, tmp_path: Path) -> None:
        manifest, _, _ = write_synthetic_research_dataset(tmp_path)
        assert manifest.preprocessing_definition == {}
        assert manifest.fitted_preprocessing_fingerprint is None
        assert_preprocessing_is_safe_for_execution(manifest)  # must not raise

    def test_fitted_preprocessing_fingerprint_alone_is_rejected(self, tmp_path: Path) -> None:
        manifest, _, _ = write_synthetic_research_dataset(tmp_path)
        unsafe = dataclasses.replace(manifest, fitted_preprocessing_fingerprint="a" * 64)
        with pytest.raises(FoldValidationError, match="fitted preprocessing"):
            assert_preprocessing_is_safe_for_execution(unsafe)

    def test_nonempty_preprocessing_definition_alone_is_rejected(self, tmp_path: Path) -> None:
        """Checked independently of the fingerprint -- a manifest could
        (hypothetically) have a non-empty `preprocessing_definition` with
        a null fingerprint; this must still be rejected, not merely
        trusting the fingerprint alone (explicit audit requirement: 'Do
        not merely trust fitted_preprocessing_fingerprint')."""
        manifest, _, _ = write_synthetic_research_dataset(tmp_path)
        unsafe = dataclasses.replace(manifest, preprocessing_definition={"f1": "standard_scale"})
        with pytest.raises(FoldValidationError, match="fitted preprocessing"):
            assert_preprocessing_is_safe_for_execution(unsafe)


class TestLabelDefinitionIdentityImplications:
    """Proves no new `ExperimentSpec` identity field is needed for
    `label_horizon_bars`: it is a pure function of the bound dataset's
    `label_definition`, which is ALREADY one of `compute_dataset_id`'s own
    inputs -- so two datasets with different label horizons always get
    different `dataset_id`s, and `ExperimentSpec.to_identity_payload`
    already embeds `dataset_binding` (which carries `dataset_id`)."""

    def test_different_horizon_bars_produces_different_dataset_id(self) -> None:
        common = {
            "symbol": "XAUUSD", "base_timeframe": Timeframe.M1, "feature_registry_fingerprint": "b" * 64,
            "split_definition": {"strategy": "single"}, "preprocessing_definition": {},
        }
        id_horizon_5 = compute_dataset_id(
            label_definition={"name": "fut", "kind": "future_return", "horizon_bars": 5, "params": {}}, **common,
        )
        id_horizon_12 = compute_dataset_id(
            label_definition={"name": "fut", "kind": "future_return", "horizon_bars": 12, "params": {}}, **common,
        )
        assert id_horizon_5 != id_horizon_12

    def test_identical_label_definition_produces_identical_dataset_id(self) -> None:
        """Contrast case: proves the difference above is caused BY the
        label definition, not incidental non-determinism."""
        common = {
            "symbol": "XAUUSD", "base_timeframe": Timeframe.M1, "feature_registry_fingerprint": "b" * 64,
            "split_definition": {"strategy": "single"}, "preprocessing_definition": {},
        }
        label_definition = {"name": "fut", "kind": "future_return", "horizon_bars": 5, "params": {}}
        assert compute_dataset_id(label_definition=label_definition, **common) == compute_dataset_id(label_definition=label_definition, **common)

    def test_real_label_definition_round_trips_through_dataset_id_inputs(self) -> None:
        """Uses the REAL `LabelDefinition.to_json_dict()` (not a hand-
        written dict) as the exact payload `compute_dataset_id` hashes,
        proving the identity argument holds against the real typed model,
        not just an assumed-equivalent literal."""
        common = {
            "symbol": "XAUUSD", "base_timeframe": Timeframe.M1, "feature_registry_fingerprint": "b" * 64,
            "split_definition": {"strategy": "single"}, "preprocessing_definition": {},
        }
        short_horizon = LabelDefinition(name="fut", kind=LabelKind.FUTURE_RETURN, horizon_bars=5)
        long_horizon = LabelDefinition(name="fut", kind=LabelKind.FUTURE_RETURN, horizon_bars=12)
        assert compute_dataset_id(label_definition=short_horizon.to_json_dict(), **common) != compute_dataset_id(
            label_definition=long_horizon.to_json_dict(), **common
        )
