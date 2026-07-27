"""Milestone 4D.1 completion, Section 7: proof that extracting
`canonical_json_bytes`/`parse_json_strict`/`sha256_hex_bytes`/
`write_json_atomic` out of `ml.persistence` and into the dependency-
neutral `quant_platform.core.json` did not change a single byte of
output for any already-valid payload -- and therefore did not change any
identity/fingerprint/content-hash computed from one.

TWO INDEPENDENT PROOF TECHNIQUES, BOTH USED BELOW
--------------------------------------------------------------------------
1. **Pre-existing pinned golden hash.** `tests/unit/ml/test_experiment_
   identity.py::TestGoldenHash::test_golden_experiment_id` already
   hardcodes `GOLDEN_EXPERIMENT_ID`, computed and checked into this
   repository BEFORE this milestone's `core.json` extraction existed.
   That test still passes unmodified (see the full-suite run in this
   milestone's delivery report) -- the strongest possible evidence,
   since it is a value fixed in the past, not something this milestone
   could tune to pass.
2. **Live "new vs. reconstructed-old" comparison.** For every fingerprint
   function actually touched by this migration (the `features` package's
   six `json.dumps(..., allow_nan=False)` identity-critical write paths,
   deliberately NOT routed through `canonical_json_bytes` -- see each
   function's own docstring and `docs/persistence_security.md`), this
   file recomputes what the OLD code (before `allow_nan=False` was added)
   would have produced, inline, and asserts byte-for-byte/hash-for-hash
   equality with what the CURRENT code produces. This is strictly
   stronger than a hardcoded golden constant: it proves WHY the values
   match (the only change was adding a keyword argument that is a no-op
   for any finite payload), not just THAT they happen to.

No hardcoded SHA-256 hex digests appear in this file for anything computed
here for the first time -- every comparison is either against the
pre-existing `GOLDEN_EXPERIMENT_ID` or against a live recomputation, to
eliminate any risk of a transcription error silently weakening the proof.
"""

from __future__ import annotations

import hashlib
import json

from tests.unit.optimization.conftest import make_optimization_spec

from quant_platform.core.json import canonical_json_bytes, sha256_hex_bytes
from quant_platform.core.types import Timeframe
from quant_platform.features.dataset_builder import _combined_fold_fingerprint
from quant_platform.features.manifests import _combined_checksum, compute_dataset_id
from quant_platform.features.models import FeatureCategory, FeatureSpec
from quant_platform.features.normalization import TransformPipeline
from quant_platform.features.registry import FeatureRegistry
from quant_platform.optimization.models import (
    OPTIMIZATION_IDENTITY_SCHEMA_VERSION,
    compute_optimization_identity,
)


class TestCanonicalJsonBytesProducesFixedBytesForAFixedPayload:
    """A literal byte-string golden vector -- safe to hardcode (unlike a
    hash digest) since a transcription error here would simply make the
    test fail to construct meaningfully, never silently pass wrong."""

    def test_fixed_representative_payload_encodes_to_the_exact_expected_bytes(self) -> None:
        payload = {"b": 2, "a": 1, "nested": {"z": 3.5, "y": [1, 2, 3]}, "flag": True, "none_val": None}
        expected = b'{"a":1,"b":2,"flag":true,"nested":{"y":[1,2,3],"z":3.5},"none_val":null}'
        assert canonical_json_bytes(payload) == expected

    def test_sha256_of_that_fixed_payload_matches_hashlib_computed_independently(self) -> None:
        payload = {"b": 2, "a": 1, "nested": {"z": 3.5, "y": [1, 2, 3]}, "flag": True, "none_val": None}
        expected_text = '{"a":1,"b":2,"flag":true,"nested":{"y":[1,2,3],"z":3.5},"none_val":null}'
        assert sha256_hex_bytes(canonical_json_bytes(payload)) == hashlib.sha256(expected_text.encode("utf-8")).hexdigest()


class TestDatasetAndContentIdsUnchangedByMigration:
    """`features.manifests.compute_dataset_id`/`_combined_checksum` were
    both touched (added `allow_nan=False`) but deliberately NOT routed
    through `canonical_json_bytes`, specifically to guarantee this."""

    def test_compute_dataset_id_matches_pre_migration_reconstruction(self) -> None:
        kwargs = {
            "symbol": "XAUUSD", "base_timeframe": Timeframe.M1, "feature_registry_fingerprint": "fp123",
            "label_definition": {"kind": "future_return", "horizon_bars": 5},
            "split_definition": {"strategy": "chronological"}, "preprocessing_definition": {"x": "standard"},
        }
        current = compute_dataset_id(**kwargs)  # type: ignore[arg-type]
        pre_migration_payload = json.dumps(
            {
                "symbol": "XAUUSD", "base_timeframe": "M1", "feature_registry_fingerprint": "fp123",
                "label_definition": {"kind": "future_return", "horizon_bars": 5},
                "split_definition": {"strategy": "chronological"}, "preprocessing_definition": {"x": "standard"},
            },
            sort_keys=True, default=str,
        )
        pre_migration_id = hashlib.sha256(pre_migration_payload.encode("utf-8")).hexdigest()[:16]
        assert current == pre_migration_id

    def test_combined_checksum_matches_pre_migration_reconstruction(self) -> None:
        per_split = {"train": "aaa111", "test": "bbb222"}
        current = _combined_checksum(per_split, "ccc333")
        pre_migration_payload = json.dumps({"splits": per_split, "preprocessing": "ccc333"}, sort_keys=True)
        pre_migration = hashlib.sha256(pre_migration_payload.encode("utf-8")).hexdigest()
        assert current == pre_migration

    def test_combined_fold_fingerprint_matches_pre_migration_reconstruction(self) -> None:
        fingerprints = {"fold_0": "aaa", "fold_1": None}
        current = _combined_fold_fingerprint(fingerprints)
        pre_migration_payload = json.dumps(fingerprints, sort_keys=True)
        pre_migration = hashlib.sha256(pre_migration_payload.encode("utf-8")).hexdigest()
        assert current == pre_migration


class TestFeatureSchemaFingerprintsUnchangedByMigration:
    def test_feature_spec_fingerprint_matches_pre_migration_reconstruction(self) -> None:
        spec = FeatureSpec(
            name="sma_20", version="1", description="a test feature", category=FeatureCategory.PRICE,
            required_inputs=("close",), source_symbols=(), source_timeframe=Timeframe.M1, output_dtype="float64",
            lookback_bars=20, warmup_bars=20, deterministic_params={"window": 20},
        )
        current = spec.fingerprint()
        pre_migration_payload = json.dumps(spec.to_json_dict(), sort_keys=True, default=str)
        pre_migration = hashlib.sha256(pre_migration_payload.encode("utf-8")).hexdigest()
        assert current == pre_migration

    def test_transform_pipeline_fingerprint_matches_pre_migration_reconstruction(self) -> None:
        pipeline = TransformPipeline()
        current = pipeline.fingerprint()
        pre_migration_payload = json.dumps(pipeline.to_json_dict(), sort_keys=True)
        pre_migration = hashlib.sha256(pre_migration_payload.encode("utf-8")).hexdigest()
        assert current == pre_migration

    def test_feature_registry_fingerprint_matches_pre_migration_reconstruction(self) -> None:
        """This IS `feature_registry_fingerprint`, embedded in every
        `FeatureBinding`/`ExperimentSpec` downstream -- the single most
        widely-propagated fingerprint this migration had to preserve."""
        registry = FeatureRegistry()
        current = registry.fingerprint()
        pre_migration_payload = json.dumps([], sort_keys=True, default=str)
        pre_migration = hashlib.sha256(pre_migration_payload.encode("utf-8")).hexdigest()
        assert current == pre_migration


class TestOptimizationIdentityUnchangedByMigration:
    """`optimization.models.compute_optimization_identity` was never
    itself modified -- it calls `ml.fingerprints.fingerprint_json`, which
    calls `ml.persistence.canonical_json_bytes`/`sha256_hex_bytes`, which
    are now object-identical re-exports of `core.json`'s (see
    `tests/unit/test_architecture_boundaries.py::
    TestMlPersistenceDelegatesToCoreJson`). This reconstructs the
    pre-migration computation independently to prove that chain produces
    byte-identical output end to end, not just that the objects match."""

    def test_optimization_id_matches_pre_migration_reconstruction(self) -> None:
        spec = make_optimization_spec()
        identity = compute_optimization_identity(spec)

        payload = dict(spec.to_identity_payload())
        payload["identity_schema_version"] = OPTIMIZATION_IDENTITY_SCHEMA_VERSION
        pre_migration_text = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True,
        )
        pre_migration_id = hashlib.sha256(pre_migration_text.encode("utf-8")).hexdigest()
        assert identity.optimization_id == pre_migration_id

    def test_optimization_id_is_deterministic_across_repeated_computation(self) -> None:
        spec = make_optimization_spec()
        first = compute_optimization_identity(spec).optimization_id
        second = compute_optimization_identity(spec).optimization_id
        assert first == second


class TestExperimentAndExecutionIdentityUnchangedByMigration:
    """`ml.experiment_identity.compute_experiment_identity` is what both
    `ExperimentSpec` identity AND (since `execution` never mints a
    separate identity of its own -- `ExecutionManifest.experiment_id` IS
    this same value) "execution IDs" ultimately are. The PRIMARY proof
    for this one is `tests/unit/ml/test_experiment_identity.py::
    TestGoldenHash::test_golden_experiment_id`'s pre-existing, pinned
    `GOLDEN_EXPERIMENT_ID` (fixed before this migration existed, still
    passing after it). This adds a second, independent, live
    reconstruction of the same underlying mechanism for a different
    spec, for defense in depth."""

    def test_experiment_identity_matches_pre_migration_reconstruction(self) -> None:
        from quant_platform.ml.experiment_identity import IDENTITY_SCHEMA_VERSION, compute_experiment_identity
        from quant_platform.ml.experiment_spec import ExperimentSpec
        from quant_platform.ml.models import (
            CodeRevisionBinding,
            DatasetBinding,
            FeatureBinding,
            LabelBinding,
            LabelType,
            ModelHyperparameters,
            ObjectiveType,
            PreprocessingBinding,
            SplitBinding,
        )
        from quant_platform.ml.seeds import SeedConfiguration

        spec = ExperimentSpec(
            dataset_binding=DatasetBinding(
                dataset_id="ds1", manifest_version="1", content_id="d" * 64, symbol="EURUSD", base_timeframe="M15",
            ),
            feature_binding=FeatureBinding(
                feature_names=("ema_9",), feature_versions={"ema_9": "1"}, feature_registry_fingerprint="e" * 64,
            ),
            label_binding=LabelBinding(name="fwd_ret_3", kind="forward_return", horizon_bars=3, label_type=LabelType.CONTINUOUS),
            split_binding=SplitBinding(strategy="time_ordered_holdout"),
            preprocessing_binding=PreprocessingBinding(),
            model_name="constant_test_model", model_version="1",
            hyperparameters=ModelHyperparameters(values={"beta": 0.2}),
            objective=ObjectiveType.REGRESSION,
            seed_configuration=SeedConfiguration(master_seed=7),
            code_revision_binding=CodeRevisionBinding(revision="f" * 40, source="git", is_dirty=False),
            primary_metric="mae",
        )
        identity = compute_experiment_identity(spec)

        payload = dict(spec.to_identity_payload())
        payload["identity_schema_version"] = IDENTITY_SCHEMA_VERSION
        pre_migration_text = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True,
        )
        pre_migration_id = hashlib.sha256(pre_migration_text.encode("utf-8")).hexdigest()
        assert identity.experiment_id == pre_migration_id
