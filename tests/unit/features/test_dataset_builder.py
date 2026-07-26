from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.exceptions import ResearchDatasetError
from quant_platform.core.types import Timeframe
from quant_platform.features.dataset_builder import ResearchDatasetBuilder, ResearchDatasetBuildRequest
from quant_platform.features.interfaces import FeatureContext, FeatureDefinition
from quant_platform.features.labels import LabelDefinition, LabelKind
from quant_platform.features.manifests import ResearchDatasetStore, ResearchManifestStore
from quant_platform.features.models import FeatureCategory, FeatureSpec, MissingPolicyKind, MissingPolicySpec
from quant_platform.features.normalization import TransformKind
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.technical.price import TechnicalWindows, register_core_technical_features


def _trend_registry() -> FeatureRegistry:
    """A tiny registry with one deterministic, linearly-trending feature --
    used to make the walk-forward per-fold-isolation guarantee directly
    observable (a monotonic trend means each fold's train-only mean is
    trivially predictable, and any cross-fold leakage would be equally
    trivial to detect)."""
    registry = FeatureRegistry()
    spec = FeatureSpec(
        name="trend", version="1", description="row index as a float", category=FeatureCategory.PRICE,
        required_inputs=(), source_symbols=(), source_timeframe=Timeframe.M1, output_dtype="float64",
        lookback_bars=0, warmup_bars=0,
    )
    registry.register(
        FeatureDefinition(spec=spec, compute=lambda ctx: pd.Series(np.arange(len(ctx.base_df), dtype="float64")))
    )
    return registry


def _builder(tmp_path, seeded_loader, registry: FeatureRegistry) -> ResearchDatasetBuilder:
    return ResearchDatasetBuilder(
        historical_loader=seeded_loader, registry=registry, research_store=ResearchDatasetStore(tmp_path / "research"),
        manifest_store=ResearchManifestStore(tmp_path / "research"),
    )


def _request(**overrides) -> ResearchDatasetBuildRequest:
    base = {
        "symbol": "XAUUSD", "base_timeframe": Timeframe.M1, "start": pd.Timestamp("2024-01-01", tz="UTC"),
        "end": pd.Timestamp("2024-01-01", tz="UTC") + Timeframe.M1.duration * 2000,
        "feature_names": ("trend",), "label_definition": LabelDefinition(name="fut", kind=LabelKind.FUTURE_RETURN, horizon_bars=5),
        "split_strategy": "chronological",
        "split_params": {"train_fraction": 0.7, "validation_fraction": 0.15, "purge_bars": 5, "embargo_bars": 5},
    }
    base.update(overrides)
    return ResearchDatasetBuildRequest(**base)


class TestBasicBuild:
    def test_produces_three_named_splits(self, tmp_path, seeded_loader) -> None:
        registry = _trend_registry()
        builder = _builder(tmp_path, seeded_loader, registry)
        manifest = builder.build(_request())
        assert set(manifest.row_counts) == {"train", "validation", "test"}
        assert all(count > 0 for count in manifest.row_counts.values())

    def test_artifacts_readable_after_build(self, tmp_path, seeded_loader) -> None:
        registry = _trend_registry()
        builder = _builder(tmp_path, seeded_loader, registry)
        manifest = builder.build(_request())
        research_store = ResearchDatasetStore(tmp_path / "research")
        splits = research_store.read_artifacts(manifest.dataset_id, manifest.content_id)
        assert splits is not None
        assert "trend" in splits["train"].columns
        assert "label" in splits["train"].columns


class TestRebuildDeterminism:
    def test_rebuilding_same_config_gives_same_dataset_id_and_version(self, tmp_path, seeded_loader) -> None:
        registry = _trend_registry()
        builder = _builder(tmp_path, seeded_loader, registry)
        manifest1 = builder.build(_request())
        manifest2 = builder.build(_request())
        assert manifest1.dataset_id == manifest2.dataset_id
        assert manifest1.version == manifest2.version
        assert manifest1.content_id == manifest2.content_id

    def test_changed_feature_selection_changes_dataset_id(self, tmp_path, seeded_loader) -> None:
        registry = _trend_registry()
        register_core_technical_features(registry, timeframe=Timeframe.M1, windows=TechnicalWindows(return_windows=(1,)))
        builder = _builder(tmp_path, seeded_loader, registry)
        manifest_trend_only = builder.build(_request(feature_names=("trend",)))
        manifest_with_return = builder.build(_request(feature_names=("trend", "return_simple_1")))
        assert manifest_trend_only.dataset_id != manifest_with_return.dataset_id

    def test_changed_label_horizon_changes_dataset_id(self, tmp_path, seeded_loader) -> None:
        registry = _trend_registry()
        builder = _builder(tmp_path, seeded_loader, registry)
        manifest_h5 = builder.build(_request())
        manifest_h10 = builder.build(
            _request(label_definition=LabelDefinition(name="fut", kind=LabelKind.FUTURE_RETURN, horizon_bars=10))
        )
        assert manifest_h5.dataset_id != manifest_h10.dataset_id

    def test_changed_preprocessing_changes_dataset_id(self, tmp_path, seeded_loader) -> None:
        registry = _trend_registry()
        builder = _builder(tmp_path, seeded_loader, registry)
        manifest_no_prep = builder.build(_request())
        manifest_with_prep = builder.build(_request(preprocessing={"trend": TransformKind.STANDARD_SCALE}))
        assert manifest_no_prep.dataset_id != manifest_with_prep.dataset_id


class TestWalkForwardPerFoldIsolation:
    def test_fold_preprocessing_never_reflects_a_later_folds_train_data(self, tmp_path, seeded_loader) -> None:
        """The regression test for the leak identified and fixed during
        development: fitting ONE global pipeline across the union of every
        fold's train rows would let fold_1's (later, larger) train range
        leak into fold_0's evaluation, since fold_1's train includes
        fold_0's test rows. With per-fold-group fitting, fold_0's fitted
        mean must be strictly less than fold_1's (the `trend` feature is
        monotonically increasing, so a later/larger train window always
        has a higher mean) -- proving each fold's pipeline is fit
        independently."""
        registry = _trend_registry()
        builder = _builder(tmp_path, seeded_loader, registry)
        request = _request(
            split_strategy="expanding_walk_forward",
            split_params={"n_splits": 3, "test_size": 200, "label_horizon": 5, "embargo": 5},
            preprocessing={"trend": TransformKind.STANDARD_SCALE},
            drop_unlabeled_rows=True,
        )
        manifest = builder.build(request)
        research_store = ResearchDatasetStore(tmp_path / "research")
        preprocessing = research_store.read_preprocessing(manifest.dataset_id, manifest.content_id)

        fold_0_mean = preprocessing["fold_0"]["fitted"]["trend"]["params"]["mean"]
        fold_1_mean = preprocessing["fold_1"]["fitted"]["trend"]["params"]["mean"]
        fold_2_mean = preprocessing["fold_2"]["fitted"]["trend"]["params"]["mean"]
        assert fold_0_mean < fold_1_mean < fold_2_mean

    def test_fold_test_split_values_are_scaled_by_that_folds_own_parameters(self, tmp_path, seeded_loader) -> None:
        registry = _trend_registry()
        builder = _builder(tmp_path, seeded_loader, registry)
        request = _request(
            split_strategy="expanding_walk_forward",
            split_params={"n_splits": 2, "test_size": 300, "label_horizon": 5, "embargo": 5},
            preprocessing={"trend": TransformKind.STANDARD_SCALE},
        )
        manifest = builder.build(request)
        research_store = ResearchDatasetStore(tmp_path / "research")
        splits = research_store.read_artifacts(manifest.dataset_id, manifest.content_id)
        # both folds' scaled test data should be roughly zero-mean under
        # THEIR OWN fold's fit -- not a shared global fit.
        assert abs(splits["fold_0_test"]["trend"].mean()) < 5
        assert abs(splits["fold_1_test"]["trend"].mean()) < 5


class TestTrainingStatisticFill:
    def test_fill_statistic_derived_only_from_train_partition(self, tmp_path, seeded_loader) -> None:
        registry = FeatureRegistry()
        spec = FeatureSpec(
            name="gappy", version="1", description="d", category=FeatureCategory.PRICE, required_inputs=(),
            source_symbols=(), source_timeframe=Timeframe.M1, output_dtype="float64", lookback_bars=0, warmup_bars=0,
            null_policy=MissingPolicySpec(kind=MissingPolicyKind.TRAINING_STATISTIC_FILL, statistic="mean"),
        )

        def _compute(ctx: FeatureContext) -> pd.Series:
            # Train range: near-constant ~1.0; everything after (val/test):
            # wildly different (~1000), so if the fill statistic leaked
            # validation/test data in, it would be nowhere near 1.0.
            n = len(ctx.base_df)
            values = np.where(np.arange(n) < int(n * 0.7), 1.0, 1000.0)
            values = values.astype("float64")
            values[::10] = np.nan  # scatter some nulls everywhere
            return pd.Series(values)

        registry.register(FeatureDefinition(spec=spec, compute=_compute))
        builder = _builder(tmp_path, seeded_loader, registry)
        manifest = builder.build(_request(feature_names=("gappy",)))
        research_store = ResearchDatasetStore(tmp_path / "research")
        splits = research_store.read_artifacts(manifest.dataset_id, manifest.content_id)
        assert not splits["train"]["gappy"].isna().any()
        # the fitted fill value must be close to 1.0 (train's own value),
        # never anywhere near 1000 (validation/test's value)
        assert splits["train"]["gappy"].between(0.5, 1.5).all()


class TestUnlabeledRowHandling:
    def test_drop_unlabeled_rows_removes_trailing_unlabeled_rows(self, tmp_path, seeded_loader) -> None:
        registry = _trend_registry()
        builder = _builder(tmp_path, seeded_loader, registry)
        manifest = builder.build(_request(drop_unlabeled_rows=True))
        research_store = ResearchDatasetStore(tmp_path / "research")
        splits = research_store.read_artifacts(manifest.dataset_id, manifest.content_id)
        for split_df in splits.values():
            assert not split_df["label"].isna().any()
            assert "label_valid" not in split_df.columns

    def test_keep_unlabeled_rows_adds_label_valid_column(self, tmp_path, seeded_loader) -> None:
        registry = _trend_registry()
        builder = _builder(tmp_path, seeded_loader, registry)
        manifest = builder.build(_request(drop_unlabeled_rows=False))
        research_store = ResearchDatasetStore(tmp_path / "research")
        splits = research_store.read_artifacts(manifest.dataset_id, manifest.content_id)
        assert "label_valid" in splits["test"].columns
        assert (~splits["test"]["label_valid"]).any()  # tail of the dataset has unresolved labels


class TestValidationGate:
    """Proves the leakage-validation gate is load-bearing: a feature that
    "cheats" by peeking at the exact future value the label is derived
    from (nothing in `FeatureContext` structurally prevents a badly-written
    feature from calling `.shift(-1)` directly -- the engine's guarantee is
    that LABELS can never reach features, not that a feature author cannot
    misuse raw future access within `ctx.base_df`; the truncation-invariance
    tests in `test_technical_price.py` are what prove the STANDARD library
    never does this) triggers a CRITICAL target-leakage finding, which
    blocks the build by default and can only proceed with an explicit,
    named override."""

    def _cheating_registry(self) -> FeatureRegistry:
        registry = FeatureRegistry()
        spec = FeatureSpec(
            name="cheat", version="1", description="peeks at the future close", category=FeatureCategory.PRICE,
            required_inputs=("close",), source_symbols=(), source_timeframe=Timeframe.M1, output_dtype="float64",
            lookback_bars=0, warmup_bars=0,
        )
        registry.register(
            FeatureDefinition(spec=spec, compute=lambda ctx: ctx.base_df["close"].pct_change(1).shift(-1))
        )
        return registry

    def test_leaked_feature_blocks_build_by_default(self, tmp_path, seeded_loader) -> None:
        registry = self._cheating_registry()
        builder = _builder(tmp_path, seeded_loader, registry)
        request = _request(
            feature_names=("cheat",),
            label_definition=LabelDefinition(name="fut", kind=LabelKind.FUTURE_RETURN, horizon_bars=1),
        )
        with pytest.raises(ResearchDatasetError, match="critical"):
            builder.build(request)

    def test_allow_critical_validation_issues_overrides_the_gate(self, tmp_path, seeded_loader) -> None:
        registry = self._cheating_registry()
        builder = _builder(tmp_path, seeded_loader, registry)
        request = _request(
            feature_names=("cheat",),
            label_definition=LabelDefinition(name="fut", kind=LabelKind.FUTURE_RETURN, horizon_bars=1),
            allow_critical_validation_issues=True,
        )
        manifest = builder.build(request)
        assert manifest.leakage_validation_result["is_valid"] is False
        assert manifest.leakage_validation_result["critical_count"] >= 1


class TestSpecIdentityVsActualClosureContent:
    """Adversarial self-audit (Section 20 'changed configuration with stale
    cache'): `feature_registry_fingerprint`/`dataset_id` are derived from
    DECLARED `FeatureSpec` metadata, not from hashing a feature's `compute`
    closure bytecode. Two different closures registered under the exact
    same (name, version, and otherwise-identical spec) therefore produce
    the SAME declared identity -- an acknowledged limitation of any
    metadata-based fingerprint (see docs/feature_engineering.md's Known
    Limitations). This test proves the failure mode stops there: the
    ACTUAL computed content still differs, so `ResearchManifestStore.save`
    still mints a genuinely NEW version rather than silently treating the
    differing output as an identical no-op -- no silent data corruption,
    even in this edge case."""

    def test_differing_closures_under_identical_declared_identity_still_get_distinct_versions(
        self, tmp_path, seeded_loader
    ) -> None:
        def _build(multiplier: float):
            registry = FeatureRegistry()
            spec = FeatureSpec(
                name="ambiguous", version="1", description="d", category=FeatureCategory.PRICE,
                required_inputs=("close",), source_symbols=(), source_timeframe=Timeframe.M1,
                output_dtype="float64", lookback_bars=0, warmup_bars=0,
            )
            registry.register(
                FeatureDefinition(spec=spec, compute=lambda ctx, m=multiplier: ctx.base_df["close"] * m)
            )
            builder = _builder(tmp_path, seeded_loader, registry)
            return builder.build(_request(feature_names=("ambiguous",)))

        manifest_a = _build(1.0)
        manifest_b = _build(2.0)  # same declared spec, genuinely different output

        assert manifest_a.dataset_id == manifest_b.dataset_id
        assert manifest_a.feature_registry_fingerprint == manifest_b.feature_registry_fingerprint
        assert manifest_a.version != manifest_b.version
        assert manifest_a.content_id != manifest_b.content_id
