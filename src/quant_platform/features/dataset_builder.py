"""`ResearchDatasetBuilder`: the single orchestrator that turns an immutable
historical dataset version into an immutable, versioned, fully-lineaged
research dataset. Ties together every other module in this package in one
fixed, leak-safe order:

  1. Load an immutable historical dataset version (`historical.loader.
     DatasetLoader` -- exact version reconstruction already guaranteed by
     Milestone 2's content-addressed canonical storage).
  2. Resolve the feature dependency graph (`registry.FeatureRegistry`).
  3. Compute point-in-time features (`engine.FeatureEngine`).
  4. Build labels SEPARATELY (`labels.build_label`) -- never fed into step 2.
  5. Trim warm-up rows; optionally drop rows without a valid label.
  6. Generate time-aware splits (`splitting`).
  7. Fit training-statistic imputation AND preprocessing on the TRAIN split
     ONLY, then apply the frozen result to every split (`missing`,
     `normalization`).
  8. Validate the assembled dataset (`validation.validate_research_dataset`).
  9. Write immutable, content-addressed artifacts (`manifests.
     ResearchDatasetStore`).
  10. Write a complete manifest (`manifests.ResearchManifestStore`).

This fixed ordering IS the leakage guarantee: preprocessing/imputation
statistics are computed in step 7, strictly after the split boundaries from
step 6 are known, and never touch anything outside the train split's own
row range.

PER-FOLD-GROUP FITTING (walk-forward)
--------------------------------------------------------------------------
For a `chronological` split plan there is exactly one train split, so step
7 is a single fit/apply pass. For expanding/rolling WALK-FORWARD plans,
each fold's `fold_k_train` range is progressively larger and, critically,
overlaps EARLIER folds' `fold_j_test` range (fold `j`'s test data becomes
part of fold `j+1`'s training data once "time has passed") -- fitting one
GLOBAL pipeline across the union of every fold's train rows would
therefore fit on data that includes an earlier fold's own test rows,
leaking into that fold's evaluation. `_fold_groups` partitions the split
plan into independent groups (one per fold, or one "global" group for a
chronological plan) and step 7 is repeated ONCE PER GROUP, using only that
group's own train split -- never data from another group.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from quant_platform.core.exceptions import ResearchDatasetError
from quant_platform.core.types import Timeframe
from quant_platform.features.engine import FeatureEngine
from quant_platform.features.labels import LabelDefinition, build_label
from quant_platform.features.manifests import (
    ResearchDatasetManifest,
    ResearchDatasetStore,
    ResearchManifestStore,
    compute_dataset_id,
)
from quant_platform.features.missing import (
    apply_missing_policy,
    compute_training_statistic,
    missingness_summary,
)
from quant_platform.features.models import MissingPolicyKind
from quant_platform.features.normalization import TransformKind, TransformPipeline
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.splitting import (
    DatasetSplit,
    SplitPlan,
    build_chronological_split,
    build_walk_forward_splits,
)
from quant_platform.features.validation import validate_research_dataset
from quant_platform.historical.code_revision import capture_code_revision
from quant_platform.historical.loader import DatasetLoader, LoadRequest

SplitStrategy = Literal["chronological", "expanding_walk_forward", "rolling_walk_forward"]


def _fold_groups(split_plan: SplitPlan) -> dict[str, list[DatasetSplit]]:
    """Partition a split plan into independent fit/apply groups: one
    "global" group for a chronological plan's `train`/`validation`/`test`,
    or one `fold_{k}` group per walk-forward fold's `fold_{k}_train`/
    `fold_{k}_test` pair. See module docstring for why walk-forward folds
    must never share a single fitted preprocessing pipeline."""
    groups: dict[str, list[DatasetSplit]] = {}
    for split in split_plan.splits:
        if split.name in ("train", "validation", "test"):
            key = "global"
        else:
            key = "_".join(split.name.split("_")[:-1]) or split.name
        groups.setdefault(key, []).append(split)
    return groups


def _combined_fold_fingerprint(fold_pipeline_fingerprints: dict[str, str | None]) -> str:
    """NOT migrated to `canonical_json_bytes`: this directly computes a
    fingerprint that propagates into `fitted_preprocessing_fingerprint`
    and ultimately `compute_dataset_id` -- see `features.manifests.
    write_artifacts`' identical note. `allow_nan=False` costs nothing
    here (every value is a fingerprint string or `None`, never a float)
    but keeps this fingerprint chain uniformly NaN-safe."""
    payload = json.dumps(fold_pipeline_fingerprints, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ResearchDatasetBuildRequest:
    symbol: str
    base_timeframe: Timeframe
    start: pd.Timestamp
    end: pd.Timestamp
    feature_names: tuple[str, ...]
    label_definition: LabelDefinition
    split_strategy: SplitStrategy
    split_params: dict[str, object] = field(default_factory=dict)
    preprocessing: dict[str, TransformKind] = field(default_factory=dict)
    dataset_version: str | None = None
    drop_unlabeled_rows: bool = True
    allow_critical_validation_issues: bool = False
    reproducibility_seed: int = 0
    aux_input_content_hashes: dict[str, str] = field(default_factory=dict)


class ResearchDatasetBuilder:
    def __init__(
        self,
        *,
        historical_loader: DatasetLoader,
        registry: FeatureRegistry,
        research_store: ResearchDatasetStore,
        manifest_store: ResearchManifestStore,
        higher_timeframe_data: Mapping[Timeframe, pd.DataFrame] | None = None,
        cross_asset_data: Mapping[str, pd.DataFrame] | None = None,
        macro_data: Mapping[str, pd.DataFrame] | None = None,
    ) -> None:
        self._historical_loader = historical_loader
        self._registry = registry
        self._research_store = research_store
        self._manifest_store = manifest_store
        self._higher_timeframe_data = higher_timeframe_data or {}
        self._cross_asset_data = cross_asset_data or {}
        self._macro_data = macro_data or {}

    def build(self, request: ResearchDatasetBuildRequest) -> ResearchDatasetManifest:
        load_request = LoadRequest(
            symbol=request.symbol, timeframe=request.base_timeframe, start=request.start, end=request.end,
            dataset_version=request.dataset_version,
        )
        historical_manifest = self._historical_loader.resolve_manifest(load_request)
        base_df = self._historical_loader.load_for_engine(load_request)

        engine = FeatureEngine(self._registry)
        result = engine.compute(
            base_df=base_df, symbol=request.symbol, timeframe=request.base_timeframe,
            feature_names=request.feature_names, higher_timeframe_data=self._higher_timeframe_data,
            cross_asset_data=self._cross_asset_data, macro_data=self._macro_data,
            source_dataset_manifest_id=historical_manifest.dataset_id,
        )
        label_result = build_label(
            request.label_definition, close=base_df["close"], high=base_df["high"], low=base_df["low"]
        )

        warmup = result.warmup_bars
        features_trimmed = result.features.iloc[warmup:].reset_index(drop=True)
        labels_trimmed = label_result.values.iloc[warmup:].reset_index(drop=True)
        is_valid_trimmed = label_result.is_valid.iloc[warmup:].reset_index(drop=True)
        open_time_trimmed = base_df["open_time"].iloc[warmup:].reset_index(drop=True)

        if request.drop_unlabeled_rows:
            keep_mask = is_valid_trimmed
            features_trimmed = features_trimmed.loc[keep_mask].reset_index(drop=True)
            labels_trimmed = labels_trimmed.loc[keep_mask].reset_index(drop=True)
            is_valid_trimmed = is_valid_trimmed.loc[keep_mask].reset_index(drop=True)
            open_time_trimmed = open_time_trimmed.loc[keep_mask].reset_index(drop=True)

        split_plan = self._build_split_plan(request, open_time_trimmed)
        fold_groups = _fold_groups(split_plan)

        named_splits: dict[str, pd.DataFrame] = {}
        fold_pipeline_fingerprints: dict[str, str | None] = {}
        fold_pipelines: dict[str, TransformPipeline] = {}
        features_filled_for_validation = features_trimmed.copy()

        for group_key, splits_in_group in fold_groups.items():
            train_split = next(
                (s for s in splits_in_group if s.name == "train" or s.name.endswith("_train")), None
            )
            if train_split is None:
                raise ResearchDatasetError(f"Fold group {group_key!r} has no train split")
            train_indices = pd.Index(train_split.indices)

            group_features = self._apply_training_statistic_fills(
                features_trimmed, feature_names=request.feature_names, train_indices=train_indices
            )
            for split in splits_in_group:
                features_filled_for_validation.loc[split.indices] = group_features.loc[split.indices]

            pipeline = TransformPipeline()
            if request.preprocessing:
                pipeline.fit(group_features.iloc[train_indices].reset_index(drop=True), specs=request.preprocessing)
            fold_pipeline_fingerprints[group_key] = pipeline.fingerprint() if request.preprocessing else None
            fold_pipelines[group_key] = pipeline

            for split in splits_in_group:
                split_features = group_features.iloc[split.indices].reset_index(drop=True)
                if request.preprocessing:
                    split_features = pipeline.apply(split_features)
                split_df = split_features.copy()
                split_df.insert(0, "open_time", open_time_trimmed.iloc[split.indices].reset_index(drop=True))
                split_df["label"] = labels_trimmed.iloc[split.indices].reset_index(drop=True)
                if not request.drop_unlabeled_rows:
                    split_df["label_valid"] = is_valid_trimmed.iloc[split.indices].reset_index(drop=True)
                named_splits[split.name] = split_df

        features_filled = features_filled_for_validation
        validation_report = validate_research_dataset(
            features_filled, timestamps=open_time_trimmed, labels=labels_trimmed,
            lineage_feature_names=set(result.lineages), split_plan=split_plan,
        )
        if not validation_report.is_valid and not request.allow_critical_validation_issues:
            raise ResearchDatasetError(
                f"Research dataset validation found {len(validation_report.critical_issues)} critical "
                f"issue(s); refusing to write artifacts. Pass allow_critical_validation_issues=True to "
                "override (not recommended). Issues: "
                + "; ".join(f"{i.issue_type.value}: {i.message}" for i in validation_report.critical_issues),
                context={"critical_count": len(validation_report.critical_issues)},
            )

        feature_versions = {name: self._registry.get(name).version for name in request.feature_names}
        feature_registry_fingerprint = self._registry.fingerprint(request.feature_names)
        preprocessing_definition = {name: kind.value for name, kind in request.preprocessing.items()}
        label_definition_json = request.label_definition.to_json_dict()
        split_definition_json = split_plan.to_json_dict()

        dataset_id = compute_dataset_id(
            symbol=request.symbol, base_timeframe=request.base_timeframe,
            feature_registry_fingerprint=feature_registry_fingerprint, label_definition=label_definition_json,
            split_definition=split_definition_json, preprocessing_definition=preprocessing_definition,
        )

        preprocessing_json = {group_key: p.to_json_dict() for group_key, p in fold_pipelines.items()}
        content_id, output_hashes = self._research_store.write_artifacts(
            dataset_id, splits=named_splits, preprocessing_json=preprocessing_json,
            extra_metadata={"symbol": request.symbol, "base_timeframe": request.base_timeframe.value},
        )

        input_hashes = {
            "historical_dataset_content_checksum": historical_manifest.content_checksum,
            **request.aux_input_content_hashes,
        }
        fitted_preprocessing_fingerprint = (
            _combined_fold_fingerprint(fold_pipeline_fingerprints) if request.preprocessing else None
        )
        manifest = ResearchDatasetManifest(
            dataset_id=dataset_id, version="", source_historical_dataset_id=historical_manifest.dataset_id,
            source_historical_manifest_version=historical_manifest.version, symbol=request.symbol,
            base_timeframe=request.base_timeframe, utc_start=request.start, utc_end=request.end,
            feature_names=tuple(request.feature_names), feature_versions=feature_versions,
            feature_registry_fingerprint=feature_registry_fingerprint, label_definition=label_definition_json,
            split_definition=split_definition_json, preprocessing_definition=preprocessing_definition,
            fitted_preprocessing_fingerprint=fitted_preprocessing_fingerprint,
            code_revision=capture_code_revision(), input_content_hashes=input_hashes,
            output_content_hashes=output_hashes, row_counts={name: len(df) for name, df in named_splits.items()},
            missing_data_summary=missingness_summary(features_filled),
            leakage_validation_result={
                "is_valid": validation_report.is_valid,
                "critical_count": len(validation_report.critical_issues),
                "warning_count": len(validation_report.warnings),
                "info_count": len(validation_report.infos),
            },
            created_at=pd.Timestamp.now(tz="UTC"), content_id=content_id,
        )
        version = self._manifest_store.save(manifest)
        return self._manifest_store.load(dataset_id, version)

    def _build_split_plan(self, request: ResearchDatasetBuildRequest, timestamps: pd.Series) -> SplitPlan:
        if request.split_strategy == "chronological":
            return build_chronological_split(timestamps, **request.split_params)  # type: ignore[arg-type]
        if request.split_strategy in ("expanding_walk_forward", "rolling_walk_forward"):
            params = dict(request.split_params)
            if request.split_strategy == "expanding_walk_forward":
                params.setdefault("max_train_size", None)
            return build_walk_forward_splits(timestamps, **params)  # type: ignore[arg-type]
        raise ValueError(f"Unknown split_strategy {request.split_strategy!r}")  # pragma: no cover

    def _apply_training_statistic_fills(
        self, features: pd.DataFrame, *, feature_names: tuple[str, ...], train_indices: pd.Index
    ) -> pd.DataFrame:
        result = features.copy()
        for name in feature_names:
            spec = self._registry.get(name).spec
            if spec.null_policy.kind is not MissingPolicyKind.TRAINING_STATISTIC_FILL:
                continue
            train_slice = result[name].iloc[train_indices]
            statistic = compute_training_statistic(train_slice, statistic=spec.null_policy.statistic)  # type: ignore[arg-type]
            filled = apply_missing_policy(result[name], policy=spec.null_policy, fitted_statistic=statistic)
            result[name] = filled.values
        return result


__all__ = ["ResearchDatasetBuildRequest", "ResearchDatasetBuilder"]
