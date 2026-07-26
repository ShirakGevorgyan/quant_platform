"""Command-line interface for the feature engineering / research dataset
platform (Milestone 3).

    python -m quant_platform.feature_cli list-features --config config.json
    python -m quant_platform.feature_cli describe-feature --config config.json --name return_simple_1
    python -m quant_platform.feature_cli build-research-dataset --config config.json
    python -m quant_platform.feature_cli validate-research-dataset --config config.json --dataset-id ID [--version V]
    python -m quant_platform.feature_cli inspect-lineage --config config.json --dataset-id ID [--version V]
    python -m quant_platform.feature_cli inspect-dataset-manifest --config config.json --dataset-id ID [--version V]
    python -m quant_platform.feature_cli compare-feature-drift --config config.json --dataset-id ID --reference train --comparison test [--version V]

Same operability conventions as `data_cli`: every command returns 0 on
success, non-zero on failure, and prints an actionable stderr message --
never a raw traceback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from quant_platform.config.feature_schemas import ResearchDatasetConfig
from quant_platform.core.exceptions import QuantPlatformError
from quant_platform.core.types import Timeframe
from quant_platform.features.cross_asset.cross_asset import CrossAssetWindows, register_cross_asset_features
from quant_platform.features.dataset_builder import ResearchDatasetBuilder, ResearchDatasetBuildRequest
from quant_platform.features.drift import compare_splits, render_drift_report
from quant_platform.features.lineage import build_lineage, render_lineage_report
from quant_platform.features.macro.macro_features import MacroSourceConfig, register_macro_features
from quant_platform.features.manifests import (
    ResearchDatasetManifest,
    ResearchDatasetStore,
    ResearchManifestStore,
)
from quant_platform.features.multi_timeframe import register_multi_timeframe_features
from quant_platform.features.registry import FeatureRegistry
from quant_platform.features.technical.price import register_core_technical_features
from quant_platform.features.temporal.calendar_features import register_core_temporal_features
from quant_platform.features.validation import validate_research_dataset
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader, LoadRequest
from quant_platform.historical.manifest import ManifestStore
from quant_platform.historical.models import sha256_file

_RESERVED_SPLIT_COLUMNS = ("open_time", "label", "label_valid")


def _load_config(path: Path) -> ResearchDatasetConfig:
    return ResearchDatasetConfig.model_validate_json(path.read_text())


def _parse_ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts if ts.tzinfo is not None else ts.tz_localize("UTC")


def build_registry(config: ResearchDatasetConfig) -> FeatureRegistry:
    """Deterministically reconstruct the exact feature registry `config`
    describes -- used both when building a dataset and when later
    inspecting its lineage, so the two always agree."""
    registry = FeatureRegistry()
    base_timeframe = config.build_base_timeframe()
    register_core_technical_features(registry, timeframe=base_timeframe, windows=config.technical.build())
    if config.temporal.enabled:
        register_core_temporal_features(registry, timeframe=base_timeframe, calendar=None)
    if config.multi_timeframe is not None:
        windows = config.multi_timeframe.build_windows()
        for higher_timeframe in config.multi_timeframe.build_timeframes():
            register_multi_timeframe_features(
                registry, base_timeframe=base_timeframe, higher_timeframe=higher_timeframe, windows=windows
            )
    for cross in config.cross_assets:
        register_cross_asset_features(
            registry, base_timeframe=base_timeframe,
            base_momentum_window=config.technical.momentum_windows[0] if config.technical.momentum_windows else 10,
            base_volatility_window=config.technical.volatility_window, cross_asset_symbol=cross.symbol,
            cross_asset_timeframe=cross.build_timeframe(),
            windows=CrossAssetWindows(
                return_window=cross.return_window, momentum_window=cross.momentum_window,
                volatility_window=cross.volatility_window, correlation_window=cross.correlation_window,
            ),
        )
    for macro in config.macro_sources:
        tolerance = pd.Timedelta(days=macro.tolerance_days) if macro.tolerance_days is not None else None
        register_macro_features(
            registry, base_timeframe=base_timeframe,
            config=MacroSourceConfig(source_name=macro.source_name, change_lookback=macro.change_lookback, tolerance=tolerance),
        )
    return registry


def _load_aux_data(
    config: ResearchDatasetConfig, historical_loader: DatasetLoader, *, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[dict[Timeframe, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, str]]:
    higher_timeframe_data: dict[Timeframe, pd.DataFrame] = {}
    aux_hashes: dict[str, str] = {}
    if config.multi_timeframe is not None:
        for higher_timeframe in config.multi_timeframe.build_timeframes():
            request = LoadRequest(symbol=config.symbol, timeframe=higher_timeframe, start=start, end=end)
            manifest = historical_loader.resolve_manifest(request)
            aux_hashes[f"higher_timeframe_{higher_timeframe.value}"] = manifest.content_checksum
            higher_timeframe_data[higher_timeframe] = historical_loader.load_for_engine(request)

    cross_asset_data: dict[str, pd.DataFrame] = {}
    for cross in config.cross_assets:
        request = LoadRequest(symbol=cross.symbol, timeframe=cross.build_timeframe(), start=start, end=end)
        manifest = historical_loader.resolve_manifest(request)
        aux_hashes[f"cross_asset_{cross.symbol}"] = manifest.content_checksum
        cross_asset_data[cross.symbol] = historical_loader.load_for_engine(request)

    macro_data: dict[str, pd.DataFrame] = {}
    for macro in config.macro_sources:
        df = pd.read_csv(macro.data_path)
        df["observation_time"] = pd.to_datetime(df["observation_time"], utc=True)
        df["release_time"] = pd.to_datetime(df["release_time"], utc=True)
        macro_data[macro.source_name] = df
        aux_hashes[f"macro_{macro.source_name}"] = sha256_file(macro.data_path)

    return higher_timeframe_data, cross_asset_data, macro_data, aux_hashes


def run_build_research_dataset(config: ResearchDatasetConfig) -> int:
    canonical_store = CanonicalStore(config.historical_storage_root)
    manifest_store = ManifestStore(config.historical_storage_root)
    historical_loader = DatasetLoader(canonical_store, manifest_store)
    start, end = _parse_ts(config.start), _parse_ts(config.end)

    registry = build_registry(config)
    higher_timeframe_data, cross_asset_data, macro_data, aux_hashes = _load_aux_data(
        config, historical_loader, start=start, end=end
    )

    research_store = ResearchDatasetStore(config.research_storage_root)
    research_manifest_store = ResearchManifestStore(config.research_storage_root)
    builder = ResearchDatasetBuilder(
        historical_loader=historical_loader, registry=registry, research_store=research_store,
        manifest_store=research_manifest_store, higher_timeframe_data=higher_timeframe_data,
        cross_asset_data=cross_asset_data, macro_data=macro_data,
    )

    feature_names = tuple(spec.name for spec in registry.list_features())
    request = ResearchDatasetBuildRequest(
        symbol=config.symbol, base_timeframe=config.build_base_timeframe(), start=start, end=end,
        feature_names=feature_names, label_definition=config.label.build(), split_strategy=config.split.strategy,
        split_params=config.split.build_params(), preprocessing=config.preprocessing.build(),
        dataset_version=config.dataset_version, drop_unlabeled_rows=config.drop_unlabeled_rows,
        allow_critical_validation_issues=config.allow_critical_validation_issues,
        aux_input_content_hashes=aux_hashes,
    )
    manifest = builder.build(request)
    print(f"Research dataset built: dataset_id={manifest.dataset_id} version={manifest.version}")
    print(f"Content id: {manifest.content_id}")
    print(f"Feature count: {len(manifest.feature_names)}")
    print(f"Row counts: {manifest.row_counts}")
    print(f"Leakage validation: {manifest.leakage_validation_result}")
    return 0


def cmd_build_research_dataset(args: argparse.Namespace) -> int:
    return run_build_research_dataset(_load_config(Path(args.config)))


def cmd_list_features(args: argparse.Namespace) -> int:
    registry = build_registry(_load_config(Path(args.config)))
    for spec in registry.list_features():
        print(f"{spec.name} (v{spec.version}) [{spec.category.value}] -- {spec.description}")
    return 0


def cmd_describe_feature(args: argparse.Namespace) -> int:
    registry = build_registry(_load_config(Path(args.config)))
    definition = registry.get(args.name, args.version)
    for key, value in definition.spec.to_json_dict().items():
        print(f"{key}: {value}")
    return 0


def cmd_inspect_dataset_manifest(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    manifest = ResearchManifestStore(config.research_storage_root).load(args.dataset_id, args.version)
    for key, value in manifest.to_json_dict().items():
        print(f"{key}: {value}")
    return 0


def _load_dataset_splits(
    config: ResearchDatasetConfig, *, dataset_id: str, version: str | None
) -> tuple[ResearchDatasetManifest, dict[str, pd.DataFrame]]:
    manifest = ResearchManifestStore(config.research_storage_root).load(dataset_id, version)
    research_store = ResearchDatasetStore(config.research_storage_root)
    splits = research_store.read_artifacts(dataset_id, manifest.content_id)
    if splits is None:
        raise QuantPlatformError(
            f"Research dataset content {manifest.content_id!r} for dataset_id={dataset_id!r} not found"
        )
    return manifest, splits


def cmd_validate_research_dataset(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    _, splits = _load_dataset_splits(config, dataset_id=args.dataset_id, version=args.version)
    combined = pd.concat(list(splits.values()), ignore_index=True)
    feature_columns = [c for c in combined.columns if c not in _RESERVED_SPLIT_COLUMNS]
    report = validate_research_dataset(
        combined[feature_columns], timestamps=combined["open_time"], labels=combined["label"]
    )
    print(report.summary())
    return 0 if report.is_valid else 2


def cmd_inspect_lineage(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    manifest = ResearchManifestStore(config.research_storage_root).load(args.dataset_id, args.version)
    registry = build_registry(config)
    lineages = [
        build_lineage(
            registry.get(name, manifest.feature_versions.get(name)).spec,
            source_dataset_manifest_id=manifest.source_historical_dataset_id,
            transformation=registry.get(name, manifest.feature_versions.get(name)).spec.description,
        )
        for name in manifest.feature_names
    ]
    print(render_lineage_report(lineages))
    return 0


def cmd_compare_feature_drift(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    _, splits = _load_dataset_splits(config, dataset_id=args.dataset_id, version=args.version)
    if args.reference not in splits or args.comparison not in splits:
        print(
            f"ERROR: split(s) not found (available: {sorted(splits)}, requested reference={args.reference!r} "
            f"comparison={args.comparison!r})", file=sys.stderr,
        )
        return 1
    report = compare_splits(
        splits[args.reference], splits[args.comparison], reference_name=args.reference, comparison_name=args.comparison
    )
    print(render_drift_report(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quant_platform.feature_cli",
        description="Feature engineering and research dataset platform CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-features", help="List every feature the config would register.")
    list_parser.add_argument("--config", required=True)
    list_parser.set_defaults(handler=cmd_list_features)

    describe_parser = subparsers.add_parser("describe-feature", help="Print one feature's full metadata.")
    describe_parser.add_argument("--config", required=True)
    describe_parser.add_argument("--name", required=True)
    describe_parser.add_argument("--version", default=None)
    describe_parser.set_defaults(handler=cmd_describe_feature)

    build_ds_parser = subparsers.add_parser("build-research-dataset", help="Build and persist a research dataset.")
    build_ds_parser.add_argument("--config", required=True)
    build_ds_parser.set_defaults(handler=cmd_build_research_dataset)

    validate_parser = subparsers.add_parser("validate-research-dataset", help="Validate an already-built research dataset.")
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument("--dataset-id", required=True)
    validate_parser.add_argument("--version", default=None)
    validate_parser.set_defaults(handler=cmd_validate_research_dataset)

    lineage_parser = subparsers.add_parser("inspect-lineage", help="Print a human-readable feature lineage report.")
    lineage_parser.add_argument("--config", required=True)
    lineage_parser.add_argument("--dataset-id", required=True)
    lineage_parser.add_argument("--version", default=None)
    lineage_parser.set_defaults(handler=cmd_inspect_lineage)

    manifest_parser = subparsers.add_parser("inspect-dataset-manifest", help="Print a research dataset manifest.")
    manifest_parser.add_argument("--config", required=True)
    manifest_parser.add_argument("--dataset-id", required=True)
    manifest_parser.add_argument("--version", default=None)
    manifest_parser.set_defaults(handler=cmd_inspect_dataset_manifest)

    drift_parser = subparsers.add_parser("compare-feature-drift", help="Compare two splits' feature distributions.")
    drift_parser.add_argument("--config", required=True)
    drift_parser.add_argument("--dataset-id", required=True)
    drift_parser.add_argument("--version", default=None)
    drift_parser.add_argument("--reference", default="train")
    drift_parser.add_argument("--comparison", default="test")
    drift_parser.set_defaults(handler=cmd_compare_feature_drift)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (QuantPlatformError, ValidationError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
