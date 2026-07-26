"""Historical market-data ingestion, normalization, validation, storage,
resampling, versioning, and reproducible loading pipeline (Milestone 2).

Broker-agnostic at its core (`historical.source.HistoricalSource`), with
MetaTrader5 as the first concrete adapter (`historical.mt5_adapter`). See
the package README for the full data lifecycle: raw source ->
immutable raw snapshot -> quality validation -> repair/quarantine policy
-> canonical Parquet storage -> leak-free resampling -> dataset manifest
-> reproducible dataset loader -> `multiframe.cursor.TimeframeCursor` /
`engine.backtest_engine.BacktestEngine`.
"""

from quant_platform.historical.calendar import (
    DailyMaintenanceBreak,
    HolidayClosure,
    TradingCalendar,
    WeeklySession,
    default_xauusd_calendar,
)
from quant_platform.historical.canonical_store import CanonicalStore, CompressionCodec, PartitionMetadata
from quant_platform.historical.loader import DatasetLoader, LoadRequest, RequiredQuality
from quant_platform.historical.manifest import DatasetManifest, ManifestStore
from quant_platform.historical.models import (
    RAW_HISTORICAL_COLUMNS,
    coerce_historical_dtypes,
    schema_fingerprint,
    spread_points_to_price,
    validate_historical_schema,
)
from quant_platform.historical.mt5_adapter import MT5AdapterConfig, Mt5ClientProtocol, MT5HistoricalSource
from quant_platform.historical.quality import (
    IssueType,
    QualityIssue,
    QualityReport,
    QualityThresholds,
    Severity,
    run_quality_checks,
)
from quant_platform.historical.raw_store import RawSnapshotStore, SnapshotMetadata
from quant_platform.historical.repair import (
    RepairAction,
    RepairLineage,
    RepairResult,
    RepairStep,
    SeverityPolicy,
    apply_repair_policy,
)
from quant_platform.historical.resampling import DerivedBarPolicy, resample_ohlcv
from quant_platform.historical.source import HistoricalSource, SourceBatch, SourceMetadata, SourceRequest
from quant_platform.historical.timezones import (
    FixedOffsetTimezone,
    NamedZoneTimezone,
    SourceTimezone,
    localize_broker_timestamps,
    require_utc,
)
from quant_platform.historical.update_pipeline import (
    RevisionPolicy,
    UpdateReport,
    apply_incremental_update,
    determine_update_start,
)

PIPELINE_VERSION = "1.0.0"
"""Version of the historical ingestion pipeline's processing logic itself
(schema + normalization + validation + repair + resampling semantics) --
recorded in every `DatasetManifest.pipeline_version` so a manifest is
always traceable to which version of this code produced it. Bump this on
any change to those semantics, independent of `quant_platform.__version__`
(the package release version)."""

__all__ = [
    "PIPELINE_VERSION",
    "RAW_HISTORICAL_COLUMNS",
    "CanonicalStore",
    "CompressionCodec",
    "DailyMaintenanceBreak",
    "DatasetLoader",
    "DatasetManifest",
    "DerivedBarPolicy",
    "FixedOffsetTimezone",
    "HistoricalSource",
    "HolidayClosure",
    "IssueType",
    "LoadRequest",
    "MT5AdapterConfig",
    "MT5HistoricalSource",
    "ManifestStore",
    "Mt5ClientProtocol",
    "NamedZoneTimezone",
    "PartitionMetadata",
    "QualityIssue",
    "QualityReport",
    "QualityThresholds",
    "RawSnapshotStore",
    "RepairAction",
    "RepairLineage",
    "RepairResult",
    "RepairStep",
    "RequiredQuality",
    "RevisionPolicy",
    "Severity",
    "SeverityPolicy",
    "SnapshotMetadata",
    "SourceBatch",
    "SourceMetadata",
    "SourceRequest",
    "SourceTimezone",
    "TradingCalendar",
    "UpdateReport",
    "WeeklySession",
    "apply_incremental_update",
    "apply_repair_policy",
    "coerce_historical_dtypes",
    "default_xauusd_calendar",
    "determine_update_start",
    "localize_broker_timestamps",
    "require_utc",
    "resample_ohlcv",
    "run_quality_checks",
    "schema_fingerprint",
    "spread_points_to_price",
    "validate_historical_schema",
]
