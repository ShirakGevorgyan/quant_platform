"""Command-line interface for the historical data ingestion pipeline.

    python -m quant_platform.data_cli ingest --config config.json --start 2024-01-01T00:00:00Z --end 2024-02-01T00:00:00Z
    python -m quant_platform.data_cli validate --config config.json --symbol XAUUSD --timeframe M1 --start ... --end ...
    python -m quant_platform.data_cli resample --config config.json --symbol XAUUSD --source-timeframe M1 --target-timeframe H1 --start ... --end ...
    python -m quant_platform.data_cli inspect-manifest --config config.json --symbol XAUUSD --timeframe M1 [--version V]

No web server, no interactive prompts -- every command is a single
non-interactive invocation. Every command returns 0 on success, a non-zero
exit code on failure, and prints an actionable error to stderr -- never a
raw traceback, and never any credential (MT5 passwords never appear in any
config repr, log line, or exception message anywhere in this pipeline; see
`config.historical_schemas.MT5SourceConfig`). A command that fails partway
through never prints a success message -- `main()`'s exception handling
runs before any "done" output, so a non-zero exit code is always a
reliable signal that the requested work did not fully complete.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from quant_platform.config.historical_schemas import IngestionConfig
from quant_platform.core.exceptions import QuantPlatformError
from quant_platform.core.time_utils import to_pandas_freq
from quant_platform.core.types import Timeframe
from quant_platform.historical import PIPELINE_VERSION
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.loader import DatasetLoader, LoadRequest
from quant_platform.historical.manifest import ManifestStore
from quant_platform.historical.models import RAW_HISTORICAL_COLUMNS
from quant_platform.historical.mt5_adapter import MT5HistoricalSource
from quant_platform.historical.quality import run_quality_checks
from quant_platform.historical.raw_store import RawSnapshotStore
from quant_platform.historical.repair import apply_repair_policy
from quant_platform.historical.resampling import resample_ohlcv
from quant_platform.historical.source import HistoricalSource, SourceRequest
from quant_platform.historical.update_pipeline import apply_incremental_update, determine_update_start

logger = logging.getLogger(__name__)


def _load_config(path: Path) -> IngestionConfig:
    return IngestionConfig.model_validate_json(path.read_text())


def run_ingest(config: IngestionConfig, source: HistoricalSource, *, start: pd.Timestamp, end: pd.Timestamp) -> int:
    """The testable core of the `ingest` command. Takes an
    already-constructed `source` -- a real `MT5HistoricalSource` in
    production, `historical.mt5_testing.FakeMt5Client`-backed in tests --
    so this orchestration logic is fully exercised without ever requiring
    a live MT5 terminal."""
    assert config.mt5 is not None  # enforced by IngestionConfig's own validator
    timeframe = config.build_requested_timeframe()
    canonical_store = CanonicalStore(config.storage.storage_root, compression=config.storage.compression)
    manifest_store = ManifestStore(config.storage.storage_root)
    raw_store = RawSnapshotStore(config.storage.storage_root)
    calendar = config.session_calendar.build() if config.session_calendar else None
    thresholds = config.validation.build_thresholds()
    severity_policy = config.validation.build_policy()

    incremental_start = determine_update_start(
        canonical_store, symbol=config.canonical_symbol, timeframe=timeframe, overlap_bars=config.update_overlap_bars
    )
    effective_start = incremental_start if incremental_start is not None else start
    if effective_start >= end:
        print(f"Nothing to do: effective start {effective_start} is not before end {end}")
        return 0

    chunk_delta = pd.Timedelta(days=config.extraction_chunk_size_days)
    total_inserted = 0
    total_conflicting = 0

    source.connect()
    try:
        cursor = effective_start
        while cursor < end:
            chunk_end = min(cursor + chunk_delta, end)
            request = SourceRequest(symbol=config.mt5.source_symbol, timeframe=timeframe, start=cursor, end=chunk_end)
            batches = list(source.fetch_all(request))
            if not batches:
                cursor = chunk_end
                continue

            raw_df = pd.concat([b.data for b in batches], ignore_index=True)
            extracted_at = pd.Timestamp.now(tz="UTC")
            snapshot_metadata = raw_store.write_snapshot(
                raw_df, source_name=config.source_name, source_version=batches[-1].metadata.source_version,
                broker=config.mt5.broker, symbol=config.canonical_symbol, source_symbol=config.mt5.source_symbol,
                timeframe=timeframe, requested_start=cursor, requested_end=chunk_end,
                server_timezone_repr=str(config.mt5.server_timezone.build()), extracted_at=extracted_at,
                is_complete=not batches[-1].is_partial,
            )

            quality_report = run_quality_checks(
                raw_df, symbol=config.canonical_symbol, timeframe=timeframe, calendar=calendar, thresholds=thresholds
            )
            repair_result = apply_repair_policy(
                raw_df, quality_report, policy=severity_policy, allow_sort=config.validation.allow_sort,
                allow_exact_duplicate_removal=config.validation.allow_exact_duplicate_removal,
                input_snapshot_id=snapshot_metadata.snapshot_id,
            )

            if len(repair_result.data) > 0:
                update_report = apply_incremental_update(
                    canonical_store, manifest_store, repair_result.data,
                    symbol=config.canonical_symbol, timeframe=timeframe, source_name=config.source_name,
                    broker=config.mt5.broker, pipeline_version=PIPELINE_VERSION,
                    parent_snapshot_ids=(snapshot_metadata.snapshot_id,),
                    requested_start=cursor, requested_end=chunk_end, revision_policy=config.build_revision_policy(),
                    quality_summary={
                        "critical": len(quality_report.critical_issues), "warning": len(quality_report.warnings),
                    },
                    repair_summary={
                        "rows_removed": repair_result.lineage.rows_in - repair_result.lineage.rows_out,
                        "rows_quarantined": repair_result.lineage.rows_quarantined,
                    },
                )
                total_inserted += update_report.rows_inserted
                total_conflicting += update_report.rows_conflicting
                logger.info(
                    "ingest chunk complete: symbol=%s timeframe=%s range=[%s, %s) inserted=%d",
                    config.canonical_symbol, timeframe.value, cursor, chunk_end, update_report.rows_inserted,
                )
            cursor = chunk_end

        if config.resampling is not None:
            _run_resampling(config, canonical_store, manifest_store, timeframe, start, end)
    finally:
        source.disconnect()

    print(
        f"Ingestion complete: symbol={config.canonical_symbol} timeframe={timeframe.value} "
        f"rows_inserted={total_inserted} rows_revised={total_conflicting}"
    )
    return 0


def _run_resampling(
    config: IngestionConfig, canonical_store: CanonicalStore, manifest_store: ManifestStore,
    source_timeframe: Timeframe, start: pd.Timestamp, end: pd.Timestamp,
) -> None:
    assert config.resampling is not None and config.mt5 is not None
    loader = DatasetLoader(canonical_store, manifest_store)
    source_df = loader.load(
        LoadRequest(symbol=config.canonical_symbol, timeframe=source_timeframe, start=start, end=end, required_quality="lenient")
    )
    for target_timeframe in config.resampling.build_targets():
        derived = resample_ohlcv(
            source_df, source_timeframe=source_timeframe, target_timeframe=target_timeframe,
            policy=config.resampling.build_policy(),
        )
        if len(derived) == 0:
            logger.info("resample %s -> %s produced no complete bars in range", source_timeframe.value, target_timeframe.value)
            continue
        derived_canonical = derived[list(RAW_HISTORICAL_COLUMNS)]
        apply_incremental_update(
            canonical_store, manifest_store, derived_canonical,
            symbol=config.canonical_symbol, timeframe=target_timeframe, source_name=config.source_name,
            broker=config.mt5.broker, pipeline_version=PIPELINE_VERSION, parent_snapshot_ids=(),
            requested_start=start, requested_end=end,
            resampling_config={"source_timeframe": source_timeframe.value, "policy": config.resampling.policy},
        )
        logger.info("resample %s -> %s wrote %d bar(s)", source_timeframe.value, target_timeframe.value, len(derived_canonical))


def cmd_ingest(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    if config.mt5 is None:
        raise QuantPlatformError("config.mt5 section is required for the 'ingest' command")
    mt5_config = config.mt5.with_credentials_from_env()
    source = MT5HistoricalSource(mt5_config.build())
    start = pd.Timestamp(args.start, tz="UTC") if pd.Timestamp(args.start).tzinfo is None else pd.Timestamp(args.start)
    end = pd.Timestamp(args.end, tz="UTC") if pd.Timestamp(args.end).tzinfo is None else pd.Timestamp(args.end)
    return run_ingest(config, source, start=start, end=end)


def run_smoke_test_mt5(config: IngestionConfig, source: HistoricalSource) -> int:
    """The testable core of the `smoke-test-mt5` command: exercises the
    real (or, in tests, injected-fake) adapter's full connect -> fetch ->
    disconnect lifecycle against a small, recent, bounded window, and
    prints diagnostics for verifying a REAL MT5 connection before trusting
    it for production ingestion:

      - connectivity: did `connect()` succeed at all.
      - symbol alias verification: did the configured `source_symbol`
        resolve -- an invalid alias raises `SourceError` inside `fetch()`,
        which propagates as a normal command failure (non-zero exit,
        actionable message), it is not silently swallowed here.
      - broker timezone verification: the most recent returned bar's UTC
        `open_time`, after conversion via the configured
        `server_timezone`, should be recent relative to "now". A
        wrong-signed or wrong-magnitude offset is the single most common
        real-world timezone misconfiguration, and it manifests exactly as
        "the most recent bar looks hours older (or newer) than it should".
      - boundary/bar-count reconciliation: how many bars were actually
        returned for a fixed recent window vs. the naive full-session
        expectation, explicitly caveated for session closures (a
        low/zero count is EXPECTED if the window fell in a weekend/
        maintenance break, not necessarily a problem).

    Never receives, logs, or prints the configured password/login -- this
    function only ever sees the already-constructed `source`.
    """
    assert config.mt5 is not None
    timeframe = config.build_requested_timeframe()
    duration = pd.Timedelta(timeframe.duration)
    now = pd.Timestamp.now(tz="UTC")
    # Floor to the timeframe grid, then step back one full bar so the most
    # recent bar requested is guaranteed to have already closed (never ask
    # for a still-forming bar).
    end = now.floor(to_pandas_freq(timeframe)) - duration
    start = end - pd.Timedelta(hours=1)

    print(f"Connecting: broker={config.mt5.broker} source_symbol={config.mt5.source_symbol} timeframe={timeframe.value}")
    source.connect()
    try:
        print("Connected.")
        request = SourceRequest(symbol=config.mt5.source_symbol, timeframe=timeframe, start=start, end=end, max_batch_size=10_000)
        batch = source.fetch(request)
    finally:
        source.disconnect()
        print("Disconnected.")

    print(f"Symbol alias {config.mt5.source_symbol!r} resolved successfully (fetch did not raise).")

    row_count = len(batch.data)
    expected_bar_count = int((end - start) / duration)
    print(f"Requested window: [{start}, {end}) -- {expected_bar_count} bar(s) expected if the market was fully open.")
    print(f"Bars received: {row_count}")
    if row_count == 0:
        print(
            "NOTE: zero bars returned. This is EXPECTED if the window fell entirely within a "
            "weekend/holiday/maintenance closure -- cross-check against your broker's published "
            "schedule before treating this as a connectivity problem."
        )
    elif row_count < expected_bar_count:
        print(
            f"NOTE: received fewer bars ({row_count}) than the full-session expectation "
            f"({expected_bar_count}) -- confirm this matches an expected session/maintenance "
            "closure for this window before treating it as a data-quality problem."
        )
    else:
        print("Bar count matches the full-session expectation.")

    if row_count > 0:
        most_recent_open_time = pd.Timestamp(batch.data["open_time"].max())
        age = pd.Timestamp.now(tz="UTC") - most_recent_open_time
        print(f"Most recent bar open_time (UTC): {most_recent_open_time} ({age} old)")
        if age > timeframe.duration * 5:
            print(
                "WARNING: the most recent bar is much older than expected relative to now. This "
                "is the classic symptom of an incorrect server_timezone configuration (wrong "
                "offset, or a fixed offset used for a broker that actually shifts with DST) -- "
                "verify config.mt5.server_timezone against your broker's published server-time "
                "documentation before trusting this connection for ingestion."
            )
        else:
            print("Broker timezone configuration looks plausible (most recent bar is recent).")

    print("Smoke test complete.")
    return 0


def cmd_smoke_test_mt5(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    if config.mt5 is None:
        raise QuantPlatformError("config.mt5 section is required for the 'smoke-test-mt5' command")
    mt5_config = config.mt5.with_credentials_from_env()
    source = MT5HistoricalSource(mt5_config.build())
    return run_smoke_test_mt5(config, source)


def cmd_validate(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    canonical_store = CanonicalStore(config.storage.storage_root, compression=config.storage.compression)
    manifest_store = ManifestStore(config.storage.storage_root)
    timeframe = Timeframe(args.timeframe)
    start = pd.Timestamp(args.start, tz="UTC") if pd.Timestamp(args.start).tzinfo is None else pd.Timestamp(args.start)
    end = pd.Timestamp(args.end, tz="UTC") if pd.Timestamp(args.end).tzinfo is None else pd.Timestamp(args.end)

    loader = DatasetLoader(canonical_store, manifest_store)
    df = loader.load(LoadRequest(symbol=args.symbol, timeframe=timeframe, start=start, end=end, required_quality="lenient"))
    calendar = config.session_calendar.build() if config.session_calendar else None
    report = run_quality_checks(
        df, symbol=args.symbol, timeframe=timeframe, calendar=calendar, thresholds=config.validation.build_thresholds()
    )
    print(report.summary())
    return 0 if report.is_valid else 2


def cmd_resample(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    if config.mt5 is None:
        raise QuantPlatformError("config.mt5 section is required for the 'resample' command (used for broker identity)")
    canonical_store = CanonicalStore(config.storage.storage_root, compression=config.storage.compression)
    manifest_store = ManifestStore(config.storage.storage_root)
    source_timeframe = Timeframe(args.source_timeframe)
    target_timeframe = Timeframe(args.target_timeframe)
    start = pd.Timestamp(args.start, tz="UTC") if pd.Timestamp(args.start).tzinfo is None else pd.Timestamp(args.start)
    end = pd.Timestamp(args.end, tz="UTC") if pd.Timestamp(args.end).tzinfo is None else pd.Timestamp(args.end)

    loader = DatasetLoader(canonical_store, manifest_store)
    source_df = loader.load(
        LoadRequest(symbol=args.symbol, timeframe=source_timeframe, start=start, end=end, required_quality="lenient")
    )
    derived = resample_ohlcv(source_df, source_timeframe=source_timeframe, target_timeframe=target_timeframe)
    if len(derived) == 0:
        print("No complete derived bars produced for the requested range.", file=sys.stderr)
        return 1

    derived_canonical = derived[list(RAW_HISTORICAL_COLUMNS)]
    report = apply_incremental_update(
        canonical_store, manifest_store, derived_canonical,
        symbol=args.symbol, timeframe=target_timeframe, source_name=config.source_name, broker=config.mt5.broker,
        pipeline_version=PIPELINE_VERSION, parent_snapshot_ids=(), requested_start=start, requested_end=end,
        resampling_config={"source_timeframe": source_timeframe.value, "policy": "REJECT_INCOMPLETE"},
    )
    print(
        f"Resampled {source_timeframe.value} -> {target_timeframe.value}: "
        f"{report.rows_inserted} row(s) inserted, manifest version {report.manifest_version}"
    )
    return 0


def cmd_inspect_manifest(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    manifest_store = ManifestStore(config.storage.storage_root)
    timeframe = Timeframe(args.timeframe)
    manifest = manifest_store.load(symbol=args.symbol, timeframe=timeframe, version=args.version)
    for key, value in manifest.to_json_dict().items():
        print(f"{key}: {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quant_platform.data_cli", description="Historical data ingestion pipeline CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest historical bars from the configured source into canonical storage."
    )
    ingest_parser.add_argument("--config", required=True, help="Path to an IngestionConfig JSON file.")
    ingest_parser.add_argument("--start", required=True, help="ISO8601 start (used only for a fresh dataset; incremental updates ignore this).")
    ingest_parser.add_argument("--end", required=True, help="ISO8601 end (exclusive).")
    ingest_parser.set_defaults(handler=cmd_ingest)

    smoke_test_parser = subparsers.add_parser(
        "smoke-test-mt5",
        help="Connect to the configured MT5 source and print connectivity/timezone/bar-count diagnostics. Never logs credentials.",
    )
    smoke_test_parser.add_argument("--config", required=True)
    smoke_test_parser.set_defaults(handler=cmd_smoke_test_mt5)

    validate_parser = subparsers.add_parser("validate", help="Run quality checks against an already-canonicalized range.")
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument("--symbol", required=True)
    validate_parser.add_argument("--timeframe", required=True, choices=[t.value for t in Timeframe])
    validate_parser.add_argument("--start", required=True)
    validate_parser.add_argument("--end", required=True)
    validate_parser.set_defaults(handler=cmd_validate)

    resample_parser = subparsers.add_parser("resample", help="Resample an existing canonical dataset to a coarser timeframe.")
    resample_parser.add_argument("--config", required=True)
    resample_parser.add_argument("--symbol", required=True)
    resample_parser.add_argument("--source-timeframe", required=True, choices=[t.value for t in Timeframe])
    resample_parser.add_argument("--target-timeframe", required=True, choices=[t.value for t in Timeframe])
    resample_parser.add_argument("--start", required=True)
    resample_parser.add_argument("--end", required=True)
    resample_parser.set_defaults(handler=cmd_resample)

    inspect_parser = subparsers.add_parser("inspect-manifest", help="Print a dataset manifest (latest version by default).")
    inspect_parser.add_argument("--config", required=True)
    inspect_parser.add_argument("--symbol", required=True)
    inspect_parser.add_argument("--timeframe", required=True, choices=[t.value for t in Timeframe])
    inspect_parser.add_argument("--version", default=None, help="Specific manifest version; omit for latest.")
    inspect_parser.set_defaults(handler=cmd_inspect_manifest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (QuantPlatformError, ValidationError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
