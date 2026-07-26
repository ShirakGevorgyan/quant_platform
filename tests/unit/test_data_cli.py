"""Tests for `quant_platform.data_cli`, the historical pipeline CLI.

`run_ingest` (the testable core of the `ingest` command) is exercised
directly against a `FakeMt5Client`-backed source, never a live MT5
terminal. The `main()` entry point is exercised through its public
argv-based interface to prove exit codes and error handling behave as a
real invocation would.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.config.historical_schemas import IngestionConfig
from quant_platform.core.exceptions import SourceError
from quant_platform.core.types import Timeframe
from quant_platform.data_cli import build_parser, main, run_ingest, run_smoke_test_mt5
from quant_platform.historical.canonical_store import CanonicalStore
from quant_platform.historical.manifest import ManifestStore
from quant_platform.historical.mt5_adapter import MT5HistoricalSource
from quant_platform.historical.mt5_testing import FakeMt5Client

UTC = "UTC"


def _write_config(tmp_path: Path, *, extra: dict | None = None) -> Path:
    config_dict = {
        "canonical_symbol": "XAUUSD",
        "source_name": "mt5",
        "mt5": {
            "broker": "TestBroker", "source_symbol": "XAUUSDm",
            "server_timezone": {"kind": "fixed_offset", "offset_minutes": 120},
        },
        "requested_timeframe": "M1",
        "extraction_chunk_size_days": 3,
        "update_overlap_bars": 5,
        "storage": {"storage_root": str(tmp_path), "compression": "zstd"},
    }
    if extra:
        config_dict.update(extra)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_dict))
    return path


def _build_source(config: IngestionConfig, **fake_kwargs) -> MT5HistoricalSource:
    assert config.mt5 is not None
    return MT5HistoricalSource(config.mt5.build(), client=FakeMt5Client(**fake_kwargs))


class TestBuildParser:
    def test_all_five_commands_are_registered(self) -> None:
        parser = build_parser()
        # argparse exposes subparser choices via the subparsers action.
        subparsers_action = next(a for a in parser._actions if a.dest == "command")
        assert set(subparsers_action.choices) == {
            "ingest", "smoke-test-mt5", "validate", "resample", "inspect-manifest",
        }

    def test_missing_command_is_an_error(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestRunIngestFreshDataset:
    def test_ingests_full_range_and_resamples(self, tmp_path) -> None:
        config_path = _write_config(tmp_path, extra={"resampling": {"target_timeframes": ["H1"], "policy": "REJECT_INCOMPLETE"}})
        config = IngestionConfig.model_validate_json(config_path.read_text())
        source = _build_source(config, seed=5, base_price=2000.0)

        start = pd.Timestamp("2024-03-04", tz=UTC)
        end = pd.Timestamp("2024-03-11", tz=UTC)  # 7 days, spans multiple 3-day chunks
        rc = run_ingest(config, source, start=start, end=end)
        assert rc == 0

        cstore = CanonicalStore(tmp_path)
        m1_df, _ = cstore.read_partition(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        assert len(m1_df) == 7 * 24 * 60

        h1_df, _ = cstore.read_partition(symbol="XAUUSD", timeframe=Timeframe.H1, year=2024)
        assert len(h1_df) == 7 * 24

    def test_manifest_records_resampling_config_for_derived_dataset(self, tmp_path) -> None:
        config_path = _write_config(tmp_path, extra={"resampling": {"target_timeframes": ["H1"]}})
        config = IngestionConfig.model_validate_json(config_path.read_text())
        source = _build_source(config, seed=5)
        run_ingest(config, source, start=pd.Timestamp("2024-03-04", tz=UTC), end=pd.Timestamp("2024-03-06", tz=UTC))

        mstore = ManifestStore(tmp_path)
        manifest = mstore.load(symbol="XAUUSD", timeframe=Timeframe.H1)
        assert manifest.resampling_config == {"source_timeframe": "M1", "policy": "REJECT_INCOMPLETE"}

    def test_no_resampling_section_means_no_derived_dataset(self, tmp_path) -> None:
        config_path = _write_config(tmp_path)
        config = IngestionConfig.model_validate_json(config_path.read_text())
        source = _build_source(config, seed=5)
        run_ingest(config, source, start=pd.Timestamp("2024-03-04", tz=UTC), end=pd.Timestamp("2024-03-06", tz=UTC))

        cstore = CanonicalStore(tmp_path)
        assert cstore.list_years(symbol="XAUUSD", timeframe=Timeframe.H1) == []


class TestRunIngestIncremental:
    def test_second_ingest_call_extends_rather_than_duplicates(self, tmp_path) -> None:
        config_path = _write_config(tmp_path)
        config = IngestionConfig.model_validate_json(config_path.read_text())

        run_ingest(
            config, _build_source(config, seed=5),
            start=pd.Timestamp("2024-03-04", tz=UTC), end=pd.Timestamp("2024-03-06", tz=UTC),
        )
        cstore = CanonicalStore(tmp_path)
        first_df, _ = cstore.read_partition(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        first_count = len(first_df)

        # Second call uses `determine_update_start`, so its own `start` is
        # effectively ignored in favor of continuing from the existing data.
        run_ingest(
            config, _build_source(config, seed=5),
            start=pd.Timestamp("2024-03-04", tz=UTC), end=pd.Timestamp("2024-03-08", tz=UTC),
        )
        second_df, _ = cstore.read_partition(symbol="XAUUSD", timeframe=Timeframe.M1, year=2024)
        assert len(second_df) > first_count
        assert second_df["open_time"].is_monotonic_increasing
        assert not second_df["open_time"].duplicated().any()


class TestMainEntryPoint:
    def test_bad_config_path_returns_nonzero_with_stderr_message(self, tmp_path, capsys) -> None:
        rc = main(["inspect-manifest", "--config", str(tmp_path / "does_not_exist.json"), "--symbol", "XAUUSD", "--timeframe", "M1"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.err
        assert captured.out == ""  # never prints a success message on failure

    def test_inspect_manifest_after_ingest_via_cli(self, tmp_path, capsys) -> None:
        config_path = _write_config(tmp_path, extra={"resampling": {"target_timeframes": ["H1"]}})
        config = IngestionConfig.model_validate_json(config_path.read_text())
        run_ingest(
            config, _build_source(config, seed=5),
            start=pd.Timestamp("2024-03-04", tz=UTC), end=pd.Timestamp("2024-03-06", tz=UTC),
        )
        rc = main(["inspect-manifest", "--config", str(config_path), "--symbol", "XAUUSD", "--timeframe", "H1"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "dataset_id:" in captured.out
        assert "resampling_config:" in captured.out

    def test_invalid_timeframe_choice_is_rejected_by_argparse(self, tmp_path) -> None:
        config_path = _write_config(tmp_path)
        with pytest.raises(SystemExit):
            main(["inspect-manifest", "--config", str(config_path), "--symbol", "XAUUSD", "--timeframe", "NOT_A_TIMEFRAME"])

    def test_validate_command_reports_clean_data_as_valid(self, tmp_path) -> None:
        config_path = _write_config(tmp_path)
        config = IngestionConfig.model_validate_json(config_path.read_text())
        run_ingest(
            config, _build_source(config, seed=5),
            start=pd.Timestamp("2024-03-04", tz=UTC), end=pd.Timestamp("2024-03-06", tz=UTC),
        )
        rc = main([
            "validate", "--config", str(config_path), "--symbol", "XAUUSD", "--timeframe", "M1",
            "--start", "2024-03-04T00:00:00", "--end", "2024-03-06T00:00:00",
        ])
        assert rc == 0

    def test_resample_command_via_cli(self, tmp_path) -> None:
        config_path = _write_config(tmp_path)
        config = IngestionConfig.model_validate_json(config_path.read_text())
        run_ingest(
            config, _build_source(config, seed=5),
            start=pd.Timestamp("2024-03-04", tz=UTC), end=pd.Timestamp("2024-03-06", tz=UTC),
        )
        rc = main([
            "resample", "--config", str(config_path), "--symbol", "XAUUSD",
            "--source-timeframe", "M1", "--target-timeframe", "H1",
            "--start", "2024-03-04T00:00:00", "--end", "2024-03-06T00:00:00",
        ])
        assert rc == 0
        cstore = CanonicalStore(tmp_path)
        h1_df, _ = cstore.read_partition(symbol="XAUUSD", timeframe=Timeframe.H1, year=2024)
        assert len(h1_df) > 0


class TestNoCredentialsInOutput:
    def test_manifest_inspection_never_prints_a_password(self, tmp_path, capsys) -> None:
        config_dict_path = _write_config(tmp_path)
        raw = json.loads(config_dict_path.read_text())
        raw["mt5"]["password"] = "super-secret-value-should-never-appear"
        config_dict_path.write_text(json.dumps(raw))
        config = IngestionConfig.model_validate_json(config_dict_path.read_text())
        run_ingest(
            config, _build_source(config, seed=5),
            start=pd.Timestamp("2024-03-04", tz=UTC), end=pd.Timestamp("2024-03-06", tz=UTC),
        )
        main(["inspect-manifest", "--config", str(config_dict_path), "--symbol", "XAUUSD", "--timeframe", "M1"])
        captured = capsys.readouterr()
        assert "super-secret-value-should-never-appear" not in captured.out
        assert "super-secret-value-should-never-appear" not in captured.err


class TestSmokeTestMt5:
    """`smoke-test-mt5` is exercised against `FakeMt5Client` here (never a
    live terminal); the real value of this command is the diagnostics it
    prints when pointed at a REAL MT5 connection, which cannot be
    exercised in this environment -- see docs/historical_data_pipeline.md's
    "Real MT5 adapter verification procedure" for the manual steps an
    operator runs once a live terminal is available."""

    def test_successful_connection_reports_diagnostics_and_returns_zero(self, tmp_path, capsys) -> None:
        config_path = _write_config(tmp_path)
        config = IngestionConfig.model_validate_json(config_path.read_text())
        source = _build_source(config, seed=1)
        rc = run_smoke_test_mt5(config, source)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Connected." in captured.out
        assert "Disconnected." in captured.out
        assert "resolved successfully" in captured.out
        assert "Bars received:" in captured.out

    def test_invalid_symbol_alias_propagates_as_a_failure_not_silently_swallowed(self, tmp_path) -> None:
        config_path = _write_config(tmp_path)
        config = IngestionConfig.model_validate_json(config_path.read_text())
        source = _build_source(config, seed=1, fail_symbol_select=True)
        with pytest.raises(SourceError):
            run_smoke_test_mt5(config, source)

    def test_never_prints_the_configured_password(self, tmp_path, capsys) -> None:
        config_dict_path = _write_config(tmp_path)
        raw = json.loads(config_dict_path.read_text())
        raw["mt5"]["password"] = "super-secret-value-should-never-appear"
        config_dict_path.write_text(json.dumps(raw))
        config = IngestionConfig.model_validate_json(config_dict_path.read_text())
        source = _build_source(config, seed=1)
        run_smoke_test_mt5(config, source)
        captured = capsys.readouterr()
        assert "super-secret-value-should-never-appear" not in captured.out
        assert "super-secret-value-should-never-appear" not in captured.err

    def test_disconnects_even_when_fetch_fails(self, tmp_path) -> None:
        config_path = _write_config(tmp_path)
        config = IngestionConfig.model_validate_json(config_path.read_text())
        fake = FakeMt5Client(seed=1, fail_symbol_select=True)
        source = MT5HistoricalSource(config.mt5.build(), client=fake)
        with pytest.raises(SourceError):
            run_smoke_test_mt5(config, source)
        assert fake.shutdown_call_count == 1

    def test_registered_as_a_cli_subcommand(self) -> None:
        parser = build_parser()
        subparsers_action = next(a for a in parser._actions if a.dest == "command")
        assert "smoke-test-mt5" in subparsers_action.choices

    def test_via_main_without_a_real_mt5_package_fails_actionably_not_silently(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        # `cmd_smoke_test_mt5` builds a REAL `MT5HistoricalSource` (no fake
        # injected) exactly like `cmd_ingest` does. Force the "package not
        # installed" branch deterministically via `sys.modules[name] = None`
        # (the standard idiom for forcing `ImportError` on a bare `import`)
        # rather than relying on the ambient environment actually lacking
        # the real `MetaTrader5` package -- that package may well be
        # pip-installed here even with no MT5 terminal running. This proves
        # the real dispatch path surfaces `MissingDependencyError` as an
        # actionable, non-zero-exit CLI failure rather than crashing with a
        # raw traceback or silently reporting success.
        monkeypatch.setitem(sys.modules, "MetaTrader5", None)
        config_path = _write_config(tmp_path)
        rc = main(["smoke-test-mt5", "--config", str(config_path)])
        assert rc == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.err
        assert "MetaTrader5" in captured.err
