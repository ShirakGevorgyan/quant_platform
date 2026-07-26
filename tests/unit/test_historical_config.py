"""Tests for `config.historical_schemas` -- the Milestone 2 ingestion
config, extending (not replacing) the existing `config.schemas` system."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from quant_platform.config.historical_schemas import (
    IngestionConfig,
    MT5SourceConfig,
    ResamplingOutputConfig,
    SessionCalendarConfig,
    StorageConfig,
    TimezoneConfig,
    ValidationPolicyConfig,
    WeeklySessionConfig,
    resolve_mt5_credentials_from_env,
)
from quant_platform.historical.repair import SeverityPolicy
from quant_platform.historical.resampling import DerivedBarPolicy
from quant_platform.historical.timezones import FixedOffsetTimezone, NamedZoneTimezone


class TestTimezoneConfig:
    def test_fixed_offset_builds_correctly(self) -> None:
        cfg = TimezoneConfig(kind="fixed_offset", offset_minutes=120, label="EET")
        built = cfg.build()
        assert isinstance(built, FixedOffsetTimezone)
        assert built.offset == pd.Timedelta(hours=2)

    def test_named_zone_builds_correctly(self) -> None:
        cfg = TimezoneConfig(kind="named_zone", zone_key="America/New_York")
        built = cfg.build()
        assert isinstance(built, NamedZoneTimezone)
        assert built.key == "America/New_York"

    def test_fixed_offset_without_offset_minutes_rejected(self) -> None:
        with pytest.raises(ValidationError, match="offset_minutes is required"):
            TimezoneConfig(kind="fixed_offset")

    def test_named_zone_without_zone_key_rejected(self) -> None:
        with pytest.raises(ValidationError, match="zone_key is required"):
            TimezoneConfig(kind="named_zone")

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            TimezoneConfig(kind="fixed_offset", offset_minutes=0, bogus_field=1)


class TestMT5SourceConfig:
    def _tz(self) -> TimezoneConfig:
        return TimezoneConfig(kind="fixed_offset", offset_minutes=120)

    def test_build_produces_adapter_config(self) -> None:
        cfg = MT5SourceConfig(broker="ICMarkets", source_symbol="XAUUSDm", server_timezone=self._tz())
        built = cfg.build()
        assert built.broker == "ICMarkets"
        assert built.source_symbol == "XAUUSDm"

    def test_password_excluded_from_repr(self) -> None:
        cfg = MT5SourceConfig(
            broker="B", source_symbol="S", server_timezone=self._tz(), password="super-secret-value"
        )
        assert "super-secret-value" not in repr(cfg)

    def test_rejects_empty_broker(self) -> None:
        with pytest.raises(ValidationError):
            MT5SourceConfig(broker="", source_symbol="S", server_timezone=self._tz())

    def test_with_credentials_from_env_fills_unset_fields(self, monkeypatch) -> None:
        monkeypatch.setenv("MT5_LOGIN", "555")
        monkeypatch.setenv("MT5_PASSWORD", "envpass")
        monkeypatch.setenv("MT5_SERVER", "EnvServer")
        cfg = MT5SourceConfig(broker="B", source_symbol="S", server_timezone=self._tz())
        filled = cfg.with_credentials_from_env()
        assert filled.login == 555
        assert filled.password == "envpass"
        assert filled.server == "EnvServer"

    def test_with_credentials_from_env_never_overrides_explicit_values(self, monkeypatch) -> None:
        monkeypatch.setenv("MT5_LOGIN", "555")
        cfg = MT5SourceConfig(broker="B", source_symbol="S", server_timezone=self._tz(), login=999)
        filled = cfg.with_credentials_from_env()
        assert filled.login == 999


class TestSessionCalendarConfig:
    def test_build_produces_trading_calendar(self) -> None:
        cfg = SessionCalendarConfig(
            name="test",
            local_timezone=TimezoneConfig(kind="fixed_offset", offset_minutes=0),
            weekly_sessions=[
                WeeklySessionConfig(open_weekday=6, open_time="23:00:00", close_weekday=4, close_time="23:00:00")
            ],
        )
        calendar = cfg.build()
        assert calendar.name == "test"
        assert len(calendar.weekly_sessions) == 1

    def test_requires_at_least_one_weekly_session(self) -> None:
        with pytest.raises(ValidationError):
            SessionCalendarConfig(
                local_timezone=TimezoneConfig(kind="fixed_offset", offset_minutes=0), weekly_sessions=[]
            )


class TestValidationPolicyConfig:
    def test_build_policy_and_thresholds(self) -> None:
        cfg = ValidationPolicyConfig(severity_policy="QUARANTINE", max_price_jump_fraction=0.1)
        assert cfg.build_policy() is SeverityPolicy.QUARANTINE
        assert cfg.build_thresholds().max_price_jump_fraction == 0.1

    def test_defaults_to_strict(self) -> None:
        cfg = ValidationPolicyConfig()
        assert cfg.build_policy() is SeverityPolicy.STRICT

    def test_rejects_non_positive_jump_fraction(self) -> None:
        with pytest.raises(ValidationError):
            ValidationPolicyConfig(max_price_jump_fraction=0.0)


class TestResamplingOutputConfig:
    def test_build_targets_and_policy(self) -> None:
        cfg = ResamplingOutputConfig(target_timeframes=["M15", "H1"])
        assert [t.value for t in cfg.build_targets()] == ["M15", "H1"]
        assert cfg.build_policy() is DerivedBarPolicy.REJECT_INCOMPLETE

    def test_requires_at_least_one_target(self) -> None:
        with pytest.raises(ValidationError):
            ResamplingOutputConfig(target_timeframes=[])

    def test_rejects_unsupported_timeframe(self) -> None:
        with pytest.raises(ValidationError):
            ResamplingOutputConfig(target_timeframes=["M1"])  # M1 is the base, not a valid resampling target here


class TestIngestionConfig:
    def _mt5(self) -> MT5SourceConfig:
        return MT5SourceConfig(
            broker="B", source_symbol="S", server_timezone=TimezoneConfig(kind="fixed_offset", offset_minutes=0)
        )

    def test_requires_mt5_section_when_source_is_mt5(self) -> None:
        with pytest.raises(ValidationError, match="mt5 config section is required"):
            IngestionConfig(canonical_symbol="XAUUSD", storage=StorageConfig(storage_root="./data"))

    def test_builds_with_mt5_section_present(self) -> None:
        cfg = IngestionConfig(
            canonical_symbol="XAUUSD", mt5=self._mt5(), storage=StorageConfig(storage_root="./data"),
        )
        assert cfg.build_requested_timeframe().value == "M1"

    def test_json_round_trip_preserves_content(self) -> None:
        cfg = IngestionConfig(
            canonical_symbol="XAUUSD", mt5=self._mt5(), storage=StorageConfig(storage_root="./data"),
        )
        reloaded = IngestionConfig.model_validate_json(cfg.model_dump_json())
        assert reloaded == cfg

    def test_rejects_unknown_top_level_fields(self) -> None:
        with pytest.raises(ValidationError):
            IngestionConfig(
                canonical_symbol="XAUUSD", mt5=self._mt5(), storage=StorageConfig(storage_root="./data"),
                unknown_field=123,
            )

    def test_extraction_chunk_size_bounds(self) -> None:
        with pytest.raises(ValidationError):
            IngestionConfig(
                canonical_symbol="XAUUSD", mt5=self._mt5(), storage=StorageConfig(storage_root="./data"),
                extraction_chunk_size_days=0,
            )
        with pytest.raises(ValidationError):
            IngestionConfig(
                canonical_symbol="XAUUSD", mt5=self._mt5(), storage=StorageConfig(storage_root="./data"),
                extraction_chunk_size_days=91,
            )


class TestSafeExampleConfig:
    """`examples/ingestion_config.example.json` is the safe, no-real-
    credentials example the Milestone 2 spec requires. This proves it
    stays valid and stays credential-free as the schema evolves."""

    _EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "ingestion_config.example.json"

    def test_example_config_file_exists(self) -> None:
        assert self._EXAMPLE_PATH.is_file(), self._EXAMPLE_PATH

    def test_example_config_parses_and_validates(self) -> None:
        config = IngestionConfig.model_validate_json(self._EXAMPLE_PATH.read_text())
        assert config.canonical_symbol == "XAUUSD"
        assert config.build_requested_timeframe().value == "M1"

    def test_example_config_builds_a_working_calendar(self) -> None:
        config = IngestionConfig.model_validate_json(self._EXAMPLE_PATH.read_text())
        assert config.session_calendar is not None
        calendar = config.session_calendar.build()
        assert len(calendar.weekly_sessions) == 1

    def test_example_config_contains_no_credentials(self) -> None:
        config = IngestionConfig.model_validate_json(self._EXAMPLE_PATH.read_text())
        assert config.mt5 is not None
        assert config.mt5.login is None
        assert config.mt5.password is None
        assert config.mt5.server is None

    def test_example_config_raw_text_contains_no_suspicious_secret_looking_values(self) -> None:
        raw_text = self._EXAMPLE_PATH.read_text().lower()
        for forbidden in ("password", "secret", "login", "token", "apikey", "api_key"):
            assert forbidden not in raw_text, f"example config unexpectedly mentions {forbidden!r}"


class TestResolveMt5CredentialsFromEnv:
    def test_reads_from_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("MT5_LOGIN", "42")
        monkeypatch.setenv("MT5_PASSWORD", "hunter2")
        monkeypatch.setenv("MT5_SERVER", "SomeServer")
        creds = resolve_mt5_credentials_from_env()
        assert creds == {"login": 42, "password": "hunter2", "server": "SomeServer"}

    def test_missing_env_vars_yield_none(self, monkeypatch) -> None:
        monkeypatch.delenv("MT5_LOGIN", raising=False)
        monkeypatch.delenv("MT5_PASSWORD", raising=False)
        monkeypatch.delenv("MT5_SERVER", raising=False)
        creds = resolve_mt5_credentials_from_env()
        assert creds == {"login": None, "password": None, "server": None}
