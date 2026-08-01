"""`examples/xauusd_cross_asset_config.example.json` is the safe,
no-real-credentials example the Milestone 10 Phase 4C spec requires.
This proves it stays valid, stays credential-free, and can actually
construct real cross-asset registry/mapping/backfill objects as the
schema evolves -- mirrors `test_curated_fred_example_config.py`'s own
established pattern for Phase 4B's own example config."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from quant_platform.market_data.collectors.cross_asset.acceptance import (
    ALPHA_VANTAGE_API_KEY_ENV_VAR,
)
from quant_platform.market_data.collectors.cross_asset.adjustment import (
    AdjustmentPolicyKind,
    create_adjustment_policy,
)
from quant_platform.market_data.collectors.cross_asset.availability import (
    BarAvailabilityPolicyKind,
    create_bar_availability_policy,
)
from quant_platform.market_data.collectors.cross_asset.instrument_form import (
    InstrumentForm,
    ProxyQuality,
    create_proxy_policy,
)
from quant_platform.market_data.collectors.cross_asset.market_backfill import (
    MarketCachePolicy,
    create_market_backfill_spec,
)
from quant_platform.market_data.collectors.cross_asset.registry import (
    create_curated_market_driver_registry,
    default_core_market_driver_specs,
    default_optional_market_driver_specs,
)
from quant_platform.market_data.collectors.cross_asset.sessions import (
    CandleTimestampConvention,
    create_timezone_session_policy,
)
from quant_platform.market_data.collectors.cross_asset.symbol_mapping import (
    create_provider_symbol_mapping,
    create_symbol_mapping_set,
)

_EXAMPLE_PATH = Path(__file__).resolve().parents[3] / "examples" / "xauusd_cross_asset_config.example.json"


def _load() -> dict:
    return json.loads(_EXAMPLE_PATH.read_text(encoding="utf-8"))


def _parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


class TestExampleConfigFileExists:
    def test_file_exists_and_is_valid_json(self) -> None:
        assert _EXAMPLE_PATH.is_file(), _EXAMPLE_PATH
        config = _load()
        assert isinstance(config, dict)


class TestExampleConfigMatchesCoreRegistry:
    def test_core_driver_ids_match_the_five_mandatory_core_concepts(self) -> None:
        config = _load()
        core_ids = {entry["canonical_driver_id"] for entry in config["curated_registry"]["core_drivers"]}
        assert core_ids == {"us_dollar_strength", "wti_crude", "brent_crude", "silver", "gold_reference"}

    def test_optional_driver_ids_match_the_five_strong_optional_concepts(self) -> None:
        config = _load()
        optional_ids = {entry["canonical_driver_id"] for entry in config["curated_registry"]["optional_drivers"]}
        assert optional_ids == {
            "us_equity_market_stress", "treasury_volatility", "broad_commodity_index", "copper_industrial_growth", "gold_miner_equity",
        }

    def test_treasury_volatility_is_honestly_disabled(self) -> None:
        config = _load()
        entry = next(e for e in config["curated_registry"]["optional_drivers"] if e["canonical_driver_id"] == "treasury_volatility")
        assert entry["enabled"] is False

    def test_nine_of_ten_concepts_are_enabled(self) -> None:
        config = _load()
        all_entries = config["curated_registry"]["core_drivers"] + config["curated_registry"]["optional_drivers"]
        assert len(all_entries) == 10
        assert sum(1 for e in all_entries if e["enabled"]) == 9

    def test_provider_mapping_entries_match_the_enabled_drivers(self) -> None:
        config = _load()
        all_entries = config["curated_registry"]["core_drivers"] + config["curated_registry"]["optional_drivers"]
        enabled_ids = {e["canonical_driver_id"] for e in all_entries if e["enabled"]}
        mapping_ids = {e["canonical_driver_id"] for e in config["provider_mappings"]["entries"]}
        assert mapping_ids == enabled_ids

    def test_every_provider_mapping_is_classified_as_a_proxy(self) -> None:
        config = _load()
        assert config["provider_mappings"]["instrument_form"] == "etf"
        # Every ETF-form mapping is structurally required to be a proxy (see
        # symbol_mapping.py's own ETF-is-always-a-proxy guard, exercised below).
        for entry in config["provider_mappings"]["entries"]:
            assert entry["proxy_quality"] in ("HIGH", "MODERATE", "LOW")


class TestExampleConfigBuildsWorkingObjects:
    def _build_registry_and_mappings(self, config: dict):
        adjustment = create_adjustment_policy(kind=AdjustmentPolicyKind[config["adjustment_policy"]["kind"].upper()])
        session_cfg = config["session_policy"]
        session = create_timezone_session_policy(
            timezone_key=session_cfg["timezone_key"], is_24_hour_session=session_cfg["is_24_hour_session"],
            timestamp_convention=CandleTimestampConvention[session_cfg["timestamp_convention"].upper()],
            provider_session_note=session_cfg["provider_session_note"],
            session_open_time=datetime.strptime(session_cfg["session_open_time"], "%H:%M:%S").time(),
            session_close_time=datetime.strptime(session_cfg["session_close_time"], "%H:%M:%S").time(),
        )
        availability_cfg = config["availability_policy"]
        availability = create_bar_availability_policy(
            kind=BarAvailabilityPolicyKind[availability_cfg["kind"].upper()], timezone_key=availability_cfg["timezone_key"],
            delay_minutes=availability_cfg["delay_minutes"],
        )

        quality_map = {"HIGH": ProxyQuality.HIGH, "MODERATE": ProxyQuality.MODERATE, "LOW": ProxyQuality.LOW}
        mapping_cfg = config["provider_mappings"]
        mappings = []
        for entry in mapping_cfg["entries"]:
            proxy = create_proxy_policy(is_proxy=True, proxy_for=entry["canonical_driver_id"], proxy_quality=quality_map[entry["proxy_quality"]])
            mappings.append(create_provider_symbol_mapping(
                provider=mapping_cfg["provider"], provider_symbol=entry["provider_symbol"], canonical_driver_id=entry["canonical_driver_id"],
                instrument_form=InstrumentForm[mapping_cfg["instrument_form"].upper()], currency=mapping_cfg["currency"],
                adjustment_policy_kind=AdjustmentPolicyKind[mapping_cfg["adjustment_policy_kind"].upper()], proxy_policy=proxy,
                exchange_or_venue=mapping_cfg["exchange_or_venue"], mapping_version=mapping_cfg["mapping_version"],
            ))
        mapping_set = create_symbol_mapping_set(tuple(mappings))
        mapping_ids_by_driver = {m.canonical_driver_id: (m.mapping_id,) for m in mappings}

        registry_cfg = config["curated_registry"]
        core_specs = default_core_market_driver_specs(
            registry_version=registry_cfg["registry_version"], adjustment_policy=adjustment, session_policy_id=session.session_policy_id,
            availability_policy_id=availability.availability_policy_id, provider_mapping_ids_by_driver=mapping_ids_by_driver,
        )
        optional_specs = default_optional_market_driver_specs(
            registry_version=registry_cfg["registry_version"], adjustment_policy=adjustment, session_policy_id=session.session_policy_id,
            availability_policy_id=availability.availability_policy_id, provider_mapping_ids_by_driver=mapping_ids_by_driver,
        )
        registry = create_curated_market_driver_registry(registry_version=registry_cfg["registry_version"], specs=core_specs + optional_specs)
        return registry, mapping_set

    def test_config_constructs_a_real_registry_and_mapping_set(self) -> None:
        config = _load()
        registry, mapping_set = self._build_registry_and_mappings(config)
        assert len(registry.specs) == 10
        assert len(registry.enabled_driver_ids()) == 9
        assert len(registry.required_driver_ids()) == 5
        assert len(mapping_set.mappings) == 9

    def test_config_constructs_a_real_backfill_spec(self) -> None:
        config = _load()
        registry, mapping_set = self._build_registry_and_mappings(config)
        backfill_cfg = config["backfill"]
        selected_mapping_ids = tuple(sorted(m.mapping_id for m in mapping_set.mappings))
        backfill_spec = create_market_backfill_spec(
            registry=registry, mapping_set=mapping_set, selected_driver_ids=tuple(backfill_cfg["selected_driver_ids"]),
            selected_mapping_ids=selected_mapping_ids, start_time=_parse_time(backfill_cfg["start_time"]),
            end_time=_parse_time(backfill_cfg["end_time"]), requested_granularity=backfill_cfg["requested_granularity"],
            target_dataset_namespace=backfill_cfg["target_dataset_namespace"], cache_policy=MarketCachePolicy[backfill_cfg["cache_policy"]],
            fail_fast=backfill_cfg["fail_fast"], max_driver_count=backfill_cfg["max_driver_count"],
            max_records_per_mapping=backfill_cfg["max_records_per_mapping"], max_total_raw_bytes=backfill_cfg["max_total_raw_bytes"],
        )
        assert len(backfill_spec.selected_driver_ids) == 9
        assert backfill_spec.curated_registry_id == registry.registry_id


class TestExampleConfigContainsNoCredentials:
    def test_no_literal_api_key_value_present(self) -> None:
        config = _load()
        credentials = config["credentials"]
        assert credentials["api_key_source"] == "environment_variable"
        assert credentials["api_key_env_var"] == ALPHA_VANTAGE_API_KEY_ENV_VAR  # references the ONE sanctioned env var, never a value
        assert "value" not in credentials

    def test_raw_text_contains_no_secret_shaped_literal(self) -> None:
        import re

        raw_text = _EXAMPLE_PATH.read_text(encoding="utf-8")
        long_literal_api_key = re.compile(r'"api_key"\s*:\s*"[A-Za-z0-9_\-]{20,}"', re.IGNORECASE)
        assert not long_literal_api_key.search(raw_text)
        for forbidden in ("password", "secret_key", "client_secret", "access_token"):
            assert forbidden not in raw_text.lower()

    def test_storage_root_is_relative_not_an_absolute_credential_bearing_path(self) -> None:
        config = _load()
        assert config["storage"]["storage_root"].startswith("./")


class TestExampleConfigWindowIsBoundedAndTzAware:
    def test_backfill_window_is_bounded_and_tz_aware(self) -> None:
        config = _load()
        start = _parse_time(config["backfill"]["start_time"])
        end = _parse_time(config["backfill"]["end_time"])
        assert start.tzinfo is not None
        assert end.tzinfo is not None
        assert start < end
        assert start.tzinfo.utcoffset(start) == timezone.utc.utcoffset(None)

    def test_no_implicit_current_date(self) -> None:
        """Both `start_time`/`end_time` are explicit ISO literals -- never
        derived from `datetime.now()` anywhere in this config-loading
        path (spec Section 17's own "reject... implicit current date")."""
        raw_text = _EXAMPLE_PATH.read_text(encoding="utf-8")
        assert "now()" not in raw_text
        assert "today" not in raw_text.lower()
