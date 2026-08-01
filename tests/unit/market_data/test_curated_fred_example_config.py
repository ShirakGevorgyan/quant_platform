"""`examples/xauusd_macro_fred_config.example.json` is the safe,
no-real-credentials example the Milestone 10 Phase 4B spec requires.
This proves it stays valid, stays credential-free, and can actually
construct real curated-FRED objects as the schema evolves -- mirroring
`tests/unit/test_historical_config.py::TestSafeExampleConfig`'s own
established pattern for `ingestion_config.example.json`."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from quant_platform.market_data.collectors.curated.acceptance import FRED_API_KEY_ENV_VAR
from quant_platform.market_data.collectors.curated.availability import (
    AvailabilityPolicyKind,
    create_availability_policy,
)
from quant_platform.market_data.collectors.curated.backfill import CachePolicy, create_curated_backfill_spec
from quant_platform.market_data.collectors.curated.registry import (
    create_curated_registry,
    default_core_series_specs,
)
from quant_platform.market_data.collectors.curated.revision_policy import (
    RevisionPolicyKind,
    create_revision_policy,
)

_EXAMPLE_PATH = Path(__file__).resolve().parents[3] / "examples" / "xauusd_macro_fred_config.example.json"


def _load() -> dict:
    return json.loads(_EXAMPLE_PATH.read_text(encoding="utf-8"))


class TestExampleConfigFileExists:
    def test_file_exists_and_is_valid_json(self) -> None:
        assert _EXAMPLE_PATH.is_file(), _EXAMPLE_PATH
        config = _load()
        assert isinstance(config, dict)


class TestExampleConfigMatchesCoreRegistry:
    def test_selected_series_ids_match_the_four_mandatory_core_series(self) -> None:
        config = _load()
        selected = set(config["backfill"]["selected_series_ids"])
        assert selected == {"DFII10", "DGS10", "CPIAUCSL", "DFF"}

    def test_core_series_canonical_names_match_the_real_registry(self) -> None:
        config = _load()
        observation_start = datetime.fromisoformat(config["backfill"]["observation_start"].replace("Z", "+00:00"))
        real_specs = default_core_series_specs(
            registry_version=config["curated_registry"]["registry_version"], revision_policy_id="a" * 64,
            release_availability_policy_id_daily="b" * 64, release_availability_policy_id_monthly="c" * 64,
            default_observation_start=observation_start,
        )
        real_by_id = {s.series_id: s.canonical_series_name for s in real_specs}
        for entry in config["curated_registry"]["core_series"]:
            assert real_by_id[entry["series_id"]] == entry["canonical_series_name"]


class TestExampleConfigBuildsWorkingObjects:
    def test_config_constructs_a_real_registry_and_backfill_spec(self) -> None:
        config = _load()
        observation_start = datetime.fromisoformat(config["backfill"]["observation_start"].replace("Z", "+00:00"))
        specs = default_core_series_specs(
            registry_version=config["curated_registry"]["registry_version"], revision_policy_id="a" * 64,
            release_availability_policy_id_daily="b" * 64, release_availability_policy_id_monthly="c" * 64,
            default_observation_start=observation_start,
        )
        registry = create_curated_registry(registry_version=config["curated_registry"]["registry_version"], specs=specs)

        revision_policy = create_revision_policy(kind=RevisionPolicyKind[config["backfill"]["revision_policy"]["kind"]])
        observation_end = datetime.fromisoformat(config["backfill"]["observation_end"].replace("Z", "+00:00"))
        backfill_spec = create_curated_backfill_spec(
            registry=registry, selected_series_ids=tuple(config["backfill"]["selected_series_ids"]), observation_start=observation_start,
            observation_end=observation_end, revision_policy_id=revision_policy.revision_policy_id,
            target_dataset_namespace=config["backfill"]["target_dataset_namespace"], cache_policy=CachePolicy[config["backfill"]["cache_policy"]],
            fail_fast=config["backfill"]["fail_fast"], page_size=config["backfill"]["page_size"],
            max_series_count=config["backfill"]["max_series_count"], max_observations_per_series=config["backfill"]["max_observations_per_series"],
            max_total_raw_bytes=config["backfill"]["max_total_raw_bytes"],
        )
        assert len(backfill_spec.selected_series_ids) == 4
        assert backfill_spec.curated_registry_id == registry.registry_id

    def test_config_constructs_real_availability_policies(self) -> None:
        config = _load()
        for policy_key in ("daily_series", "monthly_series"):
            entry = config["availability_policies"][policy_key]
            policy = create_availability_policy(
                kind=AvailabilityPolicyKind[entry["kind"]], timezone_key=entry["timezone_key"], availability_hour=entry["availability_hour"],
                availability_minute=entry["availability_minute"],
            )
            assert policy.availability_policy_id
            assert entry["applies_to"]  # non-empty series list


class TestExampleConfigContainsNoCredentials:
    def test_no_literal_api_key_value_present(self) -> None:
        config = _load()
        credentials = config["credentials"]
        assert credentials["api_key_source"] == "environment_variable"
        assert credentials["api_key_env_var"] == FRED_API_KEY_ENV_VAR  # references the ONE sanctioned env var, never a value
        assert "value" not in credentials

    def test_raw_text_contains_no_secret_shaped_literal(self) -> None:
        import re

        raw_text = _EXAMPLE_PATH.read_text(encoding="utf-8")
        # The env-var NAME reference ("FRED_API_KEY", "api_key_env_var", "api_key_source")
        # is an explicitly sanctioned, credential-free pattern (Phase 4B spec: "use env
        # var name reference... No real API key in config"). What must never appear is a
        # long opaque literal ASSIGNED to a key-shaped field -- exactly what
        # `_LONG_LITERAL_API_KEY`-style scanning in test_market_data_safety_scan.py checks.
        long_literal_api_key = re.compile(r'"api_key"\s*:\s*"[A-Za-z0-9_\-]{20,}"', re.IGNORECASE)
        assert not long_literal_api_key.search(raw_text)
        for forbidden in ("password", "secret_key", "client_secret", "access_token"):
            assert forbidden not in raw_text.lower()

    def test_storage_root_is_relative_not_an_absolute_credential_bearing_path(self) -> None:
        config = _load()
        assert config["storage"]["storage_root"].startswith("./")


class TestExampleConfigRevisionPolicyIsValid:
    def test_revision_policy_kind_is_a_real_enum_member(self) -> None:
        config = _load()
        kind_text = config["backfill"]["revision_policy"]["kind"]
        assert RevisionPolicyKind[kind_text] is RevisionPolicyKind.LATEST_AVAILABLE

    def test_observation_window_is_bounded_and_tz_aware(self) -> None:
        config = _load()
        start = datetime.fromisoformat(config["backfill"]["observation_start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(config["backfill"]["observation_end"].replace("Z", "+00:00"))
        assert start.tzinfo is not None
        assert end.tzinfo is not None
        assert start < end
        assert start.tzinfo.utcoffset(start) == timezone.utc.utcoffset(None)
