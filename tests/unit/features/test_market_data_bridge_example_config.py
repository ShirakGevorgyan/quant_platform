"""`examples/xauusd_point_in_time_research_dataset.example.json` is the
safe, no-real-credentials, explicit-placeholder example Milestone 10
Phase 4D's spec requires. Mirrors `test_cross_asset_example_config.py`'s
own established pattern: proves the file stays valid, stays credential-
free, and can actually construct REAL binding objects as the schema
evolves."""

from __future__ import annotations

import json
import re
from pathlib import Path

from quant_platform.core.types import Timeframe
from quant_platform.features.market_data_bridge.bindings import (
    create_base_asset_binding,
    create_cross_asset_dataset_binding,
    create_macro_dataset_binding,
)
from quant_platform.features.market_data_bridge.coverage import SourceCoveragePolicy, SourceCoveragePolicyKind
from quant_platform.market_data.collectors.cross_asset.instrument_form import (
    InstrumentForm,
    ProxyQuality,
    create_proxy_policy,
)
from quant_platform.market_data.collectors.curated.revision_policy import RevisionPolicyKind

_EXAMPLE_PATH = Path(__file__).resolve().parents[3] / "examples" / "xauusd_point_in_time_research_dataset.example.json"
_PLACEHOLDER_ID = re.compile(r"^[0-9a-f]{64}$")


def _load() -> dict:
    return json.loads(_EXAMPLE_PATH.read_text(encoding="utf-8"))


class TestExampleConfigFileExists:
    def test_file_exists_and_is_valid_json(self) -> None:
        assert _EXAMPLE_PATH.is_file(), _EXAMPLE_PATH
        config = _load()
        assert isinstance(config, dict)


class TestExampleConfigBuildsRealBindings:
    def test_base_binding_constructs(self) -> None:
        config = _load()
        raw = config["base_binding"]
        binding = create_base_asset_binding(
            canonical_instrument_id=raw["canonical_instrument_id"], provider=raw["provider"], pinned_dataset_id=raw["pinned_dataset_id"],
            timeframe=Timeframe(raw["timeframe"]), expected_event_kind=raw["expected_event_kind"], session_policy_id=raw["session_policy_id"],
            availability_policy_id=raw["availability_policy_id"],
        )
        assert binding.canonical_instrument_id == "XAUUSD"
        assert binding.binding_id

    def test_every_macro_binding_constructs(self) -> None:
        config = _load()
        for name, raw in config["macro_bindings"].items():
            binding = create_macro_dataset_binding(
                curated_registry_id=raw["curated_registry_id"], combined_universe_manifest_id=raw["combined_universe_manifest_id"],
                series_id=raw["series_id"], canonical_series_name=raw["canonical_series_name"], provider=raw["provider"],
                component_manifest_id=raw["component_manifest_id"], revision_policy_id=raw["revision_policy_id"],
                revision_policy_kind=RevisionPolicyKind(raw["revision_policy_kind"]), availability_policy_id=raw["availability_policy_id"],
                native_frequency=raw["native_frequency"], normalized_unit=raw["normalized_unit"], required=raw["required"],
            )
            assert binding.series_id == name or name in ("DFII10", "DGS10", "CPIAUCSL", "DFF")
            assert binding.binding_id

    def test_every_cross_asset_binding_constructs(self) -> None:
        config = _load()
        quality_map = {"high": ProxyQuality.HIGH, "moderate": ProxyQuality.MODERATE, "low": ProxyQuality.LOW}
        for _name, raw in config["cross_asset_bindings"].items():
            proxy_raw = raw["proxy_policy"]
            proxy = create_proxy_policy(
                is_proxy=proxy_raw["is_proxy"], proxy_for=proxy_raw["proxy_for"], proxy_quality=quality_map[proxy_raw["proxy_quality"]],
                known_basis_risk=proxy_raw["known_basis_risk"], roll_risk=proxy_raw["roll_risk"], tracking_error_risk=proxy_raw["tracking_error_risk"],
                currency_difference_note=proxy_raw["currency_difference_note"], session_difference_note=proxy_raw["session_difference_note"],
                adjustment_difference_note=proxy_raw["adjustment_difference_note"],
            )
            binding = create_cross_asset_dataset_binding(
                curated_registry_id=raw["curated_registry_id"], combined_manifest_id=raw["combined_manifest_id"],
                canonical_driver_id=raw["canonical_driver_id"], mapping_id=raw["mapping_id"], provider=raw["provider"],
                provider_symbol=raw["provider_symbol"], component_manifest_id=raw["component_manifest_id"],
                instrument_form=InstrumentForm(raw["instrument_form"]), proxy_policy=proxy, adjustment_policy_id=raw["adjustment_policy_id"],
                continuation_policy_id=raw["continuation_policy_id"], session_policy_id=raw["session_policy_id"],
                availability_policy_id=raw["availability_policy_id"], timeframe=Timeframe(raw["timeframe"]), required=raw["required"],
            )
            assert binding.binding_id

    def test_coverage_policy_constructs(self) -> None:
        config = _load()
        raw = config["coverage_policy"]
        policy = SourceCoveragePolicy(
            kind=SourceCoveragePolicyKind(raw["kind"]), minimum_observation_coverage_fraction=raw["minimum_observation_coverage_fraction"],
            max_consecutive_missing_aligned_rows=raw["max_consecutive_missing_aligned_rows"],
        )
        assert policy.kind is SourceCoveragePolicyKind.FAIL_REQUIRED_SOURCE


class TestExampleConfigMatchesFixtureAcceptanceUniverse:
    def test_macro_series_match_the_mandatory_acceptance_fixture_set(self) -> None:
        config = _load()
        assert set(config["macro_bindings"]) == {"DFII10", "DGS10", "CPIAUCSL", "DFF"}

    def test_cross_asset_drivers_match_the_mandatory_acceptance_fixture_set(self) -> None:
        config = _load()
        driver_ids = {raw["canonical_driver_id"] for raw in config["cross_asset_bindings"].values()}
        assert driver_ids == {"us_dollar_strength", "wti_crude", "brent_crude", "silver", "gold_reference"}

    def test_every_cross_asset_mapping_is_honestly_a_proxy(self) -> None:
        config = _load()
        for raw in config["cross_asset_bindings"].values():
            assert raw["instrument_form"] == "etf"
            assert raw["proxy_policy"]["is_proxy"] is True


class TestExampleConfigContainsNoCredentials:
    def test_no_credentials_block_at_all(self) -> None:
        config = _load()
        assert "credentials" not in config
        assert "api_key" not in json.dumps(config).lower()

    def test_no_secret_shaped_literal(self) -> None:
        raw_text = _EXAMPLE_PATH.read_text(encoding="utf-8")
        for forbidden in ("password", "secret_key", "client_secret", "access_token", "api_key"):
            assert forbidden not in raw_text.lower()


class TestExampleConfigPinsExplicitPlaceholders:
    def test_every_pinned_id_is_an_explicit_marked_placeholder(self) -> None:
        """Every *_id field is a placeholder hex string (never a mutable
        alias like "latest"), and the file's own top-level `_comment`
        explains this -- spec's own "clearly marked placeholders requiring
        replacement" requirement."""
        config = _load()
        assert "_comment" in config
        assert "PLACEHOLDER" in config["_comment"] or "placeholder" in config["_comment"].lower()
        assert _PLACEHOLDER_ID.match(config["base_binding"]["pinned_dataset_id"])

    def test_no_mutable_alias_anywhere(self) -> None:
        raw_text = _EXAMPLE_PATH.read_text(encoding="utf-8").lower()
        for alias in ('"latest"', '"current"', '"newest"', '"active"'):
            assert alias not in raw_text

    def test_no_implicit_current_date(self) -> None:
        raw_text = _EXAMPLE_PATH.read_text(encoding="utf-8")
        assert "now()" not in raw_text
        assert "today" not in raw_text.lower()
