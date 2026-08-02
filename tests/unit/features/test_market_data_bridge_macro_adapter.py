"""`features.market_data_bridge.macro_adapter`: revision-policy selection,
missing-observation handling, pin verification, and fail-closed rejection
of non-point-in-time-safe revision policies (spec Section 6)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from _market_data_bridge_test_helpers import make_macro_fixture

from quant_platform.core.exceptions import AlignmentPolicyError, SourceVerificationError
from quant_platform.features.market_data_bridge.macro_adapter import (
    resolve_macro_dataframe,
    select_observations_for_policy,
    verify_macro_binding,
)
from quant_platform.market_data.collectors.curated.revision_policy import RevisionPolicyKind


class TestVerifyMacroBinding:
    def test_resolves_expected_observations(self, tmp_path) -> None:
        fixture = make_macro_fixture(tmp_path, days=10)
        observations = verify_macro_binding(fixture.observation_store, fixture.manifest_store, fixture.binding)
        assert len(observations) == 10

    def test_stale_pin_fails_closed(self, tmp_path) -> None:
        fixture = make_macro_fixture(tmp_path, days=5)
        # Append a new observation directly, bypassing the pinned binding.
        from datetime import datetime, timedelta, timezone
        from decimal import Decimal

        from quant_platform.market_data.collectors.curated.macro_observation import (
            create_curated_macro_observation,
        )

        new_obs = create_curated_macro_observation(
            series_id="DFII10", canonical_series_name="10Y TIPS", target_macro_instrument_id="dfii10", observation_date="2024-02-01",
            value=Decimal("1.9"), is_missing=False, normalized_unit="percent", native_unit="percent", native_frequency="daily",
            realtime_start="2024-02-01", realtime_end=None, availability_time=datetime(2024, 2, 1, tzinfo=timezone.utc) + timedelta(hours=6),
            availability_policy_id="ap1", request_manifest_id="r" * 10, response_manifest_id="p" * 10, source_manifest_id="s" * 10, source_row_index=999,
        )
        fixture.observation_store.append("fred", new_obs)
        with pytest.raises(SourceVerificationError):
            verify_macro_binding(fixture.observation_store, fixture.manifest_store, fixture.binding)

    def test_no_component_manifest_fails_closed(self, tmp_path) -> None:
        fixture = make_macro_fixture(tmp_path, days=1)
        other_series_binding = replace(fixture.binding, series_id="CPIAUCSL", binding_id="")
        with pytest.raises(SourceVerificationError):
            verify_macro_binding(fixture.observation_store, fixture.manifest_store, other_series_binding)

    def test_verification_is_insensitive_to_binding_metadata_typo(self, tmp_path) -> None:
        """`canonical_series_name` is descriptive metadata, not part of
        the semantic digest recomputation -- a typo there must not
        produce a spurious verification failure (regression test for the
        false-negative bug fixed during development)."""
        fixture = make_macro_fixture(tmp_path, days=3)
        typo_binding = replace(fixture.binding, canonical_series_name="WRONG NAME", binding_id="")
        observations = verify_macro_binding(fixture.observation_store, fixture.manifest_store, typo_binding)
        assert len(observations) == 3


class TestSelectObservationsForPolicy:
    def test_vintage_series_keeps_every_revision(self, tmp_path) -> None:
        fixture = make_macro_fixture(tmp_path, days=10, with_revision_on_day=3)
        selected = select_observations_for_policy(fixture.observations, kind=RevisionPolicyKind.VINTAGE_SERIES)
        assert len(selected) == 11  # 10 base + 1 revision

    def test_first_release_only_drops_later_revisions(self, tmp_path) -> None:
        fixture = make_macro_fixture(tmp_path, days=10, with_revision_on_day=3)
        selected = select_observations_for_policy(fixture.observations, kind=RevisionPolicyKind.FIRST_RELEASE_ONLY)
        assert len(selected) == 10
        assert all(o.value == fixture.observations[0].value or o.observation_date != fixture.observations[3].observation_date for o in selected)

    @pytest.mark.parametrize("kind", [RevisionPolicyKind.LATEST_AVAILABLE, RevisionPolicyKind.AS_OF_REALTIME_DATE])
    def test_non_pit_safe_policies_are_rejected(self, tmp_path, kind: RevisionPolicyKind) -> None:
        fixture = make_macro_fixture(tmp_path, days=3)
        with pytest.raises(AlignmentPolicyError):
            select_observations_for_policy(fixture.observations, kind=kind)


class TestResolveMacroDataframe:
    def test_shape_and_columns(self, tmp_path) -> None:
        fixture = make_macro_fixture(tmp_path, days=5)
        df = resolve_macro_dataframe(fixture.observation_store, fixture.manifest_store, fixture.binding)
        assert list(df.columns) == ["value", "release_time"]
        assert len(df) == 5
        assert df["release_time"].is_monotonic_increasing

    def test_missing_observation_becomes_nan_not_dropped(self, tmp_path) -> None:
        fixture = make_macro_fixture(tmp_path, days=5, with_missing_day=2)
        df = resolve_macro_dataframe(fixture.observation_store, fixture.manifest_store, fixture.binding)
        assert len(df) == 5
        assert df["value"].isna().sum() == 1

    def test_revision_produces_two_visible_vintages(self, tmp_path) -> None:
        fixture = make_macro_fixture(tmp_path, days=5, with_revision_on_day=1)
        df = resolve_macro_dataframe(fixture.observation_store, fixture.manifest_store, fixture.binding)
        assert len(df) == 6
