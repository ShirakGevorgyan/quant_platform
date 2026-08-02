"""Adversarial tests (Milestone 10, Phase 4D, spec Section 23) -- the
22-item list, one test class per item, numbered to match the spec.
Several items are proven by machinery already exercised in the adapter/
reconciliation/verification/bindings test files; this file exists so
each of the 22 items has an explicit, independently-identifiable test
(never merely "covered incidentally"), and to add the items not yet
directly exercised elsewhere."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from datetime import timedelta
from decimal import Decimal

import pandas as pd
import pytest
from _market_data_bridge_test_helpers import (
    BASE_TIME,
    make_base_binding,
    make_cross_asset_fixture,
    make_macro_fixture,
    open_repository,
)

from quant_platform.core.exceptions import AlignmentPolicyError, SourceBindingError, SourceVerificationError
from quant_platform.core.types import Timeframe
from quant_platform.features.alignment import align_higher_timeframe, as_of_join_external
from quant_platform.features.market_data_bridge.bindings import (
    create_base_asset_binding,
)
from quant_platform.features.market_data_bridge.cross_asset_adapter import (
    resolve_cross_asset_dataframe,
    verify_cross_asset_binding,
)
from quant_platform.features.market_data_bridge.lineage import build_market_data_lineage, lineage_content_id
from quant_platform.features.market_data_bridge.macro_adapter import (
    resolve_macro_dataframe,
    select_observations_for_policy,
)
from quant_platform.features.market_data_bridge.rebuild_planner import (
    RebuildPlanKind,
    SourceChangeEvidence,
    plan_rebuild,
)
from quant_platform.features.market_data_bridge.verification import (
    verify_truncation_invariance_cross_asset,
    verify_truncation_invariance_macro,
)
from quant_platform.market_data.collectors.cross_asset.instrument_form import (
    InstrumentForm,
    create_proxy_policy,
)
from quant_platform.market_data.collectors.cross_asset.market_record import create_market_driver_bar
from quant_platform.market_data.collectors.curated.revision_policy import RevisionPolicyKind


class Test01CpiValueJoinedBeforeRelease:
    def test_value_released_after_the_row_is_never_visible(self, tmp_path) -> None:
        fixture = make_macro_fixture(tmp_path, series_id="CPIAUCSL", days=5)
        df = resolve_macro_dataframe(fixture.observation_store, fixture.manifest_store, fixture.binding)
        before_first_release = pd.Series([pd.Timestamp(BASE_TIME) - pd.Timedelta(hours=1)])
        joined = as_of_join_external(before_first_release, df, value_column="value", output_name="level")
        assert pd.isna(joined["level"].iloc[0])
        assert joined["level_is_stale"].iloc[0]


class Test02RevisedCpiNotInjectedIntoEarlierRows:
    def test_earlier_rows_unaffected_by_a_later_revision(self, tmp_path) -> None:
        fixture = make_macro_fixture(tmp_path, series_id="CPIAUCSL", days=5, with_revision_on_day=2)
        df = resolve_macro_dataframe(fixture.observation_store, fixture.manifest_store, fixture.binding)
        result = verify_truncation_invariance_macro(
            pd.Series(pd.date_range("2024-01-01", "2024-01-06", freq="6h", tz="UTC")), df, source_name="CPIAUCSL",
            truncate_after=pd.Timestamp("2024-01-06T06:00Z"),
        )
        assert result.is_invariant


class Test03FutureVintageNotSelectedInHistoricalAsOfMode:
    @pytest.mark.parametrize("kind", [RevisionPolicyKind.LATEST_AVAILABLE, RevisionPolicyKind.AS_OF_REALTIME_DATE])
    def test_non_pit_safe_kinds_are_refused(self, tmp_path, kind) -> None:
        fixture = make_macro_fixture(tmp_path, days=3)
        with pytest.raises(AlignmentPolicyError):
            select_observations_for_policy(fixture.observations, kind=kind)


class Test04CrossAssetCandleNotJoinedBeforeClose:
    def test_bar_invisible_before_its_own_close(self, tmp_path) -> None:
        fixture = make_cross_asset_fixture(tmp_path, days=3)
        df = resolve_cross_asset_dataframe(fixture.bar_store, fixture.manifest_store, fixture.binding)
        aligned = align_higher_timeframe(pd.Series([pd.Timestamp(BASE_TIME)]), df, Timeframe.D1)
        assert aligned["htf_D1_bar_index"].iloc[0] == -1


class Test05FutureCorrectedCrossAssetCandleNotInjectedIntoOldVersion:
    def test_truncation_invariance_holds(self, tmp_path) -> None:
        fixture = make_cross_asset_fixture(tmp_path, days=5)
        df = resolve_cross_asset_dataframe(fixture.bar_store, fixture.manifest_store, fixture.binding)
        result = verify_truncation_invariance_cross_asset(
            pd.Series(pd.date_range("2024-01-01", "2024-01-08", freq="6h", tz="UTC")), df, source_name="dxy", timeframe=Timeframe.D1,
            truncate_after=pd.Timestamp("2024-01-06T00:00Z"),
        )
        assert result.is_invariant


class Test06IngestionArrivalTimeNeverUsedAsAvailability:
    def test_base_asset_uses_close_time_never_arrival_time(self, tmp_path) -> None:
        """A `Candle`'s `arrival_time` (repository append time) is NEVER
        read by the base-asset adapter for availability purposes -- only
        `event_time` (open) + `timeframe.duration` (close), matching
        `FeatureEngine.compute`'s own derivation. Constructs a candle
        whose `arrival_time` is set far in the FUTURE relative to its
        `event_time` (a legitimate, late-arriving historical backfill)
        and confirms the resolved `open_time` is still the candle's real
        `event_time`, not anything arrival-time-derived."""
        from quant_platform.market_data.candles import create_candle
        from quant_platform.market_data.ingestion import ingest_raw_events
        from quant_platform.market_data.manifests import (
            DatasetKey,
            DatasetKind,
            PartitionGranularity,
            PartitioningSpec,
        )

        repo = open_repository(tmp_path)
        key = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="XAUUSD", provider="mt5")
        late_arrival = create_candle(
            instrument_id="XAUUSD", provider="mt5", symbol="XAUUSD", event_time=BASE_TIME, timeframe=Timeframe.H1, sequence=0,
            open=Decimal("2000"), high=Decimal("2005"), low=Decimal("1995"), close=Decimal("2000"), volume=Decimal("1"),
            arrival_time=BASE_TIME + timedelta(days=30),
        )
        result = ingest_raw_events(
            repository=repo, dataset_key=key, batch_id="b1", ingestion_time=BASE_TIME, events=(late_arrival,),
            partitioning=PartitioningSpec(granularity=PartitionGranularity.DAILY),
        )
        binding = create_base_asset_binding(canonical_instrument_id="XAUUSD", provider="mt5", pinned_dataset_id=result.resulting_dataset_id, timeframe=Timeframe.H1)
        from quant_platform.features.market_data_bridge.base_asset_adapter import resolve_base_asset_dataframe

        df = resolve_base_asset_dataframe(repo, binding, start=BASE_TIME, end=BASE_TIME + timedelta(hours=1))
        assert df["open_time"].iloc[0] == pd.Timestamp(BASE_TIME)


class Test07IncompatibleSessionCutoffsNotTreatedAsIdentical:
    def test_different_availability_delay_produces_different_synthetic_open_time(self, tmp_path) -> None:
        fixture_a = make_cross_asset_fixture(tmp_path / "a", days=3, extra_delay=timedelta(hours=0))
        fixture_b = make_cross_asset_fixture(tmp_path / "b", days=3, extra_delay=timedelta(hours=12))
        df_a = resolve_cross_asset_dataframe(fixture_a.bar_store, fixture_a.manifest_store, fixture_a.binding)
        df_b = resolve_cross_asset_dataframe(fixture_b.bar_store, fixture_b.manifest_store, fixture_b.binding)
        assert not df_a["open_time"].equals(df_b["open_time"])


class Test08EtfProxyNeverRepresentedAsUnderlying:
    def test_etf_form_requires_is_proxy_true(self) -> None:
        not_a_proxy = create_proxy_policy(is_proxy=False)
        with pytest.raises(SourceBindingError):
            from quant_platform.features.market_data_bridge.bindings import create_cross_asset_dataset_binding

            create_cross_asset_dataset_binding(
                curated_registry_id="r" * 64, combined_manifest_id="c" * 64, canonical_driver_id="us_dollar_strength", mapping_id="m" * 64,
                provider="alpha_vantage", provider_symbol="UUP", component_manifest_id="e" * 64, instrument_form=InstrumentForm.ETF,
                proxy_policy=not_a_proxy, adjustment_policy_id="raw", continuation_policy_id=None, session_policy_id="nyse",
                availability_policy_id="close_plus_1d", timeframe=Timeframe.D1,
            )


class Test09FuturesContinuousMetadataNotFabricated:
    def test_provider_continuous_futures_bar_requires_roll_provenance(self) -> None:
        """market_data's own structural guard (reused, not reimplemented):
        a `PROVIDER_CONTINUOUS_FUTURES` bar cannot even be CONSTRUCTED
        without roll provenance -- the bridge inherits this guarantee for
        free by never bypassing `create_market_driver_bar`."""
        from quant_platform.core.exceptions import MarketRecordError

        with pytest.raises(MarketRecordError):
            create_market_driver_bar(
                canonical_driver_id="wti_crude", provider="p", provider_symbol="CL1!", instrument_form=InstrumentForm.PROVIDER_CONTINUOUS_FUTURES,
                open_time=BASE_TIME, timeframe=Timeframe.D1, open=Decimal("75"), high=Decimal("76"), low=Decimal("74"), close=Decimal("75.5"),
                volume=None, volume_unit="native", availability_time=BASE_TIME + timedelta(days=1), availability_policy_id="a" * 64,
                session_policy_id="s" * 64, adjustment_policy_id="j" * 64, request_manifest_id="r" * 64, response_manifest_id="p" * 64,
                source_manifest_id="c" * 64, source_row_index=0, roll_provenance=None,
            )


class Test10MissingRequiredSourceNotSilentlyIgnored:
    def test_fail_required_source_raises_on_empty_required_macro_frame(self) -> None:
        from quant_platform.core.exceptions import SourceCoverageError
        from quant_platform.features.market_data_bridge.coverage import (
            SourceCoveragePolicy,
            SourceCoveragePolicyKind,
            evaluate_source_coverage,
        )

        base_df = pd.DataFrame({"open_time": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")})
        empty_macro = pd.DataFrame({"value": [], "release_time": pd.to_datetime([], utc=True)})
        with pytest.raises(SourceCoverageError):
            evaluate_source_coverage(
                base_df=base_df, base_timeframe=Timeframe.H1, macro_frames={"dfii10": empty_macro}, macro_bindings={}, cross_asset_frames={},
                cross_asset_bindings={}, requested_start=pd.Timestamp("2024-01-01", tz="UTC"), requested_end=pd.Timestamp("2024-01-01T05:00Z"),
                policy=SourceCoveragePolicy(kind=SourceCoveragePolicyKind.FAIL_REQUIRED_SOURCE),
            )


class Test11StaleMacroValueNotCarriedIndefinitelyWithoutSignal:
    def test_stale_indicator_flags_a_value_far_beyond_threshold(self) -> None:
        from quant_platform.features.market_data_bridge.staleness import evaluate_macro_staleness

        base_avail = pd.Series([pd.Timestamp("2030-01-01T00:00Z")])
        macro_df = pd.DataFrame({"value": [1.0], "release_time": pd.to_datetime(["2024-01-01T00:00Z"], utc=True)})
        finding = evaluate_macro_staleness(base_avail, macro_df, source_name="dfii10", threshold=pd.Timedelta(days=30))
        assert finding.stale_row_count == 1


class Test12StaleDailyProxyThroughLongClosureIsFlagged:
    def test_stale_indicator_flags_a_bar_far_beyond_threshold(self) -> None:
        from quant_platform.features.market_data_bridge.staleness import evaluate_cross_asset_staleness

        base_avail = pd.Series([pd.Timestamp("2024-06-01T00:00Z")])
        cross_df = pd.DataFrame({"open_time": pd.to_datetime(["2024-01-01T00:00Z"], utc=True), "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]})
        finding = evaluate_cross_asset_staleness(base_avail, cross_df, source_name="dxy", timeframe=Timeframe.D1, threshold=pd.Timedelta(days=5))
        assert finding.stale_row_count == 1


class Test13MutableAliasNeverAcceptedInsteadOfPinnedId:
    @pytest.mark.parametrize("alias", ["latest", "current", "newest", "active", "default"])
    def test_base_binding_rejects_alias(self, alias: str) -> None:
        with pytest.raises(SourceBindingError):
            create_base_asset_binding(canonical_instrument_id="XAUUSD", provider="mt5", pinned_dataset_id=alias, timeframe=Timeframe.H1)


class Test14SourceComponentSwappedAfterManifestCreationIsDetected:
    def test_verify_cross_asset_binding_fails_closed_after_swap(self, tmp_path) -> None:
        fixture = make_cross_asset_fixture(tmp_path, days=5)
        extra = create_market_driver_bar(
            canonical_driver_id="us_dollar_strength", provider="alpha_vantage", provider_symbol="UUP", instrument_form=InstrumentForm.ETF,
            open_time=BASE_TIME + timedelta(days=100), timeframe=Timeframe.D1, open=Decimal("29"), high=Decimal("29.5"), low=Decimal("28.5"),
            close=Decimal("29.2"), volume=Decimal("1"), volume_unit="shares", availability_time=BASE_TIME + timedelta(days=102),
            availability_policy_id="close_plus_conservative_delay", session_policy_id="nyse", adjustment_policy_id="raw_unadjusted",
            request_manifest_id="r" * 10, response_manifest_id="p" * 10, source_manifest_id="s" * 10, source_row_index=999,
        )
        fixture.bar_store.append_many_and_read_all("alpha_vantage", "us_dollar_strength", InstrumentForm.ETF, [extra])
        with pytest.raises(SourceVerificationError):
            verify_cross_asset_binding(fixture.bar_store, fixture.manifest_store, fixture.binding)


class Test15ManifestRehashedAfterLineageTamperingIsDetected:
    def test_tampered_lineage_no_longer_matches_recomputed_content_id(self, tmp_path) -> None:
        repo = open_repository(tmp_path / "md")
        base_binding = make_base_binding(repo, hours=5)
        lineage = build_market_data_lineage(base_binding=base_binding, macro_bindings={}, cross_asset_bindings={})
        original_id = lineage_content_id(lineage)
        tampered = dict(lineage)
        tampered["base_asset_binding"] = dict(tampered["base_asset_binding"])
        tampered["base_asset_binding"]["pinned_dataset_id"] = "9" * 64
        tampered_id = lineage_content_id(tampered)
        assert tampered_id != original_id


class Test16AlignedValuesTamperedDigestRecomputedIsDetected:
    def test_recomputed_semantic_digest_differs_after_a_value_changes(self, tmp_path) -> None:
        from quant_platform.market_data.identity import compute_content_id

        fixture = make_macro_fixture(tmp_path, days=3)
        real_digest = compute_content_id("curated_component_semantic_digest", {"observation_ids": sorted(o.observation_id for o in fixture.observations)})
        tampered_ids = sorted(o.observation_id for o in fixture.observations)
        tampered_ids[0] = "0" * 64  # simulate a tampered observation id
        tampered_digest = compute_content_id("curated_component_semantic_digest", {"observation_ids": tampered_ids})
        assert tampered_digest != real_digest
        assert real_digest == fixture.manifest.semantic_digest


class Test17DeclarationOrderDoesNotChangeOutputColumns:
    def test_macro_binding_dict_order_does_not_change_lineage_content(self, tmp_path) -> None:
        repo = open_repository(tmp_path / "md")
        base_binding = make_base_binding(repo, hours=5)
        macro_a = make_macro_fixture(tmp_path / "ma", series_id="DFII10", days=2)
        macro_b = make_macro_fixture(tmp_path / "mb", series_id="DGS10", days=2)
        lineage_1 = build_market_data_lineage(base_binding=base_binding, macro_bindings={"DFII10": macro_a.binding, "DGS10": macro_b.binding}, cross_asset_bindings={})
        lineage_2 = build_market_data_lineage(base_binding=base_binding, macro_bindings={"DGS10": macro_b.binding, "DFII10": macro_a.binding}, cross_asset_bindings={})
        assert lineage_content_id(lineage_1) == lineage_content_id(lineage_2)


class Test18PythonHashSeedDoesNotChangeLineageIdentity:
    def test_lineage_content_id_is_stable_across_hash_seeds(self, tmp_path) -> None:
        test_dir = pathlib.Path(__file__).resolve().parent
        script_lines = [
            f"import sys; sys.path.insert(0, {str(test_dir)!r})",
            "import tempfile",
            "from _market_data_bridge_test_helpers import make_base_binding, open_repository",
            "from quant_platform.features.market_data_bridge.lineage import build_market_data_lineage, lineage_content_id",
            "repo = open_repository(tempfile.mkdtemp())",
            "b = make_base_binding(repo, hours=5)",
            "lineage = build_market_data_lineage(base_binding=b, macro_bindings={}, cross_asset_bindings={})",
            "print(lineage_content_id(lineage))",
        ]
        script = "\n".join(script_lines)
        env_a = dict(os.environ, PYTHONHASHSEED="1")
        env_b = dict(os.environ, PYTHONHASHSEED="2")
        out_a = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env_a, check=True).stdout.strip()
        out_b = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env_b, check=True).stdout.strip()
        assert out_a == out_b


class Test19FilesystemRootDoesNotChangeDatasetIdentity:
    def test_same_content_different_roots_produce_the_same_lineage_content_id(self, tmp_path) -> None:
        repo_a = open_repository(tmp_path / "root_a" / "md")
        repo_b = open_repository(tmp_path / "a_totally_different_named_root" / "md")
        binding_a = make_base_binding(repo_a, hours=5)
        binding_b = make_base_binding(repo_b, hours=5)
        assert binding_a.pinned_dataset_id == binding_b.pinned_dataset_id
        lineage_a = build_market_data_lineage(base_binding=binding_a, macro_bindings={}, cross_asset_bindings={})
        lineage_b = build_market_data_lineage(base_binding=binding_b, macro_bindings={}, cross_asset_bindings={})
        assert lineage_content_id(lineage_a) == lineage_content_id(lineage_b)


class Test20LabelsNeverInfluenceSourceTrimmingOrAlignment:
    def test_coverage_evaluation_signature_has_no_label_parameter(self) -> None:
        import inspect

        from quant_platform.features.market_data_bridge.coverage import evaluate_source_coverage

        params = set(inspect.signature(evaluate_source_coverage).parameters)
        assert "labels" not in params and "label_definition" not in params

    def test_macro_adapter_module_never_imports_labels(self) -> None:
        from pathlib import Path

        bridge_root = Path(__file__).resolve().parents[3] / "src" / "quant_platform" / "features" / "market_data_bridge"
        for path in bridge_root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "features.labels" not in text and "from quant_platform.features import labels" not in text


class Test21OldManifestNotSilentlyInterpretedAsNewSchema:
    def test_manifest_without_market_data_lineage_key_loads_as_none(self) -> None:
        from quant_platform.features.manifests import ResearchDatasetManifest

        legacy_raw = {
            "dataset_id": "d" * 16, "version": "000001-abc", "source_historical_dataset_id": "h" * 16,
            "source_historical_manifest_version": "000001-xyz", "symbol": "XAUUSD", "base_timeframe": "H1",
            "utc_start": "2024-01-01T00:00:00+00:00", "utc_end": "2024-01-02T00:00:00+00:00", "feature_names": [],
            "feature_versions": {}, "feature_registry_fingerprint": "f" * 16, "label_definition": {}, "split_definition": {},
            "preprocessing_definition": {}, "fitted_preprocessing_fingerprint": None, "code_revision": "abc", "input_content_hashes": {},
            "output_content_hashes": {}, "row_counts": {}, "missing_data_summary": {}, "leakage_validation_result": {},
            "created_at": "2024-01-01T00:00:00+00:00",
            # NOTE: no "market_data_lineage" key at all -- simulates a manifest written before Phase 4D existed.
        }
        manifest = ResearchDatasetManifest.from_json_dict(legacy_raw)
        assert manifest.market_data_lineage is None


class Test22IncrementalPlannerDoesNotMissAHistoricalRevisionImpact:
    def test_a_correction_with_evidence_is_never_classified_as_no_op(self) -> None:
        ev = SourceChangeEvidence(
            source_kind="macro", source_name="dfii10", old_first_covered_time=pd.Timestamp("2024-01-01", tz="UTC"),
            old_last_covered_time=pd.Timestamp("2024-01-10", tz="UTC"), old_observation_count=10,
            new_first_covered_time=pd.Timestamp("2024-01-01", tz="UTC"), new_last_covered_time=pd.Timestamp("2024-01-10", tz="UTC"), new_observation_count=10,
        )
        # Same coverage range/count as before (a same-day revision, not an
        # extension) -- `is_append_only` must be False (new_last == old_last,
        # not >), forcing PARTIAL_RECOMPUTATION, never NO_OP or an unsafe append.
        assert not ev.is_append_only
        plan = plan_rebuild(
            existing_lineage={"x": 1}, existing_dataset_id="d1", recipe_unchanged=True, new_base_pinned_dataset_id="a", old_base_pinned_dataset_id="a",
            new_macro_component_ids={"dfii10": "new"}, old_macro_component_ids={"dfii10": "old"}, new_cross_asset_component_ids={},
            old_cross_asset_component_ids={}, evidence_by_source_name={"dfii10": ev},
        )
        assert plan.kind is RebuildPlanKind.PARTIAL_RECOMPUTATION_REQUIRED
        assert plan.kind is not RebuildPlanKind.NO_OP
