"""Milestone 6, Section 16: manifest persistence, append-only event log,
interruption/resume, and semantic-tampering detection. Proves the
central "never trust the manifest's claim alone" property: a manifest
that claims a later stage than its own artifacts actually support is
NOT believed by `verify_completed_robustness_stages`."""

from __future__ import annotations

from dataclasses import replace as dc_replace
from pathlib import Path

import pytest

from quant_platform.core.exceptions import RobustnessStateError
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.ml.models import ArtifactCategory, ArtifactReference
from quant_platform.ml.persistence import canonical_json_bytes, format_utc_timestamp, utc_now
from quant_platform.robustness.manifests import (
    ARTIFACT_KINDS,
    RobustnessEventStore,
    RobustnessEventType,
    RobustnessManifest,
    RobustnessManifestStore,
)
from quant_platform.robustness.models import RobustnessStage, is_legal_robustness_transition
from quant_platform.robustness.resume import (
    can_resume,
    require_robustness_resumable,
    resolve_resume_start_stage,
    verify_completed_robustness_stages,
)
from quant_platform.robustness.source import SourceVerificationReport

_ROBUSTNESS_ID = "a" * 64
_SOURCE_BACKTEST_ID = "b" * 64


@pytest.fixture
def stores(tmp_path: Path) -> tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore]:
    return RobustnessManifestStore(tmp_path), RobustnessEventStore(tmp_path), MLArtifactStore(tmp_path)


def _created_manifest() -> RobustnessManifest:
    now = format_utc_timestamp(utc_now())
    return RobustnessManifest(
        schema_version=1, robustness_id=_ROBUSTNESS_ID, source_backtest_id=_SOURCE_BACKTEST_ID, stage=RobustnessStage.CREATED, created_at=now, updated_at=now,
    )


class TestManifestCreateAndTransition:
    def test_create_then_load_round_trips(self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore]) -> None:
        manifest_store, _events, _artifacts = stores
        manifest_store.create(_created_manifest())
        assert manifest_store.exists(_ROBUSTNESS_ID)
        loaded = manifest_store.load(_ROBUSTNESS_ID)
        assert loaded.stage is RobustnessStage.CREATED

    def test_duplicate_create_is_rejected(self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore]) -> None:
        manifest_store, _events, _artifacts = stores
        manifest_store.create(_created_manifest())
        with pytest.raises(RobustnessStateError):
            manifest_store.create(_created_manifest())

    def test_illegal_stage_transition_is_rejected(self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore]) -> None:
        manifest_store, _events, _artifacts = stores
        manifest_store.create(_created_manifest())
        with pytest.raises(RobustnessStateError):
            manifest_store.transition(_ROBUSTNESS_ID, new_stage=RobustnessStage.COMPLETED, updated_at=format_utc_timestamp(utc_now()))

    def test_failed_stage_requires_failure_summary(self) -> None:
        now = format_utc_timestamp(utc_now())
        with pytest.raises(ValueError, match="failure_summary"):
            RobustnessManifest(schema_version=1, robustness_id=_ROBUSTNESS_ID, source_backtest_id=_SOURCE_BACKTEST_ID, stage=RobustnessStage.FAILED, created_at=now, updated_at=now)

    def test_unknown_artifact_kind_key_is_rejected(self) -> None:
        now = format_utc_timestamp(utc_now())
        ref = ArtifactReference(content_hash="c" * 64, category=ArtifactCategory.SOURCE_VERIFICATION_REPORT, size_bytes=1, created_at=now)
        with pytest.raises(ValueError, match="not_a_real_kind"):
            RobustnessManifest(
                schema_version=1, robustness_id=_ROBUSTNESS_ID, source_backtest_id=_SOURCE_BACKTEST_ID, stage=RobustnessStage.CREATED, created_at=now,
                updated_at=now, named_artifacts={"not_a_real_kind": ref},
            )

    def test_a_single_transition_can_carry_two_named_artifacts(self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore]) -> None:
        """`BOOTSTRAP_COMPLETED` alone produces two distinct artifacts
        (bootstrap_report, downside_analysis_report)."""
        manifest_store, _events, _artifacts = stores
        manifest_store.create(_created_manifest())
        now = format_utc_timestamp(utc_now())
        ref1 = ArtifactReference(content_hash="c" * 64, category=ArtifactCategory.SOURCE_VERIFICATION_REPORT, size_bytes=1, created_at=now)
        manifest_store.transition(_ROBUSTNESS_ID, new_stage=RobustnessStage.SOURCE_VERIFIED, updated_at=now, new_named_artifacts=(("source_verification_report", ref1),))
        ref2 = ArtifactReference(content_hash="d" * 64, category=ArtifactCategory.RETURN_SERIES_BUNDLE, size_bytes=1, created_at=now)
        manifest_store.transition(_ROBUSTNESS_ID, new_stage=RobustnessStage.SERIES_BUILT, updated_at=now, new_named_artifacts=(("return_series_bundle", ref2),))
        bootstrap_ref = ArtifactReference(content_hash="e" * 64, category=ArtifactCategory.BOOTSTRAP_REPORT, size_bytes=1, created_at=now)
        downside_ref = ArtifactReference(content_hash="f" * 64, category=ArtifactCategory.DOWNSIDE_ANALYSIS_REPORT, size_bytes=1, created_at=now)
        updated = manifest_store.transition(
            _ROBUSTNESS_ID, new_stage=RobustnessStage.BOOTSTRAP_COMPLETED, updated_at=now,
            new_named_artifacts=(("bootstrap_report", bootstrap_ref), ("downside_analysis_report", downside_ref)),
        )
        assert updated.artifact("bootstrap_report") == bootstrap_ref
        assert updated.artifact("downside_analysis_report") == downside_ref
        assert updated.artifact("source_verification_report") == ref1  # earlier artifacts accumulate, never overwritten

    def test_resume_count_bump(self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore]) -> None:
        manifest_store, _events, _artifacts = stores
        manifest_store.create(_created_manifest())
        bumped = manifest_store.bump_resume_count(_ROBUSTNESS_ID)
        assert bumped.resume_count == 1

    def test_json_round_trip(self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore]) -> None:
        manifest_store, _events, _artifacts = stores
        manifest_store.create(_created_manifest())
        loaded = manifest_store.load(_ROBUSTNESS_ID)
        assert RobustnessManifest.from_json_dict(loaded.to_json_dict()) == loaded

    def test_artifact_kinds_has_no_duplicates(self) -> None:
        assert len(ARTIFACT_KINDS) == len(set(ARTIFACT_KINDS))


class TestEventLog:
    def test_events_are_sequential_and_append_only(self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore]) -> None:
        manifest_store, event_store, _artifacts = stores
        manifest_store.create(_created_manifest())
        event_store.append(_ROBUSTNESS_ID, RobustnessEventType.ROBUSTNESS_CREATED)
        event_store.append(_ROBUSTNESS_ID, RobustnessEventType.SOURCE_VERIFIED)
        events = event_store.read_events(_ROBUSTNESS_ID)
        assert [e.event_type for e in events] == [RobustnessEventType.ROBUSTNESS_CREATED, RobustnessEventType.SOURCE_VERIFIED]
        assert [e.sequence for e in events] == [1, 2]


class TestSemanticTamperingDetection:
    def test_no_artifacts_verifies_through_created_only(self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore]) -> None:
        manifest_store, _events, artifact_store = stores
        manifest_store.create(_created_manifest())
        loaded = manifest_store.load(_ROBUSTNESS_ID)
        assert verify_completed_robustness_stages(loaded, artifact_store=artifact_store) is RobustnessStage.CREATED

    def test_genuine_artifact_advances_verification(self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore]) -> None:
        manifest_store, _events, artifact_store = stores
        manifest_store.create(_created_manifest())
        now = format_utc_timestamp(utc_now())
        report = SourceVerificationReport(
            schema_version=1, source_backtest_id=_SOURCE_BACKTEST_ID, verify_backtest_is_ready=True, verify_backtest_critical_count=0,
            verify_backtest_issue_codes=(), dataset_content_id_matches=True, split_plan_fingerprint_matches=True, instrument_identity_matches=True,
            bar_interval_matches=True, total_outer_folds=6, generated_at=now,
        )
        ref = artifact_store.write_artifact(canonical_json_bytes(report.to_json_dict()), category=ArtifactCategory.SOURCE_VERIFICATION_REPORT)
        manifest_store.transition(_ROBUSTNESS_ID, new_stage=RobustnessStage.SOURCE_VERIFIED, updated_at=now, new_named_artifacts=(("source_verification_report", ref),))
        loaded = manifest_store.load(_ROBUSTNESS_ID)
        assert verify_completed_robustness_stages(loaded, artifact_store=artifact_store) is RobustnessStage.SOURCE_VERIFIED

    def test_tampered_artifact_reference_demotes_verification(self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore]) -> None:
        """A manifest pointing at a content_hash that was never actually
        written to the artifact store (simulating tampering or a
        garbage-collected artifact) must NOT verify past CREATED."""
        manifest_store, _events, artifact_store = stores
        manifest_store.create(_created_manifest())
        now = format_utc_timestamp(utc_now())
        bogus_ref = ArtifactReference(content_hash="f" * 64, category=ArtifactCategory.SOURCE_VERIFICATION_REPORT, size_bytes=1, created_at=now)
        manifest_store.transition(_ROBUSTNESS_ID, new_stage=RobustnessStage.SOURCE_VERIFIED, updated_at=now, new_named_artifacts=(("source_verification_report", bogus_ref),))
        loaded = manifest_store.load(_ROBUSTNESS_ID)
        assert verify_completed_robustness_stages(loaded, artifact_store=artifact_store) is RobustnessStage.CREATED

    def test_manifest_overclaiming_its_own_stage_does_not_fool_verification(self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore]) -> None:
        """The central property this module exists to prove: `manifest.
        stage`'s own claim is NEVER trusted -- only re-verified artifacts
        are."""
        manifest_store, _events, artifact_store = stores
        manifest_store.create(_created_manifest())
        now = format_utc_timestamp(utc_now())
        report = SourceVerificationReport(
            schema_version=1, source_backtest_id=_SOURCE_BACKTEST_ID, verify_backtest_is_ready=True, verify_backtest_critical_count=0,
            verify_backtest_issue_codes=(), dataset_content_id_matches=True, split_plan_fingerprint_matches=True, instrument_identity_matches=True,
            bar_interval_matches=True, total_outer_folds=6, generated_at=now,
        )
        ref = artifact_store.write_artifact(canonical_json_bytes(report.to_json_dict()), category=ArtifactCategory.SOURCE_VERIFICATION_REPORT)
        manifest_store.transition(_ROBUSTNESS_ID, new_stage=RobustnessStage.SOURCE_VERIFIED, updated_at=now, new_named_artifacts=(("source_verification_report", ref),))
        loaded = manifest_store.load(_ROBUSTNESS_ID)
        overclaiming = dc_replace(loaded, stage=RobustnessStage.STRESS_COMPLETED)  # claims far more than its artifacts support
        assert verify_completed_robustness_stages(overclaiming, artifact_store=artifact_store) is RobustnessStage.SOURCE_VERIFIED


class TestResumeGating:
    def test_can_resume_true_for_non_terminal_stage(self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore]) -> None:
        manifest_store, _events, _artifacts = stores
        manifest_store.create(_created_manifest())
        assert can_resume(manifest_store.load(_ROBUSTNESS_ID)) is True

    def test_can_resume_false_for_missing_manifest(self) -> None:
        assert can_resume(None) is False

    def test_terminal_stage_refuses_resume(self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore]) -> None:
        manifest_store, _events, _artifacts = stores
        manifest_store.create(_created_manifest())
        loaded = manifest_store.load(_ROBUSTNESS_ID)
        completed = dc_replace(loaded, stage=RobustnessStage.COMPLETED, completed_at=format_utc_timestamp(utc_now()))
        assert can_resume(completed) is False
        with pytest.raises(Exception, match="terminal"):
            require_robustness_resumable(completed, robustness_id=_ROBUSTNESS_ID)

    def test_missing_manifest_raises_with_actionable_message(self) -> None:
        with pytest.raises(Exception, match="nothing to resume"):
            require_robustness_resumable(None, robustness_id=_ROBUSTNESS_ID)


def _valid_artifact_bytes(kind: str) -> bytes:
    """A minimal, genuinely-decodable artifact for `kind`, built directly
    from each report module's own dataclasses (never via the runner) --
    used to build a REAL chain of trustworthy artifacts through every
    `RobustnessStage` boundary, so `verify_completed_robustness_stages`'
    "never trust manifest.stage alone" guarantee can be tested at EVERY
    boundary (closure-audit Section 9) without needing to execute the
    full, expensive production pipeline."""
    from quant_platform.robustness.bootstrap import BootstrapEstimate, BootstrapReport, DownsideAnalysisReport
    from quant_platform.robustness.models import ReturnSeriesKind
    from quant_platform.robustness.promotion import GateEvaluation as PromotionGateEvaluation
    from quant_platform.robustness.promotion import PromotionDecision
    from quant_platform.robustness.selection import (
        CandidateEligibility,
        GateEvaluation,
        GateOutcomeKind,
        RankingEntry,
        SelectionReport,
    )
    from quant_platform.robustness.series import ReturnSeriesBundle
    from quant_platform.robustness.stability import ConcentrationReport, FoldStabilityReport
    from quant_platform.robustness.stress import StressReport

    now = format_utc_timestamp(utc_now())
    if kind == "source_verification_report":
        report = SourceVerificationReport(
            schema_version=1, source_backtest_id=_SOURCE_BACKTEST_ID, verify_backtest_is_ready=True, verify_backtest_critical_count=0,
            verify_backtest_issue_codes=(), dataset_content_id_matches=True, split_plan_fingerprint_matches=True, instrument_identity_matches=True,
            bar_interval_matches=True, total_outer_folds=6, generated_at=now,
        )
    elif kind == "return_series_bundle":
        report = ReturnSeriesBundle(
            schema_version=1, kind=ReturnSeriesKind.STITCHED_BAR_NET, sampling_frequency="bar", observation_count=2, effective_sample_count=2,
            values=(0.01, -0.005), fold_boundaries=(), source_artifact_content_hashes=("a" * 64,), time_range_start=None, time_range_end=None, built_at=now,
        )
    elif kind == "bootstrap_report":
        report = BootstrapReport(
            schema_version=1, series_kind="stitched_bar_net", method="stationary", repetitions=100, confidence_level=0.95, seed=0,
            estimates=(BootstrapEstimate(statistic_name="total_return", point_estimate=0.02, lower_bound=0.0, upper_bound=0.04, valid_repetitions=100, skipped_repetitions=0, failure_reasons={}),),
            generated_at=now,
        )
    elif kind == "downside_analysis_report":
        report = DownsideAnalysisReport(
            schema_version=1, probability_total_net_return_non_positive=0.1, probability_mean_return_non_positive=0.1, probability_sharpe_non_positive=0.1,
            probability_underperforms_always_flat=None, probability_underperforms_always_long=None, probability_cost_stressed_unprofitable=None,
            probability_maximum_drawdown_exceeds_limit=None, drawdown_limit=None, repetitions=100, seed=0, generated_at=now,
        )
    elif kind == "fold_stability_report":
        concentration = ConcentrationReport(
            schema_version=1, single_fold_profit_concentration=0.5, single_trade_profit_concentration=0.5, single_day_profit_concentration=0.5,
            single_direction_profit_concentration=0.5, single_confidence_bucket_profit_concentration=0.5, warning_codes=(),
        )
        report = FoldStabilityReport(
            schema_version=1, fold_count=3, profitable_fold_fraction=0.66, positive_sharpe_fold_fraction=0.66, median_fold_return=0.01,
            worst_fold_return=-0.005, fold_return_stdev=0.01, fold_sharpe_dispersion=0.1, maximum_fold_drawdown=0.05, worst_fold_cost_drag=0.001,
            fold_trade_count_dispersion=1.0, fold_exposure_dispersion=0.1, direction_consistency=0.8, benchmark_outperformance_fraction=0.6,
            concentration=concentration, generated_at=now,
        )
    elif kind == "stress_report":
        report = StressReport(
            schema_version=1, source_backtest_id=_SOURCE_BACKTEST_ID, baseline_total_net_return=0.02, baseline_total_gross_return=0.025,
            baseline_closed_trade_count=20, baseline_maximum_drawdown=0.05, scenario_results=(), breakeven_results=(), generated_at=now,
        )
    elif kind == "selection_report":
        eligibility = CandidateEligibility(
            robustness_id=_ROBUSTNESS_ID, source_backtest_id=_SOURCE_BACKTEST_ID, eligible=True,
            gate_evaluations=(GateEvaluation(gate_name="minimum_fold_count", mandatory=True, measured_value=3.0, minimum_value=3.0, maximum_value=None, outcome=GateOutcomeKind.PASS, reason=None),),
            rejection_reasons=(),
        )
        report = SelectionReport(
            schema_version=1, family_id=None, policy_identity="a" * 64, candidate_eligibility=(eligibility,),
            ranking=(RankingEntry(robustness_id=_ROBUSTNESS_ID, metric_values={"worst_fold_return": -0.005}),),
            tie_break_procedure="deterministic", selected_candidate_robustness_id=_ROBUSTNESS_ID, selection_identity="b" * 64, generated_at=now,
        )
    elif kind == "promotion_decision":
        from quant_platform.robustness.models import PromotionDecisionKind

        report = PromotionDecision(
            schema_version=1, robustness_id=_ROBUSTNESS_ID, source_backtest_id=_SOURCE_BACKTEST_ID, decision=PromotionDecisionKind.RESEARCH_ONLY,
            decision_reason="advisory gate failed", gate_evaluations=(PromotionGateEvaluation(gate_name="no_parameter_cliff", mandatory=False, measured_value=0.0, minimum_value=1.0, maximum_value=None, outcome=GateOutcomeKind.FAIL, reason="cliff detected"),),
            disclaimer="not a live-trading recommendation", generated_at=now,
        )
    elif kind in ("verification_report", "robustness_report"):
        return canonical_json_bytes({"note": "presence/readability only per resume._DECODERS", "generated_at": now})
    else:
        raise AssertionError(f"no fixture builder for artifact kind {kind!r}")
    return canonical_json_bytes(report.to_json_dict())


_STAGE_CHAIN: tuple[tuple[RobustnessStage, tuple[str, ...]], ...] = (
    (RobustnessStage.SOURCE_VERIFIED, ("source_verification_report",)),
    (RobustnessStage.SERIES_BUILT, ("return_series_bundle",)),
    (RobustnessStage.BOOTSTRAP_COMPLETED, ("bootstrap_report", "downside_analysis_report")),
    (RobustnessStage.STABILITY_COMPLETED, ("fold_stability_report",)),
    (RobustnessStage.STRESS_COMPLETED, ("stress_report",)),
    (RobustnessStage.REGIMES_COMPLETED, ()),
    (RobustnessStage.SELECTION_COMPLETED, ("selection_report",)),
    (RobustnessStage.PROMOTION_EVALUATED, ("promotion_decision",)),
    (RobustnessStage.VERIFIED, ("verification_report",)),
    (RobustnessStage.COMPLETED, ("robustness_report",)),
)


class TestCrashResumeMatrixAcrossEveryStageBoundary:
    """Closure-audit Section 9: builds a REAL chain of genuinely-
    decodable artifacts through all ten `RobustnessStage` boundaries,
    then -- for EVERY boundary in turn -- corrupts exactly that stage's
    own artifact and confirms `verify_completed_robustness_stages`
    demotes the trustworthy resume point to precisely the PRECEDING
    stage, never further, never less. This directly tests the safety
    mechanism a real crash-and-resume depends on (an interrupted process
    restarting must never resume past a point its own artifacts cannot
    independently prove it reached) without re-executing the full,
    expensive production pipeline (whose individual stage computations
    are already exhaustively hand-verified elsewhere in this suite)."""

    @pytest.mark.parametrize("corrupted_stage_index", list(range(len(_STAGE_CHAIN))))
    def test_corrupting_any_single_stage_artifact_demotes_to_exactly_its_predecessor(
        self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore], corrupted_stage_index: int,
    ) -> None:
        manifest_store, _events, artifact_store = stores
        manifest_store.create(_created_manifest())
        expected_verified_through = RobustnessStage.CREATED
        for i, (stage, kinds) in enumerate(_STAGE_CHAIN):
            now = format_utc_timestamp(utc_now())
            if i == corrupted_stage_index and kinds:
                # Corrupt EXACTLY this stage's own artifact(s): point at a
                # content hash that was never written (simulates a
                # truncated write / crash mid-flush / tampering).
                bogus = ArtifactReference(content_hash="f" * 64, category=ArtifactCategory.SOURCE_VERIFICATION_REPORT, size_bytes=1, created_at=now)
                named = tuple((kind, bogus) for kind in kinds)
            else:
                named = tuple((kind, artifact_store.write_artifact(_valid_artifact_bytes(kind), category=ArtifactCategory.SOURCE_VERIFICATION_REPORT)) for kind in kinds)
            manifest_store.transition(_ROBUSTNESS_ID, new_stage=stage, updated_at=now, new_named_artifacts=named)
            if i < corrupted_stage_index:
                expected_verified_through = stage

        loaded = manifest_store.load(_ROBUSTNESS_ID)
        stage_at_corruption, kinds_at_corruption = _STAGE_CHAIN[corrupted_stage_index]
        if not kinds_at_corruption:
            # REGIMES_COMPLETED has no artifact of its own (RegimeReport is
            # optional) -- there is nothing to corrupt at this boundary, so
            # the chain legitimately builds fully valid and verifies all the
            # way to COMPLETED, exactly like the fully-uncorrupted case.
            expected_verified_through = RobustnessStage.COMPLETED
        verified_through = verify_completed_robustness_stages(loaded, artifact_store=artifact_store)
        assert verified_through is expected_verified_through, (
            f"corrupting stage {stage_at_corruption.value!r}'s artifact must demote the trustworthy resume point to "
            f"exactly {expected_verified_through.value!r}, got {verified_through.value!r}"
        )

    def test_full_uncorrupted_chain_verifies_all_the_way_to_completed(
        self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore],
    ) -> None:
        manifest_store, _events, artifact_store = stores
        manifest_store.create(_created_manifest())
        for stage, kinds in _STAGE_CHAIN:
            now = format_utc_timestamp(utc_now())
            named = tuple((kind, artifact_store.write_artifact(_valid_artifact_bytes(kind), category=ArtifactCategory.SOURCE_VERIFICATION_REPORT)) for kind in kinds)
            manifest_store.transition(_ROBUSTNESS_ID, new_stage=stage, updated_at=now, new_named_artifacts=named)
        loaded = manifest_store.load(_ROBUSTNESS_ID)
        assert verify_completed_robustness_stages(loaded, artifact_store=artifact_store) is RobustnessStage.COMPLETED
        assert can_resume(loaded) is False  # COMPLETED is terminal -- nothing left to resume


class TestResumeRewindOnDetectedCorruptionIsLegal:
    """Release-audit regression (closure-audit Section 9): a REAL defect
    found and fixed during this pass. `RobustnessRunner._run_locked`
    re-verifies a resumed manifest via `resolve_resume_start_stage` and,
    when that finds LESS genuine progress than `manifest.stage` claims
    (an earlier artifact corrupted/lost/truncated AFTER its stage was
    legitimately reached -- the exact "process died mid-write" case this
    whole mechanism exists to survive), attempts to REWIND the manifest's
    `stage` field back down to the last independently-verified stage
    before recomputing forward. Before this fix, `is_legal_robustness_
    transition` allowed ONLY a forward-by-exactly-one-stage transition
    (or a transition to `FAILED`) -- ANY backward rewind was rejected as
    "illegal", so this recovery path raised `RobustnessStateError`
    instead of recovering, and the run was left permanently stuck
    (`can_resume` kept reporting it resumable, but every resume attempt
    hit the identical crash). Fixed in `robustness.models.is_legal_
    robustness_transition` by additionally allowing a transition to any
    STRICTLY EARLIER non-terminal stage from any non-terminal current
    stage -- an operation the normal forward-only pipeline itself never
    performs (so this cannot change ordinary forward-run behavior), and
    the ONE operation this recovery path needs. These tests fail against
    the pre-fix transition table (verified directly before applying the
    fix, reproducing the exact `RobustnessStateError`) and pass after
    it."""

    def test_rewind_to_a_strictly_earlier_non_terminal_stage_is_legal(self) -> None:
        assert is_legal_robustness_transition(RobustnessStage.BOOTSTRAP_COMPLETED, RobustnessStage.SOURCE_VERIFIED) is True
        assert is_legal_robustness_transition(RobustnessStage.STRESS_COMPLETED, RobustnessStage.CREATED) is True
        assert is_legal_robustness_transition(RobustnessStage.PROMOTION_EVALUATED, RobustnessStage.SERIES_BUILT) is True

    def test_forward_by_one_and_to_failed_remain_legal(self) -> None:
        """The fix must not narrow anything the normal forward pipeline
        already relies on."""
        assert is_legal_robustness_transition(RobustnessStage.SOURCE_VERIFIED, RobustnessStage.SERIES_BUILT) is True
        assert is_legal_robustness_transition(RobustnessStage.STRESS_COMPLETED, RobustnessStage.FAILED) is True

    def test_forward_skip_over_intermediate_stages_remains_illegal(self) -> None:
        """The fix legalizes REWIND only -- it must not accidentally also
        legalize skipping ahead without doing the intervening work."""
        assert is_legal_robustness_transition(RobustnessStage.SOURCE_VERIFIED, RobustnessStage.STRESS_COMPLETED) is False
        assert is_legal_robustness_transition(RobustnessStage.CREATED, RobustnessStage.COMPLETED) is False

    def test_terminal_stages_permit_no_transition_at_all_not_even_rewind(self) -> None:
        assert is_legal_robustness_transition(RobustnessStage.COMPLETED, RobustnessStage.SOURCE_VERIFIED) is False
        assert is_legal_robustness_transition(RobustnessStage.FAILED, RobustnessStage.SOURCE_VERIFIED) is False
        assert is_legal_robustness_transition(RobustnessStage.COMPLETED, RobustnessStage.CREATED) is False

    def test_rewind_to_self_is_not_a_legal_new_case(self) -> None:
        """A stage transitioning to itself was never legal before this
        fix (forward-by-one always targets a DIFFERENT stage) and must
        not become legal now -- `_run_locked` only invokes the rewind
        path when `start_stage is not manifest.stage` in the first place."""
        assert is_legal_robustness_transition(RobustnessStage.STRESS_COMPLETED, RobustnessStage.STRESS_COMPLETED) is False

    def test_manifest_store_transition_actually_performs_the_rewind_end_to_end(
        self, stores: tuple[RobustnessManifestStore, RobustnessEventStore, MLArtifactStore],
    ) -> None:
        """End-to-end reproduction of the exact scenario that crashed
        before this fix: a manifest claims `BOOTSTRAP_COMPLETED`, but its
        `SERIES_BUILT`-stage artifact was corrupted after the fact (a
        dangling content-hash reference). `resolve_resume_start_stage`
        correctly demotes to `SOURCE_VERIFIED`; `RobustnessManifestStore.
        transition` -- EXACTLY the call `RobustnessRunner._run_locked`
        makes to perform the rewind -- must now succeed rather than
        raising `RobustnessStateError`."""
        manifest_store, _events, artifact_store = stores
        manifest_store.create(_created_manifest())
        now = format_utc_timestamp(utc_now())
        source_ref = artifact_store.write_artifact(_valid_artifact_bytes("source_verification_report"), category=ArtifactCategory.SOURCE_VERIFICATION_REPORT)
        manifest_store.transition(_ROBUSTNESS_ID, new_stage=RobustnessStage.SOURCE_VERIFIED, updated_at=now, new_named_artifacts=(("source_verification_report", source_ref),))
        bogus_series_ref = ArtifactReference(content_hash="f" * 64, category=ArtifactCategory.RETURN_SERIES_BUNDLE, size_bytes=1, created_at=now)
        manifest_store.transition(_ROBUSTNESS_ID, new_stage=RobustnessStage.SERIES_BUILT, updated_at=now, new_named_artifacts=(("return_series_bundle", bogus_series_ref),))
        bogus_bootstrap_ref = ArtifactReference(content_hash="e" * 64, category=ArtifactCategory.BOOTSTRAP_REPORT, size_bytes=1, created_at=now)
        bogus_downside_ref = ArtifactReference(content_hash="d" * 64, category=ArtifactCategory.DOWNSIDE_ANALYSIS_REPORT, size_bytes=1, created_at=now)
        manifest_store.transition(
            _ROBUSTNESS_ID, new_stage=RobustnessStage.BOOTSTRAP_COMPLETED, updated_at=now,
            new_named_artifacts=(("bootstrap_report", bogus_bootstrap_ref), ("downside_analysis_report", bogus_downside_ref)),
        )
        loaded = manifest_store.load(_ROBUSTNESS_ID)
        assert loaded.stage is RobustnessStage.BOOTSTRAP_COMPLETED

        start_stage = resolve_resume_start_stage(loaded, artifact_store=artifact_store)
        assert start_stage is RobustnessStage.SOURCE_VERIFIED  # SERIES_BUILT's own artifact is corrupted -- demoted past it

        rewound = manifest_store.transition(_ROBUSTNESS_ID, new_stage=start_stage, updated_at=format_utc_timestamp(utc_now()))
        assert rewound.stage is RobustnessStage.SOURCE_VERIFIED
        # A fresh forward pass can now legally proceed from the rewound stage.
        assert is_legal_robustness_transition(rewound.stage, RobustnessStage.SERIES_BUILT) is True
