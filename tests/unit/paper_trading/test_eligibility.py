"""Milestone 7, Section 4: eligibility/source verification chain. These
tests exercise `verify_paper_trading_eligibility`'s own branching logic
(steps 1-4 of the chain: manifest/promotion-decision presence, content-
hash tampering, decision kind, robustness_id cross-reference) using
lightweight fakes for the manifest/artifact stores -- cheap and fast,
since those steps never need a genuine trained model. Steps 5-9 (full
`verify_robustness`/`verify_and_load_source_backtest` recomputation
against REAL evidence) are deliberately NOT re-tested here with fakes --
faking them would mean reimplementing large parts of Milestone 6's own
acceptance fixture with no real verification value. They are instead
exercised end-to-end, once, against a genuine trained model in the
Milestone 7 real-acceptance-workflow integration test (Section 33),
where a real `ELIGIBLE_FOR_PAPER_TRADING` chain already needs to exist
anyway -- avoiding duplicating that expensive fixture here."""

from __future__ import annotations

import pytest

from quant_platform.backtesting.models import (
    CommissionModelKind,
    FinancingModelKind,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.specs import CommissionSpec, FinancingSpec, SlippageSpec, SpreadSpec
from quant_platform.core.exceptions import ArtifactNotFoundError, PaperTradingEligibilityError
from quant_platform.core.types import Timeframe
from quant_platform.ml.models import ArtifactCategory, ArtifactReference
from quant_platform.ml.persistence import format_utc_timestamp, utc_now
from quant_platform.paper_trading.eligibility import (
    EligibilityVerificationEnvironment,
    EligibilityVerificationReport,
    require_paper_trading_eligibility,
    verify_paper_trading_eligibility,
)
from quant_platform.paper_trading.models import ClockMode, MarketEventMode, PartialFillPolicyKind, SessionMode
from quant_platform.paper_trading.specs import (
    DEFAULT_EXECUTION_POLICY,
    DEFAULT_POSITION_POLICY,
    DEFAULT_RISK_LIMITS,
    DEFAULT_SESSION_BOUNDARY_POLICY,
    FillPolicySpec,
    FinancingPolicySpec,
    InstrumentSpec,
    LatencyPolicySpec,
    LiquidityPolicySpec,
    OrderPolicySpec,
    PaperTradingSpec,
)
from quant_platform.robustness.manifests import RobustnessManifest
from quant_platform.robustness.models import PromotionDecisionKind, RobustnessStage
from quant_platform.robustness.promotion import PromotionDecision

_HEX_ROBUSTNESS_ID = "1" * 64
_HEX_PROMOTION_HASH = "2" * 64
_HEX_BACKTEST_ID = "3" * 64
_HEX_MODEL_ID = "4" * 64
_HEX_CALIBRATION_ID = "5" * 64
_HEX_FEATURE_ID = "6" * 64


def _instrument() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="HYPOTHETICAL_XAU", base_currency="XAU", quote_currency="USD", contract_multiplier=100.0, tick_size=0.01, tick_value=1.0,
        quantity_step=0.01, minimum_quantity=0.01, maximum_quantity=100.0, price_precision=2, quantity_precision=2,
        margin_mode="hypothetical_margin_mode", account_currency="USD", financing_convention="hypothetical_daily_swap",
        trading_timezone="UTC", session_calendar_identity="hypothetical_247",
    )


def _spec(**overrides: object) -> PaperTradingSpec:
    defaults: dict[str, object] = {
        "schema_version": 1, "verified_robustness_id": _HEX_ROBUSTNESS_ID, "verified_promotion_decision_id": _HEX_PROMOTION_HASH,
        "strategy_candidate_identity": _HEX_BACKTEST_ID, "model_artifact_identity": _HEX_MODEL_ID, "calibration_artifact_identity": _HEX_CALIBRATION_ID,
        "feature_spec_identity": _HEX_FEATURE_ID, "instrument": _instrument(), "price_precision": 2, "quantity_precision": 2,
        "session_mode": SessionMode.REPLAY_PAPER, "market_event_mode": MarketEventMode.BAR, "bar_interval": Timeframe.H1,
        "clock_mode": ClockMode.REPLAY, "starting_cash": 100_000.0, "starting_positions": (),
        "order_policy": OrderPolicySpec(close_before_reverse=True, cooldown_bars=0, maximum_orders_per_event=5, maximum_order_rate_per_window=10, order_rate_window_events=20),
        "execution_policy": DEFAULT_EXECUTION_POLICY, "fill_policy": FillPolicySpec(partial_fill_policy=PartialFillPolicyKind.FULL_FILL_ONLY),
        "spread_policy": SpreadSpec(kind=SpreadModelKind.FIXED_PRICE_UNITS, price_units=0.3),
        "slippage_policy": SlippageSpec(kind=SlippageModelKind.FIXED_BASIS_POINTS, basis_points=1.0),
        "commission_policy": CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=2.0),
        "financing_policy": FinancingPolicySpec(
            long_financing=FinancingSpec(kind=FinancingModelKind.FIXED_DAILY_BASIS_POINTS, daily_basis_points=1.5),
            short_financing=FinancingSpec(kind=FinancingModelKind.FIXED_DAILY_BASIS_POINTS, daily_basis_points=-0.5),
        ),
        "latency_policy": LatencyPolicySpec(decision_to_submit_ms=50, submit_to_accept_ms=50, accept_to_fill_eligible_ms=50),
        "liquidity_policy": LiquidityPolicySpec(trust_disclosed_size=False), "position_policy": DEFAULT_POSITION_POLICY,
        "risk_limits": DEFAULT_RISK_LIMITS, "session_boundary_policy": DEFAULT_SESSION_BOUNDARY_POLICY, "seed": 0,
    }
    defaults.update(overrides)
    return PaperTradingSpec(**defaults)  # type: ignore[arg-type]


class _FakeRobustnessManifestStore:
    def __init__(self, manifest: RobustnessManifest | None) -> None:
        self._manifest = manifest

    def load(self, robustness_id: str) -> RobustnessManifest:
        if self._manifest is None:
            raise ArtifactNotFoundError(f"no robustness manifest for {robustness_id!r}")
        return self._manifest


class _FakeArtifactStore:
    def __init__(self, content_by_hash: dict[str, bytes]) -> None:
        self._content = content_by_hash

    def read_artifact(self, content_hash: str) -> bytes:
        if content_hash not in self._content:
            raise ArtifactNotFoundError(f"no artifact for content_hash={content_hash!r}")
        return self._content[content_hash]


def _environment(*, manifest: RobustnessManifest | None, content_by_hash: dict[str, bytes] | None = None) -> EligibilityVerificationEnvironment:
    """The 8 downstream stores are never touched by the steps these tests
    exercise (short-circuit fails before `verify_robustness` is ever
    called) -- `None` stands in for "never dereferenced"."""
    return EligibilityVerificationEnvironment(
        robustness_manifest_store=_FakeRobustnessManifestStore(manifest),  # type: ignore[arg-type]
        artifact_store=_FakeArtifactStore(content_by_hash or {}),  # type: ignore[arg-type]
        backtest_manifest_store=None,  # type: ignore[arg-type]
        backtest_event_store=None,  # type: ignore[arg-type]
        calibration_manifest_store=None,  # type: ignore[arg-type]
        experiment_manifest_store=None,  # type: ignore[arg-type]
        execution_manifest_store=None,  # type: ignore[arg-type]
        research_manifest_store=None,  # type: ignore[arg-type]
        research_dataset_store=None,  # type: ignore[arg-type]
        dataset_loader=None,  # type: ignore[arg-type]
    )


def _promotion_decision(*, robustness_id: str = _HEX_ROBUSTNESS_ID, decision: PromotionDecisionKind = PromotionDecisionKind.ELIGIBLE_FOR_PAPER_TRADING) -> PromotionDecision:
    return PromotionDecision(
        schema_version=1, robustness_id=robustness_id, source_backtest_id=_HEX_BACKTEST_ID, decision=decision,
        decision_reason="test fixture", gate_evaluations=(), disclaimer="not a claim of profitability", generated_at=format_utc_timestamp(utc_now()),
    )


def _manifest_with_promotion(*, stage: RobustnessStage = RobustnessStage.COMPLETED, promotion_content_hash: str | None = _HEX_PROMOTION_HASH) -> RobustnessManifest:
    named_artifacts = {}
    if promotion_content_hash is not None:
        named_artifacts["promotion_decision"] = ArtifactReference(category=ArtifactCategory.PROMOTION_DECISION, content_hash=promotion_content_hash, size_bytes=10, created_at=format_utc_timestamp(utc_now()))
    return RobustnessManifest(
        schema_version=1, robustness_id=_HEX_ROBUSTNESS_ID, source_backtest_id=_HEX_BACKTEST_ID, stage=stage,
        created_at=format_utc_timestamp(utc_now()), updated_at=format_utc_timestamp(utc_now()), named_artifacts=named_artifacts,
    )


class TestEligibilityVerificationReportShape:
    def _eligible_report(self) -> EligibilityVerificationReport:
        return EligibilityVerificationReport(
            schema_version=1, paper_session_spec_id="a" * 64, robustness_id=_HEX_ROBUSTNESS_ID, promotion_decision_content_hash=_HEX_PROMOTION_HASH,
            promotion_decision_exists=True, promotion_decision_content_verifies=True, decision_kind_is_eligible_for_paper_trading=True,
            promotion_decision_references_requested_robustness=True, robustness_result_verifies=True, source_backtest_verifies=True,
            identities_match=True, required_source_artifacts_present=True, no_artifact_failed_incomplete_stale_or_mismatched=True,
            resolved_strategy_candidate_identity=_HEX_BACKTEST_ID, resolved_model_artifact_identity=_HEX_MODEL_ID,
            resolved_calibration_artifact_identity=_HEX_CALIBRATION_ID, resolved_feature_spec_identity=_HEX_FEATURE_ID,
            failed_step=None, failure_reason=None, generated_at=format_utc_timestamp(utc_now()),
        )

    def test_all_true_and_no_failed_step_is_eligible(self) -> None:
        assert self._eligible_report().is_eligible

    def test_any_false_step_makes_report_ineligible(self) -> None:
        import dataclasses

        report = dataclasses.replace(self._eligible_report(), identities_match=False)
        assert not report.is_eligible

    def test_failed_step_set_makes_report_ineligible_even_if_flags_are_true(self) -> None:
        import dataclasses

        report = dataclasses.replace(self._eligible_report(), failed_step="identities_match", failure_reason="mismatch")
        assert not report.is_eligible

    def test_json_round_trip(self) -> None:
        report = self._eligible_report()
        roundtripped = EligibilityVerificationReport.from_json_dict(report.to_json_dict())
        assert roundtripped == report


class TestEligibilityChainShortCircuits:
    def test_missing_robustness_manifest_fails_at_promotion_decision_exists(self) -> None:
        report = verify_paper_trading_eligibility(_spec(), environment=_environment(manifest=None))
        assert not report.is_eligible
        assert report.failed_step == "promotion_decision_exists"

    def test_incomplete_robustness_run_fails_closed(self) -> None:
        manifest = _manifest_with_promotion(stage=RobustnessStage.BOOTSTRAP_COMPLETED)
        report = verify_paper_trading_eligibility(_spec(), environment=_environment(manifest=manifest))
        assert not report.is_eligible
        assert report.failed_step == "no_artifact_failed_incomplete_stale_or_mismatched"

    def test_completed_run_missing_promotion_decision_artifact_fails(self) -> None:
        manifest = _manifest_with_promotion(promotion_content_hash=None)
        report = verify_paper_trading_eligibility(_spec(), environment=_environment(manifest=manifest))
        assert not report.is_eligible
        assert report.failed_step == "promotion_decision_exists"

    def test_tampered_promotion_decision_content_hash_reference_fails(self) -> None:
        """The manifest's own artifact reference does not match what the
        spec declares as `verified_promotion_decision_id` -- a tampered
        or stale reference, must be rejected before ever reading content."""
        manifest = _manifest_with_promotion(promotion_content_hash="9" * 64)
        report = verify_paper_trading_eligibility(_spec(), environment=_environment(manifest=manifest))
        assert not report.is_eligible
        assert report.failed_step == "promotion_decision_exists"
        assert report.promotion_decision_exists is False

    def test_undecodable_promotion_decision_content_fails_at_content_verifies(self) -> None:
        manifest = _manifest_with_promotion()
        environment = _environment(manifest=manifest, content_by_hash={_HEX_PROMOTION_HASH: b"not valid json"})
        report = verify_paper_trading_eligibility(_spec(), environment=environment)
        assert not report.is_eligible
        assert report.failed_step == "promotion_decision_content_verifies"
        assert report.promotion_decision_exists is True

    @pytest.mark.parametrize("decision_kind", [PromotionDecisionKind.REJECTED, PromotionDecisionKind.RESEARCH_ONLY, PromotionDecisionKind.MANUAL_REVIEW_REQUIRED])
    def test_non_eligible_decision_kinds_all_rejected(self, decision_kind: PromotionDecisionKind) -> None:
        manifest = _manifest_with_promotion()
        promotion = _promotion_decision(decision=decision_kind)
        environment = _environment(manifest=manifest, content_by_hash={_HEX_PROMOTION_HASH: _promotion_json_bytes(promotion)})
        report = verify_paper_trading_eligibility(_spec(), environment=environment)
        assert not report.is_eligible
        assert report.failed_step == "decision_kind_is_eligible_for_paper_trading"
        assert report.promotion_decision_content_verifies is True

    def test_promotion_decision_belonging_to_another_robustness_run_rejected(self) -> None:
        manifest = _manifest_with_promotion()
        promotion = _promotion_decision(robustness_id="8" * 64)
        environment = _environment(manifest=manifest, content_by_hash={_HEX_PROMOTION_HASH: _promotion_json_bytes(promotion)})
        report = verify_paper_trading_eligibility(_spec(), environment=environment)
        assert not report.is_eligible
        assert report.failed_step == "promotion_decision_references_requested_robustness"
        assert report.decision_kind_is_eligible_for_paper_trading is True


class TestRequirePaperTradingEligibilityRaisesFailClosed:
    def test_raises_paper_trading_eligibility_error_when_not_eligible(self) -> None:
        with pytest.raises(PaperTradingEligibilityError, match="NOT eligible"):
            require_paper_trading_eligibility(_spec(), environment=_environment(manifest=None))

    def test_error_message_includes_failed_step_and_reason(self) -> None:
        with pytest.raises(PaperTradingEligibilityError, match="promotion_decision_exists"):
            require_paper_trading_eligibility(_spec(), environment=_environment(manifest=None))


def _promotion_json_bytes(promotion: PromotionDecision) -> bytes:
    import json

    return json.dumps(promotion.to_json_dict()).encode("utf-8")
