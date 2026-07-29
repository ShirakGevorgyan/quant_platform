"""Milestone 7, Section 26: `verify_paper_session` semantic-tampering
tests. A real session's ledger (built via `run_paper_trading_session`,
reusing the same fixture pattern as `test_reconciliation.py`) must clear
every check EXCEPT eligibility (deliberately faked to fail closed here --
faking a genuine passing `ELIGIBLE_FOR_PAPER_TRADING` chain end-to-end
would mean reimplementing large parts of Milestone 6's own acceptance
fixture with no real verification value; that full chain is exercised
once, for real, in the Milestone 7 real-acceptance-workflow integration
test, Section 33 -- exactly the precedent `test_eligibility.py` already
set for the eligibility chain's own steps 5-9).

Each tampering test mutates a COPY of a real, untampered ledger/manifest
and asserts the ONE issue code it targets appears, while leaving the
targeted check's identity legible (never asserting a fully clean report,
since eligibility never genuinely passes here)."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from quant_platform.backtesting.models import (
    CommissionModelKind,
    FinancingModelKind,
    PositionDirection,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.specs import CommissionSpec, FinancingSpec, SlippageSpec, SpreadSpec
from quant_platform.core.exceptions import ArtifactNotFoundError, PaperTradingVerificationError
from quant_platform.core.types import Timeframe
from quant_platform.paper_trading.clock import ReplayClock
from quant_platform.paper_trading.eligibility import EligibilityVerificationEnvironment
from quant_platform.paper_trading.events import create_bar_event
from quant_platform.paper_trading.identity import compute_content_id
from quant_platform.paper_trading.manifests import PaperSessionManifestStore
from quant_platform.paper_trading.models import (
    ClockMode,
    LedgerEntryKind,
    MarketEventMode,
    PaperSessionStage,
    PartialFillPolicyKind,
    SessionMode,
)
from quant_platform.paper_trading.persistence import PaperSessionEventStore, create_ledger_entry
from quant_platform.paper_trading.reconciliation import reconcile_session
from quant_platform.paper_trading.runner import (
    RunnerEnvironment,
    run_paper_trading_session,
    run_shadow_session,
)
from quant_platform.paper_trading.specs import (
    DEFAULT_EXECUTION_POLICY,
    DEFAULT_POSITION_POLICY,
    DEFAULT_SESSION_BOUNDARY_POLICY,
    FillPolicySpec,
    FinancingPolicySpec,
    InstrumentSpec,
    LatencyPolicySpec,
    LiquidityPolicySpec,
    OrderPolicySpec,
    PaperTradingSpec,
    RiskLimitsSpec,
    compute_paper_session_spec_id,
)
from quant_platform.paper_trading.strategy import StrategyContext, StrategyDecision, create_strategy_decision
from quant_platform.paper_trading.verification import (
    INDEPENDENCE_CLASSIFICATION,
    require_paper_session_verified,
    verify_paper_session,
)

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
_HEX_A = "a" * 64
_HEX_B = "b" * 64
_HEX_C = "c" * 64
_HEX_D = "d" * 64
_HEX_E = "e" * 64
_HEX_OTHER = "9" * 64


@pytest.fixture(autouse=True)
def _bypass_resume_eligibility_reverification(monkeypatch: pytest.MonkeyPatch) -> None:
    """This file's `_run_real_session` fixture seeds a manifest directly
    PAST eligibility with `eligibility_environment=None` -- release-audit
    finding, fixed elsewhere: `run_paper_trading_session` now mandatorily
    re-verifies eligibility on every call that did not itself just create
    the manifest. That fix is exercised for real, against a genuine
    eligibility chain, in `test_audit_eligibility_bypass.py`; here it is
    bypassed so this file's own (unrelated) verification-tampering
    assertions don't crash on a `None` environment."""
    monkeypatch.setattr("quant_platform.paper_trading.runner.require_paper_trading_eligibility", lambda *_args, **_kwargs: None)


def _instrument() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="X", base_currency=None, quote_currency="USD", contract_multiplier=1.0, tick_size=0.01, tick_value=None, quantity_step=0.01,
        minimum_quantity=0.01, maximum_quantity=None, price_precision=2, quantity_precision=2, margin_mode="cash", account_currency="USD",
        financing_convention="none", trading_timezone="UTC", session_calendar_identity="always_open",
    )


def _spec() -> PaperTradingSpec:
    risk_limits = RiskLimitsSpec(
        maximum_signed_position=None, maximum_absolute_position=None, maximum_gross_exposure=None, maximum_order_quantity=None,
        maximum_order_notional=None, maximum_turnover=None, maximum_daily_loss=None, maximum_drawdown_fraction=None, maximum_realized_loss=None,
        maximum_unrealized_loss=None, maximum_rejected_order_count=None, maximum_consecutive_execution_failures=None,
        maximum_stale_data_seconds=None, maximum_reconciliation_discrepancy=1e-6,
    )
    return PaperTradingSpec(
        schema_version=1, verified_robustness_id=_HEX_A, verified_promotion_decision_id=_HEX_B, strategy_candidate_identity=_HEX_C,
        model_artifact_identity=_HEX_D, calibration_artifact_identity=_HEX_E, feature_spec_identity=_HEX_A, instrument=_instrument(),
        price_precision=2, quantity_precision=2, session_mode=SessionMode.REPLAY_PAPER, market_event_mode=MarketEventMode.BAR,
        bar_interval=Timeframe.H1, clock_mode=ClockMode.REPLAY, starting_cash=100_000.0, starting_positions=(),
        order_policy=OrderPolicySpec(close_before_reverse=True, cooldown_bars=0, maximum_orders_per_event=5, maximum_order_rate_per_window=100, order_rate_window_events=1000),
        execution_policy=DEFAULT_EXECUTION_POLICY, fill_policy=FillPolicySpec(partial_fill_policy=PartialFillPolicyKind.FULL_FILL_ONLY),
        spread_policy=SpreadSpec(kind=SpreadModelKind.ZERO), slippage_policy=SlippageSpec(kind=SlippageModelKind.ZERO),
        commission_policy=CommissionSpec(kind=CommissionModelKind.ZERO),
        financing_policy=FinancingPolicySpec(long_financing=FinancingSpec(kind=FinancingModelKind.NONE), short_financing=FinancingSpec(kind=FinancingModelKind.NONE)),
        latency_policy=LatencyPolicySpec(decision_to_submit_ms=0, submit_to_accept_ms=0, accept_to_fill_eligible_ms=0),
        liquidity_policy=LiquidityPolicySpec(trust_disclosed_size=False), position_policy=DEFAULT_POSITION_POLICY, risk_limits=risk_limits,
        session_boundary_policy=DEFAULT_SESSION_BOUNDARY_POLICY, seed=0,
    )


def _bars(closes: list[float]) -> list:
    events = []
    for i, close in enumerate(closes):
        open_time = _T0 + timedelta(hours=i)
        events.append(create_bar_event(instrument="X", interval=Timeframe.H1, open_time=open_time, open=close, high=close + 0.5, low=close - 0.5, close=close, sequence=i + 1, source="test"))
    return events


@dataclasses.dataclass(frozen=True, slots=True)
class _FixedDirectionStrategy:
    direction: PositionDirection
    quantity: float

    @property
    def strategy_identity(self) -> str:
        return _HEX_A

    def decide(self, context: StrategyContext) -> StrategyDecision:
        return create_strategy_decision(
            strategy_identity=self.strategy_identity, event=context.event, decision_time=context.decision_time, target_direction=self.direction,
            target_quantity=self.quantity, confidence=0.9, uncertainty=0.05, abstain=False, reason_codes=("test",), stop_target_intent=None,
        )


class _AlwaysMissingRobustnessManifestStore:
    """Cheap fake matching `test_eligibility.py`'s own precedent: the
    eligibility chain always fails at its very first step
    (`promotion_decision_exists`) without ever needing genuine robustness/
    backtest evidence -- verification.py's OWN checks are what this file
    tests, not the eligibility chain's internals (already covered by
    `test_eligibility.py`)."""

    def load(self, robustness_id: str) -> object:
        raise ArtifactNotFoundError(f"no robustness manifest for {robustness_id!r}")


def _always_ineligible_environment() -> EligibilityVerificationEnvironment:
    return EligibilityVerificationEnvironment(
        robustness_manifest_store=_AlwaysMissingRobustnessManifestStore(),  # type: ignore[arg-type]
        artifact_store=None,  # type: ignore[arg-type]
        backtest_manifest_store=None,  # type: ignore[arg-type]
        backtest_event_store=None,  # type: ignore[arg-type]
        calibration_manifest_store=None,  # type: ignore[arg-type]
        experiment_manifest_store=None,  # type: ignore[arg-type]
        execution_manifest_store=None,  # type: ignore[arg-type]
        research_manifest_store=None,  # type: ignore[arg-type]
        research_dataset_store=None,  # type: ignore[arg-type]
        dataset_loader=None,  # type: ignore[arg-type]
    )


def _append_session_transition(event_store: PaperSessionEventStore, paper_session_id: str, *, from_stage: PaperSessionStage, to_stage: PaperSessionStage, event_time: datetime) -> None:
    """Mirrors `runner._transition_with_ledger_entry`'s own ledger-append
    half exactly (without needing the manifest-transition half, which the
    caller already performs) -- kept local to the test fixture rather than
    importing a module-private helper."""
    entry = create_ledger_entry(
        session_id=paper_session_id, sequence=event_store.next_sequence(paper_session_id), kind=LedgerEntryKind.SESSION_TRANSITION,
        payload={"from_stage": from_stage.value, "to_stage": to_stage.value}, event_time=event_time, previous_entry_hash=event_store.last_entry_hash(paper_session_id),
    )
    event_store.append(paper_session_id, entry)


def _run_real_session(tmp_path) -> tuple[list, PaperTradingSpec, object, PaperSessionManifestStore]:
    """Builds a real, fully-completed session -- including the
    CREATED -> ELIGIBILITY_VERIFIED `SESSION_TRANSITION` ledger entry that
    `create_paper_session` would normally append, added here by hand since
    this fixture (like `test_reconciliation.py`'s own) bypasses
    `create_paper_session` to avoid needing a genuine eligibility chain."""
    spec = _spec()
    manifest_store = PaperSessionManifestStore(tmp_path)
    event_store = PaperSessionEventStore(tmp_path)
    dummy_eligibility_environment: EligibilityVerificationEnvironment = None  # type: ignore[assignment]
    environment = RunnerEnvironment(manifest_store=manifest_store, event_store=event_store, eligibility_environment=dummy_eligibility_environment)
    paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    manifest_store.create(paper_session_id=paper_session_id, session_mode=spec.session_mode, spec_reference=None)
    now = _T0 - timedelta(minutes=1)
    _append_session_transition(event_store, paper_session_id, from_stage=PaperSessionStage.CREATED, to_stage=PaperSessionStage.ELIGIBILITY_VERIFIED, event_time=now)
    manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)
    strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)
    events = _bars([100.0, 103.0, 106.0, 109.0])
    run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
    ledger = event_store.read_events(paper_session_id)
    manifest = manifest_store.load(paper_session_id)
    return ledger, spec, manifest, manifest_store


def _run_real_shadow_session(tmp_path) -> tuple[list, PaperTradingSpec, object]:
    """Same construction as `_run_real_session` but for `SHADOW_
    OBSERVATION` mode, via `run_shadow_session` -- used to prove the
    session-mode/ledger-content verification check works in BOTH
    directions, not just paper-ledger-contains-shadow-entries."""
    spec = dataclasses.replace(_spec(), session_mode=SessionMode.SHADOW_OBSERVATION)
    manifest_store = PaperSessionManifestStore(tmp_path)
    event_store = PaperSessionEventStore(tmp_path)
    dummy_eligibility_environment: EligibilityVerificationEnvironment = None  # type: ignore[assignment]
    environment = RunnerEnvironment(manifest_store=manifest_store, event_store=event_store, eligibility_environment=dummy_eligibility_environment)
    paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    manifest_store.create(paper_session_id=paper_session_id, session_mode=spec.session_mode, spec_reference=None)
    now = _T0 - timedelta(minutes=1)
    _append_session_transition(event_store, paper_session_id, from_stage=PaperSessionStage.CREATED, to_stage=PaperSessionStage.ELIGIBILITY_VERIFIED, event_time=now)
    manifest_store.transition(paper_session_id, target_stage=PaperSessionStage.ELIGIBILITY_VERIFIED)
    strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)
    events = _bars([100.0, 103.0, 106.0, 109.0])
    run_shadow_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
    ledger = event_store.read_events(paper_session_id)
    manifest = manifest_store.load(paper_session_id)
    return ledger, spec, manifest


def _recompute_checksum(payload: dict) -> str:
    return compute_content_id("ledger_entry_payload", payload)


class TestVerifyRealSessionSurfacesOnlyEligibilityFailure:
    def test_untampered_session_fails_only_on_eligibility(self, tmp_path) -> None:
        ledger, spec, manifest, _ = _run_real_session(tmp_path)
        report = verify_paper_session(spec, manifest=manifest, ledger=ledger, eligibility_environment=_always_ineligible_environment())
        codes = {issue.code for issue in report.criticals} | {issue.code for issue in report.errors}
        assert codes == {"eligibility_not_verified"}
        assert not report.is_ready

    def test_reconciliation_report_matches_direct_call(self, tmp_path) -> None:
        ledger, spec, manifest, _ = _run_real_session(tmp_path)
        direct = reconcile_session(ledger, session_id=manifest.paper_session_id, instrument=spec.instrument, starting_cash=spec.starting_cash)
        assert direct.is_reconciled

    def test_json_round_trip_of_report_issues(self, tmp_path) -> None:
        from quant_platform.ml.models import ValidationReport

        ledger, spec, manifest, _ = _run_real_session(tmp_path)
        report = verify_paper_session(spec, manifest=manifest, ledger=ledger, eligibility_environment=_always_ineligible_environment())
        assert ValidationReport.from_json_dict(report.to_json_dict()) == report


class TestSpecIdentityMismatch:
    def test_wrong_manifest_paper_session_id_reports_critical(self, tmp_path) -> None:
        ledger, spec, manifest, _ = _run_real_session(tmp_path)
        tampered_manifest = dataclasses.replace(manifest, paper_session_id=_HEX_OTHER)
        report = verify_paper_session(spec, manifest=tampered_manifest, ledger=ledger, eligibility_environment=_always_ineligible_environment())
        codes = {issue.code for issue in report.criticals}
        assert "spec_identity_mismatch" in codes


class TestManifestTransitionTampering:
    def test_out_of_order_transition_reports_critical(self, tmp_path) -> None:
        ledger, spec, manifest, _ = _run_real_session(tmp_path)
        transition_indices = [i for i, e in enumerate(ledger) if e.kind is LedgerEntryKind.SESSION_TRANSITION]
        assert len(transition_indices) >= 2
        tampered = list(ledger)
        first, second = transition_indices[0], transition_indices[1]
        tampered[first], tampered[second] = tampered[second], tampered[first]
        report = verify_paper_session(spec, manifest=manifest, ledger=tampered, eligibility_environment=_always_ineligible_environment())
        codes = {issue.code for issue in report.criticals}
        assert "session_transition_out_of_order" in codes

    def test_illegal_transition_reports_critical(self, tmp_path) -> None:
        ledger, spec, manifest, _ = _run_real_session(tmp_path)
        transition_indices = [i for i, e in enumerate(ledger) if e.kind is LedgerEntryKind.SESSION_TRANSITION]
        first_entry = ledger[transition_indices[0]]
        assert first_entry.payload == {"from_stage": "created", "to_stage": "eligibility_verified"}
        illegal_payload = {"from_stage": "created", "to_stage": "completed"}
        tampered = list(ledger)
        tampered[transition_indices[0]] = dataclasses.replace(first_entry, payload=illegal_payload, checksum=_recompute_checksum(illegal_payload))
        report = verify_paper_session(spec, manifest=manifest, ledger=tampered, eligibility_environment=_always_ineligible_environment())
        codes = {issue.code for issue in report.criticals}
        assert "illegal_session_transition" in codes

    def test_manifest_stage_mismatch_reports_error(self, tmp_path) -> None:
        ledger, spec, manifest, _ = _run_real_session(tmp_path)
        tampered_manifest = dataclasses.replace(manifest, stage=PaperSessionStage.RUNNING)
        report = verify_paper_session(spec, manifest=tampered_manifest, ledger=ledger, eligibility_environment=_always_ineligible_environment())
        codes = {issue.code for issue in report.errors}
        assert "manifest_stage_mismatch" in codes


class TestLedgerChainIntegrityTampering:
    def test_broken_hash_chain_reports_critical(self, tmp_path) -> None:
        ledger, spec, manifest, _ = _run_real_session(tmp_path)
        tampered = list(ledger)
        mid = len(tampered) // 2
        tampered[mid] = dataclasses.replace(tampered[mid], previous_entry_hash=_HEX_OTHER)
        report = verify_paper_session(spec, manifest=manifest, ledger=tampered, eligibility_environment=_always_ineligible_environment())
        codes = {issue.code for issue in report.criticals}
        assert "ledger_chain_broken" in codes


class TestReconciliationFailureSurfacesThroughVerification:
    def test_tampered_fill_quantity_surfaces_as_critical(self, tmp_path) -> None:
        ledger, spec, manifest, _ = _run_real_session(tmp_path)
        tampered = list(ledger)
        for i, entry in enumerate(tampered):
            if entry.kind is LedgerEntryKind.FILL:
                tampered_payload = dict(entry.payload)
                tampered_payload["quantity"] = float(str(tampered_payload["quantity"])) + 100.0
                tampered_payload["gross_notional"] = float(str(tampered_payload["price"])) * float(str(tampered_payload["quantity"]))
                tampered[i] = dataclasses.replace(entry, payload=tampered_payload, checksum=_recompute_checksum(tampered_payload))
                break
        report = verify_paper_session(spec, manifest=manifest, ledger=tampered, eligibility_environment=_always_ineligible_environment())
        codes = {issue.code for issue in report.criticals}
        assert "reconciliation_position_quantity_equals_signed_cumulative_fills_failed" in codes
        assert "reconciliation_cash_movements_match_fills_and_costs_failed" in codes


class TestPersistedReconciliationStatusMismatch:
    def test_persisted_false_against_recomputed_true_surfaces_critical(self, tmp_path) -> None:
        ledger, spec, manifest, _ = _run_real_session(tmp_path)
        last = ledger[-1]
        synthetic_payload: dict[str, object] = {"is_reconciled": False}
        synthetic = create_ledger_entry(
            session_id=manifest.paper_session_id, sequence=last.sequence + 1, kind=LedgerEntryKind.RECONCILIATION_RESULT,
            payload=synthetic_payload, event_time=last.event_time, previous_entry_hash=last.entry_id,
        )
        report = verify_paper_session(spec, manifest=manifest, ledger=[*ledger, synthetic], eligibility_environment=_always_ineligible_environment())
        codes = {issue.code for issue in report.criticals}
        assert "reconciliation_status_mismatch" in codes

    def test_persisted_true_against_recomputed_true_does_not_surface_mismatch(self, tmp_path) -> None:
        ledger, spec, manifest, _ = _run_real_session(tmp_path)
        last = ledger[-1]
        synthetic_payload: dict[str, object] = {"is_reconciled": True}
        synthetic = create_ledger_entry(
            session_id=manifest.paper_session_id, sequence=last.sequence + 1, kind=LedgerEntryKind.RECONCILIATION_RESULT,
            payload=synthetic_payload, event_time=last.event_time, previous_entry_hash=last.entry_id,
        )
        report = verify_paper_session(spec, manifest=manifest, ledger=[*ledger, synthetic], eligibility_environment=_always_ineligible_environment())
        codes = {issue.code for issue in report.criticals} | {issue.code for issue in report.errors}
        assert "reconciliation_status_mismatch" not in codes


class TestSessionModeMismatch:
    """Release-audit Area 5: `verify_paper_session` used to never check
    that a ledger's own entry KINDS are consistent with its session's
    declared `session_mode` -- `reconcile_session` against a fill-free
    shadow ledger trivially "reconciles", so nothing previously caught a
    real fill/account-snapshot spliced into a shadow session's ledger, or
    a shadow observation spliced into a real paper session's ledger."""

    def test_shadow_entry_spliced_into_a_paper_session_ledger_reports_critical(self, tmp_path) -> None:
        ledger, spec, manifest, _ = _run_real_session(tmp_path)
        last = ledger[-1]
        forged_payload: dict[str, object] = {
            "observation_id": "1" * 64, "session_id": manifest.paper_session_id, "decision_id": "2" * 64, "instrument": "X",
            "hypothetical_order_id": None, "hypothetical_fill_id": None, "hypothetical_fill_price": None, "hypothetical_fill_quantity": None,
            "counterfactual_realized_pnl_delta": None, "event_identity": "3" * 64,
            "event_time": last.to_json_dict()["event_time"], "sequence": last.sequence + 1,
        }
        forged = create_ledger_entry(
            session_id=manifest.paper_session_id, sequence=last.sequence + 1, kind=LedgerEntryKind.SHADOW_OBSERVATION,
            payload=forged_payload, event_time=last.event_time, previous_entry_hash=last.entry_id,
        )
        report = verify_paper_session(spec, manifest=manifest, ledger=[*ledger, forged], eligibility_environment=_always_ineligible_environment())
        codes = {issue.code for issue in report.criticals}
        assert "paper_session_ledger_contains_shadow_entries" in codes

    def test_fill_spliced_into_a_shadow_session_ledger_reports_critical(self, tmp_path) -> None:
        ledger, spec, manifest = _run_real_shadow_session(tmp_path)
        assert not [e for e in ledger if e.kind is LedgerEntryKind.FILL], "fixture must start as a genuine, fill-free shadow ledger"
        last = ledger[-1]
        forged_payload: dict[str, object] = {
            "fill_id": "4" * 64, "order_id": "5" * 64, "session_id": manifest.paper_session_id, "instrument": "X", "side": "buy",
            "quantity": 1.0, "price": 100.0, "contract_multiplier": 1.0, "gross_notional": 100.0, "spread_cost": 0.0, "slippage_cost": 0.0,
            "commission_cost": 0.0, "financing_component": 0.0, "execution_time": last.to_json_dict()["event_time"], "source_market_event_identity": "6" * 64,
            "liquidity_assumption": "full_fill_only", "is_final": True,
        }
        forged = create_ledger_entry(
            session_id=manifest.paper_session_id, sequence=last.sequence + 1, kind=LedgerEntryKind.FILL,
            payload=forged_payload, event_time=last.event_time, previous_entry_hash=last.entry_id,
        )
        report = verify_paper_session(spec, manifest=manifest, ledger=[*ledger, forged], eligibility_environment=_always_ineligible_environment())
        codes = {issue.code for issue in report.criticals}
        assert "shadow_session_ledger_contains_real_account_entries" in codes

    def test_untampered_shadow_session_never_reports_mode_mismatch(self, tmp_path) -> None:
        ledger, spec, manifest = _run_real_shadow_session(tmp_path)
        report = verify_paper_session(spec, manifest=manifest, ledger=ledger, eligibility_environment=_always_ineligible_environment())
        codes = {issue.code for issue in report.criticals} | {issue.code for issue in report.errors}
        assert "shadow_session_ledger_contains_real_account_entries" not in codes
        assert "paper_session_ledger_contains_shadow_entries" not in codes


class TestNonTerminalStageWarning:
    def test_non_terminal_manifest_stage_reports_warning_not_critical(self, tmp_path) -> None:
        ledger, spec, manifest, _ = _run_real_session(tmp_path)
        truncated_ledger = [e for e in ledger if e.kind is not LedgerEntryKind.SESSION_TRANSITION or e.payload.get("to_stage") not in ("completed",)]
        tampered_manifest = dataclasses.replace(manifest, stage=PaperSessionStage.VERIFIED)
        report = verify_paper_session(spec, manifest=tampered_manifest, ledger=truncated_ledger, eligibility_environment=_always_ineligible_environment())
        warning_codes = {issue.code for issue in report.issues if issue.severity.value == "warning"}
        assert "session_not_terminal" in warning_codes


class TestRequirePaperSessionVerifiedFailsClosed:
    def test_raises_when_not_ready(self, tmp_path) -> None:
        ledger, spec, manifest, _ = _run_real_session(tmp_path)
        with pytest.raises(PaperTradingVerificationError, match="eligibility_not_verified"):
            require_paper_session_verified(spec, manifest=manifest, ledger=ledger, eligibility_environment=_always_ineligible_environment())


class TestIndependenceClassification:
    def test_states_partially_independent_explicitly(self) -> None:
        assert "PARTIALLY INDEPENDENT" in INDEPENDENCE_CLASSIFICATION
        assert "STRUCTURALLY INDEPENDENT" in INDEPENDENCE_CLASSIFICATION
        assert "SOURCE-RECONSTRUCTING" in INDEPENDENCE_CLASSIFICATION
        assert "ALGORITHMICALLY INDEPENDENT" in INDEPENDENCE_CLASSIFICATION
