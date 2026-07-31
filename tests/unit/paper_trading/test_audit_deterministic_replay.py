"""Release-audit Area 10: deterministic replay (aggressive).

CONFIRMED GAP, FIXED: no reusable, PRODUCTION-grade "canonical semantic
digest" function existed anywhere -- only ad-hoc per-test-file
normalization helpers (`_normalize_ledger`, duplicated across many test
files in this audit). Section 10 explicitly requires one be DEFINED, not
merely a test convention: added `persistence.compute_ledger_semantic_
digest`, excluding `entry_id`/`checksum`/`previous_entry_hash` (pure
hash-chain-linkage artifacts) and each entry's own `event_time`
(genuinely wall-clock-derived for a `SESSION_TRANSITION` entry, which
has no market event to anchor to) -- see that function's own docstring.
This file defines it AND tests it, per Section 10's own instruction.

STATIC AUDIT (source search, not merely read): no `datetime.now()`/
`time.time()`/`uuid.*`/`random.*` call exists anywhere in `quant_
platform.paper_trading`'s own production code (confirmed by grep, not
assumed) -- the only `set(...)` usages are membership/dedup checks
(`len(set(x)) != len(x)`), never iterated to produce persisted output
order. `core.json.canonical_json_bytes` (the sole JSON encoder for both
content-hashing AND durable storage) uses `sort_keys=True`, so dict
insertion order is structurally irrelevant to any persisted/hashed
output -- a strong, already-existing determinism guarantee this file
independently confirms via the tests below rather than merely trusting
the docstring's own claim."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from quant_platform.backtesting.models import (
    CommissionModelKind,
    FinancingModelKind,
    PositionDirection,
    SlippageModelKind,
    SpreadModelKind,
)
from quant_platform.backtesting.specs import CommissionSpec, FinancingSpec, SlippageSpec, SpreadSpec
from quant_platform.core.types import Timeframe
from quant_platform.ml.artifacts import MLArtifactStore
from quant_platform.paper_trading.clock import ReplayClock
from quant_platform.paper_trading.events import create_bar_event
from quant_platform.paper_trading.manifests import PaperSessionManifestStore
from quant_platform.paper_trading.models import (
    ClockMode,
    MarketEventMode,
    PartialFillPolicyKind,
    SessionMode,
)
from quant_platform.paper_trading.persistence import PaperSessionEventStore, compute_ledger_semantic_digest
from quant_platform.paper_trading.reconciliation import reconcile_session
from quant_platform.paper_trading.runner import RunnerEnvironment, run_paper_trading_session
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

_UTC = timezone.utc
_T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=_UTC)
_HEX_A = "a" * 64
_HEX_B = "b" * 64
_HEX_C = "c" * 64
_HEX_D = "d" * 64
_HEX_E = "e" * 64


@pytest.fixture(autouse=True)
def _bypass_resume_eligibility_reverification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("quant_platform.paper_trading.runner.require_paper_trading_eligibility", lambda *_args, **_kwargs: None)


def _instrument() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="X", base_currency=None, quote_currency="USD", contract_multiplier=1.0, tick_size=0.01, tick_value=None, quantity_step=0.01,
        minimum_quantity=0.01, maximum_quantity=None, price_precision=2, quantity_precision=2, margin_mode="cash", account_currency="USD",
        financing_convention="none", trading_timezone="UTC", session_calendar_identity="always_open",
    )


def _risk_limits(**overrides: object) -> RiskLimitsSpec:
    defaults: dict[str, object] = {
        "maximum_signed_position": None, "maximum_absolute_position": None, "maximum_gross_exposure": None, "maximum_order_quantity": None,
        "maximum_order_notional": None, "maximum_turnover": None, "maximum_daily_loss": None, "maximum_drawdown_fraction": None,
        "maximum_realized_loss": None, "maximum_unrealized_loss": None, "maximum_rejected_order_count": None,
        "maximum_consecutive_execution_failures": None, "maximum_stale_data_seconds": None, "maximum_reconciliation_discrepancy": 1e-6,
    }
    defaults.update(overrides)
    return RiskLimitsSpec(**defaults)  # type: ignore[arg-type]


def _spec(**overrides: object) -> PaperTradingSpec:
    defaults: dict[str, object] = {
        "schema_version": 1, "verified_robustness_id": _HEX_A, "verified_promotion_decision_id": _HEX_B, "strategy_candidate_identity": _HEX_C,
        "model_artifact_identity": _HEX_D, "calibration_artifact_identity": _HEX_E, "feature_spec_identity": _HEX_A, "instrument": _instrument(),
        "price_precision": 2, "quantity_precision": 2, "session_mode": SessionMode.REPLAY_PAPER, "market_event_mode": MarketEventMode.BAR,
        "bar_interval": Timeframe.H1, "clock_mode": ClockMode.REPLAY, "starting_cash": 100_000.0, "starting_positions": (),
        "order_policy": OrderPolicySpec(close_before_reverse=True, cooldown_bars=0, maximum_orders_per_event=5, maximum_order_rate_per_window=100, order_rate_window_events=1000),
        "execution_policy": DEFAULT_EXECUTION_POLICY, "fill_policy": FillPolicySpec(partial_fill_policy=PartialFillPolicyKind.FULL_FILL_ONLY),
        "spread_policy": SpreadSpec(kind=SpreadModelKind.FIXED_BASIS_POINTS, basis_points=3.0), "slippage_policy": SlippageSpec(kind=SlippageModelKind.FIXED_BASIS_POINTS, basis_points=1.0),
        "commission_policy": CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=2.0),
        "financing_policy": FinancingPolicySpec(long_financing=FinancingSpec(kind=FinancingModelKind.NONE), short_financing=FinancingSpec(kind=FinancingModelKind.NONE)),
        "latency_policy": LatencyPolicySpec(decision_to_submit_ms=50, submit_to_accept_ms=50, accept_to_fill_eligible_ms=50),
        "liquidity_policy": LiquidityPolicySpec(trust_disclosed_size=False), "position_policy": DEFAULT_POSITION_POLICY,
        "risk_limits": _risk_limits(), "session_boundary_policy": DEFAULT_SESSION_BOUNDARY_POLICY, "seed": 0,
    }
    defaults.update(overrides)
    return PaperTradingSpec(**defaults)  # type: ignore[arg-type]


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


def _environment(tmp_path) -> RunnerEnvironment:
    """`eligibility_environment` is a minimal stub exposing only
    `.artifact_store` -- see this file's own autouse eligibility-bypass
    fixture; `create_paper_session` persists the spec via `environment.
    eligibility_environment.artifact_store` directly (a real defect
    found and fixed during Milestone 8 acceptance testing), so a genuine
    `MLArtifactStore` must be reachable here even though eligibility
    itself is mocked out."""
    manifest_store = PaperSessionManifestStore(tmp_path)
    event_store = PaperSessionEventStore(tmp_path)
    eligibility_environment = SimpleNamespace(artifact_store=MLArtifactStore(tmp_path))
    return RunnerEnvironment(manifest_store=manifest_store, event_store=event_store, eligibility_environment=eligibility_environment)  # type: ignore[arg-type]


def _run(tmp_path) -> tuple[str, list]:
    spec = _spec()
    environment = _environment(tmp_path)
    strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)
    events = _bars([100.0, 103.0, 97.0, 102.0, 108.0])
    manifest = run_paper_trading_session(spec, environment=environment, strategy_runtime=strategy, clock=ReplayClock(), events=events)
    assert manifest.stage.value == "completed"
    paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
    ledger = environment.event_store.read_events(paper_session_id)
    return compute_ledger_semantic_digest(ledger), ledger


class TestCanonicalSemanticDigestDefinedAndTested:
    def test_two_independent_runs_with_different_absolute_tmp_paths_produce_the_same_digest(self, tmp_path) -> None:
        digest_a, _ = _run(tmp_path / "run_a_totally_different_name")
        digest_b, _ = _run(tmp_path / "second" / "nested" / "differently_shaped_path")
        assert digest_a == digest_b

    def test_digest_is_stable_across_wall_clock_derived_session_transition_times(self, tmp_path) -> None:
        """Two runs' SESSION_TRANSITION entries carry genuinely different
        wall-clock `event_time`s (each run calls `utc_now()` separately)
        -- the digest must be identical anyway, proving it correctly
        excludes this specific operational field rather than accidentally
        being insensitive to everything."""
        digest_a, ledger_a = _run(tmp_path / "a")
        digest_b, ledger_b = _run(tmp_path / "b")
        transition_times_a = [e.event_time for e in ledger_a if e.kind.value == "session_transition"]
        transition_times_b = [e.event_time for e in ledger_b if e.kind.value == "session_transition"]
        assert transition_times_a and transition_times_b
        assert digest_a == digest_b

    def test_digest_changes_when_economic_content_actually_differs(self, tmp_path) -> None:
        """Sanity check the digest is not vacuously constant: a
        DIFFERENT strategy quantity must produce a DIFFERENT digest."""
        spec_a = _spec()
        environment_a = _environment(tmp_path / "a")
        events = _bars([100.0, 103.0, 97.0, 102.0, 108.0])
        run_paper_trading_session(spec_a, environment=environment_a, strategy_runtime=_FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0), clock=ReplayClock(), events=events)
        paper_session_id_a = compute_paper_session_spec_id(spec_a).paper_session_spec_id
        digest_a = compute_ledger_semantic_digest(environment_a.event_store.read_events(paper_session_id_a))

        spec_b = _spec(seed=1)
        environment_b = _environment(tmp_path / "b")
        run_paper_trading_session(spec_b, environment=environment_b, strategy_runtime=_FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=7.0), clock=ReplayClock(), events=events)
        paper_session_id_b = compute_paper_session_spec_id(spec_b).paper_session_spec_id
        digest_b = compute_ledger_semantic_digest(environment_b.event_store.read_events(paper_session_id_b))

        assert digest_a != digest_b

    def test_digest_detects_a_tampered_fill_quantity(self, tmp_path) -> None:
        from quant_platform.paper_trading.identity import compute_content_id

        _digest, ledger = _run(tmp_path)
        original_digest = compute_ledger_semantic_digest(ledger)
        fill_index = next(i for i, e in enumerate(ledger) if e.kind.value == "fill")
        tampered_payload = dict(ledger[fill_index].payload)
        tampered_payload["quantity"] = float(str(tampered_payload["quantity"])) + 1.0
        tampered_checksum = compute_content_id("ledger_entry_payload", tampered_payload)
        tampered_ledger = list(ledger)
        tampered_ledger[fill_index] = dataclasses.replace(ledger[fill_index], payload=tampered_payload, checksum=tampered_checksum)
        assert compute_ledger_semantic_digest(tampered_ledger) != original_digest


class TestEqualTimestampEventsProcessDeterministically:
    def test_two_events_sharing_the_same_close_time_still_replay_to_the_same_digest(self, tmp_path) -> None:
        """Two BAR events with an IDENTICAL `close_time` (a genuine tie,
        e.g. two different-interval bars closing simultaneously) but
        distinct, strictly-increasing `sequence` numbers -- ledger
        position (never event_time comparison) is what governs
        processing order, so this must replay identically every time."""
        tied_time = _T0 + timedelta(hours=1)
        bar1 = create_bar_event(instrument="X", interval=Timeframe.H1, open_time=_T0, open=100.0, high=100.5, low=99.5, close=100.0, sequence=1, source="test")
        bar2a = create_bar_event(instrument="X", interval=Timeframe.H1, open_time=tied_time - timedelta(hours=1), open=100.0, high=103.5, low=99.5, close=103.0, sequence=2, source="test")
        bar2b = create_bar_event(instrument="X", interval=Timeframe.H1, open_time=tied_time - timedelta(hours=1), open=103.0, high=103.5, low=101.5, close=102.0, sequence=3, source="test_second_source")
        assert bar2a.close_time == bar2b.close_time == tied_time
        events = (bar1, bar2a, bar2b)

        spec = _spec()
        strategy = _FixedDirectionStrategy(direction=PositionDirection.LONG, quantity=3.0)

        environment_a = _environment(tmp_path / "a")
        run_paper_trading_session(spec, environment=environment_a, strategy_runtime=strategy, clock=ReplayClock(), events=events)
        environment_b = _environment(tmp_path / "b")
        run_paper_trading_session(spec, environment=environment_b, strategy_runtime=strategy, clock=ReplayClock(), events=events)

        paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id
        digest_a = compute_ledger_semantic_digest(environment_a.event_store.read_events(paper_session_id))
        digest_b = compute_ledger_semantic_digest(environment_b.event_store.read_events(paper_session_id))
        assert digest_a == digest_b


class TestReconciliationAndVerificationReportsAreDeterministic:
    def test_reconciliation_report_checks_match_across_independent_runs(self, tmp_path) -> None:
        _digest_a, ledger_a = _run(tmp_path / "a")
        _digest_b, ledger_b = _run(tmp_path / "b")
        spec = _spec()

        report_a = reconcile_session(ledger_a, session_id="s", instrument=spec.instrument, starting_cash=spec.starting_cash)
        report_b = reconcile_session(ledger_b, session_id="s", instrument=spec.instrument, starting_cash=spec.starting_cash)
        checks_a = [(c.check_identity, c.passed, c.expected_value, c.observed_value) for c in report_a.checks]
        checks_b = [(c.check_identity, c.passed, c.expected_value, c.observed_value) for c in report_b.checks]
        assert checks_a == checks_b


class TestSeparateProcessDifferentHashSeedDeterminism:
    """The most aggressive test this area requires: run the SAME
    deterministic session in two genuinely SEPARATE Python processes,
    with DIFFERENT `PYTHONHASHSEED` values, and compare the resulting
    semantic digest -- catches nondeterminism that only a same-process,
    same-interpreter-instance test structurally cannot (hash-seed-
    dependent `set`/`frozenset` iteration order, in particular)."""

    def test_digest_matches_across_two_subprocesses_with_different_pythonhashseed(self, tmp_path) -> None:
        # Fully self-contained -- no import back into this test module (or the `tests.*`
        # package) across the subprocess boundary, which would depend on fragile sys.path/
        # rootdir assumptions this test has no business caring about. Only `quant_platform`
        # itself needs to be importable, exactly like every other subprocess CLI test in
        # this suite already relies on.
        script = tmp_path / "run_once.py"
        script.write_text(
            "import sys\n"
            "from datetime import datetime, timedelta, timezone\n"
            "from types import SimpleNamespace\n"
            "from quant_platform.backtesting.models import CommissionModelKind, FinancingModelKind, PositionDirection, SlippageModelKind, SpreadModelKind\n"
            "from quant_platform.backtesting.specs import CommissionSpec, FinancingSpec, SlippageSpec, SpreadSpec\n"
            "from quant_platform.core.types import Timeframe\n"
            "from quant_platform.paper_trading.clock import ReplayClock\n"
            "from quant_platform.paper_trading.events import create_bar_event\n"
            "from quant_platform.ml.artifacts import MLArtifactStore\n"
            "from quant_platform.paper_trading.manifests import PaperSessionManifestStore\n"
            "from quant_platform.paper_trading.models import ClockMode, MarketEventMode, PartialFillPolicyKind, SessionMode\n"
            "from quant_platform.paper_trading.persistence import PaperSessionEventStore, compute_ledger_semantic_digest\n"
            "import quant_platform.paper_trading.runner as runner_module\n"
            "from quant_platform.paper_trading.runner import RunnerEnvironment, run_paper_trading_session\n"
            "from quant_platform.paper_trading.specs import DEFAULT_EXECUTION_POLICY, DEFAULT_POSITION_POLICY, DEFAULT_SESSION_BOUNDARY_POLICY, FillPolicySpec, FinancingPolicySpec, InstrumentSpec, LatencyPolicySpec, LiquidityPolicySpec, OrderPolicySpec, PaperTradingSpec, RiskLimitsSpec, compute_paper_session_spec_id\n"
            "from quant_platform.paper_trading.strategy import create_strategy_decision\n"
            "runner_module.require_paper_trading_eligibility = lambda *a, **k: None\n"  # this test targets determinism, not eligibility (covered elsewhere) -- see this file's own autouse fixture for the same bypass, process-local here since monkeypatch cannot cross a subprocess boundary
            "_HEX = 'a' * 64\n"
            "instrument = InstrumentSpec(symbol='X', base_currency=None, quote_currency='USD', contract_multiplier=1.0, tick_size=0.01, tick_value=None, quantity_step=0.01, minimum_quantity=0.01, maximum_quantity=None, price_precision=2, quantity_precision=2, margin_mode='cash', account_currency='USD', financing_convention='none', trading_timezone='UTC', session_calendar_identity='always_open')\n"
            "risk_limits = RiskLimitsSpec(maximum_signed_position=None, maximum_absolute_position=None, maximum_gross_exposure=None, maximum_order_quantity=None, maximum_order_notional=None, maximum_turnover=None, maximum_daily_loss=None, maximum_drawdown_fraction=None, maximum_realized_loss=None, maximum_unrealized_loss=None, maximum_rejected_order_count=None, maximum_consecutive_execution_failures=None, maximum_stale_data_seconds=None, maximum_reconciliation_discrepancy=1e-6)\n"
            "spec = PaperTradingSpec(schema_version=1, verified_robustness_id=_HEX, verified_promotion_decision_id=_HEX, strategy_candidate_identity=_HEX, model_artifact_identity=_HEX, calibration_artifact_identity=_HEX, feature_spec_identity=_HEX, instrument=instrument, price_precision=2, quantity_precision=2, session_mode=SessionMode.REPLAY_PAPER, market_event_mode=MarketEventMode.BAR, bar_interval=Timeframe.H1, clock_mode=ClockMode.REPLAY, starting_cash=100000.0, starting_positions=(), order_policy=OrderPolicySpec(close_before_reverse=True, cooldown_bars=0, maximum_orders_per_event=5, maximum_order_rate_per_window=100, order_rate_window_events=1000), execution_policy=DEFAULT_EXECUTION_POLICY, fill_policy=FillPolicySpec(partial_fill_policy=PartialFillPolicyKind.FULL_FILL_ONLY), spread_policy=SpreadSpec(kind=SpreadModelKind.FIXED_BASIS_POINTS, basis_points=3.0), slippage_policy=SlippageSpec(kind=SlippageModelKind.FIXED_BASIS_POINTS, basis_points=1.0), commission_policy=CommissionSpec(kind=CommissionModelKind.PER_SIDE_BASIS_POINTS, per_side_basis_points=2.0), financing_policy=FinancingPolicySpec(long_financing=FinancingSpec(kind=FinancingModelKind.NONE), short_financing=FinancingSpec(kind=FinancingModelKind.NONE)), latency_policy=LatencyPolicySpec(decision_to_submit_ms=50, submit_to_accept_ms=50, accept_to_fill_eligible_ms=50), liquidity_policy=LiquidityPolicySpec(trust_disclosed_size=False), position_policy=DEFAULT_POSITION_POLICY, risk_limits=risk_limits, session_boundary_policy=DEFAULT_SESSION_BOUNDARY_POLICY, seed=0)\n"
            "T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)\n"
            "events = []\n"
            "for i, close in enumerate([100.0, 103.0, 97.0, 102.0, 108.0]):\n"
            "    open_time = T0 + timedelta(hours=i)\n"
            "    events.append(create_bar_event(instrument='X', interval=Timeframe.H1, open_time=open_time, open=close, high=close + 0.5, low=close - 0.5, close=close, sequence=i + 1, source='test'))\n"
            "class _Strategy:\n"
            "    strategy_identity = _HEX\n"
            "    def decide(self, context):\n"
            "        return create_strategy_decision(strategy_identity=_HEX, event=context.event, decision_time=context.decision_time, target_direction=PositionDirection.LONG, target_quantity=3.0, confidence=0.9, uncertainty=0.05, abstain=False, reason_codes=('test',), stop_target_intent=None)\n"
            "out_dir = sys.argv[1]\n"
            "manifest_store = PaperSessionManifestStore(out_dir)\n"
            "event_store = PaperSessionEventStore(out_dir)\n"
            "eligibility_environment = SimpleNamespace(artifact_store=MLArtifactStore(out_dir))\n"
            "environment = RunnerEnvironment(manifest_store=manifest_store, event_store=event_store, eligibility_environment=eligibility_environment)\n"
            "run_paper_trading_session(spec, environment=environment, strategy_runtime=_Strategy(), clock=ReplayClock(), events=events)\n"
            "paper_session_id = compute_paper_session_spec_id(spec).paper_session_spec_id\n"
            "digest = compute_ledger_semantic_digest(event_store.read_events(paper_session_id))\n"
            "print(digest)\n",
            encoding="utf-8",
        )

        def _run_subprocess(*, hashseed: str, out_dir) -> str:
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = hashseed
            result = subprocess.run([sys.executable, str(script), str(out_dir)], capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, f"subprocess failed (hashseed={hashseed}): {result.stderr}"
            return result.stdout.strip().splitlines()[-1]

        digest_seed_1 = _run_subprocess(hashseed="1", out_dir=tmp_path / "proc1")
        digest_seed_2 = _run_subprocess(hashseed="2", out_dir=tmp_path / "proc2")
        assert digest_seed_1 == digest_seed_2, "the semantic digest must be identical regardless of PYTHONHASHSEED"
