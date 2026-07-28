"""Paper-trading promotion gates (Milestone 6, Section 13). Evaluates
ONE already-analyzed candidate (the same single `RobustnessSpec` run
that produced its own `SourceVerificationReport`/`BootstrapReport`/
`DownsideAnalysisReport`/`FoldStabilityReport`/`StressReport` and,
optionally, its own `SensitivityReport`/`RegimeReport`) against a
`PromotionPolicySpec` (`robustness.specs`) and produces exactly ONE
`PromotionDecisionKind`:

- `REJECTED`: at least one MANDATORY gate was measured and FAILED
  outright -- definitive negative evidence.
- `MANUAL_REVIEW_REQUIRED`: no mandatory gate failed outright, but at
  least one MANDATORY gate could not be measured at all (`GateOutcomeKind
  .SKIP` -- missing or insufficient supporting evidence). Per Section
  13's own instruction, a skipped mandatory gate fails closed: this is
  NEVER treated as a pass, but it is also distinguished from a definitive
  failure, since a human reviewer may be able to supply the missing
  evidence.
- `RESEARCH_ONLY`: every mandatory gate passed, but at least one
  NON-mandatory (advisory) gate was measured and failed -- clears every
  hard bar, but a real, non-blocking weakness was flagged.
- `ELIGIBLE_FOR_PAPER_TRADING`: every mandatory gate passed, and every
  non-mandatory gate that could be measured also passed (an unmeasured
  non-mandatory gate does not block this outcome, unlike an unmeasured
  mandatory one).

This precedence (REJECTED > MANUAL_REVIEW_REQUIRED > RESEARCH_ONLY >
ELIGIBLE_FOR_PAPER_TRADING) is this platform's own defined rule, not an
external standard -- documented here, not merely assumed.

`ELIGIBLE_FOR_LIVE_TRADING` is not a member of `PromotionDecisionKind`
(see `robustness.models`) and can never be produced by this module.
`PromotionDecision.disclaimer` is a fixed, structurally-required field on
every decision -- it is not a comment or a UI-layer afterthought; there
is no way to construct a `PromotionDecision` without it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from quant_platform.core.exceptions import PromotionError
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)
from quant_platform.robustness.bootstrap import BootstrapReport, DownsideAnalysisReport
from quant_platform.robustness.models import GateOutcomeKind, PromotionDecisionKind, StressAxisKind
from quant_platform.robustness.regimes import RegimeReport
from quant_platform.robustness.selection import GateEvaluation
from quant_platform.robustness.sensitivity import SensitivityReport
from quant_platform.robustness.source import SourceVerificationReport
from quant_platform.robustness.specs import DEFAULT_PROMOTION_GATES, PromotionGateSpec, PromotionPolicySpec
from quant_platform.robustness.stability import FoldStabilityReport
from quant_platform.robustness.stress import StressReport

PROMOTION_DECISION_SCHEMA_VERSION = 1

DISCLAIMER = (
    "This is a resampling-based, historical-evidence promotion decision computed from the declared "
    "RobustnessSpec/PromotionPolicySpec. It is NOT financial advice, NOT a guarantee of future profitability, and "
    "ELIGIBLE_FOR_PAPER_TRADING is NOT approval for live trading with real capital -- it means the declared gates "
    "were cleared against this platform's own historical, synthetic-or-real backtest evidence, nothing more."
)

DEFAULT_PROMOTION_POLICY = PromotionPolicySpec(gates=DEFAULT_PROMOTION_GATES)


@dataclass(frozen=True, slots=True)
class PromotionEvidence:
    robustness_id: str
    source_backtest_id: str
    source_verification: SourceVerificationReport
    total_outer_folds: int
    observation_count: int
    effective_sample_count: float
    total_closed_trade_count: int
    bootstrap: BootstrapReport
    downside: DownsideAnalysisReport
    fold_stability: FoldStabilityReport
    stress: StressReport
    sensitivity: SensitivityReport | None = None
    regime: RegimeReport | None = None


def _bootstrap_lower_bound(report: BootstrapReport, statistic_name: str) -> float | None:
    for estimate in report.estimates:
        if estimate.statistic_name == statistic_name:
            return estimate.lower_bound
    return None


def _worst_stress_scenario_net_return(evidence: PromotionEvidence, *, axis_filter: StressAxisKind | None) -> float | None:
    values = [
        s.total_net_return for s in evidence.stress.scenario_results
        if s.status == "evaluated" and s.axis is not StressAxisKind.ZERO_COST and s.total_net_return is not None
        and (axis_filter is None or s.axis is axis_filter)
    ]
    return min(values) if values else None


def _no_parameter_cliff_measured(evidence: PromotionEvidence) -> float | None:
    if evidence.sensitivity is None:
        return None
    evaluated_axes = [a for a in evidence.sensitivity.axis_results if not a.skipped and a.cliff_detected is not None]
    if not evaluated_axes:
        return None
    return 0.0 if any(a.cliff_detected for a in evaluated_axes) else 1.0


def _regime_coverage(evidence: PromotionEvidence) -> float | None:
    if evidence.regime is None:
        return None
    total = 0
    covered = 0
    for dimension in evidence.regime.dimension_results:
        if dimension.unavailable:
            continue
        for bucket in dimension.buckets:
            total += 1
            covered += 0 if bucket.skipped else 1
    return (covered / total) if total > 0 else None


_GATE_MEASURERS: dict[str, Callable[[PromotionEvidence], float | None]] = {
    "verified_source_backtest": lambda e: 1.0 if e.source_verification.all_checks_passed else 0.0,
    "minimum_outer_folds": lambda e: float(e.total_outer_folds),
    "minimum_observations": lambda e: float(e.observation_count),
    "minimum_trades": lambda e: float(e.total_closed_trade_count),
    "minimum_effective_sample_size": lambda e: e.effective_sample_count,
    "profitable_fold_fraction": lambda e: e.fold_stability.profitable_fold_fraction,
    "worst_fold_return": lambda e: e.fold_stability.worst_fold_return,
    "maximum_drawdown": lambda e: e.fold_stability.maximum_fold_drawdown,
    "bootstrap_lower_bound_total_net_return": lambda e: _bootstrap_lower_bound(e.bootstrap, "total_return"),
    "probability_of_loss": lambda e: e.downside.probability_total_net_return_non_positive,
    "benchmark_outperformance": lambda e: e.fold_stability.benchmark_outperformance_fraction,
    "stressed_cost_total_net_return": lambda e: _worst_stress_scenario_net_return(e, axis_filter=None),
    "latency_stress_total_net_return": lambda e: _worst_stress_scenario_net_return(e, axis_filter=StressAxisKind.ADDITIONAL_LATENCY_BARS),
    "regime_coverage": _regime_coverage,
    "no_extreme_fold_concentration": lambda e: 1.0 if not e.fold_stability.concentration.warning_codes else 0.0,
    "no_parameter_cliff": _no_parameter_cliff_measured,
    "no_critical_verification_findings": lambda e: 1.0 if e.source_verification.verify_backtest_critical_count == 0 else 0.0,
}
"""Every gate name Section 13 lists as an example is supported here, even
though `specs.DEFAULT_PROMOTION_GATES` only enables 14 of these 17 by
default -- `minimum_observations`, `benchmark_outperformance`, and
`regime_coverage` are opt-in via a custom `PromotionPolicySpec`."""


def _evaluate_gate(gate: PromotionGateSpec, evidence: PromotionEvidence) -> GateEvaluation:
    if gate.name not in _GATE_MEASURERS:
        raise PromotionError(f"_evaluate_gate: PromotionGateSpec.name={gate.name!r} is not a gate this module knows how to measure -- must be one of {sorted(_GATE_MEASURERS)}")
    measured = _GATE_MEASURERS[gate.name](evidence)
    if measured is None:
        return GateEvaluation(
            gate_name=gate.name, mandatory=gate.mandatory, measured_value=None, minimum_value=gate.minimum_value, maximum_value=gate.maximum_value,
            outcome=GateOutcomeKind.SKIP, reason=f"gate={gate.name!r}: could not be measured (missing or insufficient supporting evidence)",
        )
    passed = (gate.minimum_value is None or measured >= gate.minimum_value) and (gate.maximum_value is None or measured <= gate.maximum_value)
    reason = None if passed else (
        f"gate={gate.name!r}: measured_value={measured!r} violates declared bound(s) minimum={gate.minimum_value!r} maximum={gate.maximum_value!r}"
    )
    return GateEvaluation(
        gate_name=gate.name, mandatory=gate.mandatory, measured_value=measured, minimum_value=gate.minimum_value, maximum_value=gate.maximum_value,
        outcome=(GateOutcomeKind.PASS if passed else GateOutcomeKind.FAIL), reason=reason,
    )


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    schema_version: int
    robustness_id: str
    source_backtest_id: str
    decision: PromotionDecisionKind
    decision_reason: str
    gate_evaluations: tuple[GateEvaluation, ...]
    disclaimer: str
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "robustness_id": self.robustness_id, "source_backtest_id": self.source_backtest_id,
            "decision": self.decision.value, "decision_reason": self.decision_reason, "gate_evaluations": [g.to_json_dict() for g in self.gate_evaluations],
            "disclaimer": self.disclaimer, "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> PromotionDecision:
        require_schema_version(raw, supported=PROMOTION_DECISION_SCHEMA_VERSION, context="PromotionDecision")
        return cls(
            schema_version=PROMOTION_DECISION_SCHEMA_VERSION, robustness_id=str(raw["robustness_id"]), source_backtest_id=str(raw["source_backtest_id"]),
            decision=PromotionDecisionKind(raw["decision"]), decision_reason=str(raw["decision_reason"]),
            gate_evaluations=tuple(GateEvaluation.from_json_dict(as_json_dict(g, field_name="gate_evaluations[]")) for g in as_json_list(raw.get("gate_evaluations") or [], field_name="gate_evaluations")),
            disclaimer=str(raw["disclaimer"]), generated_at=str(raw["generated_at"]),
        )


def evaluate_promotion(*, evidence: PromotionEvidence, policy: PromotionPolicySpec = DEFAULT_PROMOTION_POLICY) -> PromotionDecision:
    """Section 13's entry point. Every declared gate in `policy.gates` is
    evaluated and persisted, regardless of outcome -- exactly like
    `selection.compute_selection_report`'s eligibility gates, nothing
    ever silently dropped."""
    evaluations = tuple(_evaluate_gate(g, evidence) for g in policy.gates)
    mandatory_failed = [e for e in evaluations if e.mandatory and e.outcome is GateOutcomeKind.FAIL]
    mandatory_skipped = [e for e in evaluations if e.mandatory and e.outcome is GateOutcomeKind.SKIP]
    advisory_failed = [e for e in evaluations if not e.mandatory and e.outcome is GateOutcomeKind.FAIL]

    if mandatory_failed:
        decision = PromotionDecisionKind.REJECTED
        names = ", ".join(sorted(e.gate_name for e in mandatory_failed))
        decision_reason = f"REJECTED: {len(mandatory_failed)} mandatory gate(s) failed outright: {names}"
    elif mandatory_skipped:
        decision = PromotionDecisionKind.MANUAL_REVIEW_REQUIRED
        names = ", ".join(sorted(e.gate_name for e in mandatory_skipped))
        decision_reason = f"MANUAL_REVIEW_REQUIRED: {len(mandatory_skipped)} mandatory gate(s) could not be measured (fail-closed, not treated as a pass): {names}"
    elif advisory_failed:
        decision = PromotionDecisionKind.RESEARCH_ONLY
        names = ", ".join(sorted(e.gate_name for e in advisory_failed))
        decision_reason = f"RESEARCH_ONLY: every mandatory gate passed, but {len(advisory_failed)} non-mandatory (advisory) gate(s) failed: {names}"
    else:
        decision = PromotionDecisionKind.ELIGIBLE_FOR_PAPER_TRADING
        decision_reason = "ELIGIBLE_FOR_PAPER_TRADING: every mandatory gate passed, and no measured non-mandatory gate failed."

    return PromotionDecision(
        schema_version=PROMOTION_DECISION_SCHEMA_VERSION, robustness_id=evidence.robustness_id, source_backtest_id=evidence.source_backtest_id,
        decision=decision, decision_reason=decision_reason, gate_evaluations=evaluations, disclaimer=DISCLAIMER, generated_at=format_utc_timestamp(utc_now()),
    )


__all__ = [
    "DEFAULT_PROMOTION_POLICY",
    "DISCLAIMER",
    "PROMOTION_DECISION_SCHEMA_VERSION",
    "PromotionDecision",
    "PromotionEvidence",
    "evaluate_promotion",
]
