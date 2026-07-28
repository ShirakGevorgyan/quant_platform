"""Champion/challenger candidate selection (Milestone 6, Section 12). A
SEPARATE, higher-level comparison across several already-COMPLETED,
per-candidate robustness analyses -- never a re-analysis of any one
candidate, and never a step that alters any individual candidate's own
manifest (see `robustness.models.RobustnessStage`'s own docstring for
this distinction).

`CandidateEvidence` bundles exactly the already-computed reports this
module reads; it never re-derives anything, and every field is either a
persisted report or a caller-supplied scalar the runner (Section 14)
already knows (`total_closed_trade_count`, `mean_turnover_notional_
ratio`, `strategy_complexity_score`).

DETERMINISM: gate/ranking metric names are drawn from a FIXED, CLOSED
vocabulary (`_GATE_MEASURERS` / `_METRIC_EXTRACTORS`) -- a `SelectionGate`
or `SelectionPolicy.ranking_metric_order` entry naming anything else
fails closed at construction (`SelectionError`), never silently ignored.

DEVIATION FROM SECTION 12'S LITERAL WORDING: "stressed Sharpe" is not
computable from this platform's already-persisted `StressReport`
(`stress.py`'s scenario results are scalar summary outcomes -- total
return, drawdown, trade count -- not per-bar series a Sharpe ratio needs;
re-deriving bar-level dispersion at selection time would mean
re-simulating a second time, defeating the point of reusing already-
computed evidence). This module substitutes `worst_stress_scenario_net_
return` -- the minimum `total_net_return` across every EVALUATED,
non-`zero_cost` declared stress scenario -- as the concrete, honestly-
documented stand-in for "stressed result." "Bootstrap lower bound of
benchmark-relative return" is fully supported: `CandidateEvidence.
bootstrap` is expected (by convention, not enforced) to be a
`BootstrapReport` computed over a `BENCHMARK_RELATIVE`-kind return
series (see `robustness.series`); this module reads its `total_return`
statistic's `lower_bound` regardless of which series kind was actually
bootstrapped, since `BootstrapReport` itself carries no such restriction.

NO CANDIDATE DISAPPEARS: every candidate passed to `compute_selection_
report` appears in `SelectionReport.candidate_eligibility`, whether
eligible or not, with every gate's measured value and pass/fail/skip
outcome recorded. `SelectionReport.ranking` lists only ELIGIBLE
candidates, in final ranked order."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from quant_platform.core.exceptions import SelectionError
from quant_platform.ml.fingerprints import fingerprint_json
from quant_platform.ml.persistence import (
    as_json_dict,
    as_json_list,
    format_utc_timestamp,
    require_schema_version,
    utc_now,
)
from quant_platform.robustness.bootstrap import BootstrapReport
from quant_platform.robustness.models import GateOutcomeKind, StressAxisKind
from quant_platform.robustness.sensitivity import SensitivityReport
from quant_platform.robustness.source import SourceVerificationReport
from quant_platform.robustness.stability import FoldStabilityReport
from quant_platform.robustness.stress import StressReport

SELECTION_REPORT_SCHEMA_VERSION = 1

_TIE_BREAK_PROCEDURE_DESCRIPTION = (
    "Eligible candidates are compared lexicographically across SelectionPolicy.ranking_metric_order, each metric's fixed, "
    "documented higher-is-better/lower-is-better direction applied. A candidate with an unmeasurable (None) value for a "
    "given metric sorts strictly worse than any candidate with a measured value on that same metric. If every ranked "
    "metric ties exactly across two or more candidates, the final tie-break is ascending lexicographic order of the "
    "candidate's own robustness_id -- a stable, fully deterministic fallback that never depends on input order."
)


# --------------------------------------------------------------------------
# Candidate evidence bundle
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    robustness_id: str
    source_backtest_id: str
    source_verification: SourceVerificationReport
    total_outer_folds: int
    total_closed_trade_count: int
    bootstrap: BootstrapReport
    fold_stability: FoldStabilityReport
    stress: StressReport
    sensitivity: SensitivityReport | None = None
    mean_turnover_notional_ratio: float | None = None
    strategy_complexity_score: float | None = None


def _bootstrap_lower_bound(report: BootstrapReport, statistic_name: str) -> float | None:
    for estimate in report.estimates:
        if estimate.statistic_name == statistic_name:
            return estimate.lower_bound
    return None


def _worst_stress_scenario_net_return(evidence: CandidateEvidence) -> float | None:
    values = [
        s.total_net_return for s in evidence.stress.scenario_results
        if s.status == "evaluated" and s.axis is not StressAxisKind.ZERO_COST and s.total_net_return is not None
    ]
    return min(values) if values else None


def _no_parameter_cliff_measured(evidence: CandidateEvidence) -> float | None:
    if evidence.sensitivity is None:
        return None
    evaluated_axes = [a for a in evidence.sensitivity.axis_results if not a.skipped and a.cliff_detected is not None]
    if not evaluated_axes:
        return None
    return 0.0 if any(a.cliff_detected for a in evaluated_axes) else 1.0


_GATE_MEASURERS: dict[str, Callable[[CandidateEvidence], float | None]] = {
    "source_backtest_verified": lambda c: 1.0 if c.source_verification.all_checks_passed else 0.0,
    "minimum_fold_count": lambda c: float(c.total_outer_folds),
    "minimum_trade_count": lambda c: float(c.total_closed_trade_count),
    "no_critical_verification_issue": lambda c: 1.0 if c.source_verification.verify_backtest_critical_count == 0 else 0.0,
    "maximum_drawdown_under_limit": lambda c: c.fold_stability.maximum_fold_drawdown,
    "minimum_profitable_fold_fraction": lambda c: c.fold_stability.profitable_fold_fraction,
    "bootstrap_lower_bound_above_threshold": lambda c: _bootstrap_lower_bound(c.bootstrap, "total_return"),
    "cost_stressed_result_above_threshold": _worst_stress_scenario_net_return,
    "no_extreme_fold_concentration": lambda c: 1.0 if not c.fold_stability.concentration.warning_codes else 0.0,
    "no_parameter_cliff": _no_parameter_cliff_measured,
}

_METRIC_EXTRACTORS: dict[str, Callable[[CandidateEvidence], float | None]] = {
    "bootstrap_lower_bound_return": lambda c: _bootstrap_lower_bound(c.bootstrap, "total_return"),
    "worst_fold_return": lambda c: c.fold_stability.worst_fold_return,
    "worst_stress_scenario_net_return": _worst_stress_scenario_net_return,
    "maximum_drawdown": lambda c: c.fold_stability.maximum_fold_drawdown,
    "turnover": lambda c: c.mean_turnover_notional_ratio,
    "strategy_complexity_score": lambda c: c.strategy_complexity_score,
}
_METRIC_DIRECTIONS: dict[str, str] = {
    "bootstrap_lower_bound_return": "higher_is_better",
    "worst_fold_return": "higher_is_better",
    "worst_stress_scenario_net_return": "higher_is_better",
    "maximum_drawdown": "lower_is_better",
    "turnover": "lower_is_better",
    "strategy_complexity_score": "lower_is_better",
}


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SelectionGate:
    name: str
    mandatory: bool
    minimum_value: float | None = None
    maximum_value: float | None = None

    def __post_init__(self) -> None:
        if self.name not in _GATE_MEASURERS:
            raise SelectionError(f"SelectionGate.name={self.name!r} is not a known gate -- must be one of {sorted(_GATE_MEASURERS)}")
        if self.minimum_value is None and self.maximum_value is None:
            raise SelectionError(f"SelectionGate[{self.name!r}]: at least one of minimum_value/maximum_value is required")
        if self.minimum_value is not None and self.maximum_value is not None and self.minimum_value > self.maximum_value:
            raise SelectionError(f"SelectionGate[{self.name!r}]: minimum_value ({self.minimum_value}) must be <= maximum_value ({self.maximum_value})")

    def to_json_dict(self) -> dict[str, object]:
        return {"name": self.name, "mandatory": self.mandatory, "minimum_value": self.minimum_value, "maximum_value": self.maximum_value}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SelectionGate:
        def _opt(key: str) -> float | None:
            v = raw.get(key)
            return None if v is None else float(str(v))

        return cls(name=str(raw["name"]), mandatory=bool(raw["mandatory"]), minimum_value=_opt("minimum_value"), maximum_value=_opt("maximum_value"))


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    gates: tuple[SelectionGate, ...]
    ranking_metric_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.gates:
            raise SelectionError("SelectionPolicy.gates must not be empty")
        if len({g.name for g in self.gates}) != len(self.gates):
            raise SelectionError("SelectionPolicy.gates names must be unique")
        if not self.ranking_metric_order:
            raise SelectionError("SelectionPolicy.ranking_metric_order must not be empty")
        if len(set(self.ranking_metric_order)) != len(self.ranking_metric_order):
            raise SelectionError("SelectionPolicy.ranking_metric_order must not contain duplicates")
        for name in self.ranking_metric_order:
            if name not in _METRIC_EXTRACTORS:
                raise SelectionError(f"SelectionPolicy.ranking_metric_order contains unknown metric {name!r} -- must be one of {sorted(_METRIC_EXTRACTORS)}")

    def to_json_dict(self) -> dict[str, object]:
        # `gates` MUST preserve declared order here -- see `robustness.specs.
        # RobustnessSpec.to_json_dict`'s identical note: `_evaluate_candidate_
        # eligibility` builds `CandidateEligibility.gate_evaluations` by
        # iterating `policy.gates` positionally, and this durable
        # representation must round-trip faithfully for any code that reloads
        # a persisted/decoded policy and recomputes from it. Canonicalizing
        # `gates` for IDENTITY purposes only belongs in `to_identity_payload`.
        # `ranking_metric_order` is DELIBERATELY NOT sorted (never touched by
        # either method): it is a prioritized lexicographic tie-break
        # sequence where position IS semantically significant (first entry is
        # the primary sort key).
        return {"gates": [g.to_json_dict() for g in self.gates], "ranking_metric_order": list(self.ranking_metric_order)}

    def to_identity_payload(self) -> dict[str, object]:
        """`gates` is semantically an unordered set (uniqueness enforced by
        name in `__post_init__`) -- declared order must not affect
        `policy_identity`, even though it MUST be preserved by `to_json_dict`
        (see there). `ranking_metric_order` stays unsorted here too."""
        return {"gates": [g.to_json_dict() for g in sorted(self.gates, key=lambda g: g.name)], "ranking_metric_order": list(self.ranking_metric_order)}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SelectionPolicy:
        return cls(
            gates=tuple(SelectionGate.from_json_dict(as_json_dict(g, field_name="gates[]")) for g in as_json_list(raw["gates"], field_name="gates")),
            ranking_metric_order=tuple(str(m) for m in as_json_list(raw["ranking_metric_order"], field_name="ranking_metric_order")),
        )


DEFAULT_SELECTION_GATES: tuple[SelectionGate, ...] = (
    SelectionGate(name="source_backtest_verified", mandatory=True, minimum_value=1.0),
    SelectionGate(name="minimum_fold_count", mandatory=True, minimum_value=3.0),
    SelectionGate(name="minimum_trade_count", mandatory=True, minimum_value=10.0),
    SelectionGate(name="no_critical_verification_issue", mandatory=True, minimum_value=1.0),
    SelectionGate(name="maximum_drawdown_under_limit", mandatory=True, maximum_value=0.5),
    SelectionGate(name="minimum_profitable_fold_fraction", mandatory=True, minimum_value=0.5),
    SelectionGate(name="bootstrap_lower_bound_above_threshold", mandatory=True, minimum_value=0.0),
    SelectionGate(name="cost_stressed_result_above_threshold", mandatory=True, minimum_value=-0.5),
    SelectionGate(name="no_extreme_fold_concentration", mandatory=True, minimum_value=1.0),
    SelectionGate(name="no_parameter_cliff", mandatory=False, minimum_value=1.0),
)
DEFAULT_SELECTION_POLICY = SelectionPolicy(
    gates=DEFAULT_SELECTION_GATES,
    ranking_metric_order=("bootstrap_lower_bound_return", "worst_fold_return", "worst_stress_scenario_net_return", "maximum_drawdown", "turnover", "strategy_complexity_score"),
)


# --------------------------------------------------------------------------
# Report types
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class GateEvaluation:
    gate_name: str
    mandatory: bool
    measured_value: float | None
    minimum_value: float | None
    maximum_value: float | None
    outcome: GateOutcomeKind
    reason: str | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "gate_name": self.gate_name, "mandatory": self.mandatory, "measured_value": self.measured_value, "minimum_value": self.minimum_value,
            "maximum_value": self.maximum_value, "outcome": self.outcome.value, "reason": self.reason,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> GateEvaluation:
        def _opt(key: str) -> float | None:
            v = raw.get(key)
            return None if v is None else float(str(v))

        return cls(
            gate_name=str(raw["gate_name"]), mandatory=bool(raw["mandatory"]), measured_value=_opt("measured_value"), minimum_value=_opt("minimum_value"),
            maximum_value=_opt("maximum_value"), outcome=GateOutcomeKind(raw["outcome"]), reason=(None if raw.get("reason") is None else str(raw["reason"])),
        )


@dataclass(frozen=True, slots=True)
class CandidateEligibility:
    robustness_id: str
    source_backtest_id: str
    eligible: bool
    gate_evaluations: tuple[GateEvaluation, ...]
    rejection_reasons: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "robustness_id": self.robustness_id, "source_backtest_id": self.source_backtest_id, "eligible": self.eligible,
            "gate_evaluations": [g.to_json_dict() for g in self.gate_evaluations], "rejection_reasons": list(self.rejection_reasons),
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> CandidateEligibility:
        return cls(
            robustness_id=str(raw["robustness_id"]), source_backtest_id=str(raw["source_backtest_id"]), eligible=bool(raw["eligible"]),
            gate_evaluations=tuple(GateEvaluation.from_json_dict(as_json_dict(g, field_name="gate_evaluations[]")) for g in as_json_list(raw.get("gate_evaluations") or [], field_name="gate_evaluations")),
            rejection_reasons=tuple(str(r) for r in as_json_list(raw.get("rejection_reasons") or [], field_name="rejection_reasons")),
        )


@dataclass(frozen=True, slots=True)
class RankingEntry:
    robustness_id: str
    metric_values: dict[str, float | None]

    def to_json_dict(self) -> dict[str, object]:
        return {"robustness_id": self.robustness_id, "metric_values": dict(self.metric_values)}

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> RankingEntry:
        raw_metrics = as_json_dict(raw.get("metric_values") or {}, field_name="metric_values")
        return cls(robustness_id=str(raw["robustness_id"]), metric_values={k: (None if v is None else float(str(v))) for k, v in raw_metrics.items()})


@dataclass(frozen=True, slots=True)
class SelectionReport:
    schema_version: int
    family_id: str | None
    policy_identity: str
    candidate_eligibility: tuple[CandidateEligibility, ...]
    ranking: tuple[RankingEntry, ...]
    tie_break_procedure: str
    selected_candidate_robustness_id: str | None
    selection_identity: str
    generated_at: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "family_id": self.family_id, "policy_identity": self.policy_identity,
            "candidate_eligibility": [c.to_json_dict() for c in self.candidate_eligibility], "ranking": [r.to_json_dict() for r in self.ranking],
            "tie_break_procedure": self.tie_break_procedure, "selected_candidate_robustness_id": self.selected_candidate_robustness_id,
            "selection_identity": self.selection_identity, "generated_at": self.generated_at,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, object]) -> SelectionReport:
        require_schema_version(raw, supported=SELECTION_REPORT_SCHEMA_VERSION, context="SelectionReport")
        return cls(
            schema_version=SELECTION_REPORT_SCHEMA_VERSION, family_id=(None if raw.get("family_id") is None else str(raw["family_id"])),
            policy_identity=str(raw["policy_identity"]),
            candidate_eligibility=tuple(
                CandidateEligibility.from_json_dict(as_json_dict(c, field_name="candidate_eligibility[]")) for c in as_json_list(raw.get("candidate_eligibility") or [], field_name="candidate_eligibility")
            ),
            ranking=tuple(RankingEntry.from_json_dict(as_json_dict(r, field_name="ranking[]")) for r in as_json_list(raw.get("ranking") or [], field_name="ranking")),
            tie_break_procedure=str(raw["tie_break_procedure"]),
            selected_candidate_robustness_id=(None if raw.get("selected_candidate_robustness_id") is None else str(raw["selected_candidate_robustness_id"])),
            selection_identity=str(raw["selection_identity"]), generated_at=str(raw["generated_at"]),
        )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def _evaluate_gate(gate: SelectionGate, evidence: CandidateEvidence) -> GateEvaluation:
    measured = _GATE_MEASURERS[gate.name](evidence)
    if measured is None:
        return GateEvaluation(
            gate_name=gate.name, mandatory=gate.mandatory, measured_value=None, minimum_value=gate.minimum_value, maximum_value=gate.maximum_value,
            outcome=GateOutcomeKind.SKIP, reason=f"gate={gate.name!r}: could not be measured for this candidate (missing or insufficient supporting evidence)",
        )
    passed = (gate.minimum_value is None or measured >= gate.minimum_value) and (gate.maximum_value is None or measured <= gate.maximum_value)
    reason = None if passed else (
        f"gate={gate.name!r}: measured_value={measured!r} violates declared bound(s) "
        f"minimum={gate.minimum_value!r} maximum={gate.maximum_value!r}"
    )
    return GateEvaluation(
        gate_name=gate.name, mandatory=gate.mandatory, measured_value=measured, minimum_value=gate.minimum_value, maximum_value=gate.maximum_value,
        outcome=(GateOutcomeKind.PASS if passed else GateOutcomeKind.FAIL), reason=reason,
    )


def _evaluate_candidate_eligibility(evidence: CandidateEvidence, policy: SelectionPolicy) -> CandidateEligibility:
    evaluations = tuple(_evaluate_gate(g, evidence) for g in policy.gates)
    rejection_reasons: list[str] = []
    eligible = True
    for gate, evaluation in zip(policy.gates, evaluations, strict=True):
        if evaluation.outcome is GateOutcomeKind.FAIL or (evaluation.outcome is GateOutcomeKind.SKIP and gate.mandatory):
            eligible = False
            rejection_reasons.append(evaluation.reason or f"gate={gate.name!r}: mandatory gate was skipped -- fails closed")
    return CandidateEligibility(
        robustness_id=evidence.robustness_id, source_backtest_id=evidence.source_backtest_id, eligible=eligible,
        gate_evaluations=evaluations, rejection_reasons=tuple(rejection_reasons),
    )


def _sort_key(evidence: CandidateEvidence, metric_names: tuple[str, ...]) -> tuple[float | str, ...]:
    parts: list[float | str] = []
    for name in metric_names:
        value = _METRIC_EXTRACTORS[name](evidence)
        if value is None:
            parts.extend([1.0, 0.0])
            continue
        signed = value if _METRIC_DIRECTIONS[name] == "higher_is_better" else -value
        parts.extend([0.0, -signed])
    parts.append(evidence.robustness_id)
    return tuple(parts)


def compute_selection_report(*, candidates: tuple[CandidateEvidence, ...], policy: SelectionPolicy = DEFAULT_SELECTION_POLICY, family_id: str | None = None) -> SelectionReport:
    """Section 12's entry point. Every candidate in `candidates` appears
    in the returned report's `candidate_eligibility` regardless of
    outcome; `ranking` contains only eligible candidates, in final ranked
    (best-first) order."""
    if not candidates:
        raise SelectionError("compute_selection_report: candidates must not be empty")
    if len({c.robustness_id for c in candidates}) != len(candidates):
        raise SelectionError("compute_selection_report: candidate robustness_ids must be unique")

    eligibility = tuple(_evaluate_candidate_eligibility(c, policy) for c in candidates)
    eligible_ids = {e.robustness_id for e in eligibility if e.eligible}
    ranked = sorted((c for c in candidates if c.robustness_id in eligible_ids), key=lambda c: _sort_key(c, policy.ranking_metric_order))
    ranking = tuple(
        RankingEntry(robustness_id=c.robustness_id, metric_values={name: _METRIC_EXTRACTORS[name](c) for name in policy.ranking_metric_order})
        for c in ranked
    )
    selected = ranked[0].robustness_id if ranked else None
    policy_identity = fingerprint_json(policy.to_identity_payload())
    selection_identity = fingerprint_json({
        "policy_identity": policy_identity, "candidate_ids": sorted(c.robustness_id for c in candidates), "selected_candidate_robustness_id": selected,
    })
    return SelectionReport(
        schema_version=SELECTION_REPORT_SCHEMA_VERSION, family_id=family_id, policy_identity=policy_identity, candidate_eligibility=eligibility,
        ranking=ranking, tie_break_procedure=_TIE_BREAK_PROCEDURE_DESCRIPTION, selected_candidate_robustness_id=selected,
        selection_identity=selection_identity, generated_at=format_utc_timestamp(utc_now()),
    )


__all__ = [
    "DEFAULT_SELECTION_GATES",
    "DEFAULT_SELECTION_POLICY",
    "SELECTION_REPORT_SCHEMA_VERSION",
    "CandidateEligibility",
    "CandidateEvidence",
    "GateEvaluation",
    "RankingEntry",
    "SelectionGate",
    "SelectionPolicy",
    "SelectionReport",
    "compute_selection_report",
]
