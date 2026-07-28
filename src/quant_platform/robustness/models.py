"""Shared vocabulary for `quant_platform.robustness` (Milestone 6) --
stage machine, enums, and small cross-cutting value types every other
module in this package imports. Mirrors `backtesting.models`'s role one
layer up: a single place every enum/transition table lives, so no two
modules independently redefine the same closed set of names.

WHAT THIS PACKAGE DOES AND DOES NOT DO
--------------------------------------------------------------------------
`robustness` NEVER refits a model, NEVER re-simulates a fill, and NEVER
connects to a broker, MetaTrader 5, or live execution. It consumes an
ALREADY-VERIFIED, ALREADY-COMPLETED `backtesting` run's own persisted
artifacts and asks a narrower, harder question: given the evidence this
one backtest actually produced, how much of its apparent performance
survives resampling uncertainty, multiple-testing correction, fold-level
instability, cost/latency stress, and regime concentration -- and does
what survives clear a configurable, fail-closed bar for PAPER trading
(never live trading, never `ELIGIBLE_FOR_LIVE_TRADING`)."""

from __future__ import annotations

from enum import Enum


class RobustnessStage(Enum):
    """One robustness run analyzes ONE `source_backtest_id` end to end --
    unlike `BacktestStage`'s outer-fold LOOP, there is no per-candidate
    iteration here (each candidate backtest gets its OWN `RobustnessSpec`/
    manifest); `SELECTION_COMPLETED`/`PROMOTION_EVALUATED` are this
    candidate's OWN standalone eligibility/promotion evaluation -- a
    SEPARATE, higher-level `StrategyFamily` comparison (`robustness.
    selection`) can additionally rank several already-COMPLETED
    candidates against each other without altering any one of their own
    manifests."""

    CREATED = "created"
    SOURCE_VERIFIED = "source_verified"
    SERIES_BUILT = "series_built"
    BOOTSTRAP_COMPLETED = "bootstrap_completed"
    STABILITY_COMPLETED = "stability_completed"
    STRESS_COMPLETED = "stress_completed"
    REGIMES_COMPLETED = "regimes_completed"
    SELECTION_COMPLETED = "selection_completed"
    PROMOTION_EVALUATED = "promotion_evaluated"
    VERIFIED = "verified"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_ROBUSTNESS_STAGES: frozenset[RobustnessStage] = frozenset({RobustnessStage.COMPLETED, RobustnessStage.FAILED})

_ROBUSTNESS_STAGE_ORDER: tuple[RobustnessStage, ...] = (
    RobustnessStage.CREATED, RobustnessStage.SOURCE_VERIFIED, RobustnessStage.SERIES_BUILT,
    RobustnessStage.BOOTSTRAP_COMPLETED, RobustnessStage.STABILITY_COMPLETED, RobustnessStage.STRESS_COMPLETED,
    RobustnessStage.REGIMES_COMPLETED, RobustnessStage.SELECTION_COMPLETED, RobustnessStage.PROMOTION_EVALUATED,
    RobustnessStage.VERIFIED, RobustnessStage.COMPLETED,
)
_LEGAL_ROBUSTNESS_TRANSITIONS: dict[RobustnessStage, frozenset[RobustnessStage]] = {
    stage: frozenset({nxt, RobustnessStage.FAILED}) if nxt is not None else frozenset({RobustnessStage.FAILED})
    for stage, nxt in zip(_ROBUSTNESS_STAGE_ORDER, (*_ROBUSTNESS_STAGE_ORDER[1:], None), strict=True)
}
_LEGAL_ROBUSTNESS_TRANSITIONS[RobustnessStage.COMPLETED] = frozenset()
_LEGAL_ROBUSTNESS_TRANSITIONS[RobustnessStage.FAILED] = frozenset()

_ROBUSTNESS_STAGE_INDEX: dict[RobustnessStage, int] = {stage: i for i, stage in enumerate(_ROBUSTNESS_STAGE_ORDER)}
"""Closure-audit finding (Section 9): `RobustnessRunner._run_locked`
re-verifies a resumed manifest's claimed stage via `resume.resolve_
resume_start_stage` and, if that independent re-verification finds LESS
progress than the manifest claims (an artifact corrupted, truncated, or
lost AFTER its stage was genuinely reached -- exactly the "process died
mid-write" scenario this whole resume mechanism exists to survive),
attempts to REWIND `RobustnessManifest.stage` back down to the last
verifiably-good stage before recomputing forward from there. Before this
fix, `_LEGAL_ROBUSTNESS_TRANSITIONS` allowed ONLY a forward-by-exactly-
one-stage transition (or a transition to `FAILED`) -- ANY backward
rewind, even by a single stage, was rejected as an "illegal robustness
stage transition", so the exact recovery path this module's own
docstring promises ("never trust the manifest's claim alone... demotes
the resume start point") crashed instead of recovering, for any
corruption that wasn't already exactly at the manifest's current stage.
The manifest was then left permanently stuck: `can_resume` kept
reporting it as resumable, but every subsequent resume attempt hit the
identical illegal-transition error. Reproduced directly: a manifest
claiming `BOOTSTRAP_COMPLETED` with its `SERIES_BUILT`-stage artifact
corrupted independently re-verifies to `SOURCE_VERIFIED`, and attempting
`transition(new_stage=SOURCE_VERIFIED)` from `BOOTSTRAP_COMPLETED` raised
`RobustnessStateError` every time."""


def is_legal_robustness_transition(current: RobustnessStage, target: RobustnessStage) -> bool:
    if target in _LEGAL_ROBUSTNESS_TRANSITIONS[current]:
        return True
    # A REWIND to any strictly-earlier non-terminal stage is legal from any
    # non-terminal current stage -- this is the one operation `resume.py`'s
    # "demote to the last independently-verified stage" recovery needs, and
    # it is never reachable from the normal forward-only pipeline itself
    # (`_execute_pipeline` only ever transitions forward by exactly one
    # stage), so widening legality here cannot change ordinary forward-run
    # behavior.
    if current in TERMINAL_ROBUSTNESS_STAGES or target in TERMINAL_ROBUSTNESS_STAGES:
        return False
    return _ROBUSTNESS_STAGE_INDEX[target] < _ROBUSTNESS_STAGE_INDEX[current]


def is_terminal_robustness_stage(stage: RobustnessStage) -> bool:
    return stage in TERMINAL_ROBUSTNESS_STAGES


class ReturnSeriesKind(Enum):
    """Section 4: every distinct analysis series this package supports --
    NEVER silently mixed with one another (a bar-return statistic must
    never be computed over a trade-return series or vice versa)."""

    BAR_NET = "bar_net"
    BAR_GROSS = "bar_gross"
    TRADE_NET = "trade_net"
    TRADE_GROSS = "trade_gross"
    STITCHED_BAR_NET = "stitched_bar_net"
    PER_FOLD_BAR_NET = "per_fold_bar_net"
    BENCHMARK_RELATIVE = "benchmark_relative"


BAR_SAMPLED_SERIES_KINDS: frozenset[ReturnSeriesKind] = frozenset({
    ReturnSeriesKind.BAR_NET, ReturnSeriesKind.BAR_GROSS, ReturnSeriesKind.STITCHED_BAR_NET,
    ReturnSeriesKind.PER_FOLD_BAR_NET, ReturnSeriesKind.BENCHMARK_RELATIVE,
})
TRADE_SAMPLED_SERIES_KINDS: frozenset[ReturnSeriesKind] = frozenset({ReturnSeriesKind.TRADE_NET, ReturnSeriesKind.TRADE_GROSS})


class BootstrapMethodKind(Enum):
    """Section 5. `IID` is explicitly documented as weak for dependent
    (autocorrelated) financial time series -- it is implemented for
    comparison/completeness, never as this package's default or
    recommended method."""

    IID = "iid"
    MOVING_BLOCK = "moving_block"
    STATIONARY = "stationary"
    FOLD_LEVEL = "fold_level"


class MultipleTestingCorrectionKind(Enum):
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    BENJAMINI_HOCHBERG = "benjamini_hochberg"


class RegimeDimensionKind(Enum):
    """Section 11. Every dimension here is computable from information
    available strictly at-or-before the bar being classified (trailing
    windows only) -- see `regimes.py`'s own module docstring for the
    leakage-safety argument specific to each kind."""

    VOLATILITY_QUANTILE = "volatility_quantile"
    TREND_DIRECTION = "trend_direction"
    LIQUIDITY_QUANTILE = "liquidity_quantile"
    SPREAD_QUANTILE = "spread_quantile"
    SESSION = "session"
    DAY_OF_WEEK = "day_of_week"
    HOUR_OF_DAY = "hour_of_day"
    PRICE_REGIME = "price_regime"


class StressAxisKind(Enum):
    """Section 10."""

    ZERO_COST = "zero_cost"
    BASE_COST = "base_cost"
    SPREAD_MULTIPLIER = "spread_multiplier"
    SLIPPAGE_MULTIPLIER = "slippage_multiplier"
    COMMISSION_MULTIPLIER = "commission_multiplier"
    FINANCING_MULTIPLIER = "financing_multiplier"
    ADDITIONAL_LATENCY_BARS = "additional_latency_bars"
    COMBINED_ADVERSE = "combined_adverse"
    NAMED_PROFILE = "named_profile"


class PerturbationAxisKind(Enum):
    """Section 9 -- declared, bounded perturbations around the ALREADY-
    SELECTED operating point. Never a search: `sensitivity.py` never
    picks the best-performing perturbed value and never feeds one back
    into the analyzed spec."""

    PROBABILITY_THRESHOLD = "probability_threshold"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    UNCERTAINTY_THRESHOLD = "uncertainty_threshold"
    ABSTENTION_THRESHOLD = "abstention_threshold"
    HOLDING_PERIOD_BARS = "holding_period_bars"
    ENTRY_DELAY_BARS = "entry_delay_bars"
    SPREAD_BASIS_POINTS = "spread_basis_points"
    COMMISSION_BASIS_POINTS = "commission_basis_points"
    SLIPPAGE_BASIS_POINTS = "slippage_basis_points"
    FINANCING_BASIS_POINTS = "financing_basis_points"
    EXPOSURE_CAP = "exposure_cap"


class GateOutcomeKind(Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


class PromotionDecisionKind(Enum):
    """Section 13. `ELIGIBLE_FOR_LIVE_TRADING` is deliberately NOT a
    member of this enum -- it must never be constructible in this
    milestone, not merely "never selected by policy"."""

    REJECTED = "rejected"
    RESEARCH_ONLY = "research_only"
    ELIGIBLE_FOR_PAPER_TRADING = "eligible_for_paper_trading"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


__all__ = [
    "BAR_SAMPLED_SERIES_KINDS",
    "TERMINAL_ROBUSTNESS_STAGES",
    "TRADE_SAMPLED_SERIES_KINDS",
    "BootstrapMethodKind",
    "GateOutcomeKind",
    "MultipleTestingCorrectionKind",
    "PerturbationAxisKind",
    "PromotionDecisionKind",
    "RegimeDimensionKind",
    "ReturnSeriesKind",
    "RobustnessStage",
    "StressAxisKind",
    "is_legal_robustness_transition",
    "is_terminal_robustness_stage",
]
