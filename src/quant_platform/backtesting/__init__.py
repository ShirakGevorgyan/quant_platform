"""Milestone 5: leakage-safe financial evaluation, signal simulation,
transaction-cost modeling, and backtesting framework.

Depends on `quant_platform.calibration`, `quant_platform.execution`,
`quant_platform.ml`, `quant_platform.features`, and `quant_platform.
historical`. None of those packages import anything from this one -- the
dependency is one-way, and this package sits ONE LAYER ABOVE `calibration`
(it evaluates the financial usefulness of an already-frozen, already-
verified calibrated-prediction policy; it never selects features,
hyperparameters, models, or calibration/threshold/confidence/uncertainty
policies itself).

THE CENTRAL RULE
--------------------------------------------------------------------------
No bar-t-close entry by default: a prediction available at bar t's close
is not executable until at least bar t+1 (`backtesting.specs.EntrySpec`
structurally rejects same-bar-close entry unless explicitly, deliberately
permitted). Every fold starts flat and ends at its own outer-test
boundary -- no position ever crosses a fold boundary
(`backtesting.execution`'s bar-position walk is bounded by
`fold_end_position`, derived from the SAME outer-fold boundaries
`calibration`/`execution` already established, never re-derived).

Source predictions are never trusted by filename/hash alone --
`backtesting.runner.verify_and_load_predictions` independently re-derives
the frozen calibrator from its own persisted parameters and re-applies it
to the persisted raw probabilities before this package's signal mapping
ever sees a prediction.

WHY THIS PACKAGE DOES NOT REUSE `quant_platform.engine`/`costs`/`risk`/
`strategy`
--------------------------------------------------------------------------
See `backtesting.models`'s own module docstring for the full architectural
reasoning: this platform's original, cursor-driven, mutable-ABC backtest
engine is a poor fit for evaluating an already-computed, already-verified,
fold-based outer-test prediction series with content-addressed immutable
specs and resumable/verifiable artifacts -- the shape `calibration`/
`optimization`/`execution` already established, not the shape the
original engine established.

MODULE MAP
--------------------------------------------------------------------------
`models` (position/signal-mapping/overlap/entry/exit/cost enums, the
market-bar contract, the verified-prediction contract, `BacktestStage`
state machine) -> `specs` (`BacktestSpec` identity, cost/signal-mapping/
entry/exit specs) -> `costs` (adverse-sign spread/slippage price
adjustments, commission/financing return fractions, itemized
`CostBreakdown`) -> `signals` (pure prediction -> position-intent mapping,
no market data in scope) -> `fills` (the one authoritative fill-price
function, entry-kind-aware vs. exit-basis-aware reference price
selection) -> `positions` (transient `OpenPosition` carrier) -> `execution`
(chronological bar-position-walk simulation -> `TradeRecord`s) -> `trades`
(the terminal, self-verifying trade record) -> `returns` (gross/net return
math, compounded/non-compounded equity curves) -> `drawdown` (independent
peak/trough/recovery analysis) -> `metrics` (~30 financial metrics,
skip-not-NaN) -> `manifests` (`BacktestManifest` + append-only event log,
`BacktestStage` state machine) -> `runner` (`BacktestRunner`, the
top-level orchestrator + outer-fold execution, benchmarks, cost
sensitivity, confidence/uncertainty bucket analysis) -> `resume`
(verified-artifact resume planning) -> `verification` (`verify_backtest`,
including the recomputation check that proves persisted financial metrics
are actually reproducible from a persisted `TradeSet`, not just
hash-consistent) -> `reporting` (JSON/Markdown reports, mandatory
hypothetical/non-live disclaimers).

EXPLICITLY OUT OF SCOPE (see the delivery report for the complete list):
live broker integration, order submission, exchange connectivity,
portfolio allocation across multiple strategies, reinforcement learning,
deep learning, market-data ingestion, news collection, feature
engineering, automatic strategy discovery, capital leverage optimization,
production risk limits, online retraining.
"""

from __future__ import annotations

from quant_platform.backtesting.costs import (
    CostBreakdown,
    entry_slippage_price_adjustment,
    entry_spread_price_adjustment,
    exit_slippage_price_adjustment,
    exit_spread_price_adjustment,
    financing_fraction,
    per_side_commission_fraction,
)
from quant_platform.backtesting.drawdown import (
    DRAWDOWN_EPISODE_SCHEMA_VERSION,
    DrawdownEpisode,
    DrawdownReport,
    compute_drawdown_report,
)
from quant_platform.backtesting.execution import simulate_outer_fold_trades
from quant_platform.backtesting.fills import (
    FillPriceResult,
    compute_fill_price,
    select_basis_reference_price,
    select_entry_reference_price,
    select_exit_reference_price,
)
from quant_platform.backtesting.manifests import (
    BACKTEST_MANIFEST_SCHEMA_VERSION,
    BacktestEventRecord,
    BacktestEventStore,
    BacktestEventType,
    BacktestManifest,
    BacktestManifestStore,
)
from quant_platform.backtesting.metrics import SharpeAssumptions, bars_per_year, compute_financial_metrics
from quant_platform.backtesting.models import (
    MARKET_BAR_OPTIONAL_COLUMNS,
    MARKET_BAR_REQUIRED_COLUMNS,
    TERMINAL_BACKTEST_STAGES,
    VERIFIED_PREDICTION_SET_SCHEMA_VERSION,
    BacktestStage,
    CommissionModelKind,
    CompoundingPolicyKind,
    Decision,
    DecisionTimestampPolicyKind,
    EntryPolicyKind,
    ExitPolicyKind,
    ExitReasonCode,
    FinalTradePolicyKind,
    FinancingModelKind,
    OverlapPolicyKind,
    PositionDirection,
    PositionMode,
    PriceBasisKind,
    ReturnCalculationPolicyKind,
    SignalMappingPolicyKind,
    SignalReasonCode,
    SlippageModelKind,
    SpreadModelKind,
    TradeStatus,
    VerifiedPredictionSet,
    is_legal_backtest_transition,
    is_terminal_backtest_stage,
    validate_bar_interval_consistency,
    validate_market_bar_frame,
)
from quant_platform.backtesting.positions import OpenPosition
from quant_platform.backtesting.reporting import build_backtest_report_json, render_backtest_report_markdown
from quant_platform.backtesting.resume import (
    can_resume,
    require_backtest_resumable,
    resolve_resume_start_stage,
    verify_completed_backtest_outer_folds,
)
from quant_platform.backtesting.returns import (
    EQUITY_POINT_SCHEMA_VERSION,
    EquityCurve,
    EquityPoint,
    TradeReturnResult,
    build_equity_curve,
    compute_gross_return,
    compute_trade_return_result,
)
from quant_platform.backtesting.runner import (
    OUTER_FOLD_BACKTEST_RESULT_SCHEMA_VERSION,
    BacktestOutcome,
    BacktestRunner,
    BenchmarkReport,
    BenchmarkResult,
    BucketAnalysisReport,
    BucketResult,
    CostSensitivityReport,
    CostSensitivityResult,
    OuterFoldBacktestResult,
    ResolvedBacktestInputs,
    compute_benchmark_report,
    compute_bucket_analysis_report,
    compute_cost_sensitivity_report,
    resolve_backtest_inputs,
    resolve_market_bars_for_timeline,
    run_outer_fold_backtest,
    verify_and_load_predictions,
)
from quant_platform.backtesting.signals import SIGNAL_SET_SCHEMA_VERSION, Signal, SignalSet, generate_signals
from quant_platform.backtesting.specs import (
    BACKTEST_SPEC_SCHEMA_VERSION,
    DEFAULT_COST_SENSITIVITY_SCENARIOS,
    BacktestIdentity,
    BacktestSpec,
    CommissionSpec,
    CostSensitivityScenario,
    EntrySpec,
    ExitSpec,
    FinancingSpec,
    SignalMappingSpec,
    SlippageSpec,
    SpreadSpec,
    compute_backtest_identity,
)
from quant_platform.backtesting.trades import (
    TRADE_RECORD_SCHEMA_VERSION,
    TradeRecord,
    TradeSet,
    compute_trade_id,
)
from quant_platform.backtesting.verification import verify_backtest

BACKTESTING_FRAMEWORK_VERSION = "1.0.0"
"""Version of this package's own backtest-lifecycle/identity/manifest
semantics -- recorded informationally, independent of `quant_platform.
__version__` and of earlier milestones' own version constants. Bump on
any change to those semantics."""

__all__ = [
    "BACKTESTING_FRAMEWORK_VERSION",
    "BACKTEST_MANIFEST_SCHEMA_VERSION",
    "BACKTEST_SPEC_SCHEMA_VERSION",
    "DEFAULT_COST_SENSITIVITY_SCENARIOS",
    "DRAWDOWN_EPISODE_SCHEMA_VERSION",
    "EQUITY_POINT_SCHEMA_VERSION",
    "MARKET_BAR_OPTIONAL_COLUMNS",
    "MARKET_BAR_REQUIRED_COLUMNS",
    "OUTER_FOLD_BACKTEST_RESULT_SCHEMA_VERSION",
    "SIGNAL_SET_SCHEMA_VERSION",
    "TERMINAL_BACKTEST_STAGES",
    "TRADE_RECORD_SCHEMA_VERSION",
    "VERIFIED_PREDICTION_SET_SCHEMA_VERSION",
    "BacktestEventRecord",
    "BacktestEventStore",
    "BacktestEventType",
    "BacktestIdentity",
    "BacktestManifest",
    "BacktestManifestStore",
    "BacktestOutcome",
    "BacktestRunner",
    "BacktestSpec",
    "BacktestStage",
    "BenchmarkReport",
    "BenchmarkResult",
    "BucketAnalysisReport",
    "BucketResult",
    "CommissionModelKind",
    "CommissionSpec",
    "CompoundingPolicyKind",
    "CostBreakdown",
    "CostSensitivityReport",
    "CostSensitivityResult",
    "CostSensitivityScenario",
    "Decision",
    "DecisionTimestampPolicyKind",
    "DrawdownEpisode",
    "DrawdownReport",
    "EntryPolicyKind",
    "EntrySpec",
    "EquityCurve",
    "EquityPoint",
    "ExitPolicyKind",
    "ExitReasonCode",
    "ExitSpec",
    "FillPriceResult",
    "FinalTradePolicyKind",
    "FinancingModelKind",
    "FinancingSpec",
    "OpenPosition",
    "OuterFoldBacktestResult",
    "OverlapPolicyKind",
    "PositionDirection",
    "PositionMode",
    "PriceBasisKind",
    "ResolvedBacktestInputs",
    "ReturnCalculationPolicyKind",
    "SharpeAssumptions",
    "Signal",
    "SignalMappingPolicyKind",
    "SignalMappingSpec",
    "SignalReasonCode",
    "SignalSet",
    "SlippageModelKind",
    "SlippageSpec",
    "SpreadModelKind",
    "SpreadSpec",
    "TradeRecord",
    "TradeReturnResult",
    "TradeSet",
    "TradeStatus",
    "VerifiedPredictionSet",
    "bars_per_year",
    "build_backtest_report_json",
    "build_equity_curve",
    "can_resume",
    "compute_backtest_identity",
    "compute_benchmark_report",
    "compute_bucket_analysis_report",
    "compute_cost_sensitivity_report",
    "compute_drawdown_report",
    "compute_fill_price",
    "compute_financial_metrics",
    "compute_gross_return",
    "compute_trade_id",
    "compute_trade_return_result",
    "entry_slippage_price_adjustment",
    "entry_spread_price_adjustment",
    "exit_slippage_price_adjustment",
    "exit_spread_price_adjustment",
    "financing_fraction",
    "generate_signals",
    "is_legal_backtest_transition",
    "is_terminal_backtest_stage",
    "per_side_commission_fraction",
    "render_backtest_report_markdown",
    "require_backtest_resumable",
    "resolve_backtest_inputs",
    "resolve_market_bars_for_timeline",
    "resolve_resume_start_stage",
    "run_outer_fold_backtest",
    "select_basis_reference_price",
    "select_entry_reference_price",
    "select_exit_reference_price",
    "simulate_outer_fold_trades",
    "validate_bar_interval_consistency",
    "validate_market_bar_frame",
    "verify_and_load_predictions",
    "verify_backtest",
    "verify_completed_backtest_outer_folds",
]
