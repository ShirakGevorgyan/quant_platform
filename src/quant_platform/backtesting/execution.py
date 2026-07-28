"""Chronological entry/exit simulation (Milestone 5, Sections 9-11, 23-24)
-- the orchestrator that turns an already-generated `SignalSet` (no market
data in scope, see `backtesting.signals`) plus real market bars into
`TradeRecord`s, walking BAR POSITIONS in strict chronological order (never
signal order alone -- a bar-position walk is what makes overlapping
signals, opposite-signal exits, and fold-boundary forcing all resolve
unambiguously and deterministically, regardless of signal density).

WHY A BAR-POSITION WALK, NOT A SIGNAL-ONLY LOOP
--------------------------------------------------------------------------
A signal-only loop ("for each accepted signal, decide what happens") is
ambiguous the moment two signals are dense enough that the second's entry
bar arrives before the first's already-determined exit bar -- whether
that counts as "overlap" depends on comparing actual BAR POSITIONS, not
signal order. Walking bar positions one at a time and asking, at each
bar, "does the open position's exit trigger here? does a new signal's
entry trigger here?" resolves this unambiguously.

`QUEUE` (Section 9): a signal arriving while a position is open is
remembered (the MOST RECENT one, if several arrive before the position
closes -- a deterministic, documented choice) and replayed as a fresh
entry attempt at the exact bar the open position closes, natural or
forced."""

from __future__ import annotations

import pandas as pd

from quant_platform.backtesting.fills import (
    compute_fill_price,
    select_entry_reference_price,
    select_exit_reference_price,
)
from quant_platform.backtesting.models import (
    ExitPolicyKind,
    ExitReasonCode,
    FinalTradePolicyKind,
    OverlapPolicyKind,
    PositionDirection,
    TradeStatus,
)
from quant_platform.backtesting.positions import OpenPosition
from quant_platform.backtesting.returns import compute_trade_return_result
from quant_platform.backtesting.signals import Signal, SignalSet
from quant_platform.backtesting.specs import BacktestSpec, CostSensitivityScenario
from quant_platform.backtesting.trades import TradeRecord, TradeSet, compute_trade_id
from quant_platform.core.exceptions import ExecutionSimulationError
from quant_platform.ml.persistence import format_utc_timestamp, parse_utc_timestamp


def _entry_bar_position(signal: Signal, spec: BacktestSpec) -> int:
    return signal.sample_position + spec.entry_spec.delay_bars


def _fixed_exit_bar_position(entry_bar: int, spec: BacktestSpec) -> int | None:
    """Deterministic exit bar for exit policies that do NOT depend on the
    future signal stream (`FIXED_HORIZON`/`NEXT_BAR_CLOSE`); `None` for
    `OPPOSITE_SIGNAL`/`END_OF_FOLD`, both resolved dynamically by the
    caller's bar walk."""
    if spec.exit_spec.kind is ExitPolicyKind.FIXED_HORIZON:
        assert spec.exit_spec.holding_period_bars is not None
        return entry_bar + spec.exit_spec.holding_period_bars
    if spec.exit_spec.kind is ExitPolicyKind.NEXT_BAR_CLOSE:
        return entry_bar
    return None


def _natural_exit_reason(spec: BacktestSpec) -> ExitReasonCode:
    if spec.exit_spec.kind is ExitPolicyKind.FIXED_HORIZON:
        return ExitReasonCode.FIXED_HORIZON_REACHED
    if spec.exit_spec.kind is ExitPolicyKind.NEXT_BAR_CLOSE:
        return ExitReasonCode.NEXT_BAR_CLOSE
    raise ExecutionSimulationError(f"_natural_exit_reason: {spec.exit_spec.kind!r} has no self-contained natural exit")  # pragma: no cover - defensive


def _holding_days(entry_timestamp: str, exit_timestamp: str) -> float:
    entry_ts = parse_utc_timestamp(entry_timestamp)
    exit_ts = parse_utc_timestamp(exit_timestamp)
    return max((exit_ts - entry_ts).total_seconds() / 86400.0, 0.0)


def _open_from_signal(signal: Signal, bars: pd.DataFrame, spec: BacktestSpec, entry_bar: int) -> OpenPosition:
    bar = bars.iloc[entry_bar]
    reference_price = select_entry_reference_price(bar, entry_spec=spec.entry_spec, price_basis=spec.price_basis, direction=signal.direction)
    fill = compute_fill_price(
        reference_price=reference_price, direction=signal.direction, is_entry=True,
        spread_spec=spec.spread_spec, slippage_spec=spec.slippage_spec,
    )
    return OpenPosition(
        direction=signal.direction, signal_sample_position=signal.sample_position, signal_timestamp=signal.decision_timestamp,
        decision_timestamp=signal.decision_timestamp, entry_bar_position=entry_bar, entry_timestamp=format_utc_timestamp(bar["open_time"]),
        entry_observed_price=fill.observed_price, entry_effective_price=fill.effective_price, entry_spread_cost=fill.spread_adjustment,
        entry_commission=0.0, entry_slippage=fill.slippage_adjustment, confidence=signal.confidence, uncertainty=signal.uncertainty,
        calibrated_probability=signal.calibrated_probability, entry_reason=signal.reason_code,
    )


def _close_position(
    open_pos: OpenPosition, *, bars: pd.DataFrame, spec: BacktestSpec, exit_bar: int, exit_reason: ExitReasonCode,
    outer_fold_index: int, status: TradeStatus = TradeStatus.CLOSED, cost_scenario: CostSensitivityScenario | None = None,
) -> TradeRecord:
    bar = bars.iloc[exit_bar]
    reference_price = select_exit_reference_price(bar, price_basis=spec.price_basis, direction=open_pos.direction)
    fill = compute_fill_price(
        reference_price=reference_price, direction=open_pos.direction, is_entry=False,
        spread_spec=spec.spread_spec, slippage_spec=spec.slippage_spec,
    )
    exit_timestamp = format_utc_timestamp(bar["open_time"])
    notional = spec.initial_notional * spec.exposure_cap
    result = compute_trade_return_result(
        direction=open_pos.direction, entry_observed_price=open_pos.entry_observed_price, exit_observed_price=fill.observed_price,
        entry_spread_adjustment=open_pos.entry_spread_cost, exit_spread_adjustment=fill.spread_adjustment,
        entry_slippage_adjustment=open_pos.entry_slippage, exit_slippage_adjustment=fill.slippage_adjustment,
        commission_spec=spec.commission_spec, financing_spec=spec.financing_spec,
        holding_days=_holding_days(open_pos.entry_timestamp, exit_timestamp), notional=notional,
        return_calculation_policy=spec.return_calculation_policy, cost_scenario=cost_scenario,
    )
    trade_id = compute_trade_id(
        source_calibration_id=spec.source_calibration_id, outer_fold_index=outer_fold_index, signal_sample_position=open_pos.signal_sample_position,
        direction=open_pos.direction, entry_timestamp=open_pos.entry_timestamp, exit_timestamp=exit_timestamp,
    )
    return TradeRecord(
        schema_version=1, trade_id=trade_id, signal_sample_position=open_pos.signal_sample_position, outer_fold_index=outer_fold_index,
        direction=open_pos.direction, signal_timestamp=open_pos.signal_timestamp, decision_timestamp=open_pos.decision_timestamp,
        entry_timestamp=open_pos.entry_timestamp, entry_bar_position=open_pos.entry_bar_position, entry_observed_price=open_pos.entry_observed_price,
        entry_effective_price=open_pos.entry_effective_price, exit_timestamp=exit_timestamp, exit_bar_position=exit_bar,
        exit_observed_price=fill.observed_price, exit_effective_price=fill.effective_price, holding_bars=exit_bar - open_pos.entry_bar_position,
        gross_return=result.gross_return, net_return=result.net_return, cost_breakdown=result.cost_breakdown, confidence=open_pos.confidence,
        uncertainty=open_pos.uncertainty, calibrated_probability=open_pos.calibrated_probability, entry_reason=open_pos.entry_reason,
        exit_reason=exit_reason, status=status, source_calibration_id=spec.source_calibration_id, source_experiment_id=spec.source_experiment_id,
    )


def _finalize_incomplete_position(
    open_pos: OpenPosition, *, bars: pd.DataFrame, spec: BacktestSpec, fold_end_position: int, outer_fold_index: int,
    cost_scenario: CostSensitivityScenario | None,
) -> TradeRecord | None:
    """Section 11: an explicit, spec-declared policy for a trade still
    open when the outer-test partition ends."""
    if spec.exit_spec.final_trade_policy is FinalTradePolicyKind.FORCE_CLOSE_AT_FINAL_PRICE:
        return _close_position(
            open_pos, bars=bars, spec=spec, exit_bar=fold_end_position, exit_reason=ExitReasonCode.END_OF_FOLD_FORCED_CLOSE,
            outer_fold_index=outer_fold_index, status=TradeStatus.INCOMPLETE_FORCE_CLOSED, cost_scenario=cost_scenario,
        )
    return None  # DISCARD_INCOMPLETE / MARK_INCOMPLETE_EXCLUDE: neither is recorded as a TradeRecord in this reference implementation


def _simulate_independent_overlapping(
    accepted: list[Signal], *, bars: pd.DataFrame, spec: BacktestSpec, fold_end_position: int, outer_fold_index: int,
    cost_scenario: CostSensitivityScenario | None,
) -> list[TradeRecord]:
    if spec.exit_spec.kind not in (ExitPolicyKind.FIXED_HORIZON, ExitPolicyKind.NEXT_BAR_CLOSE, ExitPolicyKind.END_OF_FOLD):
        raise ExecutionSimulationError(
            "simulate_outer_fold_trades: overlap_policy=independent_overlapping requires a deterministic, "
            "self-contained exit policy (fixed_horizon/next_bar_close/end_of_fold) -- opposite_signal exits "
            "have no well-defined meaning for multiple simultaneously open, mutually independent trades"
        )
    trades: list[TradeRecord] = []
    for sig in accepted:
        entry_bar = _entry_bar_position(sig, spec)
        if entry_bar > fold_end_position:
            continue  # MISSING_MARKET_BAR: entry would require a bar beyond this fold -- silently excluded, never fabricated
        open_pos = _open_from_signal(sig, bars, spec, entry_bar)
        exit_bar = _fixed_exit_bar_position(entry_bar, spec)
        if exit_bar is None:
            exit_bar = fold_end_position  # END_OF_FOLD
        if exit_bar <= fold_end_position:
            reason = _natural_exit_reason(spec) if spec.exit_spec.kind is not ExitPolicyKind.END_OF_FOLD else ExitReasonCode.END_OF_FOLD_FORCED_CLOSE
            trades.append(_close_position(open_pos, bars=bars, spec=spec, exit_bar=exit_bar, exit_reason=reason, outer_fold_index=outer_fold_index, cost_scenario=cost_scenario))
        else:
            final = _finalize_incomplete_position(open_pos, bars=bars, spec=spec, fold_end_position=fold_end_position, outer_fold_index=outer_fold_index, cost_scenario=cost_scenario)
            if final is not None:
                trades.append(final)
    return trades


def _simulate_one_position_at_a_time(
    signal_by_entry_bar: dict[int, Signal], *, bars: pd.DataFrame, spec: BacktestSpec, fold_end_position: int, outer_fold_index: int,
    cost_scenario: CostSensitivityScenario | None,
) -> list[TradeRecord]:
    if not signal_by_entry_bar:
        return []
    trades: list[TradeRecord] = []
    open_pos: OpenPosition | None = None
    pending_exit_bar: int | None = None
    queued_signal: Signal | None = None
    first_bar = min(signal_by_entry_bar)

    for bar_position in range(first_bar, fold_end_position + 1):
        incoming = signal_by_entry_bar.get(bar_position)

        if open_pos is not None:
            natural_close = pending_exit_bar is not None and bar_position == pending_exit_bar
            opposite_signal_close = (
                spec.exit_spec.kind is ExitPolicyKind.OPPOSITE_SIGNAL and incoming is not None
                and incoming.direction is not open_pos.direction and incoming.direction is not PositionDirection.FLAT
            )
            if natural_close:
                trades.append(_close_position(open_pos, bars=bars, spec=spec, exit_bar=bar_position, exit_reason=_natural_exit_reason(spec), outer_fold_index=outer_fold_index, cost_scenario=cost_scenario))
                open_pos, pending_exit_bar = None, None
            elif opposite_signal_close:
                # The SAME opposing signal that triggered this close is
                # deliberately NOT marked "consumed": Section 11's
                # opposite-signal exit means "flip" in a long/short
                # position mode -- after closing, we are flat again and
                # this signal is evaluated fresh below, exactly like any
                # signal arriving while already flat.
                trades.append(_close_position(open_pos, bars=bars, spec=spec, exit_bar=bar_position, exit_reason=ExitReasonCode.OPPOSITE_SIGNAL, outer_fold_index=outer_fold_index, cost_scenario=cost_scenario))
                open_pos, pending_exit_bar = None, None

        if open_pos is None and queued_signal is not None:
            open_pos = _open_from_signal(queued_signal, bars, spec, bar_position)
            pending_exit_bar = _fixed_exit_bar_position(bar_position, spec)
            queued_signal = None

        if incoming is not None:
            if open_pos is None:
                open_pos = _open_from_signal(incoming, bars, spec, bar_position)
                pending_exit_bar = _fixed_exit_bar_position(bar_position, spec)
            elif spec.overlap_policy is OverlapPolicyKind.IGNORE:
                pass
            elif spec.overlap_policy is OverlapPolicyKind.CLOSE_ONLY:
                trades.append(_close_position(open_pos, bars=bars, spec=spec, exit_bar=bar_position, exit_reason=ExitReasonCode.OPPOSITE_SIGNAL, outer_fold_index=outer_fold_index, cost_scenario=cost_scenario))
                open_pos, pending_exit_bar = None, None
            elif spec.overlap_policy is OverlapPolicyKind.CLOSE_AND_REVERSE:
                trades.append(_close_position(open_pos, bars=bars, spec=spec, exit_bar=bar_position, exit_reason=ExitReasonCode.OPPOSITE_SIGNAL, outer_fold_index=outer_fold_index, cost_scenario=cost_scenario))
                open_pos = _open_from_signal(incoming, bars, spec, bar_position)
                pending_exit_bar = _fixed_exit_bar_position(bar_position, spec)
            elif spec.overlap_policy is OverlapPolicyKind.QUEUE:
                queued_signal = incoming
            # INDEPENDENT_OVERLAPPING never reaches this function.

    if open_pos is not None:
        if spec.exit_spec.kind is ExitPolicyKind.END_OF_FOLD:
            trades.append(_close_position(open_pos, bars=bars, spec=spec, exit_bar=fold_end_position, exit_reason=ExitReasonCode.END_OF_FOLD_FORCED_CLOSE, outer_fold_index=outer_fold_index, cost_scenario=cost_scenario))
        else:
            final = _finalize_incomplete_position(open_pos, bars=bars, spec=spec, fold_end_position=fold_end_position, outer_fold_index=outer_fold_index, cost_scenario=cost_scenario)
            if final is not None:
                trades.append(final)

    return trades


def simulate_outer_fold_trades(
    *, signals: SignalSet, bars: pd.DataFrame, spec: BacktestSpec, fold_end_position: int, cost_scenario: CostSensitivityScenario | None = None,
) -> TradeSet:
    """Section 23 step 6: simulate entries and exits chronologically.
    `bars` must be `iloc`-indexed by the SAME row-position space as
    `signals`' `sample_position` values (Section 24: fold boundaries are
    enforced via `fold_end_position` -- no bar past it is ever read).

    An ACCEPTED signal can still carry `direction=FLAT` (Section 8:
    `directional_long_flat`'s "predicted negative -> flat" case, and
    `probability_bands`' middle dead-zone both accept the signal while
    calling for no position) -- these never trigger an entry attempt
    (there is no side to open), exactly how `ExitPolicyKind.OPPOSITE_
    SIGNAL`'s own close-trigger check already excludes `PositionDirection.
    FLAT` below. Filtered out here, once, so every downstream helper can
    assume every signal it sees has a real side."""
    accepted = sorted((s for s in signals.accepted_signals if s.direction is not PositionDirection.FLAT), key=lambda s: s.sample_position)

    if spec.overlap_policy is OverlapPolicyKind.INDEPENDENT_OVERLAPPING:
        trades = _simulate_independent_overlapping(
            accepted, bars=bars, spec=spec, fold_end_position=fold_end_position, outer_fold_index=signals.outer_fold_index, cost_scenario=cost_scenario,
        )
    else:
        signal_by_entry_bar: dict[int, Signal] = {}
        for sig in accepted:
            entry_bar = _entry_bar_position(sig, spec)
            if entry_bar <= fold_end_position:
                signal_by_entry_bar.setdefault(entry_bar, sig)  # first signal at a given entry bar wins deterministically
            # else: MISSING_MARKET_BAR -- entry would require a bar beyond this fold, silently excluded
        trades = _simulate_one_position_at_a_time(
            signal_by_entry_bar, bars=bars, spec=spec, fold_end_position=fold_end_position, outer_fold_index=signals.outer_fold_index, cost_scenario=cost_scenario,
        )

    return TradeSet(schema_version=1, outer_fold_index=signals.outer_fold_index, trades=tuple(trades))


__all__ = ["simulate_outer_fold_trades"]
