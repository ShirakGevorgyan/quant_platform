# Deterministic Paper Trading and Shadow Execution Engine (Milestone 7)

## Scope and purpose

Milestone 6 answers "does this candidate's apparent backtest performance
survive resampling uncertainty, multiple-testing correction, fold-level
concentration, cost/latency stress, and regime narrowness?" and, if so,
issues an `ELIGIBLE_FOR_PAPER_TRADING` promotion decision — never
`ELIGIBLE_FOR_LIVE_TRADING`, which does not exist as a constructible
value anywhere in this platform. This milestone answers a different
question: **given a genuinely `ELIGIBLE_FOR_PAPER_TRADING` candidate, can
its full trading lifecycle — decisions, orders, fills, positions, costs,
account state, risk limits, a kill switch — be simulated deterministically,
broker-neutrally, and exactly, without ever transmitting a real order?**

This is infrastructure and correctness work. It does not claim
profitability, production readiness, broker readiness, or live-trading
readiness. It does not implement live order transmission or MT5
integration. There is no `LIVE` session mode, no `ELIGIBLE_FOR_LIVE_
TRADING` decision kind, and no code path anywhere in this package capable
of opening a network connection to a broker.

## Architecture

New top-level package `quant_platform.paper_trading`, depending on `core`,
`backtesting` (reuses `SpreadSpec`/`SlippageSpec`/`CommissionSpec`/
`FinancingSpec` and their cost formulas directly, never duplicated),
`robustness` (the eligibility chain re-verifies a robustness
result/promotion decision), and shared `ml` persistence utilities. Nothing
in `robustness`/`backtesting`/`execution`/`optimization` imports
`paper_trading` — the dependency direction is strictly one-way.

### Module layout (24 files)

- `models.py` — every shared enum (`SessionMode`, `ClockMode`,
  `MarketEventMode`, `OrderSide`/`OrderTypeKind`/`TimeInForceKind`/
  `OrderState`, `RejectReasonKind`, `RiskActionKind`/`RiskTriggerKind`/
  `KillSwitchState`, `PaperSessionStage`, `LedgerEntryKind`,
  `BarAmbiguityPolicyKind`, `PartialFillPolicyKind`, `MarkFieldKind`) and
  every linear/branching legal-transition table (order states, kill-switch
  states, session stages).
- `specs.py` — `PaperTradingSpec` (the content-addressed identity root)
  and every embedded policy spec: `InstrumentSpec`, `OrderPolicySpec`,
  `ExecutionPolicySpec`, `FillPolicySpec`, `FinancingPolicySpec`
  (wraps two independent `backtesting.specs.FinancingSpec` for long/short
  asymmetry), `LatencyPolicySpec`, `LiquidityPolicySpec`,
  `PositionPolicySpec`, `RiskLimitsSpec`, `SessionBoundaryPolicySpec`,
  `StartingPositionSpec`.
- `identity.py` — the one shared content-addressing primitive
  (`compute_content_id`) every domain object's `create_*` factory uses.
- `events.py` — Section 5's normalized market-event model: `QuoteEvent`,
  `BarEvent`, `SessionOpenEvent`/`SessionCloseEvent`,
  `TradingHaltEvent`/`TradingResumeEvent`, `FinancingEvent`,
  `EndOfStreamEvent`.
- `clock.py` — Section 6's explicit clock abstractions: `ReplayClock`,
  `ForwardEventClock`, `ManualTestClock`, `decision_time_for`.
- `strategy.py` — Section 7's broker-neutral `StrategyRuntime` Protocol,
  `StrategyContext`, `PortfolioSnapshot`, `RiskState`, `SessionState`,
  `StrategyDecision`.
- `orders.py` — Section 8's immutable `OrderRequest` and event-sourced
  `OrderStateEvent`/`resolve_order_state`.
- `order_policy.py` — Section 9's deterministic decision-to-order
  conversion.
- `execution.py` — Section 10's `PaperExecutionEngine` (market/limit/stop
  fills, latency, spread/slippage application, bar ambiguity policy).
- `fills.py` — Section 11's immutable `Fill` record.
- `accounting.py` — Section 12's exact position accounting
  (`PositionState`, `apply_fill_to_position`, `apply_mark_to_position`).
- `portfolio.py` — Section 13's `PortfolioState`/`PaperAccount`-level
  aggregation (cash, exposure, equity, drawdown, turnover).
- `costs.py` — Section 16's cost/financing recognition-timing wiring atop
  `backtesting`'s reused formulas.
- `risk.py` — Section 17's pre-trade/continuous risk checks and Section
  18's kill switch.
- `eligibility.py` — Section 4's fail-closed eligibility/source
  verification chain.
- `manifests.py` — Section 20's `PaperSessionManifest` state machine and
  session locking.
- `persistence.py` — Section 21's append-only, hash-chained event ledger
  (`PaperSessionEventStore`).
- `runner.py` — Section 22's `PaperTradingRunner`
  (`run_paper_trading_session`) and Section 19's shadow-mode driver
  (`run_shadow_session`).
- `shadow.py` — Section 19's shadow-observation pure functions
  (`evaluate_shadow_decision`, `ShadowObservation`).
- `reconciliation.py` — Section 25's 11 independent reconciliation checks.
- `verification.py` — Section 26's `verify_paper_session`.
- `reports.py` — Section 27's 15 durable report summaries and Section
  28's backtest-comparison diagnostic.
- `replay.py` — Section 32's bounded deterministic replay-input reader.

Plus `config/paper_trading_schemas.py` (Section 29's Pydantic
configuration schema) and 24 new domain exceptions in `core/exceptions.py`
(`PaperTradingError` and its subclasses).

## Event-time model and decision/order/fill separation

Every trading decision is computed from information available at or
before its declared `decision_time`. The clock (`clock.py`) is the single
source of "now" — no strategy code, order-policy code, or execution code
ever calls `datetime.now()`/`time.time()` directly; `ReplayClock`
advances only when explicitly told to, deterministically, independent of
wall-clock speed.

A `StrategyDecision` is **not** an order — it expresses intent (target
direction/exposure, confidence, uncertainty, abstain flag, reason codes),
and is persisted unconditionally, including abstentions. `order_policy.py`
is the ONLY place a decision becomes zero or more `OrderRequest` objects
(target-position delta calculation, close-before-reverse or atomic
target-delta, quantity rounding/clamping, cooldown, per-event/per-window
rate limits). `execution.py` is the ONLY place an accepted working order
becomes a `Fill` — order-policy code never sees future prices, and
execution code never sees strategy intent, only the order itself and the
current market event.

## Market-event contracts and clock semantics

`QuoteEvent`/`BarEvent`/`SessionOpenEvent`/`SessionCloseEvent`/
`TradingHaltEvent`/`TradingResumeEvent`/`FinancingEvent`/`EndOfStreamEvent`
are immutable, content-addressed, and self-validating (finite positive
prices, `ask >= bid`, `high >= max(open, close, low)`, `low <=
min(open, close, high)`, timezone-aware timestamps, non-repeating
quality flags). `CorporateOrInstrumentAdjustmentEvent` is deliberately
**not** implemented — Section 5 permits it "only if generically
supported," and no generic corporate-action model exists anywhere in this
repository to reuse.

`market_event_time` gives the one comparable instant every event kind
exposes (`BarEvent` uses its own `close_time` — the instant its data is
actually fully known, never `open_time`). `replay.py` validates a
BOUNDED source's cross-event properties up front (strict chronological
order, strictly increasing sequence numbers, no duplicate event_id/
sequence, single instrument unless explicitly allowed, and — since this
is what makes a `REPLAY_PAPER` session ever reach `COMPLETED` — exactly
one `EndOfStreamEvent`, as the last event). A live/forward stream's own
cross-event ordering enforcement is a separate concern this milestone
does not implement (see "Unsupported behavior" below) — Section 32
explicitly excludes building a market-data downloader or MT5 ingestion in
this milestone.

## Order model and state machine

Order types: `MARKET`, `LIMIT`, `STOP`. `STOP_LIMIT`/`MARKET_ON_CLOSE` are
**not** claimed — Section 8: "Do not claim support for an order type that
is only partially simulated." Time-in-force: `DAY`, `GTC`, `IOC`. `FOK` is
modeled but only ever legal with `FULL_FILL_ONLY` fill policy semantics.
States: `CREATED → VALIDATED → (REJECTED | ACCEPTED → WORKING →
(PARTIALLY_FILLED)* → (FILLED | CANCEL_REQUESTED → CANCELLED | EXPIRED))`,
with every transition explicit in `models._LEGAL_ORDER_TRANSITIONS` and
every rejection carrying a typed `RejectReasonKind`. Every
`ORDER_STATE_EVENT` ledger entry embeds the order's own full economic
detail (`_order_state_payload`) alongside the transition, so any single
ledger entry is enough to recover an order's side/quantity/price —
critical for `reconciliation.py`'s from-ledger reconstruction.

## Execution assumptions and bar ambiguity

Market orders fill at the current event's reference price (mid for
`QuoteEvent`, `close` for `BarEvent` — `ExecutionPolicySpec.mark_field`).
Buy fills use ask-side spread logic; sell fills use bid-side spread
logic — never mixed. Limit/stop triggering in bar mode cannot see the
true intrabar path; `BarAmbiguityPolicyKind.WORST_CASE` (the default) is
the financially conservative resolution when both a stop and a target
could plausibly have been touched in the same bar. Latency is modeled as
three separately configurable, purely arithmetic delays applied to event
time (decision-to-submit, submit-to-accept, accept-to-fill-eligible) —
never real `sleep`, so replay stays speed-independent and byte-identical.
Partial fills are either `FULL_FILL_ONLY` (fail-closed default: an order
fills completely this event or not at all) or `DETERMINISTIC_PARTIAL`
(only meaningful against a genuinely disclosed quote size — liquidity is
never invented for a bar).

## Spread, slippage, commission, and financing

Reused directly from `backtesting.specs`/`backtesting.costs` — the exact
same validated formulas, never duplicated. Financing is long/short
ASYMMETRIC (`FinancingPolicySpec` wraps two independent
`backtesting.specs.FinancingSpec`), recognized only at `FinancingEvent`
processing time (a session-boundary marker; the actual cash delta is
computed then, from the CURRENT position, never pre-computed and carried).
`Fill.financing_component` is therefore always exactly `0.0` — reserved
by Section 11's field list, unused by design, documented explicitly.

## Position accounting and account reconciliation

`accounting.py` implements exact position accounting for flat/long/short/
scale-in/partial-close/full-close/reversal, using the SAME formula shape
for LONG and SHORT (a signed-quantity convention, not two parallel
implementations). Every `PositionState` fixes its own
`contract_multiplier` once and re-validates it against the IMPLIED
multiplier of every incoming fill (`gross_notional / (price * quantity)`)
— a fill built against the wrong multiplier is rejected outright, never
silently corrupting P&L. Weighted-average-cost accounting is used (not
FIFO/LIFO lot tracking) — the standard, simplest-correct convention for a
single-instrument account, an explicit choice, not an oversight.

`portfolio.py`'s exact reconciliation identity (Section 13, verbatim):

```
equity = cash + marked_position_value - liabilities - accrued_costs
```

`liabilities` is always `0.0` — a fully cash-settled model with no
separate margin-liability line item (Section 13 explicitly permits "a
clearly documented margin/P&L abstraction" for derivatives). `cash`
tracks GROSS fill notional plus financing cash deltas only; transaction
costs (spread + slippage + commission) are subtracted exactly once, in
the `equity` formula's own `accrued_costs` term — never double-counted.
This makes `equity - starting_cash == realized_pnl + unrealized_pnl -
accrued_costs + total_financing` hold EXACTLY, verified by dedicated
round-trip reconciliation tests.

`reconciliation.py` (Section 25) implements 11 named checks, each
independently recomputed from the ledger alone: event-sequence
contiguity, no duplicate identities, legal order-state transitions,
order-quantity-equals-fills-plus-remaining, filled-orders-have-zero-
remaining, no-fill-without-a-valid-order, position-quantity-equals-
signed-cumulative-fills, realized-P&L-matches-closed-quantities, cash-
movements-match-fills-and-costs, total-costs-equal-component-sums, and
account-equity-reconciles. Every count/identity/state check is exact
(zero tolerance); accounting checks against a persisted snapshot allow a
small float-accumulation tolerance (`1e-6`) where summing many fills
makes bit-exact equality unrealistic — Section 25's own concession.

## Risk engine and kill switch

Pre-trade checks (`evaluate_pre_trade_risk`) run before every order is
accepted: trading-halted/session-not-accepting-orders (mandatory
booleans), stale-data, max order quantity/notional, max absolute/signed
position, max gross exposure — every applicable check runs and is
recorded, never short-circuited after the first failure, so
`most_severe_action` always sees the complete picture. Continuous checks
(`evaluate_continuous_risk`) run after every mark-to-market: max daily/
realized/unrealized loss, max drawdown fraction, max rejected-order
count, max consecutive execution failures, max stale-data duration, max
reconciliation discrepancy.

The kill switch (`risk.py`, Section 18) is an event-sourced state machine
(`ACTIVE → HALTING → {HALTED, FLATTENING → HALTED} → TERMINATED`) that
can never return to `ACTIVE` once left — "never silently auto-resume
after a safety halt" is a structural guarantee of
`is_legal_kill_switch_transition`, not a runtime check. On any
non-`ALLOW` continuous-risk action, `runner.py` walks the transition
graph, persists a `HALT_TRIGGERED` ledger entry for every step, sets
`trading_halted=True`, and — for `FLATTEN_SIMULATED_POSITIONS`/
`TERMINATE_SESSION` — synthesizes and immediately executes an IOC MARKET
closing order for any open position. That synthesized flatten order is
deliberately exempt from the `trading_halted`/`session_accepting_orders`
pre-trade checks it would otherwise trip: it is risk-REDUCING (an exit),
never new risk-taking, so exempting it is safe and necessary (a halt that
could never close the very position it halted over would be
self-defeating). No risk action ever sends a real order — every action in
`RiskActionKind` is a purely simulated bookkeeping/state effect.

## Shadow observation mode

`SessionMode.SHADOW_OBSERVATION` (Section 19) runs the IDENTICAL
decision → order-policy → execution → accounting pipeline every other
mode uses, but every result lands in a private `PositionState`
`run_shadow_session` owns and a `ShadowObservation` ledger entry — there
is no code path from `shadow.py` into `portfolio.apply_fill_to_portfolio`;
a shadow session's real account (if one even exists) is structurally
unreachable from here. `ShadowObservation`'s own fields are all prefixed
`hypothetical_`/`counterfactual_`, and no report ever folds them into a
real account's P&L. Shadow mode is intentionally SIMPLIFIED relative to
the real runner: it does not model trading halts, session-close, or
financing for the shadow position (Section 19 requires only that
decisions/hypothetical orders be produced and observations persisted, not
a full parallel session-lifecycle simulation), and resuming an
interrupted shadow session is not supported — `run_shadow_session` fails
closed if any `MARKET_EVENT_ACCEPTED` entry already exists in the ledger,
rather than risk the kind of resume corruption the real runner's
dedicated (and more elaborate) resume machinery exists to prevent.

## Session manifest, event ledger, and resume semantics

`PaperSessionManifest` (Section 20) moves through `CREATED →
ELIGIBILITY_VERIFIED → INITIALIZED → RUNNING → ... → END_OF_STREAM →
RECONCILING → VERIFIED → COMPLETED`, with every transition legal per an
explicit table and every transition ALSO persisted as its own
`SESSION_TRANSITION` ledger entry (not merely a manifest field) — the
manifest is a cache for a fast "what stage" read; the ledger is the
source of truth `verify_paper_session` independently replays.

The event ledger (`persistence.py`, Section 21) is append-only and
hash-chained: every `LedgerEntry.previous_entry_hash` is the prior
entry's own `entry_id`, so `verify_ledger_chain_integrity` catches a
deleted/reordered/substituted entry that a bare sequence-number check
alone would miss. Appending an entry whose `entry_id` already exists is
an idempotent no-op, never an error or a duplicate — this is what makes
resume safe: re-processing an already-applied step re-derives the
identical entry and is silently absorbed.

Resume support (Section 23) rests on this idempotency plus a fail-closed
clean-event-boundary guard (`_require_clean_event_boundary`): every
`MARKET_EVENT_ACCEPTED` entry must have a matching `ACCOUNT_SNAPSHOT`
entry (the unconditional last entry of any event's processing) before a
resume is permitted to proceed. A genuine mid-event crash (accepted but
not yet fully processed) is refused outright rather than risked — the
session requires manual investigation, never an automatic best-effort
retry. This is a coarser recovery granularity than "mid-single-event
crash-safe": a crash cleanly BETWEEN events always resumes correctly and
reproduces byte-identical output; a crash truly mid-event is a documented
scope boundary, not a silent gap (see "Known limitations" below).

## Verification model and its independence classification

`verify_paper_session` (`verification.py`, Section 26) never trusts the
persisted manifest/final report. It independently: recomputes spec
identity; re-verifies the FULL eligibility chain (a genuine second call
into `robustness.verification.verify_robustness`/
`backtesting.verification.verify_backtest`, which themselves recompute
every statistic from raw evidence); re-checks manifest transitions
against the ledger's own `SESSION_TRANSITION` entries; verifies ledger
hash-chain integrity; and re-runs `reconciliation.reconcile_session`
(itself a from-ledger recomputation).

Section 26 requires classifying verification HONESTLY rather than
claiming uniform independence, and this report is a genuine mix:

- **Spec identity, ledger chain integrity, manifest transition
  legality** are STRUCTURALLY INDEPENDENT — pure recomputation from
  hashes/enums, no financial logic, defeatable only by a wrong
  structure, never a wrong number.
- **The eligibility chain** is SOURCE-RECONSTRUCTING — it recomputes
  every statistic from Milestone 6's raw evidence, not merely re-reading
  a persisted total.
- **Reconciliation** (position/cash/costs/equity) is ALGORITHMICALLY
  INDEPENDENT — it recomputes account state using the same formulas as
  the forward run, from the ledger's own persisted fills, but does NOT
  re-invoke the original `StrategyRuntime` against raw market data a
  second time to confirm the SAME decisions would be reached again.

The overall report is therefore **PARTIALLY INDEPENDENT**, stated
explicitly via `verification.INDEPENDENCE_CLASSIFICATION` — a true
decision-level re-execution (run twice, compare digests) is the real
acceptance workflow's own property (Section 33), not `verify_paper_
session`'s.

## Reporting and the backtest-comparison diagnostic

`reports.py` (Section 27) aggregates 15 durable summaries purely from the
ledger (session, strategy-decision, order, fill, execution-quality, cost,
position, account/equity, drawdown, risk-event, rejection, halt,
reconciliation, shadow-observation, verification) — a PRESENTATION layer,
not a re-verification layer; it freely uses the last `ACCOUNT_SNAPSHOT`
for aggregate equity/P&L figures (Section 21 explicitly permits treating
a snapshot as a cache) while independently replaying win/loss counts and
maximum drawdown from the full fill/snapshot history (a cheap,
materially more informative recomputation than trusting only the final
snapshot's own numbers).

`compare_paper_to_backtest` (Section 28) is diagnostic only, never a
promotion decision. It does not require equality where execution
semantics genuinely differ, and classifies each metric mismatch via a
fixed, documented heuristic (decision/order/abstention-count mismatches
are `unexpected_decision_mismatch` — paper and backtest should reach
identical decisions from identical inputs; return/cost/drawdown
mismatches are `expected_due_to_spread`; rejected-order-count mismatches
are `expected_due_to_latency`; turnover mismatches are `expected_due_to_
partial_fills`). This module intentionally does not reach into
`backtesting.reporting`'s own report-dict internals — extracting
comparable numbers from a specific backtest report format is the
caller's job, keeping the two report formats decoupled.

## Backtest-versus-paper differences (expected sources)

A paper session's numbers are not expected to equal the source backtest's
numbers even when strategy logic is identical: paper trading applies
per-event latency (decision-to-submit/submit-to-accept/accept-to-fill),
event-driven spread/slippage against the actual event stream rather than
a backtest's own cost model timing, and a live pre-trade risk/order-rate
layer that can reject orders a simpler backtest engine never would.
Genuine decision-count or abstention-count mismatches are NOT expected
and warrant investigation — those come straight from the strategy, which
should behave identically given identical inputs.

## Instrument contract and future MT5 boundary

`InstrumentSpec` (Section 14) is broker-neutral and MT5-agnostic: symbol,
base/quote interpretation, contract multiplier, tick size/value, quantity
step/min/max, precision, margin mode, account currency, financing
convention, trading timezone, session calendar identity — general enough
to represent a hypothetical future XAUUSD contract without hardcoding any
XAUUSD-specific value. Broker-specific values (tick size/value, margin
mode, session calendar) must later come from the real MT5 symbol
specification and may differ by broker; nothing in this package is a
claim about any real broker's actual terms. This milestone is explicitly
NOT an MT5 integration — no MT5 client library is imported anywhere (see
the Section 35 safety scan), and no adapter boundary for one has been
built; a future milestone would add a translation layer at the
`MarketEvent`/`OrderRequest`/`Fill` boundary this package already defines,
never inside it.

## Unsupported behavior (explicit)

The list below is current as of the Milestone 7 release audit (see
`docs/milestone7_delivery_report.md` for the audit's full defect list —
several items below were genuine defects found and fixed by that audit;
what remains here are confirmed, deliberate scope boundaries).

- **Live/forward market-data ingestion.** `market_data.py` (suggested by
  the original module list for cross-event ordering enforcement of a
  live/forward stream) was never built — Section 32 explicitly excludes
  building a market-data downloader or MT5 data ingestion in this
  milestone. `FORWARD_PAPER` is a legal `SessionMode` and `run_paper_
  trading_session` will happily stream whatever `Iterable[MarketEvent]`
  it is given (Section 37's streaming requirement is honored — no
  unlimited stream is ever materialized into memory), but no live
  producer of that iterable exists in this codebase.
- **In-flight LIMIT/STOP orders across a resume boundary — fixed.**
  Previously, a resume unconditionally discarded every resting working
  order (`working_orders` was reset to empty rather than rebuilt). The
  Milestone 7 release audit found this, and `_reconstruct_resume_state`
  now rebuilds `working_orders` by replaying every `ORDER_STATE_EVENT` in
  the ledger up to the last fully-processed event, so a resting LIMIT/STOP
  order survives a resume exactly as it would have without one. What
  remains genuinely unsupported is order **origination** — see the next
  item.
- **Order origination is MARKET-only.** `order_policy.py` (Section 9)
  fully validates, executes, and — per the fix above — resumes `LIMIT`/
  `STOP` orders, but the deterministic decision-to-order conversion never
  itself constructs a `LIMIT` or `STOP` `OrderRequest` from a
  `StrategyDecision`; every order the automated pipeline originates is
  `MARKET`. `LIMIT`/`STOP` support exists fully at the order/execution-
  model layer for callers that construct those requests directly — the
  strategy-decision pipeline simply does not use it. Confirmed by the
  Milestone 7 release audit (Area 2) as a scope boundary, not a data-loss
  or safety gap.
- **Latency-eligibility windows are computed but not enforced as a fill
  gate.** `clock.py`'s `acceptance_time_for`/`fill_eligible_time_for`
  compute the three configured latency delays (decision-to-submit,
  submit-to-accept, accept-to-fill-eligible) against event time, but
  `execution.py` does not currently reject or defer a fill for arriving
  before its `accept_to_fill_eligible_ms` window has elapsed — the values
  are available for a future fill-timing gate but are dead code today.
  Confirmed by the Milestone 7 release audit (Area 2).
- **Mid-single-event crash recovery.** A genuine interruption strictly
  between two ledger entries of the SAME event is refused (`PaperTradingStateError`),
  never silently retried — see "Session manifest, event ledger, and
  resume semantics" above. (The Milestone 7 release audit separately
  found and fixed a related but distinct gap: crashes at the manifest's
  own TAIL transitions, between `RUNNING` and `COMPLETED`, were not
  resume-safe. That is now fixed; this mid-single-event boundary remains
  a deliberate, fail-closed scope limit.)
- **Shadow-session resume.** `run_shadow_session` refuses to resume an
  already-started shadow session outright.
- **`CANCEL_OPEN_ORDERS` is defined but unreachable.**
  `RiskActionKind.CANCEL_OPEN_ORDERS` exists in the enum and transition
  table but no pre-trade or continuous risk check in `risk.py` currently
  produces it — it is dead code, not a silently-broken safety action.
  Confirmed by the Milestone 7 release audit (Area 9).
- **`TERMINATE_SESSION` is unreachable via the live runner.**
  `RiskActionKind.TERMINATE_SESSION` is only mapped from the
  `max_consecutive_execution_failures`/`max_reconciliation_discrepancy`
  continuous-risk checks, and `runner.py` currently always calls
  `evaluate_continuous_risk` with those two inputs hardcoded neutral
  (`0`/`None`). The transition and its handling are implemented and
  directly tested, but no live code path in this milestone actually
  triggers it. Confirmed by the Milestone 7 release audit (Area 9).
- **`PaperSessionManifest.stage` does not reach a HALTED/TERMINATED
  value.** The actual order-blocking safety gate is keyed off
  `KillSwitchState` (`risk.py`), which correctly latches and never
  returns to `ACTIVE` once left (see "Risk engine and kill switch"
  above) — that guarantee is unaffected. But `PaperSessionManifest.stage`
  itself has no `HALTING`/`HALTED`/`TERMINATED` member, so a session that
  was safety-halted still reaches `COMPLETED` at end-of-stream. This is
  an observability gap — a report reader must check `KillSwitchState`,
  not `stage`, to see that a session was halted — not a safety gap.
  Confirmed by the Milestone 7 release audit (Area 9).
- **`STOP_LIMIT`/`MARKET_ON_CLOSE` order types and `CorporateOrInstrumentAdjustmentEvent`.**
  Not implemented — Section 5/8 both explicitly permit omitting anything
  not fully, exactly simulable.
- **Content-addressed identity is not self-validated on
  deserialization.** Every `create_*` factory in this package (for
  `Fill`, `OrderRequest`, `LedgerEntry`, every `MarketEvent` subclass,
  `KillSwitchTransitionEvent`) computes its object's identity field from
  a provisional placeholder, then reconstructs the object using the real
  content hash — an object's `__post_init__` therefore cannot also
  require its identity field to already equal
  `compute_content_id(...)` of its own payload, or every factory would
  break. Tamper detection for a full ledger consequently relies on the
  hash-chain (`previous_entry_hash`) and, for semantic content
  specifically, the `compute_ledger_semantic_digest` function added by
  the Milestone 7 release audit — not on any single entry self-validating
  in isolation. A fix that added self-validation was attempted and
  reverted during the audit after it broke every factory's provisional-
  construction pattern; this is a documented architectural limitation,
  not an oversight.

## Explicit disclaimers

- **Paper trading does not prove profitability.** Every report generated
  by this package carries this disclaimer verbatim
  (`reports.DIAGNOSTIC_DISCLAIMER`).
- **Simulated fills are not broker fills.** No fill in this package was
  ever accepted by a real exchange or broker; every one is a deterministic
  simulation against a normalized market event.
- **Eligibility for paper trading is not live-trading approval.**
  `ELIGIBLE_FOR_PAPER_TRADING` (Milestone 6) authorizes exactly what its
  name says — simulation, never a real order.
- **No order is ever sent to any broker.** No component in this package
  opens a network connection of any kind (verified structurally by the
  Section 35 safety scan).
- **Broker-specific XAUUSD values must be obtained later from MT5.**
  Nothing in `InstrumentSpec`'s documentation example is a claim about
  any real broker's actual contract terms.
- **This milestone is not an MT5 integration.** No MT5 client library,
  adapter, or credential field exists anywhere in this package.
