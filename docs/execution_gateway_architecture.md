# Broker-Neutral Deterministic Execution Gateway (Milestone 8)

## Scope and explicit non-live safety boundary

Milestone 7 answers "given a genuinely `ELIGIBLE_FOR_PAPER_TRADING`
candidate, can its full trading lifecycle be simulated deterministically,
broker-neutrally, and exactly?" This milestone answers a narrower,
downstream question: **given a verified, COMPLETED Milestone 7 paper
session, can its orders be routed through a broker-neutral, event-sourced
EXECUTION GATEWAY — with a real dispatch transaction, idempotency,
sequencing, health/heartbeat, a kill switch, crash recovery, and
independent reconciliation — against a deterministic, in-process dummy
broker, without ever transmitting a real order?**

This is infrastructure and correctness work, exactly like Milestone 7. It
explicitly does **not** claim profitability, broker compatibility, broker
readiness, or operational live-trading readiness, and does not authorize
real-money execution.

**Explicit, structural non-live guarantees:**

- No MT5 implementation exists anywhere in this package.
- No FxPro implementation exists anywhere in this package.
- No real broker adapter exists anywhere in this package.
- No network connection is ever opened (proven structurally — Section
  35's AST-based safety scan; see "Safety scan" below).
- No broker credential field exists anywhere in this package's domain
  objects or configuration schema.
- No live order can be transmitted — `ExecutionMode` and `AdapterKind`
  are each **single-member enums** (`TEST_ONLY` / `DETERMINISTIC_DUMMY`);
  there is no `LIVE`, `MT5`, `FXPRO`, or `REAL_BROKER` value to even
  construct.
- Only the deterministic, in-process
  `dummy_broker.DeterministicDummyBrokerAdapter` is allowed as an
  adapter — it is the *only* concrete implementation of the
  `ExecutionAdapter` Protocol in this codebase.
- Every dummy-broker fill, snapshot, and account figure is **synthetic**
  — never a claim about any real market or broker.
- Passing every test in this package does not prove profitability, does
  not prove broker compatibility, does not prove broker readiness, does
  not prove operational live-trading readiness, and does not authorize
  real-money execution.
- Milestone 9 has not been started.

## Package architecture and dependency direction

New top-level package `quant_platform.execution_gateway`, one layer above
`paper_trading`:

```
core
  ↑
ml / historical / features / backtesting / robustness
  ↑
paper_trading
  ↑
execution_gateway
```

`paper_trading`, `robustness`, `backtesting`, and every lower-level
package import nothing from `execution_gateway` — the dependency
direction is strictly one-way, exactly like `paper_trading`'s own
relationship to `robustness` one layer down. `execution_gateway` freely
reuses `paper_trading`'s own models, identity primitives, eligibility
verification, locking, and market-event types rather than duplicating
them (Section 2's own instruction: "Do not duplicate formulas or
infrastructure that already exists").

### Module layout

- `models.py` — every shared enum (`ExecutionMode`, `AdapterKind`,
  `AuthorizationMode`, `CommandType`, `ExecutionOrderState`,
  `SequencingPolicyKind`, `BrokerEventType`, `DispatchStage`,
  `ExecutionLedgerEntryKind`, `ExecutionSessionStage`,
  `AdapterHealthStatus`, `ExecutionKillSwitchState`) and every
  linear/branching legal-transition table (order states, session stages,
  kill-switch states). Re-exports `OrderSide`/`OrderTypeKind`/
  `TimeInForceKind` directly from `paper_trading.models` rather than
  redefining byte-identical vocabularies.
- `identity.py` — the shared content-addressing primitive
  (`compute_content_id`, re-exported from `paper_trading.identity`
  directly) plus `Decimal`↔JSON round-tripping helpers
  (`decimal_to_json`, `decimal_from_float`, `parse_decimal`) every
  content-addressed type in this package uses.
- `specs.py` — `ExecutionGatewaySpec` (the content-addressed identity
  root) and every embedded policy spec: `SequencingPolicySpec`,
  `IdempotencyPolicySpec`, `RecoveryPolicySpec`,
  `ReconciliationPolicySpec`, `HealthPolicySpec`, `HeartbeatPolicySpec`,
  `KillSwitchPolicySpec`, `DispatchPolicySpec`, and
  `DummyBrokerScenarioSpec` (with its own nested `RejectionRuleSpec`
  collection).
- `paper_bridge.py` — `ExecutionIntent` (Section 5), `ExecutionAuthorization`
  (Section 6), and `execution_intent_from_paper_order` (Section 6's own
  `ExecutionIntentFactory.from_paper_order`), which performs every one of
  Section 6's 13 required checks, fail-closed, before a single order is
  bridged.
- `commands.py` — the eight immutable command types
  (`SubmitOrderCommand`, `CancelOrderCommand`, `ReplaceOrderCommand`,
  `QueryOrderCommand`, `QueryOpenOrdersCommand`, `QueryPositionsCommand`,
  `QueryAccountCommand`, `HeartbeatCommand`), deterministic command
  identity, and `derive_client_order_id`.
- `state_machine.py` — `ExecutionOrderStateEvent` and
  `resolve_execution_order_state` (Section 8's event-sourced order-state
  derivation, mirroring `paper_trading.orders.resolve_order_state`
  exactly).
- `events.py` — the normalized `BrokerEvent` model (Section 14).
- `states.py` — the reconstructed `ExecutionOrder` aggregate (Section 9)
  and `ExecutionFill` (Section 10), plus `reconstruct_execution_order`.
- `adapter.py` — the `ExecutionAdapter` Protocol, `AdapterCapabilities`,
  `AdapterCallResult`, and the Section 25 broker snapshot types
  (`BrokerOrderSnapshot`, `BrokerPositionSnapshot`, `BrokerAccountSnapshot`).
- `dummy_broker.py` — `DeterministicDummyBrokerAdapter`, the one and
  only adapter implementation.
- `normalization.py` — the normalization boundary
  (`DummyBrokerRawEvent`/`normalize_dummy_broker_event`) between a
  dummy-broker-specific raw shape and the generic `BrokerEvent`.
- `sequencing.py` — `classify_broker_event` (Section 15).
- `idempotency.py` — durable idempotency-evidence reconstruction from
  the ledger (Section 16).
- `dispatcher.py` — `dispatch_command` (the dispatch transaction,
  Section 17) and `process_broker_events` (inbound classification and
  application, Section 15/18).
- `persistence.py` — the append-only, hash-chained `ExecutionLedgerEntry`
  ledger (`ExecutionSessionEventStore`, Section 18).
- `manifests.py` — `ExecutionSessionManifest`/`ExecutionSessionManifestStore`
  (Section 19) and `execution_session_lock` (a thin adapter over
  `ml.concurrency.experiment_lock`, reused unchanged).
- `health.py` / `heartbeat.py` — `AdapterHealthSnapshot`/
  `compute_health_status` and `HeartbeatOutcome`/`heartbeat_lag_status`
  (Section 21).
- `kill_switch.py` — `ExecutionKillSwitchTransitionEvent`,
  `resolve_execution_kill_switch_state`, and `authorize_dispatch` — the
  ONE centralized dispatch gate (Section 22).
- `recovery.py` — `recover_unknown_orders` (Section 23).
- `reconciliation.py` — `reconcile_execution_session` (Section 24).
- `reports.py` — `generate_execution_session_report` (Section 27).
- `verification.py` — `verify_execution_session` (Section 28).
- `replay.py` — `replay_execution_session` (Section 30), a reusable
  fresh-store wrapper around the runner used for determinism comparisons.
- `runner.py` — `run_execution_session` (Section 19/22), the single
  orchestrator, plus `pause_execution_session`.

Plus `config/execution_gateway_schemas.py` (Section 29's Pydantic
configuration schema) and 24 new domain exceptions under
`ExecutionGatewayError` in `core/exceptions.py`.

## Domain objects: intent versus command distinction

An `ExecutionIntent` (Section 5) expresses WHAT should happen
economically — it is produced exactly once, from a source paper order,
by `paper_bridge.py`, and this package never decides strategy direction:
every economic field on an `ExecutionIntent` is copied from, and
independently re-verified against, the already-decided source
`paper_trading.orders.OrderRequest`. A `Command` (Section 7) is the
concrete, adapter-facing operation that carries an intent out (or
queries/heartbeats, which carry no economic intent at all).
`commands.py` is the ONLY place a `Command` is constructed;
`dispatcher.py` is the ONLY place a `Command` is ever sent to an
adapter.

## Execution authorization and the paper-session bridge

`ExecutionAuthorization` (Section 6) is the explicit, TEST_ONLY proof
that a given source paper order may be bridged at all —
`authorization_mode` accepts only `TEST_ONLY_DUMMY_EXECUTION`.
`execution_intent_from_paper_order` performs, in order, fail-closed:

1. source paper session exists;
2. source paper session is COMPLETED and independently re-verified
   (`paper_trading.verification.require_paper_session_verified`) — not
   merely "manifest says COMPLETED";
3. source paper session belongs to the referenced paper-trading spec
   (recomputed identity, AND the manifest's own persisted spec-artifact
   content hash cross-checked against `ExecutionGatewaySpec.
   paper_trading_spec_id`);
4. source paper session still passes authoritative eligibility
   verification — a FRESH call to
   `paper_trading.eligibility.require_paper_trading_eligibility` every
   single time, never cached;
5. source order belongs to the source session;
6. strategy candidate identity matches;
7. model identity matches;
8. instrument identity matches (both the symbol string AND a recomputed
   content hash of the full `InstrumentSpec` against
   `ExecutionGatewaySpec.instrument_spec_id`);
9. quantity and side match the source order exactly (copied verbatim,
   never re-derived);
10. every other economic field (order type, prices, time-in-force,
    reduce-only) is copied verbatim from the source order;
11. the freshly-issued authorization belongs to the same source order;
12. the authorization is TEST_ONLY;
13. the adapter kind is DETERMINISTIC_DUMMY.

## Adapter protocol and capabilities

`ExecutionAdapter` is a `typing.Protocol` — `dispatcher.py`,
`recovery.py`, `reconciliation.py`, and `runner.py` talk to it
exclusively, never to `DeterministicDummyBrokerAdapter` by name. This is
what would let a future MT5 adapter (not built in this milestone) drop
in without changing a single line of strategy, risk, ledger,
reconciliation, or execution domain logic.

`AdapterCapabilities` declares 17 boolean flags (order-type support,
cancel/replace support, partial-fill support, client-order-id support,
broker-sequence support, reduce-only support, four query capabilities,
and three idempotency guarantees plus event-ordering guarantee).
`adapter.require_capability`/`require_query_capability` are called
by `dispatcher.py` **before** every adapter call — a command whose
requirements the active adapter does not declare is rejected before
dispatch, never silently downgraded.

`AdapterCallResult` is deliberately **not** the order's lifecycle — it
is proof the synchronous call itself definitely reached
(`accepted_for_processing=True`) or definitely did NOT reach
(`accepted_for_processing=False`, with a reason) the adapter. The actual
order lifecycle (acknowledged/filled/rejected) arrives later,
asynchronously, through `poll_events` as normalized `BrokerEvent`s —
this split is what lets the dummy broker model delayed, duplicated, and
out-of-order event delivery without conflating it with "did the call
itself go through."

## Execution-order state machine and `UNKNOWN` semantics

`ExecutionOrderState` has 15 members. Terminal states:
`FILLED`/`CANCELLED`/`REJECTED`/`EXPIRED`/`FAILED`. `UNKNOWN` is
deliberately **not** terminal — it always has a legal path forward (to
any state a legitimate broker response could produce, or to `FAILED` if
reconciliation proves the ambiguity can never be resolved), but a
session may never reach `COMPLETED` while any order remains `UNKNOWN`
(the runner's own completion gate, Section 19).

Every transition is explicit in
`models._LEGAL_EXECUTION_ORDER_TRANSITIONS`, mirroring
`paper_trading.orders._LEGAL_ORDER_TRANSITIONS`'s identical construction.
A synchronous adapter rejection of a fresh submit passes through
`DISPATCHED` before `REJECTED` (`REJECTED` is only a legal target from
`DISPATCHED`, never directly from `DISPATCH_PENDING`) — the adapter DID
synchronously respond, it just responded with a refusal. `CANCEL_PENDING`/
`REPLACE_PENDING` may legally revert to `ACKNOWLEDGED` or
`PARTIALLY_FILLED` (whichever the order actually was immediately before
the cancel/replace was dispatched) if the adapter synchronously refuses
the request.

## Dispatch transaction

`dispatcher.dispatch_command` is the durable dispatch transaction
(Section 17): `COMMAND_CREATED` → (capability check) → `COMMAND_VALIDATED`
→ `COMMAND_DISPATCH_INTENT` (persisted **before** the adapter is ever
called) → the adapter call itself → `COMMAND_DISPATCH_SUCCEEDED` /
`COMMAND_DISPATCH_REJECTED` / `COMMAND_MARKED_UNKNOWN`. An idempotency
check (`idempotency.is_command_already_recorded`) runs first — a
command already durably recorded with an IDENTICAL payload is a safe
no-op returning its prior resolution; a different payload under the same
`command_id` raises `ExecutionIdempotencyError` rather than silently
proceeding. Every exception from the adapter call itself is classified
as genuinely AMBIGUOUS (`COMMAND_MARKED_UNKNOWN`), never mapped to a
blanket "failed."

## Idempotency

`command_id` (Section 7) is computed from each command's own ECONOMIC
payload only — `command_sequence`/`event_time` are excluded — so
redispatching the identical economic operation (a safe retry) recomputes
the identical `command_id`. `client_order_id` is derived deterministically
from `execution_intent_id` alone (`derive_client_order_id`), never a
random UUID — one economic submit maps to exactly one stable
`client_order_id`, reused on every safe retry. `idempotency.py`'s index
builders (`build_command_index`, `build_client_order_index`,
`build_broker_order_index`, `build_broker_event_index`) reconstruct this
evidence purely from the durable ledger — never an in-memory-only set —
and raise if the ledger itself shows a genuine collision (a
`client_order_id` mapped to two different intents, a `broker_order_id`
associated with two different `client_order_id` values, and so on).

## Broker-event normalization and sequencing

`normalization.normalize_dummy_broker_event` is the ONE place
dummy-broker-specific knowledge is allowed to exist — every module
downstream (`states.py`, `sequencing.py`, `reconciliation.py`,
`verification.py`) only ever sees the generic `BrokerEvent`.
`sequencing.classify_broker_event` implements all three
`SequencingPolicyKind` values (`STRICT_SEQUENCE`, `TIMESTAMP_AND_ID`,
`ARRIVAL_ORDER_ONLY`); `ExecutionGatewaySpec.__post_init__` REQUIRES
`STRICT_SEQUENCE` whenever `adapter_kind is DETERMINISTIC_DUMMY` (the
only adapter kind that exists in this milestone) — `ARRIVAL_ORDER_ONLY`'s
lower assurance is documented and unit-tested, but no spec this milestone
constructs selects it for the one adapter it actually runs. A
sequence-slot collision (two different events claiming the same
`broker_sequence`) is classified `SEQUENCE_CONFLICT` and is CRITICAL; a
gap or an old, previously-unseen sequence is recorded and reported, never
silently skipped.

## Deterministic dummy broker

`DeterministicDummyBrokerAdapter` (Section 12) is a real, stateful,
in-process order-book simulator — not a trivial mock. No result-affecting
behavior depends on wall-clock time; every outcome is a pure function of
`DummyBrokerScenarioSpec`, the commands it receives, the market events
`advance_market_event` is fed, and its own internal event count.

**Architecture**: `submit_order`/`cancel_order`/`replace_order` are
SYNCHRONOUS acknowledgements only. `advance_market_event` (called once
per consumed market event, exactly like `paper_trading.execution`'s own
fill engine — a dummy-broker-specific method, not part of the generic
`ExecutionAdapter` Protocol) is where `ORDER_ACKNOWLEDGED`/fill/expiry
events are actually GENERATED into the broker's own authoritative
internal event log, in strict generation order. `poll_events` is where
scenario-configured duplicate/delayed/out-of-order DELIVERY is applied
ON TOP of that already-generated log — generation order and delivery
order are deliberately different concerns, which is what lets the
scenario model realistic delivery anomalies without corrupting the
underlying economic timeline.

### MARKET / LIMIT / STOP semantics

- **MARKET**: buy fills at the current event's ask (or bar close); sell
  fills at bid (or bar close). No spread/slippage is applied by the
  broker itself — see "Raw prices, no cost modeling" below.
- **LIMIT**: buy triggers when the reference price is at or below the
  limit; sell triggers when at or above. Fills AT the limit price.
- **STOP**: buy triggers when the reference price is at or above the
  stop; sell triggers when at or below. Once triggered, the order
  becomes market-like and fills at the CURRENT reference price (not the
  stop price itself) — standard stop-order convention.

### IOC / FOK / DAY / GTC semantics

- **FOK**: fills the full quantity immediately if fully fillable per the
  scenario's `partial_fill_schedule`; otherwise cancelled outright,
  NEVER partially filled — enforced both by the dummy broker's own fill
  logic and, independently, by `states.ExecutionOrder.__post_init__`
  (a FOK order observed in `PARTIALLY_FILLED` is a structural
  contradiction) and by `verification.py`'s own FOK invariant re-check.
- **IOC**: fills whatever quantity is immediately available; any
  remainder is cancelled immediately, never left working.
- **DAY**: expires when `expire_day_orders` is called (the runner calls
  this deterministically once the supplied market-event stream is
  exhausted) — never wall-clock-derived.
- **GTC**: remains working indefinitely; unaffected by DAY expiry.

### Partial fills

Governed entirely by `DummyBrokerScenarioSpec.partial_fill_schedule` — an
explicit, ORDER-SENSITIVE sequence of fractions (Section 4: "declared
durable order must be preserved"). This is a simpler, fully
deterministic-by-construction model than inferring partial fills from a
disclosed quote size; liquidity is never invented.

### Cancellation and replacement

`cancel_order` is idempotent — cancelling an already-terminal order is a
safe no-op. `replace_order` updates only the fields the
`ReplaceOrderCommand` actually specifies (quantity/limit
price/stop price/time-in-force); a replacement quantity at or below the
already-filled quantity is rejected synchronously.

### Raw prices, no cost modeling

The dummy broker reports fills at the RAW ask/bid/close reference price
only — no spread, slippage, or commission is applied inside
`dummy_broker.py`. Cost components are computed by whichever layer
constructs the durable `ExecutionFill` record (currently `dispatcher.py`,
which records `commission=spread_component=slippage_component=Decimal(0)`
for this milestone's scope — see "Known limitations" below), reusing
this platform's existing cost formulas rather than duplicating them, per
Section 12's own instruction.

### No margin modeling

The dummy broker's own account view is fully cash-settled — no margin,
no leverage, no liability line item, mirroring `paper_trading.portfolio`'s
own explicit, documented choice (Section 25 permits this: "if margin is
not modeled, document that explicitly").

## Append-only execution ledger and hash-chain integrity

`persistence.ExecutionLedgerEntry` mirrors
`paper_trading.persistence.LedgerEntry`'s architecture exactly: each
entry's `previous_entry_hash` is the prior entry's own `entry_id`;
`verify_execution_ledger_chain_integrity` walks this like a hash chain.
Appending an entry whose `entry_id` already exists at the same sequence
is an IDEMPOTENT no-op; the same `entry_id` at a different sequence, or a
different `entry_id` reused at an already-occupied sequence, is a
genuine `ExecutionIdempotencyError`/structural conflict, never silently
absorbed.

### Semantic verification, separate from hash integrity

`persistence.compute_execution_ledger_semantic_digest` normalizes each
entry to `{entry_sequence, entry_kind, payload}` before hashing —
excluding `entry_id`/`entry_hash`/`previous_entry_hash` (pure hash-chain
linkage artifacts) AND both `recorded_time` (genuinely wall-clock) and
`event_time` (excluded uniformly, mirroring
`paper_trading.persistence.compute_ledger_semantic_digest`'s own
reasoning, since a session-level transition entry has no market event to
anchor its timestamp to). Hash-chain integrity alone is insufficient to
catch semantic tampering with all hashes correctly recomputed — this
digest is the independent check for that, and is exactly what the
semantic-tampering test group targets.

## Session manifest and locking

`ExecutionSessionManifest` (Section 19) moves through `CREATED` →
`SPEC_VERIFIED` → `SOURCE_ELIGIBILITY_VERIFIED` → `ADAPTER_INITIALIZED`
→ `RECOVERY_CHECKED` → `RUNNING` → (`PAUSING` → `PAUSED` →, on resume,
back to `SOURCE_ELIGIBILITY_VERIFIED`) → `RECONCILING` → `VERIFYING` →
`COMPLETED`, with `HALTING`/`HALTED`/`FAILED`/`TERMINATED` as explicit
alternate branches. `COMPLETED` is granted only when: no unresolved
`UNKNOWN` order remains, and reconciliation found no `BLOCKING`/`CRITICAL`
issue — both checked FRESH every call from the ledger, never inferred
from a persisted boolean.

`execution_session_lock` is a thin adapter over
`ml.concurrency.experiment_lock`, reused unchanged (Section 20's own
instruction) — no second locking implementation exists in this package.
Each execution session has an exclusive lock for every mutating
operation (`create`/`transition`/`bump_resume_count`); inspection/query
commands are read-only and do not need it.

## Health and heartbeat

`health.compute_health_status` applies `HealthPolicySpec`'s thresholds
deterministically from caller-supplied counters (`consecutive_failures`,
`event_lag`, a `disconnected` flag) — never a live clock.
`AdapterHealthStatus` progresses `HEALTHY` → `DEGRADED` → `STALE` /
`UNAVAILABLE` → `RECOVERING`; `UNAVAILABLE`/`STALE` structurally forbid
`can_submit=True` (`AdapterHealthSnapshot.__post_init__` enforces this).
`heartbeat.heartbeat_lag_status` applies `HeartbeatPolicySpec`'s
thresholds to a caller-supplied consecutive-missed count.

## Kill switch and centralized dispatch gate

`ExecutionKillSwitchState` has six members: `ACTIVE`, `DEGRADED`,
`HALTING`, `HALTED`, `RECOVERING`, `TERMINATED`. Unlike Milestone 7's
paper-trading kill switch (which never resumes `ACTIVE` after a genuine
safety halt), Section 22 explicitly defines `RECOVERING` as a state whose
purpose IS to restore `ACTIVE` once broker query and reconciliation
succeed again — but `HALTED` only ever reaches `RECOVERING` via an
explicit administrative action, and `RECOVERING` may itself re-detect the
same problem and return to `HALTING` rather than `ACTIVE`.

`kill_switch.authorize_dispatch` is the ONE centralized dispatch gate
(Section 22) — every command in this package passes through it before
`dispatcher.dispatch_command`, no exceptions:

| Kill-switch state | New-exposure submit | Reduce-only submit / cancel | Replace | Query / heartbeat |
|---|---|---|---|---|
| `ACTIVE` | ✅ | ✅ | ✅ | ✅ |
| `DEGRADED` | ❌ | ✅ | ❌ | ✅ |
| `HALTING` | ❌ | ✅ | ❌ | ✅ |
| `HALTED` | ❌ | ❌ | ❌ | ✅ |
| `RECOVERING` | ❌ | ❌ | ❌ | ✅ |
| `TERMINATED` | ❌ | ❌ | ❌ | ❌ |

A safety cancel or an authorized reduce-only submit is deliberately
**never** blocked by the same new-exposure rule it exists to mitigate
(`models.REDUCE_ONLY_PERMITTING_KILL_SWITCH_STATES` is a strict superset
of `NEW_EXPOSURE_PERMITTING_KILL_SWITCH_STATES`).

## Crash recovery

`recovery.recover_unknown_orders` (Section 23) NEVER trusts cached
order state — for every order currently `UNKNOWN`, it queries the
adapter fresh (`query_order`) and resolves per Section 16's rules
exactly:

- broker confirms a recognized state → the `UNKNOWN` order transitions
  to that state, backed by a durable `ORDER_STATE_TRANSITION` entry;
- broker has no record AND the adapter guarantees idempotent submit →
  `safe_retry_authorized` (the caller may redispatch using the SAME
  `client_order_id`, never a new one);
- broker has no record and the adapter does NOT guarantee idempotent
  submit → remains `UNKNOWN`, new exposure stays halted — never a blind
  retry of a potentially-already-accepted non-idempotent submit;
- the query itself raises → recorded as a `RECONCILIATION_FAILED` ledger
  entry, remains `UNKNOWN`.

`runner.run_execution_session` calls this immediately after
`ADAPTER_INITIALIZED` on every resume, before any new command is
dispatched.

## Reconciliation

`reconciliation.reconcile_execution_session` (Section 24) NEVER raises
for an ordinary mismatch — every finding becomes a structured
`ReconciliationIssue` (`INFO`/`WARNING`/`BLOCKING`/`CRITICAL`), even
against a badly corrupted ledger; a genuine structural failure (the
adapter itself cannot be queried) is ALSO captured as a `CRITICAL` issue,
never an uncaught exception. Checks implemented: open orders vs. broker
open orders (existence, `broker_order_id`/side/quantity/status
agreement), unknown broker orders, positions vs. broker positions,
account balance/equity finiteness, unresolved `UNKNOWN` orders, broker
sequence gaps, session ownership, contract-multiplier consistency across
intent/order/fill, and duplicate-fill detection. `BLOCKING`/`CRITICAL`
issues prevent `COMPLETED` (enforced by `runner.py`).

## Contract-multiplier correctness

`ExecutionFill.gross_notional` is validated EXACTLY equal to
`quantity * price * contract_multiplier` at construction
(`__post_init__`), using `Decimal` arithmetic throughout — never `float`.
`reconciliation.py` independently cross-checks every fill's
`contract_multiplier` against its own order's declared value.

## Account and position snapshots

`BrokerOrderSnapshot`/`BrokerPositionSnapshot`/`BrokerAccountSnapshot`
(Section 25) are the dummy broker's own independently-maintained view —
constructed from the broker's OWN fills via a compact weighted-average-
cost implementation (the same convention `paper_trading.accounting` uses,
reimplemented compactly here to keep the dummy broker's bookkeeping
genuinely independent of the gateway's own from-ledger reconstruction,
which is the entire point of reconciliation).

## Semantic digest and deterministic replay

`replay.replay_execution_session` (Section 30) runs a spec end-to-end
against a FRESH, isolated store and a fresh
`DeterministicDummyBrokerAdapter`, returning a `ReplayResult` bundling
the final semantic digest, reconciliation report, verification report,
and session report. Two calls with the SAME immutable inputs — even
across separate processes, different `PYTHONHASHSEED` values, and
different temporary storage paths — produce the SAME
`ExecutionSessionManifest.semantic_digest`.

## Verification and its independence classification

`verification.verify_execution_session` (Section 28) NEVER trusts the
persisted manifest, a final report, or a cached order/position view — it
independently recomputes spec identity, ledger chain integrity, ledger
session ownership, command/client-order-id/broker-order-id idempotency
evidence, order-state-transition legality (full replay), cumulative-fill
legality, FOK-never-partial, fill-identity uniqueness, command-prohibition-
after-halt, and a structural scan for live/broker-shaped content in the
ledger's own payloads.

Classified honestly, per Section 28's own instruction:

- **Ledger chain integrity, session ownership, entry sequencing,
  order-state-transition legality, command/broker-event identity**:
  STRUCTURALLY INDEPENDENT (pure recomputation from hashes/enums, no
  financial logic).
- **Reconciliation** (positions/account vs. the dummy broker's own
  independently-maintained bookkeeping): ALGORITHMICALLY INDEPENDENT —
  but it trusts the broker's own quoted fill price as ground truth,
  exactly like a real reconciliation would trust a real broker's fill
  confirmations; it does not independently re-derive what price SHOULD
  have been quoted.
- **Source eligibility** (of the underlying Milestone 6/7 chain):
  SOURCE-RECONSTRUCTING, delegated entirely to
  `paper_trading.eligibility.verify_paper_trading_eligibility`.

Net honest classification: **PARTIALLY INDEPENDENT** — this is not a
second, independent implementation of order/fill/price determination; it
is an independent reconciliation of the same durable evidence the
forward run itself produced, plus a fresh cross-check against the
broker's own separately-maintained state.

## CLI

Sixteen additive commands (`create-execution-gateway-spec`,
`run-dummy-execution-session`, `resume-execution-session`,
`pause-execution-session`, `inspect-execution-session`,
`inspect-execution-intents`, `inspect-execution-commands`,
`inspect-execution-orders`, `inspect-execution-fills`,
`inspect-broker-events`, `inspect-execution-health`,
`inspect-execution-reconciliation`, `report-execution-session`,
`verify-execution-session`, `replay-execution-session`,
`compare-execution-to-paper`), registered exactly like every Milestone
7 command: `main()`'s existing `except (QuantPlatformError, ...)`
handler already catches every `ExecutionGatewayError` subclass and
prints `ERROR: {exc}` to stderr with no traceback — no new error-handling
code was needed for this. Every inspection command has a bounded
`--limit` (default 200, reusing Milestone 7's own
`_DEFAULT_INSPECTION_ROW_LIMIT`/`_require_non_negative_limit`, which
already rejects a negative `--limit`). `resume-execution-session`
requires `--execution-session-id` to match what `--config` resolves to,
mirroring the Milestone 7 release audit's own identity-binding fix for
`resume-paper-session`.

## Clean-install CLI testing

See `docs/milestone8_delivery_report.md`'s own dedicated section for the
exact clean-venv subprocess test result.

## Acceptance workflow

See `docs/milestone8_delivery_report.md`'s own dedicated section.

## Unsupported behavior (explicit)

- **Live/forward market-data ingestion.** No market-data downloader or
  MT5 data ingestion exists anywhere in this package — `runner.py`
  streams whatever `Iterable[MarketEvent]` it is given, exactly like
  `paper_trading.runner`'s own scope boundary.
- **Cost modeling inside the dummy broker.** The dummy broker reports
  raw fill prices only; `dispatcher.py` currently records
  `commission=spread_component=slippage_component=Decimal(0)` for every
  fill rather than wiring in `backtesting.costs`'/`paper_trading.costs`'
  existing formulas — a scope simplification, not a correctness defect
  (every reconciliation/verification check that depends on cost
  components remains internally consistent since both sides of every
  comparison use the same zero).
- **`STOP_LIMIT`/`MARKET_ON_CLOSE` order types.** Not implemented,
  mirroring Milestone 7's identical scope boundary.
- **A future MT5 adapter.** Deliberately not built in this milestone —
  the `ExecutionAdapter` Protocol is the seam a future milestone would
  implement against.

## Known limitations

See `docs/milestone8_delivery_report.md`'s dedicated section for the
full, classified list (BLOCKER / FIXED BLOCKER / NON-BLOCKING LIMITATION
/ FUTURE ENHANCEMENT / UNSUPPORTED AND FAIL-CLOSED /
DOCUMENTATION-ONLY ISSUE).

## Future MT5 adapter boundary

A future milestone would add a translation layer implementing the
`ExecutionAdapter` Protocol (`adapter.py`) at the
`Command`/`BrokerEvent`/`BrokerOrderSnapshot` boundary this package
already defines — normalizing MT5's own wire format through a NEW
`normalization.py`-equivalent boundary, never inside `dispatcher.py`,
`kill_switch.py`, `reconciliation.py`, or `verification.py`, none of
which reference `DeterministicDummyBrokerAdapter` by name anywhere.
`AdapterCapabilities` already exists precisely so a lower-capability real
adapter can declare what it does NOT support and have every consumer
fail closed accordingly, rather than silently assuming dummy-broker-level
capability.
